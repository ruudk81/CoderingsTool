#%%
import os, sys
# Ensure src/ is on sys.path so imports work regardless of cwd
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ===  IMPORTS ========================================================================================================
import time
import warnings
import pandas as pd
import numpy as np  # noqa: F401 — used by downstream utilities

# Suppress IPython exit warning when running in Streamlit
warnings.filterwarnings("ignore", message="To exit: use 'exit', 'quit', or Ctrl-D.")
import nest_asyncio
nest_asyncio.apply()

import models
from utils import dataLoader
from utils.cacheManager import CacheManager
from utils.llm import token_tracker
from config import CacheConfig, ModelConfig, DEFAULT_LANGUAGE
cache_config = CacheConfig()
cache_manager = CacheManager(cache_config)
model_config = ModelConfig()

#  ===  STANDALONE ========================================================================================================

#filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
#id_column = "DLNMID"
#var_name = "Q20"
#sample_size = 500

filename = "M000000 Associatiemonitor Merk X net databestand.sav"
id_column = "DLNMID"
var_name = "Qd1_combined"
sample_size = 2000

#filename = "M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
#id_column = "DLNMID"
#var_name = "Q15"
#sample_size = 2000

#filename = "M250127 Flitspeiling NAVOtop 0meting_153832.sav"
#id_column = "DLNMID"
#var_name = "Q10"
#sample_size = 50

RUN_UNTIL_STEP = 0
FORCE_RECALCULATE_ALL = False
VERBOSE = True
PROMPT_PRINTER = False
LANGUAGE = "nl"

STEP_NAMES = {
    0: "data",
    1: "preprocessed",
    2: "quality_filter",
    3: "extracted_ideas",
    4: "code_assignment",
    5: "export",
}

# ===================================================================================================================
# HELPERS
# ===================================================================================================================

def _resolve_step_defaults(variable_key=None, cache_manager=None, model_config=None):
    """Resolve default values for step function parameters.

    Each parameter is resolved only if passed as None:
    - variable_key: generated from module-level globals (selected_variables, etc.)
    - cache_manager: falls back to module-level global, then creates default
    - model_config: falls back to module-level global, then creates default

    Returns: (variable_key, cache_manager, model_config)
    """
    if variable_key is None:
        from utils.cacheManager import generate_enhanced_variable_key
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        _sample_size = globals().get('sample_size', None)
        _merge_config = globals().get('merge_config', None)
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged, sample_size=_sample_size, merge_config=_merge_config
        )

    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

    return variable_key, cache_manager, model_config


def _build_codebook_from_mece(mece_results_cache, category_results, var_name):
    """Build a ThemeEnrichedCodebookModel directly from MECE categories.

    This replaces steps 6-8 (codebook generation, refinement, code assignment)
    by using the MECE partition categories as the codebook directly.

    Args:
        mece_results_cache: CodingResultsCache with partition_results
        category_results: List[CodeAssignedModel] from classNcoder
        var_name: Variable name for metadata

    Returns:
        ThemeEnrichedCodebookModel
    """
    enriched_entries = []
    code_to_theme_mapping = {}
    themes_summary = []

    if not mece_results_cache or not mece_results_cache.partition_results:
        return models.ThemeEnrichedCodebookModel(
            codes=[],
            themes_summary=[],
            code_to_theme_mapping={},
            theme_methodology="Taxonomy pipeline - no MECE results available",
            source_variable=var_name
        )

    for partition_name, partition_result in mece_results_cache.partition_results.items():
        # Each partition becomes a theme
        theme_name = partition_name

        theme_codes = []
        for cat in partition_result.categories:
            code_name = cat.category_label
            definition = cat.inclusion_definition

            entry = models.ThemeEnrichedCodebookEntry(
                code=code_name,
                definition=definition,
                theme=theme_name,
                theme_description=theme_name,
                category="",
                category_description="",
                source_cluster=None,
            )
            enriched_entries.append(entry)
            theme_codes.append(code_name)
            code_to_theme_mapping[code_name] = theme_name

        themes_summary.append({
            'theme_name': theme_name,
            'theme_description': theme_name,
            'code_count': len(theme_codes)
        })

    return models.ThemeEnrichedCodebookModel(
        codes=enriched_entries,
        themes_summary=themes_summary,
        code_to_theme_mapping=code_to_theme_mapping,
        theme_methodology="Taxonomy pipeline - MECE categories as codebook",
        generation_metadata={
            "pipeline": "taxonomy",
            "total_codes": len(enriched_entries),
            "partitions": len(mece_results_cache.partition_results),
        },
        source_variable=var_name
    )


# ===================================================================================================================
# PROCESSING STEPS
# ===================================================================================================================

def step_0_load_data(
    filename,
    id_column,
    var_name=None,                    # Single variable (optional)
    var_names=None,                   # Multiple variables (optional)
    variable_key=None,                # Auto-generate if None
    cache_manager=None,               # Use global if None
    sample_size=None,                 # Limit sample size
    encoding=None,                    # SPSS file encoding
    merge_config=None,                # How to merge multiple vars
    force_recalc=False,
    verbose=True,
    streamlit_container=None,         # Optional progress updates
    data_dir=None                     # Data directory path (uses cwd-based default if None)
):
    """Step 0: Load data from SPSS file (single or multiple variables)"""
    from utils.verboseReporter import VerboseReporter

    step_name = "data"
    verbose_reporter = VerboseReporter(verbose)

    # Auto-generate variable_key if not provided
    if variable_key is None:
        from utils.cacheManager import generate_enhanced_variable_key
        if var_names and len(var_names) > 1:
            selected_vars = var_names
            is_merged = True
        else:
            selected_vars = [var_name] if var_name else (var_names if var_names else ["unknown"])
            is_merged = False
        variable_key = generate_enhanced_variable_key(selected_vars, is_merged, sample_size, merge_config)

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    if streamlit_container:
        if var_names and len(var_names) > 1:
            streamlit_container.text(f"Loading and merging {len(var_names)} variables...")
        else:
            streamlit_container.text("Loading data from SPSS file...")

    # Check cache
    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        raw_text_list = cache_manager.load_from_cache(filename, step_name, variable_key, models.ResponseModel)
        verbose_reporter.summary("DATA FROM CACHE", {"Input": f"{len(raw_text_list)} responses"})
    else:
        verbose_reporter.section_header("DATA LOADING SUMMARY")
        start_time = time.time()

        data_loader_inst = dataLoader.DataLoader(data_dir=data_dir, verbose=verbose)

        if var_names and len(var_names) > 1:
            if merge_config is None:
                merge_config = {'strategy': 'concatenate', 'separator': '; ', 'skip_empty': True}

            merge_strategy = merge_config.get('strategy', 'concatenate')
            separator = merge_config.get('separator', '; ')
            skip_empty = merge_config.get('skip_empty', True)

            verbose_reporter.stat_line(f"Loading multiple variables: {var_names}")
            verbose_reporter.stat_line(f"Merge strategy: {merge_strategy}, separator: '{separator}', skip_empty: {skip_empty}")

            raw_text_df = data_loader_inst.get_multiple_variables_with_IDs(
                filename=filename, id_column=id_column, var_names=var_names,
                merge_strategy=merge_strategy, separator=separator,
                skip_empty=skip_empty, encoding=encoding
            )
            text_column = 'merged_text'
        else:
            if var_names and len(var_names) == 1:
                var_name = var_names[0]
            elif not var_name:
                raise ValueError("Either var_name or var_names must be provided")

            raw_text_df = data_loader_inst.get_variable_with_IDs(
                filename=filename, id_column=id_column,
                var_name=var_name, encoding=encoding
            )
            text_column = var_name

        raw_unstructured = list(zip(
            [int(id_int) for id_int in raw_text_df[id_column].tolist()],
            raw_text_df[text_column].tolist()
        ))

        raw_text_list = []
        for resp_id, resp in raw_unstructured:
            if pd.isna(resp) or resp is None:
                response_type = 'nan'
                response_value = None
            elif isinstance(resp, (int, float)):
                response_type = 'numeric'
                response_value = resp
            elif isinstance(resp, str):
                response_type = 'string'
                response_value = resp
            else:
                response_type = 'unknown'
                response_value = resp
            raw_text_list.append(models.ResponseModel(
                respondent_id=resp_id, response=response_value, response_type=response_type
            ))

        original_count = len(raw_text_list)
        if sample_size and len(raw_text_list) > sample_size:
            raw_text_list = raw_text_list[:sample_size]
            verbose_reporter.stat_line(f"Applied truncation: {len(raw_text_list)} of {original_count} responses (sample size: {sample_size})")
        else:
            verbose_reporter.stat_line(f"No truncation: {len(raw_text_list)} responses (full dataset)")

        end_time = time.time()
        elapsed_time = end_time - start_time
        cache_manager.save_to_cache(raw_text_list, filename, step_name, variable_key, elapsed_time, var_lab=None)

        print("\n=== RAW DATA TYPE ANALYSIS ===")
        type_counts = {'nan': 0, 'numeric': 0, 'string': 0, 'unknown': 0}
        for item in raw_text_list:
            type_counts[item.response_type] += 1
        for data_type, count in type_counts.items():
            print(f"{data_type}: {count} items")
        print(f"\n\n'Import data' completed in {elapsed_time:.2f} seconds.\n")

        if streamlit_container:
            streamlit_container.success(f"Loaded {len(raw_text_list)} responses in {elapsed_time:.2f}s")

    return raw_text_list


def step_1_preprocess(
    raw_text_list,
    filename,
    var_lab,
    variable_key=None,
    cache_manager=None,
    model_config=None,
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None
):
    """Step 1: Preprocess text responses"""
    from utils import textNormalizer, spellChecker, textFinalizer, verboseReporter, promptPrinter
    from config_steps.config_preprocess import SpellCheckConfig

    step_name = "preprocessed"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

    if streamlit_container:
        streamlit_container.text("Preprocessing text responses...")
    spell_check_config = SpellCheckConfig(
        minimum_timeout_seconds=15.0,
        maximum_timeout_seconds=60.0)

    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=prompt_printer_enabled, print_realtime=True)

    code_meanings = {
        99999997: "Don't know (expresses uncertainty)",
        99999998: "No response (empty/NA)",
        99999999: "Meaningless answer (gibberish/irrelevant text)"}

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        preprocessed_text = cache_manager.load_from_cache(filename, step_name, variable_key, models.PreprocessedModel)
        code_counts = {}
        for item in preprocessed_text:
            code = item.quality_filter_code
            if code is not None:
                code_counts[code] = code_counts.get(code, 0) + 1
        verbose_reporter.summary("PREPROCESSED RESPONSES FROM CACHE", {"Input" : f"{len(raw_text_list)} responses"})
        for code, count in code_counts.items():
            verbose_reporter.stat_line(f"{code_meanings.get(code, 'Unknown code')} = {count} responses")
        verbose_reporter.stat_line(f"Output: {len(preprocessed_text) - sum(code_counts.values())}")

        if streamlit_container:
            streamlit_container.success("Preprocessing completed (from cache)")
    else:
        verbose_reporter.section_header("PREPROCESSING PHASE")
        text_normalizer = textNormalizer.TextNormalizer(verbose=verbose)
        spell_checker = spellChecker.SpellChecker(config=spell_check_config, model_config=model_config, verbose=verbose, prompt_printer=prompt_printer)
        text_finalizer = textFinalizer.TextFinalizer(verbose=verbose)
        start_time = time.time()

        all_responses = []
        string_responses = []
        non_string_responses = []
        for item in raw_text_list:
            preprocess_item = item.to_model(models.PreprocessedModel)
            all_responses.append(preprocess_item)
            if item.response_type == 'string':
                string_responses.append(preprocess_item)
            else:
                non_string_responses.append(preprocess_item)
        if string_responses:
            normalized_text = text_normalizer.normalize_responses(string_responses)
            normal_no_missing = [item for item in normalized_text if isinstance(item.response, str) and item.response != '<NA>']
            corrected_text = spell_checker.spell_check(normal_no_missing, var_lab)
            finalized_text = text_finalizer.finalize_responses(corrected_text)
        else:
            finalized_text = []
        processed_map = {item.respondent_id: item for item in finalized_text}
        processed_map.update({item.respondent_id: item for item in non_string_responses})
        preprocessed_text = []
        for original in raw_text_list:
            if original.respondent_id in processed_map:
                item = processed_map[original.respondent_id]
                desc_item = item.to_model(models.PreprocessedModel)
                if item.response == 'nan':
                    desc_item.quality_filter_code = 99999998
                    desc_item.quality_filter = True
                elif isinstance(item.response, int):
                    if item.response in [99999997, 99999998, 99999999]:
                        desc_item.quality_filter_code = int(item.response)
                        desc_item.quality_filter = True
                    else:
                        desc_item.quality_filter_code = None
                        desc_item.quality_filter = None
                elif isinstance(item.response, str):
                    desc_item.quality_filter_code = None
                    desc_item.quality_filter = None
                preprocessed_text.append(desc_item)
            else:
                preprocessed_text.append(models.PreprocessedModel(
                    respondent_id=original.respondent_id,
                    response='<NA>',
                    response_type='nan',
                    quality_filter_code=99999998,
                    quality_filter=True))
        end_time = time.time()
        elapsed_time = end_time - start_time

        cache_manager.save_to_cache(preprocessed_text, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

        if verbose:
            print()
            print("=== QUALITY FILTER CODE SUMMARY ===")
            code_counts = {}
            for item in preprocessed_text:
                code = item.quality_filter_code
                if code is not None:
                    code_counts[code] = code_counts.get(code, 0) + 1
            for code, count in sorted(code_counts.items()):
                meaning = code_meanings.get(code, "Unknown code")
                print(f"Code {code}: {count} items - {meaning}")
            print(f"Total items with codes: {sum(code_counts.values())}")
            print(f"Total items without codes: {len(preprocessed_text) - sum(code_counts.values())}")
            print()

        if verbose:
            print()
            print("[SAMPLES] Sample preprocessing corrections:")
            all_samples = []
            if hasattr(text_normalizer, 'transformation_examples') and text_normalizer.transformation_examples:
                all_samples.extend(text_normalizer.transformation_examples)
            if hasattr(spell_checker, 'correction_examples') and spell_checker.correction_examples:
                all_samples.extend(spell_checker.correction_examples)
            if hasattr(text_finalizer, 'transformation_examples') and text_finalizer.transformation_examples:
                all_samples.extend(text_finalizer.transformation_examples)
            if hasattr(spell_checker, 'correction_examples') and spell_checker.correction_examples:
                import random
                sample = random.choice(spell_checker.correction_examples)
                print(f'  "{sample[0]}" -> "{sample[1]}"')
            elif all_samples:
                import random
                sample = random.choice(all_samples)
                print(f'  "{sample[0]}" -> "{sample[1]}"')
            else:
                print("  No corrections made")
            print()

        print(f"\n'Preprocessing phase' completed in {elapsed_time:.2f} seconds.\n")

        if streamlit_container:
            streamlit_container.success(f"Preprocessing completed in {elapsed_time:.2f}s")

    stats = {}
    if 'text_normalizer' in locals() and hasattr(text_normalizer, 'stats'):
        stats['normalizer_stats'] = text_normalizer.stats
    if 'spell_checker' in locals() and hasattr(spell_checker, 'stats'):
        stats['spellchecker_stats'] = spell_checker.stats
    if 'text_finalizer' in locals() and hasattr(text_finalizer, 'stats'):
        stats['finalizer_stats'] = text_finalizer.stats

    return preprocessed_text, stats


def step_2_quality_filter(
    preprocessed_text,
    filename,
    var_lab,
    variable_key=None,
    cache_manager=None,
    model_config=None,
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None
):
    """Step 2: Filter low-quality responses using LLM-based quality assessment"""
    from utils import qualityFilter, verboseReporter, promptPrinter

    step_name = "quality_filter"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

    if streamlit_container:
        streamlit_container.text("Filtering low-quality responses...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=prompt_printer_enabled, print_realtime=True)

    code_meanings = {
        99999997: "Don't know (expresses uncertainty)",
        99999998: "No response (empty/NA)",
        99999999: "Meaningless answer (gibberish/irrelevant text)"}

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        quality_filtered_text = cache_manager.load_from_cache(filename, step_name, variable_key, models.QualityFilteredModel)
        input_len = len([item.response for item in quality_filtered_text if item.quality_filter_code != 99999998])
        code_counts = {}
        for item in quality_filtered_text:
            code = item.quality_filter_code
            if code is not None:
                code_counts[code] = code_counts.get(code, 0) + 1
        verbose_reporter.summary("QUALIFIED RESPONSES FROM CACHE", {"Input": f"{input_len} responses"})
        for code, count in code_counts.items():
            if code != 99999998:
                verbose_reporter.stat_line(f"{code_meanings.get(code, 'Unknown code')} = {count} responses")
        verbose_reporter.stat_line(f"Output: {len(preprocessed_text) - sum(code_counts.values())}")

        if streamlit_container:
            streamlit_container.success("Quality filtering completed (from cache)")
    else:
        verbose_reporter.section_header("QUALITY FILTERING PHASE")
        start_time = time.time()
        grader = qualityFilter.Grader(preprocessed_text, var_lab, model_config=model_config, verbose=verbose, prompt_printer=prompt_printer)
        quality_filtered_text = grader.grade()
        end_time = time.time()
        elapsed_time = end_time - start_time
        cache_manager.save_to_cache(quality_filtered_text, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

        print("\n=== MISSING CODE SUMMARY ===")
        code_counts = {}
        for item in quality_filtered_text:
            code = item.quality_filter_code
            if code is not None:
                code_counts[code] = code_counts.get(code, 0) + 1
        code_meanings_full = {
            99999997: "User missing: Don't know/only expressing uncertainty",
            99999998: "System missing: NA",
            99999999: "No answer: Empty strings/Single Characters/Only numbers/Nonsensical/gibberish/meaningless content"}
        for code, count in sorted(code_counts.items()):
            meaning = code_meanings_full.get(code, "Unknown code")
            print(f"Code {code}: {count} items - {meaning}")
        print(f"Total items with codes: {sum(code_counts.values())}")
        print(f"Total items without codes: {len(preprocessed_text) - sum(code_counts.values())}\n")
        print(f"\n\n'Quality filtering phase' completed in {elapsed_time:.2f} seconds.\n")

        if streamlit_container:
            streamlit_container.success(f"Quality filtering completed in {elapsed_time:.2f}s")

    return quality_filtered_text


def step_3_extract_ideas(
    quality_filtered_text,
    filename,
    var_lab,
    variable_key=None,
    cache_manager=None,
    model_config=None,
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None
):
    """Step 3: Extract discrete ideas from multi-idea responses"""
    from utils import ideaExtractor, verboseReporter, promptPrinter

    step_name = "extracted_ideas"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

    if streamlit_container:
        streamlit_container.text("Extracting discrete ideas from responses...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=prompt_printer_enabled, print_realtime=True)

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        encoded_text = cache_manager.load_from_cache(filename, step_name, variable_key, models.IdeasExtractedModel)
        segments = sum(item.idea_count for item in encoded_text)
        verbose_reporter.summary("IDEAS EXTRACTED FROM CACHE", {f"Input: {len(encoded_text)} filtered responses -> Output": f"{segments} response segments"})

        if streamlit_container:
            streamlit_container.success("Idea extraction completed (from cache)")
    else:
        verbose_reporter.section_header("EXTRACTION OF IDEAS EXPRESSED PHASE")
        start_time = time.time()
        filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
        verbose_reporter.stat_line(f"Input: {len(quality_filtered_text)} quality-filtered responses")
        verbose_reporter.stat_line(f"Processing: {len(filtered_text)} meaningful responses (excluded {len(quality_filtered_text) - len(filtered_text)} filtered responses)")
        encoder = ideaExtractor.IdeaExtractor(
            responses=filtered_text,
            var_lab=var_lab,
            model_config=model_config,
            verbose=verbose,
            prompt_printer=prompt_printer
        )
        encoded_text = encoder.extract()
        end_time = time.time()
        elapsed_time = end_time - start_time
        cache_manager.save_to_cache(encoded_text, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

        # Build and cache extraction metadata
        if hasattr(encoder, 'build_extraction_metadata'):
            extraction_metadata = encoder.build_extraction_metadata(
                filename=filename,
                var_name=variable_key.split('_')[0] if '_' in variable_key else variable_key
            )
            cache_manager.save_metadata_to_cache(
                metadata=extraction_metadata,
                filename=filename,
                step=step_name,
                variable_key=variable_key,
                processing_time=elapsed_time,
                var_lab=var_lab
            )
            if verbose:
                verbose_reporter.stat_line(f"Cached extraction metadata: dimension={extraction_metadata.primary_dimension}, template='{extraction_metadata.template_prefix}'")

        print(f"\n\n'Idea extraction phase' completed in {elapsed_time:.2f} seconds.\n")

        if streamlit_container:
            streamlit_container.success(f"Idea extraction completed in {elapsed_time:.2f}s")

    return encoded_text


def step_4_classNcoder(
    encoded_text,
    filename,
    var_lab,
    variable_key=None,
    cache_manager=None,
    force_recalc=False,
    verbose=True,
    streamlit_container=None
):
    """Step 4: Category Discovery & Code Assignment (taxonomy-based)

    Uses the qualitative-researcher pipeline (no embeddings needed):
      1. Domain Discovery: partition ideas by domain
      2. Qualitative Researcher: facets → attributes → codes per domain
      3. Code Assignment: assign each idea to exactly one code

    Args:
        encoded_text: List of IdeasExtractedModel instances from step 3
        filename: SPSS filename for caching
        var_lab: Survey question text (for context)
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        tuple: (category_results: List[CodeAssignedModel], mece_results_cache: CodingResultsCache)
    """
    from development.step_4_classNcoder.config_classNcoder_exp import (
        CategoriesConfig, AssignmentConfig,
    )
    from development.step_4_classNcoder.domain_discoverer import DomainDiscoverer
    from development.step_4_classNcoder.qualitative_researcher import QualitativeResearcher
    from development.step_4_classNcoder.code_assignment import CodeAssigner
    from development.step_4_classNcoder.models_exp import (
        DomainSet, DomainResultModel, CodingResultsCache, CodeAssignedModel,
    )
    from utils.verboseReporter import VerboseReporter
    from utils.promptPrinter import PromptPrinter

    step_name = "code_assignment"
    mece_step_name = "mece_categories"
    variable_key, cache_manager, _ = _resolve_step_defaults(variable_key, cache_manager)

    if streamlit_container:
        streamlit_container.text("Discovering categories and assigning codes...")
    verbose_reporter = VerboseReporter(verbose)

    # ─── Cache check ────────────────────────────────────────────────────
    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        category_results = cache_manager.load_from_cache(
            filename, step_name, variable_key, CodeAssignedModel
        )
        total_ideas = sum(
            len(resp.response_ideas)
            for resp in category_results if resp.response_ideas
        )
        assigned_count = sum(
            1 for resp in category_results if resp.response_ideas
            for idea in resp.response_ideas
            if idea.assigned_category
        )
        verbose_reporter.summary("CATEGORY ASSIGNMENT FROM CACHE", {
            "Input": f"{len(encoded_text)} responses",
            "Total ideas": f"{total_ideas}",
            "Assigned": f"{assigned_count}",
        })

        # Load MECE results cache for codebook building
        mece_results_cache = cache_manager.load_metadata_from_cache(
            filename=filename, step=mece_step_name,
            variable_key=variable_key, model_cls=CodingResultsCache,
        )

        if streamlit_container:
            streamlit_container.success(
                f"Category assignment completed (from cache): {assigned_count} ideas assigned"
            )
        return category_results, mece_results_cache

    # ─── Fresh computation ──────────────────────────────────────────────
    verbose_reporter.section_header("CATEGORY DISCOVERY & CODE ASSIGNMENT")
    start_time = time.time()

    # Load extraction metadata
    extraction_metadata = None
    try:
        extraction_metadata = cache_manager.load_metadata_from_cache(
            filename=filename, step="extracted_ideas",
            variable_key=variable_key, model_cls=models.ExtractionMetadata
        )
        if extraction_metadata and verbose:
            print(f"   Loaded extraction metadata "
                  f"(primary_dimension: {getattr(extraction_metadata, 'primary_dimension', 'N/A')})")
    except Exception as e:
        if verbose:
            print(f"   Note: Could not load extraction metadata: {e}")

    categories_config = CategoriesConfig()
    assignment_config = AssignmentConfig()

    # ═══ Stage 1: Domain Discovery ═══════════════════════════════════
    discoverer = DomainDiscoverer(categories_config, extraction_metadata)
    partition_set, label_mappings = discoverer.discover(encoded_text)

    if verbose:
        print(f"\n   Domains discovered: {len(label_mappings)}")
        for name, mapping in label_mappings.items():
            print(f"     {name}: {mapping.label_count} unique labels")

    # ═══ Stage 2: Qualitative Researcher pipeline ════════════════════
    # Build context from extraction metadata
    survey_question = var_lab or ""
    language = "Dutch"
    dataset_context = None
    dimension_name = ""
    dimension_description = ""

    if extraction_metadata:
        meta = extraction_metadata
        if getattr(meta, 'var_lab', ''):
            survey_question = meta.var_lab
        language = getattr(meta, 'lang', 'Dutch') or 'Dutch'
        dataset_context = {}
        for f in ('sector', 'entity', 'topic', 'perspective', 'intent'):
            val = getattr(meta, f, None)
            if val:
                dataset_context[f] = val
        dimension_name = getattr(meta, 'primary_dimension', '') or ''
        dimension_description = getattr(meta, 'primary_dimension_description', '') or ''

    prompt_printer = PromptPrinter(enabled=True, print_realtime=False)
    processor = QualitativeResearcher(categories_config, prompt_printer=prompt_printer)
    pipeline_result = processor.process_all_partitions(
        label_mappings=label_mappings,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=verbose,
    )

    # ═══ Cache pipeline results ══════════════════════════════════════
    # Store global codebook + per-domain results
    codebook = pipeline_result.codebook
    total_labels = sum(m.label_count for m in label_mappings.values())

    pydantic_results = {
        "__global__": DomainResultModel(
            partition_name="__global__",
            n_labels=total_labels,
            n_batches=0,
            categories=codebook,
        )
    }
    for name, result in pipeline_result.partition_results.items():
        pydantic_results[name] = DomainResultModel(
            partition_name=name,
            n_labels=result.n_labels,
            n_batches=result.n_batches,
            categories=[],
            facets=[f.model_dump() for f in result.facets],
            facet_assignments=result.facet_assignments,
            attributes={
                facet_name: [a.model_dump() for a in attrs]
                for facet_name, attrs in result.attributes.items()
            },
        )

    mece_cache = CodingResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={name: m.label_count for name, m in label_mappings.items()},
        label_source=categories_config.label_source,
        total_categories=len(codebook),
        raw_codes=[c.model_dump() for c in pipeline_result.codes],
    )
    cache_manager.save_metadata_to_cache(
        metadata=mece_cache, filename=filename,
        step=mece_step_name, variable_key=variable_key,
    )
    if verbose:
        n_codes = len(codebook)
        total_facets = sum(
            len(r.facets) for r in pipeline_result.partition_results.values()
        )
        print(f"\n   Pipeline results cached "
              f"({n_codes} codes, {total_facets} facets across "
              f"{len(pipeline_result.partition_results)} domains)")

    # ═══ Stage 3: Code Assignment ════════════════════════════════════
    assigner = CodeAssigner(
        config=assignment_config,
        ideas_models=encoded_text,
        mece_results=pydantic_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
        prompt_printer=prompt_printer,
        codes=pipeline_result.codes,
    )
    category_results = assigner.assign_all()

    end_time = time.time()
    elapsed_time = end_time - start_time

    cache_manager.save_to_cache(
        category_results, filename, step_name, variable_key,
        elapsed_time, var_lab=var_lab
    )

    total_ideas = sum(
        len(resp.response_ideas)
        for resp in category_results if resp.response_ideas
    )
    assigned_count = sum(
        1 for resp in category_results if resp.response_ideas
        for idea in resp.response_ideas
        if idea.assigned_category
    )

    if verbose:
        print(f"\n   Category discovery & code assignment completed in "
              f"{elapsed_time:.2f} seconds.")
        print(f"   Total ideas: {total_ideas}")
        print(f"   Assigned: {assigned_count}")
        print(f"   Domains: {len(pipeline_result.partition_results)}")
        print(f"   Codes: {len(codebook)}")

    if streamlit_container:
        streamlit_container.success(
            f"Code assignment completed in {elapsed_time:.2f}s: "
            f"{assigned_count}/{total_ideas} ideas assigned"
        )

    return category_results, mece_cache


def step_5_export_results(
    category_results,
    theme_enriched_codebook,
    filename,
    var_name,
    quality_filtered_text=None,
    verbose=True,
    streamlit_container=None,
):
    """Step 5: Export taxonomy results to Excel

    Args:
        category_results: List of CodeAssignedModel instances from step 4
        theme_enriched_codebook: ThemeEnrichedCodebookModel built from MECE categories
        filename: SPSS filename for export naming
        var_name: Variable name for export naming
        quality_filtered_text: List of QualityFilteredModel from step 2 (includes filtered responses)
        verbose: Enable verbose output
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        str: Path to exported Excel file
    """
    from utils.resultsExporter import ResultsExporter

    if streamlit_container:
        streamlit_container.text("Exporting results to Excel...")

    try:
        exporter = ResultsExporter(verbose=verbose)
        excel_path = exporter.export_to_excel(
            category_results,
            theme_enriched_codebook,
            filename,
            var_name,
            quality_filtered_text=quality_filtered_text,
            export_dir=None,
            include_visualizations=False,
        )
        print(f"[SUCCESS] Results exported to Excel: {excel_path}")

        if streamlit_container:
            streamlit_container.success(f"Results exported to Excel: {excel_path}")

        return excel_path
    except Exception as e:
        print(f"[WARNING] Excel export failed: {str(e)}")
        if streamlit_container:
            streamlit_container.error(f"Excel export failed: {str(e)}")
        return None



if __name__ == '__main__':
    import sys

    if RUN_UNTIL_STEP is not None and not FORCE_RECALCULATE_ALL:
        FORCE_STEP = STEP_NAMES.get(RUN_UNTIL_STEP, "")
    else:
        FORCE_STEP = ""

    data_loader = dataLoader.DataLoader(verbose=False)
    var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)

    # Start capturing all verbose output to file
    from utils.saveVerbose import VerboseCapture
    verbose_capture = VerboseCapture(
        filename=filename,
        variable_key=var_name,
        sample_size=sample_size,
        run_until_step=RUN_UNTIL_STEP if RUN_UNTIL_STEP is not None else 5
    )
    verbose_capture.__enter__()

    # Reset token tracker at start of pipeline run
    token_tracker.reset()

    print("=" * 80)
    print("CODERINGSTOOL TAXONOMY PIPELINE")
    print("=" * 80)
    print(f"Data file: {filename}")
    print(f"Variable: {var_name} - {var_lab}")
    print(f"Sample size: {sample_size if sample_size else 'All responses'}")
    print(f"Run until step: {RUN_UNTIL_STEP if RUN_UNTIL_STEP is not None else 'All (0-5)'}")
    print(f"Force recalculate: {'ALL STEPS' if FORCE_RECALCULATE_ALL else (f'Step {RUN_UNTIL_STEP} ({FORCE_STEP})' if FORCE_STEP else 'None')}")
    print(f"Verbose mode: {VERBOSE}")
    print(f"Prompt printer: {PROMPT_PRINTER}")
    print("=" * 80)

    selected_variables = globals().get('selected_variables', [var_name])
    is_merged = globals().get('is_merged', False)
    test_mode = globals().get('is_test_mode', True)
    sample_size = globals().get('test_sample_size', sample_size) if test_mode else None

    if 'variable_key' in globals():
        variable_key = globals()['variable_key']
    else:
        from utils.cacheManager import generate_enhanced_variable_key
        merge_config = globals().get('merge_config', None)
        variable_key = generate_enhanced_variable_key(
            selected_variables, is_merged,
            sample_size=sample_size, merge_config=merge_config)

    def check_execution_stop(current_step: int):
        """Check if execution should stop after current step"""
        if RUN_UNTIL_STEP is not None and current_step >= RUN_UNTIL_STEP:
            print(f"\n{'='*80}")
            print(f"EXECUTION STOPPED: RUN_UNTIL_STEP set to {RUN_UNTIL_STEP}")
            print(f"Completed steps 0-{current_step}")
            print(f"{'='*80}\n")
            if token_tracker.call_count > 0:
                print(token_tracker.get_summary())
            verbose_capture.__exit__(None, None, None)
            sys.exit(0)

    # === STEP 0: Load data ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "data"
    raw_text_list = step_0_load_data(
        filename, id_column, var_name,
        sample_size=sample_size,
        variable_key=variable_key,
        cache_manager=cache_manager,
        force_recalc=force_recalc,
        verbose=VERBOSE
    )
    check_execution_stop(0)

    # === STEP 1: Preprocess ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "preprocessed"
    preprocessed_text, stats = step_1_preprocess(
        raw_text_list, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        model_config=model_config,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(1)

    # === STEP 2: Quality filter ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "quality_filter"
    quality_filtered_text = step_2_quality_filter(
        preprocessed_text, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        model_config=model_config,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(2)

    # === STEP 3: Extract ideas ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "extracted_ideas"
    encoded_text = step_3_extract_ideas(
        quality_filtered_text, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        model_config=model_config,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(3)

    # === STEP 4: Category Discovery & Code Assignment (classNcoder) ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "code_assignment"
    category_results, mece_results_cache = step_4_classNcoder(
        encoded_text, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        force_recalc=force_recalc,
        verbose=VERBOSE,
    )
    check_execution_stop(4)

    # Build codebook from MECE categories (no step 6-8 needed)
    theme_enriched_codebook = _build_codebook_from_mece(
        mece_results_cache, category_results, var_name
    )

    if VERBOSE and theme_enriched_codebook.codes:
        print(f"\nTaxonomy codebook: {len(theme_enriched_codebook.codes)} codes in {len(theme_enriched_codebook.themes_summary)} partitions")
        for idx, entry in enumerate(theme_enriched_codebook.codes, start=1):
            print(f"  {idx}) [{entry.theme}] {entry.code}")

    # === STEP 5: Export Results ====
    excel_path = step_5_export_results(
        category_results,
        theme_enriched_codebook,
        filename,
        var_name,
        quality_filtered_text=quality_filtered_text,
        verbose=VERBOSE,
    )

    # Pipeline completed
    print(f"\n{'='*80}")
    print("TAXONOMY PIPELINE COMPLETED SUCCESSFULLY")
    print("All steps (0-5) executed")
    print(f"Results exported to: {excel_path}")
    print(f"{'='*80}\n")

    if token_tracker.call_count > 0:
        print(token_tracker.get_summary())

    verbose_capture.__exit__(None, None, None)

# %%
