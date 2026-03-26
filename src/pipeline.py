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
import numpy as np

# Suppress IPython exit warning when running in Streamlit
warnings.filterwarnings("ignore", message="To exit: use 'exit', 'quit', or Ctrl-D.")
import nest_asyncio
nest_asyncio.apply()

import models
from utils import dataLoader
from utils.cacheManager import CacheManager
from utils.llm import token_tracker
from config import CacheConfig, DEFAULT_LANGUAGE
cache_config = CacheConfig()
cache_manager = CacheManager(cache_config)

#  ===  STANDALONE ========================================================================================================

#filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
#id_column = "DLNMID"
#var_name = "Q20"
#sample_size = 500

filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
id_column = "DLNMID"
var_name = "Q20"
sample_size = 500

#filename = "M000000 Associatiemonitor Merk X net databestand.sav"
#id_column = "DLNMID"
#var_name = "Qd1_combined"
#sample_size = 2000 

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
INCLUDE_VISUALIZATIONS = True  # Add dendrogram, word clouds, and network graph to export

STEP_NAMES = {
    0: "data",
    1: "preprocessed",
    2: "quality_filter",
    3: "extracted_ideas",
    4: "taxonomy",
    5: "codebook",
    6: "code_assignment",
    7: "export",
}

# ===================================================================================================================
# HELPERS
# ===================================================================================================================

def _resolve_step_defaults(variable_key=None, cache_manager=None):
    """Resolve default values for step function parameters.

    Each parameter is resolved only if passed as None:
    - variable_key: generated from module-level globals (selected_variables, etc.)
    - cache_manager: falls back to module-level global, then creates default

    Returns: (variable_key, cache_manager)
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

    return variable_key, cache_manager


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
    """Step 0: Load data from SPSS file (single or multiple variables)

    Args:
        filename: SPSS filename to load
        id_column: Column name containing respondent IDs
        var_name: Single variable name to extract (or None if using var_names)
        var_names: List of variable names for merged loading (or None if using var_name)
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        sample_size: Limit to N responses (None = all)
        encoding: SPSS file encoding (None = auto-detect)
        merge_config: Dict with 'strategy', 'separator', 'skip_empty' for multiple vars
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.ResponseModel]: List of response models
    """
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

    # Optional Streamlit progress
    if streamlit_container:
        if var_names and len(var_names) > 1:
            streamlit_container.text(f"🔄 Loading and merging {len(var_names)} variables...")
        else:
            streamlit_container.text("🔄 Loading data from SPSS file...")

    # Check cache
    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        raw_text_list = cache_manager.load_from_cache(filename, step_name, variable_key, models.ResponseModel)
        verbose_reporter.summary("DATA FROM CACHE", {"Input": f"{len(raw_text_list)} responses"})
    else:
        verbose_reporter.section_header("DATA LOADING SUMMARY")
        start_time = time.time()

        # Determine loading mode: single or multiple variables
        data_loader_inst = dataLoader.DataLoader(data_dir=data_dir, verbose=verbose)

        if var_names and len(var_names) > 1:
            # Multiple variables mode - merge them
            if merge_config is None:
                merge_config = {'strategy': 'concatenate', 'separator': '; ', 'skip_empty': True}

            merge_strategy = merge_config.get('strategy', 'concatenate')
            separator = merge_config.get('separator', '; ')
            skip_empty = merge_config.get('skip_empty', True)

            verbose_reporter.stat_line(f"Loading multiple variables: {var_names}")
            verbose_reporter.stat_line(f"Merge strategy: {merge_strategy}, separator: '{separator}', skip_empty: {skip_empty}")

            raw_text_df = data_loader_inst.get_multiple_variables_with_IDs(
                filename=filename,
                id_column=id_column,
                var_names=var_names,
                merge_strategy=merge_strategy,
                separator=separator,
                skip_empty=skip_empty,
                encoding=encoding
            )
            text_column = 'merged_text'
        else:
            # Single variable mode
            if var_names and len(var_names) == 1:
                var_name = var_names[0]
            elif not var_name:
                raise ValueError("Either var_name or var_names must be provided")

            raw_text_df = data_loader_inst.get_variable_with_IDs(
                filename=filename,
                id_column=id_column,
                var_name=var_name,
                encoding=encoding
            )
            text_column = var_name

        # Extract data from dataframe
        raw_unstructured = list(zip(
            [int(id_int) for id_int in raw_text_df[id_column].tolist()],
            raw_text_df[text_column].tolist()
        ))

        raw_text_list = []
        # Structure data: NaN=system missing; Numeric=user missing; String=response
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
                respondent_id=resp_id,
                response=response_value,
                response_type=response_type
            ))

        # Apply sample size truncation if specified
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

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Loaded {len(raw_text_list)} responses in {elapsed_time:.2f}s")

    return raw_text_list


def step_1_preprocess(
    raw_text_list,
    filename,
    var_lab,
    variable_key=None,              # Auto-generate if None
    cache_manager=None,             # Use global if None
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None        # Optional progress updates
):
    """Step 1: Preprocess text responses

    Args:
        raw_text_list: List of ResponseModel instances from step 0
        filename: SPSS filename for caching
        var_lab: Variable label for context
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        prompt_printer_enabled: Enable prompt printing
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.PreprocessedModel]: List of preprocessed response models
    """
    from utils import textNormalizer, spellChecker, textFinalizer, verboseReporter, promptPrinter
    from config_steps.config_preprocess import SpellCheckConfig

    step_name = "preprocessed"
    variable_key, cache_manager = _resolve_step_defaults(variable_key, cache_manager)

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Preprocessing text responses...")
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
        verbose_reporter.summary("PREPROCESSED RESPONSES FROM CACHE", {"• Input" : f"{len(raw_text_list)} responses"})
        for code, count in code_counts.items():
            verbose_reporter.stat_line(f"{code_meanings.get(code, 'Unknown code')} = {count} responses")
        verbose_reporter.stat_line(f"Output: {len(preprocessed_text) - sum(code_counts.values())}")

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success("✅ Preprocessing completed (from cache)")
    else:
        verbose_reporter.section_header("PREPROCESSING PHASE")
        # intialize utils
        text_normalizer = textNormalizer.TextNormalizer(verbose=verbose)
        spell_checker = spellChecker.SpellChecker(config=spell_check_config, verbose=verbose, prompt_printer=prompt_printer)
        text_finalizer = textFinalizer.TextFinalizer(verbose=verbose)
        start_time = time.time()
        # preprocess strings
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
                    desc_item.quality_filter_code = 99999998  # System missing
                    desc_item.quality_filter = True
                elif isinstance(item.response, int):
                    # Check if it's a known missing code
                    if item.response in [99999997, 99999998, 99999999]:
                        desc_item.quality_filter_code = int(item.response)
                        desc_item.quality_filter = True
                    else:
                        # Regular numeric response - will be evaluated by qualityFilter
                        desc_item.quality_filter_code = None
                        desc_item.quality_filter = None
                elif isinstance(item.response, str):
                    # Text response - will be evaluated by qualityFilter
                    # Note: Empty strings are converted to '<NA>' by textNormalizer and handled in the else clause below
                    desc_item.quality_filter_code = None
                    desc_item.quality_filter = None
                preprocessed_text.append(desc_item)
            else:
                preprocessed_text.append(models.PreprocessedModel(
                    respondent_id=original.respondent_id,
                    response='<NA>',
                    response_type='nan',
                    quality_filter_code=99999998,  # No response (empty/NA)
                    quality_filter=True))
        end_time = time.time()
        elapsed_time = end_time - start_time

        cache_manager.save_to_cache(preprocessed_text, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

        # Quality filter summary
        if verbose:
            print()  # Empty line
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
            print()  # Empty line

        # Show consolidated sample corrections from all preprocessing steps
        if verbose:
            print()
            print("[SAMPLES] Sample preprocessing corrections:")

            # Collect samples from all processing steps
            all_samples = []

            # From text normalizer
            if hasattr(text_normalizer, 'transformation_examples') and text_normalizer.transformation_examples:
                all_samples.extend(text_normalizer.transformation_examples)

            # From spell checker (most important for user)
            if hasattr(spell_checker, 'correction_examples') and spell_checker.correction_examples:
                all_samples.extend(spell_checker.correction_examples)

            # From text finalizer
            if hasattr(text_finalizer, 'transformation_examples') and text_finalizer.transformation_examples:
                all_samples.extend(text_finalizer.transformation_examples)

            # Show one random sample from spell checker (most relevant) if available
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

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Preprocessing completed in {elapsed_time:.2f}s")

    # Collect stats from preprocessing utilities
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
    variable_key=None,              # Auto-generate if None
    cache_manager=None,             # Use global if None
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None        # Optional progress updates
):
    """Step 2: Filter low-quality responses using LLM-based quality assessment

    Args:
        preprocessed_text: List of PreprocessedModel instances from step 1
        filename: SPSS filename for caching
        var_lab: Variable label for context
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        prompt_printer_enabled: Enable prompt printing
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.QualityFilteredModel]: List of quality-filtered response models
    """
    from utils import qualityFilter, verboseReporter, promptPrinter

    step_name = "quality_filter"
    variable_key, cache_manager = _resolve_step_defaults(variable_key, cache_manager)

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Filtering low-quality responses...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=prompt_printer_enabled, print_realtime=True)

    code_meanings = {
        99999997: "Don't know (expresses uncertainty)",
        99999998: "No response (empty/NA)",
        99999999: "Meaningless answer (gibberish/irrelevant text)"}

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        quality_filtered_text = cache_manager.load_from_cache(filename, step_name, variable_key, models.QualityFilteredModel)
        input_len = len([item.response for item in quality_filtered_text if item.quality_filter_code != 99999998])
        #filtered_len = len([item.quality_filter for item in quality_filtered_text if item.quality_filter and item.quality_filter_code != 99999998])
        code_counts = {}
        for item in quality_filtered_text:
            code = item.quality_filter_code
            if code is not None:
                code_counts[code] = code_counts.get(code, 0) + 1
        verbose_reporter.summary("QUALIFIED RESPONESES FROM CACHE", {"• Input": f"{input_len} responses"})
        for code, count in code_counts.items():
            if code != 99999998:
                verbose_reporter.stat_line(f"{code_meanings.get(code, 'Unknown code')} = {count} responses")
        verbose_reporter.stat_line(f"Output: {len(preprocessed_text) - sum(code_counts.values())}")

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success("✅ Quality filtering completed (from cache)")
    else:
        verbose_reporter.section_header("QUALITY FILTERING PHASE")
        start_time = time.time()
        grader = qualityFilter.Grader(preprocessed_text, var_lab, verbose=verbose, prompt_printer=prompt_printer)
        quality_filtered_text = grader.grade()
        #grading_summary = grader.summary()
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

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Quality filtering completed in {elapsed_time:.2f}s")

    return quality_filtered_text


def step_3_extract_ideas(
    quality_filtered_text,
    filename,
    var_lab,
    variable_key=None,              # Auto-generate if None
    cache_manager=None,             # Use global if None
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None        # Optional progress updates
):
    """Step 3: Extract discrete ideas from multi-idea responses

    Args:
        quality_filtered_text: List of QualityFilteredModel instances from step 2
        filename: SPSS filename for caching
        var_lab: Variable label for context
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        prompt_printer_enabled: Enable prompt printing
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.IdeasExtractedModel]: List of models with extracted ideas
    """
    from utils import ideaExtractor, verboseReporter, promptPrinter

    step_name = "extracted_ideas"
    variable_key, cache_manager = _resolve_step_defaults(variable_key, cache_manager)

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Extracting discrete ideas from responses...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=prompt_printer_enabled, print_realtime=True)

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        encoded_text = cache_manager.load_from_cache(filename, step_name, variable_key, models.IdeasExtractedModel)
        segments = sum(item.idea_count for item in encoded_text)
        verbose_reporter.summary("IDEAS EXPRESSED AND EXTRACTED FROM RESPONSES IN CACHE", {f"Input: {len(encoded_text)} filtered responses -> Output": f"{segments} response segments"})

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success("✅ Idea extraction completed (from cache)")
    else:
        verbose_reporter.section_header("EXTRACTION OF IDEAS EXPRESSED PHASE")
        start_time = time.time()
        filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
        verbose_reporter.stat_line(f"Input: {len(quality_filtered_text)} quality-filtered responses")
        verbose_reporter.stat_line(f"Processing: {len(filtered_text)} meaningful responses (excluded {len(quality_filtered_text) - len(filtered_text)} filtered responses)")
        encoder = ideaExtractor.IdeaExtractor(
            responses=filtered_text,
            var_lab=var_lab,
            verbose=verbose,
            prompt_printer=prompt_printer
        )
        encoded_text = encoder.extract()
        end_time = time.time()
        elapsed_time = end_time - start_time
        cache_manager.save_to_cache(encoded_text, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

        # Build and cache extraction metadata (taxonomy, template prefix, context specifiers)
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

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Idea extraction completed in {elapsed_time:.2f}s")

    return encoded_text


def step_4_classify_taxonomy(
    encoded_text,
    filename,
    var_lab,
    variable_key=None,
    cache_manager=None,
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None,
):
    """Step 4: Taxonomy classification (P1-P7)

    Discovers domain partitions from step 3 ideas, then runs the full taxonomy
    pipeline: facet discovery → consolidation → assignment → attribute discovery
    → consolidation → assignment → cross-facet dedup.

    Returns:
        None (results cached as metadata for step 5)
    """
    from utils.domain_discoverer import DomainDiscoverer
    from utils.classifier import TaxonomyClassifier, TaxonomyResult
    from utils.promptPrinter import PromptPrinter
    from config_steps.config_classifier import CategoriesConfig
    from models import DomainSet, DomainResultModel, TaxonomyResultsCache, ExtractionMetadata

    step_name = "taxonomy"
    variable_key, cache_manager, _ = _resolve_step_defaults(variable_key, cache_manager)

    if streamlit_container:
        streamlit_container.text("🔄 Running taxonomy classification (P1-P7)...")

    # Check cache
    if not force_recalc and cache_manager.is_cache_valid(filename, f"{step_name}_metadata", variable_key):
        taxonomy_cache = cache_manager.load_metadata_from_cache(filename, step_name, variable_key, TaxonomyResultsCache)
        if taxonomy_cache:
            n_domains = len(taxonomy_cache.partition_results)
            n_facets = sum(len(r.facets) for r in taxonomy_cache.partition_results.values())
            print(f"\n=== TAXONOMY FROM CACHE ({n_domains} domains, {n_facets} facets) ===\n")
            if streamlit_container:
                streamlit_container.success("✅ Taxonomy classification completed (from cache)")
            return

    # Load extraction metadata from step 3
    extraction_metadata = cache_manager.load_metadata_from_cache(
        filename, "extracted_ideas", variable_key, ExtractionMetadata
    )

    # Extract survey context
    survey_question = ""
    language = "Dutch"
    dataset_context = None
    dimension_name = ""
    dimension_description = ""
    if extraction_metadata:
        survey_question = getattr(extraction_metadata, 'var_lab', '') or ''
        language = getattr(extraction_metadata, 'lang', 'Dutch') or 'Dutch'
        dataset_context = {}
        for f in ('sector', 'entity', 'topic', 'perspective', 'intent'):
            val = getattr(extraction_metadata, f, None)
            if val:
                dataset_context[f] = val
        dimension_name = getattr(extraction_metadata, 'primary_dimension', '') or ''
        dimension_description = getattr(extraction_metadata, 'primary_dimension_description', '') or ''

    # Config
    config = CategoriesConfig(
        label_source="idea",
        label_prefix="",
        include_valence=True,
    )

    start_time = time.time()

    # Stage 1: Partition discovery
    discoverer = DomainDiscoverer(config, extraction_metadata)
    partition_set, label_mappings = discoverer.discover(encoded_text)

    # Stage 2: Taxonomy classification (P1-P7)
    prompt_printer = PromptPrinter(enabled=prompt_printer_enabled, print_realtime=prompt_printer_enabled)
    classifier = TaxonomyClassifier(config, prompt_printer=prompt_printer)
    taxonomy_result = classifier.process(
        label_mappings=label_mappings,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=verbose,
    )

    elapsed_time = time.time() - start_time

    # Build per-domain pydantic results and cache
    pydantic_results = {}
    for name in taxonomy_result.partition_facets:
        facet_assigns = {
            k: v for k, v in taxonomy_result.partition_assignments.get(name, {}).items()
            if v is not None
        }
        domain_facet_ids = set(facet_assigns.keys())
        domain_attr_assigns = {
            iid: aname for iid, aname in taxonomy_result.attribute_assignments.items()
            if iid in domain_facet_ids and aname is not None
        }
        pydantic_results[name] = DomainResultModel(
            partition_name=name,
            n_labels=taxonomy_result.partition_n_labels.get(name, 0),
            n_batches=taxonomy_result.partition_n_batches.get(name, 0),
            facets=[f.model_dump() for f in taxonomy_result.partition_facets.get(name, [])],
            facet_assignments=facet_assigns,
            attributes={
                facet_name: [a.model_dump() for a in attrs]
                for facet_name, attrs in taxonomy_result.partition_attributes.get(name, {}).items()
            },
            attribute_assignments=domain_attr_assigns,
        )

    taxonomy_cache = TaxonomyResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={name: m.label_count for name, m in label_mappings.items()},
        label_source=config.label_source,
    )
    cache_manager.save_metadata_to_cache(
        metadata=taxonomy_cache,
        filename=filename,
        step=step_name,
        variable_key=variable_key,
    )

    total_facets = sum(len(taxonomy_result.partition_facets.get(n, [])) for n in taxonomy_result.partition_facets)
    total_attrs = sum(len(a) for fa in taxonomy_result.partition_attributes.values() for a in fa.values())
    print(f"\n'Taxonomy classification' completed in {elapsed_time:.2f} seconds "
          f"({len(pydantic_results)} domains, {total_facets} facets, {total_attrs} attributes).\n")

    if streamlit_container:
        streamlit_container.success(f"✅ Taxonomy classification completed in {elapsed_time:.2f}s")



def step_5_generate_codebook(
    filename,
    variable_key=None,
    cache_manager=None,
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None,
):
    """Step 5: Generate codebook from taxonomy (P8-P9)

    Loads taxonomy results from step 4 cache, reconstructs TaxonomyResult,
    and runs codebook generation (code generation per domain + cross-domain
    consolidation).

    Returns:
        None (results cached as metadata for step 6)
    """
    from utils.codeGenerator import CodebookGenerator, CodebookResult, TaxonomyResult
    from utils.promptPrinter import PromptPrinter
    from config_steps.config_codeGenerator import CodebookConfig
    from prompts_steps.prompts_classifier import DiscoveredFacet, DiscoveredAttribute
    from models import DomainSet, DomainResultModel, TaxonomyResultsCache, CodingResultsCache, ExtractionMetadata

    step_name = "mece_codes"
    variable_key, cache_manager, _ = _resolve_step_defaults(variable_key, cache_manager)

    if streamlit_container:
        streamlit_container.text("🔄 Generating codebook (P8-P9)...")

    # Check cache
    if not force_recalc and cache_manager.is_cache_valid(filename, f"{step_name}_metadata", variable_key):
        mece_cache = cache_manager.load_metadata_from_cache(filename, step_name, variable_key, CodingResultsCache)
        if mece_cache:
            n_codes = mece_cache.total_categories
            print(f"\n=== CODEBOOK FROM CACHE ({n_codes} codes) ===\n")
            if streamlit_container:
                streamlit_container.success("✅ Codebook generation completed (from cache)")
            return

    # Load taxonomy from step 4 cache
    taxonomy_cache = cache_manager.load_metadata_from_cache(
        filename, "taxonomy", variable_key, TaxonomyResultsCache
    )
    if taxonomy_cache is None:
        raise FileNotFoundError("No cached taxonomy results. Run step 4 first.")

    # Load extraction metadata from step 3
    extraction_metadata = cache_manager.load_metadata_from_cache(
        filename, "extracted_ideas", variable_key, ExtractionMetadata
    )

    # Extract survey context
    survey_question = ""
    language = "Dutch"
    dataset_context = None
    dimension_name = ""
    dimension_description = ""
    if extraction_metadata:
        survey_question = getattr(extraction_metadata, 'var_lab', '') or ''
        language = getattr(extraction_metadata, 'lang', 'Dutch') or 'Dutch'
        dataset_context = {}
        for f in ('sector', 'entity', 'topic', 'perspective', 'intent'):
            val = getattr(extraction_metadata, f, None)
            if val:
                dataset_context[f] = val
        dimension_name = getattr(extraction_metadata, 'primary_dimension', '') or ''
        dimension_description = getattr(extraction_metadata, 'primary_dimension_description', '') or ''

    partition_set = taxonomy_cache.partition_set
    pydantic_results = taxonomy_cache.partition_results

    # Reconstruct TaxonomyResult from cached partition_results
    partition_facets = {}
    partition_assignments = {}
    partition_attributes = {}
    partition_n_labels = {}
    partition_n_batches = {}
    all_attr_assignments = {}

    for name, result in pydantic_results.items():
        partition_facets[name] = [DiscoveredFacet(**f) for f in result.facets]
        partition_assignments[name] = result.facet_assignments
        partition_attributes[name] = {
            facet_name: [DiscoveredAttribute(**a) for a in attrs]
            for facet_name, attrs in result.attributes.items()
        }
        partition_n_labels[name] = result.n_labels
        partition_n_batches[name] = result.n_batches
        all_attr_assignments.update(result.attribute_assignments)

    taxonomy_result = TaxonomyResult(
        partition_n_labels=partition_n_labels,
        partition_n_batches=partition_n_batches,
        partition_facets=partition_facets,
        partition_assignments=partition_assignments,
        partition_attributes=partition_attributes,
        attribute_assignments=all_attr_assignments,
    )

    config = CodebookConfig()
    start_time = time.time()

    prompt_printer = PromptPrinter(enabled=prompt_printer_enabled, print_realtime=prompt_printer_enabled)
    generator = CodebookGenerator(config, prompt_printer=prompt_printer)
    codebook_result = generator.generate(
        taxonomy_result=taxonomy_result,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        verbose=verbose,
    )

    elapsed_time = time.time() - start_time

    # Cache codebook for step 6
    mece_cache = CodingResultsCache(
        partition_set=partition_set,
        partition_results=pydantic_results,
        label_counts={name: r.n_labels for name, r in pydantic_results.items()},
        total_categories=len(codebook_result.codes),
        raw_codes=[c.model_dump() for c in codebook_result.codes],
    )
    cache_manager.save_metadata_to_cache(
        metadata=mece_cache,
        filename=filename,
        step=step_name,
        variable_key=variable_key,
    )

    print(f"\n'Codebook generation' completed in {elapsed_time:.2f} seconds "
          f"({len(codebook_result.codes)} codes).\n")

    if streamlit_container:
        streamlit_container.success(f"✅ Codebook generation completed in {elapsed_time:.2f}s")


def step_6_assign_codes(
    encoded_text,
    filename,
    variable_key=None,
    cache_manager=None,
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None,
):
    """Step 6: Assign codes to ideas (P10)

    Loads codebook from step 5 cache and ideas from step 3, then assigns
    each idea to exactly one MECE code via LLM.

    Returns:
        List[models.CodeAssignedModel]: Ideas with assigned codes
    """
    from utils.codeAssigner import CodeAssigner
    from utils.promptPrinter import PromptPrinter
    from config_steps.config_codeAssigner import AssignmentConfig
    from prompts_steps.prompts_codeGenerator import ConsolidatedCode
    from models import CodingResultsCache, DomainSet, DomainResultModel, ExtractionMetadata

    step_name = "taxonomy_codes"
    variable_key, cache_manager, _ = _resolve_step_defaults(variable_key, cache_manager)

    if streamlit_container:
        streamlit_container.text("🔄 Assigning codes to ideas (P10)...")

    # Check cache
    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        assigned_results = cache_manager.load_from_cache(
            filename, step_name, variable_key, models.CodeAssignedModel
        )
        if assigned_results:
            total_ideas = sum(len(r.response_ideas or []) for r in assigned_results)
            print(f"\n=== CODE ASSIGNMENT FROM CACHE ({len(assigned_results)} responses, {total_ideas} ideas) ===\n")
            if streamlit_container:
                streamlit_container.success("✅ Code assignment completed (from cache)")
            return assigned_results

    # Load codebook from step 5 cache
    mece_cache = cache_manager.load_metadata_from_cache(
        filename, "mece_codes", variable_key, CodingResultsCache
    )
    if mece_cache is None:
        raise FileNotFoundError("No cached codebook results. Run step 5 first.")

    # Load extraction metadata from step 3
    extraction_metadata = cache_manager.load_metadata_from_cache(
        filename, "extracted_ideas", variable_key, ExtractionMetadata
    )

    partition_set = mece_cache.partition_set
    pydantic_results = mece_cache.partition_results

    # Reconstruct ConsolidatedCode from cached dicts
    codes = [ConsolidatedCode(**d) for d in mece_cache.raw_codes] if mece_cache.raw_codes else None

    # Collect attribute_assignments from all domains
    all_attr_assignments = {}
    for domain_result in pydantic_results.values():
        all_attr_assignments.update(domain_result.attribute_assignments)

    config = AssignmentConfig()
    start_time = time.time()

    prompt_printer = PromptPrinter(enabled=prompt_printer_enabled, print_realtime=prompt_printer_enabled)
    assigner = CodeAssigner(
        config=config,
        ideas_models=encoded_text,
        mece_results=pydantic_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
        prompt_printer=prompt_printer,
        codes=codes,
        attribute_assignments=all_attr_assignments,
    )
    assigned_results = assigner.assign_all()

    elapsed_time = time.time() - start_time

    # Cache results
    cache_manager.save_to_cache(assigned_results, filename, step_name, variable_key)

    total_ideas = sum(len(r.response_ideas or []) for r in assigned_results)
    assigned_count = sum(
        1 for r in assigned_results
        for idea in (r.response_ideas or [])
        if idea.assigned_code
    )
    print(f"\n'Code assignment' completed in {elapsed_time:.2f} seconds "
          f"({assigned_count}/{total_ideas} ideas assigned).\n")

    if streamlit_container:
        streamlit_container.success(f"✅ Code assignment completed in {elapsed_time:.2f}s")

    return assigned_results


def step_7_export(
    assigned_results=None,
    filename=None,
    var_name=None,
    variable_key=None,
    cache_manager=None,
    verbose=True,
    streamlit_container=None,
):
    """Step 7: Export results (PLACEHOLDER)

    This step is not yet implemented. It will export coded results to Excel.
    """
    print("\n=== EXPORT STEP NOT YET IMPLEMENTED ===")
    print("Step 7 (export) will be developed as part of Job 3.\n")

    if streamlit_container:
        streamlit_container.info("Export step not yet implemented")


# ===================================================================================================================
# MAIN EXECUTION
# ===================================================================================================================

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
        run_until_step=RUN_UNTIL_STEP if RUN_UNTIL_STEP is not None else 7
    )
    verbose_capture.__enter__()

    # Reset token tracker at start of pipeline run
    token_tracker.reset()

    print("=" * 80)
    print("CODERINGSTOOL PIPELINE")
    print("=" * 80)
    print(f"Data file: {filename}")
    print(f"Variable: {var_name} - {var_lab}")
    print(f"Sample size: {sample_size if sample_size else 'All responses'}")
    print(f"Run until step: {RUN_UNTIL_STEP if RUN_UNTIL_STEP is not None else 'All (0-7)'}")
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
            selected_variables,
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config)

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

    # === STEP 0 ====
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

    # === STEP 1 ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "preprocessed"
    preprocessed_text, stats = step_1_preprocess(
        raw_text_list, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,

        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(1)

    # === STEP 2 ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "quality_filter"
    quality_filtered_text = step_2_quality_filter(
        preprocessed_text, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,

        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(2)

    # === STEP 3 ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "extracted_ideas"
    encoded_text = step_3_extract_ideas(
        quality_filtered_text, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,

        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(3)

    # === STEP 4 ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "taxonomy"
    step_4_classify_taxonomy(
        encoded_text, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(4)

    # === STEP 5 ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "codebook"
    step_5_generate_codebook(
        filename,
        variable_key=variable_key,
        cache_manager=cache_manager,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(5)

    # === STEP 6 ====
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "code_assignment"
    code_assigned_results = step_6_assign_codes(
        encoded_text, filename,
        variable_key=variable_key,
        cache_manager=cache_manager,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(6)

    # === STEP 7 ====
    step_7_export(
        assigned_results=code_assigned_results,
        filename=filename,
        var_name=var_name,
        variable_key=variable_key,
        cache_manager=cache_manager,
        verbose=VERBOSE,
    )

    # Pipeline completed
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("All steps (0-7) executed")
    print(f"{'='*80}\n")

    if token_tracker.call_count > 0:
        print(token_tracker.get_summary())

    verbose_capture.__exit__(None, None, None)

# %%
