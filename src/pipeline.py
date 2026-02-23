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
from config import CacheConfig, ModelConfig, DEFAULT_LANGUAGE
cache_config = CacheConfig()
cache_manager = CacheManager(cache_config)
model_config = ModelConfig()

#  ===  STANDALONE ========================================================================================================

#filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
#id_column = "DLNMID"
#var_name = "Q20"
#sample_size = 500

#filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
#id_column = "DLNMID"
#var_name = "Q20"
#sample_size = 500

#filename = "M000000 Associatiemonitor Merk X net databestand.sav"
#id_column = "DLNMID"
#var_name = "Qd1_combined"
#sample_size = 2000 

#filename = "M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
#d_column = "DLNMID"
#var_name = "Q15"
#sample_size = 2000

filename = "M250127 Flitspeiling NAVOtop 0meting_153832.sav"
id_column = "DLNMID"
var_name = "Q10"
sample_size = 50

RUN_UNTIL_STEP = 3
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
    4: "embeddings",
    5: "initial_clusters",
    6: "codebook_generation",
    7: "codebook_refinement",
    8: "code_assignment_direct",
    9: "export"
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
    model_config=None,              # Use global if None
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
        model_config: ModelConfig instance for LLM calls (uses global if None)
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
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

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
        spell_checker = spellChecker.SpellChecker(config=spell_check_config, model_config=model_config, verbose=verbose, prompt_printer=prompt_printer)
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
    model_config=None,              # Use global if None
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
        model_config: ModelConfig instance for LLM calls (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        prompt_printer_enabled: Enable prompt printing
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.QualityFilteredModel]: List of quality-filtered response models
    """
    from utils import qualityFilter, verboseReporter, promptPrinter

    step_name = "quality_filter"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

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
        grader = qualityFilter.Grader(preprocessed_text, var_lab, model_config=model_config, verbose=verbose, prompt_printer=prompt_printer)
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
    model_config=None,              # Use global if None
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
        model_config: ModelConfig instance for LLM calls (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        prompt_printer_enabled: Enable prompt printing
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.IdeasExtractedModel]: List of models with extracted ideas
    """
    from utils import ideaExtractor, verboseReporter, promptPrinter

    step_name = "extracted_ideas"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

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
            model_config=model_config,
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
                verbose_reporter.stat_line(f"Cached extraction metadata: facet={extraction_metadata.primary_facet}, template='{extraction_metadata.template_prefix}'")

        print(f"\n\n'Idea extraction phase' completed in {elapsed_time:.2f} seconds.\n")

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Idea extraction completed in {elapsed_time:.2f}s")

    return encoded_text


def step_4_generate_embeddings(
    encoded_text,
    filename,
    var_lab,
    variable_key=None,              # Auto-generate if None
    cache_manager=None,             # Use global if None
    model_config=None,              # Use global if None
    force_recalc=False,
    verbose=True,
    streamlit_container=None        # Optional progress updates
):
    """Step 4: Generate embeddings for extracted ideas

    Args:
        encoded_text: List of IdeasExtractedModel instances from step 3
        filename: SPSS filename for caching
        var_lab: Variable label for context
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        model_config: ModelConfig instance for API configuration (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.EmbeddingsModel]: List of models with embeddings
    """
    from config_steps.config_embedder import EmbedderConfig
    from utils.embedder import Embedder
    from utils.verboseReporter import VerboseReporter

    step_name = "embeddings"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Generating embeddings for ideas...")
    verbose_reporter = VerboseReporter(verbose)

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        embedded_text = cache_manager.load_from_cache(filename, step_name, variable_key, models.EmbeddingsModel)
        total_embeddings = sum(len(resp.response_ideas) for resp in embedded_text if resp.response_ideas)
        verbose_reporter.summary("EMBEDDINGS FROM CACHE", {
            "Input": f"{len(encoded_text)} responses",
            "Total embeddings": f"{total_embeddings}"
        })

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success("✅ Embedding generation completed (from cache)")
    else:
        verbose_reporter.section_header("EMBEDDING GENERATION PHASE")
        start_time = time.time()
        verbose_reporter.step_start("Generating Embeddings", emoji="🔗")

        # Load extraction metadata from cache (contains template_prefix for embedding format)
        extraction_metadata = None
        try:
            extraction_metadata = cache_manager.load_metadata_from_cache(
                filename=filename,
                step="extracted_ideas",
                variable_key=variable_key,
                model_cls=models.ExtractionMetadata
            )
            if extraction_metadata and verbose:
                print(f"   Loaded extraction metadata (template_prefix: '{extraction_metadata.template_prefix[:30]}...')" if extraction_metadata.template_prefix and len(extraction_metadata.template_prefix) > 30 else f"   Loaded extraction metadata (template_prefix: '{extraction_metadata.template_prefix}')" if extraction_metadata.template_prefix else "   Loaded extraction metadata (no template_prefix)")
        except Exception as e:
            if verbose:
                print(f"   Note: Could not load extraction metadata: {e}")

        # Initialize embedder with v2 config (defaults: both mode, analysis enabled)
        embedder_config = EmbedderConfig(verbose=verbose)
        get_embeddings = Embedder(
            config=embedder_config,
            model_config=model_config,
            var_lab=var_lab
        )

        # Print configuration summary (matching experiment runner output)
        if verbose:
            print(f"\n📋 Embedder Configuration:")
            print(f"   Provider: {embedder_config.provider}")
            print(f"   Embedding model: {get_embeddings.embedding_model}")
            print(f"   Text format: {embedder_config.embedding_text_format}")
            print(f"   Question-aware: {embedder_config.use_question_aware}")
            print(f"   Analyze embeddings: {embedder_config.analyze_embeddings}")
            print(f"   Compute similarity stats: {embedder_config.compute_similarity_stats}")

        # Pass extraction metadata for template_prefix access
        if extraction_metadata:
            get_embeddings.set_extraction_metadata(extraction_metadata)

        # Count input statistics
        total_ideas = sum(item.idea_count for item in encoded_text)
        if verbose:
            print(f"\n📊 Input Statistics:")
            print(f"   Total responses: {len(encoded_text)}")
            print(f"   Total ideas: {total_ideas}")
            print(f"   Average ideas per response: {total_ideas / len(encoded_text):.2f}" if encoded_text else "   Average ideas per response: 0")

        input_data = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
        embedded_text = get_embeddings.get_embeddings_with_tracking(input_data, var_lab)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Count output statistics
        embeddings_count = sum(
            1 for resp in embedded_text
            if resp.response_ideas
            for idea in resp.response_ideas
            if idea.idea_embedding is not None
        )
        taxonomy_count = sum(
            1 for resp in embedded_text
            if resp.response_ideas
            for idea in resp.response_ideas
            if getattr(idea, 'taxonomy_embedding', None) is not None
        )
        ontology_count = sum(
            1 for resp in embedded_text
            if resp.response_ideas
            for idea in resp.response_ideas
            if getattr(idea, 'ontology_embedding', None) is not None
        )

        # Print final statistics (matching experiment runner output)
        if verbose:
            print(f"\n📊 Embedding Statistics:")
            print(f"   Responses processed: {len(embedded_text)}")
            print(f"   Idea embeddings generated: {embeddings_count}")
            if taxonomy_count > 0:
                print(f"   Taxonomy embeddings generated: {taxonomy_count}")
            if ontology_count > 0:
                print(f"   Ontology embeddings generated: {ontology_count}")
            print(f"   Elapsed time: {elapsed_time:.2f}s")
            print(f"   Rate: {embeddings_count / elapsed_time:.1f} embeddings/sec" if elapsed_time > 0 else "   Rate: N/A")

            # Print analysis results if available
            if get_embeddings.analysis:
                analysis = get_embeddings.analysis
                print(f"\n🔍 Embedding Analysis:")
                print(f"   Dimensions: {analysis.embedding_dim}")
                print(f"   Norm: mean={analysis.mean_norm:.4f}, std={analysis.std_norm:.4f}, "
                      f"range=[{analysis.min_norm:.4f}, {analysis.max_norm:.4f}]")
                if analysis.mean_pairwise_similarity is not None:
                    print(f"   Pairwise similarity: mean={analysis.mean_pairwise_similarity:.4f}, "
                          f"std={analysis.std_pairwise_similarity:.4f}, "
                          f"range=[{analysis.min_pairwise_similarity:.4f}, {analysis.max_pairwise_similarity:.4f}]")

        cache_manager.save_to_cache(embedded_text, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)
        print(f"\n'Embedding generation' completed in {elapsed_time:.2f} seconds.")

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Embedding generation completed in {elapsed_time:.2f}s")

    return embedded_text


def step_5_cluster(
    embedded_text,
    filename,
    var_lab,
    variable_key=None,              # Auto-generate if None
    cache_manager=None,             # Use global if None
    force_recalc=False,
    verbose=True,
    streamlit_container=None        # Optional progress updates
):
    """Step 5: Perform dimensionality reduction and clustering

    Uses Clusterer with:
    - Automatic algorithm selection (DVC + knee detection)
    - Optuna-based HDBSCAN optimization
    - c-TF-IDF keyword extraction
    - LLM cluster label generation

    Args:
        embedded_text: List of EmbeddingsModel instances from step 4
        filename: SPSS filename for caching
        var_lab: Survey question text (for context)
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        List[models.ClusterModel]: List of models with cluster assignments
    """
    from utils.clusterer import Clusterer
    from config_steps.config_clusterer import ClustererConfig
    from utils.verboseReporter import VerboseReporter

    step_name = "initial_clusters"
    representations_step_name = "cluster_representations"
    variable_key, cache_manager, _ = _resolve_step_defaults(variable_key, cache_manager)

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Clustering ideas (auto algorithm selection)...")
    verbose_reporter = VerboseReporter(verbose)

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        initial_cluster_results = cache_manager.load_from_cache(filename, step_name, variable_key, models.ClusterModel)
        cluster_ids = set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_ideas if segment.initial_cluster is not None])
        num_initial_clusters = len(cluster_ids)
        total_segments = sum(len(resp.response_ideas) for resp in initial_cluster_results if resp.response_ideas)
        verbose_reporter.summary("INITIAL CLUSTERS FROM CACHE", {
            "Input": f"{len(embedded_text)} responses",
            "Total segments": f"{total_segments}",
            "Initial clusters": f"{num_initial_clusters}"
        })

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Clustering completed (from cache): {num_initial_clusters} clusters")
    else:
        verbose_reporter.section_header("INITIAL CLUSTERING PHASE")
        start_time = time.time()

        # Load extraction metadata for taxonomy context
        extraction_metadata = None
        try:
            extraction_metadata = cache_manager.load_metadata_from_cache(
                filename=filename,
                step="extracted_ideas",
                variable_key=variable_key,
                model_cls=models.ExtractionMetadata
            )
            if extraction_metadata and verbose:
                prefix_display = extraction_metadata.template_prefix[:40] + "..." if extraction_metadata.template_prefix and len(extraction_metadata.template_prefix) > 40 else extraction_metadata.template_prefix or "(none)"
                print(f"   Loaded extraction metadata (template_prefix: '{prefix_display}')")
        except Exception as e:
            if verbose:
                print(f"   Note: Could not load extraction metadata: {e}")

        # Configure Clusterer
        config = ClustererConfig(
            # Algorithm selection: auto (DVC + knee detection)
            algorithm_mode="auto",

            # DVC thresholds
            dvc_high_threshold=0.45,
            dvc_low_threshold=0.25,
            force_agglomerative_below_dvc=0.25,

            # Knee detection
            knee_y_diff_threshold=0.6,

            # Optuna optimization
            use_optuna=True,
            max_noise_rate=0.20,
            min_clusters=3,

            # Quality-triggered re-search
            enable_research=True,
            research_max_noise_rate=0.10,
            research_min_validity=0.70,
            research_cluster_deviation_threshold=0.15,

            # Post-processing
            enable_merging=True,
            merge_centroid_threshold=0.95,
            merge_pairwise_threshold=0.98,

            # Noise reduction (BERTopic-style)
            noise_reduction_strategy="embeddings",
            noise_reduction_threshold=0.5,

            # c-TF-IDF keyword extraction
            generate_ctfidf=True,
            ctfidf_top_k=10,
            ctfidf_use_lemmatization=True,

            # LLM cluster labels (for speculative codes)
            generate_llm_labels=True,

            # Output
            verbose=verbose,
        )

        # Run clustering
        clusterer = Clusterer(embedded_text, config=config, extraction_metadata=extraction_metadata)
        clusterer.run()
        initial_cluster_results = clusterer.to_cluster_model()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Cache cluster results (primary output)
        cache_manager.save_to_cache(initial_cluster_results, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

        # Cache clustering metadata (Layer 2: keywords, labels, distributions, metrics)
        clustering_metadata = clusterer.to_metadata_model()
        cache_manager.save_to_cache(
            [clustering_metadata],
            filename,
            "clustering_metadata",
            variable_key,
            elapsed_time,
            var_lab=var_lab
        )

        # Cache cluster representations separately (for speculative codes in step 6)
        keywords = clusterer.get_cluster_keywords() or {}
        labels = clusterer.get_cluster_labels() or {}

        if keywords or labels:
            representations = []
            all_cluster_ids = set(keywords.keys()) | set(labels.keys())

            for cluster_id in sorted(all_cluster_ids):
                # Build LLM label model if available
                llm_label = None
                if cluster_id in labels:
                    label = labels[cluster_id]
                    llm_label = models.ClusterLabelModel(
                        cluster_id=label.cluster_id,
                        theme=label.theme,
                        description=label.description,
                        key_concepts=label.key_concepts,
                        n_ideas=label.n_ideas
                    )

                rep = models.ClusterRepresentationModel(
                    cluster_id=cluster_id,
                    keywords=keywords.get(cluster_id, []),
                    llm_label=llm_label
                )
                representations.append(rep)

            # Get algorithm info for metadata
            algorithm_rec = clusterer.get_algorithm_recommendation()
            metrics = clusterer.get_metrics()

            representations_model = models.ClusterRepresentationsModel(
                representations=representations,
                generation_metadata={
                    "algorithm": algorithm_rec.recommended_algorithm if algorithm_rec else "unknown",
                    "dvc_value": algorithm_rec.dvc_value if algorithm_rec else None,
                    "n_clusters": metrics.n_clusters if metrics else len(all_cluster_ids),
                    "noise_rate": metrics.noise_rate if metrics else None,
                    "mean_coherence": metrics.mean_coherence if metrics else None,
                }
            )

            # Cache representations (for step 6 speculative codes)
            cache_manager.save_to_cache(
                representations_model.model_dump(),
                filename,
                representations_step_name,
                variable_key,
                elapsed_time,
                var_lab=var_lab
            )

        print(f"\n'Initial clustering' completed in {elapsed_time:.2f} seconds.")

        # Print detailed summary (matching experiment version output)
        if verbose:
            print("\n" + "=" * 70)
            print("CLUSTERING SUMMARY")
            print("=" * 70)

            # Algorithm recommendation details
            if algorithm_rec:
                print(f"\nAlgorithm Recommendation:")
                print(f"  Recommended: {algorithm_rec.recommended_algorithm} ({algorithm_rec.confidence} confidence)")
                print(f"  DVC: {algorithm_rec.dvc_value:.3f} → {algorithm_rec.dvc_recommendation}")
                print(f"  Knee: y_diff={algorithm_rec.y_difference:.2f}, sharp={algorithm_rec.has_sharp_knee}")
                if algorithm_rec.is_forced:
                    print(f"  FORCED: Algorithm selection was forced by hard DVC rule")
                print(f"  Reasoning: {algorithm_rec.reasoning}")

            # Clustering metrics
            if metrics:
                print(f"\nClustering Metrics:")
                print(f"  Clusters: {metrics.n_clusters}")
                print(f"  Noise: {metrics.noise_count} ({metrics.noise_rate:.1%})")
                print(f"  Coherence: {metrics.mean_coherence:.3f} ({metrics.coherence_breakdown})")
                if metrics.dbcv is not None:
                    print(f"  DBCV: {metrics.dbcv:.3f}")
                if metrics.silhouette is not None and not np.isnan(metrics.silhouette):
                    print(f"  Silhouette: {metrics.silhouette:.3f}")
                if metrics.mean_persistence is not None:
                    print(f"  Persistence: mean={metrics.mean_persistence:.3f}, weighted={metrics.weighted_persistence:.3f}")
                if metrics.mean_probability is not None:
                    print(f"  Probability: mean={metrics.mean_probability:.3f}, low_ratio={metrics.low_prob_ratio:.1%}")
                if metrics.mean_outlier_score is not None:
                    print(f"  Outliers: mean_score={metrics.mean_outlier_score:.3f}, high_ratio={metrics.high_outlier_ratio:.1%}")
                print(f"  Cluster sizes: min={metrics.min_cluster_size}, median={metrics.median_cluster_size}, max={metrics.max_cluster_size}")

            # Template prefix
            template_prefix = clusterer._template_prefix
            if template_prefix:
                prefix_display = template_prefix[:60] + "..." if len(template_prefix) > 60 else template_prefix
                print(f"\nTemplate prefix: '{prefix_display}'")
            else:
                print(f"\nTemplate prefix: (none)")

            # c-TF-IDF Keywords summary
            if keywords:
                print(f"\nc-TF-IDF Keywords ({len(keywords)} clusters):")
                for cluster_id in sorted(keywords.keys()):
                    kw_list = keywords[cluster_id]
                    kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
                    print(f"  Cluster {cluster_id}: {kw_str}")

            # MMR and TF-IDF Keywords (additional methods from experimental version)
            all_keywords = clusterer.get_all_cluster_keywords()
            if all_keywords:
                for method_name in ["mmr", "tfidf"]:
                    method_keywords = all_keywords.get(method_name)
                    if method_keywords:
                        method_label = {"mmr": "MMR", "tfidf": "TF-IDF"}.get(method_name, method_name)
                        print(f"\n{method_label} Keywords ({len(method_keywords)} clusters):")
                        for cluster_id in sorted(method_keywords.keys()):
                            kw_list = method_keywords[cluster_id]
                            kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
                            print(f"  Cluster {cluster_id}: {kw_str}")

            # Print all clusters with samples (key feature from experiment version)
            clusterer.print_all_clusters(n_samples=10)

            # Cache confirmation (at end for visibility)
            print(f"\n{'='*70}")
            print(f"CACHED: {len(initial_cluster_results)} results to 'initial_clusters' (variable_key: {variable_key})")
            print(f"CACHED: {len(clustering_metadata.clusters)} clusters to 'clustering_metadata'")

        # Optional Streamlit success message
        if streamlit_container:
            num_clusters = len(set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_ideas if segment.initial_cluster is not None]))
            streamlit_container.success(f"✅ Clustering completed in {elapsed_time:.2f}s: {num_clusters} clusters")

    return initial_cluster_results


def step_6_generate_codebook(
    initial_cluster_results,
    filename,
    var_name,
    var_lab,
    variable_key=None,              # Auto-generate if None
    cache_manager=None,             # Use global if None
    model_config=None,              # Use global if None
    use_speculative_starter_codes=False,
    force_recalc=False,
    verbose=True,
    verbose_detailed=False,
    prompt_printer_enabled=False,
    cache_reasoning=True,
    streamlit_container=None        # Optional progress updates
):
    """Step 6: Generate codebook from clusters using inductive coding

    Args:
        initial_cluster_results: List of ClusterModel instances from step 5
        filename: SPSS filename for caching
        var_name: Variable name for metadata
        var_lab: Variable label for context
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        model_config: ModelConfig instance for LLM calls (uses global if None)
        use_speculative_starter_codes: Whether to use speculative starter codes
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        verbose_detailed: Enable detailed verbose output
        prompt_printer_enabled: Enable prompt printing
        cache_reasoning: Cache reasoning results for export
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        CodeGeneratorReasoningResults: Complete reasoning results with codebook and tracking data
    """
    from utils import speculativeStarterCodes, codeGenerator, clusterer, verboseReporter, promptPrinter
    
    step_name = "codebook_generation"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Generating codebook from clusters...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    # Always capture prompts when verbose, but only print realtime if prompt_printer_enabled
    prompt_printer = promptPrinter.PromptPrinter(
        enabled=verbose,  # Capture prompts whenever verbose mode is on
        print_realtime=prompt_printer_enabled  # Only print realtime if explicitly enabled
    )
    codebook_reasoning = None

    if not force_recalc and cache_manager.is_cache_valid(filename, f"{step_name}_reasoning", variable_key):
        # Load codebook_reasoning from cache
        reasoning_models = cache_manager.load_from_cache(
            filename, f"{step_name}_reasoning", variable_key, codeGenerator.CodeGeneratorReasoningResults
        )
        if reasoning_models and len(reasoning_models) > 0:
            codebook_reasoning = reasoning_models[0]
            num_codes = len(codebook_reasoning.codebook) if codebook_reasoning.codebook else 0
            verbose_reporter.summary("CODEBOOK FROM CACHE", {
                "Total codes": num_codes,
                "Source variable": var_name
            })

            # Optional Streamlit success message
            if streamlit_container:
                streamlit_container.success(f"✅ Codebook generation completed (from cache): {num_codes} codes")

            print("[OK] Loaded codebook reasoning from cache")
        else:
            print("ERROR: Failed to load codebook reasoning from cache")
            codebook_reasoning = None
    else:
        verbose_reporter.section_header("CODEBOOK GENERATION PHASE")
        start_time = time.time()

        # Phase 1: Generate starter codes (optional)
        if use_speculative_starter_codes:
            # First, try to load LLM cluster labels from step 5 clustering_metadata cache
            starter_codes = []
            metadata_step_name = "clustering_metadata"

            if cache_manager.is_cache_valid(filename, metadata_step_name, variable_key):
                try:
                    # Load ClusteringMetadataModel from cache
                    metadata_results = cache_manager.load_from_cache(
                        filename, metadata_step_name, variable_key,
                        model_cls=models.ClusteringMetadataModel
                    )
                    if metadata_results and len(metadata_results) > 0:
                        metadata = metadata_results[0]
                        # Extract starter codes from cluster labels (matches run_experiment.py)
                        for cluster_id, cluster_data in metadata.clusters.items():
                            if cluster_data.label_theme:
                                starter_codes.append({
                                    'code': cluster_data.label_theme,
                                    'definition': cluster_data.label_description or '',
                                    'cluster_id': cluster_id
                                })
                        if starter_codes:
                            print(f"Loaded {len(starter_codes)} starter codes from Clusterer LLM labels")
                except Exception as e:
                    print(f"Failed to load clustering_metadata: {e}")
                    starter_codes = []

            # Fall back to speculative starter codes generator if no LLM labels
            if not starter_codes:
                starter_generator = speculativeStarterCodes.SpeculativeStarterCodes(
                    var_lab=var_lab,
                    verbose=verbose,
                    prompt_printer=prompt_printer
                )
                starter_codes = starter_generator.generate()
        else:
            # Use empty starter codes when speculative generation is disabled
            starter_codes = []
            print("Speculative starter codes disabled - proceeding with empty starter codes")

        if not starter_codes and use_speculative_starter_codes:
            print("Error: Failed to generate starter codes. Cannot proceed with codebook generation.")
            codebook_reasoning = None
        else:
            # Clean ideas before code generation (remove brackets, normalize whitespace)
            if verbose:
                print("\nCleaning ideas for code generation...")

            cleaned_cluster_results = clusterer.clean_cluster_ideas(initial_cluster_results) #removal of meta data/context specifiers from idea text

            # Load extraction_metadata for theme extraction
            extraction_metadata = cache_manager.load_metadata_from_cache(
                filename=filename,
                step="extracted_ideas",
                variable_key=variable_key,
                model_cls=models.ExtractionMetadata
            )
            if extraction_metadata and verbose:
                print(f"[INFO] Loaded extraction metadata for theme extraction")

            # Phase 2: Inductive code generation
            generator = codeGenerator.InductiveCodeGenerator(
                cluster_results=cleaned_cluster_results,  # Use cleaned version
                starter_codes=starter_codes,
                var_lab=var_lab,
                verbose=True,
                verbose_detailed=verbose_detailed,
                prompt_printer=prompt_printer,
                extraction_metadata=extraction_metadata
            )
            results = generator.generate()

            if results and isinstance(results, codeGenerator.CodeGeneratorReasoningResults):
                # Use the codebook from results directly - it already has assignment_examples
                final_codebook = results.codebook if results.codebook else []

                # Display final codebook summary
                if verbose and final_codebook:
                    verbose_reporter.empty_line()
                    print("[STATS] FINAL CODEBOOK SUMMARY")
                    verbose_reporter.stat_line(f"Total codes: {len(final_codebook)}")
                    total_clusters_mapped = sum(len(item['source_cluster_id'].split(',')) for item in final_codebook)
                    verbose_reporter.stat_line(f"Total clusters mapped: {total_clusters_mapped}")

                    # Show sample codes
                    verbose_reporter.empty_line()
                    print("[LIST] Complete codebook:")

                idx = 1
                # Display the extracted final codebook
                for item in final_codebook:
                    if verbose:
                        definition = item['definition']
                        if len(definition) > 100:
                            definition = definition[:97] + "..."
                        cluster_count = len(item['source_cluster_id'].split(','))
                        cluster_info = f" (→ {cluster_count} clusters)" if cluster_count > 1 else ""
                        print(f"  {idx}. {item['code']}{cluster_info}")
                    idx += 1

                # Validation: Ensure all clusters are mapped
                if results and hasattr(results, 'cluster_results'):
                    total_clusters = len(results.cluster_results)

                    # Extract all cluster IDs from results
                    all_cluster_ids = {str(cr.get('cluster_id', '')) for cr in results.cluster_results}

                    # Extract mapped cluster IDs from final_codebook
                    mapped_cluster_ids = set()
                    for item in final_codebook:
                        cluster_ids = item['source_cluster_id'].split(',')
                        mapped_cluster_ids.update(cluster_ids)
                    mapped_clusters = len(mapped_cluster_ids)

                    # Calculate missing
                    missing = all_cluster_ids - mapped_cluster_ids

                    if verbose:
                        verbose_reporter.empty_line()
                        verbose_reporter.stat_line("[VALIDATION] Step 6 cluster mapping:")
                        verbose_reporter.stat_line(f"  Total clusters processed: {total_clusters}")
                        verbose_reporter.stat_line(f"  Clusters mapped to codes: {mapped_clusters}")

                    if mapped_clusters != total_clusters:
                        verbose_reporter.warning(f"  WARNING: {total_clusters - mapped_clusters} clusters not mapped to codes!")
                        verbose_reporter.warning(f"  Missing cluster IDs: {sorted(missing)}")
                    else:
                        verbose_reporter.stat_line("  ✓ All clusters successfully mapped")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Pass reasoning results if available (either from cache or newly generated)
        reasoning_for_display = None
        if 'codebook_reasoning' in locals():
            reasoning_for_display = codebook_reasoning
        elif 'results' in locals():
            reasoning_for_display = results

        # Always cache codebook reasoning if available for consistent exports
        if 'results' in locals() and results:
            try:
                codebook_reasoning = results
                cache_manager.save_to_cache([codebook_reasoning], filename, f"{step_name}_reasoning", variable_key, elapsed_time, var_lab=var_lab)
                print("Cached codebook reasoning for export consistency")
            except Exception as e:
                print(f"WARNING: Failed to cache reasoning results: {e}")
                print("   Export will fall back to basic format without reasoning columns")
        else:
            print("WARNING: No reasoning results generated to cache")
            print("   Export will fall back to basic format without reasoning columns")

        # Cache the enriched cluster results with expanded_cluster field
        cache_manager.save_to_cache(
            generator.cluster_results,  # Use updated objects with expanded_cluster populated
            filename,
            "expanded_clusters",
            variable_key,
            elapsed_time,
            var_lab=var_lab)

        print("Cached enriched clusters with expanded_cluster field")

        print(f"\n'codebook generation' completed in {elapsed_time:.2f} seconds.\n")

        # Optional Streamlit success message
        if streamlit_container:
            num_codes = len(codebook_reasoning.codebook) if codebook_reasoning and codebook_reasoning.codebook else 0
            streamlit_container.success(f"✅ Codebook generation completed in {elapsed_time:.2f}s: {num_codes} codes")

    # Display sample prompts (first of each stage) when verbose - matches run_experiment.py
    if verbose and prompt_printer.prompts:
        print("\n" + "=" * 80)
        print("SAMPLE PROMPTS (First of Each Stage)")
        print("=" * 80)
        prompt_printer.print_all_prompts()

    return codebook_reasoning


def step_7_refine_codebook(
    codebook_reasoning,
    filename,
    var_name,
    var_lab,
    variable_key=None,              # Auto-generate if None
    cache_manager=None,             # Use global if None
    model_config=None,              # Use global if None
    default_language=None,          # Use DEFAULT_LANGUAGE if None
    force_recalc=False,
    verbose=True,
    prompt_printer_enabled=False,
    streamlit_container=None        # Optional progress updates
):
    """Step 7: Refine codebook into hierarchical themes

    Args:
        codebook_reasoning: CodeGeneratorReasoningResults from step 6
        filename: SPSS filename for caching
        var_name: Variable name for metadata
        var_lab: Variable label for context
        variable_key: Cache key (auto-generated if None)
        cache_manager: CacheManager instance (uses global if None)
        model_config: ModelConfig instance for LLM calls (uses global if None)
        default_language: Language for refinement (uses DEFAULT_LANGUAGE if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        prompt_printer_enabled: Enable prompt printing
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        tuple: (refinement_results: CodeRefinementResults, theme_enriched_codebook: ThemeEnrichedCodebookModel, refinement_report: dict, prompt_printer: PromptPrinter)
    """
    from utils.codebookRefinement import refine_codebook, print_refinement_report, get_refinement_report
    from utils import verboseReporter, promptPrinter

    step_name = "codebook_refinement"
    variable_key, cache_manager, model_config = _resolve_step_defaults(variable_key, cache_manager, model_config)

    # Use DEFAULT_LANGUAGE if not provided
    if default_language is None:
        from config import DEFAULT_LANGUAGE as DEFAULT_LANG
        default_language = DEFAULT_LANG

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Refining codebook into hierarchical themes...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=True, print_realtime=prompt_printer_enabled)
    start_time = time.time()

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        refinement_results_cached = cache_manager.load_from_cache(filename, step_name, variable_key, models.CodeRefinementResults)
        if refinement_results_cached and len(refinement_results_cached) > 0:
            refinement_results = refinement_results_cached[0]
            verbose_reporter.summary("CODEBOOK REFINEMENT FROM CACHE", {
                "Original codes": refinement_results.processing_stats.get('original_code_count', 0),
                "Refined categories": refinement_results.processing_stats.get('refined_category_count', 0),
                "Total subcodes": refinement_results.processing_stats.get('total_refined_subcodes', 0)
            })

            # Optional Streamlit success message
            if streamlit_container:
                num_categories = refinement_results.processing_stats.get('refined_category_count', 0)
                streamlit_container.success(f"✅ Codebook refinement completed (from cache): {num_categories} themes")
        else:
            print("ERROR: Failed to load codebook refinement from cache")
            refinement_results = None
    else:
        verbose_reporter.section_header("CODEBOOK REFINEMENT PHASE")

        # Check if we have codebook_reasoning from step 6
        if codebook_reasoning is not None:
            verbose_reporter.step_start("Refinement", "Refining raw codes into hierarchical structure")

            # Run refinement using simple sync call
            refinement_results = refine_codebook(
                survey_question=var_lab,
                reasoning_results=codebook_reasoning,
                model_config=model_config,
                language=default_language,
                verbose=verbose,
                prompt_printer=prompt_printer
            )

            # Cache results
            elapsed_time = time.time() - start_time
            cache_manager.save_to_cache([refinement_results], filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

            if verbose:
                print_refinement_report(refinement_results)
        else:
            print("ERROR: No codebook reasoning results available for refinement")
            refinement_results = None

    elapsed_time = time.time() - start_time
    print(f"\n'codebook refinement' completed in {elapsed_time:.2f} seconds.\n")

    # Optional Streamlit success message
    if streamlit_container:
        if refinement_results:
            num_categories = refinement_results.processing_stats.get('refined_category_count', 0)
            streamlit_container.success(f"✅ Codebook refinement completed in {elapsed_time:.2f}s: {num_categories} themes")
        else:
            streamlit_container.warning("⚠️ Codebook refinement had issues")

    # Create theme enriched codebook
    if refinement_results and refinement_results.refined_codebook.refined_codebook:
        verbose_reporter.step_start("Creating theme enriched codebook", "Converting refined results for step 9")

        # Load codebook_reasoning from cache if not provided (for cache-only runs)
        if not codebook_reasoning:
            from utils import codeGenerator
            try:
                reasoning_models = cache_manager.load_from_cache(
                    filename, "codebook_generation_reasoning", variable_key,
                    codeGenerator.CodeGeneratorReasoningResults
                )
                if reasoning_models and len(reasoning_models) > 0:
                    codebook_reasoning = reasoning_models[0]
                    verbose_reporter.stat_line("Loaded codebook_reasoning from cache for examples extraction")
            except Exception as e:
                verbose_reporter.warning(f"Failed to load codebook_reasoning from cache: {e}")
                codebook_reasoning = None

        # Create ThemeEnrichedCodebookEntry objects from refined codebook
        enriched_entries = []
        code_to_theme_mapping = {}
        themes_summary = []

        # Extract assignment_examples from codebook_reasoning.codebook
        # Map by expanded cluster IDs to handle code merging/renaming
        import json
        cluster_to_assignment_examples = {}
        if codebook_reasoning and hasattr(codebook_reasoning, 'codebook'):
            for entry in codebook_reasoning.codebook:
                # Parse JSON strings to lists
                inclusion = entry.get('inclusion_examples')
                exclusion = entry.get('exclusion_examples')

                examples_data = {
                    'inclusion_examples': json.loads(inclusion) if inclusion and isinstance(inclusion, str) else inclusion,
                    'exclusion_examples': json.loads(exclusion) if exclusion and isinstance(exclusion, str) else exclusion,
                    'near_neighbor_label': entry.get('near_neighbor_label'),
                    'tell_apart_rule': entry.get('tell_apart_rule')
                }

                # Map each expanded cluster ID to these examples
                source_clusters = entry.get('source_cluster_id', '').split(',')
                for cluster_id in source_clusters:
                    cluster_id = cluster_id.strip()
                    if cluster_id:
                        cluster_to_assignment_examples[cluster_id] = examples_data

        for category in refinement_results.refined_codebook.refined_codebook:
            theme_name = category.category

            # Add to themes summary
            themes_summary.append({
                'theme_name': theme_name,
                'theme_description': theme_name,  # Use theme name as description
                'code_count': len(category.subcodes)
            })

            for subcode in category.subcodes:
                # Get assignment_examples by expanded cluster ID (handles code merging)
                source_clusters = subcode.source_cluster.split(',') if subcode.source_cluster else []
                first_cluster = source_clusters[0].strip() if source_clusters else None
                examples = cluster_to_assignment_examples.get(first_cluster, {}) if first_cluster else {}

                final_inclusion = examples.get('inclusion_examples')
                final_exclusion = examples.get('exclusion_examples')
                near_neighbor = examples.get('near_neighbor_label')
                tell_apart = examples.get('tell_apart_rule')

                # Create ThemeEnrichedCodebookEntry with category support (3-level hierarchy)
                enriched_entry = models.ThemeEnrichedCodebookEntry(
                    code=subcode.code,
                    definition=subcode.description,
                    theme=theme_name,
                    theme_description=theme_name,
                    category=subcode.category,  # Empty string for 2-level, category name for 3-level
                    category_description=subcode.category if subcode.category else "",  # Use category name as description
                    source_cluster=subcode.source_cluster,  # Use source_cluster directly from RefinedSubcode
                    inclusion_examples=final_inclusion,
                    exclusion_examples=final_exclusion,
                    near_neighbor_label=near_neighbor,
                    tell_apart_rule=tell_apart
                )
                enriched_entries.append(enriched_entry)

                # Build code-to-theme mapping
                code_to_theme_mapping[subcode.code] = theme_name

        # Create ThemeEnrichedCodebookModel
        theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
            codes=enriched_entries,
            themes_summary=themes_summary,
            code_to_theme_mapping=code_to_theme_mapping,
            theme_methodology="GPT-5 based codebook refinement with hierarchical theme organization",
            generation_metadata={
                "refinement_source": "step_8_codebook_refinement",
                "original_code_count": len(refinement_results.original_codebook),
                "refined_category_count": len(refinement_results.refined_codebook.refined_codebook),
                "total_refined_codes": len(enriched_entries)
            },
            source_variable=var_name
        )

        verbose_reporter.step_complete(f"Created theme enriched codebook: {len(enriched_entries)} codes in {len(themes_summary)} themes")

        # Validation: Ensure all cluster IDs from Step 6 are preserved after Step 7 refinement
        if codebook_reasoning and hasattr(codebook_reasoning, 'cluster_results') and verbose:
            # Get all cluster IDs from Step 6 codebook_reasoning
            step6_clusters = {str(cr.get('cluster_id', '')) for cr in codebook_reasoning.cluster_results if cr.get('cluster_id')}

            # Get all cluster IDs from Step 7 refined codebook
            step7_clusters = set()
            for entry in enriched_entries:
                if hasattr(entry, 'source_cluster') and entry.source_cluster:
                    for cluster_id in str(entry.source_cluster).split(','):
                        step7_clusters.add(cluster_id.strip())

            verbose_reporter.empty_line()
            verbose_reporter.stat_line("[VALIDATION] Step 7 cluster preservation:")
            verbose_reporter.stat_line(f"  Clusters from Step 6: {len(step6_clusters)}")
            verbose_reporter.stat_line(f"  Clusters in Step 7: {len(step7_clusters)}")

            if step6_clusters != step7_clusters:
                lost = step6_clusters - step7_clusters
                if lost:
                    verbose_reporter.warning(f"  WARNING: {len(lost)} clusters lost during refinement!")
                    verbose_reporter.warning(f"  Lost cluster IDs: {sorted(lost)}")
                added = step7_clusters - step6_clusters
                if added:
                    verbose_reporter.warning(f"  WARNING: {len(added)} unexpected clusters added!")
                    verbose_reporter.warning(f"  Added cluster IDs: {sorted(added)}")
            else:
                verbose_reporter.stat_line("  ✓ All clusters preserved through refinement")

    else:
        print("ERROR: No refinement results available to create theme enriched codebook")
        # Create empty theme enriched codebook as fallback
        theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
            codes=[],
            themes_summary=[],
            code_to_theme_mapping={},
            theme_methodology="Empty fallback - refinement failed",
            source_variable=var_name
        )

    # Cache theme_enriched_codebook separately (following step 6 pattern)
    if theme_enriched_codebook:
        try:
            cache_manager.save_to_cache([theme_enriched_codebook], filename, f"{step_name}_enriched", variable_key, elapsed_time, var_lab=var_lab)
            if verbose:
                print("Cached theme enriched codebook for step 8 access")
        except Exception as e:
            print(f"Warning: Failed to cache theme enriched codebook: {str(e)}")

    # Generate structured report for Streamlit display
    refinement_report = None
    if refinement_results:
        refinement_report = get_refinement_report(refinement_results)

    return refinement_results, theme_enriched_codebook, refinement_report, prompt_printer


def step_8_assign_codes(
    filename,
    variable_key,
    cache_manager,
    theme_enriched_codebook,
    var_lab,
    model_config=None,              # Use global if None
    force_recalc=False,
    verbose=True,
    verbose_detailed=False,
    prompt_printer_enabled=False,
    streamlit_container=None        # Optional progress updates
):
    """Step 8: Assign codes to individual ideas using enriched clusters from Step 6

    Args:
        filename: SPSS filename for caching
        variable_key: Cache key
        cache_manager: CacheManager instance
        theme_enriched_codebook: ThemeEnrichedCodebookModel from step 7
        var_lab: Variable label for context
        model_config: ModelConfig instance for LLM calls (uses global if None)
        force_recalc: Force recalculation bypassing cache
        verbose: Enable verbose output
        verbose_detailed: Enable detailed verbose output
        prompt_printer_enabled: Enable prompt printing
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        Tuple[List[models.CodeAssignedModel], Dict]:
            - List of models with code assignments
            - Dictionary with stats (total_responses, total_ideas, unique_codes_assigned,
              unique_themes_assigned, total_code_assignments, total_theme_assignments,
              avg_codes_per_idea, avg_themes_per_idea, processing_time)
    """
    from utils import codeAssigner, verboseReporter, promptPrinter

    step_name = "code_assignment_direct"

    initial_cluster_results = cache_manager.load_from_cache(
        filename,
        "expanded_clusters",
        variable_key,
        models.ClusterModel
    )

    # Use global model_config if not provided
    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Assigning codes to individual ideas...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=prompt_printer_enabled, print_realtime=True)

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        code_assigned_results = cache_manager.load_from_cache(filename, step_name, variable_key, models.CodeAssignedModel)
        total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)
        total_assignments = sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) for resp in code_assigned_results if resp.response_ideas)
        verbose_reporter.summary("DIRECT CODE ASSIGNMENTS FROM CACHE", {
            "Input responses": len(code_assigned_results),
            "Ideas processed": total_ideas,
            "Code assignments": total_assignments,
            "Theme assignments": sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_themes]) for resp in code_assigned_results if resp.response_ideas)
        })

        # Optional Streamlit success message
        if streamlit_container:
            streamlit_container.success(f"✅ Code assignment completed (from cache): {total_assignments} assignments")
    else:
        verbose_reporter.section_header("DIRECT CODE ASSIGNMENT PHASE (NO EMBEDDINGS)")
        start_time = time.time()

        if not theme_enriched_codebook or not theme_enriched_codebook.codes:
            print("Error: No enriched codebook available for direct code assignment.")
            code_assigned_results = []
        elif not initial_cluster_results:
            print("Error: No cluster results available for direct code assignment.")
            code_assigned_results = []
        else:
            print(f"\nDirect assignment: Processing ideas from {len(initial_cluster_results)} cluster results")
            print(f"Using complete codebook with {len(theme_enriched_codebook.codes)} codes")

            # Create code assigner (sends all codes to LLM)
            code_assigner_instance = codeAssigner.CodeAssigner(
                cluster_models=initial_cluster_results,
                codebook=[models.Codebook(
                    code=entry.code,
                    definition=entry.definition,
                    theme=entry.theme,
                    theme_description=entry.theme_description,
                    source_cluster=entry.source_cluster,  # Preserve source_cluster for default code mapping
                    inclusion_examples=entry.inclusion_examples,
                    exclusion_examples=entry.exclusion_examples,
                    near_neighbor_label=entry.near_neighbor_label,
                    tell_apart_rule=entry.tell_apart_rule
                ) for entry in theme_enriched_codebook.codes],
                var_lab=var_lab,
                code_to_theme_mapping=theme_enriched_codebook.code_to_theme_mapping,
                model_config=model_config,
                verbose=verbose,
                prompt_printer=prompt_printer
            )

            code_assigned_results = code_assigner_instance.assign()

            # Print assignment strategy stats (default vs fallback)
            if verbose:
                code_assigner_instance.print_assignment_stats()

        for result in code_assigned_results:
            if not hasattr(result, 'assignment_metadata') or result.assignment_metadata is None:
                result.assignment_metadata = {}
            result.assignment_metadata.update({
                "codebook_used": f"{len(theme_enriched_codebook.codes)} codes with themes",
                "assignment_method": "direct_llm_processing",
                "theme_methodology": theme_enriched_codebook.theme_methodology,
                "assignment_timestamp": start_time
            })

        end_time = time.time()
        elapsed_time = end_time - start_time

        cache_manager.save_to_cache(code_assigned_results, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)
        print(f"\n'Direct code assignment' completed in {elapsed_time:.2f} seconds.\n")

        # Optional Streamlit success message
        if streamlit_container:
            total_assignments = sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) for resp in code_assigned_results if resp.response_ideas)
            streamlit_container.success(f"✅ Code assignment completed in {elapsed_time:.2f}s: {total_assignments} assignments")

    # Calculate stats for UI display (matching PipelineSummarizer output)
    total_responses = len(code_assigned_results)
    total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)

    # Count unique codes and themes
    code_frequency = {}
    theme_frequency = {}
    for resp in code_assigned_results:
        if resp.response_ideas:
            for idea in resp.response_ideas:
                if idea and idea.assigned_codes:
                    for code in idea.assigned_codes:
                        code_frequency[code] = code_frequency.get(code, 0) + 1
                if idea and idea.assigned_themes:
                    for theme in idea.assigned_themes:
                        theme_frequency[theme] = theme_frequency.get(theme, 0) + 1

    total_code_assignments = sum(code_frequency.values())
    total_theme_assignments = sum(theme_frequency.values())
    unique_codes = len(code_frequency)
    unique_themes = len(theme_frequency)

    stats = {
        'total_responses': total_responses,
        'total_ideas': total_ideas,
        'unique_codes_assigned': unique_codes,
        'unique_themes_assigned': unique_themes,
        'total_code_assignments': total_code_assignments,
        'total_theme_assignments': total_theme_assignments,
        'avg_codes_per_idea': total_code_assignments / total_ideas if total_ideas > 0 else 0.0,
        'avg_themes_per_idea': total_theme_assignments / total_ideas if total_ideas > 0 else 0.0,
        'processing_time': elapsed_time if 'elapsed_time' in locals() else 0.0
    }

    # Print code distribution
    if verbose and code_frequency:
        print(f"\n{'='*70}")
        print("CODE DISTRIBUTION")
        print(f"{'='*70}")
        print(f"{'Code':<45} {'Count':>8} {'Percent':>10}")
        print(f"{'-'*70}")

        # Separate regular codes from fallback codes (Overig, - algemeen, etc.)
        from config import MISCELLANEOUS_CODE_LABELS, GENERAL_CODE_LABELS
        misc_labels = set(MISCELLANEOUS_CODE_LABELS.values())
        general_suffixes = [f"- {v}" for v in GENERAL_CODE_LABELS.values()]

        def is_fallback_code(code_name):
            if code_name in misc_labels:
                return True
            for suffix in general_suffixes:
                if suffix in code_name:
                    return True
            return False

        regular_codes = [(c, n) for c, n in code_frequency.items() if not is_fallback_code(c)]
        fallback_codes = [(c, n) for c, n in code_frequency.items() if is_fallback_code(c)]

        # Sort each group by count descending
        regular_codes.sort(key=lambda x: x[1], reverse=True)
        fallback_codes.sort(key=lambda x: x[1], reverse=True)

        # Display regular codes first, then fallback codes
        for code, count in regular_codes:
            pct = (count / total_code_assignments) * 100
            display_code = code[:42] + "..." if len(code) > 45 else code
            print(f"{display_code:<45} {count:>8} {pct:>9.1f}%")

        if fallback_codes:
            print(f"{'-'*70}")
            for code, count in fallback_codes:
                pct = (count / total_code_assignments) * 100
                display_code = code[:42] + "..." if len(code) > 45 else code
                print(f"{display_code:<45} {count:>8} {pct:>9.1f}%")

        print(f"{'-'*70}")
        print(f"{'TOTAL':<45} {total_code_assignments:>8} {'100.0%':>10}")
        print(f"{'='*70}\n")

    # Return instance if it exists (for prompt inspection), otherwise None
    instance_for_inspection = code_assigner_instance if 'code_assigner_instance' in locals() else None
    return code_assigned_results, stats, instance_for_inspection


def step_9_export_results(
    code_assigned_results,
    theme_enriched_codebook,
    filename,
    var_name,
    quality_filtered_text=None,
    verbose=True,
    streamlit_container=None,        # Optional progress updates
    include_visualizations=False,    # Add visualization sheets to Excel
    cache_manager=None,              # Required for loading visualization data
    variable_key=None                # Required for loading visualization data
):
    """Step 9: Export results to Excel

    Args:
        code_assigned_results: List of CodeAssignedModel instances from step 8
        theme_enriched_codebook: ThemeEnrichedCodebookModel from step 7
        filename: SPSS filename for export naming
        var_name: Variable name for export naming
        quality_filtered_text: List of QualityFilteredModel instances from step 2 (includes filtered responses)
        verbose: Enable verbose output
        streamlit_container: Optional Streamlit container for progress updates
        include_visualizations: If True, add dendrogram and word cloud sheets + generate network HTML
        cache_manager: CacheManager instance for loading visualization data (required if include_visualizations=True)
        variable_key: Cache variable key (required if include_visualizations=True)

    Returns:
        str: Path to exported Excel file
    """
    from utils.resultsExporter import ResultsExporter

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Exporting results to Excel...")

    # Load visualization data from cache if requested
    clustering_metadata = None
    extraction_metadata = None

    if include_visualizations and cache_manager and variable_key:
        try:
            # Load clustering metadata (contains c-TF-IDF keywords, LLM labels, etc.)
            clustering_metadata_list = cache_manager.load_from_cache(
                filename, "clustering_metadata", variable_key, models.ClusteringMetadataModel
            )
            if clustering_metadata_list and len(clustering_metadata_list) > 0:
                clustering_metadata = clustering_metadata_list[0]
                if verbose:
                    print(f"[INFO] Loaded clustering metadata for visualizations")

            # Load extraction metadata (contains taxonomy axis, template prefix, etc.)
            extraction_metadata = cache_manager.load_metadata_from_cache(
                filename, "extracted_ideas", variable_key, models.ExtractionMetadata
            )
            if extraction_metadata and verbose:
                print(f"[INFO] Loaded extraction metadata for visualizations")

        except Exception as e:
            if verbose:
                print(f"[WARNING] Could not load visualization data: {e}")
                print("         Visualizations may be incomplete")

    try:
        exporter = ResultsExporter(verbose=verbose)
        excel_path = exporter.export_to_excel(
            code_assigned_results,
            theme_enriched_codebook,
            filename,
            var_name,
            quality_filtered_text=quality_filtered_text,
            export_dir=None,  # Will create default export directory
            include_visualizations=include_visualizations,
            clustering_metadata=clustering_metadata,
            extraction_metadata=extraction_metadata
        )
        print(f"[SUCCESS] Code assignments exported to Excel: {excel_path}")

        # Optional Streamlit success message
        if streamlit_container:
            if include_visualizations:
                streamlit_container.success(f"✅ Results exported with visualizations: {excel_path}")
            else:
                streamlit_container.success(f"✅ Results exported to Excel: {excel_path}")

        return excel_path
    except Exception as e:
        print(f"[WARNING] Excel export failed: {str(e)}")

        # Optional Streamlit error message
        if streamlit_container:
            streamlit_container.error(f"⚠️ Excel export failed: {str(e)}")

        return None



if __name__ == '__main__':
    import sys

    if RUN_UNTIL_STEP is not None and not FORCE_RECALCULATE_ALL:
        FORCE_STEP = STEP_NAMES.get(RUN_UNTIL_STEP, "")
    else:
        FORCE_STEP = ""
    
    USE_SPECULATIVE_STARTER_CODES = False  # Uses LLM labels from Clusterer (step 5) if available
    data_loader = dataLoader.DataLoader(verbose=False)
    var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)

    # Start capturing all verbose output to file - MUST be before any print statements
    from utils.saveVerbose import VerboseCapture
    verbose_capture = VerboseCapture(
        filename=filename,
        variable_key=var_name,
        sample_size=sample_size,
        run_until_step=RUN_UNTIL_STEP if RUN_UNTIL_STEP is not None else 9
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
    print(f"Run until step: {RUN_UNTIL_STEP if RUN_UNTIL_STEP is not None else 'All (0-9)'}")
    print(f"Force recalculate: {'ALL STEPS' if FORCE_RECALCULATE_ALL else (f'Step {RUN_UNTIL_STEP} ({FORCE_STEP})' if FORCE_STEP else 'None')}")
    print(f"Speculative starter codes: {USE_SPECULATIVE_STARTER_CODES}")
    print(f"Verbose mode: {VERBOSE}")
    print(f"Prompt printer: {PROMPT_PRINTER}")
    print("=" * 80)
    
    selected_variables = globals().get('selected_variables', [var_name])
    is_merged = globals().get('is_merged', False)
    test_mode = globals().get('is_test_mode', True)
    sample_size =  globals().get('test_sample_size', sample_size) if test_mode else None
                   
    if 'variable_key' in globals():
        variable_key = globals()['variable_key']
    else:
        # Generate variable_key for standalone mode 
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
            # Print LLM usage summary before exiting
            if token_tracker.call_count > 0:
                print(token_tracker.get_summary())
            # Save captured verbose output before exiting
            verbose_capture.__exit__(None, None, None)
            sys.exit(0)
    
    # === STEP 0 ====
    """get data"""
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
    """preprocess data"""
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

    # === STEP 2 ====
    """quality filter"""
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

    # === STEP 3 ====
    """Response segments/ideas"""
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

    # === STEP 4 ====
    """Generate embeddings"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "embeddings"
    embedded_text = step_4_generate_embeddings(
        encoded_text, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        model_config=model_config,
        force_recalc=force_recalc,
        verbose=VERBOSE
    )
    check_execution_stop(4)

    # === STEP 5 ==== 
    """Reduce data/get clusters"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "initial_clusters"
    initial_cluster_results = step_5_cluster(
        embedded_text, filename,
        variable_key=variable_key,
        cache_manager=cache_manager,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        var_lab = var_lab
    )
    check_execution_stop(5)

    # === STEP 6 ====
    """Generate codes"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "codebook_generation"
    codebook_reasoning = step_6_generate_codebook(
        initial_cluster_results, filename, var_name, var_lab, variable_key, cache_manager, model_config,
        use_speculative_starter_codes=USE_SPECULATIVE_STARTER_CODES,
        force_recalc=force_recalc, verbose=VERBOSE, verbose_detailed=False,
        prompt_printer_enabled=PROMPT_PRINTER, cache_reasoning=True
    )
    check_execution_stop(6)

    # === STEP 7 ====
    """Codebook Refinement"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "codebook_refinement"
    refinement_results, theme_enriched_codebook, refinement_report, step7_prompt_printer = step_7_refine_codebook(
        codebook_reasoning, filename, var_name, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        model_config=model_config,
        default_language=DEFAULT_LANGUAGE,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=False  # Set to True to print prompt with assignment_examples
    )
    check_execution_stop(7)

    # === STEP 8 ====
    """Assign codes (and themes)"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "code_assignment_direct"
    code_assigned_results, stats, code_assigner_instance = step_8_assign_codes(
        filename,
        variable_key,
        cache_manager,
        theme_enriched_codebook,
        var_lab,
        model_config=model_config,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=True
    )
    check_execution_stop(8)

    # Print codebook summary
    for idx, entry in enumerate(theme_enriched_codebook.codes, start=1):
        print(f"{idx}) {entry.code}")

    # === STEP 9  =====
    """Export Results"""
    excel_path = step_9_export_results(
        code_assigned_results,
        theme_enriched_codebook,
        filename,
        var_name,
        quality_filtered_text=quality_filtered_text,
        verbose=VERBOSE,
        include_visualizations=INCLUDE_VISUALIZATIONS,
        cache_manager=cache_manager,
        variable_key=variable_key
    )
    
    # Pipeline completed successfully
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("All steps (0-9) executed")
    print(f"Results exported to: {excel_path}")
    print(f"{'='*80}\n")

    # Print LLM usage summary
    if token_tracker.call_count > 0:
        print(token_tracker.get_summary())

    # Save captured verbose output
    verbose_capture.__exit__(None, None, None)

# %%
