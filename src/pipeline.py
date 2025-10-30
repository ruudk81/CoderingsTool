import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# ===  IMPORTS ======================================================================================================== 
import time
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

import models
from utils import dataLoader
from utils.cacheManager import CacheManager
from config import CacheConfig, ModelConfig, DEFAULT_LANGUAGE
cache_config = CacheConfig()
cache_manager = CacheManager(cache_config)
model_config = ModelConfig()

#  ===  STANDALONE ======================================================================================================== 
# filename = "M250285 input voor coderen - met Q18Q19.sav"
# id_column = "respondentid"
# var_name = "q19"
# var_name = "Q18Q19"

# filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
# id_column = "DLNMID"
# var_name = "Q20"

filename = "M000000 Associatiemonitor Merk X net databestand.sav"
id_column = "DLNMID"
var_name = "Qd1_combined"
sample_size = 2000

# filename = "M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
# id_column = "DLNMID"
# var_name = "Q15"

# filename = "M250127 Flitspeiling NAVOtop 0meting_153832.sav"
# id_column = "DLNMID"
# var_name = "Q10"
# sample_size = 50

RUN_UNTIL_STEP = 5 # None = run all steps
FORCE_RECALCULATE_ALL = False
VERBOSE = True
PROMPT_PRINTER = False
LANGUAGE = "nl"

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
    streamlit_container=None          # NEW: Optional progress updates
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
        data_loader_inst = dataLoader.DataLoader(verbose=verbose)

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
    from config import SpellCheckConfig

    step_name = "preprocessed"

    # Auto-generate variable_key if not provided
    if variable_key is None:
        # Try to infer from context
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    # Use global model_config if not provided
    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

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

    # Auto-generate variable_key if not provided
    if variable_key is None:
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    # Use global model_config if not provided
    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

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

    # Auto-generate variable_key if not provided
    if variable_key is None:
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    # Use global model_config if not provided
    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

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
    from config import EmbeddingConfig
    from utils.embedder import Embedder
    from utils.verboseReporter import VerboseReporter

    step_name = "embeddings"

    # Auto-generate variable_key if not provided
    if variable_key is None:
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    # Use global model_config if not provided
    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

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
        embedding_config = EmbeddingConfig()
        get_embeddings = Embedder(
            config=embedding_config,
            model_config=model_config,
            provider="openai",
            verbose=verbose)
        input_data = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
        embedded_text = get_embeddings.get_embeddings_with_tracking(input_data, var_lab)

        end_time = time.time()
        elapsed_time = end_time - start_time
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
    from utils.verboseReporter import VerboseReporter
    from config import HDBSCANConfig, DEFAULT_HDBSCAN_CONFIG, DEFAULT_UMAP_CONFIG, DEFAULT_CLUSTERING_CONFIG

    step_name = "initial_clusters"

    # Auto-generate variable_key if not provided
    if variable_key is None:
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Clustering ideas with UMAP + HDBSCAN...")
    verbose_reporter = VerboseReporter(verbose)

    CLUSTERING_ALPHA = HDBSCANConfig.alpha
    CLUSTERING_EPSILON = HDBSCANConfig.cluster_selection_epsilon

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

        # Create custom HDBSCAN config if overrides specified
        hdbscan_config = None
        if CLUSTERING_ALPHA is not None or CLUSTERING_EPSILON is not None:
            hdbscan_config = HDBSCANConfig(
                min_cluster_size=DEFAULT_HDBSCAN_CONFIG.min_cluster_size,
                min_samples=DEFAULT_HDBSCAN_CONFIG.min_samples,
                cluster_selection_epsilon=DEFAULT_HDBSCAN_CONFIG.cluster_selection_epsilon,
                alpha=DEFAULT_HDBSCAN_CONFIG.alpha,
                metric=DEFAULT_HDBSCAN_CONFIG.metric,
                cluster_selection_method=DEFAULT_HDBSCAN_CONFIG.cluster_selection_method,
                prediction_data=DEFAULT_HDBSCAN_CONFIG.prediction_data,
                approx_min_span_tree=DEFAULT_HDBSCAN_CONFIG.approx_min_span_tree,
                gen_min_span_tree=DEFAULT_HDBSCAN_CONFIG.gen_min_span_tree,
                merge_similar_clusters=True,
                merge_similarity_threshold=0.95
            )

        clusterer = Clusterer(
            embedded_text,
            umap_config=DEFAULT_UMAP_CONFIG,
            clustering_config=DEFAULT_CLUSTERING_CONFIG,
            hdbscan_config=hdbscan_config,
            verbose=verbose
        )
        clusterer.run()
        initial_cluster_results = clusterer.to_cluster_model()

        end_time = time.time()
        elapsed_time = end_time - start_time
        cache_manager.save_to_cache(initial_cluster_results, filename, step_name, variable_key, elapsed_time, var_lab=var_lab)
        print(f"\n'Initial clustering' completed in {elapsed_time:.2f} seconds.")

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
        tuple: (codebook_main: CodebookModel, codebook_reasoning: CodeGeneratorReasoningResults or None)
    """
    from utils import speculativeStarterCodes, codeGenerator, verboseReporter, promptPrinter
    
    step_name = "codebook_generation"

    # Auto-generate variable_key if not provided
    if variable_key is None:
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    # Use global model_config if not provided
    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Generating codebook from clusters...")
    verbose_reporter = verboseReporter.VerboseReporter(verbose)
    prompt_printer = promptPrinter.PromptPrinter(enabled=prompt_printer_enabled, print_realtime=True)
    codebook_reasoning = None

    if not force_recalc and cache_manager.is_cache_valid(filename, step_name, variable_key):
        codebook_models = cache_manager.load_from_cache(filename, step_name, variable_key, models.CodebookModel)
        if codebook_models and len(codebook_models) > 0:
            codebook_main = codebook_models[0]
            verbose_reporter.summary("CODEBOOK FROM CACHE", {
                "Total codes": len(codebook_main.codes),
                "Source variable": codebook_main.source_variable
            })

            # Optional Streamlit success message
            if streamlit_container:
                streamlit_container.success(f"✅ Codebook generation completed (from cache): {len(codebook_main.codes)} codes")

            # Extract legacy codebook list for backward compatibility
            codebook = [models.Codebook(code=entry.code, definition=entry.definition)
                        for entry in codebook_main.codes]

            # Load reasoning cache if flag is enabled
            if cache_reasoning:
                try:
                    reasoning_models = cache_manager.load_from_cache(
                        filename, f"{step_name}_reasoning", variable_key, codeGenerator.CodeGeneratorReasoningResults
                    )
                    if reasoning_models and len(reasoning_models) > 0:
                        codebook_reasoning = reasoning_models[0]
                        print("[OK] Loaded codebook reasoning from cache")
                    else:
                        print("Note: Reasoning cache not found (run with CACHE_CODEGENERATOR_REASONING=True to create)")
                except Exception as e:
                    print(f"Warning: Failed to load reasoning cache: {e}")
        else:
            print("ERROR: Failed to load codebook from cache")
            codebook_main = models.CodebookModel(codes=[], source_variable=var_name)
            codebook = []
    else:
        verbose_reporter.section_header("CODEBOOK GENERATION PHASE")
        start_time = time.time()

        # Phase 1: Generate starter codes (optional)
        if use_speculative_starter_codes:
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
            codebook_main = models.CodebookModel(
                codes=[],
                generation_metadata={"error": "Failed to generate starter codes"},
                source_variable=var_name
            )
            codebook = []
            results = {}  # Empty results for caching check
        else:
            # Phase 2: Inductive code generation
            generator = codeGenerator.InductiveCodeGenerator(
                cluster_results=initial_cluster_results,
                starter_codes=starter_codes,
                var_lab=var_lab,
                verbose=True,
                verbose_detailed=verbose_detailed,
                prompt_printer=prompt_printer
            )
            results = generator.generate()

            codebook_entries = []
            codebook = []  # Legacy format for backward compatibility

            if results and isinstance(results, codeGenerator.CodeGeneratorReasoningResults):
                # Use the deduplicated codebook directly from results
                final_codebook = results.codebook

                # Display final codebook summary
                if verbose and final_codebook:
                    verbose_reporter.empty_line()
                    print("[STATS] FINAL CODEBOOK SUMMARY")
                    verbose_reporter.stat_line(f"Total codes: {len(final_codebook)}")

                    # Show sample codes
                    verbose_reporter.empty_line()
                    print("[LIST] Complete codebook:")

                idx = 1
                # Process the extracted final codebook
                for item in final_codebook:
                    if verbose:
                        definition = item['definition']
                        if len(definition) > 100:
                            definition = definition[:97] + "..."
                        print(f"  {idx}. {item['code']}")

                    codebook_entry = models.CodebookEntry(
                        code=item['code'],
                        definition=item['definition'],
                        source_cluster=item['source_cluster_id']
                    )
                    codebook_entries.append(codebook_entry)

                    idx += 1
            else:
                print("Warning: Codebook generator returned no results")

            codebook_main = models.CodebookModel(
                codes=codebook_entries,
                generation_metadata={
                    "methodology": "Inductive codebook generation from clusters",
                    "starter_codes_count": len(starter_codes) if starter_codes else 0,
                    "total_codes_generated": len(codebook_entries),
                },
                source_variable=var_name
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        if 'codebook_main' not in locals():
            print("ERROR: codebook_main was not created!")
            codebook_main = models.CodebookModel(
                codes=[],
                generation_metadata={"error": "Failed to create codebook model"},
                source_variable=var_name
            )

        cache_manager.save_to_cache([codebook_main], filename, step_name, variable_key, elapsed_time, var_lab=var_lab)

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

        print(f"\n'codebook generation' completed in {elapsed_time:.2f} seconds.\n")

        # Optional Streamlit success message
        if streamlit_container:
            num_codes = len(codebook_main.codes) if codebook_main else 0
            streamlit_container.success(f"✅ Codebook generation completed in {elapsed_time:.2f}s: {num_codes} codes")

    return codebook_main, codebook_reasoning


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
        streamlit_container: Optional Streamlit container for progress updates

    Returns:
        tuple: (refinement_results: CodeRefinementResults, theme_enriched_codebook: ThemeEnrichedCodebookModel, refinement_report: dict)
    """
    from utils.codebookRefinement import refine_codebook, print_refinement_report, get_refinement_report
    from utils.verboseReporter import VerboseReporter

    step_name = "codebook_refinement"

    # Auto-generate variable_key if not provided
    if variable_key is None:
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

    # Use global model_config if not provided
    if model_config is None:
        model_config = globals().get('model_config')
        if model_config is None:
            from config import ModelConfig
            model_config = ModelConfig()

    # Use DEFAULT_LANGUAGE if not provided
    if default_language is None:
        from config import DEFAULT_LANGUAGE as DEFAULT_LANG
        default_language = DEFAULT_LANG

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Refining codebook into hierarchical themes...")
    verbose_reporter = VerboseReporter(verbose)
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
            verbose_reporter.step_start("GPT-5 Refinement", "Refining raw codes into hierarchical structure")

            # Run refinement using simple sync call
            refinement_results = refine_codebook(
                survey_question=var_lab,
                reasoning_results=codebook_reasoning,
                model_config=model_config,
                language=default_language,
                verbose=verbose
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

        # Create ThemeEnrichedCodebookEntry objects from refined codebook
        enriched_entries = []
        code_to_theme_mapping = {}
        themes_summary = []

        for category in refinement_results.refined_codebook.refined_codebook:
            theme_name = category.category

            # Add to themes summary
            themes_summary.append({
                'theme_name': theme_name,
                'theme_description': theme_name,  # Use theme name as description
                'code_count': len(category.subcodes)
            })

            for subcode in category.subcodes:
                # Create ThemeEnrichedCodebookEntry with category support (3-level hierarchy)
                enriched_entry = models.ThemeEnrichedCodebookEntry(
                    code=subcode.code,
                    definition=subcode.description,
                    theme=theme_name,
                    theme_description=theme_name,
                    category=subcode.category,  # Empty string for 2-level, category name for 3-level
                    category_description=subcode.category if subcode.category else "",  # Use category name as description
                    source_cluster=subcode.id  # Use original code ID as source cluster
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

    return refinement_results, theme_enriched_codebook, refinement_report


def step_8_assign_codes(
    initial_cluster_results,
    theme_enriched_codebook,
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
    """Step 8: Assign codes to individual ideas

    Args:
        initial_cluster_results: List of ClusterModel instances from step 5
        theme_enriched_codebook: ThemeEnrichedCodebookModel from step 7
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
        Tuple[List[models.CodeAssignedModel], Dict]:
            - List of models with code assignments
            - Dictionary with stats (total_responses, total_ideas, unique_codes_assigned,
              unique_themes_assigned, total_code_assignments, total_theme_assignments,
              avg_codes_per_idea, avg_themes_per_idea, processing_time)
    """
    from utils import codeAssigner, verboseReporter, promptPrinter

    step_name = "code_assignment_direct"

    # Auto-generate variable_key if not provided
    if variable_key is None:
        selected_variables = globals().get('selected_variables', [])
        is_merged = globals().get('is_merged', False)
        sample_size = globals().get('sample_size', None)
        merge_config = globals().get('merge_config', None)

        from utils.cacheManager import generate_enhanced_variable_key
        variable_key = generate_enhanced_variable_key(
            selected_variables if selected_variables else ["unknown"],
            is_merged,
            sample_size=sample_size,
            merge_config=merge_config
        )

    # Use global cache_manager if not provided
    if cache_manager is None:
        cache_manager = globals().get('cache_manager')
        if cache_manager is None:
            from utils.cacheManager import CacheManager
            from config import CacheConfig
            cache_manager = CacheManager(CacheConfig())

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

            # Create simplified code assigner without embeddings
            code_assigner_instance = codeAssigner.CodeAssigner(
                cluster_models=initial_cluster_results,  # Use original cluster models
                codebook=[models.Codebook(
                    code=entry.code,
                    definition=entry.definition,
                    theme=entry.theme,
                    theme_description=entry.theme_description
                ) for entry in theme_enriched_codebook.codes],
                var_lab=var_lab,
                code_to_theme_mapping=theme_enriched_codebook.code_to_theme_mapping,
                cached_idea_embeddings=None,
                model_config=model_config,
                verbose=verbose,
                prompt_printer=prompt_printer
            )

            code_assigned_results = code_assigner_instance.assign()

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

    return code_assigned_results, stats


def step_9_export_results(
    code_assigned_results,
    theme_enriched_codebook,
    filename,
    var_name,
    quality_filtered_text=None,
    verbose=True,
    streamlit_container=None        # Optional progress updates
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

    Returns:
        str: Path to exported Excel file
    """
    from utils.resultsExporter import ResultsExporter

    # Optional Streamlit progress
    if streamlit_container:
        streamlit_container.text("🔄 Exporting results to Excel...")

    try:
        exporter = ResultsExporter(verbose=verbose)
        excel_path = exporter.export_to_excel(
            code_assigned_results,
            theme_enriched_codebook,
            filename,
            var_name,
            quality_filtered_text=quality_filtered_text,
            export_dir=None  # Will create default export directory
        )
        print(f"[SUCCESS] Code assignments exported to Excel: {excel_path}")

        # Optional Streamlit success message
        if streamlit_container:
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
    
    USE_SPECULATIVE_STARTER_CODES = False
    data_loader = dataLoader.DataLoader(verbose=False)
    var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
    
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
    
    if False: #debug if true
        import random
        n_samples = 5
        indices = random.sample(range(len(preprocessed_text)), n_samples)
        for i in indices:
            print("Raw structured:", raw_text_list[i])
            print("---")        
    
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
    
    # debug if true
    if False : 
        import random
        n_samples = 5
        filtered_text = [item.response for item in quality_filtered_text if item.quality_filter]
        indices = random.sample(range(len(filtered_text)), n_samples)
        for i in indices:
            print(filtered_text[i])
    
    
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
    
    if False : # debug if true
        import random
        n_samples = 1
        sampled_items = random.sample(encoded_text, n_samples)
        for item in sampled_items:
            print(item.response)
            for segment in item.response_ideas:
                print(f"- {segment.idea}")
    
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
    
    if False: #debug if true
        import random
        n_samples = 1
        sampled_items = random.sample(embedded_text, n_samples)
        for item in sampled_items:
            print(f"{item.response}\n")
            for segment in item.response_ideas:
                print(f"- {segment.idea}")
    
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
    
    if False: #debug - print random clusters  
        import random
        cluster_ids = list(set([
            response_idea.initial_cluster 
            for result in initial_cluster_results 
            for response_idea in result.response_ideas   
            if response_idea.initial_cluster is not None]))
        sampled_cluster = random.sample(cluster_ids, 1)[0]
        print(f"\nCluster {sampled_cluster}:\n")
        cluster_segments = []
        for result in initial_cluster_results:
            for response_idea in result.response_ideas:   
                if response_idea.initial_cluster == sampled_cluster:
                    cluster_segments.append(response_idea.idea)
        sampled_segments = random.sample(cluster_segments, min(10, len(cluster_segments)))
        for segment_desc in sampled_segments:
            print(f"-    {segment_desc}")
        
    if False: #debug if true - print all clusters
        cluster_ids = list(set([
            response_idea.initial_cluster 
            for result in initial_cluster_results 
            for response_idea in result.response_ideas  # This has initial_cluster
            if response_idea.initial_cluster is not None]))
        for x in range(1, round(len(cluster_ids) / 1) + 1):
            y = x * 1
            print(f"\n=== Showing clusters {y-1} to {min(y, len(cluster_ids)-1)} ===\n")
        
            for z in range(y - 1, y):
                if z < len(cluster_ids):
                    print(f"\nCluster {z}")
                    for item in initial_cluster_results:
                        for subitem in item.response_ideas:
                            if subitem.initial_cluster == z:
                                print(subitem.idea)
            input("\n🔸 Press Enter to continue to the next batch of clusters...")
    
    # === STEP 6 ====
    """Generate codes"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "codebook_generation"
    codebook_main, codebook_reasoning = step_6_generate_codebook(
        initial_cluster_results, filename, var_name, var_lab, variable_key, cache_manager, model_config,
        use_speculative_starter_codes=USE_SPECULATIVE_STARTER_CODES,
        force_recalc=force_recalc, verbose=VERBOSE, verbose_detailed=False,
        prompt_printer_enabled=PROMPT_PRINTER, cache_reasoning=True
    )
    check_execution_stop(6)
    
    if False: #debug if true (reasoning)
        if codebook_reasoning is not None:
            from utils.codegenResults import display_cluster_analysis
            display_cluster_analysis(codebook_reasoning)
        else:
            print("Note: codebook_reasoning not available for display")
    
    if False: #debug if true (prompts + reasoning)
        import random
        step3_recommendations = getattr(codebook_reasoning, 'step3_recommendations', {})
        step3_recommendations = codebook_reasoning.step3_recommendations
        available_ids = list(step3_recommendations.keys())
        cluster_id = random.choice(available_ids)

    
        from utils import codegenPromptTester
        tester = codegenPromptTester.SimplePromptTester(cluster_id = cluster_id, var_lab=var_lab)
        tester.test_prompt_1()
        tester.test_prompt_2()
        tester.test_prompt_3()
        tester.test_prompt_4()
    
        if codebook_reasoning is not None:
            from utils.codegenResults import display_cluster_analysis
            display_cluster_analysis(codebook_reasoning, cluster_id = cluster_id)
        else:
            print("Note: codebook_reasoning not available for display")
    
    # === STEP 7 ====
    """Codebook Refinement"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "codebook_refinement"
    refinement_results, theme_enriched_codebook, refinement_report = step_7_refine_codebook(
        codebook_reasoning, filename, var_name, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        model_config=model_config,
        default_language=DEFAULT_LANGUAGE,
        force_recalc=force_recalc,
        verbose=VERBOSE
    )
    check_execution_stop(7)
    
    if False: #debug
        final_codebook = refinement_results.refined_codebook
        for entry in final_codebook.refined_codebook:
            print(entry.category)
            for x in  entry.subcodes:
                print(f"- {x.code}")
            print("\n")
                    
    # === STEP 8 ====
    """Assign codes (and themes)"""
    force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == "code_assignment_direct"
    code_assigned_results, stats = step_8_assign_codes(
        initial_cluster_results, theme_enriched_codebook, filename, var_lab,
        variable_key=variable_key,
        cache_manager=cache_manager,
        model_config=model_config,
        force_recalc=force_recalc,
        verbose=VERBOSE,
        prompt_printer_enabled=PROMPT_PRINTER
    )
    check_execution_stop(8)
    
    # codebook
    for idx, entry in enumerate(theme_enriched_codebook.codes, start=1):
        print(f"{idx}) {entry.code}")
    
    # assignment stats 
    if False:
        from utils.pipelineSummarizer import PipelineSummarizer
        summarizer = PipelineSummarizer(verbose=True)
        summarizer.generate_summary(
            code_assigned_results=code_assigned_results if 'code_assigned_results' in locals() else None,
            theme_enriched_codebook=theme_enriched_codebook if 'theme_enriched_codebook' in locals() else None)
        
    # random assignments
    if False: #debug
        import random
        sampled_result = random.choice(code_assigned_results)
        print(f"Respondent ID: {sampled_result.respondent_id}")
        print(f"Response: {sampled_result.response}")
        #print(f"Idea count: {sampled_result.idea_count}")
        #print(f"Codebook: {sampled_result.assignment_metadata.get('codebook_used')}")
        #print("---- Assigned Codes ----")
        for idea in sampled_result.response_ideas:
            print("-" * 40)
            print(f"Idea ID: {idea.idea_id}")
            print(f"Idea: {idea.idea}")
            print(f"Assigned Codes: {', '.join(idea.assigned_codes)}")
            #print(f"Assigned Themes: {', '.join(idea.assigned_themes)}")
            print(f"Rationale: {idea.assignment_rationale}")
            print(f"Assignment Confidence: {idea.assignment_confidence}")
            print("-" * 40)
    
    # random prompt
    if False: #debug
    
        print("\n" + "="*80)
        print("RANDOM PROMPT TESTING (DEBUG)")
        print("="*80)
    
        # Extract ideas directly from initial_cluster_results
        all_ideas_for_debug = []
        for result in initial_cluster_results:
            if result.response_ideas:
                for idea in result.response_ideas:
                    all_ideas_for_debug.append({
                        'idea_id': idea.idea_id,
                        'idea': idea.idea,
                        'respondent_id': result.respondent_id
                    })
    
        
        from prompts import CODE_ASSIGNMENT_PROMPT
        
        if 'code_assigned_results' in locals() and 'all_ideas_for_debug' in locals() and all_ideas_for_debug:
            # Pick random idea from debug data
            random_idea = random.choice(all_ideas_for_debug)
            
            # Get idea details
            idea_id = random_idea['idea_id']
            idea_text = random_idea['idea']
            respondent_id = random_idea['respondent_id']
            
            print("[TARGET] Random Selected Idea:")
            print(f"  ID: {idea_id}")
            print(f"  Respondent: {respondent_id}")
            print(f"  Position: {all_ideas_for_debug.index(random_idea) + 1} of {len(all_ideas_for_debug)}")
            print(f"  Text ({len(idea_text)} chars): {idea_text}")
            
            # Get first 5 codes as candidate codes (simplified for demo)
            if 'theme_enriched_codebook' in locals() and theme_enriched_codebook.codes:
                similar_codes = theme_enriched_codebook.codes # First 5 codes as example
                
                # print("\nCandidate Codes (first 5):")
                # for j, code in enumerate(similar_codes, 1):
                #     print(f"  {j}. {code.code}: {code.definition}")
                
                # Format candidate codes for prompt (match CodeAssigner format)
                candidate_codes_text = "\n".join([
                    f"Code label: {code.code}\nCode description: {code.definition}\n" 
                    #f"Code: {code.definition}\n"
                    for code in similar_codes
                ])
                
                # Create prompt using same logic as CodeAssigner
                prompt = CODE_ASSIGNMENT_PROMPT.format(
                    language="Dutch",  # Match pipeline language
                    var_lab=var_lab,
                    idea_id=idea_id,
                    idea_text=idea_text,
                    candidate_codes=candidate_codes_text
                )
                
                print(f"\n{'='*60}")
                print("FORMATTED PROMPT:")
                print(f"{'='*60}")
                print(prompt)
                #print("="*60)
                
                for result in code_assigned_results:
                    segments = result.response_ideas
                    for segment in segments:
                        if segment.idea_id == idea_id:
                            print(f"\n{'='*60}")
                            print("llM RESPNSE:")
                            print(f"{'='*60}")
                            # print(f"Response: {segment.idea_id}")
                            print(f"Response: {segment.idea}")
                            print("Assigned code:\n", "".join(segment.assigned_codes))
                            print(f"\nReasoning:\n {segment.assignment_rationale}")
                            print(f"\nConfidence: {segment.assignment_confidence}")
                            #print("\n")
                
            else:
                print("ERROR: No codebook available for prompt generation")
        else:
            print("ERROR: Missing code_assigned_results or all_ideas_for_debug for random prompt test")
       
    # === STEP 9  =====
    """Export Results"""
    excel_path = step_9_export_results(code_assigned_results, theme_enriched_codebook, filename, var_name, quality_filtered_text=quality_filtered_text, verbose=VERBOSE)
    
    # Pipeline completed successfully
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("All steps (0-9) executed")
    print(f"Results exported to: {excel_path}")
    print(f"{'='*80}\n")