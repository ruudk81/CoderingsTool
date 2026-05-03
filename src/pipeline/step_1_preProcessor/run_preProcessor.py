#%%

"""
Step 1: Preprocess Step Runner

Runs the preprocessing step in isolation.
Loads Step 0 (data) results from cache and runs preprocessing.

Usage:
    cd src && python -m steps.step_1_preProcessor.run_preProcessor
"""

import sys
import time
from pathlib import Path

# Path setup
src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
data_dir = project_root / "data"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import nest_asyncio
nest_asyncio.apply()

from dataclasses import dataclass
from typing import Optional

# =============================================================================
# SHARED IMPORTS
# =============================================================================
import models
from config import CacheConfig
from pipeline.step_1_preProcessor.config_preProcessor import SpellCheckConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils import dataLoader

from test_data import TEST_DATA

# =============================================================================
# STEP CONFIGURATION
# =============================================================================
@dataclass
class StepConfig:
    # Data config from centralized test_data.py
    filename: str = TEST_DATA.filename
    id_column: str = TEST_DATA.id_column
    var_name: str = TEST_DATA.var_name
    sample_size: Optional[int] = TEST_DATA.sample_size
    # Step-specific settings
    verbose: bool = True
    prompt_printer_enabled: bool = False
    force_recalc: bool = True


STEP_CONFIG = StepConfig()

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from .spellChecker import SpellChecker
    from .textNormalizer import TextNormalizer
    from .textFinalizer import TextFinalizer
except ImportError:
    # Add step folder to path for direct execution
    exp_dir = Path(__file__).parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from spellChecker import SpellChecker
    from textNormalizer import TextNormalizer
    from textFinalizer import TextFinalizer


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step0_cache(config: StepConfig):
    """Load raw data from Step 0 cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    step_name = "data"

    if not cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raise FileNotFoundError(
            f"Cache not found: {step_name}/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=0 first to generate the cache."
        )

    data = cache_manager.load_from_cache(
        config.filename, step_name, variable_key, models.ResponseModel
    )
    return data, variable_key, cache_manager


def get_var_lab(config: StepConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN STEP RUNNER
# =============================================================================
def run_step(config: StepConfig = None):
    if config is None:
        config = STEP_CONFIG

    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size,
    )
    cache_manager = CacheManager(CacheConfig())

    if not config.force_recalc and cache_manager.is_cache_valid(config.filename, "preprocessed", variable_key):
        preprocessed_text = cache_manager.load_from_cache(
            config.filename, "preprocessed", variable_key, models.PreprocessedModel
        )
        verbose_reporter = VerboseReporter(config.verbose)
        verbose_reporter.summary("PREPROCESSED FROM CACHE", {"Input": f"{len(preprocessed_text)} responses"})
        return preprocessed_text

    raw_text_list, variable_key, cache_manager = load_step0_cache(config)
    var_lab = get_var_lab(config)

    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(enabled=config.prompt_printer_enabled, print_realtime=config.prompt_printer_enabled)
    spell_check_config = SpellCheckConfig(minimum_timeout_seconds=15.0, maximum_timeout_seconds=60.0)

    verbose_reporter.section_header("PREPROCESSING")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Input: {len(raw_text_list)} responses")

    start_time = time.time()

    # Initialize utils
    text_normalizer = TextNormalizer(verbose=config.verbose)
    spell_checker = SpellChecker(config=spell_check_config, verbose=config.verbose, prompt_printer=prompt_printer)
    text_finalizer = TextFinalizer(verbose=config.verbose)

    # Process strings
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

        # Show detailed correction examples after spell checking
        if hasattr(spell_checker, 'correction_examples') and spell_checker.correction_examples:
            print(f"\n  Sample corrections (showing up to 5):")
            for i, (orig, corrected) in enumerate(spell_checker.correction_examples[:5]):
                orig_display = orig[:60] + "..." if len(orig) > 60 else orig
                corr_display = corrected[:60] + "..." if len(corrected) > 60 else corrected
                print(f'    {i+1}. "{orig_display}" -> "{corr_display}"')

            # Show both metrics
            responses_corrected = spell_checker.stats.get('responses_corrected', 0)
            unique_corrections = spell_checker.stats.get('corrections_applied', 0)
            print(f"\n  Total: {unique_corrections} unique corrections applied to {responses_corrected} responses")

        finalized_text = text_finalizer.finalize_responses(corrected_text)
    else:
        finalized_text = []

    # Build output
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
            elif isinstance(item.response, int) and item.response in [99999997, 99999998, 99999999]:
                desc_item.quality_filter_code = int(item.response)
                desc_item.quality_filter = True
            preprocessed_text.append(desc_item)
        else:
            preprocessed_text.append(models.PreprocessedModel(
                respondent_id=original.respondent_id,
                response='<NA>',
                response_type='nan',
                quality_filter_code=99999998,
                quality_filter=True
            ))

    elapsed_time = time.time() - start_time

    cache_manager.save_to_cache(preprocessed_text, config.filename, "preprocessed", variable_key, elapsed_time, var_lab=var_lab)

    verbose_reporter.stat_line(f"Output: {len(preprocessed_text)} preprocessed responses")
    print(f"\n'Preprocessing' completed in {elapsed_time:.2f} seconds.\n")

    return preprocessed_text


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    config = STEP_CONFIG
    var_lab = get_var_lab(config)

    variable_key = generate_enhanced_variable_key([config.var_name], False, config.sample_size)
    verbose_capture = VerboseCapture(
        filename=config.filename,
        variable_key=variable_key,
        sample_size=config.sample_size,
        run_until_step=1
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("Step 1 - Preprocess")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print("=" * 70)

    try:
        results = run_step(config)

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)
