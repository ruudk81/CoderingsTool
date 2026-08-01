#%%
"""
Step 2: Quality Filter Step Runner

Runs the quality filtering step in isolation.
Loads Step 1 (preprocessed) results from cache and runs quality filtering.

Usage:
    cd src && python -m pipeline.step_2_qualityFilter.run_qualityFilter
"""

import sys
import time
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
data_dir = project_root / "data"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import nest_asyncio
nest_asyncio.apply()

from dataclasses import dataclass
from typing import Optional


EXPERIMENT_N     = None   # n or None (limit responses for quick testing)

# =============================================================================
# SHARED IMPORTS
# =============================================================================
import models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils.costTracker import CostTracker
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
    experiment_n: Optional[int] = EXPERIMENT_N


STEP_CONFIG = StepConfig()

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from .qualityFilter import Grader
except ImportError:
    exp_dir = Path(__file__).parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from qualityFilter import Grader


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step1_cache(config: StepConfig):
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    step_name = "preprocessed"

    if not cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raise FileNotFoundError(
            f"Cache not found: {step_name}/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=1 first."
        )

    data = cache_manager.load_from_cache(
        config.filename, step_name, variable_key, models.PreprocessedModel
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

    if not config.force_recalc and cache_manager.is_cache_valid(config.filename, "quality_filter", variable_key):
        quality_filtered_text = cache_manager.load_from_cache(
            config.filename, "quality_filter", variable_key, models.QualityFilteredModel
        )
        verbose_reporter = VerboseReporter(config.verbose)
        verbose_reporter.summary("QUALITY FILTER FROM CACHE", {"Input": f"{len(quality_filtered_text)} responses"})
        return quality_filtered_text

    preprocessed_text, variable_key, cache_manager = load_step1_cache(config)
    var_lab = get_var_lab(config)

    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(
        enabled=True,  # Always capture prompts for debugging
        print_realtime=config.prompt_printer_enabled  # Only print if requested
    )

    verbose_reporter.section_header("QUALITY FILTERING")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Input: {len(preprocessed_text)} preprocessed responses")

    # Optionally limit to experiment_n responses
    if config.experiment_n is not None and config.experiment_n < len(preprocessed_text):
        preprocessed_text = preprocessed_text[:config.experiment_n]
        verbose_reporter.stat_line(f"Subset: {config.experiment_n} responses")

    verbose_reporter.stat_line(f"Processing: {len(preprocessed_text)} responses")

    cost_tracker = CostTracker(filename=config.filename, variable_key=variable_key)

    start_time = time.time()

    grader = Grader(preprocessed_text, var_lab, verbose=config.verbose, prompt_printer=prompt_printer, cost_tracker=cost_tracker)
    quality_filtered_text = grader.grade()

    elapsed_time = time.time() - start_time

    cost_tracker.finalize_step("step_2_quality_filter")

    # Save captured prompts to JSON
    if prompt_printer.prompts:
        prompts_dir = project_root / "exports" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompts_file = prompts_dir / f"step2_{config.var_name}_{variable_key}.json"
        prompt_printer.save_prompts(str(prompts_file))

    cache_manager.save_to_cache(quality_filtered_text, config.filename, "quality_filter", variable_key, elapsed_time, var_lab=var_lab)

    # Summary
    code_counts = {}
    for item in quality_filtered_text:
        code = item.quality_filter_code
        if code is not None:
            code_counts[code] = code_counts.get(code, 0) + 1

    verbose_reporter.stat_line(f"Filtered: {sum(code_counts.values())} responses")
    verbose_reporter.stat_line(f"Passed: {len(quality_filtered_text) - sum(code_counts.values())} responses")
    print(f"\n'Quality filtering' completed in {elapsed_time:.2f} seconds.\n")

    return quality_filtered_text


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    config = STEP_CONFIG
    var_lab = get_var_lab(config)

    variable_key = generate_enhanced_variable_key(
        [config.var_name], False, config.sample_size
    )

    verbose_capture = VerboseCapture(
        filename=config.filename,
        var_name=config.var_name,
        sample_size=config.sample_size,
        step=2
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("Step 2 - Quality Filter")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Experiment N: {config.experiment_n or 'all'}")
    print("=" * 70)

    try:
        results = run_step(config)

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)
