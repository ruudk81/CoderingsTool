#%%
"""
Step 2: Quality Filter Experiment Runner

Runs the quality filtering step in isolation for experimentation.
Loads Step 1 (preprocessed) results from cache and runs quality filtering.

Usage:
    cd src && python -m development.step_2_qualityFilter.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental qualityFilter from this folder
    USE_EXPERIMENTAL = False -> Uses production qualityFilter from utils/
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


USE_EXPERIMENTAL = True   # Toggle between production and experimental
EXPERIMENT_N     = None   # n or None (limit responses for quick experiments)

# =============================================================================
# SHARED IMPORTS (from production)
# =============================================================================
import models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils import dataLoader

# Import centralized test data config
try:
    from development.test_data import TEST_DATA
except ImportError:
    # Fallback for direct execution
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
@dataclass
class ExperimentConfig:
    # Data config from centralized test_data.py
    filename: str = TEST_DATA.filename
    id_column: str = TEST_DATA.id_column
    var_name: str = TEST_DATA.var_name
    sample_size: Optional[int] = TEST_DATA.sample_size
    # Experiment-specific settings
    use_experimental: bool = USE_EXPERIMENTAL
    verbose: bool = True
    prompt_printer_enabled: bool = False
    force_recalc: bool = True
    experiment_n: Optional[int] = EXPERIMENT_N


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
_USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if _USE_EXPERIMENTAL:
    try:
        from .qualityFilter_exp import Grader
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from qualityFilter_exp import Grader
    print("[EXPERIMENTAL] Using qualityFilter_exp.py from development folder")
else:
    from utils.qualityFilter import Grader
    print("[PRODUCTION] Using qualityFilter.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step1_cache(config: ExperimentConfig):
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


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    preprocessed_text, variable_key, cache_manager = load_step1_cache(config)
    var_lab = get_var_lab(config)

    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(enabled=config.prompt_printer_enabled, print_realtime=config.prompt_printer_enabled)

    verbose_reporter.section_header("QUALITY FILTERING EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {_USE_EXPERIMENTAL}")
    verbose_reporter.stat_line(f"Input: {len(preprocessed_text)} preprocessed responses")

    # Optionally limit to experiment_n responses
    if config.experiment_n is not None and config.experiment_n < len(preprocessed_text):
        preprocessed_text = preprocessed_text[:config.experiment_n]
        verbose_reporter.stat_line(f"Experiment subset: {config.experiment_n} responses")

    verbose_reporter.stat_line(f"Processing: {len(preprocessed_text)} responses")

    start_time = time.time()

    grader = Grader(preprocessed_text, var_lab, verbose=config.verbose, prompt_printer=prompt_printer)
    quality_filtered_text = grader.grade()

    elapsed_time = time.time() - start_time

    cache_manager.save_to_cache(quality_filtered_text, config.filename, "quality_filter", variable_key, elapsed_time, var_lab=var_lab)

    # Summary
    code_counts = {}
    for item in quality_filtered_text:
        code = item.quality_filter_code
        if code is not None:
            code_counts[code] = code_counts.get(code, 0) + 1

    verbose_reporter.stat_line(f"Filtered: {sum(code_counts.values())} responses")
    verbose_reporter.stat_line(f"Passed: {len(quality_filtered_text) - sum(code_counts.values())} responses")
    print(f"\n'Quality filtering experiment' completed in {elapsed_time:.2f} seconds.\n")

    return quality_filtered_text


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    config = EXPERIMENT_CONFIG
    var_lab = get_var_lab(config)

    verbose_capture = VerboseCapture(
        filename=config.filename,
        variable_key=config.var_name,
        sample_size=config.sample_size,
        run_until_step=2
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("EXPERIMENT: Step 2 - Quality Filter")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {_USE_EXPERIMENTAL}")
    print(f"Experiment N: {config.experiment_n or 'all'}")
    print("=" * 70)

    try:
        results = run_experiment(config)

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)
