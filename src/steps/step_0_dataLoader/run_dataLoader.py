#%%

"""
Step 0: Data Loader Step Runner

Loads raw data from an SPSS file and writes it to cache as List[ResponseModel].
This seeds the cache for all downstream steps (step_1, step_2, etc.)
without requiring pipeline.py to run first.

Always uses the production dataLoader -- there is no step-specific variant.

Usage:
    cd src && python -m steps.step_0_dataLoader.run_dataLoader
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

import pandas as pd
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# SHARED IMPORTS
# =============================================================================
import models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils import dataLoader

# Import centralized test data config
try:
    from steps.test_data import TEST_DATA
except ImportError:
    # Fallback for direct execution
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
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
    # Runner settings
    verbose: bool = True
    force_recalc: bool = True


STEP_CONFIG = StepConfig()


# =============================================================================
# HELPERS
# =============================================================================
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
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())
    step_name = "data"

    verbose_reporter = VerboseReporter(config.verbose)
    verbose_reporter.section_header("DATA LOADING")

    # Check cache (unless force_recalc)
    if not config.force_recalc and cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raw_text_list = cache_manager.load_from_cache(config.filename, step_name, variable_key, models.ResponseModel)
        verbose_reporter.summary("DATA FROM CACHE", {"Input": f"{len(raw_text_list)} responses"})
        return raw_text_list

    start_time = time.time()

    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=config.verbose)
    raw_text_df = loader.get_variable_with_IDs(
        filename=config.filename,
        id_column=config.id_column,
        var_name=config.var_name,
    )
    text_column = config.var_name

    # Build ResponseModel list with type classification
    raw_unstructured = list(zip(
        [int(id_int) for id_int in raw_text_df[config.id_column].tolist()],
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
            respondent_id=resp_id,
            response=response_value,
            response_type=response_type
        ))

    # Apply sample size truncation
    original_count = len(raw_text_list)
    if config.sample_size and len(raw_text_list) > config.sample_size:
        raw_text_list = raw_text_list[:config.sample_size]
        verbose_reporter.stat_line(f"Truncated: {len(raw_text_list)} of {original_count} responses (sample_size={config.sample_size})")
    else:
        verbose_reporter.stat_line(f"No truncation: {len(raw_text_list)} responses (full dataset)")

    elapsed_time = time.time() - start_time

    cache_manager.save_to_cache(raw_text_list, config.filename, step_name, variable_key, elapsed_time, var_lab=None)

    # Data type summary
    type_counts = {'nan': 0, 'numeric': 0, 'string': 0, 'unknown': 0}
    for item in raw_text_list:
        type_counts[item.response_type] += 1

    print("\n=== RAW DATA TYPE ANALYSIS ===")
    for data_type, count in type_counts.items():
        print(f"  {data_type}: {count}")
    print(f"\n'Data loading' completed in {elapsed_time:.2f} seconds.\n")

    return raw_text_list


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
        run_until_step=0
    )
    verbose_capture.__enter__()

    print("=" * 70)
    print("Step 0 - Data Loader")
    print("=" * 70)
    print(f"Dataset:     {config.filename}")
    print(f"Variable:    {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Force recalc: {config.force_recalc}")
    print("=" * 70)

    try:
        results = run_step(config)

    finally:
        verbose_capture.__exit__(None, None, None)
