#%%

"""
Step 7: Export Step Runner

Runs the export step in isolation.
Loads Step 6 (taxonomy_codes) results from cache and exports to Excel.

Usage:
    cd src && python -m steps.step_7_export.run_export
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

# =============================================================================
# SHARED IMPORTS
# =============================================================================
import models
from pipeline.step_6_codeAssigner.models_codeAssigner import CodeAssignedModel
from pipeline.step_5_codeGenerator.models_codeGenerator import CodingResultsCache
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils import dataLoader

# Import centralized test data config
try:
    from pipeline.test_data import TEST_DATA
except ImportError:
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
    # Step-specific settings
    verbose: bool = True
    force_recalc: bool = True


STEP_CONFIG = StepConfig()

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from .resultsExporter import ResultsExporter
except ImportError:
    exp_dir = Path(__file__).parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from resultsExporter import ResultsExporter


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step6_cache(config: StepConfig):
    """Load step 6 (taxonomy_codes) and step 5 (mece_codes) from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    # Load code assigned results from step 6
    if not cache_manager.is_cache_valid(config.filename, "taxonomy_codes", variable_key):
        raise FileNotFoundError(
            f"Cache not found: taxonomy_codes/{variable_key}\n"
            f"Run step 6 (codeAssigner) first."
        )

    code_assigned_results = cache_manager.load_from_cache(
        config.filename, "taxonomy_codes", variable_key, CodeAssignedModel
    )

    # Load codebook from step 5 (mece_codes metadata)
    mece_cache = cache_manager.load_metadata_from_cache(
        filename=config.filename,
        step="mece_codes",
        variable_key=variable_key,
        model_cls=CodingResultsCache,
    )
    if mece_cache is None:
        raise FileNotFoundError(
            f"Cache not found: mece_codes/{variable_key}\n"
            f"Run step 5 (codeGenerator) first."
        )

    # Reconstruct ConsolidatedCode objects from cached dicts
    codes = [ConsolidatedCode(**d) for d in mece_cache.raw_codes] if mece_cache.raw_codes else []
    partition_set = mece_cache.partition_set
    partition_results = mece_cache.partition_results

    # Load quality filtered text (optional)
    quality_filtered_text = None
    try:
        quality_filtered_text = cache_manager.load_from_cache(
            config.filename, "quality_filter", variable_key, models.QualityFilteredModel
        )
    except Exception:
        pass

    return code_assigned_results, codes, partition_set, partition_results, quality_filtered_text, variable_key


def get_var_lab(config: StepConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN STEP RUNNER
# =============================================================================
def run_step(config: StepConfig = None):
    if config is None:
        config = STEP_CONFIG

    code_assigned_results, codes, partition_set, partition_results, quality_filtered_text, variable_key = load_step6_cache(config)
    var_lab = get_var_lab(config)

    verbose_reporter = VerboseReporter(config.verbose)

    verbose_reporter.section_header("EXPORT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Input: {len(code_assigned_results)} assigned results, {len(codes)} codes")

    start_time = time.time()

    # Run export
    exporter = ResultsExporter(verbose=config.verbose)
    excel_path = exporter.export_to_excel(
        code_assigned_results,
        codes,
        partition_set,
        partition_results,
        config.filename,
        config.var_name,
        quality_filtered_text=quality_filtered_text,
        export_dir=None,
    )

    elapsed_time = time.time() - start_time

    verbose_reporter.stat_line(f"Output: {excel_path}")
    print(f"\n'Export' completed in {elapsed_time:.2f} seconds.\n")

    return excel_path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    config = STEP_CONFIG
    var_lab = get_var_lab(config)

    verbose_capture = VerboseCapture(
        filename=config.filename,
        variable_key=config.var_name,
        sample_size=config.sample_size,
        run_until_step=7
    )
    verbose_capture.__enter__()

    print("=" * 70)
    print("Step 7 - Export")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print("=" * 70)

    try:
        excel_path = run_step(config)

        print("\n" + "=" * 70)
        print("EXPORT COMPLETE")
        print("=" * 70)
        print(f"Excel file: {excel_path}")
        print("=" * 70)

    finally:
        verbose_capture.__exit__(None, None, None)
