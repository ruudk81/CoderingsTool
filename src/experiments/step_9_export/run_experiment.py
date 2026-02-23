#%%

"""
Step 9: Export Experiment Runner

Runs the export step in isolation for experimentation.
Loads Step 8 (code_assignment_direct) results from cache and exports to Excel.

Usage:
    cd src && python -m experiments.step_9_export.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental resultsExporter from this folder
    USE_EXPERIMENTAL = False -> Uses production resultsExporter from utils/
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
# SHARED IMPORTS (from production)
# =============================================================================
import models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils import dataLoader

# Import centralized test data config
try:
    from experiments.test_data import TEST_DATA
except ImportError:
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
    use_experimental: bool = True
    verbose: bool = True
    include_visualizations: bool = True
    force_recalc: bool = True


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .resultsExporter_exp import ResultsExporter
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from resultsExporter_exp import ResultsExporter
    print("[EXPERIMENTAL] Using resultsExporter_exp.py from experiments folder")
else:
    from utils.resultsExporter import ResultsExporter
    print("[PRODUCTION] Using resultsExporter.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step8_cache(config: ExperimentConfig):
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    # Load code assigned results
    if not cache_manager.is_cache_valid(config.filename, "code_assignment_direct", variable_key):
        raise FileNotFoundError(
            f"Cache not found: code_assignment_direct/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=8 first."
        )

    code_assigned_results = cache_manager.load_from_cache(
        config.filename, "code_assignment_direct", variable_key, models.CodeAssignedModel
    )

    # Load theme enriched codebook
    codebook_list = cache_manager.load_from_cache(
        config.filename, "codebook_refinement_enriched", variable_key, models.ThemeEnrichedCodebookModel
    )
    theme_enriched_codebook = codebook_list[0] if codebook_list else None

    # Load quality filtered text (optional)
    quality_filtered_text = None
    try:
        quality_filtered_text = cache_manager.load_from_cache(
            config.filename, "quality_filter", variable_key, models.QualityFilteredModel
        )
    except:
        pass

    # Load visualization metadata (optional)
    clustering_metadata = None
    extraction_metadata = None
    if config.include_visualizations:
        try:
            metadata_list = cache_manager.load_from_cache(
                config.filename, "clustering_metadata", variable_key, models.ClusteringMetadataModel
            )
            clustering_metadata = metadata_list[0] if metadata_list else None
        except:
            pass
        try:
            extraction_metadata = cache_manager.load_metadata_from_cache(
                config.filename, "extracted_ideas", variable_key, models.ExtractionMetadata
            )
        except:
            pass

    return code_assigned_results, theme_enriched_codebook, quality_filtered_text, clustering_metadata, extraction_metadata, variable_key, cache_manager


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    code_assigned_results, theme_enriched_codebook, quality_filtered_text, clustering_metadata, extraction_metadata, variable_key, cache_manager = load_step8_cache(config)
    var_lab = get_var_lab(config)

    verbose_reporter = VerboseReporter(config.verbose)

    verbose_reporter.section_header("EXPORT EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")
    verbose_reporter.stat_line(f"Input: {len(code_assigned_results)} assigned results")
    verbose_reporter.stat_line(f"Include visualizations: {config.include_visualizations}")

    start_time = time.time()

    # Run export
    exporter = ResultsExporter(verbose=config.verbose)
    excel_path = exporter.export_to_excel(
        code_assigned_results,
        theme_enriched_codebook,
        config.filename,
        config.var_name,
        quality_filtered_text=quality_filtered_text,
        export_dir=None,
        include_visualizations=config.include_visualizations,
        clustering_metadata=clustering_metadata,
        extraction_metadata=extraction_metadata
    )

    elapsed_time = time.time() - start_time

    verbose_reporter.stat_line(f"Output: {excel_path}")
    print(f"\n'Export experiment' completed in {elapsed_time:.2f} seconds.\n")

    return excel_path


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
        run_until_step=9
    )
    verbose_capture.__enter__()

    print("=" * 70)
    print("EXPERIMENT: Step 9 - Export")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {USE_EXPERIMENTAL}")
    print(f"Include visualizations: {config.include_visualizations}")
    print("=" * 70)

    try:
        excel_path = run_experiment(config)

        print("\n" + "=" * 70)
        print("EXPORT COMPLETE")
        print("=" * 70)
        print(f"Excel file: {excel_path}")
        print("=" * 70)

    finally:
        verbose_capture.__exit__(None, None, None)
