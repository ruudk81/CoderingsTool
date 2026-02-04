"""
Step 6: Code Generator Experiment Runner

Runs the codebook generation step in isolation for experimentation.
Loads Step 5 (initial_clusters) results from cache and generates codebook.

Usage:
    cd src && python -m experiments.step_6_codeGenerator.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental codeGenerator from this folder
    USE_EXPERIMENTAL = False -> Uses production codeGenerator from utils/
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
from config import CacheConfig, ModelConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils import dataLoader, clusterer as clusterer_utils

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
    use_experimental: bool = False
    use_speculative_starter_codes: bool = True
    verbose: bool = True
    verbose_detailed: bool = False
    prompt_printer_enabled: bool = False
    force_recalc: bool = True


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .codeGenerator_exp import InductiveCodeGenerator
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from codeGenerator_exp import InductiveCodeGenerator
    print("[EXPERIMENTAL] Using codeGenerator_exp.py from experiments folder")
else:
    from utils.codeGenerator import InductiveCodeGenerator
    print("[PRODUCTION] Using codeGenerator.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step5_cache(config: ExperimentConfig):
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    step_name = "initial_clusters"

    if not cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raise FileNotFoundError(
            f"Cache not found: {step_name}/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=5 first."
        )

    data = cache_manager.load_from_cache(
        config.filename, step_name, variable_key, models.ClusterModel
    )

    # Load clustering metadata for starter codes
    clustering_metadata = None
    try:
        metadata_list = cache_manager.load_from_cache(
            config.filename, "clustering_metadata", variable_key, models.ClusteringMetadataModel
        )
        if metadata_list:
            clustering_metadata = metadata_list[0]
    except:
        pass

    # Load extraction metadata
    extraction_metadata = None
    try:
        extraction_metadata = cache_manager.load_metadata_from_cache(
            config.filename, "extracted_ideas", variable_key, models.ExtractionMetadata
        )
    except:
        pass

    return data, variable_key, cache_manager, clustering_metadata, extraction_metadata


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    initial_cluster_results, variable_key, cache_manager, clustering_metadata, extraction_metadata = load_step5_cache(config)
    var_lab = get_var_lab(config)

    model_config = ModelConfig()
    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(enabled=config.verbose, print_realtime=config.prompt_printer_enabled)

    verbose_reporter.section_header("CODEBOOK GENERATION EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")
    verbose_reporter.stat_line(f"Input: {len(initial_cluster_results)} cluster results")

    start_time = time.time()

    # Get starter codes from clustering metadata
    starter_codes = []
    if config.use_speculative_starter_codes and clustering_metadata:
        for cluster_id, cluster_data in clustering_metadata.clusters.items():
            if cluster_data.label_theme:
                starter_codes.append({
                    'code': cluster_data.label_theme,
                    'definition': cluster_data.label_description or '',
                    'cluster_id': cluster_id
                })
        if starter_codes:
            verbose_reporter.stat_line(f"Loaded {len(starter_codes)} starter codes from cluster labels")

    # Clean ideas
    cleaned_cluster_results = clusterer_utils.clean_cluster_ideas(initial_cluster_results)

    # Generate codebook
    generator = InductiveCodeGenerator(
        cluster_results=cleaned_cluster_results,
        starter_codes=starter_codes,
        var_lab=var_lab,
        verbose=config.verbose,
        verbose_detailed=config.verbose_detailed,
        prompt_printer=prompt_printer,
        extraction_metadata=extraction_metadata
    )
    codebook_reasoning = generator.generate()

    elapsed_time = time.time() - start_time

    # Cache results
    if codebook_reasoning:
        cache_manager.save_to_cache([codebook_reasoning], config.filename, "codebook_generation_reasoning", variable_key, elapsed_time, var_lab=var_lab)
        cache_manager.save_to_cache(generator.cluster_results, config.filename, "expanded_clusters", variable_key, elapsed_time, var_lab=var_lab)

    # Summary
    if codebook_reasoning and codebook_reasoning.codebook:
        verbose_reporter.stat_line(f"Output: {len(codebook_reasoning.codebook)} codes generated")

    print(f"\n'Codebook generation experiment' completed in {elapsed_time:.2f} seconds.\n")

    return codebook_reasoning, generator


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
        run_until_step=6
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("EXPERIMENT: Step 6 - Code Generator")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {USE_EXPERIMENTAL}")
    print(f"Speculative starter codes: {config.use_speculative_starter_codes}")
    print("=" * 70)

    try:
        codebook_reasoning, generator = run_experiment(config)

        # Print codebook
        if codebook_reasoning and codebook_reasoning.codebook:
            print("\n" + "=" * 70)
            print("GENERATED CODEBOOK")
            print("=" * 70)
            for i, code in enumerate(codebook_reasoning.codebook, 1):
                print(f"{i}. {code['code']}")
            print("=" * 70)

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)
