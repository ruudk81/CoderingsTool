#%%

"""
Step 6: Code Generator Experiment Runner

Runs the codebook generation step in isolation for experimentation.
Loads Step 5 (MECE categories) results from cache and generates codebook.

Usage:
    cd src && python -m experiments.step_6_codeGenerator.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental codeGenerator from this folder
    USE_EXPERIMENTAL = False -> Uses production codeGenerator from utils/
"""

EXPERIMENT_N = None  # n or None — limit response models for faster experiments


import pickle
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
from experiments import models_exp as models
from config import CacheConfig, ModelConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
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
    verbose_detailed: bool = False
    prompt_printer_enabled: bool = False
    force_recalc: bool = True
    # Embedding format for step 6 internal sampling/redistribution
    # Uses cached step 4 embedding if available, else computes on-the-fly.
    # "cached" = use idea_embedding as-is | "ladder" = ladder_embedding (default)
    # "concept_defined" | "idea_concept_defined" | "concept" | "concept_type"
    # Composite: "concept+concept_type_definition", "idea+concept", etc.
    step6_embedding_format: str = "ladder"
    step6_embedding_separator: str = " → "
    experiment_n: Optional[int] = EXPERIMENT_N  # Limit response models for experiment (None = all)


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .codeGenerator_exp import InductiveCodeGenerator
        from .config_exp import STAGE1_INPUT_SOURCE
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from codeGenerator_exp import InductiveCodeGenerator
        from config_exp import STAGE1_INPUT_SOURCE
    print("[EXPERIMENTAL] Using codeGenerator_exp.py from experiments folder")
else:
    from utils.codeGenerator import InductiveCodeGenerator
    print("[PRODUCTION] Using codeGenerator.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def _get_variable_key_and_cache(config: ExperimentConfig):
    """Return (variable_key, cache_manager) — always available, no step 5 dependency."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())
    return variable_key, cache_manager


def load_extraction_metadata(config: ExperimentConfig, variable_key: str, cache_manager: CacheManager):
    """Load step 3 extraction metadata (primary_facet, domain, topic, etc.)."""
    try:
        return cache_manager.load_metadata_from_cache(
            config.filename, "extracted_ideas", variable_key, models.ExtractionMetadata
        )
    except Exception:
        return None


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


def load_mece_topics(config: ExperimentConfig, variable_key: str):
    """Load MECE Phase A topics from cache if available (legacy pickle format)."""
    base_name = Path(config.filename).stem
    cache_dir = project_root / "data" / "cache"
    cache_path = cache_dir / f"mece_phase_a_{base_name}_{variable_key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            mece_topics = pickle.load(f)
        print(f"  Loaded MECE Phase A topics for {len(mece_topics)} clusters from '{cache_path.name}'")
        return mece_topics
    else:
        print(f"  WARNING: MECE Phase A cache not found: '{cache_path.name}' — falling back to idea sampling")
        return None


def load_mece_categories(config: ExperimentConfig, variable_key: str):
    """Load MECE category data from step_5_categories cache.

    Returns:
        (mece_results_cache, category_assigned_data) tuple.
        Either or both may be None if cache is not available.
    """
    cache_manager = CacheManager(CacheConfig())

    # Load MECEResultsCache (metadata cache)
    mece_results_cache = None
    try:
        mece_results_cache = cache_manager.load_metadata_from_cache(
            config.filename, "mece_categories", variable_key, models.MECEResultsCache
        )
    except Exception as e:
        print(f"  WARNING: Failed to load MECE categories metadata: {e}")

    if not mece_results_cache:
        print("  WARNING: MECE categories cache not found — falling back")
        return None, None

    # Load CategoryAssignedModel list (list cache)
    category_assigned = None
    try:
        category_assigned = cache_manager.load_from_cache(
            config.filename, "category_assignment", variable_key, models.CategoryAssignedModel
        )
    except Exception as e:
        print(f"  WARNING: Failed to load category assignments: {e}")

    if not category_assigned:
        print("  WARNING: Category assignment cache not found — falling back")
        return mece_results_cache, None

    total_cats = mece_results_cache.total_categories
    n_partitions = len(mece_results_cache.partition_results)
    print(f"  Loaded MECE categories: {total_cats} categories across {n_partitions} partitions")
    print(f"  Loaded category assignments: {len(category_assigned)} response models")

    return mece_results_cache, category_assigned


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    variable_key, cache_manager = _get_variable_key_and_cache(config)
    var_lab = get_var_lab(config)
    extraction_metadata = load_extraction_metadata(config, variable_key, cache_manager)

    model_config = ModelConfig()
    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(enabled=config.verbose, print_realtime=config.prompt_printer_enabled)

    verbose_reporter.section_header("CODEBOOK GENERATION EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")

    start_time = time.time()

    # Load input source data based on STAGE1_INPUT_SOURCE config
    mece_topics = None
    mece_results_cache = None
    category_assigned_data = None

    if USE_EXPERIMENTAL and STAGE1_INPUT_SOURCE == "mece_categories":
        verbose_reporter.stat_line(f"Input source: MECE categories (STAGE1_INPUT_SOURCE={STAGE1_INPUT_SOURCE!r})")
        mece_results_cache, category_assigned_data = load_mece_categories(config, variable_key)
        if not mece_results_cache or not category_assigned_data:
            verbose_reporter.stat_line("  Falling back to mece_topics or idea sampling")
        elif config.experiment_n is not None and config.experiment_n < len(category_assigned_data):
            full_count = len(category_assigned_data)
            category_assigned_data = category_assigned_data[:config.experiment_n]
            verbose_reporter.stat_line(f"Experiment subset: {config.experiment_n} of {full_count} response models")

    if USE_EXPERIMENTAL and STAGE1_INPUT_SOURCE == "mece_topics" or (
        STAGE1_INPUT_SOURCE == "mece_categories" and not category_assigned_data
    ):
        if STAGE1_INPUT_SOURCE == "mece_topics":
            verbose_reporter.stat_line(f"Input source: MECE topics (STAGE1_INPUT_SOURCE={STAGE1_INPUT_SOURCE!r})")
        mece_topics = load_mece_topics(config, variable_key)

    if not USE_EXPERIMENTAL or STAGE1_INPUT_SOURCE == "ideas":
        verbose_reporter.stat_line(f"Input source: idea sampling (STAGE1_INPUT_SOURCE={'ideas'!r})")

    # Generate codebook (clean slate — no starter codes)
    generator = InductiveCodeGenerator(
        cluster_results=[],
        starter_codes=[],
        var_lab=var_lab,
        verbose=config.verbose,
        verbose_detailed=config.verbose_detailed,
        prompt_printer=prompt_printer,
        extraction_metadata=extraction_metadata,
        mece_topics=mece_topics,
        mece_results_cache=mece_results_cache,
        category_assigned_data=category_assigned_data,
        embedding_text_format=config.step6_embedding_format,
        embedding_separator=config.step6_embedding_separator,
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
    if USE_EXPERIMENTAL:
        print(f"Stage 1 input source: {STAGE1_INPUT_SOURCE}")
    print(f"Embedding format: {config.step6_embedding_format}")
    print(f"Experiment N: {config.experiment_n or 'all'}")
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
