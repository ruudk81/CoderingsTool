#%%

"""
Step 8: Code Assigner Experiment Runner (Partition-Based)

Runs the code assignment step in isolation for experimentation.
Loads embeddings (step 4) and theme_enriched_codebook (step 7) from cache
and assigns codes using concept_type partition routing.

Usage:
    cd src && python -m experiments.step_8_codeAssigner.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental codeAssigner from this folder
    USE_EXPERIMENTAL = False -> Uses production codeAssigner from utils/
"""

EXPERIMENT_N = 20  # Limit number of responses to process (None = all)

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
    experiment_n: Optional[int] = EXPERIMENT_N  # Limit responses for experiment (None = use all)


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .codeAssigner_exp import CodeAssigner
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from codeAssigner_exp import CodeAssigner
    print("[EXPERIMENTAL] Using codeAssigner_exp.py from experiments folder")
else:
    from utils.codeAssigner import CodeAssigner
    print("[PRODUCTION] Using codeAssigner.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_experiment_cache(config: ExperimentConfig):
    """Load response data (step 4 embeddings) and codebook (step 7) from cache.

    Step 8 needs:
    - Response data with ladder fields (instance, concept, concept_type) → from step 4
    - Codebook with assignment instructions → from step 7
    No cluster IDs or embeddings needed for partition-based routing.
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    # Load response data from step 4 (has ladder fields from step 3)
    if not cache_manager.is_cache_valid(config.filename, "embeddings", variable_key):
        raise FileNotFoundError(
            f"Cache not found: embeddings/{variable_key}\n"
            f"Run step 4 embedder experiment first:\n"
            f"  cd src && python -m experiments.step_4_embedder.run_experiment"
        )

    response_data = cache_manager.load_from_cache(
        config.filename, "embeddings", variable_key, models.EmbeddingsModel
    )

    # Load theme enriched codebook from step 7
    if not cache_manager.is_cache_valid(config.filename, "codebook_refinement_enriched", variable_key):
        raise FileNotFoundError(
            f"Cache not found: codebook_refinement_enriched/{variable_key}\n"
            f"Run step 7 codebookRefinement experiment first:\n"
            f"  cd src && python -m experiments.step_7_codebookRefinement.run_experiment"
        )

    codebook_list = cache_manager.load_from_cache(
        config.filename, "codebook_refinement_enriched", variable_key, models.ThemeEnrichedCodebookModelExp
    )
    theme_enriched_codebook = codebook_list[0] if codebook_list else None

    return response_data, theme_enriched_codebook, variable_key, cache_manager


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    response_data, theme_enriched_codebook, variable_key, cache_manager = load_experiment_cache(config)
    var_lab = get_var_lab(config)

    model_config = ModelConfig()
    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(enabled=config.prompt_printer_enabled, print_realtime=True)

    verbose_reporter.section_header("CODE ASSIGNMENT EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")
    verbose_reporter.stat_line(f"Input: {len(response_data)} responses (source: step 4 embeddings)")

    # Optionally limit to experiment_n responses
    if config.experiment_n is not None and config.experiment_n < len(response_data):
        response_data = response_data[:config.experiment_n]
        verbose_reporter.stat_line(f"Experiment subset: {config.experiment_n} responses")
    n_partitions = len(set(getattr(e, 'concept_type', None) or '_unpartitioned' for e in theme_enriched_codebook.codes))
    verbose_reporter.stat_line(f"Codebook: {len(theme_enriched_codebook.codes)} codes across {n_partitions} partitions")

    start_time = time.time()

    # Create codebook entries with full MECE instructions
    codebook = [models.CodebookExp(
        code=entry.code,
        definition=entry.definition,
        theme=entry.theme,
        theme_description=entry.theme_description,
        source_cluster=entry.source_cluster,
        inclusion_examples=entry.inclusion_examples,
        exclusion_examples=entry.exclusion_examples,
        near_neighbor_label=entry.near_neighbor_label,
        tell_apart_rule=entry.tell_apart_rule,
        concept_type=getattr(entry, 'concept_type', None),
        boundary_test=getattr(entry, 'boundary_test', None),
        diagnostic_signals=getattr(entry, 'diagnostic_signals', None),
    ) for entry in theme_enriched_codebook.codes]

    # Remap codebook concept_types for split partitions.
    # partition_remap maps new_split_name → original_partition_name.
    # Ideas carry the original concept_type from step 3, so codebook entries
    # from split partitions need to use the old name for routing to match.
    partition_remap = getattr(theme_enriched_codebook, 'partition_remap', None) or {}
    if partition_remap:
        remapped = 0
        for entry in codebook:
            if entry.concept_type in partition_remap:
                entry.concept_type = partition_remap[entry.concept_type]
                remapped += 1
        if remapped:
            verbose_reporter.stat_line(f"Partition remap: {remapped} codebook entries remapped for {len(partition_remap)} split partitions")

    # Run code assignment
    code_assigner = CodeAssigner(
        response_models=response_data,
        codebook=codebook,
        var_lab=var_lab,
        code_to_theme_mapping=theme_enriched_codebook.code_to_theme_mapping,
        model_config=model_config,
        verbose=config.verbose,
        prompt_printer=prompt_printer
    )

    code_assigned_results = code_assigner.assign()

    elapsed_time = time.time() - start_time

    # Add metadata
    for result in code_assigned_results:
        if not hasattr(result, 'assignment_metadata') or result.assignment_metadata is None:
            result.assignment_metadata = {}
        result.assignment_metadata.update({
            "codebook_used": f"{len(theme_enriched_codebook.codes)} codes",
            "assignment_method": "partition_based_ladder",
            "experiment": True,
            "partitions": n_partitions
        })

    cache_manager.save_to_cache(code_assigned_results, config.filename, "code_assignment_direct", variable_key, elapsed_time, var_lab=var_lab)

    # Print stats
    if config.verbose:
        code_assigner.print_assignment_stats()

    # Calculate summary
    total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)
    total_assignments = sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) for resp in code_assigned_results if resp.response_ideas)

    verbose_reporter.stat_line(f"Output: {total_assignments} code assignments for {total_ideas} ideas")
    print(f"\n'Code assignment experiment' completed in {elapsed_time:.2f} seconds.\n")

    return code_assigned_results, code_assigner


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
        run_until_step=8
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("EXPERIMENT: Step 8 - Code Assigner")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {USE_EXPERIMENTAL}")
    print(f"Experiment N: {config.experiment_n or 'all'}")
    print("=" * 70)

    try:
        results, code_assigner = run_experiment(config)

        # Print sample assignment
        if results and len(results) > 0:
            import random
            sample = random.choice([r for r in results if r.response_ideas])
            if sample.response_ideas:
                idea = sample.response_ideas[0]
                print("\n" + "=" * 70)
                print("SAMPLE ASSIGNMENT")
                print("=" * 70)
                print(f"Instance: {idea.instance}")
                print(f"Concept: {idea.concept}")
                print(f"Concept Type: {idea.concept_type}")
                print(f"Codes: {idea.assigned_codes}")
                print(f"Themes: {idea.assigned_themes}")
                print("=" * 70)

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)
