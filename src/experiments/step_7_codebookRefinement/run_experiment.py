#%%
"""
Step 7: Codebook Refinement Experiment Runner

Runs the codebook refinement step in isolation for experimentation.
Loads Step 6 (codebook_generation_reasoning) results from cache and refines codebook.

Usage:
    cd src && python -m experiments.step_7_codebookRefinement.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental codebookRefinement from this folder
    USE_EXPERIMENTAL = False -> Uses production codebookRefinement from utils/
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
from experiments import models_exp as models
from config import CacheConfig, ModelConfig, DEFAULT_LANGUAGE
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils import dataLoader, codeGenerator

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
    prompt_printer_enabled: bool = False
    language: str = DEFAULT_LANGUAGE
    force_recalc: bool = True


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .codebookRefinement_exp import CodebookRefinementProcessor, CodebookRefinementConfig
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from codebookRefinement_exp import CodebookRefinementProcessor, CodebookRefinementConfig
    print("[EXPERIMENTAL] Using codebookRefinement_exp.py from experiments folder")
else:
    from utils.codebookRefinement import refine_codebook, print_refinement_report, get_refinement_report
    print("[PRODUCTION] Using codebookRefinement.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step6_cache(config: ExperimentConfig):
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    step_name = "codebook_generation_reasoning"

    if not cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raise FileNotFoundError(
            f"Cache not found: {step_name}/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=6 first."
        )

    reasoning_models = cache_manager.load_from_cache(
        config.filename, step_name, variable_key, codeGenerator.CodeGeneratorReasoningResults
    )

    codebook_reasoning = reasoning_models[0] if reasoning_models else None

    return codebook_reasoning, variable_key, cache_manager


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    codebook_reasoning, variable_key, cache_manager = load_step6_cache(config)
    var_lab = get_var_lab(config)

    model_config = ModelConfig()
    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(enabled=True, print_realtime=config.prompt_printer_enabled)

    verbose_reporter.section_header("CODEBOOK REFINEMENT EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")
    verbose_reporter.stat_line(f"Language: {config.language}")

    if codebook_reasoning and codebook_reasoning.codebook:
        verbose_reporter.stat_line(f"Input: {len(codebook_reasoning.codebook)} codes to refine")

    start_time = time.time()

    if USE_EXPERIMENTAL:
        # ── Partition-first refinement ────────────────────────────────────
        refinement_config = CodebookRefinementConfig(
            model_config=model_config,
            language=config.language,
            verbose=config.verbose,
            prompt_printer=prompt_printer,
        )
        processor = CodebookRefinementProcessor(refinement_config)

        # Single call: partition review → refine → MECE → cross-partition judge
        partition_results, judge_result, concept_type_map, partition_remap = processor.refine_codebook_partitioned(
            survey_question=var_lab,
            reasoning_results=codebook_reasoning,
        )

        # Build enriched codebook
        theme_enriched_codebook = processor.build_theme_enriched_codebook(
            partition_results=partition_results,
            judge_result=judge_result,
            concept_type_map=concept_type_map,
            source_variable=config.var_name,
            partition_remap=partition_remap,
        )

        elapsed = time.time() - start_time

        # Cache enriched codebook
        cache_manager.save_to_cache(
            [theme_enriched_codebook], config.filename,
            "codebook_refinement_enriched", variable_key, elapsed, var_lab=var_lab
        )

        # ── Report: Codebook Hierarchy ────────────────────────────────────
        total_codes = len(theme_enriched_codebook.codes)
        mece_verified = sum(1 for c in theme_enriched_codebook.codes if c.mece_verified)
        n_conflicts = len(judge_result.conflicts) if judge_result else 0

        if config.verbose:
            print(f"\n{'='*60}")
            print("REFINED CODEBOOK")
            print(f"{'='*60}")
            for i, (pname, result) in enumerate(sorted(partition_results.items()), 1):
                print(f"\n{i}. Theme: {result.theme_label} ({len(result.codes)} codes)")
                print(f"   {result.theme_description}")
                for code in result.codes:
                    print(f"   - {code.code}: {code.definition}")
            print(f"{'='*60}")

            # ── Report: Assignment Instructions ───────────────────────────────
            print(f"\n{'='*60}")
            print("MECE ASSIGNMENT INSTRUCTIONS")
            print(f"{'='*60}")
            for pname, result in sorted(partition_results.items()):
                print(f"\n--- {result.theme_label} ({pname}) ---")
                for code in result.codes:
                    print(f"\n  Code: {code.code}")
                    print(f"  Definition: {code.definition}")
                    print(f"  Boundary test: {code.boundary_test}")
                    if code.diagnostic_signals:
                        print(f"  Diagnostic signals: {', '.join(code.diagnostic_signals)}")
                    if code.inclusion_examples:
                        print(f"  Inclusion examples:")
                        for ex in code.inclusion_examples:
                            print(f"    + {ex}")
                    if code.exclusion_examples:
                        print(f"  Exclusion examples:")
                        for ex in code.exclusion_examples:
                            print(f"    - {ex}")
                    print(f"  Near neighbor: {code.near_neighbor_label}")
                    print(f"  Tell-apart rule: {code.tell_apart_rule}")
            print(f"\n{'='*60}")

        verbose_reporter.stat_line(f"\nOutput: {total_codes} refined codes across {len(partition_results)} themes")
        verbose_reporter.stat_line(f"MECE: {mece_verified}/{total_codes} codes verified")
        if judge_result:
            verbose_reporter.stat_line(f"Cross-partition: {n_conflicts} conflicts, MECE={'yes' if judge_result.is_mece_compliant else 'no'}")

    else:
        # ── Production path (legacy) ─────────────────────────────────────
        refinement_results = refine_codebook(
            survey_question=var_lab,
            reasoning_results=codebook_reasoning,
            model_config=model_config,
            language=config.language,
            verbose=config.verbose,
            prompt_printer=prompt_printer
        )
        elapsed = time.time() - start_time
        cache_manager.save_to_cache([refinement_results], config.filename, "codebook_refinement", variable_key, elapsed, var_lab=var_lab)

        if config.verbose and refinement_results:
            print_refinement_report(refinement_results)

        theme_enriched_codebook = None

    print(f"\n'Codebook refinement experiment' completed in {elapsed:.2f} seconds.\n")

    return theme_enriched_codebook, prompt_printer


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
        run_until_step=7
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("EXPERIMENT: Step 7 - Codebook Refinement")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {USE_EXPERIMENTAL}")
    print("=" * 70)

    try:
        theme_enriched_codebook, prompt_printer = run_experiment(config)

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)
