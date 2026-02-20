#%% 

"""
Step 3: Idea Extractor Experiment Runner (v5 — MECE Decision Tree Facets)

Runs the idea extraction step in isolation for experimentation.
Loads Step 2 (quality_filter) results from cache and runs idea extraction.

v5 taxonomy: 10 MECE Facets (decision tree) + Concept Types + valence

Usage:
    cd src && python -m "experiments.step_3_ideaExtractor v4.run_experiment"

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental ideaExtractor from this folder
    USE_EXPERIMENTAL = False -> Uses production ideaExtractor from utils/
"""

USE_EXPERIMENTAL = True  # Toggle between production and experimental
PRINT_PROMPTS = False  # Toggle prompt printing
EXPERIMENT_N  = None  # n or None
DISCOVER_CONCEPT_TYPES = True  # True = Phase 3 discovers types upfront; False = on-the-fly

import sys
import time
from pathlib import Path

# Path setup - ensure src directory is in path
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
# v4 uses local models with primary_facet/concept_type fields
try:
    from experiments.step_3_ideaExtractor_v4 import models_exp_v3 as models
except ImportError:
    # Fallback for direct execution
    models_v4_dir = Path(__file__).parent
    if str(models_v4_dir) not in sys.path:
        sys.path.insert(0, str(models_v4_dir))
    import models_exp_v3 as models
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
    """Configuration for the ideaExtractor experiment."""

    # Data config from centralized test_data.py
    filename: str = TEST_DATA.filename
    id_column: str = TEST_DATA.id_column
    var_name: str = TEST_DATA.var_name
    sample_size: Optional[int] = TEST_DATA.sample_size

    # Toggle: production vs experimental
    use_experimental: bool = USE_EXPERIMENTAL  # Set True to use local ideaExtractor_exp.py

    # Output settings
    verbose: bool = True
    prompt_printer_enabled: bool = PRINT_PROMPTS

    # Processing settings
    force_recalc: bool = True  # Always recalculate in experiments
    experiment_n: Optional[int] = EXPERIMENT_N  # Limit responses for experiment (None = use all)


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .ideaExtractor_exp import IdeaExtractor
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from ideaExtractor_exp import IdeaExtractor
    print("[EXPERIMENTAL] Using ideaExtractor_exp.py from experiments folder")
else:
    from utils.ideaExtractor import IdeaExtractor
    print("[PRODUCTION] Using ideaExtractor.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step2_cache(config: ExperimentConfig):
    """Load quality-filtered results from Step 2 cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    step_name = "quality_filter"

    if not cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raise FileNotFoundError(
            f"Cache not found: {step_name}/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=2 first to generate the cache."
        )

    data = cache_manager.load_from_cache(
        config.filename, step_name, variable_key, models.QualityFilteredModel
    )
    return data, variable_key, cache_manager


def get_var_lab(config: ExperimentConfig) -> str:
    """Get variable label from data file."""
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    """Run the idea extraction step."""
    if config is None:
        config = EXPERIMENT_CONFIG

    # Load previous step output
    quality_filtered_text, variable_key, cache_manager = load_step2_cache(config)

    # Get variable label
    var_lab = get_var_lab(config)

    # Initialize configs
    model_config = ModelConfig()
    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(
        enabled=True,  # Always capture prompts for debugging
        print_realtime=config.prompt_printer_enabled  # Only print if requested
    )

    verbose_reporter.section_header("IDEA EXTRACTION EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")

    # Filter to meaningful responses only
    filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
    verbose_reporter.stat_line(f"Input: {len(quality_filtered_text)} quality-filtered responses")
    verbose_reporter.stat_line(f"Meaningful responses: {len(filtered_text)}")

    # Optionally limit to experiment_n responses
    if config.experiment_n is not None and config.experiment_n < len(filtered_text):
        filtered_text = filtered_text[:config.experiment_n]
        verbose_reporter.stat_line(f"Experiment subset: {config.experiment_n} responses")

    verbose_reporter.stat_line(f"Processing: {len(filtered_text)} responses")

    start_time = time.time()

    # Run idea extraction
    extractor = IdeaExtractor(
        responses=filtered_text,
        var_lab=var_lab,
        model_config=model_config,
        verbose=config.verbose,
        prompt_printer=prompt_printer,
        discover_concept_types=DISCOVER_CONCEPT_TYPES,
    )
    encoded_text = extractor.extract()

    elapsed_time = time.time() - start_time

    # Count ideas
    total_ideas = sum(item.idea_count for item in encoded_text)
    verbose_reporter.stat_line(f"Output: {len(encoded_text)} responses with {total_ideas} ideas")
    verbose_reporter.stat_line(f"Average ideas per response: {total_ideas / len(encoded_text):.2f}")

    # Save to cache
    cache_manager.save_to_cache(
        encoded_text,
        config.filename,
        "extracted_ideas",
        variable_key,
        elapsed_time,
        var_lab=var_lab
    )

    # Save extraction metadata if available
    if hasattr(extractor, 'build_extraction_metadata'):
        extraction_metadata = extractor.build_extraction_metadata(
            filename=config.filename,
            var_name=config.var_name
        )
        cache_manager.save_metadata_to_cache(
            metadata=extraction_metadata,
            filename=config.filename,
            step="extracted_ideas",
            variable_key=variable_key,
            processing_time=elapsed_time,
            var_lab=var_lab
        )
        if config.verbose:
            verbose_reporter.stat_line(
                f"Cached extraction metadata: primary_facet={extraction_metadata.primary_facet}"
            )

    # Report any PROCESSING_ERROR failures
    if hasattr(extractor, 'failure_log') and extractor.failure_log:
        print(f"\n{'='*70}")
        print("WARNING: NOT 100% SUCCESSFUL")
        print(extractor.get_failure_report(total_responses=len(filtered_text)))
        print(f"{'='*70}")
    else:
        print(f"\nAll {len(filtered_text)} responses processed successfully (0 PROCESSING_ERROR)")

    print(f"\n'Idea extraction experiment' completed in {elapsed_time:.2f} seconds.\n")

    # Save prompts to JSON file
    if prompt_printer.prompts:
        prompts_dir = project_root / "exports" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompts_file = prompts_dir / f"step3_{config.var_name}_{variable_key}.json"
        prompt_printer.save_prompts(str(prompts_file))

    return encoded_text, extractor, prompt_printer


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    config = EXPERIMENT_CONFIG

    # Get variable label for verbose capture
    var_lab = get_var_lab(config)

    # Start verbose capture
    verbose_capture = VerboseCapture(
        filename=config.filename,
        variable_key=config.var_name,
        sample_size=config.sample_size,
        run_until_step=3
    )
    verbose_capture.__enter__()

    # Reset token tracker
    token_tracker.reset()

    print("=" * 70)
    print("EXPERIMENT: Step 3 - Idea Extractor")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {USE_EXPERIMENTAL}")
    print(f"Experiment N: {config.experiment_n or 'all'}")
    print(f"Force recalculate: {config.force_recalc}")
    print("=" * 70)

    try:
        results, extractor, prompt_printer = run_experiment(config)

        # Print sample output
        if results and len(results) > 0:
            import random
            sample = random.choice(results)
            print("\n" + "=" * 70)
            print("SAMPLE OUTPUT")
            print("=" * 70)
            print(f"Response: {sample.response}")
            if sample.template_prefix:
                print(f"Template prefix: \"{sample.template_prefix}\"")
            print(f"Ideas ({sample.idea_count}):")
            for idea in sample.response_ideas:
                print(f"  - {idea.idea}")
                ladder_parts = [v for v in (idea.instance, idea.concept, idea.concept_type, idea.concept_type_definition) if v]
                if ladder_parts:
                    print(f"    ladder: {' → '.join(ladder_parts)}")
                if idea.valence:
                    print(f"    valence: {idea.valence}")
            print("=" * 70)

        # Print all taxonomies
        if results:
            print("\n" + "=" * 70)
            print("ALL ABSTRACTION LADDERS  (instance → concept → concept_type → concept_type_definition | valence)")
            print("=" * 70)
            tax_count = 0
            for item in results:
                if not item.response_ideas:
                    continue
                for idea in item.response_ideas:
                    chain_parts = [v for v in (idea.instance, idea.concept, idea.concept_type, idea.concept_type_definition) if v]
                    sec_parts = []
                    if idea.valence:
                        sec_parts.append(idea.valence)
                    chain = " → ".join(chain_parts)
                    if sec_parts:
                        chain += f" | {', '.join(sec_parts)}"
                    if chain:
                        tax_count += 1
                        print(f"  {tax_count:3d}. {chain}")
            print(f"\nTotal: {tax_count} abstraction ladders")
            print("=" * 70)

        # Print token usage
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

        # Print captured prompts if enabled
        if PRINT_PROMPTS and prompt_printer.prompts:
            prompt_printer.print_summary()
            prompt_printer.print_all_prompts()

    finally:
        # Save verbose output
        verbose_capture.__exit__(None, None, None)

# %%
