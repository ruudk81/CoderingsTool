#%%
"""
Step 3: Idea Extraction Step Runner

Loads Step 2 (filtered) results from cache and runs idea extraction.

Usage:
    cd src && python -m steps.step_3_ideaExtractor.run_ideaExtractor
"""

PRINT_PROMPTS = False  # Toggle prompt printing
EXPERIMENT_N  = 50  # n or None
DISCOVER_DOMAINS = True  # True = Phase 3 discovers domains upfront; False = on-the-fly

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
# SHARED IMPORTS
# =============================================================================

try:
    from pipeline.step_3_ideaExtractor import models
except ImportError:
    models_dir = Path(__file__).parent
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    import models
from config import CacheConfig
from pipeline.step_3_ideaExtractor.config_ideaExtractor import DEFAULT_IDEA_EXTRACTION_CONFIG
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
from utils.promptPrinter import PromptPrinter
from utils.llm import token_tracker
from utils.costTracker import CostTracker
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
    """Configuration for the ideaExtractor step."""

    # Data config from centralized test_data.py
    filename: str = TEST_DATA.filename
    id_column: str = TEST_DATA.id_column
    var_name: str = TEST_DATA.var_name
    sample_size: Optional[int] = TEST_DATA.sample_size

    # Output settings
    verbose: bool = True
    prompt_printer_enabled: bool = PRINT_PROMPTS

    # Processing settings
    force_recalc: bool = True  # Always recalculate
    experiment_n: Optional[int] = EXPERIMENT_N  # Limit responses (None = use all)


STEP_CONFIG = StepConfig()

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from .ideaExtractor import IdeaExtractor
except ImportError:
    exp_dir = Path(__file__).parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from ideaExtractor import IdeaExtractor


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step2_cache(config: StepConfig):
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


def get_var_lab(config: StepConfig) -> str:
    """Get variable label from data file."""
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN STEP RUNNER
# =============================================================================
def run_step(config: StepConfig = None):
    """Run the idea extraction step."""
    if config is None:
        config = STEP_CONFIG

    # Load previous step output
    quality_filtered_text, variable_key, cache_manager = load_step2_cache(config)

    # Get variable label
    var_lab = get_var_lab(config)

    # Initialize configs
    verbose_reporter = VerboseReporter(config.verbose)
    prompt_printer = PromptPrinter(
        enabled=True,  # Always capture prompts for debugging
        print_realtime=config.prompt_printer_enabled  # Only print if requested
    )

    verbose_reporter.section_header("IDEA EXTRACTION")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")

    # Filter to meaningful responses only
    filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
    verbose_reporter.stat_line(f"Input: {len(quality_filtered_text)} quality-filtered responses")
    verbose_reporter.stat_line(f"Meaningful responses: {len(filtered_text)}")

    # Optionally limit to experiment_n responses
    if config.experiment_n is not None and config.experiment_n < len(filtered_text):
        filtered_text = filtered_text[:config.experiment_n]
        verbose_reporter.stat_line(f"Subset: {config.experiment_n} responses")

    verbose_reporter.stat_line(f"Processing: {len(filtered_text)} responses")

    if not filtered_text:
        raise ValueError(
            f"No meaningful responses to process. "
            f"Total quality-filtered: {len(quality_filtered_text)}, "
            f"after removing filtered-out: 0, experiment_n: {config.experiment_n}"
        )

    start_time = time.time()

    # Initialize cost tracker
    cost_tracker = CostTracker(filename=config.filename, variable_key=variable_key)

    # Dataset key for performance stats cache (filename:variable_key)
    dataset_key = f"{config.filename}:{variable_key}"

    # Run idea extraction
    extractor = IdeaExtractor(
        responses=filtered_text,
        var_lab=var_lab,
        verbose=config.verbose,
        prompt_printer=prompt_printer,
        discover_domains=DISCOVER_DOMAINS,
        cost_tracker=cost_tracker,
        dataset_key=dataset_key,
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
                f"Cached extraction metadata: primary_dimension={extraction_metadata.primary_dimension}"
            )

    # Report any PROCESSING_ERROR failures
    if hasattr(extractor, 'failure_log') and extractor.failure_log:
        print(f"\n{'='*70}")
        print("WARNING: NOT 100% SUCCESSFUL")
        print(extractor.get_failure_report(total_responses=len(filtered_text)))
        print(f"{'='*70}")
    else:
        print(f"\nAll {len(filtered_text)} responses processed successfully (0 PROCESSING_ERROR)")

    # Finalize cost tracking for this step
    cost_tracker.finalize_step("step_3_idea_extraction")

    print(f"\n'Idea extraction' completed in {elapsed_time:.2f} seconds.\n")

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
    config = STEP_CONFIG

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
    print("Step 3 - Idea Extractor")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Models: context={DEFAULT_IDEA_EXTRACTION_CONFIG.model_context}, taxonomy={DEFAULT_IDEA_EXTRACTION_CONFIG.model_taxonomy}, abstraction_ladder={DEFAULT_IDEA_EXTRACTION_CONFIG.model_abstraction_ladder}")
    print(f"Experiment N: {config.experiment_n or 'all'}")
    print(f"Force recalculate: {config.force_recalc}")
    print("=" * 70)

    try:
        results, extractor, prompt_printer = run_step(config)

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
                ladder_parts = [v for v in (idea.instance, idea.interpretation, idea.abstraction) if v]
                if ladder_parts:
                    print(f"    ladder: {' -> '.join(ladder_parts)}")
                if idea.domain:
                    print(f"    domain: {idea.domain}")
                if idea.valence:
                    print(f"    valence: {idea.valence}")
            print("=" * 70)

        # Print all taxonomies
        if results:
            print("\n" + "=" * 70)
            print("ALL TAXONOMIES  (instance -> interpretation -> abstraction | domain | valence)")
            print("=" * 70)
            tax_count = 0
            for item in results:
                if not item.response_ideas:
                    continue
                for idea in item.response_ideas:
                    ladder_parts = [v for v in (idea.instance, idea.interpretation, idea.abstraction) if v]
                    chain = " -> ".join(ladder_parts)
                    extras = []
                    if idea.domain:
                        extras.append(idea.domain)
                    if idea.valence:
                        extras.append(idea.valence)
                    if extras:
                        chain += f" | {' | '.join(extras)}"
                    if chain:
                        tax_count += 1
                        print(f"  {tax_count:3d}. {chain}")
            print(f"\nTotal: {tax_count} taxonomies")
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
