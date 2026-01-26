#%%
"""
IdeaExtractor V3 Experiment Runner

This script runs the ideaExtractor step (Step 3) in isolation for experimentation.
It loads Step 2 (quality_filter) results from cache and runs idea extraction.

NOTE: After migration, v3 (taxonomy-aware) is now the PRODUCTION ideaExtractor.
The old ideaExtractor is archived at src/utils/old/ideaExtractor_v2.py.

Usage:
    cd src && python -m experiments.ideaExtractor_v3.run_experimentv3

Toggle modes:
    USE_EXPERIMENTAL_EXTRACTOR = True  -> Uses production ideaExtractor (taxonomy-aware, migrated v3)
    USE_EXPERIMENTAL_EXTRACTOR = False -> Uses archived old ideaExtractor (v2)
"""

import os
import sys

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
import random
import re
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import nest_asyncio
nest_asyncio.apply()

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the ideaExtractor experiment."""

    # Dataset settings (matching pipeline.py selection)
    filename: str = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column: str = "DLNMID"
    var_name: str = "Q20"
    sample_size: int = 50
    

    # Experiment toggles
    use_experimental_extractor: bool = True   # Use local ideaExtractor_experimental.py (with taxonomy)
    use_experimental_prompts: bool = False    # Use local prompts.py (monkey-patched into production)

    # Output settings
    verbose: bool = True
    prompt_printer_enabled: bool = False
    sample_output_count: int = 10
    max_responses: Optional[int] = None  # Limit responses for quick experiments

    # Language setting
    language: str = "nl"

    # Output file saving
    save_results_to_file: bool = True


# Default configuration - modify this for your experiments
EXPERIMENT_CONFIG = ExperimentConfig(
    filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
    id_column="DLNMID",
    var_name="Q20",
    #sample_size=50,
    sample_size=500,

    #filename = "M250480 Associatiemonitor ASN Bank net databestand.sav",
    #id_column = "DLNMID",
    #var_name = "Qd1_combined",
    #sample_size = 2000 ,

    #filename = "M250219 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav",
    #id_column = "DLNMID",
    #var_name = "Q15",
    #sample_size = 2000,

    # Experiment toggles
    use_experimental_extractor=True,  # Enable taxonomy-aware extraction
    use_experimental_prompts=False,

    # Output settings
    verbose=True,
    prompt_printer_enabled=False,
    sample_output_count=10,
    max_responses=None,  # Set to e.g. 100 for quick experiments
)


# =============================================================================
# PROMPT INJECTION (for USE_EXPERIMENTAL_PROMPTS mode)
# =============================================================================

def inject_experimental_prompts():
    """
    [DEPRECATED] Prompt injection is no longer needed.
    After migration, taxonomy prompts are now in production prompts.py.
    This function is kept for backward compatibility but does nothing.
    """
    print("ℹ️ Prompt injection not needed - taxonomy prompts are now in production prompts.py")


# =============================================================================
# CACHE LOADING
# =============================================================================

def load_step2_cache(config: ExperimentConfig):
    """
    Load Step 2 (quality_filter) results from cache.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (quality_filtered_text, var_lab, variable_key)
    """
    import models
    from utils.cacheManager import CacheManager, generate_enhanced_variable_key
    from config import CacheConfig

    # Generate the cache key matching pipeline behavior
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )

    cache_manager = CacheManager(CacheConfig())
    step = "quality_filter"

    # Check if cache exists
    if not cache_manager.is_cache_valid(config.filename, step, variable_key):
        raise FileNotFoundError(
            f"Step 2 cache not found for:\n"
            f"  Filename: {config.filename}\n"
            f"  Variable: {config.var_name}\n"
            f"  Sample size: {config.sample_size}\n"
            f"  Variable key: {variable_key}\n\n"
            f"Please run the pipeline through Step 2 first to generate the cache."
        )

    # Load the cached data
    quality_filtered_text = cache_manager.load_from_cache(
        filename=config.filename,
        step=step,
        variable_key=variable_key,
        model_cls=models.QualityFilteredModel
    )

    # Get var_lab from cache metadata via db
    cache_info = cache_manager.db.get_cache_info(config.filename, step, variable_key)
    var_lab = cache_info.get('var_lab') if cache_info else None
    if not var_lab:
        var_lab = config.var_name  # Fallback to var_name

    return quality_filtered_text, var_lab, variable_key


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(config: ExperimentConfig = None):
    """
    Run the ideaExtractor experiment with the given configuration.

    Args:
        config: Experiment configuration (uses EXPERIMENT_CONFIG if None)

    Returns:
        List of IdeasExtractedModel instances
    """
    if config is None:
        config = EXPERIMENT_CONFIG

    print("\n" + "=" * 80)
    print("IDEA EXTRACTOR EXPERIMENT")
    print("=" * 80)

    # Print configuration
    print("\n📋 Configuration:")
    print(f"   Filename: {config.filename}")
    print(f"   Variable: {config.var_name}")
    print(f"   Sample size: {config.sample_size}")
    print(f"   Experimental extractor: {config.use_experimental_extractor}")
    print(f"   Experimental prompts: {config.use_experimental_prompts}")
    print(f"   Max responses: {config.max_responses or 'All'}")
    print(f"   Verbose: {config.verbose}")
    print(f"   Prompt printer: {config.prompt_printer_enabled}")

    # Handle prompt injection BEFORE importing IdeaExtractor
    if config.use_experimental_prompts and not config.use_experimental_extractor:
        print("\n🔧 Injecting experimental prompts into production extractor...")
        inject_experimental_prompts()

    # Import the appropriate IdeaExtractor
    # Note: After migration, v3 (taxonomy-aware) is now production
    if config.use_experimental_extractor:
        print("\n🔬 Using PRODUCTION ideaExtractor (taxonomy-aware, migrated from v3)")
        from utils.ideaExtractor import IdeaExtractor
    else:
        print("\n🏭 Using OLD ideaExtractor (archived v2)")
        from utils.old.ideaExtractor_v2 import IdeaExtractor

    # Import other utilities
    from utils import verboseReporter, promptPrinter
    from config import ModelConfig

    # Load Step 2 cache
    print("\n📂 Loading Step 2 (quality_filter) cache...")
    try:
        quality_filtered_text, var_lab, variable_key = load_step2_cache(config)
        print(f"   ✅ Loaded {len(quality_filtered_text)} responses")
        print(f"   Variable label: {var_lab}")
        print(f"   Cache key: {variable_key}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return None

    # Filter out quality-filtered responses (matching pipeline step_3)
    filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
    print(f"\n📊 Response Statistics:")
    print(f"   Total responses: {len(quality_filtered_text)}")
    print(f"   Meaningful responses: {len(filtered_text)}")
    print(f"   Filtered out: {len(quality_filtered_text) - len(filtered_text)}")

    # Apply max_responses limit if set
    if config.max_responses and len(filtered_text) > config.max_responses:
        print(f"\n⚡ Limiting to first {config.max_responses} responses for quick experiment")
        filtered_text = filtered_text[:config.max_responses]

    # Initialize utilities
    verbose_reporter = verboseReporter.VerboseReporter(config.verbose)
    prompt_printer = promptPrinter.PromptPrinter(
        enabled=config.prompt_printer_enabled,
        print_realtime=True
    )
    model_config = ModelConfig()

    # Run idea extraction
    print("\n" + "-" * 80)
    print("RUNNING IDEA EXTRACTION")
    print("-" * 80)

    start_time = time.time()

    extractor = IdeaExtractor(
        responses=filtered_text,
        var_lab=var_lab,
        model_config=model_config,
        verbose=config.verbose,
        prompt_printer=prompt_printer
    )

    extracted_ideas = extractor.extract()

    elapsed_time = time.time() - start_time

    # Cache the results for pipeline continuity
    print("\n📦 Caching extracted ideas...")
    try:
        from utils.cacheManager import CacheManager
        from config import CacheConfig
        cache_manager = CacheManager(CacheConfig())
        cache_manager.save_to_cache(
            extracted_ideas,
            config.filename,
            "extracted_ideas",  # Must match STEP_NAMES[3] in pipeline.py
            variable_key,
            elapsed_time,
            var_lab=var_lab
        )
        print(f"   ✅ Cached {len(extracted_ideas)} results to step 'extracted_ideas'")

        # NEW: Build and cache extraction metadata separately
        if config.use_experimental_extractor and hasattr(extractor, 'build_extraction_metadata'):
            print("\n📦 Caching extraction metadata...")
            extraction_metadata = extractor.build_extraction_metadata(
                filename=config.filename,
                var_name=config.var_name
            )
            cache_manager.save_metadata_to_cache(
                extraction_metadata,
                config.filename,
                "extracted_ideas",  # Same step name, will be stored with _metadata suffix
                variable_key,
                elapsed_time,
                var_lab=var_lab
            )
            print(f"   ✅ Cached extraction metadata:")
            print(f"       lang={extraction_metadata.lang}, domain={extraction_metadata.domain}")
            print(f"       topic={extraction_metadata.topic}, intent={extraction_metadata.intent}")
            print(f"       taxonomy_primary_axis={extraction_metadata.taxonomy_primary_axis}")
            if extraction_metadata.taxonomy_axis_description:
                print(f"       taxonomy_axis_description={extraction_metadata.taxonomy_axis_description}")
            if extraction_metadata.taxonomy_sample_phrases:
                print(f"       taxonomy_sample_phrases={extraction_metadata.taxonomy_sample_phrases}")
    except Exception as e:
        print(f"   ⚠️ Cache save failed: {e}")

    # Calculate statistics
    total_ideas = sum(item.idea_count for item in extracted_ideas)
    avg_ideas = total_ideas / len(extracted_ideas) if extracted_ideas else 0

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Extraction Statistics:")
    print(f"   Responses processed: {len(extracted_ideas)}")
    print(f"   Total ideas extracted: {total_ideas}")
    print(f"   Average ideas per response: {avg_ideas:.2f}")
    print(f"   Elapsed time: {elapsed_time:.2f}s")
    print(f"   Rate: {len(extracted_ideas) / elapsed_time:.1f} responses/sec")

    # Display taxonomy axis information
    if hasattr(extractor, 'taxonomy_axis') and extractor.taxonomy_axis:
        print(f"\n🎯 Taxonomy Axis:")
        print(f"   Primary axis: {extractor.taxonomy_axis}")
        if hasattr(extractor, 'secondary_axis') and extractor.secondary_axis:
            print(f"   Secondary axis: {extractor.secondary_axis}")
        if hasattr(extractor, 'taxonomy_rationale') and extractor.taxonomy_rationale:
            rationale = extractor.taxonomy_rationale[:150] + "..." if len(extractor.taxonomy_rationale) > 150 else extractor.taxonomy_rationale
            print(f"   Rationale: {rationale}")
        if hasattr(extractor, 'template_prefix') and extractor.template_prefix:
            print(f"   Template prefix: \"{extractor.template_prefix}\"")
        if hasattr(extractor, 'taxonomy_actionable_type') and extractor.taxonomy_actionable_type:
            print(f"   Actionable type: \"{extractor.taxonomy_actionable_type}\"")

    # Display sample outputs
    if extracted_ideas and config.sample_output_count > 0:
        print_sample_outputs(extracted_ideas, config.sample_output_count)

    # Print prompt summary if enabled
    if config.prompt_printer_enabled:
        print("\n")
        prompt_printer.print_summary()

    return extracted_ideas


def print_sample_outputs(results: List, n_samples: int = 10):
    """
    Print sample outputs from the extraction results.

    Args:
        results: List of IdeasExtractedModel instances
        n_samples: Number of samples to display
    """
    # Filter to responses that have ideas
    responses_with_ideas = [r for r in results if r.response_ideas]

    if not responses_with_ideas:
        print("\n⚠️ No responses with extracted ideas found")
        return

    n_samples = min(n_samples, len(responses_with_ideas))
    samples = random.sample(responses_with_ideas, n_samples)

    print("\n" + "=" * 80)
    print(f"SAMPLE OUTPUTS ({n_samples} responses)")
    print("=" * 80)

    for i, item in enumerate(samples, 1):
        print(f"\n[{i}] Respondent: {item.respondent_id}")
        print(f"    Response: {item.response[:200]}{'...' if len(item.response) > 200 else ''}")
        print(f"    Ideas ({len(item.response_ideas)}):")

        for segment in item.response_ideas:
            # Build metadata from separate fields
            metadata_parts = []
            if segment.taxonomy_phrase:
                metadata_parts.append(f"phrase=\"{segment.taxonomy_phrase}\"")
            if segment.parent_category:
                metadata_parts.append(f"category=\"{segment.parent_category}\"")
            if segment.sentiment:
                metadata_parts.append(f"sentiment={segment.sentiment}")
            if segment.sense:
                metadata_parts.append(f"sense={segment.sense}")

            metadata_str = f" ({', '.join(metadata_parts)})" if metadata_parts else ""
            print(f"      [{segment.idea_id}]: {segment.idea}{metadata_str}")

    print("\n" + "-" * 80)


# =============================================================================
# OUTPUT CAPTURE
# =============================================================================

class TeeOutput:
    """Capture stdout while also printing to console."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original_stdout.flush()

    def get_output(self) -> str:
        return self.buffer.getvalue()


def save_results_to_file(output: str, config: ExperimentConfig) -> Path:
    """
    Save experiment results to a text file.

    Args:
        output: The captured console output
        config: Experiment configuration

    Returns:
        Path to the saved file
    """
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "exports" / "ideaextractor_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    base_name = Path(config.filename).stem
    sample_str = str(config.sample_size) if config.sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_mode = "experimental" if config.use_experimental_extractor else "production"

    output_filename = f"ideaextractor_results_{base_name}_{config.var_name}_{sample_str}_{exp_mode}_{date_str}.txt"
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        results = run_experiment()
    finally:
        # Restore stdout
        sys.stdout = tee.original_stdout

    if results:
        print("\n✅ Experiment completed successfully")
        print(f"   Results contain {len(results)} processed responses")

        # Save results to file if enabled
        if EXPERIMENT_CONFIG.save_results_to_file:
            output_path = save_results_to_file(tee.get_output(), EXPERIMENT_CONFIG)
            print(f"\n📄 Results saved to: {output_path}")
    else:
        print("\n❌ Experiment failed")
        sys.exit(1)
