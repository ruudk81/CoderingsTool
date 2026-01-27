#%%
"""
CodeGenerator V2 Experiment Runner

Runs Step 6 (codebook generation) in isolation for experimentation.
Loads Step 5 (initial_clusters) from cache and runs InductiveCodeGenerator.

This matches the exact behavior of pipeline.py Step 6, allowing prompt
experimentation via local prompts.py injection.

Usage:
    cd src && python -m experiments.codeGenerator_v2.run_experiment
"""

import os
import sys

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
import io
from datetime import datetime
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

from experiments.codeGenerator_v2.config import EXPERIMENT_CONFIG, ExperimentConfig


# =============================================================================
# PROMPT INJECTION
# =============================================================================

def inject_experimental_prompts():
    """
    Inject local prompts into the production prompts module.

    This allows experimenting with prompt modifications without
    changing the production src/prompts.py file.
    """
    from experiments.codeGenerator_v2 import prompts as local_prompts
    import prompts as production_prompts

    prompt_names = [
        'CLUSTER_SUMMARY_PROMPT',
        'CODING_DECISION_PROMPT',
        'CODE_CREATION_PROMPT',
        'VERTICAL_INSTRUCTIONS',
        'HIERARCHICAL_INSTRUCTIONS',
        'CODING_MODIFICATION_PROMPT',
        'VALIDATION_PROMPT',
    ]

    for name in prompt_names:
        if hasattr(local_prompts, name):
            setattr(production_prompts, name, getattr(local_prompts, name))
            print(f"   Injected: {name}")


# =============================================================================
# CACHE LOADING (matches pipeline.py Step 5-6)
# =============================================================================

def load_step5_cache(config: ExperimentConfig):
    """
    Load Step 5 (initial_clusters) results from cache.

    Matches pipeline.py behavior exactly.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (cluster_results, var_lab, variable_key)
    """
    import models
    from utils.cacheManager import CacheManager, generate_enhanced_variable_key
    from config import CacheConfig

    # Generate cache key matching pipeline behavior
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )

    cache_manager = CacheManager(CacheConfig())
    step = "initial_clusters"

    # Check if cache exists
    if not cache_manager.is_cache_valid(config.filename, step, variable_key):
        raise FileNotFoundError(
            f"Step 5 cache not found for:\n"
            f"  Filename: {config.filename}\n"
            f"  Variable: {config.var_name}\n"
            f"  Sample size: {config.sample_size}\n"
            f"  Variable key: {variable_key}\n\n"
            f"Please run the pipeline through Step 5 first."
        )

    # Load the cached data
    cluster_results = cache_manager.load_from_cache(
        filename=config.filename,
        step=step,
        variable_key=variable_key,
        model_cls=models.ClusterModel
    )

    # Get var_lab from cache metadata
    cache_info = cache_manager.db.get_cache_info(config.filename, step, variable_key)
    var_lab = cache_info.get('var_lab') if cache_info else None
    if not var_lab:
        var_lab = config.var_name  # Fallback

    return cluster_results, var_lab, variable_key


def load_starter_codes(config: ExperimentConfig, variable_key: str):
    """
    Load starter codes from cluster_representations cache.

    Matches pipeline.py lines 1245-1263 exactly:
        for rep in rep_data['representations']:
            if rep.get('llm_label'):
                label = rep['llm_label']
                starter_codes.append({
                    'code': label.get('theme', ''),
                    'definition': label.get('description', ''),
                    'cluster_id': label.get('cluster_id')
                })

    Args:
        config: Experiment configuration
        variable_key: The cache variable key

    Returns:
        List of starter code dicts, or empty list if not found
    """
    from utils.cacheManager import CacheManager
    from config import CacheConfig

    cache_manager = CacheManager(CacheConfig())
    step = "cluster_representations"

    if not cache_manager.is_cache_valid(config.filename, step, variable_key):
        print("   No cluster_representations cache found")
        return []

    try:
        # Load as dict since it was cached with .model_dump()
        representations_data = cache_manager.load_from_cache(
            filename=config.filename,
            step=step,
            variable_key=variable_key
        )

        if not representations_data:
            return []

        # Handle list vs single dict
        rep_data = representations_data[0] if isinstance(representations_data, list) else representations_data

        if not isinstance(rep_data, dict) or 'representations' not in rep_data:
            print("   Warning: Unexpected representations format")
            return []

        # Extract starter codes from LLM labels (matches pipeline.py exactly)
        starter_codes = []
        for rep in rep_data['representations']:
            if rep.get('llm_label'):
                label = rep['llm_label']
                starter_codes.append({
                    'code': label.get('theme', ''),
                    'definition': label.get('description', ''),
                    'cluster_id': label.get('cluster_id')
                })

        return starter_codes

    except Exception as e:
        print(f"   Warning: Could not load cluster_representations: {e}")
        return []


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(config: ExperimentConfig = None):
    """
    Run the codeGenerator experiment.

    Matches pipeline.py Step 6 behavior:
    1. Load initial_clusters from Step 5 cache
    2. Load starter_codes from cluster_representations cache
    3. Clean cluster ideas
    4. Run InductiveCodeGenerator
    5. Cache results

    Args:
        config: Experiment configuration (uses EXPERIMENT_CONFIG if None)

    Returns:
        CodeGeneratorReasoningResults instance
    """
    if config is None:
        config = EXPERIMENT_CONFIG

    print("\n" + "=" * 80)
    print("CODEGENERATOR V2 EXPERIMENT")
    print("=" * 80)

    # Print configuration
    print("\n Configuration:")
    print(f"   Filename: {config.filename}")
    print(f"   Variable: {config.var_name}")
    print(f"   Sample size: {config.sample_size}")
    print(f"   Use experimental prompts: {config.use_experimental_prompts}")
    print(f"   Stages to run: {config.stages_to_run}")
    print(f"   Verbose: {config.verbose}")

    # Step 1: Inject experimental prompts if enabled
    if config.use_experimental_prompts:
        print("\n Injecting experimental prompts...")
        inject_experimental_prompts()

    # Import codeGenerator (after prompt injection)
    from utils.codeGenerator import InductiveCodeGenerator
    from utils import verboseReporter, promptPrinter, clusterer

    # Step 2: Load Step 5 cache
    print("\n Loading Step 5 (initial_clusters) cache...")
    try:
        cluster_results, var_lab, variable_key = load_step5_cache(config)
        print(f"   Loaded {len(cluster_results)} clustered items")
        print(f"   Variable label: {var_lab}")
        print(f"   Cache key: {variable_key}")
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        return None

    # Step 3: Clean cluster ideas (matching pipeline behavior)
    print("\n Cleaning cluster ideas...")
    cleaned_results = clusterer.clean_cluster_ideas(cluster_results)
    print(f"   Cleaned {len(cleaned_results)} response items")

    # Count cluster statistics
    all_clusters = set()
    noise_count = 0
    total_ideas = 0
    for result in cleaned_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                total_ideas += 1
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    all_clusters.add(idea.initial_cluster)
                elif idea.initial_cluster == -1:
                    noise_count += 1
    print(f"   Total ideas: {total_ideas}")
    print(f"   Unique clusters: {len(all_clusters)}")
    print(f"   Noise points (cluster -1): {noise_count}")

    # Step 4: Load starter codes
    print("\n Loading starter codes from cluster_representations cache...")
    starter_codes = load_starter_codes(config, variable_key)
    if starter_codes:
        print(f"   Loaded {len(starter_codes)} starter codes from ClustererV2 LLM labels")
    else:
        print("   No starter codes found - proceeding with empty starter codes")

    # Initialize utilities
    verbose_reporter = verboseReporter.VerboseReporter(config.verbose)
    prompt_printer = promptPrinter.PromptPrinter(
        enabled=config.prompt_printer_enabled,
        print_realtime=True
    )

    # Step 5: Run InductiveCodeGenerator (matches pipeline.py)
    print("\n" + "-" * 80)
    print("RUNNING CODEBOOK GENERATION")
    print("-" * 80)

    start_time = time.time()

    generator = InductiveCodeGenerator(
        cluster_results=cleaned_results,
        starter_codes=starter_codes,
        var_lab=var_lab,
        verbose=config.verbose,
        verbose_detailed=config.verbose_detailed,
        prompt_printer=prompt_printer,
        stages_to_run=config.stages_to_run
    )

    results = generator.generate()

    elapsed_time = time.time() - start_time

    # Step 6: Cache the results
    print("\n Caching codebook results...")
    try:
        from utils.cacheManager import CacheManager
        from config import CacheConfig
        cache_manager = CacheManager(CacheConfig())
        cache_manager.save_to_cache(
            results,
            config.filename,
            "codebook",  # Matches STEP_NAMES[6] in pipeline.py
            variable_key,
            elapsed_time,
            var_lab=var_lab
        )
        print(f"   Cached results to step 'codebook'")
    except Exception as e:
        print(f"   Warning: Cache save failed: {e}")

    # Print statistics
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print_statistics(results, elapsed_time)

    # Display sample codebook
    if results and config.sample_codebook_count > 0:
        print_sample_codebook(results, config.sample_codebook_count)

    # Print prompt summary if enabled
    if config.prompt_printer_enabled:
        print("\n")
        prompt_printer.print_summary()

    return results


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_statistics(results, elapsed_time: float):
    """Print statistics about the codebook generation."""
    print(f"\n Generation Statistics:")
    print(f"   Elapsed time: {elapsed_time:.2f}s")

    if results is None:
        print("   No results to display")
        return

    # Count codes
    if hasattr(results, 'codebook') and results.codebook:
        total_codes = len(results.codebook)
        print(f"   Total codes in codebook: {total_codes}")

    # Count decisions by type
    if hasattr(results, 'code_decisions') and results.code_decisions:
        decisions = results.code_decisions
        use_count = sum(1 for d in decisions if d.decision == 'USE')
        modify_count = sum(1 for d in decisions if d.decision == 'MODIFY')
        create_count = sum(1 for d in decisions if d.decision == 'CREATE')

        print(f"\n Decision Breakdown:")
        print(f"   USE (existing code): {use_count}")
        print(f"   MODIFY (extend code): {modify_count}")
        print(f"   CREATE (new code): {create_count}")

    # Count clusters processed
    if hasattr(results, 'cluster_themes') and results.cluster_themes:
        print(f"\n Clusters Processed:")
        print(f"   Clusters with themes: {len(results.cluster_themes)}")
        total_themes = sum(len(ct.themes) for ct in results.cluster_themes)
        print(f"   Total themes extracted: {total_themes}")


def print_sample_codebook(results, n_samples: int = 10):
    """Print sample codes from the generated codebook."""
    if not hasattr(results, 'codebook') or not results.codebook:
        print("\n No codebook to display")
        return

    codebook = results.codebook
    n_samples = min(n_samples, len(codebook))

    print("\n" + "=" * 80)
    print(f"SAMPLE CODEBOOK ({n_samples} of {len(codebook)} codes)")
    print("=" * 80)

    samples = codebook[:n_samples]

    for i, code in enumerate(samples, 1):
        # Handle both dict and object access patterns
        if isinstance(code, dict):
            code_label = code.get('code_label', code.get('code', 'Unknown'))
            code_def = code.get('code_definition', code.get('definition', ''))
            source = code.get('source_code')
            examples = code.get('assignment_examples', {})
        else:
            code_label = getattr(code, 'code_label', getattr(code, 'code', 'Unknown'))
            code_def = getattr(code, 'code_definition', getattr(code, 'definition', ''))
            source = getattr(code, 'source_code', None)
            examples = getattr(code, 'assignment_examples', {})

        print(f"\n[{i}] Code: {code_label}")
        print(f"    Definition: {code_def}")
        if source:
            print(f"    Source: {source}")
        if examples:
            if isinstance(examples, dict):
                inclusion = examples.get('inclusion', [])
                exclusion = examples.get('exclusion', [])
            else:
                inclusion = getattr(examples, 'inclusion', [])
                exclusion = getattr(examples, 'exclusion', [])
            if inclusion:
                print(f"    Inclusion: {', '.join(str(x) for x in inclusion[:2])}")
            if exclusion:
                print(f"    Exclusion: {', '.join(str(x) for x in exclusion[:2])}")

    print("\n" + "-" * 80)


# =============================================================================
# OUTPUT CAPTURE AND FILE SAVING
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
    """Save experiment results to a text file."""
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "exports" / "codegenerator_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    base_name = Path(config.filename).stem
    sample_str = str(config.sample_size) if config.sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_mode = "experimental" if config.use_experimental_prompts else "production"

    output_filename = f"codegenerator_{base_name}_{config.var_name}_{sample_str}_{prompt_mode}_{date_str}.txt"
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
        print("\n Experiment completed successfully")

        if hasattr(results, 'codebook') and results.codebook:
            print(f"   Codebook contains {len(results.codebook)} codes")

        # Save results to file if enabled
        if EXPERIMENT_CONFIG.save_results_to_file:
            output_path = save_results_to_file(tee.get_output(), EXPERIMENT_CONFIG)
            print(f"\n Results saved to: {output_path}")
    else:
        print("\n Experiment failed")
        sys.exit(1)
