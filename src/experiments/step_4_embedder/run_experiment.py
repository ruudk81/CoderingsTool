#%%

"""
Step 4: Embedder Experiment Runner

Runs the embedding generation step in isolation for experimentation.
Loads Step 3 (extracted_ideas) results from cache and generates embeddings.

Usage:
    cd src && python -m experiments.step_4_embedder.run_experiment

Toggle:
    USE_EXPERIMENTAL = True  -> Uses experimental embedder from this folder
    USE_EXPERIMENTAL = False -> Uses production embedder from utils/
"""

USE_EXPERIMENTAL = True

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
from config_steps.config_embedder import EmbedderConfig

# Import experimental config when enabled
if USE_EXPERIMENTAL:
    try:
        from .config_exp import EmbedderConfigExp
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from config_exp import EmbedderConfigExp
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.verboseReporter import VerboseReporter
from utils.saveVerbose import VerboseCapture
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
    use_experimental: bool = USE_EXPERIMENTAL
    verbose: bool = True
    force_recalc: bool = True


EXPERIMENT_CONFIG = ExperimentConfig()

# =============================================================================
# TOGGLE: PRODUCTION vs EXPERIMENTAL
# =============================================================================
USE_EXPERIMENTAL = EXPERIMENT_CONFIG.use_experimental

if USE_EXPERIMENTAL:
    try:
        from .embedder_exp import Embedder
    except ImportError:
        exp_dir = Path(__file__).parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from embedder_exp import Embedder
    print("[EXPERIMENTAL] Using embedder_exp.py from experiments folder")
else:
    from utils.embedder import Embedder
    print("[PRODUCTION] Using embedder.py from utils/")


# =============================================================================
# CACHE OPERATIONS
# =============================================================================
def load_step3_cache(config: ExperimentConfig):
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size
    )
    cache_manager = CacheManager(CacheConfig())

    step_name = "extracted_ideas"

    if not cache_manager.is_cache_valid(config.filename, step_name, variable_key):
        raise FileNotFoundError(
            f"Cache not found: {step_name}/{variable_key}\n"
            f"Run pipeline.py with RUN_UNTIL_STEP=3 first."
        )

    data = cache_manager.load_from_cache(
        config.filename, step_name, variable_key, models.IdeasExtractedModel
    )

    # Also load extraction metadata if available
    extraction_metadata = None
    try:
        extraction_metadata = cache_manager.load_metadata_from_cache(
            config.filename, step_name, variable_key, models.ExtractionMetadata
        )
    except:
        pass

    return data, variable_key, cache_manager, extraction_metadata


def get_var_lab(config: ExperimentConfig) -> str:
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    return loader.get_varlab(filename=config.filename, var_name=config.var_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config: ExperimentConfig = None):
    if config is None:
        config = EXPERIMENT_CONFIG

    encoded_text, variable_key, cache_manager, extraction_metadata = load_step3_cache(config)
    var_lab = get_var_lab(config)

    model_config = ModelConfig()
    verbose_reporter = VerboseReporter(config.verbose)

    verbose_reporter.section_header("EMBEDDING GENERATION EXPERIMENT")
    verbose_reporter.stat_line(f"Variable: {config.var_name} - {var_lab}")
    verbose_reporter.stat_line(f"Using experimental: {USE_EXPERIMENTAL}")

    total_ideas = sum(item.idea_count for item in encoded_text)
    verbose_reporter.stat_line(f"Input: {len(encoded_text)} responses with {total_ideas} ideas")

    start_time = time.time()

    # Initialize embedder — use experimental config when enabled
    if USE_EXPERIMENTAL:
        embedder_config = EmbedderConfigExp(verbose=config.verbose)
        verbose_reporter.stat_line(f"Config: EmbedderConfigExp (experimental)")
    else:
        embedder_config = EmbedderConfig(verbose=config.verbose)
        verbose_reporter.stat_line(f"Config: EmbedderConfig (production)")
    embedder = Embedder(
        config=embedder_config,
        model_config=model_config,
        var_lab=var_lab
    )

    if extraction_metadata:
        embedder.set_extraction_metadata(extraction_metadata)
        verbose_reporter.stat_line(f"Loaded extraction metadata (template_prefix available)")

    # Generate embeddings
    input_data = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
    embedded_text = embedder.get_embeddings_with_tracking(input_data, var_lab)

    elapsed_time = time.time() - start_time

    # Count embeddings per type
    embed_fields = ['idea_embedding', 'concept_embedding', 'concept_type_embedding', 'ladder_embedding']
    for field in embed_fields:
        count = sum(
            1 for resp in embedded_text
            if resp.response_ideas
            for idea in resp.response_ideas
            if getattr(idea, field, None) is not None
        )
        verbose_reporter.stat_line(f"  {field}: {count}")

    cache_manager.save_to_cache(embedded_text, config.filename, "embeddings", variable_key, elapsed_time, var_lab=var_lab)

    total_ideas = sum(len(resp.response_ideas) for resp in embedded_text if resp.response_ideas)
    verbose_reporter.stat_line(f"Output: {total_ideas} ideas embedded")
    verbose_reporter.stat_line(f"Rate: {total_ideas / elapsed_time:.1f} ideas/sec")
    print(f"\n'Embedding experiment' completed in {elapsed_time:.2f} seconds.\n")

    return embedded_text, embedder


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
        run_until_step=4
    )
    verbose_capture.__enter__()

    token_tracker.reset()

    print("=" * 70)
    print("EXPERIMENT: Step 4 - Embedder")
    print("=" * 70)
    print(f"Dataset: {config.filename}")
    print(f"Variable: {config.var_name} - {var_lab}")
    print(f"Sample size: {config.sample_size}")
    print(f"Using experimental: {USE_EXPERIMENTAL}")
    print("=" * 70)

    try:
        results, embedder = run_experiment(config)

        # Print embedding analysis if available
        if embedder.analysis:
            print("\n" + "=" * 70)
            print("EMBEDDING ANALYSIS")
            print("=" * 70)
            analysis = embedder.analysis
            print(f"Dimensions: {analysis.embedding_dim}")
            print(f"Norm: mean={analysis.mean_norm:.4f}, std={analysis.std_norm:.4f}")
            if analysis.mean_pairwise_similarity is not None:
                print(f"Pairwise similarity: mean={analysis.mean_pairwise_similarity:.4f}")

        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())

    finally:
        verbose_capture.__exit__(None, None, None)
