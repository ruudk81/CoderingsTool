#%%
"""
Embedder V2 Experiment Runner

This script runs the embedder step (Step 4) in isolation for experimentation.
It loads Step 3 (extracted_ideas) results from cache and runs embedding generation
with configurable settings for provider, model, text format, and analysis options.

Usage:
    cd src && python -m experiments.embedder_v2.run_experiment

Toggle modes:
    use_experimental_embedder = True  -> Uses local ExperimentalEmbedder
    use_experimental_embedder = False -> Uses production Embedder
"""

import os
import sys
import io
from pathlib import Path
from datetime import datetime

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
import random
from typing import List, Optional, Tuple

import numpy as np
import nest_asyncio
nest_asyncio.apply()

import models

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

from experiments.embedder_v2.config import (
    EmbedderExperimentConfig,
    DEFAULT_EXPERIMENT_CONFIG,
    ASN_BANK_CONFIG,
    PINKPOP_CONFIG,
    QUESTION_AWARE_CONFIG,
    GEMINI_CONFIG,
)

# Select which configuration to use
EXPERIMENT_CONFIG = EmbedderExperimentConfig(
    # Dataset settings
    filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
    id_column="DLNMID",
    var_name="Q20",
    sample_size=500,
    #sample_size=50,

    # Uncomment for larger datasets:
    #filename="M250480 Associatiemonitor ASN Bank net databestand.sav",
    #id_column="DLNMID",
    #var_name="Qd1_combined",
    #sample_size=2000,

    # Provider settings
    provider="openai",
    embedding_model=None,  # Use default from ModelConfig

    # Text format: "idea", "taxonomy_phrase", "idea_without_template_prefix", or "both"
    # using BOTH_MODE_IDEA_FORMAT logic = without prefix)
    embedding_text_format="both",

    # Question-aware embeddings
    use_question_aware=False,

    # Experiment features
    use_experimental_embedder=True,
    analyze_embeddings=True,
    compute_similarity_stats=True,

    # Output
    verbose=True,
    sample_output_count=10,
    save_results_to_file=True,
)


# =============================================================================
# CACHE LOADING
# =============================================================================

def load_step3_cache(config: EmbedderExperimentConfig):
    """
    Load Step 3 (extracted_ideas) results from cache.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (extracted_ideas, var_lab, variable_key)
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
    step = "extracted_ideas"

    # Check if cache exists
    if not cache_manager.is_cache_valid(config.filename, step, variable_key):
        raise FileNotFoundError(
            f"Step 3 cache not found for:\n"
            f"  Filename: {config.filename}\n"
            f"  Variable: {config.var_name}\n"
            f"  Sample size: {config.sample_size}\n"
            f"  Variable key: {variable_key}\n\n"
            f"Please run the pipeline through Step 3 first to generate the cache."
        )

    # Load the cached data
    extracted_ideas = cache_manager.load_from_cache(
        filename=config.filename,
        step=step,
        variable_key=variable_key,
        model_cls=models.IdeasExtractedModel
    )

    # Get var_lab from cache metadata
    cache_info = cache_manager.db.get_cache_info(config.filename, step, variable_key)
    var_lab = cache_info.get('var_lab') if cache_info else None
    if not var_lab:
        var_lab = config.var_name  # Fallback to var_name

    return extracted_ideas, var_lab, variable_key


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(config: EmbedderExperimentConfig = None):
    """
    Run the embedder experiment with the given configuration.

    Args:
        config: Experiment configuration (uses EXPERIMENT_CONFIG if None)

    Returns:
        List of EmbeddingsModel instances
    """
    if config is None:
        config = EXPERIMENT_CONFIG

    print("\n" + "=" * 80)
    print("EMBEDDER V2 EXPERIMENT")
    print("=" * 80)

    # Print configuration
    print("\n📋 Configuration:")
    print(f"   Filename: {config.filename}")
    print(f"   Variable: {config.var_name}")
    print(f"   Sample size: {config.sample_size}")
    print(f"   Provider: {config.provider}")
    print(f"   Embedding model: {config.embedding_model or '(default)'}")
    print(f"   Text format: {config.embedding_text_format}")
    print(f"   Question-aware: {config.use_question_aware}")
    print(f"   Experimental embedder: {config.use_experimental_embedder}")
    print(f"   Analyze embeddings: {config.analyze_embeddings}")
    print(f"   Compute similarity stats: {config.compute_similarity_stats}")
    print(f"   Verbose: {config.verbose}")

    # Import the appropriate Embedder
    if config.use_experimental_embedder:
        print("\n🔬 Using EXPERIMENTAL embedder (local)")
        from experiments.embedder_v2.embedder_experimental import ExperimentalEmbedder
    else:
        print("\n🏭 Using PRODUCTION embedder")
        from utils.embedder import Embedder

    # Import other utilities
    from config import ModelConfig
    import models

    # Load Step 3 cache
    print("\n📂 Loading Step 3 (extracted_ideas) cache...")
    try:
        extracted_ideas, var_lab, variable_key = load_step3_cache(config)
        print(f"   ✅ Loaded {len(extracted_ideas)} responses")
        print(f"   Variable label: {var_lab}")
        print(f"   Cache key: {variable_key}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return None

    # Count total ideas
    total_ideas = sum(item.idea_count for item in extracted_ideas)
    print(f"\n📊 Data Statistics:")
    print(f"   Total responses: {len(extracted_ideas)}")
    print(f"   Total ideas: {total_ideas}")
    print(f"   Average ideas per response: {total_ideas / len(extracted_ideas):.2f}")

    # Try to load extraction metadata (for taxonomy_with_context mode)
    extraction_metadata = None
    try:
        from utils.cacheManager import CacheManager
        from config import CacheConfig
        cache_manager = CacheManager(CacheConfig())
        if cache_manager.is_metadata_cache_valid(config.filename, "extracted_ideas", variable_key):
            extraction_metadata = cache_manager.load_metadata_from_cache(
                config.filename,
                "extracted_ideas",
                variable_key,
                models.ExtractionMetadata
            )
            if extraction_metadata:
                print(f"\n📋 Loaded extraction metadata:")
                print(f"   domain={extraction_metadata.domain}")
                print(f"   topic={extraction_metadata.topic}")
                print(f"   intent={extraction_metadata.intent}")
                print(f"   taxonomy_primary_axis={extraction_metadata.taxonomy_primary_axis}")
                if extraction_metadata.taxonomy_axis_description:
                    print(f"   taxonomy_axis_description={extraction_metadata.taxonomy_axis_description}")
                if extraction_metadata.taxonomy_sample_phrases:
                    print(f"   taxonomy_sample_phrases={extraction_metadata.taxonomy_sample_phrases}")
    except Exception as e:
        print(f"\n⚠️ Could not load extraction metadata: {e}")

    # Convert to EmbeddingsModel format
    model_config = ModelConfig()
    input_data = [item.to_model(models.EmbeddingsModel) for item in extracted_ideas]

    # Run embedding generation
    print("\n" + "-" * 80)
    print("RUNNING EMBEDDING GENERATION")
    print("-" * 80)

    start_time = time.time()

    if config.use_experimental_embedder:
        embedder = ExperimentalEmbedder(
            experiment_config=config,
            model_config=model_config,
            var_lab=var_lab
        )
        # Pass extraction metadata for taxonomy_with_context mode
        if extraction_metadata:
            embedder.extraction_metadata = extraction_metadata
    else:
        from config import EmbeddingConfig
        embedding_config = config.to_embedding_config()
        embedder = Embedder(
            config=embedding_config,
            model_config=model_config,
            provider=config.provider,
            var_lab=var_lab,
            verbose=config.verbose
        )

    embedded_data = embedder.get_embeddings_with_tracking(input_data, var_lab)

    elapsed_time = time.time() - start_time

    # Cache the results for pipeline continuity
    print("\n📦 Caching embedded data...")
    try:
        from utils.cacheManager import CacheManager
        from config import CacheConfig
        cache_manager = CacheManager(CacheConfig())
        cache_manager.save_to_cache(
            embedded_data,
            config.filename,
            "embeddings",  # Must match STEP_NAMES[4] in pipeline.py
            variable_key,
            elapsed_time,
            var_lab=var_lab
        )
        print(f"   ✅ Cached {len(embedded_data)} results to step 'embeddings'")
    except Exception as e:
        print(f"   ⚠️ Cache save failed: {e}")

    # Calculate statistics
    embeddings_count = sum(
        1 for resp in embedded_data
        if resp.response_ideas
        for idea in resp.response_ideas
        if idea.idea_embedding is not None
    )

    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Embedding Statistics:")
    print(f"   Responses processed: {len(embedded_data)}")
    print(f"   Embeddings generated: {embeddings_count}")
    print(f"   Elapsed time: {elapsed_time:.2f}s")
    print(f"   Rate: {embeddings_count / elapsed_time:.1f} embeddings/sec")

    # Print analysis if available
    if config.use_experimental_embedder and hasattr(embedder, 'analysis') and embedder.analysis:
        analysis = embedder.analysis
        print(f"\n🔍 Embedding Analysis:")
        print(f"   Dimensions: {analysis.embedding_dim}")
        print(f"   Norm: mean={analysis.mean_norm:.4f}, std={analysis.std_norm:.4f}, "
              f"range=[{analysis.min_norm:.4f}, {analysis.max_norm:.4f}]")
        if analysis.mean_pairwise_similarity is not None:
            print(f"   Pairwise similarity: mean={analysis.mean_pairwise_similarity:.4f}, "
                  f"std={analysis.std_pairwise_similarity:.4f}, "
                  f"range=[{analysis.min_pairwise_similarity:.4f}, {analysis.max_pairwise_similarity:.4f}]")

    # Display sample outputs
    if embedded_data and config.sample_output_count > 0:
        print_sample_outputs(embedded_data, config.sample_output_count, config, extraction_metadata)

    # Clean up
    if hasattr(embedder, 'close'):
        embedder.close()

    return embedded_data


def _get_embedded_text(idea, embedding_text_format: str, template_prefix: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Reconstruct the text that was embedded based on the format mode.

    Args:
        idea: Idea object with idea text and taxonomy_phrase field
        embedding_text_format: The embedding format mode used
        template_prefix: The template_prefix for stripping (only used for idea_without_template_prefix mode)

    Returns:
        Tuple of (idea_embedded_text, taxonomy_embedded_text).
        taxonomy_embedded_text is None unless format is "both".
    """
    if embedding_text_format == "taxonomy_phrase":
        taxonomy_phrase = getattr(idea, 'taxonomy_phrase', '') or ''
        return (taxonomy_phrase if taxonomy_phrase else idea.idea, None)

    if embedding_text_format == "idea_without_template_prefix":
        idea_text = idea.idea
        if template_prefix and idea_text.startswith(template_prefix):
            unique_content = idea_text[len(template_prefix):].strip()
            return (unique_content if unique_content else idea_text, None)
        return (idea_text, None)

    if embedding_text_format == "both":
        # Get idea text (using BOTH_MODE_IDEA_FORMAT logic - without prefix)
        from experiments.embedder_v2.config import BOTH_MODE_IDEA_FORMAT
        idea_text = idea.idea
        if BOTH_MODE_IDEA_FORMAT == "idea_without_template_prefix":
            if template_prefix and idea_text.startswith(template_prefix):
                unique_content = idea_text[len(template_prefix):].strip()
                idea_text = unique_content if unique_content else idea.idea
        # Get taxonomy text
        taxonomy_phrase = getattr(idea, 'taxonomy_phrase', '') or ''
        return (idea_text, taxonomy_phrase if taxonomy_phrase else None)

    # Default: "idea" mode
    return (idea.idea, None)


def print_sample_outputs(
    results: List,
    n_samples: int = 10,
    config: 'EmbedderExperimentConfig' = None,
    extraction_metadata: Optional['models.ExtractionMetadata'] = None
):
    """
    Print sample outputs from the embedding results.

    Args:
        results: List of EmbeddingsModel instances
        n_samples: Number of samples to display
        config: Experiment config to determine what text was embedded
        extraction_metadata: Extraction metadata containing template_prefix
    """
    # Filter to responses that have embeddings
    responses_with_embeddings = [
        r for r in results
        if r.response_ideas and any(idea.idea_embedding is not None for idea in r.response_ideas)
    ]

    if not responses_with_embeddings:
        print("\n⚠️ No responses with embeddings found")
        return

    n_samples = min(n_samples, len(responses_with_embeddings))
    samples = random.sample(responses_with_embeddings, n_samples)

    # Get embedding format and template_prefix
    embedding_text_format = config.embedding_text_format if config else "idea"
    template_prefix = extraction_metadata.template_prefix if extraction_metadata else None

    print("\n" + "=" * 80)
    print(f"SAMPLE OUTPUTS ({n_samples} responses)")
    print(f"Embedding format: {embedding_text_format}")
    print("=" * 80)

    for i, item in enumerate(samples, 1):
        print(f"\n[{i}] Respondent: {item.respondent_id}")
        response_preview = item.response[:100] + "..." if len(item.response) > 100 else item.response
        print(f"    Response: {response_preview}")
        print(f"    Ideas ({len(item.response_ideas)}):")

        for idea in item.response_ideas:
            # Get the actual text that was embedded
            idea_embedded_text, taxonomy_embedded_text = _get_embedded_text(idea, embedding_text_format, template_prefix)

            # Display idea text (now clean - no embedded specifiers)
            idea_text = idea.idea
            cleaned = idea_text.strip()

            # Show embedding info
            print(f"      [{idea.idea_id}]: {cleaned[:60]}{'...' if len(cleaned) > 60 else ''}")

            # Show idea_embedding
            if idea.idea_embedding is not None:
                emb = np.array(idea.idea_embedding)
                norm = np.linalg.norm(emb)
                print(f"               idea_embedding: \"{idea_embedded_text}\"")
                print(f"               dim={len(emb)}, norm={norm:.4f}")
            else:
                print(f"               idea_embedding: None")

            # Show taxonomy_embedding (only if "both" mode or if it exists)
            taxonomy_emb = getattr(idea, 'taxonomy_embedding', None)
            if taxonomy_emb is not None:
                emb2 = np.array(taxonomy_emb)
                norm2 = np.linalg.norm(emb2)
                print(f"               taxonomy_embedding: \"{taxonomy_embedded_text or '(no taxonomy_phrase)'}\"")
                print(f"               dim={len(emb2)}, norm={norm2:.4f}")

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


def save_results_to_file(output: str, config: EmbedderExperimentConfig) -> Path:
    """
    Save experiment results to a text file.

    Args:
        output: The captured console output
        config: Experiment configuration

    Returns:
        Path to the saved file
    """
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "exports" / "embedding_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    base_name = Path(config.filename).stem
    sample_str = str(config.sample_size) if config.sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    format_str = config.embedding_text_format

    output_filename = f"embedding_results_{base_name}_{config.var_name}_{sample_str}_{format_str}_{date_str}.txt"
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
