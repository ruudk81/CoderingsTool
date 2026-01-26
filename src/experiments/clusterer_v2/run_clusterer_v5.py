#%%

"""
Clusterer Run Script

Run the Clusterer pipeline on a specific dataset from cached Step 4 embeddings.

Usage:
    cd src/experiments/clusterer_v2
    python run_clusterer.py

Configure the dataset by editing the variables below.
"""

import sys
import io
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import pickle

import numpy as np

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

from clusterer_v2 import ClustererV2, ClustererV2Config


# =============================================================================
# DATASET CONFIGURATION - Edit these to match your cached Step 4 data
# =============================================================================

#FILENAME = "M250480 Associatiemonitor ASN Bank net databestand.sav"
#VARIABLE = "Qd1_combined"
#SAMPLE_SIZE = 2000

FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VARIABLE = "Q20"
SAMPLE_SIZE = 500
#SAMPLE_SIZE = 50

# =============================================================================
# EMBEDDING SOURCE CONFIGURATION
# =============================================================================
# Which embedding to use for clustering:
# - "taxonomy_embedding": Cluster on taxonomy_phrase embeddings (default, recommended)
# - "idea_embedding": Cluster on idea text embeddings (stripped of template prefix)
# Note: Requires embeddings cached with embedding_text_format="both"
EMBEDDING_SOURCE = "taxonomy_embedding"

# =============================================================================
# CLUSTERER CONFIGURATION
# =============================================================================

CONFIG = ClustererV2Config(
    # Algorithm selection: "auto", "hdbscan", "agglomerative", "kmeans"
    algorithm_mode="auto",

    # Small dataset threshold: n <= 250 → always Agglomerative, no UMAP
    # K grid: log(n) to sqrt(n), scored by silhouette
    small_dataset_threshold=0,

    # DVC thresholds for algorithm selection (for n > small_dataset_threshold)
    dvc_high_threshold=0.45,    # Above this → HDBSCAN
    dvc_low_threshold=0.25,     # Below this → Agglomerative

    # Hard rule: force Agglomerative when DVC < this
    force_agglomerative_below_dvc=0.25,

    # Knee detection
    knee_y_diff_threshold=0.6,  # Sharp knee threshold

    # Optuna optimization (for HDBSCAN)
    use_optuna=True,
    max_noise_rate=0.20,        # Maximum acceptable noise rate
    min_clusters=3,             # Minimum number of clusters

    # Quality thresholds for conditional re-search
    # Trigger: (noise > max AND validity < min) OR (cluster_deviation > threshold)
    enable_research=True,
    research_max_noise_rate=0.10,
    research_min_validity=0.70,
    research_cluster_deviation_threshold=0.15,

    # Post-processing
    enable_merging=True,
    merge_centroid_threshold=0.95,
    merge_pairwise_threshold=0.98,

    # BERTopic-style noise reduction
    noise_reduction_strategy="embeddings",
    noise_reduction_threshold=0.5,

    # c-TF-IDF keyword extraction with lemmatization
    generate_ctfidf=True,
    ctfidf_top_k=10,
    ctfidf_use_lemmatization=True,

    # Additional representations (for comparison/display)
    generate_mmr_keywords=True,
    mmr_diversity=0.3,  # 0.0 = max diversity, 1.0 = max relevance
    generate_tfidf_keywords=True,

    # LLM labels (enabled)
    generate_llm_labels=True,

    # Output
    verbose=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step4_embeddings(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
    embedding_source: str = EMBEDDING_SOURCE
) -> Tuple[np.ndarray, List[str], List[models.EmbeddingsModel]]:
    """
    Load Step 4 embeddings from cache.

    Args:
        filename: Dataset filename
        variable: Variable name
        sample_size: Sample size used for caching
        variable_key: Optional explicit cache key
        embedding_source: Which embedding to use - "taxonomy_embedding" or "idea_embedding"

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        texts: list of text strings (taxonomy_phrase or idea depending on source)
        embeddings_models: list of EmbeddingsModel objects (for pipeline compatibility)
    """
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(filename).stem
    cache_filename = f"005_embeddings_{base_name}_{variable_key}.pkl"
    cache_path = cache_dir / cache_filename

    print(f"Loading embeddings from: {cache_path}")

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run pipeline step 4 first to generate embeddings."
        )

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    # Convert serialized data to EmbeddingsModel objects
    embeddings_models = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

    # Detect embedding_text_format from first response
    embedding_format = "idea"  # Default fallback
    if embeddings_models and hasattr(embeddings_models[0], 'embedding_text_format'):
        embedding_format = embeddings_models[0].embedding_text_format or "idea"

    print(f"Cached embedding format: {embedding_format}")
    print(f"Clustering on: {embedding_source}")

    # Validate embedding_source availability
    if embedding_source == "taxonomy_embedding" and embedding_format not in ["both", "taxonomy_phrase"]:
        raise ValueError(
            f"Cannot use taxonomy_embedding: cached data has embedding_text_format='{embedding_format}'. "
            f"Re-run embedder with embedding_text_format='both' or 'taxonomy_phrase'."
        )

    # Build embeddings array and texts list based on source
    embeddings_list = []
    texts = []

    for response in embeddings_models:
        if response.response_ideas:
            for idea in response.response_ideas:
                if embedding_source == "taxonomy_embedding":
                    # Use taxonomy_embedding and taxonomy_phrase text
                    emb = getattr(idea, 'taxonomy_embedding', None)
                    text = getattr(idea, 'taxonomy_phrase', '') or idea.idea
                else:
                    # Use idea_embedding and idea text
                    emb = idea.idea_embedding
                    text = idea.idea

                if emb is not None:
                    embeddings_list.append(emb)
                    texts.append(text)

    if not embeddings_list:
        raise ValueError(f"No {embedding_source} found in cached data")

    embeddings = np.vstack(embeddings_list)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    return embeddings, texts, embeddings_models


def load_extraction_metadata(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None
) -> Optional[models.ExtractionMetadata]:
    """
    Load ExtractionMetadata from cache (if available).

    Returns:
        ExtractionMetadata or None if not found
    """
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_manager = CacheManager()
    metadata = cache_manager.load_metadata_from_cache(
        filename=filename,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=models.ExtractionMetadata
    )

    if metadata:
        print(f"Loaded ExtractionMetadata: taxonomy_axis={metadata.taxonomy_primary_axis}")
        if metadata.taxonomy_axis_description:
            print(f"  Description: {metadata.taxonomy_axis_description}")
        if metadata.var_lab:
            print(f"  Survey question (var_lab): {metadata.var_lab}")
        else:
            print(f"  Survey question (var_lab): NOT SET")
    else:
        print("ExtractionMetadata not found in cache (optional)")

    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run ClustererV2 on the configured dataset."""
    print("=" * 70)
    print("Clustering Pipeline")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Algorithm mode: {CONFIG.algorithm_mode}")
    print()

    # Load embeddings
    embeddings, idea_texts, embeddings_models = load_step4_embeddings()

    # Load extraction metadata (optional - for taxonomy context in LLM labels)
    extraction_metadata = load_extraction_metadata()

    # Run clusterer
    clusterer = ClustererV2(embeddings_models, config=CONFIG, extraction_metadata=extraction_metadata)
    clusterer.run()

    # ==========================================================================
    # CACHE RESULTS (like pipeline.py step 5)
    # ==========================================================================
    # Convert to ClusterModel list (preserves all fields + adds initial_cluster)
    cluster_results = clusterer.to_cluster_model()

    # Generate variable key for caching (consistent with Step 4)
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    # Initialize cache manager and save
    cache_manager = CacheManager()
    var_lab = extraction_metadata.var_lab if extraction_metadata else None

    cache_manager.save_to_cache(
        cluster_results,              # List[ClusterModel]
        FILENAME,                     # Dataset filename
        "initial_clusters",           # Step name (matches pipeline.py)
        variable_key,                 # Cache key
        0,                            # elapsed_time
        var_lab=var_lab
    )

    # Layer 2: Clustering metadata (keywords, labels, distributions, metrics)
    metadata = clusterer.to_metadata_model()
    cache_manager.save_to_cache(
        [metadata],                   # ClusteringMetadataModel (wrapped in list for save_to_cache)
        FILENAME,                     # Dataset filename
        "clustering_metadata",        # New step name
        variable_key,                 # Cache key
        0,                            # elapsed_time
        var_lab=var_lab
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Algorithm recommendation
    rec = clusterer.get_algorithm_recommendation()
    if rec:
        print(f"\nAlgorithm Recommendation:")
        print(f"  Recommended: {rec.recommended_algorithm} ({rec.confidence} confidence)")
        print(f"  DVC: {rec.dvc_value:.3f} → {rec.dvc_recommendation}")
        print(f"  Knee: y_diff={rec.y_difference:.2f}, sharp={rec.has_sharp_knee}")
        if rec.is_forced:
            print(f"  FORCED: Algorithm selection was forced by hard DVC rule")
        print(f"  Reasoning: {rec.reasoning}")

    # Metrics
    metrics = clusterer.get_metrics()
    if metrics:
        print(f"\nClustering Metrics:")
        print(f"  Clusters: {metrics.n_clusters}")
        print(f"  Noise: {metrics.noise_count} ({metrics.noise_rate:.1%})")
        print(f"  Coherence: {metrics.mean_coherence:.3f} ({metrics.coherence_breakdown})")
        if metrics.dbcv is not None:
            print(f"  DBCV: {metrics.dbcv:.3f}")
        if metrics.silhouette is not None and not np.isnan(metrics.silhouette):
            print(f"  Silhouette: {metrics.silhouette:.3f}")
        if metrics.mean_persistence is not None:
            print(f"  Persistence: mean={metrics.mean_persistence:.3f}, weighted={metrics.weighted_persistence:.3f}")
        if metrics.mean_probability is not None:
            print(f"  Probability: mean={metrics.mean_probability:.3f}, low_ratio={metrics.low_prob_ratio:.1%}")
        if metrics.mean_outlier_score is not None:
            print(f"  Outliers: mean_score={metrics.mean_outlier_score:.3f}, high_ratio={metrics.high_outlier_ratio:.1%}")
        print(f"  Cluster sizes: min={metrics.min_cluster_size}, median={metrics.median_cluster_size}, max={metrics.max_cluster_size}")

    # Template prefix (used for text extraction in c-TF-IDF and display)
    template_prefix = clusterer._template_prefix
    if template_prefix:
        prefix_display = template_prefix[:60] + "..." if len(template_prefix) > 60 else template_prefix
        print(f"\nTemplate prefix: '{prefix_display}'")
    else:
        print(f"\nTemplate prefix: (none)")

    # Keywords (MMR and TF-IDF only - c-TF-IDF runs internally but not displayed in summary)
    all_keywords = clusterer.get_all_cluster_keywords()
    if all_keywords:
        for method_name in ["mmr", "tfidf"]:  # Skip ctfidf in summary (still runs, used by MMR)
            method_keywords = all_keywords.get(method_name)
            if method_keywords:
                method_label = {"mmr": "MMR", "tfidf": "TF-IDF"}.get(method_name, method_name)
                print(f"\n{method_label} Keywords ({len(method_keywords)} clusters):")
                for cluster_id in sorted(method_keywords.keys()):
                    kw_list = method_keywords[cluster_id]
                    kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
                    print(f"  Cluster {cluster_id}: {kw_str}")

    # Print ALL clusters with samples
    clusterer.print_all_clusters(n_samples=10)

    # Cache confirmation (at end for visibility)
    print(f"\n{'='*70}")
    print(f"CACHED: {len(cluster_results)} results to 'initial_clusters' (variable_key: {variable_key})")
    print(f"CACHED: {len(metadata.clusters)} clusters to 'clustering_metadata'")

    # Return clusterer for further analysis
    return clusterer, embeddings_models


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


def save_results_to_file(output: str, filename: str, variable: str, sample_size: Optional[int]) -> Path:
    """
    Save clustering results to a text file.

    Args:
        output: The captured console output
        filename: Original data filename
        variable: Variable name
        sample_size: Sample size (or None)

    Returns:
        Path to the saved file
    """
    # Create output directory
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: cluster_results_filename_variable_samplesize_YYYYMMDD.txt
    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d")

    output_filename = f"cluster_results_{base_name}_{variable}_{sample_str}_{date_str}.txt"
    output_path = output_dir / output_filename

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        clusterer, embeddings_models = main()
    finally:
        # Restore stdout
        sys.stdout = tee.original_stdout

    # Save results to file
    output_path = save_results_to_file(
        output=tee.get_output(),
        filename=FILENAME,
        variable=VARIABLE,
        sample_size=SAMPLE_SIZE
    )
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")

# %%
