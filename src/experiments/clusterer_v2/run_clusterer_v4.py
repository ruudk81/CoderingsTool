#%%

"""
ClustererV4 Run Script (PaCMAP)

Run the ClustererV4 pipeline on a specific dataset from cached Step 4 embeddings.
Uses PaCMAP instead of UMAP for dimensionality reduction.

Usage:
    cd src/experiments/clusterer_v2
    python run_clusterer_v4.py

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
from utils.cacheManager import generate_enhanced_variable_key

from clusterer_v2.clusterer_v4 import ClustererV4
from clusterer_v2.config_v4 import ClustererV4Config


# =============================================================================
# DATASET CONFIGURATION - Edit these to match your cached Step 4 data
# =============================================================================

FILENAME = "M000000 Associatiemonitor Merk X net databestand.sav"
VARIABLE = "Qd1_combined"
SAMPLE_SIZE = 2000

# =============================================================================
# CLUSTERER CONFIGURATION (PaCMAP)
# =============================================================================

CONFIG = ClustererV4Config(
    # Algorithm selection: "auto", "hdbscan", "agglomerative", "kmeans"
    algorithm_mode="auto",

    # DVC thresholds for algorithm selection
    dvc_high_threshold=0.45,    # Above this -> HDBSCAN
    dvc_low_threshold=0.25,     # Below this -> Agglomerative

    # Hard rule: force Agglomerative when DVC < this
    force_agglomerative_below_dvc=0.25,

    # Knee detection
    knee_y_diff_threshold=0.6,  # Sharp knee threshold

    # ==========================================================================
    # PaCMAP CONFIGURATION (replaces UMAP)
    # ==========================================================================
    pacmap_n_neighbors_grid=(5, 10, 15),
    pacmap_mn_ratio_grid=(0.3, 0.5, 0.7),
    pacmap_fp_ratio_grid=(1.0, 2.0, 3.0),
    pacmap_n_components_grid=(10,),  # BERTopic suggests 5, but 10 often better

    # PaCMAP settings
    pacmap_random_state=42,
    pacmap_apply_pca=True,

    # Optuna optimization (for HDBSCAN)
    use_optuna=True,
    min_cluster_size_grid=(2, 5, 10, 15),  # Fixed MCS values
    max_noise_rate=0.20,        # Maximum acceptable noise rate
    min_clusters=3,             # Minimum number of clusters

    # Quality thresholds for conditional re-search
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
    variable_key: Optional[str] = None
) -> Tuple[np.ndarray, List[str], List[models.EmbeddingsModel]]:
    """
    Load Step 4 embeddings from cache.

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        idea_texts: list of idea text strings
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

    # Build embeddings array and idea texts list
    embeddings_list = []
    idea_texts = []

    for response in embeddings_models:
        if response.response_ideas:
            for idea in response.response_ideas:
                if idea.idea_embedding is not None:
                    embeddings_list.append(idea.idea_embedding)
                    idea_texts.append(idea.idea)

    if not embeddings_list:
        raise ValueError("No embeddings found in cached data")

    embeddings = np.vstack(embeddings_list)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    return embeddings, idea_texts, embeddings_models


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run ClustererV4 (PaCMAP) on the configured dataset."""
    print("=" * 70)
    print("ClustererV4 Pipeline (PaCMAP)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Algorithm mode: {CONFIG.algorithm_mode}")
    print(f"DR method: PaCMAP")
    print()

    # Load embeddings
    embeddings, idea_texts, embeddings_models = load_step4_embeddings()

    # Run clusterer
    clusterer = ClustererV4(embeddings_models, config=CONFIG)
    clusterer.run()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Algorithm recommendation
    rec = clusterer.get_algorithm_recommendation()
    if rec:
        print(f"\nAlgorithm Recommendation:")
        print(f"  Recommended: {rec.recommended_algorithm} ({rec.confidence} confidence)")
        print(f"  DVC: {rec.dvc_value:.3f} -> {rec.dvc_recommendation}")
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

    # Template prefix
    template_prefix = clusterer._template_prefix
    if template_prefix:
        prefix_display = template_prefix[:60] + "..." if len(template_prefix) > 60 else template_prefix
        print(f"\nTemplate prefix: '{prefix_display}'")
    else:
        print(f"\nTemplate prefix: (none)")

    # Keywords
    keywords = clusterer.get_cluster_keywords()
    if keywords:
        print(f"\nc-TF-IDF Keywords ({len(keywords)} clusters):")
        for cluster_id in sorted(keywords.keys()):
            kw_list = keywords[cluster_id]
            kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
            print(f"  Cluster {cluster_id}: {kw_str}")

    # Print ALL clusters with samples
    clusterer.print_all_clusters(n_samples=10)

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

    # Build filename: cluster_results_v4_filename_variable_samplesize_YYYYMMDD.txt
    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_filename = f"cluster_results_v4_{base_name}_{variable}_{sample_str}_{date_str}.txt"
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
