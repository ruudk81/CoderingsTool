"""
Clusterer-specific configuration - EXPERIMENTAL VERSION

This is an isolated copy for experimentation in step_5_clusterer.
Changes here do NOT affect the production pipeline.

Original: src/config_clusterer.py

These settings control:
- Algorithm selection (auto, HDBSCAN, Agglomerative, K-means)
- UMAP dimensionality reduction
- HDBSCAN/Optuna optimization
- Post-processing (merging, noise reduction)
- Keyword extraction (c-TF-IDF, MMR, TF-IDF)
- LLM cluster label generation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ClustererConfig:
    """
    Configuration for Clusterer with automatic algorithm selection.

    Design decisions:
    - Diagnostics always computed (even in manual mode) for debugging
    - Persistence metrics included in output
    - Visualization optional via generate_plots flag

    Default settings match the approved experiment configuration:
    - algorithm_mode="auto" for automatic selection via DVC + knee detection
    - generate_ctfidf=True for keyword extraction
    - generate_llm_labels=True for LLM-generated cluster labels
    """

    # ==========================================================================
    # ALGORITHM SELECTION
    # ==========================================================================

    # Mode: "auto", "hdbscan", "agglomerative", "kmeans"
    algorithm_mode: str = "auto"

    # Small dataset threshold: n <= this -> always Agglomerative, no UMAP
    # Set to 0 to disable and always use DVC analysis
    small_dataset_threshold: int = 0

    # DVC (Density Variation Coefficient) thresholds
    # DVC = std(d_k) / mean(d_k) where d_k = distance to k-th nearest neighbor
    dvc_high_threshold: float = 0.45   # Above this -> HDBSCAN recommended
    dvc_low_threshold: float = 0.25    # Below this -> Agglomerative recommended
    dvc_knn_k: int = 10                # k for DVC computation

    # Hard DVC rule: force Agglomerative when DVC < this threshold
    # Set enable_agglomerative_fallback=False to always use HDBSCAN in auto mode
    enable_agglomerative_fallback: bool = False
    force_agglomerative_below_dvc: float = 0.25

    # kNN Knee detection parameters
    knee_y_diff_threshold: float = 0.6  # Minimum y_difference for sharp knee
    knee_knn_k: int = 5                 # k for knee detection

    # ==========================================================================
    # UMAP CONFIGURATION
    # ==========================================================================

    # UMAP parameters - optimized for euclidean HDBSCAN
    umap_n_components_grid: Tuple[int, ...] = (5, 10)
    umap_min_dist_grid: Tuple[float, ...] = (0.0, 0.1)
    umap_min_dist: float = 0.0  # Default for non-grid-search paths
    umap_metric: str = "euclidean"
    umap_random_state: int = 42

    # UMAP precomputation for Optuna grid search
    precompute_umap: bool = True

    # n_neighbors: 0.5*sqrt(n) to 1.5*sqrt(n), log-spaced
    n_neighbors_grid_k: int = 3
    n_neighbors_low_mult: float = 0.5
    n_neighbors_high_mult: float = 1.5
    n_neighbors_min: int = 5
    n_neighbors_max: int = 50

    # ==========================================================================
    # HDBSCAN / OPTUNA OPTIMIZATION
    # ==========================================================================

    # Enable Optuna-based grid search
    use_optuna: bool = True

    # MCS (min_cluster_size) grid
    min_cluster_size_grid_k: int = 3
    mcs_low_mult: float = 0.1
    mcs_high_mult: float = 0.5
    mcs_min: int = 3

    # MS (min_samples) strategy
    min_samples_strategy: str = "half_mcs"

    # Constraints for Optuna pruning
    max_noise_rate: float = 0.20
    min_clusters: int = 3

    # Quality thresholds for conditional re-search
    enable_research: bool = True
    research_max_noise_rate: float = 0.10
    research_min_validity: float = 0.70
    research_cluster_deviation_threshold: float = 0.15

    # Extended search grid configuration
    research_mcs_multipliers: Tuple[float, ...] = (0.5, 1.0, 1.5)
    research_ms_range_multipliers: Tuple[float, float] = (0.5, 2.0)
    research_ms_grid_k: int = 4
    research_selection_methods: Tuple[str, ...] = ('eom', 'leaf')

    # ==========================================================================
    # AGGLOMERATIVE / KMEANS PARAMETERS
    # ==========================================================================

    k_selection_strategy: str = "sqrt"
    k_grid_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)
    agglomerative_linkage: str = "ward"

    # ==========================================================================
    # COHERENCE THRESHOLDS
    # ==========================================================================

    coherence_high: float = 0.95
    coherence_moderate: float = 0.90
    coherence_acceptable: float = 0.70

    # ==========================================================================
    # PROBABILITY & OUTLIER THRESHOLDS (HDBSCAN metrics)
    # ==========================================================================

    low_probability_threshold: float = 0.5
    high_outlier_threshold: float = 0.7

    # ==========================================================================
    # COMPOSITE SCORING
    # ==========================================================================

    weight_validity: float = 0.5
    tau_low_prob: float = 0.15
    lambda_low_prob: float = 1.0
    fuzzy_cluster_threshold: float = 0.30
    lambda_fuzzy: float = 0.5
    lambda_fuzzy_count: float = 0.3

    # ==========================================================================
    # POST-PROCESSING
    # ==========================================================================

    # Cluster merging
    enable_merging: bool = True
    merge_centroid_threshold: float = 0.95
    merge_pairwise_threshold: float = 0.98

    # Noise reduction strategy
    noise_reduction_strategy: str = "embeddings"
    noise_reduction_threshold: float = 0.5

    # Legacy noise reclustering settings
    enable_noise_reclustering: bool = True
    noise_parameter_strategy: str = "adaptive"
    noise_min_cluster_size: int = 3
    noise_cohesion_threshold: float = 0.70

    # ==========================================================================
    # REPRESENTATION (c-TF-IDF) - Enabled by default
    # ==========================================================================

    generate_ctfidf: bool = True
    ctfidf_top_k: int = 10
    ctfidf_ngram_range: Tuple[int, int] = (1, 2)
    ctfidf_min_df: int = 1
    ctfidf_bm25_weighting: bool = True
    ctfidf_reduce_frequent_words: bool = True

    # Lemmatization with spaCy
    ctfidf_use_lemmatization: bool = True
    ctfidf_spacy_model: str = "nl_core_news_lg"
    ctfidf_pos_pattern: str = "ADJ*_NOUN+"

    # ==========================================================================
    # ADDITIONAL REPRESENTATIONS - Enabled by default
    # ==========================================================================

    generate_mmr_keywords: bool = True
    mmr_diversity: float = 0.3
    mmr_candidate_multiplier: int = 3

    generate_tfidf_keywords: bool = True

    # ==========================================================================
    # LLM CLUSTER LABELS - Enabled by default
    # ==========================================================================

    generate_llm_labels: bool = True
    llm_labels_model: str = "gpt-4.1"
    llm_max_ideas_per_cluster: int = 10
    representative_selection_method: str = "dense_region"
    representative_min_probability: float = 0.8  # Only use ideas with cluster probability > this threshold

    # ==========================================================================
    # VISUALIZATION (OPTIONAL)
    # ==========================================================================

    generate_plots: bool = False
    plots_output_dir: Optional[Path] = None

    # ==========================================================================
    # PERFORMANCE
    # ==========================================================================

    pca_threshold: int = 10_000
    pca_variance_retained: float = 0.99
    n_jobs: int = -1

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

DEFAULT_CLUSTERER_CONFIG = ClustererConfig()


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Configuration for HDBSCAN-only mode (skip algorithm selection)
HDBSCAN_ONLY_CONFIG = ClustererConfig(
    algorithm_mode="hdbscan",
)

# Configuration for Agglomerative-only mode
AGGLOMERATIVE_ONLY_CONFIG = ClustererConfig(
    algorithm_mode="agglomerative",
)

# Configuration for fast mode (no LLM labels, no keywords)
FAST_CLUSTERING_CONFIG = ClustererConfig(
    generate_ctfidf=False,
    generate_mmr_keywords=False,
    generate_tfidf_keywords=False,
    generate_llm_labels=False,
)

# Configuration with small dataset optimization enabled
SMALL_DATASET_CONFIG = ClustererConfig(
    small_dataset_threshold=250,
)
