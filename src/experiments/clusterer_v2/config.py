"""
Clusterer Configuration Module

Defines all configuration parameters for the unified clustering utility.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ClustererV2Config:
    """
    Configuration for ClustererV2 with automatic algorithm selection.

    Design decisions:
    - Diagnostics always computed (even in manual mode) for debugging
    - Persistence metrics included in output
    - Visualization optional via generate_plots flag
    """

    # ==========================================================================
    # ALGORITHM SELECTION
    # ==========================================================================

    # Mode: "auto", "hdbscan", "agglomerative", "kmeans"
    algorithm_mode: str = "auto"

    # Small dataset threshold: n <= this → always Agglomerative, no UMAP
    small_dataset_threshold: int = 250

    # DVC (Density Variation Coefficient) thresholds
    # DVC = std(d_k) / mean(d_k) where d_k = distance to k-th nearest neighbor
    dvc_high_threshold: float = 0.45   # Above this → HDBSCAN recommended
    dvc_low_threshold: float = 0.25    # Below this → Agglomerative recommended
    dvc_knn_k: int = 10                # k for DVC computation

    # Hard DVC rule: force Agglomerative when DVC < this threshold
    # Set enable_agglomerative_fallback=False to always use HDBSCAN in auto mode
    enable_agglomerative_fallback: bool = False  # Disable auto-switch to Agglomerative
    force_agglomerative_below_dvc: float = 0.25

    # kNN Knee detection parameters
    knee_y_diff_threshold: float = 0.6  # Minimum y_difference for sharp knee
    knee_knn_k: int = 5                 # k for knee detection

    # ==========================================================================
    # UMAP CONFIGURATION
    # ==========================================================================

    # UMAP parameters - optimized for euclidean HDBSCAN
    umap_n_components_grid: Tuple[int, ...] = (5, 10)  # 5-10 dims for euclidean metric
    umap_min_dist_grid: Tuple[float, ...] = (0.0, 0.1)  # Grid for Optuna search
    umap_min_dist: float = 0.0  # Default for non-grid-search paths (0.0 packs points tighter)
    umap_metric: str = "euclidean"
    umap_random_state: int = 42

    # UMAP precomputation for Optuna grid search
    precompute_umap: bool = True

    # n_neighbors: 0.5*sqrt(n) to 1.5*sqrt(n), log-spaced
    # Controls local vs global structure preservation
    n_neighbors_grid_k: int = 3  # Number of log-spaced grid points
    n_neighbors_low_mult: float = 0.5   # Low bound = 0.5 * sqrt(n)
    n_neighbors_high_mult: float = 1.5  # High bound = 1.5 * sqrt(n)
    n_neighbors_min: int = 5   # Absolute minimum n_neighbors
    n_neighbors_max: int = 50  # Absolute maximum n_neighbors

    # ==========================================================================
    # HDBSCAN / OPTUNA OPTIMIZATION
    # ==========================================================================

    # Enable Optuna-based grid search
    use_optuna: bool = True

    # MCS (min_cluster_size) grid: max(3, 0.1×√n) to 0.5×√n, log-spaced
    # Controls granularity of clustering (smaller MCS = more/smaller clusters)
    min_cluster_size_grid_k: int = 3  # Number of log-spaced grid points
    mcs_low_mult: float = 0.1    # Low bound = max(mcs_min, 0.1 * sqrt(n))
    mcs_high_mult: float = 0.5   # High bound = 0.5 * sqrt(n)
    mcs_min: int = 3             # Absolute minimum MCS

    # MS (min_samples) strategy: ms = 0.5 * mcs
    min_samples_strategy: str = "half_mcs"  # "half_mcs" = mcs // 2

    # Constraints for Optuna pruning
    max_noise_rate: float = 0.20  # Maximum acceptable noise rate
    min_clusters: int = 3         # Minimum number of clusters required

    # Quality thresholds for conditional re-search
    # Trigger: (noise > max AND validity < min) OR (cluster_deviation > threshold)
    enable_research: bool = True
    research_max_noise_rate: float = 0.10           # Noise threshold for condition 1
    research_min_validity: float = 0.70             # Validity threshold for condition 1
    research_cluster_deviation_threshold: float = 0.15  # |k - sqrt(n)| / sqrt(n) > this triggers

    # Extended search grid configuration (triggered when quality check fails)
    # MCS: multipliers around best MCS from initial search
    research_mcs_multipliers: Tuple[float, ...] = (0.5, 1.0, 1.5)
    # MS: log-scale grid between (best_ms * low_mult) and (best_ms * high_mult)
    research_ms_range_multipliers: Tuple[float, float] = (0.5, 2.0)  # (low, high)
    research_ms_grid_k: int = 4  # Number of log-spaced points for MS
    # Cluster selection methods to try
    research_selection_methods: Tuple[str, ...] = ('eom', 'leaf')

    # ==========================================================================
    # AGGLOMERATIVE / KMEANS PARAMETERS
    # ==========================================================================

    # K selection strategy
    k_selection_strategy: str = "sqrt"  # "sqrt" = grid based on sqrt(n)
    k_grid_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)  # Multipliers for sqrt(n)

    # Agglomerative linkage
    agglomerative_linkage: str = "ward"

    # ==========================================================================
    # COHERENCE THRESHOLDS
    # ==========================================================================

    coherence_high: float = 0.95       # >= this = high coherence
    coherence_moderate: float = 0.90   # >= this = moderate coherence
    coherence_acceptable: float = 0.70 # >= this = acceptable coherence
                                        # < acceptable = unacceptable

    # ==========================================================================
    # PROBABILITY & OUTLIER THRESHOLDS (HDBSCAN metrics)
    # ==========================================================================

    # Probability thresholds (from HDBSCAN probabilities_)
    low_probability_threshold: float = 0.5  # Below this = borderline cluster member

    # Outlier thresholds (from HDBSCAN outlier_scores_ / GLOSH)
    high_outlier_threshold: float = 0.7  # Above this = potential misassignment

    # ==========================================================================
    # COMPOSITE SCORING (No Persistence)
    # ==========================================================================
    # score = w_validity * relative_validity
    #         - λ_low_prob * max(0, low_prob_ratio - τ)
    #         - λ_fuzzy * fuzzy_cluster_ratio
    #         - λ_fuzzy_count * fuzzy_cluster_fraction
    #
    # Where:
    # - low_prob_ratio: global fraction of points with prob < threshold
    # - fuzzy_cluster_ratio: fraction of points in fuzzy clusters
    # - fuzzy_cluster_fraction: n_fuzzy_clusters / n_clusters

    # Weights for composite score
    weight_validity: float = 0.5       # Weight for relative_validity_

    # Soft threshold for low_prob_ratio (global)
    tau_low_prob: float = 0.15         # Acceptable borderline ratio (no penalty below)
    lambda_low_prob: float = 1.0       # Penalty strength above tau

    # Fuzzy cluster penalty (per-cluster based)
    # A cluster is "fuzzy" if its per-cluster low_ratio > fuzzy_cluster_threshold
    fuzzy_cluster_threshold: float = 0.30  # Per-cluster low_ratio threshold for "fuzzy"
    lambda_fuzzy: float = 0.5              # Penalty for fuzzy_cluster_ratio (points)
    lambda_fuzzy_count: float = 0.3        # Penalty for fuzzy_cluster_fraction (clusters)

    # ==========================================================================
    # POST-PROCESSING
    # ==========================================================================

    # Cluster merging
    enable_merging: bool = True
    merge_centroid_threshold: float = 0.95   # Centroid similarity screening
    merge_pairwise_threshold: float = 0.98   # Final merge decision

    # Noise reduction strategy
    # "embeddings" = BERTopic-style (assign noise to nearest cluster by embedding similarity)
    # "hdbscan" = Legacy (re-run HDBSCAN on noise points)
    noise_reduction_strategy: str = "embeddings"
    noise_reduction_threshold: float = 0.5  # Min similarity to assign noise to cluster

    # Legacy noise reclustering settings (used when noise_reduction_strategy="hdbscan")
    enable_noise_reclustering: bool = True
    noise_parameter_strategy: str = "adaptive"
    noise_min_cluster_size: int = 3
    noise_cohesion_threshold: float = 0.70

    # ==========================================================================
    # REPRESENTATION (c-TF-IDF)
    # ==========================================================================

    generate_ctfidf: bool = False      # Enable c-TF-IDF keyword extraction
    ctfidf_top_k: int = 15             # Top keywords per cluster
    ctfidf_ngram_range: Tuple[int, int] = (1, 2)  # Unigrams + bigrams
    ctfidf_min_df: int = 1  # Allow cluster-unique terms (most distinctive!)
    ctfidf_bm25_weighting: bool = True
    ctfidf_reduce_frequent_words: bool = True

    # Lemmatization with spaCy (for c-TF-IDF)
    ctfidf_use_lemmatization: bool = True
    ctfidf_spacy_model: str = "nl_core_news_lg"  # Dutch large model
    # POS pattern: ADJ* + (NOUN | PROPN)+ → extracts noun phrases with optional adjectives
    ctfidf_pos_pattern: str = "ADJ*_NOUN+"

    # ==========================================================================
    # ADDITIONAL REPRESENTATIONS (displayed alongside c-TF-IDF)
    # ==========================================================================

    # MMR (Maximal Marginal Relevance) - diversity-aware keyword selection
    # Balances relevance (c-TF-IDF) with diversity (less redundant keywords)
    generate_mmr_keywords: bool = False
    mmr_diversity: float = 0.3  # Lambda: 0.0 = max diversity, 1.0 = max relevance
    mmr_candidate_multiplier: int = 3  # Pool size = top_k * multiplier

    # Basic TF-IDF per cluster (not class-based like c-TF-IDF)
    # Computes TF-IDF independently for each cluster's texts
    generate_tfidf_keywords: bool = False

    # ==========================================================================
    # LLM CLUSTER LABELS
    # ==========================================================================

    generate_llm_labels: bool = False  # Enable LLM-generated cluster labels
    llm_labels_model: str = "gpt-4.1"  # Model for label generation
    llm_max_ideas_per_cluster: int = 10  # Max ideas to include in prompt

    # Representative sample selection method for LLM prompts
    # "dense_region" = select core members using HDBSCAN probabilities_ (default)
    # "centroid" = select by cosine similarity to cluster centroid (legacy)
    representative_selection_method: str = "dense_region"

    # ==========================================================================
    # VISUALIZATION (OPTIONAL)
    # ==========================================================================

    generate_plots: bool = False
    plots_output_dir: Optional[Path] = None

    # ==========================================================================
    # PERFORMANCE
    # ==========================================================================

    # PCA for large datasets
    pca_threshold: int = 10_000  # Apply PCA when n > threshold
    pca_variance_retained: float = 0.99

    # Parallelization
    n_jobs: int = -1  # -1 = all cores

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True
