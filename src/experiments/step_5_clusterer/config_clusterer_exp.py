"""
Clusterer-specific configuration - EXPERIMENTAL VERSION

This is an isolated copy for experimentation in clusterer_v3.
Changes here do NOT affect the production pipeline.

Original: src/config_clusterer.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


# =============================================================================
# EMBEDDING SOURCE CONFIGURATION
# =============================================================================
# Which embedding to use for clustering:
# - "auto": Auto-detect from cached embedding_text_format (recommended)
# - "idea_embedding": Cluster on idea text embeddings
# - "taxonomy_embedding": Cluster on taxonomy_phrase embeddings
# - "ontology_embedding": Cluster on ontology embeddings
# "auto" reads embedding_text_format from cached data and resolves to the
# best available field. Override only when cached format is "all" and you
# want a specific field, or for explicit experimentation.
EMBEDDING_SOURCE = "auto"

# Mapping from cached embedding_text_format to the primary embedding field.
# Single-pass formats store everything in idea_embedding.
# Multi-pass formats populate additional fields per MULTI_PASS_SPECS.
FORMAT_TO_EMBEDDING_FIELD = {
    "idea": "idea_embedding",
    "idea_without_template_prefix": "idea_embedding",
    "taxonomy_phrase": "idea_embedding",           # single-pass → idea_embedding
    "ontology": "idea_embedding",                  # single-pass → idea_embedding
    "both_taxonomy_phrase": "taxonomy_embedding",   # multi-pass → use enriched field
    "both_ontology": "ontology_embedding",          # multi-pass → use enriched field
    "all": "taxonomy_embedding",                    # multi-pass → default to taxonomy
}


def resolve_embedding_source(embedding_text_format: str, config_source: str) -> str:
    """Resolve which embedding field to cluster on.

    If config_source is "auto", derive from cached embedding_text_format.
    Otherwise, return config_source as-is (explicit override).
    """
    if config_source != "auto":
        return config_source

    resolved = FORMAT_TO_EMBEDDING_FIELD.get(embedding_text_format)
    if resolved is None:
        raise ValueError(
            f"Unknown embedding_text_format '{embedding_text_format}'. "
            f"Known formats: {list(FORMAT_TO_EMBEDDING_FIELD.keys())}. "
            f"Set EMBEDDING_SOURCE explicitly in config_clusterer_exp.py."
        )
    return resolved


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
    # EMBEDDING SOURCE
    # ==========================================================================

    # Which embedding field to cluster on: "auto", "idea_embedding", "taxonomy_embedding", "ontology_embedding"
    # "auto" resolves from cached embedding_text_format at runtime.
    embedding_source: str = EMBEDDING_SOURCE

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

    # Temporary: bypass all algorithm selection and always use HDBSCAN
    force_hdbscan: bool = True

    # kNN Knee detection parameters
    knee_y_diff_threshold: float = 0.6  # Minimum y_difference for sharp knee
    knee_knn_k: int = 5                 # k for knee detection
    knee_s_denominator: int = 100       # KneeLocator S = max(1.0, n / this)
    knee_interp_threshold: int = 200    # n < this → polynomial, else interp1d

    # Index into n_neighbors grid for trial UMAP during algorithm selection
    # -1 = middle of grid (len//2)
    trial_umap_nn_index: int = -1

    # ==========================================================================
    # UMAP CONFIGURATION
    # ==========================================================================

    # UMAP parameters - optimized for euclidean HDBSCAN
    umap_n_components_grid: Tuple[int, ...] = (5, 10, 15)
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

    # HDBSCAN model parameters
    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_gen_min_span_tree: bool = True

    # Enable Optuna-based grid search
    use_optuna: bool = True

    # MCS (min_cluster_size) grid: lower=min(0.05*N, 2*ln(N)), upper=sqrt(N)
    min_cluster_size_grid_k: int = 4
    mcs_low_pct: float = 0.05       # MCS lower bound option 1: 5% of N
    mcs_low_log_mult: float = 1.0   # MCS lower bound option 2: 1 * ln(N); take min of both
    mcs_high_mult: float = 1.0      # MCS upper bound: 1.0 * sqrt(N)
    mcs_min: int = 5                # Absolute floor for MCS

    # MS (min_samples) grid: lower=ln(N), upper=sqrt(N)/2
    min_samples_grid_k: int = 4
    ms_low_log_mult: float = 1.0    # MS lower bound: 1.0 * ln(N)
    ms_high_sqrt_mult: float = 0.5  # MS upper bound: 0.5 * sqrt(N)

    # Constraints for Optuna pruning (relaxed — Pareto handles fine-grained selection)
    max_noise_rate: float = 0.40
    min_clusters: int = 2

    # Quality thresholds for conditional re-search (disabled — superseded by Pareto)
    enable_research: bool = True
    research_max_noise_rate: float = 0.10
    research_min_validity: float = 0.70
    research_cluster_deviation_threshold: float = 0.15

    # Extended search grid configuration
    research_mcs_multipliers: Tuple[float, ...] = (0.5, 1.0, 1.5)
    research_selection_methods: Tuple[str, ...] = ('eom', 'leaf')

    # ==========================================================================
    # AGGLOMERATIVE / KMEANS PARAMETERS
    # ==========================================================================

    k_selection_strategy: str = "sqrt"
    k_grid_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)
    agglomerative_linkage: str = "ward"

    # Index into n_neighbors grid for agglomerative/kmeans UMAP
    agglomerative_nn_index: int = 1
    kmeans_nn_index: int = 1

    # KMeans model parameters
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10

    # Bootstrap CI for small dataset agglomerative path
    agglomerative_bootstrap_iterations: int = 100
    agglomerative_bootstrap_confidence: float = 0.95

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
    # PARSIMONY SELECTION (legacy — superseded by Pareto frontier)
    # ==========================================================================

    enable_parsimony_selection: bool = False
    parsimony_method: str = "coherence_knee"
    coherence_knee_window_divisor: int = 5
    coherence_knee_polynomial_degree: int = 3
    parsimony_min_score: float = 0.0

    # ==========================================================================
    # PARETO FRONTIER SELECTION
    # ==========================================================================

    # Hard constraints for candidate filtering
    pareto_min_dbcv: float = 0.30           # DBCV > this to be a candidate
    # Min k: k >= 0.5 * sqrt(N)  (= N / (2 * sqrt(N)))
    pareto_min_k_sqrt_mult: float = 0.5     # k >= 0.5 * sqrt(N)
    # Max k: if N < pareto_k_small_n_threshold -> N / (2 * mcs_lower), else 0.8 * sqrt(N)
    pareto_k_small_n_threshold: int = 3000
    pareto_max_k_sqrt_mult: float = 0.8     # k <= 0.8 * sqrt(N) for large N
    pareto_max_noise_rate: float = 0.15     # Noise rate <= this to be a candidate
    pareto_max_cluster_ratio: float = 0.40  # No single cluster > 40% of items

    # Ideal-point selection weights (for picking from the Pareto front)
    # 4 Pareto objectives: DBCV (max), k (min), low_prob_ratio (min), max_cluster_ratio (min)
    # Coherence and noise handled as hard constraints, not Pareto objectives
    pareto_weight_dbcv: float = 1.0
    pareto_weight_k: float = 1.0
    pareto_weight_low_prob_ratio: float = 1.0
    pareto_weight_max_cluster_ratio: float = 1.0

    # Percentile normalization bounds (replace min-max for outlier robustness)
    pareto_norm_percentile_low: float = 5.0   # p5
    pareto_norm_percentile_high: float = 95.0  # p95

    # Enable Pareto visualization export (saves PNG to exports/)
    enable_pareto_visualization: bool = True

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
    noise_reclustering_min_total: int = 10
    noise_reclustering_cluster_selection_method: str = "leaf"

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
