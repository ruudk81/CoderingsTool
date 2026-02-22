"""
Clusterer-specific configuration - EXPERIMENTAL VERSION

This is an isolated copy for experimentation in step_5_clusterer.
Changes here do NOT affect the production pipeline.

Original: src/config_clusterer.py

These settings control:
- Algorithm selection (auto, HDBSCAN, Agglomerative)
- UMAP dimensionality reduction
- HDBSCAN grid search with Pareto selection
- Post-processing (merging, noise reduction)
- Keyword extraction (c-TF-IDF, MMR, TF-IDF)
- LLM cluster label generation
"""

from dataclasses import dataclass
from typing import Tuple


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

    # Which embedding field to cluster on:
    # "idea_embedding", "concept_embedding", "concept_type_embedding", "ladder_embedding"
    clustering_embedding_field: str = "ladder_embedding"

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

    # UMAP parameters
    umap_n_components_grid: Tuple[int, ...] = (5, 10)
    umap_min_dist_grid: Tuple[float, ...] = (0.0, 0.1)
    umap_min_dist: float = 0.0  # Default for non-grid-search paths
    umap_random_state: int = 42

    # UMAP precomputation for grid search
    precompute_umap: bool = True

    # n_neighbors: ceil(sqrt(N)/2) to max(ceil(sqrt(N)), 10), log-spaced
    n_neighbors_grid_k: int = 3
    n_neighbors_high_min: int = 10  # Floor for high bound: high = max(ceil(sqrt(N)), this)

    # ==========================================================================
    # HDBSCAN GRID SEARCH
    # ==========================================================================

    # min_samples: ceil(ln(N)) to ceil(2*ln(N)), log-spaced
    ms_grid_k: int = 3

    # min_cluster_size: 2 × ms bounds, log-spaced
    mcs_grid_k: int = 3
    mcs_ms_multiplier: float = 2.0  # mcs = this × ms

    # ==========================================================================
    # PARETO SELECTION (3-stage pipeline)
    # ==========================================================================

    # Stage 1: Hard constraint filtering (progressive fallback)
    pareto_min_dbcv: float = 0.30
    pareto_min_k_sqrt_mult: float = 0.5       # k >= 0.5 * sqrt(N)
    pareto_k_small_n_threshold: int = 3000     # N threshold for k range constraint
    pareto_max_k_sqrt_mult: float = 0.8        # k <= 0.8 * sqrt(N)
    pareto_max_noise_rate: float = 0.15
    pareto_max_cluster_ratio: float = 0.40     # no single cluster > 40%

    # Stage 3: Pareto objective weights (all equal by default)
    pareto_weight_dbcv: float = 1.0
    pareto_weight_k: float = 1.0
    pareto_weight_low_prob_ratio: float = 1.0
    pareto_weight_max_cluster_ratio: float = 1.0

    # Percentile normalization bounds (outlier-robust)
    pareto_norm_percentile_low: float = 5.0    # p5
    pareto_norm_percentile_high: float = 95.0  # p95

    # ==========================================================================
    # ITERATIVE RESIDUAL CLUSTERING
    # ==========================================================================

    enable_iterative: bool = True
    iterative_accept_probability: float = 0.7   # points need prob >= this to be accepted
    iterative_max_iterations: int = 10
    iterative_residual_ratio_stop: float = 0.10  # stop when residual ≤ 10% of group N
    iterative_min_residual_size: int = 10         # stop when residual < this

    # ==========================================================================
    # CONCEPT_TYPE GROUPING
    # ==========================================================================

    enable_concept_type_grouping: bool = False
    concept_type_min_group_size: int = 20         # smaller groups pooled into fallback
    concept_type_fallback: str = "_other"

    # ==========================================================================
    # AGGLOMERATIVE PARAMETERS
    # ==========================================================================

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
    # POST-PROCESSING
    # ==========================================================================

    # Cluster merging
    enable_merging: bool = True
    merge_centroid_threshold: float = 0.95
    merge_pairwise_threshold: float = 0.98

    # Noise reduction: assign noise points to nearest cluster by embedding similarity
    noise_reduction_threshold: float = 0.5

    # ==========================================================================
    # REPRESENTATION (c-TF-IDF) - Enabled by default
    # ==========================================================================

    generate_ctfidf: bool = True
    ctfidf_top_k: int = 10
    ctfidf_ngram_range: Tuple[int, int] = (1, 1)  # (1,1) since ADJ+NOUN compounds are pre-built
    ctfidf_min_df: int = 1
    ctfidf_bm25_weighting: bool = True
    ctfidf_reduce_frequent_words: bool = True

    # Lemmatization with spaCy
    ctfidf_use_lemmatization: bool = True
    ctfidf_spacy_model: str = "nl_core_news_lg"
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
    # PERFORMANCE
    # ==========================================================================

    pca_threshold: int = 10_000
    pca_variance_retained: float = 0.99
    n_jobs: int = -1

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True

    # Text sources — single field or composite with "+" (e.g., "idea+concept_type_definition")
    # Supported fields: "idea", "instance", "concept", "concept_type",
    #                   "concept_type_definition", "ladder"
    keyword_text_source: str = "idea"           # text for c-TF-IDF / MMR keyword extraction
    label_text_source: str = "idea"             # text for representative samples in LLM prompt
    verbose_text_source: str = "idea"           # text for print_all_clusters() display
    text_separator: str = " | "                 # separator for composite "+" fields


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

DEFAULT_CLUSTERER_CONFIG = ClustererConfig()
