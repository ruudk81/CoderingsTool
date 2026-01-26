"""
ClustererV4 Configuration Module (PaCMAP)

Replaces UMAP configuration with PaCMAP parameters.
Inherits most settings from ClustererV2Config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ClustererV4Config:
    """
    Configuration for ClustererV4 with PaCMAP dimensionality reduction.

    Key differences from ClustererV2Config:
    - Replaces UMAP parameters with PaCMAP parameters
    - Adds MN_ratio and FP_ratio grids for PaCMAP optimization
    """

    # ==========================================================================
    # ALGORITHM SELECTION
    # ==========================================================================

    algorithm_mode: str = "auto"

    # DVC thresholds
    dvc_high_threshold: float = 0.45
    dvc_low_threshold: float = 0.25
    dvc_knn_k: int = 10

    # Hard DVC rule
    enable_agglomerative_fallback: bool = False
    force_agglomerative_below_dvc: float = 0.25

    # kNN Knee detection
    knee_y_diff_threshold: float = 0.6
    knee_knn_k: int = 5

    # ==========================================================================
    # PACMAP CONFIGURATION (replaces UMAP)
    # ==========================================================================

    # PaCMAP parameter grids
    pacmap_n_neighbors_grid: Tuple[int, ...] = (10, 20, 30)
    pacmap_mn_ratio_grid: Tuple[float, ...] = (0.3, 0.5, 0.7)
    pacmap_fp_ratio_grid: Tuple[float, ...] = (1.0, 2.0, 3.0)
    pacmap_n_components_grid: Tuple[int, ...] = (10,)  # BERTopic suggests 5, but 10 often better

    # PaCMAP settings
    pacmap_random_state: int = 42
    pacmap_apply_pca: bool = True  # Apply PCA before neighbor search (helps with high-dim)

    # PaCMAP precomputation
    precompute_pacmap: bool = True

    # ==========================================================================
    # HDBSCAN / OPTUNA OPTIMIZATION
    # ==========================================================================

    use_optuna: bool = True
    min_cluster_size_grid: Optional[Tuple[int, ...]] = None  # Fixed values, overrides dynamic
    min_cluster_size_grid_k: int = 4  # Only used if min_cluster_size_grid is None
    min_samples_strategy: str = "half_mcs"

    # Constraints
    max_noise_rate: float = 0.20
    min_clusters: int = 3

    # Quality thresholds for re-search
    enable_research: bool = True
    research_max_noise_rate: float = 0.10
    research_min_validity: float = 0.70
    research_cluster_deviation_threshold: float = 0.15

    # Extended search config
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
    # PROBABILITY & OUTLIER THRESHOLDS
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

    enable_merging: bool = True
    merge_centroid_threshold: float = 0.95
    merge_pairwise_threshold: float = 0.98

    noise_reduction_strategy: str = "embeddings"
    noise_reduction_threshold: float = 0.5

    enable_noise_reclustering: bool = True
    noise_parameter_strategy: str = "adaptive"
    noise_min_cluster_size: int = 3
    noise_cohesion_threshold: float = 0.70

    # ==========================================================================
    # REPRESENTATION (c-TF-IDF)
    # ==========================================================================

    generate_ctfidf: bool = False
    ctfidf_top_k: int = 15
    ctfidf_ngram_range: Tuple[int, int] = (1, 2)
    ctfidf_min_df: int = 1
    ctfidf_bm25_weighting: bool = True
    ctfidf_reduce_frequent_words: bool = True

    ctfidf_use_lemmatization: bool = True
    ctfidf_spacy_model: str = "nl_core_news_lg"
    ctfidf_pos_pattern: str = "ADJ*_NOUN+"

    # ==========================================================================
    # LLM CLUSTER LABELS
    # ==========================================================================

    generate_llm_labels: bool = False
    llm_labels_model: str = "gpt-4.1"
    llm_max_ideas_per_cluster: int = 10

    # ==========================================================================
    # VISUALIZATION
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
