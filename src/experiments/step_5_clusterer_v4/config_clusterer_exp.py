"""
Clusterer-specific configuration - V4 EXPERIMENTAL VERSION

V4: Object-Aware Map-Reduce MECE
  Stage 1: Object Discovery (cluster categories → MECE objects)
  Stage 2: Map Objects to Ideas
  Stage 3: Object-Aware Map-Reduce MECE (per-object topic extraction)

Based on V3 config with added object discovery settings.

This is an isolated copy for experimentation in step_5_clusterer_v4.
Changes here do NOT affect the production pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


# =============================================================================
# EMBEDDING SOURCE CONFIGURATION
# =============================================================================
# Which embedding to use for clustering:
# Cached (from step 4):
# - "idea_embedding": Cluster on idea text embeddings
# - "node_embedding": Cluster on node (canonical concept) embeddings
# - "category_embedding": Cluster on semantic_category embeddings
# - "taxonomy_embedding": Cluster on taxonomy chain embeddings (node → category_label → semantic_category → root)
# - "auto": Auto-detect from cached embedding_text_format
# On-the-fly (embeds the text field via API if no cached embedding exists):
# - "category": Embed idea.semantic_category text (coarser clustering)
# - "root": Embed idea.root text (coarsest)
# - "instance": Embed idea.instance text
EMBEDDING_SOURCE = "idea_embedding"

FORMAT_TO_EMBEDDING_FIELD = {
    "idea": "idea_embedding",
    "node": "idea_embedding",
    "category": "idea_embedding",
    "taxonomy": "idea_embedding",
    "all": "idea_embedding",
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
    Configuration for Clusterer V4 — Object-Aware Map-Reduce MECE.

    V4 pipeline:
    - Phases 1-5: Clustering infrastructure (from V3)
    - Stage 1: Object Discovery (cluster categories → MECE objects)
    - Stage 2: Map Objects to Ideas
    - Stage 3: Object-Aware Map-Reduce MECE (per-object topic extraction)
    """

    # ==========================================================================
    # EMBEDDING SOURCE
    # ==========================================================================

    embedding_source: str = EMBEDDING_SOURCE

    # ==========================================================================
    # ALGORITHM SELECTION
    # ==========================================================================

    algorithm_mode: str = "auto"
    small_dataset_threshold: int = 0

    # DVC thresholds
    dvc_high_threshold: float = 0.45
    dvc_low_threshold: float = 0.25
    dvc_knn_k: int = 10

    # Hard DVC rule
    enable_agglomerative_fallback: bool = False
    force_agglomerative_below_dvc: float = 0.25

    # Force HDBSCAN bypass
    force_hdbscan: bool = True

    # kNN Knee detection
    knee_y_diff_threshold: float = 0.6
    knee_knn_k: int = 5
    knee_s_denominator: int = 100
    knee_interp_threshold: int = 200
    trial_umap_nn_index: int = -1

    # ==========================================================================
    # UMAP CONFIGURATION
    # ==========================================================================

    umap_n_components_grid: Tuple[int, ...] = (5, 10, 15)
    umap_min_dist_grid: Tuple[float, ...] = (0.0, 0.1)
    umap_min_dist: float = 0.0
    umap_metric: str = "euclidean"
    umap_random_state: int = 42
    precompute_umap: bool = True

    n_neighbors_grid_k: int = 3
    n_neighbors_low_mult: float = 0.5
    n_neighbors_high_mult: float = 1.5
    n_neighbors_min: int = 5
    n_neighbors_max: int = 50

    # ==========================================================================
    # HDBSCAN / OPTUNA OPTIMIZATION
    # ==========================================================================

    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_gen_min_span_tree: bool = True
    use_optuna: bool = True

    # MCS grid
    min_cluster_size_grid_k: int = 4
    mcs_low_pct: float = 0.05
    mcs_low_log_mult: float = 1.0
    mcs_high_mult: float = 1.0
    mcs_min: int = 5

    # MS grid
    min_samples_grid_k: int = 4
    ms_low_log_mult: float = 1.0
    ms_high_sqrt_mult: float = 0.5

    # Constraints
    max_noise_rate: float = 0.40
    min_clusters: int = 2

    # Re-search
    enable_research: bool = True
    research_max_noise_rate: float = 0.10
    research_min_validity: float = 0.70
    research_cluster_deviation_threshold: float = 0.15
    research_mcs_multipliers: Tuple[float, ...] = (0.5, 1.0, 1.5)
    research_selection_methods: Tuple[str, ...] = ('eom', 'leaf')

    # ==========================================================================
    # AGGLOMERATIVE / KMEANS PARAMETERS
    # ==========================================================================

    k_selection_strategy: str = "sqrt"
    k_grid_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)
    agglomerative_linkage: str = "ward"
    agglomerative_nn_index: int = 1
    kmeans_nn_index: int = 1
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
    # PARSIMONY SELECTION (legacy)
    # ==========================================================================

    enable_parsimony_selection: bool = False
    parsimony_method: str = "coherence_knee"
    coherence_knee_window_divisor: int = 5
    coherence_knee_polynomial_degree: int = 3
    parsimony_min_score: float = 0.0

    # ==========================================================================
    # PARETO FRONTIER SELECTION
    # ==========================================================================

    pareto_min_dbcv: float = 0.30
    pareto_min_k_sqrt_mult: float = 0.5
    pareto_k_small_n_threshold: int = 3000
    pareto_max_k_sqrt_mult: float = 0.8
    pareto_max_noise_rate: float = 0.15
    pareto_max_cluster_ratio: float = 0.40

    pareto_weight_dbcv: float = 1.0
    pareto_weight_k: float = 1.0
    pareto_weight_low_prob_ratio: float = 1.0
    pareto_weight_max_cluster_ratio: float = 1.0

    pareto_norm_percentile_low: float = 5.0
    pareto_norm_percentile_high: float = 95.0
    enable_pareto_visualization: bool = True

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
    noise_reclustering_min_total: int = 10
    noise_reclustering_cluster_selection_method: str = "leaf"

    # ==========================================================================
    # STAGE 1: OBJECT DISCOVERY
    # ==========================================================================

    # Discovery mode:
    #   "clustering"         - Default V4: cluster categories → MECE objects (Stages 1+2)
    #   "semantic_category"  - Partition ideas by semantic_category field (skip Stages 1+2)
    object_discovery_mode: str = "clustering"

    # Ontology level to cluster for object discovery (only used when mode="clustering")
    object_discovery_level: str = "category"

    # ThemeGenerator config (used by Stage 1 per-cluster theme generation)
    generate_llm_labels: bool = True
    llm_labels_model: str = "gpt-4.1"
    llm_max_ideas_per_cluster: int = 10

    # LLM model for per-cluster object theme generation
    object_theme_model: str = "gpt-4.1"
    object_theme_temperature: float = 0.3
    object_theme_max_tokens: int = 1000

    # MECE object consolidation
    object_mece_model: str = "gpt-4.1"
    object_mece_temperature: float = 0.3
    object_mece_max_tokens: int = 4000

    # ==========================================================================
    # STAGE 3: OBJECT-AWARE MAP-REDUCE MECE (per-object topic extraction)
    # ==========================================================================

    generate_mece_topics: bool = True

    # LLM model for all 3 steps (map, reduce, mece)
    mapreduce_model: str = "gpt-4.1"
    mapreduce_temperature: float = 0.3
    mapreduce_max_tokens_map: int = 4000
    mapreduce_max_tokens_reduce: int = 4000
    mapreduce_max_tokens_mece: int = 4000

    # Batching: max ideas per map batch
    mapreduce_batch_size: int = 20

    # Concurrency for parallel map batches within a cluster
    mapreduce_concurrency: int = 5
    mapreduce_rpm_limit: int = 30

    # Text source for Map-Reduce MECE topic identification
    # - "idea": idea.idea text (default, current behavior)
    # - "ontology": full chain "instance → node → category → root"
    # - "node": canonical concept name (idea.node)
    # - "category": parent grouping (idea.semantic_category)
    mapreduce_text_source: str = "idea"

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

HDBSCAN_ONLY_CONFIG = ClustererConfig(
    algorithm_mode="hdbscan",
)

AGGLOMERATIVE_ONLY_CONFIG = ClustererConfig(
    algorithm_mode="agglomerative",
)

# Fast mode: no MECE topic extraction
FAST_CLUSTERING_CONFIG = ClustererConfig(
    generate_mece_topics=False,
)

SMALL_DATASET_CONFIG = ClustererConfig(
    small_dataset_threshold=250,
)
