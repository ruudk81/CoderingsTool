"""
Configuration for Category Discovery V5.

Simplified from V4: no object discovery, no Optuna/DVC/Pareto.
Always partitions by semantic_category, then processes with MAP/REDUCE/MECE.

Two processing modes:
  "direct"   (Mode A): MAP/REDUCE/MECE on labels directly
  "clustered" (Mode B): Pre-cluster labels, then MAP/REDUCE/MECE with cluster hints
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CategoriesConfig:
    """Configuration for Category Discovery V5."""

    # ==========================================================================
    # PROCESSING MODE
    # ==========================================================================

    # "direct" (Mode A): MAP/REDUCE/MECE on labels directly
    # "clustered" (Mode B): UMAP+HDBSCAN pre-cluster, then MAP/REDUCE/MECE with hints
    processing_mode: str = "direct"

    # ==========================================================================
    # LABEL SOURCE
    # ==========================================================================

    # Which text field to collect as "labels" for MAP/REDUCE/MECE input.
    # "category_label" (default): idea.category_label — subcategory-level name
    # "node": idea.node — canonical concept
    # "idea": idea.idea — full idea text
    # "instance": idea.instance — verbatim span
    label_source: str = "category_label"

    # Optional prefix prepended to each label string before processing.
    # "" = no prefix (default)
    # "{root}: " = dynamic prefix from idea.root field
    # Any literal string = static prefix for all labels
    label_prefix: str = ""

    # ==========================================================================
    # MAP-REDUCE MECE
    # ==========================================================================

    # LLM model for all 3 steps (map, reduce, mece)
    mapreduce_model: str = "gpt-4.1"
    mapreduce_temperature: float = 0.3
    mapreduce_max_tokens_map: int = 4000
    mapreduce_max_tokens_reduce: int = 4000
    mapreduce_max_tokens_mece: int = 4000

    # Batching: max labels per map batch
    mapreduce_batch_size: int = 40

    # Fallback concurrency (used only if dynamic bootstrap fails)
    mapreduce_concurrency: int = 5
    mapreduce_rpm_limit: int = 30

    # ==========================================================================
    # PRE-CLUSTERING (Mode B only)
    # ==========================================================================
    # Simple UMAP+HDBSCAN on label embeddings within each partition.
    # Produces cluster hints injected into MAP/REDUCE/MECE prompts as context.

    # UMAP dimensionality reduction
    precluster_umap_n_components: int = 5
    precluster_umap_n_neighbors: int = 15
    precluster_umap_min_dist: float = 0.0
    precluster_umap_metric: str = "euclidean"
    precluster_umap_random_state: int = 42

    # HDBSCAN clustering
    precluster_min_cluster_size: int = 3
    precluster_min_samples: int = 2
    precluster_cluster_selection_method: str = "eom"

    # Minimum unique labels in a partition to attempt pre-clustering.
    # Below this threshold, Mode B falls back to Mode A for that partition.
    precluster_min_labels: int = 8

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True


# =============================================================================
# PRESETS
# =============================================================================

DEFAULT_CONFIG = CategoriesConfig()

CLUSTERED_CONFIG = CategoriesConfig(
    processing_mode="clustered",
)
