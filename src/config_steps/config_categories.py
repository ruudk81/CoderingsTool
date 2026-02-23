"""
Configuration for Category Discovery.

Partitions by concept_type (data-driven from step 3), then processes
with MAP/REDUCE/MECE.

Two processing modes:
  "direct"   (Mode A): MAP/REDUCE/MECE on labels directly
  "clustered" (Mode B): Pre-cluster labels, then MAP/REDUCE/MECE with cluster hints
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CategoriesConfig:
    """Configuration for Category Discovery."""

    # ==========================================================================
    # PROCESSING MODE
    # ==========================================================================

    # "direct" (Mode A): MAP/REDUCE/MECE on labels directly
    # "clustered" (Mode B): UMAP+HDBSCAN pre-cluster, then MAP/REDUCE/MECE with hints
    processing_mode: str = "direct"

    # ==========================================================================
    # PARTITION SOURCE
    # ==========================================================================

    PARTITION_SOURCE = "concept_type"

    # ==========================================================================
    # LABEL SOURCE
    # ==========================================================================

    # Which text to collect as "labels" for MAP/REDUCE/MECE input.
    #
    # Stored fields (direct attributes on EmbeddingsSubmodel):
    #   "concept_type_definition" — concept type framing
    #   "concept"                 — canonical noun phrase
    #   "concept_type"            — e.g., "recommendation"
    #   "idea"                    — full idea text incl. template prefix
    #   "instance"                — verbatim span from response
    #
    # Computed composites (assembled from stored fields by format_label()):
    #   "ladder"               — instance → concept → concept_type → concept_type_definition
    #   "idea_concept_defined" — idea → concept → concept_type_definition
    label_source: str = "idea_concept_defined"

    # Optional prefix prepended to each label string before processing.
    # "" = no prefix (default)
    # "{concept_type}: " = dynamic prefix from idea.concept_type field
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

DEFAULT_CATEGORIES_CONFIG = CategoriesConfig()

CLUSTERED_CONFIG = CategoriesConfig(
    processing_mode="clustered",
)


# =============================================================================
# CATEGORY ASSIGNMENT CONFIG
# =============================================================================

@dataclass
class AssignmentConfig:
    """Configuration for MECE category assignment to individual ideas."""

    # LLM settings
    assignment_model: str = "gpt-4.1-mini"
    assignment_temperature: float = 0.1    # Low for consistent assignment
    assignment_max_tokens: int = 4000

    # Batching: ideas per LLM call
    assignment_batch_size: int = 10

    # Fallback category for ideas that don't clearly fit any MECE category.
    # Resolved by language from extraction_metadata.lang.
    include_other_category: bool = True

    # Output
    verbose: bool = True


DEFAULT_ASSIGNMENT_CONFIG = AssignmentConfig()


# Language → "Other/Miscellaneous" label mapping
OTHER_CATEGORY_LABELS: Dict[str, str] = {
    "Dutch": "overig/anders",
    "nl-NL": "overig/anders",
    "English": "other/miscellaneous",
    "en-GB": "other/miscellaneous",
    "en-US": "other/miscellaneous",
    "German": "sonstiges",
    "de-DE": "sonstiges",
    "French": "autre/divers",
    "fr-FR": "autre/divers",
    "Spanish": "otro/varios",
    "es-ES": "otro/varios",
}
OTHER_CATEGORY_DEFAULT = "other/miscellaneous"


def get_other_category_label(language: str) -> str:
    """Resolve the Other category label for a given language."""
    return OTHER_CATEGORY_LABELS.get(language, OTHER_CATEGORY_DEFAULT)
