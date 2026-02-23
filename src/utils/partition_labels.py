"""
Label extraction and formatting utilities for Category Discovery.

Handles:
- Extracting text from idea objects based on configurable label_source
- Computing composite text from stored fields (ladder, idea_concept_defined)
- Applying optional label_prefix (static or dynamic from idea fields)
- Formatting pre-cluster results as prompt-injectable hints
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


# Composite format names that are computed from stored fields, not direct attributes.
# These mirror step 4 embedder's text format keys.
COMPOSITE_FORMATS = frozenset({"ladder", "idea_concept_defined"})


@dataclass
class PreclusterResult:
    """Result of UMAP+HDBSCAN pre-clustering on labels within a partition."""
    labels_by_cluster: Dict[int, List[str]]  # cluster_id → labels in that cluster
    noise_labels: List[str]                   # labels not assigned to any cluster
    n_clusters: int
    n_noise: int


def _compute_ladder(idea) -> str:
    """Compute 'instance → concept → concept_type → concept_type_definition'.

    Mirrors step 4 embedder's ladder format. Falls back to idea.idea
    if all component fields are empty.
    """
    parts = []
    for field in ('instance', 'concept', 'concept_type', 'concept_type_definition'):
        val = (getattr(idea, field, '') or '').strip()
        if val:
            parts.append(val)
    return " → ".join(parts) if parts else (getattr(idea, 'idea', '') or '').strip()


def _compute_idea_concept_defined(idea) -> str:
    """Compute 'idea → concept → concept_type_definition'.

    Mirrors step 4 embedder's idea_concept_defined format.
    """
    parts = []
    for field in ('idea', 'concept', 'concept_type_definition'):
        val = (getattr(idea, field, '') or '').strip()
        if val:
            parts.append(val)
    return " → ".join(parts)


def format_label(idea, label_source: str, label_prefix: str = "") -> str:
    """Extract and format a single label from an idea object.

    Args:
        idea: Idea object with taxonomy fields (idea, instance, concept,
              concept_type, concept_type_definition)
        label_source: Stored field name or composite format key.
            Stored fields: "idea", "instance", "concept",
                           "concept_type", "concept_type_definition"
            Computed composites (assembled from stored fields):
                "ladder"               — instance → concept → concept_type → concept_type_definition
                "idea_concept_defined" — idea → concept → concept_type_definition
        label_prefix: Optional prefix. "{concept_type}: " substitutes idea.concept_type.

    Returns:
        Formatted label string, or empty string if all fields are empty.
    """
    if label_source == "ladder":
        raw = _compute_ladder(idea)
    elif label_source == "idea_concept_defined":
        raw = _compute_idea_concept_defined(idea)
    else:
        raw = (getattr(idea, label_source, '') or '').strip()

    if not raw:
        return ""

    # Apply prefix
    if not label_prefix:
        return raw

    # Dynamic substitution: replace {concept_type}, {concept}, etc.
    prefix = label_prefix
    for field_name in ('concept_type', 'concept', 'concept_type_definition'):
        placeholder = '{' + field_name + '}'
        if placeholder in prefix:
            val = (getattr(idea, field_name, '') or '').strip()
            prefix = prefix.replace(placeholder, val)

    return f"{prefix}{raw}"


def collect_unique_labels(
    ideas: list,
    label_source: str = "concept_type_definition",
    label_prefix: str = "",
) -> List[str]:
    """Collect unique label strings from a list of idea objects.

    Args:
        ideas: List of idea objects (IdeasExtractedSubmodel or EmbeddingsSubmodel)
        label_source: Stored field or composite format key. See format_label().
        label_prefix: Optional prefix to prepend

    Returns:
        List of unique label strings (preserving first-seen order).
    """
    seen = {}  # dict preserves insertion order
    for idea in ideas:
        label = format_label(idea, label_source, label_prefix)
        if label and label not in seen:
            seen[label] = True
    return list(seen.keys())


def build_cluster_hints(precluster_result: Optional[PreclusterResult]) -> str:
    """Format pre-cluster results as a prompt-injectable string.

    Args:
        precluster_result: Result from UMAP+HDBSCAN pre-clustering, or None.

    Returns:
        Formatted string for prompt injection. Empty string if no result.
    """
    if precluster_result is None:
        return ""

    if precluster_result.n_clusters == 0:
        return ""

    lines = [
        "<pre_clustering>",
        "Labels pre-clustered by semantic similarity (use as hints, not constraints):",
        "",
    ]

    for cluster_id in sorted(precluster_result.labels_by_cluster.keys()):
        labels = precluster_result.labels_by_cluster[cluster_id]
        labels_str = ", ".join(labels)
        lines.append(f"Cluster {cluster_id + 1} ({len(labels)} labels): {labels_str}")

    if precluster_result.noise_labels:
        noise_str = ", ".join(precluster_result.noise_labels)
        lines.append(f"Unclustered ({len(precluster_result.noise_labels)} labels): {noise_str}")

    lines.append("</pre_clustering>")
    return "\n".join(lines)
