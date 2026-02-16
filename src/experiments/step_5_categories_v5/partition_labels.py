"""
Label extraction and formatting utilities for Category Discovery V5.

Handles:
- Extracting text from idea objects based on configurable label_source
- Applying optional label_prefix (static or dynamic from idea fields)
- Formatting pre-cluster results as prompt-injectable hints
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PreclusterResult:
    """Result of UMAP+HDBSCAN pre-clustering on labels within a partition."""
    labels_by_cluster: Dict[int, List[str]]  # cluster_id → labels in that cluster
    noise_labels: List[str]                   # labels not assigned to any cluster
    n_clusters: int
    n_noise: int


def format_label(idea, label_source: str, label_prefix: str = "") -> str:
    """Extract and format a single label from an idea object.

    Args:
        idea: Idea object with taxonomy fields (idea, instance, node,
              semantic_category, category_label, root)
        label_source: Field to read ("category_label", "node", "idea", "instance")
        label_prefix: Optional prefix. "{root}: " substitutes idea.root.

    Returns:
        Formatted label string, or empty string if field is empty.
    """
    # Extract raw text from the configured field
    raw = (getattr(idea, label_source, '') or '').strip()
    if not raw:
        return ""

    # Apply prefix
    if not label_prefix:
        return raw

    # Dynamic substitution: replace {root}, {semantic_category}, etc.
    prefix = label_prefix
    for field_name in ('root', 'semantic_category', 'node', 'category_label'):
        placeholder = '{' + field_name + '}'
        if placeholder in prefix:
            val = (getattr(idea, field_name, '') or '').strip()
            prefix = prefix.replace(placeholder, val)

    return f"{prefix}{raw}"


def collect_unique_labels(
    ideas: list,
    label_source: str = "category_label",
    label_prefix: str = "",
) -> List[str]:
    """Collect unique label strings from a list of idea objects.

    Args:
        ideas: List of idea objects (IdeasExtractedSubmodel or EmbeddingsSubmodel)
        label_source: Field to read from each idea
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
