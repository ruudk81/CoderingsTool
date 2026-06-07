"""
Neighbor stress test (cross-domain attribute proximity).

Pure Python, deterministic. Surfaces the cross-domain attribute pairs whose idea
embeddings sit closest together — the pairs most likely to make a coder hesitate,
and therefore the pairs that may need an explicit human merge/split/valence call
before P9 consolidation.

This is triage, not a verdict: high cosine similarity flags a candidate for a
human decision; it does not mean "must merge". Within-domain proximity was already
resolved by step 4 (P7 cross-facet consolidation), so only cross-domain pairs are
considered here — those are the ones P9 sees fresh.

Centroid = L2-normalized mean of the idea embeddings assigned to an attribute.
Similarity = cosine between centroids.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


# =============================================================================
# REPORT MODELS
# =============================================================================

class AttributePair(BaseModel):
    """A cross-domain attribute pair that is a candidate for a human policy call."""
    attribute_a: str
    domain_a: str
    count_a: int
    attribute_b: str
    domain_b: str
    count_b: int
    similarity: float


class NeighborStressReport(BaseModel):
    """Ranked cross-domain attribute proximity — input for human policy decisions."""
    n_attributes: int
    n_cross_domain_pairs_considered: int
    top_k: int
    min_similarity: float
    pairs: List[AttributePair] = Field(default_factory=list)


# =============================================================================
# HELPERS
# =============================================================================

def _attr(obj: Any, name: str) -> Any:
    """Read a field from a Pydantic model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def compute_attribute_centroids(
    idea_embeddings: Dict[str, Any],
    partition_results: Dict[str, Any],
) -> Dict[Tuple[str, str], Tuple[np.ndarray, int]]:
    """Compute an L2-normalized centroid per (domain, attribute).

    Keyed by (domain, attribute_name) so identical attribute names in different
    domains do not collide.

    Returns:
        {(domain, attribute): (unit_centroid_vector, idea_count)}
    """
    # Group assigned ideas by (domain, attribute)
    groups: Dict[Tuple[str, str], List[str]] = {}
    for domain, result in partition_results.items():
        assignments = _attr(result, "attribute_assignments") or {}
        for idea_id, attr_name in assignments.items():
            groups.setdefault((domain, attr_name), []).append(idea_id)

    centroids: Dict[Tuple[str, str], Tuple[np.ndarray, int]] = {}
    for key, idea_ids in groups.items():
        vectors = [
            np.asarray(idea_embeddings[i], dtype=float)
            for i in idea_ids if i in idea_embeddings
        ]
        if not vectors:
            continue
        mean = np.mean(np.stack(vectors), axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            continue
        centroids[key] = (mean / norm, len(idea_ids))
    return centroids


def nearest_cross_domain_pairs(
    centroids: Dict[Tuple[str, str], Tuple[np.ndarray, int]],
    top_k: int = 20,
    min_similarity: float = 0.0,
) -> NeighborStressReport:
    """Rank cross-domain attribute pairs by centroid cosine similarity.

    Args:
        centroids: output of compute_attribute_centroids
        top_k: max number of pairs to return
        min_similarity: only keep pairs at or above this cosine similarity
    """
    keys = list(centroids)
    pairs: List[AttributePair] = []
    considered = 0
    for (dom_a, attr_a), (dom_b, attr_b) in combinations(keys, 2):
        if dom_a == dom_b:
            continue  # within-domain proximity was resolved by step 4 (P7)
        considered += 1
        vec_a, count_a = centroids[(dom_a, attr_a)]
        vec_b, count_b = centroids[(dom_b, attr_b)]
        sim = float(np.dot(vec_a, vec_b))  # both unit vectors → cosine
        if sim < min_similarity:
            continue
        pairs.append(AttributePair(
            attribute_a=attr_a, domain_a=dom_a, count_a=count_a,
            attribute_b=attr_b, domain_b=dom_b, count_b=count_b,
            similarity=round(sim, 4),
        ))

    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return NeighborStressReport(
        n_attributes=len(keys),
        n_cross_domain_pairs_considered=considered,
        top_k=top_k,
        min_similarity=min_similarity,
        pairs=pairs[:top_k],
    )


def run_neighbor_stress_test(
    idea_embeddings: Dict[str, Any],
    partition_results: Dict[str, Any],
    top_k: int = 20,
    min_similarity: float = 0.0,
) -> Optional[NeighborStressReport]:
    """Build the cross-domain neighbor stress report. None if no embeddings."""
    if not idea_embeddings:
        return None
    centroids = compute_attribute_centroids(idea_embeddings, partition_results)
    if len(centroids) < 2:
        return None
    return nearest_cross_domain_pairs(centroids, top_k=top_k, min_similarity=min_similarity)


# =============================================================================
# FORMATTING
# =============================================================================

def format_neighbor_stress_report(report: NeighborStressReport) -> str:
    """Render the report as candidate human-decision points."""
    lines = [
        "=" * 80,
        "NEIGHBOR STRESS TEST — cross-domain attribute proximity (decision candidates)",
        "=" * 80,
        f"  Attributes: {report.n_attributes}   "
        f"Cross-domain pairs considered: {report.n_cross_domain_pairs_considered}",
        f"  Showing top {min(report.top_k, len(report.pairs))} "
        f"(min similarity {report.min_similarity})",
        "  (high similarity = coder may hesitate → candidate for a human merge/split/valence call)",
        "",
    ]
    for i, p in enumerate(report.pairs, 1):
        lines.append(
            f"  {i:>2}. {p.similarity:.3f}  "
            f"{p.attribute_a} [{p.domain_a}, {p.count_a}]"
            f"  ⇄  {p.attribute_b} [{p.domain_b}, {p.count_b}]"
        )
    if not report.pairs:
        lines.append("  (no pairs above threshold)")
    lines.append("=" * 80)
    return "\n".join(lines)
