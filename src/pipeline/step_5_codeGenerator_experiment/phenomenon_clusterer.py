"""Fase 1 — deterministische fenomeen-ontdekking.

Attribuut = eenheidscentroïde van zijn idee-embeddings. Agglomeratief
(average linkage, cosine) met een drempel-sweep over P5..P95 van de paars-
gewijze afstanden; de partitie met het langste plateau (identiek over
opeenvolgende drempels) wint. Alleen schaalvrije parameters.

Ambiguity criterion: attributes with margins below half the median of finite margins
are marked ambiguous. In perfectly separated data, no attributes are marked. Only
attributes with margins significantly below typical indicate ambiguity.

Plateau selection: Real data always has a long single-cluster tail above the highest
merge height (every threshold above that point yields 1 cluster). Plateau detection
therefore filters out degenerate partitions (1 cluster or N clusters) BEFORE searching
for the longest plateau. Only non-degenerate partitions compete. If no valid partition
exists in the sweep, DegenerateClusteringError is raised.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class DegenerateClusteringError(RuntimeError):
    pass


@dataclass
class ClusterResult:
    labels: Dict[str, int]
    clusters: Dict[int, List[str]]
    threshold: float
    plateau_len: int
    margins: Dict[str, float] = field(default_factory=dict)
    ambiguous: List[str] = field(default_factory=list)
    neighbor: Dict[str, int] = field(default_factory=dict)


def attribute_centroids(idea_embeddings, assignments) -> Dict[str, np.ndarray]:
    """Compute unit-norm centroids from idea embeddings grouped by attribute.

    Attributes without embeddings (missing ideas) are silently omitted from
    output. Use missing_attributes() to log which attributes were dropped.

    Args:
        idea_embeddings: dict of idea_id → embedding vector
        assignments: dict of idea_id → attribute name

    Returns:
        dict of attribute → unit-norm centroid vector
    """
    sums: Dict[str, np.ndarray] = {}
    for idea_id, attr in assignments.items():
        emb = idea_embeddings.get(idea_id)
        if emb is None:
            continue
        v = np.asarray(emb, dtype=np.float64)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        v = v / n
        if attr in sums:
            sums[attr] += v
        else:
            sums[attr] = v.copy()
    out = {}
    for a, s in sums.items():
        n = np.linalg.norm(s)
        if n > 0:
            out[a] = s / n
    return out


def missing_attributes(assignments, centroids) -> List[str]:
    """Return sorted list of attributes that have no centroid (missing embeddings).

    Args:
        assignments: dict of idea_id → attribute name
        centroids: dict of attribute → centroid vector

    Returns:
        sorted list of attribute names in assignments but not in centroids
    """
    attrs_assigned = set(assignments.values())
    attrs_centered = set(centroids.keys())
    return sorted(attrs_assigned - attrs_centered)


def discover_phenomena(centroids: Dict[str, np.ndarray], n_sweep: int = 40) -> ClusterResult:
    names = sorted(centroids)                      # sortering → determinisme

    # Normalize all centroids defensively (idempotent for already-normalized input)
    norm_centroids = {}
    for n in names:
        v = np.asarray(centroids[n], dtype=np.float64)
        norm = np.linalg.norm(v)
        norm_centroids[n] = v / norm if norm > 0 else v

    # Special case: exactly 2 attributes
    if len(names) == 2:
        d = _cos(norm_centroids[names[0]], norm_centroids[names[1]])
        if np.isclose(d, 0):  # identical
            raise DegenerateClusteringError("all centroids identical")
        # Two singletons
        labels = {names[0]: 1, names[1]: 2}
        clusters = {1: [names[0]], 2: [names[1]]}
        margins = {names[0]: float("inf"), names[1]: float("inf")}
        neighbor = {names[0]: 2, names[1]: 1}
        return ClusterResult(labels, clusters, float(d), 0, margins, [], neighbor)

    X = np.stack([norm_centroids[n] for n in names])
    dists = pdist(X, metric="cosine")
    if np.allclose(dists, 0):
        raise DegenerateClusteringError("all centroids identical")
    Z = linkage(dists, method="average")
    lo, hi = np.percentile(dists, 5), np.percentile(dists, 95)
    thresholds = np.linspace(lo, hi, n_sweep)
    partitions = [tuple(fcluster(Z, t, criterion="distance")) for t in thresholds]

    # Filter to non-degenerate partitions before plateau detection.
    # Real data has a long single-cluster tail above the highest merge height;
    # without filtering, plateau detection selects the degenerate tail.
    valid_indices = []
    for i, part in enumerate(partitions):
        n_clusters = len(set(part))
        if n_clusters > 1 and n_clusters < len(names):
            valid_indices.append(i)

    if not valid_indices:
        raise DegenerateClusteringError(
            f"no valid partitions in sweep (all single or all-singletons)")

    # Find longest plateau among valid partitions only
    best_start, best_len, cur_start = valid_indices[0], 1, valid_indices[0]
    for idx in range(1, len(valid_indices)):
        i = valid_indices[idx]
        prev_i = valid_indices[idx - 1]
        if partitions[i] != partitions[prev_i]:
            cur_start = i
        if i - cur_start + 1 > best_len:
            best_start, best_len = cur_start, i - cur_start + 1

    labels_arr = partitions[best_start]
    n_clusters = len(set(labels_arr))
    # Safeguard assertion (should always pass given valid_indices filtering)
    assert n_clusters > 1 and n_clusters < len(names), \
        f"Degenerate partition passed filtering: {n_clusters} clusters for {len(names)} attributes"

    labels = {n: int(c) for n, c in zip(names, labels_arr)}
    clusters: Dict[int, List[str]] = {}
    for n, c in labels.items():
        clusters.setdefault(c, []).append(n)

    # Marges: (afstand naar dichtstbijzijnde ándere clustercentroïde − afstand
    # naar eigen clustercentroïde) / afstand-naar-andere. Singletons: marge inf.
    cluster_cent = {c: _unit_mean([norm_centroids[m] for m in ms]) for c, ms in clusters.items()}
    margins, neighbor = {}, {}
    for n in names:
        own = labels[n]
        d_own = _cos(norm_centroids[n], cluster_cent[own])
        others = [(c, _cos(norm_centroids[n], cc)) for c, cc in cluster_cent.items() if c != own]
        c2, d2 = min(others, key=lambda x: x[1])
        margins[n] = float("inf") if len(clusters[own]) == 1 else (d2 - d_own) / max(d2, 1e-12)
        neighbor[n] = c2
    finite = sorted(m for m in margins.values() if m != float("inf"))
    ambiguous = []
    if finite:
        cut = 0.5 * float(np.median(finite))     # schaalvrij: half de mediaan
        ambiguous = sorted([n for n in names if margins[n] < cut])
    return ClusterResult(labels, clusters, float(thresholds[best_start]),
                         best_len, margins, ambiguous, neighbor)


def _unit_mean(vs):
    s = np.mean(np.stack(vs), axis=0)
    n = np.linalg.norm(s)
    return s / n if n > 0 else s


def _cos(a, b):
    return float(1.0 - np.dot(a, b))
