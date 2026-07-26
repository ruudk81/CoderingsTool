"""Fase 1 — deterministische fenomeen-ontdekking.

Attribuut = eenheidscentroïde van zijn idee-embeddings. Agglomeratief
(average linkage, cosine) met een drempel-sweep over P5..P95 van de paars-
gewijze afstanden; de partitie met het langste plateau (identiek over
opeenvolgende drempels) wint. Alleen schaalvrije parameters.
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


def discover_phenomena(centroids: Dict[str, np.ndarray], n_sweep: int = 40) -> ClusterResult:
    names = sorted(centroids)                      # sortering → determinisme
    X = np.stack([centroids[n] for n in names])
    dists = pdist(X, metric="cosine")
    if np.allclose(dists, 0):
        raise DegenerateClusteringError("all centroids identical")
    Z = linkage(dists, method="average")
    lo, hi = np.percentile(dists, 5), np.percentile(dists, 95)
    thresholds = np.linspace(lo, hi, n_sweep)
    partitions = [tuple(fcluster(Z, t, criterion="distance")) for t in thresholds]

    best_start, best_len, cur_start = 0, 1, 0
    for i in range(1, len(partitions)):
        if partitions[i] != partitions[cur_start]:
            cur_start = i
        if i - cur_start + 1 > best_len:
            best_start, best_len = cur_start, i - cur_start + 1
    labels_arr = partitions[best_start]
    n_clusters = len(set(labels_arr))
    if n_clusters <= 1 or n_clusters >= len(names):
        raise DegenerateClusteringError(
            f"{n_clusters} clusters for {len(names)} attributes")

    labels = {n: int(c) for n, c in zip(names, labels_arr)}
    clusters: Dict[int, List[str]] = {}
    for n, c in labels.items():
        clusters.setdefault(c, []).append(n)

    # Marges: (afstand naar dichtstbijzijnde ándere clustercentroïde − afstand
    # naar eigen clustercentroïde) / afstand-naar-andere. Singletons: marge inf.
    cluster_cent = {c: _unit_mean([centroids[m] for m in ms]) for c, ms in clusters.items()}
    margins, neighbor = {}, {}
    for n in names:
        own = labels[n]
        d_own = _cos(centroids[n], cluster_cent[own])
        others = [(c, _cos(centroids[n], cc)) for c, cc in cluster_cent.items() if c != own]
        c2, d2 = min(others, key=lambda x: x[1])
        margins[n] = float("inf") if len(clusters[own]) == 1 else (d2 - d_own) / max(d2, 1e-12)
        neighbor[n] = c2
    finite = sorted(m for m in margins.values() if m != float("inf"))
    ambiguous = []
    if finite:
        cut = float(np.percentile(finite, 10))     # schaalvrij: onderste deciel
        ambiguous = [n for n in names if margins[n] <= cut]
    return ClusterResult(labels, clusters, float(thresholds[best_start]),
                         best_len, margins, ambiguous, neighbor)


def _unit_mean(vs):
    s = np.mean(np.stack(vs), axis=0)
    n = np.linalg.norm(s)
    return s / n if n > 0 else s


def _cos(a, b):
    return float(1.0 - np.dot(a, b))
