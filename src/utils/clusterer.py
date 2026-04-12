"""
Clustering utilities for theme count estimation.

Provides UMAP dimensionality reduction + HDBSCAN clustering to estimate
the number of distinct themes in a set of embeddings. Used by step 5
to give P8 a data-driven parsimony hint.

Also provides compute_dvc() for density diagnostics (not wired into pipeline).
"""

import math
import warnings
from typing import Dict, Optional, Tuple

import hdbscan
import numpy as np
import umap
from sklearn.neighbors import NearestNeighbors


def run_umap(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP dimensionality reduction with cosine metric.

    Args:
        embeddings: Raw embeddings [N, dim] (cosine metric handles normalization)
        n_components: Target dimensionality (5 for clustering, 2 for visualization)
        n_neighbors: Balances local vs global structure
        min_dist: 0.0 for tight clusters
        random_state: Reproducibility seed

    Returns:
        UMAP-reduced embeddings [N, n_components]
    """
    warnings.filterwarnings(
        "ignore", message="n_jobs value.*overridden to 1 by setting random_state"
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Run HDBSCAN clustering on (UMAP-reduced) embeddings.

    Args:
        embeddings: Reduced embeddings [N, dim]
        min_cluster_size: Minimum points to form a cluster
        min_samples: Core point density threshold

    Returns:
        Cluster labels array [N]. -1 = noise.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    clusterer.fit(embeddings)
    return clusterer.labels_


def _count_clusters(labels: np.ndarray) -> int:
    """Count clusters excluding noise label (-1)."""
    return len(set(labels)) - (1 if -1 in labels else 0)


def compute_dvc(
    embeddings: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Compute Density Variation Coefficient.

    DVC = std(d_k) / mean(d_k), where d_k is distance to k-th nearest neighbor.
    High DVC (>0.45) indicates varying density (clustered).
    Low DVC (<0.25) indicates uniform density (diffuse).

    Not wired into the pipeline — available for diagnostics and future use.

    Args:
        embeddings: Raw or reduced embeddings [N, dim]
        k: k-th nearest neighbor to measure

    Returns:
        Dict with 'dvc', 'mean_dk', 'std_dk'. Values are NaN if insufficient data.
    """
    n = len(embeddings)
    if n < k + 1:
        return {"dvc": float("nan"), "mean_dk": float("nan"), "std_dk": float("nan")}

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    d_k = distances[:, -1]
    mean_dk = float(np.mean(d_k))
    std_dk = float(np.std(d_k))

    if mean_dk == 0:
        return {"dvc": float("nan"), "mean_dk": mean_dk, "std_dk": std_dk}

    return {"dvc": std_dk / mean_dk, "mean_dk": mean_dk, "std_dk": std_dk}


def estimate_theme_count(
    embeddings: np.ndarray,
    min_points: int = 15,
) -> Optional[Tuple[int, int]]:
    """Estimate the number of distinct themes from embeddings.

    Runs one UMAP reduction then three HDBSCAN passes (permissive, mid,
    conservative) to produce a data-driven span. Dynamic parameters scale
    with dataset size using log-based formulas.

    Args:
        embeddings: Raw embeddings [N, dim]
        min_points: Minimum embeddings required for estimation

    Returns:
        (low, high) theme count span, or None if too few points or no clear structure.
    """
    n = len(embeddings)
    if n < min_points:
        return None

    ln_n = math.log(n)

    # UMAP: single pass with capped n_neighbors
    n_neighbors = min(15, n - 1)
    reduced = run_umap(embeddings, n_neighbors=n_neighbors)

    # Three HDBSCAN passes: permissive → mid → conservative
    runs = [
        (max(2, math.ceil(2 * ln_n)), max(2, math.ceil(ln_n))),         # permissive
        (max(2, math.ceil(3 * ln_n)), max(2, math.ceil(1.5 * ln_n))),   # mid
        (max(2, math.ceil(4 * ln_n)), max(2, math.ceil(2 * ln_n))),     # conservative
    ]

    counts = []
    for mcs, ms in runs:
        labels = run_hdbscan(reduced, mcs, ms)
        counts.append(_count_clusters(labels))

    # permissive gives highest count, conservative gives lowest
    n_high = counts[0]  # permissive
    n_low = counts[2]   # conservative

    # If even permissive finds ≤1 clusters: no clear structure
    if n_high <= 1:
        return None

    # Floor conservative at 2 if permissive found clusters
    n_low = max(2, n_low)

    # Ensure low ≤ high
    if n_low > n_high:
        n_low = n_high

    return (n_low, n_high)
