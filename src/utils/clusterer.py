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
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP dimensionality reduction.

    Args:
        embeddings: Raw embeddings [N, dim]
        n_components: Target dimensionality (5 for clustering, 2 for visualization)
        n_neighbors: Balances local vs global structure
        min_dist: 0.0 for tight clusters
        metric: Distance metric ("cosine" or "euclidean")
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
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    metric: str = "euclidean",
) -> np.ndarray:
    """Run HDBSCAN clustering on (UMAP-reduced) embeddings.

    Args:
        embeddings: Reduced embeddings [N, dim]
        min_cluster_size: Minimum points to form a cluster
        min_samples: Core point density threshold
        metric: Distance metric ("euclidean" or "cosine")

    Returns:
        Cluster labels array [N]. -1 = noise.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
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
    *,
    # UMAP overrides
    n_components: Optional[int] = None,
    umap_metric: Optional[str] = None,
    min_dist: Optional[float] = None,
    # HDBSCAN overrides
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    cluster_metric: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """Estimate the number of distinct themes from embeddings.

    Runs one UMAP reduction then HDBSCAN clustering to produce a (low, high) span.

    Two modes:
    - **Caller-controlled** (min_cluster_size provided): 2-pass HDBSCAN.
      Permissive pass uses min_samples = min_cluster_size // 2 (more clusters),
      conservative pass uses min_samples = min_cluster_size (fewer clusters).
      Caller can override min_samples for the permissive pass explicitly.
    - **Auto** (no min_cluster_size): 3-pass log-scaled HDBSCAN (original behavior).

    Args:
        embeddings: Raw embeddings [N, dim]
        min_points: Minimum embeddings required for estimation
        n_components: UMAP target dimensionality (default 5)
        umap_metric: UMAP distance metric (default "cosine")
        min_dist: UMAP min_dist (default 0.0)
        min_cluster_size: HDBSCAN min_cluster_size (default: log-based auto)
        min_samples: HDBSCAN min_samples for permissive pass (default: min_cluster_size // 2)
        cluster_metric: HDBSCAN distance metric (default "euclidean")

    Returns:
        (low, high) theme count span, or None if too few points or no clear structure.
    """
    n = len(embeddings)
    if n < min_points:
        return None

    # UMAP defaults
    umap_n_components = n_components if n_components is not None else 5
    umap_met = umap_metric if umap_metric is not None else "cosine"
    umap_min_dist = min_dist if min_dist is not None else 0.0
    hdbscan_met = cluster_metric if cluster_metric is not None else "euclidean"

    # UMAP: single pass with capped n_neighbors
    n_neighbors = min(15, n - 1)
    reduced = run_umap(
        embeddings,
        n_components=umap_n_components,
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_met,
    )

    if min_cluster_size is not None:
        # Caller-controlled: 2-pass (permissive + conservative)
        mcs = max(2, min_cluster_size)
        ms_permissive = max(2, min_samples if min_samples is not None else mcs // 2)
        ms_conservative = mcs

        labels_perm = run_hdbscan(reduced, mcs, ms_permissive, metric=hdbscan_met)
        labels_cons = run_hdbscan(reduced, mcs, ms_conservative, metric=hdbscan_met)

        n_high = _count_clusters(labels_perm)
        n_low = _count_clusters(labels_cons)
    else:
        # Auto mode: 3-pass log-scaled (original behavior)
        ln_n = math.log(n)
        runs = [
            (max(2, math.ceil(2 * ln_n)), max(2, math.ceil(ln_n))),         # permissive
            (max(2, math.ceil(3 * ln_n)), max(2, math.ceil(1.5 * ln_n))),   # mid
            (max(2, math.ceil(4 * ln_n)), max(2, math.ceil(2 * ln_n))),     # conservative
        ]
        counts = []
        for mcs_auto, ms_auto in runs:
            labels = run_hdbscan(reduced, mcs_auto, ms_auto, metric=hdbscan_met)
            counts.append(_count_clusters(labels))

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
