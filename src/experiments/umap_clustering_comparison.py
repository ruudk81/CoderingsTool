#%%

"""
UMAP × Clustering Algorithm Comparison Experiment

Goal: Test different UMAP configurations with different clustering techniques
to find optimal combinations for survey response coding.

Experiment Scope (Phase 1: Small Dataset Pilot):
- Dataset: Vezet Q20 (n=50) from Step 4 cache
- UMAP grid: 9 configs (neighbors: 5, 10, 30 × components: 5, 10, 20 × min_dist: 0.1)
- Clustering: HDBSCAN (DBCV-based), Agglomerative (Ward's), K-means (silhouette-optimal)
- Output: Excel results + kNN elbow plots
"""

import os
import sys
import warnings

# Suppress UMAP n_jobs warning when using random_state
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")

# Suppress HDBSCAN validity warnings (divide by zero in edge cases - handled gracefully)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")

# Add src paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import product
import matplotlib.pyplot as plt

# Clustering and dimensionality reduction
import umap
import hdbscan
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# Local imports
import models
from utils.cacheManager import generate_enhanced_variable_key
from config import CacheConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VARIABLE = "Q20"
SAMPLE_SIZE = 50

# UMAP grid configuration
UMAP_NEIGHBORS = [5, 10, 30]
UMAP_COMPONENTS = [5, 10, 20]
UMAP_MIN_DIST = 0.1

# K-means and Agglomerative K range
K_RANGE = range(3, 16)

# Output paths
EXPORTS_DIR = Path(__file__).parent.parent.parent / "exports"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_step4_embeddings() -> Tuple[np.ndarray, List[str]]:
    """
    Load Step 4 embeddings from cache.

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        idea_texts: list of idea text strings
    """
    import pickle

    # Generate variable key (Q20 with sample_size=50 -> "Q20_50")
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    # Build cache file path directly (bypassing database lookup)
    # Cache is in project_root/data/cache
    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem
    cache_filename = f"005_embeddings_{base_name}_{variable_key}.pkl"
    cache_path = cache_dir / cache_filename

    print(f"Loading embeddings from: {cache_path}")

    if not cache_path.exists():
        raise ValueError(f"No cached embeddings found at {cache_path}. Run pipeline Step 4 first.")

    # Load pickled data
    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    # Reconstruct Pydantic models
    data = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

    if data is None:
        raise ValueError(f"No cached embeddings found for {FILENAME}/{variable_key}. Run pipeline Step 4 first.")

    # Extract embeddings and texts from all ideas
    embeddings_list = []
    idea_texts = []

    for response in data:
        if response.response_ideas:
            for idea in response.response_ideas:
                if idea.idea_embedding is not None:
                    embeddings_list.append(idea.idea_embedding)
                    idea_texts.append(idea.idea)

    if not embeddings_list:
        raise ValueError("No embeddings found in cached data")

    embeddings = np.vstack(embeddings_list)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    return embeddings, idea_texts


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    return normalize(embeddings, norm='l2', axis=1)


# =============================================================================
# UMAP FUNCTIONS
# =============================================================================

def run_umap(embeddings: np.ndarray, n_neighbors: int, n_components: int,
             min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """
    Run UMAP dimensionality reduction.

    Args:
        embeddings: L2-normalized embeddings
        n_neighbors: UMAP n_neighbors parameter
        n_components: Target dimensionality
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility

    Returns:
        reduced: UMAP-reduced embeddings
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='euclidean',
        random_state=random_state
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def compute_dbcv(labels: np.ndarray, embeddings: np.ndarray) -> float:
    """
    Compute DBCV (Density-Based Clustering Validation) score.
    Uses hdbscan's built-in validity index.

    Args:
        labels: Cluster labels (-1 for noise)
        embeddings: Data points

    Returns:
        DBCV score (higher is better, range roughly -1 to 1)
    """
    try:
        # Filter out noise points for DBCV calculation
        mask = labels >= 0
        if mask.sum() < 2:
            return -1.0

        # Use hdbscan's validity index
        # Cast to float64 to avoid dtype mismatch in hdbscan validity
        from hdbscan import validity
        embeddings_f64 = embeddings[mask].astype(np.float64)
        labels_filtered = labels[mask]
        score = validity.validity_index(embeddings_f64, labels_filtered)
        return float(score)
    except Exception as e:
        print(f"DBCV calculation failed: {e}")
        return -1.0


def run_hdbscan_grid(reduced: np.ndarray, n_samples: int) -> Dict:
    """
    Run HDBSCAN with grid search using DBCV for selection.

    Grid: mcs = [sqrt(n), 0.5*sqrt(n), 0.25*sqrt(n)]
          ms = max(1, 0.5 * mcs)

    Args:
        reduced: UMAP-reduced embeddings
        n_samples: Number of samples (for parameter calculation)

    Returns:
        dict with best labels, params, and metrics
    """
    sqrt_n = np.sqrt(n_samples)
    mcs_grid = [
        max(2, int(sqrt_n)),           # sqrt(n)
        max(2, int(0.5 * sqrt_n)),     # 0.5 * sqrt(n)
        max(2, int(0.25 * sqrt_n))     # 0.25 * sqrt(n)
    ]

    best_result = None
    best_dbcv = -2.0

    for mcs in mcs_grid:
        ms = max(1, int(0.5 * mcs))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(reduced)

        # Calculate DBCV
        dbcv = compute_dbcv(labels, reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = (labels == -1).sum()
        noise_rate = noise_count / len(labels)

        print(f"    HDBSCAN mcs={mcs}, ms={ms}: k={n_clusters}, noise={noise_rate:.2%}, DBCV={dbcv:.3f}")

        if dbcv > best_dbcv:
            best_dbcv = dbcv
            best_result = {
                'labels': labels,
                'n_clusters': n_clusters,
                'noise_rate': noise_rate,
                'dbcv': dbcv,
                'params': {'min_cluster_size': mcs, 'min_samples': ms}
            }

    return best_result


def get_k_grid(n_samples: int) -> List[int]:
    """
    Generate k grid based on sqrt(n).

    Grid: 0.5 * sqrt(n), 1 * sqrt(n), 2 * sqrt(n)

    Args:
        n_samples: Number of data points

    Returns:
        List of k values to try (sorted, unique, >= 2)
    """
    baseline = np.sqrt(n_samples)
    k_grid = [
        max(2, int(0.5 * baseline)),
        max(2, int(baseline)),
        max(2, int(2 * baseline))
    ]
    # Remove duplicates and sort
    return sorted(set(k_grid))


def run_agglomerative(reduced: np.ndarray, n_samples: int) -> List[Dict]:
    """
    Run Agglomerative clustering with Ward's linkage for all k values in grid.

    Uses k grid based on sqrt(n): [0.5*sqrt(n), sqrt(n), 2*sqrt(n)]

    Args:
        reduced: L2-normalized UMAP-reduced embeddings
        n_samples: Number of samples (for k grid calculation)

    Returns:
        List of dicts with labels, k, and metrics for each k value tested
    """
    k_grid = get_k_grid(n_samples)
    all_results = []
    best_silhouette = -2.0
    best_k = None

    for k in k_grid:
        if k >= len(reduced):
            continue

        clusterer = AgglomerativeClustering(
            n_clusters=k,
            metric='euclidean',
            linkage='ward'
        )
        labels = clusterer.fit_predict(reduced)

        # Calculate silhouette
        if len(set(labels)) > 1:
            sil = silhouette_score(reduced, labels)
        else:
            sil = -1.0

        print(f"    Agglomerative k={k}: silhouette={sil:.3f}")

        all_results.append({
            'labels': labels,
            'n_clusters': k,
            'silhouette': sil,
            'params': {'n_clusters': k}
        })

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    if best_k:
        print(f"    Agglomerative best: k={best_k}, silhouette={best_silhouette:.3f}")

    return all_results


def run_kmeans(reduced: np.ndarray, n_samples: int) -> List[Dict]:
    """
    Run K-means clustering for all k values in grid.

    Uses k grid based on sqrt(n): [0.5*sqrt(n), sqrt(n), 2*sqrt(n)]

    Args:
        reduced: L2-normalized UMAP-reduced embeddings
        n_samples: Number of samples (for k grid calculation)

    Returns:
        List of dicts with labels, k, and metrics for each k value tested
    """
    k_grid = get_k_grid(n_samples)
    all_results = []
    best_silhouette = -2.0
    best_k = None

    for k in k_grid:
        if k >= len(reduced):
            continue

        clusterer = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = clusterer.fit_predict(reduced)

        # Calculate silhouette
        if len(set(labels)) > 1:
            sil = silhouette_score(reduced, labels)
        else:
            sil = -1.0

        print(f"    K-means k={k}: silhouette={sil:.3f}")

        all_results.append({
            'labels': labels,
            'n_clusters': k,
            'silhouette': sil,
            'params': {'n_clusters': k}
        })

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    if best_k:
        print(f"    K-means best: k={best_k}, silhouette={best_silhouette:.3f}")

    return all_results


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_cluster_coherence(labels: np.ndarray, original_embeddings: np.ndarray) -> float:
    """
    Calculate mean intra-cluster cosine similarity using original embeddings.

    Args:
        labels: Cluster labels
        original_embeddings: Original (not UMAP-reduced) L2-normalized embeddings

    Returns:
        Mean coherence score (0-1, higher is better)
    """
    unique_labels = [l for l in set(labels) if l >= 0]  # Exclude noise

    if not unique_labels:
        return 0.0

    coherences = []
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = original_embeddings[mask]

        if len(cluster_embeddings) < 2:
            coherences.append(1.0)  # Single-point cluster is perfectly coherent
            continue

        # Calculate pairwise cosine similarities
        # For L2-normalized vectors: cosine_sim = dot product
        similarities = cluster_embeddings @ cluster_embeddings.T

        # Get upper triangle (excluding diagonal)
        n = len(cluster_embeddings)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_tri_indices]

        coherences.append(np.mean(pairwise_sims))

    return np.mean(coherences)


def calculate_coherence_breakdown(
    labels: np.ndarray,
    original_embeddings: np.ndarray,
    unacceptable_threshold: float = 0.70,
    low_threshold: float = 0.90,
    high_threshold: float = 0.95
) -> Dict:
    """
    Calculate per-cluster coherence and classify into unacceptable/low/moderate/high.

    Thresholds:
    - Unacceptable: coherence < 0.70 (clusters too heterogeneous)
    - Low: 0.70 ≤ coherence < 0.90 (marginal quality)
    - Moderate: 0.90 ≤ coherence < 0.95 (acceptable quality)
    - High: coherence ≥ 0.95 (excellent quality)

    Args:
        labels: Cluster labels
        original_embeddings: L2-normalized original embeddings
        unacceptable_threshold: Below this = unacceptable coherence (default 0.70)
        low_threshold: Below this (but >= unacceptable) = low coherence (default 0.90)
        high_threshold: Above this = high coherence (default 0.95)

    Returns:
        Dict with:
        - per_cluster: List of (label, size, coherence) tuples
        - n_unacceptable: Count of clusters with coherence < 0.70
        - n_low: Count of clusters with 0.70 <= coherence < 0.90
        - n_moderate: Count of clusters with 0.90 <= coherence < 0.95
        - n_high: Count of clusters with coherence >= 0.95
        - summary: String like "1 unacceptable, 2 low, 3 moderate, 4 high"
    """
    unique_labels = [l for l in set(labels) if l >= 0]  # Exclude noise

    per_cluster = []
    n_unacceptable = 0
    n_low = 0
    n_moderate = 0
    n_high = 0

    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = original_embeddings[mask]
        size = len(cluster_embeddings)

        if size < 2:
            coherence = 1.0  # Single-point cluster is perfectly coherent
        else:
            # Pairwise cosine similarity (L2-normalized → dot product)
            similarities = cluster_embeddings @ cluster_embeddings.T
            n = len(cluster_embeddings)
            upper_tri = np.triu_indices(n, k=1)
            coherence = float(np.mean(similarities[upper_tri]))

        per_cluster.append((label, size, coherence))

        if coherence < unacceptable_threshold:
            n_unacceptable += 1
        elif coherence < low_threshold:
            n_low += 1
        elif coherence < high_threshold:
            n_moderate += 1
        else:
            n_high += 1

    return {
        "per_cluster": per_cluster,
        "n_unacceptable": n_unacceptable,
        "n_low": n_low,
        "n_moderate": n_moderate,
        "n_high": n_high,
        "summary": f"{n_unacceptable} unacceptable, {n_low} low, {n_moderate} moderate, {n_high} high"
    }


def calculate_metrics(labels: np.ndarray, reduced: np.ndarray,
                      original_embeddings: np.ndarray) -> Dict:
    """
    Calculate all evaluation metrics for a clustering result.

    Args:
        labels: Cluster labels
        reduced: UMAP-reduced embeddings
        original_embeddings: Original L2-normalized embeddings

    Returns:
        dict with all metrics
    """
    # Filter out noise for some metrics
    mask = labels >= 0
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_rate = (~mask).sum() / len(labels) if len(labels) > 0 else 0

    metrics = {
        'n_clusters': n_clusters,
        'noise_rate': noise_rate,
    }

    # Coherence (uses original embeddings)
    metrics['coherence'] = calculate_cluster_coherence(labels, original_embeddings)

    # Coherence breakdown (unacceptable/low/moderate/high)
    breakdown = calculate_coherence_breakdown(labels, original_embeddings)
    metrics['coherence_n_unacceptable'] = breakdown['n_unacceptable']
    metrics['coherence_n_low'] = breakdown['n_low']
    metrics['coherence_n_moderate'] = breakdown['n_moderate']
    metrics['coherence_n_high'] = breakdown['n_high']
    metrics['coherence_breakdown'] = breakdown['summary']

    # DBCV (on reduced space)
    metrics['dbcv'] = compute_dbcv(labels, reduced)

    # Silhouette and Davies-Bouldin (only if we have valid clusters and no noise, or filter noise)
    if mask.sum() >= 2 and n_clusters >= 2:
        try:
            metrics['silhouette'] = silhouette_score(reduced[mask], labels[mask])
        except:
            metrics['silhouette'] = np.nan

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(reduced[mask], labels[mask])
        except:
            metrics['davies_bouldin'] = np.nan
    else:
        metrics['silhouette'] = np.nan
        metrics['davies_bouldin'] = np.nan

    return metrics


def apply_algorithm_metric_filter(metrics: Dict, algorithm: str) -> Dict:
    """
    Apply algorithm-appropriate metric filtering.

    HDBSCAN uses DBCV (density-based) - silhouette/DB assume convex clusters.
    Agglomerative/K-means use silhouette/DB - DBCV is for density-based methods.

    Args:
        metrics: Dict of computed metrics
        algorithm: 'HDBSCAN', 'Agglomerative', or 'K-means'

    Returns:
        Dict with inappropriate metrics set to np.nan
    """
    filtered = metrics.copy()

    if algorithm == 'HDBSCAN':
        # HDBSCAN: keep DBCV, remove silhouette and davies_bouldin
        filtered['silhouette'] = np.nan
        filtered['davies_bouldin'] = np.nan
    else:
        # Agglomerative and K-means: keep silhouette and davies_bouldin, remove DBCV
        filtered['dbcv'] = np.nan

    return filtered


# =============================================================================
# KNN ELBOW PLOTS + KNEE DETECTION
# =============================================================================

def find_knee(distances: np.ndarray) -> Tuple[int, float]:
    """
    Find knee point using max perpendicular distance to chord.

    Args:
        distances: Sorted k-NN distances

    Returns:
        knee_idx: Index of knee point
        max_dist: Maximum distance to chord line
    """
    n = len(distances)
    all_coords = np.vstack((np.arange(n), distances)).T

    # Line from first to last point
    first, last = all_coords[0], all_coords[-1]
    line_vec = last - first
    line_len = np.linalg.norm(line_vec)
    line_vec_norm = line_vec / line_len

    # Perpendicular distances to line
    vec_from_first = all_coords - first
    scalar_proj = np.dot(vec_from_first, line_vec_norm)
    proj = np.outer(scalar_proj, line_vec_norm)
    perp = vec_from_first - proj
    dist_to_line = np.linalg.norm(perp, axis=1)

    knee_idx = np.argmax(dist_to_line)
    max_dist = dist_to_line[knee_idx]

    return knee_idx, max_dist


def compute_elbow_strength(distances: np.ndarray) -> Tuple[int, float, float]:
    """
    Compute normalized elbow strength.

    Args:
        distances: Sorted k-NN distances

    Returns:
        knee_idx: Index of knee point
        knee_distance: k-NN distance at knee point
        strength: Elbow strength (0-1), higher = sharper elbow
    """
    knee_idx, max_dist = find_knee(distances)
    value_range = distances.max() - distances.min()

    if value_range == 0:
        return knee_idx, distances[knee_idx], 0.0

    strength = max_dist / value_range
    return knee_idx, distances[knee_idx], strength


def classify_density_structure(strength: float) -> str:
    """
    Classify density structure based on elbow strength.

    Args:
        strength: Elbow strength value

    Returns:
        Classification: "strong", "weak", or "none"

    Interpretation:
        - strong (>0.10): Clear density separation → HDBSCAN recommended
        - weak (0.05-0.10): Some structure → test both approaches
        - none (<0.05): Smooth density → Agglomerative/K-means recommended
    """
    if strength > 0.10:
        return "strong"
    elif strength > 0.05:
        return "weak"
    else:
        return "none"


def compute_slope_ratio(distances: np.ndarray, knee_idx: int,
                        window: int = 5, eps: float = 1e-9) -> float:
    """
    Compute slope ratio at knee point to validate if knee is "sharp" enough.

    A meaningful knee should have slope_after / slope_before >= 3,
    indicating a significant acceleration in the distance curve.

    Args:
        distances: Sorted k-NN distances
        knee_idx: Index of the detected knee point
        window: Number of points to use for slope calculation (default 5)
        eps: Small value to avoid division by zero

    Returns:
        Slope ratio (slope_after / slope_before). Values >= 3 indicate sharp knee.
        Returns 0 if knee is too close to edges for safe computation.
    """
    n = len(distances)

    # Cannot compute safely if knee is too close to edges
    if knee_idx < window or knee_idx + window >= n:
        return 0.0

    # Slope before knee: (y[knee] - y[knee - window]) / window
    slope_before = (distances[knee_idx] - distances[knee_idx - window]) / window

    # Slope after knee: (y[knee + window] - y[knee]) / window
    slope_after = (distances[knee_idx + window] - distances[knee_idx]) / window

    return slope_after / (slope_before + eps)


def choose_cluster_strategy_via_knee(
    X: np.ndarray,
    knn_k: int = 5,
    metric: str = "euclidean",
    kneedle_S: float = 1.0,
    lower_mult: float = 0.5,
    upper_mult: float = 4.0,
    tail_frac: float = 0.85,
    slope_ratio_threshold: float = 3.0,
    slope_window: int = 5
) -> Dict:
    """
    Detect knee in kNN distance curve and recommend clustering strategy.

    SEARCH WINDOW:
    - start_idx = 1 (skip first point)
    - end_idx = int(0.85 * n) (exclude extreme tail)
    - Let KneeLocator find the natural knee in the full meaningful range

    ACCEPTANCE BOUNDS (for evaluating if knee suggests HDBSCAN):
    - K_min = max(3, int(0.5 * sqrt(n)))
    - K_max = min(int(4 * sqrt(n)), int(0.85 * n))

    SLOPE RATIO TEST:
    - Validates that the knee is "sharp" enough (not just gradual curvature)
    - slope_after / slope_before >= 3.0 indicates a meaningful density transition
    - Prevents recommending HDBSCAN for nearly linear curves

    Only recommend HDBSCAN if:
    1. Knee exists
    2. Knee falls within [K_min, K_max]
    3. Slope ratio >= threshold (knee is sharp enough)

    Args:
        X: L2-normalized embeddings (n_samples, n_features)
        knn_k: Which neighbor distance to use (default 5)
        metric: Distance metric for kNN (default "euclidean")
        kneedle_S: Sensitivity parameter for KneeLocator
        lower_mult: Multiplier for lower bound (default 0.5)
        upper_mult: Multiplier for upper bound (default 4.0)
        tail_frac: Maximum fraction of n for search/acceptance (default 0.85)
        slope_ratio_threshold: Minimum slope ratio for sharp knee (default 3.0)
        slope_window: Window size for slope calculation (default 5)

    Returns:
        Dict with:
        - recommendation: "HDBSCAN" | "AGGLOMERATIVE_OR_KMEANS"
        - K: Knee point index (or None if not found)
        - K_min: Lower acceptance bound
        - K_max: Upper acceptance bound
        - start_idx: Start of search window
        - end_idx: End of search window
        - is_meaningful: Whether knee passes all tests (bounds + slope ratio)
        - in_bounds: Whether knee falls within acceptance bounds
        - has_sharp_knee: Whether slope ratio >= threshold
        - slope_ratio: Computed slope ratio at knee
        - knee_distance: kNN distance at knee point
        - distances: Sorted kNN distances array
        - elbow_strength: Normalized elbow strength (for comparison)
    """
    n = len(X)
    sqrt_n = np.sqrt(n)

    # 1. Compute kNN distances
    nn = NearestNeighbors(n_neighbors=knn_k + 1, metric=metric)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = distances[:, knn_k]

    # 2. Sort for elbow analysis
    sorted_distances = np.sort(k_distances)

    # 3. Define search window: [1, 0.85*n]
    # Search the full meaningful range - let KneeLocator find the natural knee
    start_idx = 1
    end_idx = int(tail_frac * n)

    # Ensure valid window
    if end_idx <= start_idx:
        end_idx = n - 1

    # Extract search segment for knee detection
    search_distances = sorted_distances[start_idx:end_idx]
    search_x = np.arange(len(search_distances))

    # 4. Detect knee on search window
    kneedle = KneeLocator(
        x=search_x,
        y=search_distances,
        S=kneedle_S,
        curve="convex",
        direction="increasing"
    )

    # 5. Map knee back to original coordinate system
    K_in_window = kneedle.knee  # Index within search window
    if K_in_window is not None:
        K = start_idx + K_in_window  # Map back to original index
    else:
        K = None

    # 6. Compute acceptance bounds (narrower than search window)
    # These bounds determine if the knee position suggests HDBSCAN is appropriate
    K_min = max(3, int(lower_mult * sqrt_n))
    K_max = min(int(upper_mult * sqrt_n), int(tail_frac * n))

    # 7. Check if knee is within acceptance bounds
    in_bounds = K is not None and K_min <= K <= K_max

    # 8. Compute slope ratio to validate knee sharpness
    if K is not None:
        slope_ratio = compute_slope_ratio(sorted_distances, K, window=slope_window)
        has_sharp_knee = slope_ratio >= slope_ratio_threshold
    else:
        slope_ratio = 0.0
        has_sharp_knee = False

    # 9. Determine if knee is meaningful for HDBSCAN
    # Knee must: exist AND fall within bounds AND be sharp enough
    is_meaningful = in_bounds and has_sharp_knee

    # 10. Compute elbow strength on search window
    if K is not None and len(search_distances) > 2:
        knee_distance = sorted_distances[K]
        _, max_dist = find_knee(search_distances)
        value_range = search_distances.max() - search_distances.min()
        elbow_strength = max_dist / value_range if value_range > 0 else 0.0
    else:
        knee_distance = None
        # Still compute elbow strength on search window
        if len(search_distances) > 2:
            _, _, elbow_strength = compute_elbow_strength(search_distances)
        else:
            elbow_strength = 0.0

    return {
        "recommendation": "HDBSCAN" if is_meaningful else "AGGLOMERATIVE_OR_KMEANS",
        "K": K,
        "K_min": K_min,
        "K_max": K_max,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "is_meaningful": is_meaningful,
        "in_bounds": in_bounds,
        "has_sharp_knee": has_sharp_knee,
        "slope_ratio": slope_ratio,
        "knee_distance": knee_distance,
        "distances": sorted_distances,
        "elbow_strength": elbow_strength,
        "n": n,
        "sqrt_n": sqrt_n
    }


def generate_knn_elbow_plots(embeddings: np.ndarray, umap_configs: List[Dict],
                             output_path: Path, k: int = 5) -> List[Dict]:
    """
    Generate kNN distance elbow plots with knee detection and acceptance bounds
    for each UMAP configuration.

    Args:
        embeddings: L2-normalized original embeddings
        umap_configs: List of UMAP config dicts with 'n_neighbors', 'n_components', 'reduced'
        output_path: Path to save the figure
        k: k-th nearest neighbor distance to plot

    Returns:
        List of dicts with knee detection results per config
    """
    n_configs = len(umap_configs)
    n_cols = 3
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_configs > 1 else [axes]

    elbow_results = []

    for i, config in enumerate(umap_configs):
        ax = axes[i]
        reduced = config['reduced']

        # Use new knee detection with acceptance bounds
        knee_result = choose_cluster_strategy_via_knee(reduced, knn_k=k)

        # Store results with all new fields including search window and slope ratio
        elbow_results.append({
            'n_neighbors': config['n_neighbors'],
            'n_components': config['n_components'],
            'knee_K': knee_result['K'],
            'K_min': knee_result['K_min'],
            'K_max': knee_result['K_max'],
            'start_idx': knee_result['start_idx'],
            'end_idx': knee_result['end_idx'],
            'in_bounds': knee_result['in_bounds'],
            'has_sharp_knee': knee_result['has_sharp_knee'],
            'slope_ratio': knee_result['slope_ratio'],
            'knee_meaningful': knee_result['is_meaningful'],
            'knee_recommendation': knee_result['recommendation'],
            'knee_distance': knee_result['knee_distance'],
            'elbow_strength': knee_result['elbow_strength'],
            'density_structure': classify_density_structure(knee_result['elbow_strength'])
        })

        # Plot the kNN distances
        sorted_distances = knee_result['distances']
        ax.plot(sorted_distances, linewidth=1.5, color='blue', label='k-NN distances')

        # Show truncated search window with vertical lines
        ax.axvline(x=knee_result['start_idx'], color='gray', linestyle=':', alpha=0.7)
        ax.axvline(x=knee_result['end_idx'], color='gray', linestyle=':', alpha=0.7)
        # Shade search window (where knee detection operates)
        ax.axvspan(knee_result['start_idx'], knee_result['end_idx'],
                   alpha=0.08, color='blue', label='Search window')

        # Shade acceptance region (where knee is considered meaningful)
        ax.axvspan(knee_result['K_min'], knee_result['K_max'],
                   alpha=0.15, color='green', label='Acceptance zone')

        # Mark knee point only if slope ratio >= 3 (sharp knee)
        if knee_result['K'] is not None and knee_result['has_sharp_knee']:
            color = 'green' if knee_result['is_meaningful'] else 'orange'
            marker = 'o' if knee_result['is_meaningful'] else 'X'
            ax.scatter([knee_result['K']], [knee_result['knee_distance']],
                       color=color, s=100, zorder=5, marker=marker,
                       label=f"Knee @ {knee_result['K']}")
            ax.axhline(y=knee_result['knee_distance'], color=color, linestyle='--', alpha=0.3)
            ax.axvline(x=knee_result['K'], color=color, linestyle='--', alpha=0.3)

        # Title with search window and bounds info
        window_str = f"Win:[{knee_result['start_idx']}-{knee_result['end_idx']}]"
        bounds_str = f"[{knee_result['K_min']}-{knee_result['K_max']}]"
        knee_str = f"K={knee_result['K']}" if knee_result['K'] is not None else "K=None"
        short_rec = "HDBSCAN" if knee_result['recommendation'] == "HDBSCAN" else "Agg/KM"
        ax.set_title(f"n={config['n_neighbors']}, d={config['n_components']} | "
                     f"{knee_str} {bounds_str} → {short_rec}\n{window_str}")
        ax.set_xlabel('Points (sorted)')
        ax.set_ylabel(f'{k}-NN distance')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('kNN Distance Elbow Plots with Knee Detection & Acceptance Bounds', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved kNN elbow plots to {output_path}")

    # Print elbow analysis summary with search window, acceptance bounds, and slope ratio
    print("\nKnee Detection Analysis (bounds + slope ratio test):")
    print("  Legend: bounds=IN/OUT, slope=ratio (≥3.0 = sharp)")
    for r in elbow_results:
        knee_str = f"K={r['knee_K']:2d}" if r['knee_K'] is not None else "K=None"
        bounds_str = f"[{r['K_min']}-{r['K_max']}]"
        bounds_status = "IN" if r['in_bounds'] else "OUT"
        slope_str = f"slope={r['slope_ratio']:.1f}"
        sharp_status = "sharp" if r['has_sharp_knee'] else "flat"
        print(f"  UMAP n={r['n_neighbors']:2d}, d={r['n_components']:2d}: "
              f"{knee_str} {bounds_str} {bounds_status:3s} | {slope_str} ({sharp_status}) → {r['knee_recommendation']}")

    return elbow_results


# =============================================================================
# CLUSTER INSPECTION
# =============================================================================

def print_cluster_samples(labels: np.ndarray, idea_texts: List[str],
                          algorithm: str, params: Dict, coherence: float,
                          original_embeddings: Optional[np.ndarray] = None,
                          max_samples: int = 10, random_state: int = 42):
    """
    Print random sample of responses from each cluster with per-cluster coherence.

    Args:
        labels: Cluster labels
        idea_texts: List of idea text strings
        algorithm: Algorithm name
        params: Algorithm parameters
        coherence: Overall coherence score
        original_embeddings: L2-normalized original embeddings (for per-cluster coherence)
        max_samples: Max samples per cluster
        random_state: Random seed for reproducibility
    """
    np.random.seed(random_state)

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    # Calculate per-cluster coherence if embeddings provided
    cluster_coherence_map = {}
    if original_embeddings is not None:
        breakdown = calculate_coherence_breakdown(labels, original_embeddings)
        cluster_coherence_map = {label: coh for label, size, coh in breakdown['per_cluster']}

    print(f"\n{'='*70}")
    print(f"CLUSTER SAMPLES: {algorithm}")
    print(f"Parameters: {params}")
    print(f"Coherence: {coherence:.3f} | Clusters: {n_clusters}")
    print(f"{'='*70}")

    for label in unique_labels:
        mask = labels == label
        cluster_texts = [idea_texts[i] for i in range(len(labels)) if labels[i] == label]
        cluster_size = len(cluster_texts)

        if label == -1:
            print(f"\n--- NOISE ({cluster_size} items) ---")
        else:
            # Show per-cluster coherence if available
            if label in cluster_coherence_map:
                coh = cluster_coherence_map[label]
                if coh < 0.70:
                    coh_class = "unacceptable"
                elif coh < 0.90:
                    coh_class = "low"
                elif coh < 0.95:
                    coh_class = "moderate"
                else:
                    coh_class = "high"
                print(f"\n--- Cluster {label} ({cluster_size} items, coh={coh:.3f} [{coh_class}]) ---")
            else:
                print(f"\n--- Cluster {label} ({cluster_size} items) ---")

        # Random sample
        if cluster_size <= max_samples:
            samples = cluster_texts
        else:
            indices = np.random.choice(cluster_size, max_samples, replace=False)
            samples = [cluster_texts[i] for i in indices]

        for i, text in enumerate(samples, 1):
            # Strip metadata prefix if present (format: [key=value][key=value]... actual_text)
            display_text = text
            if display_text.startswith('['):
                # Find the last ] and take text after it
                last_bracket = display_text.rfind(']')
                if last_bracket != -1:
                    display_text = display_text[last_bracket + 1:].strip()

            # Truncate long texts
            if len(display_text) > 120:
                display_text = display_text[:120] + "..."
            print(f"  {i}. {display_text}")


# =============================================================================
# COHERENCE VS K ANALYSIS
# =============================================================================

def print_coherence_by_k_table(results: List[Dict]):
    """
    Print table showing cluster coherence breakdown grouped by number of clusters (k),
    separated by algorithm.

    Shows a separate table for each algorithm (HDBSCAN, Agglomerative, K-means),
    with values averaged across UMAP configurations for each k.
    """
    algos = ['HDBSCAN', 'Agglomerative', 'K-means']

    print("\n" + "=" * 80)
    print("COHERENCE BREAKDOWN BY NUMBER OF CLUSTERS (k) - PER ALGORITHM")
    print("=" * 80)

    for algo in algos:
        algo_results = [r for r in results if r['algorithm'] == algo]
        if not algo_results:
            continue

        # Group by n_clusters
        k_groups = {}
        for r in algo_results:
            k = r['n_clusters']
            if k not in k_groups:
                k_groups[k] = {
                    'unacceptable': [],
                    'low': [],
                    'moderate': [],
                    'high': [],
                    'coherence': []
                }
            k_groups[k]['unacceptable'].append(r.get('coherence_n_unacceptable', 0))
            k_groups[k]['low'].append(r.get('coherence_n_low', 0))
            k_groups[k]['moderate'].append(r.get('coherence_n_moderate', 0))
            k_groups[k]['high'].append(r.get('coherence_n_high', 0))
            k_groups[k]['coherence'].append(r.get('coherence', 0))

        print(f"\n--- {algo} ---")
        print(f"{'k':>4} | {'Unacc (<.70)':>12} | {'Low (.70-.90)':>13} | {'Mod (.90-.95)':>13} | {'High (≥.95)':>11} | {'Avg Coh':>8}")
        print("-" * 70)

        for k in sorted(k_groups.keys()):
            g = k_groups[k]
            # Average across all UMAP configs with this k
            unacc = np.mean(g['unacceptable'])
            low = np.mean(g['low'])
            mod = np.mean(g['moderate'])
            high = np.mean(g['high'])
            avg_coh = np.mean(g['coherence'])
            print(f"{k:>4} | {unacc:>12.1f} | {low:>13.1f} | {mod:>13.1f} | {high:>11.1f} | {avg_coh:>8.3f}")

    print("-" * 80)
    print("Note: Values are averages across UMAP configs for each algorithm and k")


def generate_coherence_vs_k_plot(results: List[Dict], output_path: Path):
    """
    Generate plot showing coherence vs number of clusters (k).

    Features:
    - Line plot with coherence on Y-axis, k on X-axis
    - Separate lines per algorithm (HDBSCAN, Agglomerative, K-means)
    - Horizontal threshold lines at 0.70, 0.90, 0.95
    - Sweet spot detection: mark k where coherence drops below 0.90
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by algorithm
    algos = ['HDBSCAN', 'Agglomerative', 'K-means']
    colors = {'HDBSCAN': 'blue', 'Agglomerative': 'green', 'K-means': 'orange'}
    sweet_spots = {}

    for algo in algos:
        algo_results = [r for r in results if r['algorithm'] == algo]
        if not algo_results:
            continue

        # Group by k, compute mean coherence
        k_coherence = {}
        for r in algo_results:
            k = r['n_clusters']
            if k not in k_coherence:
                k_coherence[k] = []
            k_coherence[k].append(r['coherence'])

        ks = sorted(k_coherence.keys())
        coherences = [np.mean(k_coherence[k]) for k in ks]

        ax.plot(ks, coherences, 'o-', label=algo, color=colors[algo], linewidth=2, markersize=6)

        # Find sweet spot: highest k where coherence >= 0.90
        sweet_spot = None
        for k, coh in zip(ks, coherences):
            if coh >= 0.90:
                sweet_spot = (k, coh)

        if sweet_spot:
            sweet_spots[algo] = sweet_spot
            ax.scatter([sweet_spot[0]], [sweet_spot[1]],
                      color=colors[algo], s=200, marker='*', zorder=5,
                      edgecolors='black', linewidths=1.5)

    # Threshold lines
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='High (0.95)')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.90)')
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='Unacceptable (0.70)')

    # Shade regions
    ax.axhspan(0.95, 1.0, alpha=0.1, color='green')
    ax.axhspan(0.90, 0.95, alpha=0.1, color='yellow')
    ax.axhspan(0.70, 0.90, alpha=0.1, color='orange')
    ax.axhspan(0.0, 0.70, alpha=0.1, color='red')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Mean Coherence', fontsize=12)
    ax.set_title('Coherence vs Number of Clusters\n(★ = Sweet spot: highest k with coherence ≥ 0.90)', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved coherence vs k plot to {output_path}")

    # Print sweet spots
    if sweet_spots:
        print("\nSweet Spots (highest k with coherence ≥ 0.90):")
        for algo, (k, coh) in sweet_spots.items():
            print(f"  {algo}: k={k}, coherence={coh:.3f}")


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_results_to_excel(results: List[Dict], output_path: Path,
                            elbow_results: Optional[List[Dict]] = None):
    """
    Export experiment results to Excel.

    Args:
        results: List of experiment result dicts
        output_path: Path to save Excel file
        elbow_results: Optional list of knee detection results per UMAP config
    """
    df = pd.DataFrame(results)

    # Create lookup for elbow results by UMAP config
    elbow_lookup = {}
    if elbow_results:
        for r in elbow_results:
            key = (r['n_neighbors'], r['n_components'])
            elbow_lookup[key] = r

    # Add knee detection columns to each row
    if elbow_lookup:
        df['knee_K'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('knee_K'), axis=1)
        df['K_min'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('K_min'), axis=1)
        df['K_max'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('K_max'), axis=1)
        df['in_bounds'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('in_bounds'), axis=1)
        df['slope_ratio'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('slope_ratio'), axis=1)
        df['has_sharp_knee'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('has_sharp_knee'), axis=1)
        df['knee_meaningful'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('knee_meaningful'), axis=1)
        df['knee_recommendation'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('knee_recommendation'), axis=1)

    # Reorder columns for readability
    col_order = [
        'umap_neighbors', 'umap_components', 'algorithm',
        'n_clusters', 'noise_rate',
        'coherence', 'coherence_n_unacceptable', 'coherence_n_low', 'coherence_n_moderate', 'coherence_n_high', 'coherence_breakdown',
        'silhouette', 'davies_bouldin', 'dbcv',
        'knee_K', 'K_min', 'K_max', 'in_bounds', 'slope_ratio', 'has_sharp_knee', 'knee_meaningful', 'knee_recommendation',
        'params'
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Export to Excel with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['Results']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"Saved results to {output_path}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the full UMAP × Clustering comparison experiment."""

    print("=" * 70)
    print("UMAP × Clustering Algorithm Comparison Experiment")
    print("=" * 70)

    # Ensure exports directory exists
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load embeddings
    print("\n[1/5] Loading Step 4 embeddings from cache...")
    embeddings, idea_texts = load_step4_embeddings()
    n_samples = len(embeddings)

    # 2. L2 normalize
    print("\n[2/5] L2 normalizing embeddings...")
    embeddings_normalized = l2_normalize(embeddings)
    print(f"Normalized {n_samples} embeddings")

    # 3. Run UMAP grid
    print("\n[3/5] Running UMAP grid...")
    umap_configs = []

    for n_neighbors, n_components in product(UMAP_NEIGHBORS, UMAP_COMPONENTS):
        print(f"  UMAP: n_neighbors={n_neighbors}, n_components={n_components}")
        reduced = run_umap(
            embeddings_normalized,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=UMAP_MIN_DIST
        )
        umap_configs.append({
            'n_neighbors': n_neighbors,
            'n_components': n_components,
            'reduced': reduced
        })

    # 4. Generate kNN elbow plots
    print("\n[4/5] Generating kNN elbow plots...")
    elbow_plot_path = EXPORTS_DIR / "knn_elbow_plots.png"
    elbow_results = generate_knn_elbow_plots(embeddings_normalized, umap_configs, elbow_plot_path)

    # 5. Run clustering experiments
    print("\n[5/5] Running clustering experiments...")
    results = []

    # Track best result per algorithm (by coherence) for cluster inspection
    best_per_algo = {
        'HDBSCAN': {'coherence': -1, 'labels': None, 'params': None, 'breakdown': None},
        'Agglomerative': {'coherence': -1, 'labels': None, 'params': None, 'breakdown': None},
        'K-means': {'coherence': -1, 'labels': None, 'params': None, 'breakdown': None}
    }

    for config in umap_configs:
        n_neighbors = config['n_neighbors']
        n_components = config['n_components']
        reduced = config['reduced']

        # L2 normalize reduced embeddings for clustering
        reduced_normalized = l2_normalize(reduced)

        print(f"\nUMAP config: n_neighbors={n_neighbors}, n_components={n_components}")

        # HDBSCAN
        print("  Running HDBSCAN...")
        hdbscan_result = run_hdbscan_grid(reduced_normalized, n_samples)
        if hdbscan_result:
            metrics = calculate_metrics(
                hdbscan_result['labels'],
                reduced_normalized,
                embeddings_normalized
            )
            # Apply algorithm-appropriate metric filtering
            metrics = apply_algorithm_metric_filter(metrics, 'HDBSCAN')
            results.append({
                'umap_neighbors': n_neighbors,
                'umap_components': n_components,
                'algorithm': 'HDBSCAN',
                'params': str(hdbscan_result['params']),
                **metrics
            })
            # Track best HDBSCAN
            if metrics['coherence'] > best_per_algo['HDBSCAN']['coherence']:
                best_per_algo['HDBSCAN'] = {
                    'coherence': metrics['coherence'],
                    'labels': hdbscan_result['labels'].copy(),
                    'params': hdbscan_result['params'],
                    'breakdown': metrics['coherence_breakdown']
                }

        # Agglomerative (all k values)
        print("  Running Agglomerative...")
        agg_results = run_agglomerative(reduced_normalized, n_samples)
        for agg_result in agg_results:
            metrics = calculate_metrics(
                agg_result['labels'],
                reduced_normalized,
                embeddings_normalized
            )
            # Apply algorithm-appropriate metric filtering
            metrics = apply_algorithm_metric_filter(metrics, 'Agglomerative')
            results.append({
                'umap_neighbors': n_neighbors,
                'umap_components': n_components,
                'algorithm': 'Agglomerative',
                'params': str(agg_result['params']),
                **metrics
            })
            # Track best Agglomerative
            if metrics['coherence'] > best_per_algo['Agglomerative']['coherence']:
                best_per_algo['Agglomerative'] = {
                    'coherence': metrics['coherence'],
                    'labels': agg_result['labels'].copy(),
                    'params': agg_result['params'],
                    'breakdown': metrics['coherence_breakdown']
                }

        # K-means (all k values)
        print("  Running K-means...")
        kmeans_results = run_kmeans(reduced_normalized, n_samples)
        for kmeans_result in kmeans_results:
            metrics = calculate_metrics(
                kmeans_result['labels'],
                reduced_normalized,
                embeddings_normalized
            )
            # Apply algorithm-appropriate metric filtering
            metrics = apply_algorithm_metric_filter(metrics, 'K-means')
            results.append({
                'umap_neighbors': n_neighbors,
                'umap_components': n_components,
                'algorithm': 'K-means',
                'params': str(kmeans_result['params']),
                **metrics
            })
            # Track best K-means
            if metrics['coherence'] > best_per_algo['K-means']['coherence']:
                best_per_algo['K-means'] = {
                    'coherence': metrics['coherence'],
                    'labels': kmeans_result['labels'].copy(),
                    'params': kmeans_result['params'],
                    'breakdown': metrics['coherence_breakdown']
                }

    # Export results
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)

    excel_path = EXPORTS_DIR / "umap_clustering_comparison.xlsx"
    export_results_to_excel(results, excel_path, elbow_results=elbow_results)

    # Generate coherence vs k analysis
    print_coherence_by_k_table(results)

    coherence_plot_path = EXPORTS_DIR / "coherence_vs_k.png"
    generate_coherence_vs_k_plot(results, coherence_plot_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)

    # Best by coherence
    print("\nTop 5 by Coherence:")
    top_coherence = df.nlargest(5, 'coherence')[['algorithm', 'umap_neighbors', 'umap_components', 'coherence', 'n_clusters']]
    print(top_coherence.to_string(index=False))

    # Best per algorithm with coherence breakdown
    print("\nBest per Algorithm (by coherence):")
    for algo in ['HDBSCAN', 'Agglomerative', 'K-means']:
        algo_df = df[df['algorithm'] == algo]
        if not algo_df.empty:
            best = algo_df.loc[algo_df['coherence'].idxmax()]
            breakdown = best_per_algo[algo].get('breakdown', 'N/A')
            print(f"  {algo}: n={best['umap_neighbors']}, d={best['umap_components']}, "
                  f"coherence={best['coherence']:.3f}, k={best['n_clusters']}")
            print(f"    Breakdown: {breakdown}")

    print(f"\nTotal experiments: {len(results)}")
    print(f"Results saved to: {excel_path}")
    print(f"Elbow plots saved to: {elbow_plot_path}")
    print(f"Coherence vs k plot saved to: {coherence_plot_path}")

    # Print cluster samples for best result per algorithm
    print("\n" + "=" * 70)
    print("CLUSTER INSPECTION (Best per Algorithm by Coherence)")
    print("=" * 70)

    for algo in ['HDBSCAN', 'Agglomerative', 'K-means']:
        best = best_per_algo[algo]
        if best['labels'] is not None:
            print_cluster_samples(
                labels=best['labels'],
                idea_texts=idea_texts,
                algorithm=algo,
                params=best['params'],
                coherence=best['coherence'],
                original_embeddings=embeddings_normalized,
                max_samples=10
            )


if __name__ == "__main__":
    run_experiment()

# %%
