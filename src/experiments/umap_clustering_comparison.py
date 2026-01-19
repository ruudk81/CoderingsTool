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

# Mode: 'single' or 'all'
# - 'single': Run experiment on one specific dataset (configured below)
# - 'all': Auto-discover and run on ALL datasets with cached embeddings
RUN_MODE = 'single'

# Dataset configuration (for 'single' mode)
#FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
#VARIABLE = "Q20"
#SAMPLE_SIZE = 50

#FILENAME = "M000000 Associatiemonitor Merk X net databestand.sav"
#VARIABLE = "Qd1_combined"
#SAMPLE_SIZE = 2000

#FILENAME = "M000000 Associatiemonitor Merk X net databestand.sav"
#VARIABLE = "Qd1_combined"
#SAMPLE_SIZE = 2000

FILENAME = "M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
VARIABLE = "Q15"
SAMPLE_SIZE = 2000

# UMAP grid configuration
UMAP_NEIGHBORS = [5, 10, 30]
UMAP_COMPONENTS = [5, 10, 20]
UMAP_MIN_DIST = 0.1

# K-means and Agglomerative K range
K_RANGE = range(3, 16)

# Persistence threshold for algorithm selection
# Mean persistence >= this suggests stable HDBSCAN clusters
PERSISTENCE_THRESHOLD = 0.45

# Output paths
# For 'all' mode, creates subdirectories per dataset
EXPORTS_BASE_DIR = Path(__file__).parent.parent.parent / "exports" / "umap_clustering_comparison"
EXPORTS_DIR = Path(__file__).parent.parent.parent / "exports"  # Backwards compatible for single mode

# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def discover_cached_datasets(debug: bool = False) -> List[Dict]:
    """
    Query cache database to find all datasets with Step 4 (embeddings) cached.

    Args:
        debug: If True, print debug information

    Returns:
        List of dicts with:
        - filename: SPSS filename
        - variable_key: Cache variable key (e.g., "Q20_50")
        - var_name: Variable name parsed from key (e.g., "Q20")
        - sample_size: Sample size parsed from key (e.g., 50) or None if full dataset
        - n_ideas: Number of ideas in the dataset (from cache inspection)
    """
    import sqlite3
    import pickle

    # Use absolute path to project root to avoid path resolution issues
    # when running from different directories
    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "data" / "cache"
    db_path = cache_dir / "cache.db"

    if debug:
        print(f"DEBUG: db_path={db_path}, exists={db_path.exists()}")
        print(f"DEBUG: cache_dir={cache_dir}")

    datasets = []

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute('''
            SELECT DISTINCT filename, variable_key, cache_path
            FROM cache_metadata
            WHERE step_name = 'embeddings' AND status = 'valid'
            ORDER BY filename, variable_key
        ''')

        for filename, variable_key, cache_path in cursor.fetchall():
            # Parse sample_size from variable_key (e.g., "Q20_100" → 100)
            parts = variable_key.rsplit('_', 1)
            var_name = parts[0]

            # Check if last part is a number (sample size)
            if len(parts) > 1 and parts[1].isdigit():
                sample_size = int(parts[1])
            else:
                sample_size = None
                var_name = variable_key

            # Count ideas in the cache file
            n_ideas = 0
            full_cache_path = Path(cache_path)
            if not full_cache_path.is_absolute():
                full_cache_path = cache_dir / cache_path

            if full_cache_path.exists():
                try:
                    with open(full_cache_path, 'rb') as f:
                        data = pickle.load(f)
                    # Count ideas across all responses
                    for response in data:
                        if isinstance(response, dict):
                            ideas = response.get('response_ideas', [])
                        else:
                            ideas = getattr(response, 'response_ideas', []) or []
                        n_ideas += len([i for i in ideas if (i.get('idea_embedding') if isinstance(i, dict) else getattr(i, 'idea_embedding', None)) is not None])
                except Exception as e:
                    print(f"Warning: Could not count ideas in {cache_path}: {e}")

            datasets.append({
                'filename': filename,
                'variable_key': variable_key,
                'var_name': var_name,
                'sample_size': sample_size,
                'n_ideas': n_ideas
            })

    return datasets


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_step4_embeddings(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load Step 4 embeddings from cache.

    Args:
        filename: SPSS filename (defaults to global FILENAME)
        variable: Variable name (defaults to global VARIABLE)
        sample_size: Sample size (defaults to global SAMPLE_SIZE)
        variable_key: Pre-computed variable key (overrides variable/sample_size)

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        idea_texts: list of idea text strings
    """
    import pickle

    # Use global defaults if not provided
    _filename = filename if filename is not None else FILENAME
    _variable = variable if variable is not None else VARIABLE
    _sample_size = sample_size if sample_size is not None else SAMPLE_SIZE

    # Generate variable key if not provided
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[_variable],
            is_merged=False,
            sample_size=_sample_size
        )

    # Build cache file path directly (bypassing database lookup)
    # Cache is in project_root/data/cache
    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "data" / "cache"
    base_name = Path(_filename).stem
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
        raise ValueError(f"No cached embeddings found for {_filename}/{variable_key}. Run pipeline Step 4 first.")

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

def extract_persistence_metrics(clusterer: hdbscan.HDBSCAN, labels: np.ndarray) -> Dict:
    """
    Extract cluster persistence metrics from fitted HDBSCAN model.

    Persistence measures how stable clusters are across different density thresholds
    in the HDBSCAN dendrogram. Higher persistence = more stable/robust clusters.

    Args:
        clusterer: Fitted HDBSCAN model
        labels: Cluster labels from the model

    Returns:
        Dict with:
        - persistence_values: List of per-cluster persistence scores
        - mean_persistence: Mean persistence across all clusters
        - min_persistence: Minimum persistence value
        - max_persistence: Maximum persistence value
        - std_persistence: Standard deviation of persistence values
        - weighted_persistence: Size-weighted persistence (larger clusters weighted more)
    """
    # Try both attribute names (depends on HDBSCAN version)
    persistence = getattr(clusterer, "cluster_persistence_", None)
    if persistence is None:
        persistence = getattr(clusterer, "cluster_stability_", None)

    # Handle missing or empty persistence data
    if persistence is None or len(persistence) == 0:
        return {
            'persistence_values': [],
            'mean_persistence': np.nan,
            'min_persistence': np.nan,
            'max_persistence': np.nan,
            'std_persistence': np.nan,
            'weighted_persistence': np.nan
        }

    persistence = np.array(persistence)

    # Calculate basic statistics
    metrics = {
        'persistence_values': persistence.tolist(),
        'mean_persistence': float(np.mean(persistence)),
        'min_persistence': float(np.min(persistence)),
        'max_persistence': float(np.max(persistence)),
        'std_persistence': float(np.std(persistence)) if len(persistence) > 1 else 0.0,
    }

    # Calculate size-weighted persistence (same formula as clusterer.py _cluster_stability)
    # Formula: (1/N_non_noise) * sum(persistence[c] * |cluster_c|)
    mask = labels >= 0
    if np.any(mask):
        labels_non_noise = labels[mask]
        n = labels_non_noise.size
        max_lab = int(labels_non_noise.max())
        counts = np.bincount(labels_non_noise, minlength=max_lab + 1).astype(float)
        k = min(len(persistence), len(counts))
        if k > 0 and n > 0:
            weighted = float(np.dot(persistence[:k], counts[:k]) / n)
        else:
            weighted = np.nan
    else:
        weighted = np.nan

    metrics['weighted_persistence'] = weighted

    return metrics


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
        dict with best labels, params, metrics, AND persistence metrics
    """
    sqrt_n = np.sqrt(n_samples)
    mcs_grid = [
        max(2, int(sqrt_n)),           # sqrt(n)
        max(2, int(0.5 * sqrt_n)),     # 0.5 * sqrt(n)
        max(2, int(0.25 * sqrt_n))     # 0.25 * sqrt(n)
    ]

    best_result = None
    best_dbcv = -2.0
    best_clusterer = None  # Store the fitted model for persistence extraction

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

        # Extract persistence for logging
        persistence_metrics = extract_persistence_metrics(clusterer, labels)
        mean_pers = persistence_metrics.get('mean_persistence', np.nan)
        weighted_pers = persistence_metrics.get('weighted_persistence', np.nan)
        pers_str = ""
        if not np.isnan(mean_pers):
            pers_str = f", pers(mean={mean_pers:.3f}"
            if not np.isnan(weighted_pers):
                pers_str += f", wgt={weighted_pers:.3f})"
            else:
                pers_str += ")"

        print(f"    HDBSCAN mcs={mcs}, ms={ms}: k={n_clusters}, noise={noise_rate:.2%}, DBCV={dbcv:.3f}{pers_str}")

        if dbcv > best_dbcv:
            best_dbcv = dbcv
            best_clusterer = clusterer  # Keep reference to best model
            best_result = {
                'labels': labels,
                'n_clusters': n_clusters,
                'noise_rate': noise_rate,
                'dbcv': dbcv,
                'params': {'min_cluster_size': mcs, 'min_samples': ms}
            }

    # Extract persistence metrics from best clusterer
    if best_result and best_clusterer is not None:
        persistence_metrics = extract_persistence_metrics(best_clusterer, best_result['labels'])
        best_result.update({
            'mean_persistence': persistence_metrics['mean_persistence'],
            'min_persistence': persistence_metrics['min_persistence'],
            'max_persistence': persistence_metrics['max_persistence'],
            'std_persistence': persistence_metrics['std_persistence'],
            'weighted_persistence': persistence_metrics['weighted_persistence'],
        })

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


def choose_cluster_strategy_via_knee(
    X: np.ndarray,
    knn_k: int = 5,
    metric: str = "euclidean",
    y_diff_threshold: float = 0.6
) -> Dict:
    """
    Detect knee in kNN distance curve and recommend clustering strategy.

    ADAPTIVE PARAMETERS:
    - S (sensitivity): Scales with dataset size (S = n/100) to maintain consistent
      detection behavior across different dataset sizes. S=1 for n≤100, S=20 for n=2000.
    - interp_method: Uses 'polynomial' for small datasets (n<200) to smooth noisy curves,
      'interp1d' for larger datasets.

    KNEE SHARPNESS TEST (y_difference):
    - Uses KneeLocator's built-in y_difference metric (perpendicular distance to diagonal)
    - max(y_difference) >= 0.6 indicates a sharp knee with clear density transition
    - Values ~0.9 indicate very sharp knees, ~0.5-0.6 indicate gradual curves
    - This is more robust than slope ratio as it doesn't depend on window size

    Only recommend HDBSCAN if knee exists AND y_difference >= threshold.

    Args:
        X: L2-normalized embeddings (n_samples, n_features)
        knn_k: Which neighbor distance to use (default 5)
        metric: Distance metric for kNN (default "euclidean")
        y_diff_threshold: Minimum y_difference for sharp knee (default 0.6)

    Returns:
        Dict with:
        - recommendation: "HDBSCAN" | "AGGLOMERATIVE_OR_KMEANS"
        - K: Knee point index (or None if not found)
        - is_meaningful: Whether knee passes y_difference test
        - has_sharp_knee: Whether y_difference >= threshold
        - y_difference: Max normalized perpendicular distance (knee sharpness)
        - knee_distance: kNN distance at knee point
        - distances: Sorted kNN distances array
        - kneedle_S: Adaptive S parameter used
        - interp_method: Interpolation method used
    """
    n = len(X)
    sqrt_n = np.sqrt(n)

    # Adaptive parameters based on dataset size
    # S scales with n to maintain consistent detection behavior
    # S=1 for n≤100, S=5 for n=500, S=20 for n=2000
    kneedle_S = max(1.0, n / 100)

    # Use polynomial smoothing for small/noisy datasets
    interp_method = "polynomial" if n < 200 else "interp1d"

    # 1. Compute kNN distances
    nn = NearestNeighbors(n_neighbors=knn_k + 1, metric=metric)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = distances[:, knn_k]

    # 2. Sort for elbow analysis
    sorted_distances = np.sort(k_distances)

    # 3. Define search window: full range [1, n-1]
    # Skip first point (always 0 or very small), include everything else
    start_idx = 1
    end_idx = n - 1

    # Extract search segment for knee detection
    search_distances = sorted_distances[start_idx:end_idx]
    search_x = np.arange(len(search_distances))

    # 4. Detect knee on search window with adaptive parameters
    kneedle = KneeLocator(
        x=search_x,
        y=search_distances,
        S=kneedle_S,
        curve="convex",
        direction="increasing",
        interp_method=interp_method
    )

    # 5. Map knee back to original coordinate system
    K_in_window = kneedle.knee  # Index within search window
    if K_in_window is not None:
        K = start_idx + K_in_window  # Map back to original index
    else:
        K = None

    # 6. Compute y_difference to validate knee sharpness
    # y_difference is the normalized perpendicular distance from each point to the diagonal
    # max(y_difference) is a robust measure of knee sharpness that doesn't depend on window size
    if K is not None and kneedle.y_difference is not None and len(kneedle.y_difference) > 0:
        y_difference = float(max(kneedle.y_difference))
        has_sharp_knee = y_difference >= y_diff_threshold
    else:
        y_difference = 0.0
        has_sharp_knee = False

    # 7. Determine if knee is meaningful for HDBSCAN
    # Only requirement: knee must be sharp enough (y_difference >= threshold)
    is_meaningful = has_sharp_knee

    # 8. Get knee distance
    knee_distance = sorted_distances[K] if K is not None else None

    return {
        "recommendation": "HDBSCAN" if is_meaningful else "AGGLOMERATIVE_OR_KMEANS",
        "K": K,
        "is_meaningful": is_meaningful,
        "has_sharp_knee": has_sharp_knee,
        "y_difference": y_difference,
        "knee_distance": knee_distance,
        "distances": sorted_distances,
        "n": n,
        "sqrt_n": sqrt_n,
        "kneedle_S": kneedle_S,
        "interp_method": interp_method
    }


def choose_cluster_strategy_combined(
    knee_result: Dict,
    persistence_metrics: Dict,
    persistence_threshold: float = PERSISTENCE_THRESHOLD,
    y_diff_threshold: float = 0.6
) -> Dict:
    """
    Combined algorithm selection using BOTH knee detection AND persistence metrics.

    Decision Logic (4-quadrant matrix):
    ┌─────────────────┬─────────────────────────┬─────────────────────────┐
    │                 │ High Persistence (≥0.45)│ Low Persistence (<0.45) │
    ├─────────────────┼─────────────────────────┼─────────────────────────┤
    │ Sharp Knee      │ HDBSCAN_STRONG          │ HDBSCAN_WEAK            │
    │ (ydiff ≥ 0.6)   │ (high confidence)       │ (medium confidence)     │
    ├─────────────────┼─────────────────────────┼─────────────────────────┤
    │ Flat Knee       │ HDBSCAN_WEAK            │ AGGLOMERATIVE_OR_KMEANS │
    │ (ydiff < 0.6)   │ (medium confidence)     │ (high confidence)       │
    └─────────────────┴─────────────────────────┴─────────────────────────┘

    Args:
        knee_result: Dict from choose_cluster_strategy_via_knee()
        persistence_metrics: Dict with mean_persistence, etc. (from extract_persistence_metrics)
        persistence_threshold: Minimum mean_persistence for stable clusters (default from config)
        y_diff_threshold: Minimum y_difference for sharp knee (default 0.6)

    Returns:
        Dict with:
        - recommendation: "HDBSCAN_STRONG" | "HDBSCAN_WEAK" | "AGGLOMERATIVE_OR_KMEANS"
        - has_sharp_knee: bool
        - has_high_persistence: bool (or None if data unavailable)
        - y_difference: float
        - mean_persistence: float
        - confidence: "high" | "medium" | "low"
        - reasoning: str explaining the decision
    """
    has_sharp_knee = knee_result.get('has_sharp_knee', False)
    y_difference = knee_result.get('y_difference', 0.0)
    mean_persistence = persistence_metrics.get('mean_persistence', np.nan)

    # Handle missing persistence data - fall back to knee-only decision
    if np.isnan(mean_persistence):
        return {
            'recommendation': "HDBSCAN" if has_sharp_knee else "AGGLOMERATIVE_OR_KMEANS",
            'has_sharp_knee': has_sharp_knee,
            'has_high_persistence': None,
            'y_difference': y_difference,
            'mean_persistence': np.nan,
            'confidence': 'low',
            'reasoning': f'Persistence data unavailable, using knee detection only (ydiff={y_difference:.2f})'
        }

    has_high_persistence = mean_persistence >= persistence_threshold

    # 4-quadrant decision matrix
    if has_sharp_knee and has_high_persistence:
        # Best case: both signals agree HDBSCAN is appropriate
        recommendation = "HDBSCAN_STRONG"
        confidence = "high"
        reasoning = f"Sharp knee (ydiff={y_difference:.2f}) + stable clusters (persistence={mean_persistence:.2f}≥{persistence_threshold})"
    elif has_sharp_knee and not has_high_persistence:
        # Sharp knee but clusters not very stable - proceed with caution
        recommendation = "HDBSCAN_WEAK"
        confidence = "medium"
        reasoning = f"Sharp knee (ydiff={y_difference:.2f}) but weak cluster stability (persistence={mean_persistence:.2f}<{persistence_threshold})"
    elif not has_sharp_knee and has_high_persistence:
        # No clear knee but clusters are stable - density structure may exist
        recommendation = "HDBSCAN_WEAK"
        confidence = "medium"
        reasoning = f"Flat knee (ydiff={y_difference:.2f}) but high persistence ({mean_persistence:.2f}≥{persistence_threshold}) suggests density structure"
    else:
        # Both signals agree: no density structure
        recommendation = "AGGLOMERATIVE_OR_KMEANS"
        confidence = "high"
        reasoning = f"Flat knee (ydiff={y_difference:.2f}) + low persistence ({mean_persistence:.2f}<{persistence_threshold}) - no clear density structure"

    return {
        'recommendation': recommendation,
        'has_sharp_knee': has_sharp_knee,
        'has_high_persistence': has_high_persistence,
        'y_difference': y_difference,
        'mean_persistence': mean_persistence,
        'confidence': confidence,
        'reasoning': reasoning
    }


def generate_knn_elbow_plots(embeddings: np.ndarray, umap_configs: List[Dict],
                             output_path: Path, k: int = 5) -> List[Dict]:
    """
    Generate kNN distance elbow plots with adaptive knee detection
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

        # Use adaptive knee detection
        knee_result = choose_cluster_strategy_via_knee(reduced, knn_k=k)

        # Store results with adaptive params
        elbow_results.append({
            'n_neighbors': config['n_neighbors'],
            'n_components': config['n_components'],
            'knee_K': knee_result['K'],
            'has_sharp_knee': knee_result['has_sharp_knee'],
            'y_difference': knee_result['y_difference'],
            'knee_meaningful': knee_result['is_meaningful'],
            'knee_recommendation': knee_result['recommendation'],
            'knee_distance': knee_result['knee_distance'],
            'kneedle_S': knee_result['kneedle_S'],
            'interp_method': knee_result['interp_method']
        })

        # Plot the kNN distances
        sorted_distances = knee_result['distances']
        ax.plot(sorted_distances, linewidth=1.5, color='blue', label='k-NN distances')

        # Mark knee point if found and sharp
        if knee_result['K'] is not None and knee_result['has_sharp_knee']:
            ax.scatter([knee_result['K']], [knee_result['knee_distance']],
                       color='green', s=100, zorder=5, marker='o',
                       label=f"Knee @ {knee_result['K']}")
            ax.axhline(y=knee_result['knee_distance'], color='green', linestyle='--', alpha=0.3)
            ax.axvline(x=knee_result['K'], color='green', linestyle='--', alpha=0.3)
        elif knee_result['K'] is not None:
            # Knee found but not sharp enough - show in orange
            ax.scatter([knee_result['K']], [knee_result['knee_distance']],
                       color='orange', s=80, zorder=5, marker='x',
                       label=f"Knee @ {knee_result['K']} (flat)")

        # Simplified title
        knee_str = f"K={knee_result['K']}" if knee_result['K'] is not None else "No knee"
        ydiff_str = f"ydiff={knee_result['y_difference']:.2f}"
        short_rec = "HDBSCAN" if knee_result['recommendation'] == "HDBSCAN" else "Agg/KM"
        ax.set_title(f"n={config['n_neighbors']}, d={config['n_components']} | "
                     f"{knee_str} ({ydiff_str}) → {short_rec}")
        ax.set_xlabel('Points (sorted)')
        ax.set_ylabel(f'{k}-NN distance')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('kNN Distance Elbow Plots with Adaptive Knee Detection', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved kNN elbow plots to {output_path}")

    # Print elbow analysis summary with adaptive params
    if elbow_results:
        first_result = elbow_results[0]
        print(f"\nKnee Detection Analysis (adaptive: S={first_result['kneedle_S']:.1f}, interp={first_result['interp_method']}):")
        print("  Legend: y_difference ≥0.6 = sharp knee → HDBSCAN recommended")
    for r in elbow_results:
        knee_str = f"K={r['knee_K']:3d}" if r['knee_K'] is not None else "K=None"
        ydiff_str = f"ydiff={r['y_difference']:.2f}"
        sharp_status = "sharp" if r['has_sharp_knee'] else "flat"
        print(f"  UMAP n={r['n_neighbors']:2d}, d={r['n_components']:2d}: "
              f"{knee_str} | {ydiff_str} ({sharp_status}) → {r['knee_recommendation']}")

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
        df['y_difference'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('y_difference'), axis=1)
        df['has_sharp_knee'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('has_sharp_knee'), axis=1)
        df['knee_meaningful'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('knee_meaningful'), axis=1)
        df['knee_recommendation'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('knee_recommendation'), axis=1)
        df['kneedle_S'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('kneedle_S'), axis=1)
        df['interp_method'] = df.apply(lambda row: elbow_lookup.get(
            (row['umap_neighbors'], row['umap_components']), {}).get('interp_method'), axis=1)

    # Add combined recommendation columns for HDBSCAN rows
    # Uses both knee detection and persistence metrics
    def get_combined_recommendation(row):
        if row['algorithm'] != 'HDBSCAN':
            return {'recommendation': np.nan, 'confidence': np.nan, 'reasoning': np.nan}

        # Get knee result for this UMAP config
        elbow = elbow_lookup.get((row['umap_neighbors'], row['umap_components']), {})
        knee_result = {
            'has_sharp_knee': elbow.get('has_sharp_knee', False),
            'y_difference': elbow.get('y_difference', 0.0)
        }

        # Get persistence metrics from the row
        persistence_metrics = {
            'mean_persistence': row.get('mean_persistence', np.nan)
        }

        # Get combined recommendation
        combined = choose_cluster_strategy_combined(knee_result, persistence_metrics)
        return combined

    if elbow_lookup:
        combined_results = df.apply(get_combined_recommendation, axis=1)
        df['combined_recommendation'] = combined_results.apply(lambda x: x.get('recommendation', np.nan))
        df['combined_confidence'] = combined_results.apply(lambda x: x.get('confidence', np.nan))
        df['combined_reasoning'] = combined_results.apply(lambda x: x.get('reasoning', np.nan))

    # Reorder columns for readability
    col_order = [
        'umap_neighbors', 'umap_components', 'algorithm',
        'n_clusters', 'noise_rate',
        'coherence', 'coherence_n_unacceptable', 'coherence_n_low', 'coherence_n_moderate', 'coherence_n_high', 'coherence_breakdown',
        'silhouette', 'davies_bouldin', 'dbcv',
        # Persistence metrics (HDBSCAN only)
        'mean_persistence', 'min_persistence', 'max_persistence', 'std_persistence', 'weighted_persistence',
        # Knee detection
        'knee_K', 'y_difference', 'has_sharp_knee', 'knee_meaningful', 'knee_recommendation', 'kneedle_S', 'interp_method',
        # Combined recommendation (knee + persistence)
        'combined_recommendation', 'combined_confidence', 'combined_reasoning',
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

def run_experiment(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run the full UMAP × Clustering comparison experiment on a single dataset.

    Args:
        filename: SPSS filename (defaults to global FILENAME)
        variable: Variable name (defaults to global VARIABLE)
        sample_size: Sample size (defaults to global SAMPLE_SIZE)
        variable_key: Pre-computed variable key (overrides variable/sample_size)
        output_dir: Directory for output files (defaults to global EXPORTS_DIR)

    Returns:
        Dict with experiment summary for cross-dataset comparison:
        - dataset: variable_key used
        - n_ideas: Number of ideas processed
        - n_experiments: Total experiments run
        - best_algorithm: Algorithm with highest coherence
        - best_coherence: Highest coherence achieved
        - best_k: Number of clusters for best result
        - knee_meaningful_pct: % of UMAP configs with meaningful knee
        - best_per_algo: Dict with best result per algorithm
    """
    # Use defaults if not provided
    _filename = filename if filename is not None else FILENAME
    _variable = variable if variable is not None else VARIABLE
    _sample_size = sample_size if sample_size is not None else SAMPLE_SIZE
    _variable_key = variable_key

    # Generate variable key if not provided
    if _variable_key is None:
        _variable_key = generate_enhanced_variable_key(
            selected_variables=[_variable],
            is_merged=False,
            sample_size=_sample_size
        )

    # Set output directory (after variable_key is resolved)
    _output_dir = output_dir if output_dir is not None else EXPORTS_BASE_DIR / _variable_key

    print("=" * 70)
    print(f"UMAP × Clustering Experiment: {_variable_key}")
    print("=" * 70)

    # Ensure output directory exists
    _output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load embeddings
    print("\n[1/5] Loading Step 4 embeddings from cache...")
    embeddings, idea_texts = load_step4_embeddings(
        filename=_filename,
        variable=_variable,
        sample_size=_sample_size,
        variable_key=_variable_key
    )
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
    elbow_plot_path = _output_dir / "knn_elbow_plots.png"
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

            # Add persistence metrics (from run_hdbscan_grid)
            results.append({
                'umap_neighbors': n_neighbors,
                'umap_components': n_components,
                'algorithm': 'HDBSCAN',
                'params': str(hdbscan_result['params']),
                'mean_persistence': hdbscan_result.get('mean_persistence', np.nan),
                'min_persistence': hdbscan_result.get('min_persistence', np.nan),
                'max_persistence': hdbscan_result.get('max_persistence', np.nan),
                'std_persistence': hdbscan_result.get('std_persistence', np.nan),
                'weighted_persistence': hdbscan_result.get('weighted_persistence', np.nan),
                **metrics
            })
            # Track best HDBSCAN (include persistence)
            if metrics['coherence'] > best_per_algo['HDBSCAN']['coherence']:
                best_per_algo['HDBSCAN'] = {
                    'coherence': metrics['coherence'],
                    'labels': hdbscan_result['labels'].copy(),
                    'params': hdbscan_result['params'],
                    'breakdown': metrics['coherence_breakdown'],
                    'mean_persistence': hdbscan_result.get('mean_persistence', np.nan)
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

    excel_path = _output_dir / "umap_clustering_comparison.xlsx"
    export_results_to_excel(results, excel_path, elbow_results=elbow_results)

    # Generate coherence vs k analysis
    print_coherence_by_k_table(results)

    coherence_plot_path = _output_dir / "coherence_vs_k.png"
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
            persistence_str = ""
            if algo == 'HDBSCAN':
                mean_pers = best_per_algo[algo].get('mean_persistence', np.nan)
                if not np.isnan(mean_pers):
                    persistence_str = f", persistence={mean_pers:.3f}"
            print(f"  {algo}: n={best['umap_neighbors']}, d={best['umap_components']}, "
                  f"coherence={best['coherence']:.3f}, k={best['n_clusters']}{persistence_str}")
            print(f"    Breakdown: {breakdown}")

    # Persistence analysis summary (HDBSCAN only)
    hdbscan_df = df[df['algorithm'] == 'HDBSCAN']
    if not hdbscan_df.empty and 'mean_persistence' in hdbscan_df.columns:
        persistence_values = hdbscan_df['mean_persistence'].dropna()
        if len(persistence_values) > 0:
            mean_persistence_overall = persistence_values.mean()
            persistence_stable_count = (persistence_values >= PERSISTENCE_THRESHOLD).sum()
            persistence_stable_pct = persistence_stable_count / len(persistence_values) * 100

            print(f"\nHDBSCAN Persistence Analysis:")
            print(f"  Mean persistence across all configs: {mean_persistence_overall:.3f}")
            print(f"  Configs with stable clusters (≥{PERSISTENCE_THRESHOLD}): "
                  f"{persistence_stable_count}/{len(persistence_values)} ({persistence_stable_pct:.0f}%)")

            # Combined recommendation summary
            if 'combined_recommendation' in df.columns:
                hdbscan_recs = df[df['algorithm'] == 'HDBSCAN']['combined_recommendation'].dropna()
                if len(hdbscan_recs) > 0:
                    rec_counts = hdbscan_recs.value_counts()
                    print(f"\nCombined Recommendations (knee + persistence):")
                    for rec, count in rec_counts.items():
                        pct = count / len(hdbscan_recs) * 100
                        print(f"  {rec}: {count}/{len(hdbscan_recs)} ({pct:.0f}%)")

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

    # Compute summary for cross-dataset comparison
    # Find overall best
    best_row = df.loc[df['coherence'].idxmax()]
    best_algorithm = best_row['algorithm']
    best_coherence = best_row['coherence']
    best_k = best_row['n_clusters']

    # Compute % of UMAP configs with meaningful knee
    n_meaningful = sum(1 for r in elbow_results if r.get('knee_meaningful', False))
    knee_meaningful_pct = n_meaningful / len(elbow_results) * 100 if elbow_results else 0

    # Compute HDBSCAN persistence summary
    hdbscan_persistence_values = df[df['algorithm'] == 'HDBSCAN']['mean_persistence'].dropna()
    if len(hdbscan_persistence_values) > 0:
        hdbscan_mean_persistence = float(hdbscan_persistence_values.mean())
        hdbscan_persistence_stable_pct = float((hdbscan_persistence_values >= PERSISTENCE_THRESHOLD).sum() / len(hdbscan_persistence_values) * 100)
        # Noise rate coefficient of variation (stability measure)
        noise_rates = df[df['algorithm'] == 'HDBSCAN']['noise_rate'].dropna()
        if len(noise_rates) > 0 and noise_rates.mean() > 0:
            noise_rate_cv = float(noise_rates.std() / noise_rates.mean())
        else:
            noise_rate_cv = np.nan
    else:
        hdbscan_mean_persistence = np.nan
        hdbscan_persistence_stable_pct = np.nan
        noise_rate_cv = np.nan

    return {
        'dataset': _variable_key,
        'filename': _filename,
        'n_ideas': n_samples,
        'n_experiments': len(results),
        'best_algorithm': best_algorithm,
        'best_coherence': best_coherence,
        'best_k': best_k,
        'knee_meaningful_pct': knee_meaningful_pct,
        # Persistence summary
        'hdbscan_mean_persistence': hdbscan_mean_persistence,
        'hdbscan_persistence_stable_pct': hdbscan_persistence_stable_pct,
        'noise_rate_cv': noise_rate_cv,
        'best_per_algo': {
            algo: {
                'coherence': best_per_algo[algo]['coherence'],
                'params': best_per_algo[algo]['params'],
                'breakdown': best_per_algo[algo]['breakdown'],
                'mean_persistence': best_per_algo[algo].get('mean_persistence', np.nan) if algo == 'HDBSCAN' else np.nan
            }
            for algo in ['HDBSCAN', 'Agglomerative', 'K-means']
            if best_per_algo[algo]['labels'] is not None
        }
    }


def run_all_cached_datasets() -> List[Dict]:
    """
    Discover and run experiment on all datasets with cached embeddings.

    Returns:
        List of summary dicts, one per dataset
    """
    datasets = discover_cached_datasets()

    if not datasets:
        print("No datasets with cached embeddings found.")
        return []

    print("=" * 70)
    print("MULTI-DATASET EXPERIMENT")
    print("=" * 70)
    print(f"\nFound {len(datasets)} datasets with cached embeddings:")
    for ds in datasets:
        size_str = f"n={ds['sample_size']}" if ds['sample_size'] else "full"
        print(f"  - {ds['filename']}: {ds['variable_key']} ({size_str}, {ds['n_ideas']} ideas)")

    all_results = []

    for i, ds in enumerate(datasets, 1):
        print(f"\n{'#'*70}")
        print(f"# Dataset {i}/{len(datasets)}: {ds['variable_key']}")
        print(f"{'#'*70}")

        # Create per-dataset output directory
        output_dir = EXPORTS_BASE_DIR / ds['variable_key']

        try:
            summary = run_experiment(
                filename=ds['filename'],
                variable=ds['var_name'],
                sample_size=ds['sample_size'],
                variable_key=ds['variable_key'],
                output_dir=output_dir
            )
            all_results.append(summary)
        except Exception as e:
            print(f"ERROR: Failed to process {ds['variable_key']}: {e}")
            all_results.append({
                'dataset': ds['variable_key'],
                'filename': ds['filename'],
                'n_ideas': ds['n_ideas'],
                'error': str(e)
            })

    # Generate cross-dataset comparison report
    if all_results:
        generate_comparison_report(all_results, EXPORTS_BASE_DIR / "summary_comparison.xlsx")

    return all_results


def generate_comparison_report(all_results: List[Dict], output_path: Path):
    """
    Generate summary comparing knee detection and coherence across datasets.

    Args:
        all_results: List of summary dicts from run_experiment()
        output_path: Path to save Excel report
    """
    print("\n" + "=" * 70)
    print("CROSS-DATASET COMPARISON REPORT")
    print("=" * 70)

    # Build comparison DataFrame
    rows = []
    for r in all_results:
        if 'error' in r:
            rows.append({
                'dataset': r['dataset'],
                'filename': r['filename'],
                'n_ideas': r.get('n_ideas', 0),
                'error': r['error']
            })
            continue

        row = {
            'dataset': r['dataset'],
            'filename': r['filename'],
            'n_ideas': r['n_ideas'],
            'sqrt_n': np.sqrt(r['n_ideas']),
            'n_experiments': r['n_experiments'],
            'best_algorithm': r['best_algorithm'],
            'best_coherence': r['best_coherence'],
            'best_k': r['best_k'],
            'knee_meaningful_pct': r['knee_meaningful_pct'],
            # Persistence metrics
            'hdbscan_mean_persistence': r.get('hdbscan_mean_persistence', np.nan),
            'hdbscan_persistence_stable_pct': r.get('hdbscan_persistence_stable_pct', np.nan),
            'noise_rate_cv': r.get('noise_rate_cv', np.nan),
        }

        # Add best per algorithm
        for algo in ['HDBSCAN', 'Agglomerative', 'K-means']:
            if algo in r.get('best_per_algo', {}):
                row[f'{algo}_coherence'] = r['best_per_algo'][algo]['coherence']
                if algo == 'HDBSCAN':
                    row[f'{algo}_persistence'] = r['best_per_algo'][algo].get('mean_persistence', np.nan)
            else:
                row[f'{algo}_coherence'] = np.nan
                if algo == 'HDBSCAN':
                    row[f'{algo}_persistence'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # Print summary to console
    print("\nDataset Summary:")
    print("-" * 80)
    for _, row in df.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
            print(f"  {row['dataset']}: ERROR - {row['error']}")
        else:
            persistence_str = ""
            if pd.notna(row.get('hdbscan_mean_persistence')):
                persistence_str = f", persistence={row['hdbscan_mean_persistence']:.2f}"
            print(f"  {row['dataset']}: n={row['n_ideas']}, best={row['best_algorithm']} "
                  f"(coh={row['best_coherence']:.3f}, k={row['best_k']}), "
                  f"knee_meaningful={row['knee_meaningful_pct']:.0f}%{persistence_str}")

    # Export to Excel
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Comparison', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['Comparison']
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

    print(f"\nComparison report saved to: {output_path}")


if __name__ == "__main__":
    if RUN_MODE == 'all':
        run_all_cached_datasets()
    else:
        run_experiment()

# %%
