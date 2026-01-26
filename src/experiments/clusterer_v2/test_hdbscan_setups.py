#%%

"""
HDBSCAN Setup Comparison Test

Compare 2 clustering approaches on the same dataset:
1. UMAP + HDBSCAN (euclidean)
2. PaCMAP + HDBSCAN (euclidean)

Usage:
    cd src/experiments/clusterer_v2
    python test_hdbscan_setups.py
"""

import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import pickle

import numpy as np
import hdbscan
import umap
import pacmap
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import models
from utils.cacheManager import generate_enhanced_variable_key

# Suppress warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

FILENAME = "M000000 Associatiemonitor Merk X net databestand.sav"
VARIABLE = "Qd1_combined"
SAMPLE_SIZE = 2000


# =============================================================================
# HDBSCAN PARAMETERS (fixed for all setups)
# =============================================================================

MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 2

# UMAP parameters
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0

# PaCMAP parameters
PACMAP_N_COMPONENTS = 10
PACMAP_N_NEIGHBORS = 10

# Low probability threshold
LOW_PROB_THRESHOLD = 0.3


# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE
) -> Tuple[np.ndarray, List[str]]:
    """
    Load Step 4 embeddings from cache.

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        idea_texts: list of idea text strings
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size
    )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(filename).stem
    cache_filename = f"005_embeddings_{base_name}_{variable_key}.pkl"
    cache_path = cache_dir / cache_filename

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run pipeline step 4 first to generate embeddings."
        )

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    embeddings_models = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

    embeddings_list = []
    idea_texts = []

    for response in embeddings_models:
        if response.response_ideas:
            for idea in response.response_ideas:
                if idea.idea_embedding is not None:
                    embeddings_list.append(idea.idea_embedding)
                    idea_texts.append(idea.idea)

    if not embeddings_list:
        raise ValueError("No embeddings found in cached data")

    embeddings = np.vstack(embeddings_list)
    return embeddings, idea_texts


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    return normalize(embeddings, norm='l2', axis=1)


def strip_context_specifiers(text: str) -> str:
    """
    Strip context specifier tags and template from idea text.

    Removes patterns like:
    - [lang=nl-NL]
    - [domain=financien]
    - [sentiment=positive]
    - [sense=evaluative]
    - "Merk X heeft de associatie "

    Returns just the core idea text.
    """
    import re

    # Remove all bracketed context specifiers
    text = re.sub(r'\[[^\]]+\]', '', text)

    # Remove common template phrases
    templates = [
        "Merk X heeft de associatie ",
        "Merk X heeft de associatie",
    ]
    for template in templates:
        if template in text:
            text = text.replace(template, '')

    # Clean up whitespace
    text = text.strip()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text


def detect_template_prefix(texts: List[str]) -> str:
    """
    Detect common template prefix from idea texts.

    Returns the longest common prefix that ends with a newline.
    This handles templates like:
    "[context tags]\nMerk X heeft de associatie "
    """
    if not texts or len(texts) < 2:
        return ""

    # Find shortest text to limit prefix search
    min_len = min(len(t) for t in texts)
    if min_len == 0:
        return ""

    # Find common prefix
    prefix = ""
    for i in range(min_len):
        char = texts[0][i]
        if all(t[i] == char for t in texts):
            prefix += char
        else:
            break

    # For this dataset, the template includes everything up to and including
    # "Merk X heeft de associatie " - find the last occurrence of this pattern
    marker = "Merk X heeft de associatie "
    marker_pos = prefix.find(marker)
    if marker_pos >= 0:
        return prefix[:marker_pos + len(marker)]

    # Fallback: trim to last newline for clean break
    last_newline = prefix.rfind('\n')
    if last_newline > 0:
        return prefix[:last_newline + 1]

    return ""


def compute_knn_pairs(embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
    """
    Compute k-nearest neighbor pairs using sklearn (bypasses Annoy bug).

    This solves the PaCMAP bug on macOS + Python 3.12+ where Annoy fails silently.
    See: https://github.com/YingfanWang/PaCMAP/issues/94

    Args:
        embeddings: (N, D) array of embeddings
        n_neighbors: number of neighbors to find

    Returns:
        pair_neighbors: (N * n_neighbors, 2) array of (source_idx, neighbor_idx) pairs
    """
    N = len(embeddings)

    # Add 1 because sklearn includes the point itself as the first neighbor
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', algorithm='auto')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Remove self (first column) to get actual neighbors
    neighbor_indices = indices[:, 1:]  # Shape: (N, n_neighbors)

    # Convert to PaCMAP's expected format: (N * n_neighbors, 2) pairs
    pair_neighbors = np.zeros((N * n_neighbors, 2), dtype=np.int32)

    for i in range(N):
        for j in range(n_neighbors):
            pair_neighbors[i * n_neighbors + j, 0] = i
            pair_neighbors[i * n_neighbors + j, 1] = neighbor_indices[i, j]

    return pair_neighbors


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

def run_umap(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    random_state: int = 42
) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=random_state,
        verbose=False
    )
    return reducer.fit_transform(embeddings)


def run_pacmap(
    embeddings: np.ndarray,
    n_components: int = PACMAP_N_COMPONENTS,
    n_neighbors: int = PACMAP_N_NEIGHBORS,
    random_state: int = 42
) -> np.ndarray:
    """
    Run PaCMAP dimensionality reduction with sklearn kNN workaround.
    """
    N = len(embeddings)
    MN_ratio = 0.5
    FP_ratio = 2.0

    # Check if we need to reduce n_neighbors
    total_neighbors_needed = int(n_neighbors * (1 + MN_ratio + FP_ratio)) + 1
    if total_neighbors_needed >= N:
        max_n_neighbors = int(N / (1 + MN_ratio + FP_ratio)) - 1
        n_neighbors = max(5, max_n_neighbors)

    # Compute nearest neighbors with sklearn (bypasses Annoy bug)
    pair_neighbors = compute_knn_pairs(embeddings, n_neighbors)

    reducer = pacmap.PaCMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        pair_neighbors=pair_neighbors,
        random_state=random_state,
        verbose=False,
        apply_pca=False  # Skip PCA - embeddings are already semantically meaningful
    )

    return reducer.fit_transform(embeddings)


# =============================================================================
# CLUSTERING AND METRICS
# =============================================================================

@dataclass
class ClusteringResult:
    """Result from a clustering run."""
    labels: np.ndarray
    probabilities: np.ndarray
    n_clusters: int
    noise_count: int
    noise_rate: float
    dbcv: float
    relative_validity: float
    persistence: Optional[np.ndarray]
    mean_persistence: float
    low_prob_count: int
    low_prob_rate: float


def run_hdbscan(
    data: np.ndarray,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int = MIN_SAMPLES,
    metric: str = "euclidean"
) -> Tuple[hdbscan.HDBSCAN, ClusteringResult]:
    """
    Run HDBSCAN clustering and compute metrics.

    Args:
        data: Either reduced embeddings or precomputed distance matrix
        min_cluster_size: HDBSCAN min_cluster_size
        min_samples: HDBSCAN min_samples
        metric: "euclidean" or "precomputed"

    Returns:
        clusterer: HDBSCAN object
        result: ClusteringResult with all metrics
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        gen_min_span_tree=True,
        prediction_data=True
    )

    labels = clusterer.fit_predict(data)
    probabilities = clusterer.probabilities_

    # Basic counts
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    noise_rate = noise_count / len(labels)

    # DBCV
    try:
        dbcv = float(clusterer.relative_validity_)
    except Exception:
        dbcv = float('nan')

    # Relative validity (same as DBCV for HDBSCAN)
    relative_validity = dbcv

    # Persistence
    persistence = None
    mean_persistence = float('nan')
    if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None:
        persistence = clusterer.cluster_persistence_
        if len(persistence) > 0:
            mean_persistence = float(np.mean(persistence))

    # Low probability ratio
    low_prob_count = int(np.sum(probabilities < LOW_PROB_THRESHOLD))
    low_prob_rate = low_prob_count / len(probabilities)

    result = ClusteringResult(
        labels=labels,
        probabilities=probabilities,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_rate=noise_rate,
        dbcv=dbcv,
        relative_validity=relative_validity,
        persistence=persistence,
        mean_persistence=mean_persistence,
        low_prob_count=low_prob_count,
        low_prob_rate=low_prob_rate
    )

    return clusterer, result


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def print_setup_header(setup_num: int, description: str):
    """Print setup header."""
    print()
    print("=" * 80)
    print(f"SETUP {setup_num}: {description}")
    print("=" * 80)


def print_metrics(result: ClusteringResult, params: Dict[str, Any]):
    """Print clustering metrics."""
    print()
    print("PARAMETERS")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print()
    print("CLUSTERING METRICS")
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Noise: {result.noise_count} ({result.noise_rate:.1%})")
    print(f"  DBCV: {result.dbcv:.3f}")
    print(f"  Relative Validity: {result.relative_validity:.3f}")
    print(f"  Mean Persistence: {result.mean_persistence:.3f}")
    print(f"  Low Probability (<{LOW_PROB_THRESHOLD}): {result.low_prob_count} ({result.low_prob_rate:.1%})")


def print_cluster_samples(
    labels: np.ndarray,
    idea_texts: List[str],
    template_prefix: str,
    n_samples: int = 10
):
    """Print sample ideas for each cluster."""
    print()
    print(f"CLUSTER SAMPLES ({n_samples} per cluster)")

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            label_name = "Noise"
        else:
            label_name = f"Cluster {label}"

        cluster_indices = np.where(labels == label)[0]
        cluster_size = len(cluster_indices)

        print()
        print(f"  {label_name} (size={cluster_size}):")

        # Get unique texts for this cluster, stripping context specifiers
        cluster_texts = [strip_context_specifiers(idea_texts[i]) for i in cluster_indices]

        # Get unique texts
        seen = set()
        unique_texts = []
        for text in cluster_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        # Print samples
        for i, text in enumerate(unique_texts[:n_samples]):
            # Truncate long texts
            display_text = text[:100] + "..." if len(text) > 100 else text
            print(f"    - {display_text}")


# =============================================================================
# SETUP RUNNERS
# =============================================================================

def run_setup_1(embeddings_normalized: np.ndarray, idea_texts: List[str], template_prefix: str):
    """Setup 1: UMAP + HDBSCAN (euclidean)"""
    print_setup_header(1, "UMAP + HDBSCAN (euclidean)")

    # Dimensionality reduction
    print("\nRunning UMAP...")
    reduced = run_umap(embeddings_normalized)
    print(f"  Reduced shape: {reduced.shape}")

    # Clustering
    print("Running HDBSCAN...")
    clusterer, result = run_hdbscan(reduced, metric="euclidean")

    # Output
    params = {
        "UMAP n_components": UMAP_N_COMPONENTS,
        "UMAP n_neighbors": UMAP_N_NEIGHBORS,
        "UMAP min_dist": UMAP_MIN_DIST,
        "HDBSCAN min_cluster_size": MIN_CLUSTER_SIZE,
        "HDBSCAN min_samples": MIN_SAMPLES,
        "HDBSCAN metric": "euclidean"
    }
    print_metrics(result, params)
    print_cluster_samples(result.labels, idea_texts, template_prefix)

    return result


def run_setup_2(embeddings_normalized: np.ndarray, idea_texts: List[str], template_prefix: str):
    """Setup 2: PaCMAP + HDBSCAN (euclidean)"""
    print_setup_header(2, "PaCMAP + HDBSCAN (euclidean)")

    # Dimensionality reduction
    print("\nRunning PaCMAP (with sklearn kNN workaround)...")
    reduced = run_pacmap(embeddings_normalized)
    print(f"  Reduced shape: {reduced.shape}")

    # Clustering
    print("Running HDBSCAN...")
    clusterer, result = run_hdbscan(reduced, metric="euclidean")

    # Output
    params = {
        "PaCMAP n_components": PACMAP_N_COMPONENTS,
        "PaCMAP n_neighbors": PACMAP_N_NEIGHBORS,
        "HDBSCAN min_cluster_size": MIN_CLUSTER_SIZE,
        "HDBSCAN min_samples": MIN_SAMPLES,
        "HDBSCAN metric": "euclidean"
    }
    print_metrics(result, params)
    print_cluster_samples(result.labels, idea_texts, template_prefix)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("HDBSCAN SETUP COMPARISON TEST")
    print("=" * 80)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")

    # Load embeddings
    print("\nLoading embeddings from cache...")
    embeddings, idea_texts = load_embeddings()
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # L2 normalize
    embeddings_normalized = l2_normalize(embeddings)

    # Detect template prefix
    template_prefix = detect_template_prefix(idea_texts)
    if template_prefix:
        print(f"Detected template prefix: '{template_prefix[:50]}...'")
    else:
        print("No template prefix detected")

    # Run all setups
    results = {}

    results['setup_1'] = run_setup_1(embeddings_normalized, idea_texts, template_prefix)
    results['setup_2'] = run_setup_2(embeddings_normalized, idea_texts, template_prefix)

    # Summary comparison
    print()
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Setup':<45} {'Clusters':>10} {'Noise':>10} {'DBCV':>10} {'LowProb':>10}")
    print("-" * 85)

    setup_names = [
        "1: UMAP + HDBSCAN",
        "2: PaCMAP + HDBSCAN",
    ]

    for i, (key, name) in enumerate(zip(['setup_1', 'setup_2'], setup_names)):
        r = results[key]
        print(f"{name:<45} {r.n_clusters:>10} {r.noise_rate:>9.1%} {r.dbcv:>10.3f} {r.low_prob_rate:>9.1%}")

    print("=" * 80)

    return results


if __name__ == "__main__":
    main()

# %%
