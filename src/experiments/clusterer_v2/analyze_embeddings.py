#%% 
# 
"""
Embedding Space Diagnostic Tool

Analyze the raw pairwise cosine similarity distribution of embeddings
BEFORE any normalization, dimensionality reduction, or clustering.

This provides insight into the data regime:
- Diffuse (q90 < 0.65): spread out, weak clustering signal
- Mixed (0.65 <= q90 < 0.80): moderate structure
- Coherent (q90 >= 0.80): tight clusters expected

Usage:
    cd src/experiments/clusterer_v2
    python analyze_embeddings.py
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional
import pickle

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import models
from utils.cacheManager import generate_enhanced_variable_key


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

FILENAME = "M250480 Associatiemonitor ASN Bank net databestand.sav"
VARIABLE = "Qd1_combined"
SAMPLE_SIZE = 2000


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
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities for all embedding pairs.

    Returns upper triangle values only (excluding diagonal).
    """
    # L2 normalize for cosine similarity via dot product
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    # Full similarity matrix
    sim_matrix = embeddings_normalized @ embeddings_normalized.T

    # Extract upper triangle (excluding diagonal)
    n = len(embeddings)
    upper_tri_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[upper_tri_indices]

    return similarities


def compute_dvc(embeddings: np.ndarray, k: int = 10) -> dict:
    """
    Compute Density Variation Coefficient.

    DVC = std(d_k) / mean(d_k), where d_k is distance to k-th nearest neighbor.

    High DVC (>0.45): variable density → HDBSCAN better
    Low DVC (<0.25): uniform density → Agglomerative better
    """
    # L2 normalize
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    n = len(embeddings)
    if n < k + 1:
        return {'dvc': np.nan, 'mean_dk': np.nan, 'std_dk': np.nan}

    # Compute k-th nearest neighbor distances
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nbrs.fit(embeddings_normalized)
    distances, _ = nbrs.kneighbors(embeddings_normalized)

    # k-th neighbor distance (last column, since index 0 is self)
    d_k = distances[:, -1]

    mean_dk = float(np.mean(d_k))
    std_dk = float(np.std(d_k))

    if mean_dk == 0:
        return {'dvc': np.nan, 'mean_dk': mean_dk, 'std_dk': std_dk}

    dvc = std_dk / mean_dk

    return {
        'dvc': float(dvc),
        'mean_dk': mean_dk,
        'std_dk': std_dk
    }


def classify_regime(q90: float) -> str:
    """Classify data regime based on q90 similarity threshold."""
    if q90 < 0.65:
        return "DIFFUSE"
    elif q90 < 0.80:
        return "MIXED"
    else:
        return "COHERENT"


def print_histogram(similarities: np.ndarray, bins: int = 5, width: int = 40):
    """Print ASCII histogram of similarity distribution."""
    bin_edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(similarities, bins=bin_edges)
    percentages = counts / len(similarities) * 100

    max_pct = max(percentages)

    for i in range(bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        pct = percentages[i]
        bar_len = int(pct / max_pct * width) if max_pct > 0 else 0
        bar = "█" * bar_len
        print(f"  [{low:.1f}-{high:.1f}): {bar} {pct:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("EMBEDDING SPACE DIAGNOSTIC")
    print("=" * 80)
    print(f"Dataset: {FILENAME}")
    print(f"Variable: {VARIABLE} | Sample Size: {SAMPLE_SIZE}")
    print()

    # Load embeddings
    print("Loading embeddings from cache...")
    embeddings, idea_texts = load_embeddings()

    n_embeddings, dim = embeddings.shape
    print()
    print("EMBEDDING STATISTICS")
    print(f"  Total embeddings: {n_embeddings} | Dimension: {dim} | dtype: {embeddings.dtype}")
    print()

    # Compute pairwise similarities
    print("Computing pairwise cosine similarities...")
    similarities = compute_pairwise_similarities(embeddings)
    n_pairs = len(similarities)

    print()
    print(f"PAIRWISE COSINE SIMILARITY (upper triangle, {n_pairs:,} pairs)")

    # Quantiles
    q10 = np.percentile(similarities, 10)
    q25 = np.percentile(similarities, 25)
    q50 = np.percentile(similarities, 50)
    q75 = np.percentile(similarities, 75)
    q90 = np.percentile(similarities, 90)
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)

    print(f"  q10:  {q10:.3f}")
    print(f"  q25:  {q25:.3f}")
    print(f"  q50:  {q50:.3f}  (median)")
    print(f"  q75:  {q75:.3f}")
    print(f"  q90:  {q90:.3f}")
    print(f"  mean: {mean_sim:.3f} | std: {std_sim:.3f} | min: {min_sim:.3f} | max: {max_sim:.3f}")

    # High similarity counts
    n_above_90 = np.sum(similarities > 0.90)
    n_above_95 = np.sum(similarities > 0.95)
    pct_above_90 = n_above_90 / n_pairs * 100
    pct_above_95 = n_above_95 / n_pairs * 100
    print()
    print(f"  pairs > 0.90: {n_above_90:,} ({pct_above_90:.2f}%)")
    print(f"  pairs > 0.95: {n_above_95:,} ({pct_above_95:.2f}%)")
    print()

    # Regime classification
    regime = classify_regime(q90)
    print("REGIME CLASSIFICATION (based on q90)")
    print(f"  q90 = {q90:.3f} → {regime}", end="")
    if regime == "DIFFUSE":
        print(" (q90 < 0.65)")
    elif regime == "MIXED":
        print(" (0.65 ≤ q90 < 0.80)")
    else:
        print(" (q90 ≥ 0.80)")
    print()
    print("  Legend: Diffuse (<0.65) | Mixed (0.65-0.80) | Coherent (≥0.80)")
    print()

    # Histogram
    print("HISTOGRAM")
    print_histogram(similarities)
    print()

    # DVC with k=sqrt(n)
    k_sqrt = int(np.sqrt(n_embeddings))
    print(f"DVC (Density Variation Coefficient, k=sqrt(n)={k_sqrt})")
    dvc_result = compute_dvc(embeddings, k=k_sqrt)
    print(f"  mean_dk: {dvc_result['mean_dk']:.3f} | std_dk: {dvc_result['std_dk']:.3f} | DVC: {dvc_result['dvc']:.3f}")

    if dvc_result['dvc'] < 0.25:
        print("  → Uniform density (DVC < 0.25) → Agglomerative preferred")
    elif dvc_result['dvc'] < 0.45:
        print("  → Moderate density variation (0.25 ≤ DVC < 0.45)")
    else:
        print("  → High density variation (DVC ≥ 0.45) → HDBSCAN preferred")

    print("=" * 80)


if __name__ == "__main__":
    main()
