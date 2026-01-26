#%%

"""
ClustererV3 Run Script - PaCMAP + HDBSCAN Pipeline

Replaces UMAP with PaCMAP grid search using zadu library metrics for
dimensionality reduction quality evaluation.

Two-phase pipeline:
1. Phase 1: PaCMAP Grid Search - Find best DR config using zadu metrics
   (trustworthiness, continuity, steadiness, cohesiveness, stress)
2. Phase 2: HDBSCAN Optuna - Optimize clustering on best PaCMAP reduction

Usage:
    cd src/experiments/clusterer_v2
    python run_clusterer_v3.py

Configure the dataset by editing the variables below.
"""

import sys
import io
import math
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
import hdbscan
import optuna
from optuna.samplers import GridSampler
from sklearn.preprocessing import normalize

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

import models
from utils.cacheManager import generate_enhanced_variable_key

from clusterer_v2 import ClustererV2Config
from pacmap_optimizer import PaCMAPOptimizer, PaCMAPGridConfig, PaCMAPGridSearchResult

# Suppress warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# DATASET CONFIGURATION - Edit these to match your cached Step 4 data
# =============================================================================

FILENAME = "M000000 Associatiemonitor Merk X net databestand.sav"
VARIABLE = "Qd1_combined"
SAMPLE_SIZE = 2000

# =============================================================================
# PACMAP CONFIGURATION
# =============================================================================

PACMAP_CONFIG = PaCMAPGridConfig(
    # Parameter grid (12 total combinations - reduced for faster testing)
    n_neighbors_grid=(10, 20, 30),
    mn_ratio_grid=(0.3, 0.7),
    fp_ratio_grid=(1.0, 2.0),

    # Fixed n_components (BERTopic suggests 5, but 10 often better)
    n_components=10,

    # Zadu metric k parameter
    zadu_k=15,

    # Other settings
    random_state=42,
    apply_pca=True,
)

# =============================================================================
# HDBSCAN CONFIGURATION
# =============================================================================

# HDBSCAN optimization constraints
HDBSCAN_MAX_NOISE_RATE = 0.20
HDBSCAN_MIN_CLUSTERS = 3


# =============================================================================
# HDBSCAN OPTIMIZER (Simplified for pre-computed reductions)
# =============================================================================

@dataclass
class HDBSCANResult:
    """Result from HDBSCAN optimization."""
    min_cluster_size: int
    min_samples: int
    relative_validity: float
    n_clusters: int
    noise_rate: float
    coherence: float
    labels: np.ndarray
    model: hdbscan.HDBSCAN


@dataclass
class HDBSCANGridSearchResult:
    """Result from HDBSCAN grid search."""
    best_result: HDBSCANResult
    all_results: List[HDBSCANResult]
    n_completed: int
    n_pruned: int
    search_space: Dict[str, List]


def log_spaced_ints(low: int, high: int, k: int = 4) -> List[int]:
    """Generate k log-spaced integers between low and high."""
    if low == high:
        return [low]
    if low <= 0:
        low = 1
    vals = np.exp(np.linspace(np.log(low), np.log(high), k))
    return sorted(set(int(round(v)) for v in vals))


def mcs_grid_sqrt(N: int, k: int = 4) -> List[int]:
    """Generate min_cluster_size grid based on sqrt(N)."""
    low = max(3, int(round(0.25 * math.sqrt(N))))
    high = max(low, int(round(1.0 * math.sqrt(N))))
    return log_spaced_ints(low, high, k=k)


def calculate_coherence(labels: np.ndarray, embeddings: np.ndarray) -> float:
    """Calculate mean intra-cluster cosine similarity."""
    unique_labels = [l for l in set(labels) if l >= 0]
    if not unique_labels:
        return 0.0

    coherences = []
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) < 2:
            coherences.append(1.0)
            continue

        similarities = cluster_embeddings @ cluster_embeddings.T
        n = len(cluster_embeddings)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_tri_indices]
        coherences.append(np.mean(pairwise_sims))

    return np.mean(coherences)


def compute_dbcv(labels: np.ndarray, embeddings: np.ndarray) -> float:
    """Compute DBCV score as fallback for relative_validity_."""
    try:
        from hdbscan import validity
        mask = labels >= 0
        if mask.sum() < 2:
            return -1.0
        embeddings_f64 = embeddings[mask].astype(np.float64)
        labels_filtered = labels[mask]
        score = validity.validity_index(embeddings_f64, labels_filtered)
        return float(score)
    except Exception:
        return -1.0


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    return normalize(embeddings, norm='l2', axis=1)


def run_hdbscan_grid_search(
    reduced_embeddings: np.ndarray,
    original_embeddings: np.ndarray,
    verbose: bool = True
) -> HDBSCANGridSearchResult:
    """
    Run grid search over HDBSCAN parameters on pre-reduced embeddings.

    Args:
        reduced_embeddings: PaCMAP-reduced embeddings (N x n_components)
        original_embeddings: Original embeddings for coherence calculation
        verbose: Print progress

    Returns:
        HDBSCANGridSearchResult with best configuration
    """
    N = len(reduced_embeddings)

    # Build search space
    mcs_grid = mcs_grid_sqrt(N, k=4)
    search_space = {
        'min_cluster_size': mcs_grid,
    }

    if verbose:
        print(f"\n{'='*70}")
        print("HDBSCAN Grid Search (Optuna)")
        print('='*70)
        print(f"Dataset size: {N}")
        print(f"Reduced dimensions: {reduced_embeddings.shape[1]}")
        print(f"min_cluster_size grid: {mcs_grid}")
        print(f"Total trials: {len(mcs_grid)}")

    all_results: List[HDBSCANResult] = []
    best_result: Optional[HDBSCANResult] = None
    n_pruned = 0

    for mcs in mcs_grid:
        ms = max(1, mcs // 2)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        labels = clusterer.fit_predict(reduced_embeddings)

        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).sum() / len(labels)

        # Check constraints
        if n_clusters < HDBSCAN_MIN_CLUSTERS:
            n_pruned += 1
            if verbose:
                print(f"  mcs={mcs}: PRUNED (too few clusters: {n_clusters})")
            continue

        if noise_rate > HDBSCAN_MAX_NOISE_RATE:
            n_pruned += 1
            if verbose:
                print(f"  mcs={mcs}: PRUNED (noise too high: {noise_rate:.1%})")
            continue

        # Get relative_validity_
        try:
            relative_validity = clusterer.relative_validity_
        except AttributeError:
            relative_validity = compute_dbcv(labels, reduced_embeddings)

        # Calculate coherence on original embeddings
        coherence = calculate_coherence(labels, original_embeddings)

        result = HDBSCANResult(
            min_cluster_size=mcs,
            min_samples=ms,
            relative_validity=relative_validity,
            n_clusters=n_clusters,
            noise_rate=noise_rate,
            coherence=coherence,
            labels=labels,
            model=clusterer
        )
        all_results.append(result)

        # Track best by relative_validity
        if best_result is None or relative_validity > best_result.relative_validity:
            best_result = result
            if verbose:
                print(f"  mcs={mcs}: validity={relative_validity:.4f}, k={n_clusters}, "
                      f"noise={noise_rate:.1%}, coh={coherence:.3f} ★ NEW BEST")
        else:
            if verbose:
                print(f"  mcs={mcs}: validity={relative_validity:.4f}, k={n_clusters}, "
                      f"noise={noise_rate:.1%}, coh={coherence:.3f}")

    n_completed = len(all_results)

    if verbose:
        print(f"\n  Grid search complete: {n_completed} completed, {n_pruned} pruned")
        if best_result:
            print(f"  Best: mcs={best_result.min_cluster_size}, ms={best_result.min_samples}")
            print(f"  → validity={best_result.relative_validity:.4f}, k={best_result.n_clusters}, "
                  f"noise={best_result.noise_rate:.1%}, coh={best_result.coherence:.3f}")

    return HDBSCANGridSearchResult(
        best_result=best_result,
        all_results=all_results,
        n_completed=n_completed,
        n_pruned=n_pruned,
        search_space=search_space
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step4_embeddings(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None
) -> Tuple[np.ndarray, List[str], List[models.EmbeddingsModel]]:
    """
    Load Step 4 embeddings from cache.

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        idea_texts: list of idea text strings
        embeddings_models: list of EmbeddingsModel objects
    """
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(filename).stem
    cache_filename = f"005_embeddings_{base_name}_{variable_key}.pkl"
    cache_path = cache_dir / cache_filename

    print(f"Loading embeddings from: {cache_path}")

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run pipeline step 4 first to generate embeddings."
        )

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    # Convert serialized data to EmbeddingsModel objects
    embeddings_models = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

    # Build embeddings array and idea texts list
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
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    return embeddings, idea_texts, embeddings_models


# =============================================================================
# CLUSTER PRINTING
# =============================================================================

def print_clusters_with_samples(
    labels: np.ndarray,
    idea_texts: List[str],
    n_samples: int = 10
) -> None:
    """Print all clusters with sample ideas."""
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
    noise_count = (labels == -1).sum()

    print(f"\n{'='*70}")
    print(f"CLUSTER SAMPLES ({n_clusters} clusters, {noise_count} noise points)")
    print('='*70)

    for label in unique_labels:
        mask = labels == label
        cluster_texts = [idea_texts[i] for i in range(len(labels)) if labels[i] == label]
        cluster_size = len(cluster_texts)

        if label == -1:
            print(f"\n[NOISE] ({cluster_size} ideas)")
        else:
            print(f"\n[Cluster {label}] ({cluster_size} ideas)")

        # Print samples
        samples = cluster_texts[:n_samples]
        for i, text in enumerate(samples, 1):
            text_display = text[:80] + "..." if len(text) > 80 else text
            print(f"  {i}. {text_display}")

        if cluster_size > n_samples:
            print(f"  ... and {cluster_size - n_samples} more")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run ClustererV3 pipeline (PaCMAP + HDBSCAN)."""
    print("=" * 70)
    print("ClustererV3 Pipeline (PaCMAP + HDBSCAN)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print()

    # ==========================================================================
    # Phase 1: Load embeddings
    # ==========================================================================
    print("[Phase 1] Loading embeddings")
    embeddings, idea_texts, embeddings_models = load_step4_embeddings()
    embeddings_normalized = l2_normalize(embeddings)

    # ==========================================================================
    # Phase 2: PaCMAP Grid Search (zadu metrics)
    # ==========================================================================
    print("\n[Phase 2] PaCMAP Grid Search")
    pacmap_optimizer = PaCMAPOptimizer(
        config=PACMAP_CONFIG,
        embeddings=embeddings_normalized,
        verbose=True
    )
    pacmap_result = pacmap_optimizer.optimize()

    # Get best PaCMAP reduction
    best_reduction = pacmap_result.best_result.reduced_embeddings
    best_reduction_normalized = l2_normalize(best_reduction)

    # ==========================================================================
    # Phase 3: HDBSCAN Optimization
    # ==========================================================================
    print("\n[Phase 3] HDBSCAN Optimization")
    hdbscan_result = run_hdbscan_grid_search(
        reduced_embeddings=best_reduction_normalized,
        original_embeddings=embeddings_normalized,
        verbose=True
    )

    if hdbscan_result.best_result is None:
        print("\nERROR: No valid HDBSCAN configuration found!")
        return None, pacmap_result, None

    # ==========================================================================
    # Phase 4: Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # PaCMAP summary
    pr = pacmap_result.best_result
    print(f"\nBest PaCMAP configuration:")
    print(f"  n_neighbors={pr.n_neighbors}, MN_ratio={pr.mn_ratio}, FP_ratio={pr.fp_ratio}")
    print(f"  n_components={pr.n_components}")
    print(f"  summed_score: {pr.summed_score:.4f}")
    print(f"  Raw metrics:")
    print(f"    trustworthiness: {pr.trustworthiness:.4f}")
    print(f"    continuity: {pr.continuity:.4f}")
    print(f"    steadiness: {pr.steadiness:.4f}")
    print(f"    cohesiveness: {pr.cohesiveness:.4f}")
    print(f"    stress: {pr.stress:.4f}")

    # HDBSCAN summary
    hr = hdbscan_result.best_result
    print(f"\nBest HDBSCAN configuration:")
    print(f"  min_cluster_size={hr.min_cluster_size}, min_samples={hr.min_samples}")
    print(f"  relative_validity: {hr.relative_validity:.4f}")
    print(f"  n_clusters: {hr.n_clusters}")
    print(f"  noise_rate: {hr.noise_rate:.1%}")
    print(f"  coherence: {hr.coherence:.3f}")

    # ==========================================================================
    # Phase 5: Print cluster samples
    # ==========================================================================
    print_clusters_with_samples(
        labels=hr.labels,
        idea_texts=idea_texts,
        n_samples=10
    )

    return embeddings_models, pacmap_result, hdbscan_result


# =============================================================================
# OUTPUT CAPTURE AND EXPORT
# =============================================================================

class TeeOutput:
    """Capture stdout while also printing to console."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original_stdout.flush()

    def get_output(self) -> str:
        return self.buffer.getvalue()


def save_results_to_file(
    output: str,
    filename: str,
    variable: str,
    sample_size: Optional[int]
) -> Path:
    """Save clustering results to a text file."""
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_filename = f"cluster_results_v3_{base_name}_{variable}_{sample_str}_{date_str}.txt"
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


def save_pacmap_grid_results(
    pacmap_result: PaCMAPGridSearchResult,
    filename: str,
    variable: str,
    sample_size: Optional[int]
) -> Path:
    """Save PaCMAP grid search results to CSV."""
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_filename = f"pacmap_grid_{base_name}_{variable}_{sample_str}_{date_str}.csv"
    output_path = output_dir / output_filename

    pacmap_result.results_df.to_csv(output_path, index=False)

    return output_path


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        embeddings_models, pacmap_result, hdbscan_result = main()
    finally:
        # Restore stdout
        sys.stdout = tee.original_stdout

    if pacmap_result is not None:
        # Save results to file
        output_path = save_results_to_file(
            output=tee.get_output(),
            filename=FILENAME,
            variable=VARIABLE,
            sample_size=SAMPLE_SIZE
        )
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")

        # Save PaCMAP grid results to CSV
        csv_path = save_pacmap_grid_results(
            pacmap_result=pacmap_result,
            filename=FILENAME,
            variable=VARIABLE,
            sample_size=SAMPLE_SIZE
        )
        print(f"PaCMAP grid results saved to: {csv_path}")

# %%
