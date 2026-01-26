#%%

"""
ClustererV2 Run Script with PaCMAP Comparison

Run the ClustererV2 pipeline on a specific dataset from cached Step 4 embeddings,
with optional PaCMAP comparison to evaluate both dimensionality reduction methods.

Usage:
    cd src/experiments/clusterer_v2
    python run_clusterer_v2.py

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
import hdbscan
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

import models
from utils.cacheManager import generate_enhanced_variable_key

from clusterer_v2 import ClustererV2, ClustererV2Config

# Suppress warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")


# =============================================================================
# DATASET CONFIGURATION - Edit these to match your cached Step 4 data
# =============================================================================

FILENAME = "M250480 Associatiemonitor ASN Bank net databestand.sav"
VARIABLE = "Qd1_combined"
SAMPLE_SIZE = 2000

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

ENABLE_UMAP = False          # Set to False to skip UMAP-based ClustererV2
ENABLE_PACMAP = True         # Set to True to run PaCMAP grid search

# PaCMAP parameter grids for grid search
# Note: Start with smaller grids for testing
PACMAP_N_COMPONENTS_GRID = (5, 10)       # Reduced for testing
PACMAP_MN_RATIO_GRID = (0.5, 1.0)        # Mid-near pairs ratio (default 0.5)
PACMAP_FP_RATIO_GRID = (2.0,)            # Further pairs ratio (default 2.0) - single value for testing

# Use same n_neighbors grid as UMAP (dataset-adaptive)
PACMAP_N_NEIGHBORS_GRID_K = 4  # Number of log-spaced points

# HDBSCAN constraints for PaCMAP trials
PACMAP_MAX_NOISE_RATE = 0.20
PACMAP_MIN_CLUSTERS = 3

# =============================================================================
# CLUSTERER CONFIGURATION (UMAP-based)
# =============================================================================

CONFIG = ClustererV2Config(
    # Algorithm selection: "auto", "hdbscan", "agglomerative", "kmeans"
    algorithm_mode="auto",

    # DVC thresholds for algorithm selection
    dvc_high_threshold=0.45,    # Above this → HDBSCAN
    dvc_low_threshold=0.25,     # Below this → Agglomerative

    # Hard rule: force Agglomerative when DVC < this
    force_agglomerative_below_dvc=0.25,

    # Knee detection
    knee_y_diff_threshold=0.6,  # Sharp knee threshold

    # Optuna optimization (for HDBSCAN)
    use_optuna=True,
    max_noise_rate=0.20,        # Maximum acceptable noise rate
    min_clusters=3,             # Minimum number of clusters

    # Quality thresholds for conditional re-search
    # Trigger: (noise > max AND validity < min) OR (cluster_deviation > threshold)
    enable_research=True,
    research_max_noise_rate=0.10,
    research_min_validity=0.70,
    research_cluster_deviation_threshold=0.15,

    # Post-processing
    enable_merging=True,
    merge_centroid_threshold=0.95,
    merge_pairwise_threshold=0.98,

    # BERTopic-style noise reduction
    noise_reduction_strategy="embeddings",
    noise_reduction_threshold=0.5,

    # c-TF-IDF keyword extraction with lemmatization
    generate_ctfidf=True,
    ctfidf_top_k=10,
    ctfidf_use_lemmatization=True,

    # LLM labels (enabled)
    generate_llm_labels=True,

    # Output
    verbose=True,
)


# =============================================================================
# PACMAP FUNCTIONS
# =============================================================================

def compute_knn_pairs(embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
    """
    Compute k-nearest neighbor pairs using sklearn (bypasses Annoy).

    This function solves the known PaCMAP bug on macOS + Python 3.12+ where
    the Annoy library fails silently, returning 0 neighbors instead of
    the expected number.

    See: https://github.com/YingfanWang/PaCMAP/issues/94

    Args:
        embeddings: (N, D) array of embeddings
        n_neighbors: number of neighbors to find

    Returns:
        pair_neighbors: (N * n_neighbors, 2) array of (source_idx, neighbor_idx) pairs
                        This is the format expected by PaCMAP's pair_neighbors parameter.
    """
    from sklearn.neighbors import NearestNeighbors

    N = len(embeddings)

    # Add 1 because sklearn includes the point itself as the first neighbor
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', algorithm='auto')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Remove self (first column) to get actual neighbors
    neighbor_indices = indices[:, 1:]  # Shape: (N, n_neighbors)

    # Convert to PaCMAP's expected format: (N * n_neighbors, 2) pairs of (source, neighbor)
    # Each row i in pair_neighbors should be: [source_point_idx, neighbor_point_idx]
    pair_neighbors = np.zeros((N * n_neighbors, 2), dtype=np.int32)

    for i in range(N):
        for j in range(n_neighbors):
            pair_neighbors[i * n_neighbors + j, 0] = i
            pair_neighbors[i * n_neighbors + j, 1] = neighbor_indices[i, j]

    return pair_neighbors


def run_pacmap(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int,
    MN_ratio: float = 0.5,
    FP_ratio: float = 2.0,
    random_state: int = 42,
    apply_pca: bool = True
) -> np.ndarray:
    """
    Run PaCMAP dimensionality reduction with custom kNN (bypasses Annoy bug).

    Uses sklearn's NearestNeighbors to compute nearest neighbors instead of
    PaCMAP's internal Annoy-based computation, which fails on some systems.

    Args:
        embeddings: L2-normalized embeddings (N x D)
        n_neighbors: Number of neighbors for local structure
        n_components: Target dimensionality
        MN_ratio: Mid-near pairs ratio (controls mid-range structure)
        FP_ratio: Further pairs ratio (controls global separation)
        random_state: Random seed for reproducibility
        apply_pca: Whether to apply PCA before neighbor search (helps with high-dim data)

    Returns:
        Reduced embeddings (N x n_components)
    """
    import pacmap

    # PaCMAP needs approximately n_neighbors * (1 + MN_ratio + FP_ratio) unique neighbors
    # If this exceeds the dataset size, we need to reduce n_neighbors
    N = len(embeddings)
    total_neighbors_needed = int(n_neighbors * (1 + MN_ratio + FP_ratio)) + 1

    if total_neighbors_needed >= N:
        # Scale down n_neighbors to fit
        max_n_neighbors = int(N / (1 + MN_ratio + FP_ratio)) - 1
        max_n_neighbors = max(5, max_n_neighbors)  # At least 5
        if max_n_neighbors < n_neighbors:
            n_neighbors = max_n_neighbors

    # Compute nearest neighbors ourselves using sklearn (bypasses Annoy bug)
    pair_neighbors = compute_knn_pairs(embeddings, n_neighbors)

    # Pass precomputed neighbors in constructor to bypass Annoy
    reducer = pacmap.PaCMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        pair_neighbors=pair_neighbors,  # Pre-computed neighbors from sklearn
        random_state=random_state,
        verbose=False,
        apply_pca=apply_pca
    )

    return reducer.fit_transform(embeddings)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    return normalize(embeddings, norm='l2', axis=1)


def log_spaced_ints(low: int, high: int, k: int = 4) -> List[int]:
    """Generate k log-spaced integers between low and high."""
    if low == high:
        return [low]
    if low <= 0:
        low = 1
    vals = np.exp(np.linspace(np.log(low), np.log(high), k))
    return sorted(set(int(round(v)) for v in vals))


def n_neighbors_grid(N: int, k: int = 3) -> List[int]:
    """
    Generate n_neighbors grid for both UMAP and PaCMAP.

    Uses fixed range [10, 30] with k=3 log-spaced points.
    This range works well for both methods and typical survey datasets.

    Args:
        N: Dataset size (used for safety bounds only)
        k: Number of grid points (default 3)

    Returns:
        Log-spaced list: [10, 17, 30] for k=3
    """
    low = 10
    high = 30

    # Safety: ensure high doesn't exceed N-1
    high = min(high, N - 1)
    low = min(low, high)

    return log_spaced_ints(low, high, k=k)


def mcs_bounds_sqrt(N: int) -> Tuple[int, int]:
    """Compute min_cluster_size bounds based on sqrt(N)."""
    low = max(3, int(round(0.25 * math.sqrt(N))))
    high = max(low, int(round(1.0 * math.sqrt(N))))
    return low, high


def mcs_grid_sqrt(N: int, k: int = 4) -> List[int]:
    """Generate min_cluster_size grid for dataset of size N."""
    low, high = mcs_bounds_sqrt(N)
    return log_spaced_ints(low, high, k=k)


@dataclass
class PaCMAPTrialResult:
    """Result from a single PaCMAP + HDBSCAN trial."""
    n_neighbors: int
    n_components: int
    MN_ratio: float
    FP_ratio: float
    min_cluster_size: int
    min_samples: int
    relative_validity: float
    n_clusters: int
    noise_rate: float
    coherence: float
    labels: np.ndarray
    reduced_embeddings: np.ndarray
    pruned: bool = False
    prune_reason: str = ""


@dataclass
class PaCMAPGridSearchResult:
    """Result from PaCMAP grid search."""
    best_trial: PaCMAPTrialResult
    all_trials: List[PaCMAPTrialResult]
    n_completed: int
    n_pruned: int
    search_space: Dict[str, Any]


def precompute_pacmap_reductions(
    embeddings: np.ndarray,
    n_neighbors_list: List[int],
    n_components_list: List[int],
    MN_ratio_list: List[float],
    FP_ratio_list: List[float],
    verbose: bool = True,
    n_jobs: int = -1
) -> Dict[Tuple[int, int, float, float], np.ndarray]:
    """
    Pre-compute PaCMAP reductions for all parameter combinations in parallel.

    Args:
        embeddings: L2-normalized embeddings
        n_neighbors_list: List of n_neighbors values
        n_components_list: List of n_components values
        MN_ratio_list: List of MN_ratio values
        FP_ratio_list: List of FP_ratio values
        verbose: Print progress
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Dict mapping (n_neighbors, n_components, MN_ratio, FP_ratio) -> L2-normalized reduced embeddings
    """
    # Generate all combinations
    combinations = [
        (nn, nc, mn, fp)
        for nn in n_neighbors_list
        for nc in n_components_list
        for mn in MN_ratio_list
        for fp in FP_ratio_list
    ]

    if verbose:
        print(f"  Pre-computing {len(combinations)} PaCMAP reductions in parallel...")
        print(f"    n_neighbors: {n_neighbors_list}")
        print(f"    n_components: {n_components_list}")
        print(f"    MN_ratio: {MN_ratio_list}")
        print(f"    FP_ratio: {FP_ratio_list}")

    def compute_single_pacmap(
        n_neighbors: int,
        n_components: int,
        MN_ratio: float,
        FP_ratio: float
    ) -> Tuple[Tuple[int, int, float, float], np.ndarray]:
        try:
            reduced = run_pacmap(
                embeddings,
                n_neighbors,
                n_components,
                MN_ratio,
                FP_ratio
            )
            reduced_normalized = l2_normalize(reduced)
            return (n_neighbors, n_components, MN_ratio, FP_ratio), reduced_normalized
        except Exception as e:
            print(f"    ERROR with nn={n_neighbors}, nc={n_components}, mn={MN_ratio}, fp={FP_ratio}: {e}")
            raise

    # Run sequentially first for debugging (change n_jobs=1 to n_jobs for parallel)
    if verbose:
        print(f"    Running {len(combinations)} PaCMAP reductions sequentially for debugging...")

    results = []
    for i, (nn, nc, mn, fp) in enumerate(combinations):
        if verbose:
            print(f"    [{i+1}/{len(combinations)}] nn={nn}, nc={nc}, mn={mn}, fp={fp}...", end=" ")
        try:
            key, reduced = compute_single_pacmap(nn, nc, mn, fp)
            results.append((key, reduced))
            if verbose:
                print("OK")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            raise

    return {key: reduced for key, reduced in results}


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


def run_pacmap_grid_search(
    embeddings: np.ndarray,
    original_embeddings: np.ndarray,
    verbose: bool = True
) -> PaCMAPGridSearchResult:
    """
    Run grid search over PaCMAP + HDBSCAN parameter space.

    Args:
        embeddings: L2-normalized embeddings (for dimensionality reduction)
        original_embeddings: Original embeddings (for coherence calculation)
        verbose: Print progress

    Returns:
        PaCMAPGridSearchResult with best trial and all results
    """
    N = len(embeddings)

    # Build search space
    nn_grid = n_neighbors_grid(N, k=PACMAP_N_NEIGHBORS_GRID_K)
    nc_grid = list(PACMAP_N_COMPONENTS_GRID)
    mn_grid = list(PACMAP_MN_RATIO_GRID)
    fp_grid = list(PACMAP_FP_RATIO_GRID)
    mcs_grid = mcs_grid_sqrt(N, k=4)

    search_space = {
        'n_neighbors': nn_grid,
        'n_components': nc_grid,
        'MN_ratio': mn_grid,
        'FP_ratio': fp_grid,
        'min_cluster_size': mcs_grid,
    }

    total_pacmap_combos = len(nn_grid) * len(nc_grid) * len(mn_grid) * len(fp_grid)
    total_trials = total_pacmap_combos * len(mcs_grid)

    if verbose:
        print(f"\n{'='*70}")
        print("PaCMAP Grid Search")
        print('='*70)
        print(f"Dataset size: {N}")
        print(f"Search space:")
        print(f"  n_neighbors: {nn_grid}")
        print(f"  n_components: {nc_grid}")
        print(f"  MN_ratio: {mn_grid}")
        print(f"  FP_ratio: {fp_grid}")
        print(f"  min_cluster_size: {mcs_grid}")
        print(f"Total PaCMAP reductions: {total_pacmap_combos}")
        print(f"Total HDBSCAN trials: {total_trials}")

    # Pre-compute all PaCMAP reductions
    pacmap_cache = precompute_pacmap_reductions(
        embeddings,
        nn_grid,
        nc_grid,
        mn_grid,
        fp_grid,
        verbose=verbose
    )

    if verbose:
        print(f"\n  Running HDBSCAN trials...")

    all_trials: List[PaCMAPTrialResult] = []
    best_trial: Optional[PaCMAPTrialResult] = None

    trial_num = 0
    for nn in nn_grid:
        for nc in nc_grid:
            for mn in mn_grid:
                for fp in fp_grid:
                    # Get pre-computed PaCMAP reduction
                    reduced = pacmap_cache[(nn, nc, mn, fp)]

                    for mcs in mcs_grid:
                        trial_num += 1
                        ms = max(1, mcs // 2)

                        # Run HDBSCAN
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            metric='euclidean',
                            cluster_selection_method='eom',
                            gen_min_span_tree=True,
                        )
                        labels = clusterer.fit_predict(reduced)

                        # Calculate metrics
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        noise_rate = (labels == -1).sum() / len(labels)

                        # Check constraints
                        pruned = False
                        prune_reason = ""
                        if n_clusters < PACMAP_MIN_CLUSTERS:
                            pruned = True
                            prune_reason = f"Too few clusters: {n_clusters}"
                        elif noise_rate > PACMAP_MAX_NOISE_RATE:
                            pruned = True
                            prune_reason = f"Noise too high: {noise_rate:.1%}"

                        if pruned:
                            trial = PaCMAPTrialResult(
                                n_neighbors=nn,
                                n_components=nc,
                                MN_ratio=mn,
                                FP_ratio=fp,
                                min_cluster_size=mcs,
                                min_samples=ms,
                                relative_validity=-1.0,
                                n_clusters=n_clusters,
                                noise_rate=noise_rate,
                                coherence=0.0,
                                labels=labels,
                                reduced_embeddings=reduced,
                                pruned=True,
                                prune_reason=prune_reason
                            )
                            all_trials.append(trial)
                            continue

                        # Get relative_validity_
                        try:
                            relative_validity = clusterer.relative_validity_
                        except AttributeError:
                            relative_validity = compute_dbcv(labels, reduced)

                        # Calculate coherence on original embeddings
                        coherence = calculate_coherence(labels, original_embeddings)

                        trial = PaCMAPTrialResult(
                            n_neighbors=nn,
                            n_components=nc,
                            MN_ratio=mn,
                            FP_ratio=fp,
                            min_cluster_size=mcs,
                            min_samples=ms,
                            relative_validity=relative_validity,
                            n_clusters=n_clusters,
                            noise_rate=noise_rate,
                            coherence=coherence,
                            labels=labels,
                            reduced_embeddings=reduced
                        )
                        all_trials.append(trial)

                        # Track best
                        if best_trial is None or relative_validity > best_trial.relative_validity:
                            best_trial = trial
                            if verbose:
                                print(f"  ★ Trial {trial_num}: nn={nn}, nc={nc}, mn={mn}, fp={fp}, mcs={mcs} → "
                                      f"val={relative_validity:.4f}, k={n_clusters}, noise={noise_rate:.1%}, coh={coherence:.3f}")

    n_completed = len([t for t in all_trials if not t.pruned])
    n_pruned = len([t for t in all_trials if t.pruned])

    if verbose:
        print(f"\n  Grid search complete: {n_completed} completed, {n_pruned} pruned")
        if best_trial:
            print(f"  Best: nn={best_trial.n_neighbors}, nc={best_trial.n_components}, "
                  f"mn={best_trial.MN_ratio}, fp={best_trial.FP_ratio}, mcs={best_trial.min_cluster_size}")
            print(f"  → validity={best_trial.relative_validity:.4f}, k={best_trial.n_clusters}, "
                  f"noise={best_trial.noise_rate:.1%}, coh={best_trial.coherence:.3f}")

    return PaCMAPGridSearchResult(
        best_trial=best_trial,
        all_trials=all_trials,
        n_completed=n_completed,
        n_pruned=n_pruned,
        search_space=search_space
    )


def print_comparison_summary(
    umap_metrics: Dict[str, Any],
    pacmap_result: PaCMAPGridSearchResult
) -> None:
    """Print side-by-side comparison of UMAP vs PaCMAP results."""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY: UMAP vs PaCMAP")
    print('='*70)

    if pacmap_result.best_trial is None:
        print("  PaCMAP: No valid trials found!")
        return

    umap_validity = umap_metrics.get('relative_validity', umap_metrics.get('dbcv', 0))
    pacmap_validity = pacmap_result.best_trial.relative_validity

    umap_noise = umap_metrics.get('noise_rate', 0)
    pacmap_noise = pacmap_result.best_trial.noise_rate

    umap_coherence = umap_metrics.get('coherence', 0)
    pacmap_coherence = pacmap_result.best_trial.coherence

    umap_clusters = umap_metrics.get('n_clusters', 0)
    pacmap_clusters = pacmap_result.best_trial.n_clusters

    # Calculate deltas
    validity_delta = (pacmap_validity - umap_validity) / umap_validity * 100 if umap_validity != 0 else 0
    noise_delta = (pacmap_noise - umap_noise) / umap_noise * 100 if umap_noise != 0 else 0
    coherence_delta = (pacmap_coherence - umap_coherence) / umap_coherence * 100 if umap_coherence != 0 else 0

    print(f"\n{'Metric':<20} {'UMAP':<15} {'PaCMAP':<15} {'Δ':<15}")
    print("-" * 65)
    print(f"{'Validity':<20} {umap_validity:<15.4f} {pacmap_validity:<15.4f} {validity_delta:+.1f}%")
    print(f"{'Noise rate':<20} {umap_noise:<15.1%} {pacmap_noise:<15.1%} {noise_delta:+.1f}%")
    print(f"{'Coherence':<20} {umap_coherence:<15.3f} {pacmap_coherence:<15.3f} {coherence_delta:+.1f}%")
    print(f"{'Clusters':<20} {umap_clusters:<15} {pacmap_clusters:<15} {pacmap_clusters - umap_clusters:+d}")

    # Determine winner based on validity
    if pacmap_validity > umap_validity:
        print(f"\n  ★ Winner: PaCMAP (+{validity_delta:.1f}% validity)")
    elif umap_validity > pacmap_validity:
        print(f"\n  ★ Winner: UMAP (+{-validity_delta:.1f}% validity)")
    else:
        print(f"\n  ★ Tie: Both methods have equal validity")

    # Print PaCMAP best params
    bp = pacmap_result.best_trial
    print(f"\n  PaCMAP best params:")
    print(f"    n_neighbors={bp.n_neighbors}, n_components={bp.n_components}")
    print(f"    MN_ratio={bp.MN_ratio}, FP_ratio={bp.FP_ratio}")
    print(f"    min_cluster_size={bp.min_cluster_size}, min_samples={bp.min_samples}")


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
        embeddings_models: list of EmbeddingsModel objects (for pipeline compatibility)
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
# MAIN
# =============================================================================

def main():
    """Run ClustererV2 on the configured dataset with optional UMAP and/or PaCMAP."""
    print("=" * 70)
    print("ClustererV2 Pipeline (UMAP/PaCMAP Comparison)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"UMAP: {'Enabled' if ENABLE_UMAP else 'Disabled'}")
    print(f"PaCMAP: {'Enabled' if ENABLE_PACMAP else 'Disabled'}")
    print()

    # Load embeddings
    embeddings, idea_texts, embeddings_models = load_step4_embeddings()

    # ==========================================================================
    # PART 1: Run standard UMAP-based ClustererV2 (if enabled)
    # ==========================================================================
    clusterer = None
    umap_metrics = {}

    if ENABLE_UMAP:
        print("\n" + "=" * 70)
        print("PART 1: UMAP-based Clustering (Standard ClustererV2)")
        print("=" * 70)

        clusterer = ClustererV2(embeddings_models, config=CONFIG)
        clusterer.run()

        # Extract UMAP metrics for comparison
        metrics = clusterer.get_metrics()
        if metrics:
            umap_metrics['n_clusters'] = metrics.n_clusters
            umap_metrics['noise_rate'] = metrics.noise_rate
            umap_metrics['coherence'] = metrics.mean_coherence
            umap_metrics['dbcv'] = metrics.dbcv if metrics.dbcv is not None else 0
            umap_metrics['relative_validity'] = metrics.dbcv if metrics.dbcv is not None else 0

        # Print UMAP summary
        print(f"\n--- UMAP Results ---")
        rec = clusterer.get_algorithm_recommendation()
        if rec:
            print(f"Algorithm: {rec.recommended_algorithm} ({rec.confidence} confidence)")
            print(f"DVC: {rec.dvc_value:.3f}")

        if metrics:
            print(f"Clusters: {metrics.n_clusters}")
            print(f"Noise: {metrics.noise_count} ({metrics.noise_rate:.1%})")
            print(f"Coherence: {metrics.mean_coherence:.3f}")
            if metrics.dbcv is not None:
                print(f"DBCV: {metrics.dbcv:.3f}")
    else:
        print("\n[Skipping UMAP - disabled]")

    # ==========================================================================
    # PART 2: Run PaCMAP (if enabled)
    # ==========================================================================
    pacmap_result = None
    if ENABLE_PACMAP:
        print("\n" + "=" * 70)
        print("PART 2: PaCMAP-based Clustering")
        print("=" * 70)

        # Prepare embeddings for PaCMAP
        embeddings_normalized = l2_normalize(embeddings)

        # Run PaCMAP grid search
        pacmap_result = run_pacmap_grid_search(
            embeddings_normalized,
            embeddings_normalized,  # Use same for coherence
            verbose=True
        )

        # Print comparison (only if UMAP was also run)
        if ENABLE_UMAP:
            print_comparison_summary(umap_metrics, pacmap_result)

    # ==========================================================================
    # PART 3: Detailed UMAP results (if UMAP was run)
    # ==========================================================================
    if ENABLE_UMAP and clusterer is not None:
        print("\n" + "=" * 70)
        print("DETAILED RESULTS (UMAP-based)")
        print("=" * 70)

        # Keywords
        keywords = clusterer.get_cluster_keywords()
        if keywords:
            print(f"\nc-TF-IDF Keywords ({len(keywords)} clusters):")
            for cluster_id in sorted(keywords.keys()):
                kw_list = keywords[cluster_id]
                kw_str = ", ".join([kw for kw, _ in kw_list[:5]])
                print(f"  Cluster {cluster_id}: {kw_str}")

        # Print ALL clusters with samples
        clusterer.print_all_clusters(n_samples=10)

    # Return for further analysis
    return clusterer, embeddings_models, pacmap_result


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


def save_results_to_file(output: str, filename: str, variable: str, sample_size: Optional[int]) -> Path:
    """
    Save clustering results to a text file.

    Args:
        output: The captured console output
        filename: Original data filename
        variable: Variable name
        sample_size: Sample size (or None)

    Returns:
        Path to the saved file
    """
    # Create output directory
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: cluster_results_filename_variable_samplesize_YYYYMMDD_pacmap.txt
    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d")
    suffix = "_pacmap" if ENABLE_PACMAP else ""

    output_filename = f"cluster_results_{base_name}_{variable}_{sample_str}_{date_str}{suffix}.txt"
    output_path = output_dir / output_filename

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        clusterer, embeddings_models, pacmap_result = main()
    finally:
        # Restore stdout
        sys.stdout = tee.original_stdout

    # Save results to file
    output_path = save_results_to_file(
        output=tee.get_output(),
        filename=FILENAME,
        variable=VARIABLE,
        sample_size=SAMPLE_SIZE
    )
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")

# %%
