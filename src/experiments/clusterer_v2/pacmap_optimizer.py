"""
PaCMAP Grid Search Optimizer with Zadu Metrics

Implements exhaustive grid search over PaCMAP hyperparameters using DR quality metrics:
- Trustworthiness (local): Are close points in reduced space also close in original?
- Continuity (local): Are close points in original space still close in reduced?
- Steadiness (cluster): Are clusters in projected space also in original?
- Cohesiveness (cluster): Are clusters in original space preserved in projected?
- Stress (global): How well are pairwise distances preserved? (lower is better)

Scoring formula:
    summed_score = (trust_norm + continuity_norm + steadiness_norm + cohesiveness_norm - stress_norm) / 5

Usage:
    optimizer = PaCMAPOptimizer(config, embeddings)
    result = optimizer.optimize()
    best_reduction = result.best_result.reduced_embeddings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import pacmap

# Zadu imports for DR quality metrics
from zadu.measures import trustworthiness_continuity, steadiness_cohesiveness
from zadu.measures import stress as stress_measure

# Suppress PaCMAP warnings
warnings.filterwarnings("ignore", message="random state is set")
warnings.filterwarnings("ignore", message="n_components")


@dataclass
class PaCMAPGridConfig:
    """Configuration for PaCMAP grid search optimization."""

    # PaCMAP parameter grid
    n_neighbors_grid: Tuple[int, ...] = (10, 15, 20, 30)
    mn_ratio_grid: Tuple[float, ...] = (0.3, 0.5, 0.7)
    fp_ratio_grid: Tuple[float, ...] = (1.0, 2.0, 3.0)
    n_components: int = 10  # Fixed (configurable, BERTopic suggests 5)

    # PaCMAP settings
    random_state: int = 42
    apply_pca: bool = True  # Apply PCA before neighbor search (helps with high-dim)

    # Zadu metric parameters
    zadu_k: int = 15  # k for trustworthiness/continuity neighborhood

    # Performance optimization
    n_jobs: int = -1  # Parallel jobs (-1 = all cores)
    metric_subsample: Optional[int] = 500  # Subsample size for zadu metrics (None = use all)


@dataclass
class PaCMAPResult:
    """Result container for a single PaCMAP configuration."""

    # Parameters
    n_neighbors: int
    mn_ratio: float
    fp_ratio: float
    n_components: int

    # Raw zadu metrics
    trustworthiness: float
    continuity: float
    steadiness: float
    cohesiveness: float
    stress: float

    # Normalized scores (after MinMax scaling)
    trust_norm: float = 0.0
    continuity_norm: float = 0.0
    steadiness_norm: float = 0.0
    cohesiveness_norm: float = 0.0
    stress_norm: float = 0.0

    # Composite score
    summed_score: float = 0.0

    # The actual reduction
    reduced_embeddings: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class PaCMAPGridSearchResult:
    """Result container for full grid search."""

    best_result: PaCMAPResult
    all_results: List[PaCMAPResult]
    results_df: pd.DataFrame  # Full results as DataFrame for export
    n_configurations: int
    search_space: Dict[str, List]


def compute_knn_pairs(embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
    """
    Compute k-nearest neighbor pairs using sklearn (bypasses Annoy bug).

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
    N = len(embeddings)

    # Add 1 because sklearn includes the point itself as the first neighbor
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', algorithm='auto')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Remove self (first column) to get actual neighbors
    neighbor_indices = indices[:, 1:]  # Shape: (N, n_neighbors)

    # Convert to PaCMAP's expected format: (N * n_neighbors, 2) pairs of (source, neighbor)
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
        apply_pca: Whether to apply PCA before neighbor search

    Returns:
        Reduced embeddings (N x n_components)
    """
    N = len(embeddings)

    # PaCMAP needs approximately n_neighbors * (1 + MN_ratio + FP_ratio) unique neighbors
    # If this exceeds the dataset size, we need to reduce n_neighbors
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


def compute_zadu_metrics(
    original: np.ndarray,
    reduced: np.ndarray,
    k: int = 15,
    subsample: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute DR quality metrics using zadu library.

    Args:
        original: Original high-dimensional embeddings
        reduced: Reduced low-dimensional embeddings
        k: Neighborhood size for local metrics
        subsample: If set, use random subsample for faster computation
        random_state: Random seed for subsampling

    Returns:
        Dict with: trustworthiness, continuity, steadiness, cohesiveness, stress
    """
    # Subsample for faster computation if requested
    if subsample is not None and len(original) > subsample:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(original), size=subsample, replace=False)
        original = original[indices]
        reduced = reduced[indices]

    # Local structure metrics
    trust_sc = trustworthiness_continuity.measure(original, reduced, k=k)
    trust = trust_sc["trustworthiness"]
    continuity = trust_sc["continuity"]

    # Cluster structure metrics (slowest - benefits most from subsampling)
    steadiness_sc = steadiness_cohesiveness.measure(original, reduced, k=k)
    steadiness = steadiness_sc["steadiness"]
    cohesiveness = steadiness_sc["cohesiveness"]

    # Global structure metric
    stress_sc = stress_measure.measure(original, reduced)
    stress = stress_sc["stress"]

    return {
        "trustworthiness": trust,
        "continuity": continuity,
        "steadiness": steadiness,
        "cohesiveness": cohesiveness,
        "stress": stress
    }


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings to unit vectors."""
    return normalize(embeddings, norm='l2', axis=1)


class PaCMAPOptimizer:
    """
    Grid search optimizer for PaCMAP dimensionality reduction.

    Evaluates all configurations using zadu library metrics and selects
    the best one based on summed_score.

    Usage:
        optimizer = PaCMAPOptimizer(config, embeddings)
        result = optimizer.optimize()
        best_reduction = result.best_result.reduced_embeddings
    """

    def __init__(
        self,
        config: PaCMAPGridConfig,
        embeddings: np.ndarray,
        verbose: bool = True
    ):
        """
        Initialize PaCMAP optimizer.

        Args:
            config: PaCMAPGridConfig with parameter grids
            embeddings: Original embeddings (will be L2-normalized internally)
            verbose: Print progress
        """
        self.config = config
        self._embeddings = l2_normalize(embeddings)
        self._verbose = verbose
        self._N = len(embeddings)

    def _compute_single_config(
        self,
        n_neighbors: int,
        mn_ratio: float,
        fp_ratio: float
    ) -> PaCMAPResult:
        """Compute PaCMAP and metrics for a single configuration."""
        # Run PaCMAP on full data
        reduced = run_pacmap(
            self._embeddings,
            n_neighbors=n_neighbors,
            n_components=self.config.n_components,
            MN_ratio=mn_ratio,
            FP_ratio=fp_ratio,
            random_state=self.config.random_state,
            apply_pca=self.config.apply_pca
        )

        # Compute zadu metrics (with optional subsampling for speed)
        metrics = compute_zadu_metrics(
            self._embeddings,
            reduced,
            k=self.config.zadu_k,
            subsample=self.config.metric_subsample,
            random_state=self.config.random_state
        )

        return PaCMAPResult(
            n_neighbors=n_neighbors,
            mn_ratio=mn_ratio,
            fp_ratio=fp_ratio,
            n_components=self.config.n_components,
            trustworthiness=metrics["trustworthiness"],
            continuity=metrics["continuity"],
            steadiness=metrics["steadiness"],
            cohesiveness=metrics["cohesiveness"],
            stress=metrics["stress"],
            reduced_embeddings=reduced
        )

    def _normalize_and_score(self, results: List[PaCMAPResult]) -> List[PaCMAPResult]:
        """
        Normalize all metrics to [0, 1] using MinMaxScaler and compute summed scores.

        Formula: summed_score = (trust + continuity + steadiness + cohesiveness - stress) / 5
        Note: Stress is NOT inverted before subtraction - we subtract the normalized stress
              so that lower stress (better) contributes positively to the score.
        """
        # Extract raw metrics into arrays
        metrics_names = ['trustworthiness', 'continuity', 'steadiness', 'cohesiveness', 'stress']
        raw_metrics = np.array([
            [getattr(r, m) for m in metrics_names]
            for r in results
        ])

        # MinMax normalize each metric to [0, 1]
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(raw_metrics)

        # Update results with normalized scores and compute summed_score
        for i, result in enumerate(results):
            result.trust_norm = normalized[i, 0]
            result.continuity_norm = normalized[i, 1]
            result.steadiness_norm = normalized[i, 2]
            result.cohesiveness_norm = normalized[i, 3]
            result.stress_norm = normalized[i, 4]

            # Compute summed score: subtract stress (lower stress = better)
            result.summed_score = (
                result.trust_norm +
                result.continuity_norm +
                result.steadiness_norm +
                result.cohesiveness_norm -
                result.stress_norm
            ) / 5.0

        return results

    def _results_to_dataframe(self, results: List[PaCMAPResult]) -> pd.DataFrame:
        """Convert results to DataFrame for export."""
        records = []
        for r in results:
            records.append({
                'n_neighbors': r.n_neighbors,
                'MN_ratio': r.mn_ratio,
                'FP_ratio': r.fp_ratio,
                'n_components': r.n_components,
                'trustworthiness': r.trustworthiness,
                'continuity': r.continuity,
                'steadiness': r.steadiness,
                'cohesiveness': r.cohesiveness,
                'stress': r.stress,
                'trust_norm': r.trust_norm,
                'continuity_norm': r.continuity_norm,
                'steadiness_norm': r.steadiness_norm,
                'cohesiveness_norm': r.cohesiveness_norm,
                'stress_norm': r.stress_norm,
                'summed_score': r.summed_score
            })

        df = pd.DataFrame(records)
        df = df.sort_values('summed_score', ascending=False).reset_index(drop=True)
        df.insert(0, 'rank', range(1, len(df) + 1))
        return df

    def optimize(self) -> PaCMAPGridSearchResult:
        """
        Run exhaustive grid search over all PaCMAP configurations.

        Returns:
            PaCMAPGridSearchResult with best configuration and all results
        """
        # Build search space
        search_space = {
            'n_neighbors': list(self.config.n_neighbors_grid),
            'MN_ratio': list(self.config.mn_ratio_grid),
            'FP_ratio': list(self.config.fp_ratio_grid)
        }

        # Generate all combinations
        param_grid = list(ParameterGrid(search_space))
        n_configs = len(param_grid)

        if self._verbose:
            print(f"\n{'='*70}")
            print("PaCMAP Grid Search (zadu metrics)")
            print('='*70)
            print(f"Dataset size: {self._N}")
            print(f"n_components: {self.config.n_components} (fixed)")
            print(f"Metric subsample: {self.config.metric_subsample or 'all'}")
            print(f"Parallel jobs: {self.config.n_jobs}")
            print(f"Search space:")
            print(f"  n_neighbors: {search_space['n_neighbors']}")
            print(f"  MN_ratio: {search_space['MN_ratio']}")
            print(f"  FP_ratio: {search_space['FP_ratio']}")
            print(f"Total configurations: {n_configs}")
            print()

        # Run grid search in parallel
        if self.config.n_jobs != 1:
            if self._verbose:
                print(f"  Running {n_configs} configs in parallel...")

            results = Parallel(n_jobs=self.config.n_jobs, verbose=10 if self._verbose else 0)(
                delayed(self._compute_single_config)(
                    params['n_neighbors'],
                    params['MN_ratio'],
                    params['FP_ratio']
                )
                for params in param_grid
            )

            if self._verbose:
                print(f"\n  Completed {len(results)} configurations")
                for i, result in enumerate(results):
                    params = param_grid[i]
                    print(f"  [{i+1}/{n_configs}] nn={params['n_neighbors']}, "
                          f"MN={params['MN_ratio']}, FP={params['FP_ratio']}: "
                          f"trust={result.trustworthiness:.3f}, "
                          f"cont={result.continuity:.3f}, "
                          f"stead={result.steadiness:.3f}, "
                          f"cohes={result.cohesiveness:.3f}, "
                          f"stress={result.stress:.3f}")
        else:
            # Sequential execution
            results: List[PaCMAPResult] = []
            for i, params in enumerate(param_grid):
                nn = params['n_neighbors']
                mn = params['MN_ratio']
                fp = params['FP_ratio']

                if self._verbose:
                    print(f"  [{i+1}/{n_configs}] nn={nn}, MN={mn}, FP={fp}...", end=" ")

                try:
                    result = self._compute_single_config(nn, mn, fp)
                    results.append(result)

                    if self._verbose:
                        print(f"trust={result.trustworthiness:.3f}, "
                              f"cont={result.continuity:.3f}, "
                              f"stead={result.steadiness:.3f}, "
                              f"cohes={result.cohesiveness:.3f}, "
                              f"stress={result.stress:.3f}")
                except Exception as e:
                    if self._verbose:
                        print(f"FAILED: {e}")
                    raise

        # Normalize metrics and compute summed scores
        if self._verbose:
            print(f"\n  MinMax normalizing metrics...")
            print(f"  Computing summed_score = (trust + cont + stead + cohes - stress) / 5")

        results = self._normalize_and_score(results)

        # Find best configuration
        best_idx = np.argmax([r.summed_score for r in results])
        best_result = results[best_idx]

        if self._verbose:
            print(f"\n  Best PaCMAP configuration:")
            print(f"    n_neighbors={best_result.n_neighbors}, "
                  f"MN_ratio={best_result.mn_ratio}, "
                  f"FP_ratio={best_result.fp_ratio}")
            print(f"    summed_score: {best_result.summed_score:.4f}")
            print(f"    Normalized metrics (used for scoring):")
            print(f"      trust_norm:      {best_result.trust_norm:.4f}")
            print(f"      continuity_norm: {best_result.continuity_norm:.4f}")
            print(f"      steadiness_norm: {best_result.steadiness_norm:.4f}")
            print(f"      cohesiveness_norm: {best_result.cohesiveness_norm:.4f}")
            print(f"      stress_norm:     {best_result.stress_norm:.4f} (subtracted)")
            print(f"    Raw metrics:")
            print(f"      trustworthiness: {best_result.trustworthiness:.4f}")
            print(f"      continuity:      {best_result.continuity:.4f}")
            print(f"      steadiness:      {best_result.steadiness:.4f}")
            print(f"      cohesiveness:    {best_result.cohesiveness:.4f}")
            print(f"      stress:          {best_result.stress:.4f}")

        # Convert to DataFrame
        results_df = self._results_to_dataframe(results)

        return PaCMAPGridSearchResult(
            best_result=best_result,
            all_results=results,
            results_df=results_df,
            n_configurations=n_configs,
            search_space=search_space
        )
