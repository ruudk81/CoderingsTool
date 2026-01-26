"""
ClustererV4 Parameter Optimizer Module (PaCMAP)

Replaces UMAP with PaCMAP for dimensionality reduction.
Uses Optuna GridSampler for hyperparameter optimization of both PaCMAP and HDBSCAN.

Key differences from parameter_optimizer.py (UMAP version):
- Uses PaCMAP instead of UMAP for dimensionality reduction
- Includes PaCMAP-specific parameters: MN_ratio, FP_ratio
- Uses sklearn kNN to bypass Annoy bug on macOS + Python 3.12+
- Pre-computes PaCMAP reductions for all (n_neighbors, MN_ratio, FP_ratio, n_components) combos

Composite scoring (same as UMAP version):
    score = w_validity * relative_validity
            - lambda_low_prob * max(0, low_prob_ratio - tau)
            - lambda_fuzzy * fuzzy_cluster_ratio
            - lambda_fuzzy_count * fuzzy_cluster_fraction
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import optuna
from optuna.samplers import GridSampler
import hdbscan
import pacmap
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from .config_v4 import ClustererV4Config
from .algorithm_selector import AlgorithmSelector

# Suppress warnings during optimization
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")
warnings.filterwarnings("ignore", message="random state is set")


@dataclass
class OptunaResultPaCMAP:
    """Result container for Optuna optimization with PaCMAP."""
    best_params: Dict[str, Any]
    best_value: float
    best_labels: np.ndarray
    best_model: hdbscan.HDBSCAN
    n_trials_completed: int
    n_trials_pruned: int
    study: optuna.Study
    pacmap_embeddings: np.ndarray  # Changed from umap_embeddings
    search_space: Dict[str, List]
    persistence_metrics: Dict[str, float]


def log_spaced_ints(low: int, high: int, k: int = 4) -> List[int]:
    """Generate k log-spaced integers between low and high."""
    if low == high:
        return [low]
    if low <= 0:
        low = 1
    vals = np.exp(np.linspace(np.log(low), np.log(high), k))
    return sorted(set(int(round(v)) for v in vals))


def n_neighbors_grid_pacmap(N: int, k: int = 3) -> List[int]:
    """
    Generate n_neighbors grid for PaCMAP.

    Uses fixed range [10, 30] with k log-spaced points.
    """
    low = 10
    high = 30
    high = min(high, N - 1)
    low = min(low, high)
    return log_spaced_ints(low, high, k=k)


def mcs_bounds_sqrt(N: int) -> Tuple[int, int]:
    """Compute min_cluster_size bounds based on log(N) and sqrt(N)."""
    low = max(3, int(round(min(0.5 * math.log(N), 5))))
    high = max(low, int(round(math.sqrt(N))))
    return low, high


def mcs_grid_sqrt(N: int, k: int = 4) -> List[int]:
    """Generate min_cluster_size grid for dataset of size N."""
    low, high = mcs_bounds_sqrt(N)
    return log_spaced_ints(low, high, k=k)


def create_search_space_pacmap(
    N: int,
    config: 'ClustererV4Config'
) -> Dict[str, List]:
    """
    Create Optuna search space dict for PaCMAP + HDBSCAN GridSampler.

    Args:
        N: Dataset size
        config: ClustererV4Config with PaCMAP grids

    Returns:
        Dict with 'n_neighbors', 'MN_ratio', 'FP_ratio', 'n_components', 'min_cluster_size'
    """
    return {
        'n_neighbors': list(config.pacmap_n_neighbors_grid),
        'MN_ratio': list(config.pacmap_mn_ratio_grid),
        'FP_ratio': list(config.pacmap_fp_ratio_grid),
        'n_components': list(config.pacmap_n_components_grid),
        'min_cluster_size': (
            list(config.min_cluster_size_grid)
            if config.min_cluster_size_grid
            else mcs_grid_sqrt(N, k=config.min_cluster_size_grid_k)
        ),
    }


def compute_knn_pairs(embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
    """
    Compute k-nearest neighbor pairs using sklearn (bypasses Annoy bug).

    This function solves the known PaCMAP bug on macOS + Python 3.12+ where
    the Annoy library fails silently.

    Args:
        embeddings: (N, D) array of embeddings
        n_neighbors: number of neighbors to find

    Returns:
        pair_neighbors: (N * n_neighbors, 2) array for PaCMAP
    """
    N = len(embeddings)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', algorithm='auto')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    neighbor_indices = indices[:, 1:]  # Remove self

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

    Args:
        embeddings: L2-normalized embeddings (N x D)
        n_neighbors: Number of neighbors for local structure
        n_components: Target dimensionality
        MN_ratio: Mid-near pairs ratio
        FP_ratio: Further pairs ratio
        random_state: Random seed
        apply_pca: Whether to apply PCA before neighbor search

    Returns:
        Reduced embeddings (N x n_components)
    """
    N = len(embeddings)

    # PaCMAP needs approximately n_neighbors * (1 + MN_ratio + FP_ratio) unique neighbors
    total_neighbors_needed = int(n_neighbors * (1 + MN_ratio + FP_ratio)) + 1

    if total_neighbors_needed >= N:
        max_n_neighbors = int(N / (1 + MN_ratio + FP_ratio)) - 1
        max_n_neighbors = max(5, max_n_neighbors)
        if max_n_neighbors < n_neighbors:
            n_neighbors = max_n_neighbors

    # Compute nearest neighbors using sklearn (bypasses Annoy bug)
    pair_neighbors = compute_knn_pairs(embeddings, n_neighbors)

    reducer = pacmap.PaCMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        pair_neighbors=pair_neighbors,
        random_state=random_state,
        verbose=False,
        apply_pca=apply_pca
    )

    return reducer.fit_transform(embeddings)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    return normalize(embeddings, norm='l2', axis=1)


class ParameterOptimizerPaCMAP:
    """
    Optuna-based hyperparameter optimization for PaCMAP + HDBSCAN.

    Features:
    - GridSampler for exhaustive search over PaCMAP and HDBSCAN params
    - Pre-computed PaCMAP reductions for efficiency
    - Constraint-based pruning (noise rate, min clusters)
    - Maximizes composite score (validity - penalties)
    """

    def __init__(
        self,
        config: ClustererV4Config,
        embeddings: np.ndarray,
        original_embeddings: np.ndarray,
        verbose: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            config: ClustererV4Config
            embeddings: L2-normalized embeddings for PaCMAP
            original_embeddings: Original embeddings for coherence
            verbose: Print progress
        """
        self.config = config
        self._embeddings = embeddings
        self._original_embeddings = original_embeddings
        self._verbose = verbose
        self._N = len(embeddings)

        self._search_space: Dict[str, List] = {}
        self._pacmap_cache: Dict[Tuple, np.ndarray] = {}  # Key: (n_neighbors, MN_ratio, FP_ratio, n_components)
        self._study: Optional[optuna.Study] = None
        self._best_result: Optional[Dict[str, Any]] = None
        self._selector = AlgorithmSelector(config)

    def precompute_pacmap_reductions(self) -> Dict[Tuple, np.ndarray]:
        """
        Pre-compute PaCMAP reductions for all parameter combinations in parallel.

        Returns:
            Dict mapping (n_neighbors, MN_ratio, FP_ratio, n_components) -> L2-normalized reduced embeddings
        """
        combinations = [
            (nn, mn, fp, nc)
            for nn in self._search_space['n_neighbors']
            for mn in self._search_space['MN_ratio']
            for fp in self._search_space['FP_ratio']
            for nc in self._search_space['n_components']
        ]

        if self._verbose:
            print(f"  Pre-computing {len(combinations)} PaCMAP reductions in parallel...")
            print(f"    n_neighbors: {self._search_space['n_neighbors']}")
            print(f"    MN_ratio: {self._search_space['MN_ratio']}")
            print(f"    FP_ratio: {self._search_space['FP_ratio']}")
            print(f"    n_components: {self._search_space['n_components']}")

        def compute_single_pacmap(n_neighbors: int, mn_ratio: float, fp_ratio: float, n_components: int):
            reduced = run_pacmap(
                self._embeddings,
                n_neighbors,
                n_components,
                MN_ratio=mn_ratio,
                FP_ratio=fp_ratio,
                random_state=self.config.pacmap_random_state,
                apply_pca=self.config.pacmap_apply_pca
            )
            reduced_normalized = l2_normalize(reduced)
            return (n_neighbors, mn_ratio, fp_ratio, n_components), reduced_normalized

        n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else -1
        results = Parallel(n_jobs=n_jobs, verbose=1 if self._verbose else 0)(
            delayed(compute_single_pacmap)(nn, mn, fp, nc) for nn, mn, fp, nc in combinations
        )

        return {key: reduced for key, reduced in results}

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Get grid parameters
        n_neighbors = trial.suggest_categorical('n_neighbors', self._search_space['n_neighbors'])
        mn_ratio = trial.suggest_categorical('MN_ratio', self._search_space['MN_ratio'])
        fp_ratio = trial.suggest_categorical('FP_ratio', self._search_space['FP_ratio'])
        n_components = trial.suggest_categorical('n_components', self._search_space['n_components'])
        min_cluster_size = trial.suggest_categorical('min_cluster_size', self._search_space['min_cluster_size'])
        min_samples = max(1, min_cluster_size // 2)

        if self._verbose:
            print(f"  Trial {trial.number}: nn={n_neighbors}, MN={mn_ratio}, FP={fp_ratio}, "
                  f"nc={n_components}, mcs={min_cluster_size}")

        # Look up pre-computed PaCMAP reduction
        reduced_normalized = self._pacmap_cache[(n_neighbors, mn_ratio, fp_ratio, n_components)]

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        labels = clusterer.fit_predict(reduced_normalized)

        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).sum() / len(labels)

        # Check constraints
        if n_clusters < self.config.min_clusters:
            if self._verbose:
                print(f"    PRUNED: Too few clusters ({n_clusters})")
            raise optuna.TrialPruned(f"Too few clusters: {n_clusters}")

        if noise_rate > self.config.max_noise_rate:
            if self._verbose:
                print(f"    PRUNED: Noise too high ({noise_rate:.1%})")
            raise optuna.TrialPruned(f"Noise too high: {noise_rate:.1%}")

        # Get relative_validity_
        try:
            relative_validity = clusterer.relative_validity_
        except AttributeError:
            relative_validity = self._compute_dbcv(labels, reduced_normalized)

        # Extract metrics
        persistence_metrics = self._selector.extract_persistence_metrics(clusterer, labels)
        coherence = self._calculate_coherence(labels, self._original_embeddings)
        prob_metrics = self._compute_probability_metrics(clusterer.probabilities_, labels)
        outlier_metrics = self._compute_outlier_metrics(clusterer.outlier_scores_)

        # Compute composite score
        composite_score, score_breakdown = self._compute_composite_score(
            relative_validity,
            prob_metrics['low_prob_ratio'],
            prob_metrics['fuzzy_cluster_ratio'],
            prob_metrics['n_fuzzy_clusters'],
            n_clusters
        )

        # Log user attributes
        trial.set_user_attr('n_clusters', n_clusters)
        trial.set_user_attr('noise_rate', noise_rate)
        trial.set_user_attr('coherence', coherence)
        trial.set_user_attr('min_samples', min_samples)
        trial.set_user_attr('relative_validity', relative_validity)
        trial.set_user_attr('mean_persistence', persistence_metrics.get('mean_persistence', np.nan))
        trial.set_user_attr('weighted_persistence', persistence_metrics.get('weighted_persistence', np.nan))
        trial.set_user_attr('mean_probability', prob_metrics['mean_probability'])
        trial.set_user_attr('low_prob_ratio', prob_metrics['low_prob_ratio'])
        trial.set_user_attr('fuzzy_cluster_ratio', prob_metrics['fuzzy_cluster_ratio'])
        trial.set_user_attr('n_fuzzy_clusters', prob_metrics['n_fuzzy_clusters'])
        trial.set_user_attr('mean_outlier_score', outlier_metrics['mean_outlier_score'])
        trial.set_user_attr('high_outlier_ratio', outlier_metrics['high_outlier_ratio'])
        trial.set_user_attr('composite_score', composite_score)

        if self._verbose:
            print(f"    -> COMPOSITE={composite_score:.4f} "
                  f"(val={score_breakdown['validity_term']:.3f} - "
                  f"pen_low={score_breakdown['penalty_low_prob']:.3f} - "
                  f"pen_fuzzy={score_breakdown['penalty_fuzzy']:.3f})")
            print(f"    -> rel_validity={relative_validity:.4f}, k={n_clusters}, "
                  f"noise={noise_rate:.1%}, coh={coherence:.3f}")

        return composite_score

    def _compute_dbcv(self, labels: np.ndarray, embeddings: np.ndarray) -> float:
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

    def _calculate_coherence(self, labels: np.ndarray, embeddings: np.ndarray) -> float:
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

    def _compute_probability_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Extract metrics from HDBSCAN probabilities_."""
        mask = labels >= 0
        probs = probabilities[mask]
        labels_clustered = labels[mask]

        if len(probs) == 0:
            return {
                'mean_probability': 0.0,
                'low_prob_ratio': 0.0,
                'fuzzy_cluster_ratio': 0.0,
                'n_fuzzy_clusters': 0
            }

        mean_prob = float(np.mean(probs))
        low_prob_ratio = float((probs < self.config.low_probability_threshold).sum() / len(probs))

        fuzzy_threshold = self.config.fuzzy_cluster_threshold
        points_in_fuzzy = 0
        n_fuzzy = 0

        for label in set(labels_clustered):
            cluster_mask = labels == label
            cluster_probs = probabilities[cluster_mask]
            cluster_low_ratio = (cluster_probs < self.config.low_probability_threshold).sum() / len(cluster_probs)

            if cluster_low_ratio > fuzzy_threshold:
                points_in_fuzzy += len(cluster_probs)
                n_fuzzy += 1

        fuzzy_cluster_ratio = points_in_fuzzy / len(probs) if len(probs) > 0 else 0.0

        return {
            'mean_probability': mean_prob,
            'low_prob_ratio': low_prob_ratio,
            'fuzzy_cluster_ratio': float(fuzzy_cluster_ratio),
            'n_fuzzy_clusters': n_fuzzy
        }

    def _compute_outlier_metrics(self, outlier_scores: np.ndarray) -> Dict[str, float]:
        """Extract metrics from HDBSCAN outlier_scores_."""
        if len(outlier_scores) == 0:
            return {'mean_outlier_score': 0.0, 'high_outlier_ratio': 0.0}

        mean_score = float(np.mean(outlier_scores))
        high_ratio = float((outlier_scores > self.config.high_outlier_threshold).sum() / len(outlier_scores))

        return {
            'mean_outlier_score': mean_score,
            'high_outlier_ratio': high_ratio
        }

    def _compute_composite_score(
        self,
        relative_validity: float,
        low_prob_ratio: float,
        fuzzy_cluster_ratio: float,
        n_fuzzy_clusters: int,
        n_clusters: int
    ) -> Tuple[float, Dict[str, float]]:
        """Compute composite score."""
        w_validity = self.config.weight_validity
        tau = self.config.tau_low_prob
        lam_low_prob = self.config.lambda_low_prob
        lam_fuzzy = self.config.lambda_fuzzy
        lam_fuzzy_count = self.config.lambda_fuzzy_count

        validity_term = w_validity * relative_validity
        excess_low_prob = max(0.0, low_prob_ratio - tau)
        penalty_low_prob = lam_low_prob * excess_low_prob
        penalty_fuzzy = lam_fuzzy * fuzzy_cluster_ratio
        fuzzy_cluster_fraction = n_fuzzy_clusters / n_clusters if n_clusters > 0 else 0.0
        penalty_fuzzy_count = lam_fuzzy_count * fuzzy_cluster_fraction

        total_penalty = penalty_low_prob + penalty_fuzzy + penalty_fuzzy_count
        composite = validity_term - total_penalty

        breakdown = {
            'validity_term': validity_term,
            'penalty_low_prob': penalty_low_prob,
            'penalty_fuzzy': penalty_fuzzy,
            'penalty_fuzzy_count': penalty_fuzzy_count,
            'total_penalty': total_penalty,
            'excess_low_prob': excess_low_prob,
            'fuzzy_cluster_ratio': fuzzy_cluster_ratio,
            'fuzzy_cluster_fraction': fuzzy_cluster_fraction,
        }

        return composite, breakdown

    def optimize(self) -> OptunaResultPaCMAP:
        """
        Run Optuna grid search optimization.

        Returns:
            OptunaResultPaCMAP with best configuration and metrics
        """
        if self._verbose:
            print(f"\n[Optuna PaCMAP] Starting optimization (N={self._N})")

        # Create search space
        self._search_space = create_search_space_pacmap(self._N, self.config)

        n_pacmap_combos = (
            len(self._search_space['n_neighbors']) *
            len(self._search_space['MN_ratio']) *
            len(self._search_space['FP_ratio']) *
            len(self._search_space['n_components'])
        )
        n_trials = n_pacmap_combos * len(self._search_space['min_cluster_size'])

        if self._verbose:
            print(f"  PaCMAP combinations: {n_pacmap_combos}")
            print(f"  min_cluster_size grid: {self._search_space['min_cluster_size']}")
            print(f"  Total trials: {n_trials}")

        # Pre-compute PaCMAP reductions
        if self.config.precompute_pacmap:
            self._pacmap_cache = self.precompute_pacmap_reductions()
            if self._verbose:
                print(f"  Cached {len(self._pacmap_cache)} PaCMAP reductions")

        # Create and run Optuna study
        sampler = GridSampler(self._search_space)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study = optuna.create_study(
            study_name=f"clusterer_v4_pacmap_{id(self)}",
            direction='maximize',
            sampler=sampler,
        )

        self._study.optimize(self._objective, n_trials=None)

        # Get best trial
        best = self._study.best_trial
        n_neighbors = best.params['n_neighbors']
        mn_ratio = best.params['MN_ratio']
        fp_ratio = best.params['FP_ratio']
        n_components = best.params['n_components']
        min_cluster_size = best.params['min_cluster_size']
        min_samples = max(1, min_cluster_size // 2)

        # Re-run best configuration
        reduced_normalized = self._pacmap_cache[(n_neighbors, mn_ratio, fp_ratio, n_components)]

        best_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        best_labels = best_clusterer.fit_predict(reduced_normalized)

        persistence_metrics = self._selector.extract_persistence_metrics(best_clusterer, best_labels)

        completed = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if self._verbose:
            print(f"\n[Optuna PaCMAP] Optimization complete")
            print(f"  Best: nn={n_neighbors}, MN={mn_ratio}, FP={fp_ratio}, nc={n_components}, "
                  f"mcs={min_cluster_size}, ms={min_samples}")
            print(f"  Composite score: {best.value:.4f}")
            print(f"  Trials: {completed} completed, {pruned} pruned")

        result = OptunaResultPaCMAP(
            best_params={
                'n_neighbors': n_neighbors,
                'MN_ratio': mn_ratio,
                'FP_ratio': fp_ratio,
                'n_components': n_components,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
            },
            best_value=best.value,
            best_labels=best_labels,
            best_model=best_clusterer,
            n_trials_completed=completed,
            n_trials_pruned=pruned,
            study=self._study,
            pacmap_embeddings=reduced_normalized,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics
        )

        # Quality check and conditional re-search
        result = self._check_quality_and_research(result)

        self._best_result = result
        return result

    def get_best_result(self) -> Optional[OptunaResultPaCMAP]:
        """Get the best result from optimization."""
        return self._best_result

    def _check_quality_and_research(self, result: OptunaResultPaCMAP) -> OptunaResultPaCMAP:
        """Check quality of optimization result and trigger re-search if needed."""
        if not self.config.enable_research:
            return result

        best_trial = self._study.best_trial
        n_clusters = best_trial.user_attrs.get('n_clusters', 0)
        noise_rate = best_trial.user_attrs.get('noise_rate', 0.0)
        relative_validity = best_trial.user_attrs.get('relative_validity', result.best_value)

        sqrt_n = math.sqrt(self._N)
        max_noise = self.config.research_max_noise_rate
        min_validity = self.config.research_min_validity
        cluster_deviation_threshold = self.config.research_cluster_deviation_threshold

        cluster_deviation = abs(n_clusters - sqrt_n) / sqrt_n if sqrt_n > 0 else 0.0

        needs_research = False
        reasons = []

        if noise_rate > max_noise and relative_validity < min_validity:
            needs_research = True
            reasons.append(f"noise={noise_rate:.1%}>{max_noise:.0%} AND rel_validity={relative_validity:.3f}<{min_validity}")

        if cluster_deviation > cluster_deviation_threshold:
            needs_research = True
            reasons.append(f"cluster_deviation={cluster_deviation:.1%}>{cluster_deviation_threshold:.0%}")

        if not needs_research:
            if self._verbose:
                print(f"  Quality check PASSED: k={n_clusters}, noise={noise_rate:.1%}, "
                      f"rel_validity={relative_validity:.3f}")
            return result

        if self._verbose:
            print(f"\n[Research] Quality check FAILED: {', '.join(reasons)}")
            print(f"  Triggering extended search...")

        return self._run_extended_search(result)

    def _run_extended_search(self, initial_result: OptunaResultPaCMAP) -> OptunaResultPaCMAP:
        """Run extended search with expanded HDBSCAN parameters."""
        best_n_neighbors = initial_result.best_params['n_neighbors']
        best_mn_ratio = initial_result.best_params['MN_ratio']
        best_fp_ratio = initial_result.best_params['FP_ratio']
        best_n_components = initial_result.best_params['n_components']
        best_mcs = initial_result.best_params['min_cluster_size']
        best_ms = initial_result.best_params.get('min_samples', best_mcs // 2)
        reduced_normalized = self._pacmap_cache[(best_n_neighbors, best_mn_ratio, best_fp_ratio, best_n_components)]

        # Build extended search space (same as UMAP version)
        mcs_multipliers = self.config.research_mcs_multipliers
        mcs_options = sorted(set(max(3, int(round(best_mcs * mult))) for mult in mcs_multipliers))

        ms_low_mult, ms_high_mult = self.config.research_ms_range_multipliers
        ms_low = max(1, int(round(best_ms * ms_low_mult)))
        ms_high = max(ms_low, int(round(best_ms * ms_high_mult)))
        ms_options = log_spaced_ints(ms_low, ms_high, k=self.config.research_ms_grid_k)

        selection_methods = list(self.config.research_selection_methods)
        max_mcs = max(mcs_options)
        ms_options = [ms for ms in ms_options if ms <= max_mcs]

        extended_search_space = {
            'min_cluster_size': mcs_options,
            'min_samples': ms_options,
            'cluster_selection_method': selection_methods,
        }

        if self._verbose:
            print(f"\n[Extended Search - PaCMAP]")
            print(f"  Based on best: nn={best_n_neighbors}, MN={best_mn_ratio}, FP={best_fp_ratio}")
            print(f"  MCS grid: {mcs_options}")
            print(f"  MS grid: {ms_options}")

        def extended_objective(trial: optuna.Trial) -> float:
            mcs = trial.suggest_categorical('min_cluster_size', extended_search_space['min_cluster_size'])
            ms = trial.suggest_categorical('min_samples', extended_search_space['min_samples'])
            method = trial.suggest_categorical('cluster_selection_method', extended_search_space['cluster_selection_method'])

            if ms > mcs:
                raise optuna.TrialPruned(f"Invalid: ms={ms} > mcs={mcs}")

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric='euclidean',
                cluster_selection_method=method,
                gen_min_span_tree=True,
            )
            labels = clusterer.fit_predict(reduced_normalized)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_rate = (labels == -1).sum() / len(labels)

            if n_clusters < self.config.min_clusters:
                raise optuna.TrialPruned(f"Too few clusters: {n_clusters}")
            if noise_rate > self.config.max_noise_rate:
                raise optuna.TrialPruned(f"Noise too high: {noise_rate:.1%}")

            try:
                validity = clusterer.relative_validity_
            except AttributeError:
                validity = self._compute_dbcv(labels, reduced_normalized)

            prob_metrics = self._compute_probability_metrics(clusterer.probabilities_, labels)
            composite_score, _ = self._compute_composite_score(
                validity,
                prob_metrics['low_prob_ratio'],
                prob_metrics['fuzzy_cluster_ratio'],
                prob_metrics['n_fuzzy_clusters'],
                n_clusters
            )

            trial.set_user_attr('n_clusters', n_clusters)
            trial.set_user_attr('noise_rate', noise_rate)
            trial.set_user_attr('relative_validity', validity)

            if self._verbose:
                improved = "*" if composite_score > initial_result.best_value else " "
                print(f"  {improved} {method}, mcs={mcs}, ms={ms} -> COMPOSITE={composite_score:.4f}, k={n_clusters}")

            return composite_score

        extended_sampler = GridSampler(extended_search_space)
        extended_study = optuna.create_study(
            study_name=f"clusterer_v4_extended_{id(self)}",
            direction='maximize',
            sampler=extended_sampler,
        )
        extended_study.optimize(extended_objective, n_trials=None)

        completed = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if completed == 0:
            if self._verbose:
                print(f"  No valid trials found, keeping initial result")
            return initial_result

        best_extended = extended_study.best_trial

        if best_extended.value <= initial_result.best_value:
            if self._verbose:
                print(f"  No improvement found")
            return initial_result

        mcs = best_extended.params['min_cluster_size']
        ms = best_extended.params['min_samples']
        method = best_extended.params['cluster_selection_method']

        best_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric='euclidean',
            cluster_selection_method=method,
            gen_min_span_tree=True,
        )
        best_labels = best_clusterer.fit_predict(reduced_normalized)
        persistence_metrics = self._selector.extract_persistence_metrics(best_clusterer, best_labels)

        if self._verbose:
            print(f"  Found better: {method}, mcs={mcs}, ms={ms}, composite={best_extended.value:.4f}")

        return OptunaResultPaCMAP(
            best_params={
                'n_neighbors': best_n_neighbors,
                'MN_ratio': best_mn_ratio,
                'FP_ratio': best_fp_ratio,
                'n_components': best_n_components,
                'min_cluster_size': mcs,
                'min_samples': ms,
                'cluster_selection_method': method,
            },
            best_value=best_extended.value,
            best_labels=best_labels,
            best_model=best_clusterer,
            n_trials_completed=initial_result.n_trials_completed + completed,
            n_trials_pruned=initial_result.n_trials_pruned + pruned,
            study=self._study,
            pacmap_embeddings=reduced_normalized,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics
        )
