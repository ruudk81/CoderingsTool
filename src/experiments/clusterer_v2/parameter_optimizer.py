"""
Clusterer Parameter Optimizer Module

Implements Optuna-based hyperparameter optimization for HDBSCAN clustering:
- Log-spaced grids for n_neighbors and min_cluster_size
- Pre-computed UMAP reductions for efficiency
- Constraint-based pruning (noise rate, min clusters)
- Soft threshold composite scoring (no persistence):
  score = w_validity * relative_validity
          - λ_low_prob * max(0, low_prob_ratio - τ)
          - λ_fuzzy * fuzzy_cluster_ratio
          - λ_fuzzy_count * fuzzy_cluster_fraction

  Where:
  - low_prob_ratio: global fraction of points with probability < threshold
  - fuzzy_cluster_ratio: fraction of points in "fuzzy" clusters
  - fuzzy_cluster_fraction: n_fuzzy_clusters / n_clusters
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import optuna
from optuna.samplers import GridSampler
import hdbscan
import umap
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

from tqdm.auto import tqdm  # auto-detects Jupyter vs terminal

from .config import ClustererV2Config
from .algorithm_selector import AlgorithmSelector

# Suppress warnings during optimization
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")


@dataclass
class OptunaResult:
    """Result container for Optuna optimization."""
    best_params: Dict[str, Any]
    best_value: float
    best_labels: np.ndarray
    best_model: hdbscan.HDBSCAN
    n_trials_completed: int
    n_trials_pruned: int
    study: optuna.Study
    umap_embeddings: np.ndarray
    search_space: Dict[str, List]
    persistence_metrics: Dict[str, float]


def log_spaced_ints(low: int, high: int, k: int = 4) -> List[int]:
    """
    Generate k log-spaced integers between low and high.

    Args:
        low: Lower bound
        high: Upper bound
        k: Number of grid points

    Returns:
        Sorted list of unique integers
    """
    if low == high:
        return [low]
    if low <= 0:
        low = 1
    vals = np.exp(np.linspace(np.log(low), np.log(high), k))
    return sorted(set(int(round(v)) for v in vals))


def n_neighbors_grid(
    N: int,
    k: int = 3,
    low_mult: float = 0.5,
    high_mult: float = 1.5,
    nn_min: int = 5,
    nn_max: int = 50
) -> List[int]:
    """
    Generate n_neighbors grid based on dataset size.

    Formula: 0.5×√n to 1.5×√n, log-spaced with k points.
    Clamped to [nn_min, nn_max] and [1, N-1].

    Args:
        N: Dataset size
        k: Number of grid points (default 3)
        low_mult: Low bound multiplier for sqrt(N) (default 0.5)
        high_mult: High bound multiplier for sqrt(N) (default 1.5)
        nn_min: Absolute minimum n_neighbors (default 5)
        nn_max: Absolute maximum n_neighbors (default 50)

    Returns:
        Log-spaced list of n_neighbors values

    Examples:
        N=100:  [5, 8, 15]   (0.5×10=5 to 1.5×10=15)
        N=250:  [8, 14, 24]  (0.5×16=8 to 1.5×16=24)
        N=500:  [11, 19, 33] (0.5×22=11 to 1.5×22=33)
        N=1000: [16, 27, 47] (0.5×32=16 to 1.5×32=47)
    """
    sqrt_n = math.sqrt(N)
    low = max(nn_min, int(round(low_mult * sqrt_n)))
    high = min(nn_max, int(round(high_mult * sqrt_n)))

    # Safety: ensure bounds are valid
    high = min(high, N - 1)
    low = min(low, high)

    return log_spaced_ints(low, high, k=k)


def mcs_bounds_sqrt(
    N: int,
    low_mult: float = 0.1,
    high_mult: float = 0.5,
    mcs_min: int = 3
) -> Tuple[int, int]:
    """
    Compute min_cluster_size bounds based on sqrt(N).

    Formula:
        low = max(mcs_min, 0.1 * sqrt(N))
        high = 0.5 * sqrt(N)

    Args:
        N: Dataset size
        low_mult: Low bound multiplier for sqrt(N) (default 0.1)
        high_mult: High bound multiplier for sqrt(N) (default 0.5)
        mcs_min: Absolute minimum MCS (default 3)

    Returns:
        (low, high) bounds for min_cluster_size

    Examples:
        N=100:  (3, 5)   → grid [3, 4, 5]
        N=250:  (3, 8)   → grid [3, 5, 8]
        N=500:  (3, 11)  → grid [3, 6, 11]
        N=1000: (3, 16)  → grid [3, 7, 16]
    """
    sqrt_n = math.sqrt(N)
    low = max(mcs_min, int(round(low_mult * sqrt_n)))
    high = max(low, int(round(high_mult * sqrt_n)))
    return low, high


def mcs_grid_sqrt(
    N: int,
    k: int = 3,
    low_mult: float = 0.1,
    high_mult: float = 0.5,
    mcs_min: int = 3
) -> List[int]:
    """
    Generate min_cluster_size grid for dataset of size N.

    Args:
        N: Dataset size
        k: Number of grid points (default 3)
        low_mult: Low bound multiplier for sqrt(N) (default 0.1)
        high_mult: High bound multiplier for sqrt(N) (default 0.5)
        mcs_min: Absolute minimum MCS (default 3)

    Returns:
        Log-spaced list of min_cluster_size values
    """
    low, high = mcs_bounds_sqrt(N, low_mult, high_mult, mcs_min)
    return log_spaced_ints(low, high, k=k)


def create_search_space(N: int, config: 'ClustererV2Config') -> Dict[str, List]:
    """
    Create Optuna search space dict for GridSampler using config values.

    Args:
        N: Dataset size
        config: ClustererV2Config with grid parameters

    Returns:
        Dict with 'n_neighbors', 'n_components', 'min_dist', and 'min_cluster_size' grids

    Grid formulas (from config):
        n_neighbors: 0.5×√n to 1.5×√n, log-spaced k=3
        n_components: (5, 10)
        min_dist: (0.0, 0.1)
        min_cluster_size: max(3, 0.1×√n) to 0.5×√n, log-spaced k=3
    """
    return {
        'n_neighbors': n_neighbors_grid(
            N,
            k=config.n_neighbors_grid_k,
            low_mult=config.n_neighbors_low_mult,
            high_mult=config.n_neighbors_high_mult,
            nn_min=config.n_neighbors_min,
            nn_max=config.n_neighbors_max
        ),
        'n_components': list(config.umap_n_components_grid),
        'min_dist': list(config.umap_min_dist_grid),
        'min_cluster_size': mcs_grid_sqrt(
            N,
            k=config.min_cluster_size_grid_k,
            low_mult=config.mcs_low_mult,
            high_mult=config.mcs_high_mult,
            mcs_min=config.mcs_min
        ),
    }


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction.

    Args:
        embeddings: L2-normalized embeddings
        n_neighbors: UMAP n_neighbors
        n_components: Target dimensionality
        min_dist: UMAP min_dist
        random_state: Random seed

    Returns:
        UMAP-reduced embeddings
    """
    warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='euclidean',
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    return normalize(embeddings, norm='l2', axis=1)


class ParameterOptimizer:
    """
    Optuna-based hyperparameter optimization for HDBSCAN.

    Features:
    - GridSampler for exhaustive search
    - Pre-computed UMAP reductions
    - Constraint-based pruning (noise, min clusters)
    - Maximizes relative_validity_

    Usage:
        optimizer = ParameterOptimizer(config, embeddings, original_embeddings)
        result = optimizer.optimize()
        best = optimizer.get_best_result()
    """

    def __init__(
        self,
        config: ClustererV2Config,
        embeddings: np.ndarray,
        original_embeddings: np.ndarray,
        verbose: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            config: ClustererV2Config
            embeddings: L2-normalized embeddings for UMAP
            original_embeddings: Original embeddings for coherence (usually same)
            verbose: Print progress
        """
        self.config = config
        self._embeddings = embeddings
        self._original_embeddings = original_embeddings
        self._verbose = verbose
        self._N = len(embeddings)

        # Will be populated
        self._search_space: Dict[str, List] = {}
        self._umap_cache: Dict[Tuple[int, int, float], np.ndarray] = {}  # Key: (n_neighbors, n_components, min_dist)
        self._study: Optional[optuna.Study] = None
        self._best_result: Optional[Dict[str, Any]] = None
        self._selector = AlgorithmSelector(config)

    def precompute_umap_reductions(
        self,
        n_neighbors_list: List[int],
        n_components_list: List[int],
        min_dist_list: List[float]
    ) -> Dict[Tuple[int, int, float], np.ndarray]:
        """
        Pre-compute UMAP reductions for all (n_neighbors, n_components, min_dist) combinations in parallel.

        Args:
            n_neighbors_list: List of n_neighbors values
            n_components_list: List of n_components values
            min_dist_list: List of min_dist values

        Returns:
            Dict mapping (n_neighbors, n_components, min_dist) -> L2-normalized reduced embeddings
        """
        # Generate all combinations
        combinations = [
            (nn, nc, md)
            for nn in n_neighbors_list
            for nc in n_components_list
            for md in min_dist_list
        ]

        def compute_single_umap(n_neighbors: int, n_components: int, min_dist: float) -> Tuple[Tuple[int, int, float], np.ndarray]:
            reduced = run_umap(
                self._embeddings,
                n_neighbors,
                n_components,
                min_dist,
                self.config.umap_random_state
            )
            reduced_normalized = l2_normalize(reduced)
            return (n_neighbors, n_components, min_dist), reduced_normalized

        # Run UMAP computations in parallel with progress bar
        n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else -1

        # Use return_as='generator' to enable progress tracking while keeping parallelization
        results_gen = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(compute_single_umap)(nn, nc, md) for nn, nc, md in combinations
        )

        # Wrap generator with tqdm for progress bar
        results = list(tqdm(
            results_gen,
            total=len(combinations),
            desc="UMAP",
            disable=not self._verbose
        ))

        # Convert to dict
        return {key: reduced for key, reduced in results}

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function maximizing relative_validity_.

        Args:
            trial: Optuna trial

        Returns:
            relative_validity_ score (higher is better)
            Raises TrialPruned if constraints violated
        """
        # Get grid parameters
        n_neighbors = trial.suggest_categorical('n_neighbors', self._search_space['n_neighbors'])
        n_components = trial.suggest_categorical('n_components', self._search_space['n_components'])
        min_dist = trial.suggest_categorical('min_dist', self._search_space['min_dist'])
        min_cluster_size = trial.suggest_categorical('min_cluster_size', self._search_space['min_cluster_size'])
        min_samples = max(1, min_cluster_size // 2)  # Derived

        # Look up pre-computed UMAP reduction
        reduced_normalized = self._umap_cache[(n_neighbors, n_components, min_dist)]

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,  # Required for relative_validity_
        )
        labels = clusterer.fit_predict(reduced_normalized)

        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).sum() / len(labels)

        # Check constraints (prune if violated)
        if n_clusters < self.config.min_clusters:
            raise optuna.TrialPruned(f"Too few clusters: {n_clusters}")

        if noise_rate > self.config.max_noise_rate:
            raise optuna.TrialPruned(f"Noise too high: {noise_rate:.1%}")

        # Get relative_validity_
        try:
            relative_validity = clusterer.relative_validity_
        except AttributeError:
            # Fallback to DBCV
            relative_validity = self._compute_dbcv(labels, reduced_normalized)

        # Extract persistence metrics
        persistence_metrics = self._selector.extract_persistence_metrics(clusterer, labels)

        # Calculate coherence (on original embeddings)
        coherence = self._calculate_coherence(labels, self._original_embeddings)

        # Extract probability metrics
        prob_metrics = self._compute_probability_metrics(clusterer.probabilities_, labels)

        # Extract outlier metrics
        outlier_metrics = self._compute_outlier_metrics(clusterer.outlier_scores_)

        # Compute composite score (no persistence)
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
        """
        Extract metrics from HDBSCAN probabilities_.

        Args:
            probabilities: HDBSCAN probabilities_ array
            labels: Cluster labels

        Returns:
            Dict with mean_probability, low_prob_ratio, fuzzy_cluster_ratio, n_fuzzy_clusters
        """
        mask = labels >= 0  # Exclude noise
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

        # Compute per-cluster fuzzy ratio
        # A cluster is "fuzzy" if its per-cluster low_ratio > fuzzy_cluster_threshold
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

    def _compute_outlier_metrics(
        self,
        outlier_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract metrics from HDBSCAN outlier_scores_ (GLOSH).

        Args:
            outlier_scores: HDBSCAN outlier_scores_ array

        Returns:
            Dict with mean_outlier_score, high_outlier_ratio
        """
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
        """
        Compute soft threshold composite score (no persistence).

        Formula:
            score = w_validity * relative_validity
                    - λ_low_prob * max(0, low_prob_ratio - τ)
                    - λ_fuzzy * fuzzy_cluster_ratio
                    - λ_fuzzy_count * fuzzy_cluster_fraction

        Where:
        - fuzzy_cluster_ratio: fraction of points in fuzzy clusters
        - fuzzy_cluster_fraction: n_fuzzy_clusters / n_clusters

        Args:
            relative_validity: HDBSCAN relative_validity_ (0-1, higher=better)
            low_prob_ratio: Fraction of borderline members (0-1, lower=better)
            fuzzy_cluster_ratio: Fraction of points in fuzzy clusters (0-1, lower=better)
            n_fuzzy_clusters: Number of fuzzy clusters
            n_clusters: Total number of clusters

        Returns:
            (composite_score, breakdown_dict)
        """
        w_validity = self.config.weight_validity
        tau = self.config.tau_low_prob
        lam_low_prob = self.config.lambda_low_prob
        lam_fuzzy = self.config.lambda_fuzzy
        lam_fuzzy_count = self.config.lambda_fuzzy_count

        # Validity component
        validity_term = w_validity * relative_validity

        # Soft threshold penalty for global low_prob_ratio
        excess_low_prob = max(0.0, low_prob_ratio - tau)
        penalty_low_prob = lam_low_prob * excess_low_prob

        # Penalty for fuzzy cluster ratio (points in bad clusters)
        penalty_fuzzy = lam_fuzzy * fuzzy_cluster_ratio

        # Penalty for fuzzy cluster fraction (proportion of clusters that are bad)
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

    def optimize(self) -> OptunaResult:
        """
        Run Optuna grid search optimization.

        Returns:
            OptunaResult with best configuration and metrics
        """
        if self._verbose:
            print(f"\n[Optuna] Starting HDBSCAN optimization (N={self._N})")

        # Create search space using config values
        self._search_space = create_search_space(self._N, self.config)
        n_trials = (
            len(self._search_space['n_neighbors']) *
            len(self._search_space['n_components']) *
            len(self._search_space['min_dist']) *
            len(self._search_space['min_cluster_size'])
        )

        if self._verbose:
            self._print_search_space_table(n_trials)

        # Pre-compute UMAP reductions for all (n_neighbors, n_components, min_dist) combinations
        if self.config.precompute_umap:
            self._umap_cache = self.precompute_umap_reductions(
                self._search_space['n_neighbors'],
                self._search_space['n_components'],
                self._search_space['min_dist']
            )

        # Create and run Optuna study
        sampler = GridSampler(self._search_space)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study = optuna.create_study(
            study_name=f"clusterer_v2_{id(self)}",
            direction='maximize',
            sampler=sampler,
        )

        # Track best result for progress bar
        self._progress_best_score = 0.0
        self._progress_best_k = 0

        # Run optimization with tqdm progress bar
        pbar = tqdm(total=n_trials, desc="Optimizing", disable=not self._verbose)

        def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            """Update progress bar after each trial."""
            if trial.state == optuna.trial.TrialState.COMPLETE:
                best = study.best_trial
                best_score = best.value
                best_k = best.user_attrs.get('n_clusters', 0)
                pbar.set_postfix({'best': f'{best_score:.3f}', 'k': best_k})
            pbar.update(1)

        self._study.optimize(self._objective, n_trials=None, callbacks=[progress_callback])
        pbar.close()

        # Get best trial
        best = self._study.best_trial
        n_neighbors = best.params['n_neighbors']
        n_components = best.params['n_components']
        min_dist = best.params['min_dist']
        min_cluster_size = best.params['min_cluster_size']
        min_samples = max(1, min_cluster_size // 2)

        # Re-run best configuration to get full results
        reduced_normalized = self._umap_cache[(n_neighbors, n_components, min_dist)]

        best_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        best_labels = best_clusterer.fit_predict(reduced_normalized)

        # Extract persistence metrics from best model
        persistence_metrics = self._selector.extract_persistence_metrics(best_clusterer, best_labels)

        # Count completed/pruned trials
        completed = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if self._verbose:
            self._print_results_table()
            self._print_best_result_details(best)

        result = OptunaResult(
            best_params={
                'n_neighbors': n_neighbors,
                'n_components': n_components,
                'min_dist': min_dist,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
            },
            best_value=best.value,
            best_labels=best_labels,
            best_model=best_clusterer,
            n_trials_completed=completed,
            n_trials_pruned=pruned,
            study=self._study,
            umap_embeddings=reduced_normalized,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics
        )

        # Quality check and conditional re-search
        result = self._check_quality_and_research(result)

        self._best_result = result
        return result

    def get_best_result(self) -> Optional[OptunaResult]:
        """Get the best result from optimization (None if not run yet)."""
        return self._best_result

    def _print_search_space_table(self, n_trials: int) -> None:
        """Print compact search space configuration."""
        print(f"  n_neighbors:      {self._search_space['n_neighbors']}")
        print(f"  n_components:     {self._search_space['n_components']}")
        print(f"  min_dist:         {self._search_space['min_dist']}")
        print(f"  min_cluster_size: {self._search_space['min_cluster_size']}")
        print(f"  Total trials:     {n_trials}")

    def _print_results_table(self, top_n: int = 5) -> None:
        """Print top N results as a formatted table."""
        completed = [t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_count = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if not completed:
            print("  No completed trials")
            return

        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]

        # Print table header
        print(f"\n  Top {min(top_n, len(sorted_trials))} Results:")
        print(f"  {'nn':>4} {'nc':>4} {'md':>5} {'mcs':>4} {'k':>5} {'noise':>6} {'score':>7}")
        print(f"  {'-'*4} {'-'*4} {'-'*5} {'-'*4} {'-'*5} {'-'*6} {'-'*7}")

        for i, trial in enumerate(sorted_trials):
            marker = "★" if i == 0 else " "
            noise_rate = trial.user_attrs.get('noise_rate', 0)
            print(f"{marker} {trial.params.get('n_neighbors', '?'):>4} "
                  f"{trial.params.get('n_components', '?'):>4} "
                  f"{trial.params.get('min_dist', '?'):>5} "
                  f"{trial.params.get('min_cluster_size', '?'):>4} "
                  f"{trial.user_attrs.get('n_clusters', '?'):>5} "
                  f"{noise_rate:>5.0%} "
                  f"{trial.value:>7.3f}")

        print(f"  {len(completed)} completed, {pruned_count} pruned")

    def _print_best_result_details(self, best_trial: optuna.trial.FrozenTrial) -> None:
        """Print detailed metrics for the best configuration."""
        rel_val = best_trial.user_attrs.get('relative_validity', 0)
        low_prob = best_trial.user_attrs.get('low_prob_ratio', 0)
        n_fuzzy = best_trial.user_attrs.get('n_fuzzy_clusters', 0)
        n_clusters = best_trial.user_attrs.get('n_clusters', 0)
        mean_outlier = best_trial.user_attrs.get('mean_outlier_score', 0)

        fuzzy_frac = n_fuzzy / n_clusters if n_clusters > 0 else 0.0

        print(f"\n  Best config: rel_validity={rel_val:.4f}, low_prob={low_prob:.1%}, "
              f"fuzzy={n_fuzzy}/{n_clusters} ({fuzzy_frac:.0%}), outlier={mean_outlier:.2f}")

    def _check_quality_and_research(self, result: OptunaResult) -> OptunaResult:
        """
        Check quality of optimization result and trigger re-search if needed.

        Trigger conditions (configurable):
        - (noise > 10% AND validity < 0.7) — both signals suggest poor quality
        - OR abs(n_clusters - sqrt(n)) / sqrt(n) > 0.15 — cluster count deviates from expected

        Args:
            result: Initial optimization result

        Returns:
            Original result if quality is acceptable, or re-search result
        """
        if not self.config.enable_research:
            return result

        # Extract metrics from best trial
        best_trial = self._study.best_trial
        n_clusters = best_trial.user_attrs.get('n_clusters', 0)
        noise_rate = best_trial.user_attrs.get('noise_rate', 0.0)
        # Use relative_validity (not composite score) for quality check threshold
        relative_validity = best_trial.user_attrs.get('relative_validity', result.best_value)
        composite_score = result.best_value

        sqrt_n = math.sqrt(self._N)
        max_noise = self.config.research_max_noise_rate
        min_validity = self.config.research_min_validity
        cluster_deviation_threshold = self.config.research_cluster_deviation_threshold

        # Calculate cluster count deviation from expected sqrt(n)
        cluster_deviation = abs(n_clusters - sqrt_n) / sqrt_n if sqrt_n > 0 else 0.0

        # Check if re-search is needed
        needs_research = False
        reasons = []

        # Condition 1: High noise AND low validity (both must be true)
        if noise_rate > max_noise and relative_validity < min_validity:
            needs_research = True
            reasons.append(f"noise={noise_rate:.1%}>{max_noise:.0%} AND rel_validity={relative_validity:.3f}<{min_validity}")

        # Condition 2: Cluster count deviates significantly from expected
        if cluster_deviation > cluster_deviation_threshold:
            needs_research = True
            reasons.append(f"cluster_deviation={cluster_deviation:.1%}>{cluster_deviation_threshold:.0%} (k={n_clusters}, expected≈{sqrt_n:.0f})")

        if not needs_research:
            if self._verbose:
                print(f"  Quality check PASSED: k={n_clusters} (expected≈{sqrt_n:.0f})")
            return result

        if self._verbose:
            print(f"\n[Research] Quality check failed: {', '.join(reasons)}")
            print(f"  Triggering extended search...")

        # Run extended search
        return self._run_extended_search(result)

    def _run_extended_search(self, initial_result: OptunaResult) -> OptunaResult:
        """
        Run extended search with expanded parameters using Optuna GridSampler.

        Strategy:
        - MCS: multipliers around best (e.g., 0.5x, 1.0x, 1.5x)
        - MS: log-scale grid from (best_ms * 0.5) to (best_ms * 2.0)
        - Selection methods: try both 'eom' and 'leaf'
        - Objective: maximize relative_validity_ (same as initial search)
        - Constraints: noise > max_noise_rate, clusters < min_clusters → pruned

        Args:
            initial_result: Result from initial optimization

        Returns:
            Best result from extended search (or initial if no improvement)
        """
        # Get best params from initial search
        best_n_neighbors = initial_result.best_params['n_neighbors']
        best_n_components = initial_result.best_params.get('n_components', self.config.umap_n_components_grid[0])
        best_min_dist = initial_result.best_params.get('min_dist', self.config.umap_min_dist_grid[0])
        best_mcs = initial_result.best_params['min_cluster_size']
        best_ms = initial_result.best_params.get('min_samples', best_mcs // 2)
        reduced_normalized = self._umap_cache[(best_n_neighbors, best_n_components, best_min_dist)]

        # Build MCS grid: multipliers around best
        mcs_multipliers = self.config.research_mcs_multipliers
        mcs_options = sorted(set(
            max(3, int(round(best_mcs * mult)))  # Minimum MCS of 3
            for mult in mcs_multipliers
        ))

        # Build MS grid: log-scale from (best_ms * low) to (best_ms * high)
        ms_low_mult, ms_high_mult = self.config.research_ms_range_multipliers
        ms_low = max(1, int(round(best_ms * ms_low_mult)))
        ms_high = max(ms_low, int(round(best_ms * ms_high_mult)))
        ms_options = log_spaced_ints(ms_low, ms_high, k=self.config.research_ms_grid_k)

        # Selection methods to try
        selection_methods = list(self.config.research_selection_methods)

        # Filter MS options to only include valid combinations (ms <= max(mcs))
        max_mcs = max(mcs_options)
        ms_options = [ms for ms in ms_options if ms <= max_mcs]

        # Build extended search space
        extended_search_space = {
            'min_cluster_size': mcs_options,
            'min_samples': ms_options,
            'cluster_selection_method': selection_methods,
        }

        n_trials_total = len(mcs_options) * len(ms_options) * len(selection_methods)

        if self._verbose:
            print(f"\n[Extended Search] Based on best: nn={best_n_neighbors}, mcs={best_mcs}")
            print(f"  MCS grid:      {mcs_options}")
            print(f"  MS grid:       {ms_options}")
            print(f"  Methods:       {selection_methods}")
            print(f"  Total trials:  {n_trials_total}")

        # Define objective for extended search
        def extended_objective(trial: optuna.Trial) -> float:
            mcs = trial.suggest_categorical('min_cluster_size', extended_search_space['min_cluster_size'])
            ms = trial.suggest_categorical('min_samples', extended_search_space['min_samples'])
            method = trial.suggest_categorical('cluster_selection_method', extended_search_space['cluster_selection_method'])

            # Skip invalid combinations (ms must be <= mcs)
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

            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_rate = (labels == -1).sum() / len(labels)

            # Check constraints (prune if violated)
            if n_clusters < self.config.min_clusters:
                raise optuna.TrialPruned(f"Too few clusters: {n_clusters}")

            if noise_rate > self.config.max_noise_rate:
                raise optuna.TrialPruned(f"Noise too high: {noise_rate:.1%}")

            # Get relative_validity_
            try:
                validity = clusterer.relative_validity_
            except AttributeError:
                validity = self._compute_dbcv(labels, reduced_normalized)

            coherence = self._calculate_coherence(labels, self._original_embeddings)

            # Extract persistence metrics
            persistence_metrics = self._selector.extract_persistence_metrics(clusterer, labels)
            weighted_persistence = persistence_metrics.get('weighted_persistence', 0.0)

            # Extract probability and outlier metrics
            prob_metrics = self._compute_probability_metrics(clusterer.probabilities_, labels)
            outlier_metrics = self._compute_outlier_metrics(clusterer.outlier_scores_)

            # Compute composite score (no persistence)
            composite_score, score_breakdown = self._compute_composite_score(
                validity,
                prob_metrics['low_prob_ratio'],
                prob_metrics['fuzzy_cluster_ratio'],
                prob_metrics['n_fuzzy_clusters'],
                n_clusters
            )

            # Log user attributes for later retrieval
            trial.set_user_attr('n_clusters', n_clusters)
            trial.set_user_attr('noise_rate', noise_rate)
            trial.set_user_attr('coherence', coherence)
            trial.set_user_attr('labels', labels.tolist())
            trial.set_user_attr('relative_validity', validity)
            trial.set_user_attr('weighted_persistence', weighted_persistence)
            trial.set_user_attr('mean_probability', prob_metrics['mean_probability'])
            trial.set_user_attr('low_prob_ratio', prob_metrics['low_prob_ratio'])
            trial.set_user_attr('fuzzy_cluster_ratio', prob_metrics['fuzzy_cluster_ratio'])
            trial.set_user_attr('n_fuzzy_clusters', prob_metrics['n_fuzzy_clusters'])
            trial.set_user_attr('mean_outlier_score', outlier_metrics['mean_outlier_score'])
            trial.set_user_attr('high_outlier_ratio', outlier_metrics['high_outlier_ratio'])
            trial.set_user_attr('composite_score', composite_score)

            return composite_score

        # Create and run extended search study with progress bar
        extended_sampler = GridSampler(extended_search_space)
        extended_study = optuna.create_study(
            study_name=f"clusterer_v2_extended_{id(self)}",
            direction='maximize',
            sampler=extended_sampler,
        )

        # Run with tqdm progress bar
        ext_best_score = initial_result.best_value
        pbar = tqdm(total=n_trials_total, desc="Extended search", disable=not self._verbose)

        def ext_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            nonlocal ext_best_score
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if study.best_trial.value > ext_best_score:
                    ext_best_score = study.best_trial.value
                    pbar.set_postfix({'best': f'{ext_best_score:.3f}'})
            pbar.update(1)

        extended_study.optimize(extended_objective, n_trials=None, callbacks=[ext_progress_callback])
        pbar.close()

        # Count completed/pruned
        completed = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if self._verbose:
            print(f"  {completed} completed, {pruned} pruned")

        # Check if we found something better
        if completed == 0:
            if self._verbose:
                print(f"  No valid trials found, keeping initial result")
            return initial_result

        best_extended = extended_study.best_trial

        # Compare with initial result
        if best_extended.value <= initial_result.best_value:
            if self._verbose:
                print(f"  No improvement (extended: {best_extended.value:.4f} <= initial: {initial_result.best_value:.4f})")
            return initial_result

        # Build the better result
        mcs = best_extended.params['min_cluster_size']
        ms = best_extended.params['min_samples']
        method = best_extended.params['cluster_selection_method']

        # Re-run to get the model and labels (since we stored labels as list)
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
            improvement = best_extended.value - initial_result.best_value
            print(f"  Found better: {method}, mcs={mcs}, ms={ms}, "
                  f"score={best_extended.value:.4f} (+{improvement:.4f})")

        return OptunaResult(
            best_params={
                'n_neighbors': best_n_neighbors,
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
            study=self._study,  # Keep original study for reference
            umap_embeddings=reduced_normalized,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics
        )
