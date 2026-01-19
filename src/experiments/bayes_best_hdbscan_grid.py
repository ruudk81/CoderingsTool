#%%

"""
Bayesian HDBSCAN Grid Search Experiment

Goal: Find optimal UMAP + HDBSCAN configurations using Optuna GridSampler
by maximizing relative_validity_ with noise constraints.

Search Space:
- UMAP n_neighbors: log-spaced grid based on dataset size (k=4)
- HDBSCAN min_cluster_size: log-spaced grid based on sqrt(N) (k=4)

Fixed Parameters:
- UMAP n_components: 10
- UMAP min_dist: 0.1
- HDBSCAN min_samples: mcs // 2 (derived)

Constraints:
- MAX_NOISE_RATE: 0.20 (prune trials exceeding 20% noise)
- MIN_CLUSTERS: 3 (prune trials with fewer than 3 clusters)
"""

import os
import sys
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress UMAP n_jobs warning when using random_state
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")

# Suppress HDBSCAN validity warnings (divide by zero in edge cases)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")

# Add src paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import GridSampler
import hdbscan
import umap
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

# Local imports
import models
from utils.cacheManager import generate_enhanced_variable_key

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset configuration
FILENAME = "M000000 MOJO Bezoekersonderzoek festivalbeleving Pinkpop_153836.sav"
VARIABLE = "Q15"
SAMPLE_SIZE = 2000

# Fixed UMAP parameters
UMAP_N_COMPONENTS = 10
UMAP_MIN_DIST = 0.1

# Grid density (k=4 for each search parameter)
GRID_K = 4

# Constraints
MAX_NOISE_RATE = 0.20  # Maximum acceptable noise rate
MIN_CLUSTERS = 3       # Minimum number of clusters required

# Study configuration
STUDY_NAME = f"hdbscan_grid_{VARIABLE}_{SAMPLE_SIZE}"

# Output paths
EXPORTS_DIR = Path(__file__).parent.parent.parent / "exports" / "hdbscan_optimization"

# =============================================================================
# GRID GENERATION FUNCTIONS
# =============================================================================

def log_spaced_ints(low: int, high: int, k: int = 4) -> List[int]:
    """Generate k log-spaced integers between low and high."""
    if low == high:
        return [low]
    if low <= 0:
        low = 1
    vals = np.exp(np.linspace(np.log(low), np.log(high), k))
    return sorted(set(int(round(v)) for v in vals))


def n_neighbors_bounds(N: int) -> Tuple[int, int]:
    """Compute n_neighbors bounds based on dataset size."""
    if N < 80:
        return 5, min(15, N - 1)
    if N < 300:
        return 5, min(30, N - 1)
    low = max(5, int(round(0.005 * N)))    # ~0.5% of N
    high = min(200, int(round(0.05 * N)))  # ~5% of N, capped
    high = max(high, low)
    high = min(high, N - 1)
    low = min(low, high)
    return low, high


def n_neighbors_grid(N: int, k: int = 4) -> List[int]:
    """Generate n_neighbors grid for dataset of size N."""
    low, high = n_neighbors_bounds(N)
    return log_spaced_ints(low, high, k=k)


def mcs_bounds_sqrt(N: int) -> Tuple[int, int]:
    """Compute min_cluster_size bounds based on sqrt(N)."""
    low = max(3, int(round(0.25 * math.sqrt(N))))   # 0.25 * sqrt(N)
    high = max(low, int(round(1.0 * math.sqrt(N))))  # 1.0 * sqrt(N)
    return low, high


def mcs_grid_sqrt(N: int, k: int = 4) -> List[int]:
    """Generate min_cluster_size grid for dataset of size N."""
    low, high = mcs_bounds_sqrt(N)
    return log_spaced_ints(low, high, k=k)


def create_search_space(N: int, k: int = 4) -> Dict[str, List[int]]:
    """Create Optuna search space dict for GridSampler."""
    return {
        'n_neighbors': n_neighbors_grid(N, k=k),
        'min_cluster_size': mcs_grid_sqrt(N, k=k),
    }


# =============================================================================
# DATA LOADING (reused from umap_clustering_comparison.py)
# =============================================================================

def load_step4_embeddings(
    filename: Optional[str] = None,
    variable: Optional[str] = None,
    sample_size: Optional[int] = None,
    variable_key: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load Step 4 embeddings from cache.

    Returns:
        embeddings: numpy array of shape (n_ideas, embedding_dim)
        idea_texts: list of idea text strings
    """
    import pickle

    _filename = filename if filename is not None else FILENAME
    _variable = variable if variable is not None else VARIABLE
    _sample_size = sample_size if sample_size is not None else SAMPLE_SIZE

    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[_variable],
            is_merged=False,
            sample_size=_sample_size
        )

    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "data" / "cache"
    base_name = Path(_filename).stem
    cache_filename = f"005_embeddings_{base_name}_{variable_key}.pkl"
    cache_path = cache_dir / cache_filename

    print(f"Loading embeddings from: {cache_path}")

    if not cache_path.exists():
        raise ValueError(f"No cached embeddings found at {cache_path}. Run pipeline Step 4 first.")

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    data = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

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
# UMAP AND CLUSTERING FUNCTIONS
# =============================================================================

def run_umap(embeddings: np.ndarray, n_neighbors: int, n_components: int,
             min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    # Suppress n_jobs warning inside parallel workers
    warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='euclidean',
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)


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
    except Exception as e:
        print(f"DBCV calculation failed: {e}")
        return -1.0


def calculate_cluster_coherence(labels: np.ndarray, original_embeddings: np.ndarray) -> float:
    """Calculate mean intra-cluster cosine similarity using original embeddings."""
    unique_labels = [l for l in set(labels) if l >= 0]

    if not unique_labels:
        return 0.0

    coherences = []
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = original_embeddings[mask]

        if len(cluster_embeddings) < 2:
            coherences.append(1.0)
            continue

        similarities = cluster_embeddings @ cluster_embeddings.T
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

    Returns:
        Dict with:
        - per_cluster: List of (label, size, coherence) tuples
        - n_unacceptable, n_low, n_moderate, n_high: Counts
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
            upper_tri_indices = np.triu_indices(n, k=1)
            pairwise_sims = similarities[upper_tri_indices]
            coherence = float(np.mean(pairwise_sims))

        per_cluster.append((label, size, coherence))

        # Classify
        if coherence < unacceptable_threshold:
            n_unacceptable += 1
        elif coherence < low_threshold:
            n_low += 1
        elif coherence < high_threshold:
            n_moderate += 1
        else:
            n_high += 1

    # Sort by label
    per_cluster.sort(key=lambda x: x[0])

    summary_parts = []
    if n_unacceptable > 0:
        summary_parts.append(f"{n_unacceptable} unacceptable")
    if n_low > 0:
        summary_parts.append(f"{n_low} low")
    if n_moderate > 0:
        summary_parts.append(f"{n_moderate} moderate")
    if n_high > 0:
        summary_parts.append(f"{n_high} high")

    return {
        'per_cluster': per_cluster,
        'n_unacceptable': n_unacceptable,
        'n_low': n_low,
        'n_moderate': n_moderate,
        'n_high': n_high,
        'summary': ", ".join(summary_parts) if summary_parts else "no clusters"
    }


def print_cluster_samples(labels: np.ndarray, idea_texts: List[str],
                          params: Dict, coherence: float,
                          original_embeddings: Optional[np.ndarray] = None,
                          max_samples: int = 10, random_state: int = 42):
    """
    Print random sample of responses from each cluster with per-cluster coherence.

    Args:
        labels: Cluster labels
        idea_texts: List of idea text strings
        params: Parameters used for clustering
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
    print(f"CLUSTER SAMPLES: Best HDBSCAN Configuration")
    print(f"Parameters: {params}")
    print(f"Coherence: {coherence:.3f} | Clusters: {n_clusters}")
    print(f"{'='*70}")

    for label in unique_labels:
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


def extract_persistence_metrics(clusterer: hdbscan.HDBSCAN, labels: np.ndarray) -> Dict:
    """Extract cluster persistence metrics from fitted HDBSCAN model."""
    persistence = getattr(clusterer, "cluster_persistence_", None)
    if persistence is None:
        persistence = getattr(clusterer, "cluster_stability_", None)

    if persistence is None or len(persistence) == 0:
        return {
            'mean_persistence': np.nan,
            'weighted_persistence': np.nan,
        }

    persistence = np.array(persistence)

    metrics = {
        'mean_persistence': float(np.mean(persistence)),
    }

    # Weighted persistence
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


# =============================================================================
# OPTUNA OBJECTIVE AND OPTIMIZATION
# =============================================================================

# Global variables for objective function (set in run_optimization)
_embeddings_normalized = None
_N = None
_search_space = None
_umap_cache = {}  # Pre-computed UMAP reductions keyed by n_neighbors


def precompute_umap_reductions(embeddings: np.ndarray, n_neighbors_list: List[int],
                                n_components: int, min_dist: float) -> Dict[int, np.ndarray]:
    """
    Pre-compute UMAP reductions for all n_neighbors values in parallel.

    Args:
        embeddings: L2-normalized embeddings
        n_neighbors_list: List of n_neighbors values to compute
        n_components: UMAP n_components
        min_dist: UMAP min_dist

    Returns:
        Dict mapping n_neighbors -> L2-normalized reduced embeddings
    """
    print(f"  Pre-computing {len(n_neighbors_list)} UMAP reductions in parallel...")

    def compute_single_umap(n_neighbors: int) -> Tuple[int, np.ndarray]:
        reduced = run_umap(embeddings, n_neighbors, n_components, min_dist)
        reduced_normalized = l2_normalize(reduced)
        return n_neighbors, reduced_normalized

    # Run UMAP computations in parallel
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(compute_single_umap)(nn) for nn in n_neighbors_list
    )

    # Convert to dict
    return {nn: reduced for nn, reduced in results}


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function maximizing relative_validity_.

    Returns:
        relative_validity_ score (higher is better)
        Pruned if constraints violated (noise > 20%, clusters < 3)
    """
    global _embeddings_normalized, _N, _search_space, _umap_cache

    # Get grid parameters
    n_neighbors = trial.suggest_categorical('n_neighbors', _search_space['n_neighbors'])
    min_cluster_size = trial.suggest_categorical('min_cluster_size', _search_space['min_cluster_size'])
    min_samples = max(1, min_cluster_size // 2)  # Derived

    # Log parameters
    print(f"  Trial {trial.number}: n_neighbors={n_neighbors}, mcs={min_cluster_size}, ms={min_samples}")

    # Look up pre-computed UMAP reduction
    reduced_normalized = _umap_cache[n_neighbors]

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

    # Check constraints
    if n_clusters < MIN_CLUSTERS:
        print(f"    PRUNED: Too few clusters ({n_clusters})")
        raise optuna.TrialPruned(f"Too few clusters: {n_clusters}")

    if noise_rate > MAX_NOISE_RATE:
        print(f"    PRUNED: Noise too high ({noise_rate:.1%})")
        raise optuna.TrialPruned(f"Noise too high: {noise_rate:.1%}")

    # Get relative_validity_
    try:
        relative_validity = clusterer.relative_validity_
    except AttributeError:
        print("    Warning: relative_validity_ not available, using DBCV fallback")
        relative_validity = compute_dbcv(labels, reduced_normalized)

    # Calculate additional metrics for logging
    coherence = calculate_cluster_coherence(labels, _embeddings_normalized)
    persistence_metrics = extract_persistence_metrics(clusterer, labels)

    # Log user attributes
    trial.set_user_attr('n_clusters', n_clusters)
    trial.set_user_attr('noise_rate', noise_rate)
    trial.set_user_attr('coherence', coherence)
    trial.set_user_attr('min_samples', min_samples)
    trial.set_user_attr('mean_persistence', persistence_metrics.get('mean_persistence', np.nan))
    trial.set_user_attr('weighted_persistence', persistence_metrics.get('weighted_persistence', np.nan))

    print(f"    → rel_validity={relative_validity:.4f}, k={n_clusters}, noise={noise_rate:.1%}, coh={coherence:.3f}")

    return relative_validity


def run_optimization() -> optuna.Study:
    """Run Optuna grid search optimization and return study."""
    global _embeddings_normalized, _N, _search_space, _umap_cache

    print("=" * 70)
    print(f"HDBSCAN Grid Search Optimization: {VARIABLE}_{SAMPLE_SIZE}")
    print("=" * 70)

    # Ensure output directory exists
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare data
    print("\n[1/5] Loading Step 4 embeddings from cache...")
    embeddings, idea_texts = load_step4_embeddings()
    _embeddings_normalized = l2_normalize(embeddings)
    _N = len(embeddings)
    print(f"Loaded and normalized {_N} embeddings")

    # 2. Create search space
    print("\n[2/5] Creating search space...")
    _search_space = create_search_space(_N, k=GRID_K)
    n_trials = len(_search_space['n_neighbors']) * len(_search_space['min_cluster_size'])
    print(f"  n_neighbors grid: {_search_space['n_neighbors']}")
    print(f"  min_cluster_size grid: {_search_space['min_cluster_size']}")
    print(f"  Total trials: {n_trials}")

    # 3. Pre-compute UMAP reductions in parallel
    print("\n[3/5] Pre-computing UMAP reductions in parallel...")
    _umap_cache = precompute_umap_reductions(
        _embeddings_normalized,
        _search_space['n_neighbors'],
        UMAP_N_COMPONENTS,
        UMAP_MIN_DIST
    )
    print(f"  Cached {len(_umap_cache)} UMAP reductions")

    # 4. Create and run Optuna study
    print("\n[4/5] Running HDBSCAN optimization...")
    sampler = GridSampler(_search_space)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',
        sampler=sampler,
    )

    study.optimize(objective, n_trials=None)  # Run all grid combinations

    # 5. Print and export results
    print("\n[5/6] Exporting results...")
    print_optimization_results(study)
    export_study_results(study)
    generate_optimization_plots(study)

    # 6. Print cluster samples for best configuration
    print_best_trial_clusters(study, idea_texts)

    return study


def interpret_relative_validity(score: float) -> str:
    """Interpret relative_validity_ score."""
    if score < 0.2:
        return "Poor - clusters not well-separated by density valleys"
    elif score < 0.3:
        return "Weak - some density-based separation, but not strong"
    elif score < 0.4:
        return "Moderate - reasonable density-based clustering"
    else:
        return "Good - clear density valleys between clusters"


def print_optimization_results(study: optuna.Study):
    """Print optimization results summary."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    # Interpretation guide
    print("\nrelative_validity_ interpretation guide:")
    print("  < 0.2  : Poor - clusters not well-separated by density valleys")
    print("  0.2-0.3: Weak - some density-based separation, but not strong")
    print("  0.3-0.4: Moderate - reasonable density-based clustering")
    print("  > 0.4  : Good - clear density valleys between clusters")

    # Best trial
    best = study.best_trial
    interpretation = interpret_relative_validity(best.value)
    print(f"\nBest Trial: #{best.number}")
    print(f"  relative_validity_: {best.value:.4f} ({interpretation})")
    print(f"  Parameters:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
    print(f"  Metrics:")
    for key, value in best.user_attrs.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    # Summary statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\nTrials Summary:")
    print(f"  Completed: {len(completed_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")

    if completed_trials:
        values = [t.value for t in completed_trials]
        print(f"  relative_validity_ range: [{min(values):.4f}, {max(values):.4f}]")


def export_study_results(study: optuna.Study):
    """Export optimization results to Excel."""
    # Get trials dataframe
    trials_df = study.trials_dataframe()

    # Best parameters
    best_params = study.best_params.copy()
    best_params['min_samples'] = max(1, best_params['min_cluster_size'] // 2)
    best_params['relative_validity'] = study.best_value
    best_params['n_components'] = UMAP_N_COMPONENTS
    best_params['min_dist'] = UMAP_MIN_DIST

    # Add user attrs to best params
    for key, value in study.best_trial.user_attrs.items():
        best_params[key] = value

    # Save to Excel
    output_path = EXPORTS_DIR / f"{STUDY_NAME}_results.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        trials_df.to_excel(writer, sheet_name='Trials', index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name='Best_Params', index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
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

    print(f"Results saved to: {output_path}")


def print_best_trial_clusters(study: optuna.Study, idea_texts: List[str]):
    """Re-run best trial configuration and print cluster samples."""
    global _embeddings_normalized, _umap_cache

    best = study.best_trial
    n_neighbors = best.params['n_neighbors']
    min_cluster_size = best.params['min_cluster_size']
    min_samples = max(1, min_cluster_size // 2)

    print("\n[6/6] Printing cluster samples for best configuration...")

    # Use cached UMAP reduction
    reduced_normalized = _umap_cache[n_neighbors]

    # Re-run HDBSCAN with best parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(reduced_normalized)

    # Get coherence from best trial
    coherence = best.user_attrs.get('coherence', 0.0)

    # Build params dict for display
    params = {
        'n_neighbors': n_neighbors,
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'n_components': UMAP_N_COMPONENTS,
        'min_dist': UMAP_MIN_DIST,
    }

    # Print cluster samples
    print_cluster_samples(
        labels=labels,
        idea_texts=idea_texts,
        params=params,
        coherence=coherence,
        original_embeddings=_embeddings_normalized,
        max_samples=10
    )


def generate_optimization_plots(study: optuna.Study):
    """Generate Optuna visualization plots as interactive HTML files."""
    try:
        # Optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(str(EXPORTS_DIR / f"{STUDY_NAME}_history.html"))
        print(f"Saved optimization history plot (HTML)")

        # Parameter importances (if enough completed trials)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) >= 2:
            try:
                fig2 = optuna.visualization.plot_param_importances(study)
                fig2.write_html(str(EXPORTS_DIR / f"{STUDY_NAME}_importances.html"))
                print(f"Saved parameter importances plot (HTML)")
            except Exception as e:
                print(f"Could not generate importance plot: {e}")

        # Parallel coordinate plot
        if len(completed) >= 2:
            try:
                fig3 = optuna.visualization.plot_parallel_coordinate(study)
                fig3.write_html(str(EXPORTS_DIR / f"{STUDY_NAME}_parallel.html"))
                print(f"Saved parallel coordinate plot (HTML)")
            except Exception as e:
                print(f"Could not generate parallel coordinate plot: {e}")

    except ImportError as e:
        print(f"Warning: Could not generate plots (missing plotly): {e}")
    except Exception as e:
        print(f"Warning: Plot generation failed: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    study = run_optimization()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nBest configuration:")
    print(f"  n_neighbors: {study.best_params['n_neighbors']}")
    print(f"  min_cluster_size: {study.best_params['min_cluster_size']}")
    print(f"  min_samples: {max(1, study.best_params['min_cluster_size'] // 2)}")
    print(f"  n_components: {UMAP_N_COMPONENTS} (fixed)")
    print(f"  min_dist: {UMAP_MIN_DIST} (fixed)")
    print(f"\n  relative_validity_: {study.best_value:.4f}")

# %%
