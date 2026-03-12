"""
Clusterer Helpers Module - EXPERIMENTAL VERSION

This is an isolated copy for experimentation in step_5_clusterer.
Changes here do NOT affect the production pipeline.

Original: src/utils/clusterer_helpers.py

Consolidates all helper classes and functions for clustering:
- Section 1: Preprocessing (L2 normalization, PCA, embedding extraction)
- Section 2: Algorithm Selection (DVC, knee detection)
- Section 3: Parameter Optimization (Grid Search + Pareto Selection)
- Section 4: Quality Metrics (coherence, DBCV, silhouette)
- Section 5: Post-Processing (merging, noise reduction)
- Section 6: Representation (c-TF-IDF, MMR, TF-IDF)
- Section 7: Label Generation (LLM labels)
"""

import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_array
from kneed import KneeLocator
import hdbscan
import umap
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from pydantic import BaseModel, Field

from experiments import models_exp as models
from utils.llm import llm_create_sync, create_client

try:
    from .config_clusterer_exp import ClustererConfig
    from .prompts_exp import CLUSTER_DESCRIPTION_PROMPT, ClusterDescription
    from .placeholder_lookup import DatasetPlaceholders, build_cluster_placeholders
except ImportError:
    from config_clusterer_exp import ClustererConfig
    from prompts_exp import CLUSTER_DESCRIPTION_PROMPT, ClusterDescription
    from placeholder_lookup import DatasetPlaceholders, build_cluster_placeholders

# Suppress warnings during optimization
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")


# =============================================================================
# SECTION 1: PREPROCESSING
# =============================================================================

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings.

    Args:
        embeddings: Array of shape (n_samples, n_features)

    Returns:
        L2-normalized embeddings (unit vectors)
    """
    return normalize(embeddings, norm='l2', axis=1)


def apply_pca(
    embeddings: np.ndarray,
    n_components: float = 0.99,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_components: Variance to retain (0.99 = 99%)
        random_state: Random seed

    Returns:
        (reduced_embeddings, fitted_pca_model)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca


def extract_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererConfig
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], List[str], Optional[str], Optional[str]]:
    """
    Extract embeddings from EmbeddingsModel list.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererConfig

    Returns:
        embeddings: Array of shape (n_ideas, embedding_dim)
        idea_texts: List of idea text strings (idea.idea)
        idea_indices: List of (response_idx, idea_idx) tuples for result mapping
        concept_types: List of concept_type strings per idea
        template_prefix: The canonical phrasing prefix (if available)
        embedding_text_format: The text format used for embedding (if available)
    """
    embedding_field = getattr(config, 'clustering_embedding_field', 'idea_embedding')

    embeddings_list = []
    idea_texts = []
    idea_indices = []
    concept_types = []
    template_prefix = None
    embedding_text_format = None

    for resp_idx, response in enumerate(input_list):
        # Extract template_prefix from first response that has it
        if template_prefix is None and hasattr(response, 'template_prefix') and response.template_prefix:
            template_prefix = response.template_prefix

        # Extract embedding_text_format from first response that has it
        if embedding_text_format is None and hasattr(response, 'embedding_text_format') and response.embedding_text_format:
            embedding_text_format = response.embedding_text_format

        if response.response_ideas:
            for idea_idx, idea in enumerate(response.response_ideas):
                emb = getattr(idea, embedding_field, None)
                if emb is not None:
                    embeddings_list.append(emb)
                    idea_texts.append(idea.idea if hasattr(idea, 'idea') else str(idea))
                    idea_indices.append((resp_idx, idea_idx))
                    concept_types.append(getattr(idea, 'concept_type', '') or '')

    if not embeddings_list:
        raise ValueError(f"No embeddings found in input data (field: {embedding_field})")

    embeddings = np.vstack(embeddings_list)

    if config.verbose:
        print(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]} (field: {embedding_field})")
        if template_prefix:
            prefix_display = template_prefix[:50] + "..." if len(template_prefix) > 50 else template_prefix
            print(f"Template prefix: '{prefix_display}'")
        if embedding_text_format:
            print(f"Embedding text format: {embedding_text_format}")

    return embeddings, idea_texts, idea_indices, concept_types, template_prefix, embedding_text_format


def preprocess_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererConfig
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int]], List[str], Optional[PCA], Optional[str], Optional[str]]:
    """
    Full preprocessing pipeline: extract, optionally PCA.

    No L2 normalization is applied. OpenAI/Azure embeddings are already unit
    vectors, and UMAP uses metric='cosine' which handles raw embeddings natively.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererConfig

    Returns:
        embeddings_original: Raw embeddings (no normalization)
        embeddings_processed: Processed embeddings (may be PCA-reduced, not normalized)
        idea_texts: List of idea text strings (idea.idea)
        idea_indices: List of (response_idx, idea_idx) tuples
        concept_types: List of concept_type strings per idea
        pca_model: Fitted PCA model (or None if not applied)
        template_prefix: The canonical phrasing prefix (if available)
        embedding_text_format: The text format used for embedding (if available)
    """
    # Extract embeddings
    embeddings, idea_texts, idea_indices, concept_types, template_prefix, embedding_text_format = extract_embeddings(input_list, config)
    n_samples = len(embeddings)

    if config.verbose:
        print(f"Extracted {n_samples} embeddings (no normalization — UMAP uses cosine metric)")

    # Apply PCA for large datasets
    pca_model = None
    if n_samples > config.pca_threshold:
        if config.verbose:
            print(f"Applying PCA (n > {config.pca_threshold})...")
        embeddings_processed, pca_model = apply_pca(
            embeddings,
            n_components=config.pca_variance_retained,
            random_state=config.umap_random_state
        )
        if config.verbose:
            print(f"PCA reduced to {embeddings_processed.shape[1]} components")
    else:
        embeddings_processed = embeddings

    return embeddings, embeddings_processed, idea_texts, idea_indices, concept_types, pca_model, template_prefix, embedding_text_format


# =============================================================================
# SECTION 2: ALGORITHM SELECTION
# =============================================================================

@dataclass
class AlgorithmRecommendation:
    """Result of automatic algorithm selection."""

    # Final recommendation
    recommended_algorithm: str  # "HDBSCAN", "AGGLOMERATIVE", "KMEANS"
    confidence: str  # "high", "medium", "low"

    # DVC analysis
    dvc_value: float
    dvc_mean_dk: float
    dvc_std_dk: float
    dvc_recommendation: str  # "HDBSCAN", "AGGLOMERATIVE", "EITHER", "AGGLOMERATIVE_FORCED"

    # Knee analysis
    knee_K: Optional[int]
    has_sharp_knee: bool
    y_difference: float
    knee_recommendation: str

    # Combined analysis
    combined_recommendation: str
    reasoning: str

    # Flag for forced algorithm selection
    is_forced: bool = False


class AlgorithmSelector:
    """
    Automatic algorithm selection using DVC and kNN knee detection.

    Usage:
        selector = AlgorithmSelector(config)
        dvc_result = selector.compute_dvc(embeddings_original)
        knee_result = selector.detect_knee(embeddings_reduced)
        persistence = selector.extract_persistence_metrics(clusterer, labels)
        recommendation = selector.recommend(dvc_result, knee_result, persistence)
    """

    def __init__(self, config: ClustererConfig):
        self.config = config

    def compute_dvc(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute Density Variation Coefficient.

        DVC = std(d_k) / mean(d_k), where d_k is distance to k-th nearest neighbor.

        High DVC (>0.45) indicates varying density -> HDBSCAN better
        Low DVC (<0.25) indicates uniform density -> Agglomerative better

        Args:
            embeddings: L2-normalized embeddings
            k: k-th nearest neighbor (uses config default if None)

        Returns:
            Dict with dvc, mean_dk, std_dk, recommendation
        """
        k = k or self.config.dvc_knn_k
        n = len(embeddings)

        if n < k + 1:
            return {
                'dvc': np.nan,
                'mean_dk': np.nan,
                'std_dk': np.nan,
                'recommendation': 'INSUFFICIENT_DATA'
            }

        # Compute k-NN distances
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nbrs.fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)

        # Distance to k-th nearest neighbor (skip self at index 0)
        d_k = distances[:, -1]

        mean_dk = float(np.mean(d_k))
        std_dk = float(np.std(d_k))

        if mean_dk == 0:
            return {
                'dvc': np.nan,
                'mean_dk': mean_dk,
                'std_dk': std_dk,
                'recommendation': 'ZERO_MEAN'
            }

        dvc = std_dk / mean_dk

        # Recommendation based on thresholds
        if dvc > self.config.dvc_high_threshold:
            recommendation = 'HDBSCAN'
        elif dvc < self.config.dvc_low_threshold:
            recommendation = 'AGGLOMERATIVE'
        else:
            recommendation = 'EITHER'

        return {
            'dvc': float(dvc),
            'mean_dk': mean_dk,
            'std_dk': std_dk,
            'recommendation': recommendation
        }

    def detect_knee(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect knee in kNN distance curve using adaptive KneeLocator.

        Sharp knee (y_difference >= 0.6) indicates density transition -> HDBSCAN suitable
        Flat curve indicates uniform density -> Agglomerative/K-means better

        Args:
            embeddings: UMAP-reduced embeddings
            k: k-th nearest neighbor (uses config default if None)

        Returns:
            Dict with K, y_difference, has_sharp_knee, recommendation, etc.
        """
        k = k or self.config.knee_knn_k
        n = len(embeddings)

        # Adaptive parameters based on dataset size
        kneedle_S = max(1.0, n / 100)
        interp_method = "polynomial" if n < 200 else "interp1d"

        # Compute kNN distances
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        k_distances = distances[:, k]

        # Sort for elbow analysis
        sorted_distances = np.sort(k_distances)

        # Define search window: skip first point, include rest
        start_idx = 1
        end_idx = n - 1

        if end_idx <= start_idx:
            return {
                'K': None,
                'y_difference': 0.0,
                'has_sharp_knee': False,
                'recommendation': 'AGGLOMERATIVE_OR_KMEANS',
                'distances': sorted_distances,
                'kneedle_S': kneedle_S,
                'interp_method': interp_method
            }

        # Extract search segment
        search_distances = sorted_distances[start_idx:end_idx]
        search_x = np.arange(len(search_distances))

        # Detect knee with adaptive parameters
        kneedle = KneeLocator(
            x=search_x,
            y=search_distances,
            S=kneedle_S,
            curve="convex",
            direction="increasing",
            interp_method=interp_method
        )

        # Map knee back to original coordinate system
        K_in_window = kneedle.knee
        if K_in_window is not None:
            K = start_idx + K_in_window
        else:
            K = None

        # Compute y_difference (knee sharpness)
        if K is not None and kneedle.y_difference is not None and len(kneedle.y_difference) > 0:
            y_difference = float(max(kneedle.y_difference))
            has_sharp_knee = y_difference >= self.config.knee_y_diff_threshold
        else:
            y_difference = 0.0
            has_sharp_knee = False

        # Recommendation
        recommendation = "HDBSCAN" if has_sharp_knee else "AGGLOMERATIVE_OR_KMEANS"
        knee_distance = sorted_distances[K] if K is not None else None

        return {
            'K': K,
            'y_difference': y_difference,
            'has_sharp_knee': has_sharp_knee,
            'recommendation': recommendation,
            'knee_distance': knee_distance,
            'distances': sorted_distances,
            'kneedle_S': kneedle_S,
            'interp_method': interp_method
        }

    def extract_persistence_metrics(
        self,
        clusterer: hdbscan.HDBSCAN,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract cluster persistence metrics from fitted HDBSCAN model.

        Persistence measures cluster stability across density thresholds.
        Higher persistence = more stable/robust clusters.

        Args:
            clusterer: Fitted HDBSCAN model
            labels: Cluster labels from the model

        Returns:
            Dict with mean_persistence, weighted_persistence, etc.
        """
        # Try both attribute names (depends on HDBSCAN version)
        persistence = getattr(clusterer, "cluster_persistence_", None)
        if persistence is None:
            persistence = getattr(clusterer, "cluster_stability_", None)

        if persistence is None or len(persistence) == 0:
            return {
                'mean_persistence': np.nan,
                'min_persistence': np.nan,
                'max_persistence': np.nan,
                'std_persistence': np.nan,
                'weighted_persistence': np.nan
            }

        persistence = np.array(persistence)

        metrics = {
            'mean_persistence': float(np.mean(persistence)),
            'min_persistence': float(np.min(persistence)),
            'max_persistence': float(np.max(persistence)),
            'std_persistence': float(np.std(persistence)) if len(persistence) > 1 else 0.0,
        }

        # Calculate size-weighted persistence
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

    def recommend(
        self,
        dvc_result: Dict[str, Any],
        knee_result: Dict[str, Any]
    ) -> AlgorithmRecommendation:
        """
        Generate combined algorithm recommendation using DVC and knee signals.

        Simplified decision logic:
        1. If DVC < 0.25 -> FORCE Agglomerative (skip HDBSCAN entirely)
        2. Otherwise, use knee detection:
           - Sharp knee (ydiff >= 0.6) -> HDBSCAN
           - Flat knee (ydiff < 0.6) -> AGGLOMERATIVE

        Args:
            dvc_result: Result from compute_dvc()
            knee_result: Result from detect_knee()

        Returns:
            AlgorithmRecommendation with full details
        """
        dvc_value = dvc_result.get('dvc', np.nan)
        has_sharp_knee = knee_result.get('has_sharp_knee', False)
        y_difference = knee_result.get('y_difference', 0.0)

        # Check if agglomerative fallback is enabled
        enable_agglomerative = getattr(self.config, 'enable_agglomerative_fallback', True)

        # Hard rule: DVC < threshold forces Agglomerative (only if fallback enabled)
        force_threshold = getattr(self.config, 'force_agglomerative_below_dvc', 0.25)

        if enable_agglomerative and not np.isnan(dvc_value) and dvc_value < force_threshold:
            # HARD RULE: Force Agglomerative when density is uniform
            return AlgorithmRecommendation(
                recommended_algorithm="AGGLOMERATIVE",
                confidence="high",
                dvc_value=dvc_value,
                dvc_mean_dk=dvc_result.get('mean_dk', np.nan),
                dvc_std_dk=dvc_result.get('std_dk', np.nan),
                dvc_recommendation="AGGLOMERATIVE_FORCED",
                knee_K=knee_result.get('K'),
                has_sharp_knee=has_sharp_knee,
                y_difference=y_difference,
                knee_recommendation=knee_result.get('recommendation', 'UNKNOWN'),
                combined_recommendation="AGGLOMERATIVE_FORCED",
                reasoning=f"DVC={dvc_value:.3f} < {force_threshold} indicates uniform density -> HDBSCAN skipped",
                is_forced=True
            )

        # Standard decision based on knee detection
        if has_sharp_knee:
            combined_recommendation = "HDBSCAN"
            recommended_algorithm = "HDBSCAN"
            confidence = "high"
            reasoning = f"Sharp knee (ydiff={y_difference:.2f} >= 0.6) indicates density transitions"
        elif enable_agglomerative:
            # Flat knee -> Agglomerative (only if fallback enabled)
            combined_recommendation = "AGGLOMERATIVE"
            recommended_algorithm = "AGGLOMERATIVE"
            confidence = "medium"
            reasoning = f"Flat knee (ydiff={y_difference:.2f} < 0.6) suggests uniform density"
        else:
            # Flat knee but agglomerative disabled -> stick with HDBSCAN
            combined_recommendation = "HDBSCAN"
            recommended_algorithm = "HDBSCAN"
            confidence = "medium"
            reasoning = f"Flat knee (ydiff={y_difference:.2f} < 0.6) but agglomerative fallback disabled -> using HDBSCAN"

        return AlgorithmRecommendation(
            recommended_algorithm=recommended_algorithm,
            confidence=confidence,
            dvc_value=dvc_value,
            dvc_mean_dk=dvc_result.get('mean_dk', np.nan),
            dvc_std_dk=dvc_result.get('std_dk', np.nan),
            dvc_recommendation=dvc_result.get('recommendation', 'UNKNOWN'),
            knee_K=knee_result.get('K'),
            has_sharp_knee=has_sharp_knee,
            y_difference=y_difference,
            knee_recommendation=knee_result.get('recommendation', 'UNKNOWN'),
            combined_recommendation=combined_recommendation,
            reasoning=reasoning,
            is_forced=False
        )


# =============================================================================
# SECTION 3: PARAMETER OPTIMIZATION
# =============================================================================

@dataclass
class TrialResult:
    """Metrics from a single grid search trial."""
    params: Dict[str, Any]       # {n_neighbors, n_components, min_dist, min_samples, min_cluster_size}
    n_clusters: int
    noise_rate: float
    max_cluster_ratio: float     # largest cluster / non-noise points
    validity: float              # relative_validity_
    coherence: float             # mean pairwise cosine within clusters
    mean_persistence: float
    weighted_persistence: float
    mean_probability: float
    low_prob_ratio: float
    mean_outlier_score: float
    high_outlier_ratio: float


@dataclass
class GridSearchResult:
    """Result from exhaustive grid search with Pareto front."""
    all_trials: List[TrialResult]
    pareto_indices: List[int]    # Indices into all_trials on the Pareto front
    selected_idx: int            # Index of selected trial
    best_params: Dict[str, Any]
    best_labels: np.ndarray
    best_model: hdbscan.HDBSCAN
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
    high_min: int = 10
) -> List[int]:
    """
    Generate UMAP n_neighbors grid based on dataset size.

    Formula: ceil(sqrt(N)/2) to max(ceil(sqrt(N)), high_min), log-spaced.
    All values have a floor of 1 and are capped at N-1.

    Args:
        N: Dataset size
        k: Number of log-spaced grid points (default 3)
        high_min: Floor for high bound (default 10)

    Returns:
        Log-spaced list of n_neighbors values
    """
    sqrt_n = math.sqrt(N)
    low = max(1, math.ceil(sqrt_n / 2))
    high = max(math.ceil(sqrt_n), high_min)

    # Safety: cap at N-1
    high = min(high, N - 1)
    low = min(low, high)

    return log_spaced_ints(low, high, k=k)


def ms_grid(N: int, k: int = 3) -> List[int]:
    """
    Generate HDBSCAN min_samples grid based on dataset size.

    Formula: ceil(ln(N)) to ceil(2*ln(N)), log-spaced.
    All values have a floor of 1.

    Args:
        N: Dataset size
        k: Number of log-spaced grid points (default 3)

    Returns:
        Log-spaced list of min_samples values
    """
    ln_n = math.log(N)
    low = max(1, math.ceil(ln_n))
    high = max(low, math.ceil(2 * ln_n))
    return log_spaced_ints(low, high, k=k)


def mcs_grid(N: int, k: int = 3, ms_multiplier: float = 2.0) -> List[int]:
    """
    Generate HDBSCAN min_cluster_size grid based on dataset size.

    Formula: ceil(ms_multiplier * ln(N)) to ceil(ms_multiplier * 2*ln(N)), log-spaced.
    Derived as ms_multiplier × the ms bounds.
    All values have a floor of 1.

    Args:
        N: Dataset size
        k: Number of log-spaced grid points (default 3)
        ms_multiplier: MCS = this × MS (default 2.0)

    Returns:
        Log-spaced list of min_cluster_size values
    """
    ln_n = math.log(N)
    low = max(1, math.ceil(ms_multiplier * ln_n))
    high = max(low, math.ceil(ms_multiplier * 2 * ln_n))
    return log_spaced_ints(low, high, k=k)


def create_search_space(N: int, config: ClustererConfig) -> Dict[str, List]:
    """
    Create Optuna search space dict for GridSampler using config values.

    Grid dimensions:
    - n_neighbors: ceil(sqrt(N)/2) to max(ceil(sqrt(N)), high_min), log-spaced
    - n_components: explicit grid from config
    - min_dist: explicit grid from config
    - min_samples: ceil(ln(N)) to ceil(2*ln(N)), log-spaced
    - min_cluster_size: 2×ms bounds, log-spaced

    Args:
        N: Dataset size
        config: ClustererConfig with grid parameters

    Returns:
        Dict with 'n_neighbors', 'n_components', 'min_dist', 'min_samples',
        and 'min_cluster_size' grids
    """
    return {
        'n_neighbors': n_neighbors_grid(N, k=config.n_neighbors_grid_k, high_min=config.n_neighbors_high_min),
        'n_components': list(config.umap_n_components_grid),
        'min_dist': list(config.umap_min_dist_grid),
        'min_samples': ms_grid(N, k=config.ms_grid_k),
        'min_cluster_size': mcs_grid(N, k=config.mcs_grid_k, ms_multiplier=config.mcs_ms_multiplier),
    }


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction with cosine metric.

    Args:
        embeddings: Raw embeddings (not normalized — cosine metric handles this)
        n_neighbors: UMAP n_neighbors
        n_components: Target dimensionality
        min_dist: UMAP min_dist
        random_state: Random seed

    Returns:
        UMAP-reduced embeddings (not normalized)
    """
    warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='cosine',
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)


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
        config: ClustererConfig,
        embeddings: np.ndarray,
        original_embeddings: np.ndarray,
        verbose: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            config: ClustererConfig
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
        self._umap_cache: Dict[Tuple[int, int, float], np.ndarray] = {}
        self._best_result: Optional['GridSearchResult'] = None
        self._selector = AlgorithmSelector(config)
        self._metrics_calc = ClusterQualityMetrics(config)

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
            Dict mapping (n_neighbors, n_components, min_dist) -> raw UMAP-reduced embeddings
        """
        # Generate all combinations
        combinations = [
            (nn, nc, md)
            for nn in n_neighbors_list
            for nc in n_components_list
            for md in min_dist_list
        ]

        def compute_single_umap(n_neighbors: int, n_components: int, min_dist: float) -> Tuple[Tuple[int, int, float], np.ndarray]:
            umap_reduced = run_umap(
                self._embeddings,
                n_neighbors,
                n_components,
                min_dist,
                self.config.umap_random_state
            )
            return (n_neighbors, n_components, min_dist), umap_reduced

        # Run UMAP computations in parallel with progress bar
        n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else -1

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

        return {key: reduced for key, reduced in results}

    def _run_trial(
        self,
        umap_reduced: np.ndarray,
        umap_key: Tuple[int, int, float],
        min_samples: int,
        min_cluster_size: int
    ) -> Optional[TrialResult]:
        """
        Run single HDBSCAN trial and collect all metrics.

        Returns TrialResult or None if HDBSCAN produces < 2 clusters.
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        labels = clusterer.fit_predict(umap_reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            return None

        n_total = len(labels)
        noise_rate = float((labels == -1).sum() / n_total)

        # Max cluster ratio: largest cluster as fraction of non-noise points
        non_noise_count = (labels >= 0).sum()
        if non_noise_count > 0:
            cluster_counts = np.bincount(labels[labels >= 0])
            max_cluster_ratio = float(cluster_counts.max() / non_noise_count)
        else:
            max_cluster_ratio = 1.0

        # DBCV / relative validity
        try:
            validity = clusterer.relative_validity_
        except AttributeError:
            validity = self._metrics_calc.compute_dbcv(labels, umap_reduced)

        # Coherence (on original high-dimensional embeddings)
        coherence = self._metrics_calc.calculate_coherence(labels, self._original_embeddings)[0]

        # Persistence
        persistence_metrics = self._selector.extract_persistence_metrics(clusterer, labels)

        # Probability metrics
        prob_metrics = self._compute_probability_metrics(clusterer.probabilities_, labels)

        # Outlier metrics
        outlier_metrics = self._metrics_calc.compute_outlier_metrics(clusterer.outlier_scores_, labels)

        nn, nc, md = umap_key
        return TrialResult(
            params={
                'n_neighbors': nn,
                'n_components': nc,
                'min_dist': md,
                'min_samples': min_samples,
                'min_cluster_size': min_cluster_size,
            },
            n_clusters=n_clusters,
            noise_rate=noise_rate,
            max_cluster_ratio=max_cluster_ratio,
            validity=validity,
            coherence=coherence,
            mean_persistence=persistence_metrics.get('mean_persistence', np.nan),
            weighted_persistence=persistence_metrics.get('weighted_persistence', np.nan),
            mean_probability=prob_metrics['mean_probability'],
            low_prob_ratio=prob_metrics['low_prob_ratio'],
            mean_outlier_score=outlier_metrics['mean_outlier_score'],
            high_outlier_ratio=outlier_metrics['high_outlier_ratio'],
        )

    def _compute_probability_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Extract metrics from HDBSCAN probabilities_."""
        mask = labels >= 0
        probs = probabilities[mask]

        if len(probs) == 0:
            return {'mean_probability': 0.0, 'low_prob_ratio': 0.0}

        mean_prob = float(np.mean(probs))
        low_prob_ratio = float((probs < self.config.low_probability_threshold).sum() / len(probs))

        return {
            'mean_probability': mean_prob,
            'low_prob_ratio': low_prob_ratio,
        }

    # ── Stage 1: Hard constraint filtering ──────────────────────────────

    def _filter_candidates(self, trials: List[TrialResult]) -> List[int]:
        """
        Filter trials by hard constraints with progressive fallback.

        Returns indices into trials list.

        Fallback chain:
        1. Full constraints (DBCV, k range, noise, max_cluster_ratio)
        2. Drop DBCV
        3. Drop noise + max_cluster_ratio (k range only)
        4. All trials
        """
        cfg = self.config
        sqrt_n = math.sqrt(self._N)
        k_low = max(2, math.ceil(cfg.pareto_min_k_sqrt_mult * sqrt_n))
        k_high = math.ceil(cfg.pareto_max_k_sqrt_mult * sqrt_n)

        def passes_k_range(t: TrialResult) -> bool:
            return k_low <= t.n_clusters <= k_high

        # Full constraints
        full = [i for i, t in enumerate(trials)
                if t.validity > cfg.pareto_min_dbcv
                and passes_k_range(t)
                and t.noise_rate <= cfg.pareto_max_noise_rate
                and t.max_cluster_ratio <= cfg.pareto_max_cluster_ratio]
        if full:
            if self._verbose:
                print(f"  Stage 1: {len(full)} candidates pass full constraints "
                      f"(k in [{k_low},{k_high}])")
            return full

        # Fallback 1: drop DBCV
        fb1 = [i for i, t in enumerate(trials)
               if passes_k_range(t)
               and t.noise_rate <= cfg.pareto_max_noise_rate
               and t.max_cluster_ratio <= cfg.pareto_max_cluster_ratio]
        if fb1:
            if self._verbose:
                print(f"  Stage 1: {len(fb1)} candidates (fallback 1: dropped DBCV)")
            return fb1

        # Fallback 2: k range only
        fb2 = [i for i, t in enumerate(trials) if passes_k_range(t)]
        if fb2:
            if self._verbose:
                print(f"  Stage 1: {len(fb2)} candidates (fallback 2: k range only)")
            return fb2

        # Fallback 3: all trials
        if self._verbose:
            print(f"  Stage 1: using all {len(trials)} trials (no constraints met)")
        return list(range(len(trials)))

    # ── Stage 2: Pareto front ───────────────────────────────────────────

    @staticmethod
    def _compute_pareto_front(
        trials: List[TrialResult],
        candidate_indices: List[int]
    ) -> List[int]:
        """
        Compute Pareto front over 4 objectives.

        Objectives (all converted to maximize):
        - DBCV (maximize)
        - n_clusters (minimize → negate)
        - low_prob_ratio (minimize → negate)
        - max_cluster_ratio (minimize → negate)

        Returns positions within candidate_indices that are non-dominated.
        """
        if len(candidate_indices) <= 1:
            return list(range(len(candidate_indices)))

        # Build objective matrix: all directions = maximize
        obj_values = []
        for idx in candidate_indices:
            t = trials[idx]
            obj_values.append([
                t.validity,             # maximize
                -t.n_clusters,          # minimize → negate
                -t.low_prob_ratio,      # minimize → negate
                -t.max_cluster_ratio,   # minimize → negate
            ])

        def dominates(a, b):
            """True if a dominates b (a >= b on all, a > b on at least one)."""
            at_least_as_good = all(ai >= bi for ai, bi in zip(a, b))
            strictly_better = any(ai > bi for ai, bi in zip(a, b))
            return at_least_as_good and strictly_better

        n = len(obj_values)
        pareto_mask = [True] * n
        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                if dominates(obj_values[j], obj_values[i]):
                    pareto_mask[i] = False
                    break

        return [i for i, on_front in enumerate(pareto_mask) if on_front]

    # ── Stage 3: Ideal-point selection ──────────────────────────────────

    def _select_from_pareto(
        self,
        trials: List[TrialResult],
        candidate_indices: List[int],
        pareto_positions: List[int]
    ) -> int:
        """
        Select from Pareto front using weighted Euclidean distance to ideal point.

        Uses percentile normalization (p5/p95) from ALL candidates for robustness,
        then picks the Pareto-optimal trial closest to [1,1,1,1].

        Returns position within pareto_positions.
        """
        cfg = self.config

        # Collect raw objectives from all candidates (for normalization bounds)
        all_dbcv = [trials[candidate_indices[i]].validity for i in range(len(candidate_indices))]
        all_k = [trials[candidate_indices[i]].n_clusters for i in range(len(candidate_indices))]
        all_lp = [trials[candidate_indices[i]].low_prob_ratio for i in range(len(candidate_indices))]
        all_mcr = [trials[candidate_indices[i]].max_cluster_ratio for i in range(len(candidate_indices))]

        p_lo = cfg.pareto_norm_percentile_low
        p_hi = cfg.pareto_norm_percentile_high

        def percentile_bounds(values):
            lo = float(np.percentile(values, p_lo))
            hi = float(np.percentile(values, p_hi))
            return lo, hi

        bounds = [
            percentile_bounds(all_dbcv),
            percentile_bounds(all_k),
            percentile_bounds(all_lp),
            percentile_bounds(all_mcr),
        ]

        # Direction: True = maximize (higher is closer to ideal), False = minimize
        directions = [True, False, False, False]  # DBCV↑, k↓, low_prob↓, max_cluster↓

        weights = np.array([
            cfg.pareto_weight_dbcv,
            cfg.pareto_weight_k,
            cfg.pareto_weight_low_prob_ratio,
            cfg.pareto_weight_max_cluster_ratio,
        ])

        best_pos = pareto_positions[0]
        best_dist = float('inf')

        for pos in pareto_positions:
            cand_idx = candidate_indices[pos]
            t = trials[cand_idx]
            raw = [t.validity, t.n_clusters, t.low_prob_ratio, t.max_cluster_ratio]

            normalized = []
            for val, (lo, hi), is_max in zip(raw, bounds, directions):
                if hi - lo > 1e-12:
                    norm = (val - lo) / (hi - lo)
                else:
                    norm = 0.5
                norm = max(0.0, min(1.0, norm))
                if not is_max:
                    norm = 1.0 - norm
                normalized.append(norm)

            # Weighted Euclidean distance to ideal [1,1,1,1]
            diff = np.array([1.0 - n for n in normalized])
            dist = float(np.sqrt(np.sum(weights * diff ** 2)))

            if dist < best_dist:
                best_dist = dist
                best_pos = pos

        return best_pos

    # ── Main optimize ───────────────────────────────────────────────────

    def optimize(self) -> GridSearchResult:
        """
        Run exhaustive grid search with 3-stage Pareto selection.

        1. Enumerate all UMAP × HDBSCAN parameter combos
        2. Filter candidates by hard constraints (with fallback)
        3. Compute Pareto front on (DBCV↑, k↓, low_prob↓, max_cluster↓)
        4. Select from front via weighted ideal-point distance
        5. Reconstruct selected HDBSCAN model
        """
        if self._verbose:
            print(f"\n[Grid Search] Starting HDBSCAN optimization (N={self._N})")

        self._search_space = create_search_space(self._N, self.config)

        n_umap_combos = (
            len(self._search_space['n_neighbors']) *
            len(self._search_space['n_components']) *
            len(self._search_space['min_dist'])
        )
        n_hdbscan_combos = (
            len(self._search_space['min_samples']) *
            len(self._search_space['min_cluster_size'])
        )
        n_trials = n_umap_combos * n_hdbscan_combos

        if self._verbose:
            self._print_search_space_table(n_trials)

        # Precompute UMAP reductions
        if self.config.precompute_umap:
            self._umap_cache = self.precompute_umap_reductions(
                self._search_space['n_neighbors'],
                self._search_space['n_components'],
                self._search_space['min_dist']
            )

        # Exhaustive grid search
        all_trials: List[TrialResult] = []
        n_skipped = 0

        pbar = tqdm(total=n_trials, desc="Grid search", disable=not self._verbose)

        for nn in self._search_space['n_neighbors']:
            for nc in self._search_space['n_components']:
                for md in self._search_space['min_dist']:
                    umap_key = (nn, nc, md)
                    umap_reduced = self._umap_cache[umap_key]

                    for ms in self._search_space['min_samples']:
                        for mcs in self._search_space['min_cluster_size']:
                            result = self._run_trial(umap_reduced, umap_key, ms, mcs)
                            if result is not None:
                                all_trials.append(result)
                            else:
                                n_skipped += 1
                            pbar.update(1)

        pbar.close()

        if self._verbose:
            print(f"  {len(all_trials)} valid trials, {n_skipped} skipped (<2 clusters)")

        if not all_trials:
            raise RuntimeError("No valid clustering found across entire grid")

        # Stage 1: Hard constraint filtering
        if self._verbose:
            print("\n[Pareto Selection]")
        candidate_indices = self._filter_candidates(all_trials)

        # Stage 2: Pareto front
        pareto_positions = self._compute_pareto_front(all_trials, candidate_indices)
        if self._verbose:
            print(f"  Stage 2: {len(pareto_positions)} solutions on Pareto front")

        # Stage 3: Ideal-point selection
        selected_pos = self._select_from_pareto(all_trials, candidate_indices, pareto_positions)
        selected_idx = candidate_indices[selected_pos]
        selected = all_trials[selected_idx]

        if self._verbose:
            self._print_results_table(all_trials, candidate_indices, pareto_positions, selected_pos)

        # Reconstruct HDBSCAN model for selected trial
        p = selected.params
        umap_key = (p['n_neighbors'], p['n_components'], p['min_dist'])
        umap_reduced = self._umap_cache[umap_key]

        best_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=p['min_cluster_size'],
            min_samples=p['min_samples'],
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
        )
        best_labels = best_clusterer.fit_predict(umap_reduced)
        persistence_metrics = self._selector.extract_persistence_metrics(best_clusterer, best_labels)

        self._best_result = GridSearchResult(
            all_trials=all_trials,
            pareto_indices=[candidate_indices[pos] for pos in pareto_positions],
            selected_idx=selected_idx,
            best_params=selected.params,
            best_labels=best_labels,
            best_model=best_clusterer,
            umap_embeddings=umap_reduced,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics,
        )
        return self._best_result

    def get_best_result(self) -> Optional[GridSearchResult]:
        """Get the best result from optimization (None if not run yet)."""
        return self._best_result

    def _print_search_space_table(self, n_trials: int) -> None:
        """Print compact search space configuration."""
        print(f"  n_neighbors:      {self._search_space['n_neighbors']}")
        print(f"  n_components:     {self._search_space['n_components']}")
        print(f"  min_dist:         {self._search_space['min_dist']}")
        print(f"  min_samples:      {self._search_space['min_samples']}")
        print(f"  min_cluster_size: {self._search_space['min_cluster_size']}")
        print(f"  Total trials:     {n_trials}")

    def _print_results_table(
        self,
        trials: List[TrialResult],
        candidate_indices: List[int],
        pareto_positions: List[int],
        selected_pos: int,
    ) -> None:
        """Print Pareto front results as a formatted table."""
        selected = trials[candidate_indices[selected_pos]]

        print(f"\n  Pareto Front ({len(pareto_positions)} solutions):")
        print(f"  {'':>1} {'nn':>4} {'nc':>4} {'md':>5} {'ms':>3} {'mcs':>4} "
              f"{'k':>4} {'noise':>6} {'dbcv':>6} {'coh':>5} {'lp':>5} {'mcr':>5}")
        print(f"  {'-'*1} {'-'*4} {'-'*4} {'-'*5} {'-'*3} {'-'*4} "
              f"{'-'*4} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")

        for pos in sorted(pareto_positions, key=lambda p: trials[candidate_indices[p]].validity, reverse=True):
            t = trials[candidate_indices[pos]]
            marker = "*" if pos == selected_pos else " "
            print(f"  {marker} {t.params['n_neighbors']:>4} "
                  f"{t.params['n_components']:>4} "
                  f"{t.params['min_dist']:>5} "
                  f"{t.params['min_samples']:>3} "
                  f"{t.params['min_cluster_size']:>4} "
                  f"{t.n_clusters:>4} "
                  f"{t.noise_rate:>5.0%} "
                  f"{t.validity:>6.3f} "
                  f"{t.coherence:>5.3f} "
                  f"{t.low_prob_ratio:>5.2f} "
                  f"{t.max_cluster_ratio:>5.2f}")

        print(f"\n  Selected: k={selected.n_clusters}, dbcv={selected.validity:.3f}, "
              f"noise={selected.noise_rate:.1%}, coherence={selected.coherence:.3f}")


# =============================================================================
# SECTION 4: QUALITY METRICS
# =============================================================================

@dataclass
class ClusteringMetrics:
    """Comprehensive clustering quality metrics."""

    # Core metrics
    n_clusters: int = 0
    noise_rate: float = 0.0
    noise_count: int = 0

    # Density-based metrics (HDBSCAN)
    dbcv: Optional[float] = None
    relative_validity: Optional[float] = None
    mean_persistence: Optional[float] = None
    weighted_persistence: Optional[float] = None
    min_persistence: Optional[float] = None
    max_persistence: Optional[float] = None
    std_persistence: Optional[float] = None

    # Geometry metrics
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None

    # Coherence metrics (on original embeddings)
    mean_coherence: float = 0.0
    coherence_n_unacceptable: int = 0
    coherence_n_low: int = 0
    coherence_n_moderate: int = 0
    coherence_n_high: int = 0
    coherence_breakdown: str = ""
    per_cluster_coherence: Optional[List[Tuple[int, int, float]]] = None

    # Cluster size distribution
    cluster_sizes: Optional[List[int]] = None
    median_cluster_size: Optional[int] = None
    min_cluster_size: Optional[int] = None
    max_cluster_size: Optional[int] = None

    # Probability metrics (from HDBSCAN probabilities_)
    mean_probability: Optional[float] = None
    low_prob_ratio: Optional[float] = None
    per_cluster_mean_prob: Optional[List[Tuple[int, int, float]]] = None

    # Outlier metrics (from HDBSCAN outlier_scores_ / GLOSH)
    mean_outlier_score: Optional[float] = None
    high_outlier_ratio: Optional[float] = None

    # Algorithm info
    algorithm_used: str = ""
    algorithm_params: Optional[Dict[str, Any]] = None


class ClusterQualityMetrics:
    """
    Calculator for comprehensive clustering quality metrics.

    Usage:
        calculator = ClusterQualityMetrics(config)
        metrics = calculator.calculate_all(labels, embeddings_reduced, embeddings_original)
    """

    def __init__(self, config: ClustererConfig):
        self.config = config

    def calculate_coherence(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Tuple[float, Dict[str, int], List[Tuple[int, int, float]]]:
        """
        Calculate mean coherence and per-cluster breakdown.

        Coherence = mean pairwise cosine similarity within cluster (using original embeddings).
        """
        unique_labels = [l for l in set(labels) if l >= 0]

        if not unique_labels:
            return 0.0, {'n_unacceptable': 0, 'n_low': 0, 'n_moderate': 0, 'n_high': 0}, []

        per_cluster = []
        n_unacceptable = 0
        n_low = 0
        n_moderate = 0
        n_high = 0

        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            size = len(cluster_embeddings)

            if size < 2:
                coherence = 1.0
            else:
                similarities = cluster_embeddings @ cluster_embeddings.T
                n = len(cluster_embeddings)
                upper_tri_indices = np.triu_indices(n, k=1)
                pairwise_sims = similarities[upper_tri_indices]
                coherence = float(np.mean(pairwise_sims))

            per_cluster.append((label, size, coherence))

            if coherence < self.config.coherence_acceptable:
                n_unacceptable += 1
            elif coherence < self.config.coherence_moderate:
                n_low += 1
            elif coherence < self.config.coherence_high:
                n_moderate += 1
            else:
                n_high += 1

        per_cluster.sort(key=lambda x: x[0])
        coherences = [coh for _, _, coh in per_cluster]
        mean_coherence = float(np.mean(coherences)) if coherences else 0.0

        breakdown = {
            'n_unacceptable': n_unacceptable,
            'n_low': n_low,
            'n_moderate': n_moderate,
            'n_high': n_high
        }

        return mean_coherence, breakdown, per_cluster

    def compute_dbcv(self, labels: np.ndarray, embeddings: np.ndarray) -> float:
        """Compute DBCV (Density-Based Clustering Validation) score."""
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
            return np.nan

    def compute_geometry_metrics(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Compute geometry-based metrics (silhouette, CH, DB)."""
        mask = labels >= 0
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        metrics = {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        }

        if mask.sum() < 2 or n_clusters < 2:
            return metrics

        try:
            metrics['silhouette'] = silhouette_score(embeddings[mask], labels[mask])
        except Exception:
            pass

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings[mask], labels[mask])
        except Exception:
            pass

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(embeddings[mask], labels[mask])
        except Exception:
            pass

        return metrics

    def compute_cluster_sizes(self, labels: np.ndarray) -> Dict[str, Any]:
        """Compute cluster size distribution."""
        unique_labels = [l for l in set(labels) if l >= 0]
        sizes = [int((labels == l).sum()) for l in unique_labels]

        if not sizes:
            return {
                'cluster_sizes': [],
                'median_cluster_size': None,
                'min_cluster_size': None,
                'max_cluster_size': None
            }

        return {
            'cluster_sizes': sizes,
            'median_cluster_size': int(np.median(sizes)),
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes)
        }

    def compute_probability_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Extract metrics from HDBSCAN probabilities_."""
        mask = labels >= 0
        probs = probabilities[mask]

        if len(probs) == 0:
            return {
                'mean_probability': None,
                'low_prob_ratio': None,
                'per_cluster_mean_prob': []
            }

        mean_prob = float(np.mean(probs))
        low_prob_ratio = float((probs < self.config.low_probability_threshold).sum() / len(probs))

        per_cluster = []
        for label in sorted(set(labels[mask])):
            cluster_mask = labels == label
            cluster_probs = probabilities[cluster_mask]
            per_cluster.append((int(label), len(cluster_probs), float(np.mean(cluster_probs))))

        return {
            'mean_probability': mean_prob,
            'low_prob_ratio': low_prob_ratio,
            'per_cluster_mean_prob': per_cluster
        }

    def compute_outlier_metrics(
        self,
        outlier_scores: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Extract metrics from HDBSCAN outlier_scores_ (GLOSH algorithm)."""
        if len(outlier_scores) == 0:
            return {
                'mean_outlier_score': None,
                'high_outlier_ratio': None
            }

        mean_score = float(np.mean(outlier_scores))
        high_ratio = float((outlier_scores > self.config.high_outlier_threshold).sum() / len(outlier_scores))

        return {
            'mean_outlier_score': mean_score,
            'high_outlier_ratio': high_ratio
        }

    def calculate_all(
        self,
        labels: np.ndarray,
        embeddings_reduced: np.ndarray,
        embeddings_original: np.ndarray,
        hdbscan_model: Optional[hdbscan.HDBSCAN] = None,
        probabilities: Optional[np.ndarray] = None,
        algorithm_used: str = "",
        algorithm_params: Optional[Dict[str, Any]] = None
    ) -> ClusteringMetrics:
        """Calculate all configured metrics."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        noise_rate = noise_count / len(labels) if len(labels) > 0 else 0.0

        mean_coherence, breakdown, per_cluster = self.calculate_coherence(
            labels, embeddings_original
        )

        breakdown_parts = []
        if breakdown['n_unacceptable'] > 0:
            breakdown_parts.append(f"{breakdown['n_unacceptable']} unacceptable")
        if breakdown['n_low'] > 0:
            breakdown_parts.append(f"{breakdown['n_low']} low")
        if breakdown['n_moderate'] > 0:
            breakdown_parts.append(f"{breakdown['n_moderate']} moderate")
        if breakdown['n_high'] > 0:
            breakdown_parts.append(f"{breakdown['n_high']} high")
        coherence_breakdown_str = ", ".join(breakdown_parts) if breakdown_parts else "no clusters"

        dbcv = self.compute_dbcv(labels, embeddings_reduced)
        geometry = self.compute_geometry_metrics(labels, embeddings_reduced)
        sizes = self.compute_cluster_sizes(labels)

        persistence_metrics = {}
        relative_validity = None
        prob_metrics = {}
        outlier_metrics = {}

        if hdbscan_model is not None:
            selector = AlgorithmSelector(self.config)
            persistence_metrics = selector.extract_persistence_metrics(hdbscan_model, labels)

            try:
                relative_validity = float(hdbscan_model.relative_validity_)
            except AttributeError:
                relative_validity = None

            if hasattr(hdbscan_model, 'probabilities_') and hdbscan_model.probabilities_ is not None:
                prob_metrics = self.compute_probability_metrics(hdbscan_model.probabilities_, labels)

            if hasattr(hdbscan_model, 'outlier_scores_') and hdbscan_model.outlier_scores_ is not None:
                outlier_metrics = self.compute_outlier_metrics(hdbscan_model.outlier_scores_, labels)

        # Explicit probabilities array overrides model-based extraction
        if probabilities is not None and not prob_metrics:
            prob_metrics = self.compute_probability_metrics(probabilities, labels)

        return ClusteringMetrics(
            n_clusters=n_clusters,
            noise_rate=noise_rate,
            noise_count=noise_count,
            dbcv=dbcv,
            relative_validity=relative_validity,
            mean_persistence=persistence_metrics.get('mean_persistence'),
            weighted_persistence=persistence_metrics.get('weighted_persistence'),
            min_persistence=persistence_metrics.get('min_persistence'),
            max_persistence=persistence_metrics.get('max_persistence'),
            std_persistence=persistence_metrics.get('std_persistence'),
            silhouette=geometry['silhouette'],
            calinski_harabasz=geometry['calinski_harabasz'],
            davies_bouldin=geometry['davies_bouldin'],
            mean_coherence=mean_coherence,
            coherence_n_unacceptable=breakdown['n_unacceptable'],
            coherence_n_low=breakdown['n_low'],
            coherence_n_moderate=breakdown['n_moderate'],
            coherence_n_high=breakdown['n_high'],
            coherence_breakdown=coherence_breakdown_str,
            per_cluster_coherence=per_cluster,
            cluster_sizes=sizes['cluster_sizes'],
            median_cluster_size=sizes['median_cluster_size'],
            min_cluster_size=sizes['min_cluster_size'],
            max_cluster_size=sizes['max_cluster_size'],
            mean_probability=prob_metrics.get('mean_probability'),
            low_prob_ratio=prob_metrics.get('low_prob_ratio'),
            per_cluster_mean_prob=prob_metrics.get('per_cluster_mean_prob'),
            mean_outlier_score=outlier_metrics.get('mean_outlier_score'),
            high_outlier_ratio=outlier_metrics.get('high_outlier_ratio'),
            algorithm_used=algorithm_used,
            algorithm_params=algorithm_params
        )


# =============================================================================
# SECTION 5: POST-PROCESSING
# =============================================================================

class UnionFind:
    """Union-Find data structure for transitive closure in cluster merging."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def get_components(self) -> Dict[int, int]:
        """Return mapping from element to component representative."""
        return {e: self.find(e) for e in self.parent}


def compute_cluster_centroids(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Compute centroid for each cluster."""
    centroids = {}
    sizes = {}

    unique_labels = [l for l in set(labels) if l >= 0]
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        sizes[label] = len(cluster_embeddings)

        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid

    return centroids, sizes


def pairwise_cluster_similarity(
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    embeddings: np.ndarray
) -> Dict[str, float]:
    """Calculate pairwise similarity statistics between two clusters."""
    emb_a = embeddings[indices_a]
    emb_b = embeddings[indices_b]

    sim_matrix = emb_a @ emb_b.T
    all_sims = sim_matrix.flatten()

    return {
        'q25': float(np.percentile(all_sims, 25)),
        'q50': float(np.percentile(all_sims, 50)),
        'q75': float(np.percentile(all_sims, 75)),
        'mean': float(np.mean(all_sims))
    }


def renumber_clusters(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels to be sequential starting from 0."""
    new_labels = labels.copy()
    unique_labels = sorted(set(labels) - {-1})

    mapping = {old: new for new, old in enumerate(unique_labels)}
    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels


def reduce_noise_by_embedding_similarity(
    labels: np.ndarray,
    embeddings: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    BERTopic-style noise reduction: assign noise points to nearest cluster by embedding similarity.
    """
    noise_mask = labels == -1
    n_total_noise = noise_mask.sum()

    if n_total_noise == 0:
        if verbose:
            print("  No noise points to reduce")
        return labels, {'n_noise_initial': 0, 'n_assigned': 0, 'n_noise_final': 0}

    if verbose:
        print(f"\n[Noise Reduction - Embedding Similarity]")
        print(f"  Initial noise points: {n_total_noise}")
        print(f"  Similarity threshold: {threshold}")

    centroids, sizes = compute_cluster_centroids(labels, embeddings)

    if len(centroids) == 0:
        if verbose:
            print("  No clusters found - cannot reduce noise")
        return labels, {'n_noise_initial': n_total_noise, 'n_assigned': 0, 'n_noise_final': n_total_noise}

    cluster_ids = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])

    noise_indices = np.where(noise_mask)[0]
    noise_embeddings = embeddings[noise_indices]

    similarities = noise_embeddings @ centroid_matrix.T

    best_cluster_indices = np.argmax(similarities, axis=1)
    best_scores = np.max(similarities, axis=1)

    new_labels = labels.copy()
    n_assigned = 0

    for i, (noise_idx, cluster_idx, score) in enumerate(zip(noise_indices, best_cluster_indices, best_scores)):
        if score >= threshold:
            new_labels[noise_idx] = cluster_ids[cluster_idx]
            n_assigned += 1

    n_noise_final = (new_labels == -1).sum()
    assignment_rate = n_assigned / n_total_noise if n_total_noise > 0 else 0.0

    if verbose:
        print(f"  Assigned: {n_assigned} ({assignment_rate:.1%})")
        print(f"  Remaining noise: {n_noise_final} ({n_noise_final/len(labels):.1%})")

    stats = {
        'n_noise_initial': n_total_noise,
        'n_assigned': n_assigned,
        'n_noise_final': n_noise_final,
        'assignment_rate': assignment_rate
    }

    return new_labels, stats


def merge_similar_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    config: ClustererConfig,
    verbose: bool = True
) -> np.ndarray:
    """Merge clusters using graph-based transitive closure with union-find."""
    if not config.enable_merging:
        return labels

    centroids, sizes = compute_cluster_centroids(labels, embeddings)
    n_initial_clusters = len(centroids)

    if n_initial_clusters < 2:
        if verbose:
            print("  Less than 2 clusters - skipping merge")
        return labels

    if verbose:
        print(f"\n[Cluster Merging]")
        print(f"  Initial clusters: {n_initial_clusters}")
        print(f"  Centroid threshold: {config.merge_centroid_threshold}")
        print(f"  Pairwise threshold: {config.merge_pairwise_threshold}")

    cluster_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_to_indices[int(label)].append(i)

    cluster_ids = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
    centroid_similarities = centroid_matrix @ centroid_matrix.T

    candidates = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            sim = centroid_similarities[i, j]
            if sim >= config.merge_centroid_threshold:
                candidates.append((cluster_ids[i], cluster_ids[j], sim))

    if verbose:
        print(f"  Candidate pairs (centroid >= {config.merge_centroid_threshold}): {len(candidates)}")

    if not candidates:
        if verbose:
            print("  No similar clusters found - no merging needed")
        return labels

    merge_edges = []
    for cluster_a, cluster_b, centroid_sim in candidates:
        indices_a = np.array(cluster_to_indices[cluster_a])
        indices_b = np.array(cluster_to_indices[cluster_b])

        stats = pairwise_cluster_similarity(indices_a, indices_b, embeddings)
        quantile_mean = np.mean([stats['q25'], stats['q50'], stats['q75']])

        if quantile_mean >= config.merge_pairwise_threshold:
            merge_edges.append((cluster_a, cluster_b, centroid_sim, quantile_mean))
            if verbose:
                print(f"    + Merge {cluster_a}<->{cluster_b} | "
                      f"sizes: {sizes[cluster_a]}, {sizes[cluster_b]} | "
                      f"centroid: {centroid_sim:.3f} | quantile_mean: {quantile_mean:.3f}")

    if not merge_edges:
        if verbose:
            print("  No merge-worthy pairs found")
        return labels

    uf = UnionFind(cluster_ids)
    for cluster_a, cluster_b, _, _ in merge_edges:
        uf.union(cluster_a, cluster_b)

    component_map = uf.get_components()

    labels_merged = labels.copy()
    for old_id, component_id in component_map.items():
        labels_merged[labels == old_id] = component_id

    labels_final = renumber_clusters(labels_merged)

    unique_components = set(component_map.values())
    n_final_clusters = len(unique_components)
    n_merged = n_initial_clusters - n_final_clusters

    if verbose:
        print(f"  Merging complete:")
        print(f"    Initial: {n_initial_clusters} -> Final: {n_final_clusters}")
        print(f"    Reduction: {n_merged} clusters removed")

    return labels_final


# =============================================================================
# SECTION 6: REPRESENTATION
# =============================================================================

# Lazy-loaded spaCy model
_SPACY_NLP = None


def get_spacy_nlp(model_name: str = "nl_core_news_lg"):
    """Get or load spaCy NLP model (lazy initialization)."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load(model_name, disable=["ner", "parser"])
        except OSError:
            from spacy.cli import download
            download(model_name)
            _SPACY_NLP = spacy.load(model_name, disable=["ner", "parser"])
    return _SPACY_NLP


def extract_noun_phrases_lemmatized(
    texts: List[str],
    nlp=None,
    model_name: str = "nl_core_news_lg"
) -> List[str]:
    """Extract POS-aware keyword tokens: standalone NOUN/PROPN lemmas + ADJ*+NOUN compounds.

    Uses a forward-buffer algorithm:
    - ADJ tokens accumulate in a buffer
    - When NOUN/PROPN is encountered: emit standalone noun AND compound if ADJs preceded it
    - Any other POS / punctuation clears the ADJ buffer

    Compounds are underscore-joined internally (e.g., "groot_blauw_auto") so that
    CountVectorizer treats them as single tokens. Underscores are replaced with
    spaces at display time by RepresentationEngine._denormalize_keywords().
    """
    if nlp is None:
        nlp = get_spacy_nlp(model_name)

    processed = []

    for doc in nlp.pipe(texts, batch_size=100):
        tokens_out = []
        adj_buffer = []

        for token in doc:
            if token.is_punct or token.is_space:
                adj_buffer = []
                continue

            if token.pos_ == 'ADJ':
                adj_buffer.append(token.lemma_.lower())
            elif token.pos_ in ('NOUN', 'PROPN'):
                noun_lemma = token.lemma_.lower()
                tokens_out.append(noun_lemma)
                if adj_buffer:
                    compound = "_".join(adj_buffer + [noun_lemma])
                    tokens_out.append(compound)
                adj_buffer = []
            else:
                adj_buffer = []

        processed.append(" ".join(tokens_out))

    return processed


def format_ladder_text(idea) -> str:
    """Format full ladder chain: instance -> rung_1 -> rung_2.

    Falls back to idea.idea when all fields are empty.
    """
    parts = []
    for field_name in ('instance', 'rung_1', 'rung_2'):
        val = (getattr(idea, field_name, '') or '').strip()
        if val:
            parts.append(val)
    return " -> ".join(parts) if parts else getattr(idea, 'idea', '')


def _get_single_field_text(idea, field: str) -> str:
    """Get text for a single field from an idea object."""
    if field == "ladder":
        return format_ladder_text(idea)
    if field == "idea":
        return idea.idea
    return (getattr(idea, field, '') or '').strip()


def get_idea_field_text(idea, field: str, separator: str = " | ") -> str:
    """Get text for a field (or composite of fields) from an idea object.

    Args:
        idea: Idea object with rung_1/rung_2/ladder fields
        field: One of:
            - Single field: "idea", "instance", "rung_1", "rung_2",
              "concept_type", "ladder"
            - Composite: "idea+rung_2", "rung_1+concept_type", etc.
              Fields joined with `separator`.
            "ladder" returns the full chain
            "instance -> rung_1 -> rung_2"
        separator: Join string for composite fields (default " | ").

    Returns:
        Text string (may be empty if all fields are not populated)
    """
    if "+" in field:
        parts = [_get_single_field_text(idea, f.strip()) for f in field.split("+")]
        return separator.join(p for p in parts if p)
    return _get_single_field_text(idea, field)


# =============================================================================
# SECTION 6a: REPRESENTATION MODELS (inlined from representation/)
# =============================================================================


class BaseRepresentation(ABC):
    """Base class for cluster representation models"""

    @abstractmethod
    def extract_topics(
        self,
        cluster_id: int,
        ctfidf_scores: np.ndarray,
        vocabulary: List[str],
        cluster_texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Extract representative keywords for a cluster

        Args:
            cluster_id: Cluster identifier
            ctfidf_scores: c-TF-IDF scores for this cluster (1D array)
            vocabulary: Feature names from vectorizer
            cluster_texts: Original idea texts in this cluster
            embeddings: Optional embeddings for ideas in cluster
            **kwargs: Additional model-specific parameters

        Returns:
            List of (keyword, score) tuples, ordered by relevance
        """
        pass


class ClassTfidfTransformer(TfidfTransformer):
    """
    A Class-based TF-IDF procedure using scikit-learn's TfidfTransformer as a base.

    c-TF-IDF is a TF-IDF formula adapted for multiple classes by joining all documents per class.
    Each class is converted to a single document instead of a set of documents.

    The formula:
    1. Term Frequency: Frequency of each word x for each class c, L1 normalized
    2. Inverse Document Frequency: log(1 + (avg_words_per_class / freq_of_word_across_classes))
    3. With BM25 weighting: log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
    """

    def __init__(self):
        """Initialize with BERTopic's recommended settings"""
        self.bm25_weighting = True
        self.reduce_frequent_words = True
        super(ClassTfidfTransformer, self).__init__()

    def fit(self, X: sp.csr_matrix):
        """Learn the idf vector (global term weights)."""
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64

        if self.use_idf:
            _, n_features = X.shape
            df = np.squeeze(np.asarray(X.sum(axis=0)))
            avg_nr_samples = int(X.sum(axis=1).mean())

            if self.bm25_weighting:
                idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
            else:
                idf = np.log((avg_nr_samples / df) + 1)

            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X: sp.csr_matrix):
        """Transform a count-based matrix to c-TF-IDF."""
        if self.use_idf:
            X = normalize(X, axis=1, norm="l1", copy=False)
            if self.reduce_frequent_words:
                X.data = np.sqrt(X.data)
            X = X * self._idf_diag
        return X


class CTfidfRepresentation(BaseRepresentation):
    """
    c-TF-IDF keyword extraction for clusters.

    Wrapper around ClassTfidfTransformer that provides a complete keyword
    extraction pipeline compatible with CoderingsTool's cluster structure.
    """

    def __init__(
        self,
        top_k: int = 15,
        bm25_weighting: bool = True,
        reduce_frequent_words: bool = True,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        language: str = "nl"
    ):
        self.top_k = top_k
        self.bm25_weighting = bm25_weighting
        self.reduce_frequent_words = reduce_frequent_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.language = language

        self.transformer = ClassTfidfTransformer()
        self.transformer.bm25_weighting = bm25_weighting
        self.transformer.reduce_frequent_words = reduce_frequent_words

        self.vectorizer = None
        self.vocabulary = None
        self.ctfidf_matrix = None

    def extract_keywords(
        self,
        clusters: Dict[int, List[str]],
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Extract top keywords for each cluster using c-TF-IDF."""
        if not clusters:
            if verbose:
                print("[c-TF-IDF] Warning: No clusters provided")
            return {}

        cluster_ids = sorted(clusters.keys())
        cluster_docs = [" ".join(clusters[cid]) for cid in cluster_ids]

        if verbose:
            print(f"\n[c-TF-IDF] Processing {len(cluster_docs)} clusters")
            print(f"[c-TF-IDF] Config: ngrams={self.ngram_range}, min_df={self.min_df}, "
                  f"max_df={self.max_df}, bm25={self.bm25_weighting}")

        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b"
        )

        try:
            count_matrix = self.vectorizer.fit_transform(cluster_docs)
            self.vocabulary = self.vectorizer.get_feature_names_out()

            if verbose:
                print(f"[c-TF-IDF] Vocabulary size: {len(self.vocabulary)}")
                print(f"[c-TF-IDF] Matrix shape: {count_matrix.shape}")

        except ValueError as e:
            if verbose:
                print(f"[c-TF-IDF] Error: Vectorization failed: {e}")
            return {}

        self.ctfidf_matrix = self.transformer.fit_transform(count_matrix)

        cluster_keywords = {}
        for idx, cluster_id in enumerate(cluster_ids):
            ctfidf_scores = self.ctfidf_matrix[idx].toarray()[0]
            keywords = self.extract_topics(
                cluster_id=cluster_id,
                ctfidf_scores=ctfidf_scores,
                vocabulary=self.vocabulary,
                cluster_texts=clusters[cluster_id]
            )
            cluster_keywords[cluster_id] = keywords

        if verbose:
            print(f"[c-TF-IDF] Extracted keywords for {len(cluster_keywords)} clusters\n")

        return cluster_keywords

    def extract_topics(
        self,
        cluster_id: int,
        ctfidf_scores: np.ndarray,
        vocabulary: List[str],
        cluster_texts: List[str],
        embeddings: np.ndarray = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Extract top keywords for a single cluster."""
        top_indices = np.argsort(ctfidf_scores)[-self.top_k:][::-1]
        keywords = [
            (vocabulary[i], float(ctfidf_scores[i]))
            for i in top_indices
            if ctfidf_scores[i] > 0
        ]
        return keywords


class MMRRepresentation(BaseRepresentation):
    """
    MMR keyword selection balancing relevance and diversity.

    Formula: MMR = argmax[lambda * relevance(w) - (1-lambda) * max_similarity(w, selected)]
    """

    def __init__(
        self,
        diversity: float = 0.3,
        top_k: int = 10,
        candidate_multiplier: int = 3
    ):
        if not 0.0 <= diversity <= 1.0:
            raise ValueError(f"diversity must be between 0.0 and 1.0, got {diversity}")
        self.diversity = diversity
        self.top_k = top_k
        self.candidate_multiplier = candidate_multiplier

    def extract_topics(
        self,
        cluster_id: int,
        ctfidf_scores: np.ndarray,
        vocabulary: List[str],
        cluster_texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Extract keywords using MMR for diversity."""
        n_candidates = min(
            self.top_k * self.candidate_multiplier,
            len([s for s in ctfidf_scores if s > 0])
        )
        if n_candidates == 0:
            return []

        candidate_indices = np.argsort(ctfidf_scores)[-n_candidates:][::-1]
        candidate_keywords = [vocabulary[i] for i in candidate_indices]
        candidate_scores = ctfidf_scores[candidate_indices]

        if candidate_scores.max() > 0:
            normalized_scores = candidate_scores / candidate_scores.max()
        else:
            normalized_scores = candidate_scores

        word_similarity = self._calculate_word_similarity(candidate_keywords, cluster_texts)

        selected_keywords = []
        selected_indices = []

        for _ in range(min(self.top_k, len(candidate_keywords))):
            if len(selected_indices) == 0:
                best_idx = 0
            else:
                mmr_scores = []
                for idx in range(len(candidate_keywords)):
                    if idx in selected_indices:
                        mmr_scores.append(-np.inf)
                        continue
                    relevance = normalized_scores[idx]
                    similarities = [
                        word_similarity[idx, sel_idx]
                        for sel_idx in selected_indices
                    ]
                    max_similarity = max(similarities) if similarities else 0.0
                    mmr = self.diversity * relevance - (1 - self.diversity) * max_similarity
                    mmr_scores.append(mmr)
                best_idx = np.argmax(mmr_scores)

            selected_indices.append(best_idx)
            keyword = candidate_keywords[best_idx]
            score = float(candidate_scores[best_idx])
            selected_keywords.append((keyword, score))

        return selected_keywords

    def _calculate_word_similarity(
        self,
        keywords: List[str],
        cluster_texts: List[str]
    ) -> np.ndarray:
        """Calculate word similarity matrix based on co-occurrence in texts."""
        n_keywords = len(keywords)
        occurrence = np.zeros((len(cluster_texts), n_keywords), dtype=int)

        for text_idx, text in enumerate(cluster_texts):
            text_lower = text.lower()
            for kw_idx, keyword in enumerate(keywords):
                if keyword.lower() in text_lower:
                    occurrence[text_idx, kw_idx] = 1

        similarity = np.zeros((n_keywords, n_keywords))
        for i in range(n_keywords):
            for j in range(i, n_keywords):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    vec_i = occurrence[:, i]
                    vec_j = occurrence[:, j]
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    if norm_i > 0 and norm_j > 0:
                        sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    else:
                        sim = 0.0
                    similarity[i, j] = sim
                    similarity[j, i] = sim

        return similarity

    def get_diversity_stats(
        self,
        keywords: List[Tuple[str, float]],
        cluster_texts: List[str]
    ) -> dict:
        """Calculate diversity statistics for selected keywords."""
        if not keywords:
            return {"avg_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}

        keyword_list = [kw for kw, _ in keywords]
        similarity_matrix = self._calculate_word_similarity(keyword_list, cluster_texts)

        n = len(keyword_list)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sims.append(similarity_matrix[i, j])

        if not pairwise_sims:
            return {"avg_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}

        return {
            "avg_similarity": float(np.mean(pairwise_sims)),
            "min_similarity": float(np.min(pairwise_sims)),
            "max_similarity": float(np.max(pairwise_sims)),
            "n_keywords": len(keywords)
        }


class TfidfRepresentation(BaseRepresentation):
    """
    Basic TF-IDF keyword extraction per cluster.

    Computes TF-IDF independently for each cluster's texts, treating each
    text as a document.
    """

    def __init__(
        self,
        top_k: int = 15,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95
    ):
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

    def extract_keywords(
        self,
        clusters: Dict[int, List[str]],
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Extract top keywords for each cluster using per-cluster TF-IDF."""
        if not clusters:
            if verbose:
                print("[TF-IDF] Warning: No clusters provided")
            return {}

        if verbose:
            print(f"\n[TF-IDF] Processing {len(clusters)} clusters (per-cluster)")

        cluster_keywords = {}
        for cluster_id, texts in sorted(clusters.items()):
            if len(texts) < 2:
                if verbose:
                    print(f"[TF-IDF] Cluster {cluster_id}: skipped (only {len(texts)} text)")
                cluster_keywords[cluster_id] = []
                continue
            keywords = self._extract_cluster_keywords(cluster_id, texts, verbose)
            cluster_keywords[cluster_id] = keywords

        if verbose:
            print(f"[TF-IDF] Extracted keywords for {len(cluster_keywords)} clusters\n")

        return cluster_keywords

    def _extract_cluster_keywords(
        self,
        cluster_id: int,
        texts: List[str],
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """Extract keywords from a single cluster's texts using TF-IDF."""
        try:
            effective_min_df = min(self.min_df, max(1, len(texts) // 3))
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                min_df=effective_min_df,
                max_df=self.max_df,
                lowercase=True,
                token_pattern=r"(?u)\b\w\w+\b"
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            vocabulary = vectorizer.get_feature_names_out()

            if len(vocabulary) == 0:
                if verbose:
                    print(f"[TF-IDF] Cluster {cluster_id}: no vocabulary extracted")
                return []

            avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = np.argsort(avg_scores)[-self.top_k:][::-1]
            keywords = [
                (vocabulary[i], float(avg_scores[i]))
                for i in top_indices
                if avg_scores[i] > 0
            ]
            return keywords

        except ValueError as e:
            if verbose:
                print(f"[TF-IDF] Cluster {cluster_id}: error - {e}")
            return []

    def extract_topics(
        self,
        cluster_id: int,
        ctfidf_scores: np.ndarray,
        vocabulary: List[str],
        cluster_texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Extract keywords using BaseRepresentation interface."""
        return self._extract_cluster_keywords(cluster_id, cluster_texts, verbose=False)


class RepresentationEngine:
    """
    Keyword extraction for clusters using multiple representation methods.

    Supports c-TF-IDF (primary), MMR (diversity-aware), and basic TF-IDF.
    """

    def __init__(self, config: ClustererConfig):
        self.config = config
        self._ctfidf = None
        self._mmr = None
        self._tfidf = None
        self._current_ngram_range: Optional[Tuple[int, int]] = None

    def _get_effective_ngram_range(self, embedding_text_format: Optional[str]) -> Tuple[int, int]:
        """Determine n-gram range based on embedding text format."""
        return self.config.ctfidf_ngram_range

    def _ensure_ctfidf(self, ngram_range: Optional[Tuple[int, int]] = None):
        """Lazy initialization of c-TF-IDF model."""
        effective_range = ngram_range or self.config.ctfidf_ngram_range

        if self._ctfidf is not None and self._current_ngram_range != effective_range:
            self._ctfidf = None

        if self._ctfidf is None:
            self._ctfidf = CTfidfRepresentation(
                top_k=self.config.ctfidf_top_k,
                bm25_weighting=self.config.ctfidf_bm25_weighting,
                reduce_frequent_words=self.config.ctfidf_reduce_frequent_words,
                ngram_range=effective_range,
                min_df=self.config.ctfidf_min_df,
                max_df=0.95,
                language="nl"
            )
            self._current_ngram_range = effective_range

    def _ensure_mmr(self):
        """Lazy initialization of MMR model."""
        if self._mmr is None:
            self._mmr = MMRRepresentation(
                diversity=self.config.mmr_diversity,
                top_k=self.config.ctfidf_top_k,
                candidate_multiplier=self.config.mmr_candidate_multiplier
            )

    def _ensure_tfidf(self, ngram_range: Optional[Tuple[int, int]] = None):
        """Lazy initialization of basic TF-IDF model."""
        effective_range = ngram_range or self.config.ctfidf_ngram_range

        if self._tfidf is not None and self._current_ngram_range != effective_range:
            self._tfidf = None

        if self._tfidf is None:
            self._tfidf = TfidfRepresentation(
                top_k=self.config.ctfidf_top_k,
                ngram_range=effective_range,
                min_df=self.config.ctfidf_min_df
            )

    def extract_all_keywords(
        self,
        cluster_texts: Dict[int, List[str]],
        embedding_text_format: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
        """Extract keywords using all enabled representation methods."""
        results = {}

        effective_ngram_range = self._get_effective_ngram_range(embedding_text_format)

        if verbose:
            if effective_ngram_range != self.config.ctfidf_ngram_range:
                print(f"  N-gram range: {effective_ngram_range} (auto-detected for {embedding_text_format})")
            else:
                print(f"  N-gram range: {effective_ngram_range}")

        cleaned_clusters = self._preprocess_texts(
            cluster_texts,
            embedding_text_format,
            verbose
        )

        if self.config.generate_ctfidf:
            self._ensure_ctfidf(ngram_range=effective_ngram_range)
            if verbose:
                print("\n[c-TF-IDF] Extracting keywords...")
            results["ctfidf"] = self._ctfidf.extract_keywords(cleaned_clusters, verbose=verbose)

        if self.config.generate_mmr_keywords:
            self._ensure_ctfidf(ngram_range=effective_ngram_range)
            self._ensure_mmr()
            if verbose:
                print(f"\n[MMR] Extracting keywords (diversity={self.config.mmr_diversity})...")
            results["mmr"] = self._extract_mmr_keywords(cleaned_clusters, effective_ngram_range, verbose)

        if self.config.generate_tfidf_keywords:
            self._ensure_tfidf(ngram_range=effective_ngram_range)
            if verbose:
                print("\n[TF-IDF] Extracting keywords (per-cluster)...")
            results["tfidf"] = self._tfidf.extract_keywords(cleaned_clusters, verbose=verbose)

        return self._denormalize_keywords(results)

    @staticmethod
    def _denormalize_keywords(
        results: Dict[str, Dict[int, List[Tuple[str, float]]]]
    ) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
        """Replace underscore-joined compound tokens with space-separated for display."""
        return {
            method: {
                cid: [(kw.replace("_", " "), score) for kw, score in kws]
                for cid, kws in cluster_kws.items()
            }
            for method, cluster_kws in results.items()
        }

    def _preprocess_texts(
        self,
        cluster_texts: Dict[int, List[str]],
        embedding_text_format: Optional[str],
        verbose: bool
    ) -> Dict[int, List[str]]:
        """Preprocess texts: optional lemmatization."""
        cleaned_clusters = dict(cluster_texts)

        if verbose:
            format_display = embedding_text_format or "idea (default)"
            print(f"  Text format: {format_display}")

        if self.config.ctfidf_use_lemmatization:
            if verbose:
                print("  Applying spaCy lemmatization (NOUN/PROPN + ADJ+NOUN compounds)...")

            all_texts = []
            cluster_offsets = {}
            offset = 0
            for cluster_id, texts in cleaned_clusters.items():
                cluster_offsets[cluster_id] = (offset, offset + len(texts))
                all_texts.extend(texts)
                offset += len(texts)

            lemmatized_all = extract_noun_phrases_lemmatized(
                all_texts,
                model_name=self.config.ctfidf_spacy_model
            )

            lemmatized_clusters = {}
            for cluster_id, (start, end) in cluster_offsets.items():
                lemmatized_clusters[cluster_id] = lemmatized_all[start:end]

            cleaned_clusters = lemmatized_clusters

        return cleaned_clusters

    def _extract_mmr_keywords(
        self,
        cleaned_clusters: Dict[int, List[str]],
        ngram_range: Tuple[int, int],
        verbose: bool
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Extract keywords using MMR (diversity-aware selection)."""
        cluster_ids = sorted(cleaned_clusters.keys())
        cluster_docs = [" ".join(cleaned_clusters[cid]) for cid in cluster_ids]

        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=self.config.ctfidf_min_df,
            max_df=0.95,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b"
        )

        try:
            count_matrix = vectorizer.fit_transform(cluster_docs)
            vocabulary = list(vectorizer.get_feature_names_out())
        except ValueError:
            if verbose:
                print("[MMR] Warning: Vectorization failed")
            return {}

        self._ensure_ctfidf()
        ctfidf_matrix = self._ctfidf.transformer.fit_transform(count_matrix)

        mmr_keywords = {}
        for idx, cluster_id in enumerate(cluster_ids):
            ctfidf_scores = ctfidf_matrix[idx].toarray()[0]
            cluster_texts = cleaned_clusters[cluster_id]

            keywords = self._mmr.extract_topics(
                cluster_id=cluster_id,
                ctfidf_scores=ctfidf_scores,
                vocabulary=vocabulary,
                cluster_texts=cluster_texts
            )
            mmr_keywords[cluster_id] = keywords

        if verbose:
            print(f"[MMR] Extracted keywords for {len(mmr_keywords)} clusters")

        return mmr_keywords

    def extract_all_keywords_from_labels(
        self,
        labels: np.ndarray,
        idea_texts: List[str],
        embedding_text_format: Optional[str] = None,
        probabilities: Optional[np.ndarray] = None,
        min_probability: Optional[float] = None,
        verbose: bool = False
    ) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
        """Extract all keywords given cluster labels and idea texts.

        Args:
            labels: Cluster assignments for each idea
            idea_texts: Text of each idea
            embedding_text_format: Text format used for embeddings
            probabilities: Optional HDBSCAN cluster membership probabilities
            min_probability: Only include ideas with probability > this threshold
            verbose: Enable verbose output
        """
        cluster_texts = {}
        for i, label in enumerate(labels):
            if label >= 0:
                # Filter by probability if provided
                if probabilities is not None and min_probability is not None:
                    if probabilities[i] <= min_probability:
                        continue
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(idea_texts[i])

        return self.extract_all_keywords(
            cluster_texts,
            embedding_text_format=embedding_text_format,
            verbose=verbose
        )


# =============================================================================
# SECTION 7: LABEL GENERATION
# =============================================================================

@dataclass
class ClusterLabel:
    """Container for cluster label information."""
    cluster_id: int
    theme: str
    description: str
    key_concepts: List[str]
    n_ideas: int


class LabelGenerator:
    """
    LLM-based cluster label generator.

    Usage:
        generator = LabelGenerator(config)
        labels = generator.generate_all_labels(cluster_texts, cluster_keywords)
    """

    def __init__(self, config: ClustererConfig):
        """Initialize LabelGenerator."""
        self.config = config
        self._model = config.llm_labels_model
        self._max_ideas = config.llm_max_ideas_per_cluster

    def generate_label(
        self,
        cluster_id: int,
        ideas: List[str],
        dataset_placeholders: "DatasetPlaceholders",
        representative_samples: Optional[List[Tuple[str, float]]] = None,
        keywords: Optional[List[Tuple[str, float]]] = None,
        cluster_distributions: Optional[Dict[str, Dict[str, float]]] = None,
        verbose: bool = False,
        return_prompt: bool = False
    ) -> ClusterLabel:
        """Generate LLM-based label for a single cluster."""
        if representative_samples:
            sample_ideas = [text for text, _ in representative_samples]
        else:
            sample_ideas = ideas
            if len(ideas) > self._max_ideas:
                sample_ideas = random.sample(ideas, self._max_ideas)

        seen = set()
        unique_samples = []
        for idea in sample_ideas:
            if idea not in seen:
                seen.add(idea)
                unique_samples.append(idea)
        sample_ideas = unique_samples

        placeholders = build_cluster_placeholders(
            dataset_placeholders,
            cluster_id=cluster_id,
            num_ideas=len(ideas),
            sample_ideas=tuple(sample_ideas),
            keywords=tuple(keywords) if keywords else (),
            distributions=cluster_distributions,
        )
        prompt = CLUSTER_DESCRIPTION_PROMPT.format(**placeholders)

        try:
            client = create_client(model=self._model, async_mode=False)
            description = llm_create_sync(
                client=client,
                model=self._model,
                prompt=prompt,
                response_model=ClusterDescription,
                temperature=0.3,
                max_tokens=1000
            )

            label = ClusterLabel(
                cluster_id=cluster_id,
                theme=description.theme,
                description=description.description,
                key_concepts=description.key_concepts,
                n_ideas=len(ideas)
            )
            return (label, prompt) if return_prompt else label

        except Exception as e:
            if verbose:
                print(f"  LLM Error for cluster {cluster_id}: {type(e).__name__}: {e}")

            label = ClusterLabel(
                cluster_id=cluster_id,
                theme=f"Cluster {cluster_id}",
                description="LLM label generation failed",
                key_concepts=[],
                n_ideas=len(ideas)
            )
            return (label, prompt) if return_prompt else label

    def generate_all_labels(
        self,
        cluster_texts: Dict[int, List[str]],
        dataset_placeholders: "DatasetPlaceholders",
        cluster_keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        representative_samples: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        cluster_distributions: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
        verbose: bool = False
    ) -> Dict[int, ClusterLabel]:
        """Generate LLM-based labels for all clusters."""
        if not self.config.generate_llm_labels:
            return {}

        if verbose:
            print(f"\n[LLM Label Generation]")
            print(f"  Model: {self._model}")
            print(f"  Clusters to label: {len(cluster_texts)}")
            print(f"  Language: {dataset_placeholders.language}")
            if dataset_placeholders.facet_context:
                print(f"  Primary facet context: present")
            if dataset_placeholders.concept_types_section:
                print(f"  Concept types: present")

        labels = {}
        sample_prompt = None

        for i, (cluster_id, ideas) in enumerate(sorted(cluster_texts.items())):
            keywords = cluster_keywords.get(cluster_id) if cluster_keywords else None
            samples = representative_samples.get(cluster_id) if representative_samples else None
            distributions = cluster_distributions.get(cluster_id) if cluster_distributions else None
            is_first = (i == 0)

            if verbose:
                sample_info = f", {len(samples)} representative" if samples else ""
                print(f"  Generating label for cluster {cluster_id} ({len(ideas)} ideas{sample_info})...", end=" ")

            result = self.generate_label(
                cluster_id=cluster_id,
                ideas=ideas,
                dataset_placeholders=dataset_placeholders,
                representative_samples=samples,
                keywords=keywords,
                cluster_distributions=distributions,
                verbose=verbose,
                return_prompt=is_first
            )

            if is_first:
                label, sample_prompt = result
            else:
                label = result
            labels[cluster_id] = label

            if verbose:
                theme_display = label.theme[:50] + "..." if len(label.theme) > 50 else label.theme
                print(f"'{theme_display}'")

        if verbose and sample_prompt:
            print(f"\n  [Sample LLM Prompt (cluster 0)]")
            print("  " + "-" * 70)
            for line in sample_prompt.split('\n'):
                print(f"  {line}")
            print("  " + "-" * 70)

        return labels
