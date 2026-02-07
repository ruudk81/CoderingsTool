"""
Clusterer Helpers Module - EXPERIMENTAL VERSION

This is an isolated copy for experimentation in clusterer_v3.
Changes here do NOT affect the production pipeline.

Original: src/utils/clusterer_helpers.py

Consolidates all helper classes and functions for clustering:
- Section 1: Preprocessing (L2 normalization, PCA, embedding extraction)
- Section 2: Algorithm Selection (DVC, knee detection)
- Section 3: Parameter Optimization (Optuna HDBSCAN)
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
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator
import hdbscan
import umap
import optuna
from optuna.samplers import GridSampler
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import models
from utils.llm import llm_create_sync, create_client
from .config_clusterer_exp import ClustererConfig, resolve_embedding_source
# Toggle: use experimental or production prompts
# Keep in sync with USE_EXPERIMENTAL_CLUSTERER in run_experiment.py
USE_EXPERIMENTAL_PROMPTS = True

if USE_EXPERIMENTAL_PROMPTS:
    from .prompts_exp import CLUSTER_DESCRIPTION_PROMPT, ClusterDescription
else:
    from prompts import CLUSTER_DESCRIPTION_PROMPT, ClusterDescription

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


def format_ontology_text(idea) -> str:
    """Format ontology fields into text: 'instance - node (category)'.

    Mirrors the logic from src/utils/embedder.py _format_ontology_text().
    Falls back to idea.idea when ontology is None or all fields are empty.
    """
    ontology = getattr(idea, 'ontology', None)
    if ontology is None:
        return getattr(idea, 'idea', '')
    instance = (getattr(ontology, 'instance', '') or '').strip()
    node = (getattr(ontology, 'node', '') or '').strip()
    category = (getattr(ontology, 'category', '') or '').strip()
    parts = []
    if instance:
        parts.append(instance)
    if node:
        if category:
            parts.append(f"{node} ({category})")
        else:
            parts.append(node)
    elif category:
        parts.append(f"({category})")
    result = " - ".join(parts) if parts else ""
    return result if result else getattr(idea, 'idea', '')


def extract_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererConfig
) -> Tuple[np.ndarray, List[str], List[str], List[str], List[Tuple[int, int]], Optional[str], Optional[str]]:
    """
    Extract embeddings from EmbeddingsModel list.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererConfig (uses config.embedding_source to select embedding field)

    Returns:
        embeddings: Array of shape (n_ideas, embedding_dim)
        idea_texts: List of idea text strings (idea.idea)
        taxonomy_phrases: List of taxonomy phrase strings (idea.taxonomy_phrase)
        ontology_texts: List of ontology text strings ("instance - node (category)")
        idea_indices: List of (response_idx, idea_idx) tuples for result mapping
        template_prefix: The canonical phrasing prefix (if available)
        embedding_text_format: The text format used for embedding (if available)
    """
    # Detect embedding_text_format from cached data (first response that has it)
    embedding_text_format = None
    template_prefix = None
    for response in input_list:
        if embedding_text_format is None and hasattr(response, 'embedding_text_format') and response.embedding_text_format:
            embedding_text_format = response.embedding_text_format
        if template_prefix is None and hasattr(response, 'template_prefix') and response.template_prefix:
            template_prefix = response.template_prefix
        if embedding_text_format and template_prefix:
            break

    # Resolve "auto" → concrete field based on cached embedding_text_format
    embedding_field = resolve_embedding_source(
        embedding_text_format or "idea",
        config.embedding_source
    )

    if config.verbose and config.embedding_source == "auto":
        print(f"Auto-resolved embedding source: '{embedding_text_format}' → {embedding_field}")

    embeddings_list = []
    idea_texts = []
    taxonomy_phrases = []
    ontology_texts = []
    idea_indices = []

    for resp_idx, response in enumerate(input_list):
        if response.response_ideas:
            for idea_idx, idea in enumerate(response.response_ideas):
                emb = getattr(idea, embedding_field, None)
                if emb is not None:
                    embeddings_list.append(emb)
                    idea_texts.append(idea.idea if hasattr(idea, 'idea') else str(idea))
                    taxonomy_phrases.append(
                        idea.taxonomy_phrase if hasattr(idea, 'taxonomy_phrase') and idea.taxonomy_phrase else ""
                    )
                    ontology_texts.append(format_ontology_text(idea))
                    idea_indices.append((resp_idx, idea_idx))

    if not embeddings_list:
        raise ValueError(
            f"No embeddings found for field '{embedding_field}' in input data. "
            f"Cached embedding_text_format='{embedding_text_format}'. "
            f"Re-run step 4 with a compatible format or set EMBEDDING_SOURCE explicitly."
        )

    embeddings = np.vstack(embeddings_list)

    if config.verbose:
        print(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        print(f"Embedding source: {embedding_field}")
        if template_prefix:
            prefix_display = template_prefix[:50] + "..." if len(template_prefix) > 50 else template_prefix
            print(f"Template prefix: '{prefix_display}'")
        if embedding_text_format:
            print(f"Embedding text format: {embedding_text_format}")

    return embeddings, idea_texts, taxonomy_phrases, ontology_texts, idea_indices, template_prefix, embedding_text_format


def preprocess_embeddings(
    input_list: List[models.EmbeddingsModel],
    config: ClustererConfig
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], List[Tuple[int, int]], Optional[PCA], Optional[str], Optional[str]]:
    """
    Full preprocessing pipeline: extract, normalize, optionally PCA.

    Args:
        input_list: List of EmbeddingsModel instances
        config: ClustererConfig

    Returns:
        embeddings_normalized: L2-normalized original embeddings
        embeddings_processed: Processed embeddings (may be PCA-reduced)
        idea_texts: List of idea text strings (idea.idea)
        taxonomy_phrases: List of taxonomy phrase strings (idea.taxonomy_phrase)
        ontology_texts: List of ontology text strings ("instance - node (category)")
        idea_indices: List of (response_idx, idea_idx) tuples
        pca_model: Fitted PCA model (or None if not applied)
        template_prefix: The canonical phrasing prefix (if available)
        embedding_text_format: The text format used for embedding (if available)
    """
    # Extract embeddings
    embeddings, idea_texts, taxonomy_phrases, ontology_texts, idea_indices, template_prefix, embedding_text_format = extract_embeddings(input_list, config)
    n_samples = len(embeddings)

    # L2 normalize
    embeddings_normalized = l2_normalize(embeddings)

    if config.verbose:
        print(f"L2-normalized {n_samples} embeddings")

    # Apply PCA for large datasets
    pca_model = None
    if n_samples > config.pca_threshold:
        if config.verbose:
            print(f"Applying PCA (n > {config.pca_threshold})...")
        embeddings_processed, pca_model = apply_pca(
            embeddings_normalized,
            n_components=config.pca_variance_retained,
            random_state=config.umap_random_state
        )
        # Re-normalize after PCA
        embeddings_processed = l2_normalize(embeddings_processed)
        if config.verbose:
            print(f"PCA reduced to {embeddings_processed.shape[1]} components")
    else:
        embeddings_processed = embeddings_normalized

    return embeddings_normalized, embeddings_processed, idea_texts, taxonomy_phrases, ontology_texts, idea_indices, pca_model, template_prefix, embedding_text_format


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
        kneedle_S = max(1.0, n / self.config.knee_s_denominator)
        interp_method = "polynomial" if n < self.config.knee_interp_threshold else "interp1d"

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

    Formula: 0.5*sqrt(n) to 1.5*sqrt(n), log-spaced with k points.
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
    """
    sqrt_n = math.sqrt(N)
    low = max(nn_min, int(round(low_mult * sqrt_n)))
    high = min(nn_max, int(round(high_mult * sqrt_n)))

    # Safety: ensure bounds are valid
    high = min(high, N - 1)
    low = min(low, high)

    return log_spaced_ints(low, high, k=k)


def mcs_bounds(
    N: int,
    low_pct: float = 0.05,
    low_log_mult: float = 2.0,
    high_mult: float = 1.0,
    mcs_min: int = 5
) -> Tuple[int, int]:
    """
    Compute min_cluster_size bounds.

    Formula:
        low = max(mcs_min, min(low_pct * N, low_log_mult * ln(N)))
        high = high_mult * sqrt(N)

    Args:
        N: Dataset size
        low_pct: Low bound as percentage of N (default 0.05)
        low_log_mult: Low bound multiplier for ln(N) (default 2.0)
        high_mult: High bound multiplier for sqrt(N) (default 1.0)
        mcs_min: Absolute minimum MCS (default 5)

    Returns:
        (low, high) bounds for min_cluster_size
    """
    log_n = math.log(max(N, 2))
    sqrt_n = math.sqrt(N)
    low = max(mcs_min, int(round(min(low_pct * N, low_log_mult * log_n))))
    high = max(low, int(round(high_mult * sqrt_n)))
    return low, high


def mcs_grid(
    N: int,
    k: int = 4,
    low_pct: float = 0.05,
    low_log_mult: float = 2.0,
    high_mult: float = 1.0,
    mcs_min: int = 5
) -> List[int]:
    """
    Generate min_cluster_size grid for dataset of size N.

    Formula: lower=min(0.05*N, 2*ln(N)), upper=sqrt(N), log-spaced.

    Args:
        N: Dataset size
        k: Number of grid points (default 4)
        low_pct: Low bound as percentage of N (default 0.05)
        low_log_mult: Low bound multiplier for ln(N) (default 2.0)
        high_mult: High bound multiplier for sqrt(N) (default 1.0)
        mcs_min: Absolute minimum MCS (default 5)

    Returns:
        Log-spaced list of min_cluster_size values
    """
    low, high = mcs_bounds(N, low_pct, low_log_mult, high_mult, mcs_min)
    return log_spaced_ints(low, high, k=k)


def ms_grid(
    N: int,
    k: int = 4,
    low_log_mult: float = 1.0,
    high_sqrt_mult: float = 0.5,
    mcs_lower: int = 5
) -> List[int]:
    """
    Generate min_samples grid for dataset of size N.

    Formula: lower=ln(N), upper=sqrt(N)/2, log-spaced.
    Safety: clamp so min_ms <= max_ms and min_ms <= mcs_lower.

    Args:
        N: Dataset size
        k: Number of grid points (default 4)
        low_log_mult: Low bound multiplier for ln(N) (default 1.0)
        high_sqrt_mult: High bound multiplier for sqrt(N) (default 0.5)
        mcs_lower: Lower bound of MCS grid, used to ensure ms <= mcs

    Returns:
        Log-spaced list of min_samples values
    """
    log_n = math.log(max(N, 2))
    sqrt_n = math.sqrt(N)
    low = max(1, int(round(low_log_mult * log_n)))
    high = max(1, int(round(high_sqrt_mult * sqrt_n)))

    # Safety: ensure min_ms <= mcs_lower
    low = min(low, mcs_lower)
    high = min(high, mcs_lower) if high > mcs_lower else high

    # Safety: ensure low <= high
    if low > high:
        low = high

    return log_spaced_ints(low, high, k=k)


def create_search_space(N: int, config: ClustererConfig) -> Dict[str, List]:
    """
    Create Optuna search space dict for GridSampler using config values.

    Initial grid search uses ms = mcs // 2 (not a separate grid dimension).

    Args:
        N: Dataset size
        config: ClustererConfig with grid parameters

    Returns:
        Dict with 'n_neighbors', 'n_components', 'min_dist', 'min_cluster_size' grids
    """
    mcs_values = mcs_grid(
        N,
        k=config.min_cluster_size_grid_k,
        low_pct=config.mcs_low_pct,
        low_log_mult=config.mcs_low_log_mult,
        high_mult=config.mcs_high_mult,
        mcs_min=config.mcs_min
    )

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
        'min_cluster_size': mcs_values,
    }


# =============================================================================
# PARETO FRONTIER SELECTION
# =============================================================================


def compute_pareto_k_min(N: int, config: ClustererConfig) -> int:
    """
    Compute minimum allowed k for Pareto filtering.

    Formula: k >= pareto_min_k_sqrt_mult * sqrt(N)  (default 0.5 * sqrt(N))

    Args:
        N: Dataset size
        config: ClustererConfig with pareto_min_k_sqrt_mult

    Returns:
        Minimum allowed number of clusters
    """
    return max(1, int(config.pareto_min_k_sqrt_mult * math.sqrt(N)))


def compute_pareto_k_max(N: int, config: ClustererConfig) -> int:
    """
    Compute maximum allowed k for Pareto filtering.

    Formula:
    - N < threshold: k <= N / (4 * mcs_lower), where mcs_lower = min(0.05*N, ln(N))
    - N >= threshold: k <= 0.8 * sqrt(N)

    Args:
        N: Dataset size
        config: ClustererConfig with pareto_* and mcs_* parameters

    Returns:
        Maximum allowed number of clusters
    """
    if N < config.pareto_k_small_n_threshold:
        log_n = math.log(max(N, 2))
        mcs_lower = max(config.mcs_min, int(round(min(config.mcs_low_pct * N, config.mcs_low_log_mult * log_n))))
        k_max = int(N / (4 * mcs_lower))
    else:
        k_max = int(config.pareto_max_k_sqrt_mult * math.sqrt(N))
    return max(1, k_max)


def filter_candidates_by_hard_constraints(
    study: optuna.Study,
    N: int,
    config: ClustererConfig,
    verbose: bool = True
) -> List[optuna.trial.FrozenTrial]:
    """
    Filter completed Optuna trials by hard constraints.

    Constraints:
    - DBCV (relative_validity) > pareto_min_dbcv
    - k (n_clusters) in [k_min, k_max]
    - noise_rate <= pareto_max_noise_rate
    - max_cluster_ratio <= pareto_max_cluster_ratio

    Falls back progressively if no candidates pass:
    1. Drop DBCV constraint
    2. Drop noise + max_cluster_ratio constraints
    3. Use all completed trials

    Args:
        study: Completed Optuna study
        N: Dataset size
        config: ClustererConfig with pareto_* parameters
        verbose: Print filtering details

    Returns:
        List of trials passing constraints (with fallback if needed)
    """
    k_min = compute_pareto_k_min(N, config)
    k_max = compute_pareto_k_max(N, config)

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # Full constraints: DBCV + k range + noise + max_cluster_ratio
    candidates = [
        t for t in completed
        if t.user_attrs.get('relative_validity', -1.0) > config.pareto_min_dbcv
        and k_min <= t.user_attrs.get('n_clusters', 0) <= k_max
        and t.user_attrs.get('noise_rate', 1.0) <= config.pareto_max_noise_rate
        and t.user_attrs.get('max_cluster_ratio', 1.0) <= config.pareto_max_cluster_ratio
    ]

    if candidates:
        if verbose:
            print(f"  {len(candidates)}/{len(completed)} trials pass all hard constraints "
                  f"(DBCV>{config.pareto_min_dbcv}, k in [{k_min}, {k_max}], "
                  f"noise<={config.pareto_max_noise_rate}, "
                  f"max_cluster_ratio<={config.pareto_max_cluster_ratio})")
        return candidates

    # Fallback 1: drop DBCV, keep k + noise + max_cluster_ratio
    candidates = [
        t for t in completed
        if k_min <= t.user_attrs.get('n_clusters', 0) <= k_max
        and t.user_attrs.get('noise_rate', 1.0) <= config.pareto_max_noise_rate
        and t.user_attrs.get('max_cluster_ratio', 1.0) <= config.pareto_max_cluster_ratio
    ]
    if candidates:
        if verbose:
            print(f"  WARNING: No candidates with DBCV>{config.pareto_min_dbcv}. "
                  f"Relaxed to k in [{k_min}, {k_max}] + noise<={config.pareto_max_noise_rate} "
                  f"+ max_cluster_ratio<={config.pareto_max_cluster_ratio}: "
                  f"{len(candidates)}/{len(completed)} trials")
        return candidates

    # Fallback 2: drop DBCV + noise + max_cluster_ratio, keep k range only
    candidates = [
        t for t in completed
        if k_min <= t.user_attrs.get('n_clusters', 0) <= k_max
    ]
    if candidates:
        if verbose:
            print(f"  WARNING: No candidates with noise/max_cluster_ratio constraints. "
                  f"Relaxed to k in [{k_min}, {k_max}] only: {len(candidates)}/{len(completed)} trials")
        return candidates

    # Fallback 3: all completed trials
    if verbose:
        print(f"  WARNING: No candidates pass any constraints (k in [{k_min}, {k_max}]). "
              f"Using all {len(completed)} completed trials")
    return completed


def compute_pareto_front(
    candidates: List[optuna.trial.FrozenTrial]
) -> List[optuna.trial.FrozenTrial]:
    """
    Compute Pareto frontier over 4 objectives.

    Objectives:
    - relative_validity (DBCV): maximize
    - n_clusters: minimize
    - low_prob_ratio: minimize
    - max_cluster_ratio: minimize

    Coherence and noise are handled as hard constraints, not Pareto objectives.

    A candidate is non-dominated if no other candidate is better on ALL objectives.

    Args:
        candidates: List of trials passing hard constraints

    Returns:
        List of non-dominated trials (Pareto front)
    """
    if len(candidates) <= 1:
        return list(candidates)

    def objectives(trial):
        return (
            trial.user_attrs.get('relative_validity', 0.0),    # maximize
            -trial.user_attrs.get('n_clusters', 0),             # minimize -> negate
            -trial.user_attrs.get('low_prob_ratio', 1.0),       # minimize -> negate
            -trial.user_attrs.get('max_cluster_ratio', 1.0),    # minimize -> negate
        )

    def dominates(a_obj, b_obj):
        """True if a dominates b (a >= b on all, a > b on at least one)."""
        at_least_as_good = all(ai >= bi for ai, bi in zip(a_obj, b_obj))
        strictly_better = any(ai > bi for ai, bi in zip(a_obj, b_obj))
        return at_least_as_good and strictly_better

    obj_values = [objectives(c) for c in candidates]
    pareto = []
    for i, c in enumerate(candidates):
        is_dominated = any(
            dominates(obj_values[j], obj_values[i])
            for j in range(len(candidates)) if j != i
        )
        if not is_dominated:
            pareto.append(c)

    return pareto


def select_from_pareto_front(
    pareto_front: List[optuna.trial.FrozenTrial],
    all_candidates: List[optuna.trial.FrozenTrial],
    config: ClustererConfig,
    verbose: bool = True
) -> optuna.trial.FrozenTrial:
    """
    Select best solution from Pareto front using weighted distance to ideal point.

    Each objective is normalized using percentile bounds (p5/p95) from all_candidates
    for outlier robustness. Values are clipped to [0,1] after normalization.
    Minimize-objectives are flipped so higher=better for all.
    The solution closest (weighted Euclidean) to the ideal point [1,1,1,1] is selected.

    4 objectives:
    - DBCV (relative_validity): maximize
    - n_clusters: minimize
    - low_prob_ratio: minimize
    - max_cluster_ratio: minimize

    Coherence and noise are handled as hard constraints, not Pareto objectives.

    Args:
        pareto_front: Non-dominated solutions
        all_candidates: All solutions passing hard constraints (for normalization bounds)
        config: ClustererConfig with pareto_weight_* and pareto_norm_percentile_* fields
        verbose: Print selection details

    Returns:
        Selected trial from Pareto front
    """
    if len(pareto_front) == 1:
        if verbose:
            t = pareto_front[0]
            print(f"\n[Pareto] Single solution on front: "
                  f"DBCV={t.user_attrs.get('relative_validity', 0):.4f}, "
                  f"k={t.user_attrs.get('n_clusters', 0)}, "
                  f"low_prob={t.user_attrs.get('low_prob_ratio', 0):.3f}, "
                  f"max_clust={t.user_attrs.get('max_cluster_ratio', 0):.3f}")
        return pareto_front[0]

    def extract(trial):
        return np.array([
            trial.user_attrs.get('relative_validity', 0.0),
            trial.user_attrs.get('n_clusters', 0),
            trial.user_attrs.get('low_prob_ratio', 1.0),
            trial.user_attrs.get('max_cluster_ratio', 1.0),
        ])

    # Percentile normalization bounds from all candidates (p5/p95)
    all_vals = np.array([extract(t) for t in all_candidates])
    lo = np.percentile(all_vals, config.pareto_norm_percentile_low, axis=0)
    hi = np.percentile(all_vals, config.pareto_norm_percentile_high, axis=0)
    ranges = hi - lo
    ranges[ranges == 0] = 1.0  # avoid division by zero

    weights = np.array([
        config.pareto_weight_dbcv,
        config.pareto_weight_k,
        config.pareto_weight_low_prob_ratio,
        config.pareto_weight_max_cluster_ratio,
    ])

    best_dist = float('inf')
    best_trial = pareto_front[0]

    for trial in pareto_front:
        vals = extract(trial)
        normalized = np.clip((vals - lo) / ranges, 0.0, 1.0)

        # Flip minimize-objectives so higher=better for all
        # [0] DBCV: maximize (keep)
        # [1] n_clusters: minimize -> flip
        # [2] low_prob_ratio: minimize -> flip
        # [3] max_cluster_ratio: minimize -> flip
        normalized[1] = 1.0 - normalized[1]
        normalized[2] = 1.0 - normalized[2]
        normalized[3] = 1.0 - normalized[3]

        dist = np.sqrt(np.sum(weights * (1.0 - normalized) ** 2))

        if dist < best_dist:
            best_dist = dist
            best_trial = trial

    if verbose:
        t = best_trial
        print(f"\n[Pareto] Selected from {len(pareto_front)} Pareto-optimal solutions "
              f"(distance to ideal: {best_dist:.4f})")
        print(f"  DBCV={t.user_attrs.get('relative_validity', 0):.4f}, "
              f"k={t.user_attrs.get('n_clusters', 0)}, "
              f"low_prob={t.user_attrs.get('low_prob_ratio', 0):.3f}, "
              f"max_clust={t.user_attrs.get('max_cluster_ratio', 0):.3f}")
        print(f"  (coh={t.user_attrs.get('coherence', t.value):.4f}, "
              f"noise={t.user_attrs.get('noise_rate', 0):.3f})")
        print(f"  Params: nn={t.params.get('n_neighbors')}, "
              f"nc={t.params.get('n_components')}, "
              f"md={t.params.get('min_dist')}, "
              f"mcs={t.params.get('min_cluster_size')}")

    return best_trial


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
        self._study: Optional[optuna.Study] = None
        self._extended_study: Optional[optuna.Study] = None
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

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function maximizing DBCV (relative_validity).

        DBCV measures density-based cluster validity — how well-separated
        and internally dense the clusters are.

        Args:
            trial: Optuna trial

        Returns:
            DBCV score (higher is better)
            Raises TrialPruned if constraints violated
        """
        # Get grid parameters
        n_neighbors = trial.suggest_categorical('n_neighbors', self._search_space['n_neighbors'])
        n_components = trial.suggest_categorical('n_components', self._search_space['n_components'])
        min_dist = trial.suggest_categorical('min_dist', self._search_space['min_dist'])
        min_cluster_size = trial.suggest_categorical('min_cluster_size', self._search_space['min_cluster_size'])
        min_samples = max(1, min_cluster_size // 2)

        # Look up pre-computed UMAP reduction
        reduced_normalized = self._umap_cache[(n_neighbors, n_components, min_dist)]

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method=self.config.hdbscan_cluster_selection_method,
            gen_min_span_tree=self.config.hdbscan_gen_min_span_tree,
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

        # Calculate coherence (on original embeddings)
        coherence = self._calculate_coherence(labels, self._original_embeddings)

        # Get DBCV (relative_validity) - this is our primary optimization metric
        try:
            relative_validity = clusterer.relative_validity_
        except AttributeError:
            relative_validity = self._compute_dbcv(labels, reduced_normalized)

        # Extract persistence metrics
        persistence_metrics = self._selector.extract_persistence_metrics(clusterer, labels)

        # Extract probability metrics
        prob_metrics = self._compute_probability_metrics(clusterer.probabilities_, labels)

        # Extract outlier metrics
        outlier_metrics = self._compute_outlier_metrics(clusterer.outlier_scores_)

        # Compute max_cluster_ratio: largest cluster as fraction of all items
        non_noise_count = (labels >= 0).sum()
        if non_noise_count > 0:
            cluster_counts = np.bincount(labels[labels >= 0].astype(int))
            max_cluster_ratio = float(np.max(cluster_counts)) / len(labels)
        else:
            max_cluster_ratio = 1.0

        # Log user attributes for analysis and visualization
        trial.set_user_attr('n_clusters', n_clusters)
        trial.set_user_attr('noise_rate', noise_rate)
        trial.set_user_attr('coherence', coherence)
        trial.set_user_attr('min_samples', min_samples)
        trial.set_user_attr('relative_validity', relative_validity)
        trial.set_user_attr('max_cluster_ratio', max_cluster_ratio)
        trial.set_user_attr('mean_persistence', persistence_metrics.get('mean_persistence', np.nan))
        trial.set_user_attr('weighted_persistence', persistence_metrics.get('weighted_persistence', np.nan))
        trial.set_user_attr('mean_probability', prob_metrics['mean_probability'])
        trial.set_user_attr('low_prob_ratio', prob_metrics['low_prob_ratio'])
        trial.set_user_attr('fuzzy_cluster_ratio', prob_metrics['fuzzy_cluster_ratio'])
        trial.set_user_attr('n_fuzzy_clusters', prob_metrics['n_fuzzy_clusters'])
        trial.set_user_attr('mean_outlier_score', outlier_metrics['mean_outlier_score'])
        trial.set_user_attr('high_outlier_ratio', outlier_metrics['high_outlier_ratio'])

        return relative_validity

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
        """Calculate mean intra-cluster cosine similarity. Delegates to standalone function."""
        return calculate_coherence_score(labels, embeddings)

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

        # Compute per-cluster fuzzy ratio
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
        """Extract metrics from HDBSCAN outlier_scores_ (GLOSH)."""
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
        """Compute soft threshold composite score."""
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

    def optimize(self) -> OptunaResult:
        """Run Optuna grid search optimization."""
        if self._verbose:
            print(f"\n[Optuna] Starting HDBSCAN optimization (N={self._N})")

        self._search_space = create_search_space(self._N, self.config)
        n_trials = (
            len(self._search_space['n_neighbors']) *
            len(self._search_space['n_components']) *
            len(self._search_space['min_dist']) *
            len(self._search_space['min_cluster_size'])
        )

        if self._verbose:
            self._print_search_space_table(n_trials)

        if self.config.precompute_umap:
            self._umap_cache = self.precompute_umap_reductions(
                self._search_space['n_neighbors'],
                self._search_space['n_components'],
                self._search_space['min_dist']
            )

        sampler = GridSampler(self._search_space)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study = optuna.create_study(
            study_name=f"clusterer_v3_{id(self)}",
            direction='maximize',
            sampler=sampler,
        )

        self._progress_best_score = 0.0
        self._progress_best_k = 0

        pbar = tqdm(total=n_trials, desc="Optimizing", disable=not self._verbose)

        def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                best = study.best_trial
                best_score = best.value
                best_k = best.user_attrs.get('n_clusters', 0)
                pbar.set_postfix({'best': f'{best_score:.3f}', 'k': best_k})
            pbar.update(1)

        self._study.optimize(self._objective, n_trials=None, callbacks=[progress_callback])
        pbar.close()

        if self._verbose:
            self._print_results_table()

        # --- Pareto frontier selection ---
        candidates = filter_candidates_by_hard_constraints(
            self._study, self._N, self.config, verbose=self._verbose
        )

        # --- Extended search if no candidates pass hard constraints ---
        if candidates and self._verbose:
            print(f"\n[Extended Search] Skipped — {len(candidates)} candidates pass hard constraints")
        elif not candidates and self.config.enable_research:
            if self._verbose:
                k_min = compute_pareto_k_min(self._N, self.config)
                k_max = compute_pareto_k_max(self._N, self.config)
                print(f"\n[Extended Search] Triggered — no candidates pass hard constraints "
                      f"(DBCV>{self.config.pareto_min_dbcv}, k in [{k_min}, {k_max}])")

            best_trial = self._study.best_trial
            initial_result = OptunaResult(
                best_params=best_trial.params,
                best_value=best_trial.value,
                best_labels=np.array([]),  # placeholder
                best_model=None,
                n_trials_completed=len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                n_trials_pruned=len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                study=self._study,
                umap_embeddings=np.array([]),
                search_space=self._search_space,
                persistence_metrics={},
            )
            self._run_extended_search(initial_result)

            # Re-filter with extended search trials included
            candidates = filter_candidates_by_hard_constraints(
                self._extended_study, self._N, self.config, verbose=self._verbose
            )
        elif not candidates:
            if self._verbose:
                print(f"\n[Extended Search] Disabled — using all completed trials as fallback")

        if not candidates:
            # Final fallback: use all completed trials from both studies
            candidates = [t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if hasattr(self, '_extended_study') and self._extended_study:
                candidates += [t for t in self._extended_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if self._verbose:
                print(f"  Fallback: using all {len(candidates)} completed trials")

        if not candidates:
            raise RuntimeError("No completed trials in any study")

        pareto_front = compute_pareto_front(candidates)
        if self._verbose:
            print(f"  Pareto front: {len(pareto_front)} non-dominated solutions")

        selected_trial = select_from_pareto_front(
            pareto_front, candidates, self.config, verbose=self._verbose
        )

        # Rebuild HDBSCAN model from selected trial's params
        sel_params = selected_trial.params
        min_cluster_size = sel_params['min_cluster_size']

        # Extended search trials have min_samples as a tuned param;
        # initial grid search trials have nn/nc/md and derive ms from mcs
        is_extended = 'min_samples' in sel_params and 'n_neighbors' not in sel_params
        if is_extended:
            best_trial_initial = self._study.best_trial
            n_neighbors = best_trial_initial.params['n_neighbors']
            n_components = best_trial_initial.params['n_components']
            min_dist = best_trial_initial.params['min_dist']
            min_samples = sel_params['min_samples']
        else:
            n_neighbors = sel_params['n_neighbors']
            n_components = sel_params['n_components']
            min_dist = sel_params['min_dist']
            min_samples = max(1, min_cluster_size // 2)

        # cluster_selection_method is always fixed from config (Layer 1 = EOM)
        sel_method = self.config.hdbscan_cluster_selection_method

        reduced_normalized = self._umap_cache[(n_neighbors, n_components, min_dist)]

        best_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method=sel_method,
            gen_min_span_tree=self.config.hdbscan_gen_min_span_tree,
        )
        best_labels = best_clusterer.fit_predict(reduced_normalized)
        persistence_metrics = self._selector.extract_persistence_metrics(best_clusterer, best_labels)

        completed = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])
        if hasattr(self, '_extended_study') and self._extended_study:
            completed += len([t for t in self._extended_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned += len([t for t in self._extended_study.trials if t.state == optuna.trial.TrialState.PRUNED])

        result = OptunaResult(
            best_params={
                'n_neighbors': n_neighbors,
                'n_components': n_components,
                'min_dist': min_dist,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
            },
            best_value=selected_trial.value,
            best_labels=best_labels,
            best_model=best_clusterer,
            n_trials_completed=completed,
            n_trials_pruned=pruned,
            study=self._study,
            umap_embeddings=reduced_normalized,
            search_space=self._search_space,
            persistence_metrics=persistence_metrics
        )

        # Generate Pareto visualization if enabled
        if self.config.enable_pareto_visualization:
            self.visualize_pareto_analysis(
                self._study, candidates, pareto_front, selected_trial
            )

        self._best_result = result
        return result

    def get_best_result(self) -> Optional[OptunaResult]:
        """Get the best result from optimization (None if not run yet)."""
        return self._best_result

    def _select_by_max_delta_log(
        self,
        study: optuna.Study,
        umap_cache_key: Tuple[int, int, float]
    ) -> Tuple[optuna.trial.FrozenTrial, Optional[Dict[str, Any]]]:
        """
        Select trial using max Δlog(score)/Δk criterion (parsimony selection).

        Returns the DESTINATION k of the step with maximum proportional gain.
        This prefers solutions where adding clusters gives the best "bang for buck".

        Args:
            study: Optuna study with completed trials
            umap_cache_key: (n_neighbors, n_components, min_dist) for UMAP lookup

        Returns:
            Tuple of (selected_trial, delta_log_analysis)
            delta_log_analysis contains the full Δlog table for verbose output
        """
        # 1. Get completed trials above minimum score threshold
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value > self.config.parsimony_min_score
        ]

        if len(completed) < 2:
            if self._verbose:
                print(f"\n[Parsimony Selection] Skipped: only {len(completed)} valid trials")
            return study.best_trial, None

        # 2. Group by k (n_clusters), take best score per k
        k_to_best: Dict[int, optuna.trial.FrozenTrial] = {}
        for trial in completed:
            k = trial.user_attrs.get('n_clusters', 0)
            if k <= 0:
                continue
            if k not in k_to_best or trial.value > k_to_best[k].value:
                k_to_best[k] = trial

        # 3. Sort by k ascending
        sorted_k = sorted(k_to_best.keys())

        if len(sorted_k) < 2:
            if self._verbose:
                print(f"\n[Parsimony Selection] Skipped: only {len(sorted_k)} unique k values")
            return study.best_trial, None

        # 4-6. Compute Δlog/Δk for each consecutive pair
        delta_log_rows = []
        for i in range(len(sorted_k)):
            k_curr = sorted_k[i]
            score_curr = k_to_best[k_curr].value
            log_score = np.log(score_curr) if score_curr > 0 else float('-inf')

            if i == 0:
                # First row: no Δlog
                delta_log_rows.append({
                    'k': k_curr,
                    'score': score_curr,
                    'log_score': log_score,
                    'delta_log': None,
                    'delta_k': None,
                    'delta_log_per_k': None,
                    'trial': k_to_best[k_curr]
                })
            else:
                k_prev = sorted_k[i - 1]
                score_prev = k_to_best[k_prev].value
                log_score_prev = np.log(score_prev) if score_prev > 0 else float('-inf')

                delta_log = log_score - log_score_prev
                delta_k = k_curr - k_prev
                delta_log_per_k = delta_log / delta_k if delta_k > 0 else 0.0

                delta_log_rows.append({
                    'k': k_curr,
                    'score': score_curr,
                    'log_score': log_score,
                    'delta_log': delta_log,
                    'delta_k': delta_k,
                    'delta_log_per_k': delta_log_per_k,
                    'trial': k_to_best[k_curr]
                })

        # 7. Find max Δlog/Δk (skip first row which has no Δlog)
        valid_rows = [r for r in delta_log_rows if r['delta_log_per_k'] is not None]
        if not valid_rows:
            if self._verbose:
                print(f"\n[Parsimony Selection] Skipped: no valid Δlog computations")
            return study.best_trial, None

        best_row = max(valid_rows, key=lambda r: r['delta_log_per_k'])

        # 8. Select the DESTINATION k of the max jump
        selected_trial = best_row['trial']

        # Verbose output
        if self._verbose:
            print(f"\n[Parsimony Selection] Δlog/Δk Analysis:")
            print(f"  {'k':>5} {'score':>8} {'log(s)':>8} {'Δlog':>8} {'Δk':>4} {'Δlog/Δk':>9}")
            print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*4} {'-'*9}")

            for row in delta_log_rows:
                k = row['k']
                score = row['score']
                log_s = row['log_score']
                dl = row['delta_log']
                dk = row['delta_k']
                dlpk = row['delta_log_per_k']

                marker = " ← MAX" if row == best_row else ""
                if dl is None:
                    print(f"  {k:>5} {score:>8.4f} {log_s:>8.3f} {'—':>8} {'—':>4} {'—':>9}{marker}")
                else:
                    print(f"  {k:>5} {score:>8.4f} {log_s:>8.3f} {dl:>8.4f} {dk:>4} {dlpk:>9.5f}{marker}")

            old_best = study.best_trial
            old_k = old_best.user_attrs.get('n_clusters', 0)
            new_k = selected_trial.user_attrs.get('n_clusters', 0)

            print(f"\n  Raw max-score selection: k={old_k} (score={old_best.value:.4f})")
            print(f"  Parsimony selection:     k={new_k} (score={selected_trial.value:.4f})")

            if new_k != old_k:
                print(f"  → Changed from k={old_k} to k={new_k}")
            else:
                print(f"  → No change (same k selected)")

        return selected_trial, {'rows': delta_log_rows, 'best_row': best_row}

    def _select_by_coherence_knee(
        self,
        study: optuna.Study,
    ) -> Tuple[optuna.trial.FrozenTrial, Optional[Dict[str, Any]]]:
        """
        Select trial using Kneedle on smoothed coherence curve.

        Since coherence increases as k increases (more clusters = smaller, tighter groups),
        we find the "elbow" where coherence gains level off.
        The optimal k is at the knee - the point of diminishing returns.

        Args:
            study: Optuna study with completed trials

        Returns:
            Tuple of (selected_trial, analysis_dict)
        """
        from scipy.ndimage import uniform_filter1d

        # 1. Get completed trials
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed) < 3:
            if self._verbose:
                print(f"\n[Coherence Knee] Skipped: only {len(completed)} trials (need >= 3)")
            return study.best_trial, None

        # 2. Group by k, take best coherence per k (trial.value = coherence)
        k_to_best: Dict[int, optuna.trial.FrozenTrial] = {}
        for trial in completed:
            k = trial.user_attrs.get('n_clusters', 0)
            if k <= 0:
                continue
            if k not in k_to_best or trial.value > k_to_best[k].value:
                k_to_best[k] = trial

        sorted_k = sorted(k_to_best.keys())
        if len(sorted_k) < 3:
            if self._verbose:
                print(f"\n[Coherence Knee] Skipped: only {len(sorted_k)} unique k values (need >= 3)")
            return study.best_trial, None

        # 3. Extract coherence values
        k_values = np.array(sorted_k)
        coherence_values = np.array([k_to_best[k].user_attrs.get('coherence', 0) for k in sorted_k])

        # 4. Smooth the curve using moving average
        # Window size: ~20% of data points, minimum 3
        window_size = max(3, len(coherence_values) // self.config.coherence_knee_window_divisor)
        if window_size % 2 == 0:
            window_size += 1  # Make odd for symmetric smoothing

        # Pad edges to avoid boundary effects
        coherence_smoothed = uniform_filter1d(coherence_values, size=window_size, mode='nearest')

        # 5. Apply Kneedle to find the knee
        # Coherence INCREASES with k (more clusters = smaller, more coherent groups)
        # We want the "elbow" where gains level off: curve="concave", direction="increasing"
        try:
            kneedle = KneeLocator(
                k_values,
                coherence_smoothed,
                curve="concave",
                direction="increasing",
                interp_method="polynomial",
                polynomial_degree=self.config.coherence_knee_polynomial_degree
            )
            knee_k = kneedle.knee
            knee_y = kneedle.knee_y
        except Exception as e:
            if self._verbose:
                print(f"\n[Coherence Knee] Kneedle failed: {e}")
            knee_k = None
            knee_y = None

        # 6. Select the trial at the knee (or fallback to max coherence)
        if knee_k is not None and knee_k in k_to_best:
            selected_trial = k_to_best[knee_k]
        else:
            # Fallback: select k with highest coherence
            best_coh_idx = np.argmax(coherence_values)
            knee_k = k_values[best_coh_idx]
            knee_y = coherence_values[best_coh_idx]
            selected_trial = k_to_best[knee_k]
            if self._verbose:
                print(f"\n[Coherence Knee] No knee found, using max coherence at k={knee_k}")

        # 7. Build analysis dict
        analysis = {
            'k_values': k_values.tolist(),
            'coherence_raw': coherence_values.tolist(),
            'coherence_smoothed': coherence_smoothed.tolist(),
            'window_size': window_size,
            'knee_k': knee_k,
            'knee_coherence': knee_y,
        }

        # Verbose output
        if self._verbose:
            print(f"\n[Coherence Knee Selection]")
            print(f"  Smoothing window: {window_size} points")
            print(f"  K range: {k_values.min()} - {k_values.max()} ({len(k_values)} values)")
            print(f"  Coherence range: {coherence_values.min():.4f} - {coherence_values.max():.4f}")
            print(f"  Knee detected at: k={knee_k} (coherence={knee_y:.4f})")

            # Compare with max-score selection
            best_trial_k = study.best_trial.user_attrs.get('n_clusters', 0)
            max_score_coh = study.best_trial.user_attrs.get('coherence', 0)
            print(f"  Max-score selection: k={best_trial_k} (coherence={max_score_coh:.4f})")

            if knee_k != best_trial_k:
                print(f"  → Changed from k={best_trial_k} to k={knee_k}")

        return selected_trial, analysis

    def _print_search_space_table(self, n_trials: int) -> None:
        """Print compact search space configuration."""
        print(f"  n_neighbors:      {self._search_space['n_neighbors']}")
        print(f"  n_components:     {self._search_space['n_components']}")
        print(f"  min_dist:         {self._search_space['min_dist']}")
        print(f"  min_cluster_size: {self._search_space['min_cluster_size']}")
        print(f"  min_samples:      mcs // 2")
        print(f"  Total trials:     {n_trials}")

    def _print_results_table(self, top_n: int = 5) -> None:
        """Print top N results as a formatted table."""
        completed = [t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_count = len([t for t in self._study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if not completed:
            print("  No completed trials")
            return

        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]

        print(f"\n  Top {min(top_n, len(sorted_trials))} Results:")
        print(f"  {'nn':>4} {'nc':>4} {'md':>5} {'mcs':>4} {'ms':>4} {'k':>5} {'noise':>6} {'score':>7}")
        print(f"  {'-'*4} {'-'*4} {'-'*5} {'-'*4} {'-'*4} {'-'*5} {'-'*6} {'-'*7}")

        for i, trial in enumerate(sorted_trials):
            marker = "*" if i == 0 else " "
            noise_rate = trial.user_attrs.get('noise_rate', 0)
            ms_val = trial.params.get('min_samples', trial.user_attrs.get('min_samples', '?'))
            print(f"{marker} {trial.params.get('n_neighbors', '?'):>4} "
                  f"{trial.params.get('n_components', '?'):>4} "
                  f"{trial.params.get('min_dist', '?'):>5} "
                  f"{trial.params.get('min_cluster_size', '?'):>4} "
                  f"{ms_val:>4} "
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
        """Check quality of optimization result and trigger re-search if needed."""
        if not self.config.enable_research:
            return result

        best_trial = self._study.best_trial
        n_clusters = best_trial.user_attrs.get('n_clusters', 0)
        noise_rate = best_trial.user_attrs.get('noise_rate', 0.0)
        relative_validity = best_trial.user_attrs.get('relative_validity', result.best_value)

        k_max = compute_pareto_k_max(self._N, self.config)
        max_noise = self.config.research_max_noise_rate
        min_validity = self.config.research_min_validity
        cluster_deviation_threshold = self.config.research_cluster_deviation_threshold

        cluster_deviation = abs(n_clusters - k_max) / k_max if k_max > 0 else 0.0

        needs_research = False
        reasons = []

        if noise_rate > max_noise and relative_validity < min_validity:
            needs_research = True
            reasons.append(f"noise={noise_rate:.1%}>{max_noise:.0%} AND rel_validity={relative_validity:.3f}<{min_validity}")

        if cluster_deviation > cluster_deviation_threshold:
            needs_research = True
            reasons.append(f"cluster_deviation={cluster_deviation:.1%}>{cluster_deviation_threshold:.0%} (k={n_clusters}, k_max={k_max})")

        if not needs_research:
            if self._verbose:
                print(f"\n[Extended Search] Skipped — quality check passed "
                      f"(k={n_clusters}, k_max={k_max}, deviation={cluster_deviation:.1%}<={cluster_deviation_threshold:.0%})")
            return result

        if self._verbose:
            print(f"\n[Extended Search] Triggered — {', '.join(reasons)}")

        return self._run_extended_search(result)

    def _run_extended_search(self, initial_result: OptunaResult) -> OptunaResult:
        """Run extended search with expanded parameters using Optuna GridSampler."""
        best_n_neighbors = initial_result.best_params['n_neighbors']
        best_n_components = initial_result.best_params.get('n_components', self.config.umap_n_components_grid[0])
        best_min_dist = initial_result.best_params.get('min_dist', self.config.umap_min_dist_grid[0])
        best_mcs = initial_result.best_params['min_cluster_size']
        reduced_normalized = self._umap_cache[(best_n_neighbors, best_n_components, best_min_dist)]

        mcs_multipliers = self.config.research_mcs_multipliers
        mcs_options = sorted(set(
            max(self.config.mcs_min, int(round(best_mcs * mult)))
            for mult in mcs_multipliers
        ))

        # MS grid for extended search: lower=ln(N), upper=max_mcs/2
        max_mcs = max(mcs_options)
        ms_low = max(1, int(round(math.log(max(self._N, 2)))))
        ms_high = max(ms_low, max_mcs // 2)
        ms_options = log_spaced_ints(ms_low, ms_high, k=self.config.min_samples_grid_k)

        # NOTE: cluster_selection_method is fixed to 'eom' for Layer 1.
        # Layer 2 (analyze_leaf_overlay.py) can experiment with 'leaf' method.
        extended_search_space = {
            'min_cluster_size': mcs_options,
            'min_samples': ms_options,
        }

        n_trials_total = len(mcs_options) * len(ms_options)

        if self._verbose:
            print(f"\n[Extended Search] Based on best: nn={best_n_neighbors}, mcs={best_mcs}")
            print(f"  MCS grid:      {mcs_options}")
            print(f"  MS grid:       {ms_options}")
            print(f"  Method:        {self.config.hdbscan_cluster_selection_method} (fixed)")
            print(f"  Total trials:  {n_trials_total}")

        def extended_objective(trial: optuna.Trial) -> float:
            mcs = trial.suggest_categorical('min_cluster_size', extended_search_space['min_cluster_size'])
            ms = trial.suggest_categorical('min_samples', extended_search_space['min_samples'])

            if ms > mcs:
                raise optuna.TrialPruned(f"Invalid: ms={ms} > mcs={mcs}")

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric='euclidean',
                cluster_selection_method=self.config.hdbscan_cluster_selection_method,
                gen_min_span_tree=self.config.hdbscan_gen_min_span_tree,
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

            # Calculate coherence
            coherence = self._calculate_coherence(labels, self._original_embeddings)

            # Compute other metrics for logging
            persistence_metrics = self._selector.extract_persistence_metrics(clusterer, labels)
            prob_metrics = self._compute_probability_metrics(clusterer.probabilities_, labels)
            outlier_metrics = self._compute_outlier_metrics(clusterer.outlier_scores_)

            # Compute max_cluster_ratio: largest cluster as fraction of all items
            non_noise_count = (labels >= 0).sum()
            if non_noise_count > 0:
                cluster_counts = np.bincount(labels[labels >= 0].astype(int))
                max_cluster_ratio = float(np.max(cluster_counts)) / len(labels)
            else:
                max_cluster_ratio = 1.0

            trial.set_user_attr('n_clusters', n_clusters)
            trial.set_user_attr('noise_rate', noise_rate)
            trial.set_user_attr('coherence', coherence)
            trial.set_user_attr('labels', labels.tolist())
            trial.set_user_attr('relative_validity', validity)
            trial.set_user_attr('max_cluster_ratio', max_cluster_ratio)
            trial.set_user_attr('weighted_persistence', persistence_metrics.get('weighted_persistence', 0.0))
            trial.set_user_attr('mean_probability', prob_metrics['mean_probability'])
            trial.set_user_attr('low_prob_ratio', prob_metrics['low_prob_ratio'])
            trial.set_user_attr('fuzzy_cluster_ratio', prob_metrics['fuzzy_cluster_ratio'])
            trial.set_user_attr('n_fuzzy_clusters', prob_metrics['n_fuzzy_clusters'])
            trial.set_user_attr('mean_outlier_score', outlier_metrics['mean_outlier_score'])
            trial.set_user_attr('high_outlier_ratio', outlier_metrics['high_outlier_ratio'])

            return validity

        extended_sampler = GridSampler(extended_search_space)
        extended_study = optuna.create_study(
            study_name=f"clusterer_v3_extended_{id(self)}",
            direction='maximize',
            sampler=extended_sampler,
        )

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

        completed = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in extended_study.trials if t.state == optuna.trial.TrialState.PRUNED])

        if self._verbose:
            print(f"  {completed} completed, {pruned} pruned")

        # Store extended study for Pareto selection in optimize()
        self._extended_study = extended_study

    def visualize_pareto_analysis(
        self,
        study: optuna.Study,
        candidates: List[optuna.trial.FrozenTrial],
        pareto_front: List[optuna.trial.FrozenTrial],
        selected_trial: optuna.trial.FrozenTrial,
        output_dir: Optional[Path] = None,
        filename_prefix: str = "pareto_analysis"
    ) -> Optional[Path]:
        """
        Visualize Pareto frontier selection.

        2x2 panel:
        1. DBCV vs k scatter (Pareto objectives, candidates, front, selected)
        2. Low prob vs Max cluster scatter (Pareto objectives)
        3. Pareto front table with all metrics
        4. Summary of selected solution
        """
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = output_dir or (project_root / "exports")
            output_dir.mkdir(parents=True, exist_ok=True)

            def extract_metrics(trial):
                return {
                    'dbcv': trial.user_attrs.get('relative_validity', 0.0),
                    'coherence': trial.user_attrs.get('coherence', trial.value),
                    'k': trial.user_attrs.get('n_clusters', 0),
                    'noise': trial.user_attrs.get('noise_rate', 0.0),
                    'low_prob': trial.user_attrs.get('low_prob_ratio', 0.0),
                    'max_clust': trial.user_attrs.get('max_cluster_ratio', 0.0),
                }

            # Get all completed trials for background
            all_completed = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            all_metrics = [extract_metrics(t) for t in all_completed]
            cand_metrics = [extract_metrics(t) for t in candidates]
            pareto_metrics = [extract_metrics(t) for t in pareto_front]
            sel_metrics = extract_metrics(selected_trial)

            pareto_trial_nums = {t.number for t in pareto_front}

            fig, axes = plt.subplots(2, 2, figsize=(14, 11))

            # Panel 1: DBCV vs k (both Pareto objectives)
            ax = axes[0, 0]
            ax.scatter(
                [m['dbcv'] for m in all_metrics],
                [m['k'] for m in all_metrics],
                c='lightgrey', s=30, alpha=0.5, label=f'All ({len(all_metrics)})'
            )
            ax.scatter(
                [m['dbcv'] for m in cand_metrics],
                [m['k'] for m in cand_metrics],
                c='steelblue', s=40, alpha=0.6, label=f'Candidates ({len(cand_metrics)})'
            )
            ax.scatter(
                [m['dbcv'] for m in pareto_metrics],
                [m['k'] for m in pareto_metrics],
                c='darkorange', s=70, edgecolors='black', linewidths=0.8,
                label=f'Pareto front ({len(pareto_metrics)})', zorder=5
            )
            ax.scatter(
                [sel_metrics['dbcv']], [sel_metrics['k']],
                c='red', s=200, marker='*', edgecolors='black', linewidths=1,
                label='Selected', zorder=10
            )
            ax.set_xlabel('DBCV (relative validity)')
            ax.set_ylabel('Number of clusters (k)')
            ax.set_title('DBCV vs k')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

            # Panel 2: Low prob vs Max cluster (both Pareto objectives)
            ax = axes[0, 1]
            ax.scatter(
                [m['low_prob'] for m in all_metrics],
                [m['max_clust'] for m in all_metrics],
                c='lightgrey', s=30, alpha=0.5, label=f'All ({len(all_metrics)})'
            )
            ax.scatter(
                [m['low_prob'] for m in cand_metrics],
                [m['max_clust'] for m in cand_metrics],
                c='steelblue', s=40, alpha=0.6, label=f'Candidates ({len(cand_metrics)})'
            )
            ax.scatter(
                [m['low_prob'] for m in pareto_metrics],
                [m['max_clust'] for m in pareto_metrics],
                c='darkorange', s=70, edgecolors='black', linewidths=0.8,
                label=f'Pareto front ({len(pareto_metrics)})', zorder=5
            )
            ax.scatter(
                [sel_metrics['low_prob']], [sel_metrics['max_clust']],
                c='red', s=200, marker='*', edgecolors='black', linewidths=1,
                label='Selected', zorder=10
            )
            ax.set_xlabel('Low probability ratio')
            ax.set_ylabel('Max cluster ratio')
            ax.set_title('Low Prob vs Max Cluster')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

            # Panel 3: Pareto front table
            ax = axes[1, 0]
            ax.axis('off')
            if pareto_metrics:
                # Sort by DBCV descending for readability
                sorted_pareto = sorted(
                    zip(pareto_front, pareto_metrics),
                    key=lambda x: x[1]['dbcv'],
                    reverse=True
                )
                col_labels = ['Trial', 'DBCV', 'k', 'LowP', 'MaxCl', 'Coh', 'Noise']
                n_cols = len(col_labels)
                table_data = []
                cell_colors = []
                for trial, m in sorted_pareto:
                    is_selected = (trial.number == selected_trial.number)
                    row = [
                        f"#{trial.number}{'*' if is_selected else ''}",
                        f"{m['dbcv']:.4f}",
                        f"{m['k']}",
                        f"{m['low_prob']:.3f}",
                        f"{m['max_clust']:.3f}",
                        f"{m['coherence']:.4f}",
                        f"{m['noise']:.3f}",
                    ]
                    table_data.append(row)
                    if is_selected:
                        cell_colors.append(['#ffcccc'] * n_cols)
                    else:
                        cell_colors.append(['white'] * n_cols)

                table = ax.table(
                    cellText=table_data,
                    colLabels=col_labels,
                    cellColours=cell_colors,
                    colColours=['#e6e6e6'] * n_cols,
                    loc='center',
                    cellLoc='center',
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.0, 1.3)
                ax.set_title('Pareto Front Solutions (* = selected)', fontsize=11)

            # Panel 4: Summary
            ax = axes[1, 1]
            ax.axis('off')
            summary_lines = [
                f"Dataset: N = {self._N}",
                f"Grid trials: {len(all_completed)} completed, "
                f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned",
                f"Hard constraint filter: {len(candidates)} candidates",
                f"Pareto front: {len(pareto_front)} non-dominated",
                "",
                "Selected (Pareto objectives):",
                f"  DBCV = {sel_metrics['dbcv']:.4f}",
                f"  k = {sel_metrics['k']}",
                f"  Low prob = {sel_metrics['low_prob']:.3f}",
                f"  Max cluster = {sel_metrics['max_clust']:.3f}",
                "Context (hard constraints):",
                f"  Coherence = {sel_metrics['coherence']:.4f}",
                f"  Noise = {sel_metrics['noise']:.3f}",
                "",
                f"  nn={selected_trial.params.get('n_neighbors')}, "
                f"nc={selected_trial.params.get('n_components')}, "
                f"md={selected_trial.params.get('min_dist')}, "
                f"mcs={selected_trial.params.get('min_cluster_size')}",
            ]
            ax.text(
                0.05, 0.95, '\n'.join(summary_lines),
                transform=ax.transAxes,
                verticalalignment='top',
                fontfamily='monospace',
                fontsize=10,
            )
            ax.set_title('Selection Summary', fontsize=11)

            fig.suptitle(
                f'Pareto Frontier Analysis (N={self._N})',
                fontsize=14, fontweight='bold', y=0.98
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"{filename_prefix}_{timestamp}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)

            if self._verbose:
                print(f"\n  Pareto visualization saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"  WARNING: Pareto visualization failed: {e}")
            return None

    def visualize_elbow_analysis(
        self,
        study: Optional[optuna.Study] = None,
        output_dir: Optional[Path] = None,
        filename_prefix: str = "elbow_analysis"
    ) -> Optional[Path]:
        """
        Visualize elbow analysis comparing Coherence vs Mean Probability.

        Creates a 3x2 panel plot showing:
        - Row 1: Coherence vs K, Mean Probability vs K
        - Row 2: Δlog/Δk for Composite, Δlog/Δk for Mean Probability
        - Row 3: Relative Validity vs K, Summary

        Args:
            study: Optuna study (uses self._study if None)
            output_dir: Directory for output (defaults to exports/)
            filename_prefix: Prefix for output filename

        Returns:
            Path to saved PNG file, or None if visualization failed
        """
        study = study or self._study
        if study is None:
            print("[Elbow Visualization] No study available")
            return None

        # Set output directory
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "exports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Get completed trials
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value > self.config.parsimony_min_score
        ]

        if len(completed) < 2:
            print(f"[Elbow Visualization] Not enough trials: {len(completed)}")
            return None

        # 2. Group by k (n_clusters), take best coherence per k (trial.value = coherence)
        k_to_best: Dict[int, optuna.trial.FrozenTrial] = {}
        for trial in completed:
            k = trial.user_attrs.get('n_clusters', 0)
            if k <= 0:
                continue
            if k not in k_to_best or trial.value > k_to_best[k].value:
                k_to_best[k] = trial

        sorted_k = sorted(k_to_best.keys())
        if len(sorted_k) < 2:
            print(f"[Elbow Visualization] Not enough unique k values: {len(sorted_k)}")
            return None

        # 3. Build data arrays for all metrics
        k_values = []
        coherence_scores = []  # trial.value = coherence
        mean_probs = []
        relative_validity = []

        for k in sorted_k:
            trial = k_to_best[k]
            k_values.append(k)
            coherence_scores.append(trial.value)  # Now coherence, not composite
            mean_probs.append(trial.user_attrs.get('mean_probability', 0))
            relative_validity.append(trial.user_attrs.get('relative_validity', 0))

        # 4. Compute Δlog/Δk for both metrics
        def compute_delta_log(values):
            delta_log_per_k = [0]  # First value has no delta
            for i in range(1, len(values)):
                val = values[i]
                val_prev = values[i - 1]
                if val > 0 and val_prev > 0:
                    delta_log = np.log(val) - np.log(val_prev)
                    delta_k = k_values[i] - k_values[i - 1]
                    delta_log_per_k.append(delta_log / delta_k if delta_k > 0 else 0)
                else:
                    delta_log_per_k.append(0)
            return delta_log_per_k

        coherence_delta = compute_delta_log(coherence_scores)
        prob_delta = compute_delta_log(mean_probs)

        # 5. Find elbow for each metric
        valid_indices = list(range(1, len(k_values)))

        # Composite score elbow
        coherence_elbow_idx = max(valid_indices, key=lambda i: coherence_delta[i])
        coherence_elbow_k = k_values[coherence_elbow_idx]
        coherence_best_idx = max(range(len(coherence_scores)), key=lambda i: coherence_scores[i])
        coherence_best_k = k_values[coherence_best_idx]

        # Mean probability elbow
        prob_elbow_idx = max(valid_indices, key=lambda i: prob_delta[i])
        prob_elbow_k = k_values[prob_elbow_idx]
        prob_best_idx = max(range(len(mean_probs)), key=lambda i: mean_probs[i])
        prob_best_k = k_values[prob_best_idx]

        # Relative validity best
        validity_best_idx = max(range(len(relative_validity)), key=lambda i: relative_validity[i])
        validity_best_k = k_values[validity_best_idx]

        # 6. Create the visualization (3x2 grid)
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'Elbow Analysis: Coherence vs Mean Probability\n(N={self._N} ideas, {len(sorted_k)} unique k values)',
                     fontsize=14, fontweight='bold')

        # Color scheme
        color_coherence = '#2E86AB'  # Blue
        color_prob = '#A23B72'       # Purple
        color_validity = '#F18F01'   # Orange
        color_elbow = '#E94F37'      # Red
        color_best = '#2ECC71'       # Green

        # Row 1, Left: Coherence vs K
        ax1 = axes[0, 0]
        ax1.plot(k_values, coherence_scores, 'o-', color=color_coherence, linewidth=2, markersize=8)
        ax1.axvline(x=coherence_elbow_k, color=color_elbow, linestyle='--', linewidth=2, alpha=0.7, label=f'Elbow k={coherence_elbow_k}')
        ax1.axvline(x=coherence_best_k, color=color_best, linestyle=':', linewidth=2, alpha=0.7, label=f'Best k={coherence_best_k}')
        ax1.scatter([coherence_elbow_k], [coherence_scores[coherence_elbow_idx]], color=color_elbow, s=150, zorder=5, edgecolors='black', linewidth=2)
        ax1.scatter([coherence_best_k], [coherence_scores[coherence_best_idx]], color=color_best, s=150, zorder=5, edgecolors='black', linewidth=2, marker='*')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax1.set_ylabel('Coherence', fontsize=11)
        ax1.set_title('Coherence vs K', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Row 1, Right: Mean Probability vs K
        ax2 = axes[0, 1]
        ax2.plot(k_values, mean_probs, 'o-', color=color_prob, linewidth=2, markersize=8)
        ax2.axvline(x=prob_elbow_k, color=color_elbow, linestyle='--', linewidth=2, alpha=0.7, label=f'Elbow k={prob_elbow_k}')
        ax2.axvline(x=prob_best_k, color=color_best, linestyle=':', linewidth=2, alpha=0.7, label=f'Best k={prob_best_k}')
        ax2.scatter([prob_elbow_k], [mean_probs[prob_elbow_idx]], color=color_elbow, s=150, zorder=5, edgecolors='black', linewidth=2)
        ax2.scatter([prob_best_k], [mean_probs[prob_best_idx]], color=color_best, s=150, zorder=5, edgecolors='black', linewidth=2, marker='*')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax2.set_ylabel('Mean Probability', fontsize=11)
        ax2.set_title('Mean Cluster Probability vs K', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Row 2, Left: Δlog/Δk for Coherence
        ax3 = axes[1, 0]
        bar_colors = [color_elbow if i == coherence_elbow_idx else color_coherence for i in range(len(k_values))]
        ax3.bar(k_values, coherence_delta, color=bar_colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax3.set_ylabel('Δlog(score) / Δk', fontsize=11)
        ax3.set_title(f'Δlog/Δk for Coherence (elbow at k={coherence_elbow_k})', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Row 2, Right: Δlog/Δk for Mean Probability
        ax4 = axes[1, 1]
        bar_colors = [color_elbow if i == prob_elbow_idx else color_prob for i in range(len(k_values))]
        ax4.bar(k_values, prob_delta, color=bar_colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax4.set_ylabel('Δlog(prob) / Δk', fontsize=11)
        ax4.set_title(f'Δlog/Δk for Mean Probability (elbow at k={prob_elbow_k})', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Row 3, Left: Relative Validity vs K
        ax5 = axes[2, 0]
        ax5.plot(k_values, relative_validity, 'o-', color=color_validity, linewidth=2, markersize=8)
        ax5.axvline(x=validity_best_k, color=color_best, linestyle=':', linewidth=2, alpha=0.7, label=f'Best k={validity_best_k}')
        ax5.scatter([validity_best_k], [relative_validity[validity_best_idx]], color=color_best, s=150, zorder=5, edgecolors='black', linewidth=2, marker='*')
        ax5.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax5.set_ylabel('Relative Validity (DBCV)', fontsize=11)
        ax5.set_title('Relative Validity vs K', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)

        # Row 3, Right: Summary
        ax6 = axes[2, 1]
        ax6.axis('off')

        summary_lines = [
            "SELECTION SUMMARY",
            "=" * 50,
            "",
            "COMPOSITE SCORE:",
            f"  Best k={coherence_best_k} (score={coherence_scores[coherence_best_idx]:.4f})",
            f"  Elbow k={coherence_elbow_k} (Δlog/Δk={coherence_delta[coherence_elbow_idx]:.5f})",
            "",
            "MEAN PROBABILITY:",
            f"  Best k={prob_best_k} (prob={mean_probs[prob_best_idx]:.4f})",
            f"  Elbow k={prob_elbow_k} (Δlog/Δk={prob_delta[prob_elbow_idx]:.5f})",
            "",
            "RELATIVE VALIDITY:",
            f"  Best k={validity_best_k} (validity={relative_validity[validity_best_idx]:.4f})",
            "",
            "=" * 50,
            "",
            "K VALUES (sorted by coherence):",
        ]

        # Add top 10 k values
        sorted_by_score = sorted(range(len(k_values)), key=lambda i: coherence_scores[i], reverse=True)[:10]
        for rank, i in enumerate(sorted_by_score, 1):
            k = k_values[i]
            summary_lines.append(f"  {rank}. k={k:3d}: coh={coherence_scores[i]:.4f}, prob={mean_probs[i]:.4f}, val={relative_validity[i]:.4f}")

        summary_text = "\n".join(summary_lines)
        ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes,
                 fontsize=9, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 7. Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        output_path = output_dir / filename

        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        if self._verbose:
            print(f"\n[Elbow Visualization] Saved to: {output_path}")

        return output_path

    def visualize_metric_comparison(
        self,
        study: Optional[optuna.Study] = None,
        output_dir: Optional[Path] = None,
        filename_prefix: str = "metric_comparison"
    ) -> Optional[Path]:
        """
        Compare different metrics vs K to find which has best elbow shape.

        Creates a 2x3 panel plot showing:
        - relative_validity vs K
        - mean_probability vs K
        - coherence vs K
        - composite_score vs K
        - noise_rate vs K
        - low_prob_ratio vs K

        Args:
            study: Optuna study (uses self._study if None)
            output_dir: Directory for output (defaults to exports/)
            filename_prefix: Prefix for output filename

        Returns:
            Path to saved PNG file, or None if visualization failed
        """
        study = study or self._study
        if study is None:
            print("[Metric Comparison] No study available")
            return None

        # Set output directory
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "exports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get completed trials
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed) < 2:
            print(f"[Metric Comparison] Not enough trials: {len(completed)}")
            return None

        # Group by k, take best coherence per k (trial.value = coherence)
        k_to_best: Dict[int, optuna.trial.FrozenTrial] = {}
        for trial in completed:
            k = trial.user_attrs.get('n_clusters', 0)
            if k <= 0:
                continue
            if k not in k_to_best or trial.value > k_to_best[k].value:
                k_to_best[k] = trial

        sorted_k = sorted(k_to_best.keys())
        if len(sorted_k) < 2:
            print(f"[Metric Comparison] Not enough unique k values: {len(sorted_k)}")
            return None

        # Extract metrics for each k
        k_values = []
        relative_validity_values = []
        mean_probability_values = []
        coherence_values = []  # Primary metric (trial.value = coherence)
        persistence_values = []
        noise_rate_values = []
        low_prob_ratio_values = []

        for k in sorted_k:
            trial = k_to_best[k]
            k_values.append(k)
            relative_validity_values.append(trial.user_attrs.get('relative_validity', 0))
            mean_probability_values.append(trial.user_attrs.get('mean_probability', 0))
            coherence_values.append(trial.value)  # trial.value is now coherence
            persistence_values.append(trial.user_attrs.get('weighted_persistence', 0))
            noise_rate_values.append(trial.user_attrs.get('noise_rate', 0))
            low_prob_ratio_values.append(trial.user_attrs.get('low_prob_ratio', 0))

        # Find best k for each metric
        best_validity_idx = max(range(len(relative_validity_values)), key=lambda i: relative_validity_values[i])
        best_prob_idx = max(range(len(mean_probability_values)), key=lambda i: mean_probability_values[i])
        best_coherence_idx = max(range(len(coherence_values)), key=lambda i: coherence_values[i])
        best_persistence_idx = max(range(len(persistence_values)), key=lambda i: persistence_values[i])

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Metric Comparison vs K\n(N={self._N} ideas, {len(sorted_k)} unique k values)',
                     fontsize=14, fontweight='bold')

        colors = {
            'validity': '#2E86AB',
            'probability': '#A23B72',
            'coherence': '#F18F01',
            'persistence': '#C73E1D',
            'noise': '#6B717E',
            'low_prob': '#3B1F2B'
        }

        def plot_metric(ax, k_vals, metric_vals, title, ylabel, color, best_idx, higher_is_better=True):
            ax.plot(k_vals, metric_vals, 'o-', color=color, linewidth=2, markersize=8)
            best_k = k_vals[best_idx]
            best_val = metric_vals[best_idx]
            ax.scatter([best_k], [best_val], color='red', s=200, zorder=5,
                      edgecolors='black', linewidth=2, marker='*')
            ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            direction = "↑ higher=better" if higher_is_better else "↓ lower=better"
            ax.set_title(f'{title}\n(best k={best_k}, {direction})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add trend annotation
            if len(metric_vals) > 2:
                trend = "monotonic ↗" if all(metric_vals[i] <= metric_vals[i+1] for i in range(len(metric_vals)-1)) else \
                        "monotonic ↘" if all(metric_vals[i] >= metric_vals[i+1] for i in range(len(metric_vals)-1)) else \
                        "non-monotonic"
                ax.annotate(trend, xy=(0.02, 0.98), xycoords='axes fraction',
                           fontsize=9, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # Plot each metric
        plot_metric(axes[0, 0], k_values, relative_validity_values,
                   'Relative Validity (DBCV)', 'relative_validity', colors['validity'], best_validity_idx)

        plot_metric(axes[0, 1], k_values, mean_probability_values,
                   'Mean Cluster Probability', 'mean_probability', colors['probability'], best_prob_idx)

        plot_metric(axes[0, 2], k_values, coherence_values,
                   'Coherence (Intra-cluster Similarity)', 'coherence', colors['coherence'], best_coherence_idx)

        plot_metric(axes[1, 0], k_values, persistence_values,
                   'Weighted Persistence', 'weighted_persistence', colors['persistence'], best_persistence_idx)

        # For noise and low_prob, lower is better
        best_noise_idx = min(range(len(noise_rate_values)), key=lambda i: noise_rate_values[i])
        plot_metric(axes[1, 1], k_values, noise_rate_values,
                   'Noise Rate', 'noise_rate', colors['noise'], best_noise_idx, higher_is_better=False)

        best_lowprob_idx = min(range(len(low_prob_ratio_values)), key=lambda i: low_prob_ratio_values[i])
        plot_metric(axes[1, 2], k_values, low_prob_ratio_values,
                   'Low Probability Ratio', 'low_prob_ratio', colors['low_prob'], best_lowprob_idx, higher_is_better=False)

        plt.tight_layout()

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        output_path = output_dir / filename

        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        if self._verbose:
            print(f"\n[Metric Comparison] Saved to: {output_path}")
            print(f"  Best k by metric:")
            print(f"    relative_validity: k={k_values[best_validity_idx]} ({relative_validity_values[best_validity_idx]:.4f})")
            print(f"    mean_probability:  k={k_values[best_prob_idx]} ({mean_probability_values[best_prob_idx]:.4f})")
            print(f"    coherence:         k={k_values[best_coherence_idx]} ({coherence_values[best_coherence_idx]:.4f})")
            print(f"    persistence:       k={k_values[best_persistence_idx]} ({persistence_values[best_persistence_idx]:.4f})")

        return output_path

    def visualize_coherence_knee(
        self,
        study: Optional[optuna.Study] = None,
        output_dir: Optional[Path] = None,
        filename_prefix: str = "coherence_knee"
    ) -> Optional[Path]:
        """
        Visualize coherence-based knee detection with smoothing.

        Creates a 2x2 panel plot showing:
        - Top-left: Raw coherence vs K
        - Top-right: Smoothed coherence vs K with knee point
        - Bottom-left: First derivative of smoothed coherence
        - Bottom-right: Summary

        Args:
            study: Optuna study (uses self._study if None)
            output_dir: Directory for output (defaults to exports/)
            filename_prefix: Prefix for output filename

        Returns:
            Path to saved PNG file, or None if visualization failed
        """
        from scipy.ndimage import uniform_filter1d

        study = study or self._study
        if study is None:
            print("[Coherence Knee Viz] No study available")
            return None

        # Set output directory
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "exports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get completed trials
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed) < 3:
            print(f"[Coherence Knee Viz] Not enough trials: {len(completed)}")
            return None

        # Group by k, take best coherence per k (trial.value = coherence)
        k_to_best: Dict[int, optuna.trial.FrozenTrial] = {}
        for trial in completed:
            k = trial.user_attrs.get('n_clusters', 0)
            if k <= 0:
                continue
            if k not in k_to_best or trial.value > k_to_best[k].value:
                k_to_best[k] = trial

        sorted_k = sorted(k_to_best.keys())
        if len(sorted_k) < 3:
            print(f"[Coherence Knee Viz] Not enough unique k values: {len(sorted_k)}")
            return None

        # Extract data
        k_values = np.array(sorted_k)
        coherence_values = np.array([k_to_best[k].user_attrs.get('coherence', 0) for k in sorted_k])

        # Smooth the curve
        window_size = max(3, len(coherence_values) // self.config.coherence_knee_window_divisor)
        if window_size % 2 == 0:
            window_size += 1
        coherence_smoothed = uniform_filter1d(coherence_values, size=window_size, mode='nearest')

        # Compute first derivative (rate of change)
        coherence_derivative = np.gradient(coherence_smoothed, k_values)

        # Apply Kneedle
        try:
            kneedle = KneeLocator(
                k_values,
                coherence_smoothed,
                curve="concave",
                direction="increasing",
                interp_method="polynomial",
                polynomial_degree=self.config.coherence_knee_polynomial_degree
            )
            knee_k = kneedle.knee
            knee_y = kneedle.knee_y
        except Exception:
            knee_k = None
            knee_y = None

        # Find max coherence k (from user_attrs)
        max_coh_idx = np.argmax(coherence_values)
        max_coh_k = k_values[max_coh_idx]

        # Find best trial k (max coherence = trial.value)
        best_trial_k = study.best_trial.user_attrs.get('n_clusters', 0)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Coherence Knee Detection (Smoothed)\n(N={self._N} ideas, window={window_size})',
                     fontsize=14, fontweight='bold')

        color_raw = '#A0A0A0'
        color_smooth = '#F18F01'
        color_knee = '#E94F37'
        color_max_coh = '#2ECC71'
        color_max_score = '#2E86AB'

        # Top-left: Raw coherence
        ax1 = axes[0, 0]
        ax1.plot(k_values, coherence_values, 'o-', color=color_raw, linewidth=1.5, markersize=6, alpha=0.7, label='Raw')
        ax1.scatter([max_coh_k], [coherence_values[max_coh_idx]], color=color_max_coh, s=150, zorder=5,
                   edgecolors='black', linewidth=2, marker='*', label=f'Max coh k={max_coh_k}')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax1.set_ylabel('Coherence', fontsize=11)
        ax1.set_title('Raw Coherence vs K', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Top-right: Smoothed coherence with knee
        ax2 = axes[0, 1]
        ax2.plot(k_values, coherence_values, 'o', color=color_raw, markersize=4, alpha=0.4, label='Raw')
        ax2.plot(k_values, coherence_smoothed, '-', color=color_smooth, linewidth=3, label='Smoothed')
        if knee_k is not None:
            ax2.axvline(x=knee_k, color=color_knee, linestyle='--', linewidth=2, alpha=0.7, label=f'Knee k={knee_k}')
            knee_idx = np.where(k_values == knee_k)[0]
            if len(knee_idx) > 0:
                ax2.scatter([knee_k], [coherence_smoothed[knee_idx[0]]], color=color_knee, s=200, zorder=5,
                           edgecolors='black', linewidth=2, marker='D')
        ax2.axvline(x=best_trial_k, color=color_max_score, linestyle=':', linewidth=2, alpha=0.7, label=f'Max score k={best_trial_k}')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax2.set_ylabel('Coherence (smoothed)', fontsize=11)
        ax2.set_title('Smoothed Coherence with Knee Detection', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Bottom-left: First derivative
        ax3 = axes[1, 0]
        ax3.plot(k_values, coherence_derivative, 'o-', color=color_smooth, linewidth=2, markersize=6)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        if knee_k is not None:
            ax3.axvline(x=knee_k, color=color_knee, linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax3.set_ylabel('d(Coherence)/dk', fontsize=11)
        ax3.set_title('Rate of Change (Derivative)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Bottom-right: Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_lines = [
            "COHERENCE KNEE SELECTION",
            "=" * 45,
            "",
            f"Smoothing window: {window_size} points",
            f"K range: {k_values.min()} - {k_values.max()} ({len(k_values)} values)",
            "",
            f"Raw coherence range: {coherence_values.min():.4f} - {coherence_values.max():.4f}",
            f"Smoothed range: {coherence_smoothed.min():.4f} - {coherence_smoothed.max():.4f}",
            "",
            "=" * 45,
            "",
            "SELECTION RESULTS:",
            f"  Knee detected:    k={knee_k}" + (f" (coherence={knee_y:.4f})" if knee_y else " (none)"),
            f"  Max coherence:    k={max_coh_k} (coherence={coherence_values[max_coh_idx]:.4f})",
            f"  Max score:        k={best_trial_k}",
            "",
            "=" * 45,
            "",
            "INTERPRETATION:",
            "  The knee is where coherence gains",
            "  level off - the point of diminishing",
            "  returns for adding more clusters.",
        ]

        summary_text = "\n".join(summary_lines)
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=10, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        output_path = output_dir / filename

        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        if self._verbose:
            print(f"\n[Coherence Knee Viz] Saved to: {output_path}")

        return output_path


# =============================================================================
# SECTION 4: QUALITY METRICS
# =============================================================================


def calculate_coherence_score(labels: np.ndarray, embeddings: np.ndarray) -> float:
    """
    Calculate mean intra-cluster cosine similarity on original embeddings.

    Standalone function usable by both ParameterOptimizer (HDBSCAN) and
    Clusterer._run_agglomerative() for consistent K selection.

    Args:
        labels: Cluster labels (noise = -1 is excluded)
        embeddings: Original (non-UMAP) embeddings, assumed L2-normalized

    Returns:
        Mean coherence across all clusters (higher = better)
    """
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


def assess_noise_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cohesion_threshold: float
) -> List[int]:
    """Assess quality of noise-derived clusters and return valid ones."""
    valid_clusters = []
    unique_labels = [l for l in set(labels) if l >= 0]

    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) < 2:
            valid_clusters.append(label)
            continue

        similarities = cluster_embeddings @ cluster_embeddings.T
        n = len(cluster_embeddings)
        upper_tri = np.triu_indices(n, k=1)
        mean_sim = float(np.mean(similarities[upper_tri]))

        if mean_sim >= cohesion_threshold:
            valid_clusters.append(label)

    return valid_clusters


def recluster_noise(
    labels: np.ndarray,
    umap_embeddings: np.ndarray,
    original_embeddings: np.ndarray,
    config: ClustererConfig,
    verbose: bool = True
) -> np.ndarray:
    """Two-pass clustering: Attempt to find viable clusters among noise points."""
    if not config.enable_noise_reclustering:
        return labels

    noise_mask = labels == -1
    n_total_noise = noise_mask.sum()

    min_total = config.noise_reclustering_min_total
    if n_total_noise < min_total:
        if verbose:
            print(f"  Skipping noise reclustering: only {n_total_noise} noise points (minimum: {min_total})")
        return labels

    if verbose:
        print(f"\n[Noise Reclustering]")
        print(f"  Total noise points: {n_total_noise}")

    noise_mcs = max(3, config.noise_min_cluster_size)
    noise_ms = max(1, noise_mcs // 2)

    if verbose:
        print(f"  Parameters: min_cluster_size={noise_mcs}, min_samples={noise_ms}")

    U_noise = umap_embeddings[noise_mask]

    noise_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=noise_mcs,
        min_samples=noise_ms,
        metric='euclidean',
        cluster_selection_method=config.noise_reclustering_cluster_selection_method,
        gen_min_span_tree=config.hdbscan_gen_min_span_tree
    )
    noise_labels = noise_hdbscan.fit_predict(U_noise)

    original_noise = original_embeddings[noise_mask]
    valid_noise_clusters = assess_noise_cluster_quality(
        original_noise, noise_labels, config.noise_cohesion_threshold
    )

    if len(valid_noise_clusters) == 0:
        if verbose:
            print(f"  No viable clusters found in noise")
        return labels

    labels_updated = labels.copy()
    max_main_cluster = labels[labels >= 0].max() if np.any(labels >= 0) else -1
    next_cluster_id = max_main_cluster + 1

    noise_indices = np.where(noise_mask)[0]
    n_recovered = 0

    for old_noise_cluster_id in valid_noise_clusters:
        cluster_mask_in_noise = noise_labels == old_noise_cluster_id
        global_indices = noise_indices[cluster_mask_in_noise]
        labels_updated[global_indices] = next_cluster_id
        n_recovered += len(global_indices)
        next_cluster_id += 1

    n_noise_clusters = len(valid_noise_clusters)
    recovery_rate = n_recovered / n_total_noise if n_total_noise > 0 else 0.0
    final_noise = (labels_updated == -1).sum()

    if verbose:
        print(f"  Viable clusters discovered: {n_noise_clusters}")
        print(f"  Points recovered: {n_recovered} ({recovery_rate:.1%})")
        print(f"  Residual noise: {final_noise} ({final_noise/len(labels):.1%})")

    return labels_updated


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
    """Extract lemmatized content words: ADJ, NOUN, PROPN (standalone or in phrases)."""
    if nlp is None:
        nlp = get_spacy_nlp(model_name)

    processed = []

    for doc in nlp.pipe(texts, batch_size=100):
        phrases = []
        current_phrase = []

        for token in doc:
            if token.is_punct or token.is_space:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []
                continue

            if token.pos_ in ('ADJ', 'NOUN', 'PROPN'):
                current_phrase.append(token.lemma_.lower())
            else:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []

        if current_phrase:
            phrases.append(' '.join(current_phrase))

        processed.append(' '.join(phrases))

    return processed


def extract_text_for_display(idea_text: str, template_prefix: Optional[str] = None) -> str:
    """Extract clean text for display (strip template prefix if present)."""
    if template_prefix and idea_text.startswith(template_prefix):
        unique_content = idea_text[len(template_prefix):].strip()
        return unique_content if unique_content else idea_text
    return idea_text


def extract_text_for_format(
    idea_text: str,
    taxonomy_phrase: str,
    embedding_text_format: Optional[str],
    ontology_text: str = ""
) -> str:
    """Extract text matching what was actually embedded based on embedding_text_format."""
    if embedding_text_format == "taxonomy_phrase":
        return taxonomy_phrase if taxonomy_phrase else idea_text
    if embedding_text_format == "ontology":
        return ontology_text if ontology_text else idea_text
    return idea_text


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
        if embedding_text_format in ("taxonomy_phrase", "ontology"):
            return (1, 1)
        return self.config.ctfidf_ngram_range

    def _ensure_ctfidf(self, ngram_range: Optional[Tuple[int, int]] = None):
        """Lazy initialization of c-TF-IDF model."""
        effective_range = ngram_range or self.config.ctfidf_ngram_range

        if self._ctfidf is not None and self._current_ngram_range != effective_range:
            self._ctfidf = None

        if self._ctfidf is None:
            try:
                from .representation.ctfidf_representation import CTfidfRepresentation

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
            except ImportError as e:
                raise ImportError(
                    f"Could not import CTfidfRepresentation: {e}. "
                    "Make sure the representation module is available."
                )

    def _ensure_mmr(self):
        """Lazy initialization of MMR model."""
        if self._mmr is None:
            try:
                from .representation.mmr_representation import MMRRepresentation

                self._mmr = MMRRepresentation(
                    diversity=self.config.mmr_diversity,
                    top_k=self.config.ctfidf_top_k,
                    candidate_multiplier=self.config.mmr_candidate_multiplier
                )
            except ImportError as e:
                raise ImportError(
                    f"Could not import MMRRepresentation: {e}. "
                    "Make sure the representation module is available."
                )

    def _ensure_tfidf(self, ngram_range: Optional[Tuple[int, int]] = None):
        """Lazy initialization of basic TF-IDF model."""
        effective_range = ngram_range or self.config.ctfidf_ngram_range

        if self._tfidf is not None and self._current_ngram_range != effective_range:
            self._tfidf = None

        if self._tfidf is None:
            try:
                from .representation.tfidf_representation import TfidfRepresentation

                self._tfidf = TfidfRepresentation(
                    top_k=self.config.ctfidf_top_k,
                    ngram_range=effective_range,
                    min_df=self.config.ctfidf_min_df
                )
            except ImportError as e:
                raise ImportError(
                    f"Could not import TfidfRepresentation: {e}. "
                    "Make sure the representation module is available."
                )

    def extract_keywords(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_taxonomy_phrases: Optional[Dict[int, List[str]]] = None,
        cluster_ontology_texts: Optional[Dict[int, List[str]]] = None,
        embedding_text_format: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Extract top keywords for each cluster using c-TF-IDF."""
        if not self.config.generate_ctfidf:
            return {}

        self._ensure_ctfidf()

        cleaned_clusters = {}
        for cluster_id, texts in cluster_texts.items():
            taxonomy_list = cluster_taxonomy_phrases.get(cluster_id, []) if cluster_taxonomy_phrases else []
            ontology_list = cluster_ontology_texts.get(cluster_id, []) if cluster_ontology_texts else []
            cleaned_texts = []
            for i, text in enumerate(texts):
                taxonomy = taxonomy_list[i] if i < len(taxonomy_list) else ""
                ontology = ontology_list[i] if i < len(ontology_list) else ""
                cleaned_texts.append(
                    extract_text_for_format(text, taxonomy, embedding_text_format, ontology_text=ontology)
                )
            cleaned_clusters[cluster_id] = cleaned_texts

        if verbose:
            format_display = embedding_text_format or "idea (default)"
            print(f"  Text format: {format_display}")

        if self.config.ctfidf_use_lemmatization:
            if verbose:
                print("  Applying spaCy lemmatization (ADJ | NOUN | PROPN)...")

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

        keywords = self._ctfidf.extract_keywords(cleaned_clusters, verbose=verbose)

        return keywords

    def extract_keywords_from_labels(
        self,
        labels: np.ndarray,
        idea_texts: List[str],
        taxonomy_phrases: Optional[List[str]] = None,
        ontology_texts: Optional[List[str]] = None,
        embedding_text_format: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Extract keywords given cluster labels and idea texts."""
        cluster_texts = {}
        cluster_taxonomy_phrases = {}
        cluster_ontology_texts = {}
        for i, label in enumerate(labels):
            if label >= 0:
                if label not in cluster_texts:
                    cluster_texts[label] = []
                    cluster_taxonomy_phrases[label] = []
                    cluster_ontology_texts[label] = []
                cluster_texts[label].append(idea_texts[i])
                taxonomy = taxonomy_phrases[i] if taxonomy_phrases and i < len(taxonomy_phrases) else ""
                cluster_taxonomy_phrases[label].append(taxonomy)
                ontology = ontology_texts[i] if ontology_texts and i < len(ontology_texts) else ""
                cluster_ontology_texts[label].append(ontology)

        return self.extract_keywords(
            cluster_texts,
            cluster_taxonomy_phrases=cluster_taxonomy_phrases,
            cluster_ontology_texts=cluster_ontology_texts,
            embedding_text_format=embedding_text_format,
            verbose=verbose
        )

    def get_cluster_summary(
        self,
        cluster_id: int,
        keywords: List[Tuple[str, float]],
        max_keywords: int = 10
    ) -> str:
        """Generate formatted text summary for a cluster."""
        if not keywords:
            return f"Cluster {cluster_id}: (no keywords)"

        kw_strs = [f"{kw} ({score:.3f})" for kw, score in keywords[:max_keywords]]
        return f"Cluster {cluster_id}: {', '.join(kw_strs)}"

    def extract_all_keywords(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_taxonomy_phrases: Optional[Dict[int, List[str]]] = None,
        cluster_ontology_texts: Optional[Dict[int, List[str]]] = None,
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
            cluster_taxonomy_phrases,
            cluster_ontology_texts,
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

        return results

    def _preprocess_texts(
        self,
        cluster_texts: Dict[int, List[str]],
        cluster_taxonomy_phrases: Optional[Dict[int, List[str]]],
        cluster_ontology_texts: Optional[Dict[int, List[str]]],
        embedding_text_format: Optional[str],
        verbose: bool
    ) -> Dict[int, List[str]]:
        """Preprocess texts: format extraction and optional lemmatization."""
        cleaned_clusters = {}
        for cluster_id, texts in cluster_texts.items():
            taxonomy_list = cluster_taxonomy_phrases.get(cluster_id, []) if cluster_taxonomy_phrases else []
            ontology_list = cluster_ontology_texts.get(cluster_id, []) if cluster_ontology_texts else []
            cleaned_texts = []
            for i, text in enumerate(texts):
                taxonomy = taxonomy_list[i] if i < len(taxonomy_list) else ""
                ontology = ontology_list[i] if i < len(ontology_list) else ""
                cleaned_texts.append(
                    extract_text_for_format(text, taxonomy, embedding_text_format, ontology_text=ontology)
                )
            cleaned_clusters[cluster_id] = cleaned_texts

        if verbose:
            format_display = embedding_text_format or "idea (default)"
            print(f"  Text format: {format_display}")

        if self.config.ctfidf_use_lemmatization:
            if verbose:
                print("  Applying spaCy lemmatization (ADJ | NOUN | PROPN)...")

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
        taxonomy_phrases: Optional[List[str]] = None,
        ontology_texts: Optional[List[str]] = None,
        embedding_text_format: Optional[str] = None,
        probabilities: Optional[np.ndarray] = None,
        min_probability: Optional[float] = None,
        verbose: bool = False
    ) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
        """Extract all keywords given cluster labels and idea texts.

        Args:
            labels: Cluster assignments for each idea
            idea_texts: Text of each idea
            taxonomy_phrases: Optional taxonomy phrases for each idea
            ontology_texts: Optional ontology text strings for each idea
            embedding_text_format: Text format used for embeddings
            probabilities: Optional HDBSCAN cluster membership probabilities
            min_probability: Only include ideas with probability > this threshold
            verbose: Enable verbose output
        """
        cluster_texts = {}
        cluster_taxonomy_phrases = {}
        cluster_ontology_texts = {}

        # Track filtering stats
        total_per_cluster = {}

        for i, label in enumerate(labels):
            if label >= 0:
                total_per_cluster[label] = total_per_cluster.get(label, 0) + 1
                # Filter by probability if provided
                if probabilities is not None and min_probability is not None:
                    if probabilities[i] <= min_probability:
                        continue
                if label not in cluster_texts:
                    cluster_texts[label] = []
                    cluster_taxonomy_phrases[label] = []
                    cluster_ontology_texts[label] = []
                cluster_texts[label].append(idea_texts[i])
                taxonomy = taxonomy_phrases[i] if taxonomy_phrases and i < len(taxonomy_phrases) else ""
                cluster_taxonomy_phrases[label].append(taxonomy)
                ontology = ontology_texts[i] if ontology_texts and i < len(ontology_texts) else ""
                cluster_ontology_texts[label].append(ontology)

        # Print probability filtering stats
        if verbose and probabilities is not None and min_probability is not None:
            total_before = sum(total_per_cluster.values())
            total_after = sum(len(texts) for texts in cluster_texts.values())
            print(f"  Probability filter (>{min_probability}): {total_after}/{total_before} ideas pass threshold")
            for cluster_id in sorted(total_per_cluster.keys()):
                kept = len(cluster_texts.get(cluster_id, []))
                total = total_per_cluster[cluster_id]
                filtered = total - kept
                if filtered > 0:
                    print(f"    Cluster {cluster_id}: {kept}/{total} kept ({filtered} filtered)")

        return self.extract_all_keywords(
            cluster_texts,
            cluster_taxonomy_phrases=cluster_taxonomy_phrases,
            cluster_ontology_texts=cluster_ontology_texts,
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

    def _get_sample_terminology(
        self,
        embedding_text_format: Optional[str]
    ) -> Tuple[str, str]:
        """Get terminology for samples based on embedding text format."""
        if embedding_text_format == "taxonomy_phrase":
            return ("taxonomy_phrases", "taxonomy phrases")
        if embedding_text_format == "ontology":
            return ("ontology_concepts", "ontology concepts")
        return ("response_ideas", "response ideas")

    def _build_dataset_context_section(self, dataset_context: Optional[Dict[str, str]]) -> str:
        """Build dataset context section for prompt."""
        if not dataset_context:
            return ""
        parts = []
        if dataset_context.get('domain'):
            parts.append(f"Domain: {dataset_context['domain']}")
        if dataset_context.get('entity'):
            parts.append(f"Entity: {dataset_context['entity']}")
        if dataset_context.get('topic'):
            parts.append(f"Topic: {dataset_context['topic']}")
        if dataset_context.get('perspective'):
            parts.append(f"Perspective: {dataset_context['perspective']}")
        if dataset_context.get('intent'):
            parts.append(f"Intent: {dataset_context['intent']}")
        if not parts:
            return ""
        return "\n" + "\n".join(parts)

    def _build_cluster_profile_section(self, distributions: Optional[Dict[str, Dict[str, float]]]) -> str:
        """Build cluster profile section showing sentiment/sense distributions."""
        if not distributions:
            return ""
        parts = []

        sent = distributions.get('sentiment', {})
        if sent and not (len(sent) == 1 and 'neutral' in sent):
            sent_str = ", ".join(f"{int(v*100)}% {k}" for k, v in sent.items())
            parts.append(f"Sentiment: {sent_str}")

        sense = distributions.get('sense', {})
        if sense and not (len(sense) == 1 and 'factual' in sense):
            sense_str = ", ".join(f"{int(v*100)}% {k}" for k, v in sense.items())
            parts.append(f"Sense: {sense_str}")

        if not parts:
            return ""
        return f"""
<cluster_profile>
{chr(10).join(parts)}
Note: Do NOT encode sentiment or evaluation into the theme label.
</cluster_profile>
"""

    def generate_label(
        self,
        cluster_id: int,
        ideas: List[str],
        representative_samples: Optional[List[Tuple[str, float]]] = None,
        keywords: Optional[List[Tuple[str, float]]] = None,
        taxonomy_axis: Optional[str] = None,
        taxonomy_description: Optional[str] = None,
        taxonomy_actionable_type: Optional[str] = None,
        embedding_text_format: Optional[str] = None,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
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

        samples_tag, sample_type = self._get_sample_terminology(embedding_text_format)
        ideas_formatted = "\n".join(f"{i+1}. {idea}" for i, idea in enumerate(sample_ideas))

        if taxonomy_axis:
            actionable_type = taxonomy_actionable_type or "concepts"
            entity = dataset_context.get('entity', 'the entity') if dataset_context else 'the entity'
            taxonomy_context = f"""
<taxonomy_context>
Primary coding dimension: {taxonomy_axis}
Definition: {taxonomy_description or 'Not specified'}
Actionable type: {actionable_type}
Labels MUST describe content within this dimension ONLY.
Do NOT include sentiment, evaluation, tone, or respondent intent in the label.
</taxonomy_context>
"""
            taxonomy_task_guidance = f" ({taxonomy_axis}: {actionable_type})"
            taxonomy_output_constraint = f" within the {taxonomy_axis} dimension"
            taxonomy_task_constraint = (
                f"6. Ensure the theme names a SINGLE {actionable_type} ({taxonomy_axis} dimension)."
            )
            taxonomy_alignment_section = f"""
<taxonomy_alignment_check>
Before finalizing the theme, confirm in one sentence:
"This label fits the {taxonomy_axis} dimension because it names a single {actionable_type} attributed to {entity}."
</taxonomy_alignment_check>
"""
        else:
            taxonomy_context = ""
            taxonomy_task_guidance = ""
            taxonomy_output_constraint = ""
            taxonomy_task_constraint = ""
            taxonomy_alignment_section = ""

        if keywords:
            kw_formatted = "\n".join(f"{i+1}. {kw}" for i, (kw, score) in enumerate(keywords[:10]))
            keywords_section = f"""
<statistical_keywords>
These terms statistically differentiate this cluster from others (c-TF-IDF).
Use to refine - but not override - the representative {sample_type}:
{kw_formatted}
</statistical_keywords>
"""
        else:
            keywords_section = ""

        dataset_context_section = self._build_dataset_context_section(dataset_context)
        cluster_profile_section = self._build_cluster_profile_section(cluster_distributions)

        prompt = CLUSTER_DESCRIPTION_PROMPT.format(
            language=language,
            survey_question=survey_question,
            cluster_id=cluster_id,
            num_ideas=len(ideas),
            samples_tag=samples_tag,
            sample_type=sample_type,
            taxonomy_context=taxonomy_context,
            taxonomy_task_guidance=taxonomy_task_guidance,
            taxonomy_output_constraint=taxonomy_output_constraint,
            taxonomy_task_constraint=taxonomy_task_constraint,
            taxonomy_alignment_section=taxonomy_alignment_section,
            keywords_section=keywords_section,
            dataset_context_section=dataset_context_section,
            cluster_profile_section=cluster_profile_section,
            ideas_list=ideas_formatted
        )

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
        cluster_keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        representative_samples: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        extraction_metadata=None,
        embedding_text_format: Optional[str] = None,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        cluster_distributions: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
        verbose: bool = False
    ) -> Dict[int, ClusterLabel]:
        """Generate LLM-based labels for all clusters."""
        if not self.config.generate_llm_labels:
            return {}

        taxonomy_axis = None
        taxonomy_description = None
        taxonomy_actionable_type = None
        if extraction_metadata:
            taxonomy_axis = getattr(extraction_metadata, 'taxonomy_primary_axis', None)
            taxonomy_description = getattr(extraction_metadata, 'taxonomy_axis_description', None)
            taxonomy_actionable_type = getattr(extraction_metadata, 'taxonomy_actionable_type', None)

        if verbose:
            print(f"\n[LLM Label Generation]")
            print(f"  Model: {self._model}")
            print(f"  Clusters to label: {len(cluster_texts)}")
            if taxonomy_axis:
                print(f"  Taxonomy axis: {taxonomy_axis}")
            if dataset_context:
                context_parts = [f"{k}={v}" for k, v in dataset_context.items() if v]
                if context_parts:
                    print(f"  Dataset context: {', '.join(context_parts)}")
            if cluster_distributions:
                print(f"  Including cluster profiles (sentiment/sense distributions)")

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
                representative_samples=samples,
                keywords=keywords,
                taxonomy_axis=taxonomy_axis,
                taxonomy_description=taxonomy_description,
                taxonomy_actionable_type=taxonomy_actionable_type,
                embedding_text_format=embedding_text_format,
                survey_question=survey_question,
                language=language,
                dataset_context=dataset_context,
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
