"""
Clusterer Algorithm Selector Module

Implements automatic algorithm selection using:
1. DVC (Density Variation Coefficient) - computed on original embeddings
2. kNN Knee Detection - computed on UMAP-reduced embeddings

Note: Persistence is now computed per-trial in the optimizer, not in this phase.

Decision Logic (Simplified):
┌─────────────────┬─────────────────────────────────────────────┐
│ DVC < 0.25      │ FORCE Agglomerative (skip HDBSCAN entirely) │
├─────────────────┼─────────────────────────────────────────────┤
│ DVC ≥ 0.25      │ Continue to knee detection:                 │
│                 │   - Sharp knee (ydiff ≥ 0.6) → HDBSCAN      │
│                 │   - Flat knee (ydiff < 0.6) → AGGLOMERATIVE │
└─────────────────┴─────────────────────────────────────────────┘
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import hdbscan

from .config import ClustererV2Config


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

    def __init__(self, config: ClustererV2Config):
        self.config = config

    def compute_dvc(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute Density Variation Coefficient.

        DVC = std(d_k) / mean(d_k), where d_k is distance to k-th nearest neighbor.

        High DVC (>0.45) indicates varying density → HDBSCAN better
        Low DVC (<0.25) indicates uniform density → Agglomerative better

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

        Sharp knee (y_difference >= 0.6) indicates density transition → HDBSCAN suitable
        Flat curve indicates uniform density → Agglomerative/K-means better

        Adaptive parameters based on dataset size:
        - S (sensitivity) scales with n/100
        - interp_method: polynomial for small datasets, interp1d for larger

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
        1. If DVC < 0.25 → FORCE Agglomerative (skip HDBSCAN entirely)
        2. Otherwise, use knee detection:
           - Sharp knee (ydiff ≥ 0.6) → HDBSCAN
           - Flat knee (ydiff < 0.6) → AGGLOMERATIVE

        Note: Persistence is now computed per-trial in the optimizer, not here.

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
                reasoning=f"DVC={dvc_value:.3f} < {force_threshold} indicates uniform density → HDBSCAN skipped",
                is_forced=True
            )

        # Standard decision based on knee detection
        if has_sharp_knee:
            combined_recommendation = "HDBSCAN"
            recommended_algorithm = "HDBSCAN"
            confidence = "high"
            reasoning = f"Sharp knee (ydiff={y_difference:.2f} ≥ 0.6) indicates density transitions"
        elif enable_agglomerative:
            # Flat knee → Agglomerative (only if fallback enabled)
            combined_recommendation = "AGGLOMERATIVE"
            recommended_algorithm = "AGGLOMERATIVE"
            confidence = "medium"
            reasoning = f"Flat knee (ydiff={y_difference:.2f} < 0.6) suggests uniform density"
        else:
            # Flat knee but agglomerative disabled → stick with HDBSCAN
            combined_recommendation = "HDBSCAN"
            recommended_algorithm = "HDBSCAN"
            confidence = "medium"
            reasoning = f"Flat knee (ydiff={y_difference:.2f} < 0.6) but agglomerative fallback disabled → using HDBSCAN"

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
