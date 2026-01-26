"""
Clusterer Quality Metrics Module

Implements comprehensive clustering quality metrics:
- Coherence (mean intra-cluster cosine similarity on original embeddings)
- DBCV (Density-Based Clustering Validation)
- Silhouette score
- Persistence metrics (from HDBSCAN)
- Noise quality assessment
- Cluster size distribution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan

from .config import ClustererV2Config


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
    low_prob_ratio: Optional[float] = None  # % of clustered points with prob < threshold
    per_cluster_mean_prob: Optional[List[Tuple[int, int, float]]] = None  # (cluster_id, size, mean_prob)

    # Outlier metrics (from HDBSCAN outlier_scores_ / GLOSH)
    mean_outlier_score: Optional[float] = None
    high_outlier_ratio: Optional[float] = None  # % of points with score > threshold

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

    def __init__(self, config: ClustererV2Config):
        self.config = config

    def calculate_coherence(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Tuple[float, Dict[str, int], List[Tuple[int, int, float]]]:
        """
        Calculate mean coherence and per-cluster breakdown.

        Coherence = mean pairwise cosine similarity within cluster (using original embeddings).

        Thresholds:
        - Unacceptable: coherence < 0.70
        - Low: 0.70 <= coherence < 0.90
        - Moderate: 0.90 <= coherence < 0.95
        - High: coherence >= 0.95

        Args:
            labels: Cluster labels
            embeddings: L2-normalized original embeddings

        Returns:
            (mean_coherence, breakdown_counts, per_cluster_list)
        """
        unique_labels = [l for l in set(labels) if l >= 0]  # Exclude noise

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
            if coherence < self.config.coherence_acceptable:
                n_unacceptable += 1
            elif coherence < self.config.coherence_moderate:
                n_low += 1
            elif coherence < self.config.coherence_high:
                n_moderate += 1
            else:
                n_high += 1

        # Sort by label
        per_cluster.sort(key=lambda x: x[0])

        # Calculate mean coherence
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
        """
        Compute DBCV (Density-Based Clustering Validation) score.

        Args:
            labels: Cluster labels (-1 for noise)
            embeddings: Data points

        Returns:
            DBCV score (higher is better, range roughly -1 to 1)
        """
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
        """
        Compute geometry-based metrics (silhouette, CH, DB).

        Args:
            labels: Cluster labels
            embeddings: Data points (reduced space)

        Returns:
            Dict with silhouette, calinski_harabasz, davies_bouldin
        """
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
        """
        Compute cluster size distribution.

        Args:
            labels: Cluster labels

        Returns:
            Dict with sizes list and statistics
        """
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
        """
        Extract metrics from HDBSCAN probabilities_.

        probabilities_ indicates membership strength: 0 = noise/not in cluster,
        1 = at the heart of the cluster.

        Args:
            probabilities: HDBSCAN probabilities_ array
            labels: Cluster labels

        Returns:
            Dict with mean_probability, low_prob_ratio, per_cluster_mean_prob
        """
        mask = labels >= 0  # Exclude noise (prob=0 by definition)
        probs = probabilities[mask]

        if len(probs) == 0:
            return {
                'mean_probability': None,
                'low_prob_ratio': None,
                'per_cluster_mean_prob': []
            }

        mean_prob = float(np.mean(probs))
        low_prob_ratio = float((probs < self.config.low_probability_threshold).sum() / len(probs))

        # Per-cluster breakdown
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
        """
        Extract metrics from HDBSCAN outlier_scores_ (GLOSH algorithm).

        Higher outlier score = more likely to be a local outlier.

        Args:
            outlier_scores: HDBSCAN outlier_scores_ array
            labels: Cluster labels (used for context, but scores apply to all points)

        Returns:
            Dict with mean_outlier_score, high_outlier_ratio
        """
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
        """
        Calculate all configured metrics.

        Args:
            labels: Cluster labels
            embeddings_reduced: UMAP-reduced embeddings
            embeddings_original: Original L2-normalized embeddings
            hdbscan_model: Optional fitted HDBSCAN (for persistence)
            algorithm_used: Name of algorithm used
            algorithm_params: Parameters used

        Returns:
            ClusteringMetrics with all computed values
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        noise_rate = noise_count / len(labels) if len(labels) > 0 else 0.0

        # Coherence
        mean_coherence, breakdown, per_cluster = self.calculate_coherence(
            labels, embeddings_original
        )

        # Build breakdown string
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

        # DBCV
        dbcv = self.compute_dbcv(labels, embeddings_reduced)

        # Geometry metrics
        geometry = self.compute_geometry_metrics(labels, embeddings_reduced)

        # Cluster sizes
        sizes = self.compute_cluster_sizes(labels)

        # Persistence metrics (if HDBSCAN model provided)
        persistence_metrics = {}
        relative_validity = None
        prob_metrics = {}
        outlier_metrics = {}

        if hdbscan_model is not None:
            from .algorithm_selector import AlgorithmSelector
            selector = AlgorithmSelector(self.config)
            persistence_metrics = selector.extract_persistence_metrics(hdbscan_model, labels)

            try:
                relative_validity = float(hdbscan_model.relative_validity_)
            except AttributeError:
                relative_validity = None

            # Probability metrics
            if hasattr(hdbscan_model, 'probabilities_') and hdbscan_model.probabilities_ is not None:
                prob_metrics = self.compute_probability_metrics(hdbscan_model.probabilities_, labels)

            # Outlier metrics
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
