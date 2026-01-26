"""
ClustererV2 - Unified Clustering Utility with Bayesian Optimization

This module consolidates the best practices from clustering experiments:
1. Bayesian HDBSCAN Grid Search (Optuna)
2. kNN Distance Analysis for Noise Detection
3. DVC (Density Variation Coefficient) for Algorithm Selection
4. Coherence-Based Algorithm Comparison
5. c-TF-IDF Cluster Representations

Usage:
    from clusterer_v2 import ClustererV2, ClustererV2Config

    config = ClustererV2Config(algorithm_mode="auto", verbose=True)
    clusterer = ClustererV2(embeddings_data, config=config)
    clusterer.run()

    # Get results
    cluster_models = clusterer.to_cluster_model()
    metrics = clusterer.get_metrics()
    recommendation = clusterer.get_algorithm_recommendation()
"""

from .config import ClustererV2Config
from .clusterer import ClustererV2
from .algorithm_selector import AlgorithmSelector, AlgorithmRecommendation
from .quality_metrics import ClusterQualityMetrics, ClusteringMetrics
from .label_generator import LabelGenerator, ClusterLabel

__all__ = [
    "ClustererV2",
    "ClustererV2Config",
    "AlgorithmSelector",
    "AlgorithmRecommendation",
    "ClusterQualityMetrics",
    "ClusteringMetrics",
    "LabelGenerator",
    "ClusterLabel",
]
