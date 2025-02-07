"""Configuration for clustering module"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import json


@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters"""
    
    # Dimensionality reduction settings
    reducer_type: str = "umap"  # Options: "umap", "tsne", "pca"
    n_components: int = 10
    n_neighbors: int = 15
    min_dist: float = 0.1
    reducer_metric: str = "cosine"
    random_state: int = 42
    
    # Clustering settings
    clusterer_type: str = "hdbscan"  # Options: "hdbscan", "dbscan", "kmeans"
    min_cluster_size: int = 5
    min_samples: int = 3
    clustering_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    
    # Text processing settings
    ngram_range: Tuple[int, int] = (1, 3)
    max_features: int = 1000
    min_df: int = 1
    
    # Meta-clustering settings
    meta_min_cluster_size: int = 2
    meta_metric: str = "euclidean"
    
    # Quality control settings
    min_silhouette_score: float = 0.3
    max_noise_ratio: float = 0.5
    calculate_quality_metrics: bool = True
    
    # Processing settings
    verbose: bool = True
    n_jobs: int = 1
    low_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        # Convert tuple to list for JSON serialization
        if 'ngram_range' in config_dict:
            config_dict['ngram_range'] = list(config_dict['ngram_range'])
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig':
        """Create configuration from dictionary"""
        # Convert list back to tuple
        if 'ngram_range' in config_dict:
            config_dict['ngram_range'] = tuple(config_dict['ngram_range'])
        return cls(**config_dict)
    
    def get_hash(self) -> str:
        """Generate hash of configuration for cache invalidation"""
        import hashlib
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_reducer_params(self) -> Dict[str, Any]:
        """Get parameters for dimensionality reduction model"""
        if self.reducer_type == "umap":
            return {
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "min_dist": self.min_dist,
                "metric": self.reducer_metric,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "low_memory": self.low_memory,
                "transform_seed": self.random_state
            }
        elif self.reducer_type == "tsne":
            return {
                "n_components": min(self.n_components, 3),  # TSNE max 3 components
                "metric": self.reducer_metric,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs
            }
        elif self.reducer_type == "pca":
            return {
                "n_components": self.n_components,
                "random_state": self.random_state
            }
        else:
            raise ValueError(f"Unknown reducer type: {self.reducer_type}")
    
    def get_clusterer_params(self) -> Dict[str, Any]:
        """Get parameters for clustering model"""
        if self.clusterer_type == "hdbscan":
            return {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "metric": self.clustering_metric,
                "cluster_selection_method": self.cluster_selection_method,
                "prediction_data": True,
                "approx_min_span_tree": False,
                "gen_min_span_tree": True
            }
        elif self.clusterer_type == "dbscan":
            return {
                "eps": 0.5,  # This might need to be configurable
                "min_samples": self.min_samples,
                "metric": self.clustering_metric,
                "n_jobs": self.n_jobs
            }
        elif self.clusterer_type == "kmeans":
            return {
                "n_clusters": 10,  # This needs to be configurable
                "random_state": self.random_state,
                "n_jobs": self.n_jobs
            }
        else:
            raise ValueError(f"Unknown clusterer type: {self.clusterer_type}")


# Preset configurations
PRESETS = {
    "default": ClusteringConfig(),
    
    "large_dataset": ClusteringConfig(
        min_cluster_size=10,
        min_samples=5,
        n_neighbors=30,
        low_memory=True
    ),
    
    "small_dataset": ClusteringConfig(
        min_cluster_size=3,
        min_samples=2,
        n_neighbors=5,
        n_components=5
    ),
    
    "high_quality": ClusteringConfig(
        min_cluster_size=8,
        min_samples=4,
        max_noise_ratio=0.3,
        min_silhouette_score=0.4,
        calculate_quality_metrics=True
    ),
    
    "fast_processing": ClusteringConfig(
        reducer_type="pca",
        n_components=5,
        calculate_quality_metrics=False,
        low_memory=False
    )
}


def get_preset_config(preset_name: str) -> ClusteringConfig:
    """Get a preset configuration by name"""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset_name]