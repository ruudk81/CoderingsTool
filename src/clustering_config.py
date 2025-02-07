"""Configuration for automatic clustering"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class ClusteringConfig:
    """Configuration for automatic clustering - minimal user configuration"""
    
    # User-configurable settings
    embedding_type: str = "description"  # Options: "code" or "description"
    language: str = "nl"  # For text processing: "nl" or "en"
    
    # Quality thresholds for automatic decisions
    min_quality_score: float = 0.3  # Minimum acceptable quality (0-1)
    max_noise_ratio: float = 0.5  # Maximum acceptable outlier ratio
    
    # Processing settings
    verbose: bool = True
    calculate_quality_metrics: bool = True
    
    # Fixed settings for automatic mode (not user-configurable)
    random_state: int = 42
    n_jobs: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def get_hash(self) -> str:
        """Generate hash of configuration for cache invalidation"""
        import hashlib
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_auto_params(self, data_size: int) -> Dict[str, Any]:
        """
        Automatically determine optimal parameters based on data size
        
        Args:
            data_size: Number of items to cluster
            
        Returns:
            Dictionary of parameters for HDBSCAN
        """
        # Automatic parameter selection based on data size
        if data_size < 100:
            min_cluster_size = 3
            min_samples = 2
        elif data_size < 500:
            min_cluster_size = 5
            min_samples = 3
        elif data_size < 1000:
            min_cluster_size = 10
            min_samples = 5
        else:
            min_cluster_size = 15
            min_samples = 8
        
        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
            "approx_min_span_tree": False,
            "gen_min_span_tree": True
        }
    
    def get_reducer_params(self, data_size: int) -> Dict[str, Any]:
        """
        Automatically determine UMAP parameters based on data size
        
        Args:
            data_size: Number of items to reduce
            
        Returns:
            Dictionary of parameters for UMAP
        """
        # Automatic UMAP parameters
        if data_size < 100:
            n_neighbors = 5
            n_components = 5
        elif data_size < 500:
            n_neighbors = 15
            n_components = 10
        else:
            n_neighbors = 30
            n_components = 10
            
        return {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "low_memory": True,
            "transform_seed": self.random_state
        }
    
    def get_text_processing_params(self) -> Dict[str, Any]:
        """Get text processing parameters based on language"""
        return {
            "ngram_range": (1, 3),
            "max_features": 1000,
            "min_df": 1,
            "language": self.language
        }


# Preset configurations for different scenarios
PRESETS = {
    "default": ClusteringConfig(),
    
    "english": ClusteringConfig(
        language="en"
    ),
    
    "code_embeddings": ClusteringConfig(
        embedding_type="code"
    ),
    
    "strict_quality": ClusteringConfig(
        min_quality_score=0.4,
        max_noise_ratio=0.3
    ),
    
    "lenient_quality": ClusteringConfig(
        min_quality_score=0.2,
        max_noise_ratio=0.7
    )
}


def get_preset_config(preset_name: str) -> ClusteringConfig:
    """Get a preset configuration by name"""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset_name]