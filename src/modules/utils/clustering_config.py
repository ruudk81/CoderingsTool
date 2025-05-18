"""Configuration for automatic clustering."""
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ClusteringConfig:
    """Configuration for automatic clustering mode."""
    
    # User-configurable options
    embedding_type: str = "description"  # Default to description as requested
    language: str = "nl"  # Default to Dutch
    
    # Quality thresholds for automatic decisions
    min_quality_score: float = 0.3  # Minimum acceptable quality
    max_noise_ratio: float = 0.5    # Maximum acceptable noise before micro-clustering
    
    # Optional overrides (mostly for testing)
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    
    def get_reducer_params(self, data_size: int) -> Dict:
        """Get UMAP parameters based on data size."""
        n_neighbors = min(30, max(15, data_size // 50))
        min_dist = 0.0 if data_size < 1000 else 0.1
        
        return {
            'n_neighbors': n_neighbors,
            'n_components': 10,
            'min_dist': min_dist,
            'metric': 'cosine',
            'random_state': 42,
            'n_jobs': 1
        }
    
    def get_auto_params(self, data_size: int) -> Dict:
        """Get clustering parameters based on data size."""
        # Use override values if provided
        if self.min_cluster_size is not None and self.min_samples is not None:
            return {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'
            }
        
        # Auto-calculate based on data size
        if data_size < 100:
            min_cluster_size = 2
            min_samples = 1
        elif data_size < 500:
            min_cluster_size = 3
            min_samples = 2
        elif data_size < 1000:
            min_cluster_size = 5
            min_samples = 3
        else:
            min_cluster_size = 10
            min_samples = 5
            
        return {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        }