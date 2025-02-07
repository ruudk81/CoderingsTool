"""Configuration for automatic clustering."""
from dataclasses import dataclass
from typing import Optional

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