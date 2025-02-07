import os

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict #, List

from dataclasses import dataclass, field

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# File handling
ALLOWED_EXTENSIONS = ['.sav']
MAX_FILE_SIZE_MB = 50

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
CONTEXT_WINDOW = 128000
MAX_OUTPUT_TOKENS = 16000

BATCH_SIZE = 100

# Hunspell path detection
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'utils':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'hunspell'))
elif os.path.basename(current_dir) == 'src':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', 'hunspell'))
elif os.path.basename(current_dir) == 'Coderingstool':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, 'hunspell'))

# Hunspell settings
SUPPORTED_LANGUAGES = ["nl", "en_GB"]  # Dutch and English
HUNSPELL_PATH = os.path.join(hunspell_dir, "hunspell.exe")
DUTCH_DICT_PATH = os.path.join(hunspell_dir, "dict", "nl_NL")
ENGLISH_DICT_PATH = os.path.join(hunspell_dir, "dict", "en_GB")
DEFAULT_LANGUAGE = "Dutch"

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

def get_default_cache_dir():
    """Get the default cache directory relative to project root"""
    src_dir = Path(__file__).parent
    return src_dir.parent / "data" / "cache"


@dataclass
class CacheConfig:
    """Configuration for cache management system"""
    
    # Base cache directory - absolute path to project_root/data/cache
    cache_dir: Path = field(default_factory=get_default_cache_dir)
    
    # SQLite database name
    db_name: str = "cache.db"
    
    # Step prefixes for file naming
    step_prefixes: Dict[str, str] = field(default_factory=lambda: {
        "data": "001",
        "preprocessed": "002", 
        "segmented_descriptions": "003",
        "embeddings": "004",
        "clusters": "005",
        "labels": "006",
        "results": "007"
        })
    
    # Cache validity settings
    max_cache_age_days: int = 30
    check_file_hash: bool = True
    
    # File handling settings
    enable_compression: bool = False
    compression_level: int = 6  # 1-9, higher = more compression
    use_atomic_writes: bool = True
    
    # Performance settings
    batch_size: int = 1000
    memory_limit_mb: int = 500
    
    # Cleanup settings
    auto_cleanup: bool = True
    cleanup_interval_days: int = 7
    max_cache_size_gb: float = 10.0
    
    # Logging settings
    log_cache_operations: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Ensure cache directory exists and adjust settings for platform"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Disable atomic writes on Windows to avoid file locking issues
        import platform
        if platform.system() == 'Windows':
            self.use_atomic_writes = False
        
    @property
    def db_path(self) -> Path:
        """Full path to the SQLite database"""
        return self.cache_dir / self.db_name
    
    def get_step_prefix(self, step_name: str) -> str:
        """Get the numeric prefix for a given step"""
        return self.step_prefixes.get(step_name, "999")
    
    def get_cache_filename(self, original_filename: str, step_name: str) -> str:
        """Generate cache filename with prefix"""
        base_name = Path(original_filename).stem
        prefix = self.get_step_prefix(step_name)
        return f"{prefix}_{step_name}_{base_name}.csv"
    
    def get_cache_filepath(self, original_filename: str, step_name: str) -> Path:
        """Get full path for cached file"""
        cache_filename = self.get_cache_filename(original_filename, step_name)
        return self.cache_dir / cache_filename


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters that affect cache validity"""
    
    # Language settings
    language: str = "nl"
    spell_check_enabled: bool = True
    
    # Quality filter settings
    quality_threshold: float = 0.5
    min_response_length: int = 5
    
    # Clustering settings
    clustering_algorithm: str = "hdbscan"
    min_cluster_size: int = 5
    min_samples: int = 3
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    
    # Labeling settings
    labeling_model: str = "gpt-4o-mini"
    labeling_temperature: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def get_hash(self) -> str:
        """Generate hash of configuration for cache invalidation"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

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


# =============================================================================
# LABELLING CONFIGURATION
# =============================================================================


@dataclass
class LabellerConfig:
    """Configuration for hierarchical labeller"""
    model: str = "gpt-4o-mini"   
    temperature: float = 0.0   
    max_tokens: int = 4000
    language: str = DEFAULT_LANGUAGE
    max_retries: int = 3
    batch_size: int = 8
    top_k_representatives: int = 3
    concurrent_requests: int = 5
    seed: int = 42  # Fixed seed for reproducibility
    use_sequential_processing: bool = True
    validation_threshold: float = 0.95


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Global configuration instances
DEFAULT_CACHE_CONFIG = CacheConfig()
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()
DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()
DEFAULT_LABELLER_CONFIG = LabellerConfig()