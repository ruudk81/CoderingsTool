"""Cache configuration for the CoderingsTool pipeline"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import os


@dataclass
class CacheConfig:
    """Configuration for cache management system"""
    
    # Base cache directory
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    
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
        import hashlib
        import json
        
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


# Global configuration instance
DEFAULT_CACHE_CONFIG = CacheConfig()
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()

# Environment-based overrides
if os.getenv("CODERINGSTOOL_CACHE_DIR"):
    DEFAULT_CACHE_CONFIG.cache_dir = Path(os.getenv("CODERINGSTOOL_CACHE_DIR"))

if os.getenv("CODERINGSTOOL_MAX_CACHE_AGE"):
    DEFAULT_CACHE_CONFIG.max_cache_age_days = int(os.getenv("CODERINGSTOOL_MAX_CACHE_AGE"))

if os.getenv("CODERINGSTOOL_VERBOSE"):
    DEFAULT_CACHE_CONFIG.verbose = True