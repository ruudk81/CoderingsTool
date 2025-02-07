import os
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, field

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# File handling (only keep what's used)
ALLOWED_EXTENSIONS = ['.sav']

# LLM settings (core settings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# Hunspell path detection
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'utils':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'hunspell'))
elif os.path.basename(current_dir) == 'modules':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'hunspell'))
elif os.path.basename(current_dir) == 'src':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', 'hunspell'))
elif os.path.basename(current_dir) == 'Coderingstool':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, 'hunspell'))

# Hunspell settings (only keep what's used)
HUNSPELL_PATH = os.path.join(hunspell_dir, "hunspell.exe")
DUTCH_DICT_PATH = os.path.join(hunspell_dir, "dict", "nl_NL")
ENGLISH_DICT_PATH = os.path.join(hunspell_dir, "dict", "en_GB")
DEFAULT_LANGUAGE = "Dutch"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for different models used in each pipeline stage"""
    
    # Stage-specific models
    spell_check_model: str = "gpt-4o-mini"           # Fast model for spell checking
    quality_filter_model: str = "gpt-4o-mini"        # Fast model for quality filtering
    segmentation_model: str = "gpt-4o-mini"          # Fast model for response segmentation
    description_model: str = "gpt-4o-mini"           # Fast model for description generation
    embedding_model: str = "text-embedding-3-large"  # Embedding model
    cluster_merge_model: str = "gpt-4o-mini"         # Fast model for cluster merging
    labelling_model: str = "gpt-4o"                  # High-quality model for final labelling
    
    # Global parameters
    seed: int = 42
    default_temperature: float = 0.0  # Default to deterministic
    max_tokens: int = 16000 # 4o-mini
    
    # Stage-specific temperatures (override default if needed)
    spell_check_temperature: float = 0.0
    quality_filter_temperature: float = 0.0
    segmentation_temperature: float = 0.0
    description_temperature: float = 0.0
    cluster_merge_temperature: float = 0.0   
    labelling_temperature: float = 0.0      
    
    def get_model_for_stage(self, stage: str) -> str:
        """Get the appropriate model for a pipeline stage"""
        stage_models = {
            'spell_check': self.spell_check_model,
            'quality_filter': self.quality_filter_model,
            'segmentation': self.segmentation_model,
            'description': self.description_model,
            'embedding': self.embedding_model,
            'cluster_merge': self.cluster_merge_model,
            'labelling': self.labelling_model,
        }
        return stage_models.get(stage, DEFAULT_MODEL)
    
    def get_temperature_for_stage(self, stage: str) -> float:
        """Get the appropriate temperature for a pipeline stage"""
        stage_temperatures = {
            'spell_check': self.spell_check_temperature,
            'quality_filter': self.quality_filter_temperature,
            'segmentation': self.segmentation_temperature,
            'description': self.description_temperature,
            'cluster_merge': self.cluster_merge_temperature,
            'labelling': self.labelling_temperature,
        }
        return stage_temperatures.get(stage, self.default_temperature)

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
    
    cache_dir: Path = field(default_factory=get_default_cache_dir)
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


# @dataclass
# class ProcessingConfig:
#     """Configuration for processing parameters that affect cache validity"""
    
#     # Language settings
#     language: str = "nl"
#     spell_check_enabled: bool = True
    
#     # Output control
#     verbose: bool = False
   
#     # Clustering settings
#     clustering_algorithm: str = "hdbscan"
#     min_cluster_size: int = 5
#     min_samples: int = 3
    
#     # Model configuration
#     models: ModelConfig = None
    
#     def __post_init__(self):
#         if self.models is None:
#             self.models = ModelConfig()
    
#     def to_dict(self) -> dict:
#         """Convert to dictionary for serialization"""
#         return {
#             k: v for k, v in self.__dict__.items()
#             if not k.startswith('_')
#         }
    
#     def get_hash(self) -> str:
#         """Generate hash of configuration for cache invalidation"""
#         config_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
#         return hashlib.md5(config_str.encode()).hexdigest()


# @dataclass
# class ClusteringConfig:
#     """Configuration for automatic clustering mode."""
    
#     embedding_type: str = "description"  # Default to description as requested
#     language: str = "nl"  # Default to Dutch
    
#     # Optional overrides (mostly for testing)
#     min_cluster_size: Optional[int] = None
#     min_samples: Optional[int] = None
    
#     def get_reducer_params(self, data_size: int) -> Dict:
#         """Get UMAP parameters based on data size."""
#         n_neighbors = min(30, max(15, data_size // 50))
#         min_dist = 0.0 if data_size < 1000 else 0.1
        
#         return {
#             'n_neighbors': n_neighbors,
#             'n_components': 10,
#             'min_dist': min_dist,
#             'metric': 'cosine',
#             'random_state': 42,
#             'n_jobs': 1
#         }
    
#     def get_auto_params(self, data_size: int) -> Dict:
#         """Get clustering parameters based on data size."""
#         # Use override values if provided
#         if self.min_cluster_size is not None and self.min_samples is not None:
#             return {
#                 'min_cluster_size': self.min_cluster_size,
#                 'min_samples': self.min_samples,
#                 'metric': 'euclidean',
#                 'cluster_selection_method': 'eom'
#             }
        
#         # Auto-calculate based on data size
#         if data_size < 100:
#             min_cluster_size = 2
#             min_samples = 1
#         elif data_size < 500:
#             min_cluster_size = 3
#             min_samples = 2
#         elif data_size < 1000:
#             min_cluster_size = 5
#             min_samples = 3
#         else:
#             min_cluster_size = 10
#             min_samples = 5
            
#         return {
#             'min_cluster_size': min_cluster_size,
#             'min_samples': min_samples,
#             'metric': 'euclidean',
#             'cluster_selection_method': 'eom'
#         }


@dataclass
class LabellerConfig:
    """Configuration for hierarchical labeller"""
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 4000
    language: str = DEFAULT_LANGUAGE
    max_retries: int = 3
    batch_size: int = 8  # Micro-clusters per batch
    top_k_representatives: int = 3  # Representative codes per cluster
    concurrent_requests: int = 5

@dataclass
class SpellCheckConfig:
    """Configuration for spell checking step"""
    batch_size: int = 20
    temperature: float = 0.0
    max_tokens: int = 4000
    retries: int = 3
    retry_delay: int = 2
    max_batch_size: int = 5
    completion_reserve: int = 1000
    cache_size: int = 10000
    spacy_batch_size: int = 32
    repeated_char_threshold: int = 5  # Characters repeated 5+ times
    max_correction_examples: int = 10  # For verbose output
    seed: int = 42
    context_chars: int = 20  # Characters of context for spell checking
    spell_check_threshold: float = 0.7  # Confidence threshold for corrections
    max_concurrent_requests: int = 5  # For API rate limiting


@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering step"""
    batch_size: int = 20
    temperature: float = 0.0
    max_tokens: int = 4000
    retries: int = 3
    instructor_retries: int = 3
    high_quality_threshold: float = 0.7
    medium_quality_threshold: float = 0.4
    max_filter_examples: int = 5  # For verbose output
    model: str = "gpt-4o-mini"  # Model for quality assessment
    max_concurrent_requests: int = 5  # For API rate limiting


@dataclass
class SegmentationConfig:
    """Configuration for segmentation and description step"""
    max_tokens: int = 16000
    completion_reserve: int = 1000
    max_batch_size: int = 5
    retry_delay: int = 2
    max_retries: int = 3
    spacy_batch_size: int = 32
    umap_n_jobs: int = 1
    max_code_examples: int = 5  # For verbose output
    max_sample_responses: int = 3  # For verbose output
    model: str = "gpt-4o-mini"  # Model for segmentation
    temperature: float = 0.0  # Temperature for generation
    max_concurrent_requests: int = 5  # For API rate limiting


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation step"""
    batch_size: int = 100
    max_concurrent_requests: int = 5
    embedding_model: str = "text-embedding-3-large"
    max_sample_responses: int = 3  # For verbose output


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Global configuration instances
DEFAULT_CACHE_CONFIG = CacheConfig()
# DEFAULT_PROCESSING_CONFIG = ProcessingConfig()
# DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()
DEFAULT_LABELLER_CONFIG = LabellerConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()

# Pipeline step configurations
DEFAULT_SPELLCHECK_CONFIG = SpellCheckConfig()
DEFAULT_QUALITY_FILTER_CONFIG = QualityFilterConfig()
DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()

# Environment-based model overrides
if os.getenv("CODERINGSTOOL_SPELL_MODEL"):
    DEFAULT_MODEL_CONFIG.spell_check_model = os.getenv("CODERINGSTOOL_SPELL_MODEL")
if os.getenv("CODERINGSTOOL_LABELLING_MODEL"):
    DEFAULT_MODEL_CONFIG.labelling_model = os.getenv("CODERINGSTOOL_LABELLING_MODEL")
if os.getenv("CODERINGSTOOL_EMBEDDING_MODEL"):
    DEFAULT_MODEL_CONFIG.embedding_model = os.getenv("CODERINGSTOOL_EMBEDDING_MODEL")