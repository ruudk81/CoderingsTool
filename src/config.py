import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# File handling (only keep what's used)
ALLOWED_EXTENSIONS = ['.sav']

# LLM settings (core settings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"

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



@dataclass
class LabellerConfig:
    """Configuration for hierarchical labelling"""
    # Model settings
    model: str = "gpt-4o-mini"  # Primary model for labelling
    temperature: float = 0.1  # Lower for more consistent output
    max_tokens: int = 4000
    seed: int = 42  # For reproducibility
    api_key: Optional[str] = None  # Will use env var if not provided
    
    # Language and localization
    language: str = DEFAULT_LANGUAGE
    
    # Processing parameters
    top_k_representatives: int = 3  # Representative examples per cluster
    map_reduce_threshold: int = 30  # Use MapReduce if more clusters
    batch_size: int = 10  # Clusters per batch in MapReduce
    assignment_threshold: float = 0.5  # Minimum probability for assignment (lowered for better coverage)
    
    # LLM refinement option
    use_llm_refinement: bool = False  # For Phase 4 enhancement
    
    # Retry and concurrency settings
    max_retries: int = 3
    concurrent_requests: int = 10  # Increased for better performance
    retry_delay: int = 2  # Seconds between retries

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
# CLUSTERING CONFIGURATION
# =============================================================================

@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction"""
    n_neighbors: int = 5
    n_components: int = 10
    min_dist: float = 0.0
    metric: str = "cosine"
    random_state: int = 42
    n_jobs: int = 1
    low_memory: bool = True
    transform_seed: int = 42


@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN clustering"""
    min_cluster_size: Optional[int] = None  # Use algorithm default if None
    min_samples: Optional[int] = None  # Use algorithm default if None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = False
    approx_min_span_tree: bool = False
    gen_min_span_tree: bool = True


@dataclass
class VectorizerConfig:
    """Configuration for CountVectorizer"""
    ngram_range: Tuple[int, int] = (1, 3)
    min_df: int = 1
    max_df: float = 1.0
    max_features: Optional[int] = None
    use_language_stop_words: bool = True  # Use spacy stop words based on DEFAULT_LANGUAGE


@dataclass
class ClusterMergerConfig:
    """Configuration for cluster merging step"""
    model: str = DEFAULT_MODEL
    max_concurrent_requests: int = 5
    batch_size: int = 5
    similarity_threshold: float = 0.95
    max_retries: int = 3
    retry_delay: int = 2
    temperature: float = 0.3
    max_tokens: int = 4000
    language: str = DEFAULT_LANGUAGE
    verbose: bool = True


@dataclass
class ClusteringConfig:
    """Master configuration for clustering step (Step 5)"""
    # Embedding selection
    embedding_type: str = "description"  # Options: "description" or "code"
    
    # Sub-configurations for different models
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    merger: ClusterMergerConfig = field(default_factory=ClusterMergerConfig)
    
    # Quality and filtering settings
    enable_quality_metrics: bool = True
    filter_na_items: bool = True
    remap_cluster_ids: bool = True
    
    # General settings
    verbose: bool = True


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Global configuration instances
DEFAULT_LABELLER_CONFIG = LabellerConfig()

# Pipeline step configurations
DEFAULT_SPELLCHECK_CONFIG = SpellCheckConfig()
DEFAULT_QUALITY_FILTER_CONFIG = QualityFilterConfig()
DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()

