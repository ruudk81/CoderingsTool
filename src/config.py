import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

# File handling (only keep what's used)
ALLOWED_EXTENSIONS = ['.sav']

# =============================================================================
# HUNSPELL CONFIGURATION
# =============================================================================

current_dir = os.getcwd()
if os.path.basename(current_dir) == 'utils':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'hunspell'))
elif os.path.basename(current_dir) == 'modules':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'hunspell'))
elif os.path.basename(current_dir) == 'src':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', 'hunspell'))
elif os.path.basename(current_dir) == 'Coderingstool':
    hunspell_dir = os.path.abspath(os.path.join(current_dir, 'hunspell'))

HUNSPELL_PATH = os.path.join(hunspell_dir, "hunspell.exe")
DUTCH_DICT_PATH = os.path.join(hunspell_dir, "dict", "nl_NL")
ENGLISH_DICT_PATH = os.path.join(hunspell_dir, "dict", "en_GB")
DEFAULT_LANGUAGE = "Dutch"

# =============================================================================
# MODEL CONFIGURATION - CENTRALIZED
# =============================================================================

# LLM settings (core settings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4.1-mini"

# =============================================================================
# OPENAI RATE LIMITS (Official limits as of 2025)
# =============================================================================

@dataclass
class OpenAIRateLimits:
    """Official OpenAI API rate limits by model"""
    tokens_per_minute: int
    requests_per_minute: int
    tokens_per_day: int

# Rate limits for different models (Tier 4/5 paid accounts)
OPENAI_RATE_LIMITS = {
    "gpt-4.1": OpenAIRateLimits(
        tokens_per_minute=800_000,
        requests_per_minute=5_000,
        tokens_per_day=100_000_000
    ),
    "gpt-4.1-mini": OpenAIRateLimits(
        tokens_per_minute=4_000_000,
        requests_per_minute=5_000,
        tokens_per_day=40_000_000
    ),
    "gpt-4.1-nano": OpenAIRateLimits(
        tokens_per_minute=4_000_000,
        requests_per_minute=5_000,
        tokens_per_day=40_000_000
    ),
    "o4-mini": OpenAIRateLimits(
        tokens_per_minute=4_000_000,
        requests_per_minute=5_000,
        tokens_per_day=40_000_000
    ),
    "gpt-4o": OpenAIRateLimits(
        tokens_per_minute=800_000,
        requests_per_minute=5_000,
        tokens_per_day=100_000_000
    ),
    "gpt-4o-mini": OpenAIRateLimits(
        tokens_per_minute=4_000_000,
        requests_per_minute=5_000,
        tokens_per_day=40_000_000
    ),
    # Fallback for unknown models (conservative limits)
    "default": OpenAIRateLimits(
        tokens_per_minute=800_000,
        requests_per_minute=5_000,
        tokens_per_day=40_000_000
    )
}

def get_openai_rate_limits(model: str) -> OpenAIRateLimits:
    """Get rate limits for a specific OpenAI model"""
    # Handle model variations (e.g., gpt-4o-mini-2024-07-18)
    base_model = model.split('-')[0:2]  # Get base model name
    base_model_str = '-'.join(base_model)
    
    # Try exact match first, then base model, then default
    if model in OPENAI_RATE_LIMITS:
        return OPENAI_RATE_LIMITS[model]
    elif base_model_str in OPENAI_RATE_LIMITS:
        return OPENAI_RATE_LIMITS[base_model_str]
    else:
        return OPENAI_RATE_LIMITS["default"]

# =============================================================================
# EMBEDDING MODEL DIMENSIONS
# =============================================================================

# Embedding dimensions for different OpenAI embedding models
EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

def get_embedding_dimensions(model: str) -> int:
    """Get embedding dimensions for a specific OpenAI embedding model"""
    return EMBEDDING_MODEL_DIMENSIONS.get(model, 1536)  # Default to 1536 if unknown

@dataclass
class ModelConfig:
    """Centralized configuration for all models used throughout the pipeline"""
    
    # =============================================================================
    # STAGE-SPECIFIC MODELS
    # =============================================================================
    
    # Step 2: Text preprocessing models
    spell_check_model: str = DEFAULT_MODEL
    token_model: str = "gpt-4o-mini"
    tiktoken_spellChecker: str = "gpt-4o-mini"  # For token counting in spell checker      
    
    # Step 3: Quality filtering and segmentation models  
    quality_filter_model: str = DEFAULT_MODEL      
    segmentation_model: str = DEFAULT_MODEL        
    description_model: str = DEFAULT_MODEL         
    
    # Step 4: Embedding model
    embedding_model: str = "text-embedding-3-large"  
    
    # Step 6: Codebook generation
    speculative_codes_model: str = DEFAULT_MODEL  

    # Step 7: Codebook generation
    token_codebook_generation_model: str = "gpt-4o-mini"
    codebook_analysis_model: str = DEFAULT_MODEL     
    response_summary_model: str = DEFAULT_MODEL     
    match_and_recommend_model: str = DEFAULT_MODEL     
    validation_model: str = DEFAULT_MODEL      
    
    # Step 8: Hierarchical organisation of codebook
    hierarchical_organisation_model: str = DEFAULT_MODEL
    domain_clustering_model: str = DEFAULT_MODEL
    theme_synthesis_model: str = DEFAULT_MODEL
    
    # Step 9: Code assignment
    code_assignment_model: str = DEFAULT_MODEL

    # =============================================================================
    # GLOBAL PARAMETERS
    # =============================================================================
    
    seed: int = 42
    default_temperature: float = 0.0  # Default to deterministic
    default_max_tokens: int = 32000   # Default token limit
    
    # =============================================================================
    # STAGE-SPECIFIC TEMPERATURES
    # =============================================================================
    
    spell_check_temperature: float = 0.0
    quality_filter_temperature: float = 0.0
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def get_model_for_stage(self, stage: str) -> str:
        """Get the appropriate model for a pipeline stage"""
        stage_models = {
            'spell_check': self.spell_check_model,
            'quality_filter': self.quality_filter_model,
            'segmentation': self.segmentation_model,
            'description': self.description_model,
            'embedding': self.embedding_model,
            'speculative_codes': self.speculative_codes_model,
            'tiktoken': self.token_codebook_generation_model,
            'tiktoken_spellChecker': self.tiktoken_spellChecker,
            'codes_analysis': self.codebook_analysis_model,
            'cluster_analysis': self.response_summary_model,
            'recommend': self.match_and_recommend_model,
            'review': self.validation_model,
            'hierarchical_organisation': self.hierarchical_organisation_model,
            'domain_clustering': self.domain_clustering_model,
            'theme_synthesis': self.theme_synthesis_model,
            'code_assignment': self.code_assignment_model
        }
        return stage_models.get(stage, DEFAULT_MODEL)
    
    def get_temperature_for_stage(self, stage: str) -> float:
        """Get the appropriate temperature for a pipeline stage"""
        stage_temperatures = {
            'spell_check': self.spell_check_temperature,
            'quality_filter': self.quality_filter_temperature,
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
        "quality_filter": "003",
        "extracted_ideas": "004",         
        "embeddings": "005",     
        "initial_clusters": "006",
        "codebook_generation": "007",
        "theme_identification": "008",
        "code_assignment": "009",
        "export": "010"
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

# =============================================================================
# PREPROCESS CONFIGURATION
# =============================================================================

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

# =============================================================================
# SEGMENT CONFIGURATION
# =============================================================================

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
    # Model configuration - will be overridden by ModelConfig
    model: str = "gpt-4o-mini"  # Fallback model
    max_concurrent_requests: int = 5  # For API rate limiting


@dataclass
class SegmentationConfig:
    """Configuration for segmentation and description step"""
    max_tokens: int = 16000
    completion_reserve: int = 1000
    min_batch_size: int = 5  # Minimum responses per batch for efficiency
    max_batch_size: int = 20  # Maximum responses per batch for manageability
    target_token_utilization: float = 0.8  # Use 80% of available tokens per batch
    retry_delay: int = 2
    max_retries: int = 3
    spacy_batch_size: int = 32
    umap_n_jobs: int = 1
    max_code_examples: int = 5  # For verbose output
    max_sample_responses: int = 3  # For verbose output
    # Model configuration - will be overridden by ModelConfig
    model: str = "gpt-4o-mini"  # Fallback model
    temperature: float = 0.0  # Temperature for generation
    max_concurrent_requests: int = 8  # Optimized for better throughput while respecting rate limits

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation step"""
    batch_size: int = 100
    max_concurrent_requests: int = 5
    # Model configuration - will be overridden by ModelConfig
    embedding_model: str = "text-embedding-3-large"  # Fallback model
    max_sample_responses: int = 3  # For verbose output
    
    # Question-aware embedding configuration
    use_question_aware: bool = False  # Enable question-aware embeddings
    response_weight: float = 0.6  # Weight for response embeddings
    question_weight: float = 0.3  # Weight for question embeddings
    domain_anchor_weight: float = 0.1  # Weight for domain-relative positioning


# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction"""
    n_neighbors: int = 10  # Higher for better semantic relationships
    n_components: int = 5  # More dimensions to preserve semantic nuances
    min_dist: float = 0.0  # Slight separation for better cluster distinction
    metric: str = "cosine"  # Consistent with HDBSCAN for semantic similarity
    random_state: int = 42
    n_jobs: int = 1
    low_memory: bool = True
    transform_seed: int = 42


@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN clustering"""
    min_cluster_size: Optional[int] = None  # Smaller clusters for better semantic coherence
    min_samples: Optional[int] = None # Lower threshold for more selective clustering
    metric: str = "euclidean"  # Better for semantic embeddings than euclidean
    cluster_selection_method: str = "eom"
    prediction_data: bool = True
    approx_min_span_tree: bool = False
    gen_min_span_tree: bool = True


@dataclass
class VectorizerConfig:
    """Configuration for CountVectorizer"""
    ngram_range: Tuple[int, int] = (1, 1)
    min_df: int = 1
    max_df: float = 1.0
    max_features: Optional[int] = None
    use_language_stop_words: bool = True  # Use spacy stop words based on DEFAULT_LANGUAGE


# =============================================================================
# LABELLING CONFIGURATION
# =============================================================================

@dataclass
class LabellerConfig:
    """Configuration for hierarchical labelling"""
    # Model settings - will be overridden by ModelConfig
    model: str = "gpt-4o-mini"  # Fallback base model
    temperature: float = 0.0  # Lower for more consistent output
    max_tokens: int = 16000  # Increased for gpt-4o's higher capacity
    seed: int = 42  # For reproducibility
    api_key: Optional[str] = None  # Will use env var if not provided
    
    # Language and localization
    language: str = DEFAULT_LANGUAGE
    
    # Processing parameters
    top_k_representatives: int = 3  # Representative examples per cluster
    map_reduce_threshold: int = 100  # Use MapReduce if more clusters
    batch_size: int = 10  # Clusters per batch in MapReduce
    assignment_threshold: float = 0.5  # Minimum probability for assignment (lowered for better coverage)
    
    # Retry and concurrency settings
    max_retries: int = 3
    concurrent_requests: int = 10  # Increased for better performance
    retry_delay: int = 2  # Seconds between retries
    
    # Confidence scoring settings
    use_confidence_scoring: bool = True  # Enable confidence-based assignment
    confidence_threshold: float = 0.49  # Minimum confidence for assignment
    confidence_batch_size: int = 10  # Clusters to process per confidence scoring batch

# =============================================================================
# CODE ASSIGNMENT CONFIGURATION
# =============================================================================

@dataclass
class CodeAssignmentConfig:
    """Configuration for code assignment step"""
    batch_size: int = 20  # Increased from 10 for better throughput
    temperature: float = 0.0
    max_tokens: int = 4000
    retries: int = 3
    retry_delay: int = 2
    max_concurrent_requests: int = 20  # Increased from 5 (though semaphore removed)
    top_k_similar_codes: int = 5  # Number of most similar codes to present
    min_confidence_threshold: float = 0.3  # Minimum confidence for valid assignment
    # Model configuration - will be overridden by ModelConfig
    model: str = "gpt-4o-mini"  # Fallback model
    max_assignment_examples: int = 3  # For verbose output

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

@dataclass
class ExportConfig:
    """Configuration for results export functionality"""
    
    # Output directory settings
    export_dir: Optional[str] = None  # Will use data dir if None
    create_subdirs: bool = True  # Create subdirectories by survey variable
    
    # File naming patterns
    spss_suffix: str = "_codes"  # Suffix for SPSS file with codes
    excel_suffix: str = "_results"  # Suffix for Excel results file
    
    # Excel export settings
    enable_codebook_tab: bool = True
    enable_dendrogram_tab: bool = True
    enable_frequency_tab: bool = True
    enable_wordcloud_tab: bool = True
    
    # Visualization settings
    chart_width: int = 12
    chart_height: int = 8
    wordcloud_width: int = 800
    wordcloud_height: int = 600
    max_wordcloud_words: int = 100
    
    # Data formatting
    include_descriptions: bool = True
    include_frequencies: bool = True
    include_percentages: bool = True
    
    # Quality settings
    min_frequency_for_chart: int = 1  # Minimum frequency to include in charts
    max_categories_in_chart: int = 50  # Maximum categories to show in frequency charts
    
    # Output verbosity
    verbose: bool = True
    
    def get_export_dir(self, base_data_dir: str) -> str:
        """Get the export directory path"""
        if self.export_dir:
            return self.export_dir
        return os.path.join(base_data_dir, "exports")

# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Central model configuration - configure all models here
DEFAULT_MODEL_CONFIG = ModelConfig()

# Step-specific configurations
DEFAULT_SPELLCHECK_CONFIG = SpellCheckConfig()
DEFAULT_QUALITY_FILTER_CONFIG = QualityFilterConfig()
DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_LABELLER_CONFIG = LabellerConfig()
DEFAULT_CODE_ASSIGNMENT_CONFIG = CodeAssignmentConfig()
DEFAULT_EXPORT_CONFIG = ExportConfig()


