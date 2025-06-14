import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
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
DEFAULT_MODEL = "gpt-4o-mini"

@dataclass
class ModelConfig:
    """Centralized configuration for all models used throughout the pipeline"""
    
    # =============================================================================
    # STAGE-SPECIFIC MODELS
    # =============================================================================
    
    # Step 2: Text preprocessing models
    spell_check_model: str = "gpt-4o-mini"           # Fast model for spell checking
    
    # Step 3: Quality filtering and segmentation models  
    quality_filter_model: str = "gpt-4o-mini"        # Fast model for quality filtering
    segmentation_model: str = "gpt-4o-mini"          # Fast model for response segmentation
    description_model: str = "gpt-4o-mini"           # Fast model for description generation
    
    # Step 4: Embedding model
    embedding_model: str = "text-embedding-3-large"  # Embedding model
    
    # Step 5: Clustering models
    cluster_merge_model: str = "gpt-4o-mini"         # Fast model for cluster merging
    
    # Step 6: Hierarchical labelling models (6 phases)
    labelling_base_model: str = "gpt-4.1-mini"        # Base model for most labelling phases
    phase1_descriptive_model: str = "gpt-4.1-mini"    # Phase 1: Descriptive coding
    phase2_merger_model: str = "gpt-4.1-mini"         # Phase 2: Label merger
    phase2_5_confidence_model: str = "gpt-4o-mini"    # Phase 2.5: Confidence scoring
    phase3_themes_model: str = "gpt-4o"              # Phase 3: Extract themes (premium model)
    phase4_codebook_model: str = "gpt-4o-mini"       # Phase 4: Create codebook
    phase5_refinement_model: str = "gpt-4o"     # Phase 5: Label refinement
    phase6_assignment_model: str = "gpt-4o-mini"     # Phase 6: Assignment
    
    # =============================================================================
    # GLOBAL PARAMETERS
    # =============================================================================
    
    seed: int = 42
    default_temperature: float = 0.0  # Default to deterministic
    default_max_tokens: int = 16000   # Default token limit
    
    # =============================================================================
    # STAGE-SPECIFIC TEMPERATURES
    # =============================================================================
    
    spell_check_temperature: float = 0.0
    quality_filter_temperature: float = 0.0
    segmentation_temperature: float = 0.0
    description_temperature: float = 0.0
    cluster_merge_temperature: float = 0.0   
    labelling_temperature: float = 0.0
    phase3_themes_temperature: float = 0.0  # Keep deterministic even for premium model
    
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
            'cluster_merge': self.cluster_merge_model,
            'labelling': self.labelling_base_model,
        }
        return stage_models.get(stage, DEFAULT_MODEL)
    
    def get_model_for_phase(self, phase: str) -> str:
        """Get the appropriate model for a specific labelling phase"""
        phase_models = {
            'phase1_descriptive': self.phase1_descriptive_model,
            'phase2_merger': self.phase2_merger_model,
            'phase2_5_confidence': self.phase2_5_confidence_model,
            'phase3_themes': self.phase3_themes_model,
            'phase4_codebook': self.phase4_codebook_model,
            'phase5_refinement': self.phase5_refinement_model,
            'phase6_assignment': self.phase6_assignment_model,
        }
        return phase_models.get(phase, self.labelling_base_model)
    
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
    
    def get_temperature_for_phase(self, phase: str) -> float:
        """Get the appropriate temperature for a specific labelling phase"""
        phase_temperatures = {
            'phase3_themes': self.phase3_themes_temperature,
        }
        return phase_temperatures.get(phase, self.labelling_temperature)

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
        "segmented_descriptions": "004",
        "initial_clusters": "005",
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
class TfidfConfig:
    """Configuration for TF-IDF component of ensemble embeddings"""
    max_features: int = 1000  # Number of TF-IDF features
    ngram_range: Tuple[int, int] = (1, 2)  # Unigrams and bigrams
    min_df: int = 2  # Minimum document frequency
    max_df: float = 0.95  # Maximum document frequency (remove too common terms)
    use_idf: bool = True  # Use IDF weighting
    sublinear_tf: bool = True  # Use log(TF) instead of raw TF
    # POS tags to keep (None = no filtering)
    allowed_pos_tags: Optional[List[str]] = field(default_factory=lambda: ['NOUN', 'PROPN', 'ADJ'])
    # Ensure TF-IDF vectors are normalized
    norm: str = 'l2'

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation step"""
    batch_size: int = 100
    max_concurrent_requests: int = 5
    # Model configuration - will be overridden by ModelConfig
    embedding_model: str = "text-embedding-3-large"  # Fallback model
    max_sample_responses: int = 3  # For verbose output
    
    # Ensemble embedding configuration
    use_ensemble: bool = False  # Enable ensemble embeddings
    openai_weight: float = 0.7  # Weight for OpenAI embeddings
    tfidf_weight: float = 0.3  # Weight for TF-IDF embeddings
    ensemble_combination: str = "weighted_concat"  # Options: "weighted_concat", "weighted_average"
    reduce_dimensions: bool = False  # Let UMAP handle dimensionality reduction
    target_dimensions: int = 768  # Target dimensions if reducing
    
    # TF-IDF configuration
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)


# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction"""
    n_neighbors: int = 15 #5
    n_components: int = 3 #10
    min_dist: float = 0.0
    metric: str = "euclidean" # vs "cosine"
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
    prediction_data: bool = True
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
    # Model configuration - will be overridden by ModelConfig
    model: str = DEFAULT_MODEL  # Fallback model
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
class NoiseRescueConfig:
    """Configuration for noise point rescue using cosine similarity or HDBSCAN methods"""
    enabled: bool = True
    rescue_threshold: float = 0.3  # For HDBSCAN methods
    max_rescue_attempts: int = 1000
    
    # Cosine similarity rescue parameters
    use_cosine_rescue: bool = False
    cosine_similarity_threshold: float = 0.7


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
    noise_rescue: NoiseRescueConfig = field(default_factory=NoiseRescueConfig)
    
    # Quality and filtering settings
    enable_quality_metrics: bool = True
    filter_na_items: bool = True
    remap_cluster_ids: bool = True
    
    # General settings
    verbose: bool = True


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
DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()
DEFAULT_LABELLER_CONFIG = LabellerConfig()
DEFAULT_EXPORT_CONFIG = ExportConfig()


