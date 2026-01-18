import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

# Load .env file if it exists (simple loader, no dependencies)
def _load_dotenv():
    """Load environment variables from .env file in project root."""
    env_paths = [
        Path(__file__).parent.parent / '.env',  # src/../.env
        Path.cwd() / '.env',
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and not os.environ.get(key):
                            os.environ[key] = value
            break

_load_dotenv()

# File handling (only keep what's used)
ALLOWED_EXTENSIONS = ['.sav']

# =============================================================================
# HUNSPELL CONFIGURATION (Cross-platform)
# =============================================================================
import platform
import shutil

def _get_hunspell_paths() -> Tuple[str, str]:
    """
    Get Hunspell executable and dictionary directory paths.
    Supports Windows (bundled .exe) and macOS/Linux (system install via brew/apt).

    Returns:
        Tuple of (hunspell_executable_path, hunspell_dict_directory)
    """
    # Determine project hunspell directory (for dictionaries and Windows exe)
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'utils':
        hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'hunspell'))
    elif os.path.basename(current_dir) == 'modules':
        hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'hunspell'))
    elif os.path.basename(current_dir) == 'src':
        hunspell_dir = os.path.abspath(os.path.join(current_dir, '..', 'hunspell'))
    elif os.path.basename(current_dir) == 'Coderingstool':
        hunspell_dir = os.path.abspath(os.path.join(current_dir, 'hunspell'))
    else:
        hunspell_dir = os.path.abspath(os.path.join(current_dir, 'hunspell'))

    system = platform.system()

    if system == "Windows":
        # Use bundled Windows executable
        hunspell_exe = os.path.join(hunspell_dir, "hunspell.exe")
    else:
        # macOS or Linux: try system-installed hunspell
        # Check common locations in order of preference
        system_hunspell = shutil.which("hunspell")

        if system_hunspell:
            hunspell_exe = system_hunspell
        elif system == "Darwin":  # macOS
            # Homebrew paths (Apple Silicon and Intel)
            brew_paths = [
                "/opt/homebrew/bin/hunspell",  # Apple Silicon
                "/usr/local/bin/hunspell",      # Intel Mac
            ]
            hunspell_exe = next((p for p in brew_paths if os.path.exists(p)), None)
            if not hunspell_exe:
                # Fallback to bundled (won't work but provides clear error)
                hunspell_exe = os.path.join(hunspell_dir, "hunspell")
        else:  # Linux
            linux_paths = [
                "/usr/bin/hunspell",
                "/usr/local/bin/hunspell",
            ]
            hunspell_exe = next((p for p in linux_paths if os.path.exists(p)), None)
            if not hunspell_exe:
                hunspell_exe = os.path.join(hunspell_dir, "hunspell")

    return hunspell_exe, hunspell_dir

# Initialize cross-platform paths
HUNSPELL_PATH, _hunspell_dir = _get_hunspell_paths()
DUTCH_DICT_PATH = os.path.join(_hunspell_dir, "dict", "nl_NL")
ENGLISH_DICT_PATH = os.path.join(_hunspell_dir, "dict", "en_GB")
DEFAULT_LANGUAGE = "Dutch"

# Language-specific labels for miscellaneous/catch-all code
MISCELLANEOUS_CODE_LABELS = {
    "Dutch": "Overig",
    "English": "Other",
    "German": "Sonstiges",
    "French": "Autre",
    "Spanish": "Otro",
}

# Language-specific labels for general/theme-level assignments
GENERAL_CODE_LABELS = {
    "Dutch": "algemeen",
    "English": "overall",
    "German": "allgemein",
    "French": "général",
    "Spanish": "general",
}

# =============================================================================
# MODEL CONFIGURATION - CENTRALIZED
# =============================================================================

# =============================================================================
# API PROVIDER CONFIGURATION
# =============================================================================
# Toggle between "openai" and "azure" here
API_PROVIDER = "azure"  # Options: "openai" or "azure"

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING", "text-embedding-3-large")
# Deployment for codeGenerator (uses chat completion without reasoning)
# Defaults to same deployment as main model if not specified
AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER", AZURE_OPENAI_DEPLOYMENT_NAME)

# Azure ARM access (for dynamic limit fetching - optional)
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")

# =============================================================================
# MODEL LIMITS (context window + max output tokens)
# Adjust based on your subscription tier if needed
# =============================================================================
OPENAI_MODEL_LIMITS = {
    # GPT-4.1 family - 1M context window
    "gpt-4.1": {"context_window": 1_000_000, "max_output": 32_000},
    "gpt-4.1-mini": {"context_window": 1_000_000, "max_output": 32_000},
    "gpt-4.1-nano": {"context_window": 1_000_000, "max_output": 32_000},
    # GPT-5 family - 400K total (272K input + 128K output)
    "gpt-5": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5.1": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5.2": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5-mini": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5-nano": {"context_window": 128_000, "max_output": 32_000},
    "gpt-5-chat-latest": {"context_window": 272_000, "max_output": 128_000},
    # GPT-4o family (legacy)
    "gpt-4o": {"context_window": 128_000, "max_output": 16_000},
    "gpt-4o-mini": {"context_window": 128_000, "max_output": 16_000},
    # Embeddings
    "text-embedding-3-large": {"context_window": 8_191, "max_output": 0},
    "text-embedding-3-small": {"context_window": 8_191, "max_output": 0},
}

# Default models (used for both providers)
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# =============================================================================
# CLIENT FACTORY FUNCTIONS
# =============================================================================
# These create the appropriate client based on API_PROVIDER setting

def create_instructor_client(model: str, async_mode: bool = True) -> Any:
    """
    Create instructor client based on API_PROVIDER setting.

    Args:
        model: Model name (e.g., 'gpt-4.1-mini')
        async_mode: Whether to create async client (default True)

    Returns:
        Instructor-wrapped client for structured outputs
    """
    import instructor
    from openai import OpenAI, AsyncOpenAI

    if API_PROVIDER == "azure":
        # Azure v1 API: use standard OpenAI client with custom base_url
        # This gives access to the Responses API (responses.create)
        azure_base_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/"
        base_client = AsyncOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            base_url=azure_base_url
        ) if async_mode else OpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            base_url=azure_base_url
        )
        # Use RESPONSES_TOOLS mode since v1 API supports Responses API
        return instructor.from_openai(base_client, mode=instructor.Mode.RESPONSES_TOOLS)
    else:
        # OpenAI uses the Responses API
        return instructor.from_provider(
            f"openai/{model}",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=async_mode,
            api_key=OPENAI_API_KEY
        )


def create_embedding_client(async_mode: bool = True) -> Any:
    """
    Create embedding client based on API_PROVIDER setting.

    Args:
        async_mode: Whether to create async client (default True)

    Returns:
        OpenAI client for embeddings (with custom base_url for Azure)
    """
    from openai import OpenAI, AsyncOpenAI

    if API_PROVIDER == "azure":
        # Azure v1 API: use standard OpenAI client with custom base_url
        azure_base_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/"
        if async_mode:
            return AsyncOpenAI(api_key=AZURE_OPENAI_API_KEY, base_url=azure_base_url)
        return OpenAI(api_key=AZURE_OPENAI_API_KEY, base_url=azure_base_url)
    else:
        if async_mode:
            return AsyncOpenAI(api_key=OPENAI_API_KEY)
        return OpenAI(api_key=OPENAI_API_KEY)


def get_model_for_api(model: str) -> str:
    """
    Get the appropriate model/deployment name for the current API provider.

    For Azure, maps model names to deployment names.
    For OpenAI, returns the model name as-is.
    """
    if API_PROVIDER == "azure":
        # Azure uses deployment names - map common models
        # For now, use the configured deployment name
        return AZURE_OPENAI_DEPLOYMENT_NAME
    return model


def get_embedding_model_for_api() -> str:
    """Get the appropriate embedding model/deployment for the current API provider."""
    if API_PROVIDER == "azure":
        return AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING
    return DEFAULT_EMBEDDING_MODEL


# =============================================================================
# RATE LIMIT FALLBACKS (Used when API headers are unavailable)
# =============================================================================
# Rate limits are now fetched dynamically from API response headers.
# These fallback values are only used if headers are not present.
# Set via environment variables or use conservative defaults.

FALLBACK_TPM = int(os.getenv("FALLBACK_TPM", "100000"))  # Conservative: 100K tokens/min
FALLBACK_RPM = int(os.getenv("FALLBACK_RPM", "100"))     # Conservative: 100 requests/min

# =============================================================================
# EMBEDDING MODEL DIMENSIONS
# =============================================================================

# Embedding dimensions for different OpenAI embedding models
EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    # Gemini
    "text-embedding-004": 768,
    "models/text-embedding-004": 768,   
    # Gemini large
    "gemini-embedding-001": 3072,              
    "models/gemini-embedding-001": 3072,       
    }

def get_embedding_dimensions(model: str) -> int:
    """Get embedding dimensions for a specific OpenAI embedding model"""
    return EMBEDDING_MODEL_DIMENSIONS.get(model)   

@dataclass
class ModelConfig:
    """Centralized configuration for all models used throughout the pipeline"""
    
    # =============================================================================
    # MODEL TYPE MAPPING
    # =============================================================================
    MODEL_TYPES = {
        # GPT-4 family (chat models)
        "gpt-4": "chat",
        "gpt-4o": "chat",
        "gpt-4o-mini": "chat",
        "gpt-4.1": "chat",
        "gpt-4.1-mini": "chat",
        "gpt-4.1-nano": "chat",
        "gpt-5-chat-latest": "chat",
        
        # GPT-5 family (reasoning models)
        "gpt-5": "reasoning",
        "gpt-5-mini": "reasoning",
        "gpt-5-nano": "reasoning",

    }
    
    # =============================================================================
    # STAGE-SPECIFIC MODELS
    # =============================================================================
    
    # Text preprocessing models
    spell_check_model: str = DEFAULT_MODEL
    token_model: str = "gpt-4o-mini"
    tiktoken_spellChecker: str = "gpt-4o-mini"  # 
    
    # Quality filtering and segmentation models  
    quality_filter_model: str = DEFAULT_MODEL      
    segmentation_model: str = DEFAULT_MODEL        
    description_model: str = DEFAULT_MODEL         
    
    # Embedding model
    embedding_model: str = "text-embedding-3-large"
    #embedding_model: str = "gemini-embedding-001"
   
    speculative_codes_model: str = DEFAULT_MODEL  

    # Codebook generation
    token_codebook_generation_model: str = "gpt-4o-mini"
    thematic_summary_model: str = "gpt-5-chat-latest"      
    candidate_selection_model: str = "gpt-5-chat-latest"           
    code_generation_model: str ="gpt-5-chat-latest"               
    validation_model: str = "gpt-5-chat-latest"
    
    # Codebook refinement
    codebook_refinement_model: str = "gpt-5-mini" 

    # # theme identification
    # thematic_organizer_model : str = "gpt-5-mini"   
    # theme_extraction_reasoning_effort: str = "low"       
    # theme_extraction_text_verbosity: str = "medium"      

    # Code assignment
    code_assignment_model: str = "gpt-5-chat-latest"
  

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
    refinement_temperature: float = 0.2 
    
    # =============================================================================
    # GPT-5 SPECIFIC PARAMETERS - STAGE-SPECIFIC
    # =============================================================================
    
    # Theme Extraction (Step 1 - Cluster Summary)
    theme_extraction_reasoning_effort: str = "minimal"       
    theme_extraction_text_verbosity: str = "medium"      

    # Candidate Selection (Step 2 - Code Selection)  
    candidate_selection_reasoning_effort: str = "minimal"   
    candidate_selection_text_verbosity: str = "medium"        

    # Code Generation (Step 3 - Code Recommendation)
    code_generation_reasoning_effort: str = "minimal"      
    code_generation_text_verbosity: str = "medium"     

    # Validation (Step 4 - Code Validation)
    validation_reasoning_effort: str = "minimal"         
    validation_text_verbosity: str = "medium"         
    
    # Codebook Refinement  
    refinement_reasoning_effort: str = "minimal"
    refinement_text_verbosity: str = "medium"

    # Keep global defaults as fallback
    gpt5_reasoning_effort: str = "minimal"  # Global default
    gpt5_text_verbosity: str = "medium"     # Global default
    
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
            'theme_extraction': self.thematic_summary_model,
            'candidate_selection': self.candidate_selection_model,
            'code_recommendation': self.code_generation_model,
            'recommendation_validation': self.validation_model,
            'codebook_refinement': self.codebook_refinement_model,
            'code_assignment': self.code_assignment_model
            }
        return stage_models.get(stage, DEFAULT_MODEL)
    
    def get_temperature_for_stage(self, stage: str) -> float:
        
        stage_temperatures = {
            'spell_check': self.spell_check_temperature,
            'quality_filter': self.quality_filter_temperature,
            'refinement': self.refinement_temperature}
    
        if stage in stage_temperatures:
            return stage_temperatures[stage]
    
        model_name = self.get_model_for_stage(stage)
        model_type = self.MODEL_TYPES.get(model_name, "chat")
    
        if model_type == "chat":
            return 0.0
        elif model_type == "reasoning":
            return 1.0
        else:
            return self.default_temperature

    
    def get_reasoning_effort_for_stage(self, stage: str) -> str:
        """Get GPT-5 reasoning effort for specific stage"""
        stage_efforts = {
            'theme_extraction': self.theme_extraction_reasoning_effort,
            'candidate_selection': self.candidate_selection_reasoning_effort,
            'code_recommendation': self.code_generation_reasoning_effort,
            'recommendation_validation': self.validation_reasoning_effort,
            'codebook_refinement': self.refinement_reasoning_effort
        }
        return stage_efforts.get(stage, self.gpt5_reasoning_effort)

    def get_text_verbosity_for_stage(self, stage: str) -> str:
        """Get GPT-5 text verbosity for specific stage"""
        stage_verbosities = {
            'theme_extraction': self.theme_extraction_text_verbosity,
            'candidate_selection': self.candidate_selection_text_verbosity,
            'code_recommendation': self.code_generation_text_verbosity,
            'recommendation_validation': self.validation_text_verbosity,
            'codebook_refinement': self.refinement_text_verbosity
        }
        return stage_verbosities.get(stage, self.gpt5_text_verbosity)
    
    def get_langchain_config_for_stage(self, stage: str) -> Dict[str, Any]:
        """Get complete LangChain configuration for a stage."""
        model_name = self.get_model_for_stage(stage)
        model_type = self.MODEL_TYPES.get(model_name, "chat")
    
        temperature = (
            0.0 if model_type == "chat"
            else self.refinement_temperature if model_name == "gpt-5-chat-latest"
            else 1.0
        )
    
        return {
            "api_key": OPENAI_API_KEY,
            "model": model_name,
            "temperature": temperature,
        }
    
  
    
# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

@dataclass
class ProcessingConfig:
    """Global processing parameters affecting cache validity and performance"""

    # Rate limiting
    rate_limit_headroom: float = 0.9  # Use 90% of API limits for safety

    # Concurrency bounds
    concurrency_cap_default: int = 300
    concurrency_cap_permissive: int = 10000
    concurrency_min_default: int = 100
    concurrency_min_permissive: int = 0
    concurrency_min_conservative: int = 10

    # Adaptive timeout bounds
    adaptive_timeout_min_seconds: float = 15.0
    adaptive_timeout_max_seconds: float = 60.0
    adaptive_timeout_margin: float = 1.5

    # Latency tracking
    latency_tracker_ema_alpha: float = 0.1
    latency_tracker_samples_window: int = 100  # Keep last N samples for percentiles

    # Bootstrap measurement
    bootstrap_probe_count: int = 3

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
        "expanded_clusters": "006",
        "codebook_generation": "007",
        "codebook_refinement": "008",
        "code_assignment": "009",
        "export": "010"
        })
    
    # Cache validity settings
    max_cache_age_days: int = 30
    auto_cleanup: bool = True  # Automatically cleanup old cache on initialization

    # File handling settings
    use_atomic_writes: bool = True
    
    # Performance settings
    batch_size: int = 1000
    
    # Logging settings
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
    spacy_batch_size: int = 64  # Increased for better performance
    repeated_char_threshold: int = 5  # Characters repeated 5+ times
    max_correction_examples: int = 10  # For verbose output
    seed: int = 42
    context_chars: int = 20  # Characters of context for spell checking
    max_concurrent_requests: int = 5  # For API rate limiting
    
    # Performance optimization settings
    max_words_to_check: int = 100000  # Skip spell checking if more words than this
    enable_word_frequency_cache: bool = True  # Cache common words
    progress_report_interval: int = 10000  # Report progress every N words
    max_unique_oov_words: int = 5000  # Limit unique OOV words to process
    enable_early_termination: bool = True  # Allow early termination for large datasets
    
    # Aggressive parallel processing settings for suggestion generation
    max_concurrent_suggestion_chunks: int = 20  # Number of concurrent chunks for OOV processing (increased for better parallelism)
    max_words_per_chunk: int = 1200  # Maximum words per chunk
    enable_adaptive_chunking: bool = True  # Dynamic chunk sizing based on performance
    chunk_progress_reporting: bool = True  # Show progress per chunk
    suggestion_processing_semaphore_limit: int = 100  # Limit concurrent Hunspell processes (increased for aggressive parallelism)
    
    # Timeout configuration for API calls
    minimum_timeout_seconds: float = 15.0  # Minimum timeout for API calls (safety net)
    maximum_timeout_seconds: float = 60.0  # Maximum timeout for API calls (prevents excessive waits)
    
    # New optimization parameters
    hunspell_concurrent_sessions: int = 20  # Number of concurrent Hunspell sessions for OOV detection (increased from 5)
    hunspell_batch_size: int = 1000  # Words per Hunspell batch (reduced from 1000 for better distribution)
    enable_streaming_oov_detection: bool = True  # Enable producer-consumer pattern for OOV detection
    oov_detection_queue_size: int = 10000  # Size of queue for streaming OOV detection
    
    # Rate limiting optimization parameters
    rate_limit_safety_factor: float = 0.95  # Use 95% of theoretical maximum (was 0.85)
    rate_limit_utilization: float = 0.98  # Use 98% of actual rate limits (was 0.95)
    concurrent_burst_multiplier: float = 3.0  # Burst capacity multiplier (was 2.0)
    
    # Suggestion validation parameters
    enable_suggestion_pre_validation: bool = True  # Pre-validate suggestions before LLM calls
    disable_pre_validation_above_oov_words: int = 2000  # Auto-disable pre-validation for very large datasets
    enable_suggestion_caching: bool = True  # Cache validated suggestions
    
    # Performance optimization parameters
    hunspell_pool_size: int = 20  # Number of persistent Hunspell processes in pool
    ultra_batch_threshold: int = 1000  # Use ultra-optimized batch processing above this many OOV words
    ultra_batch_size: int = 10000  # Batch size for ultra-optimized processing (increased for better performance)

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
    model: str = DEFAULT_MODEL  # Fallback model
    max_concurrent_requests: int = 5  # For API rate limiting
    # Timeout configuration for API calls
    minimum_timeout_seconds: float = 15.0  # Minimum timeout for API calls (safety net)
    maximum_timeout_seconds: float = 60.0  # Maximum timeout for API calls (prevents excessive waits)
   

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
    # Timeout configuration for API calls
    minimum_timeout_seconds: float = 15.0  # Minimum timeout for API calls (safety net)
    maximum_timeout_seconds: float = 60.0  # Maximum timeout for API calls (prevents excessive waits)

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation step"""
    batch_size: int = 100
    max_concurrent_requests: int = 5
    embedding_model: str = "text-embedding-3-large"  # Fallback model
    max_sample_responses: int = 3  # For verbose output
    
    # Provider-specific optimizations
    gemini_batch_size: int = 20  # Smaller batches for Gemini (individual API calls)
    gemini_max_concurrent: int = 10  # Optimized concurrency for Gemini - works well in practice
    openai_batch_size: int = 100  # Large batches for OpenAI (true batch API)
    openai_max_concurrent: int = 5  # OpenAI handles higher concurrency
    
    # Question-aware embedding configuration
    use_question_aware: bool = False  # Enable question-aware embeddings
    response_weight: float = 0.6  # Weight for response embeddings
    question_weight: float = 0.3  # Weight for question embeddings
    domain_anchor_weight: float = 0.1  # Weight for domain-relative positioning


# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

@dataclass
class ClusteringConfig:
    """Configuration for the complete clustering pipeline"""

    # PCA configuration
    pca_components: int = .99   # keep 99% of variance
    pca_random_state: int = 42  # random state for re-calc

    # Metrics
    enable_dbcv       = True
    enable_meanp      = True
    centroid_distance = True

    # Calc Metrics
    CLUSTER_METRIC = "euclidean"
    DBCV_D = 1  # safe for DBCV (avoid overflow)
    similarity_analysis_thresholds: list = field(default_factory=lambda: [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])     # Similarity analysis thresholds
    default_merge_threshold: float = 0.95     # Default merge threshold for similarity-based merging
    grid_search_max_workers: Optional[int] = None  # None=auto, -1=all cores
    grid_search_timeout_seconds: float = 300.0
    ctfidf_top_k: int = 15
    ctfidf_min_df: int = 2
    ctfidf_ngram_range: Tuple[int,int] = (1,2)

    # Post-clustering merge configuration
    merge_similar_clusters: bool = True  # Enable automatic merging of similar clusters
    merge_centroid_threshold: float = 0.95  # Centroid similarity threshold for candidate screening
    merge_pairwise_threshold: float = 0.98  # Pairwise similarity threshold for merge decision. .98 = "Essentially duplicates or rephrasings"
    merge_quantiles: Tuple[float, float, float] = (0.25, 0.50, 0.75)  # Quantiles for similarity evaluation

    # Noise assessment configuration
    noise_assignability_threshold: float = 0.95  # Similarity threshold for classifying noise as soft (assignable) vs hard (true outliers)

    # Noise reclustering configuration (two-pass clustering)
    enable_noise_reclustering: bool = True  # Enable second clustering pass on noise points
    noise_parameter_strategy: str = "adaptive"  # Parameter strategy: "adaptive", "aggressive", "fixed"
    noise_min_cluster_size: int = 3  # Minimum points for viable noise-derived cluster
    noise_min_total_points: int = 10  # Skip noise reclustering if fewer noise points
    noise_cluster_cohesion_threshold: float = 0.70  # Internal similarity threshold for quality filtering
    noise_min_clusters: int = 1  # Minimum viable clusters to accept reclustering result

    # Parameter strategy settings
    noise_mcs_divisor: int = 3  # For "aggressive": main_mcs // divisor
    noise_ms_divisor: int = 2  # For "aggressive": main_ms // divisor
    noise_fixed_mcs: int = 5  # For "fixed": fixed min_cluster_size
    noise_fixed_ms: int = 3  # For "fixed": fixed min_samples

@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction"""
    n_neighbors: int = 10  # default is 15, but 10 provides more detail, could be a better sweet spot in compination with clustering method "leaf"
    n_components: int = 10    # More dimensions to preserve semantic nuances
    min_dist: float = 0.1  # Slight separation for better cluster distinction
    metric: str = "cosine"  # Consistent with HDBSCAN for semantic similarity
    random_state: int = 42
    n_jobs: int = 1
    low_memory: bool = True
    transform_seed: int = 42
    
    # Parallel processing configuration
    n_epochs = 200
    use_parallel_umap: bool = False  # False = reproducible (single-threaded), True = faster (parallel)
    parallel_jobs: int = -1  # Number of cores to use when parallel enabled (-1 = all cores)


@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN clustering"""
    min_cluster_size: Optional[int] = 5  # Smaller clusters for better semantic coherence
    min_samples: Optional[int] = None # if none, fallback is min_cluster_size
    cluster_selection_epsilon: Optional[float] = 0.0
    alpha: Optional[float] = 1.0  # Default stability weighting as requested
    metric: str = "euclidean"  # Better for semantic embeddings than euclidean
    cluster_selection_method: str = "leaf"  #good for semantic purity & granularity; eom good for broad themes
    prediction_data: bool = True
    approx_min_span_tree: bool = False
    gen_min_span_tree: bool = True
    
    # Cluster merging configuration
    merge_similar_clusters: bool = True
    merge_similarity_threshold: float = 0.95  # Cosine similarity threshold for merging


@dataclass
class VectorizerConfig:
    """Configuration for CountVectorizer"""
    ngram_range: Tuple[int, int] = (1, 1)
    min_df: int = 1
    max_df: float = 1.0
    max_features: Optional[int] = None
    use_language_stop_words: bool = True  # Use spacy stop words based on DEFAULT_LANGUAGE


# =============================================================================
# CODEDESIGNER CONFIGURATION
# =============================================================================

@dataclass
class CodeDesignerConfig:
    """Configuration for the new CodeGenerator system"""
    
    # Model configuration
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 4000
    seed: Optional[int] = 42
    
    # Theme-based similarity batching
    similarity_threshold: float = 0.7  # Cosine similarity threshold for dissimilarity batching
    max_sub_batch_size: int = 10  # Maximum clusters per sub-batch
    
    # Rate limiting and performance
    batch_size: int = 20  # Base batch size for API calls
    max_concurrent_requests: int = 15  # Maximum concurrent API requests
    async_concurrency_limit: int = 16  # Async concurrency limit for codeGenerator
    enable_aggressive_parallelism: bool = True  # Enable concurrent processing within batches
    
    # Processing strategy
    enable_sequential_batch_processing: bool = True  # Process dissimilarity batches sequentially
    enable_sub_batch_processing: bool = True  # Split large batches into sub-batches
    
    # Monitoring and reporting
    enable_similarity_distribution_analysis: bool = True  # Report similarity statistics
    enable_batch_analytics: bool = True  # Report batch formation statistics
    enable_performance_monitoring: bool = True  # Monitor processing performance
    
    # Idea sampling settings
    max_ideas_per_cluster: int = 30  # Maximum ideas to include per cluster for LLM processing
    
    # SharedCodebook settings
    enable_version_tracking: bool = True  # Track codebook versions
    enable_embedding_cache: bool = True  # Cache code embeddings per version
    max_cached_versions: int = 5  # Maximum cached codebook versions
    
    # Modification leak recovery settings
    enable_concurrent_leak_recovery: bool = True  # Use concurrent batch processing for modification leak recovery
    modification_leak_batch_size: int = 10  # Batch size for concurrent leak recovery



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
    top_k_similar_codes: int = 10  # Number of most similar codes to present
    min_confidence_threshold: float = 0.3  # Minimum confidence for valid assignment
    miscellaneous_confidence_threshold: float = 0.6  # Threshold below which to assign miscellaneous code
    # Model configuration - will be overridden by ModelConfig
    model: str = DEFAULT_MODEL  # Fallback model
    max_assignment_examples: int = 3  # For verbose output
    # Timeout configuration for API calls
    minimum_timeout_seconds: float = 15.0  # Minimum timeout for API calls (safety net)
    maximum_timeout_seconds: float = 60.0  # Maximum timeout for API calls (prevents excessive waits)
  

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

# Processing configuration
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()

# Step-specific configurations
DEFAULT_SPELLCHECK_CONFIG = SpellCheckConfig()
DEFAULT_QUALITY_FILTER_CONFIG = QualityFilterConfig()
DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_UMAP_CONFIG = UMAPConfig()
DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()
DEFAULT_HDBSCAN_CONFIG = HDBSCANConfig()
DEFAULT_LABELLER_CONFIG = LabellerConfig()
DEFAULT_CODE_ASSIGNMENT_CONFIG = CodeAssignmentConfig()
DEFAULT_EXPORT_CONFIG = ExportConfig()
DEFAULT_CODEDESIGNER_CONFIG = CodeDesignerConfig()


