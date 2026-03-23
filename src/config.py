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
API_PROVIDER = "openai"  # Options: "openai" or "azure"

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
# Defaults to DEFAULT_MODEL (gpt-4.1-mini) to match CodeDesignerConfig.model
AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER", "gpt-4.1-mini")

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

    # Quality filtering and segmentation models
    quality_filter_model: str = DEFAULT_MODEL
    segmentation_model: str = "gpt-4.1-mini"

    # Embedding model
    embedding_model: str = "text-embedding-3-large"

    speculative_codes_model: str = DEFAULT_MODEL

    # Codebook generation
    thematic_summary_model: str = "gpt-5-chat-latest"
    candidate_selection_model: str = "gpt-5-chat-latest"
    code_generation_model: str ="gpt-5-chat-latest"
    validation_model: str = "gpt-5-chat-latest"

    # Codebook refinement
    codebook_refinement_model: str = "gpt-5-mini"

    # Code assignment
    code_assignment_model: str = "gpt-4.1-nano"

  

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
            'embedding': self.embedding_model,
            'speculative_codes': self.speculative_codes_model,
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
    

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

@dataclass
class ProcessingConfig:
    """Global processing parameters affecting cache validity and performance"""

    # Rate limiting
    rate_limit_headroom: float = 0.9  # Use 80% of API limits for safety

    # Concurrency bounds
    concurrency_cap_default: int = 1000
    concurrency_cap_permissive: int = 10000
    concurrency_min_default: int = 100
    concurrency_min_permissive: int = 0
    concurrency_min_conservative: int = 10

    # Adaptive timeout bounds
    adaptive_timeout_min_seconds: float = 15.0
    adaptive_timeout_max_seconds: float = 120.0
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
        # Step 0-2: production pipeline
        "data": "001",
        "preprocessed": "002",
        "quality_filter": "003",
        # Step 3: idea extraction (dev)
        "extracted_ideas": "004",
        "extracted_ideas_metadata": "004",
        # Step 4: taxonomy classifier (dev, P1-P7)
        "taxonomy": "005",
        "taxonomy_metadata": "005",
        # Step 5: codebook generator (dev, P8-P10)
        "mece_codes": "006",
        "mece_codes_metadata": "006",
        "taxonomy_codes": "006",
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

# QualityFilterConfig moved to config_steps/config_qualityFilter.py

# CodeDesignerConfig moved to config_steps/config_codeGenerator.py

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
# EXPORT CLEANUP CONFIGURATION
# =============================================================================

@dataclass
class ExportCleanupConfig:
    """Configuration for automatic cleanup of exports/ subdirectories."""
    enabled: bool = True               # Set False to disable auto-cleanup
    max_age_days: int = 30             # Delete files older than this
    keep_latest_n: int = 3             # Always keep N newest per group key
    silent: bool = True                # Suppress console output when auto-running


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Central model configuration - configure all models here
DEFAULT_MODEL_CONFIG = ModelConfig()

# Processing configuration
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()

# Step-specific configurations (SpellCheckConfig → config_preprocess.py, QualityFilterConfig → config_qualityFilter.py)
DEFAULT_CODE_ASSIGNMENT_CONFIG = CodeAssignmentConfig()
DEFAULT_EXPORT_CLEANUP_CONFIG = ExportCleanupConfig()


