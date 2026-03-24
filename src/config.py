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
# MODEL CONFIGURATION - CENTRALIZED FOR DEVELOPMENT PIPELINE
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

# =============================================================================
# MODEL FAMILY TOGGLE
# =============================================================================
# Switch this to change all pipeline models at once.
# Each step uses get_model(tier) to resolve the actual model name.
#
# Supported families: "gpt-4.1", "gpt-5"
# Tiers: "default" (full model), "mini", "nano"
#
# Examples:
#   MODEL_FAMILY = "gpt-4.1"  →  gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
#   MODEL_FAMILY = "gpt-5"    →  gpt-5, gpt-5-mini, gpt-5-nano

MODEL_FAMILY = "gpt-5"


def get_model(tier: str = "default") -> str:
    """Resolve a model name from the current MODEL_FAMILY and tier.

    Args:
        tier: "default", "mini", or "nano"
    """
    if tier == "default":
        return MODEL_FAMILY
    return f"{MODEL_FAMILY}-{tier}"


# Reasoning model families require reasoning_effort and text_verbosity parameters.
# These are hardcoded defaults (minimal reasoning, medium verbosity) applied
# automatically when using a reasoning model family like gpt-5.
_REASONING_FAMILIES = {"gpt-5"}

REASONING_EFFORT = "minimal"   # minimal, low, medium, high
TEXT_VERBOSITY = "medium"      # minimal, low, medium, high


def get_reasoning_params(model: str = None) -> dict:
    """Return reasoning API params if the model is a reasoning model, else empty dict.

    Usage in _llm_call: pass **get_reasoning_params(model) as kwargs to llm_create_async.
    For chat models (gpt-4.1 family): returns {} — no extra params.
    For reasoning models (gpt-5 family): returns {reasoning: {effort: ...}}.

    NOTE: We only pass 'reasoning' (effort), NOT 'text' (verbosity).
    The 'text' parameter conflicts with instructor's structured output format.
    Instructor controls the output format; adding text.format overrides it
    and causes InstructorRetryException on every call.
    """
    if model is None:
        model = get_model()
    for rf in _REASONING_FAMILIES:
        if model == rf or model.startswith(rf + "-"):
            return {
                "reasoning": {"effort": REASONING_EFFORT},
            }
    return {}


# Default models (derived from MODEL_FAMILY)
DEFAULT_MODEL = get_model("mini")
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

FALLBACK_TPM = int(os.getenv("FALLBACK_TPM", "100000"))  # Conservative: 100K tokens/min
FALLBACK_RPM = int(os.getenv("FALLBACK_RPM", "100"))     # Conservative: 100 requests/min

# =============================================================================
# EMBEDDING MODEL DIMENSIONS
# =============================================================================

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
    """Centralized model configuration.

    Development pipeline (steps 1-6): uses MODEL_FAMILY toggle via get_model().
    Old production pipeline (pipeline.py, app.py): uses legacy stage models below.
    """

    # =============================================================================
    # MODEL TYPE MAPPING (shared by both pipelines)
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
    # SHARED PARAMETERS
    # =============================================================================

    # Embedding model (not family-dependent)
    embedding_model: str = "text-embedding-3-large"

    # Global parameters
    seed: int = 42
    default_temperature: float = 0.0
    default_max_tokens: int = 32000






    # =============================================================================
    # OLD PRODUCTION PIPELINE MODELS — USED BY pipeline.py / app.py — TO BE CLEANED
    # These models are referenced by utils/codeGenerator.py, utils/codebookRefinement.py,
    # utils/codeAssigner.py, and app.py. They will be removed when the old production
    # pipeline is migrated to use the development steps.
    # =============================================================================

    speculative_codes_model: str = get_model("mini")
    thematic_summary_model: str = "gpt-5-chat-latest"
    candidate_selection_model: str = "gpt-5-chat-latest"
    code_generation_model: str = "gpt-5-chat-latest"
    validation_model: str = "gpt-5-chat-latest"
    codebook_refinement_model: str = "gpt-5-mini"
    code_assignment_model: str = get_model("nano")
    refinement_temperature: float = 0.2

    # GPT-5 reasoning/verbosity parameters (old production pipeline only)
    theme_extraction_reasoning_effort: str = "minimal"
    theme_extraction_text_verbosity: str = "medium"
    candidate_selection_reasoning_effort: str = "minimal"
    candidate_selection_text_verbosity: str = "medium"
    code_generation_reasoning_effort: str = "minimal"
    code_generation_text_verbosity: str = "medium"
    validation_reasoning_effort: str = "minimal"
    validation_text_verbosity: str = "medium"
    refinement_reasoning_effort: str = "minimal"
    refinement_text_verbosity: str = "medium"
    gpt5_reasoning_effort: str = "minimal"
    gpt5_text_verbosity: str = "medium"

    # =============================================================================
    # HELPER METHODS — OLD PRODUCTION PIPELINE — TO BE CLEANED
    # =============================================================================

    def get_model_for_stage(self, stage: str) -> str:
        """Get the appropriate model for a pipeline stage.

        Used by old production pipeline (app.py, utils/codeGenerator.py).
        Development steps 1-3 now use their own config.model field directly.
        """
        stage_models = {
            'spell_check': get_model("mini"),          # fallback if old callers still use this
            'quality_filter': get_model("mini"),        # fallback if old callers still use this
            'segmentation': get_model("mini"),          # fallback if old callers still use this
            'embedding': self.embedding_model,
            'speculative_codes': self.speculative_codes_model,
            'theme_extraction': self.thematic_summary_model,
            'candidate_selection': self.candidate_selection_model,
            'code_recommendation': self.code_generation_model,
            'recommendation_validation': self.validation_model,
            'codebook_refinement': self.codebook_refinement_model,
            'code_assignment': self.code_assignment_model,
        }
        return stage_models.get(stage, DEFAULT_MODEL)

    def get_temperature_for_stage(self, stage: str) -> float:
        stage_temperatures = {
            'spell_check': 0.0,
            'quality_filter': 0.0,
            'refinement': self.refinement_temperature,
        }
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
        """Get GPT-5 reasoning effort for specific stage (old production pipeline)."""
        stage_efforts = {
            'theme_extraction': self.theme_extraction_reasoning_effort,
            'candidate_selection': self.candidate_selection_reasoning_effort,
            'code_recommendation': self.code_generation_reasoning_effort,
            'recommendation_validation': self.validation_reasoning_effort,
            'codebook_refinement': self.refinement_reasoning_effort,
        }
        return stage_efforts.get(stage, self.gpt5_reasoning_effort)

    def get_text_verbosity_for_stage(self, stage: str) -> str:
        """Get GPT-5 text verbosity for specific stage (old production pipeline)."""
        stage_verbosities = {
            'theme_extraction': self.theme_extraction_text_verbosity,
            'candidate_selection': self.candidate_selection_text_verbosity,
            'code_recommendation': self.code_generation_text_verbosity,
            'recommendation_validation': self.validation_text_verbosity,
            'codebook_refinement': self.refinement_text_verbosity,
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
    # Cap at 200: even with 30K RPM, a single Python process can't efficiently
    # manage 500+ in-flight HTTP connections. The rate limiter paces requests,
    # but the concurrency gate caps how many are in-flight simultaneously.
    concurrency_cap_default: int = 200
    concurrency_cap_permissive: int = 500
    concurrency_min_default: int = 10
    concurrency_min_permissive: int = 0
    concurrency_min_conservative: int = 5

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
        # Step 5: code generator (dev, P8-P9)
        "mece_codes": "006",
        "mece_codes_metadata": "006",
        # Step 6: code assigner (dev, P10)
        "taxonomy_codes": "007",
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


