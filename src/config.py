import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# =============================================================================
# DEPLOYMENT - provider + family
# =============================================================================

API_PROVIDER = "openai"  # Options: "openai" or "azure"
MODEL_FAMILY = "gpt-5.4"

#API_PROVIDER = "azure"  # Options: "openai" or "azure"
#MODEL_FAMILY = "gpt-4.1"

# Examples:
#   MODEL_FAMILY = "gpt-4.1"  →  gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
#   MODEL_FAMILY = "gpt-5"    →  gpt-5, gpt-5-mini, gpt-5-nano
#   MODEL_FAMILY = "gpt-5.4"   


def get_model(tier: str = "default") -> str:
    """Resolve a model name from the current MODEL_FAMILY and tier.

    Applies FAMILY_TIER_OVERRIDES when the current MODEL_FAMILY has a mapping
    (e.g. gpt-4.1 "nano" → "mini" because gpt-4.1-nano is too weak).

    Args:
        tier: "default", "mini", or "nano"
    """
    overrides = FAMILY_TIER_OVERRIDES.get(MODEL_FAMILY, {})
    tier = overrides.get(tier, tier)
    if tier == "default":
        return MODEL_FAMILY
    return f"{MODEL_FAMILY}-{tier}"

# =============================================================================
# STEP MODEL TIERS 
# =============================================================================

STEP_MODEL_TIERS = {
    # Step 1: Preprocessing
    "spell_check":      "nano",
    # Step 2: Quality Filter
    "quality_filter":   "nano",
    # Step 3: Idea Extraction
    "idea_extraction_context": "default",           # specifiers + dimension discovery
    "idea_extraction_taxonomy": "default",          # domain discovery + consolidation
    "idea_extraction_abstraction_ladder": "nano",   # main extraction + retry
    # Step 4: Taxonomy Classifier (P1-P8)
    "classifier_p1":    "mini",      # Facet Discovery
    "classifier_p2":    "default",   # Facet Consolidation
    "classifier_p3":    "nano",      # Facet Assignment
    "classifier_p4":    "mini",      # Attribute Discovery
    "classifier_p5":    "default",   # Attribute Consolidation
    "classifier_p6":    "nano",      # Attribute Assignment
    "classifier_p7":    "default",   # Cross-facet Attribute Consolidation
    "classifier_p8":    "default",   # Cross-domain Attribute Consolidation
    # Step 5: Code Generator (P8-P9)
    "codegen_p8":       "default",
    "codegen_p9":       "default",
    # Step 6: Code Assigner
    "code_assignment":  "nano",
}

# Override tiers per model family (when target family needs different tier).
# E.g. gpt-4.1 has no nano-quality equivalent to gpt-5.4-nano → use mini instead.
FAMILY_TIER_OVERRIDES = {
    "gpt-4.1": {
        "nano": "mini",       # gpt-4.1-nano < gpt-5.4-nano → bump to mini
        "mini": "default",    # gpt-4.1-mini < gpt-5.4-mini → bump to default
    }
}


def get_step_model(phase: str) -> str:
    """Resolve model name for a pipeline phase from the central tier mapping."""
    return get_model(STEP_MODEL_TIERS[phase])

# =============================================================================
# REASONING PARAMS & VERBOSITY
# =============================================================================

REASONING_EFFORT = "none"   # none, minimal, low, medium, high — none only for ≥5.4
TEXT_VERBOSITY = "medium"      # low, medium, high — default for all steps

# Per-step verbosity overrides (None or absent = use TEXT_VERBOSITY default)
STEP_VERBOSITY = {
    # Step 4: discovery/consolidation phases have scratchpad → low saves tokens
    "classifier_p1": "low",
    "classifier_p2": "low",
    "classifier_p4": "low",
    "classifier_p5": "low",
    "classifier_p7": "low",
    "classifier_p8": "low",
    # All other steps: fall back to TEXT_VERBOSITY
}


def get_step_verbosity(phase: str) -> str:
    """Return verbosity for a pipeline phase. Falls back to TEXT_VERBOSITY."""
    return STEP_VERBOSITY.get(phase, TEXT_VERBOSITY)


def get_reasoning_params(model: str = None, phase: str = None) -> dict:
    """Return reasoning API params if the model is a reasoning model, else empty dict.

    Args:
        model: Model name. If None, uses default model.
        phase: Pipeline phase key (e.g. "classifier_p1"). If provided, uses
               per-step verbosity from STEP_VERBOSITY.
    """
    if model is None:
        model = get_model()
    if ModelConfig.MODEL_TYPES.get(model) == "reasoning":
        verbosity = get_step_verbosity(phase) if phase else TEXT_VERBOSITY
        return {
            "reasoning": {"effort": REASONING_EFFORT},
            "text": {"verbosity": verbosity},
        }
    return {}


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


# =============================================================================
# MODEL CONFIGURATION - CENTRALIZED FOR DEVELOPMENT PIPELINE
# =============================================================================

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
AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER", "gpt-4.1-mini")

# Azure ARM access (for dynamic limit fetching - optional)
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")

# =============================================================================
# HARDCODED MODEL LIMITS 
# =============================================================================
OPENAI_MODEL_LIMITS = {
    # GPT-4.1 family  
    "gpt-4.1": {"context_window": 1_000_000, "max_output": 32_000},
    "gpt-4.1-mini": {"context_window": 1_000_000, "max_output": 32_000},
    "gpt-4.1-nano": {"context_window": 1_000_000, "max_output": 32_000},
    # GPT-5 family -  
    "gpt-5.4": {"context_window": 1_000_000, "max_output": 128_000},
    "gpt-5.4-mini": {"context_window": 400_000, "max_output": 128_000},
    "gpt-5.4-nano": {"context_window": 400_000, "max_output": 128_000},
    "gpt-5": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5.1": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5.2": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5-mini": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5-nano": {"context_window": 128_000, "max_output": 32_000},
    "gpt-5-chat-latest": {"context_window": 272_000, "max_output": 128_000},
    # GPT-4o family  
    "gpt-4o": {"context_window": 128_000, "max_output": 16_000},
    "gpt-4o-mini": {"context_window": 128_000, "max_output": 16_000},
    # Embeddings
    "text-embedding-3-large": {"context_window": 8_191, "max_output": 0},
    "text-embedding-3-small": {"context_window": 8_191, "max_output": 0},
}

# =============================================================================
# MODEL PRICING 
# Update when OpenAI changes pricing: https://openai.com/api/pricing/
# =============================================================================
MODEL_PRICING = {
    # GPT-4.1 family
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # GPT-5 family
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5.2": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-chat-latest": {"input": 1.25, "output": 10.00},
    # GPT-4o family (legacy)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Embeddings
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 1.00, "output": 4.00}


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
    """Centralized model configuration."""

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
        "gpt-5.4": "reasoning",
        "gpt-5.4-mini": "reasoning",
        "gpt-5.4-nano": "reasoning",
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
# PROCESSING CONFIGURATION
# =============================================================================

@dataclass
class ProcessingConfig:
    """Global processing parameters affecting cache validity and performance"""

    # Rate limiting
    rate_limit_headroom: float = 0.9  # Use 90% of API limits for safety

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
        "taxonomy_classified": "005",  # growing model with enriched facet/attribute
        "taxonomy_xdomain": "005",             # cross-domain consolidated metadata
        "taxonomy_classified_xdomain": "005",  # cross-domain consolidated growing model
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

DEFAULT_MODEL_CONFIG = ModelConfig()

# Processing configuration
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()

DEFAULT_EXPORT_CLEANUP_CONFIG = ExportCleanupConfig()


# =============================================================================
# MISC
# =============================================================================

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
