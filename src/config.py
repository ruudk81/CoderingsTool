import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# Load .env before anything reads os.getenv() below (simple loader, no dependencies).
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


# =============================================================================
# DEPLOYMENT - provider + family
# =============================================================================

API_PROVIDER = "azure"     # "openai" (own account) or "azure" (deployments below)
MODEL_FAMILY = "gpt-5.4"   # "gpt-5.4" (reasoning) or "gpt-4.1" (chat)

# The two are independent: azure+gpt-5.4, azure+gpt-4.1 and openai+gpt-5.4 all work.
# On Azure the family is capped by what AZURE_DEPLOYMENTS maps to a real deployment.

# Examples:
#   MODEL_FAMILY = "gpt-4.1"  →  gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
#   MODEL_FAMILY = "gpt-5.4"  →  gpt-5.4, and the mini/nano tiers via
#                                FAMILY_TIER_OVERRIDES (gpt-5.6-luna on Azure)


def get_model(tier: str = "default") -> str:
    """Resolve a model name from the current MODEL_FAMILY and tier.

    Applies FAMILY_TIER_OVERRIDES, preferring a provider-specific entry
    (API_PROVIDER, MODEL_FAMILY) over a family-wide one.

    Args:
        tier: "default", "mini", or "nano"
    """
    overrides = (FAMILY_TIER_OVERRIDES.get((API_PROVIDER, MODEL_FAMILY))
                 or FAMILY_TIER_OVERRIDES.get(MODEL_FAMILY, {}))
    tier = overrides.get(tier, tier)
    if tier.startswith("gpt-"):
        # An override may name a full model (e.g. "mini": "gpt-5.6-luna") to serve
        # a tier with a model outside the family.
        return tier
    if tier == "default":
        return MODEL_FAMILY
    return f"{MODEL_FAMILY}-{tier}"

# =============================================================================
# STEP MODEL TIERS 
# =============================================================================

STEP_MODEL_TIERS = {
    # Step 1: Preprocessing
    "spell_check":      "mini",
    # Step 2: Quality Filter
    "quality_filter":   "mini",
    # Step 3: Idea Extraction
    "idea_extraction_context": "default",           # specifiers + dimension discovery
    "idea_extraction_taxonomy": "default",          # domain discovery + consolidation
    "idea_extraction_abstraction_ladder": "mini",   # main extraction + retry
    # Step 4: Taxonomy Classifier (P1-P9). P3 (facet discovery zonder assen)
    # is dezelfde dispatch als P2 met een andere prompt en heeft geen eigen key.
    "classifier_p1":    "default",   # Axis Discovery
    "classifier_p2":    "mini",      # Facet Discovery (met én zonder assen)
    "classifier_p4":    "mini",      # Facet Assignment
    "classifier_p5":    "default",   # Facet Consolidation (in-axis, post-assignment)
    "classifier_p6":    "mini",      # Attribute Discovery
    "classifier_p7":    "mini",      # Attribute Assignment
    "classifier_p8":    "default",   # Attribute Consolidation (in-facet, post-assignment)
    "classifier_p9":    "default",   # Valence-neutral merge
    # Step 5: Code Generator (P8-P9)
    "codegen_p8":       "default",
    "codegen_p9":       "default",
    # Step 6: Code Assigner
    "code_assignment":  "mini",
}

# Override tiers per model family, or per (provider, family) when the reason is
# provider-specific. A (provider, family) key wins over a family-wide one.
FAMILY_TIER_OVERRIDES = {
    "gpt-4.1": {
        "nano": "mini",       # gpt-4.1-nano < gpt-5.4-nano → bump to mini
        "mini": "default",    # gpt-4.1-mini < gpt-5.4-mini → bump to default
    },
    # Benchmark 2026-07-31 (exports/diagnostics/2026-07-31-luna-vs-mini-benchmark):
    # luna ≥ 5.4-mini on steps 2/3/4 — better filter verdicts, cleaner taxonomy
    # (solo-facet 12.5% vs 23-24%, placement errors 7.5% vs 11-12.5%) — at 2-4×
    # lower cost. Default tier stays gpt-5.4.
    ("azure", "gpt-5.4"): {
        "mini": "gpt-5.6-luna",
        "nano": "gpt-5.6-luna",
    },
}


def get_step_model(phase: str) -> str:
    """Resolve model name for a pipeline phase from the central tier mapping."""
    return get_model(STEP_MODEL_TIERS[phase])

# =============================================================================
# REASONING PARAMS & VERBOSITY
# =============================================================================

REASONING_EFFORT = "none"   # none, low, medium, high — the floor for bulk phases
TEXT_VERBOSITY = "medium"      # low, medium, high — default for all steps

# Per-step reasoning effort (absent = use REASONING_EFFORT default).
#
# Measured 2026-08-01 on a consolidation task, reasoning tokens per call:
#   gpt-5.4  none 0 | low 66 | medium 248 | high 215
#   luna     none 0 | low 24 | medium  33 | high  31
# "minimal" is rejected by both models, and medium ≈ high, so the real choice is
# none / low / medium.
#
# Why low and not medium: the step that matters is none -> low, where the model
# starts reasoning at all. low -> medium costs ~4x the reasoning tokens on 5.4
# and no measurement here shows it classifies better — both produced the same
# grouping. Raise a phase to medium when a measurement justifies it, not before.
#
# Which phases get it: the ones that build the taxonomy rather than apply it.
# They are ~1.7% of all calls (~278 of 16,700 on a full ASN run). The bulk phases
# — spell check, quality filter, extraction, both assignments — stay at the
# default, because they place one item into a structure that already exists.
STEP_EFFORT = {
    # Step 3: what the dimensions and domains ARE
    "idea_extraction_context":  "low",
    "idea_extraction_taxonomy": "low",
    # Step 4: discovery and consolidation of axes, facets + attributes.
    # medium op P1 (assen), P2/P3 en P6 (discovery) en P8 (in-facet
    # consolidatie) — besluit Ruud 2026-08-03 bij de herordende pijplijn.
    "classifier_p1": "medium",
    "classifier_p2": "medium",
    "classifier_p5": "low",
    "classifier_p6": "medium",
    "classifier_p8": "medium",
    "classifier_p9": "low",
    # Step 5: writing and consolidating the codebook
    "codegen_p8": "low",
    "codegen_p9": "low",
    # Absent on purpose (high volume, mechanical): spell_check, quality_filter,
    # idea_extraction_abstraction_ladder, classifier_p4, classifier_p7,
    # code_assignment.
}

# Per-step verbosity overrides (None or absent = use TEXT_VERBOSITY default)
STEP_VERBOSITY = {
    # Step 4: discovery/consolidation phases have scratchpad → low saves tokens
    "classifier_p1": "low",
    "classifier_p2": "low",
    "classifier_p5": "low",
    "classifier_p6": "low",
    "classifier_p8": "low",
    "classifier_p9": "low",
    # All other steps: fall back to TEXT_VERBOSITY
}


def get_step_verbosity(phase: str) -> str:
    """Return verbosity for a pipeline phase. Falls back to TEXT_VERBOSITY."""
    return STEP_VERBOSITY.get(phase, TEXT_VERBOSITY)


def get_step_effort(phase: str) -> str:
    """Return reasoning effort for a pipeline phase. Falls back to REASONING_EFFORT."""
    return STEP_EFFORT.get(phase, REASONING_EFFORT)


def get_reasoning_params(model: str = None, phase: str = None) -> dict:
    """Return reasoning API params if the model is a reasoning model, else empty dict.

    Args:
        model: Model name. If None, uses default model.
        phase: Pipeline phase key (e.g. "classifier_p1"). If provided, uses
               per-step effort from STEP_EFFORT and verbosity from STEP_VERBOSITY.
    """
    if model is None:
        model = get_model()
    if ModelConfig.MODEL_TYPES.get(model) != "reasoning":
        return {}

    effort = get_step_effort(phase) if phase else REASONING_EFFORT
    verbosity = get_step_verbosity(phase) if phase else TEXT_VERBOSITY
    # One shape for both providers: the Responses API takes reasoning effort and
    # verbosity nested. This is also what gives the gpt-5.6 family effort control —
    # on Chat Completions those rejected reasoning_effort next to function tools.
    return {
        "reasoning": {"effort": effort},
        "text": {"verbosity": verbosity},
    }


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


# =============================================================================
# MODEL CONFIGURATION - CENTRALIZED FOR DEVELOPMENT PIPELINE
# =============================================================================

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Azure OpenAI settings — deployments are spread over two resources in the
# Motivaction tenant: "prod" (mot-azure-open-ai) and "dev"
# (mot-azure-openai-dev-resource, carrying gpt-5.4-mini/-nano and gpt-5.6-luna).
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING", "text-embedding-3-large")

AZURE_RESOURCES = {
    "prod": {"endpoint": AZURE_OPENAI_ENDPOINT, "api_key": AZURE_OPENAI_API_KEY},
    "dev":  {"endpoint": os.getenv("AZURE_OPENAI_DEV_ENDPOINT"),
             "api_key":  os.getenv("AZURE_OPENAI_DEV_API_KEY")},
}

# Model name -> (resource, deployment name). Deployment names do not always match
# the models they serve, so the code keeps reasoning in model names and this map
# translates. Unmapped models fall back to ("prod", AZURE_OPENAI_DEPLOYMENT_NAME).
# Add a line here when a new deployment appears; nothing else needs to change.
AZURE_DEPLOYMENTS = {
    "gpt-5.4":      ("prod", "gpt-5.4"),
    "gpt-5.4-mini": ("dev",  "gpt-5.4-mini"),
    "gpt-5.4-nano": ("dev",  "gpt-5.4-nano"),
    "gpt-5.6-luna": ("dev",  "gpt-5.6-luna"),
    "gpt-4.1":      ("prod", "gpt-4.1"),
    "gpt-4.1-mini": ("prod", "Test_data_analytics"),  # serves gpt-4.1-mini
    "gpt-4.1-nano": ("prod", "Test_data_analytics"),
    "text-embedding-3-large": ("prod", "text-embedding-3-large"),
    "text-embedding-ada-002": ("prod", "text-embedding-ada-002"),
}


def get_azure_route(model: str) -> Tuple[str, str, str]:
    """Resolve (endpoint, api_key, deployment) for an Azure call with this model.

    Raises when the target resource's credentials are missing from .env, so a
    dev-routed model fails loudly instead of silently hitting the wrong resource.
    """
    resource, deployment = AZURE_DEPLOYMENTS.get(model, ("prod", AZURE_OPENAI_DEPLOYMENT_NAME))
    creds = AZURE_RESOURCES[resource]
    if not creds["endpoint"] or not creds["api_key"]:
        raise RuntimeError(
            f"Model '{model}' routes to Azure resource '{resource}', "
            f"but its endpoint/key are missing from .env"
        )
    return creds["endpoint"], creds["api_key"], deployment

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
    # GPT-5.6 family (Sol > Terra > Luna)
    "gpt-5.6-luna": {"context_window": 1_050_000, "max_output": 128_000},
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
    # GPT-5.6 family
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},  # per 2026-07-30 price cut (-80%)
    # GPT-4o family (legacy)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Embeddings
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 1.00, "output": 4.00}


def get_model_for_api(model: str) -> str:
    """
    Get the appropriate model/deployment name for the current API provider.

    For Azure, maps model names to deployment names.
    For OpenAI, returns the model name as-is.
    """
    if API_PROVIDER == "azure":
        return AZURE_DEPLOYMENTS.get(model, ("prod", AZURE_OPENAI_DEPLOYMENT_NAME))[1]
    return model


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

        # GPT-5 family (reasoning models)
        "gpt-5.4": "reasoning",
        "gpt-5.4-mini": "reasoning",
        "gpt-5.4-nano": "reasoning",
        "gpt-5.6-luna": "reasoning",
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
        # Step 4: taxonomy classifier (dev, P1-P8)
        "taxonomy": "005",
        "taxonomy_metadata": "005",
        "taxonomy_classified": "005",  # growing model with enriched facet/attribute
        "taxonomy_xdomain": "005",             # cross-domain consolidated metadata
        "taxonomy_classified_xdomain": "005",  # cross-domain consolidated growing model
        "taxonomy_corrected": "005",             # legacy P9-era over-merge corrected metadata; old chains only, nothing writes these anymore
        "taxonomy_classified_corrected": "005",  # post-hoc over-merge corrected growing model
        # Step 5: code generator (dev, P8-P9)
        "mece_codes": "006",
        "mece_codes_metadata": "006",
        # Step 6: code assigner (dev, P10)
        "taxonomy_codes": "007",
    })
    
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
# DEFAULT INSTANCES
# =============================================================================

DEFAULT_MODEL_CONFIG = ModelConfig()

# Processing configuration
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()


# =============================================================================
# MISC
# =============================================================================

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
