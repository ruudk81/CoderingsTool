import os
from pathlib import Path
from typing import Dict, Tuple
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
# CREDENTIALS
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AZURE_RESOURCES = {
    "prod": {"endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
             "api_key":  os.getenv("AZURE_OPENAI_API_KEY")},
    "dev":  {"endpoint": os.getenv("AZURE_OPENAI_DEV_ENDPOINT"),
             "api_key":  os.getenv("AZURE_OPENAI_DEV_API_KEY")},
}

AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING", "text-embedding-3-large")


# =============================================================================
# DEPLOYMENT — which model runs which phase
# =============================================================================

API_PROVIDER = "azure"           # "openai" (own account) or "azure" (deployments below)


@dataclass(frozen=True)
class Model:
    name: str            # what the API calls it
    deployment: str      # Azure deployment name; may differ from `name`
    resource: str        # which Azure resource hosts it: "prod" | "dev"
    reasoning: bool      # True -> reasoning effort + verbosity, no temperature
    price_in: float      # $ per 1M input tokens
    price_out: float     # $ per 1M output tokens
    context: int
    max_output: int

MODELS: Dict[Tuple[str, int], Model] = {
    #             name              deployment             resource  reason  in    out    context    max_output
    ("5.4", 5): Model("gpt-5.4",      "gpt-5.4",             "prod",  True,   2.50, 15.00, 1_000_000, 128_000),
    ("5.4", 4): Model("gpt-5.4-mini", "gpt-5.4-mini",        "dev",   True,   0.75,  4.50,   400_000, 128_000),
    ("5.4", 3): Model("gpt-5.4-nano", "gpt-5.4-nano",        "dev",   True,   0.20,  1.25,   400_000, 128_000),
    ("5.6", 3): Model("gpt-5.6-luna", "gpt-5.6-luna",        "dev",   True,   0.20,  1.20, 1_050_000, 128_000),
    ("4.1", 5): Model("gpt-4.1",      "gpt-4.1",             "prod",  False,  2.00,  8.00, 1_000_000,  32_000),
    ("4.1", 4): Model("gpt-4.1-mini", "Test_data_analytics", "prod",  False,  0.40,  1.60, 1_000_000,  32_000),
    ("4.1", 3): Model("gpt-4.1-nano", "Test_data_analytics", "prod",  False,  0.10,  0.40, 1_000_000,  32_000),
}

# Embeddings have no ladder — one entry per model, same fields.
EMBEDDINGS: Dict[str, Model] = {
    "text-embedding-3-large": Model("text-embedding-3-large", "text-embedding-3-large","prod", False, 0.13, 0.0, 8_191, 0),
}

STEP_MODEL: Dict[str, Tuple[str, int]] = {
    # Step 1: Preprocessing
    "spell_check":                        ("5.6", 3),
    # Step 2: Quality Filter
    "quality_filter":                     ("5.6", 3),
    # Step 3: Idea Extraction
    "idea_extraction_context":            ("5.6", 3),   # specifiers + dimension discovery
    "idea_extraction_taxonomy":           ("5.6", 3),   # domain discovery + consolidation
    "idea_extraction_abstraction_ladder": ("5.6", 3),   # main extraction + retry
    # Step 4: Taxonomy Classifier — discovery, consolidation, assignment,
    # refinement per level, then the valence merge. Named by function, not by
    # number: a reordering should not force a rename here or in the perf model.
    "classifier_facet_discovery":         ("5.6", 3),
    "classifier_facet_consolidation":     ("5.6", 3),
    "classifier_facet_assignment":        ("5.6", 3),
    "classifier_facet_refinement":        ("5.6", 3),
    "classifier_attribute_discovery":     ("5.6", 3),
    "classifier_attribute_consolidation": ("5.6", 3),
    "classifier_attribute_assignment":    ("5.6", 3),
    "classifier_attribute_refinement":    ("5.6", 3),
    "classifier_valence_merge":           ("5.6", 3),
    # Step 5: Code Generator (P8-P9)
    "codegen_p8":                         ("5.6", 3),
    "codegen_p9":                         ("5.6", 3),
    # Step 6: Code Assigner
    "code_assignment":                    ("5.6", 3),
}

# =============================================================================
# RESOLUTION — lookups only, no synthesis
# =============================================================================

_BY_NAME: Dict[str, Model] = {m.name: m for m in (*MODELS.values(), *EMBEDDINGS.values())}


def _rung(generation: str, level: int) -> Model:
    """The model at one rung of the ladder.

    A rung that isn't deployed is an error, never a fallback: silently serving a
    different model is exactly the failure this table exists to prevent.
    """
    try:
        return MODELS[(generation, level)]
    except KeyError:
        levels = sorted((lvl for gen, lvl in MODELS if gen == generation), reverse=True)
        detail = (f"generation {generation!r} has levels {levels}" if levels
                  else f"known generations: {sorted({gen for gen, _ in MODELS})}")
        raise RuntimeError(f"no model deployed at rung ({generation!r}, {level}) — {detail}") from None


def _model(name: str) -> Model:
    """The model with this name, from either table."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise RuntimeError(f"unknown model {name!r} — known: {sorted(_BY_NAME)}") from None


def get_step_model(phase: str) -> str:
    """Resolve the model name for a pipeline phase."""
    try:
        generation, level = STEP_MODEL[phase]
    except KeyError:
        raise RuntimeError(f"unknown phase {phase!r} — known: {sorted(STEP_MODEL)}") from None
    return _rung(generation, level).name


def get_azure_route(model: str) -> Tuple[str, str, str]:
    """Resolve (endpoint, api_key, deployment) for an Azure call with this model.

    Credentials are checked here rather than at import, so that importing config
    without a .env still works — but a model whose resource has no credentials
    fails loudly instead of silently hitting the other resource.
    """
    m = _model(model)
    creds = AZURE_RESOURCES[m.resource]
    if not creds["endpoint"] or not creds["api_key"]:
        raise RuntimeError(
            f"model {model!r} routes to Azure resource {m.resource!r}, "
            f"but its endpoint/key are missing from .env"
        )
    return creds["endpoint"], creds["api_key"], m.deployment


def get_model_for_api(model: str) -> str:
    """The name the API answers to: the deployment on Azure, the model on OpenAI."""
    return _model(model).deployment if API_PROVIDER == "azure" else model


# Which generations this configuration actually uses — reported alongside costs.
ACTIVE_GENERATIONS = "+".join(sorted({gen for gen, _ in STEP_MODEL.values()}))


# =============================================================================
# DERIVED MODEL REGISTERS
# =============================================================================

MODEL_PRICING = {n: {"input": m.price_in, "output": m.price_out}
                 for n, m in _BY_NAME.items()}

OPENAI_MODEL_LIMITS = {n: {"context_window": m.context, "max_output": m.max_output}
                       for n, m in _BY_NAME.items()}

DEFAULT_PRICING = {"input": 1.00, "output": 4.00}


# =============================================================================
# REASONING PARAMS & VERBOSITY
# =============================================================================

# "minimal" is absent on purpose: both gpt-5.4 and gpt-5.6-luna reject it.
VALID_EFFORTS = ("none", "low", "medium", "high")
VALID_VERBOSITIES = ("low", "medium", "high")

REASONING_EFFORT = "none"      # the floor for bulk phases
TEXT_VERBOSITY = "medium"      # default for all steps

STEP_EFFORT = {
    # Step 3: what the dimensions and domains ARE
    "idea_extraction_context":  "medium",
    "idea_extraction_taxonomy": "medium",
    # Step 4: the phases that BUILD the taxonomy reason; the two assignment
    # phases apply it and fall back to REASONING_EFFORT, which is where ~98% of
    # the calls are.
    "classifier_facet_discovery": "medium",
    "classifier_facet_consolidation": "medium",
    "classifier_facet_refinement": "medium",
    "classifier_attribute_discovery": "medium",
    "classifier_attribute_consolidation": "medium",
    "classifier_attribute_refinement": "medium",
    "classifier_valence_merge": "low",
    # Step 5: writing and consolidating the codebook
    "codegen_p8": "medium",
    "codegen_p9": "medium",
    # All other steps: fall back to REASONING_EFFORT
}

STEP_VERBOSITY = {
    # Step 4: discovery/consolidation phases have scratchpad → low saves tokens
    "classifier_facet_discovery": "low",
    "classifier_facet_consolidation": "low",
    "classifier_facet_refinement": "low",
    "classifier_attribute_discovery": "low",
    "classifier_attribute_consolidation": "low",
    "classifier_attribute_refinement": "low",
    "classifier_valence_merge": "low",
    # All other steps: fall back to TEXT_VERBOSITY
}


def _validate() -> None:
    """Fail at import if the three per-phase tables disagree.

    This has to run at import: six config_*.py files call get_step_model() as a
    dataclass default, so by the time a phase actually runs it is far too late to
    notice that its model was never deployed.

    It also covers what the getters cannot. STEP_EFFORT and STEP_VERBOSITY are
    read with .get(), so a renamed phase does not raise — it silently serves the
    default, and the setting is simply lost. An unsupported value is worse: it
    survives import and 400s on a live call, mid-run.
    """
    for phase, (generation, level) in STEP_MODEL.items():
        try:
            _rung(generation, level)
        except RuntimeError as exc:
            raise RuntimeError(f"STEP_MODEL[{phase!r}]: {exc}") from None

    names = [m.name for m in MODELS.values()]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise RuntimeError(f"model on more than one rung of MODELS: {duplicates}")

    for label, table, default, valid in (
        ("STEP_EFFORT", STEP_EFFORT, REASONING_EFFORT, VALID_EFFORTS),
        ("STEP_VERBOSITY", STEP_VERBOSITY, TEXT_VERBOSITY, VALID_VERBOSITIES),
    ):
        if default not in valid:
            raise RuntimeError(f"{label} default {default!r} is not one of {list(valid)}")
        for phase, value in table.items():
            if phase not in STEP_MODEL:
                raise RuntimeError(
                    f"{label}[{phase!r}]: not a phase in STEP_MODEL — renamed? "
                    f"the phase would silently fall back to {default!r}"
                )
            if value not in valid:
                raise RuntimeError(
                    f"{label}[{phase!r}]: {value!r} is rejected by the API; "
                    f"use one of {list(valid)}"
                )


_validate()


def get_step_verbosity(phase: str) -> str:
    """Return verbosity for a pipeline phase. Falls back to TEXT_VERBOSITY."""
    return STEP_VERBOSITY.get(phase, TEXT_VERBOSITY)


def get_step_effort(phase: str) -> str:
    """Return reasoning effort for a pipeline phase. Falls back to REASONING_EFFORT."""
    return STEP_EFFORT.get(phase, REASONING_EFFORT)


def get_reasoning_params(model: str, phase: str = None) -> dict:
    """Return reasoning API params if the model is a reasoning model, else empty dict.

    Args:
        model: Model name, as returned by get_step_model().
        phase: Pipeline phase key (e.g. "classifier_facet_discovery"). If provided, uses
               per-step effort from STEP_EFFORT and verbosity from STEP_VERBOSITY.
    """
    if not _model(model).reasoning:
        return {}

    effort = get_step_effort(phase) if phase else REASONING_EFFORT
    verbosity = get_step_verbosity(phase) if phase else TEXT_VERBOSITY
   
    return {
        "reasoning": {"effort": effort},
        "text": {"verbosity": verbosity},
    }


# =============================================================================
# RATE LIMIT FALLBACKS (Used when API headers are unavailable)
# =============================================================================

FALLBACK_TPM = int(os.getenv("FALLBACK_TPM", "100000"))  # Conservative: 100K tokens/min
FALLBACK_RPM = int(os.getenv("FALLBACK_RPM", "100"))     # Conservative: 100 requests/min


# =============================================================================
# MODEL & PROCESSING CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Centralized model configuration."""

    MODEL_TYPES = {n: ("reasoning" if m.reasoning else "chat") for n, m in _BY_NAME.items()}

    # Embedding model (not family-dependent)
    embedding_model: str = "text-embedding-3-large"

    # Global parameters
    seed: int = 42
    default_temperature: float = 0.0
    default_max_tokens: int = 32000


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


DEFAULT_PROCESSING_CONFIG = ProcessingConfig()


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
        "taxonomy_classified": "005",            # growing model with enriched facet/attribute
        "taxonomy_xdomain": "005",               # cross-domain consolidated metadata
        "taxonomy_classified_xdomain": "005",    # cross-domain consolidated growing model
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
# MISC
# =============================================================================

DEFAULT_LANGUAGE = "Dutch"

# Language-specific labels for miscellaneous/catch-all code
MISCELLANEOUS_CODE_LABELS = {
    "Dutch": "Overig",
    "English": "Other",
    "German": "Sonstiges",
    "French": "Autre",
    "Spanish": "Otro",
}
