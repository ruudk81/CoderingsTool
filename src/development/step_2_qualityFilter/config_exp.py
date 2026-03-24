"""
Experimental Configuration for Step 2: Quality Filter

Re-exports production config items that don't need modification.
Defines experimental constants for quality filtering.

Pattern follows step_1_preProcessor/config_exp.py
"""

from dataclasses import dataclass

# =============================================================================
# RE-EXPORTS FROM PRODUCTION CONFIG (read-only items)
# =============================================================================
from config import (
    OPENAI_API_KEY,
    DEFAULT_LANGUAGE,
    ModelConfig,
    ProcessingConfig,
    DEFAULT_PROCESSING_CONFIG,
    API_PROVIDER,
    FALLBACK_TPM,
    FALLBACK_RPM,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)
from utils.llm import _is_reasoning_model
from config_steps.config_qualityFilter import (
    QualityFilterConfig,
    DEFAULT_QUALITY_FILTER_CONFIG,
)
from config_steps.config_ideaExtractor import (
    RampUpConfig, DEFAULT_RAMP_UP_CONFIG,
    CircuitBreakerConfig, DEFAULT_CIRCUIT_BREAKER_CONFIG,
    WarmUpConfig, DEFAULT_WARM_UP_CONFIG,
)

# =============================================================================
# EXPERIMENTAL CONSTANTS (moved from qualityFilter_exp.py)
# =============================================================================

# Token estimation history windows
INPUT_HISTORY_MAXLEN = 3              # EMA input token history window
OUTPUT_HISTORY_MAXLEN = 5             # EMA output token history window
ERROR_WINDOW_SIZE = 50                # Token estimation error tracking window

# Timeout and latency defaults
DEFAULT_TIMEOUT_SECONDS = 180.0       # Cold-start timeout (generous for reasoning models)
DEFAULT_LATENCY_SECONDS = 10.0        # Fallback for LatencyTracker when no data (overridden by model-tier)

# Model-tier latency estimates (cold start only, replaced by measured values at warm-up)
# See optimal_api_request_strategy.md Section C1
MODEL_TIER_LATENCY = {
    "nano": 2.0,        # Tiny model, fast inference
    "mini": 5.0,        # Small model, moderate inference
    "default": 10.0,    # Full model, longer inference
    "reasoning": 20.0,  # Reasoning overhead doubles latency
}

# Output token ratios by model type
OUTPUT_RATIO_CHAT = 1.15              # Chat models: ~15% output overhead
OUTPUT_RATIO_REASONING = 2.5          # Reasoning models: 150% output overhead

# Progress reporting
PROGRESS_REPORT_INTERVAL = 5          # Seconds between progress reports
DIAGNOSTIC_INTERVAL = 30              # Seconds between diagnostic reports

# Rate limiting
MAX_TOKEN_ACQUIRE_ATTEMPTS = 1000     # Max attempts to acquire tokens before failing

# Throughput adjustment (feedback loop)
THROUGHPUT_ADJUSTMENT_THRESHOLD = 1.1     # Trigger when actual > 110% of estimate
THROUGHPUT_ADJUSTMENT_MIN_SAMPLES = 10    # Min data points before adjusting
ADJUSTMENT_INTERVAL = 15                  # Seconds between adjustment checks

# =============================================================================
# OPTIMAL API REQUEST STRATEGY CONSTANTS
# See optimal_api_request_strategy.md for rationale
# =============================================================================

# Cold start
COLD_START_CAP = 50                   # Max initial concurrency (undocumented API ceiling)

# Warm-up
WARM_UP_WINDOW_SECONDS = 10.0         # Time-based warm-up (not count-based)
WARM_UP_MIN_COMPLETIONS = 3           # Minimum completions needed for calibration

# Signal-based ramp
RAMP_INTERVAL_SECONDS = 5.0           # Evaluate signals every 5s
RAMP_INCREASE_FACTOR = 1.25           # +25% concurrency when all signals green
RAMP_DECREASE_FACTOR = 0.80           # -20% concurrency when any signal red

# Signal thresholds (as fraction of 90%-headroom limit)
SIGNAL_GREEN_THRESHOLD = 0.80         # < 80% utilization = green (ramp up)
SIGNAL_YELLOW_THRESHOLD = 0.90        # 80-90% = yellow (hold), > 90% = red (throttle)


# =============================================================================
# HELPER: Model-tier latency lookup
# =============================================================================
def get_model_tier_latency(model: str) -> float:
    """Return estimated latency for cold-start calculations based on model name."""
    model_lower = model.lower()
    if 'nano' in model_lower:
        return MODEL_TIER_LATENCY["nano"]
    if 'mini' in model_lower:
        return MODEL_TIER_LATENCY["mini"]
    if _is_reasoning_model(model):
        return MODEL_TIER_LATENCY["reasoning"]
    return MODEL_TIER_LATENCY["default"]


def get_output_ratio(model: str) -> float:
    """Return output token ratio for token estimation based on model type."""
    return OUTPUT_RATIO_REASONING if _is_reasoning_model(model) else OUTPUT_RATIO_CHAT


# =============================================================================
# EXPERIMENTAL CONFIG CLASS (for future use)
# =============================================================================
@dataclass
class QualityFilterConfigExp(QualityFilterConfig):
    """Experimental quality filter config - override fields as needed."""
    pass


DEFAULT_QUALITY_FILTER_CONFIG_EXP = QualityFilterConfigExp()
