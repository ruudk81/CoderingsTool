"""
Quality-filter-specific configuration — separate from main config.py

This module contains all configuration for the Step 2 quality filtering pipeline:
- QualityFilterConfig dataclass
- Processing constants for token estimation, timeouts, and throughput adjustment
- Optimal API request strategy constants
"""
from dataclasses import dataclass
from config import get_step_model
from utils.llm import _is_reasoning_model


# =============================================================================
# QUALITY FILTER CONFIGURATION
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
    # Model (derived from MODEL_FAMILY toggle in config.py)
    model: str = get_step_model("quality_filter")
    max_concurrent_requests: int = 5  # For API rate limiting
    # Timeout configuration for API calls
    minimum_timeout_seconds: float = 15.0  # Minimum timeout for API calls (safety net)
    maximum_timeout_seconds: float = 60.0  # Maximum timeout for API calls (prevents excessive waits)


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

DEFAULT_QUALITY_FILTER_CONFIG = QualityFilterConfig()


# =============================================================================
# PROCESSING CONSTANTS
# =============================================================================

# Token estimation history windows
INPUT_HISTORY_MAXLEN = 3              # EMA input token history window
OUTPUT_HISTORY_MAXLEN = 5             # EMA output token history window
ERROR_WINDOW_SIZE = 50                # Token estimation error tracking window

# Timeout and latency defaults
DEFAULT_TIMEOUT_SECONDS = 180.0       # Cold-start timeout (generous for reasoning models)
DEFAULT_LATENCY_SECONDS = 10.0        # Fallback for LatencyTracker when no data (overridden by model-tier)

# Model-tier latency estimates (cold start only, replaced by measured values at warm-up)
MODEL_TIER_LATENCY = {
    "nano": 2.0,
    "mini": 5.0,
    "default": 10.0,
    "reasoning": 20.0,
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
# =============================================================================

# Cold start
COLD_START_CAP = 50                   # Max initial concurrency

# Warm-up
WARM_UP_WINDOW_SECONDS = 10.0         # Time-based warm-up
WARM_UP_MIN_COMPLETIONS = 3           # Minimum completions needed for calibration

# Signal-based ramp
RAMP_INTERVAL_SECONDS = 5.0           # Evaluate signals every 5s
RAMP_INCREASE_FACTOR = 1.10           # +10% concurrency when all signals green
RAMP_DECREASE_FACTOR = 0.80           # -20% concurrency when any signal red

# Signal thresholds (as fraction of 90%-headroom limit)
SIGNAL_GREEN_THRESHOLD = 0.80         # < 80% utilization = green (ramp up)
SIGNAL_YELLOW_THRESHOLD = 0.90        # 80-90% = yellow (hold), > 90% = red (throttle)


# =============================================================================
# HELPERS
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
