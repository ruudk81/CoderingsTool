"""
Experimental Configuration for Step 2: Quality Filter

Re-exports production config items that don't need modification.
Defines experimental constants for quality filtering.

Pattern follows step_1_preprocess/config_exp.py
"""

from dataclasses import dataclass

# =============================================================================
# RE-EXPORTS FROM PRODUCTION CONFIG (read-only items)
# =============================================================================
from config import (
    OPENAI_API_KEY,
    DEFAULT_LANGUAGE,
    ModelConfig,
    QualityFilterConfig,
    DEFAULT_QUALITY_FILTER_CONFIG,
    ProcessingConfig,
    DEFAULT_PROCESSING_CONFIG,
    API_PROVIDER,
    FALLBACK_TPM,
    FALLBACK_RPM,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)

# =============================================================================
# EXPERIMENTAL CONSTANTS (moved from qualityFilter_exp.py)
# =============================================================================

# Token estimation history windows
INPUT_HISTORY_MAXLEN = 3              # EMA input token history window
OUTPUT_HISTORY_MAXLEN = 5             # EMA output token history window
ERROR_WINDOW_SIZE = 50                # Token estimation error tracking window

# Timeout and latency defaults
DEFAULT_TIMEOUT_SECONDS = 30.0        # Default timeout when no latency data
DEFAULT_LATENCY_SECONDS = 2.0         # Default latency estimate

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
# EXPERIMENTAL CONFIG CLASS (for future use)
# =============================================================================
@dataclass
class QualityFilterConfigExp(QualityFilterConfig):
    """Experimental quality filter config - override fields as needed."""
    pass


DEFAULT_QUALITY_FILTER_CONFIG_EXP = QualityFilterConfigExp()
