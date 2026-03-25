"""
Quality-filter-specific configuration — separate from main config.py

This module contains all configuration for the Step 2 quality filtering pipeline:
- QualityFilterConfig dataclass
- Processing constants for token estimation, timeouts, and throughput adjustment
"""
from dataclasses import dataclass
from config import get_model


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
    model: str = get_model("mini")
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
