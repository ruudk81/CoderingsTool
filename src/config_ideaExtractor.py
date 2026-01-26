"""
IdeaExtractor-specific configuration - separate from main config.py

This module contains all configuration constants for the IdeaExtractor,
organized into logical dataclasses for easier management and tuning.

These settings control:
- Token estimation and history tracking
- Tiktoken-to-API offset learning
- Timeouts and latency handling
- Progress reporting intervals
- Bootstrap measurement
- PID controller for rate limiting
- TPM tracking and utilization
- Throughput adjustment
- Generic specifier extraction
"""
from dataclasses import dataclass


# =============================================================================
# TOKEN HISTORY CONFIGURATION
# =============================================================================

@dataclass
class TokenHistoryConfig:
    """Configuration for token estimation history tracking."""
    input_history_maxlen: int = 10          # EMA input token history window
    output_history_maxlen: int = 15         # EMA output token history window
    output_ratio_history_maxlen: int = 20   # Track output/input ratios for learning
    default_output_ratio: float = 0.25      # Initial fallback ratio (learned from data)
    error_window_size: int = 50             # Token estimation error tracking window


# =============================================================================
# TIKTOKEN OFFSET CONFIGURATION
# =============================================================================

@dataclass
class TiktokenOffsetConfig:
    """Configuration for learning tiktoken-to-API token offset.

    The API always reports more tokens than tiktoken due to:
    - System messages added by the API
    - Instructor/structured output overhead
    - Message formatting tokens
    """
    api_offset_default: int = 300           # Default offset (instructor overhead ~300 tokens)
    offset_history_maxlen: int = 30         # Samples to learn the offset
    offset_min_samples: int = 5             # Min samples before using learned offset


# =============================================================================
# TIMEOUT CONFIGURATION
# =============================================================================

@dataclass
class TimeoutConfig:
    """Configuration for timeouts and latency handling."""
    default_timeout_seconds: float = 30.0   # Default timeout when no latency data
    default_latency_seconds: float = 2.0    # Default latency estimate
    max_token_acquire_attempts: int = 1000  # Max attempts to acquire tokens before failing
    bootstrap_timeout_seconds: float = 30.0 # Timeout for bootstrap probe calls


# =============================================================================
# REPORTING CONFIGURATION
# =============================================================================

@dataclass
class ReportingConfig:
    """Configuration for progress and diagnostic reporting intervals."""
    progress_report_interval: int = 5       # Seconds between progress reports
    diagnostic_interval: int = 10           # Seconds between diagnostic outputs
    adjustment_interval: int = 20           # Seconds between throughput adjustments


# =============================================================================
# BOOTSTRAP CONFIGURATION
# =============================================================================

@dataclass
class BootstrapConfig:
    """Configuration for bootstrap measurement phase."""
    num_probes: int = 5                     # Number of probe calls for accuracy
    default_avg_tokens: int = 1500          # Default token estimate fallback
    sample_size_for_token_estimation: int = 10  # Sample size for initial token calculation


# =============================================================================
# PID CONTROLLER CONFIGURATION
# =============================================================================

@dataclass
class PIDControllerConfig:
    """Configuration for PID-style throughput controller.

    Uses ASYMMETRIC gains:
    - kp_up: Aggressive when under-utilizing (speed up faster)
    - kp_down: Gentle when over-utilizing (slow down carefully)

    The controller tracks:
    - Error: difference between target and actual TPM utilization
    - Integral: accumulated error over time (handles persistent bias)
    - Derivative: rate of change (dampens oscillations)
    """
    kp_up: float = 0.4                      # Proportional gain when under-utilizing
    kp_down: float = 0.2                    # Proportional gain when over-utilizing
    ki: float = 0.05                        # Integral gain (accumulated error correction)
    kd: float = 0.1                         # Derivative gain (dampen oscillations)
    min_adjustment: float = 0.02            # Minimum adjustment to apply (2%)
    max_adjustment: float = 0.15            # Maximum single adjustment (15%)


# =============================================================================
# TPM TRACKING CONFIGURATION
# =============================================================================

@dataclass
class TPMTrackingConfig:
    """Configuration for real-time TPM (Tokens Per Minute) tracking.

    Tracks actual consumption in a sliding window to provide
    accurate utilization metrics for PID control.
    """
    sliding_window_seconds: float = 60.0    # Track TPM over last 60 seconds
    sample_interval: float = 1.0            # Sample TPM every 1 second
    target_utilization: float = 0.85        # Target 85% TPM utilization (15% buffer)


# =============================================================================
# THROUGHPUT ADJUSTMENT CONFIGURATION
# =============================================================================

@dataclass
class ThroughputConfig:
    """Configuration for threshold-based throughput adjustment (fallback to PID)."""
    adjustment_min_samples: int = 10        # Min samples before adjustment
    adjustment_threshold: float = 1.05      # Sensitivity threshold (5%)


# =============================================================================
# GENERIC SPECIFIER CONFIGURATION
# =============================================================================

@dataclass
class SpecifierConfig:
    """Configuration for generic specifier extraction.

    Controls how context specifiers (lang, domain, topic, perspective,
    entity, intent) are extracted from response samples.
    """
    sample_min: int = 50                    # Min samples for generic specifiers
    sample_max: int = 1000                  # Max samples for generic specifiers
    chunk_size: int = 100                   # Chunk size for specifier extraction
    max_workers: int = 10                   # Max workers for specifier extraction


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

DEFAULT_TOKEN_HISTORY_CONFIG = TokenHistoryConfig()
DEFAULT_TIKTOKEN_OFFSET_CONFIG = TiktokenOffsetConfig()
DEFAULT_TIMEOUT_CONFIG = TimeoutConfig()
DEFAULT_REPORTING_CONFIG = ReportingConfig()
DEFAULT_BOOTSTRAP_CONFIG = BootstrapConfig()
DEFAULT_PID_CONTROLLER_CONFIG = PIDControllerConfig()
DEFAULT_TPM_TRACKING_CONFIG = TPMTrackingConfig()
DEFAULT_THROUGHPUT_CONFIG = ThroughputConfig()
DEFAULT_SPECIFIER_CONFIG = SpecifierConfig()
