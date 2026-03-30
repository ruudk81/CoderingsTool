"""
IdeaExtractor-specific configuration - separate from main config.py

This module contains all configuration constants for the IdeaExtractor,
organized into logical dataclasses for easier management and tuning.

These settings control:
- Token estimation and history tracking
- Tiktoken-to-API offset learning
- Timeouts and latency handling
- Progress reporting intervals
- Concurrency discovery and rate limiting
- Throughput adjustment
- Generic specifier extraction
"""
from dataclasses import dataclass
from config import get_step_model

# =============================================================================
# SEGMENTATION CONFIGURATION  
# =============================================================================

@dataclass
class SegmentationConfig:
    """Configuration for segmentation and description step"""
    max_tokens: int = 16000
    completion_reserve: int = 1000
    min_batch_size: int = 5  # Minimum responses per batch for efficiency
    max_batch_size: int = 20  # Maximum responses per batch for manageability
    target_token_utilization: float = 0.8  # Use 80% of available tokens per batch
    retry_delay: int = 2
    max_retries: int = 3
    spacy_batch_size: int = 32
    umap_n_jobs: int = 1
    max_code_examples: int = 5  # For verbose output
    max_sample_responses: int = 3  # For verbose output
    # Model (derived from MODEL_FAMILY toggle in config.py)
    model: str = get_step_model("segmentation")
    temperature: float = 0.0  # Temperature for generation
    max_concurrent_requests: int = 8  # Optimized for better throughput while respecting rate limits
    # Timeout configuration for API calls
    minimum_timeout_seconds: float = 15.0  # Minimum timeout for API calls (safety net)
    maximum_timeout_seconds: float = 60.0  # Maximum timeout for API calls (prevents excessive waits)

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
    """Configuration for timeouts and latency handling.

    Per strategy doc: single-processing steps (1, 2, 3, 8) use 20s cold-start floor;
    chunk-processing steps (4, 5, 6) use 45s. Adaptive after warm-up: max(floor, min(P95×3, 180)).
    """
    timeout_floor_seconds: float = 10.0     # Cold-start floor for timeout safety net
    default_timeout_seconds: float = 10.0   # Cold-start default timeout
    default_latency_seconds: float = 2.0    # Default latency estimate
    max_token_acquire_attempts: int = 1000  # Max attempts to acquire tokens before failing


# =============================================================================
# REPORTING CONFIGURATION
# =============================================================================

@dataclass
class ReportingConfig:
    """Configuration for progress and diagnostic reporting intervals."""
    progress_report_interval: int = 5       # Seconds between progress reports
    diagnostic_interval: int = 10           # Seconds between diagnostic outputs
    adjustment_interval: int = 20           # Seconds between concurrency/throughput adjustments


# =============================================================================
# BOOTSTRAP CONFIGURATION
# =============================================================================

@dataclass
class BootstrapConfig:
    """Configuration for initial token estimation defaults."""
    default_avg_tokens: int = 1500          # Default token estimate fallback
    sample_size_for_token_estimation: int = 10  # Sample size for initial token calculation


# =============================================================================
# RAMP-UP CONFIGURATION
# =============================================================================

@dataclass
class ConcurrencyControlConfig:
    """State machine concurrency controller.

    Monitors throughput (concurrency/P50) and in-flight P100 (max latency per tick).
    Ramps up gently, holds steady at sweet spot, backs off on stress, recovers to healthy level.

    States: RAMP-UP → STEADY ↔ BACKOFF → RECOVER → STEADY
    """
    ramp_step_pct: float = 0.025           # +2.5% of starting concurrency per tick (min 2)
    backoff_pct: float = 0.85              # cut to 85% of last healthy concurrency on BACKOFF
    steady_ratio: float = 2.0             # inflight_P95/P50 > 2× → STEADY
    inflight_ratio: float = 5.0           # inflight_P100/P50 > 5× → BACKOFF (after 2 consecutive ticks)
    min_concurrency: int = 5              # hard floor


# =============================================================================
# CIRCUIT BREAKER CONFIGURATION
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Concurrency circuit breaker — detects sustained timeout rate spikes.

    Trigger-only mechanism: detects timeout rate > threshold, signals the caller
    to trigger BACKOFF. Does not manage concurrency or recovery itself.

    Lifecycle: CLOSED → detects spike → trips (signals caller) → cooldown → CLOSED.
    """
    window_size: int = 100                # Count-based: last N completions (not time-based)
    trip_threshold: float = 0.05          # Trip if >5% of events are timeouts
    min_events_to_trip: int = 10          # Need enough events to be statistically meaningful
    cooldown_drain_multiplier: float = 3.0  # Cooldown = drain_time × this multiplier


# =============================================================================
# PID CONTROLLER CONFIGURATION
# =============================================================================

@dataclass
class PIDControllerConfig:
    """PID controller for arrival rate adjustment.

    Uses ASYMMETRIC gains:
    - kp_up: Aggressive when under-utilizing (speed up faster)
    - kp_down: Gentle when over-utilizing (slow down carefully)

    The controller adjusts the arrival rate (requests/second) based on
    real-time TPM utilization, keeping throughput near optimal without
    hitting rate limits.
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
    """Real-time TPM (Tokens Per Minute) tracking for PID input.

    Tracks actual token consumption in a sliding window to provide
    accurate utilization metrics for PID control.
    """
    sliding_window_seconds: float = 60.0    # Track TPM over last 60 seconds
    target_utilization: float = 0.80        # Target 80% TPM utilization (20% buffer)


# =============================================================================
# THROUGHPUT ADJUSTMENT CONFIGURATION
# =============================================================================

@dataclass
class ThroughputConfig:
    """Configuration for threshold-based token estimate correction."""
    adjustment_min_samples: int = 10        # Min samples before adjustment
    adjustment_threshold: float = 1.05      # Sensitivity threshold (5%)


# =============================================================================
# WARM-UP CALIBRATION CONFIGURATION
# =============================================================================

@dataclass
class WarmUpConfig:
    """Configuration for warm-up token calibration.

    During the first N completions, we measure actual token usage
    and latency to calibrate estimates. After calibration, Little's Law
    concurrency is recomputed with measured data.
    """
    sample_min: int = 15               # Min completions before token calibration
    sample_max: int = 30               # Max completions before forced calibration


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
    chunk_size: int = 50                    # Chunk size for specifier extraction
    max_workers: int = 10                   # Max workers for specifier extraction




# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()
DEFAULT_TOKEN_HISTORY_CONFIG = TokenHistoryConfig()
DEFAULT_TIKTOKEN_OFFSET_CONFIG = TiktokenOffsetConfig()
DEFAULT_TIMEOUT_CONFIG = TimeoutConfig()
DEFAULT_REPORTING_CONFIG = ReportingConfig()
DEFAULT_BOOTSTRAP_CONFIG = BootstrapConfig()
DEFAULT_THROUGHPUT_CONFIG = ThroughputConfig()
DEFAULT_WARM_UP_CONFIG = WarmUpConfig()
DEFAULT_CONCURRENCY_CONTROL_CONFIG = ConcurrencyControlConfig()
DEFAULT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig()
DEFAULT_PID_CONTROLLER_CONFIG = PIDControllerConfig()
DEFAULT_TPM_TRACKING_CONFIG = TPMTrackingConfig()
DEFAULT_SPECIFIER_CONFIG = SpecifierConfig()
