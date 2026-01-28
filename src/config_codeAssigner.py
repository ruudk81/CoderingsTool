"""
CodeAssigner-specific configuration - separate from main config.py

This module contains all configuration constants for the CodeAssigner,
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
    default_output_ratio: float = 0.15      # Initial fallback ratio (code assignment has smaller outputs)
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
    num_probes: int = 3                     # Number of probe calls for measurement
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
    target_utilization: float = 0.80        # Target 80% TPM utilization (20% buffer)


# =============================================================================
# THROUGHPUT ADJUSTMENT CONFIGURATION
# =============================================================================

@dataclass
class ThroughputConfig:
    """Configuration for threshold-based throughput adjustment (fallback to PID)."""
    adjustment_min_samples: int = 10        # Min samples before adjustment
    adjustment_threshold: float = 1.05      # Sensitivity threshold (5%)


# =============================================================================
# ADAPTIVE CONFIDENCE THRESHOLD CONFIGURATION
# =============================================================================

@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive confidence threshold in Stage 1 evaluation.

    Instead of a fixed 0.7 threshold, this tracks the running distribution
    of confidence scores and uses percentile-based acceptance.

    Benefits:
    - Adapts to codebook complexity (fine-grained vs coarse)
    - Accounts for model uncertainty patterns
    - Reduces unnecessary Stage 2 fallbacks
    """
    use_adaptive: bool = False              # Enable adaptive threshold (False = backward compatible)
    fixed_threshold: float = 0.7            # Default fixed threshold when adaptive disabled
    adaptive_percentile: int = 25           # Accept if above this percentile of scores
    adaptive_floor: float = 0.5             # Never go below this absolute threshold
    warmup_samples: int = 20                # Min samples before adapting


# =============================================================================
# DYNAMIC TOP-K CONFIGURATION
# =============================================================================

@dataclass
class DynamicTopKConfig:
    """Configuration for dynamic similarity-based code selection in Stage 2.

    Instead of fixed top-k, this supports intelligent selection:
    - "fixed": Return exactly top_k codes (default, backward compatible)
    - "threshold": Return all codes above similarity_threshold
    - "dropoff": Return codes until similarity drops significantly

    Benefits:
    - Includes all relevant codes when similarity scores are close
    - Avoids arbitrary cutoffs that might exclude good matches
    """
    mode: str = "fixed"                     # "fixed" | "threshold" | "dropoff"
    similarity_threshold: float = 0.75      # For "threshold" mode: include codes above this
    dropoff_ratio: float = 0.85             # For "dropoff": stop when sim < best_sim * ratio
    min_codes: int = 3                      # Always include at least this many codes
    max_codes: int = 20                     # Never exceed this many codes


# =============================================================================
# PATTERN TRACKING CONFIGURATION
# =============================================================================

@dataclass
class PatternTrackingConfig:
    """Configuration for pattern learning and diagnostics.

    Tracks patterns during code assignment for post-run analysis:
    - Code co-occurrence (which codes appear together)
    - Cluster fallback rates (which clusters have poor default codes)
    - Confidence calibration (are predictions well-calibrated?)
    """
    enabled: bool = True                    # Enable pattern tracking (minimal overhead)
    track_cooccurrence: bool = True         # Track code co-occurrence patterns
    track_cluster_fallback: bool = True     # Track fallback rates per cluster
    track_confidence_calibration: bool = True  # Track confidence distribution


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
DEFAULT_ADAPTIVE_THRESHOLD_CONFIG = AdaptiveThresholdConfig()
DEFAULT_DYNAMIC_TOPK_CONFIG = DynamicTopKConfig()
DEFAULT_PATTERN_TRACKING_CONFIG = PatternTrackingConfig()
