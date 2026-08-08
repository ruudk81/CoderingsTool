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
# IDEA EXTRACTION CONFIGURATION
# =============================================================================

@dataclass
class IdeaExtractionConfig:
    """Configuration for idea extraction step"""
    max_tokens: int = 16000
    spacy_batch_size: int = 32
    max_code_examples: int = 5  # For verbose output
    # Models per stage
    model_context: str = get_step_model("idea_extraction_context")
    model_taxonomy: str = get_step_model("idea_extraction_taxonomy")
    model_abstraction_ladder: str = get_step_model("idea_extraction_abstraction_ladder")
    temperature: float = 0.0  # Temperature for generation

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
    backoff_throughput_pct: float = 0.90   # BACKOFF = 0.9 × healthy_throughput × healthy_p50 (cooling room)
    inflight_ratio: float = 5.0           # inflight_P100/P50 > 5× → BACKOFF (after 2 consecutive ticks)
    min_concurrency: int = 2              # hard floor


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


@dataclass
class DomainDiscoveryConfig:
    """Configuration for domain discovery (phase 3).

    Domain discovery does NOT run on the specifier sample. Reading a dataset's
    properties (language, sector, perspective) from a fifth of the responses is
    sound — those do not change by reading more. Finding which THEMES exist is a
    different question: a theme that misses the draw gets no domain, and every idea
    about it then falls through to 'Other' for the whole dataset.

    Measured on the response counts in use here, for a theme carried by 1% of
    respondents, chance it appears at least twice in at least one chunk:

        4586 responses, 917 sampled (the old rule)   0.81
        4586 responses, all                          1.00
       50000 responses, 1000 sampled                 0.41
       50000 responses, all                          1.00

    So discovery reads every response, chunked with overlap — the same treatment
    step 4 gives facet and attribute discovery one level down.
    """
    chunk_size_min: int = 100          # no splitting below this (single chunk)
    chunk_size_max: int = 150          # ceiling per chunk, keeps prompt quality high
    target_chunks: int = 6             # ideal chunk count before the ceiling bites
    chunk_overlap: float = 0.2         # overlap between adjacent chunks


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

DEFAULT_IDEA_EXTRACTION_CONFIG = IdeaExtractionConfig()
DEFAULT_TOKEN_HISTORY_CONFIG = TokenHistoryConfig()
DEFAULT_TIKTOKEN_OFFSET_CONFIG = TiktokenOffsetConfig()
DEFAULT_TIMEOUT_CONFIG = TimeoutConfig()
DEFAULT_BOOTSTRAP_CONFIG = BootstrapConfig()
DEFAULT_WARM_UP_CONFIG = WarmUpConfig()
DEFAULT_CONCURRENCY_CONTROL_CONFIG = ConcurrencyControlConfig()
DEFAULT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig()
DEFAULT_PID_CONTROLLER_CONFIG = PIDControllerConfig()
DEFAULT_SPECIFIER_CONFIG = SpecifierConfig()
DEFAULT_DOMAIN_DISCOVERY_CONFIG = DomainDiscoveryConfig()
