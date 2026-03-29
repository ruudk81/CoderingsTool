"""
IdeaExtractor  - Dimension-based idea extraction with hybrid rate limiting

Extracts structured ideas from survey responses using LLM with:
- Primary dimension selection (10 MECE dimensions via decision tree, per-dataset)
- Data-driven domain discovery (5-15 domains per dimension)
- 4-layer hierarchy: Instance → Interpretation → Abstraction → Domain → Primary Dimension
- Secondary dimension: valence
- Rate limiting: state machine concurrency + PID arrival rate + circuit breaker
- Template prefix enforcement for normalized idea phrasing

Rate limiting strategy:
1. RPM: AsyncLimiter (PID-adjustable arrival rate)
2. TPM: TokenBucket (self-regulating via acquire/wait/reconcile)
3. Concurrency: ConcurrencyGate (state machine: ramp, hold, repair, recover)
4. Circuit breaker: monitors timeout RATE, adjusts concurrency on sustained pressure

Key features:
1. Learned tiktoken→API token offset (accounts for ~300 token system overhead)
2. PID arrival rate optimization (asymmetric: aggressive up, gentle down)
3. State machine concurrency: ramp gently, hold at sweet spot, repair on stress
4. Circuit breaker for concurrency (reacts to rate, not individual timeouts)
5. Live warm-up with one-shot calibration from production data
"""

# === MODULES ========================================================================================================
import asyncio
import random
import re
import time
import statistics
import itertools
import logging
import unicodedata
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import numpy as np

import nest_asyncio
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
from pipeline.step_3_ideaExtractor import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, FALLBACK_TPM, FALLBACK_RPM, get_reasoning_params
from pipeline.step_3_ideaExtractor.config_ideaExtractor import SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG
from utils.llm import create_client, llm_create_async, RateLimits, extract_rate_limits_from_response
from utils.modelPerfStats import (
    load_stats, save_stats, update_phase_stats, get_phase_stats, STATS_FILE,
    COLD_START_P95_MULTIPLIER, COLD_START_P99_MULTIPLIER, COLD_START_TIMEOUT_FACTOR,
)

# === PROMPTS (builders + response models) =========================================================================
from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import (
    build_context_specifier_group1_prompt,
    build_context_specifier_group2_prompt,
    build_consolidate_specifiers_group1_prompt,
    build_consolidate_specifiers_group2_prompt,
    build_primary_dimension_decision_tree_prompt,
    build_primary_dimension_consolidation_prompt,
    build_domain_discovery_prompt,
    build_domain_consolidation_prompt,
    build_taxonomy_enriched_extraction_prompt,
    GenericSpecifierGroup1Response,
    GenericSpecifierGroup2Response,
    PrimaryDimensionChunkResponse,
    PrimaryDimensionConsolidatedResponse,
    DomainItem,
    DomainChunkResponse,
    DomainConsolidatedResponse,
    create_extraction_model,
    consolidate_primary_dimension_by_majority,
)

# === DIMENSION DATA ===============================================================================================
from pipeline.step_3_ideaExtractor.dimension_data import get_dimension, DimensionDefinition

# === STEP-SPECIFIC CONFIG =============================================================================================
from pipeline.step_3_ideaExtractor.config_ideaExtractor import (
    DEFAULT_TOKEN_HISTORY_CONFIG,
    DEFAULT_TIKTOKEN_OFFSET_CONFIG,
    DEFAULT_TIMEOUT_CONFIG,
    DEFAULT_REPORTING_CONFIG,
    DEFAULT_BOOTSTRAP_CONFIG,
    DEFAULT_THROUGHPUT_CONFIG,
    DEFAULT_WARM_UP_CONFIG,
    DEFAULT_SPECIFIER_CONFIG,
    DEFAULT_CONCURRENCY_CONTROL_CONFIG,
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
    DEFAULT_PID_CONTROLLER_CONFIG,
    DEFAULT_TPM_TRACKING_CONFIG,
    WarmUpConfig,
)

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cached_resources import get_tiktoken_encoding



# (Helper functions _escape_braces_for_format, _resolve_slot_type, _resolve_schema_data,
#  _format_lookup_for_dimension removed — replaced by dimension_data.py + prompt_builders.py)


# === CONSTANTS (from config_ideaExtractor.py) =========================================================================
# Token history windows
INPUT_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.input_history_maxlen
OUTPUT_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.output_history_maxlen
OUTPUT_RATIO_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.output_ratio_history_maxlen
DEFAULT_OUTPUT_RATIO = DEFAULT_TOKEN_HISTORY_CONFIG.default_output_ratio
ERROR_WINDOW_SIZE = DEFAULT_TOKEN_HISTORY_CONFIG.error_window_size

# Tiktoken → API token offset learning
TIKTOKEN_API_OFFSET_DEFAULT = DEFAULT_TIKTOKEN_OFFSET_CONFIG.api_offset_default
TIKTOKEN_OFFSET_HISTORY_MAXLEN = DEFAULT_TIKTOKEN_OFFSET_CONFIG.offset_history_maxlen
TIKTOKEN_OFFSET_MIN_SAMPLES = DEFAULT_TIKTOKEN_OFFSET_CONFIG.offset_min_samples

# Timeouts and latency
TIMEOUT_FLOOR_SECONDS = DEFAULT_TIMEOUT_CONFIG.timeout_floor_seconds
DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_timeout_seconds
COLD_START_CAP = 50  # Max initial concurrency when no empirical data exists
DEFAULT_LATENCY_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_latency_seconds
MAX_TOKEN_ACQUIRE_ATTEMPTS = DEFAULT_TIMEOUT_CONFIG.max_token_acquire_attempts

# Reporting intervals
PROGRESS_REPORT_INTERVAL = DEFAULT_REPORTING_CONFIG.progress_report_interval
DIAGNOSTIC_INTERVAL = DEFAULT_REPORTING_CONFIG.diagnostic_interval
ADJUSTMENT_INTERVAL = DEFAULT_REPORTING_CONFIG.adjustment_interval

# Bootstrap settings
DEFAULT_AVG_TOKENS = DEFAULT_BOOTSTRAP_CONFIG.default_avg_tokens
SAMPLE_SIZE_FOR_TOKEN_ESTIMATION = DEFAULT_BOOTSTRAP_CONFIG.sample_size_for_token_estimation

# Threshold-based token estimate correction
THROUGHPUT_ADJUSTMENT_MIN_SAMPLES = DEFAULT_THROUGHPUT_CONFIG.adjustment_min_samples
THROUGHPUT_ADJUSTMENT_THRESHOLD = DEFAULT_THROUGHPUT_CONFIG.adjustment_threshold

# Generic specifier settings
GENERIC_SPECIFIER_SAMPLE_MIN = DEFAULT_SPECIFIER_CONFIG.sample_min
GENERIC_SPECIFIER_SAMPLE_MAX = DEFAULT_SPECIFIER_CONFIG.sample_max
GENERIC_SPECIFIER_CHUNK_SIZE = DEFAULT_SPECIFIER_CONFIG.chunk_size
MAX_SPECIFIER_WORKERS = DEFAULT_SPECIFIER_CONFIG.max_workers


# === RATE LIMITING CLASSES  ========================================================================================================
class TokenBucket:
    """Simple token bucket for TPM limiting"""
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens_needed):
        """Acquire tokens, returning wait time if not available"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            # Regenerate tokens based on time elapsed
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now

            if self.available >= tokens_needed:
                self.available -= tokens_needed
                return True
            else:
                # Calculate wait time
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
                return wait_seconds

    async def wait_and_acquire(self, tokens_needed):
        """Wait if necessary and acquire tokens"""
        attempts = 0
        while attempts < MAX_TOKEN_ACQUIRE_ATTEMPTS:
            attempts += 1
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            else:
                # result is wait_seconds
                await asyncio.sleep(result)

        raise RuntimeError(f"Failed to acquire {tokens_needed} tokens after {MAX_TOKEN_ACQUIRE_ATTEMPTS} attempts")

    async def reconcile(self, delta_tokens: int) -> None:
        """Reconcile actual token usage against estimate."""
        if delta_tokens < 0:
            async with self.lock:
                self.available = min(self.tpm, self.available - delta_tokens)


class ConcurrencyGate:
    """Concurrency limiter that tracks exact active count and supports
    both increase and decrease at runtime.

    When limit decreases: in-flight requests drain naturally (no cancellation),
    but new requests block until active < limit.
    When limit increases: blocked requests wake immediately.

    Uses asyncio futures for zero-overhead waiting (no polling/busy-wait).
    asyncio is single-threaded (cooperative), so no locks needed.
    """

    def __init__(self, limit: int):
        self._limit = limit
        self._active = 0
        self._waiters: deque = deque()  # asyncio.Future objects

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def active(self) -> int:
        return self._active

    def set_limit(self, new_limit: int):
        """Change limit (up or down). Wakes blocked waiters if room opened."""
        self._limit = max(1, new_limit)
        self._wake_waiters()

    def _wake_waiters(self):
        """Wake waiting coroutines that can now proceed."""
        while self._waiters and self._active < self._limit:
            fut = self._waiters.popleft()
            if not fut.done():
                self._active += 1
                fut.set_result(True)

    async def acquire(self):
        """Acquire a concurrency slot. Blocks if at limit."""
        if self._active < self._limit:
            self._active += 1
            return
        # At limit — create a future and wait
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # We were granted a slot but got cancelled — release it
                self._active -= 1
                self._wake_waiters()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self):
        """Release a concurrency slot and wake next waiter."""
        self._active -= 1
        self._wake_waiters()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        self.release()


class LatencyTracker:
    """EMA tracker for latencies with step-type-aware timeout strategy.

    Per strategy doc:
    - Cold start: max(timeout_floor, default_timeout) — 20s for single-processing steps
    - Adaptive: max(timeout_floor, min(P95×3, 180)) after warm-up data
    - Retry: adaptive with 60s floor (latency history is fully populated at retry time)
    """
    def __init__(self, processing_config: Optional[ProcessingConfig] = None,
                 timeout_floor: float = None, default_timeout: float = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.timeout_floor = timeout_floor if timeout_floor is not None else TIMEOUT_FLOOR_SECONDS
        self.default_timeout = default_timeout if default_timeout is not None else DEFAULT_TIMEOUT_SECONDS
        self.ema = None
        self.alpha = self.processing_config.latency_tracker_ema_alpha
        self.values = deque(maxlen=self.processing_config.latency_tracker_samples_window)
        self.retry_mode = False  # Set True for batch retry

    def add(self, value):
        """Add a latency measurement"""
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

    def get_timeout(self, est_tokens=None):
        """Safety-net timeout — only catches truly stuck requests.

        Normal latency variance (1-30s) is NOT a problem. We only want to
        catch requests that are genuinely stuck (network issues, server errors).
        Aggressive timeouts are counterproductive: they defer legitimate slow
        tasks, double API costs on retry, and trigger the circuit breaker.
        """
        if self.retry_mode:
            # Adaptive at retry time — latency history is fully populated
            if not self.values:
                return 180.0  # Edge case: no data at all
            p95 = float(np.percentile(list(self.values), 95))
            return max(60.0, min(p95 * 3.0, 180.0))

        if not self.values:
            return max(self.timeout_floor, self.default_timeout)  # Step-type-aware cold start

        # Safety net: 3× P95, bounded by floor and ceiling
        p95 = float(np.percentile(list(self.values), 95))
        return max(self.timeout_floor, min(p95 * 3.0, 180.0))

    def get_avg_latency(self):
        """Get average latency for concurrency calculations."""
        if not self.values:
            return DEFAULT_LATENCY_SECONDS
        return self.ema if self.ema is not None else DEFAULT_LATENCY_SECONDS

    def get_p50(self) -> float:
        """Return P50 (median) latency, or EMA fallback if fewer than 2 samples."""
        if len(self.values) >= 2:
            return float(np.percentile(list(self.values), 50))
        return self.get_avg_latency()

    def get_p95(self) -> float:
        """Return P95 latency, or EMA fallback if fewer than 2 samples."""
        if len(self.values) >= 2:
            return float(np.percentile(list(self.values), 95))
        return self.get_avg_latency()

    def get_p99(self) -> float:
        """Return P99 latency, or EMA fallback if fewer than 2 samples."""
        if len(self.values) >= 2:
            return float(np.percentile(list(self.values), 99))
        return self.get_avg_latency()


# === OPTIMAL STRATEGY CLASSES ========================================================================================================

class TiktokenOffsetLearner:
    """Learns the offset between tiktoken counts and actual API token counts.

    The API always reports more tokens than tiktoken because of:
    - System messages added by the API
    - Instructor/structured output overhead
    - Message formatting tokens

    This class learns the average offset and applies it to estimates.
    """
    def __init__(self, default_offset: int = TIKTOKEN_API_OFFSET_DEFAULT):
        self.default_offset = default_offset
        self.offsets = deque(maxlen=TIKTOKEN_OFFSET_HISTORY_MAXLEN)
        self._learned_offset = None

    def record(self, tiktoken_count: int, api_count: int):
        """Record a tiktoken vs API count pair to learn the offset."""
        offset = api_count - tiktoken_count
        self.offsets.append(offset)

        # Update learned offset when we have enough samples
        if len(self.offsets) >= TIKTOKEN_OFFSET_MIN_SAMPLES:
            self._learned_offset = int(sum(self.offsets) / len(self.offsets))

    def get_offset(self) -> int:
        """Get the current offset to add to tiktoken counts."""
        if self._learned_offset is not None:
            return self._learned_offset
        return self.default_offset

    def is_learned(self) -> bool:
        """Check if we have enough samples to trust the learned offset."""
        return len(self.offsets) >= TIKTOKEN_OFFSET_MIN_SAMPLES

    def get_stats(self) -> dict:
        """Get statistics about the offset learning."""
        return {
            "samples": len(self.offsets),
            "learned_offset": self._learned_offset,
            "using_offset": self.get_offset(),
            "is_learned": self.is_learned(),
            "min_offset": min(self.offsets) if self.offsets else None,
            "max_offset": max(self.offsets) if self.offsets else None,
        }



# === LITTLE'S LAW & CONCURRENCY CALCULATION ========================================================================================================
@dataclass
class ApiLimits:
    """API limits structure for Little's Law calculations"""
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(limits: ApiLimits, latency_seconds: float, avg_tokens: float,
                                 headroom: float = 0.9) -> int:
    """Compute optimal concurrency using Little's Law: N = lambda * W

    Returns the raw Little's Law number (floor 5). Caller applies additional
    headroom (e.g. * 0.90) and ramp-up strategy.
    """
    latency_seconds = max(float(latency_seconds or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)

    rpm_throughput = limits.requests_per_minute * headroom / 60
    tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
    allowed_rps = max(min(rpm_throughput, tpm_throughput), 0.0)
    target = allowed_rps * latency_seconds   # Little's Law

    return int(max(target, 5))


def detect_bottleneck(rpm_limit: int, tpm_limit: int, avg_tokens: int,
                      starting_concurrency: int) -> str:
    """Determine the binding constraint before processing starts.

    If both RPM/s and TPM/s are far above the starting concurrency,
    the bottleneck is server throughput (latency-bound), not rate limits.
    """
    rpm_per_second = rpm_limit / 60
    tpm_per_second = tpm_limit / 60 / max(avg_tokens, 1)
    api_throughput = min(rpm_per_second, tpm_per_second)

    if api_throughput > starting_concurrency * 5:
        return "throughput"
    elif rpm_per_second <= tpm_per_second:
        return "rpm"
    else:
        return "tpm"


# === CONCURRENCY STATE MACHINE =========================================================================

class ConcurrencyState(Enum):
    RAMPING = "RAMPING"
    HOLDING = "HOLDING"
    REPAIRING = "REPAIRING"
    RECOVERED = "RECOVERED"


class ConcurrencyStateMachine:
    """State machine concurrency controller.

    Monitors throughput (concurrency/P50) and interval P100 (max latency per tick).
    Ramps gently, holds at sweet spot, repairs on stress, recovers to healthy level.

    States: RAMPING → HOLDING ↔ REPAIRING → RECOVERED → HOLDING
    """

    def __init__(self, starting: int, bottleneck: str = "throughput",
                 config: 'ConcurrencyControlConfig' = None):
        from pipeline.step_3_ideaExtractor.config_ideaExtractor import DEFAULT_CONCURRENCY_CONTROL_CONFIG
        self.config = config or DEFAULT_CONCURRENCY_CONTROL_CONFIG
        self.current = starting
        self.starting = starting
        self.bottleneck = bottleneck
        self.ramp_step = max(2, int(starting * self.config.ramp_step_pct))

        self.state = ConcurrencyState.RAMPING
        self.last_healthy_concurrency = starting
        self.holding_concurrency = None

        # Consecutive tick counters for sustained ratio checks
        self.consecutive_p95_stressed = 0   # P95/P50 > holding_ratio
        self.consecutive_p100_stressed = 0  # P100/P50 > stress_ratio

        # Per-tick signal tracking
        self.interval_latencies = []

    def record_latency(self, latency: float):
        """Called from worker on each completion. Appends to interval buffer."""
        self.interval_latencies.append(latency)

    def _get_interval_p100(self) -> float:
        """Return max latency since last evaluation, then clear buffer."""
        if not self.interval_latencies:
            return 0.0
        p100 = max(self.interval_latencies)
        self.interval_latencies.clear()
        return p100

    def evaluate(self, p50: float, p95: float, now: float) -> int:
        """Main tick evaluation. Returns new concurrency.

        Two ratio signals:
          P95/P50 > holding_ratio for N consecutive ticks → HOLDING
          P100/P50 > stress_ratio for N consecutive ticks → REPAIRING
        """
        if self.bottleneck != "throughput":
            return self.current

        interval_p100 = self._get_interval_p100()
        if interval_p100 == 0 or p50 <= 0:
            return self.current

        # Compute ratios
        self.p95_ratio = p95 / p50
        self.p100_ratio = interval_p100 / p50

        # Track consecutive ticks above thresholds
        if self.p95_ratio > self.config.holding_ratio:
            self.consecutive_p95_stressed += 1
        else:
            self.consecutive_p95_stressed = 0

        if self.p100_ratio > self.config.stress_ratio:
            self.consecutive_p100_stressed += 1
        else:
            self.consecutive_p100_stressed = 0

        should_hold = self.consecutive_p95_stressed >= self.config.stress_consecutive
        should_repair = self.consecutive_p100_stressed >= self.config.stress_consecutive

        if self.state == ConcurrencyState.RAMPING:
            if should_repair:
                self.state = ConcurrencyState.REPAIRING
                self.current = max(self.config.min_concurrency,
                                   int(self.current * self.config.repair_pct))
            elif should_hold:
                self.state = ConcurrencyState.HOLDING
                self.holding_concurrency = self.current
                self.last_healthy_concurrency = self.current
            else:
                self.last_healthy_concurrency = self.current
                self.current = self.current + self.ramp_step

        elif self.state == ConcurrencyState.HOLDING:
            self.holding_concurrency = self.current
            if should_repair:
                self.state = ConcurrencyState.REPAIRING
                self.current = max(self.config.min_concurrency,
                                   int(self.current * self.config.repair_pct))
            elif not should_hold:
                # P95 ratio dropped back — resume ramping
                self.state = ConcurrencyState.RAMPING
                self.current = self.current + self.ramp_step

        elif self.state == ConcurrencyState.REPAIRING:
            if self.p100_ratio <= self.config.recovery_ratio:
                self.state = ConcurrencyState.RECOVERED

        elif self.state == ConcurrencyState.RECOVERED:
            if should_repair:
                self.state = ConcurrencyState.REPAIRING
                self.current = max(self.config.min_concurrency,
                                   int(self.current * self.config.repair_pct))
            elif should_hold:
                self.state = ConcurrencyState.HOLDING
                self.holding_concurrency = self.current
                self.last_healthy_concurrency = self.current
            elif self.current >= self.last_healthy_concurrency:
                self.state = ConcurrencyState.HOLDING
                self.current = self.last_healthy_concurrency
                self.holding_concurrency = self.current
            else:
                self.current = min(self.current + self.ramp_step,
                                   self.last_healthy_concurrency)

        return self.current

# === REAL-TIME TPM TRACKER ========================================================================================================
class RealTimeTPMTracker:
    """Tracks actual TPM usage in a sliding window for PID feedback."""
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.samples = deque()  # (timestamp, tokens) pairs
        self.lock = asyncio.Lock()

    async def record(self, tokens: int):
        async with self.lock:
            now = time.monotonic()
            self.samples.append((now, tokens))
            self._prune_old_samples(now)

    def _prune_old_samples(self, now: float):
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    async def get_current_tpm(self) -> float:
        async with self.lock:
            now = time.monotonic()
            self._prune_old_samples(now)
            if not self.samples:
                return 0.0
            total_tokens = sum(t for _, t in self.samples)
            elapsed = max(now - self.samples[0][0], 1.0)
            return (total_tokens / elapsed) * 60


# === REAL-TIME RPM TRACKER ========================================================================================================
class RealTimeRPMTracker:
    """Tracks actual RPM (requests per minute) in a sliding window for constraint visibility."""
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.samples = deque()  # timestamps of completed requests
        self.lock = asyncio.Lock()

    async def record(self):
        async with self.lock:
            now = time.monotonic()
            self.samples.append(now)
            self._prune(now)

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0] < cutoff:
            self.samples.popleft()

    async def get_current_rpm(self) -> float:
        async with self.lock:
            now = time.monotonic()
            self._prune(now)
            if not self.samples:
                return 0.0
            elapsed = max(now - self.samples[0], 1.0)
            return (len(self.samples) / elapsed) * 60


# === PID THROUGHPUT CONTROLLER ========================================================================================================
class PIDThroughputController:
    """PID controller for smooth arrival rate adjustment.

    Asymmetric gains: aggressive when under-utilizing, gentle when over-utilizing.
    Adjusts the arrival rate (requests/second) based on real-time TPM utilization.
    """
    def __init__(self, target_utilization: float = 0.80,
                 kp_up: float = 0.4, kp_down: float = 0.2,
                 ki: float = 0.05, kd: float = 0.1,
                 min_adjustment: float = 0.02, max_adjustment: float = 0.15):
        self.target = target_utilization
        self.kp_up = kp_up
        self.kp_down = kp_down
        self.ki = ki
        self.kd = kd
        self.min_adjustment = min_adjustment
        self.max_adjustment = max_adjustment
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.adjustment_history = deque(maxlen=20)

    def compute_adjustment(self, current_utilization: float) -> float:
        """Returns multiplier for arrival rate. >1.0 = speed up, <1.0 = slow down."""
        now = time.monotonic()
        error = self.target - current_utilization

        dt = 1.0
        if self.last_time is not None:
            dt = max(now - self.last_time, 0.1)
        self.last_time = now

        self.integral += error * dt
        self.integral = max(-0.5, min(0.5, self.integral))

        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        kp = self.kp_up if error > 0 else self.kp_down
        output = (kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        output = max(-self.max_adjustment, min(self.max_adjustment, output))

        if abs(output) < self.min_adjustment:
            adjustment = 1.0
        else:
            adjustment = 1.0 + output

        self.adjustment_history.append({
            "time": now, "utilization": current_utilization,
            "error": error, "output": output, "adjustment": adjustment
        })
        return adjustment

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


# === CONCURRENCY CIRCUIT BREAKER ========================================================================================================
class ConcurrencyCircuitBreaker:
    """Monitors timeout rate in sliding window. Only adjusts concurrency on sustained pressure.

    State machine:
      CLOSED     — Normal operation. Monitoring timeout rate.
      OPEN       — Tripped. Concurrency reduced. In cooldown (no further changes).
      RECOVERING — Cooldown expired, rate OK. Gradually ramping back to baseline.

    Individual timeouts are invisible — tenacity retries them.
    Only the RATE of timeouts in the window triggers action.
    """

    def __init__(self, config, gate: ConcurrencyGate, baseline: int):
        self.config = config
        self.gate = gate
        self.baseline = baseline
        self._events: deque = deque()  # (timestamp, 'ok'|'timeout')
        self._state = 'CLOSED'
        self._last_trip_time: Optional[float] = None
        self._last_recovery_check: Optional[float] = None
        self._trip_count: int = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def trip_count(self) -> int:
        return self._trip_count

    def record_completion(self):
        self._events.append((time.monotonic(), 'ok'))
        self._prune_window()

    def record_timeout(self):
        self._events.append((time.monotonic(), 'timeout'))
        self._prune_window()

    def _prune_window(self):
        cutoff = time.monotonic() - self.config.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _get_timeout_rate(self) -> Tuple[float, int]:
        """Returns (timeout_rate, total_events) in current window."""
        self._prune_window()
        total = len(self._events)
        if total == 0:
            return 0.0, 0
        timeouts = sum(1 for _, t in self._events if t == 'timeout')
        return timeouts / total, total

    def check_and_adjust(self) -> Optional[str]:
        """Called every 1s. Returns 'tripped', 'recovering', 'recovered', or None."""
        now = time.monotonic()
        rate, total = self._get_timeout_rate()

        if self._state == 'CLOSED':
            if total >= self.config.min_events_to_trip and rate > self.config.trip_threshold:
                return self._trip(now, rate, total)
            return None

        elif self._state == 'OPEN':
            elapsed = now - self._last_trip_time if self._last_trip_time else 0
            if elapsed < self.config.cooldown_seconds:
                return None  # Still in cooldown
            # Cooldown expired — check if still bad
            if total >= self.config.min_events_to_trip and rate > self.config.trip_threshold:
                return self._trip(now, rate, total)
            # Rate normalized — enter recovery
            self._state = 'RECOVERING'
            self._last_recovery_check = now
            return 'recovering'

        elif self._state == 'RECOVERING':
            if now - (self._last_recovery_check or now) < self.config.recovery_interval_seconds:
                return None
            self._last_recovery_check = now
            # Check if rate spiked again
            if total >= self.config.min_events_to_trip and rate > self.config.trip_threshold:
                return self._trip(now, rate, total)
            # Rate is good — step up toward baseline
            current = self.gate.limit
            target = min(self.baseline, int(current * (1.0 + self.config.recovery_step_pct)))
            target = max(target, current + 1)  # At least +1
            if target >= self.baseline:
                self.gate.set_limit(self.baseline)
                self._state = 'CLOSED'
                self._trip_count = 0
                print(f"✅ Circuit breaker recovered: concurrency restored to {self.baseline}")
                return 'recovered'
            self.gate.set_limit(target)
            print(f"📈 Circuit breaker recovering: {current} → {target} (target: {self.baseline})")
            return 'recovering'

        return None

    def _trip(self, now: float, rate: float, total: int) -> str:
        pre_trip = self.gate.limit
        new_limit = max(self.config.min_concurrency,
                        int(self.gate.limit * self.config.reduction_factor))
        self.gate.set_limit(new_limit)
        self._state = 'OPEN'
        self._last_trip_time = now
        self._trip_count += 1
        print(f"⚡ CIRCUIT BREAKER TRIPPED: timeout rate {rate:.1%} "
              f"({total} events in {self.config.window_seconds}s) | "
              f"concurrency {pre_trip} → {new_limit} "
              f"(cooldown {self.config.cooldown_seconds}s)")
        return 'tripped'


# === MAIN IDEA EXTRACTOR CLASS ========================================================================================================
class IdeaExtractor:
    def __init__(
        self,
        responses: List[models.QualityFilteredModel],
        var_lab: str,
        config: Optional[SegmentationConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        verbose: bool = False,
        prompt_printer=None,
        verbose_reporter: Optional['VerboseReporter'] = None,
        discover_domains: bool = False):

        self.responses = responses
        self.var_lab = var_lab
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.model_config = model_config or ModelConfig()  # kept for backward compat
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.warm_up_config = DEFAULT_WARM_UP_CONFIG
        self.model = self.config.model
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.IdeasExtractedModel] = []
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        # Capture flags for each prompt type (only capture first instance)
        self._captured_context_specifier1 = False
        self._captured_context_specifier2 = False
        self._captured_taxonomy_chunk = False
        self._captured_consolidate1 = False
        self._captured_consolidate2 = False
        self._captured_taxonomy_consolidation = False
        self._captured_domain_chunk = False
        self._captured_domain_consolidation = False
        # Initialize tokenizer for token estimation (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Initialize OpenAI client with instructor (supports OpenAI and Azure)
        self.client = create_client(self.model, async_mode=True)

        # Rate limiting setup - use fallback values for initial setup
        self.rate_limits = RateLimits(
            tokens_per_minute=FALLBACK_TPM,
            requests_per_minute=FALLBACK_RPM,
            tokens_per_day=FALLBACK_TPM * 60 * 24
        )

        # Token bucket for TPM limiting
        self.tpm_bucket = TokenBucket(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

        # Adaptive token estimation
        self.input_token_history = deque(maxlen=INPUT_HISTORY_MAXLEN)
        self.output_token_history = deque(maxlen=OUTPUT_HISTORY_MAXLEN)
        self.output_ratio_history = deque(maxlen=OUTPUT_RATIO_HISTORY_MAXLEN)  # Track output/input ratios
        self.estimation_errors = deque(maxlen=ERROR_WINDOW_SIZE)

        # Rolling average of actual total tokens
        self.actual_total_tokens = deque(maxlen=ERROR_WINDOW_SIZE)

        # Load persistent performance stats for cold-start calibration
        self._perf_stats = load_stats()
        _stored = get_phase_stats(self._perf_stats, self.model, "step3_idea_extraction")
        if _stored and _stored.get("sample_count", 0) >= 10:
            if "p99_latency_s" in _stored:
                _p99 = _stored["p99_latency_s"]
                _timeout_factor = COLD_START_TIMEOUT_FACTOR if _stored.get("had_timeouts") else 0
                _stored_timeout = max(COLD_START_P99_MULTIPLIER * _p99, _timeout_factor * _p99)
            elif "p95_latency_s" in _stored:
                _stored_timeout = _stored["p95_latency_s"] * COLD_START_P95_MULTIPLIER
            else:
                _stored_timeout = None
        else:
            _stored_timeout = None
        _stored_tiktoken_offset = (
            int(_stored["tiktoken_offset"])
            if _stored and _stored.get("sample_count", 0) >= 10 and "tiktoken_offset" in _stored
            else None
        )
        self._stored_empirical_capacity = (
            _stored["empirical_capacity"]
            if _stored and _stored.get("sample_count", 0) >= 10 and "empirical_capacity" in _stored
            else None
        )
        self._stored_p50 = (
            _stored["p50_latency_s"]
            if _stored and _stored.get("sample_count", 0) >= 10 and "p50_latency_s" in _stored
            else None
        )

        # Latency tracking (use stored P95 as cold-start floor if available)
        self.latency_tracker = LatencyTracker(
            processing_config=self.processing_config,
            timeout_floor=_stored_timeout if _stored_timeout else TIMEOUT_FLOOR_SECONDS,
            default_timeout=_stored_timeout if _stored_timeout else DEFAULT_TIMEOUT_SECONDS,
        )

        # Generic specifiers (must be initialized before _calculate_avg_tokens)
        self.generic_specifiers = {}

        # Taxonomy dimension (must be initialized before _calculate_avg_tokens)
        self.primary_dimension = None
        self.primary_dimension_rationale = None
        self.primary_dimension_description = None  # Dynamic context-specific description
        self.decision_tree_stop_position = 0   # Which decision tree step triggered selection
        # Template prefix for embedding (V3: restored for normalized clustering)
        self.template_prefix = None

        # Phase 3 toggle: True = discover domains upfront; False = on-the-fly
        self.discover_domains = discover_domains

        # Calculate initial average tokens estimate
        self.avg_tokens = self._calculate_avg_tokens()

        # Rate limiting components (will be initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None

        # === RATE LIMITING COMPONENTS ===
        # Tiktoken→API offset learning (accounts for system overhead)
        # Seed default_offset from stored stats so warm-up starts from empirical value
        _tiktoken_default = _stored_tiktoken_offset if _stored_tiktoken_offset is not None else TIKTOKEN_API_OFFSET_DEFAULT
        self.tiktoken_offset_learner = TiktokenOffsetLearner(default_offset=_tiktoken_default)

        # Config for new rate limiting components
        self.concurrency_control_config = DEFAULT_CONCURRENCY_CONTROL_CONFIG
        self.circuit_breaker_config = DEFAULT_CIRCUIT_BREAKER_CONFIG
        self.pid_config = DEFAULT_PID_CONTROLLER_CONFIG
        self.tpm_tracking_config = DEFAULT_TPM_TRACKING_CONFIG

        # Components initialized in _initialize_rate_limiters()
        self.circuit_breaker = None
        self.tpm_tracker = None
        self.rpm_tracker = None
        self.pid_controller = None
        self.current_arrival_rate = None

        # Concurrency state machine
        self._concurrency_sm = None  # Initialized in _initialize_rate_limiters

        # Initial avg_tokens preserved for diagnostics (set from tiktoken Phase 4, never updated)
        self.bootstrap_avg_tokens = None

        # Stats tracking
        self.v3_stats = {
            'adjustments_made': 0,
            'threshold_adjustments': 0,
            'pid_adjustments': 0,
            'max_tpm_utilization': 0.0,
            'min_tpm_utilization': 100.0,
            'circuit_breaker_trips': 0,
        }

        # Stats
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'retries': 0,
            'rate_limits': 0,
            'timeouts': 0
        }

        # Failure tracking: set for O(1) lookup + list for detailed reporting
        self.failed_task_ids: set = set()
        self.failure_log = []  # List of {respondent_id, reason, error_type, response_preview}

    def _calculate_avg_tokens(self) -> int:
        """Calculate average tokens per request for rate limiting.

        V3: Uses placeholder template values for estimation.
        """
        if not self.responses:
            return DEFAULT_AVG_TOKENS

        sample_size = min(SAMPLE_SIZE_FOR_TOKEN_ESTIMATION, len(self.responses))
        sample_responses = self.responses[:sample_size]

        # Store original values to restore after estimation
        original_primary_dimension = self.primary_dimension
        original_primary_dimension_description = self.primary_dimension_description
        original_generic_specifiers = self.generic_specifiers

        # Set placeholder values for token estimation
        self.primary_dimension = "ATTRIBUTES_ASSOCIATIONS"
        self.primary_dimension_description = "general concepts and ideas"
        self.generic_specifiers = {
            "lang": "nl-NL",
            "perspective": "consumer",
            "intent": "evaluate",
            "domain": "general",
            "topic": "feedback",
            "entity": "unknown",
        }

        # Placeholder values for template estimation
        placeholder_subject = "the subject"
        placeholder_phrasing_template = "the subject is [ACTIONABLE_TAXONOMY_DIMENSION]"

        token_counts = []
        for response in sample_responses:
            prompt = self._build_taxonomy_enriched_prompt(
                response.response,
                placeholder_subject,
                placeholder_phrasing_template
            )
            prompt_tokens = len(self.encoding.encode(prompt))
            completion_tokens = int(prompt_tokens * 0.25)
            token_counts.append(prompt_tokens + completion_tokens)

        # Restore original values
        self.primary_dimension = original_primary_dimension
        self.primary_dimension_description = original_primary_dimension_description
        self.generic_specifiers = original_generic_specifiers

        return int(statistics.mean(token_counts)) if token_counts else DEFAULT_AVG_TOKENS

    async def _consolidate_specifiers(self, group: int, chunk_results: List[Dict]) -> Dict[str, str]:
        """Consolidate specifier results from multiple chunks using LLM."""
        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            group_name = "Group1 (lang/perspective/intent)" if group == 1 else "Group2 (domain/topic/entity)"
            self.verbose_reporter.stat_line(f"  Consolidating {len(chunk_results)} {group_name} results via LLM...")

        formatted_results = []
        for idx, result in enumerate(chunk_results, 1):
            response_obj = result['response']
            if group == 1:
                formatted_results.append(
                    f"Chunk {idx}:\n"
                    f"  - lang: {response_obj.lang}\n"
                    f"  - perspective: {response_obj.perspective}\n"
                    f"  - intent: {response_obj.intent}"
                )
            else:
                formatted_results.append(
                    f"Chunk {idx}:\n"
                    f"  - sector: {response_obj.sector}\n"
                    f"  - topic: {response_obj.topic}\n"
                    f"  - entity: {response_obj.entity}"
                )

        chunk_results_text = "\n\n".join(formatted_results)

        if group == 1:
            prompt = build_consolidate_specifiers_group1_prompt(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text,
            )
            response_model = GenericSpecifierGroup1Response
            if self.prompt_printer and not self._captured_consolidate1:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_consolidate_specifiers",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="consolidate_specifiers_group1",
                    metadata={"model": self.model, "survey_question": self.var_lab}
                )
                self._captured_consolidate1 = True
        else:
            prompt = build_consolidate_specifiers_group2_prompt(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text,
            )
            response_model = GenericSpecifierGroup2Response
            if self.prompt_printer and not self._captured_consolidate2:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_consolidate_specifiers",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="consolidate_specifiers_group2",
                    metadata={"model": self.model, "survey_question": self.var_lab}
                )
                self._captured_consolidate2 = True

        est_tokens = self._estimate_preprocessed_tokens(prompt)
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=response_model,
                prompt=prompt,
                temperature=0.0,
                **get_reasoning_params(self.model),
            )

        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            if group == 1:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: lang={response.lang}, perspective={response.perspective}, intent={response.intent}"
                )
            else:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: sector={response.sector}, topic={response.topic}, entity={response.entity}"
                )

        if group == 1:
            return {
                "lang": response.lang,
                "perspective": response.perspective,
                "intent": response.intent
            }
        else:
            return {
                "domain": response.sector,
                "topic": response.topic,
                "entity": response.entity
            }

    async def _consolidate_primary_dimension(
        self,
        chunk_results: List[Dict],
        context_specifiers: Dict,
        sample_responses: Optional[List] = None,
    ) -> PrimaryDimensionConsolidatedResponse:
        """Consolidate primary dimension selection from chunks.

        Uses majority rule when >50% of chunks agree. Falls back to LLM
        consolidation with actual response data when there is no majority.

        Args:
            chunk_results: List of dicts with 'response' containing PrimaryDimensionChunkResponse
            context_specifiers: Dict with domain, entity, topic, perspective, intent
            sample_responses: Response objects for tie-breaking (used when no majority)

        Returns:
            PrimaryDimensionConsolidatedResponse with selected dimension and description
        """
        # Try majority rule first
        chunk_response_objects = [r['response'] for r in chunk_results]
        majority_result = consolidate_primary_dimension_by_majority(chunk_response_objects)

        if majority_result is not None:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Primary dimension: majority rule -> {majority_result.primary_dimension}")
                self.verbose_reporter.stat_line(f"    {majority_result.primary_dimension_rationale}")
            return majority_result

        # No majority — run LLM consolidation with response data for grounding
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Primary dimension: no majority, running LLM consolidation with response sample")

        # Format chunk results for consolidation prompt
        formatted_results = []
        for idx, result in enumerate(chunk_results):
            chunk_response = result['response']
            evidence_text = "\n".join([f'    - "{e}"' for e in chunk_response.evidence])
            stop_pos = getattr(chunk_response, 'decision_tree_stop_position', 0)
            chunk_text = f"""Chunk {idx + 1}:
  Primary dimension: {chunk_response.primary_dimension} (decision tree step {stop_pos})
  Evidence:
{evidence_text}
  Clarification: {chunk_response.clarification}"""
            formatted_results.append(chunk_text)

        # Build response sample for tie-breaking
        chunk_responses_text = ""
        if sample_responses:
            grounding_sample = random.sample(
                sample_responses,
                min(GENERIC_SPECIFIER_CHUNK_SIZE, len(sample_responses))
            )
            chunk_responses_text = "\n".join([f"- {r.response}" for r in grounding_sample])

        prompt = build_primary_dimension_consolidation_prompt(
            language=self.language,
            survey_question=self.var_lab,
            sector=context_specifiers['domain'],
            entity=context_specifiers['entity'],
            topic=context_specifiers['topic'],
            perspective=context_specifiers['perspective'],
            intent=context_specifiers['intent'],
            chunk_results="\n\n".join(formatted_results),
            chunk_responses=chunk_responses_text,
        )

        # Capture first taxonomy consolidation prompt
        if self.prompt_printer and not self._captured_taxonomy_consolidation:
            self.prompt_printer.capture_prompt(
                step_name="idea_extraction_taxonomy_consolidation",
                utility_name="IdeaExtractor",
                prompt_content=prompt,
                prompt_type="dimension_consolidation",
                metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
            )
            self._captured_taxonomy_consolidation = True

        est_tokens = self._estimate_preprocessed_tokens(prompt)
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=PrimaryDimensionConsolidatedResponse,
                prompt=prompt,
                temperature=0.0,
                **get_reasoning_params(self.model),
            )

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Taxonomy consolidated (LLM):")
            self.verbose_reporter.stat_line(f"    Primary: {response.primary_dimension}")
            self.verbose_reporter.stat_line(f"    Rationale: {response.primary_dimension_rationale[:100]}...")

        return response

    async def _consolidate_domains(self, chunk_results: List[Dict], context_specifiers: Dict) -> DomainConsolidatedResponse:
        """Consolidate chunk-level domain discoveries into a single set."""
        dimension = get_dimension(self.primary_dimension)

        # Format chunk results for the consolidation prompt
        formatted_results = []
        for idx, result in enumerate(chunk_results):
            chunk_response = result['response']
            cats_text = "\n".join([
                f'    - {c.key}: "{c.label}" — {c.definition}'
                for c in chunk_response.domains
            ])
            chunk_text = f"""Chunk {idx + 1}:
  Domains:
{cats_text}"""
            formatted_results.append(chunk_text)

        prompt = build_domain_consolidation_prompt(
            language=self.language,
            survey_question=self.var_lab,
            sector=context_specifiers['domain'],
            entity=context_specifiers['entity'],
            topic=context_specifiers['topic'],
            perspective=context_specifiers['perspective'],
            intent=context_specifiers['intent'],
            primary_dimension=self.primary_dimension,
            chunk_results="\n\n".join(formatted_results),
            domain_diagnostic=dimension.prompt_rules.domain_diagnostic,
        )

        if self.prompt_printer and not self._captured_domain_consolidation:
            self.prompt_printer.capture_prompt(
                step_name="idea_extraction_domains",
                utility_name="IdeaExtractor",
                prompt_content=prompt,
                prompt_type="domain_consolidation",
                metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
            )
            self._captured_domain_consolidation = True

        est_tokens = self._estimate_preprocessed_tokens(prompt)
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=DomainConsolidatedResponse,
                prompt=prompt,
                temperature=0.0,
                **get_reasoning_params(self.model),
            )

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Domains consolidated:")
            for c in response.domains:
                self.verbose_reporter.stat_line(f"    {c.key}: {c.label}")

        return response

    async def _extract_generic_specifiers(self) -> Tuple[Dict[str, str], PrimaryDimensionConsolidatedResponse, DomainConsolidatedResponse]:
        """Extract context specifiers first, then primary dimension with context awareness, then domains.

        Two-phase extraction:
        - Phase 1: Extract context specifiers (Group 1 + Group 2) in parallel
        - Phase 2: Extract taxonomy axis scoring with context specifiers available

        Returns:
            Tuple of (context_specifiers dict, PrimaryDimensionConsolidatedResponse, DomainConsolidatedResponse)
        """
        sample_size = min(GENERIC_SPECIFIER_SAMPLE_MAX, max(int(0.2 * len(self.responses)), GENERIC_SPECIFIER_SAMPLE_MIN))
        sample = random.sample(self.responses, min(sample_size, len(self.responses)))

        chunk_size = GENERIC_SPECIFIER_CHUNK_SIZE
        chunks = [sample[i:i+chunk_size] for i in range(0, len(sample), chunk_size)]
        chunk_texts = ["\n".join([f"- {r.response}" for r in chunk]) for chunk in chunks]

        self.verbose_reporter.stat_line(f"Context + Taxonomy: {len(sample)} samples, {len(chunks)} chunks")

        # === PHASE 1: Context specifiers (parallel) ===
        self.verbose_reporter.stat_line(f"  Phase 1: Extracting context specifiers...")

        context_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            # Group 1: lang/perspective/intent
            context_tasks.append({
                'task_id': f"group1_chunk{chunk_idx}",
                'group': 1,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk)
            })
            # Group 2: domain/topic/entity
            context_tasks.append({
                'task_id': f"group2_chunk{chunk_idx}",
                'group': 2,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk)
            })

        context_results = await self._process_generic_specifier_tasks(context_tasks)

        group1_results = [r for r in context_results if r['group'] == 1]
        group2_results = [r for r in context_results if r['group'] == 2]

        if self.verbose_reporter.enabled and group1_results and group2_results:
            self.verbose_reporter.stat_line(f"  Phase 1 chunk-level results:")
            for r in group1_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group1): "
                    f"lang={r['response'].lang}, perspective={r['response'].perspective}, intent={r['response'].intent}"
                )
            for r in group2_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group2): "
                    f"sector={r['response'].sector}, topic={r['response'].topic}, entity={r['response'].entity}"
                )

        # Hard failure if context specifier extraction produced no results
        if not group1_results or not group2_results:
            raise RuntimeError(
                f"Context specifier extraction failed: "
                f"{len(group1_results)} group1 results, {len(group2_results)} group2 results. "
                f"Cannot proceed without context specifiers. "
                f"Check LLM connectivity, rate limits, and model availability."
            )

        # Consolidate Group 1
        if len(group1_results) == 1:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Single chunk - skipping consolidation for Group1")
            group1_consolidated = {
                "lang": group1_results[0]['response'].lang,
                "perspective": group1_results[0]['response'].perspective,
                "intent": group1_results[0]['response'].intent
            }
        else:
            group1_consolidated = await self._consolidate_specifiers(1, group1_results)

        # Consolidate Group 2
        if len(group2_results) == 1:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Single chunk - skipping consolidation for Group2")
            group2_consolidated = {
                "domain": group2_results[0]['response'].sector,
                "topic": group2_results[0]['response'].topic,
                "entity": group2_results[0]['response'].entity
            }
        else:
            group2_consolidated = await self._consolidate_specifiers(2, group2_results)

        context_specifiers = {**group1_consolidated, **group2_consolidated}
        self.verbose_reporter.stat_line(f"  Phase 1 complete. Context specifiers: {context_specifiers}")

        # === PHASE 2: Taxonomy scoring (with context awareness) ===
        self.verbose_reporter.stat_line(f"  Phase 2: Scoring taxonomy axes with context (perspective={context_specifiers.get('perspective')}, intent={context_specifiers.get('intent')})...")

        taxonomy_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            taxonomy_tasks.append({
                'task_id': f"taxonomy_chunk{chunk_idx}",
                'group': 3,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk),
                'context_specifiers': context_specifiers  # Pass context to taxonomy scoring
            })

        taxonomy_results = await self._process_generic_specifier_tasks(taxonomy_tasks)

        if self.verbose_reporter.enabled and taxonomy_results:
            self.verbose_reporter.stat_line(f"  Phase 2 chunk-level results:")
            for r in taxonomy_results:
                chunk_resp = r['response']
                stop_pos = getattr(chunk_resp, 'decision_tree_stop_position', '?')
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Taxonomy): Dimension={chunk_resp.primary_dimension} (tree step {stop_pos})"
                )

        # Consolidate Taxonomy — hard failure if no results
        if not taxonomy_results:
            raise RuntimeError(
                "Primary dimension selection produced no results from any chunk. "
                "Check LLM connectivity, rate limits, and model availability."
            )
        taxonomy_consolidated = await self._consolidate_primary_dimension(taxonomy_results, context_specifiers, sample_responses=sample)

        self.verbose_reporter.stat_line(f"  Context results: {context_specifiers}")
        self.verbose_reporter.stat_line(f"  Taxonomy: primary={taxonomy_consolidated.primary_dimension}")

        # Set primary dimension early so Phase 3 domain discovery can use it
        self.primary_dimension = taxonomy_consolidated.primary_dimension
        self.primary_dimension_description = taxonomy_consolidated.primary_dimension_description
        # Capture the most common decision tree stop position from chunks
        stop_positions = [
            getattr(r['response'], 'decision_tree_stop_position', 0)
            for r in taxonomy_results
        ]
        self.decision_tree_stop_position = max(set(stop_positions), key=stop_positions.count) if stop_positions else 0

        # === PHASE 3: Domain discovery (optional) ===
        if self.discover_domains:
            self.verbose_reporter.stat_line(f"  Phase 3: Discovering domains from response data...")

            category_tasks = []
            for chunk_idx, chunk in enumerate(chunks):
                category_tasks.append({
                    'task_id': f"topical_cat_chunk{chunk_idx}",
                    'group': 4,
                    'chunk_idx': chunk_idx,
                    'chunk_text': chunk_texts[chunk_idx],
                    'chunk_size': len(chunk),
                    'context_specifiers': context_specifiers
                })

            category_results = await self._process_generic_specifier_tasks(category_tasks)

            if self.verbose_reporter.enabled and category_results:
                self.verbose_reporter.stat_line(f"  Phase 3 chunk-level results:")
                for r in category_results:
                    chunk_resp = r['response']
                    cat_keys = [c.key for c in chunk_resp.domains]
                    self.verbose_reporter.stat_line(
                        f"    Chunk {r['chunk_idx']+1}: {len(cat_keys)} domains: {cat_keys}"
                    )

            # Consolidate domains — hard failure if no results
            if not category_results:
                raise RuntimeError(
                    "Domain discovery produced no results from any chunk. "
                    "Check LLM connectivity, rate limits, and model availability."
                )
            elif len(category_results) == 1:
                # Single chunk — use directly
                categories_consolidated = DomainConsolidatedResponse(
                    domains=category_results[0]['response'].domains
                )
            else:
                categories_consolidated = await self._consolidate_domains(category_results, context_specifiers)

            self.verbose_reporter.stat_line(f"  Domains: {[c.key for c in categories_consolidated.domains]}")
        else:
            self.verbose_reporter.stat_line(f"  Phase 3: Skipped (on-the-fly domains)")
            categories_consolidated = DomainConsolidatedResponse(domains=[])

        return context_specifiers, taxonomy_consolidated, categories_consolidated

    async def _process_generic_specifier_tasks(self, tasks: List[Dict]) -> List[Dict]:
        queue = asyncio.Queue()
        results = []

        for task in tasks:
            await queue.put(task)

        num_workers = min(MAX_SPECIFIER_WORKERS, len(tasks))
        for _ in range(num_workers):
            await queue.put(None)

        workers = [
            asyncio.create_task(self._generic_specifier_worker(queue, results))
            for _ in range(num_workers)
        ]

        await asyncio.gather(*workers)

        return results

    async def _generic_specifier_worker(self, queue: asyncio.Queue, results: List):
        """Worker for processing generic specifier AND taxonomy tasks."""
        while True:
            task = await queue.get()
            if task is None:
                break

            try:
                if self.semaphore is None or self.rate_limiter is None:
                    raise RuntimeError(
                        f"Rate limiters not initialized before worker started. "
                        f"semaphore={self.semaphore}, rate_limiter={self.rate_limiter}"
                    )

                # === Build prompt BEFORE acquiring rate limit resources ===
                # Prompts are pure string construction — no I/O, no semaphore needed.
                if task['group'] == 1:
                    # Group 1: lang/perspective/intent
                    prompt = build_context_specifier_group1_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                    )
                    response_model = GenericSpecifierGroup1Response
                    if self.prompt_printer and not self._captured_context_specifier1:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_context_specifiers",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="context_specifier_group1",
                            metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
                        )
                        self._captured_context_specifier1 = True
                elif task['group'] == 2:
                    # Group 2: domain/topic/entity
                    prompt = build_context_specifier_group2_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                    )
                    response_model = GenericSpecifierGroup2Response
                    if self.prompt_printer and not self._captured_context_specifier2:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_context_specifiers",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="context_specifier_group2",
                            metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
                        )
                        self._captured_context_specifier2 = True
                elif task['group'] == 3:
                    # Group 3: Taxonomy dimension selection (decision tree, context-aware)
                    ctx = task['context_specifiers']
                    prompt = build_primary_dimension_decision_tree_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                        perspective=ctx['perspective'],
                        intent=ctx['intent'],
                        sector=ctx['domain'],
                        entity=ctx['entity'],
                        topic=ctx['topic'],
                    )
                    response_model = PrimaryDimensionChunkResponse
                    if self.prompt_printer and not self._captured_taxonomy_chunk:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_taxonomy_chunk",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="dimension_chunk_decision_tree",
                            metadata={
                                "model": self.model,
                                "survey_question": self.var_lab,
                                "language": self.language,
                                "perspective": ctx['perspective'],
                                "intent": ctx['intent'],
                            }
                        )
                        self._captured_taxonomy_chunk = True
                else:  # group == 4: Domain discovery
                    ctx = task['context_specifiers']

                    dimension = get_dimension(self.primary_dimension)
                    prompt = build_domain_discovery_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                        perspective=ctx['perspective'],
                        intent=ctx['intent'],
                        sector=ctx['domain'],
                        entity=ctx['entity'],
                        topic=ctx['topic'],
                        primary_dimension=self.primary_dimension,
                        primary_dimension_description=self.primary_dimension_description,
                        domain_diagnostic=dimension.prompt_rules.domain_diagnostic,
                        domain_instruction=dimension.prompt_rules.domain_instruction,
                    )
                    response_model = DomainChunkResponse
                    if self.prompt_printer and not self._captured_domain_chunk:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_domains",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="domain_chunk",
                            metadata={"model": self.model, "survey_question": self.var_lab, "language": self.language}
                        )
                        self._captured_domain_chunk = True

                # === Count tokens from actual prompt, then acquire ===
                est_tokens = self._estimate_preprocessed_tokens(prompt)

                async with self.semaphore:
                    await self.tpm_bucket.wait_and_acquire(est_tokens)
                    await self.rate_limiter.acquire()

                    response = await llm_create_async(
                        client=self.client,
                        model=self.model,
                        response_model=response_model,
                        prompt=prompt,
                        temperature=0.0,
                        **get_reasoning_params(self.model),
                    )

                    results.append({
                        'task_id': task['task_id'],
                        'group': task['group'],
                        'chunk_idx': task['chunk_idx'],
                        'response': response
                    })

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Generic specifier task {task['task_id']} failed: {e}", exc_info=True)
                self.verbose_reporter.stat_line(f"Generic specifier task {task['task_id']} failed: {e}")
            finally:
                queue.task_done()

    def _build_canonical_phrasing(self, primary_dimension: str) -> tuple:
        """Build canonical_term and canonical_phrasing programmatically.

        Replaces the LLM-based subject extraction -- the anchor slot is always
        the entity, so we just do a string substitution.

        Returns:
            (canonical_term, canonical_phrasing)
        """
        dimension = get_dimension(primary_dimension)
        entity = self.generic_specifiers['entity']
        canonical_term = entity.replace("_", " ").title()
        canonical_phrasing = dimension.pattern.replace("[ANCHOR_SUBJECT]", canonical_term)
        return canonical_term, canonical_phrasing

    def _build_taxonomy_enriched_prompt(
        self,
        response: str,
        subject: str,
        phrasing_template: str,
    ) -> str:
        """Build taxonomy-enriched prompt for idea extraction.

        Args:
            response: The response text to extract ideas from
            subject: The canonical subject/entity from survey question
            phrasing_template: Template with domain marker placeholder

        Returns:
            Formatted prompt string
        """
        assert self.primary_dimension is not None, "primary_dimension must be set before building extraction prompt"
        dimension = get_dimension(self.primary_dimension)

        # Build domain table from discovered thematic domains
        discovered_domains = getattr(self, 'domains', None)
        if discovered_domains:
            domain_table = (
                "- Choose the most specific applicable domain from this predefined set; otherwise select Other:\n"
                + "\n".join(
                    f"  • {c.label} = \"{c.definition}\"" for c in discovered_domains
                )
                + '\n  Other = "Does not fit any of the above thematic domains"'
            )
        else:
            # During token estimation (_calculate_avg_tokens), domains haven't been
            # discovered yet — use a placeholder table for sizing purposes.
            domain_table = "- (domains will be discovered during extraction)"

        return build_taxonomy_enriched_extraction_prompt(
            language=self.language,
            var_lab=self.var_lab,
            perspective=self.generic_specifiers['perspective'],
            sector=self.generic_specifiers['domain'],
            entity=self.generic_specifiers['entity'],
            topic=self.generic_specifiers['topic'],
            intent=self.generic_specifiers['intent'],
            response=response,
            canonical_phrasing=phrasing_template,
            dimension=dimension,
            domain_table=domain_table,
        )

    def estimate_tokens(self, prompt: str) -> int:
        """V3: Estimate total tokens using optimal adaptive strategy.

        V3 Improvements:
        - Applies learned tiktoken→API offset upfront
        - Reduced safety margins (offset handles the gap)
        - Faster convergence with smaller margins when learned
        """
        # Count tokens with tiktoken
        tiktoken_count = len(self.encoding.encode(prompt))

        # V3: Apply learned offset (accounts for system overhead)
        offset = self.tiktoken_offset_learner.get_offset()
        actual_input_tokens = tiktoken_count + offset

        # V3: Reduced safety margins (offset already accounts for gap)
        num_samples = len(self.estimation_errors)
        if num_samples < 5:
            safety_margin = 1.15  # V3: Reduced from 1.30 (offset handles gap)
        elif num_samples < 15:
            safety_margin = 1.10  # V3: Reduced from 1.20
        else:
            safety_margin = 1.05  # V3: Reduced from 1.15 (tight when learned)

        # Input estimation: use history average if available, blend with current
        if len(self.input_token_history) >= 5:
            avg_input = sum(self.input_token_history) / len(self.input_token_history)
            # Weighted blend: 70% history, 30% current for stability
            estimated_input = int(0.7 * avg_input + 0.3 * actual_input_tokens)
        else:
            # Early phase: use current with safety margin
            estimated_input = int(actual_input_tokens * safety_margin)

        # Always update input history (larger window now handles this better)
        self.input_token_history.append(actual_input_tokens)

        # Output estimation: use learned ratio if available
        if len(self.output_ratio_history) >= 5:
            # Use learned output/input ratio
            learned_ratio = sum(self.output_ratio_history) / len(self.output_ratio_history)
            estimated_output = int(estimated_input * learned_ratio * safety_margin)
        elif len(self.output_token_history) >= 3:
            # Use output history average
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output * safety_margin)
        else:
            # Fallback to default ratio with safety margin
            estimated_output = int(estimated_input * DEFAULT_OUTPUT_RATIO * safety_margin)

        # Cap output to max_tokens
        estimated_output = min(self.config.max_tokens, estimated_output)

        return estimated_input + estimated_output

    def _estimate_preprocessed_tokens(self, prompt: str) -> int:
        """Simple token estimate for pre-processing calls (non-adaptive).

        Used for context extraction, dimension selection, consolidation, and subject
        extraction — calls that don't need adaptive estimation.
        """
        tiktoken_count = len(self.encoding.encode(prompt))
        return int((tiktoken_count + TIKTOKEN_API_OFFSET_DEFAULT) * (1 + DEFAULT_OUTPUT_RATIO))

    def get_token_estimation_stats(self) -> dict:
        """Get token estimation accuracy statistics including learned ratio."""
        if not self.estimation_errors:
            return {"status": "collecting_data", "samples": 0}

        avg_error = sum(self.estimation_errors) / len(self.estimation_errors)
        avg_input = sum(self.input_token_history) / len(self.input_token_history) if self.input_token_history else 0
        avg_output = sum(self.output_token_history) / len(self.output_token_history) if self.output_token_history else 0
        avg_actual_total = sum(self.actual_total_tokens) / len(self.actual_total_tokens) if self.actual_total_tokens else 0

        # Calculate learned output ratio
        learned_ratio = (sum(self.output_ratio_history) / len(self.output_ratio_history)) if self.output_ratio_history else DEFAULT_OUTPUT_RATIO

        return {
            "status": "learning",
            "samples": len(self.estimation_errors),
            "avg_estimation_error": avg_error,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "avg_actual_total_tokens": avg_actual_total,
            "initial_avg_tokens": self.bootstrap_avg_tokens if self.bootstrap_avg_tokens is not None else self.avg_tokens,
            "current_avg_tokens": self.avg_tokens,
            "adjustments_made": self.v3_stats['adjustments_made'],
            "threshold_adjustments": self.v3_stats['threshold_adjustments'],
            "input_samples": len(self.input_token_history),
            "output_samples": len(self.output_token_history),
            "actual_samples": len(self.actual_total_tokens),
            "learned_output_ratio": learned_ratio,
            "ratio_samples": len(self.output_ratio_history)
        }

    def get_token_bucket_status(self) -> dict:
        """Get current token bucket status"""
        available_pct = (self.tpm_bucket.available / self.tpm_bucket.tpm) * 100

        if len(self.actual_total_tokens) >= 10:
            recent_avg = sum(list(self.actual_total_tokens)[-10:]) / 10
            consumption_rate_per_sec = recent_avg / 2.0
            real_utilization_pct = (consumption_rate_per_sec / (self.tpm_bucket.tpm / 60)) * 100
        else:
            real_utilization_pct = 100 - available_pct
            consumption_rate_per_sec = 0

        return {
            "available_tokens": int(self.tpm_bucket.available),
            "capacity": self.tpm_bucket.tpm,
            "utilization_pct": real_utilization_pct,
            "low_tokens": available_pct < 10,
            "consumption_rate": consumption_rate_per_sec
        }

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            InternalServerError,
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def process_task(self, task: Dict) -> models.IdeasExtractedModel:
        """Process a single idea extraction task.
        Timeouts are NOT retried here — they're collected and reprocessed as a batch."""
        task_start = time.perf_counter()

        try:
            # Use taxonomy-aware subject extraction for template prefix
            assert self.primary_dimension is not None, "primary_dimension must be set before processing tasks"
            subject, phrasing_template = self._build_canonical_phrasing(self.primary_dimension)

            # Extract template prefix (everything before the domain marker)
            dimension = get_dimension(self.primary_dimension)
            dim_marker = dimension.domain_marker
            template_prefix = phrasing_template.split(dim_marker)[0].strip() if dim_marker in phrasing_template else phrasing_template
            if self.template_prefix is None:
                self.template_prefix = template_prefix

            # V3: Build prompt with subject and phrasing template
            prompt = self._build_taxonomy_enriched_prompt(
                task['response'],
                subject,
                phrasing_template,
            )

            if self.prompt_printer and not self._captured_prompt:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="idea_extraction_v3",
                    metadata={
                        "model": self.model,
                        "var_lab": self.var_lab,
                        "language": self.language,
                        "respondent_id": task['respondent_id'],
                        "primary_dimension": self.primary_dimension,
                        "template_prefix": template_prefix,
                        "primary_dimension_description": self.primary_dimension_description
                    }
                )
                self._captured_prompt = True

            est_tokens = self.estimate_tokens(prompt)

            if task.get('task_index', 0) < 5:
                logger.info(f"[ESTIMATION DEBUG] Task {task.get('task_index', 0)}: estimated {est_tokens} tokens")

            self.stats['tasks_processed'] += 1

            # Create dimension-specific response model (no ClassVar mutation — baked in)
            assert self.primary_dimension is not None, "primary_dimension must be set before processing tasks"
            dimension = get_dimension(self.primary_dimension)
            AxisExtractionModel = create_extraction_model(
                dimension=dimension,
                template_prefix=template_prefix,
                domains=getattr(self, 'domains', None),
                model=self.model,
            )

            async with self.semaphore:
                # No timeout for last batch — when remaining tasks fit within concurrency,
                # they're all in the final wave. Timeouts would only add wall time.
                last_batch = (hasattr(self, '_task_queue') and self._task_queue is not None
                              and self._task_queue.qsize() < self.optimal_concurrency)
                timeout = None if last_batch else self.latency_tracker.get_timeout(est_tokens)
                await self.tpm_bucket.wait_and_acquire(est_tokens)
                api_start = time.perf_counter()
                async with self.rate_limiter:
                    response = await asyncio.wait_for(
                        llm_create_async(
                            client=self.client,
                            model=self.model,
                            response_model=List[AxisExtractionModel],
                            prompt=prompt,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            max_retries=3,
                            **get_reasoning_params(self.model),
                        ),
                        timeout=timeout
                    )

                    latency = time.perf_counter() - api_start
                    self.latency_tracker.add(latency)
                    if self._concurrency_sm:
                        self._concurrency_sm.record_latency(latency)

                    # Record successful completion to circuit breaker
                    if self.circuit_breaker:
                        self.circuit_breaker.record_completion()

                    usage = getattr(response, '_raw_response', None)
                    if usage:
                        usage = getattr(usage, 'usage', None)
                    if not usage:
                        usage = getattr(response, 'usage', None)

                    if usage:
                        actual_input_tokens = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0)
                        actual_output_tokens = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0)
                        actual_total_tokens = getattr(usage, 'total_tokens', 0) or (actual_input_tokens + actual_output_tokens)

                        # Always update output history (input updated in estimate_tokens)
                        self.output_token_history.append(actual_output_tokens)

                        # Track output/input ratio for learning
                        if actual_input_tokens > 0:
                            ratio = actual_output_tokens / actual_input_tokens
                            self.output_ratio_history.append(ratio)

                        self.actual_total_tokens.append(actual_total_tokens)

                        estimation_error = abs(actual_total_tokens - est_tokens)
                        self.estimation_errors.append(estimation_error)

                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)

                        # Learn tiktoken→API offset (for input tokens only)
                        tiktoken_input = len(self.encoding.encode(prompt))
                        self.tiktoken_offset_learner.record(tiktoken_input, actual_input_tokens)

                        # Feed trackers for constraint visibility
                        if self.tpm_tracker:
                            await self.tpm_tracker.record(actual_total_tokens)
                        if self.rpm_tracker:
                            await self.rpm_tracker.record()

                    ideas = []
                    for i, idea_response in enumerate(response):
                        normalized = self._normalize_idea_text(idea_response.idea) if idea_response.idea else ""
                        if normalized and normalized not in ["", "NA", "N/A"]:
                            # Extract taxonomy fields (flat)
                            taxonomy_resp = getattr(idea_response, 'abstraction_ladder', None)

                            # Clean idea text
                            idea_text = self._format_idea_text(normalized)
                            response_idea_id = str(i + 1)
                            ideas.append(models.IdeasExtractedSubmodel(
                                idea_id=f"{task['respondent_id']}_{response_idea_id}",
                                idea=idea_text,
                                instance=taxonomy_resp.instance if taxonomy_resp else "",
                                interpretation=taxonomy_resp.interpretation if taxonomy_resp else "",
                                abstraction=taxonomy_resp.abstraction if taxonomy_resp else "",
                                domain=taxonomy_resp.domain if taxonomy_resp else "",
                                valence=getattr(idea_response, 'valence', "") or "",
                            ))

                    if ideas:
                        self.stats['tasks_successful'] += 1
                        return models.IdeasExtractedModel(
                            respondent_id=task['respondent_id'],
                            response=task['response'],
                            quality_filter=task.get('quality_filter', True),
                            quality_filter_code=task.get('quality_filter_code', 0),
                            response_ideas=ideas,
                            idea_count=len(ideas),
                            template_prefix=self.template_prefix or ""  # V3: Use extracted template prefix
                        )
                    else:
                        # Empty ideas: retry up to 2 more times before falling back
                        logger.warning(f"Task {task['respondent_id']}: LLM returned empty ideas, retrying...")
                        for empty_retry in range(2):
                            await self.tpm_bucket.wait_and_acquire(est_tokens)
                            retry_response = await asyncio.wait_for(
                                llm_create_async(
                                    client=self.client,
                                    model=self.model,
                                    response_model=List[AxisExtractionModel],
                                    prompt=prompt,
                                    temperature=self.config.temperature,
                                    max_tokens=self.config.max_tokens,
                                    max_retries=3,
                                    **get_reasoning_params(self.model),
                                ),
                                timeout=timeout
                            )
                            retry_ideas = []
                            for i, idea_response in enumerate(retry_response):
                                normalized = self._normalize_idea_text(idea_response.idea) if idea_response.idea else ""
                                if normalized and normalized not in ["", "NA", "N/A"]:
                                    taxonomy_resp = getattr(idea_response, 'abstraction_ladder', None)
                                    idea_text = self._format_idea_text(normalized)
                                    response_idea_id = str(i + 1)
                                    retry_ideas.append(models.IdeasExtractedSubmodel(
                                        idea_id=f"{task['respondent_id']}_{response_idea_id}",
                                        idea=idea_text,
                                        instance=taxonomy_resp.instance if taxonomy_resp else "",
                                        interpretation=taxonomy_resp.interpretation if taxonomy_resp else "",
                                        abstraction=taxonomy_resp.abstraction if taxonomy_resp else "",
                                        facet="",
                                        domain=taxonomy_resp.domain if taxonomy_resp else "",
                                        valence=getattr(idea_response, 'valence', "") or "",
                                    ))
                            if retry_ideas:
                                logger.info(f"Task {task['respondent_id']}: empty-ideas retry {empty_retry+1} succeeded ({len(retry_ideas)} ideas)")
                                self.stats['tasks_successful'] += 1
                                return models.IdeasExtractedModel(
                                    respondent_id=task['respondent_id'],
                                    response=task['response'],
                                    quality_filter=task.get('quality_filter', True),
                                    quality_filter_code=task.get('quality_filter_code', 0),
                                    response_ideas=retry_ideas,
                                    idea_count=len(retry_ideas),
                                    template_prefix=self.template_prefix or ""
                                )
                        # All retries exhausted — log and fall back
                        self.stats['tasks_failed'] += 1
                        self.failed_task_ids.add(str(task['respondent_id']))
                        self.failure_log.append({
                            'respondent_id': task['respondent_id'],
                            'reason': 'empty_ideas',
                            'error_type': None,
                            'response_preview': task['response'][:80]
                        })
                        logger.error(f"Task {task['respondent_id']}: empty ideas after 2 retries, creating PROCESSING_ERROR fallback")
                        return self.create_fallback_response(task, reason="empty_ideas")

        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - task_start
            self.stats['timeouts'] += 1
            if self.circuit_breaker:
                self.circuit_breaker.record_timeout()
            print(f"DEFERRED: task {task['respondent_id']} after {elapsed:.1f}s (timeout was {timeout:.1f}s)")
            # Return None — caller collects for batch reprocessing
            return None

        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['respondent_id']} hit rate limit")
            raise

        except InstructorRetryException as e:
            # Concise output for 429 errors wrapped in InstructorRetryException
            error_str = str(e)
            if "429" in error_str or "RateLimitReached" in error_str:
                if "token rate limit" in error_str.lower():
                    limit_type = "TPM"
                elif "call rate limit" in error_str.lower():
                    limit_type = "RPM"
                else:
                    limit_type = "rate"
                print(f"⚠️ 429 {limit_type} limit hit (task {task['respondent_id']})")
            else:
                logger.error(f"Task {task['respondent_id']} failed: {type(e).__name__}")
            raise

        except Exception as e:
            logger.error(f"Task {task['respondent_id']} failed: {type(e).__name__}: {e}")
            raise

    def create_fallback_response(self, task: Dict, reason: str = "unknown") -> models.IdeasExtractedModel:
        """Create fallback response for failed tasks"""
        return models.IdeasExtractedModel(
            respondent_id=task['respondent_id'],
            response=task['response'],
            quality_filter=task.get('quality_filter', True),
            quality_filter_code=task.get('quality_filter_code', 0),
            response_ideas=[
                models.IdeasExtractedSubmodel(
                    idea_id=f"{task['respondent_id']}_1",
                    idea=f"PROCESSING_ERROR: {reason}"
                )
            ],
            idea_count=1,
            template_prefix=self.template_prefix or ""  # V3: Use extracted template prefix
        )

    def get_failure_report(self, total_responses: int = None) -> str:
        """Return a formatted report of all PROCESSING_ERROR failures."""
        total = total_responses or self.stats.get('tasks_processed', 0)
        n_failures = len(self.failure_log)

        if n_failures == 0:
            return f"PROCESSING ERRORS: 0 of {total} responses (0%)"

        lines = [f"PROCESSING ERRORS: {n_failures} of {total} responses ({n_failures/max(total,1)*100:.1f}%)"]

        # Group by reason
        from collections import Counter
        reason_counts = Counter()
        for f in self.failure_log:
            key = f['error_type'] if f['reason'] == 'exception' else f['reason']
            reason_counts[key] += 1

        lines.append(f"  Breakdown: {', '.join(f'{count}x {reason}' for reason, count in reason_counts.most_common())}")
        lines.append("")

        for f in self.failure_log:
            reason_str = f['error_type'] if f['reason'] == 'exception' else f['reason']
            preview = f['response_preview']
            lines.append(f"  Respondent {f['respondent_id']}: {reason_str} | \"{preview}...\"")

        return "\n".join(lines)

    def _normalize_idea_text(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)
        text = text.strip()
        text = ' '.join(text.split())

        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            text = text.replace(char, '')

        # Strip dimension marker token (e.g., "[EXPERIENCE_PERCEPTION]") that the
        # LLM sometimes includes despite prompt instructions not to.
        # We add the prefix programmatically in _format_idea_text, so the marker
        # must not be in the LLM's output.
        original = text
        if self.primary_dimension:
            dimension = get_dimension(self.primary_dimension)
            marker = dimension.domain_marker  # e.g., "[EXPERIENCE_PERCEPTION]"
            if marker and marker in text:
                text = text.replace(marker, '').strip()
                text = ' '.join(text.split())  # collapse any double spaces

        # Strip template prefix if the LLM included it (we add it in _format_idea_text)
        if self.template_prefix and text.lower().startswith(self.template_prefix.lower()):
            text = text[len(self.template_prefix):].strip()

        # If stripping left nothing, keep the original — don't discard real content
        if not text:
            return original

        return text

    def _format_idea_text(self, normalized_text: str) -> str:
        """Prepend the template prefix to the LLM's verbatim span.

        The LLM outputs just the idea content (e.g. "goede sfeer").
        We prepend the canonical prefix (e.g. "Pinkpop →") to produce
        the full idea statement: "Pinkpop → goede sfeer".
        """
        if self.template_prefix and not normalized_text.lower().startswith(self.template_prefix.lower()):
            return f"{self.template_prefix} {normalized_text}"
        return normalized_text

    def build_extraction_metadata(self, filename: str = "", var_name: str = "") -> 'models.ExtractionMetadata':
        """Build ExtractionMetadata from extracted context specifiers and taxonomy info.

        This creates a single metadata object that captures all extraction-level
        information that applies to the entire dataset (not per-idea).

        Args:
            filename: The source data filename
            var_name: The variable name being extracted

        Returns:
            ExtractionMetadata instance with all fields populated
        """
        return models.ExtractionMetadata(
            # File/variable info
            filename=filename,
            var_name=var_name,
            var_lab=self.var_lab,

            # Template
            template_prefix=self.template_prefix or "",

            # Context specifiers (6 fields)
            lang=self.generic_specifiers.get('lang', ''),
            sector=self.generic_specifiers.get('domain', ''),
            topic=self.generic_specifiers.get('topic', ''),
            perspective=self.generic_specifiers.get('perspective', ''),
            entity=self.generic_specifiers.get('entity', ''),
            intent=self.generic_specifiers.get('intent', ''),

            # Taxonomy (these should always be set by the time metadata is built)
            primary_dimension=self.primary_dimension or '',
            primary_dimension_description=self.primary_dimension_description or '',
            decision_tree_stop_position=self.decision_tree_stop_position,
            # Domains
            domains=[
                {"key": c.key, "label": c.label, "definition": c.definition}
                for c in getattr(self, 'domains', []) or []
            ],
        )

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
        """Make a minimal API call to fetch rate limits from response headers."""
        from openai import AsyncOpenAI
        from config import API_PROVIDER, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY

        if API_PROVIDER == "azure":
            # Use self.model (the actual deployment) rather than the hardcoded default
            deployment = self.model
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{deployment}/",
                default_query={"api-version": "2024-10-21"},
            )
            model = deployment
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self.model

        if API_PROVIDER == "azure":
            response = await client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_completion_tokens=5,
            )
        else:
            response = await client.responses.with_raw_response.create(
                model=model,
                input="Hi",
            )

        return extract_rate_limits_from_response(response)

    def _initialize_rate_limiters(self, limits, num_tasks: int, avg_latency_s: float = None) -> int:
        """Initialize four-layer rate limiting with gradual ramp-up.

        Layer 1: RPM — AsyncLimiter with PID-adjustable arrival rate
        Layer 2: TPM — TokenBucket (self-regulating)
        Layer 3: Concurrency — ConcurrencyGate via Little's Law, with gradual ramp-up
        Layer 4: Circuit breaker — monitors timeout rate, adjusts Layer 3 on sustained pressure

        Returns target concurrency (what we're ramping toward).
        """
        headroom = self.processing_config.rate_limit_headroom
        avg_latency_s = avg_latency_s or DEFAULT_LATENCY_SECONDS

        # Layer 1: RPM — AsyncLimiter (PID adjusts this via current_arrival_rate)
        arrival_rate = min(
            limits.requests_per_minute * headroom / 60,
            limits.tokens_per_minute * headroom / self.avg_tokens / 60
        )
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / arrival_rate)
        self.current_arrival_rate = arrival_rate

        # Layer 2: TPM — TokenBucket (self-regulating via acquire/wait)
        self.tpm_bucket = TokenBucket(int(limits.tokens_per_minute * headroom))

        # Layer 3: Concurrency — capacity-relative starting + ramp to 90% of Little's Law
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law = compute_optimal_concurrency(
            api_limits, avg_latency_s, self.avg_tokens,
            headroom=headroom
        )
        little_law_cap = min(little_law, num_tasks)

        # Starting concurrency = min(empirical capacity or cold start cap, num_tasks)
        if getattr(self, '_stored_empirical_capacity', None) is not None:
            target = min(int(self._stored_empirical_capacity), num_tasks)
        else:
            target = min(COLD_START_CAP, num_tasks)
        target = max(target, DEFAULT_CONCURRENCY_CONTROL_CONFIG.min_concurrency)

        self.semaphore = ConcurrencyGate(target)
        self.optimal_concurrency = target

        # State machine concurrency controller
        bottleneck = detect_bottleneck(
            limits.requests_per_minute, limits.tokens_per_minute,
            self.avg_tokens, target
        )
        self._concurrency_sm = ConcurrencyStateMachine(
            starting=target, bottleneck=bottleneck,
            config=DEFAULT_CONCURRENCY_CONTROL_CONFIG
        )

        # Layer 4: Circuit breaker (baseline updated when ramp stops)
        self.circuit_breaker = ConcurrencyCircuitBreaker(
            config=self.circuit_breaker_config,
            gate=self.semaphore,
            baseline=target
        )

        # PID components for arrival rate adjustment
        self.tpm_tracker = RealTimeTPMTracker(
            window_seconds=self.tpm_tracking_config.sliding_window_seconds
        )
        self.pid_controller = PIDThroughputController(
            target_utilization=self.tpm_tracking_config.target_utilization,
            kp_up=self.pid_config.kp_up,
            kp_down=self.pid_config.kp_down,
            ki=self.pid_config.ki,
            kd=self.pid_config.kd,
            min_adjustment=self.pid_config.min_adjustment,
            max_adjustment=self.pid_config.max_adjustment,
        )
        self.rpm_tracker = RealTimeRPMTracker(
            window_seconds=self.tpm_tracking_config.sliding_window_seconds
        )

        return target

    def _initialize_conservative_rate_limiters(self, limits: 'RateLimits', num_tasks: int = 20) -> None:
        """Initialize conservative rate limiters for context extraction phase.

        Uses very conservative settings since we don't have accurate token estimates yet.
        """
        # Conservative arrival rate (50% of normal headroom)
        conservative_tokens = DEFAULT_AVG_TOKENS * 1.5  # 2250 tokens
        arrival_rate = min(
            limits.requests_per_minute * 0.5 / 60,
            limits.tokens_per_minute * 0.5 / conservative_tokens / 60
        )

        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / max(arrival_rate, 0.1))
        self.semaphore = ConcurrencyGate(min(num_tasks, 10))
        self.optimal_concurrency = min(num_tasks, 10)

        # Token bucket at full rate
        self.tpm_bucket = TokenBucket(int(limits.tokens_per_minute * self.processing_config.rate_limit_headroom))

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Conservative setup: concurrency={self.optimal_concurrency}, tokens={conservative_tokens}")

    def _calculate_warm_up_sample_size(self, num_tasks: int) -> int:
        """Adaptive sample size: more samples for larger datasets, capped."""
        if num_tasks <= 50:
            return self.warm_up_config.sample_min
        elif num_tasks >= 500:
            return self.warm_up_config.sample_max
        else:
            # Linear interpolation between min and max
            fraction = (num_tasks - 50) / (500 - 50)
            return int(self.warm_up_config.sample_min + fraction * (self.warm_up_config.sample_max - self.warm_up_config.sample_min))

    def _calibrate_from_warm_up(self, num_tasks: int) -> None:
        """One-shot calibration: update token estimate AND recompute Little's Law concurrency.

        Fires once after enough warm-up completions. Uses measured latency and
        token counts to recalculate optimal concurrency, update circuit breaker
        baseline, and recalibrate PID arrival rate.
        """
        measured_avg_tokens = int(np.mean(list(self.actual_total_tokens)))
        # P10 latency: median includes queuing time at high concurrency,
        # which inflates Little's Law cap → positive feedback loop.
        measured_latency = float(np.percentile(list(self.latency_tracker.values), 10))

        old_avg = self.avg_tokens
        old_conc = self.optimal_concurrency

        self.avg_tokens = measured_avg_tokens
        self.bootstrap_avg_tokens = measured_avg_tokens

        # Recalculate Little's Law with measured data
        api_limits = ApiLimits(
            self.rate_limits.tokens_per_minute,
            self.rate_limits.requests_per_minute
        )
        headroom = self.processing_config.rate_limit_headroom
        new_little_law = compute_optimal_concurrency(
            api_limits, measured_latency, measured_avg_tokens,
            headroom=headroom
        )
        new_little_law_cap = min(new_little_law, num_tasks)

        # Recalculate arrival rate (tokens changed, so TPM rail changes)
        new_arrival_rate = min(
            self.rate_limits.requests_per_minute * headroom / 60,
            self.rate_limits.tokens_per_minute * headroom / measured_avg_tokens / 60
        )
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / new_arrival_rate)
        self.current_arrival_rate = new_arrival_rate

        # Reset PID (we just recalibrated tokens)
        if self.pid_controller:
            self.pid_controller.reset()

        print(f"\n[WARM-UP] Token calibration from {len(self.actual_total_tokens)} samples: "
              f"avg_tokens {old_avg} → {measured_avg_tokens}")

        self._warm_up_calibrated = True

    def _adjust_throughput_if_needed(self) -> bool:
        """Threshold-based token estimate correction.

        When actual token usage significantly exceeds the current estimate,
        update avg_tokens so the TPM bucket allocates correctly. The RPM rail
        and concurrency semaphore are independent and don't need adjustment.

        Returns True if adjustment was made, False otherwise.
        """
        if len(self.actual_total_tokens) < THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
            return False

        actual_avg = sum(self.actual_total_tokens) / len(self.actual_total_tokens)
        current_avg = self.avg_tokens

        ratio = actual_avg / current_avg if current_avg > 0 else 1.0

        if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
            return False

        # Update token estimate
        old_avg = self.avg_tokens
        self.avg_tokens = int(actual_avg)

        self.v3_stats['threshold_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1

        print(f"\n⚡ TOKEN ESTIMATE CORRECTION")
        print(f"   Actual tokens ({actual_avg:.0f}) exceeded estimate ({current_avg:.0f}) by {(ratio-1)*100:.0f}%")
        print(f"   avg_tokens: {old_avg} → {self.avg_tokens}")
        print(f"   Tiktoken offset: {self.tiktoken_offset_learner.get_offset()} (learned: {self.tiktoken_offset_learner.is_learned()})")

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Token estimate correction: {old_avg} → {self.avg_tokens}")

        return True

    async def _apply_pid_adjustment(self) -> bool:
        """Apply PID-style continuous throughput adjustment based on real-time TPM utilization.

        Uses asymmetric gains: aggressive when under-utilizing, gentle when over-utilizing.
        Returns True if adjustment was applied, False otherwise.
        """
        if self.current_arrival_rate is None or self.tpm_tracker is None:
            return False

        current_tpm = await self.tpm_tracker.get_current_tpm()
        tpm_limit = self.rate_limits.tokens_per_minute
        utilization = current_tpm / tpm_limit if tpm_limit > 0 else 0.0

        # Track utilization stats
        self.v3_stats['max_tpm_utilization'] = max(self.v3_stats['max_tpm_utilization'], utilization * 100)
        self.v3_stats['min_tpm_utilization'] = min(self.v3_stats['min_tpm_utilization'], utilization * 100)

        adjustment = self.pid_controller.compute_adjustment(utilization)

        if abs(adjustment - 1.0) < 0.01:
            return False

        old_rate = self.current_arrival_rate
        new_rate = old_rate * adjustment

        # Clamp to reasonable bounds
        rpm_max = self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        new_rate = max(0.5, min(rpm_max, new_rate))

        # Only apply if change is meaningful (>2%)
        if abs(new_rate - old_rate) / max(old_rate, 0.001) < 0.02:
            return False

        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / new_rate)
        self.current_arrival_rate = new_rate
        self.v3_stats['pid_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1

        return True

    async def worker(self, queue: asyncio.Queue, results: List, timed_out: List):
        """Worker coroutine that processes tasks from queue.
        Timed-out tasks are collected in `timed_out` for batch reprocessing.
        """
        while True:
            task = None
            try:
                task = await queue.get()
                if task is None:
                    break

                task_index, task_data = task
                result = await self.process_task(task_data)
                if result is None:
                    # Timeout — collect for batch retry
                    timed_out.append((task_index, task_data))
                else:
                    results[task_index] = result

            except Exception as e:
                # Extract concise error info for rate limit errors
                error_str = str(e)
                error_type = type(e).__name__
                if "429" in error_str or "RateLimitReached" in error_str:
                    # Determine if RPM or TPM limit
                    if "token rate limit" in error_str.lower():
                        limit_type = "TPM"
                    elif "call rate limit" in error_str.lower():
                        limit_type = "RPM"
                    else:
                        limit_type = "rate"
                    error_type = f"RateLimit_{limit_type}"
                    task_id = task_data.get('respondent_id', 'unknown') if task else 'unknown'
                    print(f"⚠️ 429 {limit_type} limit hit (task {task_id})")
                else:
                    # Non-rate-limit errors: show full details
                    logger.error(f"Task failed after retries: {e}")
                self.stats['tasks_failed'] += 1
                if task is not None:
                    task_index, task_data = task
                    self.failed_task_ids.add(str(task_data.get('respondent_id', 'unknown')))
                    self.failure_log.append({
                        'respondent_id': task_data.get('respondent_id', 'unknown'),
                        'reason': 'exception',
                        'error_type': error_type,
                        'response_preview': task_data.get('response', '')[:80]
                    })
                    results[task_index] = self.create_fallback_response(task_data, reason=error_type)
            finally:
                if task is not None:
                    queue.task_done()

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[models.IdeasExtractedModel]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []

        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Fetching rate limits from API...")

        limits = await self._fetch_rate_limits_from_api()

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Warning: Using fallback rate limits (TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            limits = RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
                tokens_per_day=FALLBACK_TPM * 60 * 24
            )
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Fetched from API: TPM={limits.tokens_per_minute:,}, RPM={limits.requests_per_minute:,}")

        self.rate_limits = limits

        # === PHASE 2: Initialize CONSERVATIVE rate limiters for context extraction ===
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Initializing conservative rate limiters for context extraction...")
        self._initialize_conservative_rate_limiters(limits, num_tasks=30)

        # === PHASE 3: Extract context specifiers, primary dimension, AND domains ===
        self.verbose_reporter.stat_line("Extracting context specifiers, primary dimension, and domains...")
        self.generic_specifiers, taxonomy_result, categories_result = await self._extract_generic_specifiers()

        # Store taxonomy axis info for use in idea extraction
        self.primary_dimension = taxonomy_result.primary_dimension
        self.primary_dimension_rationale = taxonomy_result.primary_dimension_rationale
        self.primary_dimension_description = taxonomy_result.primary_dimension_description  # Dynamic context-specific description

        # Store domains for use in per-response extraction model
        # Empty list (Phase 3 skipped) → None to trigger on-the-fly mode in model factories
        self.domains = categories_result.domains or None

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"\nTaxonomy axis selected: {self.primary_dimension}")
            if self.primary_dimension_description:
                self.verbose_reporter.stat_line(f"Description: {self.primary_dimension_description}")
            if self.domains:
                self.verbose_reporter.stat_line(f"Domains: {[c.key for c in self.domains]}")
            else:
                self.verbose_reporter.stat_line(f"Domains: on-the-fly (no pre-discovered domains)")

        # === PHASE 4: Recalculate avg_tokens with REAL context ===
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("\nRecalculating token estimates with real context...")
        old_avg = self.avg_tokens
        self.avg_tokens = self._calculate_avg_tokens()
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Updated avg_tokens: {old_avg} → {self.avg_tokens}")

        # === PHASE 5: INITIALIZE RATE LIMITING (Little's Law + PID + Circuit Breaker) ===
        self.bootstrap_avg_tokens = self.avg_tokens

        target_conc = self._initialize_rate_limiters(limits, len(tasks))
        warm_up_samples = self._calculate_warm_up_sample_size(len(tasks))
        self._warm_up_calibrated = False
        self._warm_up_target_samples = warm_up_samples

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"\nRate limiting: Theoretical + Empirical concurrency + PID + Circuit Breaker")
            self.verbose_reporter.stat_line(f"Target concurrency: {self.optimal_concurrency}, calibration after {warm_up_samples} completions")

        # === PHASE 6: Print setup info and launch workers ===
        headroom = self.processing_config.rate_limit_headroom
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / self.avg_tokens / 60
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"
        expected_throughput = min(rpm_throughput, tpm_throughput)

        # Report every ~5% of tasks, min 10
        report_every_n = max(len(tasks) // 20, 10)

        # Compute Little's Law components for reporting
        latency_used = getattr(self, '_stored_p50', None) or DEFAULT_LATENCY_SECONDS
        latency_source = "stored P50" if getattr(self, '_stored_p50', None) else "default"
        little_law_raw = expected_throughput * latency_used

        sm = self._concurrency_sm
        print("\nRATE LIMITING SETUP")
        print(f"- Model: {self.model}")
        print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * headroom:,.0f} with headroom)")
        print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * headroom:,.0f} with headroom)")
        print(f"- Initial avg_tokens (tiktoken): {self.avg_tokens}")
        print(f"- Theoretical throughput: RPM={rpm_throughput:.1f}/s, TPM={tpm_throughput:.1f}/s → {expected_throughput:.1f}/s ({bottleneck} bound)")
        print(f"- Latency: {latency_used:.2f}s ({latency_source})")
        print(f"- Little's Law: L = {expected_throughput:.1f}/s × {latency_used:.2f}s = {little_law_raw:.0f}")
        print(f"- Theoretical concurrency: min(Little's Law, tasks) = min({little_law_raw:.0f}, {len(tasks)}) = {min(int(little_law_raw), len(tasks))}")
        if getattr(self, '_stored_empirical_capacity', None) is not None:
            print(f"- Target concurrency: {self.optimal_concurrency} (empirical capacity, measured over time)")
        else:
            print(f"- Target concurrency: {self.optimal_concurrency} (cold start cap, no empirical data)")
        print(f"- Bottleneck: {sm.bottleneck}")
        print(f"- Controller: state machine (signals: throughput, interval P100) ramp step: +{sm.ramp_step}")
        print(f"- Arrival rate: {self.current_arrival_rate:.2f}/s (PID-adjusted)")
        print(f"- Token calibration: after {warm_up_samples} completions")
        print(f"- Processing {len(tasks):,} tasks")

        # Workers = target concurrency so they can fill slots
        num_workers = min(self.optimal_concurrency, len(tasks))

        print(f"\nWorkers: {num_workers}, Target concurrency: {self.optimal_concurrency}")

        t_phase_start = time.time()
        print(f"\n⏱ T+0.0s: Starting task processing")

        queue = asyncio.Queue()
        self._task_queue = queue  # shared with process_task for tail mode detection
        results = [None] * len(tasks)
        timed_out = []  # Collect timed-out tasks for batch retry

        for i, task in enumerate(tasks):
            task['result_index'] = i
            task['task_index'] = i
            await queue.put((i, task))

        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(self.worker(queue, results, timed_out))
            workers.append(w)

        start_time = time.time()
        last_report = start_time
        last_report_completed = 0
        last_diagnostics = start_time
        last_adjustment = start_time

        while not queue.empty():
            await asyncio.sleep(0.1)  # 10 Hz — enough to track ramp and progress
            now = time.time()

            # EVERY 1s: Circuit breaker evaluates timeout RATE (not individual events)
            if self.circuit_breaker:
                action = self.circuit_breaker.check_and_adjust()
                if action == 'tripped':
                    self.v3_stats['circuit_breaker_trips'] += 1
                    self.optimal_concurrency = self.semaphore.limit
                elif action in ('recovering', 'recovered'):
                    self.optimal_concurrency = self.semaphore.limit

            # Progress reporting + concurrency evaluation — synchronized
            completed = self.stats['tasks_processed']
            completions_since_report = completed - last_report_completed
            if now - last_report >= 2.0 or completions_since_report >= report_every_n:
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                timeouts = self.stats['timeouts']

                # TPM/RPM utilization (for reporting and PID)
                tpm_pct = rpm_pct = 0.0
                if self.tpm_tracker and self.rate_limits:
                    current_tpm = await self.tpm_tracker.get_current_tpm()
                    tpm_pct = current_tpm / self.rate_limits.tokens_per_minute * 100 if self.rate_limits.tokens_per_minute else 0
                if self.rpm_tracker and self.rate_limits:
                    current_rpm = await self.rpm_tracker.get_current_rpm()
                    rpm_pct = current_rpm / self.rate_limits.requests_per_minute * 100 if self.rate_limits.requests_per_minute else 0

                # Latency and throughput
                latency_str = ""
                throughput_str = ""
                p50 = 0.0
                if self.latency_tracker.values:
                    vals = list(self.latency_tracker.values)
                    p50 = float(np.percentile(vals, 50))
                    p95 = float(np.percentile(vals, 95))
                    throughput = self.optimal_concurrency / p50 if p50 > 0 else 0
                    throughput_str = f" thru:{throughput:.0f}/s"

                # State machine concurrency evaluation (same tick as report)
                state_str = ""
                interval_p100 = 0.0
                if self._concurrency_sm and p50 > 0:
                    # Sync with circuit breaker (may have changed concurrency externally)
                    self._concurrency_sm.current = self.semaphore.limit
                    new_conc = self._concurrency_sm.evaluate(p50=p50, p95=p95, now=time.monotonic())
                    if new_conc != self.optimal_concurrency:
                        self.semaphore.set_limit(new_conc)
                        self.optimal_concurrency = new_conc
                    sm = self._concurrency_sm
                    p95r = getattr(sm, 'p95_ratio', 0)
                    p100r = getattr(sm, 'p100_ratio', 0)
                    state_str = (f" {sm.state.value}"
                                 f" p95:{p95r:.1f}x/{sm.consecutive_p95_stressed}"
                                 f" p100:{p100r:.1f}x/{sm.consecutive_p100_stressed}")

                # Latency percentiles (P100 = interval max, from state machine)
                if self.latency_tracker.values:
                    iv_p100 = getattr(sm, 'p100_ratio', 0) * p50 if sm else float(max(vals))
                    latency_str = f" P50:{p50:.1f}s P95:{p95:.1f}s P100:{iv_p100:.1f}s"

                # Bottleneck-dependent constraint info
                active = self.semaphore.active
                bn = self._concurrency_sm.bottleneck if self._concurrency_sm else "throughput"
                if bn == "throughput":
                    constraint_str = f"inflight:{active}/{self.optimal_concurrency}"
                elif bn == "rpm":
                    constraint_str = f"RPM:{rpm_pct:.0f}%"
                else:
                    constraint_str = f"TPM:{tpm_pct:.0f}%"

                timeout_info = f" deferred:{timeouts}" if timeouts > 0 else ""
                print(f"[STEP3] {completed}/{len(tasks)} |{throughput_str} | "
                      f"{constraint_str} |{latency_str} |{state_str}{timeout_info}")
                last_report = now
                last_report_completed = completed

            # One-shot token calibration after warm-up
            if (not self._warm_up_calibrated
                    and len(self.actual_total_tokens) >= self._warm_up_target_samples
                    and len(self.latency_tracker.values) >= self._warm_up_target_samples):
                self._calibrate_from_warm_up(len(tasks))

            # Spawn extra workers if state machine increased concurrency beyond current worker count
            if self.optimal_concurrency > num_workers:
                extra = self.optimal_concurrency - num_workers
                for _ in range(extra):
                    w = asyncio.create_task(self.worker(queue, results, timed_out))
                    workers.append(w)
                num_workers = self.optimal_concurrency

            # INTERVAL: PID arrival rate adjustment + token correction (20s)
            if now - last_adjustment >= ADJUSTMENT_INTERVAL:
                if not self._adjust_throughput_if_needed():
                    await self._apply_pid_adjustment()
                last_adjustment = now

            # Diagnostics
            if self.verbose_reporter.enabled and now - last_diagnostics >= DIAGNOSTIC_INTERVAL:
                bucket_status = self.get_token_bucket_status()
                token_stats = self.get_token_estimation_stats()

                if bucket_status['low_tokens']:
                    self.verbose_reporter.stat_line(f"Token bucket low: {bucket_status['available_tokens']:,} tokens ({bucket_status['utilization_pct']:.1f}% utilized)")

                if token_stats['status'] == 'learning' and token_stats['actual_samples'] >= 10:
                    actual_avg = token_stats['avg_actual_total_tokens']
                    current_avg = token_stats['current_avg_tokens']
                    difference = actual_avg - current_avg
                    pct_change = (difference / current_avg * 100) if current_avg > 0 else 0

                    cb_state = self.circuit_breaker.state if self.circuit_breaker else 'N/A'
                    self.verbose_reporter.stat_line(
                        f"Tokens: est={current_avg:.0f}, actual={actual_avg:.0f} ({pct_change:+.1f}%) | "
                        f"Concurrency: {self.semaphore.active}/{self.optimal_concurrency} (CB:{cb_state})"
                    )

                last_diagnostics = now

        await queue.join()

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        t_main_done = time.time()
        # Snapshot main-batch metrics before retry pass contaminates them
        self._main_batch_successful = self.stats['tasks_successful']
        self._main_batch_wall_time = t_main_done - t_phase_start
        self._main_batch_p50 = self.latency_tracker.get_p50() if self.latency_tracker.values else 0.0
        print(f"⏱ T+{t_main_done - t_phase_start:.1f}s: Main batch done — {self.stats['tasks_successful']} succeeded, {self.stats['timeouts']} deferred")
        sm_state = self._concurrency_sm.state.value if self._concurrency_sm else "N/A"
        print(f"  Final concurrency: {self.optimal_concurrency} (state: {sm_state})")

        # === RETRY PASS: retry timed-out + failed tasks with reduced concurrency ===
        # Collect all failed tasks: timed-out (from worker) + exceptions (from failure_log)
        failed_tasks_for_retry = []

        # Timed-out tasks (returned None from process_task)
        if timed_out:
            for task_index, task_data in timed_out:
                failed_tasks_for_retry.append((task_index, task_data, 'timeout'))

        # Exception failures (from failed_task_ids set — O(1) lookup)
        if self.failed_task_ids:
            for i, task in enumerate(tasks):
                if str(task.get('respondent_id', '')) in self.failed_task_ids:
                    failed_tasks_for_retry.append((i, task, 'exception'))

        if failed_tasks_for_retry:
            print(f"\n[RETRY PASS] Retrying {len(failed_tasks_for_retry)} failed tasks with reduced concurrency...")

            # Save pre-retry state and clear for retry tracking
            pre_retry_failure_log = list(self.failure_log)
            pre_retry_timed_out_count = len(timed_out)
            self.failed_task_ids.clear()
            self.failure_log.clear()

            # Reduced concurrency: 10% of workers, min 5
            retry_workers_count = max(5, min(len(failed_tasks_for_retry), num_workers // 10))

            # Generous timeout for retry
            self.latency_tracker.retry_mode = True

            retry_queue = asyncio.Queue()
            retry_timed_out = []
            retry_results_map = {}  # task_index -> result

            for orig_index, task_data, reason in failed_tasks_for_retry:
                await retry_queue.put((orig_index, task_data))

            retry_worker_tasks = []
            for _ in range(retry_workers_count):
                w = asyncio.create_task(self.worker(retry_queue, results, retry_timed_out))
                retry_worker_tasks.append(w)

            await retry_queue.join()
            for _ in retry_worker_tasks:
                await retry_queue.put(None)
            await asyncio.gather(*retry_worker_tasks)

            self.latency_tracker.retry_mode = False

            # Count recoveries: check if result is a real response (not fallback with PROCESSING_ERROR)
            def _is_fallback(result):
                if result is None:
                    return True
                ideas = getattr(result, 'response_ideas', [])
                return any(getattr(idea, 'idea', '').startswith('PROCESSING_ERROR') for idea in ideas)

            recovered = 0
            for orig_index, task_data, reason in failed_tasks_for_retry:
                result = results[orig_index]
                if not _is_fallback(result):
                    recovered += 1

            # Create permanent fallback for tasks still failed after retry (including retry timed-out)
            for task_index, task_data in retry_timed_out:
                if results[task_index] is None:
                    self.stats['tasks_failed'] += 1
                    self.failed_task_ids.add(str(task_data.get('respondent_id', 'unknown')))
                    self.failure_log.append({
                        'respondent_id': task_data.get('respondent_id', 'unknown'),
                        'reason': 'timeout_after_retry',
                        'error_type': 'Timeout',
                        'response_preview': task_data.get('response', '')[:80]
                    })
                    results[task_index] = self.create_fallback_response(task_data, reason='timeout')

            # Fallback for originally timed-out tasks that weren't recovered
            for orig_index, task_data, reason in failed_tasks_for_retry:
                if results[orig_index] is None:
                    self.stats['tasks_failed'] += 1
                    results[orig_index] = self.create_fallback_response(task_data, reason=reason)

            still_failed = len(self.failure_log)
            print(f"[RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")
            if still_failed > 0:
                failed_ids_list = sorted(str(f['respondent_id']) for f in self.failure_log)
                print(f"[RETRY PASS] Permanently failed IDs: {failed_ids_list[:20]}{'...' if still_failed > 20 else ''}")
        else:
            # No failures — create fallback for any remaining None results (shouldn't happen)
            pass

        elapsed = time.time() - start_time
        print(f"\nCompleted {len(tasks)} tasks in {elapsed:.1f}s")
        print(f"- Successful: {self.stats['tasks_successful']}")
        print(f"- Failed: {self.stats['tasks_failed']}")
        print(f"- Rate limits: {self.stats['rate_limits']}")
        timeouts = self.stats['timeouts']
        if timeouts > 0:
            print(f"- Timeouts: {timeouts}")
        print(f"- Average: {elapsed/len(tasks):.2f}s/task")
        if self._warm_up_calibrated:
            print(f"- Token calibration: after {self._warm_up_target_samples} samples")
        if self.v3_stats['threshold_adjustments'] > 0:
            print(f"- Token corrections: {self.v3_stats['threshold_adjustments']}")
            print(f"  - Initial avg_tokens: {self.bootstrap_avg_tokens}, Final: {self.avg_tokens}")
        cb_trips = self.v3_stats.get('circuit_breaker_trips', 0)
        cb_state = self.circuit_breaker.state if self.circuit_breaker else 'N/A'
        print(f"- Concurrency: {self.optimal_concurrency} (CB:{cb_state}, trips:{cb_trips})")
        if self.v3_stats.get('pid_adjustments', 0) > 0:
            print(f"- PID adjustments: {self.v3_stats['pid_adjustments']}")
            print(f"  - TPM utilization: {self.v3_stats['min_tpm_utilization']:.0f}% - {self.v3_stats['max_tpm_utilization']:.0f}%")

        # Print PROCESSING_ERROR failure report
        if self.failure_log:
            print(f"\n{'='*70}")
            print(self.get_failure_report(total_responses=len(tasks)))
            print(f"{'='*70}")
        else:
            print(f"\nPROCESSING ERRORS: 0 of {len(tasks)} responses (0%)")

        # Print strategy stats
        offset_stats = self.tiktoken_offset_learner.get_stats()
        print(f"\nSTRATEGY STATS:")
        print(f"- Tiktoken offset: {offset_stats['using_offset']} tokens (learned: {offset_stats['is_learned']}, samples: {offset_stats['samples']})")
        if offset_stats['min_offset'] is not None:
            print(f"  - Offset range: {offset_stats['min_offset']} to {offset_stats['max_offset']}")
        print(f"- Token corrections: {self.v3_stats['threshold_adjustments']}")
        print(f"- PID adjustments: {self.v3_stats.get('pid_adjustments', 0)}")
        cb_state = self.circuit_breaker.state if self.circuit_breaker else 'N/A'
        print(f"- Final concurrency: {self.optimal_concurrency} (CB:{cb_state}, trips:{self.v3_stats.get('circuit_breaker_trips', 0)})")

        if self.verbose_reporter.enabled:
            token_stats = self.get_token_estimation_stats()

            if token_stats['status'] == 'learning':
                accuracy = max(0, 100 - (token_stats['avg_estimation_error'] / max(1, token_stats['avg_input_tokens'] + token_stats['avg_output_tokens']) * 100))
                self.verbose_reporter.stat_line(f"Token estimation accuracy: {accuracy:.1f}% (avg error: {token_stats['avg_estimation_error']:.0f} tokens)")

                if token_stats['actual_samples'] >= 10:
                    actual_avg = token_stats['avg_actual_total_tokens']
                    initial_avg = token_stats['initial_avg_tokens']
                    self.verbose_reporter.stat_line(f"Token usage: Initial {initial_avg:.0f} → Actual {actual_avg:.0f}")

        return results

    def extract(self) -> List[models.IdeasExtractedModel]:
        """Main method to extract ideas from responses using bootstrap measurement and unified processing"""
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)

        if not self.responses:
            self.verbose_reporter.stat_line("No responses to process")
            return []

        tasks = []
        for response in self.responses:
            tasks.append({
                'respondent_id': response.respondent_id,
                'response': response.response,
                'quality_filter': response.quality_filter,
                'quality_filter_code': response.quality_filter_code
            })

        nest_asyncio.apply()
        self._results = asyncio.run(self.process_all_tasks_async(tasks))

        # Strip canonical_phrasing: leak from idea texts before further processing/caching
        import re as _re
        _canonical_pattern = _re.compile(r'\bcanonical_phrasing:\s*')
        for result in self._results:
            if result.response_ideas:
                for idea in result.response_ideas:
                    if idea.idea and 'canonical_phrasing:' in idea.idea:
                        idea.idea = _canonical_pattern.sub('', idea.idea).strip()

        # Persist empirical stats for cold-start calibration on next run
        if len(self.latency_tracker.values) >= 5 and self.actual_total_tokens:
            tokens = list(self.actual_total_tokens)
            total_tasks = self.stats['tasks_successful'] + self.stats['tasks_failed'] + self.stats['timeouts']
            measurements = {
                "p50_latency_s": self.latency_tracker.get_p50(),
                "p95_latency_s": self.latency_tracker.get_p95(),
                "p99_latency_s": self.latency_tracker.get_p99(),
                "avg_tokens": sum(tokens) / len(tokens),
                "had_timeouts": self.stats['timeouts'] > 0,
            }
            if self.tiktoken_offset_learner.is_learned():
                measurements["tiktoken_offset"] = float(self.tiktoken_offset_learner.get_offset())
            if total_tasks > 0:
                measurements["timeout_rate"] = self.stats['timeouts'] / total_tasks
            # Empirical capacity: the discovered sweet spot (HOLDING concurrency), not the panic-cut value
            holding = self._concurrency_sm.holding_concurrency if self._concurrency_sm else None
            measurements["empirical_capacity"] = float(holding or self.optimal_concurrency)
            update_phase_stats(self._perf_stats, self.model, "step3_idea_extraction",
                               measurements, len(self.actual_total_tokens))
            save_stats(self._perf_stats)

        self._stats.output_count = len(self._results)
        self._stats.end_timing()

        unique_ideas = set()
        multi_idea_responses = 0
        total_idea_length = 0
        idea_count = 0

        response_examples = []
        for resp in self._results:
            if resp.response_ideas and len(resp.response_ideas) > 0:
                if len(resp.response_ideas) > 1:
                    multi_idea_responses += 1

                valid_ideas = []
                for idea in resp.response_ideas:
                    if idea.idea and not idea.idea.startswith("PROCESSING_ERROR") and idea.idea not in ["NA", "NOT_PROCESSED"]:
                        unique_ideas.add(idea.idea)
                        idea_words = idea.idea.split()
                        total_idea_length += len(idea_words)
                        idea_count += 1
                        # Store full idea info including taxonomy
                        valid_ideas.append({
                            'idea': idea.idea,
                            'instance': idea.instance,
                            'facet': idea.facet,
                            'domain': idea.domain,
                            'valence': idea.valence,
                        })

                if valid_ideas and len(response_examples) < self.config.max_code_examples:
                    response_examples.append({
                        'response': resp.response,
                        'ideas': valid_ideas
                    })

        self.verbose_reporter.stat_line(f"Total responses processed: {len(self._results)}")
        self.verbose_reporter.stat_line(f"Total ideas extracted: {idea_count}")
        self.verbose_reporter.stat_line(f"Unique ideas identified: {len(unique_ideas)}")
        if multi_idea_responses > 0:
            single_idea_responses = len([r for r in self._results if r.response_ideas and len(r.response_ideas) == 1])
            self.verbose_reporter.stat_line(f"Single idea responses: {single_idea_responses} ({single_idea_responses/len(self._results)*100:.1f}%)")
            self.verbose_reporter.stat_line(f"Multiple idea responses: {multi_idea_responses} ({multi_idea_responses/len(self._results)*100:.1f}%)")

        single_idea_responses = len([r for r in self._results if r.response_ideas and len(r.response_ideas) == 1]) if multi_idea_responses > 0 else 0
        self.stats = {
            'total_responses': len(self._results),
            'total_ideas': idea_count,
            'unique_ideas': len(unique_ideas),
            'single_idea_responses': single_idea_responses,
            'multi_idea_responses': multi_idea_responses,
            'single_idea_percentage': (single_idea_responses / len(self._results) * 100) if len(self._results) > 0 and multi_idea_responses > 0 else 0,
            'multi_idea_percentage': (multi_idea_responses / len(self._results) * 100) if len(self._results) > 0 else 0
        }

        self.verbose_reporter.step_complete("Idea extraction completed")

        return self._results
