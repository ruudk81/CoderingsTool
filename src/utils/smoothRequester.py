"""
SmoothRequester — orchestrator, assembler and executor of concurrent LLM API requests.

Manages rate pacing (RPM/TPM) AND concurrency control simultaneously.
Two systems, selected by header availability from one probe call:
  System A (server-side data): residual latency drift + passive rate rails
  System B (client-side data): P50-drift + PID rate pacing

The caller provides: task list, process function, config params.
Everything else is internal: workers, dispatch, gates, monitoring, retry, cache.

Usage:
    requester = SmoothRequester(
        model="gpt-5.4-nano",
        phase_key="step3_idea_extraction",
    )
    results = await requester.process_all(tasks, process_fn, fallback_fn)
"""

import asyncio
import logging
import math
import time
from collections import deque, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from aiolimiter import AsyncLimiter
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException

from config import (
    API_PROVIDER, OPENAI_API_KEY,
    FALLBACK_TPM, FALLBACK_RPM, ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
)
from utils.llm import (
    create_client, llm_create_async, RateLimits,
    fetch_rate_limits as llm_fetch_rate_limits,
    HeaderCaptureTransport,
)
from utils.perfModel import perf_model
from utils.cached_resources import get_tiktoken_encoding

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DEFAULTS (overridable via constructor)
# =============================================================================

COLD_START_CAP = 50
DEFAULT_AVG_TOKENS = 1500
DEFAULT_LATENCY_SECONDS = 2.0
TIMEOUT_FLOOR_SECONDS = 10.0
DEFAULT_TIMEOUT_SECONDS = 10.0
HEADROOM = 0.9
WARM_UP_SAMPLE_MIN = 15
WARM_UP_SAMPLE_MAX = 30
THROUGHPUT_ADJUSTMENT_MIN_SAMPLES = 10
THROUGHPUT_ADJUSTMENT_THRESHOLD = 1.05
ADJUSTMENT_INTERVAL = 20  # seconds between PID adjustments (System B)
DEFAULT_OUTPUT_RATIO = 0.25
DISPATCH_DELAY_P50_THRESHOLD = 5.0   # seconds — no delay below this P50
DISPATCH_DELAY_SPREAD_FACTOR = 12    # proportionality: delay = (p50 - threshold) / factor


# =============================================================================
# BUILDING BLOCKS — rate pacing and concurrency control components
# =============================================================================

class TokenBucket:
    """Token bucket for TPM limiting."""

    def __init__(self, tokens_per_minute: int, max_acquire_attempts: int = 1000):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
        self._max_attempts = max_acquire_attempts

    async def acquire(self, tokens_needed):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now
            if self.available >= tokens_needed:
                self.available -= tokens_needed
                return True
            else:
                return (tokens_needed - self.available) * 60 / self.tpm

    async def wait_and_acquire(self, tokens_needed):
        for _ in range(self._max_attempts):
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            await asyncio.sleep(result)
        raise RuntimeError(f"Failed to acquire {tokens_needed} tokens")

    async def reconcile(self, delta_tokens: int):
        if delta_tokens < 0:
            async with self.lock:
                self.available = min(self.tpm, self.available - delta_tokens)


class ConcurrencyGate:
    """Semaphore with dynamic limit adjustment."""

    def __init__(self, limit: int):
        self._limit = limit
        self._active = 0
        self._waiters: deque = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def active(self) -> int:
        return self._active

    def set_limit(self, new_limit: int):
        self._limit = max(1, new_limit)
        self._wake_waiters()

    def _wake_waiters(self):
        while self._waiters and self._active < self._limit:
            fut = self._waiters.popleft()
            if not fut.done():
                self._active += 1
                fut.set_result(True)

    async def acquire(self):
        if self._active < self._limit:
            self._active += 1
            return
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self._active -= 1
                self._wake_waiters()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self):
        self._active -= 1
        self._wake_waiters()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        self.release()


class LatencyTracker:
    """EMA tracker with adaptive timeout."""

    def __init__(self, ema_alpha=0.1, samples_window=100,
                 timeout_floor=TIMEOUT_FLOOR_SECONDS, default_timeout=DEFAULT_TIMEOUT_SECONDS,
                 timeout_multiplier=6.0):
        self.timeout_floor = timeout_floor
        self.default_timeout = default_timeout
        self.timeout_multiplier = timeout_multiplier
        self.ema = None
        self.alpha = ema_alpha
        self.values = deque(maxlen=samples_window)
        self.retry_mode = False

    def add(self, value):
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

    def get_timeout(self):
        m = self.timeout_multiplier
        if self.retry_mode:
            if not self.values:
                return 180.0
            p50 = float(np.percentile(list(self.values), 50))
            retry_floor = self.retry_mode if isinstance(self.retry_mode, (int, float)) else 60.0
            return max(retry_floor, min(p50 * m, 180.0))
        if not self.values:
            return max(self.timeout_floor, self.default_timeout)
        p50 = float(np.percentile(list(self.values), 50))
        return max(self.timeout_floor, min(p50 * m, 180.0))

    def get_p50(self):
        if len(self.values) >= 2:
            return float(np.percentile(list(self.values), 50))
        return self.ema or DEFAULT_LATENCY_SECONDS


class TiktokenOffsetLearner:
    """Learns tiktoken-to-API token offset."""

    def __init__(self, default_offset=300, history_maxlen=30, min_samples=5):
        self.default_offset = default_offset
        self.offsets = deque(maxlen=history_maxlen)
        self._learned_offset = None
        self._min_samples = min_samples

    def record(self, tiktoken_count, api_count):
        self.offsets.append(api_count - tiktoken_count)
        if len(self.offsets) >= self._min_samples:
            self._learned_offset = int(sum(self.offsets) / len(self.offsets))

    def get_offset(self):
        return self._learned_offset if self._learned_offset is not None else self.default_offset

    def is_learned(self):
        return len(self.offsets) >= self._min_samples

    def get_stats(self):
        return {
            "samples": len(self.offsets),
            "using_offset": self.get_offset(),
            "is_learned": self.is_learned(),
            "min_offset": min(self.offsets) if self.offsets else None,
            "max_offset": max(self.offsets) if self.offsets else None,
        }


class SimplifiedCircuitBreaker:
    """Timeout rate monitoring — defense-in-depth."""

    def __init__(self, window=100, trip_threshold=0.05, cooldown_s=10.0):
        self._events: deque = deque(maxlen=window)
        self._trip_threshold = trip_threshold
        self._cooldown_s = cooldown_s
        self._state = 'CLOSED'
        self._last_trip_time = None
        self._trip_count = 0

    @property
    def state(self):
        return self._state

    @property
    def trip_count(self):
        return self._trip_count

    def record_completion(self):
        self._events.append('ok')

    def record_timeout(self):
        self._events.append('timeout')

    def check(self):
        now = time.monotonic()
        if self._state == 'CLOSED':
            total = len(self._events)
            if total < 10:
                return None
            rate = sum(1 for e in self._events if e == 'timeout') / total
            if rate > self._trip_threshold:
                self._state = 'OPEN'
                self._last_trip_time = now
                self._trip_count += 1
                print(f"CIRCUIT BREAKER TRIPPED: timeout rate {rate:.1%} ({total} events)")
                return 'tripped'
        elif self._state == 'OPEN':
            if self._last_trip_time and (now - self._last_trip_time) >= self._cooldown_s:
                self._state = 'CLOSED'
        return None


class ConcurrencyCircuitBreaker:
    """Monitors timeout rate in sliding window. Adjusts concurrency on sustained pressure.

    State machine:
      CLOSED     — Normal operation. Monitoring timeout rate.
      OPEN       — Tripped. Concurrency reduced. In cooldown.
      RECOVERING — Cooldown expired, rate OK. Gradually ramping back to baseline.
    """

    def __init__(self, config, gate: ConcurrencyGate, baseline: int):
        self.config = config
        self.gate = gate
        self.baseline = baseline
        self._events: deque = deque()
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
        self._prune_window()
        total = len(self._events)
        if total == 0:
            return 0.0, 0
        timeouts = sum(1 for _, t in self._events if t == 'timeout')
        return timeouts / total, total

    def check_and_adjust(self) -> Optional[str]:
        """Returns 'tripped', 'recovering', 'recovered', or None."""
        now = time.monotonic()
        rate, total = self._get_timeout_rate()

        if self._state == 'CLOSED':
            if total >= self.config.min_events_to_trip and rate > self.config.trip_threshold:
                return self._trip(now, rate, total)
            return None

        elif self._state == 'OPEN':
            elapsed = now - self._last_trip_time if self._last_trip_time else 0
            if elapsed < self.config.cooldown_seconds:
                return None
            if total >= self.config.min_events_to_trip and rate > self.config.trip_threshold:
                return self._trip(now, rate, total)
            self._state = 'RECOVERING'
            self._last_recovery_check = now
            return 'recovering'

        elif self._state == 'RECOVERING':
            if now - (self._last_recovery_check or now) < self.config.recovery_interval_seconds:
                return None
            self._last_recovery_check = now
            if total >= self.config.min_events_to_trip and rate > self.config.trip_threshold:
                return self._trip(now, rate, total)
            current = self.gate.limit
            target = min(self.baseline, int(current * (1.0 + self.config.recovery_step_pct)))
            target = max(target, current + 1)
            if target >= self.baseline:
                self.gate.set_limit(self.baseline)
                self._state = 'CLOSED'
                self._trip_count = 0
                print(f"Circuit breaker recovered: concurrency restored to {self.baseline}")
                return 'recovered'
            self.gate.set_limit(target)
            print(f"Circuit breaker recovering: {current} -> {target} (target: {self.baseline})")
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
        print(f"CIRCUIT BREAKER TRIPPED: timeout rate {rate:.1%} "
              f"({total} events in {self.config.window_seconds}s) | "
              f"concurrency {pre_trip} -> {new_limit} "
              f"(cooldown {self.config.cooldown_seconds}s)")
        return 'tripped'


# =============================================================================
# LITTLE'S LAW
# =============================================================================

@dataclass
class ApiLimits:
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(limits, latency_s, avg_tokens, headroom=0.9):
    latency_s = max(float(latency_s or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)
    rpm_thr = limits.requests_per_minute * headroom / 60
    tpm_thr = limits.tokens_per_minute * headroom / avg_tokens / 60
    return max(math.ceil(min(rpm_thr, tpm_thr) * latency_s), 2)


# =============================================================================
# SYSTEM A: Server-Side Data — residual latency + header pressure
# =============================================================================

class ConcurrencyState(Enum):
    RAMP_UP = "RAMP-UP"
    STEADY = "STEADY"
    BACKOFF = "BACKOFF"
    RECOVER = "RECOVER"


class ResidualLatencyTracker:
    """Tracks residual = observed_latency - openai-processing-ms."""

    def __init__(self, window=200, median_window=20, trend_recent=10, trend_previous=10):
        self._median_window = median_window
        self._trend_recent = trend_recent
        self._trend_previous = trend_previous
        self._entries: deque = deque(maxlen=window)

    def add(self, observed_s, processing_ms):
        residual = max(0.0, observed_s * 1000 - processing_ms)
        self._entries.append((time.monotonic(), residual, processing_ms))

    def _recent_n(self, n):
        return [(r, p) for _, r, p in list(self._entries)[-n:]]

    def _median(self, vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        mid = len(s) // 2
        return (s[mid - 1] + s[mid]) / 2.0 if len(s) % 2 == 0 and len(s) >= 2 else s[mid]

    def median_residual(self):
        return self._median([r for r, _ in self._recent_n(self._median_window)])

    def median_processing(self):
        return self._median([p for _, p in self._recent_n(self._median_window)])

    def normalized_residual(self):
        mp = self.median_processing()
        return self.median_residual() / mp if mp > 0 else 0.0

    def trend(self):
        total = self._trend_recent + self._trend_previous
        if len(self._entries) < total:
            return 1.0
        entries = list(self._entries)
        recent = self._median([r for _, r, _ in entries[-self._trend_recent:]])
        previous = self._median([r for _, r, _ in entries[-(total):-self._trend_recent]])
        return recent / previous if previous > 0 else 1.0

    @property
    def sample_count(self):
        return len(self._entries)


class HeaderAwareConcurrencyController:
    """3-state machine + BACKOFF event, driven by residual latency drift."""

    def __init__(self, starting, ramp_step_pct=0.025, backoff_pct=0.90,
                 min_concurrency=2, drift_steady=1.2, drift_backoff=1.5,
                 drift_resume=1.1, budget_pressure_threshold=0.9):
        self.current = starting
        self.ramp_step = max(2, int(starting * ramp_step_pct))
        self._backoff_pct = backoff_pct
        self._min = min_concurrency
        self._drift_steady = drift_steady
        self._drift_backoff = drift_backoff
        self._drift_resume = drift_resume
        self._budget_threshold = budget_pressure_threshold

        self.state = ConcurrencyState.RAMP_UP
        self.last_healthy_concurrency = starting
        self.residual_baseline = 0.0
        self.residual_drift = 0.0
        self.backoff_ticks = 0
        self.signal_cutoff = 0.0

        self.last_healthy_throughput = 0.0
        self.last_healthy_p50 = 0.0

    def _backoff_cut(self, from_conc):
        return max(self._min, int(from_conc * self._backoff_pct))

    def evaluate(self, median_residual_ms, header_pressure=0.0,
                 throughput=0.0, p50=0.0):
        if throughput > 0 and p50 > 0:
            self.last_healthy_throughput = throughput
            self.last_healthy_p50 = p50

        if self.residual_baseline == 0 and median_residual_ms > 0:
            self.residual_baseline = median_residual_ms

        self.residual_drift = (median_residual_ms / self.residual_baseline
                               if self.residual_baseline > 0 else 1.0)

        drift = self.residual_drift
        stressed = drift > self._drift_backoff or header_pressure > self._budget_threshold

        if stressed:
            self.backoff_ticks += 1
        else:
            self.backoff_ticks = 0

        if self.state == ConcurrencyState.RAMP_UP:
            if stressed:
                self.current = self._backoff_cut(self.last_healthy_concurrency)
                self.signal_cutoff = time.perf_counter()
                self.state = ConcurrencyState.RECOVER
            elif drift > self._drift_steady:
                self.state = ConcurrencyState.STEADY
                self.last_healthy_concurrency = self.current
            else:
                self.last_healthy_concurrency = self.current
                self.current += self.ramp_step

        elif self.state == ConcurrencyState.STEADY:
            if stressed:
                self.current = self._backoff_cut(self.last_healthy_concurrency)
                self.signal_cutoff = time.perf_counter()
                self.state = ConcurrencyState.RECOVER
            elif drift < self._drift_resume:
                self.state = ConcurrencyState.RAMP_UP
                self.current += self.ramp_step

        elif self.state == ConcurrencyState.RECOVER:
            if stressed and self.backoff_ticks >= 4:
                self.current = self._backoff_cut(self.current)
                self.backoff_ticks = 0
                self.signal_cutoff = time.perf_counter()
            elif drift < self._drift_resume:
                self.state = ConcurrencyState.STEADY
                self.last_healthy_concurrency = self.current

        return self.current


# =============================================================================
# SYSTEM B: Client-Side Data — P50-drift concurrency + PID rate pacing
#
# Fallback path for when server-side headers (openai-processing-ms) are not
# available — e.g., Azure OpenAI, proxy setups, or future API changes.
#
# System A uses residual latency (observed - server processing time) as a
# signal independent of our own dispatch decisions. System B only has P50
# latency, which is contaminated by our concurrency changes: reducing
# concurrency lowers P50 (fewer requests = less server load), making it look
# like stress resolved, while increasing concurrency raises P50 from batch
# scheduling noise. The controller therefore optimizes against its own shadow
# rather than discovering the server's true capacity.
#
# Retained as a functional fallback so the pipeline works regardless of
# header availability.
# =============================================================================


class P50DriftConcurrencyController:
    """State machine concurrency controller driven by P50 latency drift.

    Monitors in-flight latency signals and observed throughput to estimate the
    server's concurrency limit. BACKOFF and RECOVER targets are grounded in
    measured throughput × P50, not arbitrary percentages.

    States: RAMP-UP → STEADY ↔ BACKOFF → RECOVER → STEADY

    Signal: P50 drift from baseline.
      >20% drift → STEADY (hold)
      >50% drift for 2 consecutive ticks → BACKOFF
    """

    def __init__(self, starting: int, bottleneck: str = "throughput",
                 config=None):
        from pipeline.step_3_ideaExtractor.config_ideaExtractor import DEFAULT_CONCURRENCY_CONTROL_CONFIG
        self.config = config or DEFAULT_CONCURRENCY_CONTROL_CONFIG
        self.current = starting
        self.starting = starting
        self.bottleneck = bottleneck
        self.ramp_step = max(2, int(starting * self.config.ramp_step_pct))

        self.state = ConcurrencyState.RAMP_UP
        self.last_healthy_concurrency = starting
        self.steady_concurrency = None

        # Throughput-grounded targets
        self.last_healthy_throughput = 0.0
        self.last_healthy_p50 = 0.0

        # P50 drift detection
        self.p50_baseline = 0.0

        # Ratios for reporting (diagnostic only, don't drive transitions)
        self.p95_ratio = 0.0
        self.p100_ratio = 0.0
        self.backoff_ticks = 0
        self.stress_ticks = 0
        self.signal_cutoff = 0.0

    def _throughput_grounded_target(self, fraction: float) -> int:
        """Compute concurrency target from measured throughput × P50."""
        if self.last_healthy_throughput > 0 and self.last_healthy_p50 > 0:
            target = int(fraction * self.last_healthy_throughput * self.last_healthy_p50)
        else:
            target = int(fraction * self.last_healthy_concurrency)
        return max(self.config.min_concurrency, target)

    def evaluate(self, p50: float, inflight_p95: float, inflight_p100: float,
                 now: float, throughput: float = 0.0, inflight: int = 0) -> int:
        """Main tick evaluation. Returns new concurrency."""
        if p50 <= 0:
            return self.current

        if inflight_p100 > 0:
            self.p95_ratio = inflight_p95 / p50
            self.p100_ratio = inflight_p100 / p50

        if self.p50_baseline == 0:
            self.p50_baseline = p50

        p50_drift = p50 / self.p50_baseline
        should_hold = p50_drift > 1.2
        stressed = p50_drift > 1.5

        if stressed:
            self.stress_ticks += 1
        else:
            self.stress_ticks = 0
        should_backoff = self.stress_ticks >= 2

        if throughput > 0 and p50 > 0:
            self.last_healthy_throughput = throughput
            self.last_healthy_p50 = p50

        if self.state == ConcurrencyState.RAMP_UP:
            if should_backoff:
                self.stress_ticks = 0
                self.state = ConcurrencyState.BACKOFF
                self.backoff_ticks = 0
                self.signal_cutoff = time.perf_counter()
                self.current = self._throughput_grounded_target(self.config.backoff_throughput_pct)
            elif should_hold:
                self.state = ConcurrencyState.STEADY
                self.steady_concurrency = self.current
                self.last_healthy_concurrency = self.current
            else:
                self.last_healthy_concurrency = self.current
                self.current = self.current + self.ramp_step

        elif self.state == ConcurrencyState.STEADY:
            self.steady_concurrency = self.current
            if should_backoff:
                self.stress_ticks = 0
                self.state = ConcurrencyState.BACKOFF
                self.backoff_ticks = 0
                self.signal_cutoff = time.perf_counter()
                self.current = self._throughput_grounded_target(self.config.backoff_throughput_pct)
            elif not should_hold:
                self.state = ConcurrencyState.RAMP_UP
                self.current = self.current + self.ramp_step

        elif self.state == ConcurrencyState.BACKOFF:
            self.backoff_ticks += 1
            if self.p100_ratio <= self.config.inflight_ratio:
                self.state = ConcurrencyState.RECOVER
                self.backoff_ticks = 0
            elif self.backoff_ticks >= 3:
                self.current = max(self.config.min_concurrency,
                                   int(self.current * self.config.backoff_throughput_pct))
                self.backoff_ticks = 0
                self.signal_cutoff = time.perf_counter()

        elif self.state == ConcurrencyState.RECOVER:
            if should_backoff:
                self.stress_ticks = 0
                self.state = ConcurrencyState.BACKOFF
                self.backoff_ticks = 0
                self.signal_cutoff = time.perf_counter()
                self.current = self._throughput_grounded_target(self.config.backoff_throughput_pct)
            elif should_hold:
                self.state = ConcurrencyState.STEADY
                self.steady_concurrency = self.current
                self.last_healthy_concurrency = self.current
            else:
                recovery_target = self._throughput_grounded_target(1.0)
                if self.current >= recovery_target:
                    self.state = ConcurrencyState.STEADY
                    self.current = recovery_target
                    self.steady_concurrency = self.current
                    self.last_healthy_concurrency = self.current
                else:
                    recovery_step = max(1, self.ramp_step // 2)
                    self.current = min(self.current + recovery_step, recovery_target)

        return self.current


class P50DriftCircuitBreaker:
    """Timeout rate monitor for System B (P50-drift path).

    Trigger-only — signals the caller to engage BACKOFF, does not manage
    concurrency or recovery itself. Uses a fixed-size event window.

    Lifecycle: CLOSED → detects spike → trips → cooldown → CLOSED.
    """

    def __init__(self, config):
        self.config = config
        self._events: deque = deque(maxlen=config.window_size)
        self._state = 'CLOSED'
        self._last_trip_time: Optional[float] = None
        self._trip_count: int = 0
        self._cooldown_seconds: float = 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def trip_count(self) -> int:
        return self._trip_count

    def record_completion(self):
        self._events.append('ok')

    def record_timeout(self):
        self._events.append('timeout')

    def _get_timeout_rate(self) -> Tuple[float, int]:
        total = len(self._events)
        if total == 0:
            return 0.0, 0
        timeouts = sum(1 for t in self._events if t == 'timeout')
        return timeouts / total, total

    def check(self, drain_time: float = 0.0) -> Optional[str]:
        """Called every tick. Returns 'tripped' or None."""
        now = time.monotonic()
        rate, total = self._get_timeout_rate()

        if self._state == 'CLOSED':
            if total >= self.config.min_events_to_trip and rate > self.config.trip_threshold:
                return self._trip(now, rate, total, drain_time)
            return None

        elif self._state == 'OPEN':
            elapsed = now - self._last_trip_time if self._last_trip_time else 0
            if elapsed < self._cooldown_seconds:
                return None
            self._state = 'CLOSED'
            return None

        return None

    def _trip(self, now: float, rate: float, total: int, drain_time: float) -> str:
        self._cooldown_seconds = max(5.0, drain_time * self.config.cooldown_drain_multiplier)
        self._state = 'OPEN'
        self._last_trip_time = now
        self._trip_count += 1
        print(f"CIRCUIT BREAKER TRIPPED: timeout rate {rate:.1%} "
              f"({total} events) | cooldown {self._cooldown_seconds:.1f}s")
        return 'tripped'


class PIDThroughputController:
    """Asymmetric PID for arrival rate adjustment."""

    def __init__(self, target_utilization=1.0, kp_up=0.4, kp_down=0.2,
                 ki=0.05, kd=0.1, min_adj=0.02, max_adj=0.15):
        self.target = target_utilization
        self.kp_up, self.kp_down = kp_up, kp_down
        self.ki, self.kd = ki, kd
        self.min_adj, self.max_adj = min_adj, max_adj
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def compute_adjustment(self, utilization):
        now = time.monotonic()
        error = self.target - utilization
        dt = max(now - self.last_time, 0.1) if self.last_time else 1.0
        self.last_time = now
        self.integral = max(-0.5, min(0.5, self.integral + error * dt))
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        kp = self.kp_up if error > 0 else self.kp_down
        output = max(-self.max_adj, min(self.max_adj,
                     kp * error + self.ki * self.integral + self.kd * derivative))
        return 1.0 if abs(output) < self.min_adj else 1.0 + output

    def reset(self):
        self.integral = self.last_error = 0.0
        self.last_time = None


class RealTimeTPMTracker:
    """Sliding-window TPM for PID feedback."""
    def __init__(self, window_s=60.0):
        self.window_s = window_s
        self.samples = deque()
        self.lock = asyncio.Lock()

    async def record(self, tokens):
        async with self.lock:
            now = time.monotonic()
            self.samples.append((now, tokens))
            self._prune(now)

    def _prune(self, now):
        cutoff = now - self.window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    async def get_current_tpm(self):
        async with self.lock:
            now = time.monotonic()
            self._prune(now)
            if not self.samples:
                return 0.0
            return sum(t for _, t in self.samples) / max(now - self.samples[0][0], 1.0) * 60


class RealTimeRPMTracker:
    """Sliding-window RPM."""
    def __init__(self, window_s=60.0):
        self.window_s = window_s
        self.samples = deque()
        self.lock = asyncio.Lock()

    async def record(self):
        async with self.lock:
            self.samples.append(time.monotonic())
            self._prune(self.samples[-1])

    def _prune(self, now):
        cutoff = now - self.window_s
        while self.samples and self.samples[0] < cutoff:
            self.samples.popleft()

    async def get_current_rpm(self):
        async with self.lock:
            now = time.monotonic()
            self._prune(now)
            if not self.samples:
                return 0.0
            return len(self.samples) / max(now - self.samples[0], 1.0) * 60


# Archived P50-drift fallback (imported conditionally in _setup_concurrency when no headers)


# =============================================================================
# SMOOTH REQUESTER — the orchestrator
# =============================================================================

class SmoothRequester:
    """Orchestrates concurrent API request processing.

    Manages rate pacing (RPM/TPM) AND concurrency control simultaneously.
    The caller provides tasks and a processing function. Everything else is internal.
    """

    def __init__(
        self,
        model: str,
        phase_key: str = "default",
        num_tasks: int = 0,
        verbose: bool = True,
        processing_config: Optional[ProcessingConfig] = None,
        known_limits: Optional[RateLimits] = None,
        show_setup: bool = True,
        quiet: bool = False,
    ):
        self.model = model
        self.phase_key = phase_key
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self._known_limits = known_limits
        self._show_setup = show_setup
        self._quiet = quiet
        self._headroom = self.processing_config.rate_limit_headroom

        # Tiktoken encoding for token estimation
        self.encoding = get_tiktoken_encoding(model)

        # Token estimation state
        self.avg_tokens = DEFAULT_AVG_TOKENS
        self.input_token_history = deque(maxlen=10)
        self.output_token_history = deque(maxlen=15)
        self.output_ratio_history = deque(maxlen=20)
        self.actual_total_tokens = deque(maxlen=50)
        self.estimation_errors = deque(maxlen=50)

        # Warm start from the parametric performance model.
        self._pred = perf_model.predict(model, phase_key)
        if self._pred.avg_tokens:
            self.avg_tokens = self._pred.avg_tokens
        self.tiktoken_offset_learner = TiktokenOffsetLearner(default_offset=self._pred.tiktoken_offset)

        # Dispatch delay: stagger heavy tasks to avoid server batch congestion
        p50_estimate = self._pred.p50_latency_s or DEFAULT_TIMEOUT_SECONDS / 5
        self._dispatch_delay = max(0.0, (p50_estimate - DISPATCH_DELAY_P50_THRESHOLD) / DISPATCH_DELAY_SPREAD_FACTOR)

        # Latency tracker — timeout strategy depends on whether we have a calibrated
        # concurrency ceiling. Without one, the phase is unconstrained (no rate or
        # throughput pressure) so we use 180s and let the server finish. With a
        # calibrated ceiling, we use the adaptive multiplier to cut outliers.
        _timeout = self._pred.timeout_s or TIMEOUT_FLOOR_SECONDS
        if self._pred.concurrency is None:
            # Unconstrained: no calibrated ceiling → fixed 180s timeout
            _timeout = 180.0
            _multiplier = 1.0
        else:
            # Calibrated: adaptive multiplier scales with task count
            _multiplier = min(6, round(math.log(max(num_tasks, 1)) + 1))
        self.latency_tracker = LatencyTracker(
            ema_alpha=self.processing_config.latency_tracker_ema_alpha,
            samples_window=self.processing_config.latency_tracker_samples_window,
            timeout_floor=_timeout,
            default_timeout=_timeout,
            timeout_multiplier=float(_multiplier),
        )

        # Rate limiting components (initialized in _setup)
        self.rate_limiter = None
        self.tpm_bucket = None
        self.semaphore = None
        self.optimal_concurrency = 0
        self.current_arrival_rate = None
        self.rate_limits = RateLimits(FALLBACK_TPM, FALLBACK_RPM)

        # System selection (set after probe)
        self._has_server_headers = False
        self._header_transport = None
        self._concurrency_controller = None
        self.circuit_breaker = None
        self._residual_tracker = None
        self.tpm_tracker = None
        self.rpm_tracker = None
        self.pid_controller = None

        # State tracking
        self._inflight_starts = {}
        self._warm_up_calibrated = False
        self._warm_up_target_samples = WARM_UP_SAMPLE_MIN
        self._last_remaining_requests = 0
        self._last_limit_requests = 0

        # Stats
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'timeouts': 0,
            'rate_limits': 0,
        }
        self.failed_task_ids = set()
        self.failure_log = []

        self.verbose = verbose

    # === PROBE + SETUP ========================================================

    async def _probe_and_setup(self, num_tasks: int):
        """Probe API for rate limits + headers, then set up all components."""
        # Create client with header capture
        self.client = create_client(self.model, async_mode=True, capture_headers=True)
        self._header_transport = getattr(self.client, '_header_transport', None)

        if self._known_limits is not None:
            # Skip probe — use caller-provided limits; assume header support
            limits = self._known_limits
            has_server_headers = True
        else:
            # Probe call
            limits, has_server_headers = await self._fetch_rate_limits()

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            limits = RateLimits(FALLBACK_TPM, FALLBACK_RPM)
        self.rate_limits = limits
        self._has_server_headers = has_server_headers

        # Set up rate pacing + concurrency
        self._setup_rate_pacing(limits)
        self._setup_concurrency(limits, num_tasks, has_server_headers)

        # Warm-up sample size
        if num_tasks <= 50:
            self._warm_up_target_samples = WARM_UP_SAMPLE_MIN
        elif num_tasks >= 500:
            self._warm_up_target_samples = WARM_UP_SAMPLE_MAX
        else:
            frac = (num_tasks - 50) / 450
            self._warm_up_target_samples = int(WARM_UP_SAMPLE_MIN + frac * (WARM_UP_SAMPLE_MAX - WARM_UP_SAMPLE_MIN))

    async def _fetch_rate_limits(self) -> Tuple[RateLimits, bool]:
        """Probe call: discover rate limits + header availability."""
        return await llm_fetch_rate_limits(self.model)

    def _setup_rate_pacing(self, limits):
        """Create TokenBucket + AsyncLimiter (always active, both systems)."""
        headroom = self._headroom
        arrival_rate = min(
            limits.requests_per_minute * headroom / 60,
            limits.tokens_per_minute * headroom / self.avg_tokens / 60
        )
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / arrival_rate)
        self.current_arrival_rate = arrival_rate
        self.tpm_bucket = TokenBucket(int(limits.tokens_per_minute * headroom))

    def _setup_concurrency(self, limits, num_tasks, has_server_headers):
        """Set up integrated rate + concurrency control.

        Both controls always created. The binding constraint (rate vs concurrency)
        is determined continuously by min(rate_limit_concurrency, server_concurrency).
        """
        headroom = self._headroom
        avg_latency = self._pred.p50_latency_s or DEFAULT_LATENCY_SECONDS

        # Rate-limit concurrency: what rate limits allow (Little's Law)
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        self._rate_limit_concurrency = compute_optimal_concurrency(
            api_limits, avg_latency, self.avg_tokens, headroom
        )

        # Server concurrency: from predicted capacity or cold start
        if self._pred.concurrency is not None:
            self._server_concurrency = int(self._pred.concurrency)
        else:
            self._server_concurrency = COLD_START_CAP

        # Effective = min(rate, server, tasks)
        effective = min(self._rate_limit_concurrency, self._server_concurrency, num_tasks)
        effective = max(effective, 2)

        self.semaphore = ConcurrencyGate(effective)
        self.optimal_concurrency = effective

        # Which rate limit is tighter (for display)
        rpm_thr = limits.requests_per_minute * headroom / 60
        tpm_thr = limits.tokens_per_minute * headroom / self.avg_tokens / 60
        self._rate_bottleneck = "RPM" if rpm_thr < tpm_thr else "TPM"

        # --- Concurrency controller (depends on header availability) ---
        if has_server_headers:
            self._residual_tracker = ResidualLatencyTracker()
            self._concurrency_controller = HeaderAwareConcurrencyController(starting=effective)
            self.circuit_breaker = SimplifiedCircuitBreaker()
        else:
            from pipeline.step_3_ideaExtractor.config_ideaExtractor import DEFAULT_CONCURRENCY_CONTROL_CONFIG, DEFAULT_CIRCUIT_BREAKER_CONFIG
            self._concurrency_controller = P50DriftConcurrencyController(
                starting=effective, bottleneck="throughput",
                config=DEFAULT_CONCURRENCY_CONTROL_CONFIG,
            )
            self.circuit_breaker = P50DriftCircuitBreaker(config=DEFAULT_CIRCUIT_BREAKER_CONFIG)
            self._residual_tracker = None

        # --- Rate pacing controller (always created) ---
        self.tpm_tracker = RealTimeTPMTracker()
        self.rpm_tracker = RealTimeRPMTracker()
        self.pid_controller = PIDThroughputController()


    def _recalculate_rate_limit_concurrency(self):
        """Recalculate rate-limit concurrency from current avg_tokens and latency."""
        headroom = self._headroom
        latency = self.latency_tracker.get_p50() if self.latency_tracker.values else DEFAULT_LATENCY_SECONDS
        api_limits = ApiLimits(self.rate_limits.tokens_per_minute, self.rate_limits.requests_per_minute)
        self._rate_limit_concurrency = compute_optimal_concurrency(
            api_limits, latency, self.avg_tokens, headroom
        )

    # === TOKEN ESTIMATION =====================================================

    def estimate_tokens(self, prompt: str) -> int:
        """Adaptive token estimation with learned offset and safety margins."""
        tiktoken_count = len(self.encoding.encode(prompt))
        offset = self.tiktoken_offset_learner.get_offset()
        actual_input = tiktoken_count + offset

        n = len(self.estimation_errors)
        margin = 1.15 if n < 5 else (1.10 if n < 15 else 1.05)

        if len(self.input_token_history) >= 5:
            avg_in = sum(self.input_token_history) / len(self.input_token_history)
            est_input = int(0.7 * avg_in + 0.3 * actual_input)
        else:
            est_input = int(actual_input * margin)

        self.input_token_history.append(actual_input)

        if len(self.output_ratio_history) >= 5:
            ratio = sum(self.output_ratio_history) / len(self.output_ratio_history)
            est_output = int(est_input * ratio * margin)
        elif len(self.output_token_history) >= 3:
            est_output = int(sum(self.output_token_history) / len(self.output_token_history) * margin)
        else:
            est_output = int(est_input * DEFAULT_OUTPUT_RATIO * margin)

        return est_input + min(4000, est_output)

    # === EXECUTE TASK (gate + outcome recording) ==============================

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def _execute_task(self, task, prepare_fn, parse_fn):
        """Full task execution: prepare → gate → LLM call → headers → tokens → parse.

        The smoothRequester owns the LLM call so it can read headers and reconcile tokens.
        prepare_fn: builds prompt + call parameters (step-specific)
        parse_fn: parses raw LLM response into step result (step-specific)
        """
        task_id = task.get('respondent_id', task.get('task_index', id(task)))

        # Step 1: prepare (build prompt, response model — step-specific)
        call_params = prepare_fn(task)
        prompt = call_params['prompt']
        est_tokens = self.estimate_tokens(prompt)
        est_in = len(self.encoding.encode(prompt))

        async with self.semaphore:
            timeout = self.latency_tracker.get_timeout()
            await self.tpm_bucket.wait_and_acquire(est_tokens)

            try:
                # Step 2: LLM call (owned by smoothRequester for header access)
                # api_start is AFTER all gates so latency measures only API time
                async with self.rate_limiter:
                    api_start = time.perf_counter()
                    self._inflight_starts[task_id] = api_start
                    conc_at_dispatch = len(self._inflight_starts)
                    response = await asyncio.wait_for(
                        llm_create_async(
                            client=self.client,
                            model=self.model,
                            prompt=prompt,
                            response_model=call_params.get('response_model'),
                            temperature=call_params.get('temperature', 0.0),
                            max_tokens=call_params.get('max_tokens', 4000),
                            max_retries=call_params.get('max_retries', 3),
                            **call_params.get('extra_kwargs', {}),
                        ),
                        timeout=timeout
                    )

                latency = time.perf_counter() - api_start
                self._inflight_starts.pop(task_id, None)

                if self.circuit_breaker:
                    self.circuit_breaker.record_completion()

                # Step 3: Header reading (System A)
                server_processing_ms = 0
                if self._header_transport and self._residual_tracker is not None:
                    client_id = getattr(response, '_client_request_id', None)
                    if client_id:
                        entry = self._header_transport.get(client_id)
                        if entry and entry['processing_ms'] > 0:
                            server_processing_ms = entry['processing_ms']
                            self._residual_tracker.add(latency, server_processing_ms)
                        if entry and entry.get('remaining_requests', 0) > 0:
                            self._last_remaining_requests = entry['remaining_requests']
                        if entry and entry.get('limit_requests', 0) > 0:
                            self._last_limit_requests = entry['limit_requests']

                # Step 4: Token reconciliation
                raw = getattr(response, '_raw_response', None)
                usage = getattr(raw, 'usage', None) if raw else getattr(response, 'usage', None)
                if usage:
                    actual_in = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0)
                    actual_out = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0)
                    actual_total = getattr(usage, 'total_tokens', 0) or (actual_in + actual_out)

                    self.output_token_history.append(actual_out)
                    if actual_in > 0:
                        self.output_ratio_history.append(actual_out / actual_in)
                    self.actual_total_tokens.append(actual_total)
                    self.estimation_errors.append(abs(actual_total - est_tokens))

                    delta = actual_total - est_tokens
                    await self.tpm_bucket.reconcile(delta)

                    self.tiktoken_offset_learner.record(est_in, actual_in)
                    perf_model.observe(self.model, self.phase_key, actual_in, actual_out,
                                        latency, conc_at_dispatch, False, est_in)

                    if self.tpm_tracker:
                        await self.tpm_tracker.record(actual_total)
                    if self.rpm_tracker:
                        await self.rpm_tracker.record()

                # Step 5: Parse response (step-specific)
                result = parse_fn(task, response)

                self.stats['tasks_processed'] += 1
                if result is not None:
                    self.stats['tasks_successful'] += 1
                    # Only record latency for clean successes — keeps P50 clean
                    # for Little's Law and timeout calculation
                    self.latency_tracker.add(latency)
                return result

            except asyncio.TimeoutError:
                self._inflight_starts.pop(task_id, None)
                self.stats['tasks_processed'] += 1
                self.stats['timeouts'] += 1
                if self.circuit_breaker:
                    self.circuit_breaker.record_timeout()
                perf_model.observe(self.model, self.phase_key, est_in, 0, timeout, conc_at_dispatch, True)
                return None  # collected for retry

            except (RateLimitError, APIConnectionError, InternalServerError):
                self._inflight_starts.pop(task_id, None)
                raise  # tenacity retries these

            except InstructorRetryException:
                self._inflight_starts.pop(task_id, None)
                raise  # propagates to worker

            except Exception:
                self._inflight_starts.pop(task_id, None)
                raise

    # === WORKER ===============================================================

    async def _worker(self, queue, results, timed_out, prepare_fn, parse_fn, fallback_fn):
        """Generic worker: pull task, execute, handle outcomes."""
        while True:
            task = None
            try:
                task = await queue.get()
                if task is None:
                    break

                task_index, task_data = task

                # Stagger dispatch for heavy tasks (P50 > threshold)
                if self._dispatch_delay > 0:
                    async with self._dispatch_lock:
                        seq = self._dispatch_seq
                        self._dispatch_seq += 1
                    if seq > 0:
                        target_time = self._dispatch_start + (seq * self._dispatch_delay)
                        now = time.perf_counter()
                        if target_time > now:
                            await asyncio.sleep(target_time - now)

                result = await self._execute_task(task_data, prepare_fn, parse_fn)
                if result is None:
                    timed_out.append((task_index, task_data))
                else:
                    results[task_index] = result

            except Exception as e:
                error_type = type(e).__name__
                error_str = str(e)
                if "429" in error_str:
                    limit_type = "TPM" if "token" in error_str.lower() else "RPM"
                    error_type = f"RateLimit_{limit_type}"
                    task_id = task_data.get('respondent_id', '?') if task else '?'
                    print(f"429 {limit_type} limit (task {task_id})")
                else:
                    logger.error(f"Task failed: {e}")

                self.stats['tasks_failed'] += 1
                if task is not None:
                    task_index, task_data = task
                    self.failed_task_ids.add(str(task_data.get('respondent_id', '?')))
                    self.failure_log.append({
                        'task_id': task_data.get('respondent_id', '?'),
                        'reason': 'exception',
                        'error_type': error_type,
                    })
                    if fallback_fn:
                        results[task_index] = fallback_fn(task_data, error_type)
            finally:
                if task is not None:
                    queue.task_done()

    # === TICK (both controllers always active, one report line) ==================

    def _tick(self, completed, total, tick_rate, p50, throughput, active, concurrency,
              current_tpm, effective_tpm, current_rpm, effective_rpm, num_tasks):
        """Both controllers always active. One unified report line.

        PID adjusts arrival rate (rate pacing).
        State machine adjusts semaphore (concurrency control).
        Effective concurrency = min(rate_limit, server_limit).
        Neither interferes with the other because latency is measured after all gates.
        """
        warmup_elapsed = time.time() - self._start_time
        drain = active / throughput if throughput > 0 else 0
        sm = self._concurrency_controller

        # --- Rate pacing: recalculate rate_limit_concurrency ---
        self._recalculate_rate_limit_concurrency()

        # --- Concurrency control: state machine evaluation ---
        state_str = ""
        if self._has_server_headers:
            state_str = self._evaluate_concurrency_header_aware(sm, throughput, p50, warmup_elapsed)
        else:
            state_str = self._evaluate_concurrency_p50_drift(sm, p50, throughput, active, warmup_elapsed, drain)

        # --- Circuit breaker ---
        if self.circuit_breaker:
            if self._has_server_headers:
                cb = self.circuit_breaker.check()
            else:
                cb = self.circuit_breaker.check(drain_time=drain)
            if cb == 'tripped':
                if hasattr(sm, '_backoff_cut'):
                    sm.current = sm._backoff_cut(sm.last_healthy_concurrency)
                    sm.signal_cutoff = time.perf_counter()
                    sm.state = ConcurrencyState.RECOVER
                elif hasattr(sm, '_throughput_grounded_target'):
                    sm.state = ConcurrencyState.BACKOFF
                    sm.backoff_ticks = 0
                    sm.signal_cutoff = time.perf_counter()
                    sm.current = sm._throughput_grounded_target(sm.config.backoff_throughput_pct)
                self._server_concurrency = sm.current

        # --- Effective concurrency: min(rate, server) ---
        effective = min(self._rate_limit_concurrency, self._server_concurrency, num_tasks)
        effective = max(effective, 2)
        if effective != self.optimal_concurrency:
            old = self.optimal_concurrency
            self.semaphore.set_limit(effective)
            self.optimal_concurrency = effective

        # --- Rate pacing display ---
        tpm_pct = current_tpm / effective_tpm * 100 if effective_tpm else 0
        rpm_pct = current_rpm / effective_rpm * 100 if effective_rpm else 0
        if self._rate_bottleneck == "TPM":
            rate_val = current_tpm / 60
            limit_val = effective_tpm / 60
            fmt = lambda v: f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}"
            pace_str = f"tok:{fmt(rate_val)}/{fmt(limit_val)} ({tpm_pct:.0f}%)"
        else:
            pace_str = f"req:{current_rpm/60:.1f}/{effective_rpm/60:.1f} ({rpm_pct:.0f}%)"

        # --- Concurrency display ---
        conc_str = f" conc:{concurrency}" if concurrency != active else ""

        # --- Latency display: residual current/baseline (drift%) ---
        # System A: residual = observed - openai-processing-ms (server-side)
        # System B: residual = P50 observed latency (client-side)
        # Same format, same label, different data source
        latency_str = ""
        if self._has_server_headers and self._residual_tracker and self._residual_tracker.sample_count >= 5:
            current = self._residual_tracker.median_residual()
            baseline = sm.residual_baseline if sm else 0.0
            drift = sm.residual_drift if sm else 0.0
            drift_pct = int((drift - 1) * 100)
            drift_str = f"+{drift_pct}%" if drift_pct >= 0 else f"{drift_pct}%"
            latency_str = f" | thru:{current:.0f}ms/{baseline:.0f}ms ({drift_str})"
        elif not self._has_server_headers and self.latency_tracker.values:
            current_ms = p50 * 1000
            baseline_val = sm.p50_baseline if sm and hasattr(sm, 'p50_baseline') and sm.p50_baseline > 0 else p50
            baseline_ms = baseline_val * 1000
            drift_pct = int((p50 / baseline_val - 1) * 100) if baseline_val > 0 else 0
            drift_str = f"+{drift_pct}%" if drift_pct >= 0 else f"{drift_pct}%"
            latency_str = f" | thru:{current_ms:.0f}ms/{baseline_ms:.0f}ms ({drift_str})"

        # --- Dynamic state: which constraint actually binds ---
        is_rate_capped = (effective <= self._rate_limit_concurrency
                          and self._rate_limit_concurrency <= self._server_concurrency)
        if warmup_elapsed < 5.0:
            control_str = "WARM-UP"
        elif is_rate_capped:
            control_str = "RATE-CAPPED"
        else:
            control_str = state_str.strip() if state_str.strip() else "—"

        timeout_info = f" | deferred:{self.stats['timeouts']}" if self.stats['timeouts'] > 0 else ""

        return (f"[{self.phase_key}] {completed}/{total} | inflight:{active}{conc_str} | {pace_str}"
                f"{latency_str} | completing:{tick_rate:.0f}/s | {control_str}{timeout_info}")

    def _evaluate_concurrency_header_aware(self, sm, throughput, p50, warmup_elapsed):
        """Evaluate header-aware state machine. Updates server_concurrency. Returns state string."""
        tracker = self._residual_tracker
        if not sm or not tracker or tracker.sample_count < 5 or warmup_elapsed < 5.0:
            return " WARM-UP" if warmup_elapsed < 5.0 else ""

        med_res = tracker.median_residual()
        header_pressure = 0.0
        if self._last_limit_requests > 0:
            header_pressure = 1.0 - (self._last_remaining_requests / self._last_limit_requests)

        old_state = sm.state
        old_conc = sm.current
        new_conc = sm.evaluate(med_res, header_pressure, throughput, p50)
        self._server_concurrency = new_conc

        # Show BACKOFF as state on the tick when the cut happens
        if sm.state == ConcurrencyState.RECOVER and old_state != ConcurrencyState.RECOVER and new_conc < old_conc:
            return f" BACKOFF ({old_conc}→{new_conc})"

        return f" {sm.state.value}"

    def _evaluate_concurrency_p50_drift(self, sm, p50, throughput, active, warmup_elapsed, drain):
        """Evaluate P50-drift state machine. Updates server_concurrency. Returns state string."""
        if not sm or p50 <= 0 or warmup_elapsed < 5.0:
            return " WARM-UP" if warmup_elapsed < 5.0 else ""

        # In-flight P95/P100
        now_perf = time.perf_counter()
        cutoff = sm.signal_cutoff if sm else 0.0
        fresh = {k: v for k, v in self._inflight_starts.items() if v >= cutoff}
        inflight_p95 = inflight_p100 = 0.0
        if fresh:
            durations = sorted(now_perf - s for s in fresh.values())
            inflight_p100 = durations[-1]
            inflight_p95 = durations[min(int(len(durations) * 0.95), len(durations) - 1)]

        old_state = sm.state
        old_conc = sm.current
        new_conc = sm.evaluate(
            p50=p50, inflight_p95=inflight_p95, inflight_p100=inflight_p100,
            now=time.monotonic(), throughput=throughput, inflight=active)
        self._server_concurrency = new_conc

        # Show BACKOFF as state on the tick when the cut happens
        if sm.state.value in ("RECOVER", "BACKOFF") and old_state.value not in ("RECOVER", "BACKOFF") and new_conc < old_conc:
            return f" BACKOFF ({old_conc}→{new_conc})"

        return f" {sm.state.value}"

    # === WARM-UP + PID ========================================================

    def _calibrate_tokens(self):
        """Shared warm-up: recalibrate avg_tokens and arrival rate."""
        measured_avg = int(np.mean(list(self.actual_total_tokens)))
        old_avg = self.avg_tokens
        self.avg_tokens = measured_avg
        self._warm_up_measured_latency = float(np.percentile(list(self.latency_tracker.values), 10))

        headroom = self._headroom
        new_rate = min(
            self.rate_limits.requests_per_minute * headroom / 60,
            self.rate_limits.tokens_per_minute * headroom / measured_avg / 60
        )
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / new_rate)
        self.current_arrival_rate = new_rate

        if self.pid_controller:
            self.pid_controller.reset()
        self._warm_up_calibrated = True
        return old_avg, measured_avg

    def _calibrate_concurrency(self, num_tasks):
        """Warm-up: recalibrate tokens, then recalculate effective concurrency."""
        old_avg, new_avg = self._calibrate_tokens()
        old_conc = self.optimal_concurrency

        # Recalculate rate_limit_concurrency with measured data
        self._recalculate_rate_limit_concurrency()

        # Effective = min(rate, server, tasks) — same logic as tick
        new_conc = min(self._rate_limit_concurrency, self._server_concurrency, num_tasks)
        new_conc = max(new_conc, 2)

        if new_conc != old_conc:
            self.semaphore.set_limit(new_conc)
            self.optimal_concurrency = new_conc
            print(f"[WARM-UP] concurrency {old_conc} → {new_conc} (avg_tokens {old_avg} → {new_avg})")
        else:
            print(f"[WARM-UP] avg_tokens {old_avg} → {new_avg}, concurrency unchanged at {old_conc}")

    def _adjust_throughput_if_needed(self):
        """Token estimate correction when actual exceeds estimate by >5%."""
        if len(self.actual_total_tokens) < THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
            return False
        actual_avg = sum(self.actual_total_tokens) / len(self.actual_total_tokens)
        ratio = actual_avg / self.avg_tokens if self.avg_tokens > 0 else 1.0
        if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
            return False
        old = self.avg_tokens
        self.avg_tokens = int(actual_avg)
        print(f"[TOKEN CORRECTION] avg_tokens: {old} → {self.avg_tokens} (actual {actual_avg:.0f}, +{(ratio-1)*100:.0f}%)")
        return True

    async def _apply_pid(self):
        """System B: PID arrival rate adjustment."""
        if not self.tpm_tracker or not self.pid_controller:
            return
        current_tpm = await self.tpm_tracker.get_current_tpm()
        effective = self.rate_limits.tokens_per_minute * self._headroom
        utilization = current_tpm / effective if effective > 0 else 0.0
        adjustment = self.pid_controller.compute_adjustment(utilization)
        if abs(adjustment - 1.0) < 0.01:
            return
        old_rate = self.current_arrival_rate
        new_rate = max(0.5, min(
            self.rate_limits.requests_per_minute * self._headroom / 60,
            old_rate * adjustment
        ))
        if abs(new_rate - old_rate) / max(old_rate, 0.001) < 0.02:
            return
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / new_rate)
        self.current_arrival_rate = new_rate

    # === PROCESS ALL ==========================================================

    async def process_all(self, tasks: List[Dict], prepare_fn, parse_fn,
                          fallback_fn=None) -> List:
        """Main entry point. Processes all tasks with rate pacing + concurrency control.

        Args:
            tasks: list of task dicts
            prepare_fn: fn(task) -> dict with {prompt, response_model, temperature, max_tokens, ...}
            parse_fn: fn(task, response) -> result or None
            fallback_fn: fn(task, reason) -> fallback result (optional)

        Returns: list of results (same order as tasks), None for permanently failed.
        """
        if not tasks:
            return []

        num_tasks = len(tasks)
        await self._probe_and_setup(num_tasks)

        # Print setup (gated by show_setup)
        if self._show_setup:
            headroom = self._headroom
            print("\nRATE LIMITING SETUP")
            print(f"- Model: {self.model}")
            print(f"- RPM limit: {self.rate_limits.requests_per_minute:,} ({self.rate_limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"- TPM limit: {self.rate_limits.tokens_per_minute:,} ({self.rate_limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"- Warm start: {self._pred.origin_line()}")
            token_src = self._pred.origins.get("avg_tokens", "default")
            print(f"- Initial avg_tokens ({token_src}): {self.avg_tokens}")
            print(f"- Target concurrency: {self.optimal_concurrency}")
            print(f"- System: {'A (header-aware)' if self._has_server_headers else 'B (client-side)'}")
            print(f"- Rate limit concurrency: {self._rate_limit_concurrency} | Server concurrency: {self._server_concurrency}")
            if self._dispatch_delay > 0:
                spread = self._dispatch_delay * (num_tasks - 1)
                p50_src = self._pred.origins.get("p50_latency_s", "default")
                print(f"- Dispatch delay ({p50_src}): {self._dispatch_delay:.2f}s/task → {spread:.1f}s spread over {num_tasks} tasks")
            print(f"- Processing {num_tasks:,} tasks")

        # Queue + workers
        queue = asyncio.Queue()
        results = [None] * num_tasks
        timed_out = []

        for i, task in enumerate(tasks):
            await queue.put((i, task))

        # Dispatch delay tracking for staggered initial burst
        self._dispatch_seq = 0
        self._dispatch_lock = asyncio.Lock()
        self._dispatch_start = time.perf_counter()

        num_workers = min(self.optimal_concurrency, num_tasks)
        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(self._worker(queue, results, timed_out, prepare_fn, parse_fn, fallback_fn))
            workers.append(w)

        if not self._quiet:
            print(f"\nWorkers: {num_workers}, Target concurrency: {self.optimal_concurrency}")

        # Monitoring loop
        self._start_time = start_time = time.time()
        self._last_conc_check = start_time
        last_report = start_time
        last_adjustment = start_time
        last_tick_successful = 0
        healthy_completions = 0
        healthy_elapsed = 0.0
        last_healthy_successful = 0
        last_healthy_time = start_time
        report_every_n = max(num_tasks // 20, 10)

        if not self._quiet:
            print(f"\n⏱ T+0.0s: Starting task processing")

        while not queue.empty():
            await asyncio.sleep(0.1)
            now = time.time()
            completed = self.stats['tasks_processed']
            completions_since = completed - (last_report - start_time) if False else (completed - last_tick_successful)  # approximate

            if now - last_report >= 2.0 or (completed - last_tick_successful) >= report_every_n:
                tick_duration = now - last_report
                current_successful = self.stats['tasks_successful']
                tick_successful = current_successful - last_tick_successful
                tick_rate = tick_successful / tick_duration if tick_duration > 0 else 0.0

                p50 = self.latency_tracker.get_p50() if self.latency_tracker.values else 0.0

                # Healthy throughput
                sm = self._concurrency_controller
                sm_state_val = sm.state.value if sm and hasattr(sm, 'state') else None
                in_healthy = sm_state_val in ("RAMP-UP", "STEADY") or sm_state_val is None
                if in_healthy and (current_successful - last_healthy_successful) > 0:
                    healthy_completions += current_successful - last_healthy_successful
                    healthy_elapsed += now - last_healthy_time
                last_healthy_successful = current_successful
                last_healthy_time = now
                throughput = healthy_completions / healthy_elapsed if healthy_elapsed > 0 else 0

                active = self.semaphore.active
                concurrency = self.semaphore.limit
                drain = active / throughput if throughput > 0 else 0

                # Get TPM/RPM utilization (always, for tick + PID)
                current_tpm = await self.tpm_tracker.get_current_tpm() if self.tpm_tracker else 0
                effective_tpm = self.rate_limits.tokens_per_minute * self._headroom
                current_rpm = await self.rpm_tracker.get_current_rpm() if self.rpm_tracker else 0
                effective_rpm = self.rate_limits.requests_per_minute * self._headroom

                # Unified tick: determines binding constraint, activates right controller
                line = self._tick(completed, num_tasks, tick_rate, p50, throughput,
                                  active, concurrency, current_tpm, effective_tpm,
                                  current_rpm, effective_rpm, num_tasks)
                if not self._quiet:
                    print(line)

                last_report = now
                last_tick_successful = current_successful

            # Warm-up calibration (always recalibrate both tokens and concurrency)
            if (not self._warm_up_calibrated
                    and len(self.actual_total_tokens) >= self._warm_up_target_samples
                    and len(self.latency_tracker.values) >= self._warm_up_target_samples):
                self._calibrate_concurrency(num_tasks)

            # Token correction + PID (both always active)
            if now - last_adjustment >= 0:  # every tick
                self._adjust_throughput_if_needed()
                await self._apply_pid()
                last_adjustment = now

            # Spawn extra workers if concurrency increased
            if self.optimal_concurrency > num_workers:
                extra = self.optimal_concurrency - num_workers
                for _ in range(extra):
                    w = asyncio.create_task(self._worker(queue, results, timed_out, prepare_fn, parse_fn, fallback_fn))
                    workers.append(w)
                num_workers = self.optimal_concurrency

        # Drain
        t_loop = time.time()
        if not self._quiet:
            print(f"⏱ T+{t_loop - start_time:.1f}s: Monitoring loop exited, awaiting queue.join()...")
        await queue.join()
        t_join = time.time()
        if not self._quiet:
            print(f"⏱ T+{t_join - start_time:.1f}s: queue.join() done ({t_join - t_loop:.1f}s drain)")

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        if not self._quiet:
            print(f"⏱ T+{time.time() - start_time:.1f}s: Main batch done — {self.stats['tasks_successful']} succeeded, {self.stats['timeouts']} deferred")
            is_rate_capped = self._rate_limit_concurrency <= self._server_concurrency
            if is_rate_capped:
                print(f"  RATE-CAPPED at {self.optimal_concurrency} (rate_conc={self._rate_limit_concurrency}, server_conc={self._server_concurrency})")
            else:
                sm_state = self._concurrency_controller.state.value if self._concurrency_controller and hasattr(self._concurrency_controller, 'state') else "N/A"
                print(f"  Final concurrency: {self.optimal_concurrency} (state: {sm_state})")

        # === RETRY PASS ===
        failed_for_retry = []
        if timed_out:
            for idx, data in timed_out:
                failed_for_retry.append((idx, data, 'timeout'))
        if self.failed_task_ids:
            for i, task in enumerate(tasks):
                if str(task.get('respondent_id', '')) in self.failed_task_ids:
                    failed_for_retry.append((i, task, 'exception'))

        recovered = 0
        if failed_for_retry:
            if not self._quiet:
                print(f"\n[RETRY PASS] Retrying {len(failed_for_retry)} failed tasks with reduced concurrency...")
            self.failed_task_ids.clear()
            self.failure_log.clear()

            retry_workers_n = max(5, min(len(failed_for_retry), num_workers // 10))
            self.latency_tracker.retry_mode = 60.0 if self._has_server_headers else self.latency_tracker.timeout_floor

            retry_queue = asyncio.Queue()
            retry_timed_out = []
            for idx, data, _ in failed_for_retry:
                await retry_queue.put((idx, data))

            retry_tasks = []
            for _ in range(retry_workers_n):
                w = asyncio.create_task(self._worker(retry_queue, results, retry_timed_out, prepare_fn, parse_fn, fallback_fn))
                retry_tasks.append(w)

            await retry_queue.join()
            for _ in retry_tasks:
                await retry_queue.put(None)
            await asyncio.gather(*retry_tasks)
            self.latency_tracker.retry_mode = False

            # Count recoveries
            for idx, data, reason in failed_for_retry:
                if results[idx] is not None:
                    recovered += 1

            # Fallback for still-failed
            for idx, data in retry_timed_out:
                if results[idx] is None:
                    self.stats['tasks_failed'] += 1
                    if fallback_fn:
                        results[idx] = fallback_fn(data, 'timeout')
            for idx, data, reason in failed_for_retry:
                if results[idx] is None:
                    self.stats['tasks_failed'] += 1
                    if fallback_fn:
                        results[idx] = fallback_fn(data, reason)

            still_failed = sum(1 for r in results if r is None)
            if not self._quiet:
                print(f"[RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")

        # Summary
        wall_time = time.time() - start_time
        self.stats['recovered'] = recovered
        self.stats['wall_time'] = wall_time

        if not self._quiet:
            rate_capped_final = self._rate_limit_concurrency <= self._server_concurrency
            print(f"\nCompleted {num_tasks} tasks in {wall_time:.1f}s")
            print(f"- Successful: {self.stats['tasks_successful']}")
            if recovered > 0:
                print(f"- Recovered: {recovered} (retried successfully)")
            print(f"- Rate limits: {self.stats['rate_limits']}")
            print(f"- Timeouts: {self.stats['timeouts']}")
            if rate_capped_final:
                print(f"- Rate-capped at concurrency {self.optimal_concurrency}")
            else:
                print(f"- Final concurrency: {self.optimal_concurrency}")

        # Save stats
        perf_model.save()

        return results
