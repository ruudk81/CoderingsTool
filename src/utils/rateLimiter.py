"""
Centralized rate limiting and concurrency control for LLM API calls.

Two systems, selected at startup based on header availability:
  System A (server-side data): OpenAI provides openai-processing-ms and
    remaining-requests/tokens headers. Concurrency controlled by residual
    latency drift. Rate pacing via passive TokenBucket + AsyncLimiter rails.
  System B (client-side data): No server-side headers (Azure, proxies).
    Concurrency controlled by P50-drift state machine. Rate pacing via
    PID-adjusted TokenBucket + AsyncLimiter.

Both systems always manage BOTH rate pacing AND concurrency. Neither is
"rate-limited" vs "throughput-bound" — both controls are always active.
Whichever constraint binds first gates naturally.

Usage:
    from utils.rateLimiter import (
        TokenBucket, ConcurrencyGate, LatencyTracker, TiktokenOffsetLearner,
        SimplifiedCircuitBreaker, ResidualLatencyTracker,
        HeaderAwareConcurrencyController, ConcurrencyState,
        PIDThroughputController, RealTimeTPMTracker, RealTimeRPMTracker,
        ApiLimits, compute_optimal_concurrency,
    )
"""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# SHARED INFRASTRUCTURE (both systems)
# =============================================================================

class TokenBucket:
    """Token bucket for TPM limiting. Regenerates tokens continuously."""

    def __init__(self, tokens_per_minute: int, max_acquire_attempts: int = 1000):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
        self._max_attempts = max_acquire_attempts

    async def acquire(self, tokens_needed):
        """Acquire tokens, returning True or wait_seconds."""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now

            if self.available >= tokens_needed:
                self.available -= tokens_needed
                return True
            else:
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
                return wait_seconds

    async def wait_and_acquire(self, tokens_needed):
        """Wait if necessary and acquire tokens."""
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            else:
                await asyncio.sleep(result)
        raise RuntimeError(f"Failed to acquire {tokens_needed} tokens after {self._max_attempts} attempts")

    async def reconcile(self, delta_tokens: int) -> None:
        """Reconcile actual token usage against estimate."""
        if delta_tokens < 0:
            async with self.lock:
                self.available = min(self.tpm, self.available - delta_tokens)


class ConcurrencyGate:
    """Concurrency limiter with dynamic limit adjustment.

    When limit decreases: in-flight requests drain naturally (no cancellation).
    When limit increases: blocked requests wake immediately.
    asyncio is single-threaded (cooperative), so no locks needed.
    """

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
        """Change limit. Wakes blocked waiters if room opened."""
        self._limit = max(1, new_limit)
        self._wake_waiters()

    def _wake_waiters(self):
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
        """Release a concurrency slot and wake next waiter."""
        self._active -= 1
        self._wake_waiters()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        self.release()


class LatencyTracker:
    """EMA tracker for latencies with adaptive timeout.

    Timeout strategy: max(floor, min(P50 × 6, 180)).
    Retry mode: configurable floor for retry passes.
    """

    def __init__(self, ema_alpha: float = 0.2, samples_window: int = 200,
                 timeout_floor: float = 10.0, default_timeout: float = 10.0,
                 default_latency: float = 2.0):
        self.timeout_floor = timeout_floor
        self.default_timeout = default_timeout
        self.default_latency = default_latency
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

    def get_timeout(self, est_tokens=None):
        """Safety-net timeout: max(floor, min(P50 × 6, 180))."""
        if self.retry_mode:
            if not self.values:
                return 180.0
            p50 = float(np.percentile(list(self.values), 50))
            retry_floor = self.retry_mode if isinstance(self.retry_mode, (int, float)) else 60.0
            return max(retry_floor, min(p50 * 6.0, 180.0))

        if not self.values:
            return max(self.timeout_floor, self.default_timeout)

        p50 = float(np.percentile(list(self.values), 50))
        return max(self.timeout_floor, min(p50 * 6.0, 180.0))

    def get_avg_latency(self):
        if not self.values:
            return self.default_latency
        return self.ema if self.ema is not None else self.default_latency

    def get_p50(self) -> float:
        if len(self.values) >= 2:
            return float(np.percentile(list(self.values), 50))
        return self.get_avg_latency()

    def get_p95(self) -> float:
        if len(self.values) >= 2:
            return float(np.percentile(list(self.values), 95))
        return self.get_avg_latency()

    def get_p99(self) -> float:
        if len(self.values) >= 2:
            return float(np.percentile(list(self.values), 99))
        return self.get_avg_latency()


class TiktokenOffsetLearner:
    """Learns the offset between tiktoken counts and actual API token counts."""

    def __init__(self, default_offset: int = 300, history_maxlen: int = 30,
                 min_samples: int = 5):
        self.default_offset = default_offset
        self.offsets = deque(maxlen=history_maxlen)
        self._learned_offset = None
        self._min_samples = min_samples

    def record(self, tiktoken_count: int, api_count: int):
        offset = api_count - tiktoken_count
        self.offsets.append(offset)
        if len(self.offsets) >= self._min_samples:
            self._learned_offset = int(sum(self.offsets) / len(self.offsets))

    def get_offset(self) -> int:
        if self._learned_offset is not None:
            return self._learned_offset
        return self.default_offset

    def is_learned(self) -> bool:
        return len(self.offsets) >= self._min_samples

    def get_stats(self) -> dict:
        return {
            "samples": len(self.offsets),
            "learned_offset": self._learned_offset,
            "using_offset": self.get_offset(),
            "is_learned": self.is_learned(),
            "min_offset": min(self.offsets) if self.offsets else None,
            "max_offset": max(self.offsets) if self.offsets else None,
        }


class SimplifiedCircuitBreaker:
    """Timeout rate monitoring. Trips when timeout rate exceeds threshold.

    Defense-in-depth: catches pathological scenarios that the concurrency
    controller's primary signal can't detect.
    """

    def __init__(self, window: int = 100, trip_threshold: float = 0.05,
                 cooldown_s: float = 10.0):
        self._window = window
        self._trip_threshold = trip_threshold
        self._cooldown_s = cooldown_s
        self._events: deque = deque(maxlen=window)
        self._state = 'CLOSED'
        self._last_trip_time: Optional[float] = None
        self._trip_count: int = 0

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

    def check(self) -> Optional[str]:
        """Called every tick. Returns 'tripped' or None."""
        now = time.monotonic()

        if self._state == 'CLOSED':
            total = len(self._events)
            if total < 10:
                return None
            timeouts = sum(1 for e in self._events if e == 'timeout')
            rate = timeouts / total
            if rate > self._trip_threshold:
                self._state = 'OPEN'
                self._last_trip_time = now
                self._trip_count += 1
                print(f"CIRCUIT BREAKER TRIPPED: timeout rate {rate:.1%} ({total} events)")
                return 'tripped'
            return None

        elif self._state == 'OPEN':
            elapsed = now - self._last_trip_time if self._last_trip_time else 0
            if elapsed >= self._cooldown_s:
                self._state = 'CLOSED'
            return None

        return None


# =============================================================================
# LITTLE'S LAW
# =============================================================================

@dataclass
class ApiLimits:
    """API limits for Little's Law calculations."""
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(limits: ApiLimits, latency_seconds: float,
                                avg_tokens: float, headroom: float = 0.9) -> int:
    """Little's Law: N = lambda × W. Returns raw concurrency (floor 2)."""
    latency_seconds = max(float(latency_seconds or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)

    rpm_throughput = limits.requests_per_minute * headroom / 60
    tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
    allowed_rps = max(min(rpm_throughput, tpm_throughput), 0.0)
    target = allowed_rps * latency_seconds

    return max(math.ceil(target), 2)


# =============================================================================
# SYSTEM A: Server-Side Data (OpenAI with headers)
# =============================================================================

class ConcurrencyState(Enum):
    RAMP_UP = "RAMP-UP"
    STEADY = "STEADY"
    BACKOFF = "BACKOFF"
    RECOVER = "RECOVER"


class ResidualLatencyTracker:
    """Tracks residual latency (observed - server processing) per API call.

    Residual = observed_latency_ms - openai-processing-ms.
    An inference signal: lower residual = less overhead outside server processing.
    Rising residual can indicate server queuing or network degradation.
    """

    def __init__(self, window: int = 200, median_window: int = 20,
                 trend_recent: int = 10, trend_previous: int = 10):
        self._window = window
        self._median_window = median_window
        self._trend_recent = trend_recent
        self._trend_previous = trend_previous
        self._entries: deque = deque(maxlen=window)

    def add(self, observed_latency_s: float, processing_ms: float):
        observed_ms = observed_latency_s * 1000.0
        residual_ms = max(0.0, observed_ms - processing_ms)
        self._entries.append((time.monotonic(), residual_ms, processing_ms))

    def _recent_n(self, n: int) -> list:
        entries = list(self._entries)
        return [(r, p) for _, r, p in entries[-n:]]

    def _median(self, values: list) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        mid = len(s) // 2
        if len(s) % 2 == 0 and len(s) >= 2:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    def median_residual(self) -> float:
        recent = self._recent_n(self._median_window)
        return self._median([r for r, _ in recent])

    def median_processing(self) -> float:
        recent = self._recent_n(self._median_window)
        return self._median([p for _, p in recent])

    def normalized_residual(self) -> float:
        """Residual / processing ratio."""
        med_proc = self.median_processing()
        if med_proc <= 0:
            return 0.0
        return self.median_residual() / med_proc

    def trend(self) -> float:
        """Ratio of recent median vs previous median. >1.0 = growing."""
        total_needed = self._trend_recent + self._trend_previous
        if len(self._entries) < total_needed:
            return 1.0
        entries = list(self._entries)
        recent = [r for _, r, _ in entries[-self._trend_recent:]]
        previous = [r for _, r, _ in entries[-(self._trend_recent + self._trend_previous):-self._trend_recent]]
        med_recent = self._median(recent)
        med_previous = self._median(previous)
        if med_previous <= 0:
            return 1.0
        return med_recent / med_previous

    @property
    def sample_count(self) -> int:
        return len(self._entries)


class HeaderAwareConcurrencyController:
    """Concurrency controller driven by residual latency drift.

    Three states + one event:
      States: RAMP-UP, STEADY, RECOVER
      Event: BACKOFF (cut concurrency, then → RECOVER)

    Drift = current_median_residual / baseline_residual.
    Baseline learned from first evaluate() after warm-up.

      < 1.1 → healthy (RAMP-UP)
      1.2-1.5 → slowing (STEADY / RECOVER hold)
      > 1.5 → BACKOFF event: cut to 0.9 × last_healthy, then RECOVER

    BACKOFF is a single-tick event. Consecutive ticks above 1.5 are counted
    cross-state; 4th consecutive tick triggers a re-cut.
    """

    def __init__(self, starting: int, ramp_step_pct: float = 0.025,
                 backoff_pct: float = 0.90, min_concurrency: int = 2,
                 drift_steady: float = 1.2, drift_backoff: float = 1.5,
                 drift_resume: float = 1.1, budget_pressure_threshold: float = 0.9):
        self.current = starting
        self.starting = starting
        self.ramp_step = max(2, int(starting * ramp_step_pct))
        self._backoff_pct = backoff_pct
        self._min_concurrency = min_concurrency
        self._drift_steady = drift_steady
        self._drift_backoff = drift_backoff
        self._drift_resume = drift_resume
        self._budget_pressure_threshold = budget_pressure_threshold

        self.state = ConcurrencyState.RAMP_UP
        self.last_healthy_concurrency = starting
        self.steady_concurrency = None

        self.last_healthy_throughput = 0.0
        self.last_healthy_p50 = 0.0

        self.residual_baseline = 0.0
        self.residual_drift = 0.0
        self.backoff_ticks = 0
        self.signal_cutoff = 0.0

    def _backoff_cut(self, from_concurrency: int) -> int:
        return max(self._min_concurrency,
                   int(from_concurrency * self._backoff_pct))

    def evaluate(self, median_residual_ms: float, normalized_residual: float,
                 residual_trend: float, header_pressure: float,
                 now: float, throughput: float = 0.0, p50: float = 0.0) -> int:
        """Main tick evaluation. Returns new concurrency."""
        if throughput > 0 and p50 > 0:
            self.last_healthy_throughput = throughput
            self.last_healthy_p50 = p50

        if self.residual_baseline == 0 and median_residual_ms > 0:
            self.residual_baseline = median_residual_ms

        if self.residual_baseline > 0:
            self.residual_drift = median_residual_ms / self.residual_baseline
        else:
            self.residual_drift = 1.0

        drift = self.residual_drift
        is_stressed = drift > self._drift_backoff or header_pressure > self._budget_pressure_threshold

        # Cross-state consecutive stress counter
        if is_stressed:
            self.backoff_ticks += 1
        else:
            self.backoff_ticks = 0

        if self.state == ConcurrencyState.RAMP_UP:
            if is_stressed:
                self.current = self._backoff_cut(self.last_healthy_concurrency)
                self.signal_cutoff = time.perf_counter()
                self.state = ConcurrencyState.RECOVER
            elif drift > self._drift_steady:
                self.state = ConcurrencyState.STEADY
                self.steady_concurrency = self.current
                self.last_healthy_concurrency = self.current
            else:
                self.last_healthy_concurrency = self.current
                self.current = self.current + self.ramp_step

        elif self.state == ConcurrencyState.STEADY:
            self.steady_concurrency = self.current
            if is_stressed:
                self.current = self._backoff_cut(self.last_healthy_concurrency)
                self.signal_cutoff = time.perf_counter()
                self.state = ConcurrencyState.RECOVER
            elif drift < self._drift_resume:
                self.state = ConcurrencyState.RAMP_UP
                self.current = self.current + self.ramp_step

        elif self.state == ConcurrencyState.RECOVER:
            if is_stressed and self.backoff_ticks >= 4:
                self.current = self._backoff_cut(self.current)
                self.backoff_ticks = 0
                self.signal_cutoff = time.perf_counter()
            elif drift < self._drift_resume:
                self.state = ConcurrencyState.STEADY
                self.steady_concurrency = self.current
                self.last_healthy_concurrency = self.current

        return self.current


# =============================================================================
# SYSTEM B: Client-Side Data (Azure fallback, no headers)
# =============================================================================

class PIDThroughputController:
    """PID controller for arrival rate adjustment.

    Asymmetric gains: aggressive when under-utilizing, gentle when over-utilizing.
    Adjusts the arrival rate based on real-time TPM utilization.
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


class RealTimeTPMTracker:
    """Tracks actual TPM usage in a sliding window for PID feedback."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.samples = deque()
        self.lock = asyncio.Lock()

    async def record(self, tokens: int):
        async with self.lock:
            now = time.monotonic()
            self.samples.append((now, tokens))
            self._prune(now)

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    async def get_current_tpm(self) -> float:
        async with self.lock:
            now = time.monotonic()
            self._prune(now)
            if not self.samples:
                return 0.0
            total_tokens = sum(t for _, t in self.samples)
            elapsed = max(now - self.samples[0][0], 1.0)
            return (total_tokens / elapsed) * 60


class RealTimeRPMTracker:
    """Tracks actual RPM in a sliding window."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.samples = deque()
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
