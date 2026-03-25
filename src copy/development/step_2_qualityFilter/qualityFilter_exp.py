import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from dataclasses import dataclass
import numpy as np

import re
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from aiolimiter import AsyncLimiter

# === MODELS ========================================================================================================
import models

# === CONFIG (from experimental config_exp.py) ========================================================================================================
try:
    from .config_exp import (
        OPENAI_API_KEY, DEFAULT_LANGUAGE,
        ModelConfig, QualityFilterConfig, DEFAULT_QUALITY_FILTER_CONFIG,
        ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
        API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME,
        # Experimental constants
        INPUT_HISTORY_MAXLEN, OUTPUT_HISTORY_MAXLEN, ERROR_WINDOW_SIZE,
        DEFAULT_TIMEOUT_SECONDS, DEFAULT_LATENCY_SECONDS,
        PROGRESS_REPORT_INTERVAL, DIAGNOSTIC_INTERVAL, MAX_TOKEN_ACQUIRE_ATTEMPTS,
        THROUGHPUT_ADJUSTMENT_THRESHOLD, THROUGHPUT_ADJUSTMENT_MIN_SAMPLES, ADJUSTMENT_INTERVAL,
        # Ramp-up and warm-up configs (from config_ideaExtractor)
        RampUpConfig, DEFAULT_RAMP_UP_CONFIG,
        CircuitBreakerConfig, DEFAULT_CIRCUIT_BREAKER_CONFIG,
        WarmUpConfig, DEFAULT_WARM_UP_CONFIG,
        # Optimal API request strategy constants
        get_model_tier_latency, get_output_ratio,
        COLD_START_CAP, WARM_UP_WINDOW_SECONDS, WARM_UP_MIN_COMPLETIONS,
        RAMP_INTERVAL_SECONDS, RAMP_INCREASE_FACTOR, RAMP_DECREASE_FACTOR,
        SIGNAL_GREEN_THRESHOLD, SIGNAL_YELLOW_THRESHOLD,
    )
    from .prompts_exp import GRADER_INSTRUCTIONS
except ImportError:
    from config_exp import (
        OPENAI_API_KEY, DEFAULT_LANGUAGE,
        ModelConfig, QualityFilterConfig, DEFAULT_QUALITY_FILTER_CONFIG,
        ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
        API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME,
        # Experimental constants
        INPUT_HISTORY_MAXLEN, OUTPUT_HISTORY_MAXLEN, ERROR_WINDOW_SIZE,
        DEFAULT_TIMEOUT_SECONDS, DEFAULT_LATENCY_SECONDS,
        PROGRESS_REPORT_INTERVAL, DIAGNOSTIC_INTERVAL, MAX_TOKEN_ACQUIRE_ATTEMPTS,
        THROUGHPUT_ADJUSTMENT_THRESHOLD, THROUGHPUT_ADJUSTMENT_MIN_SAMPLES, ADJUSTMENT_INTERVAL,
        # Ramp-up and warm-up configs (from config_ideaExtractor)
        RampUpConfig, DEFAULT_RAMP_UP_CONFIG,
        CircuitBreakerConfig, DEFAULT_CIRCUIT_BREAKER_CONFIG,
        WarmUpConfig, DEFAULT_WARM_UP_CONFIG,
        # Optimal API request strategy constants
        get_model_tier_latency, get_output_ratio,
        COLD_START_CAP, WARM_UP_WINDOW_SECONDS, WARM_UP_MIN_COMPLETIONS,
        RAMP_INTERVAL_SECONDS, RAMP_INCREASE_FACTOR, RAMP_DECREASE_FACTOR,
        SIGNAL_GREEN_THRESHOLD, SIGNAL_YELLOW_THRESHOLD,
    )
    from prompts_exp import GRADER_INSTRUCTIONS

from utils.llm import RateLimits, extract_rate_limits_from_response
from config import get_reasoning_params


# =============================================================================
# RAW RESPONSE PARSING (no instructor — nano trips on structured output schemas)
# =============================================================================
# Category → quality_filter_code mapping:
#   1, 2     → 99999997  (don't know / no answer)
#   3        → 99999998  (no text / empty)
#   4        → 99999999  (gibberish / nonsense)
#   no flag  → None      (keep for analysis)
CATEGORY_TO_CODE = {"1": 99999997, "2": 99999997, "3": 99999998, "4": 99999999}

def parse_quality_code(raw_text: str) -> Optional[int]:
    """Parse <category> tag from LLM scratchpad output into a quality filter code.

    Returns 99999997, 99999999, or None (= keep the response).
    Any unexpected output defaults to None (conservative: don't flag).
    """
    # Extract content from <category>...</category> tag
    match = re.search(r'<category>\s*(.*?)\s*</category>', raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        value = match.group(1).strip().lower()
        if value in CATEGORY_TO_CODE:
            return CATEGORY_TO_CODE[value]
        # "no flag" or any other text → keep
        return None
    # No tag found — fallback: scan for a standalone category number
    match = re.search(r'\b([1-4])\b', raw_text[-50:])  # check end of response
    if match:
        return CATEGORY_TO_CODE.get(match.group(1))
    # Unparseable → conservative default: keep
    return None

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cached_resources import get_openai_client, get_tiktoken_encoding

try:
    import nest_asyncio #for Spyder
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries during retries
# Our own error handling provides concise, actionable output instead
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.ERROR)

# Note: Constants (INPUT_HISTORY_MAXLEN, etc.) now imported from config_exp.py
# Note: MIN/MAX_CONCURRENCY and MIN/MAX_WORKERS come from ProcessingConfig

# === RATE LIMITING HELPER CLASSES  ========================================================================================================

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
        """Wait if necessary and acquire tokens with safeguard against infinite loops."""
        logger.debug(f"[TOKEN BUCKET] Requesting {tokens_needed} tokens")

        attempts = 0
        while attempts < MAX_TOKEN_ACQUIRE_ATTEMPTS:
            attempts += 1
            result = await self.acquire(tokens_needed)
            if result is True:
                logger.debug(f"[TOKEN BUCKET] Acquired {tokens_needed} tokens, {self.available:.0f} remaining")
                return
            else:
                # result is wait_seconds
                logger.debug(f"[TOKEN BUCKET] Insufficient tokens, waiting {result:.1f}s")
                await asyncio.sleep(result)

        raise RuntimeError(f"Failed to acquire {tokens_needed} tokens after {MAX_TOKEN_ACQUIRE_ATTEMPTS} attempts")
    
    async def reconcile(self, delta_tokens: int) -> None:
        """Reconcile actual token usage against estimate.

        Args:
            delta_tokens: Difference between actual and estimated tokens.
                         Negative = overestimated (return tokens to bucket).
                         Positive = underestimated (already consumed).
        """
        if delta_tokens < 0:
            async with self.lock:
                old_available = self.available
                self.available = min(self.tpm, self.available - delta_tokens)
                logger.debug(f"[TOKEN BUCKET] Reconciled {-delta_tokens} tokens back, {old_available:.0f} → {self.available:.0f}")
        else:
            logger.debug(f"[TOKEN BUCKET] No reconciliation needed for +{delta_tokens} tokens (underestimated)")


TIMEOUT_FLOOR_SECONDS = 60.0  # Per strategy doc: 60s floor, 180s cold-start/ceiling


class LatencyTracker:
    """EMA tracker for latencies with generous timeout strategy"""
    def __init__(self, processing_config: Optional[ProcessingConfig] = None,
                 timeout_floor: float = None, default_timeout: float = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.timeout_floor = timeout_floor if timeout_floor is not None else TIMEOUT_FLOOR_SECONDS
        self.default_timeout = default_timeout if default_timeout is not None else DEFAULT_TIMEOUT_SECONDS
        self.ema = None
        self.alpha = self.processing_config.latency_tracker_ema_alpha
        self.values = deque(maxlen=self.processing_config.latency_tracker_samples_window)
        self.retry_mode = False  # When True, use very generous timeout for retry pass

    def add(self, value):
        """Add a latency measurement"""
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

    def get_timeout(self, est_tokens=None):
        """Calculate timeout: generous safety net, not aggressive cutoff"""
        config = self.processing_config
        if self.retry_mode:
            return 180.0  # Very generous for retry pass
        if not self.values:
            return max(self.timeout_floor, self.default_timeout)  # Cold start

        # P95 × margin as safety net, bounded by floor and ceiling
        p95 = float(np.percentile(list(self.values), 95))
        return max(self.timeout_floor, min(config.adaptive_timeout_max_seconds, p95 * config.adaptive_timeout_margin))

    def get_avg_latency(self):
        """Get average latency for concurrency calculations"""
        if not self.values:
            return DEFAULT_LATENCY_SECONDS
        return self.ema if self.ema is not None else DEFAULT_LATENCY_SECONDS


# === CONCURRENCY GATE (replaces asyncio.Semaphore) ========================================================================================================

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


# === CONCURRENCY RAMP (LINEAR WITH CONGESTION DETECTION) ========================================================================================================

class ConcurrencyRamp:
    """Completion-based concurrency ramp with congestion detection.

    Concurrency scales linearly with completion progress:
      0% complete → start (50% of Little's Law)
      100% complete → target (90% of Little's Law)

    Checks for two stop signals:
      1. Throughput drop — completion rate declining vs previous window
      2. Queue backing up — timeout rate >5% in a window

    After warm-up calibration, Little's Law is recalculated and the ramp
    adjusts start/target but preserves congestion detection state.
    """
    def __init__(self, config: 'RampUpConfig', little_law_cap: int, num_tasks: int):
        self.config = config
        self._num_tasks = num_tasks
        self._done = False
        self._stopped_concurrency = None
        self._stop_reason = None

        # Compute start and target from Little's Law
        self._little_law_cap = little_law_cap
        start = max(config.min_initial, int(little_law_cap * config.start_fraction))
        target = min(int(little_law_cap * config.target_fraction), num_tasks)
        self._start = start
        self._target = max(target, start)  # target >= start
        self._current = start

        # Throughput tracking (rolling window)
        self._prev_throughput = None
        self._declining_steps = 0

        # Queue depth tracking
        self._prev_completions_total = 0
        self._prev_timeouts_total = 0

    @property
    def cap(self) -> int:
        return self._target

    def current_target(self) -> int:
        return self._current

    def is_done(self) -> bool:
        return self._done

    def stopped_concurrency(self) -> Optional[int]:
        return self._stopped_concurrency

    def recalibrate(self, new_little_law_cap: int):
        """Called after warm-up calibration with updated Little's Law.

        Updates start/target from new cap. Preserves congestion detection
        state (_prev_throughput, _declining_steps) so that ongoing throughput
        decline is not forgotten across recalibration.
        """
        self._little_law_cap = new_little_law_cap
        new_start = max(self.config.min_initial, int(new_little_law_cap * self.config.start_fraction))
        new_target = min(int(new_little_law_cap * self.config.target_fraction), self._num_tasks)
        self._start = new_start
        self._target = max(new_target, new_start)
        self._current = new_start
        self._done = False
        self._stopped_concurrency = None
        self._stop_reason = None
        # NOTE: _prev_throughput and _declining_steps intentionally preserved
        print(f"RAMP RECALIBRATED: {new_start} → {self._target} "
              f"(Little's Law: {new_little_law_cap})")

    def record_measurement(self, throughput: float, tpm_pct: float, rpm_pct: float,
                           completions_total: int, timeouts_total: int, duration: float):
        """Called every measurement window with current metrics."""
        if self._done:
            return

        # --- Stop signal 1: throughput dropping ---
        if self._prev_throughput is not None and self._prev_throughput > 0:
            growth = (throughput - self._prev_throughput) / self._prev_throughput
            if growth < -0.10:  # throughput dropped >10%
                self._declining_steps += 1
            else:
                self._declining_steps = max(0, self._declining_steps - 1)

            if self._declining_steps >= 2:  # 2 consecutive drops
                self._stop('throughput_drop', self._current)
                return

        # --- Stop signal 2: queue congestion (timeouts appearing) ---
        new_timeouts = timeouts_total - self._prev_timeouts_total
        new_completions = completions_total - self._prev_completions_total
        if new_timeouts > 0 and new_completions > 0:
            timeout_rate = new_timeouts / (new_completions + new_timeouts)
            if timeout_rate > 0.05:  # >5% of this window timed out
                self._stop('queue_congestion', self._current)
                return

        self._prev_throughput = throughput
        self._prev_completions_total = completions_total
        self._prev_timeouts_total = timeouts_total

        # --- Completion-based ramp: advance proportional to progress ---
        ramp_fraction = min(completions_total / self._num_tasks, 1.0)
        new_conc = int(self._start + (self._target - self._start) * ramp_fraction)
        new_conc = max(new_conc, self._current)  # never decrease during ramp

        if new_conc >= self._target:
            self._current = self._target
            self._done = True
            self._stopped_concurrency = self._target
            self._stop_reason = 'target_reached'
            print(f"RAMP COMPLETE: concurrency {self._target} "
                  f"({self.config.target_fraction*100:.0f}% of Little's Law {self._little_law_cap})")
        else:
            self._current = new_conc

    def _stop(self, reason: str, concurrency: int):
        """Stop ramping due to congestion signal."""
        self._done = True
        self._stopped_concurrency = concurrency
        self._stop_reason = reason
        label = 'THROUGHPUT DROP' if reason == 'throughput_drop' else 'QUEUE CONGESTION'
        print(f"RAMP STOPPED ({label}): locking concurrency at {concurrency} "
              f"(was ramping toward {self._target})")


# === CIRCUIT BREAKER ========================================================================================================

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
        """Called every tick. Returns 'tripped', 'recovering', 'recovered', or None."""
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


# === CONCURRENCY INITIALIZATION ========================================================================================================

@dataclass
class ApiLimits:
    """API limits structure for Little's Law calculations"""
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(limits: ApiLimits, latency_seconds: float, avg_tokens: float, processing_config: Optional[ProcessingConfig] = None, cap: Optional[int] = None, min_conc: Optional[int] = None, headroom: Optional[float] = None) -> int:
    """Compute optimal concurrency using Little's Law"""
    config = processing_config or DEFAULT_PROCESSING_CONFIG
    cap = cap if cap is not None else config.concurrency_cap_default
    min_conc = min_conc if min_conc is not None else config.concurrency_min_default
    headroom = headroom if headroom is not None else config.rate_limit_headroom

    latency_seconds = max(float(latency_seconds or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)

    rpm_throughput = limits.requests_per_minute * headroom / 60
    tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
    candidates = [rpm_throughput, tpm_throughput]
    allowed_rps = max(min(candidates), 0.0)
    target = allowed_rps * latency_seconds   # Little's Law

    return int(max(min(target, cap), min_conc))



# === MAIN GRADER CLASS ========================================================================================================

class Grader:
    def __init__(
        self,
        responses: List[models.PreprocessedModel],
        var_lab: str,
        config: Optional[QualityFilterConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        verbose: bool = False,
        prompt_printer = None):

        self.responses = responses
        self.question = var_lab
        self.config = config or DEFAULT_QUALITY_FILTER_CONFIG
        self.model_config = model_config or ModelConfig()  # kept for backward compat
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.config.model
        self.grader_instructions = GRADER_INSTRUCTIONS
        self._results: List[models.QualityFilteredModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer

        # Initialize tokenizer for token counting (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Raw async client (no instructor — nano produces correct results without schema injection)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        # Rate limiting setup - use fallback values for initial setup
        # Actual rate limits will be fetched from API during process_all_tasks_async
        self.rate_limits = RateLimits(
            tokens_per_minute=FALLBACK_TPM,
            requests_per_minute=FALLBACK_RPM,
            tokens_per_day=FALLBACK_TPM * 60 * 24
        )

        # Token bucket for TPM limiting (will be re-initialized with actual limits during bootstrap)
        self.tpm_bucket = TokenBucket(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom)
        
        # Adaptive token estimation (following user's strategy)
        self.input_token_history = deque(maxlen=INPUT_HISTORY_MAXLEN)
        self.output_token_history = deque(maxlen=OUTPUT_HISTORY_MAXLEN)
        self.estimation_errors = deque(maxlen=ERROR_WINDOW_SIZE)
        self.first_prompt_tokens = None  # Cache first prompt calculation
        
        # Rolling average of actual total tokens for comparison
        self.actual_total_tokens = deque(maxlen=ERROR_WINDOW_SIZE)
        
        # Latency tracking
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)
        
        # Calculate initial average tokens estimate for bootstrapping
        self.avg_tokens = self._calculate_avg_tokens()
        
        # Rate limiting components (will be initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None
        
        # Stats
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'retries': 0,
            'rate_limits': 0,
            'timeouts': 0
        }

        # Failure log: tracks each permanent failure with details
        self.failure_log = []  # List of {respondent_id, reason, error_type, response_preview}

        # Throughput adjustment state
        self.current_arrival_rate = None      # Set during initialization, updated on adjustment
        self.bootstrap_avg_tokens = None      # Preserved original estimate for diagnostics
        self.adjustment_stats = {
            'adjustments_made': 0,
            'last_avg_tokens': None,
        }

        # Warm-up ramp configuration (Option B)
        self.ramp_up_config = DEFAULT_RAMP_UP_CONFIG
        self.circuit_breaker_config = DEFAULT_CIRCUIT_BREAKER_CONFIG
        self.warm_up_config = DEFAULT_WARM_UP_CONFIG
        self.circuit_breaker = None
        self._concurrency_ramp = None
        self._ramp_complete = True  # No ramp until initialized
        self._warm_up_calibrated = False
        self._warm_up_target_samples = 15

    def _calculate_avg_tokens(self) -> int:
        """Calculate average token count for requests"""
        sample_size = min(10, len(self.responses))
        if sample_size == 0:
            return 200  # Default estimate
        
        total_tokens = 0
        for i in range(sample_size):
            prompt = self._build_individual_prompt(
                self.question,
                self.responses[i].respondent_id,
                self.responses[i].response
            )
            total_tokens += len(self.encoding.encode(prompt))
        
        avg_input = total_tokens / sample_size
        # Output ratio depends on model type (chat vs reasoning)
        output_ratio = get_output_ratio(self.model)
        return int(avg_input * output_ratio)

    def _build_individual_prompt(self, var_lab: str, response_id: str, response_text: str) -> str:
        """Build prompt for individual response assessment"""
        return self.grader_instructions.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            response_text=response_text
        )
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate total tokens using adaptive strategy"""
        actual_input_tokens = len(self.encoding.encode(prompt))
        
        # Input estimation: first prompt + 15%, then average of first 3
        if self.first_prompt_tokens is None:
            # First prompt: use actual + 15% margin
            self.first_prompt_tokens = actual_input_tokens
            estimated_input = int(actual_input_tokens * 1.15)
        elif len(self.input_token_history) < 3:
            # Still collecting data: use actual + 15%
            estimated_input = int(actual_input_tokens * 1.15)
        else:
            # Use average of first 3 actual inputs
            avg_input = sum(self.input_token_history) / len(self.input_token_history)
            estimated_input = int(avg_input)
        
        # Track input tokens for learning
        if len(self.input_token_history) < 3:
            self.input_token_history.append(actual_input_tokens)
        
        # Output estimation: 15% of input, then average of first 5 responses
        if len(self.output_token_history) < 5:
            # Use 15% of input as estimate
            estimated_output = int(estimated_input * 0.15)
        else:
            # Use average of first 5 actual outputs
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output)
        
        # Ensure we don't exceed max_tokens
        estimated_output = min(self.config.max_tokens, estimated_output)
        
        total_estimate = estimated_input + estimated_output
        
        return total_estimate
    
    def get_token_estimation_stats(self) -> dict:
        """Get token estimation accuracy statistics"""
        if not self.estimation_errors:
            return {"status": "collecting_data", "samples": 0}
        
        avg_error = sum(self.estimation_errors) / len(self.estimation_errors)
        avg_input = sum(self.input_token_history) / len(self.input_token_history) if self.input_token_history else 0
        avg_output = sum(self.output_token_history) / len(self.output_token_history) if self.output_token_history else 0
        avg_actual_total = sum(self.actual_total_tokens) / len(self.actual_total_tokens) if self.actual_total_tokens else 0
        
        return {
            "status": "learning",
            "samples": len(self.estimation_errors),
            "avg_estimation_error": avg_error,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "avg_actual_total_tokens": avg_actual_total,
            "initial_avg_tokens": self.bootstrap_avg_tokens if self.bootstrap_avg_tokens is not None else self.avg_tokens,
            "current_avg_tokens": self.avg_tokens,
            "adjustments_made": self.adjustment_stats['adjustments_made'],
            "input_samples": len(self.input_token_history),
            "output_samples": len(self.output_token_history),
            "actual_samples": len(self.actual_total_tokens)
        }
    
    def get_token_bucket_status(self) -> dict:
        """Get current token bucket status with corrected utilization calculation"""
        available_pct = (self.tpm_bucket.available / self.tpm_bucket.tpm) * 100
        
        # Calculate real utilization based on consumption rate vs capacity
        if len(self.actual_total_tokens) >= 10:
            # Use actual consumption rate over last 10 samples
            recent_avg = sum(list(self.actual_total_tokens)[-10:]) / 10
            # Convert to per-second rate (assuming ~2s per request for rough estimate)
            consumption_rate_per_sec = recent_avg / 2.0
            # Calculate utilization as percentage of per-second capacity (60k tokens/sec)
            real_utilization_pct = (consumption_rate_per_sec / (self.tpm_bucket.tpm / 60)) * 100
        else:
            # Fallback to bucket level method for early samples
            real_utilization_pct = 100 - available_pct
        
        return {
            "available_tokens": int(self.tpm_bucket.available),
            "capacity": self.tpm_bucket.tpm,
            "utilization_pct": real_utilization_pct,
            "low_tokens": available_pct < 10,
            "consumption_rate": consumption_rate_per_sec if len(self.actual_total_tokens) >= 10 else 0
        }

    def _adjust_throughput_if_needed(self) -> bool:
        """Threshold-based throughput adjustment.

        When actual token usage significantly exceeds bootstrap estimate,
        reinstall rate_limiter and token bucket with corrected values.
        Pattern ported from ideaExtractor_exp.py.

        Returns True if adjustment was made, False otherwise.
        """
        if len(self.actual_total_tokens) < THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
            return False

        actual_avg = sum(self.actual_total_tokens) / len(self.actual_total_tokens)
        bootstrap_avg = self.avg_tokens
        ratio = actual_avg / bootstrap_avg if bootstrap_avg > 0 else 1.0

        if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
            return False

        # Calculate new arrival rate using actual tokens
        rpm_throughput = self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        new_tpm_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / actual_avg / 60
        new_arrival_rate = min(rpm_throughput, new_tpm_throughput)

        # Reinstall rate limiter with adjusted rate
        self.rate_limiter = AsyncLimiter(1, time_period=1.0/new_arrival_rate)

        # Reinitialize token bucket with fresh state
        old_bucket_available = self.tpm_bucket.available
        self.tpm_bucket = TokenBucket(int(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom))

        # Update avg_tokens for future estimation and rate calculations
        old_avg = self.avg_tokens
        old_arrival_rate = self.current_arrival_rate or 0
        self.avg_tokens = int(actual_avg)
        self.current_arrival_rate = new_arrival_rate

        # Track adjustment
        self.adjustment_stats['last_avg_tokens'] = old_avg
        self.adjustment_stats['adjustments_made'] += 1

        print(f"\n>> THROUGHPUT ADJUSTMENT #{self.adjustment_stats['adjustments_made']}")
        print(f"   Actual tokens ({actual_avg:.0f}) exceeded estimate ({bootstrap_avg:.0f}) by {(ratio-1)*100:.0f}%")
        print(f"   Arrival rate: {old_arrival_rate:.2f}/s -> {new_arrival_rate:.2f}/s")
        print(f"   avg_tokens: {old_avg} -> {self.avg_tokens}")
        print(f"   Token bucket reset (was {old_bucket_available:,.0f} available)")

        return True

# --- OPTION B: WARM-UP + CONSERVATIVE RAMP METHODS ---

    def _initialize_rate_limiters(self, limits: RateLimits, num_tasks: int) -> int:
        """Initialize rate limiting per optimal API request strategy.

        Layer 1: RPM — AsyncLimiter (request spacing)
        Layer 2: TPM — TokenBucket (token budget guard)
        Layer 3: Concurrency — ConcurrencyGate (cold start from RPM, capped at 50)
        Layer 4: Circuit breaker — monitors timeout rate

        Returns target_semaphore (what we're ramping toward after warm-up).
        """
        headroom = self.processing_config.rate_limit_headroom

        # Store limits for utilization tracking
        self._rpm_limit = limits.requests_per_minute * headroom
        self._tpm_limit = limits.tokens_per_minute * headroom

        # Layer 1: RPM — rate limiter
        arrival_rate = min(
            self._rpm_limit / 60,
            self._tpm_limit / self.avg_tokens / 60
        )
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / arrival_rate)
        self.current_arrival_rate = arrival_rate

        # Layer 2: TPM — token bucket
        self.tpm_bucket = TokenBucket(int(self._tpm_limit))

        # Layer 3: Concurrency — cold start per strategy doc
        # Cold semaphore: min(RPM/60, 50), floored at min_initial
        cold_semaphore = min(int(limits.requests_per_minute / 60), COLD_START_CAP)
        cold_semaphore = max(cold_semaphore, self.ramp_up_config.min_initial)
        cold_semaphore = min(cold_semaphore, num_tasks)
        self._cold_start_semaphore = cold_semaphore

        self.semaphore = ConcurrencyGate(cold_semaphore)
        self.optimal_concurrency = cold_semaphore

        # Compute Little's Law target (for post-warm-up ramp target)
        latency_est = get_model_tier_latency(self.model)
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law = compute_optimal_concurrency(
            api_limits, latency_est, self.avg_tokens,
            processing_config=self.processing_config,
        )
        little_law_cap = min(little_law, num_tasks)
        self._little_law_target = little_law_cap

        # Target semaphore: 90% of Little's Law (set after warm-up recalibration)
        self._target_semaphore = min(int(little_law_cap * 0.90), num_tasks)

        # Keep ConcurrencyRamp for backward compat (bypassed by signal-based ramp)
        self._concurrency_ramp = ConcurrencyRamp(
            self.ramp_up_config, little_law_cap, num_tasks
        )
        self._ramp_complete = False
        self._signal_ramp_active = False  # Activates after warm-up

        # RPM/TPM utilization tracking (sliding window, 60s)
        self._rpm_window = deque()  # (timestamp,) entries
        self._tpm_window = deque()  # (timestamp, tokens) entries
        self._prev_queue_depth = 0  # For queue health trend
        self._prev_p95 = None       # For latency trend detection

        # Layer 4: Circuit breaker
        self.circuit_breaker = ConcurrencyCircuitBreaker(
            config=self.circuit_breaker_config,
            gate=self.semaphore,
            baseline=cold_semaphore
        )

        return little_law_cap

    def _calculate_warm_up_sample_size(self, num_tasks: int) -> int:
        """Adaptive sample size: more samples for larger datasets, capped."""
        if num_tasks <= 50:
            return self.warm_up_config.sample_min
        elif num_tasks >= 500:
            return self.warm_up_config.sample_max
        else:
            fraction = (num_tasks - 50) / (500 - 50)
            return int(self.warm_up_config.sample_min + fraction * (self.warm_up_config.sample_max - self.warm_up_config.sample_min))

    def _rpm_utilization(self) -> float:
        """Current RPM as fraction of limit (0.0 to 1.0+)."""
        now = time.time()
        cutoff = now - 60.0
        while self._rpm_window and self._rpm_window[0] < cutoff:
            self._rpm_window.popleft()
        current_rpm = len(self._rpm_window)
        return current_rpm / self._rpm_limit if self._rpm_limit > 0 else 0.0

    def _tpm_utilization(self) -> float:
        """Current TPM as fraction of limit (0.0 to 1.0+)."""
        now = time.time()
        cutoff = now - 60.0
        while self._tpm_window and self._tpm_window[0][0] < cutoff:
            self._tpm_window.popleft()
        current_tpm = sum(t[1] for t in self._tpm_window)
        return current_tpm / self._tpm_limit if self._tpm_limit > 0 else 0.0

    def _record_api_call(self, actual_tokens: int):
        """Record an API call for RPM/TPM utilization tracking."""
        now = time.time()
        self._rpm_window.append(now)
        self._tpm_window.append((now, actual_tokens))

    def _evaluate_signals(self, queue_depth: int) -> str:
        """Evaluate queue health + RPM% + TPM% signals.

        Returns: 'green' (all green), 'yellow' (any yellow), 'red' (any red).
        """
        # Queue health: compare with previous depth
        if queue_depth > self._prev_queue_depth + 5:
            queue_signal = 'red'
        elif queue_depth > self._prev_queue_depth:
            queue_signal = 'yellow'
        else:
            queue_signal = 'green'

        # RPM utilization
        rpm_pct = self._rpm_utilization()
        if rpm_pct > SIGNAL_YELLOW_THRESHOLD:
            rpm_signal = 'red'
        elif rpm_pct > SIGNAL_GREEN_THRESHOLD:
            rpm_signal = 'yellow'
        else:
            rpm_signal = 'green'

        # TPM utilization
        tpm_pct = self._tpm_utilization()
        if tpm_pct > SIGNAL_YELLOW_THRESHOLD:
            tpm_signal = 'red'
        elif tpm_pct > SIGNAL_GREEN_THRESHOLD:
            tpm_signal = 'yellow'
        else:
            tpm_signal = 'green'

        # Latency trend (P95 increasing → API under pressure)
        latency_signal = 'green'
        if len(self.latency_tracker.values) >= 5:
            current_p95 = float(np.percentile(list(self.latency_tracker.values), 95))
            if self._prev_p95 is not None:
                if current_p95 > self._prev_p95 * 1.25:  # P95 jumped >25%
                    latency_signal = 'red'
                elif current_p95 > self._prev_p95 * 1.10:  # P95 crept up >10%
                    latency_signal = 'yellow'
            self._prev_p95 = current_p95

        self._prev_queue_depth = queue_depth

        # Any red → red, any yellow → yellow, all green → green
        signals = [queue_signal, rpm_signal, tpm_signal, latency_signal]
        if 'red' in signals:
            return 'red'
        if 'yellow' in signals:
            return 'yellow'
        return 'green'

    def _apply_signal_ramp(self, queue: asyncio.Queue):
        """Signal-based ramp: +25% if all green, -20% if any red, hold if yellow."""
        if not self._signal_ramp_active:
            return

        queue_depth = queue.qsize()
        signal = self._evaluate_signals(queue_depth)
        current = self.semaphore.limit

        if signal == 'green' and current < self._target_semaphore:
            new = min(int(current * RAMP_INCREASE_FACTOR), self._target_semaphore)
            new = max(new, current + 1)  # Always advance by at least 1
            self.semaphore.set_limit(new)
            self.optimal_concurrency = new
        elif signal == 'red' and current > self.ramp_up_config.min_initial:
            new = max(int(current * RAMP_DECREASE_FACTOR), self.ramp_up_config.min_initial)
            self.semaphore.set_limit(new)
            self.optimal_concurrency = new
        # yellow: hold

    def _calibrate_from_warm_up(self, num_tasks: int) -> None:
        """Post-warm-up calibration: measure reality, jump, activate signal-based ramp.

        After 10s of real work, uses measured latency and token counts to:
        1. Recalculate Little's Law with real data
        2. Jump concurrency based on stress detection
        3. Activate signal-based ramp toward target
        """
        measured_avg_tokens = int(np.mean(list(self.actual_total_tokens)))
        measured_latency = float(np.median(list(self.latency_tracker.values)))

        old_avg = self.avg_tokens
        self.avg_tokens = measured_avg_tokens
        self.bootstrap_avg_tokens = measured_avg_tokens

        # Recalculate Little's Law with measured data
        api_limits = ApiLimits(
            self.rate_limits.tokens_per_minute,
            self.rate_limits.requests_per_minute
        )
        new_little_law = compute_optimal_concurrency(
            api_limits, measured_latency, measured_avg_tokens,
            processing_config=self.processing_config,
        )
        new_little_law_cap = min(new_little_law, num_tasks)
        self._little_law_target = new_little_law_cap
        self._target_semaphore = min(int(new_little_law_cap * 0.90), num_tasks)

        # Stress detection: any timeouts during warm-up?
        warm_up_timeouts = self.stats.get('timeouts', 0)
        latency_values = list(self.latency_tracker.values)
        latency_increasing = (len(latency_values) >= 4 and
            np.mean(latency_values[-2:]) > np.mean(latency_values[:2]) * 1.5)
        stress = warm_up_timeouts > 0 or latency_increasing

        # Post-warm-up jump: min(100, Little's Law) if no stress
        if not stress:
            post_warmup = min(100, new_little_law_cap)
        else:
            post_warmup = self._cold_start_semaphore  # Hold if stress detected

        post_warmup = min(post_warmup, num_tasks)
        self.semaphore.set_limit(post_warmup)
        self.optimal_concurrency = post_warmup
        if self.circuit_breaker:
            self.circuit_breaker.baseline = post_warmup

        # Activate signal-based ramp
        self._signal_ramp_active = True
        self._ramp_complete = False

        # Recalculate arrival rate (tokens changed, so TPM rail changes)
        headroom = self.processing_config.rate_limit_headroom
        new_arrival_rate = min(
            self.rate_limits.requests_per_minute * headroom / 60,
            self.rate_limits.tokens_per_minute * headroom / measured_avg_tokens / 60
        )
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / new_arrival_rate)
        self.current_arrival_rate = new_arrival_rate

        stress_label = " (STRESS DETECTED — holding)" if stress else ""
        print(f"\n{'='*60}")
        print(f"WARM-UP CALIBRATION (from {len(self.actual_total_tokens)} samples, {WARM_UP_WINDOW_SECONDS}s window){stress_label}")
        print(f"   Latency: {measured_latency:.2f}s (median measured)")
        print(f"   avg_tokens: {old_avg} (tiktoken) -> {measured_avg_tokens} (measured)")
        print(f"   Little's Law: {new_little_law_cap}")
        print(f"   Concurrency: {self._cold_start_semaphore} (cold) -> {post_warmup} (jump) -> {self._target_semaphore} (target)")
        print(f"   Arrival rate: {new_arrival_rate:.2f}/s")
        print(f"   Signal ramp: +{int((RAMP_INCREASE_FACTOR-1)*100)}% / {RAMP_INTERVAL_SECONDS}s toward {self._target_semaphore}")
        print(f"{'='*60}")

        self._warm_up_calibrated = True

    async def _check_ramp_up(self):
        """Completion-based ramp with congestion detection.

        Called every 0.1s. Every measurement_window (0.5s), feeds throughput and
        timeout count to ConcurrencyRamp. Concurrency advances with completion progress.
        """
        if self._ramp_complete:
            return

        now = time.monotonic()
        if not hasattr(self, '_ramp_last_check_time'):
            self._ramp_last_check_time = now
            self._ramp_last_completions = self.stats['tasks_successful']
            return

        elapsed = now - self._ramp_last_check_time
        if elapsed < self.ramp_up_config.measurement_window_seconds:
            return

        completions_this_window = self.stats['tasks_successful'] - self._ramp_last_completions
        if completions_this_window < self.ramp_up_config.min_completions_per_step:
            return  # extend window — not enough data yet

        throughput = completions_this_window / elapsed

        # Feed to ramp (no TPM/RPM % tracking for qualityFilter)
        self._concurrency_ramp.record_measurement(
            throughput, 0.0, 0.0,
            completions_total=self.stats['tasks_successful'],
            timeouts_total=self.stats['timeouts'],
            duration=elapsed,
        )

        # Reset window
        self._ramp_last_check_time = now
        self._ramp_last_completions = self.stats['tasks_successful']

        if self._concurrency_ramp.is_done():
            final = self._concurrency_ramp.stopped_concurrency()
            self.semaphore.set_limit(final)
            self.optimal_concurrency = final
            self._ramp_complete = True
            if self.circuit_breaker:
                self.circuit_breaker.baseline = final
        else:
            next_conc = self._concurrency_ramp.current_target()
            self.semaphore.set_limit(next_conc)
            self.optimal_concurrency = next_conc

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            asyncio.TimeoutError
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def process_task(self, task: Dict) -> models.QualityFilteredModel:
        """Process a single quality assessment task via raw API call + parse."""
        task_start = time.perf_counter()

        try:
            prompt = self._build_individual_prompt(
                self.question,
                task['task_id'],
                task['response_text']
            )
            est_tokens = self.estimate_tokens(prompt)

            if task.get('task_index', 0) < 5:
                logger.info(f"[ESTIMATION DEBUG] Task {task.get('task_index', 0)}: estimated {est_tokens} tokens")

            if self.prompt_printer and task.get('task_index', 0) == 0:
                self.prompt_printer.capture_prompt(
                    step_name="quality_filter",
                    utility_name="QualityFilter",
                    prompt_content=prompt,
                    prompt_type="quality_assessment",
                    metadata={
                        "model": self.model,
                        "var_lab": self.question,
                        "language": DEFAULT_LANGUAGE,
                        "estimated_tokens": est_tokens
                    }
                )

            async with self.semaphore:
                timeout = self.latency_tracker.get_timeout(est_tokens)
                await self.tpm_bucket.wait_and_acquire(est_tokens)
                async with self.rate_limiter:

                    # Raw API call — no instructor schema injection
                    api_params = {
                        "model": self.model,
                        "input": prompt,
                        "max_output_tokens": self.config.max_tokens,
                        **get_reasoning_params(self.model),
                    }
                    response = await asyncio.wait_for(
                        self.client.responses.create(**api_params),
                        timeout=timeout
                    )

                    latency = time.perf_counter() - task_start
                    self.latency_tracker.add(latency)

                    # Track token usage
                    usage = getattr(response, 'usage', None)
                    if usage:
                        input_tokens = getattr(usage, 'input_tokens', 0)
                        output_tokens = getattr(usage, 'output_tokens', 0)
                        actual_total_tokens = getattr(usage, 'total_tokens', 0) or (input_tokens + output_tokens)

                        if len(self.output_token_history) < 5:
                            self.output_token_history.append(output_tokens)
                        self.actual_total_tokens.append(actual_total_tokens)
                        self._record_api_call(actual_total_tokens)

                        estimation_error = abs(actual_total_tokens - est_tokens)
                        self.estimation_errors.append(estimation_error)
                        await self.tpm_bucket.reconcile(actual_total_tokens - est_tokens)

                    # Parse raw text response into quality code
                    raw_text = response.output_text if hasattr(response, 'output_text') else str(response)
                    quality_code = parse_quality_code(raw_text)

                    result = models.QualityFilteredModel(
                        respondent_id=task['task_id'],
                        response=task['response_text'],
                        quality_filter=quality_code is not None,
                        quality_filter_code=quality_code
                    )
                    self.stats['tasks_successful'] += 1
                    return result

        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.warning(f"Task {task['task_id']} timed out")
            raise

        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['task_id']} hit rate limit")
            raise

        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {type(e).__name__}: {e}")
            raise
    
    def create_fallback_response(self, task: Dict) -> models.QualityFilteredModel:
        """Create fallback response for failed tasks"""
        original = task['original_response']
        # Mark as meaningful (conservative) but with error code to track
        original.quality_filter = False
        original.quality_filter_code = -1  # Distinguishable error code (not 0=meaningful, not 97/99=filtered)
        return original

    def get_failure_report(self, total_responses: int = None) -> str:
        """Return a formatted report of all processing failures."""
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

    async def worker(self, queue: asyncio.Queue, results: List) -> None:
        """Worker coroutine that processes tasks from queue"""
        while True:
            task = None
            try:
                task = await queue.get()
                if task is None:  # Sentinel
                    break

                try:
                    result = await self.process_task(task)
                    results[task['result_index']] = result
                    if self.circuit_breaker:
                        self.circuit_breaker.record_completion()
                except Exception as e:
                    # Record timeout for circuit breaker
                    if isinstance(e, (asyncio.TimeoutError, APITimeoutError)):
                        if self.circuit_breaker:
                            self.circuit_breaker.record_timeout()
                    elif self.circuit_breaker:
                        self.circuit_breaker.record_completion()  # Non-timeout failures are not congestion

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
                        print(f"429 {limit_type} limit hit — task {task['task_id']} failed permanently")
                    else:
                        # Non-rate-limit errors: show type only (not full str(e) which can be huge)
                        logger.error(f"Task {task['task_id']} failed after retries: {error_type}")

                    self.stats['tasks_failed'] += 1
                    self.failure_log.append({
                        'respondent_id': task['task_id'],
                        'reason': 'exception',
                        'error_type': error_type,
                        'response_preview': task.get('response_text', '')[:80]
                    })
                    results[task['result_index']] = self.create_fallback_response(task)
                finally:
                    self.stats['tasks_processed'] += 1
                    queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker fatal error (will terminate): {e}", exc_info=True)
                self.stats['worker_failures'] = self.stats.get('worker_failures', 0) + 1
                break

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
        """Make a minimal API call to fetch rate limits from response headers."""
        from config import API_PROVIDER, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

        if API_PROVIDER == "azure":
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/",
                default_query={"api-version": "2024-10-21"},
            )
            model = AZURE_OPENAI_DEPLOYMENT_NAME
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self.model

        # Make minimal API call with raw response to get headers
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

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[Optional[models.QualityFilteredModel]]:
        """Process all tasks using Option B: warm-up with conservative ramp."""
        if not tasks:
            return []

        self.verbose_reporter.step_start("Quality Assessment")

        # Phase 1: Fetch rate limits from API
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

        # Phase 2: Initialize rate limiters per optimal API request strategy
        self.bootstrap_avg_tokens = self.avg_tokens  # Preserve tiktoken estimate for diagnostics
        self._initialize_rate_limiters(limits, len(tasks))
        self._warm_up_calibrated = False

        # Workers: enough to cover target (spawned upfront, gated by semaphore)
        num_workers = min(self._target_semaphore, len(tasks))
        num_workers = max(num_workers, 5)  # Floor of 5 workers

        headroom = self.processing_config.rate_limit_headroom
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / self.avg_tokens / 60
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"
        latency_est = get_model_tier_latency(self.model)

        print("[RATE LIMITING SETUP] - Optimal API Request Strategy")
        print(f"- Model: {self.model}")
        print(f"- RPM limit: {limits.requests_per_minute:,} ({self._rpm_limit:,.0f} with headroom)")
        print(f"- TPM limit: {limits.tokens_per_minute:,} ({self._tpm_limit:,.0f} with headroom)")
        print(f"- Tiktoken avg_tokens: {self.avg_tokens} (output ratio: {get_output_ratio(self.model):.2f}x)")
        print(f"- Expected throughput: {min(rpm_throughput, tpm_throughput):.1f}/s ({bottleneck} limited)")
        print(f"- Latency estimate: {latency_est}s ({self.model})")
        print(f"- Little's Law (theoretical): {self._little_law_target}")
        print(f"- Cold start: {self._cold_start_semaphore} (min(RPM/60, {COLD_START_CAP}))")
        print(f"- Target: {self._target_semaphore} (90% of Little's Law, capped at {len(tasks)} tasks)")
        print(f"- Warm-up: {WARM_UP_WINDOW_SECONDS}s window, then signal-based ramp (+{int((RAMP_INCREASE_FACTOR-1)*100)}%/{RAMP_INTERVAL_SECONDS}s)")
        print(f"- Processing {len(tasks):,} tasks")
        print(f"- Workers: {num_workers}")

        # Create queue and results list
        queue = asyncio.Queue()
        results = [None] * len(tasks)

        for i, task in enumerate(tasks):
            task['result_index'] = i
            task['task_index'] = i
            await queue.put(task)

        # Start workers
        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(self.worker(queue, results))
            workers.append(w)

        # Phase 4: Main processing loop
        start_time = time.time()
        last_report = start_time
        last_adjustment = start_time
        last_ramp_check = start_time
        prev_completed = 0  # For suppressing output when idle

        while not queue.empty():
            await asyncio.sleep(0.1)
            now = time.time()
            elapsed = now - start_time

            # Circuit breaker (every tick)
            if self.circuit_breaker:
                action = self.circuit_breaker.check_and_adjust()
                if action in ('tripped', 'recovering', 'recovered'):
                    self.optimal_concurrency = self.semaphore.limit

            # Warm-up calibration (time-based: after WARM_UP_WINDOW_SECONDS + min completions)
            if (not self._warm_up_calibrated
                    and elapsed >= WARM_UP_WINDOW_SECONDS
                    and len(self.actual_total_tokens) >= WARM_UP_MIN_COMPLETIONS
                    and len(self.latency_tracker.values) >= WARM_UP_MIN_COMPLETIONS):
                self._calibrate_from_warm_up(len(tasks))
                # Spawn extra workers if target increased
                if self._target_semaphore > num_workers:
                    extra = self._target_semaphore - num_workers
                    for _ in range(extra):
                        w = asyncio.create_task(self.worker(queue, results))
                        workers.append(w)
                    num_workers = self._target_semaphore
                    print(f"Workers: {num_workers} (+{extra} after calibration)")

            # Signal-based ramp check (every RAMP_INTERVAL_SECONDS)
            if self._signal_ramp_active and now - last_ramp_check >= RAMP_INTERVAL_SECONDS:
                self._apply_signal_ramp(queue)
                last_ramp_check = now

            # Throughput adjustment check
            if now - last_adjustment >= ADJUSTMENT_INTERVAL:
                self._adjust_throughput_if_needed()
                last_adjustment = now

            # Progress report (every PROGRESS_REPORT_INTERVAL seconds)
            if now - last_report >= PROGRESS_REPORT_INTERVAL:
                completed = self.stats['tasks_processed']
                # Suppress output when no new completions (consolidation tail)
                if completed > prev_completed:
                    rate = completed / elapsed if elapsed > 0 else 0
                    rpm_pct = self._rpm_utilization() * 100
                    tpm_pct = self._tpm_utilization() * 100
                    conc = self.semaphore.active
                    conc_limit = self.optimal_concurrency
                    q_depth = queue.qsize()
                    p50 = float(np.median(list(self.latency_tracker.values))) if self.latency_tracker.values else 0
                    p95 = float(np.percentile(list(self.latency_tracker.values), 95)) if len(self.latency_tracker.values) >= 2 else 0
                    target_info = f"→{self._target_semaphore}" if self._signal_ramp_active else ""
                    print(f"[STEP2] {completed}/{len(tasks)} ({rate:.1f}/s) | "
                          f"TPM:{tpm_pct:.0f}% RPM:{rpm_pct:.0f}% "
                          f"Conc:{conc}/{conc_limit}{target_info} Queue:{q_depth} | "
                          f"P50:{p50:.1f}s P95:{p95:.1f}s")
                    prev_completed = completed
                last_report = now

        # Wait for all tasks to complete
        await queue.join()

        # Stop workers
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        # Phase 5: Stats after main pass
        elapsed = time.time() - start_time
        print(f"\nMain batch: {len(tasks)} tasks in {elapsed:.1f}s")
        print(f"- Successful: {self.stats['tasks_successful']}")
        print(f"- Failed: {self.stats['tasks_failed']}")
        print(f"- Rate limits: {self.stats['rate_limits']}")
        print(f"- Timeouts: {self.stats['timeouts']}")
        print(f"- Average: {elapsed/len(tasks):.2f}s/task")
        if self.adjustment_stats['adjustments_made'] > 0:
            print(f"- Throughput adjustments: {self.adjustment_stats['adjustments_made']}")

        # Phase 6: Retry pass for true failures
        if self.failure_log:
            failed_tasks = []
            failed_ids = {str(f['respondent_id']) for f in self.failure_log}
            for task in tasks:
                if str(task['task_id']) in failed_ids:
                    failed_tasks.append(task)

            if failed_tasks:
                print(f"\n[RETRY PASS] Retrying {len(failed_tasks)} failed tasks with reduced concurrency...")

                pre_retry_failure_log = list(self.failure_log)
                self.failure_log.clear()

                # Reduced concurrency: 10% of workers, min 5
                retry_workers = max(5, min(len(failed_tasks), num_workers // 10))

                # Generous timeout for retry
                self.latency_tracker.retry_mode = True

                retry_queue = asyncio.Queue()
                retry_results = [None] * len(failed_tasks)

                for i, task in enumerate(failed_tasks):
                    task['result_index'] = i
                    await retry_queue.put(task)

                retry_worker_tasks = []
                for _ in range(retry_workers):
                    w = asyncio.create_task(self.worker(retry_queue, retry_results))
                    retry_worker_tasks.append(w)

                await retry_queue.join()
                for _ in retry_worker_tasks:
                    await retry_queue.put(None)
                await asyncio.gather(*retry_worker_tasks)

                self.latency_tracker.retry_mode = False

                # Merge retry results back into main results
                recovered = 0
                for i, task in enumerate(failed_tasks):
                    retry_result = retry_results[i]
                    if retry_result and getattr(retry_result, 'quality_filter_code', -1) != -1:
                        # Find original index and overwrite fallback
                        original_idx = next(
                            j for j, t in enumerate(tasks) if str(t['task_id']) == str(task['task_id'])
                        )
                        results[original_idx] = retry_result
                        recovered += 1

                still_failed = len(self.failure_log)
                print(f"[RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")
                if still_failed > 0:
                    failed_ids_list = sorted(str(f['respondent_id']) for f in self.failure_log)
                    print(f"[RETRY PASS] Permanently failed IDs: {failed_ids_list[:20]}{'...' if still_failed > 20 else ''}")

        # Failure report (if any remain after retry)
        if self.failure_log:
            print(f"\n{'='*70}")
            print(self.get_failure_report(total_responses=len(tasks)))
            print(f"{'='*70}")

        return results

    def _prepare_individual_tasks(self) -> List[Dict]:
        """Prepare individual tasks for processing"""
        items_to_process = [r for r in self.responses if r.quality_filter_code is None]
        
        tasks = []
        for i, response in enumerate(items_to_process):
            # Handle various data types safely (including numpy types)
            response_text = response.response
            if isinstance(response_text, (float, int, np.floating, np.integer)):
                # Check for NaN/Inf using proper numeric functions
                try:
                    if math.isnan(float(response_text)) or math.isinf(float(response_text)):
                        response_text = ''
                    else:
                        response_text = str(response_text)
                except (ValueError, TypeError):
                    response_text = str(response_text)
            elif response_text is None:
                response_text = ''
            
            tasks.append({
                'task_id': response.respondent_id,
                'response_text': response_text,
                'original_response': response
            })
        
        return tasks

    def grade(self) -> List[models.QualityFilteredModel]:
        """Main entry point for quality filtering"""
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)

        # Pre-filter empty/None responses (code 99999998 = no response)
        # Dev pipeline step 1 doesn't always mark these, so we catch them here
        empty_values = {'none', 'nan', '<na>', 'na', ''}
        pre_filter_count = 0
        for r in self.responses:
            if r.quality_filter_code is None:
                response_text = str(r.response).strip() if r.response else ""
                if not response_text or response_text.lower() in empty_values:
                    r.quality_filter_code = 99999998
                    r.quality_filter = True
                    pre_filter_count += 1
        if pre_filter_count > 0:
            print(f"Pre-filtered {pre_filter_count} empty/None responses (code 99999998)")

        # Separate items that need processing from pre-filtered
        items_to_process = [r for r in self.responses if r.quality_filter_code is None]
        pre_filtered_items = [r for r in self.responses if r.quality_filter_code is not None]
        
        self.verbose_reporter.step_start("Quality Assessment")
        # Use fallback rate limits for initial display (actual limits fetched during processing)
        self.verbose_reporter.stat_line(f"Model: {self.model} (Initial limits: {self.rate_limits.requests_per_minute} RPM, {self.rate_limits.tokens_per_minute:,} TPM)")
        self.verbose_reporter.stat_line(f"Items needing LLM evaluation: {len(items_to_process)}")
        self.verbose_reporter.stat_line(f"Pre-filtered items: {len(pre_filtered_items)}")
        
        # Process items that need LLM evaluation
        if items_to_process:
            # Prepare tasks
            tasks = self._prepare_individual_tasks()
            
            # Process with queue + workers
            if nest_asyncio:
                nest_asyncio.apply()
            llm_results = asyncio.run(self.process_all_tasks_async(tasks))
            
            # Store results
            self._results = llm_results
        else:
            self.verbose_reporter.stat_line("No items require LLM evaluation")
            self._results = []

        # Create mapping for efficient lookup (normalize to str for type-safe lookup)
        llm_results_map = {str(result.respondent_id): result for result in self._results if result}

        # Merge results in original order
        merged_results = []
        for original_item in self.responses:
            if original_item.quality_filter_code is not None:
                # Keep pre-filtered item
                merged_results.append(original_item)
            else:
                # Use LLM result if available (normalize to str for lookup)
                if str(original_item.respondent_id) in llm_results_map:
                    merged_results.append(llm_results_map[str(original_item.respondent_id)])
                else:
                    # Fallback if not processed
                    original_item.quality_filter = False
                    original_item.quality_filter_code = 0
                    merged_results.append(original_item)
        
        # Update results
        self._results = merged_results

        # Calculate statistics
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        filtered_examples = []
        
        for result in self._results:
            if hasattr(result, 'quality_score'):
                if result.quality_score >= self.config.high_quality_threshold:
                    quality_counts["high"] += 1
                elif result.quality_score >= self.config.medium_quality_threshold:
                    quality_counts["medium"] += 1
                else:
                    quality_counts["low"] += 1
            
            # Collect meaningful filter examples
            if (result.quality_filter and 
                len(filtered_examples) < self.config.max_filter_examples and
                result.quality_filter_code is not None and
                (result.quality_filter_code % 100 == 97 or result.quality_filter_code % 100 == 99)):
                code_label = "don't know" if result.quality_filter_code == 99999997 else "gibberish/off-topic"
                filtered_examples.append(f'"{result.response}" ({code_label})')
        
        self._stats.output_count = len([r for r in self._results if not r.quality_filter])
        self._stats.end_timing()
        
        # Report statistics — clear separation of total vs LLM-processed
        total = len(self._results)
        llm_processed = len(items_to_process)
        pre_filtered_count = len(pre_filtered_items)
        llm_filtered = sum(1 for r in self._results
                           if r.quality_filter and r.quality_filter_code in (99999997, 99999999))
        llm_dont_know = sum(1 for r in self._results if r.quality_filter_code == 99999997)
        llm_gibberish = sum(1 for r in self._results if r.quality_filter_code == 99999999)
        meaningful = sum(1 for r in self._results if not r.quality_filter)

        print(f"\n{'─'*60}")
        print(f"SUMMARY ({total} total responses)")
        print(f"{'─'*60}")
        print(f"  Pre-filtered (empty/NA, code 99999998): {pre_filtered_count:>5}")
        print(f"  LLM evaluated:                          {llm_processed:>5}")
        print(f"    → Don't know (99999997):              {llm_dont_know:>5}")
        print(f"    → Gibberish  (99999999):              {llm_gibberish:>5}")
        print(f"    → Meaningful (null):                  {meaningful:>5}")
        print(f"{'─'*60}")
        print(f"  Total filtered out:                     {pre_filtered_count + llm_filtered:>5}  ({(pre_filtered_count + llm_filtered)/total*100:.1f}%)")
        print(f"  Total meaningful (passed):              {meaningful:>5}  ({meaningful/total*100:.1f}%)")
        print(f"{'─'*60}")

        # Show filtered examples (LLM-filtered only, not pre-filtered)
        if filtered_examples:
            self.verbose_reporter.sample_list("Sample LLM-filtered responses", filtered_examples)

        self.verbose_reporter.step_complete("Quality filtering completed")

        # Report any processing failures prominently
        if self.failure_log:
            print(f"\n{'='*70}")
            print("WARNING: NOT 100% SUCCESSFUL")
            print(self.get_failure_report(total_responses=llm_processed))
            print(f"{'='*70}")
        else:
            print(f"\nAll {llm_processed} LLM-evaluated responses processed successfully (0 errors)")

        return self._results

    def filter(self) -> List[models.QualityFilteredModel]:
        """Return only meaningful responses"""
        return [r for r in self._results if not r.quality_filter]

    def summary(self) -> Dict[str, Union[int, float]]:
        """Get summary statistics"""
        total = len(self._results)
        meaningless = sum(1 for r in self._results if r.quality_filter)
        meaningful = total - meaningless
        
        # Count by processing type
        llm_processed = sum(1 for r in self._results if hasattr(r, 'quality_score'))
        pre_filtered = total - llm_processed

        return {
            "total_responses": total,
            "meaningful_responses": meaningful,
            "meaningless_responses": meaningless,
            "meaningful_percentage": round((meaningful / total) * 100, 2) if total > 0 else 0,
            "llm_processed": llm_processed,
            "pre_filtered": pre_filtered
        }