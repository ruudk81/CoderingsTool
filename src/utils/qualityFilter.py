import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import math
import time
import logging
import itertools
from typing import Dict, List, Optional, Union
from collections import deque
from dataclasses import dataclass
import numpy as np

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter

# === MODELS ========================================================================================================
import models

# === CONFIG — generic/universal ========================================================================================================
from config import (
    OPENAI_API_KEY, DEFAULT_LANGUAGE,
    ModelConfig, ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
    FALLBACK_TPM, FALLBACK_RPM,
)

# === CONFIG — step-specific ========================================================================================================
from config_steps.config_qualityFilter import (
    QualityFilterConfig, DEFAULT_QUALITY_FILTER_CONFIG,
    INPUT_HISTORY_MAXLEN, OUTPUT_HISTORY_MAXLEN, ERROR_WINDOW_SIZE,
    DEFAULT_TIMEOUT_SECONDS, DEFAULT_LATENCY_SECONDS,
    PROGRESS_REPORT_INTERVAL, DIAGNOSTIC_INTERVAL, MAX_TOKEN_ACQUIRE_ATTEMPTS,
    THROUGHPUT_ADJUSTMENT_THRESHOLD, THROUGHPUT_ADJUSTMENT_MIN_SAMPLES, ADJUSTMENT_INTERVAL,
)

from utils.llm import create_client, llm_create_async, ProbeResponse, RateLimits, extract_rate_limits_from_response

# === PROMPTS + RESPONSE MODELS ========================================================================================================
from prompts import GRADER_INSTRUCTIONS, QualityFilterLLMResponse

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding

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

# Note: Constants (INPUT_HISTORY_MAXLEN, etc.) now imported from config_steps
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


class LatencyTracker:
    """Simple EMA tracker for latencies"""
    def __init__(self, processing_config: Optional[ProcessingConfig] = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.ema = None
        self.alpha = self.processing_config.latency_tracker_ema_alpha
        self.values = deque(maxlen=self.processing_config.latency_tracker_samples_window)
    
    def add(self, value):
        """Add a latency measurement"""
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
    
    def get_timeout(self, est_tokens):
        """Calculate timeout based on EMA and token count with configurable bounds"""
        config = self.processing_config
        if not self.values:
            return max(config.adaptive_timeout_min_seconds, DEFAULT_TIMEOUT_SECONDS)

        # Use P95 latency as base
        p95 = np.percentile(list(self.values), 95)
        # Simple linear scaling with token count
        # Assume ~100ms per 1000 tokens as baseline
        token_factor = est_tokens / 1000
        timeout = p95 + (token_factor * 0.1)
        # Apply margin and configurable bounds
        return max(config.adaptive_timeout_min_seconds, min(config.adaptive_timeout_max_seconds, timeout * config.adaptive_timeout_margin))
    
    def get_avg_latency(self):
        """Get average latency for concurrency calculations"""
        if not self.values:
            return DEFAULT_LATENCY_SECONDS
        return self.ema if self.ema is not None else DEFAULT_LATENCY_SECONDS


# === BOOTSTRAP MEASUREMENT SYSTEM ========================================================================================================

@dataclass
class ApiLimits:
    """API limits structure for bootstrap calculations"""
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


async def bootstrap_measure_async(call_fn, n_probes: int = 3):
    """Run n_probes serial calls and return (avg_latency_s, avg_tokens). call_fn() -> usage dict."""
    latencies, tokens = [], []
    for _ in range(n_probes):
        t0 = time.perf_counter()
        usage = await call_fn()  # Let tenacity handle timeouts and retries
        t1 = time.perf_counter()
        latencies.append(max(t1 - t0, 0.001))
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        tokens.append(max(pt + ct, 1))
    return sum(latencies)/len(latencies), sum(tokens)/len(tokens)


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
        self.model_config = model_config or ModelConfig()
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.model_config.get_model_for_stage('quality_filter')
        self.grader_instructions = GRADER_INSTRUCTIONS
        self._results: List[models.QualityFilteredModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer

        # Initialize tokenizer for token counting (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Instructor-patched async client for structured output (supports OpenAI and Azure)
        self.client = create_client(self.model, async_mode=True)

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

        # Failure log: tracks each permanent failure with details
        self.failure_log = []  # List of {respondent_id, reason, error_type, response_preview}

        # Throughput adjustment state
        self.current_arrival_rate = None      # Set after bootstrap, updated on adjustment
        self.bootstrap_avg_tokens = None      # Preserved original bootstrap value for diagnostics
        self.adjustment_stats = {
            'adjustments_made': 0,
            'last_avg_tokens': None,
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
        # Assume 15% output ratio initially
        return int(avg_input * 1.15)

    def _build_individual_prompt(self, var_lab: str, response_id: str, response_text: str) -> str:
        """Build prompt for individual response assessment"""
        responses_text = f"respondent_id: {response_id}, response: \"{response_text}\""
        return self.grader_instructions.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses=responses_text
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

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            InstructorRetryException,
            asyncio.TimeoutError
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def process_task(self, task: Dict) -> models.QualityFilteredModel:
        """Process a single quality assessment task"""
        task_start = time.perf_counter()
        
        try:
            # Build prompt
            prompt = self._build_individual_prompt(
                self.question,
                task['task_id'],
                task['response_text']
            )
            
            # Estimate tokens
            est_tokens = self.estimate_tokens(prompt)
            
            # Log estimation for first few tasks
            if task.get('task_index', 0) < 5:
                logger.info(f"[ESTIMATION DEBUG] Task {task.get('task_index', 0)}: estimated {est_tokens} tokens")
            
            # Capture prompt for first task
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
           
            # Calculate dynamic timeout BEFORE entering rate-limited section
            timeout = self.latency_tracker.get_timeout(est_tokens)

            # Semaphore FIRST to prevent convoy effect, then token bucket, then rate limiter
            async with self.semaphore:
                await self.tpm_bucket.wait_and_acquire(est_tokens)
                async with self.rate_limiter:
                    
                    # Make API call
                    response = await asyncio.wait_for(
                        llm_create_async(
                            client=self.client,
                            model=self.model,
                            response_model=List[QualityFilterLLMResponse],
                            prompt=prompt,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens
                        ),
                        timeout=timeout
                    )
                    
                    # Record latency
                    latency = time.perf_counter() - task_start
                    self.latency_tracker.add(latency)
                    
                    # Track actual token usage for learning and reconciliation
                    usage = getattr(response, '_raw_response', None)
                    if usage:
                        usage = getattr(usage, 'usage', None)
                    if not usage:
                        usage = getattr(response, 'usage', None)

                    if usage:
                        # Handle both Responses API (input_tokens/output_tokens) and Chat API (prompt_tokens/completion_tokens)
                        input_tokens = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0)
                        output_tokens = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0)
                        actual_total_tokens = getattr(usage, 'total_tokens', 0) or (input_tokens + output_tokens)
                        actual_output_tokens = output_tokens

                        # Update output token history for estimation learning
                        if len(self.output_token_history) < 5:
                            self.output_token_history.append(actual_output_tokens)

                        # Track actual total tokens for rolling average
                        self.actual_total_tokens.append(actual_total_tokens)

                        # Track estimation accuracy
                        estimation_error = abs(actual_total_tokens - est_tokens)
                        self.estimation_errors.append(estimation_error)

                        # Reconcile token difference with bucket
                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)
                    
                    # Extract result and convert strict LLM response to pipeline model
                    if response and len(response) > 0:
                        llm_result = response[0]
                        # AUDIT: Log if LLM returned different ID (drift detection)
                        if str(llm_result.respondent_id) != str(task['task_id']):
                            logger.warning(
                                f"ID drift detected: LLM returned '{llm_result.respondent_id}' "
                                f"but input was '{task['task_id']}'"
                            )

                        # OVERRIDE: Always use original values, only take classification from LLM
                        result = models.QualityFilteredModel(
                            respondent_id=task['task_id'],           # FROM ORIGINAL
                            response=task['response_text'],          # FROM ORIGINAL
                            quality_filter=llm_result.quality_filter,
                            quality_filter_code=llm_result.quality_filter_code
                        )
                        self.stats['tasks_successful'] += 1
                        return result
                    else:
                        return self.create_fallback_response(task)
                    
        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.warning(f"Task {task['task_id']} timed out")
            raise  # Let tenacity retry
            
        except InstructorRetryException as e:
            # Concise output for 429 errors wrapped in InstructorRetryException
            error_str = str(e)
            if "429" in error_str or "RateLimitReached" in error_str:
                self.stats['rate_limits'] += 1
                if "token rate limit" in error_str.lower():
                    limit_type = "TPM"
                elif "call rate limit" in error_str.lower():
                    limit_type = "RPM"
                else:
                    limit_type = "rate"
                print(f"429 {limit_type} limit hit (task {task['task_id']})")
            else:
                logger.error(f"Task {task['task_id']} failed: {type(e).__name__}")
            raise  # Let tenacity retry

        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['task_id']} hit rate limit")
            raise  # Let tenacity retry

        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {type(e).__name__}: {e}")
            raise  # Let tenacity retry

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

    async def probe_call_no_structured(self, task_dict):
        """Probe call with minimal response model for bootstrap measurement"""
        prompt = self._build_individual_prompt(
            task_dict.get('var_lab', self.question),
            task_dict['task_id'],
            task_dict['response_text']
        )

        # Use minimal ProbeResponse model for Azure compatibility (instructor requires response_model)
        resp = await llm_create_async(
            client=self.client,
            model=self.model,
            prompt=prompt,
            response_model=ProbeResponse,
            temperature=self.config.temperature,
            track_usage=False,  # Manual tracking for probes
        )

        # Extract usage from instructor's _raw_response
        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        # Handle both Responses API (input_tokens) and Chat API (prompt_tokens)
        input_tokens = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0)
        output_tokens = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0)
        return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}

    async def worker(self, queue: asyncio.Queue, results: List) -> None:
        """Worker coroutine that processes tasks from queue"""
        task = None
        while True:
            try:
                task = await queue.get()
                if task is None:  # Sentinel
                    break

                try:
                    result = await self.process_task(task)
                    results[task['result_index']] = result
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
                        print(f"429 {limit_type} limit hit — task {task['task_id']} failed permanently")
                    else:
                        # Non-rate-limit errors: show type only (not full str(e) which can be huge)
                        logger.error(f"Task {task['task_id']} failed after retries: {error_type}")

                    self.stats['tasks_failed'] += 1
                    results[task['result_index']] = self.create_fallback_response(task)
                    self.failure_log.append({
                        'respondent_id': task['task_id'],
                        'reason': 'exception',
                        'error_type': error_type,
                        'response_preview': task.get('response_text', '')[:80]
                    })
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
        from openai import AsyncOpenAI
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
        response = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )

        return extract_rate_limits_from_response(response)

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[Optional[models.QualityFilteredModel]]:
        """Process all tasks using queue + workers pattern with bootstrap measurement."""
        if not tasks:
            return []

        self.verbose_reporter.step_start("Quality Assessment")

        # Fetch rate limits dynamically from API response headers
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Fetching rate limits from API...")

        limits = await self._fetch_rate_limits_from_api()

        # Fallback if headers not available
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

        # Store rate limits on self for use in diagnostics/reporting
        self.rate_limits = limits

        # Bootstrap measurement with probe calls
        sample_tasks = tasks[:min(3, len(tasks))]
        if len(sample_tasks) < 3:
            sample_tasks = sample_tasks * 3
            sample_tasks = sample_tasks[:3]

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Running bootstrap measurement (3 probe calls)...")

        start_time = time.time()
        task_cycle = itertools.cycle(sample_tasks)

        async def probe_with_different_tasks():
            return await self.probe_call_no_structured(next(task_cycle))

        avg_latency_s, avg_tokens = await bootstrap_measure_async(probe_with_different_tasks, n_probes=3)

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Probe time: {time.time() - start_time:.3f}s")
            self.verbose_reporter.stat_line(f"Bootstrap results: {avg_latency_s:.3f}s avg latency, {avg_tokens:.0f} avg tokens")

        # Initialize latency tracker with bootstrap measurements
        for i in range(3):
            self.latency_tracker.add(avg_latency_s)

        # Update avg_tokens with bootstrap measurement
        self.avg_tokens = int(avg_tokens)
        self.bootstrap_avg_tokens = self.avg_tokens

        # Calculate optimal concurrency using Little's Law
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law_concurrency = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, processing_config=self.processing_config, cap=self.processing_config.concurrency_cap_permissive, min_conc=self.processing_config.concurrency_min_permissive)

        # Adaptive minimum: don't force high minimum when RPM-limited
        # (e.g., when Little's Law says 1-5, forcing 100 causes 429s)
        # Pattern from ideaExtractor_exp.py: 3x calculated optimal for burst headroom, floor of 5
        max_concurrency = self.processing_config.concurrency_cap_default
        adaptive_min = min(
            self.processing_config.concurrency_min_default,
            max(little_law_concurrency * 3, 5)
        )
        optimal = min(max_concurrency, max(little_law_concurrency, adaptive_min))

        # Initialize rate limiting components
        arrival_rate = min(
            limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
            limits.tokens_per_minute * self.processing_config.rate_limit_headroom / avg_tokens / 60
        )

        self.rate_limiter = AsyncLimiter(1, time_period=1.0/arrival_rate)

        self.semaphore = asyncio.Semaphore(min(len(tasks), optimal))
        self.optimal_concurrency = min(len(tasks), optimal)

        # Re-initialize TokenBucket with actual rate limits
        self.tpm_bucket = TokenBucket(limits.tokens_per_minute * self.processing_config.rate_limit_headroom)
        self.current_arrival_rate = arrival_rate

        print("[RATE LIMITING SETUP]")
        print(f"- Model: {self.model}")
        print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- Bootstrap measured avg_tokens: {self.avg_tokens}")

        # Show expected throughput breakdown
        rpm_throughput = limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        tpm_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"
        print(f"- Expected throughput: {min(rpm_throughput, tpm_throughput):.1f}/s ({bottleneck} limited)")
        print(f"- Optimal by Little's law: {little_law_concurrency}")
        print(f"- Constrained optimum: {optimal} (adaptive_min={adaptive_min}, max={max_concurrency})")

        print(f"- Processing {len(tasks):,} tasks")

        # Calculate number of workers using ProcessingConfig bounds
        expected_throughput = min(rpm_throughput, tpm_throughput)
        # Adaptive min workers: 2x optimal concurrency as floor (at least 10)
        max_workers = self.processing_config.max_workers if hasattr(self.processing_config, 'max_workers') else 200
        adaptive_min_workers = max(10, optimal * 2)
        num_workers = min(max_workers, max(adaptive_min_workers, int(expected_throughput * avg_latency_s * 2.0)))
       
        print(f"- Workers launched: (concurrent subroutines): {num_workers}")
        print(f"- API calls in flight (concurrency ceiling/semaphore): {self.optimal_concurrency}")
        
        # Create queue and results list
        queue = asyncio.Queue()
        results = [None] * len(tasks)
        
        # Add tasks to queue with result indices
        for i, task in enumerate(tasks):
            task['result_index'] = i
            task['task_index'] = i
            await queue.put(task)
        
        # Start workers
        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(self.worker(queue, results))
            workers.append(w)
        
        # Progress monitoring with diagnostics
        start_time = time.time()
        last_report = start_time
        last_diagnostics = start_time
        last_adjustment = start_time

        while not queue.empty():
            await asyncio.sleep(1)
            now = time.time()

            # Throughput adjustment check
            if now - last_adjustment >= ADJUSTMENT_INTERVAL:
                self._adjust_throughput_if_needed()
                last_adjustment = now
            
            # Regular progress report
            if now - last_report >= PROGRESS_REPORT_INTERVAL:
                completed = self.stats['tasks_processed']
                remaining = queue.qsize()
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%), "
                      f"Rate: {rate:.1f}/s, Queue: {remaining}")
                last_report = now
            
            # Diagnostic report (if verbose)
            if self.verbose_reporter.enabled and now - last_diagnostics >= DIAGNOSTIC_INTERVAL:
                bucket_status = self.get_token_bucket_status()
                token_stats = self.get_token_estimation_stats()
                
                # Token bucket diagnostics
                if bucket_status['low_tokens']:
                    self.verbose_reporter.stat_line(f"⚠️ Token bucket low: {bucket_status['available_tokens']:,} tokens ({bucket_status['utilization_pct']:.1f}% utilized)")
                
                # Token estimation diagnostics
                if token_stats['status'] == 'learning' and token_stats['samples'] >= 5:
                    self.verbose_reporter.stat_line(f"Token estimation: {token_stats['avg_estimation_error']:.0f} avg error, "
                                                  f"Input: {token_stats['avg_input_tokens']:.0f} avg ({token_stats['input_samples']}/3), "
                                                  f"Output: {token_stats['avg_output_tokens']:.0f} avg ({token_stats['output_samples']}/5)")
                    
                    # Show comparison of initial vs learned average tokens
                    if token_stats['actual_samples'] >= 10:
                        actual_avg = token_stats['avg_actual_total_tokens']
                        initial_avg = token_stats['initial_avg_tokens']
                        current_avg = token_stats['current_avg_tokens']
                        difference = actual_avg - current_avg

                        learned_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(actual_avg, 1) / 60
                        current_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(current_avg, 1) / 60

                        pct_change = (difference / current_avg * 100) if current_avg > 0 else 0
                        threshold_pct = int((THROUGHPUT_ADJUSTMENT_THRESHOLD - 1) * 100)
                        threshold_note = f"below {threshold_pct}% threshold" if abs(pct_change) <= threshold_pct else f"exceeds {threshold_pct}% threshold"
                        if token_stats['adjustments_made'] > 0:
                            self.verbose_reporter.stat_line(f"Token usage: Bootstrap {initial_avg:.0f}, Adjusted {current_avg:.0f}, Actual {actual_avg:.0f} "
                                                          f"({difference:+.0f} from current, {pct_change:+.1f}%) — {threshold_note}")
                            self.verbose_reporter.stat_line(f"Throughput: pacing at {current_throughput:.1f}/s (adjusted from {self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(initial_avg, 1) / 60:.1f}/s bootstrap)")
                        else:
                            self.verbose_reporter.stat_line(f"Token usage: Initial estimate {initial_avg:.0f}, Learned average {actual_avg:.0f} "
                                                          f"({difference:+.0f} tokens, {pct_change:+.1f}%) — {threshold_note}")
                            self.verbose_reporter.stat_line(f"Throughput: pacing at {current_throughput:.1f}/s (bootstrap), optimal {learned_throughput:.1f}/s (learned)")
                
                last_diagnostics = now
        
        # Wait for all tasks to complete
        await queue.join()
        
        # Stop workers
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        
        # Final stats with diagnostics
        elapsed = time.time() - start_time
        print(f"\nCompleted {len(tasks)} tasks in {elapsed:.1f}s")
        print(f"- Successful: {self.stats['tasks_successful']}")
        print(f"- Failed: {self.stats['tasks_failed']}")
        print(f"- Rate limits: {self.stats['rate_limits']}")
        print(f"- Timeouts: {self.stats['timeouts']}")
        print(f"- Average: {elapsed/len(tasks):.2f}s/task")
        if self.adjustment_stats['adjustments_made'] > 0:
            print(f"- Throughput adjustments: {self.adjustment_stats['adjustments_made']}")
            print(f"  - Bootstrap avg_tokens: {self.bootstrap_avg_tokens}, Final avg_tokens: {self.avg_tokens}")

        # Failure report
        if self.failure_log:
            print(f"\n{'='*70}")
            print(self.get_failure_report(total_responses=len(tasks)))
            print(f"{'='*70}")

        # Final diagnostic summary (if verbose)
        if self.verbose_reporter.enabled:
            token_stats = self.get_token_estimation_stats()

            if token_stats['status'] == 'learning':
                accuracy = max(0, 100 - (token_stats['avg_estimation_error'] / max(1, token_stats['avg_input_tokens'] + token_stats['avg_output_tokens']) * 100))
                self.verbose_reporter.stat_line(f"Token estimation accuracy: {accuracy:.1f}% (avg error: {token_stats['avg_estimation_error']:.0f} tokens)")
                self.verbose_reporter.stat_line(f"Learned averages - Input: {token_stats['avg_input_tokens']:.0f}, Output: {token_stats['avg_output_tokens']:.0f}")
                
                # Final comparison of initial vs learned token usage
                if token_stats['actual_samples'] >= 10:
                    actual_avg = token_stats['avg_actual_total_tokens']
                    initial_avg = token_stats['initial_avg_tokens']
                    current_avg = token_stats['current_avg_tokens']
                    difference = actual_avg - initial_avg

                    optimal_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(actual_avg, 1) / 60
                    initial_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(initial_avg, 1) / 60

                    pct_change = (difference / initial_avg * 100) if initial_avg > 0 else 0
                    threshold_pct = int((THROUGHPUT_ADJUSTMENT_THRESHOLD - 1) * 100)
                    residual = actual_avg - current_avg
                    residual_pct = (residual / current_avg * 100) if current_avg > 0 else 0
                    residual_note = f"below {threshold_pct}% threshold" if abs(residual_pct) <= threshold_pct else f"exceeds {threshold_pct}% threshold"
                    self.verbose_reporter.stat_line(f"Token usage summary: Bootstrap {initial_avg:.0f} -> Actual {actual_avg:.0f} "
                                                  f"({difference:+.0f} tokens, {pct_change:+.1f}%)")
                    if token_stats['adjustments_made'] > 0:
                        self.verbose_reporter.stat_line(f"Adjustments applied: {token_stats['adjustments_made']} (final avg_tokens: {current_avg}, residual drift {residual_pct:+.1f}% — {residual_note})")
                    self.verbose_reporter.stat_line(f"Throughput analysis: Bootstrap {initial_throughput:.1f}/s -> Optimal {optimal_throughput:.1f}/s with perfect estimation")
        
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
                filtered_examples.append(f'"{result.response}" (quality filter: meaningless)')
        
        self._stats.output_count = len([r for r in self._results if not r.quality_filter])
        self._stats.end_timing()
        
        # Report statistics
        total = len(self._results)
        filtered_count = sum(1 for r in self._results if r.quality_filter)
        llm_processed = len(items_to_process)
        
        self.verbose_reporter.stat_line(f"Total responses: {total}")
        self.verbose_reporter.stat_line(f"LLM processed: {llm_processed}")
        self.verbose_reporter.stat_line(f"Pre-filtered: {len(pre_filtered_items)}")
        
        if quality_counts["high"] > 0:
            self.verbose_reporter.stat_line(f"High quality: {quality_counts['high']} responses ({quality_counts['high']/llm_processed*100:.1f}% of LLM processed)" if llm_processed > 0 else "High quality: 0 responses")
        if quality_counts["medium"] > 0:
            self.verbose_reporter.stat_line(f"Medium quality: {quality_counts['medium']} responses ({quality_counts['medium']/llm_processed*100:.1f}% of LLM processed)" if llm_processed > 0 else "Medium quality: 0 responses")
        if quality_counts["low"] > 0:
            self.verbose_reporter.stat_line(f"Low quality: {quality_counts['low']} responses ({quality_counts['low']/llm_processed*100:.1f}% of LLM processed)" if llm_processed > 0 else "Low quality: 0 responses")
        
        self.verbose_reporter.stat_line(f"Total filtered out: {filtered_count} responses ({filtered_count/total*100:.1f}%)")
        
        # Show filtered examples
        if filtered_examples:
            self.verbose_reporter.sample_list("Sample filtered responses", filtered_examples)
        
        self.verbose_reporter.step_complete("Quality filtering completed")

        # Report any processing failures prominently
        if self.failure_log:
            print(f"\n{'='*70}")
            print("WARNING: NOT 100% SUCCESSFUL")
            print(self.get_failure_report(total_responses=llm_processed))
            print(f"{'='*70}")
        else:
            print(f"\nAll {llm_processed} responses processed successfully (0 errors)")

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