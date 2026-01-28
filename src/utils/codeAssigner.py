import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import logging
import itertools
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np

from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
from pydantic import BaseModel, field_validator
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, GENERAL_CODE_LABELS, MISCELLANEOUS_CODE_LABELS, API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM
from utils.llm import create_client, llm_create_async, create_embedding_client, ProbeResponse, RateLimits, extract_rate_limits_from_response
from prompts import DEFAULT_CODE_EVALUATION_PROMPT, FALLBACK_CODE_ASSIGNMENT_PROMPT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding

try:
    import nest_asyncio  # for Spyder
    nest_asyncio.apply()
except ImportError:
    pass

# === STEP-SPECIFIC CONFIG =============================================================================================
from config_codeAssigner import (
    DEFAULT_TOKEN_HISTORY_CONFIG,
    DEFAULT_TIKTOKEN_OFFSET_CONFIG,
    DEFAULT_TIMEOUT_CONFIG,
    DEFAULT_REPORTING_CONFIG,
    DEFAULT_BOOTSTRAP_CONFIG,
    DEFAULT_PID_CONTROLLER_CONFIG,
    DEFAULT_TPM_TRACKING_CONFIG,
    DEFAULT_THROUGHPUT_CONFIG,
    DEFAULT_ADAPTIVE_THRESHOLD_CONFIG,
    DEFAULT_DYNAMIC_TOPK_CONFIG,
    DEFAULT_PATTERN_TRACKING_CONFIG,
)

# === CONSTANTS (from config_codeAssigner.py) =========================================================================
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
DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_timeout_seconds
DEFAULT_LATENCY_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_latency_seconds
MAX_TOKEN_ACQUIRE_ATTEMPTS = DEFAULT_TIMEOUT_CONFIG.max_token_acquire_attempts
BOOTSTRAP_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_CONFIG.bootstrap_timeout_seconds

# Reporting intervals
PROGRESS_REPORT_INTERVAL = DEFAULT_REPORTING_CONFIG.progress_report_interval
DIAGNOSTIC_INTERVAL = DEFAULT_REPORTING_CONFIG.diagnostic_interval
ADJUSTMENT_INTERVAL = DEFAULT_REPORTING_CONFIG.adjustment_interval

# Bootstrap settings
BOOTSTRAP_NUM_PROBES = DEFAULT_BOOTSTRAP_CONFIG.num_probes
DEFAULT_AVG_TOKENS = DEFAULT_BOOTSTRAP_CONFIG.default_avg_tokens
SAMPLE_SIZE_FOR_TOKEN_ESTIMATION = DEFAULT_BOOTSTRAP_CONFIG.sample_size_for_token_estimation

# PID-style continuous adjustment (asymmetric gains)
PID_KP_UP = DEFAULT_PID_CONTROLLER_CONFIG.kp_up
PID_KP_DOWN = DEFAULT_PID_CONTROLLER_CONFIG.kp_down
PID_KI = DEFAULT_PID_CONTROLLER_CONFIG.ki
PID_KD = DEFAULT_PID_CONTROLLER_CONFIG.kd
PID_MIN_ADJUSTMENT = DEFAULT_PID_CONTROLLER_CONFIG.min_adjustment
PID_MAX_ADJUSTMENT = DEFAULT_PID_CONTROLLER_CONFIG.max_adjustment

# Real-time TPM tracking
TPM_SLIDING_WINDOW_SECONDS = DEFAULT_TPM_TRACKING_CONFIG.sliding_window_seconds
TPM_SAMPLE_INTERVAL = DEFAULT_TPM_TRACKING_CONFIG.sample_interval
TPM_TARGET_UTILIZATION = DEFAULT_TPM_TRACKING_CONFIG.target_utilization

# Threshold-based adjustment (fallback to PID)
THROUGHPUT_ADJUSTMENT_MIN_SAMPLES = DEFAULT_THROUGHPUT_CONFIG.adjustment_min_samples
THROUGHPUT_ADJUSTMENT_THRESHOLD = DEFAULT_THROUGHPUT_CONFIG.adjustment_threshold


# === RATE LIMITING HELPER CLASSES ========================================================================================================

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
        logger.debug(f"[TOKEN BUCKET] Requesting {tokens_needed} tokens")
        
        while True:
            result = await self.acquire(tokens_needed)
            if result is True:
                logger.debug(f"[TOKEN BUCKET] Acquired {tokens_needed} tokens, {self.available:.0f} remaining")
                return
            else:
                # result is wait_seconds
                logger.debug(f"[TOKEN BUCKET] Insufficient tokens, waiting {result:.1f}s")
                await asyncio.sleep(result)
    
    async def reconcile(self, delta_tokens):
        """Reconcile actual vs estimated tokens"""
        # If we overestimated (delta < 0), return tokens
        # If we underestimated (delta > 0), we already used them
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
            return max(config.adaptive_timeout_min_seconds, 30.0)

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
            return 2.0  # Default 2s
        return self.ema if self.ema is not None else 2.0


# === V3 OPTIMAL STRATEGY CLASSES ========================================================================================================

class TiktokenOffsetLearner:
    """V3: Learns the offset between tiktoken counts and actual API token counts.

    The API always reports more tokens than tiktoken because of:
    - System messages added by the API
    - Instructor/structured output overhead
    - Message formatting tokens

    This class learns the average offset and applies it to estimates.
    """
    def __init__(self, default_offset: int = 300, history_maxlen: int = 30, min_samples: int = 5):
        self.default_offset = default_offset
        self.offsets = deque(maxlen=history_maxlen)
        self.min_samples = min_samples
        self._learned_offset = None

    def record(self, tiktoken_count: int, api_count: int):
        """Record a tiktoken vs API count pair to learn the offset."""
        offset = api_count - tiktoken_count
        self.offsets.append(offset)

        # Update learned offset when we have enough samples
        if len(self.offsets) >= self.min_samples:
            self._learned_offset = int(sum(self.offsets) / len(self.offsets))

    def get_offset(self) -> int:
        """Get the current offset to add to tiktoken counts."""
        if self._learned_offset is not None:
            return self._learned_offset
        return self.default_offset

    def is_learned(self) -> bool:
        """Check if we have enough samples to trust the learned offset."""
        return len(self.offsets) >= self.min_samples

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


class RealTimeTPMTracker:
    """V3: Tracks actual TPM usage in a sliding window for real-time feedback.

    Unlike the token bucket (which tracks available capacity), this tracks
    actual consumption to provide accurate utilization metrics.
    """
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.samples = deque()  # (timestamp, tokens) pairs
        self.lock = asyncio.Lock()

    async def record(self, tokens: int):
        """Record token usage at current time."""
        async with self.lock:
            now = time.monotonic()
            self.samples.append((now, tokens))
            self._prune_old_samples(now)

    def _prune_old_samples(self, now: float):
        """Remove samples outside the window."""
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    async def get_current_tpm(self) -> float:
        """Get current TPM rate based on sliding window."""
        async with self.lock:
            now = time.monotonic()
            self._prune_old_samples(now)

            if not self.samples:
                return 0.0

            total_tokens = sum(t for _, t in self.samples)
            elapsed = now - self.samples[0][0] if self.samples else 1.0
            elapsed = max(elapsed, 1.0)  # Avoid division by zero

            # Extrapolate to per-minute rate
            return (total_tokens / elapsed) * 60

    async def get_utilization(self, tpm_limit: int) -> float:
        """Get current TPM utilization as a percentage."""
        current_tpm = await self.get_current_tpm()
        return (current_tpm / tpm_limit) * 100 if tpm_limit > 0 else 0.0


class PIDThroughputController:
    """V3: PID controller for smooth, continuous throughput adjustment.

    Instead of step-based threshold adjustments, this provides gradual
    corrections that converge smoothly to optimal throughput.

    Uses ASYMMETRIC gains:
    - kp_up (0.4): Aggressive when under-utilizing (speed up faster)
    - kp_down (0.2): Gentle when over-utilizing (slow down carefully)

    The controller tracks:
    - Error: difference between target and actual TPM utilization
    - Integral: accumulated error over time (handles persistent bias)
    - Derivative: rate of change (dampens oscillations)
    """
    def __init__(
        self,
        target_utilization: float = 0.85,
        kp_up: float = 0.4,
        kp_down: float = 0.2,
        ki: float = 0.05,
        kd: float = 0.1,
        min_adjustment: float = 0.02,
        max_adjustment: float = 0.15
    ):
        self.target = target_utilization
        self.kp_up = kp_up      # Gain when under-utilizing (speed up)
        self.kp_down = kp_down  # Gain when over-utilizing (slow down)
        self.ki = ki
        self.kd = kd
        self.min_adjustment = min_adjustment
        self.max_adjustment = max_adjustment

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.adjustment_history = deque(maxlen=20)

    def compute_adjustment(self, current_utilization: float) -> float:
        """Compute throughput adjustment factor based on current utilization.

        Args:
            current_utilization: Current TPM utilization (0.0 to 1.0+)

        Returns:
            Adjustment factor to multiply current arrival rate by.
            - >1.0 means speed up (under-utilizing)
            - <1.0 means slow down (over-utilizing)
            - 1.0 means no change
        """
        now = time.monotonic()

        # Error: positive means under-utilizing, negative means over-utilizing
        error = self.target - current_utilization

        # Time delta for integral/derivative
        dt = 1.0  # Default
        if self.last_time is not None:
            dt = max(now - self.last_time, 0.1)
        self.last_time = now

        # Integral term (accumulated error)
        self.integral += error * dt
        # Clamp integral to prevent windup
        self.integral = max(-0.5, min(0.5, self.integral))

        # Derivative term (rate of change)
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        # Asymmetric proportional gain: aggressive up, gentle down
        kp = self.kp_up if error > 0 else self.kp_down

        # PID output
        output = (kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Clamp to reasonable adjustment range
        output = max(-self.max_adjustment, min(self.max_adjustment, output))

        # Convert to multiplier (1.0 + output)
        # Ignore tiny adjustments
        if abs(output) < self.min_adjustment:
            adjustment = 1.0
        else:
            adjustment = 1.0 + output

        self.adjustment_history.append({
            "time": now,
            "utilization": current_utilization,
            "error": error,
            "output": output,
            "adjustment": adjustment
        })

        return adjustment

    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def get_stats(self) -> dict:
        """Get controller statistics."""
        recent = list(self.adjustment_history)[-5:] if self.adjustment_history else []
        return {
            "target_utilization": self.target,
            "integral": self.integral,
            "last_error": self.last_error,
            "recent_adjustments": recent,
            "kp_up": self.kp_up,
            "kp_down": self.kp_down,
            "ki": self.ki,
            "kd": self.kd
        }


# === ADAPTIVE CONFIDENCE THRESHOLD CLASSES ========================================================================================================

class ConfidenceTracker:
    """Tracks running confidence distribution for adaptive thresholding.

    Instead of a fixed 0.7 threshold, this collects Stage 1 confidence scores
    and provides percentile-based adaptive thresholds.

    Benefits:
    - Adapts to codebook complexity (fine-grained codebooks naturally have lower confidence)
    - Accounts for model uncertainty patterns
    - Reduces unnecessary Stage 2 fallbacks when the codebook is working well
    """

    def __init__(
        self,
        percentile: int = 25,
        floor: float = 0.5,
        warmup: int = 20,
        history_maxlen: int = 500
    ):
        self.percentile = percentile
        self.floor = floor
        self.warmup = warmup
        self.confidences = deque(maxlen=history_maxlen)

    def record(self, confidence: float):
        """Record a confidence score from Stage 1 evaluation."""
        self.confidences.append(confidence)

    def get_adaptive_threshold(self, fixed_threshold: float) -> float:
        """Get adaptive threshold based on distribution.

        Returns fixed_threshold if warmup not complete.
        Otherwise returns the percentile of the distribution, floored at self.floor.
        """
        if len(self.confidences) < self.warmup:
            return fixed_threshold

        percentile_value = np.percentile(list(self.confidences), self.percentile)
        return max(self.floor, percentile_value)

    def is_warmed_up(self) -> bool:
        """Check if warmup period is complete."""
        return len(self.confidences) >= self.warmup

    def get_stats(self) -> dict:
        """Get distribution statistics for diagnostics."""
        if not self.confidences:
            return {"samples": 0}

        conf_list = list(self.confidences)
        return {
            "samples": len(conf_list),
            "mean": float(np.mean(conf_list)),
            "median": float(np.median(conf_list)),
            "p25": float(np.percentile(conf_list, 25)),
            "p75": float(np.percentile(conf_list, 75)),
            "min": float(min(conf_list)),
            "max": float(max(conf_list)),
            "current_threshold": self.get_adaptive_threshold(0.7),
            "is_warmed_up": self.is_warmed_up()
        }


# === PATTERN TRACKING CLASSES ========================================================================================================

@dataclass
class ClusterDiagnostics:
    """Diagnostics for a single cluster."""
    cluster_id: str
    total_ideas: int = 0
    used_default: int = 0
    used_fallback: int = 0
    confidences: List[float] = field(default_factory=list)

    @property
    def fallback_rate(self) -> float:
        if self.total_ideas == 0:
            return 0.0
        return self.used_fallback / self.total_ideas

    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


class PatternTracker:
    """Tracks patterns for learning and diagnostics.

    Provides insights on:
    - Code co-occurrence patterns (which codes appear together for same respondent)
    - Cluster-specific fallback rates (which clusters have poor default codes)
    - Confidence calibration (distribution of confidence scores by bucket)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        # Code co-occurrence: {(code_a, code_b): count}
        self.code_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        # Cluster diagnostics: {cluster_id: ClusterDiagnostics}
        self.cluster_diagnostics: Dict[str, ClusterDiagnostics] = {}

        # Confidence calibration buckets
        self.confidence_buckets = {
            "0.5-0.6": {"count": 0, "sum_confidence": 0.0},
            "0.6-0.7": {"count": 0, "sum_confidence": 0.0},
            "0.7-0.8": {"count": 0, "sum_confidence": 0.0},
            "0.8-0.9": {"count": 0, "sum_confidence": 0.0},
            "0.9-1.0": {"count": 0, "sum_confidence": 0.0},
        }

        # Assignment history for co-occurrence analysis
        self._respondent_codes: Dict[str, List[str]] = defaultdict(list)

    def record_assignment(
        self,
        respondent_id: str,
        cluster_id: str,
        assigned_code: str,
        confidence: float,
        used_default: bool,
        fallback_triggered: bool
    ):
        """Record a code assignment for pattern tracking."""
        if not self.enabled:
            return

        # Track cluster diagnostics
        if cluster_id:
            if cluster_id not in self.cluster_diagnostics:
                self.cluster_diagnostics[cluster_id] = ClusterDiagnostics(cluster_id=cluster_id)
            diag = self.cluster_diagnostics[cluster_id]
            diag.total_ideas += 1
            diag.confidences.append(confidence)
            if used_default:
                diag.used_default += 1
            if fallback_triggered:
                diag.used_fallback += 1

        # Track confidence distribution
        bucket = self._get_confidence_bucket(confidence)
        if bucket:
            self.confidence_buckets[bucket]["count"] += 1
            self.confidence_buckets[bucket]["sum_confidence"] += confidence

        # Track code co-occurrence (same respondent, multiple ideas)
        self._respondent_codes[respondent_id].append(assigned_code)

    def _get_confidence_bucket(self, confidence: float) -> Optional[str]:
        if 0.5 <= confidence < 0.6:
            return "0.5-0.6"
        elif 0.6 <= confidence < 0.7:
            return "0.6-0.7"
        elif 0.7 <= confidence < 0.8:
            return "0.7-0.8"
        elif 0.8 <= confidence < 0.9:
            return "0.8-0.9"
        elif 0.9 <= confidence <= 1.0:
            return "0.9-1.0"
        return None

    def finalize_cooccurrence(self):
        """Calculate code co-occurrence after all assignments complete."""
        from itertools import combinations

        for respondent_id, codes in self._respondent_codes.items():
            unique_codes = list(set(codes))
            if len(unique_codes) >= 2:
                for code_a, code_b in combinations(sorted(unique_codes), 2):
                    self.code_cooccurrence[(code_a, code_b)] += 1

    def get_problematic_clusters(self, fallback_threshold: float = 0.5, min_ideas: int = 3) -> List[Dict]:
        """Identify clusters with high fallback rates."""
        problematic = []
        for cluster_id, diag in self.cluster_diagnostics.items():
            if diag.total_ideas >= min_ideas and diag.fallback_rate >= fallback_threshold:
                problematic.append({
                    "cluster_id": cluster_id,
                    "total_ideas": diag.total_ideas,
                    "fallback_rate": diag.fallback_rate,
                    "avg_confidence": diag.avg_confidence
                })
        return sorted(problematic, key=lambda x: x["fallback_rate"], reverse=True)

    def get_top_cooccurrences(self, top_n: int = 10) -> List[Dict]:
        """Get most common code co-occurrences."""
        self.finalize_cooccurrence()
        sorted_pairs = sorted(
            self.code_cooccurrence.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        return [
            {"code_a": pair[0], "code_b": pair[1], "count": count}
            for (pair, count) in sorted_pairs
        ]

    def get_confidence_calibration(self) -> Dict:
        """Get confidence calibration statistics."""
        result = {}
        for bucket, data in self.confidence_buckets.items():
            if data["count"] > 0:
                result[bucket] = {
                    "count": data["count"],
                    "avg_confidence": data["sum_confidence"] / data["count"]
                }
        return result


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


# === PYDANTIC MODELS ========================================================================================================

class CodeAssignmentResponse(BaseModel):
    idea_id: str
    idea: str
    assigned_codes: List[str]
    assignment_confidence: float
    assignment_rationale: str
    assigned_themes: Optional[List[str]] = None

class DefaultCodeEvaluationResponse(BaseModel):
    """Stage 1: Evaluating default code from cluster"""
    idea_id: str
    confidence: float
    rationale: str

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        """Coerce string numbers to float (common with LLM JSON output)"""
        if isinstance(v, str):
            return float(v)
        return v

class FallbackCodeAssignmentResponse(BaseModel):
    """Stage 2: Selecting from all codes"""
    idea_id: str
    assigned_codes: List[str]
    assignment_confidence: float
    assignment_rationale: str

    @field_validator('assignment_confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        """Coerce string numbers to float (common with LLM JSON output)"""
        if isinstance(v, str):
            return float(v)
        return v


class CodeAssigner:
    """
    Two-stage code assignment using embedding-based similarity filtering.
    Stage 2 presents top-10 most similar codes instead of entire codebook.
    """
    
    def __init__(
        self,
        cluster_models: List[models.ClusterModel],
        codebook: List[models.Codebook],
        var_lab: str,
        code_to_theme_mapping: Optional[Dict[str, str]] = None,
        config: Optional[CodeAssignmentConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        adaptive_threshold_config = None,
        dynamic_topk_config = None,
        pattern_config = None,
        verbose: bool = False,
        prompt_printer = None):

        self.cluster_models = cluster_models
        self.codebook = codebook
        self.var_lab = var_lab
        self.config = config or DEFAULT_CODE_ASSIGNMENT_CONFIG
        self.model_config = model_config or ModelConfig()
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.model_config.get_model_for_stage('code_assignment')
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.CodeAssignedModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        self._captured_prompt = False

        # Theme mapping for code-to-theme assignments
        self.code_to_theme_mapping = code_to_theme_mapping or {}

        # Initialize tokenizer for token counting (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Instructor-patched async client for structured output (Azure/OpenAI abstracted)
        self.client = create_client(model=self.model, async_mode=True)

        # Embedding client for code similarity (plain OpenAI client)
        self.embedding_client = create_embedding_client(async_mode=False)
        self.embedding_model = self.model_config.embedding_model

        # Rate limiting setup - use fallback values for initial setup
        # Actual rate limits will be fetched from API during processing
        self.rate_limits = RateLimits(
            tokens_per_minute=FALLBACK_TPM,
            requests_per_minute=FALLBACK_RPM,
            tokens_per_day=FALLBACK_TPM * 60 * 24
        )

        # Token bucket for TPM limiting (will be re-initialized with actual limits during bootstrap)
        self.tpm_bucket = TokenBucket(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

        # Progressive token estimation (V3: updated to use config constants)
        self.input_token_history = deque(maxlen=INPUT_HISTORY_MAXLEN)
        self.output_token_history = deque(maxlen=OUTPUT_HISTORY_MAXLEN)
        self.output_ratio_history = deque(maxlen=OUTPUT_RATIO_HISTORY_MAXLEN)  # V3: Track output/input ratios for learning
        self.estimation_errors = deque(maxlen=ERROR_WINDOW_SIZE)
        self.first_prompt_tokens = None  # Cache first prompt calculation

        # Rolling average of actual total tokens for comparison
        self.actual_total_tokens = deque(maxlen=ERROR_WINDOW_SIZE)

        # Latency tracking
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)

        # === V3 OPTIMAL STRATEGY COMPONENTS ===
        # Tiktoken→API offset learning (accounts for system overhead)
        self.tiktoken_offset_learner = TiktokenOffsetLearner(
            default_offset=TIKTOKEN_API_OFFSET_DEFAULT,
            history_maxlen=TIKTOKEN_OFFSET_HISTORY_MAXLEN,
            min_samples=TIKTOKEN_OFFSET_MIN_SAMPLES
        )

        # Real-time TPM tracking (sliding window)
        self.tpm_tracker = RealTimeTPMTracker(window_seconds=TPM_SLIDING_WINDOW_SECONDS)

        # PID throughput controller (continuous adjustment)
        self.pid_controller = PIDThroughputController(
            target_utilization=TPM_TARGET_UTILIZATION,
            kp_up=PID_KP_UP,
            kp_down=PID_KP_DOWN,
            ki=PID_KI,
            kd=PID_KD,
            min_adjustment=PID_MIN_ADJUSTMENT,
            max_adjustment=PID_MAX_ADJUSTMENT
        )

        # Track current arrival rate for PID adjustment
        self.current_arrival_rate = None

        # V3 stats tracking
        self.v3_stats = {
            'adjustments_made': 0,
            'pid_adjustments': 0,
            'threshold_adjustments': 0,
            'max_tpm_utilization': 0.0,
            'min_tpm_utilization': 100.0,
        }

        # === CODE ASSIGNMENT STRATEGY IMPROVEMENTS ===
        # Store configs (use defaults if not provided)
        self.adaptive_threshold_config = adaptive_threshold_config or DEFAULT_ADAPTIVE_THRESHOLD_CONFIG
        self.dynamic_topk_config = dynamic_topk_config or DEFAULT_DYNAMIC_TOPK_CONFIG
        self.pattern_config = pattern_config or DEFAULT_PATTERN_TRACKING_CONFIG

        # Confidence tracker for adaptive threshold
        self.confidence_tracker = ConfidenceTracker(
            percentile=self.adaptive_threshold_config.adaptive_percentile,
            floor=self.adaptive_threshold_config.adaptive_floor,
            warmup=self.adaptive_threshold_config.warmup_samples
        )

        # Pattern tracker for diagnostics
        self.pattern_tracker = PatternTracker(enabled=self.pattern_config.enabled)

        # Track last similarity stats for diagnostics
        self._last_selected_count = 0
        self._last_top_similarity = 0.0
        
        # Calculate initial average tokens estimate for bootstrapping
        self.avg_tokens = self._calculate_avg_tokens()
        
        # Rate limiting components (will be initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None
        
        # Stats
        self._stats = ProcessingStats()
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'retries': 0,
            'rate_limits': 0,
            'timeouts': 0,
            'error_types': {}  # Track error types: {error_type: count}
        }

        # Two-stage assignment stats
        self.stage_1_calls = 0
        self.stage_2_calls = 0
        self.used_default_count = 0
        self.used_fallback_count = 0

        # Prompt/Response logging for debugging
        self.prompt_responses = []
        self.last_prompt = ""  # Track the last prompt used for assignment
        self.verbose = verbose

        # Build cluster→codes mapping
        self.cluster_to_codes = self._build_cluster_code_mapping()

        # Code embeddings for similarity filtering (lazy-load on first use)
        self._code_embeddings = None

        self.verbose_reporter.stat_line(f"Model: {self.model}")
        self.verbose_reporter.stat_line(f"API Limits: {self.rate_limits.requests_per_minute} RPM, {self.rate_limits.tokens_per_minute:,} TPM")

    def _calculate_avg_tokens(self) -> int:
        """Calculate average token count for code assignment requests"""
        if not self.cluster_models:
            return 1500  # Default estimate
        
        sample_size = min(10, len(self.cluster_models))
        total_tokens = 0
        sample_count = 0
        
        for i in range(sample_size):
            model = self.cluster_models[i]
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea in model.response_ideas:
                    if hasattr(idea, 'idea') and idea.idea:
                        # Create sample prompt
                        prompt = self._create_prompt_for_estimation(idea.idea_id, idea.idea)
                        total_tokens += len(self.encoding.encode(prompt))
                        sample_count += 1
                        break  # Only sample first idea per model
        
        if sample_count == 0:
            return 1500  # Fallback
        
        avg_input = total_tokens / sample_count
        # Assume 15% output ratio initially
        return int(avg_input * 1.15)

    def _build_cluster_code_mapping(self) -> Dict[str, List[models.Codebook]]:
        """Build mapping from expanded_cluster ID to codes generated from that cluster"""
        from collections import defaultdict
        mapping = defaultdict(list)
        merged_codes_count = 0

        for code in self.codebook:
            if hasattr(code, 'source_cluster') and code.source_cluster:
                # Split comma-separated cluster IDs (e.g., "8,11,23" → ["8", "11", "23"])
                cluster_ids = str(code.source_cluster).split(',')

                # Log if multiple clusters share this code
                if len(cluster_ids) > 1:
                    merged_codes_count += 1
                    # if self.verbose:
                    #     cluster_list = [c.strip() for c in cluster_ids]
                    #     #self.verbose_reporter.info(f"  Code '{code.code[:50]}...' mapped to {len(cluster_list)} clusters: {cluster_list}")

                # Create mapping for each individual cluster ID
                for cluster_id in cluster_ids:
                    cluster_id = cluster_id.strip()  # Remove whitespace
                    if cluster_id:  # Skip empty strings
                        mapping[cluster_id].append(code)

        # Convert to regular dict and log stats
        cluster_dict = dict(mapping)
        if self.verbose:
            total_clusters_with_codes = len(cluster_dict)
            avg_codes_per_cluster = sum(len(codes) for codes in cluster_dict.values()) / total_clusters_with_codes if total_clusters_with_codes > 0 else 0
            self.verbose_reporter.stat_line(f"Cluster→Code mapping: {total_clusters_with_codes} clusters, avg {avg_codes_per_cluster:.1f} codes/cluster")
            if merged_codes_count > 0:
                self.verbose_reporter.stat_line(f"  {merged_codes_count} codes shared across multiple clusters")

        return cluster_dict

    def _create_prompt_for_estimation(self, idea_id: str, idea_text: str) -> str:
        """Create prompt for token estimation using top-k codes"""
        top_k = self.config.top_k_similar_codes
        all_codes_text = "\n".join([
            f"Code: {code.code}\nDefinition: {code.definition}\n"
            for code in self.codebook[:top_k]
        ])

        unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")

        return FALLBACK_CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_confidence=0.0,
            all_codes=all_codes_text,
            unknown_label=unknown_label
        )

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
        """Make a minimal API call to fetch rate limits from response headers."""
        from openai import AsyncOpenAI
        from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

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

    def estimate_tokens(self, prompt: str) -> int:
        """V3: Estimate total tokens using optimal adaptive strategy.

        V3 Improvements:
        - Applies learned tiktoken→API offset upfront
        - Reduced safety margins (offset handles the gap)
        - Uses output ratio learning for more accurate output estimation
        - Weighted blend: 70% history, 30% current for stability
        """
        # Count tokens with tiktoken
        tiktoken_count = len(self.encoding.encode(prompt))

        # V3: Apply learned offset (accounts for system overhead)
        offset = self.tiktoken_offset_learner.get_offset()
        actual_input_tokens = tiktoken_count + offset

        # V3: Reduced safety margins (offset already accounts for gap)
        num_samples = len(self.estimation_errors)
        if num_samples < 5:
            safety_margin = 1.15  # V3: Reduced from higher margins (offset handles gap)
        elif num_samples < 15:
            safety_margin = 1.10  # V3: Reduced once we have some data
        else:
            safety_margin = 1.05  # V3: Tight when learned

        # Input estimation: use history average if available, blend with current
        if len(self.input_token_history) >= 5:
            avg_input = sum(self.input_token_history) / len(self.input_token_history)
            # Weighted blend: 70% history, 30% current for stability
            estimated_input = int(0.7 * avg_input + 0.3 * actual_input_tokens)
        else:
            # Early phase: use current with safety margin
            estimated_input = int(actual_input_tokens * safety_margin)

        # Always update input history
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

    def _assign_themes_to_codes(self, assigned_codes: List[str]) -> List[str]:
        """Map assigned codes to their themes using cached mapping"""
        themes = []
        for code in assigned_codes:
            theme = self.code_to_theme_mapping.get(code)
            if theme and theme not in themes:
                themes.append(theme)
        return themes

    def _build_general_codes(self) -> List[Dict[str, any]]:
        """Build synthetic general codes for theme and category fallbacks"""
        general_codes = []
        general_label = GENERAL_CODE_LABELS.get(self.language, "overall")

        # Theme-level general codes
        unique_themes = set()
        for code in self.codebook:
            if hasattr(code, 'theme') and code.theme:
                theme_desc = getattr(code, 'theme_description', code.theme)
                unique_themes.add((code.theme, theme_desc))

        for theme, theme_desc in unique_themes:
            # Collect specific codes in this theme for exclusion examples
            specific_codes_in_theme = [code.code for code in self.codebook
                                      if hasattr(code, 'theme') and code.theme == theme]

            general_codes.append({
                'code': f"{theme} - {general_label}",
                'definition': f"Algemene verwijzing naar thema '{theme}': {theme_desc}",
                'inclusion_examples': [
                    f"Algemene of vage verwijzing naar {theme}",
                    f"Niet-specifieke uitspraak over {theme}",
                    f"Vaag verband met {theme} zonder concrete details"
                ],
                'exclusion_examples': specific_codes_in_theme,
                'near_neighbor_label': "Specifieke codes binnen dit thema",
                'tell_apart_rule': f"Gebruik deze algemene code alleen als geen enkele specifieke code past. Specifieke codes in dit thema: {', '.join(specific_codes_in_theme[:3])}{'...' if len(specific_codes_in_theme) > 3 else ''}",
                'type': 'theme_general'
            })

        # Category-level general codes (if 3-level hierarchy exists)
        unique_categories = set()
        for code in self.codebook:
            if hasattr(code, 'category') and code.category:
                cat_desc = getattr(code, 'category_description', code.category)
                theme = getattr(code, 'theme', '')
                unique_categories.add((code.category, cat_desc, theme))

        for category, cat_desc, theme in unique_categories:
            # Collect specific codes in this category for exclusion examples
            specific_codes_in_category = [code.code for code in self.codebook
                                         if hasattr(code, 'category') and code.category == category]

            general_codes.append({
                'code': f"{category} - {general_label}",
                'definition': f"Algemene verwijzing naar categorie '{category}' binnen {theme}: {cat_desc}",
                'inclusion_examples': [
                    f"Algemene of vage verwijzing naar {category}",
                    f"Niet-specifieke uitspraak over {category} binnen {theme}",
                    f"Vaag verband met {category} zonder concrete details"
                ],
                'exclusion_examples': specific_codes_in_category,
                'near_neighbor_label': "Specifieke codes binnen deze categorie",
                'tell_apart_rule': f"Gebruik deze algemene code alleen als geen enkele specifieke code past. Specifieke codes in deze categorie: {', '.join(specific_codes_in_category[:3])}{'...' if len(specific_codes_in_category) > 3 else ''}",
                'type': 'category_general'
            })

        return general_codes

    def _generate_code_embeddings(self) -> np.ndarray:
        """Generate embeddings for all codes (code + definition)"""
        code_texts = [f"Code: {code.code}. Definition: {code.definition}"
                      for code in self.codebook]

        embeddings = []
        for text in code_texts:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embeddings.append(response.data[0].embedding)

        return np.array(embeddings)

    @property
    def code_embeddings(self):
        """Lazy-load code embeddings on first use"""
        if self._code_embeddings is None:
            self._code_embeddings = self._generate_code_embeddings()
        return self._code_embeddings

    def _find_similar_codes(self, idea_embedding: np.ndarray, top_k: int = 10) -> List[models.Codebook]:
        """Find similar codes using configurable selection strategy.

        Supports three modes via dynamic_topk_config:
        - "fixed": Return exactly top_k codes (default, backward compatible)
        - "threshold": Return all codes above similarity_threshold
        - "dropoff": Return codes until similarity drops significantly from best
        """
        similarities = cosine_similarity([idea_embedding], self.code_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_similarities = similarities[sorted_indices]

        mode = self.dynamic_topk_config.mode

        if mode == "fixed":
            # Original behavior: return exactly top_k
            k = self.config.top_k_similar_codes
            selected = sorted_indices[:k]
        elif mode == "threshold":
            # Return all codes above similarity threshold
            above_threshold = sorted_similarities >= self.dynamic_topk_config.similarity_threshold
            count = int(np.sum(above_threshold))
            count = min(count, self.dynamic_topk_config.max_codes)
            count = max(count, self.dynamic_topk_config.min_codes)
            selected = sorted_indices[:count]
        elif mode == "dropoff":
            # Return codes until similarity drops significantly from best
            best_sim = sorted_similarities[0]
            cutoff = best_sim * self.dynamic_topk_config.dropoff_ratio
            count = sum(1 for s in sorted_similarities if s >= cutoff)
            count = max(count, self.dynamic_topk_config.min_codes)
            count = min(count, self.dynamic_topk_config.max_codes)
            selected = sorted_indices[:count]
        else:
            # Fallback to original behavior
            selected = sorted_indices[:self.config.top_k_similar_codes]

        # Track for diagnostics
        self._last_selected_count = len(selected)
        self._last_top_similarity = float(sorted_similarities[0]) if len(sorted_similarities) > 0 else 0.0

        return [self.codebook[i] for i in selected]

    def _format_examples_list(self, examples: Optional[List[str]]) -> str:
        """Format examples list for prompt display"""
        if not examples:
            return "No specific examples provided"
        return "\n".join([f"  • {ex}" for ex in examples])

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all individual ideas for processing with expanded_cluster info"""
        all_ideas = []

        for model in self.cluster_models:
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    if hasattr(idea_submodel, 'idea_embedding') and idea_submodel.idea_embedding is not None:
                        # Extract expanded_cluster (fallback to initial_cluster if not available)
                        expanded_cluster = getattr(idea_submodel, 'expanded_cluster', None) or \
                                         getattr(idea_submodel, 'initial_cluster', None)

                        all_ideas.append((
                            model.respondent_id,
                            idea_submodel.idea_id,
                            idea_submodel.idea,
                            idea_submodel.idea_embedding,
                            expanded_cluster
                        ))
                    else:
                        self.verbose_reporter.stat_line(f"Warning: No embedding for idea {idea_submodel.idea_id}")
            else:
                self.verbose_reporter.stat_line(f"Warning: No response_ideas found for respondent {model.respondent_id}")

        return all_ideas

    def _create_prompt(self, idea_id: str, idea_text: str) -> str:
        """Create prompt for probe calls using Stage 2 (all codes) prompt"""
        # Format all codes
        all_codes_text = "\n".join([
            f"Code: {code.code}\nDefinition: {code.definition}\n"
            for code in self.codebook
        ])

        unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")

        prompt = FALLBACK_CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_confidence=0.0,
            all_codes=all_codes_text,
            unknown_label=unknown_label
        )

        return prompt
    
    async def probe_call_no_structured(self, task_dict):
        """Probe call with minimal response model for bootstrap measurement"""
        idea_data = task_dict['idea_data']
        respondent_id, idea_id, idea_text, idea_embedding, expanded_cluster = idea_data

        prompt = self._create_prompt(idea_id, idea_text)

        # Use minimal ProbeResponse model for Azure compatibility (instructor requires response_model)
        resp = await llm_create_async(
            client=self.client,
            model=self.model,
            prompt=prompt,
            response_model=ProbeResponse,
            temperature=self.config.temperature,
            track_usage=False  # We're extracting usage manually for bootstrap
        )

        # Extract usage from instructor's _raw_response
        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        # Handle both Azure (prompt_tokens) and OpenAI (input_tokens) response formats
        if u:
            prompt_tokens = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0)
            completion_tokens = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0)
            return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        return {"prompt_tokens": 0, "completion_tokens": 0}

    async def evaluate_default_code(self, idea_id: str, idea_text: str, default_code: models.Codebook):
        """Stage 1: Evaluate how well the default code from cluster fits the idea

        Returns:
            tuple: (DefaultCodeEvaluationResponse, str) - response and prompt used
        """

        prompt = DEFAULT_CODE_EVALUATION_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_code=default_code.code,
            default_definition=default_code.definition,
            inclusion_examples=self._format_examples_list(default_code.inclusion_examples),
            exclusion_examples=self._format_examples_list(default_code.exclusion_examples),
            near_neighbor_label=default_code.near_neighbor_label or "Unknown",
            tell_apart_rule=default_code.tell_apart_rule or "N/A"
        )

        self.last_prompt = prompt  # Store for backward compatibility

        # Estimate tokens for rate limiting
        est_tokens = self.estimate_tokens(prompt)

        # Calculate adaptive timeout
        timeout = self.latency_tracker.get_timeout(est_tokens)

        # FIX CONVOY EFFECT: Acquire semaphore FIRST to bound waiters,
        # then acquire token bucket and rate limiter
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            async with self.rate_limiter:
                start_time = time.perf_counter()

                response = await asyncio.wait_for(
                    llm_create_async(
                        client=self.client,
                        model=self.model,
                        prompt=prompt,
                        response_model=DefaultCodeEvaluationResponse,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        track_usage=True
                    ),
                    timeout=timeout
                )

                # Track latency for adaptive timeout adjustment
                latency = time.perf_counter() - start_time
                self.latency_tracker.add(latency)

                # Token reconciliation: reconcile actual vs estimated
                if hasattr(response, '_raw_response'):
                    usage = response._raw_response.usage
                    if usage:
                        actual_input_tokens = getattr(usage, 'prompt_tokens', None) or getattr(usage, 'input_tokens', 0)
                        actual_output_tokens = getattr(usage, 'completion_tokens', None) or getattr(usage, 'output_tokens', 0)
                        actual_total_tokens = getattr(usage, 'total_tokens', 0) or (actual_input_tokens + actual_output_tokens)

                        # Reconcile token bucket
                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)

                        # V3: Track output tokens for learning
                        self.output_token_history.append(actual_output_tokens)
                        self.actual_total_tokens.append(actual_total_tokens)

                        # V3: Track output ratio for learning
                        if actual_input_tokens > 0:
                            ratio = actual_output_tokens / actual_input_tokens
                            self.output_ratio_history.append(ratio)

                        # V3: Track estimation error
                        estimation_error = abs(actual_total_tokens - est_tokens)
                        self.estimation_errors.append(estimation_error)

                        # V3: Record to TPM tracker (real-time sliding window)
                        await self.tpm_tracker.record(actual_total_tokens)

                        # V3: Learn tiktoken→API offset
                        tiktoken_input = len(self.encoding.encode(prompt))
                        self.tiktoken_offset_learner.record(tiktoken_input, actual_input_tokens)

        self.stage_1_calls += 1
        return response, prompt

    async def assign_from_all_codes(self, idea_id: str, idea_text: str, idea_embedding: np.ndarray, default_confidence: float):

        # Build list of codes: top-10 similar + general + unknown
        all_codes_list = []

        # 1. Add top-10 most similar codes using embeddings
        top_k = self.config.top_k_similar_codes
        similar_codes = self._find_similar_codes(idea_embedding, top_k=top_k)

        for code in similar_codes:
            all_codes_list.append({
                'code': code.code,
                'definition': code.definition,
                'inclusion_examples': code.inclusion_examples,
                'exclusion_examples': code.exclusion_examples,
                'near_neighbor_label': code.near_neighbor_label,
                'tell_apart_rule': code.tell_apart_rule,
                'type': 'specific'
            })

        # 2. Add general codes (theme/category level)
        all_codes_list.extend(self._build_general_codes())

        # 3. Add unknown fallback
        unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")
        all_codes_list.append({
            'code': unknown_label,
            'definition': "Geen duidelijke relatie met thema's in codebook",
            'inclusion_examples': None,
            'exclusion_examples': None,
            'near_neighbor_label': None,
            'tell_apart_rule': None,
            'type': 'unknown'
        })

        # Format all codes for prompt
        all_codes_text = "\n".join([
            f"Code: {c['code']}\n"
            f"Definition: {c['definition']}\n"
            f"Include when: {self._format_examples_list(c.get('inclusion_examples'))}\n"
            f"Exclude when: {self._format_examples_list(c.get('exclusion_examples'))}\n"
            f"Boundary: Differs from '{c.get('near_neighbor_label') or 'N/A'}' - {c.get('tell_apart_rule') or 'N/A'}\n"
            for c in all_codes_list
        ])

        prompt = FALLBACK_CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_confidence=default_confidence,
            all_codes=all_codes_text,
            unknown_label=unknown_label
        )

        self.last_prompt = prompt  # Store for backward compatibility

        # Estimate tokens for rate limiting
        est_tokens = self.estimate_tokens(prompt)

        # Calculate adaptive timeout
        timeout = self.latency_tracker.get_timeout(est_tokens)

        # FIX CONVOY EFFECT: Acquire semaphore FIRST to bound waiters,
        # then acquire token bucket and rate limiter
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            async with self.rate_limiter:
                start_time = time.perf_counter()

                response = await asyncio.wait_for(
                    llm_create_async(
                        client=self.client,
                        model=self.model,
                        prompt=prompt,
                        response_model=FallbackCodeAssignmentResponse,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        track_usage=True
                    ),
                    timeout=timeout
                )

                # Track latency for adaptive timeout adjustment
                latency = time.perf_counter() - start_time
                self.latency_tracker.add(latency)

                # Token reconciliation: reconcile actual vs estimated
                if hasattr(response, '_raw_response'):
                    usage = response._raw_response.usage
                    if usage:
                        actual_input_tokens = getattr(usage, 'prompt_tokens', None) or getattr(usage, 'input_tokens', 0)
                        actual_output_tokens = getattr(usage, 'completion_tokens', None) or getattr(usage, 'output_tokens', 0)
                        actual_total_tokens = getattr(usage, 'total_tokens', 0) or (actual_input_tokens + actual_output_tokens)

                        # Reconcile token bucket
                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)

                        # V3: Track output tokens for learning
                        self.output_token_history.append(actual_output_tokens)
                        self.actual_total_tokens.append(actual_total_tokens)

                        # V3: Track output ratio for learning
                        if actual_input_tokens > 0:
                            ratio = actual_output_tokens / actual_input_tokens
                            self.output_ratio_history.append(ratio)

                        # V3: Track estimation error
                        estimation_error = abs(actual_total_tokens - est_tokens)
                        self.estimation_errors.append(estimation_error)

                        # V3: Record to TPM tracker (real-time sliding window)
                        await self.tpm_tracker.record(actual_total_tokens)

                        # V3: Learn tiktoken→API offset
                        tiktoken_input = len(self.encoding.encode(prompt))
                        self.tiktoken_offset_learner.record(tiktoken_input, actual_input_tokens)

        self.stage_2_calls += 1
        return response, prompt

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
    async def process_task(self, task: Dict) -> CodeAssignmentResponse:
        """Two-stage code assignment: evaluate default from cluster, fallback to all codes if needed"""
        #task_start = time.perf_counter()

        try:
            idea_data = task['idea_data']
            respondent_id, idea_id, idea_text, idea_embedding, expanded_cluster = idea_data

            # Metadata for tracking default vs fallback
            metadata = {
                'used_default': False,
                'fallback_triggered': False,
                'default_confidence': None,
                'expanded_cluster': expanded_cluster
            }

            # Local variable to capture the prompt for this specific task (avoid race conditions)
            prompt_used = ""

            # Get default code(s) from this idea's cluster
            default_codes = self.cluster_to_codes.get(str(expanded_cluster), [])

            if not default_codes:
                # No codes from this cluster - go straight to fallback (Stage 2)
                metadata['fallback_triggered'] = True
                stage_2_result, prompt_used = await self.assign_from_all_codes(idea_id, idea_text, idea_embedding, default_confidence=0.0)

                assigned_code = stage_2_result.assigned_codes[0]
                confidence = stage_2_result.assignment_confidence
                rationale = f"No default code available. {stage_2_result.assignment_rationale}"
                self.used_fallback_count += 1

            else:
                # Stage 1: Evaluate default code from cluster
                default_code = default_codes[0]  # Use first code from cluster
                stage_1_result, prompt_used = await self.evaluate_default_code(idea_id, idea_text, default_code)

                metadata['default_confidence'] = stage_1_result.confidence

                # Record for adaptive threshold tracking
                self.confidence_tracker.record(stage_1_result.confidence)

                # Determine threshold (adaptive or fixed)
                if self.adaptive_threshold_config.use_adaptive:
                    threshold = self.confidence_tracker.get_adaptive_threshold(
                        self.adaptive_threshold_config.fixed_threshold
                    )
                else:
                    threshold = self.adaptive_threshold_config.fixed_threshold
                metadata['threshold_used'] = threshold

                if stage_1_result.confidence >= threshold:
                    # Use default code
                    metadata['used_default'] = True
                    assigned_code = default_code.code
                    confidence = stage_1_result.confidence
                    rationale = stage_1_result.rationale
                    self.used_default_count += 1

                else:
                    # Stage 2: Fallback to similar codes (dynamic top-k)
                    metadata['fallback_triggered'] = True
                    stage_2_result, prompt_used = await self.assign_from_all_codes(idea_id, idea_text, idea_embedding, stage_1_result.confidence)

                    assigned_code = stage_2_result.assigned_codes[0]
                    confidence = stage_2_result.assignment_confidence
                    rationale = f"Default: {stage_1_result.rationale} | Fallback: {stage_2_result.assignment_rationale}"
                    self.used_fallback_count += 1

            # Create response
            response = CodeAssignmentResponse(
                idea_id=idea_id,
                idea=idea_text,
                assigned_codes=[assigned_code],
                assignment_confidence=confidence,
                assignment_rationale=rationale
            )

            # Add theme mapping
            response.assigned_themes = self._assign_themes_to_codes([assigned_code])

            # Record pattern for diagnostics
            self.pattern_tracker.record_assignment(
                respondent_id=respondent_id,
                cluster_id=str(expanded_cluster) if expanded_cluster else "",
                assigned_code=assigned_code,
                confidence=confidence,
                used_default=metadata.get('used_default', False),
                fallback_triggered=metadata.get('fallback_triggered', False)
            )

            # Capture for debugging (only if verbose)
            if self.verbose:
                self.prompt_responses.append({
                    'prompt': prompt_used,  # Use local variable to avoid race conditions with concurrent tasks
                    'respondent_id': respondent_id,
                    'idea_id': idea_id,
                    'idea_text': idea_text,
                    'expanded_cluster': expanded_cluster,
                    'assigned_codes': [assigned_code],
                    'confidence': confidence,
                    'rationale': rationale,
                    'metadata': metadata
                })

            self.stats['tasks_successful'] += 1
            return response

        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.warning(f"Task {task['task_id']} timed out")
            raise  # Let tenacity retry

        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['task_id']} hit rate limit")
            raise  # Let tenacity retry

        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {type(e).__name__}: {e}")
            raise  # Let tenacity retry
    
    def create_fallback_response(self, task: Dict) -> CodeAssignmentResponse:
        """Create fallback response for failed tasks"""
        idea_data = task['idea_data']
        respondent_id, idea_id, idea_text, idea_embedding, expanded_cluster = idea_data
        
        # Return fallback response (first available code)
        fallback_code = self.codebook[0].code if self.codebook else "Unknown"
        fallback_themes = self._assign_themes_to_codes([fallback_code]) if fallback_code != "Unknown" else []
        
        return CodeAssignmentResponse(
            idea_id=idea_id,
            idea=idea_text,
            assigned_codes=[fallback_code],
            assigned_themes=fallback_themes,
            assignment_confidence=0.1,
            assignment_rationale="Processing failed, using fallback code"
        )
    
    async def worker(self, queue: asyncio.Queue, results: List):
        """Worker coroutine that processes tasks from queue"""
        #worker_id = id(asyncio.current_task())
        task_count = 0
        
        while True:
            try:
                task = await queue.get()
                if task is None:  # Sentinel
                    #print(f"[DEBUG] Worker {worker_id} received sentinel, processed {task_count} tasks")
                    break
                
                task_count += 1
                #print(f"[DEBUG] Worker {worker_id} processing task {task_count}: {task.get('task_id', 'unknown')}")
                
                try:
                    result = await self.process_task(task)
                    results[task['result_index']] = result
                    #print(f"[DEBUG] Worker {worker_id} task {task_count} SUCCESS")
                except Exception as e:
                    # After all retries failed
                    #print(f"[DEBUG] Worker {worker_id} task {task_count} FAILED: {type(e).__name__}: {e}")
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # Track error types
                    if error_type not in self.stats['error_types']:
                        self.stats['error_types'][error_type] = {'count': 0, 'sample_messages': []}
                    self.stats['error_types'][error_type]['count'] += 1
                    # Store up to 3 sample error messages per type
                    if len(self.stats['error_types'][error_type]['sample_messages']) < 3:
                        self.stats['error_types'][error_type]['sample_messages'].append(error_msg[:200])

                    logger.error(f"Task {task['task_id']} failed after retries: {error_type}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    self.stats['tasks_failed'] += 1
                    results[task['result_index']] = self.create_fallback_response(task)
                finally:
                    self.stats['tasks_processed'] += 1
                    queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker error: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"Worker traceback: {traceback.format_exc()}")
                break

    async def _process_single_idea(self, idea_data: tuple) -> CodeAssignmentResponse:
        """Deprecated - use process_task instead"""
        # This method is now deprecated - it's kept for compatibility
        # All processing now goes through the worker queue pattern
        pass

    def _merge_results_into_models(self, assignment_results: List[CodeAssignmentResponse]) -> List[models.CodeAssignedModel]:
        """Merge assignment results back into model structure"""

        # Create lookup for assignments by idea_id
        assignments_lookup = {result.idea_id: result for result in assignment_results}
        
        coded_models = []
        
        for original_model in self.cluster_models:
            # Convert to CodeAssignedModel
            coded_model = original_model.to_model(models.CodeAssignedModel)
            
            # Update response_ideas with assignments
            if coded_model.response_ideas:
                updated_ideas = []
                for idea_submodel in coded_model.response_ideas:
                    # Convert to AssignedIdeaSubmodel
                    assigned_idea = models.AssignedIdeaSubmodel(
                        idea_id=idea_submodel.idea_id,
                        idea=idea_submodel.idea,
                        initial_cluster=getattr(idea_submodel, 'initial_cluster', None),
                        expanded_cluster=getattr(idea_submodel, 'expanded_cluster', None),
                        cluster_theme=getattr(idea_submodel, 'cluster_theme', None),
                        idea_embedding=getattr(idea_submodel, 'idea_embedding', None)
                    )
                    
                    # Add assignment data if available
                    if idea_submodel.idea_id in assignments_lookup:
                        assignment = assignments_lookup[idea_submodel.idea_id]
                        assigned_idea.assigned_codes = assignment.assigned_codes
                        assigned_idea.assigned_themes = assignment.assigned_themes
                        assigned_idea.assignment_confidence = assignment.assignment_confidence
                        assigned_idea.assignment_rationale = assignment.assignment_rationale
                    else:
                        # Fallback if no assignment found
                        assigned_idea.assigned_codes = ["Unassigned"]
                        assigned_idea.assigned_themes = []
                        assigned_idea.assignment_confidence = 0.0
                        assigned_idea.assignment_rationale = "No assignment found"
                    
                    updated_ideas.append(assigned_idea)
                
                coded_model.response_ideas = updated_ideas
            
            coded_models.append(coded_model)

        return coded_models

    def get_random_samples(self, n: int = 3, seed: int = None) -> List[Dict]:
        """
        Get n random prompt/response samples for inspection.

        Args:
            n: Number of samples to return (default 3)
            seed: Random seed for reproducibility (default None)

        Returns:
            List of dictionaries containing sample data
        """
        if not self.prompt_responses:
            return []

        # Use numpy random for consistent behavior
        rng = np.random.default_rng(seed)

        # Sample without replacement (or all if n > total)
        n_samples = min(n, len(self.prompt_responses))
        indices = rng.choice(len(self.prompt_responses), size=n_samples, replace=False)

        samples = [self.prompt_responses[i] for i in indices]
        return samples

    def print_samples(self, samples: List[Dict]):
        """Pretty-print samples for inspection"""
        if not samples:
            print("\n⚠️ No samples available (verbose mode may be disabled)")
            return

        print(f"\n{'='*80}")
        print(f"RANDOM CODE ASSIGNMENT SAMPLES (n={len(samples)})")
        print(f"{'='*80}")

        for i, sample in enumerate(samples, 1):
            print(f"\n{'─'*80}")
            print(f"SAMPLE #{i}")
            print(f"{'─'*80}")
            print(f"Respondent ID: {sample['respondent_id']}")
            print(f"Idea ID: {sample['idea_id']}")
            print("\nIdea Text:")
            print(f"  {sample['idea_text']}")
            print(f"\nAssigned Codes: {', '.join(sample['assigned_codes'])}")
            print(f"Assigned Themes: {', '.join(sample['assigned_themes']) if sample['assigned_themes'] else 'None'}")
            print(f"Confidence: {sample['confidence']:.2f}")
            print("\nRationale:")
            print(f"  {sample['rationale']}")
            print(f"\n{'─'*40}")
            print("FULL PROMPT:")
            print(f"{'─'*40}")
            print(sample['prompt'])
            print(f"{'─'*80}\n")

    def print_assignment_stats(self):
        """Print detailed stats about default vs fallback usage"""
        total = self.used_default_count + self.used_fallback_count

        if total == 0:
            print("\n⚠️ No assignment stats available")
            return

        default_pct = (self.used_default_count / total) * 100
        fallback_pct = (self.used_fallback_count / total) * 100

        print(f"\n{'='*80}")
        print("CODE ASSIGNMENT STRATEGY BREAKDOWN")
        print(f"{'='*80}")
        print(f"Total ideas processed: {total}")
        print("")
        print(f"Used default (cluster code): {self.used_default_count} ({default_pct:.1f}%)")
        print(f"Used fallback (all codes):   {self.used_fallback_count} ({fallback_pct:.1f}%)")
        print("")
        print("API calls:")
        print(f"  Stage 1 (evaluate default): {self.stage_1_calls}")
        print(f"  Stage 2 (fallback):         {self.stage_2_calls}")
        print(f"  Total API calls:            {self.stage_1_calls + self.stage_2_calls}")
        print(f"  Avg calls per idea:         {(self.stage_1_calls + self.stage_2_calls) / total:.2f}")
        print(f"{'='*80}\n")

    def print_learning_insights(self):
        """Print diagnostic insights from pattern tracking and adaptive threshold."""
        if not self.pattern_config.enabled:
            return

        print(f"\n{'─'*60}")
        print("PATTERN LEARNING INSIGHTS")
        print(f"{'─'*60}")

        # Adaptive threshold stats
        if self.adaptive_threshold_config.use_adaptive:
            stats = self.confidence_tracker.get_stats()
            print(f"\n📊 Adaptive Threshold:")
            print(f"   Current threshold: {stats.get('current_threshold', 0.7):.3f}")
            print(f"   Samples collected: {stats.get('samples', 0)}")
            print(f"   Warmed up: {'Yes' if stats.get('is_warmed_up', False) else 'No'}")
            if stats.get('samples', 0) > 0:
                print(f"   Confidence range: [{stats.get('min', 0):.3f}, {stats.get('max', 0):.3f}]")
                print(f"   Mean confidence: {stats.get('mean', 0):.3f}")
        else:
            print(f"\n📊 Fixed Threshold: {self.adaptive_threshold_config.fixed_threshold}")

        # Dynamic top-k mode info
        print(f"\n🎯 Dynamic Top-K Mode: {self.dynamic_topk_config.mode}")
        if self.dynamic_topk_config.mode == "threshold":
            print(f"   Similarity threshold: {self.dynamic_topk_config.similarity_threshold}")
        elif self.dynamic_topk_config.mode == "dropoff":
            print(f"   Dropoff ratio: {self.dynamic_topk_config.dropoff_ratio}")

        # Problematic clusters (high fallback rates)
        if self.pattern_config.track_cluster_fallback:
            problematic = self.pattern_tracker.get_problematic_clusters(fallback_threshold=0.5, min_ideas=3)
            if problematic:
                print(f"\n⚠️  Clusters with high fallback rates (>50%):")
                for c in problematic[:5]:
                    print(f"   • Cluster {c['cluster_id'][:30]}: {c['fallback_rate']*100:.0f}% fallback ({c['total_ideas']} ideas)")
            else:
                print(f"\n✅ No clusters with problematic fallback rates")

        # Code co-occurrence patterns
        if self.pattern_config.track_cooccurrence:
            cooc = self.pattern_tracker.get_top_cooccurrences(top_n=5)
            if cooc:
                print(f"\n🔗 Top code co-occurrences (same respondent):")
                for c in cooc:
                    print(f"   • {c['code_a'][:25]} + {c['code_b'][:25]}: {c['count']}x")

        # Confidence calibration
        if self.pattern_config.track_confidence_calibration:
            cal = self.pattern_tracker.get_confidence_calibration()
            if cal:
                print(f"\n📈 Confidence distribution:")
                for bucket, data in sorted(cal.items()):
                    if data['count'] > 0:
                        print(f"   {bucket}: {data['count']} assignments")

        print(f"{'─'*60}\n")

    # === V3 THROUGHPUT ADJUSTMENT METHODS ========================================================================================================

    async def _apply_pid_adjustment(self) -> bool:
        """V3: Apply PID-style continuous throughput adjustment based on real-time TPM utilization.

        This provides smooth, gradual adjustments that converge to optimal throughput
        without the oscillations of threshold-based step changes.

        Returns True if adjustment was applied, False otherwise.
        """
        if self.current_arrival_rate is None:
            return False

        # Get real-time TPM utilization
        current_tpm = await self.tpm_tracker.get_current_tpm()
        tpm_limit = self.rate_limits.tokens_per_minute
        utilization = current_tpm / tpm_limit if tpm_limit > 0 else 0.0

        # Track min/max utilization for stats
        self.v3_stats['max_tpm_utilization'] = max(self.v3_stats['max_tpm_utilization'], utilization * 100)
        self.v3_stats['min_tpm_utilization'] = min(self.v3_stats['min_tpm_utilization'], utilization * 100)

        # Compute PID adjustment
        adjustment = self.pid_controller.compute_adjustment(utilization)

        # Skip if adjustment is negligible (1.0 = no change)
        if abs(adjustment - 1.0) < 0.01:
            return False

        # Apply adjustment to arrival rate
        old_rate = self.current_arrival_rate
        new_rate = old_rate * adjustment

        # Clamp to reasonable bounds
        rpm_max = self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        new_rate = max(0.5, min(rpm_max, new_rate))

        # Only apply if change is meaningful
        if abs(new_rate - old_rate) / old_rate < 0.02:
            return False

        # Update rate limiter
        if new_rate < 1:
            self.rate_limiter = AsyncLimiter(1, time_period=1/new_rate)
        else:
            self.rate_limiter = AsyncLimiter(int(new_rate), time_period=1.0)

        self.current_arrival_rate = new_rate
        self.v3_stats['pid_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1

        return True

    def _adjust_throughput_if_needed(self) -> bool:
        """V3: Threshold-based adjustment (fallback for large corrections).

        This is kept as a fallback for when the token estimate is significantly wrong
        and a larger step correction is needed before PID can fine-tune.

        Returns True if adjustment was made, False otherwise.
        """
        # Need enough samples to make a reliable decision
        if len(self.actual_total_tokens) < THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
            return False

        actual_avg = sum(self.actual_total_tokens) / len(self.actual_total_tokens)
        bootstrap_avg = self.avg_tokens

        # Calculate ratio of actual to bootstrap
        ratio = actual_avg / bootstrap_avg if bootstrap_avg > 0 else 1.0

        # V3: Only trigger threshold adjustment for significant underestimation
        # PID handles fine-tuning; this is for coarse correction
        if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
            return False

        # Calculate new arrival rate using actual tokens
        old_tpm_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / bootstrap_avg / 60
        new_tpm_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / actual_avg / 60
        rpm_throughput = self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60

        new_arrival_rate = min(rpm_throughput, new_tpm_throughput)

        # Reinstall rate limiter with adjusted rate
        if new_arrival_rate < 1:
            self.rate_limiter = AsyncLimiter(1, time_period=1/new_arrival_rate)
        else:
            self.rate_limiter = AsyncLimiter(int(new_arrival_rate), time_period=1.0)

        # V3: Track current arrival rate for PID
        self.current_arrival_rate = new_arrival_rate

        # Reinitialize token bucket with fresh state
        old_bucket_available = self.tpm_bucket.available
        self.tpm_bucket = TokenBucket(int(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom))

        # Reset PID controller (we just made a step change)
        self.pid_controller.reset()

        # Update avg_tokens for future calculations
        old_avg = self.avg_tokens
        self.avg_tokens = int(actual_avg)

        self.v3_stats['threshold_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1

        # Log the adjustment
        print(f"\n⚡ THROUGHPUT ADJUSTMENT (threshold)")
        print(f"   Actual tokens ({actual_avg:.0f}) exceeded bootstrap ({bootstrap_avg:.0f}) by {(ratio-1)*100:.0f}%")
        print(f"   Arrival rate: {old_tpm_throughput:.2f}/s → {new_arrival_rate:.2f}/s")
        print(f"   avg_tokens: {old_avg} → {self.avg_tokens}")
        print(f"   Token bucket reset (was {old_bucket_available:,.0f} available)")
        print(f"   Tiktoken offset: {self.tiktoken_offset_learner.get_offset()} (learned: {self.tiktoken_offset_learner.is_learned()})")

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Threshold adjustment: {old_tpm_throughput:.2f}/s → {new_arrival_rate:.2f}/s")

        return True

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[CodeAssignmentResponse]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []

        try:
            #print(f"[DEBUG] Starting process_all_tasks_async with {len(tasks)} tasks")

            self.verbose_reporter.step_start("Code Assignment")

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

            # Re-initialize TokenBucket with actual rate limits
            self.tpm_bucket = TokenBucket(limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

            # Bootstrap measurement with probe calls (following qualityFilter.py pattern)
            sample_tasks = tasks[:min(3, len(tasks))]
            if len(sample_tasks) < 3:
                # Duplicate tasks if we have fewer than 3
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
            for i in range(3):  # Add 3 samples to get started
                self.latency_tracker.add(avg_latency_s)
            
            # Update avg_tokens with bootstrap measurement
            self.avg_tokens = int(avg_tokens)
        
            # Calculate optimal concurrency using Little's Law
            api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
            Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, processing_config=self.processing_config, cap=self.processing_config.concurrency_cap_permissive, min_conc=self.processing_config.concurrency_min_permissive)
            # Use ProcessingConfig for bounds instead of hardcoded constants
            min_concurrency = self.processing_config.concurrency_min_default
            max_concurrency = self.processing_config.concurrency_cap_default
            optimal = min(max_concurrency, max(Little, min_concurrency))

            # Initialize rate limiting components
            arrival_rate = min(
                limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
                limits.tokens_per_minute * self.processing_config.rate_limit_headroom / avg_tokens / 60
                )

            if arrival_rate < 1:
                self.rate_limiter = AsyncLimiter(1, time_period=1/arrival_rate)
            else:
                self.rate_limiter = AsyncLimiter(int(arrival_rate), time_period=1.0)

            self.semaphore = asyncio.Semaphore(min(len(tasks), optimal))
            self.optimal_concurrency = min(len(tasks), optimal)

            # V3: Track current arrival rate for PID adjustment
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
            print(f"- Optimal by Little's law: {Little}")
            print(f"- Constrained optimum: {optimal} (min={min_concurrency}, max={max_concurrency})")

            print(f"- Processing {len(tasks):,} tasks")

            # Calculate number of workers using ProcessingConfig bounds
            expected_throughput = min(rpm_throughput, tpm_throughput)
            max_workers = self.processing_config.max_workers if hasattr(self.processing_config, 'max_workers') else 200
            min_workers = self.processing_config.min_workers if hasattr(self.processing_config, 'min_workers') else 50
            num_workers = min(max_workers, max(min_workers, int(expected_throughput * avg_latency_s * 2.0)))
            
            print(f"- Workers launched: (concurrent subroutines): {num_workers}")
            print(f"- API calls in flight (concurrency ceiling/semaphore): {self.optimal_concurrency}")
            
            # Create queue and results list
            queue = asyncio.Queue()
            results = [None] * len(tasks)
            
            # Add tasks to queue with result indices
            for i, task in enumerate(tasks):
                task['result_index'] = i
                task['task_index'] = i
                task['task_id'] = task['idea_data'][1]  # idea_id
                await queue.put(task)
        
            # Start workers
            workers = []
            #print(f"[DEBUG] Starting {num_workers} workers...")
            for i in range(num_workers):
                w = asyncio.create_task(self.worker(queue, results))
                workers.append(w)
            #print(f"[DEBUG] All {len(workers)} workers started")
        
            # Progress monitoring
            start_time = time.time()
            last_report = start_time
            last_adjustment = start_time  # V3: Track last adjustment time

            #print(f"[DEBUG] Starting progress monitoring, queue size: {queue.qsize()}")

            # Monitor progress until all tasks are processed
            while self.stats['tasks_processed'] < len(tasks):
                await asyncio.sleep(1)
                now = time.time()

                # Regular progress report every 5s
                if now - last_report >= PROGRESS_REPORT_INTERVAL:
                    completed = self.stats['tasks_processed']
                    remaining = queue.qsize()
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0

                    print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%), "
                      f"Rate: {rate:.1f}/s, Queue: {remaining}")
                    last_report = now

                # V3: Apply PID adjustment periodically
                if now - last_adjustment >= ADJUSTMENT_INTERVAL:
                    await self._apply_pid_adjustment()
                    last_adjustment = now

                    # V3: Also check threshold-based adjustment (fallback for large corrections)
                    if self.stats['tasks_processed'] >= THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
                        self._adjust_throughput_if_needed()

                # Check if queue is empty but not all tasks processed (potential deadlock)
                if queue.empty() and self.stats['tasks_processed'] < len(tasks):
                    #print(f"[DEBUG] Queue empty but only {self.stats['tasks_processed']}/{len(tasks)} processed")
                    break
        
            #print("[DEBUG] Progress monitoring complete, waiting for queue.join()")
        
            # Wait for all tasks to complete
            await queue.join()
        
            #print("[DEBUG] Queue.join() complete")
        
            # Stop workers
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)
        
            # Final stats
            elapsed = time.time() - start_time
            print(f"\nCompleted {len(tasks)} tasks in {elapsed:.1f}s")
            print(f"- Successful: {self.stats['tasks_successful']}")
            print(f"- Failed: {self.stats['tasks_failed']}")
            print(f"- Rate limits: {self.stats['rate_limits']}")
            print(f"- Timeouts: {self.stats['timeouts']}")
            print(f"- Average: {elapsed/len(tasks):.2f}s/task")

            # Report error types if any failures occurred
            if self.stats['error_types']:
                print(f"\nError Types ({len(self.stats['error_types'])} unique):")
                for error_type, error_data in sorted(self.stats['error_types'].items(),
                                                     key=lambda x: x[1]['count'], reverse=True):
                    print(f"  - {error_type}: {error_data['count']} occurrences")
                    if error_data['sample_messages']:
                        print("    Sample errors:")
                        for i, msg in enumerate(error_data['sample_messages'], 1):
                            print(f"      {i}. {msg}")

            # V3: Report V3 stats
            if self.v3_stats['adjustments_made'] > 0 or self.tiktoken_offset_learner.is_learned():
                print(f"\nV3 Rate Limiting Stats:")
                print(f"- PID adjustments: {self.v3_stats['pid_adjustments']}")
                print(f"- Threshold adjustments: {self.v3_stats['threshold_adjustments']}")
                if self.v3_stats['max_tpm_utilization'] > 0:
                    print(f"- TPM utilization range: {self.v3_stats['min_tpm_utilization']:.1f}% - {self.v3_stats['max_tpm_utilization']:.1f}%")
                offset_stats = self.tiktoken_offset_learner.get_stats()
                print(f"- Tiktoken offset: {offset_stats['using_offset']} (learned: {offset_stats['is_learned']}, samples: {offset_stats['samples']})")

            return results
        
        except Exception as e:
            logger.error(f"[CRITICAL ERROR] process_all_tasks_async failed: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            print(f"\n❌ CODE ASSIGNMENT FAILED: {type(e).__name__}: {e}")
            print("Returning fallback responses for all tasks...\n")
            fallback_results = []
            for task in tasks:
                fallback_results.append(self.create_fallback_response(task))
            return fallback_results

    def _prepare_individual_tasks(self, all_ideas: List[tuple]) -> List[Dict]:
        """Prepare individual tasks for processing"""
        tasks = []
        for i, idea_data in enumerate(all_ideas):
            tasks.append({
                'idea_data': idea_data
            })
        return tasks
    
    async def assign_codes(self) -> List[models.CodeAssignedModel]:
        """Main method to assign codes using standardized processing patterns"""
        self._stats.start_timing()
        
        # Extract all ideas
        all_ideas = self._extract_all_ideas()
        total_ideas = len(all_ideas)
        
        if total_ideas == 0:
            self.verbose_reporter.stat_line("No ideas found for code assignment")
            return []
        
        # Use fallback rate limits for initial display (actual limits fetched during processing)
        self.verbose_reporter.stat_line(f"Model: {self.model} (Initial limits: {self.rate_limits.requests_per_minute} RPM, {self.rate_limits.tokens_per_minute:,} TPM)")
        self.verbose_reporter.stat_line(f"Processing {total_ideas} ideas with {len(self.codebook)} available codes")
        
        # Prepare tasks
        tasks = self._prepare_individual_tasks(all_ideas)
        
        # Process with queue + workers pattern
        if nest_asyncio:
            nest_asyncio.apply()
        all_results = await self.process_all_tasks_async(tasks)
        
        # Merge results back into model structure
        self._results = self._merge_results_into_models(all_results)
        
        # Report summary
        if all_results:
            valid_results = [r for r in all_results if r is not None]
            if valid_results:
                avg_confidence = np.mean([r.assignment_confidence for r in valid_results])
                high_confidence = sum(1 for r in valid_results if r.assignment_confidence >= 0.7)
                low_confidence = sum(1 for r in valid_results if r.assignment_confidence < 0.5)
                
                self.verbose_reporter.summary("CODE ASSIGNMENT COMPLETED", {
                    "Total ideas processed": len(valid_results),
                    "Average confidence": f"{avg_confidence:.2f}",
                    "High confidence (≥0.7)": high_confidence,
                    "Low confidence (<0.5)": low_confidence
                })

        # Print learning insights (assignment stats printed by pipeline.py)
        self.print_learning_insights()

        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        if nest_asyncio:
            nest_asyncio.apply()
        
        return asyncio.run(self.assign_codes())
