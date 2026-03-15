import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import logging
import itertools
import difflib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
from pydantic import BaseModel, field_validator
from development import models_exp as models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, MISCELLANEOUS_CODE_LABELS, API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM
from utils.llm import create_client, llm_create_async, create_embedding_client, ProbeResponse, RateLimits, extract_rate_limits_from_response
from development.step_8_codeAssigner.config_exp import SimilarityRoutingConfig

# === PROMPTS (partition-based, ladder) ============================================================================
from development.step_8_codeAssigner.prompts_exp import (
    SINGLE_CODE_EVALUATION_PROMPT,
    PARTITION_EVALUATION_PROMPT,
    SIMILARITY_EVALUATION_PROMPT,
)

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cached_resources import get_tiktoken_encoding

try:
    import nest_asyncio  # for Spyder
    nest_asyncio.apply()
except ImportError:
    pass

# === STEP-SPECIFIC CONFIG =============================================================================================
from config_steps.config_codeAssigner import (
    DEFAULT_TOKEN_HISTORY_CONFIG,
    DEFAULT_TIKTOKEN_OFFSET_CONFIG,
    DEFAULT_TIMEOUT_CONFIG,
    DEFAULT_REPORTING_CONFIG,
    DEFAULT_BOOTSTRAP_CONFIG,
    DEFAULT_PID_CONTROLLER_CONFIG,
    DEFAULT_TPM_TRACKING_CONFIG,
    DEFAULT_THROUGHPUT_CONFIG,
    DEFAULT_ADAPTIVE_THRESHOLD_CONFIG,
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
        """Wait if necessary and acquire tokens, with safeguard against infinite loops."""
        attempts = 0
        while attempts < MAX_TOKEN_ACQUIRE_ATTEMPTS:
            attempts += 1
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            else:
                await asyncio.sleep(result)
        raise RuntimeError(
            f"Failed to acquire {tokens_needed} tokens after {MAX_TOKEN_ACQUIRE_ATTEMPTS} attempts"
        )

    async def reconcile(self, delta_tokens):
        if delta_tokens < 0:
            async with self.lock:
                self.available = min(self.tpm, self.available - delta_tokens)


class LatencyTracker:
    """Simple EMA tracker for latencies"""
    def __init__(self, processing_config: Optional[ProcessingConfig] = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.ema = None
        self.alpha = self.processing_config.latency_tracker_ema_alpha
        self.values = deque(maxlen=self.processing_config.latency_tracker_samples_window)

    def add(self, value):
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

    def get_timeout(self, est_tokens):
        config = self.processing_config
        if not self.values:
            return max(config.adaptive_timeout_min_seconds, 30.0)
        p95 = np.percentile(list(self.values), 95)
        token_factor = est_tokens / 1000
        timeout = p95 + (token_factor * 0.1)
        return max(config.adaptive_timeout_min_seconds, min(config.adaptive_timeout_max_seconds, timeout * config.adaptive_timeout_margin))

    def get_avg_latency(self):
        if not self.values:
            return 2.0
        return self.ema if self.ema is not None else 2.0


# === V3 OPTIMAL STRATEGY CLASSES ========================================================================================================

class TiktokenOffsetLearner:
    """Learns the offset between tiktoken counts and actual API token counts."""
    def __init__(self, default_offset: int = 300, history_maxlen: int = 30, min_samples: int = 5):
        self.default_offset = default_offset
        self.offsets = deque(maxlen=history_maxlen)
        self.min_samples = min_samples
        self._learned_offset = None

    def record(self, tiktoken_count: int, api_count: int):
        offset = api_count - tiktoken_count
        self.offsets.append(offset)
        if len(self.offsets) >= self.min_samples:
            self._learned_offset = int(sum(self.offsets) / len(self.offsets))

    def get_offset(self) -> int:
        if self._learned_offset is not None:
            return self._learned_offset
        return self.default_offset

    def is_learned(self) -> bool:
        return len(self.offsets) >= self.min_samples

    def get_stats(self) -> dict:
        return {
            "samples": len(self.offsets),
            "learned_offset": self._learned_offset,
            "using_offset": self.get_offset(),
            "is_learned": self.is_learned(),
        }


class RealTimeTPMTracker:
    """Tracks actual TPM usage in a sliding window."""
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
            elapsed = max(now - self.samples[0][0], 1.0) if self.samples else 1.0
            return (total_tokens / elapsed) * 60

    async def get_utilization(self, tpm_limit: int) -> float:
        current_tpm = await self.get_current_tpm()
        return (current_tpm / tpm_limit) * 100 if tpm_limit > 0 else 0.0


class PIDThroughputController:
    """PID controller for smooth throughput adjustment."""
    def __init__(self, target_utilization=0.85, kp_up=0.4, kp_down=0.2, ki=0.05, kd=0.1, min_adjustment=0.02, max_adjustment=0.15):
        self.target = target_utilization
        self.kp_up, self.kp_down = kp_up, kp_down
        self.ki, self.kd = ki, kd
        self.min_adjustment, self.max_adjustment = min_adjustment, max_adjustment
        self.integral, self.last_error, self.last_time = 0.0, 0.0, None
        self.adjustment_history = deque(maxlen=20)

    def compute_adjustment(self, current_utilization: float) -> float:
        now = time.monotonic()
        error = self.target - current_utilization
        dt = max(now - self.last_time, 0.1) if self.last_time is not None else 1.0
        self.last_time = now
        self.integral = max(-0.5, min(0.5, self.integral + error * dt))
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        kp = self.kp_up if error > 0 else self.kp_down
        output = max(-self.max_adjustment, min(self.max_adjustment, (kp * error) + (self.ki * self.integral) + (self.kd * derivative)))
        adjustment = 1.0 if abs(output) < self.min_adjustment else 1.0 + output
        self.adjustment_history.append({"time": now, "utilization": current_utilization, "adjustment": adjustment})
        return adjustment

    def reset(self):
        self.integral, self.last_error, self.last_time = 0.0, 0.0, None


# === ADAPTIVE CONFIDENCE THRESHOLD ========================================================================================================

class ConfidenceTracker:
    """Tracks running confidence distribution for adaptive thresholding."""
    def __init__(self, percentile=25, floor=0.5, warmup=20, history_maxlen=500):
        self.percentile, self.floor, self.warmup = percentile, floor, warmup
        self.confidences = deque(maxlen=history_maxlen)

    def record(self, confidence: float):
        self.confidences.append(confidence)

    def get_adaptive_threshold(self, fixed_threshold: float) -> float:
        if len(self.confidences) < self.warmup:
            return fixed_threshold
        return max(self.floor, np.percentile(list(self.confidences), self.percentile))

    def is_warmed_up(self) -> bool:
        return len(self.confidences) >= self.warmup

    def get_stats(self) -> dict:
        if not self.confidences:
            return {"samples": 0}
        conf_list = list(self.confidences)
        return {
            "samples": len(conf_list), "mean": float(np.mean(conf_list)),
            "median": float(np.median(conf_list)),
            "current_threshold": self.get_adaptive_threshold(0.7),
            "is_warmed_up": self.is_warmed_up()
        }


# === PATTERN TRACKING ========================================================================================================

@dataclass
class PartitionDiagnostics:
    """Diagnostics for a single concept_type partition."""
    partition_name: str
    total_ideas: int = 0
    matched: int = 0
    unknown: int = 0
    confidences: List[float] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.matched / self.total_ideas if self.total_ideas else 0.0

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0


class PatternTracker:
    """Tracks patterns for learning and diagnostics."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.code_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self.partition_diagnostics: Dict[str, PartitionDiagnostics] = {}
        self.confidence_buckets = {
            "0.5-0.6": {"count": 0, "sum_confidence": 0.0},
            "0.6-0.7": {"count": 0, "sum_confidence": 0.0},
            "0.7-0.8": {"count": 0, "sum_confidence": 0.0},
            "0.8-0.9": {"count": 0, "sum_confidence": 0.0},
            "0.9-1.0": {"count": 0, "sum_confidence": 0.0},
        }
        self._respondent_codes: Dict[str, List[str]] = defaultdict(list)

    def record_assignment(self, respondent_id, partition_name, assigned_code, confidence, matched):
        if not self.enabled:
            return
        if partition_name:
            if partition_name not in self.partition_diagnostics:
                self.partition_diagnostics[partition_name] = PartitionDiagnostics(partition_name=partition_name)
            diag = self.partition_diagnostics[partition_name]
            diag.total_ideas += 1
            diag.confidences.append(confidence)
            if matched:
                diag.matched += 1
            else:
                diag.unknown += 1
        bucket = self._get_confidence_bucket(confidence)
        if bucket:
            self.confidence_buckets[bucket]["count"] += 1
            self.confidence_buckets[bucket]["sum_confidence"] += confidence
        self._respondent_codes[respondent_id].append(assigned_code)

    def _get_confidence_bucket(self, confidence):
        if 0.5 <= confidence < 0.6: return "0.5-0.6"
        elif 0.6 <= confidence < 0.7: return "0.6-0.7"
        elif 0.7 <= confidence < 0.8: return "0.7-0.8"
        elif 0.8 <= confidence < 0.9: return "0.8-0.9"
        elif 0.9 <= confidence <= 1.0: return "0.9-1.0"
        return None

    def get_problematic_partitions(self, unknown_threshold=0.5, min_ideas=3):
        problematic = []
        for name, diag in self.partition_diagnostics.items():
            unknown_rate = 1.0 - diag.match_rate
            if diag.total_ideas >= min_ideas and unknown_rate >= unknown_threshold:
                problematic.append({"partition": name, "total_ideas": diag.total_ideas, "unknown_rate": unknown_rate, "avg_confidence": diag.avg_confidence})
        return sorted(problematic, key=lambda x: x["unknown_rate"], reverse=True)

    def get_confidence_calibration(self):
        return {bucket: {"count": d["count"], "avg_confidence": d["sum_confidence"] / d["count"]} for bucket, d in self.confidence_buckets.items() if d["count"] > 0}


# === BOOTSTRAP ========================================================================================================

@dataclass
class ApiLimits:
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(limits, latency_seconds, avg_tokens, processing_config=None, cap=None, min_conc=None, headroom=None):
    config = processing_config or DEFAULT_PROCESSING_CONFIG
    cap = cap if cap is not None else config.concurrency_cap_default
    min_conc = min_conc if min_conc is not None else config.concurrency_min_default
    headroom = headroom if headroom is not None else config.rate_limit_headroom
    latency_seconds = max(float(latency_seconds or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)
    rpm_throughput = limits.requests_per_minute * headroom / 60
    tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
    allowed_rps = max(min(rpm_throughput, tpm_throughput), 0.0)
    return int(max(min(allowed_rps * latency_seconds, cap), min_conc))


async def bootstrap_measure_async(call_fn, n_probes=3):
    latencies, tokens = [], []
    for _ in range(n_probes):
        t0 = time.perf_counter()
        usage = await call_fn()
        t1 = time.perf_counter()
        latencies.append(max(t1 - t0, 0.001))
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        tokens.append(max(pt + ct, 1))
    return sum(latencies) / len(latencies), sum(tokens) / len(tokens)


# === PYDANTIC MODELS ========================================================================================================

class CodeAssignmentResponse(BaseModel):
    idea_id: str
    idea: str
    assigned_codes: List[str]
    assignment_confidence: float
    assignment_rationale: str
    assigned_themes: Optional[List[str]] = None

class DefaultCodeEvaluationResponse(BaseModel):
    """Single code evaluation response"""
    idea_id: str
    confidence: float
    rationale: str

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        return float(v) if isinstance(v, str) else v


class SimpleClassificationResponse(BaseModel):
    """Simplified classification response — pick one code from candidates."""
    idea_id: str
    code: str
    confidence: float
    rationale: str

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        return float(v) if isinstance(v, str) else v


class CodeEvaluation(BaseModel):
    """Single code evaluation within multi-candidate evaluation"""
    code: str
    confidence: float
    rationale: str

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        return float(v) if isinstance(v, str) else v


class PartitionEvaluationResponse(BaseModel):
    """Multi-code evaluation from partition"""
    idea_id: str
    evaluations: List[CodeEvaluation]
    best_match: CodeEvaluation

    @field_validator('evaluations', mode='before')
    @classmethod
    def ensure_list(cls, v):
        return [v] if not isinstance(v, list) else v


# === MAIN CODE ASSIGNER CLASS ========================================================================================================

class CodeAssigner:
    """
    Partition-based code assignment using concept_type partitions and abstraction ladder.
    No embeddings, no fallback paths — LLM evaluates all codes in each partition directly.
    """

    def __init__(
        self,
        response_models: List[models.EmbeddingsModel],
        codebook: List[models.CodebookExp],
        var_lab: str,
        partition_remap: Optional[Dict[str, str]] = None,
        code_to_theme_mapping: Optional[Dict[str, str]] = None,
        dominance_axes: Optional[Dict[str, str]] = None,
        other_idea_assignments: Optional[Dict[str, str]] = None,
        similarity_config: Optional[SimilarityRoutingConfig] = None,
        config: Optional[CodeAssignmentConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        adaptive_threshold_config=None,
        pattern_config=None,
        verbose: bool = False,
        prompt_printer=None):

        self.response_models = response_models
        self.codebook = codebook
        self.var_lab = var_lab
        self.other_idea_assignments = other_idea_assignments or {}
        self.similarity_config = similarity_config or SimilarityRoutingConfig(routing_mode="partition")
        self.config = config or DEFAULT_CODE_ASSIGNMENT_CONFIG
        self.model_config = model_config or ModelConfig()
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.model_config.get_model_for_stage('code_assignment')
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.CodeAssignedModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        self._captured_prompt = False

        # Theme mapping
        self.code_to_theme_mapping = code_to_theme_mapping or {}

        # Dominance axes for partition-level routing dimensions
        self.dominance_axes = dominance_axes or {}

        # Tokenizer (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Instructor-patched async client
        self.client = create_client(model=self.model, async_mode=True)

        # Rate limiting setup
        self.rate_limits = RateLimits(
            tokens_per_minute=FALLBACK_TPM,
            requests_per_minute=FALLBACK_RPM,
            tokens_per_day=FALLBACK_TPM * 60 * 24
        )
        self.tpm_bucket = TokenBucket(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

        # Progressive token estimation
        self.input_token_history = deque(maxlen=INPUT_HISTORY_MAXLEN)
        self.output_token_history = deque(maxlen=OUTPUT_HISTORY_MAXLEN)
        self.output_ratio_history = deque(maxlen=OUTPUT_RATIO_HISTORY_MAXLEN)
        self.estimation_errors = deque(maxlen=ERROR_WINDOW_SIZE)
        self.actual_total_tokens = deque(maxlen=ERROR_WINDOW_SIZE)

        # Latency tracking
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)

        # V3 components
        self.tiktoken_offset_learner = TiktokenOffsetLearner(
            default_offset=TIKTOKEN_API_OFFSET_DEFAULT,
            history_maxlen=TIKTOKEN_OFFSET_HISTORY_MAXLEN,
            min_samples=TIKTOKEN_OFFSET_MIN_SAMPLES
        )
        self.tpm_tracker = RealTimeTPMTracker(window_seconds=TPM_SLIDING_WINDOW_SECONDS)
        self.pid_controller = PIDThroughputController(
            target_utilization=TPM_TARGET_UTILIZATION,
            kp_up=PID_KP_UP, kp_down=PID_KP_DOWN,
            ki=PID_KI, kd=PID_KD,
            min_adjustment=PID_MIN_ADJUSTMENT, max_adjustment=PID_MAX_ADJUSTMENT
        )
        self.current_arrival_rate = None
        self.v3_stats = {
            'adjustments_made': 0, 'pid_adjustments': 0,
            'threshold_adjustments': 0, 'max_tpm_utilization': 0.0,
            'min_tpm_utilization': 100.0,
        }

        # Configs
        self.adaptive_threshold_config = adaptive_threshold_config or DEFAULT_ADAPTIVE_THRESHOLD_CONFIG
        self.pattern_config = pattern_config or DEFAULT_PATTERN_TRACKING_CONFIG

        # Confidence tracker
        self.confidence_tracker = ConfidenceTracker(
            percentile=self.adaptive_threshold_config.adaptive_percentile,
            floor=self.adaptive_threshold_config.adaptive_floor,
            warmup=self.adaptive_threshold_config.warmup_samples
        )

        # Pattern tracker
        self.pattern_tracker = PatternTracker(enabled=self.pattern_config.enabled)

        # Rate limiting components (initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None

        # Stats
        self._stats = ProcessingStats()
        self.stats = {
            'tasks_processed': 0, 'tasks_successful': 0,
            'tasks_failed': 0, 'retries': 0, 'rate_limits': 0,
            'timeouts': 0, 'error_types': {}
        }

        # Failure tracking and retry queue
        self.failure_log = []                          # per-task audit trail
        self._rate_limit_retry_queue: List[Dict] = []  # 429-failed tasks for retry pass

        # Partition-based assignment stats
        self.partition_eval_calls = 0
        self.partition_match_count = 0
        self.unknown_count = 0

        # Prompt/Response logging
        self.prompt_responses = []
        self.last_prompt = ""
        self.verbose = verbose

        # === PARTITION ROUTING ===
        # partition_remap: refined_name → original_name (from step 7 splits/renames)
        self.partition_remap = partition_remap or {}
        # reverse_remap: original_name → [refined_name_1, refined_name_2, ...]
        self.reverse_remap: Dict[str, List[str]] = defaultdict(list)
        for refined_name, original_name in self.partition_remap.items():
            self.reverse_remap[original_name].append(refined_name)

        # Build partition codebooks using REFINED concept_types (no collapse)
        self.partition_codebooks = self._partition_codebook_by_concept_type()
        self.partition_stats: Dict[str, Dict] = {}

        if self.reverse_remap and self.verbose:
            self.verbose_reporter.stat_line(f"Reverse remap: {len(self.reverse_remap)} original partitions → {sum(len(v) for v in self.reverse_remap.values())} refined sub-partitions")
            for orig, refined_list in sorted(self.reverse_remap.items()):
                self.verbose_reporter.stat_line(f"  '{orig}' → {refined_list}")

        self.verbose_reporter.stat_line(f"Model: {self.model}")
        self.verbose_reporter.stat_line(f"API Limits: {self.rate_limits.requests_per_minute} RPM, {self.rate_limits.tokens_per_minute:,} TPM")

        # === SIMILARITY ROUTING (initialized lazily in assign_codes) ===
        self._code_embeddings = None          # np.ndarray [n_codes, dims]
        self._idea_embeddings_lookup = {}     # {idea_id: np.ndarray}
        self._similarity_stats = {'codes_embedded': 0, 'embedding_time': 0.0}

        if self.similarity_config.routing_mode != "partition":
            self.verbose_reporter.stat_line(f"Routing mode: {self.similarity_config.routing_mode} (top_k={self.similarity_config.top_k})")
        else:
            self.verbose_reporter.stat_line(f"Routing mode: partition (classic)")

    # === PARTITION ROUTING ========================================================================================================

    def _partition_codebook_by_concept_type(self) -> Dict[str, List[models.CodebookExp]]:
        """Group codebook entries by concept_type."""
        partitions: Dict[str, List[models.CodebookExp]] = defaultdict(list)
        for code in self.codebook:
            concept_type = getattr(code, 'concept_type', None) or '_unpartitioned'
            partitions[concept_type].append(code)

        partition_dict = dict(partitions)
        if self.verbose:
            self.verbose_reporter.stat_line(f"Partition codebook: {len(partition_dict)} partitions")
            for name, codes in sorted(partition_dict.items()):
                self.verbose_reporter.stat_line(f"  {name}: {len(codes)} codes")

        return partition_dict

    def _get_codes_for_idea(self, concept_type: str) -> Tuple[List[models.CodebookExp], Dict[str, List[models.CodebookExp]]]:
        """Get codes for an idea, handling refined partition routing.

        Returns:
            (flat_codes, grouped_codes)
            - flat_codes: all matching codes in a flat list
            - grouped_codes: {refined_partition_name: [codes]} for prompt grouping.
              Empty dict if direct match (no sub-partitions).
        """
        # Direct match: concept_type exists as-is in partition_codebooks
        # (unsplit partition where name didn't change)
        if concept_type in self.partition_codebooks:
            codes = self.partition_codebooks[concept_type]
            return codes, {}

        # Reverse remap: concept_type was split/renamed into refined sub-partitions
        if concept_type in self.reverse_remap:
            refined_names = self.reverse_remap[concept_type]
            grouped = {}
            all_codes = []
            for rname in sorted(refined_names):
                sub_codes = self.partition_codebooks.get(rname, [])
                if sub_codes:
                    grouped[rname] = sub_codes
                    all_codes.extend(sub_codes)
            return all_codes, grouped

        # No match at all
        return [], {}

    def _format_grouped_partition_codes(self, grouped_codes: Dict[str, List[models.CodebookExp]]) -> str:
        """Format codes grouped by refined sub-partition for the LLM prompt."""
        sections = []
        for partition_name, codes in grouped_codes.items():
            header = f"=== Sub-partition: {partition_name} ({len(codes)} codes) ==="
            code_blocks = self._format_partition_codes(codes)
            sections.append(f"{header}\n{code_blocks}")
        return "\n\n".join(sections)

    # === SEMANTIC SIMILARITY ROUTING ========================================================================================================

    def _build_code_embedding_text(self, code: models.CodebookExp) -> str:
        """Compose text for embedding a code entry."""
        if self.similarity_config.code_embedding_text == "simple":
            return f"{code.code}: {code.definition}"
        # "rich": include boundary test and diagnostic signals for discriminative power
        boundary = getattr(code, 'boundary_test', '') or ''
        signals = getattr(code, 'diagnostic_signals', None) or []
        signals_str = ", ".join(signals) if signals else ""
        parts = [f"{code.code}: {code.definition}"]
        if boundary:
            parts.append(f"Boundary: {boundary}")
        if signals_str:
            parts.append(f"Signals: {signals_str}")
        return ". ".join(parts)

    def _generate_code_embeddings(self) -> np.ndarray:
        """Generate embeddings for all codebook entries. Single batch API call."""
        embedding_client = create_embedding_client(async_mode=False)
        embedding_model = self.model_config.embedding_model

        code_texts = [self._build_code_embedding_text(code) for code in self.codebook]

        t0 = time.perf_counter()
        embeddings = []
        for text in code_texts:
            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=text
            )
            embeddings.append(response.data[0].embedding)

        elapsed = time.perf_counter() - t0
        result = np.array(embeddings)
        self._similarity_stats['codes_embedded'] = len(code_texts)
        self._similarity_stats['embedding_time'] = elapsed
        self.verbose_reporter.stat_line(
            f"Code embeddings: {len(code_texts)} codes embedded in {elapsed:.1f}s "
            f"(dims={result.shape[1]})"
        )
        return result

    def _build_idea_embeddings_lookup(self):
        """Build a lookup dict mapping idea_id → embedding vector from response models."""
        field = self.similarity_config.idea_embedding_field + "_embedding"
        count = 0
        for model in self.response_models:
            if not hasattr(model, 'response_ideas') or not model.response_ideas:
                continue
            for idea in model.response_ideas:
                emb = getattr(idea, field, None)
                if emb is None:
                    emb = getattr(idea, 'idea_embedding', None)
                if emb is not None:
                    self._idea_embeddings_lookup[idea.idea_id] = emb
                    count += 1
        self.verbose_reporter.stat_line(f"Idea embeddings loaded: {count} ideas with '{field}'")

    def _find_similar_codes(self, idea_embedding: np.ndarray, concept_type: str = None) -> List[models.CodebookExp]:
        """Find top-K similar codes using cosine similarity.

        Supports dropoff mode: include codes within dropoff_ratio of best similarity,
        clamped between min_codes and max_codes.
        """
        cfg = self.similarity_config
        similarities = cosine_similarity([idea_embedding], self._code_embeddings)[0]

        # Hybrid mode: boost same-partition codes
        if cfg.routing_mode == "hybrid" and concept_type:
            for i, code in enumerate(self.codebook):
                code_ct = getattr(code, 'concept_type', None) or ''
                # Direct match or reverse-mapped match
                if code_ct == concept_type or concept_type in self.reverse_remap and code_ct in self.reverse_remap[concept_type]:
                    similarities[i] += cfg.partition_boost

        sorted_indices = np.argsort(similarities)[::-1]
        sorted_sims = similarities[sorted_indices]

        # Apply dropoff from best
        best_sim = sorted_sims[0] if len(sorted_sims) > 0 else 0.0
        cutoff = best_sim * cfg.dropoff_ratio
        cutoff = max(cutoff, cfg.similarity_floor)

        count = sum(1 for s in sorted_sims if s >= cutoff)
        count = max(count, cfg.min_codes)
        count = min(count, cfg.max_codes, len(self.codebook))

        selected = sorted_indices[:count]

        # Store for diagnostics
        self._last_similarity_scores = {
            self.codebook[i].code: float(similarities[i])
            for i in selected
        }

        return [self.codebook[i] for i in selected]

    def _format_similarity_context(self, codes: List[models.CodebookExp]) -> str:
        """Format similarity ranking scores for inclusion in the LLM prompt."""
        if not hasattr(self, '_last_similarity_scores') or not self._last_similarity_scores:
            return ""
        lines = []
        for code in codes:
            score = self._last_similarity_scores.get(code.code, 0.0)
            lines.append(f"  {code.code}: {score:.3f}")
        return "\n".join(lines)

    # === TOKEN ESTIMATION & HELPERS ========================================================================================================

    def _calculate_avg_tokens(self) -> int:
        """Calculate average token count from sample prompts."""
        if not self.response_models:
            return 1500

        sample_size = min(10, len(self.response_models))
        total_tokens = 0
        sample_count = 0

        for i in range(sample_size):
            model = self.response_models[i]
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea in model.response_ideas:
                    if hasattr(idea, 'idea') and idea.idea:
                        prompt = self._create_probe_prompt(
                            idea.idea_id, idea.idea,
                            getattr(idea, 'instance', '') or '',
                            getattr(idea, 'rung_1', '') or '',
                            getattr(idea, 'concept_type', '') or '',
                        )
                        total_tokens += len(self.encoding.encode(prompt))
                        sample_count += 1
                        break
        if sample_count == 0:
            return 1500
        return int((total_tokens / sample_count) * 1.15)

    def _create_probe_prompt(self, idea_id, idea_text, instance, rung_1, concept_type) -> str:
        """Create a representative prompt for bootstrap probe calls."""
        first_partition = next(iter(self.partition_codebooks), None)
        if not first_partition:
            return f"Evaluate idea: {idea_text}"

        codes = self.partition_codebooks[first_partition]
        if len(codes) == 1:
            code = codes[0]
            return SINGLE_CODE_EVALUATION_PROMPT.format(
                language=self.language, var_lab=self.var_lab,
                idea_id=idea_id,
                instance=instance or idea_text,
                rung_1=rung_1 or idea_text,
                concept_type=concept_type,
                code=code.code, definition=code.definition,
                boundary_test=getattr(code, 'boundary_test', 'N/A') or 'N/A',
                diagnostic_signals=self._format_diagnostic_signals(code),
                inclusion_examples=self._format_examples_list(code.inclusion_examples),
                exclusion_examples=self._format_examples_list(code.exclusion_examples),
                near_neighbor_label=code.near_neighbor_label or "Unknown",
                tell_apart_rule=code.tell_apart_rule or "N/A"
            )
        else:
            return PARTITION_EVALUATION_PROMPT.format(
                language=self.language, var_lab=self.var_lab,
                idea_id=idea_id,
                instance=instance or idea_text,
                rung_1=rung_1 or idea_text,
                concept_type=concept_type,
                dominance_axis_block="",
                partition_codes_formatted=self._format_partition_codes(codes[:3])
            )

    def _format_examples_list(self, examples: Optional[List[str]]) -> str:
        if not examples:
            return "No specific examples provided"
        return "\n".join([f"  - {ex}" for ex in examples])

    def _format_diagnostic_signals(self, code: models.CodebookExp) -> str:
        signals = getattr(code, 'diagnostic_signals', None)
        return ", ".join(signals) if signals else "(none)"

    _CATCHALL_MARKERS = {"overig/anders", "other/miscellaneous", "sonstiges", "autre", "otro"}

    @staticmethod
    def _is_catchall_code(code_label: str) -> bool:
        """Detect partition-level catch-all codes (overig/anders, other/miscellaneous, etc.)."""
        lower = code_label.lower()
        return any(marker in lower for marker in CodeAssigner._CATCHALL_MARKERS)

    @staticmethod
    def _normalize_code_name(returned_code: str, candidate_codes: list) -> str:
        """Match the LLM-returned code name to the closest candidate from the list.

        Handles: slight string variants (spaces vs underscores), partition names
        returned instead of code names, trailing artifacts like '(0)'.
        Returns 'NONE' if no reasonable match is found.
        """
        if returned_code == "NONE":
            return "NONE"

        candidate_names = [c.code for c in candidate_codes]

        # 1. Exact match
        if returned_code in candidate_names:
            return returned_code

        # 2. Clean up common artifacts: strip (0), trailing whitespace
        cleaned = returned_code.strip().rstrip("(0)").rstrip().rstrip("(").rstrip()
        if cleaned in candidate_names:
            return cleaned

        # 3. Normalize: lowercase, replace underscores with spaces, collapse whitespace
        def norm(s):
            return " ".join(s.lower().replace("_", " ").split())

        norm_returned = norm(cleaned)
        for name in candidate_names:
            if norm(name) == norm_returned:
                return name

        # 4. Substring match: if returned string contains a candidate name (or vice versa)
        for name in candidate_names:
            if norm(name) in norm_returned or norm_returned in norm(name):
                return name

        # 5. Fuzzy match: use difflib to find closest candidate (threshold 0.6)
        norm_candidates = {norm(name): name for name in candidate_names}
        matches = difflib.get_close_matches(norm_returned, norm_candidates.keys(), n=1, cutoff=0.6)
        if matches:
            return norm_candidates[matches[0]]

        # No match — return original (will fall into unknown bucket)
        return returned_code

    def _format_partition_codes(self, codes: List[models.CodebookExp]) -> str:
        """Format all codes from a partition for the evaluation prompt.

        Catch-all codes (overig/anders) are tagged as LAST RESORT so the LLM
        deprioritizes them in favor of specific codes.
        """
        formatted = []
        for code in codes:
            is_catchall = self._is_catchall_code(code.code)
            tag = " [LAST RESORT - only if no specific code fits]" if is_catchall else ""
            entry = (
                f"Code: {code.code}{tag}\n"
                f"Definition: {code.definition}\n"
                f"Boundary test (primary focus check): {getattr(code, 'boundary_test', 'N/A') or 'N/A'}\n"
                f"Diagnostic signals: {self._format_diagnostic_signals(code)}\n"
                f"Inclusion Examples (valid references for this code):\n    {self._format_examples_list(code.inclusion_examples)}\n"
                f"Routing redirects (ideas that belong to a neighboring code instead):\n    {self._format_examples_list(code.exclusion_examples)}\n"
                f"Routing rule: Differs from '{code.near_neighbor_label or 'N/A'}' - {code.tell_apart_rule or 'N/A'}"
            )
            formatted.append(entry)
        return "\n---\n".join(formatted)

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
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

        probe_kwargs = dict(model=model, messages=[{"role": "user", "content": "Hi"}])
        model_type = ModelConfig.MODEL_TYPES.get(self.model, "chat")
        if model_type == "reasoning":
            probe_kwargs["max_completion_tokens"] = 5
        else:
            probe_kwargs["max_tokens"] = 5
        response = await client.chat.completions.with_raw_response.create(**probe_kwargs)
        return extract_rate_limits_from_response(response)

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate total tokens using adaptive strategy."""
        tiktoken_count = len(self.encoding.encode(prompt))
        offset = self.tiktoken_offset_learner.get_offset()
        actual_input_tokens = tiktoken_count + offset

        num_samples = len(self.estimation_errors)
        safety_margin = 1.15 if num_samples < 5 else 1.10 if num_samples < 15 else 1.05

        if len(self.input_token_history) >= 5:
            avg_input = sum(self.input_token_history) / len(self.input_token_history)
            estimated_input = int(0.7 * avg_input + 0.3 * actual_input_tokens)
        else:
            estimated_input = int(actual_input_tokens * safety_margin)

        self.input_token_history.append(actual_input_tokens)

        if len(self.output_ratio_history) >= 5:
            learned_ratio = sum(self.output_ratio_history) / len(self.output_ratio_history)
            estimated_output = int(estimated_input * learned_ratio * safety_margin)
        elif len(self.output_token_history) >= 3:
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output * safety_margin)
        else:
            estimated_output = int(estimated_input * DEFAULT_OUTPUT_RATIO * safety_margin)

        estimated_output = min(self.config.max_tokens, estimated_output)
        return estimated_input + estimated_output

    def _assign_themes_to_codes(self, assigned_codes: List[str]) -> List[str]:
        themes = []
        for code in assigned_codes:
            theme = self.code_to_theme_mapping.get(code)
            if theme and theme not in themes:
                themes.append(theme)
        return themes

    # === IDEA EXTRACTION ========================================================================================================

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all ideas with ladder fields for processing."""
        all_ideas = []
        for model in self.response_models:
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea in model.response_ideas:
                    all_ideas.append((
                        model.respondent_id,
                        idea.idea_id,
                        idea.idea,
                        getattr(idea, 'concept_type', '') or '',
                        getattr(idea, 'instance', '') or '',
                        getattr(idea, 'rung_1', '') or '',
                        getattr(idea, 'rung_2', '') or '',
                    ))
            else:
                self.verbose_reporter.stat_line(f"Warning: No response_ideas for respondent {model.respondent_id}")
        return all_ideas

    # === LLM EVALUATION METHODS ========================================================================================================

    async def _reconcile_tokens(self, response, prompt: str, est_tokens: int):
        """Common token reconciliation logic after an LLM call."""
        if hasattr(response, '_raw_response'):
            usage = response._raw_response.usage
            if usage:
                actual_input = getattr(usage, 'prompt_tokens', None) or getattr(usage, 'input_tokens', 0)
                actual_output = getattr(usage, 'completion_tokens', None) or getattr(usage, 'output_tokens', 0)
                actual_total = getattr(usage, 'total_tokens', 0) or (actual_input + actual_output)

                await self.tpm_bucket.reconcile(actual_total - est_tokens)
                self.output_token_history.append(actual_output)
                self.actual_total_tokens.append(actual_total)

                if actual_input > 0:
                    self.output_ratio_history.append(actual_output / actual_input)

                self.estimation_errors.append(abs(actual_total - est_tokens))
                await self.tpm_tracker.record(actual_total)

                tiktoken_input = len(self.encoding.encode(prompt))
                self.tiktoken_offset_learner.record(tiktoken_input, actual_input)

    async def evaluate_single_code(self, idea_id, idea_text, instance, rung_1, concept_type, code):
        """Evaluate a single code against an idea's abstraction ladder."""
        prompt = SINGLE_CODE_EVALUATION_PROMPT.format(
            language=self.language, var_lab=self.var_lab,
            idea_id=idea_id,
            instance=instance or idea_text,
            rung_1=rung_1 or idea_text,
            concept_type=concept_type,
            code=code.code, definition=code.definition,
            boundary_test=getattr(code, 'boundary_test', 'N/A') or 'N/A',
            diagnostic_signals=self._format_diagnostic_signals(code),
            inclusion_examples=self._format_examples_list(code.inclusion_examples),
            exclusion_examples=self._format_examples_list(code.exclusion_examples),
            near_neighbor_label=code.near_neighbor_label or "Unknown",
            tell_apart_rule=code.tell_apart_rule or "N/A"
        )

        self.last_prompt = prompt
        est_tokens = self.estimate_tokens(prompt)
        timeout = self.latency_tracker.get_timeout(est_tokens)

        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            async with self.rate_limiter:
                start_time = time.perf_counter()
                response = await asyncio.wait_for(
                    llm_create_async(
                        client=self.client, model=self.model, prompt=prompt,
                        response_model=DefaultCodeEvaluationResponse,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens, track_usage=True
                    ),
                    timeout=timeout
                )
                self.latency_tracker.add(time.perf_counter() - start_time)
                await self._reconcile_tokens(response, prompt, est_tokens)

        self.partition_eval_calls += 1
        return response, prompt

    async def evaluate_partition_codes(self, idea_id, idea_text, instance, rung_1, concept_type, codes, grouped_codes=None):
        """Evaluate all codes from a partition against an idea's abstraction ladder."""
        # Use grouped formatting when refined sub-partitions exist
        if grouped_codes:
            codes_formatted = self._format_grouped_partition_codes(grouped_codes)
        else:
            codes_formatted = self._format_partition_codes(codes)

        # Build dominance axis block for this partition — procedural gate
        dominance_axis = self.dominance_axes.get(concept_type, "")
        if dominance_axis:
            dominance_axis_block = (
                f"\n<routing_dimension>\n"
                f"MANDATORY ROUTING GATE — Answer this question in Step 0(c) before evaluating any code:\n"
                f"{dominance_axis}\n"
                f"Your answer determines which code receives this idea. A code whose primary focus "
                f"contradicts your answer here CANNOT be the best match.\n"
                f"</routing_dimension>\n"
            )
        else:
            dominance_axis_block = ""

        prompt = PARTITION_EVALUATION_PROMPT.format(
            language=self.language, var_lab=self.var_lab,
            idea_id=idea_id,
            instance=instance or idea_text,
            rung_1=rung_1 or idea_text,
            concept_type=concept_type,
            dominance_axis_block=dominance_axis_block,
            partition_codes_formatted=codes_formatted
        )

        self.last_prompt = prompt
        est_tokens = self.estimate_tokens(prompt)
        timeout = self.latency_tracker.get_timeout(est_tokens)

        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            async with self.rate_limiter:
                start_time = time.perf_counter()
                response = await asyncio.wait_for(
                    llm_create_async(
                        client=self.client, model=self.model, prompt=prompt,
                        response_model=PartitionEvaluationResponse,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens, track_usage=True
                    ),
                    timeout=timeout
                )
                self.latency_tracker.add(time.perf_counter() - start_time)
                await self._reconcile_tokens(response, prompt, est_tokens)

        self.partition_eval_calls += 1
        return response, prompt

    async def evaluate_similarity_codes(self, idea_id, idea_text, instance, rung_1, concept_type, codes, similarity_context, rung_2=None):
        """Evaluate similarity-selected codes — simple classification prompt."""
        codes_formatted = self._format_partition_codes(codes)

        prompt = SIMILARITY_EVALUATION_PROMPT.format(
            language=self.language, var_lab=self.var_lab,
            idea_id=idea_id,
            instance=instance or idea_text,
            rung_1=rung_1 or idea_text,
            rung_2=rung_2 or concept_type or '',
            concept_type=concept_type,
            candidate_codes_formatted=codes_formatted
        )

        self.last_prompt = prompt
        est_tokens = self.estimate_tokens(prompt)
        timeout = self.latency_tracker.get_timeout(est_tokens)

        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            async with self.rate_limiter:
                start_time = time.perf_counter()
                response = await asyncio.wait_for(
                    llm_create_async(
                        client=self.client, model=self.model, prompt=prompt,
                        response_model=SimpleClassificationResponse,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens, track_usage=True
                    ),
                    timeout=timeout
                )
                self.latency_tracker.add(time.perf_counter() - start_time)
                await self._reconcile_tokens(response, prompt, est_tokens)

        self.partition_eval_calls += 1
        return response, prompt

    async def _retry_with_correction(self, idea_id, idea_text, instance, rung_1, rung_2, concept_type, candidate_codes, invalid_code):
        """Retry when the LLM returned a code not in the candidate list.

        Sends a short correction prompt listing only the valid code names.
        """
        valid_names = [c.code for c in candidate_codes]
        prompt = (
            f"You previously returned \"{invalid_code}\" but that is not a valid code. "
            f"Pick exactly one code from this list:\n"
            f"{chr(10).join(f'- {name}' for name in valid_names)}\n"
            f"- NONE\n\n"
            f"Idea: \"{instance or idea_text}\"\n"
            f"Respond with JSON: {{\"idea_id\": \"{idea_id}\", \"code\": \"PICK_ONE\", "
            f"\"confidence\": 0.00, \"rationale\": \"brief\"}}"
        )

        est_tokens = self.estimate_tokens(prompt)
        timeout = self.latency_tracker.get_timeout(est_tokens)

        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            async with self.rate_limiter:
                start_time = time.perf_counter()
                response = await asyncio.wait_for(
                    llm_create_async(
                        client=self.client, model=self.model, prompt=prompt,
                        response_model=SimpleClassificationResponse,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens, track_usage=True
                    ),
                    timeout=timeout
                )
                self.latency_tracker.add(time.perf_counter() - start_time)
                await self._reconcile_tokens(response, prompt, est_tokens)

        return response, prompt

    async def probe_call_no_structured(self, task_dict):
        """Probe call for bootstrap measurement."""
        idea_data = task_dict['idea_data']
        respondent_id, idea_id, idea_text, concept_type, instance, rung_1, rung_2 = idea_data

        prompt = self._create_probe_prompt(idea_id, idea_text, instance, rung_1, concept_type)

        resp = await llm_create_async(
            client=self.client, model=self.model, prompt=prompt,
            response_model=ProbeResponse,
            temperature=self.config.temperature, track_usage=False
        )

        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        if u:
            prompt_tokens = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0)
            completion_tokens = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0)
            return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        return {"prompt_tokens": 0, "completion_tokens": 0}

    # === TASK PROCESSING ========================================================================================================

    @retry(
        retry=retry_if_exception_type((
            RateLimitError, APIConnectionError, APITimeoutError,
            InternalServerError, InstructorRetryException, asyncio.TimeoutError
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def process_task(self, task: Dict) -> CodeAssignmentResponse:
        """Code assignment: route via partition or semantic similarity, then LLM evaluation."""
        try:
            idea_data = task['idea_data']
            respondent_id, idea_id, idea_text, concept_type, instance, rung_1, rung_2 = idea_data

            unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")
            prompt_used = ""
            use_similarity = self.similarity_config.routing_mode != "partition"

            # === CODE SELECTION: similarity or partition ===
            if use_similarity:
                idea_embedding = self._idea_embeddings_lookup.get(idea_id)
                if idea_embedding is not None:
                    candidate_codes = self._find_similar_codes(idea_embedding, concept_type)
                    grouped_codes = None  # no sub-partition grouping in similarity mode
                else:
                    # Fallback to partition routing if no embedding
                    candidate_codes, grouped_codes = self._get_codes_for_idea(concept_type)
            else:
                candidate_codes, grouped_codes = self._get_codes_for_idea(concept_type)

            # === LLM EVALUATION ===
            if not candidate_codes:
                assigned_code = unknown_label
                confidence = 0.0
                rationale = f"EXCLUDE: No codes available for idea '{idea_id}'"
                self.unknown_count += 1
                logger.info(f"Idea {idea_id}: no candidate codes")

            elif len(candidate_codes) == 1:
                code = candidate_codes[0]
                eval_result, prompt_used = await self.evaluate_single_code(
                    idea_id, idea_text, instance, rung_1, concept_type, code
                )
                self.confidence_tracker.record(eval_result.confidence)

                threshold = (self.confidence_tracker.get_adaptive_threshold(self.adaptive_threshold_config.fixed_threshold)
                             if self.adaptive_threshold_config.use_adaptive
                             else self.adaptive_threshold_config.fixed_threshold)

                if eval_result.confidence >= threshold:
                    assigned_code = code.code
                    confidence = eval_result.confidence
                    rationale = eval_result.rationale
                    self.partition_match_count += 1
                else:
                    assigned_code = unknown_label
                    confidence = eval_result.confidence
                    rationale = f"Below threshold ({threshold:.2f}). {eval_result.rationale}"
                    self.unknown_count += 1

            else:
                # Multiple codes → LLM evaluates
                if use_similarity and grouped_codes is None:
                    # Similarity mode: simple classification prompt
                    similarity_context = self._format_similarity_context(candidate_codes)
                    eval_result, prompt_used = await self.evaluate_similarity_codes(
                        idea_id, idea_text, instance, rung_1, concept_type,
                        candidate_codes, similarity_context, rung_2=rung_2
                    )

                    # SimpleClassificationResponse: .code, .confidence, .rationale
                    # Normalize LLM-returned code name against candidate list
                    candidate_names = [c.code for c in candidate_codes]
                    normalized = self._normalize_code_name(eval_result.code, candidate_codes)

                    # If normalization didn't resolve to a valid code, retry with correction prompt
                    if normalized != "NONE" and normalized not in candidate_names:
                        logger.info(f"Idea {idea_id}: code '{eval_result.code}' not in candidates, retrying")
                        retry_result, _ = await self._retry_with_correction(
                            idea_id, idea_text, instance, rung_1, rung_2, concept_type,
                            candidate_codes, eval_result.code
                        )
                        normalized = self._normalize_code_name(retry_result.code, candidate_codes)

                    eval_result.code = normalized
                    self.confidence_tracker.record(eval_result.confidence)

                    threshold = (self.confidence_tracker.get_adaptive_threshold(self.adaptive_threshold_config.fixed_threshold)
                                 if self.adaptive_threshold_config.use_adaptive
                                 else self.adaptive_threshold_config.fixed_threshold)

                    if eval_result.code != "NONE" and eval_result.confidence >= threshold:
                        assigned_code = eval_result.code
                        confidence = eval_result.confidence
                        rationale = eval_result.rationale
                        self.partition_match_count += 1
                    else:
                        assigned_code = unknown_label
                        confidence = eval_result.confidence
                        rationale = f"Below threshold ({threshold:.2f}). {eval_result.rationale}"
                        self.unknown_count += 1
                else:
                    # Partition mode: use partition evaluation (grouped by sub-partition)
                    eval_result, prompt_used = await self.evaluate_partition_codes(
                        idea_id, idea_text, instance, rung_1, concept_type,
                        candidate_codes, grouped_codes
                    )

                    # Normalize LLM-returned code name against candidate list
                    eval_result.best_match.code = self._normalize_code_name(eval_result.best_match.code, candidate_codes)
                    best_confidence = eval_result.best_match.confidence
                    self.confidence_tracker.record(best_confidence)

                    threshold = (self.confidence_tracker.get_adaptive_threshold(self.adaptive_threshold_config.fixed_threshold)
                                 if self.adaptive_threshold_config.use_adaptive
                                 else self.adaptive_threshold_config.fixed_threshold)

                    if eval_result.best_match.code != "NONE" and best_confidence >= threshold:
                        assigned_code = eval_result.best_match.code
                        confidence = best_confidence
                        rationale = eval_result.best_match.rationale
                        self.partition_match_count += 1
                    else:
                        assigned_code = unknown_label
                        confidence = best_confidence
                        rationale = f"Below threshold ({threshold:.2f}). {eval_result.best_match.rationale}"
                        self.unknown_count += 1

            # Create response
            response = CodeAssignmentResponse(
                idea_id=idea_id, idea=idea_text,
                assigned_codes=[assigned_code],
                assignment_confidence=confidence,
                assignment_rationale=rationale
            )
            response.assigned_themes = self._assign_themes_to_codes([assigned_code])

            # Record for diagnostics
            matched = assigned_code != unknown_label
            self.pattern_tracker.record_assignment(
                respondent_id=respondent_id, partition_name=concept_type,
                assigned_code=assigned_code, confidence=confidence, matched=matched,
            )

            # Debug logging
            if self.verbose:
                self.prompt_responses.append({
                    'prompt': prompt_used, 'respondent_id': respondent_id,
                    'idea_id': idea_id, 'idea_text': idea_text,
                    'instance': instance, 'rung_1': rung_1,
                    'concept_type': concept_type,
                    'assigned_codes': [assigned_code],
                    'confidence': confidence, 'rationale': rationale,
                })

            self.stats['tasks_successful'] += 1
            return response

        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            raise
        except RateLimitError:
            self.stats['rate_limits'] += 1
            raise
        except InstructorRetryException as e:
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
            raise  # let tenacity retry
        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {type(e).__name__}: {e}")
            raise

    def create_fallback_response(self, task: Dict) -> CodeAssignmentResponse:
        """Create fallback response for failed tasks."""
        idea_data = task['idea_data']
        respondent_id, idea_id, idea_text, concept_type, instance, rung_1, rung_2 = idea_data
        unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")
        return CodeAssignmentResponse(
            idea_id=idea_id, idea=idea_text,
            assigned_codes=[unknown_label], assigned_themes=[],
            assignment_confidence=0.0,
            assignment_rationale=f"Processing failed in partition '{concept_type}'"
        )

    def get_failure_report(self, total_tasks: int = None) -> str:
        """Return a formatted report of all processing failures."""
        from collections import Counter
        total = total_tasks or self.stats.get('tasks_processed', 0)
        n_failures = len(self.failure_log)

        if n_failures == 0:
            return f"PROCESSING ERRORS: 0 of {total} tasks (0%)"

        lines = [
            f"PROCESSING ERRORS: {n_failures} of {total} tasks "
            f"({n_failures / max(total, 1) * 100:.1f}%)"
        ]

        reason_counts = Counter(f['error_type'] for f in self.failure_log)
        lines.append(
            f"  Breakdown: {', '.join(f'{count}x {reason}' for reason, count in reason_counts.most_common())}"
        )
        lines.append("")

        for f in self.failure_log:
            lines.append(
                f"  Idea {f['idea_id']} (resp {f['respondent_id']}): "
                f"{f['error_type']} | \"{f['idea_preview']}...\""
            )

        return "\n".join(lines)

    async def worker(self, queue: asyncio.Queue, results: List):
        while True:
            try:
                task = await queue.get()
                if task is None:
                    break
                try:
                    result = await self.process_task(task)
                    results[task['result_index']] = result
                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__

                    # Classify: is this a 429 rate-limit failure?
                    is_rate_limit = (
                        "429" in error_str
                        or "RateLimitReached" in error_str
                        or isinstance(e, RateLimitError)
                    )

                    if is_rate_limit:
                        if "token rate limit" in error_str.lower():
                            limit_type = "TPM"
                        elif "call rate limit" in error_str.lower():
                            limit_type = "RPM"
                        else:
                            limit_type = "rate"
                        classified_type = f"RateLimit_{limit_type}"
                        print(f"429 {limit_type} — task {task['task_id']} queued for retry")
                        self._rate_limit_retry_queue.append(task)
                    else:
                        classified_type = error_type
                        logger.error(f"Task {task['task_id']} failed after retries: {error_type}")

                    # Aggregated stats
                    if classified_type not in self.stats['error_types']:
                        self.stats['error_types'][classified_type] = {'count': 0, 'sample_messages': []}
                    self.stats['error_types'][classified_type]['count'] += 1
                    if len(self.stats['error_types'][classified_type]['sample_messages']) < 3:
                        self.stats['error_types'][classified_type]['sample_messages'].append(error_str[:200])

                    self.stats['tasks_failed'] += 1
                    results[task['result_index']] = self.create_fallback_response(task)

                    # Per-task audit trail
                    self.failure_log.append({
                        'idea_id': task['task_id'],
                        'respondent_id': task['idea_data'][0],
                        'reason': 'exception',
                        'error_type': classified_type,
                        'idea_preview': task['idea_data'][2][:80],
                    })
                finally:
                    self.stats['tasks_processed'] += 1
                    queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {type(e).__name__}: {e}")
                break

    # === MERGE RESULTS ========================================================================================================

    def _merge_results_into_models(self, assignment_results: List[CodeAssignmentResponse]) -> List[models.CodeAssignedModel]:
        """Merge assignment results back into model structure."""
        assignments_lookup = {result.idea_id: result for result in assignment_results}
        coded_models = []

        for original_model in self.response_models:
            coded_model = original_model.to_model(models.CodeAssignedModel)
            if coded_model.response_ideas:
                updated_ideas = []
                for idea_submodel in coded_model.response_ideas:
                    idea_data = idea_submodel.model_dump()
                    if idea_submodel.idea_id in assignments_lookup:
                        assignment = assignments_lookup[idea_submodel.idea_id]
                        idea_data['assigned_codes'] = assignment.assigned_codes
                        idea_data['assigned_themes'] = assignment.assigned_themes
                        idea_data['assignment_confidence'] = assignment.assignment_confidence
                        idea_data['assignment_rationale'] = assignment.assignment_rationale
                    else:
                        idea_data['assigned_codes'] = ["Unassigned"]
                        idea_data['assigned_themes'] = []
                        idea_data['assignment_confidence'] = 0.0
                        idea_data['assignment_rationale'] = "No assignment found"
                    updated_ideas.append(models.AssignedIdeaSubmodel(**idea_data))
                coded_model.response_ideas = updated_ideas
            coded_models.append(coded_model)

        return coded_models

    # === STATS & REPORTING ========================================================================================================

    def print_assignment_stats(self):
        total = self.partition_match_count + self.unknown_count
        if total == 0:
            print("\nNo assignment stats available")
            return

        match_pct = (self.partition_match_count / total) * 100
        unknown_pct = (self.unknown_count / total) * 100

        print(f"\n{'='*80}")
        print("CODE ASSIGNMENT BREAKDOWN (PARTITION-BASED, LADDER)")
        print(f"{'='*80}")
        print(f"Total ideas processed: {total}")
        print(f"Partition match:  {self.partition_match_count} ({match_pct:.1f}%)")
        print(f"Unknown:          {self.unknown_count} ({unknown_pct:.1f}%)")
        print(f"API calls: {self.partition_eval_calls}")
        print(f"{'='*80}")
        self.print_partition_stats()

    def print_partition_stats(self):
        diagnostics = self.pattern_tracker.partition_diagnostics
        if not diagnostics:
            return
        print(f"\n{'_'*60}")
        print("PER-PARTITION BREAKDOWN")
        print(f"{'_'*60}")
        for name in sorted(diagnostics.keys()):
            diag = diagnostics[name]
            print(f"  {name}: {diag.total_ideas} ideas, "
                  f"{diag.matched} matched ({diag.match_rate*100:.0f}%), "
                  f"{diag.unknown} unknown, avg conf: {diag.avg_confidence:.2f}")
        print(f"{'_'*60}\n")

    def print_learning_insights(self):
        if not self.pattern_config.enabled:
            return
        print(f"\n{'_'*60}")
        print("PATTERN LEARNING INSIGHTS")
        print(f"{'_'*60}")

        if self.adaptive_threshold_config.use_adaptive:
            stats = self.confidence_tracker.get_stats()
            print(f"\nAdaptive Threshold: {stats.get('current_threshold', 0.7):.3f} "
                  f"({stats.get('samples', 0)} samples, warmed up: {stats.get('is_warmed_up', False)})")
        else:
            print(f"\nFixed Threshold: {self.adaptive_threshold_config.fixed_threshold}")

        if self.pattern_config.track_cluster_fallback:
            problematic = self.pattern_tracker.get_problematic_partitions(unknown_threshold=0.5, min_ideas=3)
            if problematic:
                print(f"\nPartitions with high unknown rates (>50%):")
                for p in problematic[:5]:
                    print(f"   - {p['partition']}: {p['unknown_rate']*100:.0f}% unknown ({p['total_ideas']} ideas)")

        if self.pattern_config.track_confidence_calibration:
            cal = self.pattern_tracker.get_confidence_calibration()
            if cal:
                print(f"\nConfidence distribution:")
                for bucket, data in sorted(cal.items()):
                    print(f"   {bucket}: {data['count']} assignments")

        print(f"{'_'*60}\n")

    # === THROUGHPUT ADJUSTMENT ========================================================================================================

    async def _apply_pid_adjustment(self) -> bool:
        if self.current_arrival_rate is None:
            return False
        current_tpm = await self.tpm_tracker.get_current_tpm()
        tpm_limit = self.rate_limits.tokens_per_minute
        utilization = current_tpm / tpm_limit if tpm_limit > 0 else 0.0

        self.v3_stats['max_tpm_utilization'] = max(self.v3_stats['max_tpm_utilization'], utilization * 100)
        self.v3_stats['min_tpm_utilization'] = min(self.v3_stats['min_tpm_utilization'], utilization * 100)

        adjustment = self.pid_controller.compute_adjustment(utilization)
        if abs(adjustment - 1.0) < 0.01:
            return False

        old_rate = self.current_arrival_rate
        new_rate = max(0.5, min(self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60, old_rate * adjustment))

        if abs(new_rate - old_rate) / old_rate < 0.02:
            return False

        self.rate_limiter = AsyncLimiter(1, time_period=1 / new_rate) if new_rate < 1 else AsyncLimiter(int(new_rate), time_period=1.0)
        self.current_arrival_rate = new_rate
        self.v3_stats['pid_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1
        return True

    def _adjust_throughput_if_needed(self) -> bool:
        if len(self.actual_total_tokens) < THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
            return False

        actual_avg = sum(self.actual_total_tokens) / len(self.actual_total_tokens)
        bootstrap_avg = self._avg_tokens
        ratio = actual_avg / bootstrap_avg if bootstrap_avg > 0 else 1.0
        if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
            return False

        new_tpm_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / actual_avg / 60
        rpm_throughput = self.rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        new_arrival_rate = min(rpm_throughput, new_tpm_throughput)

        self.rate_limiter = AsyncLimiter(1, time_period=1 / new_arrival_rate) if new_arrival_rate < 1 else AsyncLimiter(int(new_arrival_rate), time_period=1.0)
        self.current_arrival_rate = new_arrival_rate
        self.tpm_bucket = TokenBucket(int(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom))
        self.pid_controller.reset()
        self._avg_tokens = int(actual_avg)
        self.v3_stats['threshold_adjustments'] += 1
        self.v3_stats['adjustments_made'] += 1
        return True

    # === MAIN PROCESSING PIPELINE ========================================================================================================

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[CodeAssignmentResponse]:
        """Process all tasks using queue + workers with bootstrap measurement."""
        if not tasks:
            return []

        try:
            self.verbose_reporter.step_start("Code Assignment")

            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("Fetching rate limits from API...")

            limits = await self._fetch_rate_limits_from_api()
            if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
                limits = RateLimits(tokens_per_minute=FALLBACK_TPM, requests_per_minute=FALLBACK_RPM, tokens_per_day=FALLBACK_TPM * 60 * 24)
            else:
                self.verbose_reporter.stat_line(f"Fetched from API: TPM={limits.tokens_per_minute:,}, RPM={limits.requests_per_minute:,}")

            self.rate_limits = limits
            self.tpm_bucket = TokenBucket(limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

            # Bootstrap measurement
            sample_tasks = tasks[:min(3, len(tasks))]
            if len(sample_tasks) < 3:
                sample_tasks = (sample_tasks * 3)[:3]

            self.verbose_reporter.stat_line("Running bootstrap measurement (3 probe calls)...")
            start_time = time.time()
            task_cycle = itertools.cycle(sample_tasks)

            async def probe():
                return await self.probe_call_no_structured(next(task_cycle))

            avg_latency_s, avg_tokens = await bootstrap_measure_async(probe, n_probes=3)
            self.verbose_reporter.stat_line(f"Bootstrap: {avg_latency_s:.3f}s avg latency, {avg_tokens:.0f} avg tokens")

            for _ in range(3):
                self.latency_tracker.add(avg_latency_s)
            self._avg_tokens = int(avg_tokens)

            # Compute optimal concurrency
            api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
            Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, processing_config=self.processing_config, cap=self.processing_config.concurrency_cap_permissive, min_conc=self.processing_config.concurrency_min_permissive)
            optimal = min(self.processing_config.concurrency_cap_default, max(Little, self.processing_config.concurrency_min_default))

            arrival_rate = min(
                limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
                limits.tokens_per_minute * self.processing_config.rate_limit_headroom / avg_tokens / 60
            )

            self.rate_limiter = AsyncLimiter(1, time_period=1 / arrival_rate) if arrival_rate < 1 else AsyncLimiter(int(arrival_rate), time_period=1.0)
            self.semaphore = asyncio.Semaphore(min(len(tasks), optimal))
            self.optimal_concurrency = min(len(tasks), optimal)
            self.current_arrival_rate = arrival_rate

            rpm_throughput = limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
            tpm_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self._avg_tokens / 60
            bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

            print(f"[RATE LIMITING] {self.model} | {bottleneck} limited | "
                  f"{min(rpm_throughput, tpm_throughput):.1f}/s | concurrency={optimal} | {len(tasks)} tasks")

            # Queue + workers
            expected_throughput = min(rpm_throughput, tpm_throughput)
            max_workers = getattr(self.processing_config, 'max_workers', 200)
            min_workers = getattr(self.processing_config, 'min_workers', 50)
            num_workers = min(max_workers, max(min_workers, int(expected_throughput * avg_latency_s * 2.0)))

            queue = asyncio.Queue()
            results = [None] * len(tasks)

            for i, task in enumerate(tasks):
                task['result_index'] = i
                task['task_index'] = i
                task['task_id'] = task['idea_data'][1]  # idea_id
                await queue.put(task)

            workers = [asyncio.create_task(self.worker(queue, results)) for _ in range(num_workers)]

            # Progress monitoring
            start_time = time.time()
            last_report = start_time
            last_adjustment = start_time

            while self.stats['tasks_processed'] < len(tasks):
                await asyncio.sleep(1)
                now = time.time()

                if now - last_report >= PROGRESS_REPORT_INTERVAL:
                    completed = self.stats['tasks_processed']
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%), Rate: {rate:.1f}/s")
                    last_report = now

                if now - last_adjustment >= ADJUSTMENT_INTERVAL:
                    await self._apply_pid_adjustment()
                    if self.stats['tasks_processed'] >= THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
                        self._adjust_throughput_if_needed()
                    last_adjustment = now

                if queue.empty() and self.stats['tasks_processed'] < len(tasks):
                    break

            await queue.join()
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)

            elapsed = time.time() - start_time
            print(f"\nPass 1 completed {len(tasks)} tasks in {elapsed:.1f}s "
                  f"({self.stats['tasks_successful']} ok, {self.stats['tasks_failed']} failed, "
                  f"{elapsed/len(tasks):.2f}s/task)")

            if self.stats['error_types']:
                for error_type, data in self.stats['error_types'].items():
                    print(f"  {error_type}: {data['count']}x")

            # === RETRY PASS: re-process tasks that failed due to 429 rate limits ===
            if self._rate_limit_retry_queue:
                retry_tasks = list(self._rate_limit_retry_queue)
                self._rate_limit_retry_queue.clear()

                RETRY_COOLDOWN_SECONDS = 30
                print(f"\n[RETRY PASS] {len(retry_tasks)} tasks failed due to rate limits.")
                print(f"[RETRY PASS] Waiting {RETRY_COOLDOWN_SECONDS}s cooldown before retrying...")
                await asyncio.sleep(RETRY_COOLDOWN_SECONDS)

                # Remove retried tasks from failure_log (they get a fresh chance)
                retry_idea_ids = {t['task_id'] for t in retry_tasks}
                self.failure_log = [f for f in self.failure_log if f['idea_id'] not in retry_idea_ids]
                # Adjust stats so retry-pass processing counts correctly
                self.stats['tasks_failed'] -= len(retry_tasks)
                self.stats['tasks_processed'] -= len(retry_tasks)

                # Re-populate a fresh queue
                retry_queue = asyncio.Queue()
                retry_results = [None] * len(retry_tasks)
                for i, task in enumerate(retry_tasks):
                    task['result_index'] = i  # local index into retry_results
                    await retry_queue.put(task)

                retry_num_workers = min(num_workers, len(retry_tasks))
                retry_workers = [
                    asyncio.create_task(self.worker(retry_queue, retry_results))
                    for _ in range(retry_num_workers)
                ]

                # Simple progress monitoring for retry pass
                retry_start = time.time()
                retry_target = self.stats['tasks_processed'] + len(retry_tasks)
                while self.stats['tasks_processed'] < retry_target:
                    await asyncio.sleep(1)
                    if retry_queue.empty():
                        break

                await retry_queue.join()
                for _ in retry_workers:
                    await retry_queue.put(None)
                await asyncio.gather(*retry_workers)

                # Merge successful retries back into main results
                recovered = 0
                for i, task in enumerate(retry_tasks):
                    original_index = task['task_index']
                    if retry_results[i] is not None and retry_results[i].assignment_confidence > 0.0:
                        results[original_index] = retry_results[i]
                        recovered += 1

                retry_elapsed = time.time() - retry_start
                print(f"[RETRY PASS] Completed in {retry_elapsed:.1f}s: "
                      f"{recovered}/{len(retry_tasks)} recovered")

            # Final failure report
            if self.failure_log:
                print(f"\n{'_' * 60}")
                print(self.get_failure_report(total_tasks=len(tasks)))
                print(f"{'_' * 60}")

            return results

        except Exception as e:
            logger.error(f"[CRITICAL] process_all_tasks_async failed: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [self.create_fallback_response(task) for task in tasks]

    def _prepare_individual_tasks(self, all_ideas: List[tuple]) -> List[Dict]:
        return [{'idea_data': idea_data} for idea_data in all_ideas]

    def _create_other_assignments(self, other_ideas: List[tuple]) -> List[CodeAssignmentResponse]:
        """Create synthetic CodeAssignmentResponse for pre-assigned 'other' ideas.

        These ideas were assigned DIRECT_OTHER codes in step 6 and should skip
        LLM assignment. The code is looked up from self.other_idea_assignments.
        """
        results = []
        for idea_data in other_ideas:
            respondent_id, idea_id, idea_text, *_ = idea_data
            code_label = self.other_idea_assignments.get(idea_id, "")
            theme = self.code_to_theme_mapping.get(code_label, "")

            results.append(CodeAssignmentResponse(
                idea_id=idea_id,
                idea=idea_text,
                assigned_codes=[code_label],
                assignment_confidence=1.0,
                assignment_rationale="Pre-assigned: DIRECT_OTHER from step 6",
                assigned_themes=[theme] if theme else [],
            ))
        return results

    async def assign_codes(self) -> List[models.CodeAssignedModel]:
        """Main method: assign codes using partition-based or similarity-based routing."""
        self._stats.start_timing()

        # Initialize similarity infrastructure if needed
        if self.similarity_config.routing_mode != "partition":
            self._code_embeddings = self._generate_code_embeddings()
            self._build_idea_embeddings_lookup()

        all_ideas = self._extract_all_ideas()

        # Separate pre-assigned "other" ideas from ideas needing LLM assignment
        if self.other_idea_assignments:
            other_idea_ids = set(self.other_idea_assignments.keys())
            normal_ideas = [i for i in all_ideas if i[1] not in other_idea_ids]  # i[1] = idea_id
            other_ideas = [i for i in all_ideas if i[1] in other_idea_ids]
            self.verbose_reporter.stat_line(
                f"Pre-assigned 'other' ideas: {len(other_ideas)} (skipping LLM) | "
                f"Ideas for LLM assignment: {len(normal_ideas)}"
            )
        else:
            normal_ideas = all_ideas
            other_ideas = []

        total_ideas = len(normal_ideas)

        if total_ideas == 0 and not other_ideas:
            self.verbose_reporter.stat_line("No ideas found for code assignment")
            return []

        self.verbose_reporter.stat_line(f"Processing {total_ideas} ideas with {len(self.codebook)} codes across {len(self.partition_codebooks)} partitions")

        if normal_ideas:
            tasks = self._prepare_individual_tasks(normal_ideas)

            if nest_asyncio:
                nest_asyncio.apply()
            all_results = await self.process_all_tasks_async(tasks)
        else:
            all_results = []

        # Add synthetic results for pre-assigned "other" ideas
        if other_ideas:
            other_results = self._create_other_assignments(other_ideas)
            all_results.extend(other_results)

        self._results = self._merge_results_into_models(all_results)

        if all_results:
            valid_results = [r for r in all_results if r is not None]
            if valid_results:
                avg_confidence = np.mean([r.assignment_confidence for r in valid_results])
                high_conf = sum(1 for r in valid_results if r.assignment_confidence >= 0.7)
                low_conf = sum(1 for r in valid_results if r.assignment_confidence < 0.5)

                self.verbose_reporter.summary("CODE ASSIGNMENT COMPLETED", {
                    "Total ideas": len(valid_results),
                    "Avg confidence": f"{avg_confidence:.2f}",
                    "High confidence (>=0.7)": high_conf,
                    "Low confidence (<0.5)": low_conf,
                    "Partition matches": self.partition_match_count,
                    "Unknown": self.unknown_count,
                })

        self.print_learning_insights()

        if self.failure_log:
            print(f"\n{'=' * 70}")
            print("WARNING: NOT ALL TASKS COMPLETED SUCCESSFULLY")
            print(self.get_failure_report(total_tasks=len(all_results) if all_results else 0))
            print(f"{'=' * 70}")

        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes."""
        if nest_asyncio:
            nest_asyncio.apply()
        return asyncio.run(self.assign_codes())
