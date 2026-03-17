"""
Dual Assignment: Code + Attribute per Idea.

Assigns each idea to exactly one MECE code and its best-matching attribute
via a single LLM call. All partitions are processed concurrently with shared
rate limiting.

Rate limiting strategy (aligned with step 3 prompt processing strategy):
  1. ConcurrencyGate: completion-based ramp from 50% → 90% of Little's Law
  2. TokenBucket: TPM safety rail with reconciliation
  3. AsyncLimiter: PID-adjusted RPM arrival rate
  4. Timeout: generous safety net (60s floor, P95×3 adaptive)
  + Circuit breaker: monitors timeout RATE, adjusts concurrency on sustained pressure
  + Warm-up calibration: first 15-30 tasks calibrate tokens + latency → recalculate Little's Law

Pipeline:
  1. Group ideas by partition (domain)
  2. Fetch rate limits + estimate tokens via tiktoken (no probe calls)
  3. Initialize 4-layer rate limiting with ramp
  4. Queue + workers with warm-up calibration trigger
  5. Collect assignments, build CodeAssignedModel list

Usage:
    from .code_assignment import CodeAssigner
    from .config_classNcoder_exp import AssignmentConfig

    assigner = CodeAssigner(
        config=AssignmentConfig(),
        ideas_models=ideas_models,
        mece_results=mece_results,
        partition_set=partition_set,
        extraction_metadata=extraction_metadata,
    )
    results = assigner.assign_all()
"""

import asyncio
import logging
import re
import time
import statistics
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import nest_asyncio
from aiolimiter import AsyncLimiter
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from instructor.exceptions import InstructorRetryException

from utils.llm import (
    create_client, llm_create_async,
    RateLimits, extract_rate_limits_from_response,
)
from utils.cached_resources import get_tiktoken_encoding
from config import (
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
    OPENAI_API_KEY, API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
)

from development.step_3_ideaExtractor import models_exp as models
from development.step_3_ideaExtractor.ideaExtractor_exp import (
    TokenBucket,
    ConcurrencyGate,
    LatencyTracker,
    TiktokenOffsetLearner,
    ConcurrencyRamp,
    RealTimeTPMTracker,
    RealTimeRPMTracker,
    PIDThroughputController,
    ConcurrencyCircuitBreaker,
)
from config_steps.config_ideaExtractor import (
    DEFAULT_RAMP_UP_CONFIG,
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
    DEFAULT_PID_CONTROLLER_CONFIG,
    DEFAULT_TPM_TRACKING_CONFIG,
    DEFAULT_WARM_UP_CONFIG,
)

from .config_classNcoder_exp import AssignmentConfig, get_other_category_label
from .models_exp import DomainSet, DomainResultModel, CodeAssignedSubmodel, CodeAssignedModel
from .prompts_exp import (
    build_single_dual_assignment_prompt,
    CodeAssignment,
    CodeAssignmentBatch,
    CodeAttributeAssignment,
    CodeFromAttributes,
    MECECode,
)

# Reuse Little's Law calculation from qualitative_researcher
from .qualitative_researcher import (
    ApiLimits, compute_optimal_concurrency,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()

logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries during retries
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.ERROR)

# === PROCESSING CONSTANTS ===================================================

PROGRESS_REPORT_INTERVAL = 2           # Seconds between progress reports
PID_ADJUSTMENT_INTERVAL = 20           # Seconds between PID adjustments
THROUGHPUT_ADJUSTMENT_THRESHOLD = 1.3  # Trigger threshold-based adjustment
THROUGHPUT_ADJUSTMENT_MIN_SAMPLES = 10 # Min data points before adjusting
ERROR_WINDOW_SIZE = 100                # Rolling window for token tracking
DEFAULT_LATENCY_SECONDS = 0.8          # Default latency for nano (before warm-up)


class CodeAssigner:
    """
    Assigns each idea to exactly one MECE category within its domain
    partition. All partitions are processed concurrently through a shared
    queue + workers pattern with TokenBucket, rate limiter, and retry.
    """

    def __init__(
        self,
        config: AssignmentConfig,
        ideas_models: List[models.IdeasExtractedModel],
        mece_results: Dict[str, DomainResultModel],
        partition_set: DomainSet,
        extraction_metadata: Optional[models.ExtractionMetadata] = None,
        prompt_printer=None,
        codes: List[CodeFromAttributes] = None,
    ):
        self._config = config
        self._ideas_models = ideas_models
        self._mece_results = mece_results
        self._partition_set = partition_set
        self._extraction_metadata = extraction_metadata
        self._codes = codes or []

        # Prompt capture (optional — pass PromptPrinter to enable)
        self._prompt_printer = prompt_printer
        self._captured_assign_gates: set = set()

        # Shared async resources — initialized in _assign_all_async()
        self._client = None
        self._rate_limiter = None
        self._tpm_bucket = None
        self._rate_limits = None

        # 4-layer rate limiting components (initialized in _initialize_rate_limiters)
        self._gate = None                   # ConcurrencyGate (replaces Semaphore)
        self._latency_tracker = None        # LatencyTracker
        self._tiktoken_learner = None       # TiktokenOffsetLearner
        self._concurrency_ramp = None       # ConcurrencyRamp
        self._tpm_tracker = None            # RealTimeTPMTracker
        self._rpm_tracker = None            # RealTimeRPMTracker
        self._pid_controller = None         # PIDThroughputController
        self._circuit_breaker = None        # ConcurrencyCircuitBreaker
        self._warm_up_done = False          # One-shot calibration flag

        # Tokenizer for local token estimation
        self._encoding = get_tiktoken_encoding(config.assignment_model)

        # ID-based resolution maps — populated in _assign_all_async()
        self._id_to_label: Dict[str, str] = {}
        self._id_to_parent: Dict[str, str] = {}
        self._other_id: Optional[str] = None
        self._other_label: Optional[str] = None

        # Attribute assignments from dual mode (idea_id -> attribute name)
        self._attribute_assignments: Dict[str, str] = {}

        # Processing stats
        self._stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'rate_limits': 0,
            'timeouts': 0,
        }
        self._failure_log: List[Dict] = []

        # Token tracking
        self._actual_total_tokens: deque = deque(maxlen=ERROR_WINDOW_SIZE)
        self._avg_tokens: int = 0
        self._current_arrival_rate: float = 0.0
        self._adjustment_count: int = 0

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Canonical partition key: lowercase, underscores→spaces."""
        return (key or '').strip().lower().replace('_', ' ')

    def assign_all(self) -> List[CodeAssignedModel]:
        """Sync entry point. Returns list of CodeAssignedModel."""
        return asyncio.run(self._assign_all_async())

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _assign_all_async(self) -> List[CodeAssignedModel]:
        """Main async orchestration with 4-layer rate limiting + warm-up calibration."""
        verbose = self._config.verbose
        processing_config = DEFAULT_PROCESSING_CONFIG
        headroom = processing_config.rate_limit_headroom

        # ── Phase 1: Setup ──────────────────────────────────────────────────

        # 1a. Create async instructor client
        self._client = create_client(
            model=self._config.assignment_model, async_mode=True
        )

        # 1b. Build global facet lookup from P2 facet assignments
        self._facet_lookup: Dict[str, str] = {}
        for name, mece_res in self._mece_results.items():
            if mece_res.facet_assignments:
                self._facet_lookup.update(mece_res.facet_assignments)
        if verbose:
            print(f"  Facet lookup: {len(self._facet_lookup)} entries")

        # 1c. Group all ideas by partition (domain)
        partition_ideas = self._group_ideas_by_partition()
        total_ideas = sum(len(ideas) for ideas in partition_ideas.values())

        # 1d. Resolve codebook: __global__ → shared across all partitions
        if "__global__" in self._mece_results:
            global_mece = self._mece_results["__global__"]
        else:
            all_cats = []
            seen_labels = set()
            for mece_res in self._mece_results.values():
                if mece_res and mece_res.categories:
                    for cat in mece_res.categories:
                        if cat.category_label not in seen_labels:
                            seen_labels.add(cat.category_label)
                            all_cats.append(cat)
            total_labels = sum(
                r.n_labels for r in self._mece_results.values() if r
            )
            global_mece = DomainResultModel(
                partition_name="__global__",
                n_labels=total_labels,
                n_batches=0,
                categories=all_cats,
            )

        self._resolved_mece = {
            pname: global_mece for pname in partition_ideas
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"CATEGORY ASSIGNMENT")
            print(f"{'='*70}")
            print(f"  Ideas: {total_ideas} across {len(partition_ideas)} partitions")
            print(f"  Codebook: global")
            print(f"  Model: {self._config.assignment_model}")

        if total_ideas == 0:
            print("  WARNING: No ideas to assign")
            return self._build_output_models({}, {})

        # 1e. Pre-build ID maps
        self._build_id_maps()

        # ── Phase 2: Fetch rate limits ───────────────────────────────────────

        if verbose:
            print("  Fetching rate limits from API...")
        limits = await self._fetch_rate_limits()
        self._rate_limits = limits

        if verbose:
            print(f"  Rate limits: TPM={limits.tokens_per_minute:,}, "
                  f"RPM={limits.requests_per_minute:,}")

        # ── Phase 3: Token estimation via tiktoken (no probe calls) ──────────

        self._avg_tokens = self._estimate_avg_tokens(partition_ideas)
        if verbose:
            print(f"  Token estimate (tiktoken): {self._avg_tokens}")

        # ── Phase 4: Build task list ─────────────────────────────────────────

        task_list = []
        for partition_name in sorted(partition_ideas.keys()):
            ideas = partition_ideas[partition_name]
            mece_result = self._resolved_mece.get(partition_name)

            if not mece_result or not mece_result.categories:
                if verbose:
                    print(f"  WARNING: No MECE categories for "
                          f"'{partition_name}', skipping {len(ideas)} ideas")
                continue

            for idea_idx, idea in enumerate(ideas):
                task_list.append({
                    'idea': idea,
                    'partition_name': partition_name,
                    'batch_idx': idea_idx,
                    'n_batches': len(ideas),
                })

        total_batches = len(task_list)

        # ── Phase 5: Initialize 4-layer rate limiting ────────────────────────

        little_law_cap = self._initialize_rate_limiters(
            limits, total_batches, headroom
        )

        # Workers = initial ramp target (not static 200)
        initial_conc = self._concurrency_ramp.current_target()
        num_workers = min(total_batches, initial_conc)

        # Warm-up target: 15-30 completions
        warm_up_config = DEFAULT_WARM_UP_CONFIG
        self._warm_up_target = min(
            warm_up_config.sample_max,
            max(warm_up_config.sample_min, total_batches // 10)
        )

        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / max(self._avg_tokens, 1) / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

        if verbose:
            print(f"\n  [RATE LIMITING SETUP] — Completion-Based Ramp + PID + Circuit Breaker")
            print(f"  Model: {self._config.assignment_model}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"  Token estimate (tiktoken): {self._avg_tokens}")
            print(f"  Expected throughput: {arrival_rate:.1f}/s ({bottleneck} limited)")
            print(f"  Little's Law cap: {little_law_cap}")
            print(f"  Ramp: {initial_conc} (50%) → {self._concurrency_ramp.cap} (90%)")
            print(f"  Timeout: 60s safety net (P95×3 adaptive)")
            print(f"  Workers: {num_workers}")
            print(f"  Warm-up: calibrate after {self._warm_up_target} completions")
            print(f"  Total tasks: {total_batches}")

        # ── Phase 6: Queue + workers + main loop ─────────────────────────────

        queue = asyncio.Queue()
        results = [None] * total_batches
        timed_out = []  # (index, task) tuples

        for i, task in enumerate(task_list):
            task['result_index'] = i
            await queue.put(task)

        workers = [
            asyncio.create_task(self._worker(queue, results, timed_out))
            for _ in range(num_workers)
        ]

        start_time = time.time()
        last_report = start_time
        last_pid = start_time
        last_ramp = start_time

        while self._stats['tasks_processed'] < total_batches:
            await asyncio.sleep(0.1)
            now = time.time()
            elapsed = now - start_time
            completed = self._stats['tasks_processed']

            # Every 1s: circuit breaker + ramp
            if now - last_ramp >= 1.0:
                if self._circuit_breaker:
                    self._circuit_breaker.check_and_adjust()
                self._check_ramp_up(completed, self._stats['timeouts'], elapsed)
                last_ramp = now

            # Progress report with constraint visibility
            if now - last_report >= PROGRESS_REPORT_INTERVAL:
                await self._print_progress(
                    completed, total_batches, total_ideas, elapsed
                )
                last_report = now

            # One-shot warm-up calibration
            if (not self._warm_up_done
                    and len(self._actual_total_tokens) >= self._warm_up_target
                    and len(self._latency_tracker.values) >= self._warm_up_target):
                extra = self._calibrate_from_warm_up(limits, headroom, total_batches)
                if extra > 0 and extra > len(workers):
                    new_count = extra - len(workers)
                    for _ in range(new_count):
                        workers.append(
                            asyncio.create_task(self._worker(queue, results, timed_out))
                        )
                    if verbose:
                        print(f"  Workers: {len(workers)} (+{new_count} after calibration)")

            # Every 20s: PID adjustment
            if now - last_pid >= PID_ADJUSTMENT_INTERVAL:
                await self._apply_pid_adjustment(headroom)
                last_pid = now

        # Drain and stop workers
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        elapsed = time.time() - start_time

        # ── Phase 7: Handle timed-out tasks ──────────────────────────────────

        if timed_out and verbose:
            print(f"\n  Timed out: {len(timed_out)} tasks (fallback, no retry)")

        # ── Phase 8: Collect assignments ─────────────────────────────────────

        assignment_lookup = {}
        for i, result in enumerate(results):
            if result is None or isinstance(result, Exception):
                continue
            for assignment in result.assignments:
                assignment_lookup[assignment.idea_id] = assignment

        assigned_count = len(assignment_lookup)

        # Resolve category IDs to labels
        id_resolution: Dict[str, str] = {}
        resolve_stats = {"resolved": 0, "fallback": 0, "unresolved": 0}
        for idea_id, assignment in assignment_lookup.items():
            raw_id = getattr(assignment, 'assigned_category_id', '') or ''
            cat_id = self._normalize_id(raw_id)
            label = self._id_to_label.get(cat_id)
            if label:
                id_resolution[idea_id] = label
                resolve_stats["resolved"] += 1
            elif raw_id:
                id_resolution[idea_id] = self._other_label or ""
                resolve_stats["fallback"] += 1
            else:
                id_resolution[idea_id] = ""
                resolve_stats["unresolved"] += 1

        # Final stats
        if verbose:
            print(f"\n  Completed {total_batches} tasks in {elapsed:.1f}s")
            print(f"  - Successful: {self._stats['tasks_successful']}")
            print(f"  - Failed: {self._stats['tasks_failed']}")
            if self._stats['rate_limits']:
                print(f"  - Rate limits hit: {self._stats['rate_limits']}")
            if self._stats['timeouts']:
                print(f"  - Timeouts: {self._stats['timeouts']} (fallback)")
            print(f"  - Average: {elapsed / max(total_batches, 1):.2f}s/task")
            print(f"  - Assigned: {assigned_count}/{total_ideas} ideas")
            if self._adjustment_count > 0:
                print(f"  - Throughput adjustments: {self._adjustment_count}")

            print(f"\n  [ID RESOLUTION]")
            print(f"    Resolved: {resolve_stats['resolved']}")
            if resolve_stats['fallback']:
                print(f"    Fallback (invalid ID): {resolve_stats['fallback']}")
            if resolve_stats['unresolved']:
                print(f"    Unresolved (no ID): {resolve_stats['unresolved']}")

        if self._failure_log:
            from collections import Counter
            print(f"\n  PROCESSING ERRORS: {len(self._failure_log)} of {total_batches}")
            for reason, count in Counter(f['error_type'] for f in self._failure_log).most_common():
                print(f"    {count}x {reason}")

        output = self._build_output_models(assignment_lookup, id_resolution)
        if verbose:
            self._print_assignment_summary(output)
        return output

    # =========================================================================
    # WORKER
    # =========================================================================

    async def _worker(
        self,
        queue: asyncio.Queue,
        results: List,
        timed_out: List,
    ) -> None:
        """Worker coroutine: pulls tasks from queue, processes with retry."""
        while True:
            try:
                task = await queue.get()
                if task is None:  # Sentinel
                    break

                try:
                    result = await self._process_task_with_retry(task)
                    if result is None:
                        # Timeout — collect for fallback
                        timed_out.append((task['result_index'], task))
                    else:
                        results[task['result_index']] = result
                        self._stats['tasks_successful'] += 1
                except Exception as e:
                    error_type = type(e).__name__
                    error_str = str(e)

                    if "429" in error_str or "RateLimitReached" in error_str:
                        if "token rate limit" in error_str.lower():
                            error_type = "RateLimit_TPM"
                        elif "call rate limit" in error_str.lower():
                            error_type = "RateLimit_RPM"
                        else:
                            error_type = "RateLimit"
                        self._stats['rate_limits'] += 1

                    self._stats['tasks_failed'] += 1
                    self._failure_log.append({
                        'partition': task['partition_name'],
                        'batch_idx': task['batch_idx'],
                        'error_type': error_type,
                    })
                finally:
                    self._stats['tasks_processed'] += 1
                    queue.task_done()

            except asyncio.CancelledError:
                break

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            InstructorRetryException,
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _process_task_with_retry(
        self,
        task: Dict,
    ) -> Optional[CodeAssignmentBatch]:
        """Process a single idea with 4-layer rate limiting + timeout.

        Returns None on timeout (collected for fallback, no retry).
        """
        partition_name = task['partition_name']
        idea = task['idea']
        prompt = self._build_dual_assignment_prompt(idea)
        response_model = CodeAttributeAssignment

        # Prompt capture (first task per partition)
        _assign_key = f"assign_{partition_name}"
        if (self._prompt_printer is not None
                and _assign_key not in self._captured_assign_gates):
            self._prompt_printer.capture_prompt(
                step_name="code_assignment",
                utility_name="CodeAssigner",
                prompt_content=prompt,
                prompt_type="dual_assignment",
                metadata={
                    "model": self._config.assignment_model,
                    "language": (
                        self._extraction_metadata.lang
                        if self._extraction_metadata else "Dutch"
                    ),
                    "partition_name": partition_name,
                    "n_codes": len(self._codes),
                }
            )
            self._captured_assign_gates.add(_assign_key)

        est_tokens = self._avg_tokens

        # 4-layer rate limiting: Gate → TPM bucket → RPM limiter → Timeout
        async with self._gate:
            # Compute timeout AFTER gate (uses current latency data)
            timeout = self._latency_tracker.get_timeout(est_tokens)
            await self._tpm_bucket.wait_and_acquire(est_tokens)
            api_start = time.perf_counter()

            async with self._rate_limiter:
                try:
                    result = await asyncio.wait_for(
                        llm_create_async(
                            client=self._client,
                            model=self._config.assignment_model,
                            prompt=prompt,
                            response_model=response_model,
                            temperature=self._config.assignment_temperature,
                            max_tokens=self._config.assignment_max_tokens,
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    self._stats['timeouts'] += 1
                    if self._circuit_breaker:
                        self._circuit_breaker.record_timeout()
                    return None  # Collected for fallback

                # Record latency
                latency = time.perf_counter() - api_start
                self._latency_tracker.add(latency)

                # Circuit breaker feedback
                if self._circuit_breaker:
                    self._circuit_breaker.record_completion()

                # Track actual token usage + reconcile
                usage = getattr(result, '_raw_response', None)
                if usage:
                    usage = getattr(usage, 'usage', None)
                if not usage:
                    usage = getattr(result, 'usage', None)

                if usage:
                    input_tokens = (
                        getattr(usage, 'input_tokens', 0)
                        or getattr(usage, 'prompt_tokens', 0)
                    )
                    output_tokens = (
                        getattr(usage, 'output_tokens', 0)
                        or getattr(usage, 'completion_tokens', 0)
                    )
                    actual_total = (
                        getattr(usage, 'total_tokens', 0)
                        or (input_tokens + output_tokens)
                    )

                    self._actual_total_tokens.append(actual_total)

                    # Reconcile token bucket
                    delta = actual_total - est_tokens
                    await self._tpm_bucket.reconcile(delta)

                    # Learn tiktoken→API offset
                    tiktoken_count = len(self._encoding.encode(prompt))
                    self._tiktoken_learner.record(tiktoken_count, input_tokens)

                    # Feed PID trackers
                    if self._tpm_tracker:
                        await self._tpm_tracker.record(actual_total)
                    if self._rpm_tracker:
                        await self._rpm_tracker.record()

                # Store attribute assignment
                if result.assigned_attribute:
                    self._attribute_assignments[idea.idea_id] = result.assigned_attribute

                # Wrap into batch format
                wrapped = CodeAssignmentBatch(
                    assignments=[CodeAssignment(
                        idea_id=idea.idea_id,
                        assigned_category_id=result.assigned_code_id,
                        confidence=result.confidence,
                        rationale=result.rationale,
                    )]
                )
                return wrapped

    # =========================================================================
    # RATE LIMITING: INITIALIZATION + CALIBRATION + ADJUSTMENT
    # =========================================================================

    def _estimate_avg_tokens(self, partition_ideas: Dict) -> int:
        """Estimate avg tokens per request via tiktoken (no API calls)."""
        # Sample up to 20 ideas across partitions
        all_ideas = []
        for ideas in partition_ideas.values():
            all_ideas.extend(ideas)
        sample = all_ideas[:min(20, len(all_ideas))]

        if not sample:
            return 3000  # Conservative fallback

        token_counts = []
        for idea in sample:
            prompt = self._build_dual_assignment_prompt(idea)
            prompt_tokens = len(self._encoding.encode(prompt))
            # Estimate output at ~15% of input (structured JSON response)
            completion_tokens = int(prompt_tokens * 0.15)
            token_counts.append(prompt_tokens + completion_tokens)

        return int(statistics.mean(token_counts))

    def _initialize_rate_limiters(
        self,
        limits: RateLimits,
        num_tasks: int,
        headroom: float,
    ) -> int:
        """Set up 4-layer rate limiting with completion-based ramp.

        Returns the Little's Law cap (target concurrency).
        """
        avg_tokens = max(self._avg_tokens, 1)

        # Arrival rate from RPM and TPM
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        self._current_arrival_rate = arrival_rate

        # Layer 1: RPM (AsyncLimiter)
        self._rate_limiter = AsyncLimiter(
            1, time_period=1.0 / max(arrival_rate, 0.01)
        )

        # Layer 2: TPM (TokenBucket)
        self._tpm_bucket = TokenBucket(int(limits.tokens_per_minute * headroom))

        # Little's Law cap
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law_cap = compute_optimal_concurrency(
            api_limits, DEFAULT_LATENCY_SECONDS, avg_tokens,
            headroom=headroom,
        )

        # Layer 3: Concurrency (ConcurrencyGate + ramp)
        initial_conc = max(5, int(little_law_cap * DEFAULT_RAMP_UP_CONFIG.start_fraction))
        self._gate = ConcurrencyGate(initial_conc)
        self._concurrency_ramp = ConcurrencyRamp(
            DEFAULT_RAMP_UP_CONFIG, little_law_cap, num_tasks
        )

        # Layer 4: Circuit breaker
        self._circuit_breaker = ConcurrencyCircuitBreaker(
            DEFAULT_CIRCUIT_BREAKER_CONFIG, self._gate, initial_conc
        )

        # PID components
        self._tpm_tracker = RealTimeTPMTracker(
            window_seconds=DEFAULT_TPM_TRACKING_CONFIG.sliding_window_seconds
        )
        self._rpm_tracker = RealTimeRPMTracker(window_seconds=60.0)
        self._pid_controller = PIDThroughputController(
            target_utilization=DEFAULT_TPM_TRACKING_CONFIG.target_utilization,
            kp_up=DEFAULT_PID_CONTROLLER_CONFIG.kp_up,
            kp_down=DEFAULT_PID_CONTROLLER_CONFIG.kp_down,
            ki=DEFAULT_PID_CONTROLLER_CONFIG.ki,
            kd=DEFAULT_PID_CONTROLLER_CONFIG.kd,
            min_adjustment=DEFAULT_PID_CONTROLLER_CONFIG.min_adjustment,
            max_adjustment=DEFAULT_PID_CONTROLLER_CONFIG.max_adjustment,
        )

        # Latency + tiktoken offset
        self._latency_tracker = LatencyTracker()
        self._tiktoken_learner = TiktokenOffsetLearner()

        return little_law_cap

    def _calibrate_from_warm_up(
        self,
        limits: RateLimits,
        headroom: float,
        num_tasks: int,
    ) -> int:
        """One-shot calibration from first N real completions.

        Returns new worker target (for spawning extra workers if needed).
        """
        self._warm_up_done = True

        # Measured values
        actual_avg_tokens = int(np.mean(list(self._actual_total_tokens)))
        p10_latency = float(np.percentile(list(self._latency_tracker.values), 10))

        # Recalculate Little's Law with measured data
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        new_cap = compute_optimal_concurrency(
            api_limits, p10_latency, actual_avg_tokens,
            headroom=headroom,
        )

        # Update token estimate
        old_avg = self._avg_tokens
        self._avg_tokens = actual_avg_tokens

        # Recalibrate ramp
        self._concurrency_ramp.recalibrate(new_cap)

        # Update circuit breaker baseline
        new_initial = self._concurrency_ramp.current_target()
        self._circuit_breaker.baseline = new_initial

        # Recalculate arrival rate
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / max(actual_avg_tokens, 1) / 60
        new_arrival_rate = min(rpm_throughput, tpm_throughput)
        self._current_arrival_rate = new_arrival_rate
        self._rate_limiter = AsyncLimiter(
            1, time_period=1.0 / max(new_arrival_rate, 0.01)
        )

        # Reset PID
        self._pid_controller.reset()

        if self._config.verbose:
            print(f"\n  [WARM-UP CALIBRATION]")
            print(f"    Tokens: {old_avg} → {actual_avg_tokens} (measured)")
            print(f"    Latency P10: {p10_latency:.2f}s")
            print(f"    Little's Law: {new_cap}")
            print(f"    Arrival rate: {new_arrival_rate:.1f}/s")

        # Return new ramp target for worker scaling
        return min(num_tasks, self._concurrency_ramp.cap)

    def _check_ramp_up(self, completions: int, timeouts: int, elapsed: float):
        """Check and advance completion-based concurrency ramp."""
        if self._concurrency_ramp is None or self._concurrency_ramp.is_done():
            return

        rate = completions / elapsed if elapsed > 0 else 0

        self._concurrency_ramp.record_measurement(
            throughput=rate,
            tpm_pct=0,  # PID handles TPM, ramp uses throughput
            rpm_pct=0,
            completions_total=completions,
            timeouts_total=timeouts,
            duration=elapsed,
        )

        new_target = self._concurrency_ramp.current_target()
        if new_target != self._gate.limit:
            self._gate.set_limit(new_target)

    async def _apply_pid_adjustment(self, headroom: float):
        """PID-based arrival rate adjustment using real-time TPM utilization."""
        if not self._tpm_tracker or not self._rate_limits:
            return

        current_tpm = await self._tpm_tracker.get_current_tpm()
        tpm_limit = self._rate_limits.tokens_per_minute * headroom
        if tpm_limit <= 0:
            return

        utilization = current_tpm / tpm_limit
        adjustment = self._pid_controller.compute_adjustment(utilization)

        if adjustment != 1.0:
            self._current_arrival_rate *= adjustment
            self._rate_limiter = AsyncLimiter(
                1, time_period=1.0 / max(self._current_arrival_rate, 0.01)
            )
            self._adjustment_count += 1

    async def _print_progress(
        self,
        completed: int,
        total: int,
        total_ideas: int,
        elapsed: float,
    ):
        """Print progress with constraint visibility (TPM%, RPM%, Conc%)."""
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = total - completed
        eta_s = remaining / rate if rate > 0 else 0
        eta_str = f"{eta_s / 60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"

        # Constraint utilization
        tpm_pct = rpm_pct = conc_pct = 0.0
        if self._tpm_tracker and self._rate_limits:
            current_tpm = await self._tpm_tracker.get_current_tpm()
            tpm_limit = self._rate_limits.tokens_per_minute * DEFAULT_PROCESSING_CONFIG.rate_limit_headroom
            tpm_pct = (current_tpm / tpm_limit * 100) if tpm_limit > 0 else 0
        if self._rpm_tracker and self._rate_limits:
            current_rpm = await self._rpm_tracker.get_current_rpm()
            rpm_limit = self._rate_limits.requests_per_minute * DEFAULT_PROCESSING_CONFIG.rate_limit_headroom
            rpm_pct = (current_rpm / rpm_limit * 100) if rpm_limit > 0 else 0
        if self._gate:
            conc_pct = (self._gate.active / max(self._gate.limit, 1) * 100)

        cb_state = self._circuit_breaker.state if self._circuit_breaker else "N/A"

        failed_str = (
            f" Failed:{self._stats['tasks_failed']}"
            if self._stats['tasks_failed'] else ""
        )
        deferred_str = (
            f" Deferred:{self._stats['timeouts']}"
            if self._stats['timeouts'] else ""
        )

        print(
            f"  Progress: {completed}/{total} ({completed/total*100:.1f}%) "
            f"Rate: {rate:.0f}/s | "
            f"TPM:{tpm_pct:.0f}% RPM:{rpm_pct:.0f}% "
            f"Conc:{self._gate.active}/{self._gate.limit}({conc_pct:.0f}%) "
            f"CB:{cb_state} "
            f"ETA: {eta_str}{failed_str}{deferred_str}"
        )

    # =========================================================================
    # ID-BASED RESOLUTION
    # =========================================================================

    _RE_NORMALIZE_ID = re.compile(r'\s+')

    @staticmethod
    def _normalize_id(raw_id: str) -> str:
        """Normalize a raw category ID: 'c7' -> 'C7', '7' -> 'C7'."""
        cat_id = CodeAssigner._RE_NORMALIZE_ID.sub('', raw_id.strip().upper())
        if not cat_id.startswith('C') and cat_id.isdigit():
            cat_id = f"C{cat_id}"
        return cat_id

    def _build_id_maps(self) -> None:
        """Build ID-to-label and ID-to-parent maps for all leaf categories.

        Uses depth-first traversal matching the prompt numbering order.
        Populates self._id_to_label, self._id_to_parent, self._other_id,
        self._other_label.
        """
        id_to_label: Dict[str, str] = {}
        id_to_parent: Dict[str, str] = {}
        counter = [0]

        def _walk(cats: List[MECECode], parent_label: Optional[str] = None):
            for cat in cats:
                if cat.subcategories:
                    _walk(cat.subcategories, cat.category_label)
                else:
                    counter[0] += 1
                    cat_id = f"C{counter[0]}"
                    id_to_label[cat_id] = cat.category_label
                    if parent_label:
                        id_to_parent[cat_id] = parent_label

        # Use first resolved partition (all share same global codebook)
        for mece_res in self._resolved_mece.values():
            if mece_res and mece_res.categories:
                _walk(mece_res.categories)
                break

        # Add "other" category as final entry
        if self._config.include_other_category:
            language = "Dutch"
            if self._extraction_metadata:
                language = getattr(self._extraction_metadata, 'lang', 'Dutch') or 'Dutch'
            other_label = get_other_category_label(language)
            counter[0] += 1
            other_id = f"C{counter[0]}"
            id_to_label[other_id] = other_label
            self._other_id = other_id
            self._other_label = other_label
        else:
            self._other_id = None
            self._other_label = None

        self._id_to_label = id_to_label
        self._id_to_parent = id_to_parent

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_dual_assignment_prompt(self, idea) -> str:
        """Build prompt for dual assignment (code + attribute) using raw codes."""
        survey_question = ""
        language = "Dutch"
        dataset_context_section = ""

        if self._extraction_metadata:
            survey_question = self._extraction_metadata.var_lab or ""
            language = self._extraction_metadata.lang or "Dutch"
            parts = []
            for f in ('domain', 'entity', 'topic', 'perspective', 'intent'):
                val = getattr(self._extraction_metadata, f, None)
                if val:
                    parts.append(f"{f.capitalize()}: {val}")
            if parts:
                dataset_context_section = "\n".join(parts)

        other_label = get_other_category_label(language)

        return build_single_dual_assignment_prompt(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            codes=self._codes,
            other_label=other_label if self._config.include_other_category else None,
            idea=idea,
            facet_lookup=self._facet_lookup,
        )

    # =========================================================================
    # IDEA GROUPING & BATCHING
    # =========================================================================

    def _group_ideas_by_partition(
        self,
    ) -> Dict[str, List[models.IdeasExtractedSubmodel]]:
        """Group all ideas by their domain (= partition)."""
        partitions: Dict[str, List] = {}
        for resp in self._ideas_models:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                ct = self._normalize_key(idea.domain)
                if not ct:
                    continue
                if ct not in partitions:
                    partitions[ct] = []
                partitions[ct].append(idea)
        return partitions

    # =========================================================================
    # OUTPUT MODEL CONSTRUCTION
    # =========================================================================

    def _build_output_models(
        self,
        assignment_lookup: dict,
        id_resolution: Dict[str, str],
    ) -> List[CodeAssignedModel]:
        """Build CodeAssignedModel list preserving response structure."""
        facet_lookup = self._facet_lookup

        output = []
        for resp in self._ideas_models:
            new_ideas = []
            if resp.response_ideas:
                for idea in resp.response_ideas:
                    assignment = assignment_lookup.get(idea.idea_id)
                    ct = self._normalize_key(idea.domain)

                    resolved_label = id_resolution.get(idea.idea_id)

                    # Parent lookup via ID
                    parent_cat = None
                    if assignment:
                        raw_id = getattr(assignment, 'assigned_category_id', '') or ''
                        cat_id = self._normalize_id(raw_id)
                        parent_cat = self._id_to_parent.get(cat_id)

                    idea_data = idea.model_dump()
                    explicit_fields = {
                        'assigned_category', 'category_confidence',
                        'category_rationale', 'assigned_attribute',
                        'partition_name', 'parent_category', 'facet',
                    }
                    new_idea = CodeAssignedSubmodel(
                        **{k: v for k, v in idea_data.items()
                           if k in CodeAssignedSubmodel.model_fields
                           and k not in explicit_fields},
                        assigned_category=(
                            resolved_label or None
                        ),
                        category_confidence=(
                            assignment.confidence
                            if assignment else None
                        ),
                        category_rationale=(
                            assignment.rationale
                            if assignment else None
                        ),
                        assigned_attribute=(
                            self._attribute_assignments.get(idea.idea_id)
                        ),
                        partition_name=ct if ct else None,
                        parent_category=parent_cat,
                        facet=facet_lookup.get(idea.idea_id, idea_data.get('facet', '')),
                    )
                    new_ideas.append(new_idea)

            resp_data = resp.model_dump()
            new_resp = CodeAssignedModel(
                **{k: v for k, v in resp_data.items()
                   if k in CodeAssignedModel.model_fields
                   and k != 'response_ideas'},
                response_ideas=new_ideas,
            )
            output.append(new_resp)

        return output

    # =========================================================================
    # RATE LIMIT HELPERS
    # =========================================================================

    async def _fetch_rate_limits(self) -> RateLimits:
        """Fetch rate limits from API headers."""
        from openai import AsyncOpenAI

        if API_PROVIDER == "azure":
            from config import (
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_DEPLOYMENT_NAME,
            )
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=(
                    f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
                    f"{AZURE_OPENAI_DEPLOYMENT_NAME}/"
                ),
                default_query={"api-version": "2024-10-21"},
            )
            model = AZURE_OPENAI_DEPLOYMENT_NAME
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self._config.assignment_model

        response = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        limits = extract_rate_limits_from_response(response)

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if self._config.verbose:
                print(f"  WARNING: Using fallback rate limits "
                      f"(TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            return RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
            )
        return limits


    # =========================================================================
    # REPORTING
    # =========================================================================

    @staticmethod
    def _print_assignment_summary(
        output: List[CodeAssignedModel],
    ):
        """Print per-partition assignment summary, grouped by parent when available."""
        partition_stats: Dict[str, Dict] = {}

        for resp in output:
            if not resp.response_ideas:
                continue
            for idea in resp.response_ideas:
                pt = idea.partition_name or "(unknown)"
                if pt not in partition_stats:
                    partition_stats[pt] = {
                        "total": 0,
                        "assigned": 0,
                        "confidences": [],
                        "categories": {},
                        "parent_groups": {},
                        "has_parents": False,
                    }
                stats = partition_stats[pt]
                stats["total"] += 1
                if idea.assigned_category:
                    stats["assigned"] += 1
                    stats["confidences"].append(
                        idea.category_confidence or 0.0
                    )
                    cat = idea.assigned_category
                    stats["categories"][cat] = (
                        stats["categories"].get(cat, 0) + 1
                    )
                    if idea.parent_category:
                        stats["has_parents"] = True
                        parent = idea.parent_category
                        if parent not in stats["parent_groups"]:
                            stats["parent_groups"][parent] = {}
                        stats["parent_groups"][parent][cat] = (
                            stats["parent_groups"][parent].get(cat, 0) + 1
                        )

        print(f"\n  {'─'*60}")
        print(f"  ASSIGNMENT SUMMARY")
        print(f"  {'─'*60}")

        for pt in sorted(partition_stats.keys()):
            stats = partition_stats[pt]
            avg_conf = (
                sum(stats["confidences"]) / len(stats["confidences"])
                if stats["confidences"] else 0.0
            )
            print(f"\n  Partition: {pt}")
            print(f"    Assigned: {stats['assigned']}/{stats['total']}")
            print(f"    Avg confidence: {avg_conf:.2f}")

            if stats["has_parents"] and stats["parent_groups"]:
                print(f"    Categories ({len(stats['categories'])}):")
                parent_totals = {
                    p: sum(cats.values())
                    for p, cats in stats["parent_groups"].items()
                }
                sorted_parents = sorted(
                    parent_totals, key=lambda p: -parent_totals[p]
                )
                for parent in sorted_parents:
                    print(f"      {parent} ({parent_totals[parent]}):")
                    children = stats["parent_groups"][parent]
                    for cat, count in sorted(
                        children.items(), key=lambda x: -x[1]
                    ):
                        print(f"        {cat}: {count}")
                # Orphan categories (e.g., "overig/anders" — no parent)
                parented_cats = set()
                for children in stats["parent_groups"].values():
                    parented_cats.update(children.keys())
                orphans = {
                    c: n for c, n in stats["categories"].items()
                    if c not in parented_cats
                }
                if orphans:
                    for cat, count in sorted(
                        orphans.items(), key=lambda x: -x[1]
                    ):
                        print(f"      {cat}: {count}")
            else:
                print(f"    Categories ({len(stats['categories'])}):")
                for cat, count in sorted(
                    stats["categories"].items(),
                    key=lambda x: -x[1],
                ):
                    print(f"      {cat}: {count}")
