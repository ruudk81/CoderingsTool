"""
Dual Assignment: Code + Attribute per Idea.

Assigns each idea to exactly one MECE code and its best-matching attribute
via a single LLM call. All partitions are processed concurrently with shared
rate limiting.

Pipeline:
  1. Group ideas by partition (domain)
  2. Bootstrap: fetch rate limits + probe calls for latency/token measurement
  3. Little's Law → optimal concurrency
  4. Queue + workers: process single ideas with TokenBucket + rate limiter + retry
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
from collections import deque
from typing import Dict, List, Optional

import nest_asyncio
from aiolimiter import AsyncLimiter
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from instructor.exceptions import InstructorRetryException

from utils.llm import (
    create_client, llm_create_async, ProbeResponse,
    RateLimits, extract_rate_limits_from_response,
)
from config import (
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG,
    OPENAI_API_KEY, API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
)

from development.step_3_ideaExtractor import models_exp as models
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

# Reuse bootstrap utilities from qualitative_researcher
from .qualitative_researcher import (
    ApiLimits, compute_optimal_concurrency, bootstrap_measure_async,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()

logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries during retries
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.ERROR)

# === PROCESSING CONSTANTS ===================================================

PROGRESS_REPORT_INTERVAL = 5           # Seconds between progress reports
THROUGHPUT_ADJUSTMENT_THRESHOLD = 1.1  # Trigger when actual > 110% of estimate
THROUGHPUT_ADJUSTMENT_MIN_SAMPLES = 10 # Min data points before adjusting
ADJUSTMENT_INTERVAL = 15               # Seconds between adjustment checks
MAX_TOKEN_ACQUIRE_ATTEMPTS = 1000      # Max attempts to acquire tokens
ERROR_WINDOW_SIZE = 100                # Rolling window for token tracking


# === TOKEN BUCKET ===========================================================

class TokenBucket:
    """Token bucket for TPM limiting. Ported from qualityFilter.py."""

    def __init__(self, tokens_per_minute: int):
        self.tpm = tokens_per_minute
        self.available = float(tokens_per_minute)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens_needed: int):
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
                return deficit * 60 / self.tpm  # wait_seconds

    async def wait_and_acquire(self, tokens_needed: int):
        attempts = 0
        while attempts < MAX_TOKEN_ACQUIRE_ATTEMPTS:
            attempts += 1
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            else:
                await asyncio.sleep(result)
        raise RuntimeError(
            f"Failed to acquire {tokens_needed} tokens "
            f"after {MAX_TOKEN_ACQUIRE_ATTEMPTS} attempts"
        )

    async def reconcile(self, delta_tokens: int):
        """Return overestimated tokens to bucket (negative delta = overestimate)."""
        if delta_tokens < 0:
            async with self.lock:
                self.available = min(self.tpm, self.available - delta_tokens)


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
        self._semaphore = None
        self._rate_limiter = None
        self._tpm_bucket = None

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

        # Token tracking for throughput adjustment
        self._actual_total_tokens: deque = deque(maxlen=ERROR_WINDOW_SIZE)
        self._avg_tokens: int = 0
        self._bootstrap_avg_tokens: int = 0
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
        """Main async orchestration: bootstrap, queue+workers, collect."""
        verbose = self._config.verbose
        processing_config = DEFAULT_PROCESSING_CONFIG
        headroom = processing_config.rate_limit_headroom

        # 1. Create async instructor client
        self._client = create_client(
            model=self._config.assignment_model, async_mode=True
        )

        # 2. Build global facet lookup from P2 facet assignments
        self._facet_lookup: Dict[str, str] = {}
        for mece_res in self._mece_results.values():
            if mece_res.facet_assignments:
                self._facet_lookup.update(mece_res.facet_assignments)

        # 3. Group all ideas by partition (domain)
        partition_ideas = self._group_ideas_by_partition()
        total_ideas = sum(len(ideas) for ideas in partition_ideas.values())

        # Resolve codebook: __global__ → shared across all partitions
        if "__global__" in self._mece_results:
            global_mece = self._mece_results["__global__"]
        else:
            # Legacy cache: per-partition keys. Merge all categories into
            # a single global codebook (deduplicated by category_label).
            all_cats = []
            seen_labels = set()
            for mece_res in self._mece_results.values():
                if mece_res and mece_res.categories:
                    for cat in mece_res.categories:
                        if cat.category_label not in seen_labels:
                            seen_labels.add(cat.category_label)
                            all_cats.append(cat)
            total_labels = sum(
                r.n_labels for r in self._mece_results.values()
                if r
            )
            global_mece = DomainResultModel(
                partition_name="__global__",
                n_labels=total_labels,
                n_batches=0,
                categories=all_cats,
            )
            if verbose:
                print(f"  Legacy cache: merged {len(self._mece_results)} "
                      f"partition codebooks → {len(all_cats)} global categories")

        # Broadcast global codebook to all idea partitions
        self._resolved_mece = {
            pname: global_mece for pname in partition_ideas
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"CATEGORY ASSIGNMENT")
            print(f"{'='*70}")
            codebook_mode = "global"
            print(f"  Ideas: {total_ideas} across {len(partition_ideas)} partitions")
            print(f"  Codebook: {codebook_mode}")
            print(f"  Model: {self._config.assignment_model}")

        if total_ideas == 0:
            print("  WARNING: No ideas to assign")
            return self._build_output_models({}, {})

        # 3. Fetch rate limits from API
        if verbose:
            print("  Fetching rate limits from API...")
        limits = await self._fetch_rate_limits()

        if verbose:
            print(f"  Rate limits: TPM={limits.tokens_per_minute:,}, "
                  f"RPM={limits.requests_per_minute:,}")

        # 4. Pre-build ID maps
        self._build_id_maps()

        # 5. Bootstrap measurement (3 probe calls)
        first_partition = next(iter(sorted(partition_ideas.keys())))
        first_idea = partition_ideas[first_partition][0]
        probe_prompt = self._build_dual_assignment_prompt(first_idea)

        if verbose:
            print("  Running bootstrap measurement (3 probe calls)...")

        probe_start = time.time()

        async def probe_fn():
            return await self._probe_call(probe_prompt)

        avg_latency, avg_tokens = await bootstrap_measure_async(
            probe_fn, n_probes=3
        )

        self._avg_tokens = int(avg_tokens)
        self._bootstrap_avg_tokens = self._avg_tokens

        if verbose:
            print(f"  Probe time: {time.time() - probe_start:.1f}s")
            print(f"  Bootstrap: {avg_latency:.2f}s avg latency, "
                  f"{avg_tokens:.0f} avg tokens")

        # 6. Little's Law → optimal concurrency
        api_limits = ApiLimits(
            limits.tokens_per_minute, limits.requests_per_minute
        )
        little_law_conc = compute_optimal_concurrency(
            api_limits, avg_latency, avg_tokens,
            processing_config=processing_config,
            cap=processing_config.concurrency_cap_permissive,
            min_conc=processing_config.concurrency_min_permissive,
        )

        max_concurrency = processing_config.concurrency_cap_default
        adaptive_min = min(
            processing_config.concurrency_min_default,
            max(little_law_conc * 3, 5),
        )
        optimal = min(max_concurrency, max(little_law_conc, adaptive_min))

        # Arrival rate for rate limiter
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"
        self._current_arrival_rate = arrival_rate

        # 7. Build flat task list across all partitions
        task_list = []  # List of dicts with batch/idea info
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

        # 8. Initialize shared rate-limiting resources
        self._semaphore = asyncio.Semaphore(min(total_batches, optimal))
        self._rate_limiter = AsyncLimiter(
            1, time_period=1.0 / max(arrival_rate, 0.01)
        )
        self._tpm_bucket = TokenBucket(
            int(limits.tokens_per_minute * headroom)
        )
        self._rate_limits = limits

        # Workers: 2x optimal concurrency, at least 10
        num_workers = min(200, max(10, optimal * 2))

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Model: {self._config.assignment_model}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"  Bootstrap avg_tokens: {self._avg_tokens}")
            print(f"  Expected throughput: {arrival_rate:.1f}/s "
                  f"({bottleneck} limited)")
            print(f"  Optimal by Little's Law: {little_law_conc}")
            print(f"  Concurrency (semaphore): {min(total_batches, optimal)}")
            print(f"  Workers: {num_workers}")
            print(f"  Total batches: {total_batches}")

        # 9. Queue + workers execution
        queue = asyncio.Queue()
        results = [None] * total_batches

        for i, task in enumerate(task_list):
            task['result_index'] = i
            await queue.put(task)

        workers = [
            asyncio.create_task(self._worker(queue, results))
            for _ in range(num_workers)
        ]

        # Progress monitoring loop
        start_time = time.time()
        last_report = start_time
        last_adjustment = start_time

        while not queue.empty() or self._stats['tasks_processed'] < total_batches:
            await asyncio.sleep(1)
            now = time.time()

            # Throughput adjustment check
            if now - last_adjustment >= ADJUSTMENT_INTERVAL:
                self._adjust_throughput_if_needed(limits, headroom)
                last_adjustment = now

            # Progress report
            if now - last_report >= PROGRESS_REPORT_INTERVAL:
                completed = self._stats['tasks_processed']
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = total_batches - completed

                # ETA calculation
                if rate > 0:
                    eta_s = remaining / rate
                    if eta_s >= 60:
                        eta_str = f"{eta_s / 60:.1f}m"
                    else:
                        eta_str = f"{eta_s:.0f}s"
                else:
                    eta_str = "?"

                ideas_done = sum(
                    len(task_list[i]['ideas'])
                    for i in range(total_batches)
                    if results[i] is not None
                )

                failed_str = (
                    f", Failed: {self._stats['tasks_failed']}"
                    if self._stats['tasks_failed'] else ""
                )
                print(
                    f"  Progress: {completed}/{total_batches} batches "
                    f"({ideas_done}/{total_ideas} ideas), "
                    f"Rate: {rate:.1f}/s, "
                    f"ETA: {eta_str}{failed_str}"
                )
                last_report = now

            # Exit early if all tasks processed
            if self._stats['tasks_processed'] >= total_batches:
                break

        # Wait for queue to drain fully
        await queue.join()

        # Stop workers with sentinels
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        elapsed = time.time() - start_time

        # 10. Collect assignments into lookup: idea_id → CodeAssignment
        assignment_lookup = {}
        failed_count = self._stats['tasks_failed']

        for i, result in enumerate(results):
            if result is None:
                failed_count += 1
                continue
            if isinstance(result, Exception):
                failed_count += 1
                continue
            for assignment in result.assignments:
                assignment_lookup[assignment.idea_id] = assignment

        assigned_count = len(assignment_lookup)

        # 10b. Resolve category IDs to labels
        id_resolution: Dict[str, str] = {}  # idea_id → resolved label
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
            print(f"\n  Completed {total_batches} batches in {elapsed:.1f}s")
            print(f"  - Successful: {self._stats['tasks_successful']}")
            print(f"  - Failed: {self._stats['tasks_failed']}")
            if self._stats['rate_limits']:
                print(f"  - Rate limits hit: {self._stats['rate_limits']}")
            if self._stats['timeouts']:
                print(f"  - Timeouts: {self._stats['timeouts']}")
            print(f"  - Average: {elapsed / max(total_batches, 1):.2f}s/batch")
            print(f"  - Assigned: {assigned_count}/{total_ideas} ideas")
            if self._adjustment_count > 0:
                print(f"  - Throughput adjustments: {self._adjustment_count}")
                print(f"    Bootstrap avg_tokens: {self._bootstrap_avg_tokens}, "
                      f"Final avg_tokens: {self._avg_tokens}")

        # ID resolution diagnostics
        if verbose:
            print(f"\n  [ID RESOLUTION]")
            print(f"    Resolved: {resolve_stats['resolved']}")
            if resolve_stats['fallback']:
                print(f"    Fallback (invalid ID): {resolve_stats['fallback']}")
            if resolve_stats['unresolved']:
                print(f"    Unresolved (no ID): {resolve_stats['unresolved']}")

        # Failure report
        if self._failure_log:
            print(f"\n  PROCESSING ERRORS: {len(self._failure_log)} of "
                  f"{total_batches} batches")
            from collections import Counter
            reason_counts = Counter(
                f['error_type'] for f in self._failure_log
            )
            for reason, count in reason_counts.most_common():
                print(f"    {count}x {reason}")

        # 11. Build output models
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
    ) -> None:
        """Worker coroutine: pulls tasks from queue, processes with retry."""
        while True:
            try:
                task = await queue.get()
                if task is None:  # Sentinel
                    break

                try:
                    result = await self._process_batch_with_retry(task)
                    results[task['result_index']] = result
                    self._stats['tasks_successful'] += 1
                except Exception as e:
                    error_type = type(e).__name__
                    error_str = str(e)

                    # Classify error
                    if "429" in error_str or "RateLimitReached" in error_str:
                        if "token rate limit" in error_str.lower():
                            error_type = "RateLimit_TPM"
                        elif "call rate limit" in error_str.lower():
                            error_type = "RateLimit_RPM"
                        else:
                            error_type = "RateLimit"
                        self._stats['rate_limits'] += 1
                    elif isinstance(e, asyncio.TimeoutError):
                        error_type = "Timeout"
                        self._stats['timeouts'] += 1

                    self._stats['tasks_failed'] += 1
                    self._failure_log.append({
                        'partition': task['partition_name'],
                        'batch_idx': task['batch_idx'],
                        'error_type': error_type,
                    })

                    if self._config.verbose:
                        print(f"    FAILED: {task['partition_name']} batch "
                              f"{task['batch_idx']+1}/{task['n_batches']}: "
                              f"{error_type}")
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
            asyncio.TimeoutError,
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _process_batch_with_retry(
        self,
        task: Dict,
    ) -> CodeAssignmentBatch:
        """Process a single idea with rate limiting, token bucket, and retry.

        Performs dual assignment (code + attribute) for one idea, then wraps
        the result in a CodeAssignmentBatch for uniform downstream handling.
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

        # Estimate tokens for this call
        est_tokens = self._avg_tokens

        # Rate limiting: semaphore → token bucket → rate limiter
        async with self._semaphore:
            await self._tpm_bucket.wait_and_acquire(est_tokens)
            async with self._rate_limiter:
                result = await llm_create_async(
                    client=self._client,
                    model=self._config.assignment_model,
                    prompt=prompt,
                    response_model=response_model,
                    temperature=self._config.assignment_temperature,
                    max_tokens=self._config.assignment_max_tokens,
                )

                # Track actual token usage for learning + reconciliation
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

                # Store attribute assignment
                if result.assigned_attribute:
                    self._attribute_assignments[idea.idea_id] = result.assigned_attribute

                # Wrap into batch format for uniform downstream handling
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
    # THROUGHPUT ADJUSTMENT
    # =========================================================================

    def _adjust_throughput_if_needed(
        self,
        limits: RateLimits,
        headroom: float,
    ) -> bool:
        """Threshold-based throughput adjustment when actual tokens exceed bootstrap."""
        if len(self._actual_total_tokens) < THROUGHPUT_ADJUSTMENT_MIN_SAMPLES:
            return False

        actual_avg = sum(self._actual_total_tokens) / len(self._actual_total_tokens)
        bootstrap_avg = self._avg_tokens
        ratio = actual_avg / bootstrap_avg if bootstrap_avg > 0 else 1.0

        if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
            return False

        # Recalculate arrival rate with actual tokens
        rpm_throughput = limits.requests_per_minute * headroom / 60
        new_tpm_throughput = limits.tokens_per_minute * headroom / actual_avg / 60
        new_arrival_rate = min(rpm_throughput, new_tpm_throughput)

        # Reinstall rate limiter
        self._rate_limiter = AsyncLimiter(
            1, time_period=1.0 / max(new_arrival_rate, 0.01)
        )

        # Reset token bucket
        self._tpm_bucket = TokenBucket(int(limits.tokens_per_minute * headroom))

        old_avg = self._avg_tokens
        old_rate = self._current_arrival_rate
        self._avg_tokens = int(actual_avg)
        self._current_arrival_rate = new_arrival_rate
        self._adjustment_count += 1

        print(f"\n  >> THROUGHPUT ADJUSTMENT #{self._adjustment_count}")
        print(f"     Actual tokens ({actual_avg:.0f}) exceeded estimate "
              f"({bootstrap_avg:.0f}) by {(ratio-1)*100:.0f}%")
        print(f"     Arrival rate: {old_rate:.2f}/s -> {new_arrival_rate:.2f}/s")
        print(f"     avg_tokens: {old_avg} -> {self._avg_tokens}")

        return True

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
                    new_idea = CodeAssignedSubmodel(
                        **{k: v for k, v in idea_data.items()
                           if k in CodeAssignedSubmodel.model_fields},
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

    async def _probe_call(self, prompt: str):
        """Probe call for bootstrap measurement — returns usage dict."""
        resp = await llm_create_async(
            client=self._client,
            model=self._config.assignment_model,
            prompt=prompt,
            response_model=ProbeResponse,
            temperature=self._config.assignment_temperature,
            track_usage=False,
        )
        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        input_tokens = (
            getattr(u, "input_tokens", 0)
            or getattr(u, "prompt_tokens", 0)
        )
        output_tokens = (
            getattr(u, "output_tokens", 0)
            or getattr(u, "completion_tokens", 0)
        )
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }

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
