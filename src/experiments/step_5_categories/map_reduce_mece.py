"""
Map-Reduce MECE engine for Theme Discovery.

Per-partition theme extraction via 3-step LLM pipeline with partition
context and optional pre-cluster hints.

All partitions are processed CONCURRENTLY. All LLM calls share a single
semaphore and rate limiter for efficient throughput control.

Pipeline for each partition:
  1. MAP:    Batch labels, identify candidate themes per batch
  2. REDUCE: Synthesize themes across batches (skipped for single-batch)
  3. MECE:   Apply MECE constraints with boundary criteria

All prompts include {cluster_hints} (empty for Mode A, populated for Mode B).
Operates on label strings (not idea text).

Usage:
    from .map_reduce_mece import MapReduceMECE
    from .config_categories_exp import CategoriesConfig

    processor = MapReduceMECE(config)
    results = processor.process_all_partitions(
        label_mappings={"identity": mapping, ...},
        partition_set=partition_set,
        ...
    )
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import nest_asyncio
from aiolimiter import AsyncLimiter

from utils.llm import create_client, llm_create_async, ProbeResponse, RateLimits, extract_rate_limits_from_response
from config import (
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG, OPENAI_API_KEY,
    API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
)

from .config_categories_exp import CategoriesConfig
from .partition_discoverer import PartitionLabelMapping
from .partition_labels import PreclusterResult, build_cluster_hints
from .prompts_exp import (
    MAP_CATEGORIES_PROMPT,
    REDUCE_THEMES_PROMPT,
    MECE_BOUNDARIES_PROMPT,
    MapBatchThemes,
    SynthesizedThemeList,
    MECECategorySet,
    MECECategory,
    MECEVerification,
    PartitionSet,
    PartitionDescription,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


# =============================================================================
# BOOTSTRAP UTILITIES (adapted from qualityFilter_exp.py)
# =============================================================================

@dataclass
class ApiLimits:
    """API limits structure for bootstrap calculations."""
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(
    limits: ApiLimits,
    latency_seconds: float,
    avg_tokens: float,
    processing_config: Optional[ProcessingConfig] = None,
    cap: Optional[int] = None,
    min_conc: Optional[int] = None,
    headroom: Optional[float] = None,
) -> int:
    """Compute optimal concurrency using Little's Law."""
    config = processing_config or DEFAULT_PROCESSING_CONFIG
    cap = cap if cap is not None else config.concurrency_cap_default
    min_conc = min_conc if min_conc is not None else config.concurrency_min_default
    headroom = headroom if headroom is not None else config.rate_limit_headroom

    latency_seconds = max(float(latency_seconds or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)

    rpm_throughput = limits.requests_per_minute * headroom / 60
    tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
    allowed_rps = max(min(rpm_throughput, tpm_throughput), 0.0)
    target = allowed_rps * latency_seconds  # Little's Law

    return int(max(min(target, cap), min_conc))


async def bootstrap_measure_async(call_fn, n_probes: int = 3):
    """Run n_probes serial calls and return (avg_latency_s, avg_tokens)."""
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


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class PromptContext:
    """Shared context passed to all prompt formatting methods."""
    survey_question: str
    language: str
    dataset_context_section: str
    taxonomy_context: str


@dataclass
class PartitionContext:
    """Partition-specific context injected into all prompts."""
    partition_name: str
    partition_inclusion: str
    partition_boundary_test: str
    peer_partitions_list: str
    grouping_instruction: str = ""
    cluster_hints: str = ""


@dataclass
class PartitionMECEResult:
    """Complete map-reduce MECE result for a single partition."""
    partition_name: str
    n_labels: int
    n_batches: int
    reduce_skipped: bool
    categories: List[MECECategory]
    mece_verifications: List[MECEVerification] = field(default_factory=list)


class MapReduceMECE:
    """
    Per-partition Map-Reduce MECE theme extractor with partition context.

    All partitions are processed concurrently. All LLM calls share a single
    async client, semaphore, and rate limiter for efficient throughput.

    Three-step pipeline for each partition:
    1. MAP:    Batch labels, identify candidate themes (async)
    2. REDUCE: Synthesize themes across batches (async)
    3. MECE:   Apply MECE boundaries with self-verification (async)
    """

    def __init__(self, config: CategoriesConfig, prompt_printer=None):
        self._model = config.mapreduce_model
        self._temperature = config.mapreduce_temperature
        self._max_tokens_map = config.mapreduce_max_tokens_map
        self._max_tokens_reduce = config.mapreduce_max_tokens_reduce
        self._max_tokens_mece = config.mapreduce_max_tokens_mece
        self._batch_size = config.mapreduce_batch_size
        self._concurrency = config.mapreduce_concurrency
        self._rpm_limit = config.mapreduce_rpm_limit

        # Prompt capture (optional — pass PromptPrinter to enable)
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()

        # Shared async resources — initialized in _process_all_async()
        self._client = None
        self._semaphore = None
        self._rate_limiter = None

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def process_all_partitions(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_set: PartitionSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        primary_facet: Optional[str] = None,
        facet_description: Optional[str] = None,
        grouping_instructions: Optional[Dict[str, str]] = None,
        precluster_results: Optional[Dict[str, PreclusterResult]] = None,
        verbose: bool = False,
    ) -> Dict[str, PartitionMECEResult]:
        """
        Process all partitions through the map-reduce-MECE pipeline.

        All partitions are processed concurrently. All LLM calls share
        one semaphore + rate limiter for throughput control.

        Args:
            label_mappings: Dict mapping partition_name → PartitionLabelMapping
            partition_set: Full PartitionSet (for building peer partition context)
            survey_question: The survey question for prompt context
            language: Language for output (default: Dutch)
            dataset_context: Optional dict with domain, entity, topic, etc.
            primary_facet: Optional primary facet from step 3
            facet_description: Optional primary facet description
            grouping_instructions: Optional dict of partition_name → instruction
            precluster_results: Optional dict of partition_name → PreclusterResult
            verbose: Print progress info

        Returns:
            Dict mapping partition_name → PartitionMECEResult
        """
        print(f"\n{'='*70}")
        print(f"MAP-REDUCE MECE: Category Discovery")
        print(f"{'='*70}")

        # Build shared prompt context
        dataset_context_section = self._build_dataset_context_section(dataset_context)
        taxonomy_context = self._build_taxonomy_context(
            primary_facet, facet_description,
        )

        prompt_context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            taxonomy_context=taxonomy_context,
        )

        # Build per-partition context (with peer partitions, grouping, cluster hints)
        partition_contexts = self._build_all_partition_contexts(
            partition_set,
            grouping_instructions=grouping_instructions,
            precluster_results=precluster_results,
        )

        # Filter empty mappings
        active_partitions = {
            name: mapping for name, mapping in label_mappings.items()
            if mapping.labels
        }

        if verbose:
            total_labels = sum(m.label_count for m in active_partitions.values())
            total_batches = sum(
                len(self._create_batches(m.labels))
                for m in active_partitions.values()
            )
            print(f"  Processing {len(active_partitions)} partitions concurrently "
                  f"({total_labels} labels, ~{total_batches} map batches)")
            print(f"  Batch size: {self._batch_size} labels")

        return asyncio.run(
            self._process_all_async(
                active_partitions, partition_contexts, prompt_context, verbose
            )
        )

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _probe_call(self, prompt: str):
        """Probe call for bootstrap measurement — returns usage dict."""
        resp = await llm_create_async(
            client=self._client,
            model=self._model,
            prompt=prompt,
            response_model=ProbeResponse,
            temperature=self._temperature,
            track_usage=False,
        )
        # Extract usage from instructor's _raw_response
        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        input_tokens = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0)
        output_tokens = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0)
        return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}

    async def _process_all_async(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, PartitionContext],
        prompt_context: PromptContext,
        verbose: bool,
    ) -> Dict[str, PartitionMECEResult]:
        """Main async entry: bootstrap rate limits, then gather all partitions."""
        self._client = create_client(model=self._model, async_mode=True)

        processing_config = DEFAULT_PROCESSING_CONFIG
        headroom = processing_config.rate_limit_headroom

        # --- Fetch real rate limits from API headers ---
        if verbose:
            print("  Fetching rate limits from API...")
        limits = await self._fetch_rate_limits_from_api()

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if verbose:
                print(f"  WARNING: Using fallback rate limits "
                      f"(TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            limits = RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
            )
        elif verbose:
            print(f"  Fetched from API: TPM={limits.tokens_per_minute:,}, "
                  f"RPM={limits.requests_per_minute:,}")

        # --- Bootstrap probe calls to measure latency + token usage ---
        # Build a representative MAP prompt for probing
        first_name = next(iter(sorted(label_mappings.keys())))
        first_labels = label_mappings[first_name].labels
        probe_batch = first_labels[:min(self._batch_size, len(first_labels))]
        probe_prompt = self._build_map_prompt(
            probe_batch, 0, 1,
            partition_contexts[first_name], prompt_context,
        )

        if verbose:
            print("  Running bootstrap measurement (3 probe calls)...")

        probe_start = time.time()

        async def probe_fn():
            return await self._probe_call(probe_prompt)

        avg_latency_s, avg_tokens = await bootstrap_measure_async(probe_fn, n_probes=3)

        if verbose:
            print(f"  Probe time: {time.time() - probe_start:.1f}s")
            print(f"  Bootstrap: {avg_latency_s:.2f}s avg latency, "
                  f"{avg_tokens:.0f} avg tokens")

        # --- Compute optimal concurrency via Little's Law ---
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law_conc = compute_optimal_concurrency(
            api_limits, avg_latency_s, avg_tokens,
            processing_config=processing_config,
            cap=processing_config.concurrency_cap_permissive,
            min_conc=processing_config.concurrency_min_permissive,
        )

        # Adaptive minimum: 3x Little's Law for burst headroom, floor of 5
        max_concurrency = processing_config.concurrency_cap_default
        adaptive_min = min(
            processing_config.concurrency_min_default,
            max(little_law_conc * 3, 5),
        )
        optimal = min(max_concurrency, max(little_law_conc, adaptive_min))

        # Compute arrival rate from the binding constraint (RPM or TPM)
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

        # Initialize shared rate-limiting resources
        total_tasks = sum(
            len(self._create_batches(m.labels)) + 2  # MAP batches + REDUCE + MECE
            for m in label_mappings.values()
        )
        self._semaphore = asyncio.Semaphore(min(total_tasks, optimal))
        self._rate_limiter = AsyncLimiter(1, time_period=1.0 / max(arrival_rate, 0.01))

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Model: {self._model}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"  Expected throughput: {arrival_rate:.1f}/s ({bottleneck} limited)")
            print(f"  Optimal by Little's Law: {little_law_conc}")
            print(f"  Concurrency (semaphore): {min(total_tasks, optimal)}")
            print(f"  Total LLM calls: ~{total_tasks}")

        # --- Process all partitions concurrently ---
        start_time = time.time()

        tasks = {
            name: self._process_single_partition(
                name, mapping.labels, partition_contexts[name],
                prompt_context, verbose
            )
            for name, mapping in sorted(label_mappings.items())
        }

        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

        results = {}
        for name, result in zip(tasks.keys(), results_list):
            if isinstance(result, Exception):
                print(f"  Partition '{name}' FAILED: "
                      f"{type(result).__name__}: {result}")
            else:
                results[name] = result

        elapsed = time.time() - start_time
        if verbose:
            total_categories = sum(len(r.categories) for r in results.values())
            print(f"\n  All partitions done in {elapsed:.1f}s → "
                  f"{total_categories} MECE categories across "
                  f"{len(results)} partitions")

        return results

    # =========================================================================
    # SINGLE PARTITION PIPELINE (async)
    # =========================================================================

    async def _process_single_partition(
        self,
        partition_name: str,
        labels: List[str],
        part_context: PartitionContext,
        prompt_context: PromptContext,
        verbose: bool = False,
    ) -> PartitionMECEResult:
        """Run the 3-step pipeline for a single partition (async)."""

        # Step 0: Batch labels (already deduplicated by collect_unique_labels)
        batches = self._create_batches(labels)
        n_batches = len(batches)

        if verbose:
            hints_note = " +cluster hints" if part_context.cluster_hints else ""
            print(f"    Partition '{partition_name}': {len(labels)} labels, "
                  f"{n_batches} batch(es){hints_note}")

        # Step 1: MAP
        t_map = time.time()
        map_results = await self._run_map_step(
            partition_name, batches, part_context, prompt_context
        )
        t_map = time.time() - t_map
        total_candidates = sum(len(r.themes) for r in map_results)

        # Step 2: REDUCE (skip for single batch)
        reduce_skipped = (n_batches == 1)
        t_reduce = time.time()

        if reduce_skipped:
            consolidated = SynthesizedThemeList(
                themes=[
                    {
                        "theme_label": t.theme_label,
                        "central_organizing_idea": t.central_organizing_idea,
                        "description": t.description,
                        "integrated_from": [t.theme_label],
                    }
                    for t in map_results[0].themes
                ]
            )
        else:
            consolidated = await self._run_reduce_step(
                partition_name, map_results, part_context, prompt_context
            )
        t_reduce = time.time() - t_reduce

        # Step 3: MECE
        t_mece = time.time()
        mece_result = await self._run_mece_step(
            partition_name, consolidated, len(labels),
            part_context, prompt_context
        )
        t_mece = time.time() - t_mece

        if verbose:
            reduce_info = ('reduce skipped'
                           if reduce_skipped
                           else f'synthesized to {len(consolidated.themes)}')
            t_total = t_map + t_reduce + t_mece
            print(f"    Partition '{partition_name}' → "
                  f"{len(mece_result.categories)} MECE categories "
                  f"(map: {total_candidates} candidates, {reduce_info}) "
                  f"[MAP {t_map:.1f}s, REDUCE {t_reduce:.1f}s, "
                  f"MECE {t_mece:.1f}s → {t_total:.1f}s]")

        return PartitionMECEResult(
            partition_name=partition_name,
            n_labels=len(labels),
            n_batches=n_batches,
            reduce_skipped=reduce_skipped,
            categories=mece_result.categories,
            mece_verifications=(
                mece_result.mece_verifications
                if hasattr(mece_result, 'mece_verifications')
                else []
            ),
        )

    # =========================================================================
    # DYNAMIC RATE LIMIT DISCOVERY
    # =========================================================================

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
        """Make a minimal API call to fetch rate limits from response headers."""
        from openai import AsyncOpenAI

        if API_PROVIDER == "azure":
            from config import (
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_DEPLOYMENT_NAME,
            )
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/",
                default_query={"api-version": "2024-10-21"},
            )
            model = AZURE_OPENAI_DEPLOYMENT_NAME
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self._model

        response = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        return extract_rate_limits_from_response(response)

    # =========================================================================
    # SHARED LLM CALL
    # =========================================================================

    async def _llm_call(self, prompt: str, response_model, max_tokens: int):
        """Make a rate-limited LLM call through the shared semaphore."""
        async with self._semaphore:
            async with self._rate_limiter:
                return await llm_create_async(
                    client=self._client,
                    model=self._model,
                    prompt=prompt,
                    response_model=response_model,
                    temperature=self._temperature,
                    max_tokens=max_tokens,
                )

    # =========================================================================
    # MAP STEP
    # =========================================================================

    async def _run_map_step(
        self,
        partition_name: str,
        batches: List[List[str]],
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> List[MapBatchThemes]:
        """Send each batch to LLM concurrently through shared rate limiter."""
        results = [None] * len(batches)

        async def process_batch(batch_idx: int, labels: List[str]):
            prompt = self._build_map_prompt(
                labels, batch_idx, len(batches),
                part_context, prompt_context
            )
            # Prompt capture (first batch per partition only)
            gate_key = "map"
            if (self._prompt_printer is not None
                    and batch_idx == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="theme_discovery_map",
                    utility_name="MapReduceMECE",
                    prompt_content=prompt,
                    prompt_type="map_themes",
                    metadata={
                        "model": self._model,
                        "language": prompt_context.language,
                        "partition_name": partition_name,
                        "batch_number": 1,
                        "total_batches": len(batches),
                        "survey_question": prompt_context.survey_question,
                    }
                )
                self._captured_gates.add(gate_key)
            try:
                result = await self._llm_call(
                    prompt, MapBatchThemes, self._max_tokens_map
                )
                results[batch_idx] = result
            except Exception as e:
                print(f"    MAP '{partition_name}' batch "
                      f"{batch_idx + 1}/{len(batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                results[batch_idx] = MapBatchThemes(themes=[])

        await asyncio.gather(*(
            process_batch(i, batch) for i, batch in enumerate(batches)
        ))

        return [r for r in results if r is not None]

    # =========================================================================
    # REDUCE STEP
    # =========================================================================

    async def _run_reduce_step(
        self,
        partition_name: str,
        map_results: List[MapBatchThemes],
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> SynthesizedThemeList:
        """Synthesize candidate themes from all batches into overarching themes."""
        prompt = self._build_reduce_prompt(
            map_results, part_context, prompt_context
        )

        # Prompt capture
        gate_key = "reduce"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="theme_discovery_reduce",
                utility_name="MapReduceMECE",
                prompt_content=prompt,
                prompt_type="reduce_themes",
                metadata={
                    "model": self._model,
                    "language": prompt_context.language,
                    "partition_name": partition_name,
                    "n_batches": len(map_results),
                    "survey_question": prompt_context.survey_question,
                }
            )
            self._captured_gates.add(gate_key)

        try:
            return await self._llm_call(
                prompt, SynthesizedThemeList, self._max_tokens_reduce
            )
        except Exception as e:
            print(f"    REDUCE '{partition_name}' FAILED: "
                  f"{type(e).__name__}: {e}")
            return SynthesizedThemeList(
                themes=[
                    {
                        "theme_label": t.theme_label,
                        "central_organizing_idea": t.central_organizing_idea,
                        "description": t.description,
                        "integrated_from": [t.theme_label],
                    }
                    for mr in map_results
                    for t in mr.themes
                ]
            )

    # =========================================================================
    # MECE STEP
    # =========================================================================

    async def _run_mece_step(
        self,
        partition_name: str,
        synthesized_themes: SynthesizedThemeList,
        n_labels: int,
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> MECECategorySet:
        """Apply MECE constraints with boundary criteria."""
        prompt = self._build_mece_prompt(
            synthesized_themes, n_labels, part_context, prompt_context
        )

        # Prompt capture
        gate_key = "mece"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="theme_discovery_mece",
                utility_name="MapReduceMECE",
                prompt_content=prompt,
                prompt_type="mece_boundaries",
                metadata={
                    "model": self._model,
                    "language": prompt_context.language,
                    "partition_name": partition_name,
                    "n_themes": len(synthesized_themes.themes),
                    "n_labels": n_labels,
                    "survey_question": prompt_context.survey_question,
                }
            )
            self._captured_gates.add(gate_key)

        try:
            return await self._llm_call(
                prompt, MECECategorySet, self._max_tokens_mece
            )
        except Exception as e:
            print(f"    MECE '{partition_name}' FAILED: "
                  f"{type(e).__name__}: {e}")
            return MECECategorySet(
                categories=[
                    MECECategory(
                        category_label=t.theme_label,
                        inclusion_definition=t.description,
                        boundary_test="",
                        diagnostic_signals=[],
                        key_expressions=[],
                        tiebreaker_rules=[],
                    )
                    for t in synthesized_themes.themes
                ],
                mece_verifications=[],
            )

    # =========================================================================
    # PROMPT BUILDERS
    # =========================================================================

    def _build_map_prompt(
        self,
        labels: List[str],
        batch_idx: int,
        total_batches: int,
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> str:
        """Build prompt for the MAP step with partition context."""
        labels_list = "\n".join(f"- {label}" for label in labels)

        gi = part_context.grouping_instruction
        grouping_block = f"\n{gi}" if gi else ""

        return MAP_CATEGORIES_PROMPT.format(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            taxonomy_context=prompt_context.taxonomy_context,
            partition_name=part_context.partition_name,
            partition_inclusion=part_context.partition_inclusion,
            partition_boundary_test=part_context.partition_boundary_test,
            peer_partitions_list=part_context.peer_partitions_list,
            cluster_hints=part_context.cluster_hints,
            grouping_instruction=grouping_block,
            batch_number=batch_idx + 1,
            total_batches=total_batches,
            n_labels=len(labels),
            labels_list=labels_list,
        )

    def _build_reduce_prompt(
        self,
        map_results: List[MapBatchThemes],
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> str:
        """Build prompt for the REDUCE step with partition context."""
        sections = []
        for batch_idx, batch_result in enumerate(map_results):
            section_lines = [f"Batch {batch_idx + 1}:"]
            for theme in batch_result.themes:
                section_lines.append(
                    f"  - {theme.theme_label}: {theme.description}"
                )
                if hasattr(theme, 'central_organizing_idea') and theme.central_organizing_idea:
                    section_lines.append(
                        f"    Central idea: {theme.central_organizing_idea}"
                    )
            sections.append("\n".join(section_lines))

        batch_categories_list = "\n\n".join(sections)
        n_total_themes = sum(len(r.themes) for r in map_results)

        gi = part_context.grouping_instruction
        grouping_block = f"\n{gi}" if gi else ""

        return REDUCE_THEMES_PROMPT.format(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            taxonomy_context=prompt_context.taxonomy_context,
            partition_name=part_context.partition_name,
            partition_inclusion=part_context.partition_inclusion,
            partition_boundary_test=part_context.partition_boundary_test,
            peer_partitions_list=part_context.peer_partitions_list,
            cluster_hints=part_context.cluster_hints,
            grouping_instruction=grouping_block,
            n_batches=len(map_results),
            n_total_themes=n_total_themes,
            batch_categories_list=batch_categories_list,
        )

    def _build_mece_prompt(
        self,
        synthesized_themes: SynthesizedThemeList,
        n_labels: int,
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> str:
        """Build prompt for the MECE step with partition context."""
        cat_lines = []
        for i, theme in enumerate(synthesized_themes.themes, 1):
            cat_lines.append(
                f"{i}. {theme.theme_label}\n"
                f"   Central idea: {theme.central_organizing_idea}\n"
                f"   Description: {theme.description}\n"
                f"   Integrated from: {', '.join(theme.integrated_from)}"
            )
        categories_list = "\n\n".join(cat_lines)

        gi = part_context.grouping_instruction
        grouping_block = f"\n{gi}" if gi else ""

        return MECE_BOUNDARIES_PROMPT.format(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            taxonomy_context=prompt_context.taxonomy_context,
            partition_name=part_context.partition_name,
            partition_inclusion=part_context.partition_inclusion,
            partition_boundary_test=part_context.partition_boundary_test,
            peer_partitions_list=part_context.peer_partitions_list,
            cluster_hints=part_context.cluster_hints,
            grouping_instruction=grouping_block,
            n_labels=n_labels,
            n_categories=len(synthesized_themes.themes),
            categories_list=categories_list,
        )

    # =========================================================================
    # PARTITION CONTEXT BUILDERS
    # =========================================================================

    def _build_all_partition_contexts(
        self,
        partition_set: PartitionSet,
        grouping_instructions: Optional[Dict[str, str]] = None,
        precluster_results: Optional[Dict[str, PreclusterResult]] = None,
    ) -> Dict[str, PartitionContext]:
        """Build PartitionContext for each partition with peer partitions listed."""
        contexts = {}
        all_partitions = partition_set.partitions

        for part in all_partitions:
            # Build peer partitions list (all partitions except this one)
            peer_lines = []
            for peer in all_partitions:
                if peer.partition_name == part.partition_name:
                    continue
                peer_lines.append(
                    f"- {peer.partition_name}: {peer.inclusion_definition}"
                )

            # Look up grouping instruction for this partition
            gi = ""
            if grouping_instructions:
                gi = grouping_instructions.get(part.partition_name, "")

            # Build cluster hints for this partition (Mode B)
            hints = ""
            if precluster_results:
                pc_result = precluster_results.get(part.partition_name)
                if pc_result:
                    hints = build_cluster_hints(pc_result)

            contexts[part.partition_name] = PartitionContext(
                partition_name=part.partition_name,
                partition_inclusion=part.inclusion_definition,
                partition_boundary_test=part.boundary_test,
                peer_partitions_list=(
                    "\n".join(peer_lines) if peer_lines
                    else "(no peer partitions)"
                ),
                grouping_instruction=gi,
                cluster_hints=hints,
            )

        return contexts

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _create_batches(self, labels: List[str]) -> List[List[str]]:
        """Split labels into batches of max batch_size."""
        return [
            labels[i:i + self._batch_size]
            for i in range(0, len(labels), self._batch_size)
        ]

    @staticmethod
    def _build_dataset_context_section(
        dataset_context: Optional[Dict[str, str]],
    ) -> str:
        """Build dataset context block for prompts."""
        if not dataset_context:
            return ""

        parts = []
        for key in ["domain", "entity", "topic", "perspective", "intent"]:
            value = dataset_context.get(key, "")
            if value:
                parts.append(f"{key.capitalize()}: {value}")

        if not parts:
            return ""

        return "<dataset_context>\n" + "\n".join(parts) + "\n</dataset_context>"

    @staticmethod
    def _build_taxonomy_context(
        primary_facet: Optional[str],
        facet_description: Optional[str],
    ) -> str:
        """Build taxonomy context block for prompts from ExtractionMetadata fields."""
        if not primary_facet:
            return ""

        desc = facet_description or "Not specified"
        return (
            f"<taxonomy_context>\n"
            f"Primary coding facet (applies across all concept types): {primary_facet}\n"
            f"Definition in survey language: {desc}\n"
            f"Categories must describe content within this facet ONLY.\n"
            f"</taxonomy_context>"
        )
