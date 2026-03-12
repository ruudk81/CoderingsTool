"""
Qualitative Researcher pipeline v2 for Category Discovery.

Pipeline:
  1.  Theme Discovery (chunked, per partition) — overlapping chunks + taxonomy context
  1.5 Theme Consolidation (per partition) — LLM-based deduplication
  2a. Concept Discovery (per partition) — organizing concepts from descriptive codes
  3.  COC Consolidation (cross-partition) — merge COCs into minimum set
  4.  Hierarchical Codebook (single call) — 2-3 level codebook from consolidated COCs

Per-partition steps (1, 1.5, 2a) run CONCURRENTLY. Steps 3 and 4 are sequential.

Usage:
    from .qualitative_researcher import QualitativeResearcher
    from .config_categories_exp import CategoriesConfig

    processor = QualitativeResearcher(config)
    result = processor.process_all_partitions(
        label_mappings={"identity": mapping, ...},
        partition_set=partition_set,
        dimension_name="EVALUATION_PRIORITIZATION",
        dimension_description="...",
        ...
    )
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import nest_asyncio
from aiolimiter import AsyncLimiter

from utils.llm import (
    create_client, llm_create_async, ProbeResponse, RateLimits,
    extract_rate_limits_from_response,
)
from config import (
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG, OPENAI_API_KEY,
    API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM,
)

from .config_categories_exp import CategoriesConfig
from .partition_discoverer import PartitionLabelMapping
from .models_exp import PartitionSet, PartitionDescription
from .prompts_exp import (
    MECECategory,
    build_theme_discovery_prompt,
    ThemeDiscoveryResult,
    build_theme_consolidation_prompt,
    ConsolidatedThemesResult,
    build_concept_discovery_prompt,
    ConceptDiscoveryResult,
    build_coc_consolidation_prompt,
    COCConsolidationResult,
    build_hierarchical_codebook_prompt,
    HierarchicalCodebookResult,
    convert_hierarchical_to_mece_categories,
    ThematicAnalysisResult,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


# =============================================================================
# SHARED DATACLASSES
# =============================================================================

@dataclass
class ApiLimits:
    """API limits structure for bootstrap calculations."""
    tokens_per_minute: int
    requests_per_minute: int


@dataclass
class PromptContext:
    """Shared context passed to all prompt formatting methods."""
    survey_question: str
    language: str
    dataset_context_section: str
    dimension_name: str
    dimension_description: str


@dataclass
class PartitionContext:
    """Partition-specific context."""
    partition_name: str
    partition_definition: str


@dataclass
class PartitionResult:
    """Theme discovery result for a single partition."""
    partition_name: str
    n_labels: int
    n_batches: int
    themes: List[str]


@dataclass
class PartitionConceptResult:
    """Concept discovery result for a single partition."""
    partition_name: str
    concept_discovery: ConceptDiscoveryResult


@dataclass
class PipelineResult:
    """Complete v2 pipeline output."""
    partition_themes: Dict[str, List[str]]
    codebook_categories: List[MECECategory]
    codebook_narrative: str
    partition_results: Dict[str, PartitionResult]
    partition_concepts: Optional[Dict[str, PartitionConceptResult]] = None
    coc_consolidation: Optional[COCConsolidationResult] = None
    hierarchical_codebook: Optional[HierarchicalCodebookResult] = None

    @property
    def codebook(self) -> List[MECECategory]:
        """Returns the final MECE codebook entries."""
        return self.codebook_categories

    @property
    def thematic_analysis(self) -> ThematicAnalysisResult:
        """Backward compat shim for run_experiment.py and cache_mece_results."""
        return ThematicAnalysisResult(
            themes=self.codebook_categories,
            thematic_map=self.codebook_narrative,
        )


# =============================================================================
# BOOTSTRAP UTILITIES
# =============================================================================

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
# MAIN PROCESSOR
# =============================================================================

class QualitativeResearcher:
    """
    Per-partition Qualitative Researcher pipeline v2.

    Pipeline:
    1.  THEME DISCOVERY:          Per partition, chunked with overlap (concurrent)
    1b. PROGRAMMATIC DEDUP:       Per partition, case-insensitive (no LLM)
    1.5 THEME CONSOLIDATION:      Per partition, LLM-based dedup (concurrent)
    2a. CONCEPT DISCOVERY:        Per partition, organizing concepts (concurrent)
    3.  COC CONSOLIDATION:        Cross-partition, merge COCs into minimum set
    4.  HIERARCHICAL CODEBOOK:    Single call, build 2-3 level codebook from consolidated COCs

    LLM calls: sum(N_chunks per partition) + N_partitions + N_partitions + 1 + 1
    """

    def __init__(self, config: CategoriesConfig, prompt_printer=None):
        self._model = config.qr_model
        self._temperature = config.qr_temperature
        self._max_tokens_themes = config.qr_max_tokens_theme_discovery
        self._max_tokens_consolidation = config.qr_max_tokens_consolidation
        self._max_tokens_concept_discovery = config.qr_max_tokens_concept_discovery
        self._max_tokens_coc_consolidation = config.qr_max_tokens_coc_consolidation
        self._max_tokens_hierarchical_codebook = config.qr_max_tokens_hierarchical_codebook

        # Batch sizing
        self._batch_size_min = config.batch_size_min
        self._batch_size_max = config.batch_size_max
        self._target_batches = config.target_batches
        self._chunk_overlap = config.chunk_overlap
        self._consolidation_chunk_size = config.consolidation_chunk_size
        self._consolidation_max_rounds = config.consolidation_max_rounds

        # Prompt capture (optional)
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
        dimension_name: str = "",
        dimension_description: str = "",
        verbose: bool = False,
    ) -> PipelineResult:
        """Process all partitions through the v2 pipeline."""
        print(f"\n{'='*70}")
        print(f"QUALITATIVE RESEARCHER v2: Category Discovery")
        print(f"{'='*70}")

        # Build shared prompt context
        dataset_context_section = self._build_dataset_context_section(dataset_context)

        prompt_context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            dimension_name=dimension_name,
            dimension_description=dimension_description,
        )

        # Build per-partition context
        partition_contexts = self._build_all_partition_contexts(partition_set)

        # Filter empty mappings
        active_partitions = {
            name: mapping for name, mapping in label_mappings.items()
            if mapping.labels
        }

        if verbose:
            total_labels = sum(m.label_count for m in active_partitions.values())
            partition_info = []
            total_chunks = 0
            for name, mapping in sorted(active_partitions.items()):
                n = len(mapping.labels)
                bs = self._compute_batch_size(n)
                nb = len(self._create_batches(mapping.labels))
                total_chunks += nb
                partition_info.append(f"{name}: {n}→{nb}×{bs}")
            n_partitions = len(active_partitions)
            total_llm_calls = total_chunks + n_partitions + n_partitions + 2
            print(f"  Processing {n_partitions} partitions concurrently "
                  f"({total_labels} labels)")
            print(f"  Adaptive batch sizes: {', '.join(partition_info)}")
            print(f"  Chunk overlap: {self._chunk_overlap:.0%}")
            print(f"  LLM calls: ~{total_llm_calls} "
                  f"({total_chunks} theme discovery + "
                  f"{n_partitions} consolidation + "
                  f"{n_partitions} concept discovery + "
                  f"1 COC consolidation + 1 codebook)")
            if dimension_name:
                print(f"  Taxonomy: {dimension_name}")

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
    ) -> PipelineResult:
        """Main async entry: bootstrap → theme discovery → MECE → codebook."""
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
        first_name = next(iter(sorted(label_mappings.keys())))
        first_mapping = label_mappings[first_name]
        first_labels = first_mapping.labels
        probe_batch_size = self._compute_batch_size(len(first_labels))
        probe_n = min(probe_batch_size, len(first_labels))
        probe_batch = first_labels[:probe_n]
        probe_domains = first_mapping.label_domains[:probe_n] if first_mapping.label_domains else None
        first_part_ctx = partition_contexts[first_name]
        probe_prompt = build_theme_discovery_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            partition_name=first_part_ctx.partition_name,
            partition_definition=first_part_ctx.partition_definition,
            labels=probe_batch,
            label_domains=probe_domains,
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

        max_concurrency = processing_config.concurrency_cap_default
        adaptive_min = min(
            processing_config.concurrency_min_default,
            max(little_law_conc * 3, 5),
        )
        optimal = min(max_concurrency, max(little_law_conc, adaptive_min))

        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

        # Total tasks: theme chunks + consolidation + concept discovery + COC consolidation + codebook
        total_theme_chunks = sum(
            len(self._create_batches(m.labels))
            for m in label_mappings.values()
        )
        total_tasks = total_theme_chunks + len(label_mappings) + len(label_mappings) + 2
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

        # =================================================================
        # PHASE 1: Per-partition theme discovery (concurrent)
        # =================================================================
        start_time = time.time()

        tasks = {
            name: self._discover_partition_themes(
                name, mapping.labels, partition_contexts[name],
                prompt_context, verbose,
                label_domains=mapping.label_domains or None,
            )
            for name, mapping in sorted(label_mappings.items())
        }

        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

        partition_results = {}
        partition_themes = {}
        for name, result in zip(tasks.keys(), results_list):
            if isinstance(result, Exception):
                print(f"  Partition '{name}' FAILED: "
                      f"{type(result).__name__}: {result}")
            else:
                partition_results[name] = result
                partition_themes[name] = result.themes

        phase1_elapsed = time.time() - start_time
        if verbose:
            total_themes = sum(len(t) for t in partition_themes.values())
            print(f"\n  Phase 1 done in {phase1_elapsed:.1f}s → "
                  f"{total_themes} raw themes across "
                  f"{len(partition_results)} partitions")

        # =================================================================
        # PHASE 1.5: Per-partition theme consolidation (concurrent)
        # =================================================================
        if verbose:
            print(f"\n  Phase 1.5: Per-partition Theme Consolidation...")

        t_consolidation = time.time()

        consolidation_tasks = {
            name: self._run_theme_consolidation(
                name, themes, partition_contexts[name], prompt_context
            )
            for name, themes in sorted(partition_themes.items())
            if themes
        }

        consolidation_results = await asyncio.gather(
            *consolidation_tasks.values(), return_exceptions=True
        )

        for name, result in zip(consolidation_tasks.keys(), consolidation_results):
            if isinstance(result, Exception):
                print(f"  Consolidation '{name}' FAILED: "
                      f"{type(result).__name__}: {result}")
                # Keep raw themes as fallback
            else:
                old_count = len(partition_themes[name])
                partition_themes[name] = result.themes
                if verbose:
                    print(f"    {name}: {old_count} → {len(result.themes)} themes")

        t_consolidation = time.time() - t_consolidation
        if verbose:
            total_consolidated = sum(len(t) for t in partition_themes.values())
            print(f"  Phase 1.5 done in {t_consolidation:.1f}s → "
                  f"{total_consolidated} consolidated themes")

        # =================================================================
        # PHASE 2a: Per-partition Concept Discovery (concurrent)
        # =================================================================
        if verbose:
            print(f"\n  Phase 2a: Per-partition Concept Discovery "
                  f"({len(partition_themes)} partitions)...")

        t_phase2a = time.time()

        concept_tasks = {
            name: self._run_concept_discovery(
                name, themes, partition_contexts[name], prompt_context
            )
            for name, themes in sorted(partition_themes.items())
            if themes
        }

        concept_results_list = await asyncio.gather(
            *concept_tasks.values(), return_exceptions=True
        )

        partition_concepts: Dict[str, PartitionConceptResult] = {}
        concept_results_for_consolidation: Dict[str, ConceptDiscoveryResult] = {}
        for name, result in zip(concept_tasks.keys(), concept_results_list):
            if isinstance(result, Exception):
                print(f"  Concept discovery '{name}' FAILED: "
                      f"{type(result).__name__}: {result}")
            else:
                partition_concepts[name] = PartitionConceptResult(
                    partition_name=name,
                    concept_discovery=result,
                )
                concept_results_for_consolidation[name] = result

        t_phase2a = time.time() - t_phase2a

        if verbose:
            total_concepts = sum(
                len(pcr.concept_discovery.compressed_concepts)
                for pcr in partition_concepts.values()
            )
            print(f"\n  Phase 2a done in {t_phase2a:.1f}s → "
                  f"{total_concepts} COCs across "
                  f"{len(partition_concepts)} partitions")
            for name, pcr in sorted(partition_concepts.items()):
                n_concepts = len(pcr.concept_discovery.compressed_concepts)
                print(f"    {name}: {n_concepts} concepts")
                for i, c in enumerate(pcr.concept_discovery.compressed_concepts, 1):
                    print(f"      {i}. {c}")

        # =================================================================
        # PHASE 3: Cross-partition COC Consolidation (single call)
        # =================================================================
        if verbose:
            print(f"\n  Phase 3: Cross-partition COC Consolidation...")

        t_phase3 = time.time()

        # Collect partition definitions for the consolidation prompt
        partition_definitions = {
            name: partition_contexts[name].partition_definition
            for name in concept_results_for_consolidation
        }

        consolidated = await self._run_coc_consolidation(
            concept_results_for_consolidation, partition_definitions, prompt_context
        )

        t_phase3 = time.time() - t_phase3

        if verbose:
            n_input = sum(
                len(cr.compressed_concepts)
                for cr in concept_results_for_consolidation.values()
            )
            n_output = len(consolidated.consolidated_concepts)
            print(f"\n  Phase 3 done in {t_phase3:.1f}s → "
                  f"{n_input} per-partition COCs → {n_output} consolidated COCs")
            for i, c in enumerate(consolidated.consolidated_concepts, 1):
                sources = ", ".join(c.source_partitions)
                print(f"    {i}. {c.concept_name} [{sources}]")

        # =================================================================
        # PHASE 4: Hierarchical Codebook Construction (single call)
        # =================================================================
        if verbose:
            print(f"\n  Phase 4: Hierarchical Codebook Construction...")

        t_phase4 = time.time()

        codebook_result = await self._run_hierarchical_codebook(
            consolidated, prompt_context
        )

        # Bridge to MECECategory
        codebook = convert_hierarchical_to_mece_categories(codebook_result)
        codebook_narrative = codebook_result.mece_validation

        t_phase4 = time.time() - t_phase4

        if verbose:
            n_themes = len(codebook_result.themes)
            n_codes = sum(len(t.codes) for t in codebook_result.themes)
            n_subcodes = sum(
                len(sc) for t in codebook_result.themes
                for c in t.codes for sc in [c.subcodes]
            )
            print(f"\n  Phase 4 done in {t_phase4:.1f}s → "
                  f"{n_themes} themes, {n_codes} subthemes"
                  + (f", {n_subcodes} valence codes" if n_subcodes else ""))
            for theme in codebook_result.themes:
                print(f"    L1: {theme.theme_label}: {theme.theme_definition}")
                for code in theme.codes:
                    print(f"      L2: {code.code_label}: {code.definition}")
                    for subcode in code.subcodes:
                        print(f"        L3: {subcode.code_label}: {subcode.definition}")

        total_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Pipeline complete in {total_elapsed:.1f}s")

        return PipelineResult(
            partition_themes=partition_themes,
            codebook_categories=codebook,
            codebook_narrative=codebook_narrative,
            partition_results=partition_results,
            partition_concepts=partition_concepts,
            coc_consolidation=consolidated,
            hierarchical_codebook=codebook_result,
        )

    # =========================================================================
    # PHASE 1: PER-PARTITION THEME DISCOVERY
    # =========================================================================

    async def _discover_partition_themes(
        self,
        partition_name: str,
        labels: List[str],
        part_context: PartitionContext,
        prompt_context: PromptContext,
        verbose: bool = False,
        label_domains: Optional[List] = None,
    ) -> PartitionResult:
        """Run theme discovery + programmatic dedup for a single partition."""

        # Step 0: Create overlapping batches
        batches = self._create_batches(labels)
        domain_batches = self._create_batches(label_domains) if label_domains else None
        n_batches = len(batches)

        if verbose:
            batch_size = self._compute_batch_size(len(labels))
            print(f"    Partition '{partition_name}': {len(labels)} labels, "
                  f"{n_batches} chunk(s) of ~{batch_size} "
                  f"(overlap {self._chunk_overlap:.0%})")

        # Step 1: THEME DISCOVERY (chunked, concurrent)
        t_themes = time.time()
        all_themes = await self._run_theme_discovery(
            partition_name, batches, part_context, prompt_context,
            domain_batches=domain_batches,
        )
        t_themes = time.time() - t_themes

        # Step 2: Programmatic dedup (case-insensitive, strip whitespace)
        seen = set()
        unique_themes = []
        for theme in all_themes:
            key = theme.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_themes.append(theme.strip())

        if verbose:
            print(f"    Partition '{partition_name}' themes: "
                  f"{len(all_themes)} raw → {len(unique_themes)} unique "
                  f"[{t_themes:.1f}s]")

        return PartitionResult(
            partition_name=partition_name,
            n_labels=len(labels),
            n_batches=n_batches,
            themes=unique_themes,
        )

    async def _run_theme_discovery(
        self,
        partition_name: str,
        batches: List[List[str]],
        part_context: PartitionContext,
        prompt_context: PromptContext,
        domain_batches: Optional[List[List]] = None,
    ) -> List[str]:
        """Discover themes from chunked labels (concurrent)."""
        results = [None] * len(batches)

        async def process_chunk(chunk_idx: int, labels: List[str]):
            domains = domain_batches[chunk_idx] if domain_batches else None
            prompt = build_theme_discovery_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                partition_name=part_context.partition_name,
                partition_definition=part_context.partition_definition,
                labels=labels,
                label_domains=domains,
            )

            # Prompt capture (first chunk per partition)
            gate_key = f"qr_themes_{partition_name}"
            if (self._prompt_printer is not None
                    and chunk_idx == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="theme_discovery",
                    metadata={
                        "model": self._model,
                        "language": prompt_context.language,
                        "partition_name": partition_name,
                        "batch_number": chunk_idx + 1,
                        "total_batches": len(batches),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            try:
                result = await self._llm_call(
                    prompt, ThemeDiscoveryResult, self._max_tokens_themes
                )
                results[chunk_idx] = result
            except Exception as e:
                print(f"    THEME DISCOVERY '{partition_name}' chunk "
                      f"{chunk_idx + 1}/{len(batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                results[chunk_idx] = ThemeDiscoveryResult(themes=[])

        await asyncio.gather(*(
            process_chunk(i, batch) for i, batch in enumerate(batches)
        ))

        # Flatten all themes from all chunks
        all_themes = []
        for r in results:
            if r is not None:
                all_themes.extend(r.themes)
        return all_themes

    # =========================================================================
    # PHASE 1.5: PER-PARTITION THEME CONSOLIDATION
    # =========================================================================

    async def _run_theme_consolidation(
        self,
        partition_name: str,
        themes: List[str],
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> ConsolidatedThemesResult:
        """Consolidate raw themes into distinct set.

        Uses hierarchical chunking when themes exceed consolidation_chunk_size:
        split into batches, consolidate each concurrently, dedup merged results,
        repeat until the list fits in a single final call.
        """
        max_per_call = self._consolidation_chunk_size
        max_rounds = self._consolidation_max_rounds
        current_themes = list(themes)

        for round_idx in range(max_rounds):
            # Fits in one call → single final consolidation
            if len(current_themes) <= max_per_call:
                return await self._consolidate_single_batch(
                    partition_name, current_themes, part_context, prompt_context,
                    round_idx=round_idx, is_final=True,
                )

            # Chunk, consolidate each chunk concurrently, dedup, loop
            chunks = self._chunk_themes_for_consolidation(
                current_themes, max_per_call
            )

            if round_idx > 0:
                print(f"      {partition_name} round {round_idx + 1}: "
                      f"{len(current_themes)} themes in {len(chunks)} chunks")

            chunk_results = await asyncio.gather(*(
                self._consolidate_single_batch(
                    partition_name, chunk, part_context, prompt_context,
                    round_idx=round_idx, chunk_idx=i, total_chunks=len(chunks),
                )
                for i, chunk in enumerate(chunks)
            ))

            # Flatten + programmatic dedup
            merged: List[str] = []
            seen: set = set()
            for result in chunk_results:
                for theme in result.themes:
                    key = theme.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        merged.append(theme.strip())

            current_themes = merged

        # Exhausted max_rounds — do one final call with whatever remains
        return await self._consolidate_single_batch(
            partition_name, current_themes, part_context, prompt_context,
            round_idx=max_rounds, is_final=True,
        )

    async def _consolidate_single_batch(
        self,
        partition_name: str,
        themes: List[str],
        part_context: PartitionContext,
        prompt_context: PromptContext,
        *,
        round_idx: int = 0,
        chunk_idx: int | None = None,
        total_chunks: int | None = None,
        is_final: bool = False,
    ) -> ConsolidatedThemesResult:
        """Single consolidation LLM call."""
        prompt = build_theme_consolidation_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            partition_name=part_context.partition_name,
            partition_definition=part_context.partition_definition,
            themes=themes,
        )

        # Prompt capture (only first chunk of first round)
        gate_key = f"qr_consolidation_{partition_name}"
        if (self._prompt_printer is not None
                and round_idx == 0
                and (chunk_idx is None or chunk_idx == 0)
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="theme_consolidation",
                metadata={
                    "model": self._model,
                    "language": prompt_context.language,
                    "partition_name": partition_name,
                    "n_raw_themes": len(themes),
                    "dimension_name": prompt_context.dimension_name,
                    "round": round_idx,
                    "chunk": chunk_idx,
                    "total_chunks": total_chunks,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, ConsolidatedThemesResult, self._max_tokens_consolidation
        )

    @staticmethod
    def _chunk_themes_for_consolidation(
        themes: List[str], chunk_size: int,
    ) -> List[List[str]]:
        """Split themes into non-overlapping chunks of at most chunk_size."""
        return [themes[i:i + chunk_size]
                for i in range(0, len(themes), chunk_size)]

    # =========================================================================
    # PHASE 2a: PER-PARTITION CONCEPT DISCOVERY
    # =========================================================================

    async def _run_concept_discovery(
        self,
        partition_name: str,
        themes: List[str],
        part_context: PartitionContext,
        prompt_context: PromptContext,
    ) -> ConceptDiscoveryResult:
        """Identify organizing concepts from a single partition's themes."""
        prompt = build_concept_discovery_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            partition_name=part_context.partition_name,
            partition_definition=part_context.partition_definition,
            themes=themes,
        )

        # Prompt capture
        gate_key = f"qr_concept_discovery_{partition_name}"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="concept_discovery",
                metadata={
                    "model": self._model,
                    "language": prompt_context.language,
                    "partition_name": partition_name,
                    "n_themes": len(themes),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, ConceptDiscoveryResult, self._max_tokens_concept_discovery
        )

    # =========================================================================
    # PHASE 3: CROSS-PARTITION COC CONSOLIDATION
    # =========================================================================

    async def _run_coc_consolidation(
        self,
        concept_results: Dict[str, ConceptDiscoveryResult],
        partition_definitions: Dict[str, str],
        prompt_context: PromptContext,
    ) -> COCConsolidationResult:
        """Consolidate per-partition COCs into minimum set for full coverage."""
        prompt = build_coc_consolidation_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            partition_concepts=concept_results,
            partition_definitions=partition_definitions,
        )

        # Prompt capture
        gate_key = "qr_coc_consolidation"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="coc_consolidation",
                metadata={
                    "model": self._model,
                    "language": prompt_context.language,
                    "n_partitions": len(concept_results),
                    "n_total_concepts": sum(
                        len(cr.compressed_concepts)
                        for cr in concept_results.values()
                    ),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, COCConsolidationResult, self._max_tokens_coc_consolidation
        )

    # =========================================================================
    # PHASE 4: HIERARCHICAL CODEBOOK CONSTRUCTION
    # =========================================================================

    async def _run_hierarchical_codebook(
        self,
        consolidated: COCConsolidationResult,
        prompt_context: PromptContext,
    ) -> HierarchicalCodebookResult:
        """Build hierarchical codebook from consolidated COCs."""
        prompt = build_hierarchical_codebook_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            consolidated_concepts=consolidated.consolidated_concepts,
        )

        # Prompt capture
        gate_key = "qr_hierarchical_codebook"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="hierarchical_codebook",
                metadata={
                    "model": self._model,
                    "language": prompt_context.language,
                    "n_consolidated_concepts": len(consolidated.consolidated_concepts),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, HierarchicalCodebookResult, self._max_tokens_hierarchical_codebook
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
    # HELPERS
    # =========================================================================

    def _compute_batch_size(self, n_labels: int) -> int:
        """Compute adaptive batch size."""
        if n_labels <= self._batch_size_min:
            return n_labels
        ideal = max(n_labels // self._target_batches, 1)
        return max(self._batch_size_min, min(ideal, self._batch_size_max))

    def _create_batches(self, labels: List[str]) -> List[List[str]]:
        """Split labels into overlapping batches.

        Each batch overlaps with the previous by chunk_overlap * batch_size
        labels. First batch starts at 0, subsequent batches step forward
        by (1 - overlap) * batch_size.
        """
        batch_size = self._compute_batch_size(len(labels))
        if len(labels) <= batch_size:
            return [labels]

        overlap = int(batch_size * self._chunk_overlap)
        step = max(batch_size - overlap, 1)

        batches = []
        i = 0
        while i < len(labels):
            batches.append(labels[i:i + batch_size])
            i += step
            # Avoid a tiny trailing batch
            if i < len(labels) and i + batch_size > len(labels):
                # Last batch: take the final batch_size items
                batches.append(labels[-batch_size:])
                break

        return batches

    def _build_all_partition_contexts(
        self,
        partition_set: PartitionSet,
    ) -> Dict[str, PartitionContext]:
        """Build PartitionContext for each partition."""
        contexts = {}
        for part in partition_set.partitions:
            contexts[part.partition_name] = PartitionContext(
                partition_name=part.partition_name,
                partition_definition=part.inclusion_definition,
            )
        return contexts

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
