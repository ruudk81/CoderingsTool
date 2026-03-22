"""
Inductive Code Generation pipeline for Category Discovery v3.

Pipeline (10 stages):
  P1.  Facet Discovery (chunked, per domain) — dimension-specific semantics
  P2.  Facet Consolidation (per domain, hierarchical merge)
  P3.  Facet Assignment (batched, per domain) — assign ideas to discovered facets
  P4.  Attribute Discovery (per facet within domain) — concrete observables
  P5.  Attribute Chunk Consolidation (per facet, hierarchical merge)
  P6.  Attribute Assignment (per facet) — assign ideas to discovered attributes
  P7.  Cross-facet Attribute Consolidation (per domain) — dedup across facets
  P8.  Code Generation from Attributes (per domain) — derive codebook codes
  P9.  Codebook Consolidation (cross-domain) — merge into final MECE codebook
  P10. Code Assignment (per idea) — assign codes to ideas

Per-domain steps (P1–P7) run CONCURRENTLY. P8–P9 are sequential.

Usage:
    from .qualitative_researcher import QualitativeResearcher
    from .config_classNcoder_exp import CategoriesConfig

    processor = QualitativeResearcher(config)
    result = processor.process_all_partitions(
        label_mappings={"identity": mapping, ...},
        partition_set=partition_set,
        dimension_name="ATTRIBUTES_ASSOCIATIONS",
        dimension_description="...",
        ...
    )
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, create_model

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

from development.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)

from .config_classNcoder_exp import CategoriesConfig
from .domain_discoverer import PartitionLabelMapping
from .partition_labels import format_label
from .models_exp import DomainSet, DomainDescription
from .prompts_exp import (
    # P1: Facet Discovery
    build_facet_discovery_prompt,
    FacetDiscoveryResult,
    DiscoveredFacet,
    # P2: Facet Consolidation
    build_facet_consolidation_prompt,
    FacetConsolidatedResponse,
    # P3: Facet Assignment
    build_facet_assignment_prompt,
    FacetAssignmentBatch,
    # P4: Attribute Discovery
    build_attribute_discovery_prompt,
    AttributeDiscoveryResult,
    DiscoveredAttribute,
    # P5: Attribute Chunk Consolidation
    build_attribute_chunk_consolidation_prompt,
    AttributeChunkConsolidatedResponse,
    # P6: Attribute Assignment
    build_attribute_assignment_prompt,
    AttributeAssignmentBatch,
    # P7: Cross-facet Attribute Consolidation
    build_attribute_consolidation_prompt,
    AttributeConsolidatedResponse,
    ConsolidatedAttribute,
    # P8: Code Generation from Attributes
    build_code_from_attributes_prompt,
    CodeGenerationFromAttributesResult,
    CodeFromAttributes,
    # P9: Codebook Consolidation
    build_codebook_consolidation_prompt,
    CodebookConsolidationResult,
    ConsolidatedCode,
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
    dimension_def: Optional[DimensionDefinition] = None


@dataclass
class DomainContext:
    """Partition-specific context."""
    partition_name: str
    partition_definition: str


@dataclass
class DomainResult:
    """Per-domain pipeline result (v3)."""
    partition_name: str
    n_labels: int
    n_batches: int
    facets: List[DiscoveredFacet]
    facet_assignments: Dict[str, str]  # idea_id -> facet_name
    attributes: Dict[str, List[DiscoveredAttribute]]  # facet_name -> attributes
    attribute_assignments: Dict[str, str] = field(default_factory=dict)  # idea_id -> attribute_name


@dataclass
class TaxonomyResult:
    """Output of taxonomy stages P1-P7."""
    partition_n_labels: Dict[str, int]
    partition_n_batches: Dict[str, int]
    partition_facets: Dict[str, List[DiscoveredFacet]]
    partition_assignments: Dict[str, Dict[str, str]]  # domain -> {idea_id -> facet_name}
    partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]]  # domain -> {facet -> [attrs]}
    attribute_assignments: Dict[str, str]  # idea_id -> attribute_name


@dataclass
class PipelineResult:
    """Complete pipeline output (v3)."""
    partition_results: Dict[str, DomainResult]
    codebook_narrative: str
    codes: List[ConsolidatedCode]


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
    Inductive Code Generation pipeline for Category Discovery v3.

    Pipeline (10 stages):
    P1.  FACET DISCOVERY:                   Per domain, chunked with overlap (concurrent)
    P2.  FACET CONSOLIDATION:               Per domain, hierarchical merge
    P3.  FACET ASSIGNMENT:                  Per domain, assign ideas to facets (concurrent)
    P4.  ATTRIBUTE DISCOVERY:               Per (domain, facet), discover attributes (concurrent)
    P5.  ATTRIBUTE CHUNK CONSOLIDATION:     Per facet, hierarchical merge
    P6.  ATTRIBUTE ASSIGNMENT:              Per facet, assign ideas to attributes (concurrent)
    P7.  CROSS-FACET ATTR CONSOLIDATION:    Per domain, dedup across facets
    P8.  CODE GENERATION:                   Per domain, derive codes from attributes
    P9.  CODEBOOK CONSOLIDATION:            Cross-domain, merge into MECE codebook
    P10. CODE ASSIGNMENT:                   Per idea, assign codes (separate module)
    """

    def __init__(self, config: CategoriesConfig, prompt_printer=None):
        self._model_p1 = config.qr_model_p1
        self._model_p2 = config.qr_model_p2
        self._model_p3 = config.qr_model_p3
        self._model_p4 = config.qr_model_p4
        self._model_p5 = config.qr_model_p5
        self._model_p6 = config.qr_model_p6
        self._model_p7 = config.qr_model_p7
        self._model_p8 = config.qr_model_p8
        self._model_p9 = config.qr_model_p9
        self._temperature = config.qr_temperature
        self._max_tokens_facet_discovery = config.qr_max_tokens_facet_discovery
        self._max_tokens_facet_assignment = config.qr_max_tokens_facet_assignment
        self._max_tokens_attribute_discovery = config.qr_max_tokens_attribute_discovery
        self._max_tokens_code_from_attributes = config.qr_max_tokens_code_from_attributes
        self._max_tokens_codebook_consolidation = config.qr_max_tokens_codebook_consolidation
        self._facet_assignment_batch_size = config.facet_assignment_batch_size

        # Batch sizing — P1 (facet discovery)
        self._batch_size_min = config.batch_size_min
        self._batch_size_max = config.batch_size_max
        self._target_batches = config.target_batches
        self._chunk_overlap = config.chunk_overlap
        self._consolidation_max_chunks_per_call = config.consolidation_max_chunks_per_call
        self._consolidation_max_items_per_call = config.consolidation_max_items_per_call
        self._consolidation_max_rounds = config.consolidation_max_rounds

        # Batch sizing — P4 (attribute discovery)
        self._p4_batch_size_min = config.p4_batch_size_min
        self._p4_batch_size_max = config.p4_batch_size_max
        self._p4_target_batches = config.p4_target_batches
        self._p4_chunk_overlap = config.p4_chunk_overlap

        # Label source for observation formatting
        self._label_source = config.label_source
        self._label_prefix = config.label_prefix
        self._include_valence = config.include_valence

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

    def _prepare_context(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_set: DomainSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        dimension_name: str = "",
        dimension_description: str = "",
        verbose: bool = False,
    ):
        """Shared setup: resolve dimension, build contexts, filter empty mappings."""
        # Resolve dimension definition from dimension_data.py
        dimension_def = None
        if dimension_name:
            dimension_def = get_dimension(dimension_name)
            if dimension_def and verbose:
                print(f"  Dimension: {dimension_name}")
                print(f"  Facet diagnostic: {dimension_def.prompt_rules.facet_diagnostic}")
                print(f"  Domain diagnostic: {dimension_def.prompt_rules.domain_diagnostic}")
            elif not dimension_def and verbose:
                print(f"  WARNING: No DimensionDefinition found for '{dimension_name}'")
                print(f"  Falling back to generic taxonomy language")

        dataset_context_section = self._build_dataset_context_section(dataset_context)

        prompt_context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            dimension_def=dimension_def,
        )

        partition_contexts = self._build_all_partition_contexts(partition_set)

        active_partitions = {
            name: mapping for name, mapping in label_mappings.items()
            if mapping.labels
        }

        return prompt_context, partition_contexts, active_partitions

    def process_all_partitions(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_set: DomainSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        dimension_name: str = "",
        dimension_description: str = "",
        verbose: bool = False,
    ) -> PipelineResult:
        """Run full pipeline: taxonomy (P1-P7) + codebook (P8-P9)."""
        print(f"\n{'='*70}")
        print(f"INDUCTIVE CODE GENERATION v3: Category Discovery")
        print(f"{'='*70}")

        prompt_context, partition_contexts, active_partitions = self._prepare_context(
            label_mappings, partition_set, survey_question, language,
            dataset_context, dimension_name, dimension_description, verbose,
        )

        if verbose:
            total_labels = sum(m.label_count for m in active_partitions.values())
            total_ideas = sum(len(m.ideas) for m in active_partitions.values())
            n_partitions = len(active_partitions)
            print(f"  Processing {n_partitions} domains concurrently "
                  f"({total_labels} observations, {total_ideas} ideas)")
            print(f"  Pipeline: P1-P7 taxonomy → P8-P9 codebook")

        return asyncio.run(
            self._process_all_async(
                active_partitions, partition_contexts, prompt_context, verbose
            )
        )

    def process_taxonomy_only(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_set: DomainSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        dimension_name: str = "",
        dimension_description: str = "",
        verbose: bool = False,
    ) -> TaxonomyResult:
        """Run taxonomy stages only (P1-P7): facets, attributes, assignments."""
        print(f"\n{'='*70}")
        print(f"TAXONOMY DISCOVERY (P1-P7)")
        print(f"{'='*70}")

        prompt_context, partition_contexts, active_partitions = self._prepare_context(
            label_mappings, partition_set, survey_question, language,
            dataset_context, dimension_name, dimension_description, verbose,
        )

        if verbose:
            total_labels = sum(m.label_count for m in active_partitions.values())
            total_ideas = sum(len(m.ideas) for m in active_partitions.values())
            n_partitions = len(active_partitions)
            print(f"  Processing {n_partitions} domains concurrently "
                  f"({total_labels} observations, {total_ideas} ideas)")
            print(f"  Pipeline: P1 facet discovery → P3 facet assignment → "
                  f"P4 attribute discovery → P7 consolidation")

        async def _run():
            await self._initialize_async_resources(
                active_partitions, partition_contexts, prompt_context, verbose
            )
            return await self._process_taxonomy_async(
                active_partitions, partition_contexts, prompt_context, verbose
            )

        return asyncio.run(_run())

    def process_codebook_only(
        self,
        taxonomy: TaxonomyResult,
        partition_set: DomainSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        dimension_name: str = "",
        dimension_description: str = "",
        label_mappings: Optional[Dict[str, PartitionLabelMapping]] = None,
        verbose: bool = False,
    ) -> PipelineResult:
        """Run codebook stages only (P8-P9) from a TaxonomyResult."""
        print(f"\n{'='*70}")
        print(f"CODEBOOK GENERATION (P8-P9)")
        print(f"{'='*70}")

        # Need label_mappings for bootstrap; use empty if not provided
        if label_mappings is None:
            label_mappings = {}

        prompt_context, partition_contexts, active_partitions = self._prepare_context(
            label_mappings, partition_set, survey_question, language,
            dataset_context, dimension_name, dimension_description, verbose,
        )

        async def _run():
            # Initialize clients and rate limiters
            # For codebook-only, bootstrap uses fallback if no labels available
            if active_partitions:
                await self._initialize_async_resources(
                    active_partitions, partition_contexts, prompt_context, verbose
                )
            else:
                # No labels available — use fallback rate limits
                unique_models = {self._model_p1, self._model_p2, self._model_p3,
                                 self._model_p4, self._model_p5, self._model_p6,
                                 self._model_p7, self._model_p8, self._model_p9}
                self._clients = {m: create_client(model=m, async_mode=True) for m in unique_models}
                self._semaphore = asyncio.Semaphore(5)
                self._rate_limiter = AsyncLimiter(1, time_period=0.1)
                if verbose:
                    print("  Using fallback rate limits (no labels for bootstrap)")

            return await self._process_codebook_async(
                taxonomy, partition_contexts, prompt_context, verbose
            )

        return asyncio.run(_run())

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _probe_call(self, prompt: str):
        """Probe call for bootstrap measurement — returns usage dict."""
        resp = await llm_create_async(
            client=self._clients[self._model_p1],
            model=self._model_p1,
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

    async def _initialize_async_resources(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
    ):
        """Bootstrap: create clients, probe rate limits, set up concurrency."""
        # Create one client per unique model
        unique_models = {self._model_p1, self._model_p2, self._model_p3, self._model_p4, self._model_p5, self._model_p6, self._model_p7, self._model_p8, self._model_p9}
        self._clients = {m: create_client(model=m, async_mode=True) for m in unique_models}

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
        first_part_ctx = partition_contexts[first_name]
        probe_prompt = build_facet_discovery_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_def=prompt_context.dimension_def,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            partition_name=first_part_ctx.partition_name,
            partition_definition=first_part_ctx.partition_definition,
            observations=probe_batch,
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

        # Estimate total tasks across all phases
        total_p1_chunks = sum(
            len(self._create_batches(m.labels))
            for m in label_mappings.values()
        )
        n_domains = len(label_mappings)
        est_p3_batches = n_domains * 3
        est_p4_tasks = n_domains * 5
        total_tasks = total_p1_chunks + est_p3_batches + est_p4_tasks + 1

        self._semaphore = asyncio.Semaphore(min(total_tasks, optimal))
        self._rate_limiter = AsyncLimiter(1, time_period=1.0 / max(arrival_rate, 0.01))

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Models: P1={self._model_p1}, P2={self._model_p2}, "
                  f"P3={self._model_p3}, P4={self._model_p4}, P5={self._model_p5}, "
                  f"P6={self._model_p6}, P7={self._model_p7}, P8={self._model_p8}, "
                  f"P9={self._model_p9}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"  Expected throughput: {arrival_rate:.1f}/s ({bottleneck} limited)")
            print(f"  Optimal by Little's Law: {little_law_conc}")
            print(f"  Concurrency (semaphore): {min(total_tasks, optimal)}")

    async def _process_all_async(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
    ) -> PipelineResult:
        """Main async entry: runs taxonomy (P1-P7) then codebook (P8-P9)."""
        await self._initialize_async_resources(
            label_mappings, partition_contexts, prompt_context, verbose
        )

        start_time = time.time()

        taxonomy = await self._process_taxonomy_async(
            label_mappings, partition_contexts, prompt_context, verbose
        )

        codebook_result = await self._process_codebook_async(
            taxonomy, partition_contexts, prompt_context, verbose
        )

        total_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Pipeline complete in {total_elapsed:.1f}s")

        return codebook_result

    async def _process_taxonomy_async(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
    ) -> TaxonomyResult:
        """Taxonomy stages P1-P7: facets, attributes, assignments."""
        start_time = time.time()

        # =================================================================
        # PHASE 1 (P1): Per-domain Facet Discovery (concurrent)
        # =================================================================
        if verbose:
            print(f"\n  Phase 1: Per-domain Facet Discovery...")

        facet_tasks = {}
        for name, mapping in sorted(label_mappings.items()):
            # Build excluded domains: all other domains
            excluded = [
                (other_name, partition_contexts[other_name].partition_definition)
                for other_name in partition_contexts
                if other_name != name
            ]
            facet_tasks[name] = self._discover_partition_facets(
                name, mapping.labels, partition_contexts[name],
                prompt_context, verbose,
                excluded_domains=excluded,
            )

        facet_results_list = await asyncio.gather(
            *facet_tasks.values(), return_exceptions=True
        )

        partition_facets: Dict[str, List[DiscoveredFacet]] = {}
        partition_n_labels: Dict[str, int] = {}
        partition_n_batches: Dict[str, int] = {}
        for name, result in zip(facet_tasks.keys(), facet_results_list):
            if isinstance(result, Exception):
                print(f"  Domain '{name}' FAILED: "
                      f"{type(result).__name__}: {result}")
            else:
                facets, n_labels, n_batches = result
                partition_facets[name] = facets
                partition_n_labels[name] = n_labels
                partition_n_batches[name] = n_batches

        phase1_elapsed = time.time() - start_time
        if verbose:
            total_facets = sum(len(f) for f in partition_facets.values())
            print(f"\n  Phase 1 done in {phase1_elapsed:.1f}s → "
                  f"{total_facets} facets across "
                  f"{len(partition_facets)} domains")
            for name in sorted(partition_facets.keys()):
                facet_names = [f.facet_name for f in partition_facets[name]]
                print(f"    {name}: {facet_names}")

        # =================================================================
        # PHASE 3 (P3): Per-domain Facet Assignment (concurrent)
        # =================================================================
        if verbose:
            print(f"\n  Phase 3: Per-domain Facet Assignment...")

        t_phase3 = time.time()

        assignment_tasks = {
            name: self._run_facet_assignment(
                name, partition_facets[name],
                label_mappings[name].ideas,
                partition_contexts[name], prompt_context,
            )
            for name in sorted(partition_facets.keys())
            if partition_facets[name] and label_mappings[name].ideas
        }

        assignment_results_list = await asyncio.gather(
            *assignment_tasks.values(), return_exceptions=True
        )

        # idea_id -> facet_name per domain
        partition_assignments: Dict[str, Dict[str, str]] = {}
        for name, result in zip(assignment_tasks.keys(), assignment_results_list):
            if isinstance(result, Exception):
                print(f"  Facet assignment '{name}' FAILED: "
                      f"{type(result).__name__}: {result}")
                partition_assignments[name] = {}
            else:
                partition_assignments[name] = result
                if verbose:
                    n_assigned = len(result)
                    n_ideas = len(label_mappings[name].ideas)
                    print(f"    {name}: {n_assigned}/{n_ideas} ideas assigned")

        t_phase3 = time.time() - t_phase3
        if verbose:
            total_assigned = sum(len(a) for a in partition_assignments.values())
            print(f"  Phase 3 done in {t_phase3:.1f}s → "
                  f"{total_assigned} ideas assigned to facets")

        # =================================================================
        # PHASE 4 (P4): Per-facet Attribute Discovery (concurrent)
        # =================================================================
        if verbose:
            print(f"\n  Phase 4: Per-facet Attribute Discovery...")

        t_phase4 = time.time()

        # Group ideas by (domain, facet) using P3 assignments
        domain_facet_ideas = self._group_ideas_by_facet(
            label_mappings, partition_facets, partition_assignments
        )

        attr_tasks = {}
        for (domain_name, facet_name), ideas in domain_facet_ideas.items():
            # Find the facet object for description
            facet_obj = None
            for f in partition_facets.get(domain_name, []):
                if f.facet_name == facet_name:
                    facet_obj = f
                    break

            if not facet_obj or not ideas:
                continue

            # Collect observations (labels) from the ideas in this facet
            observations = []
            for idea in ideas:
                label = format_label(idea, self._label_source, self._label_prefix, self._include_valence)
                if label:
                    observations.append(label)

            if not observations:
                continue

            # Build excluded facets: all other facets in the same domain
            excluded_f = [
                (f.facet_name, f.facet_description)
                for f in partition_facets.get(domain_name, [])
                if f.facet_name != facet_name
            ]

            task_key = f"{domain_name}::{facet_name}"
            attr_tasks[task_key] = self._discover_facet_attributes(
                domain_name=domain_name,
                facet_name=facet_name,
                facet_description=facet_obj.facet_description,
                observations=observations,
                part_context=partition_contexts[domain_name],
                prompt_context=prompt_context,
                excluded_facets=excluded_f,
            )

        attr_results_list = await asyncio.gather(
            *attr_tasks.values(), return_exceptions=True
        )

        # Collect attributes: domain -> facet -> [attributes]
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        # Also per-domain flat: facet -> [attributes]
        partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}

        for key, result in zip(attr_tasks.keys(), attr_results_list):
            domain_name, facet_name = key.split("::", 1)
            if isinstance(result, Exception):
                print(f"  Attribute discovery '{key}' FAILED: "
                      f"{type(result).__name__}: {result}")
            else:
                if domain_name not in domain_facet_attributes:
                    domain_facet_attributes[domain_name] = {}
                domain_facet_attributes[domain_name][facet_name] = result

                if domain_name not in partition_attributes:
                    partition_attributes[domain_name] = {}
                partition_attributes[domain_name][facet_name] = result

                if verbose:
                    print(f"    {domain_name}/{facet_name}: "
                          f"{len(result)} attributes")

        t_phase4 = time.time() - t_phase4
        if verbose:
            total_attrs = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            print(f"  Phase 4 done in {t_phase4:.1f}s → "
                  f"{total_attrs} attributes across "
                  f"{len(attr_tasks)} facets")

        # =================================================================
        # PHASE 6 (P6): Per-facet Attribute Assignment
        # =================================================================
        if verbose:
            print(f"\n  Phase 6: Per-facet Attribute Assignment...")

        t_phase6 = time.time()

        # Assign attributes to ideas, grouped by facet
        attribute_assignments: Dict[str, str] = {}  # idea_id -> attribute_name
        assign_tasks = {}

        for domain_name, facet_attrs in domain_facet_attributes.items():
            for facet_name, attributes in facet_attrs.items():
                if not attributes:
                    continue

                # Get ideas assigned to this facet
                facet_ideas = domain_facet_ideas.get(
                    (domain_name, facet_name), []
                )
                if not facet_ideas:
                    continue

                # Find facet description
                facet_obj = None
                for f in partition_facets.get(domain_name, []):
                    if f.facet_name == facet_name:
                        facet_obj = f
                        break

                if not facet_obj:
                    continue

                task_key = f"{domain_name}::{facet_name}"
                assign_tasks[task_key] = self._assign_attributes_to_ideas(
                    domain_name=domain_name,
                    facet_name=facet_name,
                    facet_description=facet_obj.facet_description,
                    attributes=attributes,
                    ideas=facet_ideas,
                    part_context=partition_contexts[domain_name],
                    prompt_context=prompt_context,
                )

        if assign_tasks:
            assign_results = await asyncio.gather(
                *assign_tasks.values(), return_exceptions=True
            )

            for task_key, result in zip(assign_tasks.keys(), assign_results):
                if isinstance(result, Exception):
                    print(f"  Attribute assignment '{task_key}' FAILED: "
                          f"{type(result).__name__}: {result}")
                else:
                    attribute_assignments.update(result)
                    if verbose:
                        domain_name, facet_name = task_key.split("::", 1)
                        print(f"    {domain_name}/{facet_name}: "
                              f"{len(result)} ideas assigned")

        t_phase6 = time.time() - t_phase6
        if verbose:
            print(f"  Phase 6 done in {t_phase6:.1f}s → "
                  f"{len(attribute_assignments)} ideas with attributes")

        # =================================================================
        # PHASE 7 (P7): Cross-facet Attribute Consolidation per domain
        # (now with frequency data from attribute assignments)
        # =================================================================
        if verbose:
            print(f"\n  Phase 7: Cross-facet Attribute Consolidation...")

        t_phase7 = time.time()

        consolidation_tasks = {}
        for domain_name, facet_attrs in domain_facet_attributes.items():
            # Only consolidate if domain has 2+ facets with attributes
            if len(facet_attrs) < 2:
                continue
            # Filter attribute_assignments to this domain
            domain_facet_ids = set(partition_assignments.get(domain_name, {}).keys())
            domain_attr_assigns = {
                iid: aname for iid, aname in attribute_assignments.items()
                if iid in domain_facet_ids
            }
            consolidation_tasks[domain_name] = self._consolidate_domain_attributes(
                domain_name=domain_name,
                facet_attributes=facet_attrs,
                partition_facets=partition_facets.get(domain_name, []),
                part_context=partition_contexts[domain_name],
                prompt_context=prompt_context,
                attribute_assignments=domain_attr_assigns,
            )

        if consolidation_tasks:
            consolidation_results = await asyncio.gather(
                *consolidation_tasks.values(), return_exceptions=True
            )

            for domain_name, result in zip(
                consolidation_tasks.keys(), consolidation_results
            ):
                if isinstance(result, Exception):
                    print(f"  P7 '{domain_name}' FAILED: "
                          f"{type(result).__name__}: {result}")
                    continue

                # Rebuild facet -> [attributes] from consolidated result
                before_count = sum(
                    len(a) for a in domain_facet_attributes[domain_name].values()
                )
                new_facet_attrs: Dict[str, List[DiscoveredAttribute]] = {}
                for attr in result:
                    facet = attr.parent_facet
                    if facet not in new_facet_attrs:
                        new_facet_attrs[facet] = []
                    # Convert ConsolidatedAttribute back to DiscoveredAttribute
                    new_facet_attrs[facet].append(DiscoveredAttribute(
                        attribute_name=attr.attribute_name,
                        attribute_description=attr.attribute_description,
                        parent_facet=attr.parent_facet,
                        example_observations=attr.example_observations,
                    ))

                domain_facet_attributes[domain_name] = new_facet_attrs
                partition_attributes[domain_name] = new_facet_attrs

                # Remap attribute assignments: old names → consolidated names
                remap = {}
                for attr in result:
                    for src in attr.source_attributes:
                        if src != attr.attribute_name:
                            remap[src] = attr.attribute_name
                if remap:
                    remapped = 0
                    for idea_id, attr_name in list(attribute_assignments.items()):
                        if attr_name in remap:
                            attribute_assignments[idea_id] = remap[attr_name]
                            remapped += 1

                if verbose:
                    after_count = sum(len(a) for a in new_facet_attrs.values())
                    remap_msg = f", {remapped} remapped" if remap else ""
                    print(f"    {domain_name}: {before_count} → "
                          f"{after_count} attributes "
                          f"({len(new_facet_attrs)} facets{remap_msg})")

        t_phase7 = time.time() - t_phase7
        if verbose:
            total_attrs_after = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            print(f"  Phase 7 done in {t_phase7:.1f}s → "
                  f"{total_attrs_after} consolidated attributes")

        taxonomy_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Taxonomy (P1-P7) complete in {taxonomy_elapsed:.1f}s")

        return TaxonomyResult(
            partition_n_labels=partition_n_labels,
            partition_n_batches=partition_n_batches,
            partition_facets=partition_facets,
            partition_assignments=partition_assignments,
            partition_attributes=partition_attributes,
            attribute_assignments=attribute_assignments,
        )

    async def _process_codebook_async(
        self,
        taxonomy: TaxonomyResult,
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
    ) -> PipelineResult:
        """Codebook stages P8-P9: code generation + consolidation."""
        # Unpack taxonomy result
        partition_facets = taxonomy.partition_facets
        partition_assignments = taxonomy.partition_assignments
        domain_facet_attributes = taxonomy.partition_attributes
        attribute_assignments = taxonomy.attribute_assignments

        start_time = time.time()

        # =================================================================
        # PHASE 8 (P8): Per-domain Code Generation
        # =================================================================
        if verbose:
            print(f"\n  Phase 8: Per-domain Code Generation...")

        t_phase8 = time.time()

        # Build one task per domain (no valence split — codes emerge naturally)
        p8_tasks = {}
        for domain_name in domain_facet_attributes:
            domain_attrs = domain_facet_attributes.get(domain_name, {})
            if not domain_attrs:
                continue

            # Filter attribute_assignments to this domain
            domain_facet_ids = set(partition_assignments.get(domain_name, {}).keys())
            domain_attr_assigns = {
                iid: aname for iid, aname in attribute_assignments.items()
                if iid in domain_facet_ids
            }

            # Build excluded domains: all other domains
            excluded = [
                (other_name, partition_contexts[other_name].partition_definition)
                for other_name in partition_contexts
                if other_name != domain_name
            ]

            p8_tasks[domain_name] = self._run_code_generation_from_attributes(
                {domain_name: domain_attrs}, prompt_context,
                attribute_assignments=domain_attr_assigns,
                domain_name=domain_name,
                domain_definition=partition_contexts[domain_name].partition_definition,
                excluded_domains=excluded,
            )

        p8_results = await asyncio.gather(*p8_tasks.values(), return_exceptions=True)

        # Collect all codes with provenance tracking
        all_codes = []
        code_provenance = {}  # code index -> domain_name
        codebook_narratives = []
        for key, result in zip(p8_tasks.keys(), p8_results):
            if isinstance(result, Exception):
                print(f"  P8 '{key}' FAILED: {type(result).__name__}: {result}")
            else:
                for code in result.codes:
                    code_provenance[len(all_codes)] = key
                    all_codes.append(code)
                codebook_narratives.append(f"[{key}] {result.scratchpad}")
                if verbose:
                    print(f"    {key}: {len(result.codes)} codes")

        t_phase8 = time.time() - t_phase8

        if verbose:
            print(f"\n  Phase 8 done in {t_phase8:.1f}s → {len(all_codes)} raw codes "
                  f"from {len(p8_tasks)} calls")

        # Compute idea frequencies per code (from attribute assignments)
        # Each code has source_attributes; count how many ideas map to those attrs
        attr_to_count: Dict[str, int] = {}
        for attr_name in attribute_assignments.values():
            attr_to_count[attr_name] = attr_to_count.get(attr_name, 0) + 1

        code_frequencies: Dict[int, int] = {}
        for idx, code in enumerate(all_codes):
            freq = sum(
                attr_to_count.get(attr, 0)
                for attr in (code.source_attributes or [])
            )
            code_frequencies[idx] = freq

        # =================================================================
        # PHASE 9 (P9): Cross-domain Codebook Consolidation
        # =================================================================
        if verbose:
            print(f"\n  Phase 9: Codebook Consolidation...")

        t_phase9 = time.time()

        if len(all_codes) > 0:
            consolidation_result = await self._consolidate_codebook(
                all_codes, code_provenance, prompt_context,
                code_frequencies=code_frequencies,
            )
            all_codes = consolidation_result.codes
            codebook_narratives.append(
                f"[consolidation] {consolidation_result.evaluation}"
            )

        codebook_narrative = "\n".join(codebook_narratives)

        t_phase9 = time.time() - t_phase9

        if verbose:
            print(f"\n  Phase 9 done in {t_phase9:.1f}s → {len(all_codes)} codes "
                  f"(after consolidation)")
            for i, code in enumerate(all_codes, 1):
                print(f"    {i}. {code.code_name}: {code.definition}")

        codebook_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Codebook (P8-P9) complete in {codebook_elapsed:.1f}s")

        # Build DomainResult for each domain
        partition_results = {}
        for name in partition_facets:
            # Collect attribute assignments for this domain
            domain_facet_ids = set(partition_assignments.get(name, {}).keys())
            domain_attr_assigns = {
                idea_id: attr_name
                for idea_id, attr_name in attribute_assignments.items()
                if idea_id in domain_facet_ids
            }
            partition_results[name] = DomainResult(
                partition_name=name,
                n_labels=taxonomy.partition_n_labels.get(name, 0),
                n_batches=taxonomy.partition_n_batches.get(name, 0),
                facets=partition_facets.get(name, []),
                facet_assignments=partition_assignments.get(name, {}),
                attributes=taxonomy.partition_attributes.get(name, {}),
                attribute_assignments=domain_attr_assigns,
            )

        return PipelineResult(
            partition_results=partition_results,
            codebook_narrative=codebook_narrative,
            codes=all_codes,
        )

    # =========================================================================
    # PHASE 1 (P1): PER-DOMAIN FACET DISCOVERY
    # =========================================================================

    async def _discover_partition_facets(
        self,
        partition_name: str,
        labels: List[str],
        part_context: DomainContext,
        prompt_context: PromptContext,
        verbose: bool = False,
        excluded_domains: Optional[List[tuple]] = None,
    ) -> tuple:
        """Run facet discovery + LLM consolidation for a single domain.

        Returns: (facets, n_labels, n_batches)
        """
        # Step 0: Create overlapping batches
        batches = self._create_batches(labels)
        n_batches = len(batches)

        if verbose:
            batch_size = self._compute_batch_size(len(labels))
            print(f"    Domain '{partition_name}': {len(labels)} observations, "
                  f"{n_batches} chunk(s) of ~{batch_size} "
                  f"(overlap {self._chunk_overlap:.0%})")

        # Step 1: FACET DISCOVERY (chunked, concurrent)
        t_discovery = time.time()
        chunk_facets = await self._run_facet_discovery(
            partition_name, batches, part_context, prompt_context,
            excluded_domains=excluded_domains,
        )
        t_discovery = time.time() - t_discovery

        # Count raw facets across all chunks
        n_raw = sum(len(cf) for cf in chunk_facets)
        non_empty_chunks = [cf for cf in chunk_facets if cf]

        # Step 2: Consolidation (hierarchical when needed)
        if len(non_empty_chunks) <= 1:
            # Single chunk: use directly, no consolidation needed
            facets = non_empty_chunks[0] if non_empty_chunks else []
        else:
            async def _facet_consolidate_fn(chunks):
                return await self._consolidate_facets(
                    partition_name, chunks, part_context, prompt_context,
                    excluded_domains=excluded_domains,
                )

            facets = await self._hierarchical_consolidate(
                chunk_facets, _facet_consolidate_fn,
                label=f"facets/{partition_name}",
            )

        if verbose:
            print(f"    Domain '{partition_name}' facets: "
                  f"{n_raw} raw → {len(facets)} consolidated "
                  f"[{t_discovery:.1f}s]")

        return facets, len(labels), n_batches

    async def _run_facet_discovery(
        self,
        partition_name: str,
        batches: List[List[str]],
        part_context: DomainContext,
        prompt_context: PromptContext,
        excluded_domains: Optional[List[tuple]] = None,
    ) -> List[List[DiscoveredFacet]]:
        """Discover facets from chunked observations (concurrent).

        Returns per-chunk facet lists (not flattened) so the consolidation
        step can show chunk provenance.
        """
        results = [None] * len(batches)

        async def process_chunk(chunk_idx: int, observations: List[str]):
            prompt = build_facet_discovery_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                partition_name=part_context.partition_name,
                partition_definition=part_context.partition_definition,
                observations=observations,
                excluded_domains=excluded_domains,
            )

            # Prompt capture (first chunk per domain)
            gate_key = f"qr_facets_{partition_name}"
            if (self._prompt_printer is not None
                    and chunk_idx == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="facet_discovery",
                    metadata={
                        "model": self._model_p1,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_facet_discovery,
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
                    prompt, FacetDiscoveryResult, self._max_tokens_facet_discovery
                )
                results[chunk_idx] = result
            except Exception as e:
                print(f"    FACET DISCOVERY '{partition_name}' chunk "
                      f"{chunk_idx + 1}/{len(batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                results[chunk_idx] = FacetDiscoveryResult(facets=[])

        await asyncio.gather(*(
            process_chunk(i, batch) for i, batch in enumerate(batches)
        ))

        return [r.facets for r in results if r is not None]

    # =========================================================================
    # PHASE 2 (P2): FACET CONSOLIDATION (per-domain, LLM-based)
    # =========================================================================

    async def _consolidate_facets(
        self,
        partition_name: str,
        chunk_facets: List[List[DiscoveredFacet]],
        part_context: DomainContext,
        prompt_context: PromptContext,
        excluded_domains: Optional[List[tuple]] = None,
    ) -> List[DiscoveredFacet]:
        """Consolidate chunk-level facet discoveries into a single coherent set.

        Follows the same pattern as step 3's _consolidate_domains().
        """
        # Format chunk results for the consolidation prompt
        formatted_chunks = []
        for idx, facets in enumerate(chunk_facets):
            if not facets:
                continue
            facet_lines = []
            for f in facets:
                examples = "; ".join(f.example_observations[:3])
                facet_lines.append(
                    f'    - "{f.facet_name}" — {f.facet_description} (examples: {examples})'
                )
            formatted_chunks.append(
                f"Chunk {idx + 1}:\n  Facets:\n" + "\n".join(facet_lines)
            )

        prompt = build_facet_consolidation_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_def=prompt_context.dimension_def,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            domain_name=part_context.partition_name,
            domain_definition=part_context.partition_definition,
            chunk_results="\n\n".join(formatted_chunks),
            excluded_domains=excluded_domains,
        )

        # Prompt capture
        gate_key = f"qr_facet_consolidation_{partition_name}"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="facet_consolidation",
                metadata={
                    "model": self._model_p2,
                    "temperature": 0.0,
                    "max_tokens": self._max_tokens_facet_discovery,
                    "language": prompt_context.language,
                    "partition_name": partition_name,
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        result = await self._llm_call(
            prompt, FacetConsolidatedResponse, self._max_tokens_facet_discovery,
            temperature=0.0, model=self._model_p2,
        )
        return result.facets

    # =========================================================================
    # HIERARCHICAL CONSOLIDATION (shared by P2 and P5)
    # =========================================================================

    async def _hierarchical_consolidate(
        self,
        chunk_results: List[list],
        consolidate_fn,
        label: str,
        _round: int = 0,
    ) -> list:
        """Recursively consolidate chunk results respecting capacity limits.

        Groups chunks into sub-groups of at most max_chunks_per_call,
        consolidates each group concurrently, then recurses on the
        intermediate results until everything fits in a single call.

        Works for both DiscoveredFacet (P2) and DiscoveredAttribute
        (P5) via the callback pattern — consolidate_fn encapsulates
        the prompt-building and LLM call logic.

        Args:
            chunk_results: Per-chunk discovery results (List[List[T]]).
            consolidate_fn: async (List[List[T]]) -> List[T]
            label: Human-readable label for logging.
            _round: Internal recursion counter (do not set externally).
        """
        max_c = self._consolidation_max_chunks_per_call
        max_i = self._consolidation_max_items_per_call
        max_r = self._consolidation_max_rounds

        non_empty = [c for c in chunk_results if c]
        if not non_empty:
            return []

        n_chunks = len(non_empty)
        total_items = sum(len(c) for c in non_empty)

        # Base case: fits in a single consolidation call
        if n_chunks <= max_c and total_items <= max_i:
            return await consolidate_fn(non_empty)

        # Safety cap: force merge if we've hit max rounds
        if _round >= max_r:
            print(f"    WARNING: {label}: hit max consolidation rounds ({max_r}), "
                  f"forcing final merge of {n_chunks} chunks / {total_items} items")
            return await consolidate_fn(non_empty)

        # Determine group size: respect both chunk count and item count limits.
        # Floor at 2: grouping by 1 would create no reduction and loop forever.
        group_size = max_c
        avg_items = total_items / n_chunks if n_chunks > 0 else 0
        while group_size > 2 and group_size * avg_items > max_i:
            group_size -= 1

        # Split into groups
        groups = [
            non_empty[i:i + group_size]
            for i in range(0, n_chunks, group_size)
        ]

        print(f"    {label}: hierarchical round {_round + 1}: "
              f"{n_chunks} chunks ({total_items} items) → "
              f"{len(groups)} groups of ≤{group_size}")

        # Consolidate each group concurrently
        intermediate = await asyncio.gather(*[
            consolidate_fn(group) for group in groups
        ], return_exceptions=True)

        # Handle failures: keep failed groups as-is (flatten their chunks)
        next_chunks = []
        for i, result in enumerate(intermediate):
            if isinstance(result, Exception):
                print(f"    WARNING: {label}: group {i + 1} consolidation failed: "
                      f"{type(result).__name__}: {result}")
                # Fall back to raw chunks from this group
                for chunk in groups[i]:
                    next_chunks.append(chunk)
            else:
                next_chunks.append(result)

        # Recurse with intermediate results as new chunks
        return await self._hierarchical_consolidate(
            next_chunks, consolidate_fn, label, _round=_round + 1,
        )

    # =========================================================================
    # PHASE 3 (P3): PER-DOMAIN FACET ASSIGNMENT
    # =========================================================================

    async def _run_facet_assignment(
        self,
        domain_name: str,
        facets: List[DiscoveredFacet],
        ideas: List,
        part_context: DomainContext,
        prompt_context: PromptContext,
    ) -> Dict[str, str]:
        """Assign all ideas in a domain to discovered facets.

        Returns: Dict[idea_id, facet_name]
        """
        # Build facet ID -> name mapping
        facet_id_to_name = {}
        for i, facet in enumerate(facets, 1):
            facet_id_to_name[f"F{i}"] = facet.facet_name

        # Batch ideas
        batch_size = self._facet_assignment_batch_size
        idea_batches = [
            ideas[i:i + batch_size]
            for i in range(0, len(ideas), batch_size)
        ]

        all_assignments: Dict[str, str] = {}

        async def process_batch(batch_idx: int, batch_ideas: List):
            prompt = build_facet_assignment_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=domain_name,
                domain_definition=part_context.partition_definition,
                facets=facets,
                other_label=None,  # No "other" for facet assignment
                ideas=batch_ideas,
            )

            # Prompt capture (first batch per domain)
            gate_key = f"qr_facet_assign_{domain_name}"
            if (self._prompt_printer is not None
                    and batch_idx == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="facet_assignment",
                    metadata={
                        "model": self._model_p3,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_facet_assignment,
                        "language": prompt_context.language,
                        "partition_name": domain_name,
                        "batch_number": batch_idx + 1,
                        "total_batches": len(idea_batches),
                        "n_facets": len(facets),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            try:
                result = await self._llm_call(
                    prompt, FacetAssignmentBatch, self._max_tokens_facet_assignment,
                    model=self._model_p3,
                )
                return result.assignments
            except Exception as e:
                print(f"    FACET ASSIGNMENT '{domain_name}' batch "
                      f"{batch_idx + 1}/{len(idea_batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                return []

        batch_results = await asyncio.gather(*(
            process_batch(i, batch)
            for i, batch in enumerate(idea_batches)
        ))

        # Retry pass: re-run failed batches (returned empty [])
        failed_batch_indices = [i for i, r in enumerate(batch_results) if len(r) == 0 and len(idea_batches[i]) > 0]
        if failed_batch_indices:
            print(f"    [RETRY PASS] Retrying {len(failed_batch_indices)} failed batches (facet assignment)...")
            retry_results = await asyncio.gather(*(
                process_batch(i, idea_batches[i])
                for i in failed_batch_indices
            ))
            recovered = 0
            for orig_idx, retry_result in zip(failed_batch_indices, retry_results):
                if len(retry_result) > 0:
                    batch_results[orig_idx] = retry_result
                    recovered += 1
            still_failed = len(failed_batch_indices) - recovered
            print(f"    [RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")

        # BP1: Build original idea lookup per batch for validation + content cross-check
        from difflib import SequenceMatcher
        batch_idea_lookups = [
            {idea.idea_id: idea for idea in batch} for batch in idea_batches
        ]

        for batch_idx, assignments in enumerate(batch_results):
            original_lookup = batch_idea_lookups[batch_idx] if batch_idx < len(batch_idea_lookups) else {}

            for assignment in assignments:
                # BP1: Validate returned idea_id exists in original batch
                original_idea = original_lookup.get(assignment.idea_id)
                if original_idea is None:
                    print(f"    ID DRIFT: LLM returned unexpected idea_id "
                          f"'{assignment.idea_id}' in batch {batch_idx} — skipping")
                    continue

                # Content cross-validation: compare returned instance vs original
                original_text = getattr(original_idea, 'idea', '') or getattr(original_idea, 'instance', '') or ''
                returned_text = getattr(assignment, 'idea', '') or ''
                if original_text and returned_text:
                    similarity = SequenceMatcher(
                        None, returned_text.lower(), original_text.lower()
                    ).ratio()
                    if similarity < 0.7:
                        print(f"    CONTENT DRIFT: idea '{original_idea.idea_id}' — "
                              f"returned '{returned_text}' doesn't match "
                              f"original '{original_text}' (similarity: {similarity:.2f}) — skipping")
                        continue

                # Fix 2 (BP6): Reject invalid facet_id — no raw string fallback
                facet_name = facet_id_to_name.get(assignment.assigned_facet_id)
                #if facet_name is None:
                #    print(f"    WARNING: Invalid facet_id '{assignment.assigned_facet_id}' "
                #          f"for idea '{original_idea.idea_id}' — skipping")
                #    continue

                # Fix 3: Detect duplicate assignments
                if original_idea.idea_id in all_assignments:
                    print(f"    WARNING: Duplicate assignment for '{original_idea.idea_id}' — "
                          f"overwriting '{all_assignments[original_idea.idea_id]}' with '{facet_name}'")

                # BP1: Always store under ORIGINAL idea_id
                all_assignments[original_idea.idea_id] = facet_name

        # BP3 + BP4: Iterate ALL originals, create fallback for missing, count reconciliation
        expected_all = {idea.idea_id for idea in ideas}
        missing = expected_all - set(all_assignments.keys())
        if missing:
            print(f"    WARNING: {len(missing)}/{len(ideas)} ideas received no facet assignment")
            for idea_id in missing:
                all_assignments[idea_id] = "__UNASSIGNED__"

        return all_assignments

    # =========================================================================
    # PHASE 6 (P6): PER-FACET ATTRIBUTE ASSIGNMENT
    # =========================================================================

    async def _assign_attributes_to_ideas(
        self,
        domain_name: str,
        facet_name: str,
        facet_description: str,
        attributes: List[DiscoveredAttribute],
        ideas: List,
        part_context: 'DomainContext',
        prompt_context: 'PromptContext',
    ) -> Dict[str, str]:
        """Assign each idea to an attribute within its facet.

        Returns dict of idea_id -> attribute_name.
        """
        # Build ID-to-name map
        attr_id_to_name = {}
        for i, attr in enumerate(attributes, 1):
            attr_id_to_name[f"A{i}"] = attr.attribute_name

        # Batch ideas (reuse facet assignment batch size)
        batch_size = self._facet_assignment_batch_size
        idea_batches = [
            ideas[i:i + batch_size]
            for i in range(0, len(ideas), batch_size)
        ]

        all_assignments: Dict[str, str] = {}

        async def process_batch(batch_idx: int, batch_ideas: List):
            prompt = build_attribute_assignment_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=domain_name,
                domain_definition=part_context.partition_definition,
                facet_name=facet_name,
                facet_description=facet_description,
                attributes=attributes,
                ideas=batch_ideas,
            )

            # Prompt capture (first batch per facet)
            gate_key = f"qr_attr_assign_{domain_name}_{facet_name}"
            if (self._prompt_printer is not None
                    and batch_idx == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="attribute_assignment",
                    metadata={
                        "model": self._model_p6,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_facet_assignment,
                        "language": prompt_context.language,
                        "partition_name": domain_name,
                        "facet_name": facet_name,
                        "batch_number": batch_idx + 1,
                        "total_batches": len(idea_batches),
                        "n_attributes": len(attributes),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            try:
                result = await self._llm_call(
                    prompt, AttributeAssignmentBatch,
                    self._max_tokens_facet_assignment,
                    model=self._model_p6,
                )
                return result.assignments
            except Exception as e:
                print(f"    ATTR ASSIGNMENT '{domain_name}/{facet_name}' "
                      f"batch {batch_idx + 1}/{len(idea_batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                return []

        batch_results = await asyncio.gather(*(
            process_batch(i, batch)
            for i, batch in enumerate(idea_batches)
        ))

        # Retry pass: re-run failed batches (returned empty [])
        failed_batch_indices = [i for i, r in enumerate(batch_results) if len(r) == 0 and len(idea_batches[i]) > 0]
        if failed_batch_indices:
            print(f"    [RETRY PASS] Retrying {len(failed_batch_indices)} failed batches (attribute assignment)...")
            retry_results = await asyncio.gather(*(
                process_batch(i, idea_batches[i])
                for i in failed_batch_indices
            ))
            recovered = 0
            for orig_idx, retry_result in zip(failed_batch_indices, retry_results):
                if len(retry_result) > 0:
                    batch_results[orig_idx] = retry_result
                    recovered += 1
            still_failed = len(failed_batch_indices) - recovered
            print(f"    [RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")

        # BP1: Build original idea lookup per batch for validation + content cross-check
        from difflib import SequenceMatcher
        batch_idea_lookups = [
            {idea.idea_id: idea for idea in batch} for batch in idea_batches
        ]

        for batch_idx, assignments in enumerate(batch_results):
            original_lookup = batch_idea_lookups[batch_idx] if batch_idx < len(batch_idea_lookups) else {}

            for assignment in assignments:
                # BP1: Validate returned idea_id exists in original batch
                original_idea = original_lookup.get(assignment.idea_id)
                if original_idea is None:
                    print(f"    ID DRIFT: LLM returned unexpected idea_id "
                          f"'{assignment.idea_id}' in attr batch {batch_idx} — skipping")
                    continue

                # Content cross-validation: compare returned instance vs original
                original_text = getattr(original_idea, 'idea', '') or getattr(original_idea, 'instance', '') or ''
                returned_text = getattr(assignment, 'idea', '') or ''
                if original_text and returned_text:
                    similarity = SequenceMatcher(
                        None, returned_text.lower(), original_text.lower()
                    ).ratio()
                    if similarity < 0.7:
                        print(f"    CONTENT DRIFT: idea '{original_idea.idea_id}' — "
                              f"returned '{returned_text}' doesn't match "
                              f"original '{original_text}' (similarity: {similarity:.2f}) — skipping")
                        continue

                # Fix 6 (BP6): Reject invalid attribute_id — no raw string fallback
                attr_name = attr_id_to_name.get(assignment.assigned_attribute_id)
                if attr_name is None:
                    print(f"    WARNING: Invalid attribute_id '{assignment.assigned_attribute_id}' "
                          f"for idea '{original_idea.idea_id}' — skipping")
                    continue

                # Detect duplicate assignments
                if original_idea.idea_id in all_assignments:
                    print(f"    WARNING: Duplicate attr assignment for '{original_idea.idea_id}' — "
                          f"overwriting '{all_assignments[original_idea.idea_id]}' with '{attr_name}'")

                # BP1: Always store under ORIGINAL idea_id
                all_assignments[original_idea.idea_id] = attr_name

        # BP3 + BP4: Iterate ALL originals, create fallback for missing
        expected_all = {idea.idea_id for idea in ideas}
        missing = expected_all - set(all_assignments.keys())
        if missing:
            print(f"    WARNING: {len(missing)}/{len(ideas)} ideas received no attribute assignment "
                  f"in facet '{facet_name}'")
            for idea_id in missing:
                all_assignments[idea_id] = "__UNASSIGNED__"

        return all_assignments

    # =========================================================================
    # PHASE 4 (P4): PER-FACET ATTRIBUTE DISCOVERY
    # =========================================================================

    async def _discover_facet_attributes(
        self,
        domain_name: str,
        facet_name: str,
        facet_description: str,
        observations: List[str],
        part_context: DomainContext,
        prompt_context: PromptContext,
        excluded_facets: Optional[List[tuple]] = None,
    ) -> List[DiscoveredAttribute]:
        """Discover attributes (L4) within a single facet.

        Mirrors P1's chunking strategy: when observations exceed
        batch_size_min, they are split into overlapping chunks, each
        chunk is processed independently, and results are consolidated
        via an LLM merge pass.
        """
        # Step 0: Create overlapping batches using P4-specific sizing
        batches = self._create_batches(
            observations,
            size_min=self._p4_batch_size_min,
            size_max=self._p4_batch_size_max,
            target=self._p4_target_batches,
            overlap=self._p4_chunk_overlap,
        )
        n_batches = len(batches)

        # Step 1: ATTRIBUTE DISCOVERY (chunked, concurrent)
        chunk_attributes = await self._run_attribute_discovery_chunks(
            domain_name, facet_name, facet_description,
            batches, part_context, prompt_context,
            excluded_facets=excluded_facets,
        )

        # Count raw attributes across all chunks
        n_raw = sum(len(ca) for ca in chunk_attributes)
        non_empty_chunks = [ca for ca in chunk_attributes if ca]

        # Step 2: Consolidation (hierarchical when needed)
        if len(non_empty_chunks) <= 1:
            # Single chunk or no results: use directly
            attributes = non_empty_chunks[0] if non_empty_chunks else []
        else:
            async def _attr_consolidate_fn(chunks):
                return await self._consolidate_attribute_chunks(
                    domain_name, facet_name, facet_description,
                    chunks, part_context, prompt_context,
                    excluded_facets=excluded_facets,
                )

            attributes = await self._hierarchical_consolidate(
                chunk_attributes, _attr_consolidate_fn,
                label=f"attrs/{domain_name}/{facet_name}",
            )

        if n_batches > 1:
            print(f"      {domain_name}/{facet_name}: "
                  f"{len(observations)} obs, {n_batches} chunks, "
                  f"{n_raw} raw → {len(attributes)} consolidated")

        return attributes

    async def _run_attribute_discovery_chunks(
        self,
        domain_name: str,
        facet_name: str,
        facet_description: str,
        batches: List[List[str]],
        part_context: DomainContext,
        prompt_context: PromptContext,
        excluded_facets: Optional[List[tuple]] = None,
    ) -> List[List[DiscoveredAttribute]]:
        """Discover attributes from chunked observations (concurrent).

        Returns per-chunk attribute lists (not flattened) so the consolidation
        step can show chunk provenance.
        """
        results = [None] * len(batches)

        async def process_chunk(chunk_idx: int, observations: List[str]):
            prompt = build_attribute_discovery_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=domain_name,
                domain_definition=part_context.partition_definition,
                facet_name=facet_name,
                facet_description=facet_description,
                observations=observations,
                excluded_facets=excluded_facets,
            )

            # Prompt capture (first chunk per facet)
            gate_key = f"qr_attributes_{domain_name}_{facet_name}"
            if (self._prompt_printer is not None
                    and chunk_idx == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="attribute_discovery",
                    metadata={
                        "model": self._model_p4,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_attribute_discovery,
                        "language": prompt_context.language,
                        "partition_name": domain_name,
                        "facet_name": facet_name,
                        "n_observations": len(observations),
                        "batch_number": chunk_idx + 1,
                        "total_batches": len(batches),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            try:
                result = await self._llm_call(
                    prompt, AttributeDiscoveryResult,
                    self._max_tokens_attribute_discovery,
                    model=self._model_p4,
                )
                results[chunk_idx] = result.attributes
            except Exception as e:
                print(f"    ATTRIBUTE DISCOVERY '{domain_name}/{facet_name}' "
                      f"chunk {chunk_idx + 1} FAILED: "
                      f"{type(e).__name__}: {e}")
                results[chunk_idx] = []

        tasks = [
            process_chunk(i, batch)
            for i, batch in enumerate(batches)
        ]
        await asyncio.gather(*tasks)
        return results

    async def _consolidate_attribute_chunks(
        self,
        domain_name: str,
        facet_name: str,
        facet_description: str,
        chunk_attributes: List[List[DiscoveredAttribute]],
        part_context: DomainContext,
        prompt_context: PromptContext,
        excluded_facets: Optional[List[tuple]] = None,
    ) -> List[DiscoveredAttribute]:
        """Consolidate chunk-level attribute discoveries into a single set.

        Mirrors _consolidate_facets() pattern.
        """
        # Format chunk results for the consolidation prompt
        formatted_chunks = []
        for idx, attributes in enumerate(chunk_attributes):
            if not attributes:
                continue
            attr_lines = []
            for a in attributes:
                examples = "; ".join(a.example_observations[:3])
                attr_lines.append(
                    f'    - "{a.attribute_name}" — {a.attribute_description} '
                    f'(examples: {examples})'
                )
            formatted_chunks.append(
                f"Chunk {idx + 1}:\n  Attributes:\n" + "\n".join(attr_lines)
            )

        prompt = build_attribute_chunk_consolidation_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_def=prompt_context.dimension_def,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            domain_name=domain_name,
            facet_name=facet_name,
            facet_description=facet_description,
            chunk_results="\n\n".join(formatted_chunks),
            excluded_facets=excluded_facets,
        )

        # Prompt capture
        gate_key = f"qr_attribute_chunk_consolidation_{domain_name}_{facet_name}"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="attribute_chunk_consolidation",
                metadata={
                    "model": self._model_p5,
                    "temperature": 0.0,
                    "max_tokens": self._max_tokens_attribute_discovery,
                    "language": prompt_context.language,
                    "domain_name": domain_name,
                    "facet_name": facet_name,
                    "n_chunks": len([c for c in chunk_attributes if c]),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        result = await self._llm_call(
            prompt, AttributeChunkConsolidatedResponse,
            self._max_tokens_attribute_discovery,
            temperature=0.0, model=self._model_p5,
        )
        return result.attributes

    # =========================================================================
    # PHASE 7 (P7): CROSS-FACET ATTRIBUTE CONSOLIDATION
    # =========================================================================

    async def _consolidate_domain_attributes(
        self,
        domain_name: str,
        facet_attributes: Dict[str, List[DiscoveredAttribute]],
        partition_facets: List[DiscoveredFacet],
        part_context: DomainContext,
        prompt_context: PromptContext,
        attribute_assignments: Optional[Dict[str, str]] = None,
    ) -> List[ConsolidatedAttribute]:
        """Consolidate attributes across facets within a domain.

        Takes all facets and their attributes for one domain, deduplicates
        overlapping attributes, and assigns each to its best-fitting facet.
        When attribute_assignments is provided, frequency counts are included.
        """
        # Compute attribute frequencies from assignments
        attr_counts: Dict[str, int] = {}
        if attribute_assignments:
            for attr_name in attribute_assignments.values():
                attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

        # Format facet->attributes block for the prompt
        lines = []
        for facet_name, attributes in sorted(facet_attributes.items()):
            # Find facet description
            facet_desc = ""
            for f in partition_facets:
                if f.facet_name == facet_name:
                    facet_desc = f.facet_description
                    break

            lines.append(f'Facet: "{facet_name}" — {facet_desc}')
            for attr in attributes:
                examples = "; ".join(attr.example_observations[:2])
                count = attr_counts.get(attr.attribute_name, 0)
                freq_tag = f" ({count} ideas)" if attribute_assignments else ""
                lines.append(
                    f'  - "{attr.attribute_name}"{freq_tag} — '
                    f'{attr.attribute_description} '
                    f'(examples: {examples})'
                )
            lines.append("")  # blank line between facets

        facet_attributes_block = "\n".join(lines)

        prompt = build_attribute_consolidation_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_def=prompt_context.dimension_def,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            domain_name=domain_name,
            domain_definition=part_context.partition_definition,
            facet_attributes_block=facet_attributes_block,
        )

        # Prompt capture
        gate_key = f"qr_attribute_consolidation_{domain_name}"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="attribute_consolidation",
                metadata={
                    "model": self._model_p7,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens_attribute_discovery,
                    "language": prompt_context.language,
                    "domain_name": domain_name,
                    "n_facets": len(facet_attributes),
                    "n_attributes_before": sum(
                        len(a) for a in facet_attributes.values()
                    ),
                }
            )
            self._captured_gates.add(gate_key)

        result = await self._llm_call(
            prompt, AttributeConsolidatedResponse,
            self._max_tokens_attribute_discovery,
            model=self._model_p7,
        )
        return result.attributes

    # =========================================================================
    # PHASE 8 (P8): CODE GENERATION FROM ATTRIBUTES
    # =========================================================================

    @staticmethod
    def _build_constrained_response_model(
        attribute_names: List[str],
    ):
        """Build a CodeGenerationFromAttributesResult with source_attributes
        constrained to an enum of valid attribute names."""
        if not attribute_names:
            return CodeGenerationFromAttributesResult

        # Create Literal type from known attribute names
        AttrLiteral = Literal[tuple(attribute_names)]

        # Dynamic CodeFromAttributes with constrained source_attributes
        ConstrainedCode = create_model(
            "CodeFromAttributes",
            code_name=(str, Field(..., description="Short code name (3-5 word noun phrase)")),
            definition=(str, Field(..., description="Clear definition of what this code covers (1-2 sentences)")),
            typical_indicators=(List[str], Field(..., description="Words or phrases that signal this code")),
            source_attributes=(List[AttrLiteral], Field(
                default_factory=list,
                description="Attribute names this code is derived from (must be exact names from the inventory)",
            )),
        )

        # Dynamic result model using the constrained code model
        ConstrainedResult = create_model(
            "CodeGenerationFromAttributesResult",
            scratchpad=(str, CodeGenerationFromAttributesResult.model_fields["scratchpad"]),
            codes=(List[ConstrainedCode], Field(..., description="Formal codes derived from the attribute inventory")),
        )

        return ConstrainedResult

    async def _run_code_generation_from_attributes(
        self,
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        prompt_context: PromptContext,
        attribute_assignments: Optional[Dict[str, str]] = None,
        domain_name: str = "",
        domain_definition: str = "",
        excluded_domains: Optional[List[tuple]] = None,
    ) -> CodeGenerationFromAttributesResult:
        """Generate codes from an attribute inventory (per-domain)."""
        prompt = build_code_from_attributes_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_def=prompt_context.dimension_def,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            domain_name=domain_name,
            domain_definition=domain_definition,
            domain_attributes=domain_facet_attributes,
            attribute_assignments=attribute_assignments,
            excluded_domains=excluded_domains,
        )

        # Collect all attribute names for enum constraint
        all_attr_names = [
            attr.attribute_name
            for facet_attrs in domain_facet_attributes.values()
            for attrs in facet_attrs.values()
            for attr in attrs
        ]
        response_model = self._build_constrained_response_model(all_attr_names)

        # Prompt capture
        domain_key = "::".join(domain_facet_attributes.keys())
        gate_key = f"qr_code_gen_{domain_key}"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="code_generation_from_attributes",
                metadata={
                    "model": self._model_p8,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens_code_from_attributes,
                    "language": prompt_context.language,
                    "n_domains": len(domain_facet_attributes),
                    "n_total_attributes": len(all_attr_names),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, response_model,
            self._max_tokens_code_from_attributes,
            model=self._model_p8,
        )

    # =========================================================================
    # PHASE 9 (P9): CODEBOOK CONSOLIDATION
    # =========================================================================

    async def _consolidate_codebook(
        self,
        raw_codes: list,
        code_provenance: dict,
        prompt_context: PromptContext,
        code_frequencies: Optional[Dict[int, int]] = None,
    ) -> CodebookConsolidationResult:
        """Consolidate per-domain codes into a final parsimonious codebook."""
        prompt = build_codebook_consolidation_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            raw_codes=raw_codes,
            code_provenance=code_provenance,
            code_frequencies=code_frequencies,
        )

        # Prompt capture
        gate_key = "qr_codebook_consolidation"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="codebook_consolidation",
                metadata={
                    "model": self._model_p9,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens_codebook_consolidation,
                    "language": prompt_context.language,
                    "n_raw_codes": len(raw_codes),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, CodebookConsolidationResult,
            self._max_tokens_codebook_consolidation,
            model=self._model_p9,
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
            model = self._model_p1

        response = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        return extract_rate_limits_from_response(response)

    # =========================================================================
    # SHARED LLM CALL
    # =========================================================================

    async def _llm_call(self, prompt: str, response_model, max_tokens: int,
                        temperature: float | None = None, model: str | None = None,
                        timeout: float = 120.0):
        """Make a rate-limited LLM call through the shared semaphore.

        Timeout is a generous safety net (default 120s for batched prompts).
        Only catches truly stuck requests — not slow-but-legitimate responses.
        """
        use_model = model or self._model_p1
        client = self._clients[use_model]
        async with self._semaphore:
            async with self._rate_limiter:
                return await asyncio.wait_for(
                    llm_create_async(
                        client=client,
                        model=use_model,
                        prompt=prompt,
                        response_model=response_model,
                        temperature=temperature if temperature is not None else self._temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _compute_batch_size(
        self, n_labels: int,
        *, size_min: Optional[int] = None, size_max: Optional[int] = None,
        target: Optional[int] = None,
    ) -> int:
        """Compute adaptive batch size.

        Accepts optional overrides; defaults to P1 config values.
        """
        bmin = size_min if size_min is not None else self._batch_size_min
        bmax = size_max if size_max is not None else self._batch_size_max
        tgt = target if target is not None else self._target_batches

        if n_labels <= bmin:
            return n_labels
        ideal = max(n_labels // tgt, 1)
        return max(bmin, min(ideal, bmax))

    def _create_batches(
        self, labels: List[str],
        *, size_min: Optional[int] = None, size_max: Optional[int] = None,
        target: Optional[int] = None, overlap: Optional[float] = None,
    ) -> List[List[str]]:
        """Split labels into overlapping batches.

        Each batch overlaps with the previous by chunk_overlap * batch_size
        labels. First batch starts at 0, subsequent batches step forward
        by (1 - overlap) * batch_size.

        Accepts optional overrides; defaults to P1 config values.
        """
        chunk_overlap = overlap if overlap is not None else self._chunk_overlap

        batch_size = self._compute_batch_size(
            len(labels), size_min=size_min, size_max=size_max, target=target,
        )
        if len(labels) <= batch_size:
            return [labels]

        ovlp = int(batch_size * chunk_overlap)
        step = max(batch_size - ovlp, 1)

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

    def _group_ideas_by_facet(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_facets: Dict[str, List[DiscoveredFacet]],
        partition_assignments: Dict[str, Dict[str, str]],
    ) -> Dict[tuple, List]:
        """Group ideas by (domain, facet) using P3 assignments.

        Returns: {(domain_name, facet_name): [idea_objects]}
        """
        groups: Dict[tuple, List] = {}

        for domain_name, assignments in partition_assignments.items():
            mapping = label_mappings.get(domain_name)
            if not mapping:
                continue

            # Build idea_id -> idea object lookup
            idea_lookup = {
                idea.idea_id: idea for idea in mapping.ideas
            }

            for idea_id, facet_name in assignments.items():
                idea = idea_lookup.get(idea_id)
                if idea is None:
                    continue
                key = (domain_name, facet_name)
                if key not in groups:
                    groups[key] = []
                groups[key].append(idea)

        return groups

    def _build_all_partition_contexts(
        self,
        partition_set: DomainSet,
    ) -> Dict[str, DomainContext]:
        """Build DomainContext for each partition."""
        contexts = {}
        for part in partition_set.partitions:
            contexts[part.partition_name] = DomainContext(
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
