"""
Inductive Code Generation pipeline for Category Discovery v3.

Pipeline:
  P1. Facet Discovery (chunked, per domain) — dimension-specific semantics
  P2. Facet Assignment (batched, per domain) — assign ideas to discovered facets
  P3. Attribute Discovery (per facet within domain) — concrete observables
  P4. Code Generation from Attributes (cross-domain) — derive codebook codes

Per-domain steps (P1, P2, P3) run CONCURRENTLY. P4 is sequential.

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

from development.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)

from .config_classNcoder_exp import CategoriesConfig
from .domain_discoverer import PartitionLabelMapping
from .partition_labels import format_label
from .models_exp import DomainSet, DomainDescription
from .prompts_exp import (
    MECECode,
    # P1: Facet Discovery
    build_facet_discovery_prompt,
    FacetDiscoveryResult,
    DiscoveredFacet,
    # P1.5: Facet Consolidation
    build_facet_consolidation_prompt,
    FacetConsolidatedResponse,
    # P2: Facet Assignment
    build_facet_assignment_prompt,
    FacetAssignmentBatch,
    # P3: Attribute Discovery
    build_attribute_discovery_prompt,
    AttributeDiscoveryResult,
    DiscoveredAttribute,
    # P3 chunk consolidation
    build_attribute_chunk_consolidation_prompt,
    AttributeChunkConsolidatedResponse,
    # P3.5: Attribute Consolidation (cross-facet within domain)
    build_attribute_consolidation_prompt,
    AttributeConsolidatedResponse,
    ConsolidatedAttribute,
    # P4: Code Generation from Attributes
    build_code_from_attributes_prompt,
    CodeGenerationFromAttributesResult,
    CodeFromAttributes,
    FormalCode,
    # P4.5: Codebook Consolidation
    build_codebook_consolidation_prompt,
    CodebookConsolidationResult,
    # Bridge
    convert_codes_to_mece_categories,
    convert_formal_codes_to_mece_categories,
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


@dataclass
class PipelineResult:
    """Complete pipeline output (v3)."""
    partition_results: Dict[str, DomainResult]
    codebook_categories: List[MECECode]
    codebook_narrative: str
    codes: List[FormalCode]

    @property
    def codebook(self) -> List[MECECode]:
        """Returns the final MECE codebook entries."""
        return self.codebook_categories

    # Backward compatibility alias
    @property
    def consolidated_codes(self) -> List[FormalCode]:
        return self.codes


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

    Pipeline:
    P1. FACET DISCOVERY:        Per domain, chunked with overlap (concurrent)
    P1b. PROGRAMMATIC DEDUP:    Per domain, case-insensitive facet name merge (no LLM)
    P2. FACET ASSIGNMENT:       Per domain, assign ideas to facets (concurrent)
    P3. ATTRIBUTE DISCOVERY:    Per (domain, facet), discover attributes (concurrent)
    P4. CODE GENERATION:        Cross-domain, derive codes from attributes
    """

    def __init__(self, config: CategoriesConfig, prompt_printer=None):
        self._model_p1 = config.qr_model_p1
        self._model_p1_5 = config.qr_model_p1_5
        self._model_p2 = config.qr_model_p2
        self._model_p3 = config.qr_model_p3
        self._model_p4 = config.qr_model_p4
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
        self._consolidation_chunk_size = config.consolidation_chunk_size
        self._consolidation_max_rounds = config.consolidation_max_rounds

        # Batch sizing — P3 (attribute discovery)
        self._p3_batch_size_min = config.p3_batch_size_min
        self._p3_batch_size_max = config.p3_batch_size_max
        self._p3_target_batches = config.p3_target_batches
        self._p3_chunk_overlap = config.p3_chunk_overlap

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
        """Process all partitions through the v3 inductive coding pipeline."""
        print(f"\n{'='*70}")
        print(f"INDUCTIVE CODE GENERATION v3: Category Discovery")
        print(f"{'='*70}")

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

        # Build shared prompt context
        dataset_context_section = self._build_dataset_context_section(dataset_context)

        prompt_context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            dimension_def=dimension_def,
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
            total_ideas = sum(len(m.ideas) for m in active_partitions.values())
            n_partitions = len(active_partitions)
            print(f"  Processing {n_partitions} domains concurrently "
                  f"({total_labels} observations, {total_ideas} ideas)")
            print(f"  Pipeline: P1 facet discovery → P2 facet assignment → "
                  f"P3 attribute discovery → P4 code generation")

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

    async def _process_all_async(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
    ) -> PipelineResult:
        """Main async entry: bootstrap → P1 facet discovery → P2 facet assignment →
        P3 attribute discovery → P4 code generation."""
        # Create one client per unique model
        unique_models = {self._model_p1, self._model_p1_5, self._model_p2, self._model_p3, self._model_p4}
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
        # P2 tasks estimated per domain (will refine after P1)
        est_p2_batches = n_domains * 3  # rough estimate
        est_p3_tasks = n_domains * 5  # rough estimate
        total_tasks = total_p1_chunks + est_p2_batches + est_p3_tasks + 1  # +1 for P4

        self._semaphore = asyncio.Semaphore(min(total_tasks, optimal))
        self._rate_limiter = AsyncLimiter(1, time_period=1.0 / max(arrival_rate, 0.01))

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Models: P1={self._model_p1}, P1.5={self._model_p1_5}, "
                  f"P2={self._model_p2}, P3={self._model_p3}, P4={self._model_p4}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"  Expected throughput: {arrival_rate:.1f}/s ({bottleneck} limited)")
            print(f"  Optimal by Little's Law: {little_law_conc}")
            print(f"  Concurrency (semaphore): {min(total_tasks, optimal)}")

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
        # PHASE 2 (P2): Per-domain Facet Assignment (concurrent)
        # =================================================================
        if verbose:
            print(f"\n  Phase 2: Per-domain Facet Assignment...")

        t_phase2 = time.time()

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

        t_phase2 = time.time() - t_phase2
        if verbose:
            total_assigned = sum(len(a) for a in partition_assignments.values())
            print(f"  Phase 2 done in {t_phase2:.1f}s → "
                  f"{total_assigned} ideas assigned to facets")

        # =================================================================
        # PHASE 3 (P3): Per-facet Attribute Discovery (concurrent)
        # =================================================================
        if verbose:
            print(f"\n  Phase 3: Per-facet Attribute Discovery...")

        t_phase3 = time.time()

        # Group ideas by (domain, facet) using P2 assignments
        domain_facet_ideas = self._group_ideas_by_facet(
            label_mappings, partition_facets, partition_assignments
        )

        # Track valences per (domain, facet) for P4 valence split
        facet_valences: Dict[tuple, set] = {}
        for (domain_name, facet_name), ideas in domain_facet_ideas.items():
            valences = {getattr(idea, 'valence', '0') or '0' for idea in ideas}
            facet_valences[(domain_name, facet_name)] = valences

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

        t_phase3 = time.time() - t_phase3
        if verbose:
            total_attrs = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            print(f"  Phase 3 done in {t_phase3:.1f}s → "
                  f"{total_attrs} attributes across "
                  f"{len(attr_tasks)} facets")

        # =================================================================
        # PHASE 3.5 (P3.5): Cross-facet Attribute Consolidation per domain
        # =================================================================
        if verbose:
            print(f"\n  Phase 3.5: Cross-facet Attribute Consolidation...")

        t_phase35 = time.time()

        consolidation_tasks = {}
        for domain_name, facet_attrs in domain_facet_attributes.items():
            # Only consolidate if domain has 2+ facets with attributes
            if len(facet_attrs) < 2:
                continue
            consolidation_tasks[domain_name] = self._consolidate_domain_attributes(
                domain_name=domain_name,
                facet_attributes=facet_attrs,
                partition_facets=partition_facets.get(domain_name, []),
                part_context=partition_contexts[domain_name],
                prompt_context=prompt_context,
            )

        if consolidation_tasks:
            consolidation_results = await asyncio.gather(
                *consolidation_tasks.values(), return_exceptions=True
            )

            for domain_name, result in zip(
                consolidation_tasks.keys(), consolidation_results
            ):
                if isinstance(result, Exception):
                    print(f"  P3.5 '{domain_name}' FAILED: "
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

                if verbose:
                    after_count = sum(len(a) for a in new_facet_attrs.values())
                    print(f"    {domain_name}: {before_count} → "
                          f"{after_count} attributes "
                          f"({len(new_facet_attrs)} facets)")

        t_phase35 = time.time() - t_phase35
        if verbose:
            total_attrs_after = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            print(f"  Phase 3.5 done in {t_phase35:.1f}s → "
                  f"{total_attrs_after} consolidated attributes")

        # =================================================================
        # PHASE 4 (P4): Per-domain Code Generation with valence split
        # =================================================================
        if verbose:
            print(f"\n  Phase 4: Per-domain Code Generation (valence split)...")

        t_phase4 = time.time()

        # Build per-domain tasks, split by valence
        p4_tasks = {}
        for domain_name in domain_facet_attributes:
            pos_attrs, neg_attrs = self._split_attributes_by_valence(
                domain_facet_attributes, facet_valences, domain_name
            )
            if pos_attrs:
                p4_tasks[f"{domain_name}::pos"] = self._run_code_generation_from_attributes(
                    {domain_name: pos_attrs}, prompt_context, valence_label="positive"
                )
            if neg_attrs:
                p4_tasks[f"{domain_name}::neg"] = self._run_code_generation_from_attributes(
                    {domain_name: neg_attrs}, prompt_context, valence_label="negative"
                )

        p4_results = await asyncio.gather(*p4_tasks.values(), return_exceptions=True)

        # Collect all codes with provenance tracking
        all_codes = []
        code_provenance = {}  # code index -> "domain::valence"
        codebook_narratives = []
        for key, result in zip(p4_tasks.keys(), p4_results):
            if isinstance(result, Exception):
                print(f"  P4 '{key}' FAILED: {type(result).__name__}: {result}")
            else:
                for code in result.codes:
                    code_provenance[len(all_codes)] = key
                    all_codes.append(code)
                codebook_narratives.append(f"[{key}] {result.evaluation}")
                if verbose:
                    print(f"    {key}: {len(result.codes)} codes")

        t_phase4 = time.time() - t_phase4

        if verbose:
            print(f"\n  Phase 4 done in {t_phase4:.1f}s → {len(all_codes)} raw codes "
                  f"from {len(p4_tasks)} calls")

        # =================================================================
        # PHASE 4.5: Cross-domain Codebook Consolidation
        # =================================================================
        if verbose:
            print(f"\n  Phase 4.5: Codebook Consolidation...")

        t_phase45 = time.time()

        if len(all_codes) > 0:
            consolidation_result = await self._consolidate_codebook(
                all_codes, code_provenance, prompt_context
            )
            all_codes = consolidation_result.codes
            codebook_narratives.append(
                f"[consolidation] {consolidation_result.evaluation}"
            )

        codebook = convert_codes_to_mece_categories(all_codes)
        codebook_narrative = "\n".join(codebook_narratives)

        t_phase45 = time.time() - t_phase45

        if verbose:
            print(f"\n  Phase 4.5 done in {t_phase45:.1f}s → {len(all_codes)} codes "
                  f"(after consolidation)")
            for i, code in enumerate(all_codes, 1):
                print(f"    {i}. {code.code_name}: {code.definition}")

        total_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Pipeline complete in {total_elapsed:.1f}s")

        # Build DomainResult for each domain
        partition_results = {}
        for name in partition_facets:
            partition_results[name] = DomainResult(
                partition_name=name,
                n_labels=partition_n_labels.get(name, 0),
                n_batches=partition_n_batches.get(name, 0),
                facets=partition_facets.get(name, []),
                facet_assignments=partition_assignments.get(name, {}),
                attributes=partition_attributes.get(name, {}),
            )

        return PipelineResult(
            partition_results=partition_results,
            codebook_categories=codebook,
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

        # Step 2: Consolidation
        if len(non_empty_chunks) <= 1:
            # Single chunk: use directly, no consolidation needed
            facets = non_empty_chunks[0] if non_empty_chunks else []
        else:
            # Multiple chunks: LLM consolidation
            facets = await self._consolidate_facets(
                partition_name, chunk_facets, part_context, prompt_context,
                excluded_domains=excluded_domains,
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
    # PHASE 1.5: FACET CONSOLIDATION (per-domain, LLM-based)
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
            partition_name=part_context.partition_name,
            partition_definition=part_context.partition_definition,
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
                    "model": self._model_p1_5,
                    "language": prompt_context.language,
                    "partition_name": partition_name,
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        result = await self._llm_call(
            prompt, FacetConsolidatedResponse, self._max_tokens_facet_discovery,
            temperature=0.0, model=self._model_p1_5,
        )
        return result.facets

    # =========================================================================
    # PHASE 2 (P2): PER-DOMAIN FACET ASSIGNMENT
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
                        "model": self._model_p2,
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
                    model=self._model_p2,
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

        for assignments in batch_results:
            for assignment in assignments:
                facet_name = facet_id_to_name.get(
                    assignment.assigned_facet_id,
                    assignment.assigned_facet_id,
                )
                all_assignments[assignment.idea_id] = facet_name

        return all_assignments

    # =========================================================================
    # PHASE 3 (P3): PER-FACET ATTRIBUTE DISCOVERY
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
        # Step 0: Create overlapping batches using P3-specific sizing
        batches = self._create_batches(
            observations,
            size_min=self._p3_batch_size_min,
            size_max=self._p3_batch_size_max,
            target=self._p3_target_batches,
            overlap=self._p3_chunk_overlap,
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

        # Step 2: Consolidation (mirrors P1.5 pattern)
        if len(non_empty_chunks) <= 1:
            # Single chunk or no results: use directly
            attributes = non_empty_chunks[0] if non_empty_chunks else []
        else:
            # Multiple chunks: LLM consolidation
            attributes = await self._consolidate_attribute_chunks(
                domain_name, facet_name, facet_description,
                chunk_attributes, part_context, prompt_context,
                excluded_facets=excluded_facets,
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
                        "model": self._model_p3,
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
                    model=self._model_p3,
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
                    "model": self._model_p1_5,
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
            temperature=0.0, model=self._model_p1_5,
        )
        return result.attributes

    # =========================================================================
    # PHASE 3.5 (P3.5): CROSS-FACET ATTRIBUTE CONSOLIDATION
    # =========================================================================

    async def _consolidate_domain_attributes(
        self,
        domain_name: str,
        facet_attributes: Dict[str, List[DiscoveredAttribute]],
        partition_facets: List[DiscoveredFacet],
        part_context: DomainContext,
        prompt_context: PromptContext,
    ) -> List[ConsolidatedAttribute]:
        """Consolidate attributes across facets within a domain.

        Takes all facets and their attributes for one domain, deduplicates
        overlapping attributes, and assigns each to its best-fitting facet.
        """
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
                lines.append(
                    f'  - "{attr.attribute_name}" — {attr.attribute_description} '
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
                    "model": self._model_p3,
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
            model=self._model_p3,
        )
        return result.attributes

    # =========================================================================
    # PHASE 4 (P4): CODE GENERATION FROM ATTRIBUTES
    # =========================================================================

    def _split_attributes_by_valence(
        self,
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        facet_valences: Dict[tuple, set],
        domain_name: str,
    ) -> tuple:
        """Split a domain's facet->attributes into positive/neutral vs negative."""
        pos_attrs = {}
        neg_attrs = {}
        for facet_name, attributes in domain_facet_attributes.get(domain_name, {}).items():
            valences = facet_valences.get((domain_name, facet_name), {"0"})
            has_pos = bool(valences & {"+", "0"})
            has_neg = "-" in valences
            if has_pos:
                pos_attrs[facet_name] = attributes
            if has_neg:
                neg_attrs[facet_name] = attributes
        return pos_attrs, neg_attrs

    async def _run_code_generation_from_attributes(
        self,
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        prompt_context: PromptContext,
        valence_label: str = "",
    ) -> CodeGenerationFromAttributesResult:
        """Generate codes from an attribute inventory (per-domain, valence-scoped)."""
        prompt = build_code_from_attributes_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            domain_attributes=domain_facet_attributes,
            valence_label=valence_label,
        )

        # Prompt capture
        domain_key = "::".join(domain_facet_attributes.keys())
        gate_key = f"qr_code_gen_{domain_key}_{valence_label}"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            total_attrs = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="code_generation_from_attributes",
                metadata={
                    "model": self._model_p4,
                    "language": prompt_context.language,
                    "n_domains": len(domain_facet_attributes),
                    "n_total_attributes": total_attrs,
                    "valence": valence_label or "all",
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, CodeGenerationFromAttributesResult,
            self._max_tokens_code_from_attributes,
            model=self._model_p4,
        )

    # =========================================================================
    # PHASE 4.5: CODEBOOK CONSOLIDATION
    # =========================================================================

    async def _consolidate_codebook(
        self,
        raw_codes: list,
        code_provenance: dict,
        prompt_context: PromptContext,
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
                    "model": self._model_p4,
                    "language": prompt_context.language,
                    "n_raw_codes": len(raw_codes),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        return await self._llm_call(
            prompt, CodebookConsolidationResult,
            self._max_tokens_codebook_consolidation,
            model=self._model_p4,
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
                        temperature: float | None = None, model: str | None = None):
        """Make a rate-limited LLM call through the shared semaphore."""
        use_model = model or self._model_p1
        client = self._clients[use_model]
        async with self._semaphore:
            async with self._rate_limiter:
                return await llm_create_async(
                    client=client,
                    model=use_model,
                    prompt=prompt,
                    response_model=response_model,
                    temperature=temperature if temperature is not None else self._temperature,
                    max_tokens=max_tokens,
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
        """Group ideas by (domain, facet) using P2 assignments.

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
