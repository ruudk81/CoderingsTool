"""
Taxonomy Classifier: Inductive taxonomy discovery pipeline (P1-P7).

Pipeline (7 stages):
  P1.  Facet Discovery (chunked, per domain) — dimension-specific semantics
  P2.  Facet Consolidation (per domain, hierarchical merge)
  P3.  Facet Assignment (batched, per domain) — assign ideas to discovered facets
  P4.  Attribute Discovery (per facet within domain) — concrete observables
  P5.  Attribute Chunk Consolidation (per facet, hierarchical merge)
  P6.  Attribute Assignment (per facet) — assign ideas to discovered attributes
  P7.  Cross-facet Attribute Consolidation (per domain) — dedup across facets

Per-domain steps (P1–P7) run CONCURRENTLY.

Usage:
    from .classifier import TaxonomyClassifier
    from .config_classifier import CategoriesConfig

    processor = TaxonomyClassifier(config)
    result = processor.process(
        label_mappings={"identity": mapping, ...},
        partition_set=partition_set,
        dimension_name="ATTRIBUTES_ASSOCIATIONS",
        dimension_description="...",
        ...
    )
"""

import asyncio
import time
import numpy as np
import tiktoken
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

import nest_asyncio
from aiolimiter import AsyncLimiter

from utils.llm import (
    create_client, llm_create_async, RateLimits,
    extract_rate_limits_from_response,
)
from config import (
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG, OPENAI_API_KEY,
    API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM, get_reasoning_params,
)

from pipeline.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)
from pipeline.step_3_ideaExtractor.ideaExtractor import (
    ConcurrencyGate, ConcurrencyRamp,
    RealTimeTPMTracker, RealTimeRPMTracker,
    ApiLimits, compute_optimal_concurrency,
    PIDThroughputController,
    TiktokenOffsetLearner,
)
from utils.smoothRequester import (
    TokenBucket, LatencyTracker, ConcurrencyCircuitBreaker,
)
from pipeline.step_3_ideaExtractor.config_ideaExtractor import (
    RampUpConfig,
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
)

from pipeline.step_4_classifier.config_classifier import CategoriesConfig, ClassifierRampConfig
from utils.modelPerfStats import (
    load_stats, save_stats, update_phase_stats, apply_to_ramp_config, STATS_FILE,
)
from .domain_discoverer import PartitionLabelMapping
from .partition_labels import format_label
from .models_classifier import DomainSet, DomainDescription
from .prompts_classifier import (
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
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


# =============================================================================
# SHARED DATACLASSES
# =============================================================================

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
class PhaseRampState:
    """Per-phase 4-layer rate limiting state for gather-based dispatch.

    Full stack (large phases): all 4 layers active.
    Light mode (small phases): token_bucket/latency_tracker/circuit_breaker = None.
    """
    # Layer 1: Concurrency gate + completion-based ramp
    gate: ConcurrencyGate
    ramp: ConcurrencyRamp
    rpm_tracker: RealTimeRPMTracker
    tpm_tracker: RealTimeTPMTracker
    phase_name: str = ""
    total_tasks: int = 0
    completions: int = 0
    timeouts: int = 0
    done: bool = False

    # Layer 2: TPM token bucket (None for small phases)
    token_bucket: Optional[TokenBucket] = None

    # Layer 4: Adaptive timeout + circuit breaker (None for small phases)
    latency_tracker: Optional[LatencyTracker] = None
    circuit_breaker: Optional[ConcurrencyCircuitBreaker] = None

    # Warm-up calibration tracking
    actual_total_tokens: Optional[deque] = field(default_factory=lambda: deque(maxlen=100))
    warm_up_calibrated: bool = False
    warm_up_target_samples: int = 0
    estimated_avg_tokens: int = 3000


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class TaxonomyClassifier:
    """
    Taxonomy Classifier: Inductive taxonomy discovery pipeline (P1-P7).

    Pipeline (7 stages):
    P1.  FACET DISCOVERY:                   Per domain, chunked with overlap (concurrent)
    P2.  FACET CONSOLIDATION:               Per domain, hierarchical merge
    P3.  FACET ASSIGNMENT:                  Per domain, assign ideas to facets (concurrent)
    P4.  ATTRIBUTE DISCOVERY:               Per (domain, facet), discover attributes (concurrent)
    P5.  ATTRIBUTE CHUNK CONSOLIDATION:     Per facet, hierarchical merge
    P6.  ATTRIBUTE ASSIGNMENT:              Per facet, assign ideas to attributes (concurrent)
    P7.  CROSS-FACET ATTR CONSOLIDATION:    Per domain, dedup across facets
    """

    def __init__(self, config: CategoriesConfig, prompt_printer=None):
        self._model_p1 = config.qr_model_p1
        self._model_p2 = config.qr_model_p2
        self._model_p3 = config.qr_model_p3
        self._model_p4 = config.qr_model_p4
        self._model_p5 = config.qr_model_p5
        self._model_p6 = config.qr_model_p6
        self._model_p7 = config.qr_model_p7
        self._temperature = config.qr_temperature
        self._max_tokens_facet_discovery = config.qr_max_tokens_facet_discovery
        self._max_tokens_facet_assignment = config.qr_max_tokens_facet_assignment
        self._max_tokens_attribute_discovery = config.qr_max_tokens_attribute_discovery
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

        # Failure tracking for retry pass (P3, P6)
        self.failed_task_ids: set = set()

        # Concurrency ramp config
        self._ramp_config = config.ramp_config

        # Shared async resources — initialized in _initialize_async_resources()
        self._client = None
        self._semaphore = None
        self._rate_limiter = None
        self._fetched_limits = None
        self._pid_controller = None          # PID controller for arrival rate adjustment
        self._current_arrival_rate = None    # Tracks current arrival rate for PID
        self._tiktoken_offset_learner = TiktokenOffsetLearner()  # Learns tiktoken→API token offset
        self._perf_stats: dict = {}                              # Persistent stats loaded per run

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

    def process(
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
        """Run taxonomy stages (P1-P7): facets, attributes, assignments."""
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

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _initialize_async_resources(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
    ):
        """Initialize clients and rate limiters.

        Fetches real rate limits from API headers, computes Little's Law-based
        default concurrency. Per-phase gates (P1/P3/P4/P6) override the default
        with completion-based ramps.
        """
        # Load persistent performance stats (used for cold-start calibration per phase)
        self._perf_stats = load_stats()

        # Create one client per unique model
        unique_models = {self._model_p1, self._model_p2, self._model_p3, self._model_p4, self._model_p5, self._model_p6, self._model_p7}
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

        # Store for per-phase ramp creation
        self._fetched_limits = limits

        # --- Compute Little's Law-based concurrency ---
        cfg = self._ramp_config
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law = compute_optimal_concurrency(
            api_limits, cfg.estimated_latency_seconds, cfg.estimated_avg_tokens,
        )

        est_avg_tokens = cfg.estimated_avg_tokens
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / est_avg_tokens / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

        # Default semaphore for small phases (P2, P5, P7) — uses start_fraction
        default_conc = max(cfg.min_initial, int(little_law * cfg.start_fraction))
        self._semaphore = asyncio.Semaphore(default_conc)
        self._rate_limiter = AsyncLimiter(1, time_period=1.0 / max(arrival_rate, 0.01))
        self._current_arrival_rate = arrival_rate
        self._pid_controller = PIDThroughputController(target_utilization=0.80)

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Models: P1={self._model_p1}, P2={self._model_p2}, "
                  f"P3={self._model_p3}, P4={self._model_p4}, P5={self._model_p5}, "
                  f"P6={self._model_p6}, P7={self._model_p7}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"  Expected throughput: {arrival_rate:.1f}/s ({bottleneck} limited)")
            print(f"  Little's Law: {little_law} | "
                  f"Default concurrency: {default_conc} (small phases P2/P5/P7)")

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

        # Estimate total P1 chunks for ramp sizing
        total_p1_chunks = sum(
            len(self._create_batches(mapping.labels))
            for mapping in label_mappings.values()
        )
        p1_state = self._create_phase_ramp("P1", total_p1_chunks, model=self._model_p1,
                                            phase_key="step4_p1_facet_discovery")

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
                phase_state=p1_state,
            )

        facet_results_list = await self._run_with_ramp(
            facet_tasks.values(), p1_state
        )
        self._collect_phase_stats(p1_state, self._model_p1, "step4_p1_facet_discovery")

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

        # Per-phase ramp: estimate total batches across all domains
        total_p3_ideas = sum(
            len(label_mappings[name].ideas)
            for name in partition_facets
            if partition_facets[name] and label_mappings[name].ideas
        )
        est_p3_batches = max(1, total_p3_ideas // self._facet_assignment_batch_size)
        p3_state = self._create_phase_ramp("P3", est_p3_batches, model=self._model_p3,
                                            phase_key="step4_p3_facet_assignment")

        assignment_tasks = {
            name: self._run_facet_assignment(
                name, partition_facets[name],
                label_mappings[name].ideas,
                partition_contexts[name], prompt_context,
                gate=p3_state.gate,
                phase_state=p3_state,
            )
            for name in sorted(partition_facets.keys())
            if partition_facets[name] and label_mappings[name].ideas
        }

        assignment_results_list = await self._run_with_ramp(
            assignment_tasks.values(), p3_state
        )
        self._collect_phase_stats(p3_state, self._model_p3, "step4_p3_facet_assignment")

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
            attr_tasks[task_key] = (domain_name, facet_name, facet_obj, observations, excluded_f)

        # Create ramp — estimate inner chunks across all facets
        total_p4_chunks = 0
        for task_key, (dn, fn, fo, obs, ef) in attr_tasks.items():
            total_p4_chunks += max(1, len(self._create_batches(
                obs, size_min=self._p4_batch_size_min,
                size_max=self._p4_batch_size_max,
                target=self._p4_target_batches,
            )))
        p4_state = self._create_phase_ramp("P4", total_p4_chunks, model=self._model_p4,
                                            phase_key="step4_p4_attribute_discovery")

        # Build actual coroutines
        attr_coros = {}
        for task_key, (dn, fn, fo, obs, ef) in attr_tasks.items():
            attr_coros[task_key] = self._discover_facet_attributes(
                domain_name=dn,
                facet_name=fn,
                facet_description=fo.facet_description,
                observations=obs,
                part_context=partition_contexts[dn],
                prompt_context=prompt_context,
                excluded_facets=ef,
                phase_state=p4_state,
            )

        attr_results_list = await self._run_with_ramp(
            attr_coros.values(), p4_state
        )
        self._collect_phase_stats(p4_state, self._model_p4, "step4_p4_attribute_discovery")

        # Collect attributes: domain -> facet -> [attributes]
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        # Also per-domain flat: facet -> [attributes]
        partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}

        for key, result in zip(attr_coros.keys(), attr_results_list):
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
                  f"{len(attr_coros)} facets")

        # =================================================================
        # PHASE 6 (P6): Per-facet Attribute Assignment
        # =================================================================
        if verbose:
            print(f"\n  Phase 6: Per-facet Attribute Assignment...")

        t_phase6 = time.time()

        # Per-phase ramp: estimate total batches across all facets
        total_p6_ideas = sum(
            len(domain_facet_ideas.get((dn, fn), []))
            for dn, fa in domain_facet_attributes.items()
            for fn in fa
        )
        est_p6_batches = max(1, total_p6_ideas // self._facet_assignment_batch_size)
        p6_state = self._create_phase_ramp("P6", est_p6_batches, model=self._model_p6,
                                            phase_key="step4_p6_attribute_assignment")

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
                    gate=p6_state.gate,
                    phase_state=p6_state,
                )

        if assign_tasks:
            assign_results = await self._run_with_ramp(
                assign_tasks.values(), p6_state
            )
            self._collect_phase_stats(p6_state, self._model_p6, "step4_p6_attribute_assignment")

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
            if not facet_attrs:
                continue
            # Filter attribute_assignments to this domain
            domain_facet_ids = set(partition_assignments.get(domain_name, {}).keys())
            domain_attr_assigns = {
                iid: aname for iid, aname in attribute_assignments.items()
                if iid in domain_facet_ids
            }
            excluded = [
                (other_name, partition_contexts[other_name].partition_definition)
                for other_name in partition_contexts
                if other_name != domain_name
            ]
            consolidation_tasks[domain_name] = self._consolidate_domain_attributes(
                domain_name=domain_name,
                facet_attributes=facet_attrs,
                partition_facets=partition_facets.get(domain_name, []),
                part_context=partition_contexts[domain_name],
                prompt_context=prompt_context,
                attribute_assignments=domain_attr_assigns,
                excluded_domains=excluded,
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

                if not new_facet_attrs and before_count > 0:
                    print(f"    WARNING: P7 '{domain_name}' returned 0 valid attributes "
                          f"(had {before_count}) — keeping pre-consolidation state")
                else:
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

        save_stats(self._perf_stats)

        return TaxonomyResult(
            partition_n_labels=partition_n_labels,
            partition_n_batches=partition_n_batches,
            partition_facets=partition_facets,
            partition_assignments=partition_assignments,
            partition_attributes=partition_attributes,
            attribute_assignments=attribute_assignments,
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
        phase_state: PhaseRampState = None,
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
            phase_state=phase_state,
        )
        t_discovery = time.time() - t_discovery

        # Count raw facets across all chunks
        n_raw = sum(len(cf) for cf in chunk_facets)
        non_empty_chunks = [cf for cf in chunk_facets if cf]

        # Step 2: Consolidation (always run for quality refinement)
        if not non_empty_chunks:
            facets = []
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
        phase_state: PhaseRampState = None,
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
                    prompt, FacetDiscoveryResult, self._max_tokens_facet_discovery,
                    timeout=180.0,
                    gate=phase_state.gate if phase_state else None,
                    phase_state=phase_state,
                )
                results[chunk_idx] = result
            except Exception as e:
                print(f"    FACET DISCOVERY '{partition_name}' chunk "
                      f"{chunk_idx + 1}/{len(batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                results[chunk_idx] = FacetDiscoveryResult(scratchpad="chunk failed", facets=[])

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
            temperature=0.0, model=self._model_p2, timeout=180.0,
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
        gate=None,
        phase_state: PhaseRampState = None,
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
                    model=self._model_p3, timeout=60.0, gate=gate,
                    phase_state=phase_state,
                )
                return result.assignments
            except Exception as e:
                print(f"    FACET ASSIGNMENT '{domain_name}' batch "
                      f"{batch_idx + 1}/{len(idea_batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                self.failed_task_ids.update(idea.idea_id for idea in batch_ideas)
                return []

        self.failed_task_ids.clear()  # Reset before main pass
        batch_results = await asyncio.gather(*(
            process_batch(i, batch)
            for i, batch in enumerate(idea_batches)
        ))

        # Retry pass: re-run truly failed batches with reduced concurrency.
        # NOTE: Intentional divergence from strategy doc retry pattern.
        # Strategy says: reuse the same processing function with reduced concurrency.
        # P3/P6 use batch-level retry because assignment is batched (10 ideas per call);
        # individual-task retry doesn't apply — the unit of failure is the batch.
        failed_batch_indices = [
            i for i, batch in enumerate(idea_batches)
            if any(idea.idea_id in self.failed_task_ids for idea in batch)
        ]
        if failed_batch_indices:
            print(f"    [RETRY PASS] Retrying {len(failed_batch_indices)} failed batches (facet assignment)...")
            pre_retry_failed = set(self.failed_task_ids)
            self.failed_task_ids.clear()

            # Reduced concurrency: 10% of total batches, min 5
            retry_sem = asyncio.Semaphore(max(5, len(failed_batch_indices) // 10))

            async def retry_batch(batch_idx, batch_ideas):
                async with retry_sem:
                    return await process_batch(batch_idx, batch_ideas)

            retry_results = await asyncio.gather(*(
                retry_batch(i, idea_batches[i])
                for i in failed_batch_indices
            ))
            recovered = 0
            for orig_idx, retry_result in zip(failed_batch_indices, retry_results):
                if len(retry_result) > 0:
                    batch_results[orig_idx] = retry_result
                    recovered += 1
            still_failed = len(failed_batch_indices) - recovered
            print(f"    [RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")
            if self.failed_task_ids:
                print(f"    [RETRY PASS] Permanently failed ideas: {len(self.failed_task_ids)}")

        # BP1: Build original idea lookup per batch for validation + content cross-check
        from difflib import SequenceMatcher
        import re as _re

        def _normalize_for_comparison(text: str) -> str:
            """Strip template prefix and canonical_phrasing: for similarity comparison."""
            if ' → ' in text:
                text = text.split(' → ', 1)[1]
            text = _re.sub(r'\bcanonical_phrasing:\s*', '', text)
            return text.strip().lower()

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
                        None,
                        _normalize_for_comparison(returned_text),
                        _normalize_for_comparison(original_text),
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
        gate=None,
        phase_state: PhaseRampState = None,
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
                    model=self._model_p6, timeout=60.0, gate=gate,
                    phase_state=phase_state,
                )
                return result.assignments
            except Exception as e:
                print(f"    ATTR ASSIGNMENT '{domain_name}/{facet_name}' "
                      f"batch {batch_idx + 1}/{len(idea_batches)} FAILED: "
                      f"{type(e).__name__}: {e}")
                self.failed_task_ids.update(idea.idea_id for idea in batch_ideas)
                return []

        self.failed_task_ids.clear()  # Reset before main pass
        batch_results = await asyncio.gather(*(
            process_batch(i, batch)
            for i, batch in enumerate(idea_batches)
        ))

        # Retry pass: re-run truly failed batches with reduced concurrency.
        # NOTE: Intentional divergence from strategy doc retry pattern.
        # Strategy says: reuse the same processing function with reduced concurrency.
        # P3/P6 use batch-level retry because assignment is batched (10 ideas per call);
        # individual-task retry doesn't apply — the unit of failure is the batch.
        failed_batch_indices = [
            i for i, batch in enumerate(idea_batches)
            if any(idea.idea_id in self.failed_task_ids for idea in batch)
        ]
        if failed_batch_indices:
            print(f"    [RETRY PASS] Retrying {len(failed_batch_indices)} failed batches (attribute assignment)...")
            pre_retry_failed = set(self.failed_task_ids)
            self.failed_task_ids.clear()

            # Reduced concurrency: 10% of total batches, min 5
            retry_sem = asyncio.Semaphore(max(5, len(failed_batch_indices) // 10))

            async def retry_batch(batch_idx, batch_ideas):
                async with retry_sem:
                    return await process_batch(batch_idx, batch_ideas)

            retry_results = await asyncio.gather(*(
                retry_batch(i, idea_batches[i])
                for i in failed_batch_indices
            ))
            recovered = 0
            for orig_idx, retry_result in zip(failed_batch_indices, retry_results):
                if len(retry_result) > 0:
                    batch_results[orig_idx] = retry_result
                    recovered += 1
            still_failed = len(failed_batch_indices) - recovered
            print(f"    [RETRY PASS] Recovered: {recovered}, Still failed: {still_failed}")
            if self.failed_task_ids:
                print(f"    [RETRY PASS] Permanently failed ideas: {len(self.failed_task_ids)}")

        # BP1: Build original idea lookup per batch for validation + content cross-check
        from difflib import SequenceMatcher
        import re as _re

        def _normalize_for_comparison(text: str) -> str:
            """Strip template prefix and canonical_phrasing: for similarity comparison."""
            if ' → ' in text:
                text = text.split(' → ', 1)[1]
            text = _re.sub(r'\bcanonical_phrasing:\s*', '', text)
            return text.strip().lower()

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
                        None,
                        _normalize_for_comparison(returned_text),
                        _normalize_for_comparison(original_text),
                    ).ratio()
                    if similarity < 0.7:
                        print(f"    CONTENT DRIFT: idea '{original_idea.idea_id}' — "
                              f"returned '{returned_text}' doesn't match "
                              f"original '{original_text}' (similarity: {similarity:.2f}) — skipping")
                        continue

                # Fix 6 (BP6): Reject invalid attribute_id — single-attribute fallback
                attr_name = attr_id_to_name.get(assignment.assigned_attribute_id)
                if attr_name is None:
                    if len(attr_id_to_name) == 1:
                        attr_name = next(iter(attr_id_to_name.values()))
                    else:
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
        phase_state: PhaseRampState = None,
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
            phase_state=phase_state,
        )

        # Count raw attributes across all chunks
        n_raw = sum(len(ca) for ca in chunk_attributes)
        non_empty_chunks = [ca for ca in chunk_attributes if ca]

        # Step 2: Consolidation (always run for quality refinement)
        if not non_empty_chunks:
            attributes = []
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
        phase_state: PhaseRampState = None,
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
                    model=self._model_p4, timeout=90.0,
                    gate=phase_state.gate if phase_state else None,
                    phase_state=phase_state,
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
            temperature=0.0, model=self._model_p5, timeout=180.0,
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
        excluded_domains: Optional[List[tuple]] = None,
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
            excluded_domains=excluded_domains,
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
            model=self._model_p7, timeout=180.0,
        )
        return result.attributes

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

    # =========================================================================
    # 4-LAYER RATE LIMITING (per-phase)
    # =========================================================================

    def _collect_phase_stats(
        self, state: PhaseRampState, model: str, phase_key: str
    ) -> None:
        """Extract latency/token measurements from a completed phase and update perf stats."""
        if state.latency_tracker is None or len(state.latency_tracker.values) < 5:
            return
        tokens = list(state.actual_total_tokens)
        if not tokens:
            return
        measurements = {
            "p50_latency_s": state.latency_tracker.get_p50(),
            "p95_latency_s": state.latency_tracker.get_p95(),
            "avg_tokens": sum(tokens) / len(tokens),
        }
        if phase_key == "step4_p1_facet_discovery" and self._tiktoken_offset_learner.is_learned():
            measurements["tiktoken_offset"] = float(self._tiktoken_offset_learner.get_offset())
        update_phase_stats(self._perf_stats, model, phase_key, measurements, state.completions)

    def _create_phase_ramp(self, phase_name: str, num_tasks: int,
                           model: str = None, phase_key: str = None) -> PhaseRampState:
        """Create per-phase 4-layer rate limiting stack.

        Large phases (>=min_tasks): full stack with TokenBucket, LatencyTracker, CircuitBreaker.
        Small phases: gate + ramp only (layers 2-4 skipped).
        """
        from copy import copy
        cfg = copy(self._ramp_config)
        # Apply empirical cold-start estimates from previous runs if available
        if phase_key and model:
            apply_to_ramp_config(self._perf_stats, model, phase_key, cfg)
        headroom = DEFAULT_PROCESSING_CONFIG.rate_limit_headroom
        api_limits = ApiLimits(
            self._fetched_limits.tokens_per_minute,
            self._fetched_limits.requests_per_minute,
        )
        little_law = compute_optimal_concurrency(
            api_limits, cfg.estimated_latency_seconds, cfg.estimated_avg_tokens,
        )
        little_law_cap = min(little_law, num_tasks)

        # Reset PID so each phase starts fresh
        if self._pid_controller:
            self._pid_controller.reset()

        ramp_up_cfg = RampUpConfig(
            start_fraction=cfg.start_fraction,
            target_fraction=cfg.target_fraction,
            min_initial=cfg.min_initial,
            measurement_window_seconds=cfg.monitor_poll_interval,
            min_completions_per_step=cfg.min_completions_per_step,
        )

        half_little_law = int(little_law * cfg.start_fraction)
        initial = max(half_little_law, num_tasks)  # small phases: all at once
        initial = min(initial, num_tasks)           # never exceed task count
        target = min(int(little_law_cap * cfg.target_fraction), num_tasks)

        gate = ConcurrencyGate(initial)
        ramp = ConcurrencyRamp(ramp_up_cfg, little_law_cap, num_tasks)

        is_large = num_tasks >= cfg.circuit_breaker_min_tasks

        # Layer 2: TPM token bucket
        token_bucket = None
        if is_large:
            token_bucket = TokenBucket(int(self._fetched_limits.tokens_per_minute * headroom))

        # Layer 4a: Adaptive timeout via latency tracker
        latency_tracker = None
        if is_large:
            latency_tracker = LatencyTracker(
                DEFAULT_PROCESSING_CONFIG,
                timeout_floor=cfg.timeout_floor_seconds,
                default_timeout=cfg.default_timeout_seconds,
            )

        # Layer 4b: Circuit breaker
        circuit_breaker = None
        if cfg.circuit_breaker_enabled and is_large:
            circuit_breaker = ConcurrencyCircuitBreaker(
                DEFAULT_CIRCUIT_BREAKER_CONFIG, gate, initial,
            )

        # Warm-up calibration target (adaptive sample size)
        warm_up_target = 0
        if num_tasks >= cfg.warm_up_min_tasks_to_enable:
            if num_tasks <= 50:
                warm_up_target = cfg.warm_up_sample_min
            elif num_tasks >= 500:
                warm_up_target = cfg.warm_up_sample_max
            else:
                fraction = (num_tasks - 50) / (500 - 50)
                warm_up_target = int(cfg.warm_up_sample_min
                                     + fraction * (cfg.warm_up_sample_max - cfg.warm_up_sample_min))

        mode = "4-layer" if is_large else "light"
        print(f"    [{phase_name}] Little's Law: {little_law} | "
              f"Start: {initial} → Target: {target} ({num_tasks} tasks, {mode})")

        return PhaseRampState(
            gate=gate, ramp=ramp,
            rpm_tracker=RealTimeRPMTracker(window_seconds=60.0),
            tpm_tracker=RealTimeTPMTracker(window_seconds=60.0),
            phase_name=phase_name,
            total_tasks=num_tasks,
            token_bucket=token_bucket,
            latency_tracker=latency_tracker,
            circuit_breaker=circuit_breaker,
            warm_up_target_samples=warm_up_target,
            estimated_avg_tokens=cfg.estimated_avg_tokens,
        )

    async def _phase_monitor(self, state: PhaseRampState):
        """Background monitor: ramp + circuit breaker + warm-up + PID + progress."""
        start_time = time.monotonic()
        last_report_time = start_time
        last_reported_completions = -1
        last_pid_time = start_time

        while not state.done:
            await asyncio.sleep(self._ramp_config.monitor_poll_interval)
            now = time.monotonic()
            elapsed = now - start_time

            # --- Circuit breaker check (every tick) ---
            if state.circuit_breaker:
                action = state.circuit_breaker.check_and_adjust()
                if action and action in ('tripped', 'recovering', 'recovered'):
                    pass  # CB already adjusted gate.limit internally

            # --- PID arrival rate adjustment (every 20s, large phases only) ---
            if state.token_bucket and now - last_pid_time >= 20.0:
                await self._apply_pid_adjustment(state)
                last_pid_time = now

            # --- Feed completions to ramp ---
            if not state.ramp.is_done() and state.completions >= self._ramp_config.min_completions_per_step:
                rate = state.completions / elapsed if elapsed > 0 else 0
                state.ramp.record_measurement(
                    throughput=rate,
                    tpm_pct=0, rpm_pct=0,
                    completions_total=state.completions,
                    timeouts_total=state.timeouts,
                    duration=elapsed,
                )
                new_target = state.ramp.current_target()
                if new_target != state.gate.limit:
                    state.gate.set_limit(new_target)
                    if state.circuit_breaker:
                        state.circuit_breaker.baseline = new_target

            # --- Warm-up calibration (one-shot) ---
            if (not state.warm_up_calibrated
                    and state.warm_up_target_samples > 0
                    and state.actual_total_tokens is not None
                    and len(state.actual_total_tokens) >= state.warm_up_target_samples
                    and state.latency_tracker
                    and len(state.latency_tracker.values) >= state.warm_up_target_samples):
                self._calibrate_from_warm_up(state)

            # --- Progress line every 2s (suppress stale) ---
            if now - last_report_time >= 2.0:
                if state.completions != last_reported_completions:
                    last_report_time = now
                    last_reported_completions = state.completions
                    rate = state.completions / elapsed if elapsed > 0 else 0

                    current_tpm = await state.tpm_tracker.get_current_tpm()
                    current_rpm = await state.rpm_tracker.get_current_rpm()

                    tpm_limit = self._fetched_limits.tokens_per_minute if self._fetched_limits else 0
                    rpm_limit = self._fetched_limits.requests_per_minute if self._fetched_limits else 0
                    tpm_pct = (current_tpm / tpm_limit * 100) if tpm_limit > 0 else 0
                    rpm_pct = (current_rpm / rpm_limit * 100) if rpm_limit > 0 else 0

                    latency_info = ""
                    if state.latency_tracker and len(state.latency_tracker.values) >= 2:
                        vals = list(state.latency_tracker.values)
                        p50 = float(np.percentile(vals, 50))
                        p95 = float(np.percentile(vals, 95))
                        latency_info = f" | P50:{p50:.1f}s P95:{p95:.1f}s"

                    timeout_info = f" timeouts:{state.timeouts}" if state.timeouts > 0 else ""
                    cb_info = ""
                    if state.circuit_breaker and state.circuit_breaker.state != 'CLOSED':
                        cb_info = f" CB:{state.circuit_breaker.state}"
                    ramp_target = state.ramp._target
                    print(f"    [{state.phase_name}] {state.completions}/{state.total_tasks} "
                          f"({rate:.1f}/s) | "
                          f"TPM:{tpm_pct:.0f}% RPM:{rpm_pct:.0f}% "
                          f"Conc:{state.gate.active}/{state.gate.limit}→{ramp_target}"
                          f"{latency_info}{timeout_info}{cb_info}")

    async def _run_with_ramp(self, coros, state: PhaseRampState):
        """Run coroutines via gather with a background ramp monitor.

        Returns gather results. Monitor exits when gather completes.
        """
        async def _work():
            results = await asyncio.gather(*coros, return_exceptions=True)
            state.done = True
            return results

        results, _ = await asyncio.gather(_work(), self._phase_monitor(state))

        # Phase summary: latency distribution + token stats
        if state.latency_tracker and state.latency_tracker.values:
            vals = list(state.latency_tracker.values)
            p10 = float(np.percentile(vals, 10))
            p50 = float(np.percentile(vals, 50))
            p95 = float(np.percentile(vals, 95))
            avg_tok = int(np.mean(list(state.actual_total_tokens))) if state.actual_total_tokens else 0
            print(f"    [{state.phase_name}] Latency: P10={p10:.1f}s P50={p50:.1f}s P95={p95:.1f}s | "
                  f"avg_tokens={avg_tok:,} | "
                  f"{state.completions} ok, {state.timeouts} timeouts")

        return results

    def _calibrate_from_warm_up(self, state: PhaseRampState) -> None:
        """One-shot calibration: update token estimate and recompute Little's Law.

        Fires once per phase after enough completions. Uses measured latency (P10)
        and token counts to recalculate optimal concurrency and arrival rate.
        """
        measured_avg_tokens = int(np.mean(list(state.actual_total_tokens)))
        measured_latency = float(np.percentile(list(state.latency_tracker.values), 10))

        old_avg = state.estimated_avg_tokens
        state.estimated_avg_tokens = measured_avg_tokens

        # Recalculate Little's Law with measured data
        api_limits = ApiLimits(
            self._fetched_limits.tokens_per_minute,
            self._fetched_limits.requests_per_minute,
        )
        new_little_law = compute_optimal_concurrency(
            api_limits, measured_latency, measured_avg_tokens,
        )
        new_little_law_cap = min(new_little_law, state.total_tasks)

        # Recalibrate ramp (preserves congestion detection state)
        if not state.ramp.is_done():
            state.ramp.recalibrate(new_little_law_cap)
            new_start = state.ramp.current_target()
            state.gate.set_limit(new_start)
            if state.circuit_breaker:
                state.circuit_breaker.baseline = new_start

        # Recalculate arrival rate
        headroom = DEFAULT_PROCESSING_CONFIG.rate_limit_headroom
        new_arrival_rate = min(
            self._fetched_limits.requests_per_minute * headroom / 60,
            self._fetched_limits.tokens_per_minute * headroom / measured_avg_tokens / 60,
        )
        self._rate_limiter = AsyncLimiter(1, time_period=1.0 / max(new_arrival_rate, 0.01))
        self._current_arrival_rate = new_arrival_rate  # Keep PID in sync after warm-up

        conc_target = state.ramp._target
        print(f"\n    {'='*60}")
        print(f"    WARM-UP CALIBRATION [{state.phase_name}] "
              f"({len(state.actual_total_tokens)} samples)")
        print(f"      Latency: {measured_latency:.1f}s (P10 measured)")
        print(f"      avg_tokens: {old_avg} (estimate) -> {measured_avg_tokens} (measured)")
        print(f"      Little's Law: {new_little_law_cap}")
        print(f"      Concurrency: {state.gate.limit} → {conc_target}")
        print(f"      Arrival rate: {new_arrival_rate:.2f}/s")
        print(f"    {'='*60}")

        state.warm_up_calibrated = True

    async def _apply_pid_adjustment(self, state: PhaseRampState) -> None:
        """Adjust AsyncLimiter arrival rate via PID controller based on TPM utilization.

        Called every 20s from _phase_monitor. Asymmetric: aggressive when under-utilizing,
        gentle when over-utilizing. No-ops when limits are unknown or adjustment is trivial.
        """
        if self._current_arrival_rate is None or self._pid_controller is None:
            return
        if not self._fetched_limits or not self._fetched_limits.tokens_per_minute:
            return

        current_tpm = await state.tpm_tracker.get_current_tpm()
        tpm_limit = self._fetched_limits.tokens_per_minute
        utilization = current_tpm / tpm_limit if tpm_limit > 0 else 0.0

        adjustment = self._pid_controller.compute_adjustment(utilization)
        if abs(adjustment - 1.0) < 0.01:
            return

        old_rate = self._current_arrival_rate
        new_rate = old_rate * adjustment
        headroom = DEFAULT_PROCESSING_CONFIG.rate_limit_headroom
        rpm_max = self._fetched_limits.requests_per_minute * headroom / 60
        new_rate = max(0.5, min(rpm_max, new_rate))

        if abs(new_rate - old_rate) / max(old_rate, 0.001) < 0.02:
            return

        self._rate_limiter = AsyncLimiter(1, time_period=1.0 / new_rate)
        self._current_arrival_rate = new_rate

    # =========================================================================
    # SHARED LLM CALL
    # =========================================================================

    async def _llm_call(self, prompt: str, response_model, max_tokens: int,
                        temperature: float | None = None, model: str | None = None,
                        timeout: float = 120.0, gate=None, phase_state: PhaseRampState = None):
        """Make a rate-limited LLM call through the 4-layer stack.

        Layer ordering (outside → inside):
          1. ConcurrencyGate — limits in-flight requests
          2. Adaptive timeout — computed AFTER gate (uses live latency data)
          3. TokenBucket — TPM pacing (blocks until tokens available)
          4. AsyncLimiter — RPM pacing (inter-request spacing)
          5. Circuit breaker — records completion/timeout

        Small phases (phase_state layers are None) skip layers 2-5 gracefully.
        """
        use_model = model or self._model_p1
        client = self._clients[use_model]
        concurrency_ctx = gate if gate is not None else self._semaphore

        # Token estimate for TPM bucket (conservative until warm-up fires)
        est_tokens = max_tokens
        if phase_state and phase_state.estimated_avg_tokens:
            est_tokens = phase_state.estimated_avg_tokens
        # Apply learned tiktoken→API offset to improve bucket pre-acquisition accuracy
        est_tokens += self._tiktoken_offset_learner.get_offset()

        async with concurrency_ctx:                                     # Layer 1: Concurrency
            # Adaptive timeout: compute AFTER gate (fresh latency data)
            effective_timeout = timeout
            if phase_state and phase_state.latency_tracker:
                effective_timeout = phase_state.latency_tracker.get_timeout()

            # Layer 2: TPM token bucket
            if phase_state and phase_state.token_bucket:
                await phase_state.token_bucket.wait_and_acquire(est_tokens)

            async with self._rate_limiter:                              # Layer 3: RPM
                task_start = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        llm_create_async(
                            client=client,
                            model=use_model,
                            prompt=prompt,
                            response_model=response_model,
                            temperature=temperature if temperature is not None else self._temperature,
                            max_tokens=max_tokens,
                            **get_reasoning_params(use_model),
                        ),
                        timeout=effective_timeout,
                    )

                    # Record latency for adaptive timeout
                    elapsed = time.monotonic() - task_start
                    if phase_state and phase_state.latency_tracker:
                        phase_state.latency_tracker.add(elapsed)

                    # Layer 4: Circuit breaker — record success
                    if phase_state and phase_state.circuit_breaker:
                        phase_state.circuit_breaker.record_completion()

                    if phase_state is not None:
                        phase_state.completions += 1
                        await phase_state.rpm_tracker.record()

                        # Extract actual tokens from response
                        raw = getattr(result, '_raw_response', None)
                        usage = getattr(raw, 'usage', None) if raw else None
                        actual_tokens = None
                        if usage:
                            actual_tokens = (
                                getattr(usage, 'prompt_tokens', 0)
                                + getattr(usage, 'completion_tokens', 0)
                                + getattr(usage, 'input_tokens', 0)
                                + getattr(usage, 'output_tokens', 0)
                            )
                            await phase_state.tpm_tracker.record(actual_tokens)
                        else:
                            await phase_state.tpm_tracker.record(max_tokens)

                        # Track for warm-up calibration
                        if actual_tokens and phase_state.actual_total_tokens is not None:
                            phase_state.actual_total_tokens.append(actual_tokens)

                        # Learn tiktoken→API offset for future estimates
                        if actual_tokens:
                            try:
                                encoding = tiktoken.encoding_for_model(use_model)
                                tiktoken_count = len(encoding.encode(prompt))
                                input_tokens = (
                                    getattr(usage, 'input_tokens', 0)
                                    or getattr(usage, 'prompt_tokens', 0)
                                ) if usage else 0
                                if input_tokens > 0:
                                    self._tiktoken_offset_learner.record(tiktoken_count, input_tokens)
                            except Exception:
                                pass  # Non-critical — don't disrupt the pipeline

                        # Reconcile token bucket (return overestimate)
                        if actual_tokens and phase_state.token_bucket:
                            delta = actual_tokens - est_tokens
                            if delta != 0:
                                await phase_state.token_bucket.reconcile(delta)

                    return result

                except asyncio.TimeoutError:
                    # Layer 4: Circuit breaker — record timeout
                    if phase_state and phase_state.circuit_breaker:
                        phase_state.circuit_breaker.record_timeout()
                    if phase_state is not None:
                        phase_state.timeouts += 1
                    raise

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
