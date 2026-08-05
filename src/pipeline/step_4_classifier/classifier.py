"""
Taxonomy Classifier: inductive taxonomy discovery (P1-P9).

Pipeline (9 stages, P3/P7 optional):
  P1.  Facet Discovery (chunked, per domain) — dimension-specific semantics
  P2.  Facet Consolidation (per domain) — conceptual merge + orthogonal formulation
  P3.  Facet Review (per domain, optional) — rewrite name/description for
       orthogonality, write a boundary test, flag suspected concept overlap.
       Schema-enforced rewrite-and-flag only: the facet SET cannot change.
       Skipped for the two standing drain domains and for single-facet domains.
  P4.  Facet Assignment (per domain) — assign ideas to consolidated facets
  P5.  Attribute Discovery (per facet within domain) — concrete observables
  P6.  Attribute Consolidation, round 1 (per facet) — dedup the chunk discoveries
  P7.  Attribute Review (per domain, optional) — rewrite name/description for
       orthogonality, flag suspected concept overlap across the domain's
       consolidated attribute set. Schema-enforced rewrite-and-flag only: the
       attribute SET cannot change. Skipped for the two standing drain domains
       and for domains with fewer than 2 attributes in total.
  P8.  Attribute Assignment (per facet) — assign ideas to attributes
  P9.  Attribute Consolidation, round 2 (per facet, AFTER assignment) — judged on
       real counts and real contents, with four actions: merge / split / widen /
       move. Scope is one facet: no cross-facet or cross-domain structural
       consolidation exists, so a structure merge can never relocate an idea's
       facet or domain — because per-idea (domain, facet) is DERIVED from where
       the attribute lives, that relocation would otherwise move every idea in
       the bucket at once. When a group of ideas belongs elsewhere the IDEAS
       move and the structure stays put.

Per-domain steps run CONCURRENTLY; P9 runs per facet after P8.

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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import nest_asyncio

from utils.llm import (
    RateLimits, token_tracker,
    fetch_rate_limits as llm_fetch_rate_limits,
)
from config import (
    API_PROVIDER, DEFAULT_PROCESSING_CONFIG, FALLBACK_TPM, FALLBACK_RPM,
    get_azure_route, get_reasoning_params,
)

from pipeline.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)
from utils.smoothRequester import SmoothRequester

from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from .domain_discoverer import PartitionLabelMapping
from .partition_labels import format_label
from .taxonomy_health import drain_domains
from .dedup import dedup_exact_attributes, dedup_exact_facets
from utils.embedder import SharedEmbedder
from .assignment_batching import (
    LabelRep, facet_card_text, group_label_reps, make_batches,
    shortlist_indices, validate_batch_response,
)
from models import DomainSet, DomainDescription
from .prompts_classifier import (
    # P1: Axis Discovery
    build_axis_discovery_prompt,
    AxisSystemResponse,
    # P1b: Tagged Facet Discovery
    build_tagged_facet_discovery_prompt,
    build_tagged_facet_discovery_model,
    # P1: Facet Discovery
    build_facet_discovery_prompt,
    FacetDiscoveryResult,
    DiscoveredFacet,
    # Facet Assignment
    build_facet_assignment_prompt_single,
    FacetAssignmentResult,
    # Facet Assignment (batch)
    build_batch_facet_assignment_model,
    build_facet_assignment_prompt_batch,
    # Facet Consolidation (in-axis, post-assignment)
    build_in_axis_consolidation_prompt,
    InAxisConsolidatedResponse,
    # Attribute Discovery
    build_attribute_discovery_prompt,
    AttributeDiscoveryResult,
    DiscoveredAttribute,
    # Attribute Assignment
    build_attribute_assignment_prompt_single,
    AttributeAssignmentResult,
    # P9: In-facet Attribute Consolidation (post-assignment)
    build_in_facet_consolidation_prompt,
    build_neighbour_block,
    InFacetConsolidatedResponse,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


# =============================================================================
# HELPERS
# =============================================================================

def sample_axis_observations(
    labels: List[str], *, target: int = 120, cap: int = 150,
) -> List[str]:
    """Deterministic spread sample across a domain's ordered unique labels
    (P1 input). Every k-th label, arithmetic stride — no randomness, so the
    same input always yields the same sample, and the stride covers the full
    range instead of just the labels that happen to land in the first chunk.

    n <= cap: return every label (already small enough). n > cap: take
    `target` evenly-spaced indices from 0 to n-1, then clip to `cap`.
    """
    n = len(labels)
    if n <= cap:
        return list(labels)

    stride = n / target
    seen: Set[int] = set()
    idxs: List[int] = []
    for i in range(target):
        idx = min(int(i * stride), n - 1)
        if idx not in seen:
            seen.add(idx)
            idxs.append(idx)
    return [labels[i] for i in idxs[:cap]]


def validate_and_repair_axis_system(
    response: Optional[AxisSystemResponse],
) -> Optional[AxisSystemResponse]:
    """Validate a P1 response. `DiscoveredAxis` carries no sub-structure to
    repair (axis_name, axis_description, value_range only), so the only
    remaining check is the axis count: not 1-4 axes fails the whole response
    (returns None) so the caller falls back to the pre-existing untagged path
    for that domain.
    """
    if response is None or not response.axes:
        return None
    if not (1 <= len(response.axes) <= 4):
        return None

    return response


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
    boundary_test: str = ""
    exclusions: List[str] = field(default_factory=list)


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
    """Output of taxonomy stages P1-P9."""
    partition_n_labels: Dict[str, int]
    partition_n_batches: Dict[str, int]
    partition_facets: Dict[str, List[DiscoveredFacet]]
    partition_assignments: Dict[str, Dict[str, str]]  # domain -> {idea_id -> facet_name}
    partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]]  # domain -> {facet -> [attrs]}
    attribute_assignments: Dict[str, str]  # idea_id -> attribute_name
    # P1: discovered axis system per domain (model_dump, verbatim). Empty
    # unless axis_first_enabled. Written to a JSON log by the runner —
    # deliberately NOT part of the shared TaxonomyResultsCache model.
    axis_systems: Dict[str, dict] = field(default_factory=dict)
    # Pre-P9 snapshots (before the post-assignment consolidation round remaps)
    raw_partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = field(default_factory=dict)
    raw_attribute_assignments: Dict[str, str] = field(default_factory=dict)
    # Assignment confidence scores (0.0-1.0)
    facet_confidence: Dict[str, float] = field(default_factory=dict)
    attribute_confidence: Dict[str, float] = field(default_factory=dict)
    # Assignment valence (+, -, 0)
    facet_valence: Dict[str, str] = field(default_factory=dict)
    attribute_valence: Dict[str, str] = field(default_factory=dict)
    # P9 provenance: one entry per action taken, so every merge/split/move is
    # auditable after the fact. Written to a JSON file by the runner; deliberately
    # NOT put in the shared cache model, which production also uses.
    consolidation_log: List[Dict] = field(default_factory=list)
    # P7 overlap flags: attr_a/facet_a/attr_b/facet_b/reason/decision_rule, one

# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class TaxonomyClassifier:
    """
    Taxonomy Classifier: inductive taxonomy discovery (P1-P9).

    Pipeline (9 stages, P3/P7 optional):
    P1.  FACET DISCOVERY:              Per domain, chunked with overlap (concurrent)
    P2.  FACET CONSOLIDATION:          Per domain, conceptual merge + orthogonal labels
    P3.  FACET REVIEW (optional):      Per domain, rewrite + flag only — sharpens
                                       names/descriptions, writes a boundary test,
                                       flags suspected overlap. Skipped for drain
                                       domains and single-facet domains.
    P4.  FACET ASSIGNMENT:             Per domain, assign ideas to facets (concurrent)
    P5.  ATTRIBUTE DISCOVERY:          Per (domain, facet), discover attributes (concurrent)
    P6.  ATTRIBUTE CONSOLIDATION r1:   Per facet, dedup the chunk discoveries
    P7.  ATTRIBUTE REVIEW (optional):  Per domain, rewrite + flag only — sharpens
                                       names/descriptions across the domain's
                                       consolidated attribute set, flags suspected
                                       overlap. Skipped for drain domains and
                                       domains with fewer than 2 attributes total.
    P8.  ATTRIBUTE ASSIGNMENT:         Per facet, assign ideas to attributes (concurrent)
    P9.  ATTRIBUTE CONSOLIDATION r2:   Per facet, AFTER assignment — real counts and
                                       real contents; merge / split / widen / move.
                                       The facet is fixed and is not in the schema:
                                       no cross-facet or cross-domain structural
                                       consolidation exists, so a merge can never
                                       relocate an idea across facets or domains.
    """

    def __init__(self, config: CategoriesConfig, prompt_printer=None, cost_tracker=None):
        self.cost_tracker = cost_tracker
        self._config = config
        self._axis_first_enabled = config.axis_first_enabled
        self._model_p1 = config.qr_model_p1
        self._model_p2 = config.qr_model_p2
        self._model_p4 = config.qr_model_p4
        self._model_p5 = config.qr_model_p5
        self._model_p6 = config.qr_model_p6
        self._model_p7 = config.qr_model_p7
        self._model_p8 = config.qr_model_p8
        self._model_p9 = config.qr_model_p9

        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_4_taxonomy_classifier", {
                "p1_axis_discovery": self._model_p1,
                "p2_facet_discovery": self._model_p2,
                "p4_facet_assignment": self._model_p4,
                "p5_facet_consolidation": self._model_p5,
                "p6_attribute_discovery": self._model_p6,
                "p7_attribute_assignment": self._model_p7,
                "p8_attribute_consolidation": self._model_p8,
                "p9_valence_merge": self._model_p9,
            })

        self._temperature = config.qr_temperature
        self._max_tokens_facet_discovery = config.qr_max_tokens_facet_discovery
        self._max_tokens_facet_assignment = config.qr_max_tokens_facet_assignment
        self._max_tokens_attribute_discovery = config.qr_max_tokens_attribute_discovery
        self._max_tokens_consolidation = config.qr_max_tokens_consolidation
        self._p9_contents_top_n = config.p9_contents_top_n

        # Batch sizing — P1 (facet discovery)
        self._batch_size_min = config.batch_size_min
        self._batch_size_max = config.batch_size_max
        self._target_batches = config.target_batches
        self._chunk_overlap = config.chunk_overlap
        self._consolidation_max_chunks_per_call = config.consolidation_max_chunks_per_call
        self._consolidation_max_items_per_call = config.consolidation_max_items_per_call
        self._consolidation_max_rounds = config.consolidation_max_rounds

        # Batch sizing — P5 (attribute discovery)
        self._p4_batch_size_min = config.p4_batch_size_min
        self._p4_batch_size_max = config.p4_batch_size_max
        self._p4_target_batches = config.p4_target_batches
        self._p4_chunk_overlap = config.p4_chunk_overlap

        # Label source for observation formatting
        self._label_source = config.label_source
        self._label_prefix = config.label_prefix

        # P4 facet assignment — batch mode
        self._assign_batch_enabled = config.facet_assignment_batch_enabled
        self._assign_batch_k = config.facet_assignment_batch_k
        self._assign_shortlist_enabled = config.facet_assignment_shortlist_enabled
        self._assign_shortlist_k = config.facet_assignment_shortlist_k
        self._assign_label_dedup = config.facet_assignment_label_dedup

        # Prompt capture (optional)
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()

        self._debug_stop_after_phase = config.debug_stop_after_phase

        # Assignment confidence scores and valence (populated by P4/P8 parse_fns)
        self._facet_confidence: Dict[str, float] = {}
        self._attribute_confidence: Dict[str, float] = {}
        self._facet_valence: Dict[str, str] = {}
        self._attribute_valence: Dict[str, str] = {}

        # P1: validated axis system per domain (populated by _process_taxonomy_async
        # when axis_first_enabled; empty otherwise). Carried on the instance because
        # TaxonomyResult only needs the model_dump for the final return.
        self.axis_systems: Dict[str, AxisSystemResponse] = {}

        # Rate limits — fetched in _initialize_async_resources(), one probe per
        # unique deployment; each phase reads the limits of its own model
        self._limits_by_model: Dict[str, RateLimits] = {}
        self._has_headers_by_model: Dict[str, bool] = {}

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
        extraction_metadata=None,
    ) -> TaxonomyResult:
        """Run taxonomy stages: facets, attributes, assignments.

        `extraction_metadata` (models.ExtractionMetadata, optional) feeds the
        drain-domain skip (taxonomy_health.drain_domains) for axis discovery.
        """
        print(f"\n{'='*70}")
        print(f"TAXONOMY DISCOVERY (5 phases)")
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
            print(f"  Pipeline: facet discovery → facet assignment → "
                  f"attribute discovery → attribute assignment → cross-facet consolidation")

        async def _run():
            await self._initialize_async_resources(
                active_partitions, partition_contexts, prompt_context, verbose
            )
            return await self._process_taxonomy_async(
                active_partitions, partition_contexts, prompt_context, verbose,
                extraction_metadata=extraction_metadata,
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
        """Fetch rate limits from API — one probe per unique deployment.

        Quota is per deployment, so each phase must read the limits of the
        deployment its own model resolves to — not whatever P2 happens to run
        on. Probes for distinct deployments run in parallel, so wall-clock cost
        stays that of a single probe.
        """
        phase_models = [
            self._model_p1, self._model_p2, self._model_p4, self._model_p5,
            self._model_p6, self._model_p7, self._model_p8,
        ]

        def route_key(model):
            if API_PROVIDER == "azure":
                endpoint, _, deployment = get_azure_route(model)
                return (endpoint, deployment)
            return (model,)

        representatives: Dict[tuple, str] = {}
        for m in phase_models:
            representatives.setdefault(route_key(m), m)

        if verbose:
            print(f"  Fetching rate limits from API "
                  f"({len(representatives)} deployment(s))...")
        probes = await asyncio.gather(
            *(llm_fetch_rate_limits(m) for m in representatives.values())
        )
        by_route = dict(zip(representatives.keys(), probes))

        for m in set(phase_models):
            limits, has_headers = by_route[route_key(m)]
            if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
                if verbose:
                    print(f"  WARNING: {m}: using fallback rate limits "
                          f"(TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
                limits = RateLimits(
                    tokens_per_minute=FALLBACK_TPM,
                    requests_per_minute=FALLBACK_RPM,
                )
            self._limits_by_model[m] = limits
            self._has_headers_by_model[m] = has_headers

        if verbose:
            headroom = DEFAULT_PROCESSING_CONFIG.rate_limit_headroom
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Models: P1={self._model_p1}, P2/P3={self._model_p2}, "
                  f"P4={self._model_p4}, P5={self._model_p5}, "
                  f"P6={self._model_p6}, P7={self._model_p7}, "
                  f"P8={self._model_p8}, P9={self._model_p9}")
            for key, m in representatives.items():
                limits = self._limits_by_model[m]
                label = key[-1] if API_PROVIDER == "azure" else m
                print(f"  {label}: TPM={limits.tokens_per_minute:,} "
                      f"({limits.tokens_per_minute * headroom:,.0f} with headroom) | "
                      f"RPM={limits.requests_per_minute:,} "
                      f"({limits.requests_per_minute * headroom:,.0f} with headroom)")

    async def _process_taxonomy_async(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
        extraction_metadata=None,
    ) -> TaxonomyResult:
        """Taxonomy stages P1-P9: facets, attributes, assignments."""
        start_time = time.time()
        self._facet_confidence.clear()
        self._attribute_confidence.clear()
        self._facet_valence.clear()
        self._attribute_valence.clear()
        self.axis_systems.clear()

        # P9 action log — declared here (not at its historical P3 site) because
        # P1, which runs first, needs it too.
        consolidation_log: List[Dict] = []

        # =================================================================
        # PHASE 1 (P1): Per-domain Axis Discovery (optional, SmoothRequester,
        # light mode — mirrors the P2/P3 dispatch). Establishes 1-4 axes per
        # domain from a deterministic sample of its observations, before P1
        # facet discovery runs. Behind axis_first_enabled; off is
        # byte-identical to the pre-existing chain — this whole block is
        # skipped, axis_systems stays empty, no domain is touched.
        # Skips the standing drain domains and domains too small to be worth
        # an axis call (< 2 chunks AND < 20 labels).
        # =================================================================
        if self._axis_first_enabled:
            _snap_p1a = token_tracker.snapshot() if self.cost_tracker else None
            t_p1a = time.time()

            drain_p1a = drain_domains(extraction_metadata)

            p1a_tasks = []
            for name, mapping in sorted(label_mappings.items()):
                if name in drain_p1a:
                    continue
                n_labels = len(mapping.labels)
                # Recomputed here (domain_chunk_info builds it too, just below,
                # for P1) — cheap list slicing, and P1 must gate before facet discovery exists.
                n_chunks = len(self._create_batches(mapping.labels))
                if n_chunks < 2 and n_labels < 20:
                    continue
                p1a_tasks.append({
                    'domain_name': name,
                    'part_context': partition_contexts[name],
                    'sample_observations': sample_axis_observations(mapping.labels),
                })

            if p1a_tasks:
                if verbose:
                    print(f"\n  Phase 1a: Axis Discovery")

                p1a_requester = SmoothRequester(
                    model=self._model_p1,
                    phase_key="step4_p1_axis_discovery",
                    num_tasks=len(p1a_tasks),
                    verbose=verbose,
                    known_limits=self._limits_by_model[self._model_p1],
                    has_server_headers=self._has_headers_by_model[self._model_p1],
                    show_setup=False,
                    quiet=True,
                )
                p1a_results = await p1a_requester.process_all(
                    p1a_tasks,
                    self._p1_prepare_fn(prompt_context),
                    self._p1_parse_fn(),
                    self._p1_fallback_fn(),
                )

                for task, response in zip(p1a_tasks, p1a_results):
                    name = task['domain_name']
                    validated = validate_and_repair_axis_system(response)
                    if validated is None:
                        consolidation_log.append({
                            "action": "axis_system_failed", "domain": name,
                            "reason": "no response" if response is None else "invalid axis system",
                        })
                        continue
                    self.axis_systems[name] = validated
                    consolidation_log.append({
                        "action": "axis_system_discovered", "domain": name,
                        "n_axes": len(validated.axes),
                    })

                if verbose:
                    s = p1a_requester.stats
                    print(f"    P1 axis discovery: {len(p1a_tasks)} tasks, "
                          f"{time.time() - t_p1a:.1f}s ({s.get('tasks_successful', 0)} ok, "
                          f"{s.get('timeouts', 0)} timeouts, {s.get('recovered', 0)} retries)")
                    print(f"    {len(self.axis_systems)}/{len(p1a_tasks)} domains got an axis system")
                    for name in sorted(self.axis_systems):
                        system = self.axis_systems[name]
                        print(f"      {name} ({len(system.axes)} axes):")
                        for axis in system.axes:
                            print(f"        - {axis.axis_name}: {axis.value_range}")
                    failed = [t['domain_name'] for t in p1a_tasks
                              if t['domain_name'] not in self.axis_systems]
                    if failed:
                        print(f"      no axis system (P3 path): {', '.join(sorted(failed))}")

            if self.cost_tracker and _snap_p1a is not None:
                self.cost_tracker.record_phase(
                    "step_4_taxonomy_classifier", "p1_axis_discovery",
                    _snap_p1a, token_tracker.snapshot(), self._model_p1)

        if self._debug_stop_after_phase == 1:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P1 — "
                      f"{len(self.axis_systems)} axis system(s), no facets built")
            return TaxonomyResult(
                partition_n_labels={},
                partition_n_batches={},
                partition_facets={},
                partition_assignments={},
                partition_attributes={},
                attribute_assignments={},
                consolidation_log=consolidation_log,
                axis_systems=self._dump_axis_systems(),
            )

        # =================================================================
        # PHASE 2/3 (P2/P3): Per-domain Facet Discovery (SmoothRequester)
        # =================================================================
        _snap_p1 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 1-2: Facet Discovery + Consolidation")

        t_phase1 = time.time()

        # Build flat task list: one task per (domain, chunk)
        p1_tasks = []
        domain_chunk_info: Dict[str, Dict] = {}  # domain → {labels, n_batches, excluded}

        for name, mapping in sorted(label_mappings.items()):
            batches = self._create_batches(mapping.labels)
            excluded = [
                (other_name, partition_contexts[other_name].partition_definition)
                for other_name in partition_contexts
                if other_name != name
            ]
            domain_chunk_info[name] = {
                'n_labels': len(mapping.labels),
                'n_batches': len(batches),
                'excluded': excluded,
            }

            for chunk_idx, observations in enumerate(batches):
                p1_tasks.append({
                    'domain_name': name,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(batches),
                    'observations': observations,
                    'part_context': partition_contexts[name],
                    'excluded_domains': excluded,
                })

        if verbose:
            total_obs = sum(info['n_labels'] for info in domain_chunk_info.values())
            max_chunks = max(info['n_batches'] for info in domain_chunk_info.values())
            chunk_desc = "1 chunk each" if max_chunks == 1 else f"up to {max_chunks} chunks"
            print(f"    Input: {len(domain_chunk_info)} domains, {total_obs} observations ({chunk_desc})")

        # P1 discovery via SmoothRequester
        p1_requester = SmoothRequester(
            model=self._model_p2,
            phase_key="step4_p2_facet_discovery",
            num_tasks=len(p1_tasks),
            verbose=verbose,
            known_limits=self._limits_by_model[self._model_p2],
            has_server_headers=self._has_headers_by_model[self._model_p2],
            show_setup=False,
            quiet=True,
        )
        p1_results = await p1_requester.process_all(
            p1_tasks,
            self._p2_prepare_fn(prompt_context),
            self._p2_parse_fn(),
            self._p2_fallback_fn(),
        )

        # Group chunk results by domain
        domain_chunk_facets: Dict[str, List[List[DiscoveredFacet]]] = {}
        for task, result in zip(p1_tasks, p1_results):
            name = task['domain_name']
            if name not in domain_chunk_facets:
                domain_chunk_facets[name] = []
            domain_chunk_facets[name].append(result or [])

        t_discovery = time.time() - t_phase1
        if verbose:
            s = p1_requester.stats
            print(f"    P1 discovery: {len(p1_tasks)} tasks, {t_discovery:.1f}s "
                  f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries)")

        if self.cost_tracker and _snap_p1 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p2_facet_discovery",
                _snap_p1, token_tracker.snapshot(), self._model_p2)

        if self._debug_stop_after_phase == 2:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P2/P3 — raw chunk facets follow")
                for name, chunks in domain_chunk_facets.items():
                    for ci, chunk in enumerate(chunks, 1):
                        print(f"\n  RAW P1  {name}  — chunk {ci}/{len(chunks)}: {len(chunk)} facet(s)")
                        for f in chunk:
                            print(f"    - {f.facet_name}: {f.facet_description}")
            return TaxonomyResult(
                partition_n_labels={n: i['n_labels'] for n, i in domain_chunk_info.items()},
                partition_n_batches={n: i['n_batches'] for n, i in domain_chunk_info.items()},
                partition_facets={n: [f for ch in chs for f in ch]
                                  for n, chs in domain_chunk_facets.items()},
                partition_assignments={},
                partition_attributes={},
                attribute_assignments={},
                consolidation_log=consolidation_log,
                axis_systems=self._dump_axis_systems(),
            )

        # Facet discovery flows straight to assignment: every domain's chunk
        # proposals are flattened as-is. Consolidation happens AFTER
        # assignment, per axis, on real contents (the in-axis phase below).
        partition_facets: Dict[str, List[DiscoveredFacet]] = {}
        partition_n_labels: Dict[str, int] = {}
        partition_n_batches: Dict[str, int] = {}
        for name in sorted(domain_chunk_facets.keys()):
            raw_facets = [f for chunk in domain_chunk_facets[name] for f in chunk]
            all_facets = dedup_exact_facets(raw_facets)
            if len(all_facets) < len(raw_facets):
                consolidation_log.append({
                    "action": "facet_exact_dedup", "domain": name,
                    "before": len(raw_facets), "after": len(all_facets),
                })
            partition_facets[name] = all_facets
            if all_facets:
                partition_n_labels[name] = domain_chunk_info[name]['n_labels']
                partition_n_batches[name] = domain_chunk_info[name]['n_batches']

        phase1_elapsed = time.time() - t_phase1
        if verbose:
            total_facets = sum(len(f) for f in partition_facets.values())
            print(f"    Raw facets to assignment ({phase1_elapsed:.1f}s -> {total_facets} facets):")
            for name in sorted(partition_facets.keys()):
                facets = partition_facets.get(name, [])
                facet_names = ", ".join(f.facet_name for f in facets) if facets else "(none)"
                print(f"      {name}: {len(facets)} facet(s): {facet_names}")



        # =================================================================
        # PHASE 4 (P4): Per-domain Facet Assignment (SmoothRequester)
        # =================================================================
        _snap_p3 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 4: Facet Assignment")

        t_phase3 = time.time()

        # Single-facet domains are auto-assigned without LLM call
        p3_tasks = []
        p3_batch_tasks = []
        partition_assignments: Dict[str, Dict[str, str]] = {}
        p3_auto_assigned: Dict[str, int] = {}  # domain → idea count (for reporting)
        p3_pending_singles: List[Dict] = []    # escalated reps → full-menu single pass

        embedder = (SharedEmbedder()
                    if self._assign_batch_enabled and self._assign_shortlist_enabled
                    else None)

        for domain_name in sorted(partition_facets.keys()):
            if not partition_facets[domain_name] or not label_mappings[domain_name].ideas:
                continue
            facets = partition_facets[domain_name]
            ideas = label_mappings[domain_name].ideas

            # Single-facet domain: auto-assign all ideas
            if len(facets) == 1:
                partition_assignments[domain_name] = {
                    idea.idea_id: facets[0].facet_name for idea in ideas
                }
                for idea in ideas:
                    self._facet_confidence[idea.idea_id] = 1.0
                p3_auto_assigned[domain_name] = len(ideas)
                continue

            if not self._assign_batch_enabled:
                # Multi-facet: one task per idea (pre-batch path, byte-identical)
                facet_id_to_name = {f"F{i}": f.facet_name for i, f in enumerate(facets, 1)}
                for idea in ideas:
                    idea_label = format_label(idea, self._label_source, self._label_prefix)
                    p3_tasks.append({
                        'domain_name': domain_name,
                        'idea_id': idea.idea_id,
                        'idea_label': idea_label,
                        'facets': facets,
                        'facet_id_to_name': facet_id_to_name,
                        'part_context': partition_contexts[domain_name],
                    })
                continue

            # Batch mode: unique-label reps, K per call, optional shortlist menu
            reps = group_label_reps(ideas, self._label_source, self._label_prefix,
                                    dedup=self._assign_label_dedup)
            label_vectors = card_vectors = None
            if embedder is not None and len(facets) > self._assign_shortlist_k:
                card_vectors = await embedder.embed_texts(
                    [facet_card_text(f.model_dump()) for f in facets])
                label_vectors = await embedder.embed_texts([rep.label for rep in reps])
            for index_group in make_batches(len(reps), self._assign_batch_k):
                menu = facets
                if card_vectors is not None:
                    keep = shortlist_indices(
                        label_vectors[index_group], card_vectors, self._assign_shortlist_k)
                    menu = [facets[i] for i in keep]
                p3_batch_tasks.append({
                    'domain_name': domain_name,
                    'reps': [reps[i] for i in index_group],
                    'menu_facets': menu,
                    'full_facets': facets,
                    'facet_id_to_name': {f"F{i}": f.facet_name for i, f in enumerate(menu, 1)},
                    'part_context': partition_contexts[domain_name],
                })

        if p3_batch_tasks:
            batch_requester = SmoothRequester(
                model=self._model_p4,
                phase_key="step4_p4_facet_assignment",
                num_tasks=len(p3_batch_tasks),
                verbose=verbose,
                known_limits=self._limits_by_model[self._model_p4],
                has_server_headers=self._has_headers_by_model[self._model_p4],
                show_setup=False,
                quiet=False,
            )
            batch_results = await batch_requester.process_all(
                p3_batch_tasks,
                self._p4_batch_prepare_fn(prompt_context),
                self._p4_batch_parse_fn(p3_pending_singles),
                self._p4_batch_fallback_fn(p3_pending_singles),
            )
            for task, result in zip(p3_batch_tasks, batch_results):
                domain_name = task['domain_name']
                if domain_name not in partition_assignments:
                    partition_assignments[domain_name] = {}
                if result:
                    partition_assignments[domain_name].update(result)

            if p3_pending_singles:
                single_tasks = []
                for pending in p3_pending_singles:
                    facets = pending['facets']
                    single_tasks.append({
                        'domain_name': pending['domain_name'],
                        'idea_id': pending['rep'].idea_ids[0],
                        'idea_label': pending['rep'].label,
                        'facets': facets,
                        'facet_id_to_name': {f"F{i}": f.facet_name
                                             for i, f in enumerate(facets, 1)},
                        'part_context': pending['part_context'],
                    })
                escalation_requester = SmoothRequester(
                    model=self._model_p4,
                    phase_key="step4_p4_facet_assignment",
                    num_tasks=len(single_tasks),
                    verbose=verbose,
                    known_limits=self._limits_by_model[self._model_p4],
                    has_server_headers=self._has_headers_by_model[self._model_p4],
                    show_setup=False,
                    quiet=True,
                )
                single_results = await escalation_requester.process_all(
                    single_tasks,
                    self._p4_prepare_fn(prompt_context),
                    self._p4_parse_fn(),
                    self._p4_fallback_fn(),
                )
                for pending, result in zip(p3_pending_singles, single_results):
                    domain_name = pending['domain_name']
                    if domain_name not in partition_assignments:
                        partition_assignments[domain_name] = {}
                    if result:
                        rep = pending['rep']
                        facet_name = next(iter(result.values()))
                        anchor = rep.idea_ids[0]
                        for idea_id in rep.idea_ids:
                            partition_assignments[domain_name][idea_id] = facet_name
                            self._facet_confidence[idea_id] = \
                                self._facet_confidence.get(anchor, 0.0)
                            self._facet_valence[idea_id] = \
                                self._facet_valence.get(anchor, "0")

                escalation_reasons: Dict[str, Counter] = defaultdict(Counter)
                for pending in p3_pending_singles:
                    escalation_reasons[pending['domain_name']][pending['reason']] += 1
                for domain_name, reasons in sorted(escalation_reasons.items()):
                    consolidation_log.append({
                        "action": "p4_batch_escalation", "domain": domain_name,
                        "reasons": dict(reasons),
                    })

            if verbose:
                s = batch_requester.stats
                n_reps = sum(len(t['reps']) for t in p3_batch_tasks)
                print(f"    Assignment (batch): {len(p3_batch_tasks)} calls voor "
                      f"{n_reps} unieke labels, {s.get('wall_time', 0):.1f}s "
                      f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts); "
                      f"{len(p3_pending_singles)} geëscaleerd naar losse calls")

        if p3_tasks:
            p3_requester = SmoothRequester(
                model=self._model_p4,
                phase_key="step4_p4_facet_assignment",
                num_tasks=len(p3_tasks),
                verbose=verbose,
                known_limits=self._limits_by_model[self._model_p4],
                has_server_headers=self._has_headers_by_model[self._model_p4],
                show_setup=False,
                quiet=False,
            )
            p3_results = await p3_requester.process_all(
                p3_tasks,
                self._p4_prepare_fn(prompt_context),
                self._p4_parse_fn(),
                self._p4_fallback_fn(),
            )

            if verbose:
                s = p3_requester.stats
                t_sr = s.get('wall_time', 0)
                auto_msg = f" + {len(p3_auto_assigned)} auto-assigned" if p3_auto_assigned else ""
                print(f"    Assignment: {len(p3_tasks)} tasks, {t_sr:.1f}s "
                      f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                      f"{s.get('recovered', 0)} retries){auto_msg}")

            # Reassemble: merge per-idea results into per-domain assignments
            for task, result in zip(p3_tasks, p3_results):
                domain_name = task['domain_name']
                if domain_name not in partition_assignments:
                    partition_assignments[domain_name] = {}
                if result:
                    partition_assignments[domain_name].update(result)
        elif verbose and p3_auto_assigned:
            print(f"    Assignment: all {len(p3_auto_assigned)} domains auto-assigned (1 facet each)")

        # __UNASSIGNED__ fallback for missing ideas (LLM-assigned domains only)
        for domain_name in partition_assignments:
            if domain_name in p3_auto_assigned:
                continue
            ideas = label_mappings[domain_name].ideas
            expected = {idea.idea_id for idea in ideas}
            assigned = set(partition_assignments[domain_name].keys())
            missing = expected - assigned
            if missing:
                print(f"    WARNING: {len(missing)}/{len(ideas)} ideas received no facet assignment")
                for idea_id in missing:
                    partition_assignments[domain_name][idea_id] = "__UNASSIGNED__"
                    self._facet_confidence[idea_id] = 0.0

        t_phase3 = time.time() - t_phase3
        if verbose:
            total_assigned = sum(len(a) for a in partition_assignments.values())
            print(f"    Results ({t_phase3:.1f}s → {total_assigned} ideas assigned):")
            for domain_name in sorted(partition_assignments):
                n_assigned = len(partition_assignments[domain_name])
                n_ideas = len(label_mappings[domain_name].ideas)
                auto_tag = " (auto)" if domain_name in p3_auto_assigned else ""
                print(f"      {domain_name}: {n_assigned}/{n_ideas}{auto_tag}")

        if self.cost_tracker and _snap_p3 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p4_facet_assignment",
                _snap_p3, token_tracker.snapshot(), self._model_p4)

        if self._debug_stop_after_phase == 4:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P4 — skipping P5-P9")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes={},
                attribute_assignments={},
                facet_confidence=self._facet_confidence,
                facet_valence=self._facet_valence,
                consolidation_log=consolidation_log,
                axis_systems=self._dump_axis_systems(),
            )

        # =================================================================
        # Facet Consolidation (in-axis, post-assignment): judge every axis
        # group's facets on their real idea counts and real contents, mirror
        # of the in-facet attribute round one level up. The axis is fixed —
        # a merge can never move a facet to another axis; when a group of
        # ideas belongs elsewhere, the IDEAS move and the structure stays.
        # =================================================================
        _snap_fc = token_tracker.snapshot() if self.cost_tracker else None
        if verbose:
            print(f"\n  Phase 5: Facet Consolidation (in-axis)")
        t_fc = time.time()

        facet_ideas_now = self._group_ideas_by_facet(
            label_mappings, partition_facets, partition_assignments
        )

        fc_tasks = []
        for domain_name in sorted(partition_facets.keys()):
            facets = partition_facets[domain_name]
            if len(facets) < 2:
                continue
            part_ctx = partition_contexts[domain_name]

            # Group facets by their axis tag; untagged facets form one group
            # per domain, scoped by the domain itself.
            groups: Dict[str, List[DiscoveredFacet]] = {}
            for f in facets:
                groups.setdefault(self._norm_text(f.axis) if f.axis else "", []).append(f)

            axis_sys = self.axis_systems.get(domain_name)
            axis_name_by_norm: Dict[str, str] = {}
            axis_desc_by_norm: Dict[str, str] = {}
            if axis_sys:
                for ax in axis_sys.axes:
                    axis_name_by_norm[self._norm_text(ax.axis_name)] = ax.axis_name
                    axis_desc_by_norm[self._norm_text(ax.axis_name)] = ax.axis_description

            for gkey, gfacets in sorted(groups.items()):
                if len(gfacets) < 2:
                    continue
                axis_name = axis_name_by_norm.get(gkey) or (gfacets[0].axis or domain_name)
                axis_description = axis_desc_by_norm.get(
                    gkey, part_ctx.partition_definition)
                neighbour_lines = []
                for okey, ofacets in sorted(groups.items()):
                    if okey == gkey:
                        continue
                    oname = axis_name_by_norm.get(okey) or (ofacets[0].axis or domain_name)
                    onames = ", ".join(f.facet_name for f in ofacets)
                    neighbour_lines.append(f"Axis: {oname} — facets: {onames}")
                fc_tasks.append({
                    'domain_name': domain_name,
                    'axis_name': axis_name,
                    'axis_description': axis_description,
                    'axis_tag_raw': gfacets[0].axis,
                    'part_context': part_ctx,
                    'facets': gfacets,
                    'facets_block': self._build_axis_facets_block(
                        gfacets, facet_ideas_now, domain_name),
                    'neighbour_axes_block': "\n".join(neighbour_lines),
                })

        if fc_tasks:
            fc_requester = SmoothRequester(
                model=self._model_p5,
                phase_key="step4_p5_facet_consolidation",
                num_tasks=len(fc_tasks),
                verbose=verbose,
                known_limits=self._limits_by_model[self._model_p5],
                has_server_headers=self._has_headers_by_model[self._model_p5],
                show_setup=False,
                quiet=True,
            )
            fc_results = await fc_requester.process_all(
                fc_tasks,
                self._p5_prepare_fn(prompt_context),
                self._p5_parse_fn(),
                self._p5_fallback_fn(),
            )
            partition_assignments, fc_log = self._apply_in_axis_results(
                tasks=fc_tasks,
                results=fc_results,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                label_mappings=label_mappings,
                verbose=verbose,
            )
            consolidation_log.extend(fc_log)

            if verbose:
                s = fc_requester.stats
                total_facets = sum(len(f) for f in partition_facets.values())
                print(f"    Facet consolidation: {len(fc_tasks)} tasks, "
                      f"{time.time() - t_fc:.1f}s ({s.get('tasks_successful', 0)} ok, "
                      f"{s.get('timeouts', 0)} timeouts, {s.get('recovered', 0)} retries) "
                      f"→ {total_facets} facets")
        elif verbose:
            print(f"    Facet consolidation: nothing to consolidate (no axis group ≥ 2 facets)")

        if self.cost_tracker and _snap_fc is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p5_facet_consolidation",
                _snap_fc, token_tracker.snapshot(), self._model_p5)

        # =================================================================
        # Attribute Discovery (per facet, chunked)
        # =================================================================
        _snap_p4 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 6: Attribute Discovery")

        t_phase4 = time.time()

        # Group ideas by (domain, facet) using P4 assignments
        domain_facet_ideas = self._group_ideas_by_facet(
            label_mappings, partition_facets, partition_assignments
        )

        # Build flat task list: one task per (domain, facet, chunk)
        p4_tasks = []
        facet_meta: Dict[str, Dict] = {}  # facet_key → {observations, excluded_facets, facet_obj}

        for (domain_name, facet_name), ideas in domain_facet_ideas.items():
            facet_obj = None
            for f in partition_facets.get(domain_name, []):
                if f.facet_name == facet_name:
                    facet_obj = f
                    break
            if not facet_obj or not ideas:
                continue

            observations = []
            for idea in ideas:
                label = format_label(idea, self._label_source, self._label_prefix)
                if label:
                    observations.append(label)
            if not observations:
                continue

            excluded_f = [
                (f.facet_name, f.facet_description)
                for f in partition_facets.get(domain_name, [])
                if f.facet_name != facet_name
            ]

            facet_key = f"{domain_name}::{facet_name}"
            batches = self._create_batches(
                observations,
                size_min=self._p4_batch_size_min,
                size_max=self._p4_batch_size_max,
                target=self._p4_target_batches,
                overlap=self._p4_chunk_overlap,
            )
            facet_meta[facet_key] = {
                'observations': observations,
                'excluded_facets': excluded_f,
                'facet_obj': facet_obj,
                'n_batches': len(batches),
                'chunk_observations': batches,
            }

            for chunk_idx, chunk_obs in enumerate(batches):
                p4_tasks.append({
                    'domain_name': domain_name,
                    'facet_name': facet_name,
                    'facet_description': facet_obj.facet_description,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(batches),
                    'observations': chunk_obs,
                    'part_context': partition_contexts[domain_name],
                    'facet_obj': facet_obj,
                    'excluded_facets': excluded_f,
                    'facet_key': facet_key,
                })

        if p4_tasks:
            p4_requester = SmoothRequester(
                model=self._model_p6,
                phase_key="step4_p6_attribute_discovery",
                num_tasks=len(p4_tasks),
                verbose=verbose,
                known_limits=self._limits_by_model[self._model_p6],
                has_server_headers=self._has_headers_by_model[self._model_p6],
                show_setup=False,
                quiet=True,
            )
            p4_results = await p4_requester.process_all(
                p4_tasks,
                self._p6_prepare_fn(prompt_context),
                self._p6_parse_fn(),
                self._p6_fallback_fn(),
            )
        else:
            p4_results = []

        # Group chunk results by facet_key
        facet_chunk_attrs: Dict[str, List[List[DiscoveredAttribute]]] = {}
        for task, result in zip(p4_tasks, p4_results):
            fk = task['facet_key']
            if fk not in facet_chunk_attrs:
                facet_chunk_attrs[fk] = []
            facet_chunk_attrs[fk].append(result or [])

        t_p4_discovery = time.time() - t_phase4
        if verbose and p4_tasks:
            s = p4_requester.stats
            print(f"    P6 discovery: {len(p4_tasks)} tasks, {t_p4_discovery:.1f}s "
                  f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries)")

        if self.cost_tracker and _snap_p4 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p6_attribute_discovery",
                _snap_p4, token_tracker.snapshot(), self._model_p6)

        if self._debug_stop_after_phase == 6:
            fk_home = {}
            for task in p4_tasks:
                fk_home.setdefault(task['facet_key'], (task['domain_name'], task['facet_name']))
            raw_attrs: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
            for fk, chunks in facet_chunk_attrs.items():
                dom, fac = fk_home[fk]
                raw_attrs.setdefault(dom, {})[fac] = [a for ch in chunks for a in ch]
            if verbose:
                print(f"\n  [DEBUG] Early stop after P6 — raw chunk attributes follow")
                for fk, chunks in facet_chunk_attrs.items():
                    dom, fac = fk_home[fk]
                    for ci, chunk in enumerate(chunks, 1):
                        print(f"\n  RAW P5  {dom} > {fac}  — chunk {ci}/{len(chunks)}: {len(chunk)} attribute(s)")
                        for a in chunk:
                            print(f"    - {a.attribute_name}: {a.attribute_description}")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes=raw_attrs,
                attribute_assignments={},
                facet_confidence=self._facet_confidence,
                facet_valence=self._facet_valence,
                consolidation_log=consolidation_log,
                axis_systems=self._dump_axis_systems(),
            )

        # Attribute discovery flows straight to assignment: each facet's chunk
        # proposals are flattened as-is. Consolidation happens AFTER
        # assignment, in-facet, on real contents (the in-facet phase below).
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        for facet_key, chunk_attributes in sorted(facet_chunk_attrs.items()):
            domain_name, facet_name = facet_key.split("::", 1)
            raw_attrs = [a for chunk in chunk_attributes for a in chunk]
            flat = dedup_exact_attributes(raw_attrs)
            if len(flat) < len(raw_attrs):
                consolidation_log.append({
                    "action": "attribute_exact_dedup", "domain": domain_name,
                    "facet": facet_name, "before": len(raw_attrs), "after": len(flat),
                })
            domain_facet_attributes.setdefault(domain_name, {})[facet_name] = flat
            partition_attributes.setdefault(domain_name, {})[facet_name] = flat

        if verbose:
            total_attrs = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            print(f"    Raw attributes to assignment: {total_attrs} across {len(facet_chunk_attrs)} facets")



        # =================================================================
        # PHASE 7 (P7): Per-facet Attribute Assignment (SmoothRequester)
        # =================================================================
        _snap_p6 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 7: Attribute Assignment")

        t_phase6 = time.time()

        # Build flat task list: one task per (domain, facet, batch)
        # Single-attribute facets are auto-assigned without LLM call
        p6_tasks = []
        attribute_assignments: Dict[str, str] = {}
        p6_auto_assigned: Dict[str, int] = {}  # facet_key → idea count (for reporting)
        facet_idea_sets: Dict[str, List] = {}

        for domain_name, facet_attrs in domain_facet_attributes.items():
            for facet_name, attributes in facet_attrs.items():
                if not attributes:
                    continue
                facet_ideas = domain_facet_ideas.get((domain_name, facet_name), [])
                if not facet_ideas:
                    continue

                facet_key = f"{domain_name}::{facet_name}"
                facet_idea_sets[facet_key] = facet_ideas

                # Single-attribute facet: auto-assign all ideas
                if len(attributes) == 1:
                    for idea in facet_ideas:
                        attribute_assignments[idea.idea_id] = attributes[0].attribute_name
                        self._attribute_confidence[idea.idea_id] = 1.0
                    p6_auto_assigned[facet_key] = len(facet_ideas)
                    continue

                # Multi-attribute: one task per idea
                facet_obj = None
                for f in partition_facets.get(domain_name, []):
                    if f.facet_name == facet_name:
                        facet_obj = f
                        break
                if not facet_obj:
                    continue

                attr_id_to_name = {f"A{i}": a.attribute_name for i, a in enumerate(attributes, 1)}
                for idea in facet_ideas:
                    idea_label = format_label(idea, self._label_source, self._label_prefix)
                    p6_tasks.append({
                        'domain_name': domain_name,
                        'facet_name': facet_name,
                        'facet_description': facet_obj.facet_description,
                        'idea_id': idea.idea_id,
                        'idea_label': idea_label,
                        'attributes': attributes,
                        'attr_id_to_name': attr_id_to_name,
                        'facet_key': facet_key,
                    })

        if p6_tasks:
            p6_requester = SmoothRequester(
                model=self._model_p7,
                phase_key="step4_p7_attribute_assignment",
                num_tasks=len(p6_tasks),
                verbose=verbose,
                known_limits=self._limits_by_model[self._model_p7],
                has_server_headers=self._has_headers_by_model[self._model_p7],
                show_setup=False,
                quiet=False,
            )
            p6_results = await p6_requester.process_all(
                p6_tasks,
                self._p7_prepare_fn(prompt_context),
                self._p7_parse_fn(),
                self._p7_fallback_fn(),
            )

            if verbose:
                s = p6_requester.stats
                t_sr = s.get('wall_time', 0)
                auto_msg = f" + {len(p6_auto_assigned)} auto-assigned" if p6_auto_assigned else ""
                print(f"    Assignment: {len(p6_tasks)} tasks, {t_sr:.1f}s "
                      f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                      f"{s.get('recovered', 0)} retries){auto_msg}")

            # Reassemble: merge per-idea results
            for task, result in zip(p6_tasks, p6_results):
                if result:
                    attribute_assignments.update(result)

            # __UNASSIGNED__ fallback per facet (only for LLM-assigned facets)
            for facet_key, facet_ideas in facet_idea_sets.items():
                if facet_key in p6_auto_assigned:
                    continue
                expected = {idea.idea_id for idea in facet_ideas}
                assigned = {iid for iid in expected if iid in attribute_assignments}
                missing = expected - assigned
                if missing:
                    facet_name = facet_key.split("::", 1)[1]
                    print(f"    WARNING: {len(missing)}/{len(facet_ideas)} ideas received no attribute "
                          f"assignment in facet '{facet_name}'")
                    for idea_id in missing:
                        attribute_assignments[idea_id] = "__UNASSIGNED__"
                        self._attribute_confidence[idea_id] = 0.0
        elif verbose and p6_auto_assigned:
            print(f"    Assignment: all {len(p6_auto_assigned)} facets auto-assigned (1 attribute each)")

        t_phase6 = time.time() - t_phase6
        if verbose:
            print(f"    Results ({t_phase6:.1f}s → {len(attribute_assignments)} ideas assigned):")
            for facet_key, facet_ideas in sorted(facet_idea_sets.items()):
                n_assigned = sum(1 for idea in facet_ideas if idea.idea_id in attribute_assignments)
                domain_name, facet_name = facet_key.split("::", 1)
                auto_tag = " (auto)" if facet_key in p6_auto_assigned else ""
                print(f"      {domain_name}/{facet_name}: {n_assigned}/{len(facet_ideas)}{auto_tag}")

        if self.cost_tracker and _snap_p6 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p7_attribute_assignment",
                _snap_p6, token_tracker.snapshot(), self._model_p7)

        # Snapshot P8 state before the post-assignment consolidation round remaps.
        # This is what makes a bad merge diagnosable after the fact — keep it.
        raw_attribute_assignments = dict(attribute_assignments)
        raw_partition_attributes = {
            d: {f: list(attrs) for f, attrs in facets.items()}
            for d, facets in partition_attributes.items()
        }

        if self._debug_stop_after_phase == 7:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P7 — skipping P8-P9")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes=partition_attributes,
                attribute_assignments=attribute_assignments,
                raw_partition_attributes=raw_partition_attributes,
                raw_attribute_assignments=raw_attribute_assignments,
                facet_confidence=self._facet_confidence,
                attribute_confidence=self._attribute_confidence,
                facet_valence=self._facet_valence,
                attribute_valence=self._attribute_valence,
                consolidation_log=consolidation_log,
                axis_systems=self._dump_axis_systems(),
            )

        # =================================================================
        # PHASE 8 (P8): In-facet Attribute Consolidation (post-assignment)
        # Replaces the earlier cross-facet and cross-domain consolidation rounds
        # that used to follow attribute assignment.
        # Scope is ONE facet, so no merge can relocate an idea's facet; when a
        # group of ideas belongs elsewhere the IDEAS move and the structure stays.
        # =================================================================
        _snap_p7 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 8: Attribute Consolidation (in-facet)")

        t_phase9 = time.time()
        # `consolidation_log` already exists (initialised before P3, which may
        # have appended to it); P9's own actions are appended below, not a
        # replacement, so P3's rewrite/flag entries survive into the final result.

        p7_tasks = []
        for domain_name, facet_attrs in domain_facet_attributes.items():
            for facet_name, attributes in facet_attrs.items():
                facet_ideas = domain_facet_ideas.get((domain_name, facet_name), [])
                if not attributes or not facet_ideas:
                    continue

                facet_obj = next(
                    (f for f in partition_facets.get(domain_name, [])
                     if f.facet_name == facet_name), None)
                if facet_obj is None:
                    continue

                # Adjacent facets in the SAME domain, with their real sizes — context
                # for boundary-writing and move targets, never merge candidates.
                neighbours = []
                for other_facet, other_attrs in facet_attrs.items():
                    if other_facet == facet_name or not other_attrs:
                        continue
                    other_ideas = domain_facet_ideas.get((domain_name, other_facet), [])
                    counts = Counter(
                        attribute_assignments.get(i.idea_id) for i in other_ideas
                    )
                    neighbours.append((
                        other_facet,
                        [(a.attribute_name, counts.get(a.attribute_name, 0))
                         for a in other_attrs],
                    ))

                p7_tasks.append({
                    'domain_name': domain_name,
                    'facet_name': facet_name,
                    'facet_description': facet_obj.facet_description,
                    'attributes': attributes,
                    'facet_ideas': facet_ideas,
                    'part_context': partition_contexts[domain_name],
                    # Rendered here, where the assignments are in scope: the real
                    # contents of each bucket, not the examples discovery guessed at.
                    'attributes_block': self._build_facet_contents_block(
                        attributes, facet_ideas, attribute_assignments),
                    'neighbour_block': build_neighbour_block(neighbours),
                })

        p7_results = []
        if p7_tasks:
            p7_requester = SmoothRequester(
                model=self._model_p8,
                phase_key="step4_p8_attribute_consolidation",
                num_tasks=len(p7_tasks),
                verbose=verbose,
                known_limits=self._limits_by_model[self._model_p8],
                has_server_headers=self._has_headers_by_model[self._model_p8],
                show_setup=False,
                quiet=True,
            )
            p7_results = await p7_requester.process_all(
                p7_tasks,
                self._p8_prepare_fn(prompt_context),
                self._p8_parse_fn(),
                self._p8_fallback_fn(),
            )

            if verbose:
                s = p7_requester.stats
                print(f"    P8 consolidation: {len(p7_tasks)} tasks, "
                      f"{s.get('wall_time', 0):.1f}s ({s['tasks_successful']} ok, "
                      f"{s.get('timeouts', 0)} timeouts, {s.get('recovered', 0)} retries)")

            attribute_assignments, partition_assignments, p9_log = (
                self._apply_in_facet_results(
                    tasks=p7_tasks,
                    results=p7_results,
                    domain_facet_attributes=domain_facet_attributes,
                    partition_attributes=partition_attributes,
                    attribute_assignments=attribute_assignments,
                    partition_assignments=partition_assignments,
                    verbose=verbose,
                )
            )
            consolidation_log.extend(p9_log)

        t_phase9 = time.time() - t_phase9
        if verbose:
            n_after = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            acts = Counter(e["action"] for e in p9_log) if p7_tasks else Counter()
            print(f"    Results ({t_phase9:.1f}s → {n_after} attributes): "
                  + (", ".join(f"{k}={v}" for k, v in sorted(acts.items())) or "no changes"))

        if self.cost_tracker and _snap_p7 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p8_attribute_consolidation",
                _snap_p7, token_tracker.snapshot(), self._model_p8)

        taxonomy_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Taxonomy complete in {taxonomy_elapsed:.1f}s")

        return TaxonomyResult(
            partition_n_labels=partition_n_labels,
            partition_n_batches=partition_n_batches,
            partition_facets=partition_facets,
            partition_assignments=partition_assignments,
            partition_attributes=partition_attributes,
            attribute_assignments=attribute_assignments,
            raw_partition_attributes=raw_partition_attributes,
            raw_attribute_assignments=raw_attribute_assignments,
            facet_confidence=self._facet_confidence,
            attribute_confidence=self._attribute_confidence,
            facet_valence=self._facet_valence,
            attribute_valence=self._attribute_valence,
            consolidation_log=consolidation_log,
            axis_systems=self._dump_axis_systems(),
        )

    # =========================================================================
    # PHASE 4 (P4): PER-DOMAIN FACET ASSIGNMENT (SmoothRequester)
    # =========================================================================

    def _p4_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P4 facet assignment (single idea)."""
        def prepare_fn(task: Dict) -> Dict:
            axis_system = self.axis_systems.get(task['domain_name'])
            axis_descriptions = (
                {axis.axis_name: axis.axis_description for axis in axis_system.axes}
                if axis_system is not None else None
            )
            prompt = build_facet_assignment_prompt_single(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facets=task['facets'],
                idea_label=task['idea_label'],
                axis_descriptions=axis_descriptions,
            )

            # Prompt capture (first idea per domain)
            gate_key = f"qr_facet_assign_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="facet_assignment",
                    metadata={
                        "model": self._model_p4,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_facet_assignment,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "n_facets": len(task['facets']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': FacetAssignmentResult,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_facet_assignment,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p4, phase="classifier_p4"),
            }
        return prepare_fn

    def _p4_parse_fn(self):
        """Return parse_fn closure for P4 facet assignment (single idea)."""
        def parse_fn(task: Dict, response) -> Optional[Dict[str, str]]:
            facet_name = task['facet_id_to_name'].get(response.assigned_facet_id)
            if facet_name is None:
                return {}
            self._facet_confidence[task['idea_id']] = response.confidence
            self._facet_valence[task['idea_id']] = response.valence
            return {task['idea_id']: facet_name}
        return parse_fn

    @staticmethod
    def _p4_fallback_fn():
        """Return fallback_fn closure for P4 facet assignment."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            return {}
        return fallback_fn

    def _p4_batch_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P4 facet assignment (batch of reps)."""
        def prepare_fn(task: Dict) -> Dict:
            axis_system = self.axis_systems.get(task['domain_name'])
            axis_descriptions = (
                {axis.axis_name: axis.axis_description for axis in axis_system.axes}
                if axis_system is not None else None
            )
            ideas = [(rep.idea_ids[0], rep.label) for rep in task['reps']]
            prompt = build_facet_assignment_prompt_batch(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facets=task['menu_facets'],
                ideas=ideas,
                axis_descriptions=axis_descriptions,
            )

            # Prompt capture (first batch per domain)
            gate_key = f"qr_facet_assign_batch_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="facet_assignment_batch",
                    metadata={
                        "model": self._model_p4,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_facet_assignment * 2,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "n_facets": len(task['menu_facets']),
                        "n_ideas": len(ideas),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': build_batch_facet_assignment_model(
                    list(task['facet_id_to_name'].keys()),
                    [idea_id for idea_id, _ in ideas]),
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_facet_assignment * 2,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p4, phase="classifier_p4"),
            }
        return prepare_fn

    def _p4_batch_parse_fn(self, pending_singles: List[Dict]):
        """Return parse_fn: accept validated items (fanned out over the rep's
        instances), route the rest to the escalation list with a reason."""
        def parse_fn(task: Dict, response) -> Optional[Dict[str, str]]:
            rep_by_id = {rep.idea_ids[0]: rep for rep in task['reps']}
            ok, escalate = validate_batch_response(list(rep_by_id.keys()), response)
            out: Dict[str, str] = {}
            for rep_id, item in ok.items():
                facet_name = task['facet_id_to_name'][item.assigned_facet_id]
                for idea_id in rep_by_id[rep_id].idea_ids:
                    out[idea_id] = facet_name
                    self._facet_confidence[idea_id] = item.confidence
                    self._facet_valence[idea_id] = item.valence
            for rep_id, reason in escalate.items():
                pending_singles.append({
                    'domain_name': task['domain_name'],
                    'rep': rep_by_id[rep_id],
                    'reason': reason,
                    'facets': task['full_facets'],
                    'part_context': task['part_context'],
                })
            return out
        return parse_fn

    @staticmethod
    def _p4_batch_fallback_fn(pending_singles: List[Dict]):
        """Return fallback_fn: a definitively failed batch escalates whole."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            for rep in task['reps']:
                pending_singles.append({
                    'domain_name': task['domain_name'],
                    'rep': rep,
                    'reason': 'batch_failed',
                    'facets': task['full_facets'],
                    'part_context': task['part_context'],
                })
            return {}
        return fallback_fn

    # =========================================================================
    # PHASE 7 (P7): PER-FACET ATTRIBUTE ASSIGNMENT (SmoothRequester)
    # =========================================================================

    def _p7_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P8 attribute assignment (single idea)."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_attribute_assignment_prompt_single(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                facet_name=task['facet_name'],
                facet_description=task['facet_description'],
                attributes=task['attributes'],
                idea_label=task['idea_label'],
            )

            # Prompt capture (first idea per facet)
            gate_key = f"qr_attr_assign_{task['domain_name']}_{task['facet_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="attribute_assignment",
                    metadata={
                        "model": self._model_p7,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_facet_assignment,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "facet_name": task['facet_name'],
                        "n_attributes": len(task['attributes']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': AttributeAssignmentResult,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_facet_assignment,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p7, phase="classifier_p7"),
            }
        return prepare_fn

    def _p7_parse_fn(self):
        """Return parse_fn closure for P8 attribute assignment (single idea)."""
        def parse_fn(task: Dict, response) -> Optional[Dict[str, str]]:
            attr_name = task['attr_id_to_name'].get(response.assigned_attribute_id)
            if attr_name is None:
                return {}
            self._attribute_confidence[task['idea_id']] = response.confidence
            self._attribute_valence[task['idea_id']] = response.valence
            return {task['idea_id']: attr_name}
        return parse_fn

    @staticmethod
    def _p7_fallback_fn():
        """Return fallback_fn closure for P8 attribute assignment."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            return {}
        return fallback_fn

    # =========================================================================
    # PHASE 1 (P1): PER-DOMAIN AXIS DISCOVERY (SmoothRequester, light mode)
    # =========================================================================

    def _p1_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P1 axis discovery."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_axis_discovery_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                primary_dimension=prompt_context.dimension_name,
                noun_phrase=(
                    prompt_context.dimension_def.noun_phrase_descriptor
                    if prompt_context.dimension_def else prompt_context.dimension_name
                ),
                domain_label=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                domain_boundary_test=task['part_context'].boundary_test,
                sample_observations=task['sample_observations'],
            )

            gate_key = f"qr_axis_discovery_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="axis_discovery",
                    metadata={
                        "model": self._model_p1,
                        "temperature": 0.0,
                        "max_tokens": self._max_tokens_consolidation,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "n_sample_observations": len(task['sample_observations']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': AxisSystemResponse,
                'temperature': 0.0,
                'max_tokens': self._max_tokens_consolidation,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p1, phase="classifier_p1"),
            }
        return prepare_fn

    def _p1_parse_fn(self):
        """Return parse_fn closure for P1 axis discovery."""
        def parse_fn(task: Dict, response) -> Optional[AxisSystemResponse]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _p1_fallback_fn():
        """Return fallback_fn closure for P1. On failure the domain simply gets
        no axis system — it runs the old untagged path for the whole run."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    # =========================================================================
    # PHASE 2/3 (P2/P3): PER-DOMAIN FACET DISCOVERY (SmoothRequester)
    # =========================================================================

    def _p2_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P1 facet discovery. A domain with a
        validated axis system (axis_first_enabled) gets the tagged P1b prompt
        and response model; a domain without one (flag off / axis failed /
        drain) gets the untouched untagged P1 path — byte-identical to before."""
        def prepare_fn(task: Dict) -> Dict:
            domain_name = task['domain_name']
            axis_system = self.axis_systems.get(domain_name)

            if axis_system is not None:
                prompt = build_tagged_facet_discovery_prompt(
                    survey_question=prompt_context.survey_question,
                    language=prompt_context.language,
                    noun_phrase=(
                        prompt_context.dimension_def.noun_phrase_descriptor
                        if prompt_context.dimension_def else prompt_context.dimension_name
                    ),
                    domain_label=domain_name,
                    domain_definition=task['part_context'].partition_definition,
                    axis_system=axis_system,
                    chunk_observations=task['observations'],
                )
                response_model = build_tagged_facet_discovery_model(
                    [axis.axis_name for axis in axis_system.axes]
                )
                prompt_type = "tagged_facet_discovery"
            else:
                prompt = build_facet_discovery_prompt(
                    survey_question=prompt_context.survey_question,
                    language=prompt_context.language,
                    dataset_context_section=prompt_context.dataset_context_section,
                    dimension_def=prompt_context.dimension_def,
                    dimension_name=prompt_context.dimension_name,
                    dimension_description=prompt_context.dimension_description,
                    partition_name=domain_name,
                    partition_definition=task['part_context'].partition_definition,
                    boundary_test=task['part_context'].boundary_test,
                    exclusions=task['part_context'].exclusions,
                    observations=task['observations'],
                    excluded_domains=task['excluded_domains'],
                )
                response_model = FacetDiscoveryResult
                prompt_type = "facet_discovery"

            # Prompt capture (first chunk per domain)
            gate_key = f"qr_facets_{domain_name}"
            if (self._prompt_printer is not None
                    and task['chunk_idx'] == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type=prompt_type,
                    metadata={
                        "model": self._model_p2,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_facet_discovery,
                        "language": prompt_context.language,
                        "partition_name": domain_name,
                        "batch_number": task['chunk_idx'] + 1,
                        "total_batches": task['total_chunks'],
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': response_model,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_facet_discovery,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p2, phase="classifier_p2"),
            }
        return prepare_fn

    def _p2_parse_fn(self):
        """Return parse_fn closure for P1 facet discovery. Tagged responses
        (domain has a validated axis system) are grouped by axis in the
        response schema itself — `build_tagged_facet_discovery_model` binds
        `axis_name` to a Literal over the domain's known axes, so every
        proposal is inherently tagged to a valid axis; no rejection step is
        needed. Each proposal becomes a DiscoveredFacet carrying its axis for
        downstream provenance (segment-less: P1 axes carry no
        sub-structure). Untagged responses (domain without a system) pass
        through unchanged, as before."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredFacet]]:
            if response is None:
                return []

            domain_name = task['domain_name']
            axis_system = self.axis_systems.get(domain_name)
            if axis_system is None:
                return response.facets

            observations = task['observations']
            facets: List[DiscoveredFacet] = []
            for axis_facets in response.axes:
                for proposal in axis_facets.facets:
                    facets.append(DiscoveredFacet(
                        facet_name=proposal.facet_name,
                        facet_description=proposal.facet_definition,
                        inclusion_rule=proposal.inclusion_rule,
                        exclusion_rule=proposal.exclusion_rule,
                        example_observations=[
                            observations[i - 1] for i in proposal.example_observations
                            if 1 <= i <= len(observations)
                        ],
                        axis=axis_facets.axis_name,
                    ))
            return facets
        return parse_fn

    @staticmethod
    def _p2_fallback_fn():
        """Return fallback_fn closure for P1 facet discovery."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredFacet]:
            return []
        return fallback_fn

    @staticmethod
    def _round_robin_examples(proposals: List, *, limit: int = 5) -> List[str]:
        """Pool example observations from a group's consumed proposals
        (facet proposals in P2, attribute proposals in P6), one per proposal
        per pass (round-robin), preserving each proposal's own example
        order, stopping at `limit`. Unlike a flat concatenation, this can't
        exhaust the pool on one proposal — every proposal that has an
        example contributes before any proposal contributes a second one,
        echoing the old path's model-curated "representative across the
        merged [sources]" spread."""
        examples: List[str] = []
        round_idx = 0
        while len(examples) < limit:
            added = False
            for p in proposals:
                if round_idx < len(p.example_observations):
                    examples.append(p.example_observations[round_idx])
                    added = True
                    if len(examples) == limit:
                        break
            if not added:
                break
            round_idx += 1
        return examples

    # =========================================================================
    # PHASE 6 (P6): PER-FACET ATTRIBUTE DISCOVERY (SmoothRequester)
    # =========================================================================

    def _p6_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for attribute discovery."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_attribute_discovery_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facet_name=task['facet_name'],
                facet_description=task['facet_description'],
                observations=task['observations'],
                excluded_facets=task['excluded_facets'],
            )
            response_model = AttributeDiscoveryResult
            prompt_type = "attribute_discovery"

            # Prompt capture (first chunk per facet)
            gate_key = f"qr_attributes_{task['domain_name']}_{task['facet_name']}"
            if (self._prompt_printer is not None
                    and task['chunk_idx'] == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type=prompt_type,
                    metadata={
                        "model": self._model_p6,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens_attribute_discovery,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "facet_name": task['facet_name'],
                        "n_observations": len(task['observations']),
                        "batch_number": task['chunk_idx'] + 1,
                        "total_batches": task['total_chunks'],
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': response_model,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_attribute_discovery,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p6, phase="classifier_p6"),
            }
        return prepare_fn

    def _p6_parse_fn(self):
        """Return parse_fn closure for attribute discovery."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredAttribute]]:
            if response is None:
                return []
            return response.attributes
        return parse_fn

    @staticmethod
    def _p6_fallback_fn():
        """Return fallback_fn closure for P5 attribute discovery."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredAttribute]:
            return []
        return fallback_fn

    # =========================================================================
    # PHASE 8 (P8): IN-FACET ATTRIBUTE CONSOLIDATION (SmoothRequester)
    # =========================================================================

    @staticmethod
    def _norm_text(text: Optional[str]) -> str:
        """Normalise a response text for matching. Case- and padding-insensitive
        only — no stemming, no stopwords, nothing language-specific, so this stays
        use-case agnostic and every match is checkable by eye."""
        return (text or "").strip().lower()

    def _dump_axis_systems(self) -> Dict[str, dict]:
        """Model-dump the discovered axis systems, verbatim, for TaxonomyResult
        and the runner's axes log. Empty unless axis_first_enabled produced any."""
        return {name: system.model_dump() for name, system in self.axis_systems.items()}

    def _build_axis_facets_block(
        self,
        facets: List[DiscoveredFacet],
        facet_ideas: Dict[tuple, List],
        domain_name: str,
        top_n: Optional[int] = None,
    ) -> str:
        """Render each facet on this axis with its real size, its share of the
        axis, and the response texts it actually holds — the facet-level mirror
        of `_build_facet_contents_block`."""
        if top_n is None:
            top_n = self._p9_contents_top_n
        counts = {f.facet_name: facet_ideas.get((domain_name, f.facet_name), [])
                  for f in facets}
        total = sum(len(v) for v in counts.values())

        lines = []
        for f in facets:
            mine = counts[f.facet_name]
            pct = round(100 * len(mine) / total) if total else 0
            texts = Counter(
                (i.instance or "").strip() for i in mine if (i.instance or "").strip()
            )
            shown = " · ".join(f'"{t}" x{c}' for t, c in texts.most_common(top_n))
            more = (f" · ... {len(texts) - top_n} further distinct texts"
                    if len(texts) > top_n else "")
            lines.append(
                f'- "{f.facet_name}" — {len(mine)} ideas, {pct}% of this axis — '
                f'{f.facet_description}'
            )
            lines.append(f'    actually contains: {shown}{more}' if shown
                         else '    actually contains: (no ideas assigned)')
        return "\n".join(lines)

    def _p5_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for in-axis facet consolidation."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_in_axis_consolidation_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                noun_phrase=(
                    prompt_context.dimension_def.noun_phrase_descriptor
                    if prompt_context.dimension_def else prompt_context.dimension_name
                ),
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                axis_name=task['axis_name'],
                axis_description=task['axis_description'],
                facets_block=task['facets_block'],
                neighbour_axes_block=task['neighbour_axes_block'],
            )

            gate_key = f"qr_in_axis_consolidation_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="in_axis_consolidation",
                    metadata={
                        "model": self._model_p5,
                        "temperature": 0.0,
                        "max_tokens": self._max_tokens_consolidation,
                        "language": prompt_context.language,
                        "domain_name": task['domain_name'],
                        "axis_name": task['axis_name'],
                        "n_facets": len(task['facets']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': InAxisConsolidatedResponse,
                'temperature': 0.0,
                'max_tokens': self._max_tokens_consolidation,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p5, phase="classifier_p5"),
            }
        return prepare_fn

    def _p5_parse_fn(self):
        """Return parse_fn closure for in-axis facet consolidation."""
        def parse_fn(task: Dict, response) -> Optional[InAxisConsolidatedResponse]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _p5_fallback_fn():
        """Return fallback_fn closure. On failure the axis group is left
        exactly as discovery produced it — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_in_axis_results(
        self,
        *,
        tasks: List[Dict],
        results: List,
        partition_facets: Dict[str, List[DiscoveredFacet]],
        partition_assignments: Dict[str, Dict[str, str]],
        label_mappings: Dict[str, "PartitionLabelMapping"],
        verbose: bool,
    ) -> Tuple[Dict[str, Dict[str, str]], List[Dict]]:
        """Apply every in-axis result, then remap ideas — structure first,
        ideas second. The facet-level mirror of `_apply_in_facet_results`: splits
        route by exact normalised response text, merges are a wholesale
        rename, misfit moves are per-text within the domain, and a facet
        dropped without being claimed is restored so no idea points at a
        name absent from the structure."""
        log: List[Dict] = []

        pre_facets: Dict[str, List[DiscoveredFacet]] = {
            dom: list(facets) for dom, facets in partition_facets.items()
        }

        # ---- 1. structure ------------------------------------------------
        remap: Dict[Tuple[str, str], str] = {}            # (dom, src) -> new name
        splits: Dict[Tuple[str, str, str], str] = {}      # (dom, src, text) -> child
        moves: Dict[Tuple[str, str, str], Optional[str]] = {}
        renamed_to: Dict[Tuple[str, str], str] = {}       # (dom, norm old) -> new
        split_children: Dict[Tuple[str, str], List[str]] = {}
        consumed: Dict[str, Set[str]] = {}                # dom -> source facet names replaced
        new_group_facets: Dict[str, List[DiscoveredFacet]] = {}  # dom -> new facets, task order

        for task, result in zip(tasks, results):
            dom = task['domain_name']
            group = task['facets']
            before = [f.facet_name for f in group]

            if result is None or not result.facets:
                log.append({"action": "facet_consolidation_failed", "domain": dom,
                            "axis": task['axis_name'],
                            "note": "no result — axis group left as discovered",
                            "facets_before": before})
                continue

            by_norm = {self._norm_text(b): b for b in before}

            def _resolve(src: str) -> Optional[str]:
                return by_norm.get(self._norm_text(src))

            unmatched = sorted({
                s for item in result.facets for s in (item.source_facets or [])
                if _resolve(s) is None
            })
            if unmatched:
                log.append({"action": "unknown_source_facet", "domain": dom,
                            "axis": task['axis_name'], "sources": unmatched})

            claims: Dict[str, int] = {}
            for item in result.facets:
                for src in (item.source_facets or []):
                    real = _resolve(src)
                    if real:
                        claims[real] = claims.get(real, 0) + 1
            contested = {
                src for src, n in claims.items() if n > 1
                and not any(_resolve(s) == src for it in result.facets
                            if it.instance_texts for s in (it.source_facets or []))
            }
            if contested:
                log.append({"action": "unroutable_facet_claim", "domain": dom,
                            "axis": task['axis_name'], "sources": sorted(contested),
                            "note": "claimed by several returned facets without "
                                    "instance_texts — ideas left on the source"})

            group_new: List[DiscoveredFacet] = []
            group_consumed: Set[str] = set()
            for item in result.facets:
                group_new.append(DiscoveredFacet(
                    facet_name=item.facet_name,
                    facet_description=item.facet_description,
                    inclusion_rule=item.inclusion_rule,
                    exclusion_rule=item.exclusion_rule,
                    example_observations=item.example_observations,
                    axis=task['axis_tag_raw'],   # fixed by scope, not by the model
                ))

                sources = [r for r in (_resolve(s) for s in (item.source_facets or []))
                           if r is not None]
                group_consumed.update(sources)
                if item.action == "split" and item.instance_texts:
                    for src in (sources or before):
                        for txt in item.instance_texts:
                            splits[(dom, src, self._norm_text(txt))] = item.facet_name
                        split_children.setdefault(
                            (dom, self._norm_text(src)), []).append(item.facet_name)
                    log.append({"action": "facet_split", "domain": dom,
                                "axis": task['axis_name'], "into": item.facet_name,
                                "sources": sources,
                                "n_texts": len(item.instance_texts),
                                "texts": item.instance_texts})
                else:
                    for src in sources:
                        if src != item.facet_name and src not in contested:
                            remap[(dom, src)] = item.facet_name
                            renamed_to[(dom, self._norm_text(src))] = item.facet_name
                    if item.action in ("merge", "widen") or (
                            sources and sources != [item.facet_name]):
                        log.append({"action": f"facet_{item.action}", "domain": dom,
                                    "axis": task['axis_name'],
                                    "result": item.facet_name, "sources": sources})

            # Sources never claimed by any returned facet: keep them — dropping
            # a facet silently would orphan every idea assigned to it.
            unclaimed = [f for f in group if f.facet_name not in group_consumed
                         and self._norm_text(f.facet_name) not in {
                             self._norm_text(n.facet_name) for n in group_new}]
            for f in unclaimed:
                group_new.append(f)
                log.append({"action": "facet_kept_unclaimed", "domain": dom,
                            "axis": task['axis_name'], "facet": f.facet_name})

            consumed.setdefault(dom, set()).update(fn.facet_name for fn in group)
            new_group_facets.setdefault(dom, []).extend(group_new)

            for m in (result.misfits or []):
                real_from = _resolve(m.from_facet) or m.from_facet
                for txt in (m.instance_texts or []):
                    moves[(dom, real_from, self._norm_text(txt))] = (
                        m.target_facet if m.verdict == "move" else None)
                log.append({"action": f"facet_misfit_{m.verdict}", "domain": dom,
                            "axis": task['axis_name'], "from_facet": real_from,
                            "target": m.target_facet,
                            "n_texts": len(m.instance_texts or []),
                            "texts": m.instance_texts, "reason": m.reason})

        # Rebuild each touched domain's facet list: untouched facets keep their
        # position; consolidated groups are replaced by their new facets.
        for dom, gone in consumed.items():
            kept = [f for f in partition_facets.get(dom, [])
                    if f.facet_name not in gone]
            partition_facets[dom] = kept + new_group_facets.get(dom, [])

        # A source split into exactly one child was renamed, not divided.
        split_ambiguous: Dict[Tuple[str, str], List[str]] = {}
        for key, children in split_children.items():
            uniq = sorted(set(children))
            if len(uniq) == 1:
                renamed_to.setdefault(key, uniq[0])
            else:
                split_ambiguous[key] = uniq

        home: Dict[str, Set[str]] = {
            dom: {f.facet_name for f in facets}
            for dom, facets in partition_facets.items()
        }

        # ---- 2-4. ideas --------------------------------------------------
        text_of: Dict[str, str] = {}
        for dom in consumed:
            mapping = label_mappings.get(dom)
            if mapping:
                for idea in mapping.ideas:
                    text_of[idea.idea_id] = self._norm_text(getattr(idea, "instance", ""))

        n_split = n_remap = n_moved = n_out = n_unresolved = 0
        unresolved_targets: Counter = Counter()
        for dom in consumed:
            assigns = partition_assignments.get(dom, {})
            for iid, cur in list(assigns.items()):
                txt = text_of.get(iid, "")

                mkey = (dom, cur, txt)
                if mkey in moves:
                    target = moves[mkey]
                    if target is None:
                        n_out += 1
                        continue
                    if target not in home.get(dom, set()):
                        target = renamed_to.get((dom, self._norm_text(target)), target)
                    if target in home.get(dom, set()):
                        assigns[iid] = target
                        n_moved += 1
                    else:
                        n_unresolved += 1
                        unresolved_targets[target] += 1
                    continue

                skey = (dom, cur, txt)
                if skey in splits:
                    assigns[iid] = splits[skey]
                    n_split += 1
                    continue

                rkey = (dom, cur)
                if rkey in remap:
                    assigns[iid] = remap[rkey]
                    n_remap += 1

        # ---- 5. self-check ----------------------------------------------
        orphans: Counter = Counter()
        for dom in consumed:
            names = home.get(dom, set())
            for iid, fac in partition_assignments.get(dom, {}).items():
                if fac and fac != "__UNASSIGNED__" and fac not in names:
                    orphans[(dom, fac)] += 1

        restored = 0
        for (dom, fac), count in orphans.items():
            src = next((f for f in pre_facets.get(dom, [])
                        if f.facet_name == fac), None)
            if src is not None:
                if all(f.facet_name != fac for f in partition_facets[dom]):
                    partition_facets[dom].append(src)
                home.setdefault(dom, set()).add(fac)
                restored += 1
        if orphans:
            log.append({"action": "orphaned_facet_assignment",
                        "restored_nodes": restored,
                        "ideas_affected": sum(orphans.values()),
                        "facets": sorted({f for (_, f) in orphans})})
            if verbose:
                print(f"    SELF-CHECK: {sum(orphans.values())} ideas pointed at "
                      f"{len(orphans)} facet(s) missing from the structure — "
                      f"{restored} node(s) restored")

        log.append({"action": "_facet_totals", "ideas_split": n_split,
                    "ideas_remapped": n_remap, "ideas_moved": n_moved,
                    "flagged_contentless_left_in_place": n_out,
                    "moves_with_unresolvable_target": n_unresolved,
                    "unresolved_target_names": dict(unresolved_targets.most_common(20))})

        if verbose:
            print(f"    Ideas: {n_remap} remapped, {n_split} split, {n_moved} moved "
                  f"across facets, {n_out} flagged contentless (left in place)")

        return partition_assignments, log

    def _build_facet_contents_block(
        self,
        attributes: List[DiscoveredAttribute],
        facet_ideas: List,
        attribute_assignments: Dict[str, str],
        top_n: Optional[int] = None,
    ) -> str:
        """Render each attribute with its real size, its share of the facet, and the
        response texts it actually holds.

        This is the input change the whole phase turns on: P9 was shown 2 examples
        picked during discovery, before a single idea was assigned, so it reasoned
        about what the label promised. Here it sees what the bucket contains.
        """
        if top_n is None:
            top_n = self._p9_contents_top_n
        assigned = [i for i in facet_ideas if attribute_assignments.get(i.idea_id)]
        total = len(assigned)

        lines = []
        for attr in attributes:
            name = attr.attribute_name
            mine = [i for i in assigned if attribute_assignments.get(i.idea_id) == name]
            pct = round(100 * len(mine) / total) if total else 0
            texts = Counter(
                (i.instance or "").strip() for i in mine if (i.instance or "").strip()
            )
            shown = " · ".join(f'"{t}" x{c}' for t, c in texts.most_common(top_n))
            more = (f" · ... {len(texts) - top_n} further distinct texts"
                    if len(texts) > top_n else "")
            lines.append(
                f'- "{name}" — {len(mine)} ideas, {pct}% of this facet — '
                f'{attr.attribute_description}'
            )
            lines.append(f'    actually contains: {shown}{more}' if shown
                         else '    actually contains: (no ideas assigned)')
        return "\n".join(lines)

    def _p8_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P9 in-facet attribute consolidation."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_in_facet_consolidation_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facet_name=task['facet_name'],
                facet_description=task['facet_description'],
                attributes_block=task['attributes_block'],
                neighbour_block=task['neighbour_block'],
            )

            gate_key = f"qr_in_facet_consolidation_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="in_facet_consolidation",
                    metadata={
                        "model": self._model_p8,
                        "temperature": 0.0,
                        "max_tokens": self._max_tokens_consolidation,
                        "language": prompt_context.language,
                        "domain_name": task['domain_name'],
                        "facet_name": task['facet_name'],
                        "n_attributes": len(task['attributes']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': InFacetConsolidatedResponse,
                'temperature': 0.0,
                'max_tokens': self._max_tokens_consolidation,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p8, phase="classifier_p8"),
            }
        return prepare_fn

    def _p8_parse_fn(self):
        """Return parse_fn closure for P9 in-facet attribute consolidation."""
        def parse_fn(task: Dict, response) -> Optional[InFacetConsolidatedResponse]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _p8_fallback_fn():
        """Return fallback_fn closure for P9. On failure the facet is left exactly
        as P8 produced it — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_in_facet_results(
        self,
        *,
        tasks: List[Dict],
        results: List,
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        attribute_assignments: Dict[str, str],
        partition_assignments: Dict[str, Dict[str, str]],
        verbose: bool,
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], List[Dict]]:
        """Apply every P9 result, then remap ideas — structure first, ideas second.

        Order matters and is deliberate:
          1. rebuild the structure for every facet (so move targets can be resolved
             against the FINAL names, not the names a concurrent call has renamed)
          2. splits   — route by exact response text to a named child
          3. merges   — wholesale rename of a source attribute
          4. moves    — per-text, may cross a facet boundary; the structure does not
                        follow, so no other idea is dragged along
        Every action lands in the log with the texts it touched.
        """
        log: List[Dict] = []

        # keep the pre-P9 attribute objects reachable, so the self-check below can
        # put back a node P9 dropped without ever naming it
        pre_p7_attrs: Dict[Tuple[str, str], List[DiscoveredAttribute]] = {
            (t['domain_name'], t['facet_name']): list(t['attributes']) for t in tasks
        }

        # ---- 1. structure ----------------------------------------------------
        # scope key is (domain, facet): attribute names are not unique across facets
        remap: Dict[Tuple[str, str, str], str] = {}
        splits: Dict[Tuple[str, str, str, str], str] = {}
        moves: Dict[Tuple[str, str, str, str], Optional[str]] = {}
        renamed_to: Dict[str, str] = {}   # normalised old name -> new name, global
        split_children: Dict[str, List[str]] = {}   # normalised old name -> child names

        for task, result in zip(tasks, results):
            dom, fac = task['domain_name'], task['facet_name']
            before = [a.attribute_name for a in task['attributes']]

            if result is None or not result.attributes:
                log.append({"action": "failed", "domain": dom, "facet": fac,
                            "note": "no result — facet left as P8 produced it",
                            "attributes_before": before})
                continue

            # Match the model's source names against the real ones case- and
            # padding-insensitively. A strict equality check silently dropped
            # sources that differed only in capitalisation, leaving their ideas on
            # a pre-P9 name that no longer exists in the structure.
            by_norm = {self._norm_text(b): b for b in before}

            def _resolve(src: str) -> Optional[str]:
                return by_norm.get(self._norm_text(src))

            unmatched = sorted({
                s for item in result.attributes for s in (item.source_attributes or [])
                if _resolve(s) is None
            })
            if unmatched:
                log.append({"action": "unknown_source_name", "domain": dom, "facet": fac,
                            "sources": unmatched,
                            "note": "named as a source but not among this facet's attributes"})

            # A source claimed by more than one returned attribute is only routable
            # when the claimants carry instance_texts (a split). Without that the
            # remap would silently let the last writer win and move the whole bucket.
            claims: Dict[str, int] = {}
            for item in result.attributes:
                for src in (item.source_attributes or []):
                    real = _resolve(src)
                    if real:
                        claims[real] = claims.get(real, 0) + 1
            contested = {
                src for src, n in claims.items() if n > 1
                and not any(_resolve(s) == src for it in result.attributes
                            if it.instance_texts for s in (it.source_attributes or []))
            }
            if contested:
                log.append({"action": "unroutable_claim", "domain": dom, "facet": fac,
                            "sources": sorted(contested),
                            "note": ("claimed by several returned attributes with no "
                                     "instance_texts — ideas left on the source")})

            new_attrs: List[DiscoveredAttribute] = []
            for item in result.attributes:
                new_attrs.append(DiscoveredAttribute(
                    attribute_name=item.attribute_name,
                    attribute_description=item.attribute_description,
                    parent_facet=fac,          # fixed by scope, not by the model
                    example_observations=item.example_observations,
                ))

                sources = [r for r in (_resolve(s) for s in (item.source_attributes or []))
                           if r is not None]
                if item.action == "split" and item.instance_texts:
                    for src in (sources or before):
                        for txt in item.instance_texts:
                            splits[(dom, fac, src, self._norm_text(txt))] = item.attribute_name
                        split_children.setdefault(
                            self._norm_text(src), []).append(item.attribute_name)
                    log.append({"action": "split", "domain": dom, "facet": fac,
                                "into": item.attribute_name, "sources": sources,
                                "n_texts": len(item.instance_texts),
                                "texts": item.instance_texts})
                else:
                    for src in sources:
                        if src != item.attribute_name and src not in contested:
                            remap[(dom, fac, src)] = item.attribute_name
                            # Facets are consolidated concurrently, so a move may name
                            # a target by the name it had in the neighbour block while
                            # its own facet is renaming it. Keep the trail.
                            renamed_to[self._norm_text(src)] = item.attribute_name
                    if item.action in ("merge", "widen") or (
                            sources and sources != [item.attribute_name]):
                        log.append({"action": item.action, "domain": dom, "facet": fac,
                                    "result": item.attribute_name, "sources": sources})

            domain_facet_attributes[dom][fac] = new_attrs
            partition_attributes.setdefault(dom, {})[fac] = new_attrs

            for m in (result.misfits or []):
                for txt in (m.instance_texts or []):
                    moves[(dom, fac, m.from_attribute, self._norm_text(txt))] = (
                        m.target_attribute if m.verdict == "move" else None)
                log.append({"action": f"misfit_{m.verdict}", "domain": dom, "facet": fac,
                            "from_attribute": m.from_attribute,
                            "target": m.target_attribute,
                            "n_texts": len(m.instance_texts or []),
                            "texts": m.instance_texts, "reason": m.reason})

        # A source split into exactly ONE child was renamed, not divided — follow it.
        # A source split into several children is genuinely ambiguous as a move target:
        # picking a child would be a guess, so those are reported, not resolved.
        split_ambiguous: Dict[str, List[str]] = {}
        for src, children in split_children.items():
            uniq = sorted(set(children))
            if len(uniq) == 1:
                renamed_to.setdefault(src, uniq[0])
            else:
                split_ambiguous[src] = uniq

        # where does every surviving attribute live now?
        home: Dict[str, Tuple[str, str]] = {}
        ambiguous: Set[str] = set()
        for dom, facets in domain_facet_attributes.items():
            for fac, attrs in facets.items():
                for a in attrs:
                    if a.attribute_name in home and home[a.attribute_name] != (dom, fac):
                        ambiguous.add(a.attribute_name)
                    home[a.attribute_name] = (dom, fac)

        # ---- 2-4. ideas ------------------------------------------------------
        idea_facet: Dict[str, Tuple[str, str]] = {}
        for dom, assigns in partition_assignments.items():
            for iid, fac in assigns.items():
                idea_facet[iid] = (dom, fac)

        text_of = {}
        for task in tasks:
            for idea in task['facet_ideas']:
                text_of[idea.idea_id] = self._norm_text(getattr(idea, "instance", ""))

        n_split = n_remap = n_moved = n_out = n_unresolved = n_target_split = 0
        unresolved_targets: Counter = Counter()
        for iid, cur in list(attribute_assignments.items()):
            place = idea_facet.get(iid)
            if not place:
                continue
            dom, fac = place
            txt = text_of.get(iid, "")

            mkey = (dom, fac, cur, txt)
            if mkey in moves:
                target = moves[mkey]
                if target is None:
                    n_out += 1                      # flagged contentless; left in place
                    continue
                # The neighbour block showed pre-P9 names, and facets consolidate
                # concurrently, so a valid target may already have been renamed by
                # its own facet. Follow the rename before giving up.
                if target not in home:
                    target = renamed_to.get(self._norm_text(target), target)
                if target in home and target not in ambiguous:
                    attribute_assignments[iid] = target
                    t_dom, t_fac = home[target]
                    partition_assignments.setdefault(t_dom, {})[iid] = t_fac
                    if t_dom != dom:
                        partition_assignments.get(dom, {}).pop(iid, None)
                    n_moved += 1
                elif self._norm_text(target) in split_ambiguous:
                    n_target_split += 1     # target was divided; choosing a child would guess
                    unresolved_targets[target] += 1
                else:
                    n_unresolved += 1
                    unresolved_targets[target] += 1
                continue

            skey = (dom, fac, cur, txt)
            if skey in splits:
                attribute_assignments[iid] = splits[skey]
                n_split += 1
                continue

            rkey = (dom, fac, cur)
            if rkey in remap:
                attribute_assignments[iid] = remap[rkey]
                n_remap += 1

        # ---- 5. self-check: no idea may point at a node that does not exist -----
        # P9 can drop an attribute it never mentions as a source. Its ideas would
        # then carry a name absent from the structure, and everything downstream
        # (codebook, export) silently loses them. Restore the node instead.
        orphans: Counter = Counter()
        for iid, name in attribute_assignments.items():
            if name and name not in home:
                orphans[(idea_facet.get(iid, ("?", "?")), name)] += 1

        restored = 0
        for (place, name), count in orphans.items():
            dom, fac = place
            src = next((a for a in (pre_p7_attrs.get((dom, fac)) or [])
                        if a.attribute_name == name), None)
            if src is not None and dom in domain_facet_attributes:
                # Both structures usually hold the SAME list object for this facet
                # (assigned together in step 1), so a bare append to each would
                # insert the node twice.
                for attrs in (domain_facet_attributes[dom].setdefault(fac, []),
                              partition_attributes.setdefault(dom, {}).setdefault(fac, [])):
                    if all(a.attribute_name != name for a in attrs):
                        attrs.append(src)
                home[name] = (dom, fac)
                restored += 1
        if orphans:
            log.append({"action": "orphaned_assignment", "restored_nodes": restored,
                        "ideas_affected": sum(orphans.values()),
                        "attributes": sorted({n for (_, n) in orphans}),
                        "note": ("P9 returned no attribute claiming these, so their "
                                 "ideas kept a name absent from the structure; the "
                                 "node was put back to keep the two consistent")})
            if verbose:
                print(f"    SELF-CHECK: {sum(orphans.values())} ideas pointed at "
                      f"{len(orphans)} attribute(s) missing from the structure — "
                      f"{restored} node(s) restored")

        log.append({"action": "_totals", "ideas_split": n_split, "ideas_remapped": n_remap,
                    "ideas_moved": n_moved, "flagged_contentless_left_in_place": n_out,
                    "moves_with_unresolvable_target": n_unresolved,
                    "moves_whose_target_was_itself_split": n_target_split,
                    "unresolved_target_names": dict(unresolved_targets.most_common(20)),
                    "ambiguous_attribute_names": sorted(ambiguous)})

        if verbose:
            print(f"    Ideas: {n_remap} remapped, {n_split} split, {n_moved} moved "
                  f"across attributes, {n_out} flagged contentless (left in place)")
            if n_unresolved or n_target_split:
                print(f"    {n_unresolved} moves named an unknown target, "
                      f"{n_target_split} named a target that was itself split — "
                      f"both left in place (see the log)")

        return attribute_assignments, partition_assignments, log

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
        """Group ideas by (domain, facet) using P4 assignments.

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
                boundary_test=part.boundary_test,
                exclusions=part.exclusions,
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
