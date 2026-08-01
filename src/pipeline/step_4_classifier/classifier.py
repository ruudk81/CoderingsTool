"""
Taxonomy Classifier: inductive taxonomy discovery (P1-P10).

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
    DEFAULT_PROCESSING_CONFIG, FALLBACK_TPM, FALLBACK_RPM, get_reasoning_params,
)

from pipeline.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)
from utils.smoothRequester import SmoothRequester

from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from .domain_discoverer import PartitionLabelMapping
from .partition_labels import format_label
from .taxonomy_health import drain_domains
from models import DomainSet, DomainDescription
from .prompts_classifier import (
    # P1: Facet Discovery
    build_facet_discovery_prompt,
    FacetDiscoveryResult,
    DiscoveredFacet,
    # P2: Facet Consolidation
    build_facet_consolidation_prompt,
    FacetConsolidatedResponse,
    # P3: Facet Review
    build_facet_review_prompt,
    FacetReviewResponse,
    # P4: Facet Assignment
    build_facet_assignment_prompt_single,
    FacetAssignmentResult,
    # P5: Attribute Discovery
    build_attribute_discovery_prompt,
    AttributeDiscoveryResult,
    DiscoveredAttribute,
    # P6: Attribute Chunk Consolidation
    build_attribute_chunk_consolidation_prompt,
    AttributeChunkConsolidatedResponse,
    # P7: Attribute Review
    build_attribute_review_prompt,
    AttributeReviewResponse,
    # P8: Attribute Assignment
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
    """Output of taxonomy stages P1-P10."""
    partition_n_labels: Dict[str, int]
    partition_n_batches: Dict[str, int]
    partition_facets: Dict[str, List[DiscoveredFacet]]
    partition_assignments: Dict[str, Dict[str, str]]  # domain -> {idea_id -> facet_name}
    partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]]  # domain -> {facet -> [attrs]}
    attribute_assignments: Dict[str, str]  # idea_id -> attribute_name
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
    # entry per flagged pair — meant to travel in-memory to P9 as a
    # suspected-overlap hint (consumed downstream). Also mirrored into
    # consolidation_log as attribute_review_flag entries.
    attribute_review_flags: List[Dict] = field(default_factory=list)

# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class TaxonomyClassifier:
    """
    Taxonomy Classifier: inductive taxonomy discovery (P1-P10).

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
        self._model_p1 = config.qr_model_p1
        self._model_p2 = config.qr_model_p2
        self._model_p3 = config.qr_model_p3
        self._model_p4 = config.qr_model_p4
        self._model_p5 = config.qr_model_p5
        self._model_p6 = config.qr_model_p6
        self._model_p7 = config.qr_model_p7
        self._model_p8 = config.qr_model_p8
        self._model_p9 = config.qr_model_p9
        self._model_p10 = config.qr_model_p10

        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_4_taxonomy_classifier", {
                "p1_facet_discovery": self._model_p1,
                "p2_facet_consolidation": self._model_p2,
                "p3_facet_review": self._model_p3,
                "p4_facet_assignment": self._model_p4,
                "p5_attribute_discovery": self._model_p5,
                "p6_attribute_consolidation": self._model_p6,
                "p7_attribute_review": self._model_p7,
                "p8_attribute_assignment": self._model_p8,
                "p9_in_facet_consolidation": self._model_p9,
                "p10_valence_merge": self._model_p10,
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

        # Prompt capture (optional)
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()

        self._debug_stop_after_phase = config.debug_stop_after_phase
        self._facet_review_enabled = config.facet_review_enabled
        self._attribute_review_enabled = config.attribute_review_enabled

        # Assignment confidence scores and valence (populated by P4/P8 parse_fns)
        self._facet_confidence: Dict[str, float] = {}
        self._attribute_confidence: Dict[str, float] = {}
        self._facet_valence: Dict[str, str] = {}
        self._attribute_valence: Dict[str, str] = {}

        # Rate limits — fetched once in _initialize_async_resources()
        self._fetched_limits = None
        self._fetched_has_headers = None

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
        """Run taxonomy stages (P1-P10): facets, attributes, assignments.

        `extraction_metadata` (models.ExtractionMetadata, optional) feeds P3/P7's
        drain-domain skip (taxonomy_health.drain_domains) — without it every domain
        is treated as reviewable, same as a legacy cache with no standing keys.
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
        """Fetch rate limits from API. All LLM calls go through SmoothRequester."""
        # --- Fetch real rate limits from API headers ---
        if verbose:
            print("  Fetching rate limits from API...")
        limits, has_headers = await llm_fetch_rate_limits(self._model_p1)

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

        self._fetched_limits = limits
        self._fetched_has_headers = has_headers
        headroom = DEFAULT_PROCESSING_CONFIG.rate_limit_headroom

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Models: P1={self._model_p1}, P2={self._model_p2}, "
                  f"P3={self._model_p3}, P4={self._model_p4}, P5={self._model_p5}, "
                  f"P6={self._model_p6}, P8={self._model_p8}, P9={self._model_p9}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")

    async def _process_taxonomy_async(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
        extraction_metadata=None,
    ) -> TaxonomyResult:
        """Taxonomy stages P1-P10: facets, attributes, assignments."""
        start_time = time.time()
        self._facet_confidence.clear()
        self._attribute_confidence.clear()
        self._facet_valence.clear()
        self._attribute_valence.clear()

        # =================================================================
        # PHASE 1 (P1): Per-domain Facet Discovery (SmoothRequester)
        # + PHASE 2 (P2): Facet Consolidation (per-domain, sequential)
        # =================================================================
        _snap_p1 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 1: Facet Discovery + Consolidation")

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
            model=self._model_p1,
            phase_key="step4_p1_facet_discovery",
            num_tasks=len(p1_tasks),
            verbose=verbose,
            known_limits=self._fetched_limits,
            has_server_headers=self._fetched_has_headers,
            show_setup=False,
            quiet=True,
        )
        p1_results = await p1_requester.process_all(
            p1_tasks,
            self._p1_prepare_fn(prompt_context),
            self._p1_parse_fn(),
            self._p1_fallback_fn(),
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
                "step_4_taxonomy_classifier", "p1_facet_discovery",
                _snap_p1, token_tracker.snapshot(), self._model_p1)

        if self._debug_stop_after_phase == 1:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P1 — skipping P2-P9")
            return TaxonomyResult(
                partition_n_labels={},
                partition_n_batches={},
                partition_facets={},
                partition_assignments={},
                partition_attributes={},
                attribute_assignments={},
            )

        # P2 consolidation per domain (SmoothRequester, concurrent)
        _snap_p2 = token_tracker.snapshot() if self.cost_tracker else None
        t_consolidation = time.time()
        partition_facets: Dict[str, List[DiscoveredFacet]] = {}
        partition_n_labels: Dict[str, int] = {}
        partition_n_batches: Dict[str, int] = {}

        max_c = self._consolidation_max_chunks_per_call
        max_i = self._consolidation_max_items_per_call

        # Build P2 task list — one task per domain (single-round) or per group (multi-round)
        p2_tasks = []
        for name in sorted(domain_chunk_facets.keys()):
            chunk_facets = domain_chunk_facets[name]
            non_empty = [cf for cf in chunk_facets if cf]
            if not non_empty:
                partition_facets[name] = []
                continue

            n_chunks = len(non_empty)
            total_items = sum(len(cf) for cf in non_empty)
            partition_n_labels[name] = domain_chunk_info[name]['n_labels']
            partition_n_batches[name] = domain_chunk_info[name]['n_batches']

            if n_chunks <= max_c and total_items <= max_i:
                # Single-round: one task for this domain
                p2_tasks.append({
                    'domain_name': name,
                    'chunk_facets': non_empty,
                    'part_context': partition_contexts[name],
                    'excluded_domains': domain_chunk_info[name]['excluded'],
                    'round': 1,
                })
            else:
                # Multi-round: split into groups
                group_size = max_c
                avg_items = total_items / n_chunks
                while group_size > 2 and group_size * avg_items > max_i:
                    group_size -= 1
                groups = [non_empty[i:i + group_size] for i in range(0, n_chunks, group_size)]
                for group in groups:
                    p2_tasks.append({
                        'domain_name': name,
                        'chunk_facets': group,
                        'part_context': partition_contexts[name],
                        'excluded_domains': domain_chunk_info[name]['excluded'],
                        'round': 1,
                        'is_group': True,
                    })

        # Run P2 round 1
        if p2_tasks:
            p2_requester = SmoothRequester(
                model=self._model_p2,
                phase_key="step4_p2_facet_consolidation",
                num_tasks=len(p2_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                has_server_headers=self._fetched_has_headers,
                show_setup=False,
                quiet=True,
            )
            p2_results = await p2_requester.process_all(
                p2_tasks,
                self._p2_prepare_fn(prompt_context),
                self._p2_parse_fn(),
                self._p2_fallback_fn(),
            )

            # Collect results per domain
            domain_round1: Dict[str, List[List[DiscoveredFacet]]] = {}
            for task, result in zip(p2_tasks, p2_results):
                name = task['domain_name']
                if name not in domain_round1:
                    domain_round1[name] = []
                domain_round1[name].append(result or [])

            # Check if any domain needs round 2 (multi-round: intermediate results)
            needs_round2 = {}
            for name, results_list in domain_round1.items():
                if len(results_list) == 1 and not any(t.get('is_group') for t in p2_tasks if t['domain_name'] == name):
                    # Single-round domain — done
                    partition_facets[name] = results_list[0]
                else:
                    # Multi-round: results_list has one result per group → check if fits in one call now
                    non_empty = [r for r in results_list if r]
                    n = len(non_empty)
                    total = sum(len(r) for r in non_empty)
                    if n <= max_c and total <= max_i:
                        needs_round2[name] = non_empty
                    else:
                        # Still too big — flatten and take what we have
                        all_facets = [f for group in non_empty for f in group]
                        partition_facets[name] = all_facets

            # Round 2 if needed
            if needs_round2:
                r2_tasks = []
                for name, intermediate in needs_round2.items():
                    r2_tasks.append({
                        'domain_name': name,
                        'chunk_facets': intermediate,
                        'part_context': partition_contexts[name],
                        'excluded_domains': domain_chunk_info[name]['excluded'],
                        'round': 2,
                    })
                r2_requester = SmoothRequester(
                    model=self._model_p2,
                    phase_key="step4_p2_facet_consolidation",
                    num_tasks=len(r2_tasks),
                    verbose=verbose,
                    known_limits=self._fetched_limits,
                    has_server_headers=self._fetched_has_headers,
                    show_setup=False,
                    quiet=True,
                )
                r2_results = await r2_requester.process_all(
                    r2_tasks,
                    self._p2_prepare_fn(prompt_context),
                    self._p2_parse_fn(),
                    self._p2_fallback_fn(),
                )
                for task, result in zip(r2_tasks, r2_results):
                    partition_facets[task['domain_name']] = result or []

        t_consolidation = time.time() - t_consolidation
        if verbose:
            s = p2_requester.stats if p2_tasks else {}
            print(f"    P2 consolidation: {len(p2_tasks)} tasks, {t_consolidation:.1f}s "
                  f"({s.get('tasks_successful', 0)} ok, {s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries)")

        phase1_elapsed = time.time() - t_phase1
        if verbose:
            total_facets = sum(len(f) for f in partition_facets.values())
            print(f"    Results ({phase1_elapsed:.1f}s → {total_facets} facets):")
            for name in sorted(partition_facets.keys()):
                n_raw = sum(len(cf) for cf in domain_chunk_facets.get(name, []))
                facets = partition_facets.get(name, [])
                facet_names = ", ".join(f.facet_name for f in facets) if facets else "(none)"
                print(f"      {name}: {n_raw} raw → {len(facets)} facet(s): {facet_names}")

        if self.cost_tracker and _snap_p2 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p2_facet_consolidation",
                _snap_p2, token_tracker.snapshot(), self._model_p2)

        if self._debug_stop_after_phase == 2:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P2 — skipping P4–P9")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments={},
                partition_attributes={},
                attribute_assignments={},
            )

        # =================================================================
        # PHASE 3 (P3): Per-domain Facet Review (optional, SmoothRequester,
        # light mode — mirrors the P2 dispatch). Rewrite + flag only: the
        # facet SET cannot change, enforced by the response schema. Skips
        # the standing drain domains and domains with fewer than 2 facets
        # (nothing to review against).
        # =================================================================
        consolidation_log: List[Dict] = []
        attribute_review_flags: List[Dict] = []

        if self._facet_review_enabled:
            _snap_p3r = token_tracker.snapshot() if self.cost_tracker else None
            t_p3r = time.time()

            drain = drain_domains(extraction_metadata)
            if verbose and not drain:
                print("    P3 facet review: no standing drain-domain keys in "
                      "metadata (legacy cache) — every domain is reviewable")

            p3r_tasks = []
            for name in sorted(partition_facets.keys()):
                facets = partition_facets.get(name, [])
                if name in drain or len(facets) < 2:
                    continue
                p3r_tasks.append({
                    'domain_name': name,
                    'facets': facets,
                    'part_context': partition_contexts[name],
                })

            if p3r_tasks:
                if verbose:
                    print(f"\n  Phase 3: Facet Review")

                p3r_requester = SmoothRequester(
                    model=self._model_p3,
                    phase_key="step4_p3_facet_review",
                    num_tasks=len(p3r_tasks),
                    verbose=verbose,
                    known_limits=self._fetched_limits,
                    has_server_headers=self._fetched_has_headers,
                    show_setup=False,
                    quiet=True,
                )
                p3r_results = await p3r_requester.process_all(
                    p3r_tasks,
                    self._p3_review_prepare_fn(prompt_context),
                    self._p3_review_parse_fn(),
                    self._p3_review_fallback_fn(),
                )

                p3r_domain_counts: Dict[str, Dict[str, int]] = {}
                for task, response in zip(p3r_tasks, p3r_results):
                    p3r_domain_counts[task['domain_name']] = self._apply_p3_review(
                        task, response, consolidation_log,
                    )

                if verbose:
                    s = p3r_requester.stats
                    print(f"    P3 review: {len(p3r_tasks)} tasks, "
                          f"{time.time() - t_p3r:.1f}s ({s.get('tasks_successful', 0)} ok, "
                          f"{s.get('timeouts', 0)} timeouts, {s.get('recovered', 0)} retries)")
                    for name in sorted(p3r_domain_counts.keys()):
                        c = p3r_domain_counts[name]
                        if c['failed']:
                            print(f"      {name}: review failed (facet set mismatch) — domain unchanged")
                        else:
                            print(f"      {name}: {c['reviewed']} facets reviewed, "
                                  f"{c['rewritten']} rewritten, {c['flagged']} flagged")

            if self.cost_tracker and _snap_p3r is not None:
                self.cost_tracker.record_phase(
                    "step_4_taxonomy_classifier", "p3_facet_review",
                    _snap_p3r, token_tracker.snapshot(), self._model_p3)

        # =================================================================
        # PHASE 4 (P4): Per-domain Facet Assignment (SmoothRequester)
        # =================================================================
        _snap_p3 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 2: Facet Assignment")

        t_phase3 = time.time()

        # Build flat task list: one task per idea
        # Single-facet domains are auto-assigned without LLM call
        p3_tasks = []
        partition_assignments: Dict[str, Dict[str, str]] = {}
        p3_auto_assigned: Dict[str, int] = {}  # domain → idea count (for reporting)
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

            # Multi-facet: one task per idea
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

        if p3_tasks:
            p3_requester = SmoothRequester(
                model=self._model_p4,
                phase_key="step4_p4_facet_assignment",
                num_tasks=len(p3_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                has_server_headers=self._fetched_has_headers,
                show_setup=False,
                quiet=False,
            )
            p3_results = await p3_requester.process_all(
                p3_tasks,
                self._p3_prepare_fn(prompt_context),
                self._p3_parse_fn(),
                self._p3_fallback_fn(),
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
        elif verbose and p3_auto_assigned:
            print(f"    Assignment: all {len(p3_auto_assigned)} domains auto-assigned (1 facet each)")

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
                "step_4_taxonomy_classifier", "p3_facet_assignment",
                _snap_p3, token_tracker.snapshot(), self._model_p4)

        if self._debug_stop_after_phase == 3:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P4 — skipping P5–P9")
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
            )

        # =================================================================
        # PHASE 5 (P5): Per-facet Attribute Discovery (SmoothRequester)
        # + PHASE 6 (P6): Attribute Consolidation (per-facet, sequential)
        # =================================================================
        _snap_p4 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 3: Attribute Discovery + Consolidation")

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
                    'excluded_facets': excluded_f,
                    'facet_key': facet_key,
                })

        # P5 discovery via SmoothRequester
        if p4_tasks:
            p4_requester = SmoothRequester(
                model=self._model_p5,
                phase_key="step4_p5_attribute_discovery",
                num_tasks=len(p4_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                has_server_headers=self._fetched_has_headers,
                show_setup=False,
                quiet=True,
            )
            p4_results = await p4_requester.process_all(
                p4_tasks,
                self._p4_prepare_fn(prompt_context),
                self._p4_parse_fn(),
                self._p4_fallback_fn(),
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
            print(f"    P5 discovery: {len(p4_tasks)} tasks, {t_p4_discovery:.1f}s "
                  f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries)")

        if self.cost_tracker and _snap_p4 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p4_attribute_discovery",
                _snap_p4, token_tracker.snapshot(), self._model_p5)

        if self._debug_stop_after_phase == 4:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P5 — skipping P6–P9")
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
            )

        # P6 consolidation per facet (SmoothRequester, concurrent)
        _snap_p5 = token_tracker.snapshot() if self.cost_tracker else None
        t_consolidation = time.time()
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}

        # Build P6 task list
        p5_tasks = []
        for facet_key, chunk_attributes in sorted(facet_chunk_attrs.items()):
            domain_name, facet_name = facet_key.split("::", 1)
            meta = facet_meta[facet_key]
            non_empty = [ca for ca in chunk_attributes if ca]

            if not non_empty:
                if domain_name not in domain_facet_attributes:
                    domain_facet_attributes[domain_name] = {}
                domain_facet_attributes[domain_name][facet_name] = []
                if domain_name not in partition_attributes:
                    partition_attributes[domain_name] = {}
                partition_attributes[domain_name][facet_name] = []
                continue

            n_chunks = len(non_empty)
            total_items = sum(len(ca) for ca in non_empty)

            if n_chunks <= max_c and total_items <= max_i:
                p5_tasks.append({
                    'domain_name': domain_name,
                    'facet_name': facet_name,
                    'facet_description': meta['facet_obj'].facet_description,
                    'chunk_attributes': non_empty,
                    'excluded_facets': meta['excluded_facets'],
                    'facet_key': facet_key,
                })
            else:
                # Multi-round: split into groups
                group_size = max_c
                avg_items = total_items / n_chunks
                while group_size > 2 and group_size * avg_items > max_i:
                    group_size -= 1
                groups = [non_empty[i:i + group_size] for i in range(0, n_chunks, group_size)]
                for group in groups:
                    p5_tasks.append({
                        'domain_name': domain_name,
                        'facet_name': facet_name,
                        'facet_description': meta['facet_obj'].facet_description,
                        'chunk_attributes': group,
                        'excluded_facets': meta['excluded_facets'],
                        'facet_key': facet_key,
                        'is_group': True,
                    })

        if p5_tasks:
            p5_requester = SmoothRequester(
                model=self._model_p6,
                phase_key="step4_p6_attribute_consolidation",
                num_tasks=len(p5_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                has_server_headers=self._fetched_has_headers,
                show_setup=False,
                quiet=True,
            )
            p5_results = await p5_requester.process_all(
                p5_tasks,
                self._p5_prepare_fn(prompt_context),
                self._p5_parse_fn(),
                self._p5_fallback_fn(),
            )

            # Collect results per facet
            facet_round1: Dict[str, List[List[DiscoveredAttribute]]] = {}
            for task, result in zip(p5_tasks, p5_results):
                fk = task['facet_key']
                if fk not in facet_round1:
                    facet_round1[fk] = []
                facet_round1[fk].append(result or [])

            # Check multi-round + assemble results
            needs_round2 = {}
            for fk, results_list in facet_round1.items():
                dn, fn = fk.split("::", 1)
                if len(results_list) == 1 and not any(t.get('is_group') for t in p5_tasks if t['facet_key'] == fk):
                    attributes = results_list[0]
                else:
                    non_empty = [r for r in results_list if r]
                    n = len(non_empty)
                    total = sum(len(r) for r in non_empty)
                    if n <= max_c and total <= max_i:
                        needs_round2[fk] = non_empty
                        continue
                    else:
                        attributes = [a for group in non_empty for a in group]

                if dn not in domain_facet_attributes:
                    domain_facet_attributes[dn] = {}
                domain_facet_attributes[dn][fn] = attributes
                if dn not in partition_attributes:
                    partition_attributes[dn] = {}
                partition_attributes[dn][fn] = attributes

            # Round 2 if needed
            if needs_round2:
                r2_tasks = []
                for fk, intermediate in needs_round2.items():
                    dn, fn = fk.split("::", 1)
                    meta = facet_meta[fk]
                    r2_tasks.append({
                        'domain_name': dn,
                        'facet_name': fn,
                        'facet_description': meta['facet_obj'].facet_description,
                        'chunk_attributes': intermediate,
                        'excluded_facets': meta['excluded_facets'],
                        'facet_key': fk,
                    })
                r2_requester = SmoothRequester(
                    model=self._model_p6,
                    phase_key="step4_p6_attribute_consolidation",
                    num_tasks=len(r2_tasks),
                    verbose=verbose,
                    known_limits=self._fetched_limits,
                    has_server_headers=self._fetched_has_headers,
                    show_setup=False,
                    quiet=True,
                )
                r2_results = await r2_requester.process_all(
                    r2_tasks,
                    self._p5_prepare_fn(prompt_context),
                    self._p5_parse_fn(),
                    self._p5_fallback_fn(),
                )
                for task, result in zip(r2_tasks, r2_results):
                    dn, fn = task['facet_key'].split("::", 1)
                    if dn not in domain_facet_attributes:
                        domain_facet_attributes[dn] = {}
                    domain_facet_attributes[dn][fn] = result or []
                    if dn not in partition_attributes:
                        partition_attributes[dn] = {}
                    partition_attributes[dn][fn] = result or []

        t_consolidation = time.time() - t_consolidation
        if verbose:
            s = p5_requester.stats if p5_tasks else {}
            print(f"    P6 consolidation: {len(p5_tasks)} tasks, {t_consolidation:.1f}s "
                  f"({s.get('tasks_successful', 0)} ok, {s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries)")

        t_phase4 = time.time() - t_phase4
        if verbose:
            total_attrs = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            print(f"    Results ({t_phase4:.1f}s → {total_attrs} attributes across {len(facet_chunk_attrs)} facets):")
            for facet_key in sorted(facet_chunk_attrs.keys()):
                domain_name, facet_name = facet_key.split("::", 1)
                n_raw = sum(len(ca) for ca in facet_chunk_attrs[facet_key])
                attrs = domain_facet_attributes.get(domain_name, {}).get(facet_name, [])
                attr_names = ", ".join(a.attribute_name for a in attrs) if attrs else "(none)"
                print(f"      {domain_name}/{facet_name}: {n_raw} raw → {len(attrs)} attr(s): {attr_names}")

        if self.cost_tracker and _snap_p5 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p5_attribute_consolidation",
                _snap_p5, token_tracker.snapshot(), self._model_p6)

        if self._debug_stop_after_phase == 5:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P6 — skipping P7–P9")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes=partition_attributes,
                attribute_assignments={},
                facet_confidence=self._facet_confidence,
                facet_valence=self._facet_valence,
                consolidation_log=consolidation_log,
            )

        # =================================================================
        # PHASE 7 (P7): Per-domain Attribute Review (optional, SmoothRequester,
        # light mode — mirrors the P3/P6 dispatch). Rewrite + flag only: the
        # attribute SET cannot change, enforced by the response schema. Skips
        # the standing drain domains and domains with fewer than 2 attributes
        # in total across their facets (nothing to review against).
        # =================================================================
        if self._attribute_review_enabled:
            _snap_p7r = token_tracker.snapshot() if self.cost_tracker else None
            t_p7r = time.time()

            drain = drain_domains(extraction_metadata)
            if verbose and not drain:
                print("    P7 attribute review: no standing drain-domain keys in "
                      "metadata (legacy cache) — every domain is reviewable")

            p7r_tasks = []
            for name in sorted(partition_attributes.keys()):
                facet_attrs = partition_attributes.get(name, {})
                total_attrs = sum(len(attrs) for attrs in facet_attrs.values())
                if name in drain or total_attrs < 2:
                    continue
                p7r_tasks.append({
                    'domain_name': name,
                    'facets': partition_facets.get(name, []),
                    'facet_attributes': facet_attrs,
                    'part_context': partition_contexts[name],
                })

            if p7r_tasks:
                if verbose:
                    print(f"\n  Phase 7: Attribute Review")

                p7r_requester = SmoothRequester(
                    model=self._model_p7,
                    phase_key="step4_p7_attribute_review",
                    num_tasks=len(p7r_tasks),
                    verbose=verbose,
                    known_limits=self._fetched_limits,
                    has_server_headers=self._fetched_has_headers,
                    show_setup=False,
                    quiet=True,
                )
                p7r_results = await p7r_requester.process_all(
                    p7r_tasks,
                    self._p7_review_prepare_fn(prompt_context),
                    self._p7_review_parse_fn(),
                    self._p7_review_fallback_fn(),
                )

                p7r_domain_counts: Dict[str, Dict[str, int]] = {}
                for task, response in zip(p7r_tasks, p7r_results):
                    p7r_domain_counts[task['domain_name']] = self._apply_p7_review(
                        task, response, consolidation_log, attribute_review_flags,
                    )

                if verbose:
                    s = p7r_requester.stats
                    print(f"    P7 review: {len(p7r_tasks)} tasks, "
                          f"{time.time() - t_p7r:.1f}s ({s.get('tasks_successful', 0)} ok, "
                          f"{s.get('timeouts', 0)} timeouts, {s.get('recovered', 0)} retries)")
                    for name in sorted(p7r_domain_counts.keys()):
                        c = p7r_domain_counts[name]
                        if c['failed']:
                            print(f"      {name}: review failed (attribute set mismatch) — domain unchanged")
                        else:
                            print(f"      {name}: {c['reviewed']} attributes reviewed, "
                                  f"{c['rewritten']} rewritten, {c['flagged']} flagged")

            if self.cost_tracker and _snap_p7r is not None:
                self.cost_tracker.record_phase(
                    "step_4_taxonomy_classifier", "p7_attribute_review",
                    _snap_p7r, token_tracker.snapshot(), self._model_p7)

        # =================================================================
        # PHASE 8 (P8): Per-facet Attribute Assignment (SmoothRequester)
        # =================================================================
        _snap_p6 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 4: Attribute Assignment")

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
                model=self._model_p8,
                phase_key="step4_p8_attribute_assignment",
                num_tasks=len(p6_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                has_server_headers=self._fetched_has_headers,
                show_setup=False,
                quiet=False,
            )
            p6_results = await p6_requester.process_all(
                p6_tasks,
                self._p6_prepare_fn(prompt_context),
                self._p6_parse_fn(),
                self._p6_fallback_fn(),
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
                "step_4_taxonomy_classifier", "p6_attribute_assignment",
                _snap_p6, token_tracker.snapshot(), self._model_p8)

        # Snapshot P8 state before the post-assignment consolidation round remaps.
        # This is what makes a bad merge diagnosable after the fact — keep it.
        raw_attribute_assignments = dict(attribute_assignments)
        raw_partition_attributes = {
            d: {f: list(attrs) for f, attrs in facets.items()}
            for d, facets in partition_attributes.items()
        }

        if self._debug_stop_after_phase == 6:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P8 — skipping P9")
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
                attribute_review_flags=attribute_review_flags,
            )

        # =================================================================
        # PHASE 9: In-facet Attribute Consolidation (post-assignment)
        # Replaces the earlier cross-facet and cross-domain consolidation rounds
        # that used to follow attribute assignment.
        # Scope is ONE facet, so no merge can relocate an idea's facet; when a
        # group of ideas belongs elsewhere the IDEAS move and the structure stays.
        # =================================================================
        _snap_p7 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 9: In-facet Attribute Consolidation")

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
                model=self._model_p9,
                phase_key="step4_p9_in_facet_consolidation",
                num_tasks=len(p7_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                has_server_headers=self._fetched_has_headers,
                show_setup=False,
                quiet=True,
            )
            p7_results = await p7_requester.process_all(
                p7_tasks,
                self._p9_prepare_fn(prompt_context),
                self._p9_parse_fn(),
                self._p9_fallback_fn(),
            )

            if verbose:
                s = p7_requester.stats
                print(f"    P9 consolidation: {len(p7_tasks)} tasks, "
                      f"{s.get('wall_time', 0):.1f}s ({s['tasks_successful']} ok, "
                      f"{s.get('timeouts', 0)} timeouts, {s.get('recovered', 0)} retries)")

            attribute_assignments, partition_assignments, p9_log = (
                self._apply_p9_results(
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
                "step_4_taxonomy_classifier", "p7_in_facet_consolidation",
                _snap_p7, token_tracker.snapshot(), self._model_p9)

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
            attribute_review_flags=attribute_review_flags,
        )

    # =========================================================================
    # PHASE 4 (P4): PER-DOMAIN FACET ASSIGNMENT (SmoothRequester)
    # =========================================================================

    def _p3_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P4 facet assignment (single idea)."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_facet_assignment_prompt_single(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facets=task['facets'],
                idea_label=task['idea_label'],
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

    def _p3_parse_fn(self):
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
    def _p3_fallback_fn():
        """Return fallback_fn closure for P4 facet assignment."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            return {}
        return fallback_fn

    # =========================================================================
    # PHASE 8 (P8): PER-FACET ATTRIBUTE ASSIGNMENT (SmoothRequester)
    # =========================================================================

    def _p6_prepare_fn(self, prompt_context: PromptContext):
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
                        "model": self._model_p8,
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
                'extra_kwargs': get_reasoning_params(self._model_p8, phase="classifier_p8"),
            }
        return prepare_fn

    def _p6_parse_fn(self):
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
    def _p6_fallback_fn():
        """Return fallback_fn closure for P8 attribute assignment."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            return {}
        return fallback_fn

    # =========================================================================
    # PHASE 1 (P1): PER-DOMAIN FACET DISCOVERY (SmoothRequester)
    # =========================================================================

    def _p1_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P1 facet discovery."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_facet_discovery_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                partition_name=task['domain_name'],
                partition_definition=task['part_context'].partition_definition,
                boundary_test=task['part_context'].boundary_test,
                exclusions=task['part_context'].exclusions,
                observations=task['observations'],
                excluded_domains=task['excluded_domains'],
            )

            # Prompt capture (first chunk per domain)
            gate_key = f"qr_facets_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and task['chunk_idx'] == 0
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
                        "partition_name": task['domain_name'],
                        "batch_number": task['chunk_idx'] + 1,
                        "total_batches": task['total_chunks'],
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': FacetDiscoveryResult,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_facet_discovery,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p1, phase="classifier_p1"),
            }
        return prepare_fn

    def _p1_parse_fn(self):
        """Return parse_fn closure for P1 facet discovery."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredFacet]]:
            return response.facets if response else []
        return parse_fn

    @staticmethod
    def _p1_fallback_fn():
        """Return fallback_fn closure for P1 facet discovery."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredFacet]:
            return []
        return fallback_fn

    # =========================================================================
    # PHASE 2 (P2): FACET CONSOLIDATION (SmoothRequester)
    # =========================================================================

    @staticmethod
    def _format_chunk_facets(chunk_facets: List[List[DiscoveredFacet]]) -> str:
        """Format chunk-level facet discoveries into text for consolidation prompt."""
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
        return "\n\n".join(formatted_chunks)

    def _p2_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P2 facet consolidation."""
        def prepare_fn(task: Dict) -> Dict:
            chunk_results_text = self._format_chunk_facets(task['chunk_facets'])

            prompt = build_facet_consolidation_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                chunk_results=chunk_results_text,
                excluded_domains=task['excluded_domains'],
            )

            # Prompt capture (first call per domain)
            gate_key = f"qr_facet_consolidation_{task['domain_name']}"
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
                        "max_tokens": self._max_tokens_consolidation,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': FacetConsolidatedResponse,
                'temperature': 0.0,
                'max_tokens': self._max_tokens_consolidation,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p2, phase="classifier_p2"),
            }
        return prepare_fn

    def _p2_parse_fn(self):
        """Return parse_fn closure for P2 facet consolidation."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredFacet]]:
            return response.facets if response else []
        return parse_fn

    @staticmethod
    def _p2_fallback_fn():
        """Return fallback_fn closure for P2 facet consolidation."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredFacet]:
            return []
        return fallback_fn

    # =========================================================================
    # PHASE 3 (P3): PER-DOMAIN FACET REVIEW (SmoothRequester, optional)
    # =========================================================================

    def _p3_review_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P3 facet review."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_facet_review_prompt(
                survey_question=prompt_context.survey_question,
                primary_dimension=prompt_context.dimension_name,
                domain_label=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                domain_boundary_test=task['part_context'].boundary_test,
                facets=task['facets'],
            )

            gate_key = f"qr_facet_review_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="facet_review",
                    metadata={
                        "model": self._model_p3,
                        "temperature": 0.0,
                        "max_tokens": self._max_tokens_consolidation,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': FacetReviewResponse,
                'temperature': 0.0,
                'max_tokens': self._max_tokens_consolidation,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p3, phase="classifier_p3"),
            }
        return prepare_fn

    def _p3_review_parse_fn(self):
        """Return parse_fn closure for P3 facet review."""
        def parse_fn(task: Dict, response) -> Optional[FacetReviewResponse]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _p3_review_fallback_fn():
        """Return fallback_fn closure for P3. On failure the domain's facets are
        left exactly as P2 produced them — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_p3_review(
        self,
        task: Dict,
        response: Optional[FacetReviewResponse],
        consolidation_log: List[Dict],
    ) -> Dict[str, int]:
        """Apply one domain's P3 review in place. Mandate is rewrite-and-flag
        only — the facet SET is fixed. Match is on `original_name`, case-/
        whitespace-insensitive (`_norm_text`, same helper P9's `_resolve` uses);
        a set mismatch (missing or extra names) leaves the domain's facets
        untouched and is logged as `facet_review_failed` with the diff.
        """
        domain = task['domain_name']
        facets = task['facets']  # same list object as partition_facets[domain]
        counts = {'reviewed': len(facets), 'rewritten': 0, 'flagged': 0, 'failed': 0}

        if response is None or not response.facets:
            consolidation_log.append({
                "action": "facet_review_failed", "domain": domain,
                "note": "no response",
            })
            counts['failed'] = 1
            return counts

        by_norm = {self._norm_text(f.facet_name): f for f in facets}
        resp_by_norm = {self._norm_text(r.original_name): r for r in response.facets}

        missing = sorted(f.facet_name for n, f in by_norm.items() if n not in resp_by_norm)
        extra = sorted(r.original_name for n, r in resp_by_norm.items() if n not in by_norm)
        if missing or extra:
            consolidation_log.append({
                "action": "facet_review_failed", "domain": domain,
                "missing": missing, "extra": extra,
            })
            counts['failed'] = 1
            return counts

        for norm_name, facet in by_norm.items():
            reviewed = resp_by_norm[norm_name]
            before = {"facet_name": facet.facet_name, "facet_description": facet.facet_description}
            changed = (reviewed.facet_name != facet.facet_name
                       or reviewed.facet_description != facet.facet_description)
            facet.facet_name = reviewed.facet_name
            facet.facet_description = reviewed.facet_description
            facet.boundary_test = reviewed.boundary_test
            if changed:
                counts['rewritten'] += 1
                consolidation_log.append({
                    "action": "facet_review_rewrite", "domain": domain,
                    "before": before,
                    "after": {"facet_name": facet.facet_name, "facet_description": facet.facet_description},
                })

        for flag in response.overlap_flags:
            counts['flagged'] += 1
            consolidation_log.append({
                "action": "facet_review_flag", "domain": domain,
                "facet_a": flag.facet_a, "facet_b": flag.facet_b, "reason": flag.reason,
            })

        return counts

    # =========================================================================
    # PHASE 5 (P5): PER-FACET ATTRIBUTE DISCOVERY (SmoothRequester)
    # =========================================================================

    def _p4_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P5 attribute discovery."""
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

            # Prompt capture (first chunk per facet)
            gate_key = f"qr_attributes_{task['domain_name']}_{task['facet_name']}"
            if (self._prompt_printer is not None
                    and task['chunk_idx'] == 0
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="attribute_discovery",
                    metadata={
                        "model": self._model_p5,
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
                'response_model': AttributeDiscoveryResult,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_attribute_discovery,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p5, phase="classifier_p5"),
            }
        return prepare_fn

    def _p4_parse_fn(self):
        """Return parse_fn closure for P5 attribute discovery."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredAttribute]]:
            return response.attributes if response else []
        return parse_fn

    @staticmethod
    def _p4_fallback_fn():
        """Return fallback_fn closure for P5 attribute discovery."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredAttribute]:
            return []
        return fallback_fn

    # =========================================================================
    # PHASE 6 (P6): ATTRIBUTE CONSOLIDATION (SmoothRequester)
    # =========================================================================

    @staticmethod
    def _format_chunk_attributes(chunk_attributes: List[List[DiscoveredAttribute]]) -> str:
        """Format chunk-level attribute discoveries into text for consolidation prompt."""
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
        return "\n\n".join(formatted_chunks)

    def _p5_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P6 attribute consolidation."""
        def prepare_fn(task: Dict) -> Dict:
            chunk_results_text = self._format_chunk_attributes(task['chunk_attributes'])

            prompt = build_attribute_chunk_consolidation_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=task['domain_name'],
                facet_name=task['facet_name'],
                facet_description=task['facet_description'],
                chunk_results=chunk_results_text,
                excluded_facets=task['excluded_facets'],
            )

            # Prompt capture (first call per facet)
            gate_key = f"qr_attribute_chunk_consolidation_{task['domain_name']}_{task['facet_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="attribute_chunk_consolidation",
                    metadata={
                        "model": self._model_p6,
                        "temperature": 0.0,
                        "max_tokens": self._max_tokens_consolidation,
                        "language": prompt_context.language,
                        "domain_name": task['domain_name'],
                        "facet_name": task['facet_name'],
                        "n_chunks": len([c for c in task['chunk_attributes'] if c]),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': AttributeChunkConsolidatedResponse,
                'temperature': 0.0,
                'max_tokens': self._max_tokens_consolidation,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p6, phase="classifier_p6"),
            }
        return prepare_fn

    def _p5_parse_fn(self):
        """Return parse_fn closure for P6 attribute consolidation."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredAttribute]]:
            return response.attributes if response else []
        return parse_fn

    @staticmethod
    def _p5_fallback_fn():
        """Return fallback_fn closure for P6 attribute consolidation."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredAttribute]:
            return []
        return fallback_fn

    # =========================================================================
    # PHASE 7 (P7): PER-DOMAIN ATTRIBUTE REVIEW (SmoothRequester, optional)
    # =========================================================================

    def _p7_review_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P7 attribute review."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_attribute_review_prompt(
                survey_question=prompt_context.survey_question,
                domain_label=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facets=task['facets'],
                facet_attributes=task['facet_attributes'],
            )

            gate_key = f"qr_attribute_review_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and gate_key not in self._captured_gates):
                self._prompt_printer.capture_prompt(
                    step_name="qualitative_researcher",
                    utility_name="QualitativeResearcher",
                    prompt_content=prompt,
                    prompt_type="attribute_review",
                    metadata={
                        "model": self._model_p7,
                        "temperature": 0.0,
                        "max_tokens": self._max_tokens_consolidation,
                        "language": prompt_context.language,
                        "partition_name": task['domain_name'],
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': AttributeReviewResponse,
                'temperature': 0.0,
                'max_tokens': self._max_tokens_consolidation,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p7, phase="classifier_p7"),
            }
        return prepare_fn

    def _p7_review_parse_fn(self):
        """Return parse_fn closure for P7 attribute review."""
        def parse_fn(task: Dict, response) -> Optional[AttributeReviewResponse]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _p7_review_fallback_fn():
        """Return fallback_fn closure for P7. On failure the domain's attributes
        are left exactly as P6 produced them — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_p7_review(
        self,
        task: Dict,
        response: Optional[AttributeReviewResponse],
        consolidation_log: List[Dict],
        attribute_review_flags: List[Dict],
    ) -> Dict[str, int]:
        """Apply one domain's P7 review in place. Mandate is rewrite-and-flag
        only — the attribute SET is fixed. Match key is (facet_name,
        original_name), whitespace-/case-insensitive (`_norm_text`, same
        helper P3/P9 use) — the spec's codewaarborg is 1-op-1-dekking per
        (facet, original_name), so coverage is checked per attribute
        INSTANCE, not per distinct name: two attributes that share a
        normalized name across different facets are two separate instances
        that each need their own covering response row.

        Two-pass instance matching (index-based, so duplicate normalized
        names never collapse into one slot):
        - Pass 1: exact (facet, name) match — the normal, well-behaved case.
        - Pass 2: on what's left, name-only match — recovers the case where
          the row names a real attribute but echoes the wrong facet (facets
          are read-only context, so this is not a coverage gap by itself).
        Whatever remains unmatched after both passes is a genuine coverage
        gap: the whole domain's attributes are left untouched, logged as
        `attribute_review_failed` with a `facet > name` diff (Pass 2 already
        absorbed the "known name, wrong facet" case, so anything still
        unmatched here is a real missing/extra instance, not a facet typo).
        Pass-2 pairs are NOT applied — the row is left unchanged, logged as
        `attribute_review_failed` with a note; the rest of the domain (pass-1
        pairs) still applies normally.
        """
        domain = task['domain_name']
        facet_attributes = task['facet_attributes']  # same dict object as partition_attributes[domain]
        actual_items: List[Tuple[str, DiscoveredAttribute]] = [
            (facet_name, attr)
            for facet_name, attrs in facet_attributes.items()
            for attr in attrs
        ]
        counts = {'reviewed': len(actual_items), 'rewritten': 0, 'flagged': 0, 'failed': 0}

        if response is None or not response.attributes:
            consolidation_log.append({
                "action": "attribute_review_failed", "domain": domain,
                "note": "no response",
            })
            counts['failed'] = 1
            return counts

        resp_rows = list(response.attributes)

        # Pass 1: exact (norm_facet, norm_name) match, by index — a plain
        # Dict keyed on name alone silently drops duplicate-named instances
        # living in other facets, so indices (not a collapsing dict) carry
        # the multiplicity.
        actual_by_key: Dict[Tuple[str, str], List[int]] = {}
        for i, (facet_name, attr) in enumerate(actual_items):
            key = (self._norm_text(facet_name), self._norm_text(attr.attribute_name))
            actual_by_key.setdefault(key, []).append(i)

        exact_pairs: List[Tuple[int, int]] = []
        leftover_resp_idx: List[int] = []
        for j, row in enumerate(resp_rows):
            key = (self._norm_text(row.facet_name), self._norm_text(row.original_name))
            candidates = actual_by_key.get(key)
            if candidates:
                exact_pairs.append((candidates.pop(0), j))
            else:
                leftover_resp_idx.append(j)

        matched_actual_idx = {i for i, _ in exact_pairs}
        leftover_actual_idx = [i for i in range(len(actual_items)) if i not in matched_actual_idx]

        # Pass 2: name-only match on what's left — recovers "right name,
        # wrong facet" without treating it as a coverage gap.
        actual_by_name: Dict[str, List[int]] = {}
        for i in leftover_actual_idx:
            actual_by_name.setdefault(self._norm_text(actual_items[i][1].attribute_name), []).append(i)

        facet_mismatch_pairs: List[Tuple[int, int]] = []
        still_unmatched_resp_idx: List[int] = []
        for j in leftover_resp_idx:
            row = resp_rows[j]
            candidates = actual_by_name.get(self._norm_text(row.original_name))
            if candidates:
                facet_mismatch_pairs.append((candidates.pop(0), j))
            else:
                still_unmatched_resp_idx.append(j)

        name_matched_actual_idx = {i for i, _ in facet_mismatch_pairs}
        unmatched_actual_idx = [i for i in leftover_actual_idx if i not in name_matched_actual_idx]
        unmatched_resp_idx = still_unmatched_resp_idx

        if unmatched_actual_idx or unmatched_resp_idx:
            missing = sorted(
                f"{actual_items[i][0]} > {actual_items[i][1].attribute_name}"
                for i in unmatched_actual_idx
            )
            extra = sorted(
                f"{resp_rows[j].facet_name} > {resp_rows[j].original_name}"
                for j in unmatched_resp_idx
            )
            consolidation_log.append({
                "action": "attribute_review_failed", "domain": domain,
                "missing": missing, "extra": extra,
            })
            counts['failed'] = 1
            return counts

        for i, j in exact_pairs:
            facet_name, attr = actual_items[i]
            row = resp_rows[j]
            before = {"attribute_name": attr.attribute_name, "attribute_description": attr.attribute_description}
            changed = (row.attribute_name != attr.attribute_name
                       or row.attribute_description != attr.attribute_description)
            attr.attribute_name = row.attribute_name
            attr.attribute_description = row.attribute_description
            if changed:
                counts['rewritten'] += 1
                consolidation_log.append({
                    "action": "attribute_review_rewrite", "domain": domain, "facet": facet_name,
                    "before": before,
                    "after": {"attribute_name": attr.attribute_name, "attribute_description": attr.attribute_description},
                })

        for i, j in facet_mismatch_pairs:
            facet_name, attr = actual_items[i]
            row = resp_rows[j]
            consolidation_log.append({
                "action": "attribute_review_failed", "domain": domain,
                "facet": facet_name,
                "note": (
                    f"facet mismatch for '{attr.attribute_name}': response "
                    f"facet_name '{row.facet_name}' != actual facet "
                    f"'{facet_name}' — attribute left unchanged"
                ),
            })

        for flag in response.overlap_flags:
            counts['flagged'] += 1
            flag_dict = {
                "attr_a": flag.attr_a, "facet_a": flag.facet_a,
                "attr_b": flag.attr_b, "facet_b": flag.facet_b,
                "reason": flag.reason, "decision_rule": flag.decision_rule,
            }
            attribute_review_flags.append(flag_dict)
            consolidation_log.append({
                "action": "attribute_review_flag", "domain": domain, **flag_dict,
            })

        return counts

    # =========================================================================
    # PHASE 9: IN-FACET ATTRIBUTE CONSOLIDATION (SmoothRequester)
    # =========================================================================

    @staticmethod
    def _norm_text(text: Optional[str]) -> str:
        """Normalise a response text for matching. Case- and padding-insensitive
        only — no stemming, no stopwords, nothing language-specific, so this stays
        use-case agnostic and every match is checkable by eye."""
        return (text or "").strip().lower()

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

    def _p9_prepare_fn(self, prompt_context: PromptContext):
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
                        "model": self._model_p9,
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
                'extra_kwargs': get_reasoning_params(self._model_p9, phase="classifier_p9"),
            }
        return prepare_fn

    def _p9_parse_fn(self):
        """Return parse_fn closure for P9 in-facet attribute consolidation."""
        def parse_fn(task: Dict, response) -> Optional[InFacetConsolidatedResponse]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _p9_fallback_fn():
        """Return fallback_fn closure for P9. On failure the facet is left exactly
        as P8 produced it — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_p9_results(
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
