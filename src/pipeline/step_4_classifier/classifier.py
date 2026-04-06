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
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

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
# HELPERS
# =============================================================================

def _normalize_for_comparison(text: str) -> str:
    """Strip template prefix and canonical_phrasing for similarity comparison."""
    if ' → ' in text:
        text = text.split(' → ', 1)[1]
    text = re.sub(r'\bcanonical_phrasing:\s*', '', text)
    return text.strip().lower()


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

    def __init__(self, config: CategoriesConfig, prompt_printer=None, dataset_key: str = "", cost_tracker=None):
        self.cost_tracker = cost_tracker
        self._model_p1 = config.qr_model_p1
        self._model_p2 = config.qr_model_p2
        self._model_p3 = config.qr_model_p3
        self._model_p4 = config.qr_model_p4
        self._model_p5 = config.qr_model_p5
        self._model_p6 = config.qr_model_p6
        self._model_p7 = config.qr_model_p7

        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_4_taxonomy_classifier", {
                "p1_facet_discovery": self._model_p1,
                "p2_facet_consolidation": self._model_p2,
                "p3_facet_assignment": self._model_p3,
                "p4_attribute_discovery": self._model_p4,
                "p5_attribute_consolidation": self._model_p5,
                "p6_attribute_assignment": self._model_p6,
                "p7_cross_facet_consolidation": self._model_p7,
            })

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

        # Dataset key for empirical stats cache (SmoothRequester)
        self._dataset_key = dataset_key

        # Prompt capture (optional)
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()

        self._debug_stop_after_phase = config.debug_stop_after_phase

        # Rate limits — fetched once in _initialize_async_resources()
        self._fetched_limits = None

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
        """Fetch rate limits from API. All LLM calls go through SmoothRequester."""
        # --- Fetch real rate limits from API headers ---
        if verbose:
            print("  Fetching rate limits from API...")
        limits, _ = await llm_fetch_rate_limits(self._model_p1)

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
        headroom = DEFAULT_PROCESSING_CONFIG.rate_limit_headroom

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Models: P1={self._model_p1}, P2={self._model_p2}, "
                  f"P3={self._model_p3}, P4={self._model_p4}, P5={self._model_p5}, "
                  f"P6={self._model_p6}, P7={self._model_p7}")
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
    ) -> TaxonomyResult:
        """Taxonomy stages P1-P7: facets, attributes, assignments."""
        start_time = time.time()

        # =================================================================
        # PHASE 1 (P1): Per-domain Facet Discovery (SmoothRequester)
        # + PHASE 2 (P2): Facet Consolidation (per-domain, sequential)
        # =================================================================
        _snap_p1p2 = token_tracker.snapshot() if self.cost_tracker else None

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
            dataset_key=self._dataset_key,
            phase_key="step4_p1_facet_discovery",
            num_tasks=len(p1_tasks),
            verbose=verbose,
            known_limits=self._fetched_limits,
            show_setup=False,
            default_timeout=60.0,
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

        if self._debug_stop_after_phase == 1:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P1 — skipping P2-P7")
            return TaxonomyResult(
                partition_n_labels={},
                partition_n_batches={},
                partition_facets={},
                partition_assignments={},
                partition_attributes={},
                attribute_assignments={},
            )

        # P2 consolidation per domain (SmoothRequester, concurrent)
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
                dataset_key=self._dataset_key,
                phase_key="step4_p2_facet_consolidation",
                num_tasks=len(p2_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                show_setup=False,
                default_timeout=60.0,
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
                    dataset_key=self._dataset_key,
                    phase_key="step4_p2_facet_consolidation",
                    num_tasks=len(r2_tasks),
                    verbose=verbose,
                    known_limits=self._fetched_limits,
                    show_setup=False,
                    default_timeout=60.0,
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

        if self.cost_tracker and _snap_p1p2 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p1_p2_facet_discovery_and_consolidation",
                _snap_p1p2, token_tracker.snapshot(), self._model_p1)

        if self._debug_stop_after_phase == 2:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P2 — skipping P3–P7")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments={},
                partition_attributes={},
                attribute_assignments={},
            )

        # =================================================================
        # PHASE 3 (P3): Per-domain Facet Assignment (SmoothRequester)
        # =================================================================
        _snap_p3 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 2: Facet Assignment")

        t_phase3 = time.time()

        # Build flat task list: one task per (domain, batch)
        # Single-facet domains are auto-assigned without LLM call
        p3_tasks = []
        partition_assignments: Dict[str, Dict[str, str]] = {}
        p3_auto_assigned: Dict[str, int] = {}  # domain → idea count (for reporting)
        batch_size = self._facet_assignment_batch_size
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
                p3_auto_assigned[domain_name] = len(ideas)
                continue

            # Multi-facet: create SR tasks
            facet_id_to_name = {f"F{i}": f.facet_name for i, f in enumerate(facets, 1)}
            idea_batches = [ideas[j:j + batch_size] for j in range(0, len(ideas), batch_size)]
            for batch_idx, batch_ideas in enumerate(idea_batches):
                p3_tasks.append({
                    'domain_name': domain_name,
                    'batch_idx': batch_idx,
                    'total_batches': len(idea_batches),
                    'batch_ideas': batch_ideas,
                    'facets': facets,
                    'facet_id_to_name': facet_id_to_name,
                    'part_context': partition_contexts[domain_name],
                })

        if p3_tasks:
            p3_requester = SmoothRequester(
                model=self._model_p3,
                dataset_key=self._dataset_key,
                phase_key="step4_p3_facet_assignment",
                num_tasks=len(p3_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                show_setup=False,
                quiet=True,
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

            # Reassemble: merge batch-level dicts into per-domain assignments
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
                _snap_p3, token_tracker.snapshot(), self._model_p3)

        if self._debug_stop_after_phase == 3:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P3 — skipping P4–P7")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes={},
                attribute_assignments={},
            )

        # =================================================================
        # PHASE 4 (P4): Per-facet Attribute Discovery (SmoothRequester)
        # + PHASE 5 (P5): Attribute Consolidation (per-facet, sequential)
        # =================================================================
        _snap_p4p5 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 3: Attribute Discovery + Consolidation")

        t_phase4 = time.time()

        # Group ideas by (domain, facet) using P3 assignments
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
                label = format_label(idea, self._label_source, self._label_prefix, self._include_valence)
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

        # P4 discovery via SmoothRequester
        if p4_tasks:
            p4_requester = SmoothRequester(
                model=self._model_p4,
                dataset_key=self._dataset_key,
                phase_key="step4_p4_attribute_discovery",
                num_tasks=len(p4_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                show_setup=False,
                default_timeout=60.0,
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
            print(f"    P4 discovery: {len(p4_tasks)} tasks, {t_p4_discovery:.1f}s "
                  f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries)")

        if self._debug_stop_after_phase == 4:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P4 — skipping P5–P7")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes={},
                attribute_assignments={},
            )

        # P5 consolidation per facet (SmoothRequester, concurrent)
        t_consolidation = time.time()
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}

        # Build P5 task list
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
                model=self._model_p5,
                dataset_key=self._dataset_key,
                phase_key="step4_p5_attribute_consolidation",
                num_tasks=len(p5_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                show_setup=False,
                default_timeout=60.0,
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
                    model=self._model_p5,
                    dataset_key=self._dataset_key,
                    phase_key="step4_p5_attribute_consolidation",
                    num_tasks=len(r2_tasks),
                    verbose=verbose,
                    known_limits=self._fetched_limits,
                    show_setup=False,
                    default_timeout=60.0,
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
            print(f"    P5 consolidation: {len(p5_tasks)} tasks, {t_consolidation:.1f}s "
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

        if self.cost_tracker and _snap_p4p5 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p4_p5_attribute_discovery_and_consolidation",
                _snap_p4p5, token_tracker.snapshot(), self._model_p4)

        if self._debug_stop_after_phase == 5:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P5 — skipping P6–P7")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes=partition_attributes,
                attribute_assignments={},
            )

        # =================================================================
        # PHASE 6 (P6): Per-facet Attribute Assignment (SmoothRequester)
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
        batch_size = self._facet_assignment_batch_size
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
                    p6_auto_assigned[facet_key] = len(facet_ideas)
                    continue

                # Multi-attribute: create SR tasks
                facet_obj = None
                for f in partition_facets.get(domain_name, []):
                    if f.facet_name == facet_name:
                        facet_obj = f
                        break
                if not facet_obj:
                    continue

                attr_id_to_name = {f"A{i}": a.attribute_name for i, a in enumerate(attributes, 1)}
                idea_batches = [facet_ideas[j:j + batch_size] for j in range(0, len(facet_ideas), batch_size)]

                for batch_idx, batch_ideas in enumerate(idea_batches):
                    p6_tasks.append({
                        'domain_name': domain_name,
                        'facet_name': facet_name,
                        'facet_description': facet_obj.facet_description,
                        'batch_idx': batch_idx,
                        'total_batches': len(idea_batches),
                        'batch_ideas': batch_ideas,
                        'attributes': attributes,
                        'attr_id_to_name': attr_id_to_name,
                        'facet_key': facet_key,
                    })

        if p6_tasks:
            p6_requester = SmoothRequester(
                model=self._model_p6,
                dataset_key=self._dataset_key,
                phase_key="step4_p6_attribute_assignment",
                num_tasks=len(p6_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                show_setup=False,
                quiet=True,
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

            # Reassemble: merge batch-level dicts
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
                _snap_p6, token_tracker.snapshot(), self._model_p6)

        if self._debug_stop_after_phase == 6:
            if verbose:
                print(f"\n  [DEBUG] Early stop after P6 — skipping P7")
            return TaxonomyResult(
                partition_n_labels=partition_n_labels,
                partition_n_batches=partition_n_batches,
                partition_facets=partition_facets,
                partition_assignments=partition_assignments,
                partition_attributes=partition_attributes,
                attribute_assignments=attribute_assignments,
            )

        # =================================================================
        # PHASE 7 (P7): Cross-facet Attribute Consolidation per domain
        # (now with frequency data from attribute assignments)
        # =================================================================
        _snap_p7 = token_tracker.snapshot() if self.cost_tracker else None

        if verbose:
            print(f"\n  Phase 5: Cross-facet Attribute Consolidation")

        t_phase7 = time.time()

        # Build P7 task list — one per domain
        p7_tasks = []
        p7_domain_names = []
        for domain_name, facet_attrs in domain_facet_attributes.items():
            if not facet_attrs:
                continue
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
            p7_tasks.append({
                'domain_name': domain_name,
                'facet_attributes': facet_attrs,
                'domain_facets': partition_facets.get(domain_name, []),
                'part_context': partition_contexts[domain_name],
                'domain_attr_assigns': domain_attr_assigns,
                'excluded_domains': excluded,
            })
            p7_domain_names.append(domain_name)

        if p7_tasks:
            p7_requester = SmoothRequester(
                model=self._model_p7,
                dataset_key=self._dataset_key,
                phase_key="step4_p7_attribute_consolidation",
                num_tasks=len(p7_tasks),
                verbose=verbose,
                known_limits=self._fetched_limits,
                show_setup=False,
                default_timeout=60.0,
                quiet=True,
            )
            p7_results = await p7_requester.process_all(
                p7_tasks,
                self._p7_prepare_fn(prompt_context),
                self._p7_parse_fn(),
                self._p7_fallback_fn(),
            )

            if verbose:
                s = p7_requester.stats
                t_sr = s.get('wall_time', 0)
                print(f"    P7 consolidation: {len(p7_tasks)} tasks, {t_sr:.1f}s "
                      f"({s['tasks_successful']} ok, {s.get('timeouts', 0)} timeouts, "
                      f"{s.get('recovered', 0)} retries)")

            _p7_domain_results = {}
            for task, result in zip(p7_tasks, p7_results):
                domain_name = task['domain_name']
                if not result:
                    print(f"    WARNING: P7 '{domain_name}' returned empty")
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
                    _p7_domain_results[domain_name] = (before_count, after_count, len(new_facet_attrs), remap_msg)

        t_phase7 = time.time() - t_phase7
        if verbose:
            total_attrs_after = sum(
                len(attrs)
                for facet_attrs in domain_facet_attributes.values()
                for attrs in facet_attrs.values()
            )
            print(f"    Results ({t_phase7:.1f}s → {total_attrs_after} consolidated attributes):")
            for domain_name, (before, after, n_facets, remap_msg) in sorted(_p7_domain_results.items()):
                print(f"      {domain_name}: {before} → {after} attributes ({n_facets} facets{remap_msg})")

        taxonomy_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Taxonomy complete in {taxonomy_elapsed:.1f}s")

        if self.cost_tracker and _snap_p7 is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", "p7_cross_facet_consolidation",
                _snap_p7, token_tracker.snapshot(), self._model_p7)

        return TaxonomyResult(
            partition_n_labels=partition_n_labels,
            partition_n_batches=partition_n_batches,
            partition_facets=partition_facets,
            partition_assignments=partition_assignments,
            partition_attributes=partition_attributes,
            attribute_assignments=attribute_assignments,
        )

    # =========================================================================
    # PHASE 3 (P3): PER-DOMAIN FACET ASSIGNMENT (SmoothRequester)
    # =========================================================================

    def _p3_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P3 facet assignment."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_facet_assignment_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facets=task['facets'],
                other_label=None,
                ideas=task['batch_ideas'],
            )

            # Prompt capture (first batch per domain)
            gate_key = f"qr_facet_assign_{task['domain_name']}"
            if (self._prompt_printer is not None
                    and task['batch_idx'] == 0
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
                        "partition_name": task['domain_name'],
                        "batch_number": task['batch_idx'] + 1,
                        "total_batches": task['total_batches'],
                        "n_facets": len(task['facets']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': FacetAssignmentBatch,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_facet_assignment,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p3),
            }
        return prepare_fn

    def _p3_parse_fn(self):
        """Return parse_fn closure for P3 facet assignment."""
        def parse_fn(task: Dict, response) -> Optional[Dict[str, str]]:
            original_lookup = {idea.idea_id: idea for idea in task['batch_ideas']}
            facet_id_to_name = task['facet_id_to_name']
            assignments: Dict[str, str] = {}

            for assignment in response.assignments:
                # Validate returned idea_id exists in original batch
                original_idea = original_lookup.get(assignment.idea_id)
                if original_idea is None:
                    print(f"    ID DRIFT: LLM returned unexpected idea_id "
                          f"'{assignment.idea_id}' in batch {task['batch_idx']} — skipping")
                    continue

                # Content cross-validation
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

                # Map facet_id to name
                facet_name = facet_id_to_name.get(assignment.assigned_facet_id)

                # Detect duplicate assignments
                if original_idea.idea_id in assignments:
                    print(f"    WARNING: Duplicate assignment for '{original_idea.idea_id}' — "
                          f"overwriting '{assignments[original_idea.idea_id]}' with '{facet_name}'")

                assignments[original_idea.idea_id] = facet_name

            return assignments
        return parse_fn

    @staticmethod
    def _p3_fallback_fn():
        """Return fallback_fn closure for P3 facet assignment."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            return {}
        return fallback_fn

    # =========================================================================
    # PHASE 6 (P6): PER-FACET ATTRIBUTE ASSIGNMENT (SmoothRequester)
    # =========================================================================

    def _p6_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P6 attribute assignment."""
        def prepare_fn(task: Dict) -> Dict:
            prompt = build_attribute_assignment_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                facet_name=task['facet_name'],
                facet_description=task['facet_description'],
                attributes=task['attributes'],
                ideas=task['batch_ideas'],
            )

            # Prompt capture (first batch per facet)
            gate_key = f"qr_attr_assign_{task['domain_name']}_{task['facet_name']}"
            if (self._prompt_printer is not None
                    and task['batch_idx'] == 0
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
                        "partition_name": task['domain_name'],
                        "facet_name": task['facet_name'],
                        "batch_number": task['batch_idx'] + 1,
                        "total_batches": task['total_batches'],
                        "n_attributes": len(task['attributes']),
                        "dimension_name": prompt_context.dimension_name,
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': AttributeAssignmentBatch,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_facet_assignment,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p6),
            }
        return prepare_fn

    def _p6_parse_fn(self):
        """Return parse_fn closure for P6 attribute assignment."""
        def parse_fn(task: Dict, response) -> Optional[Dict[str, str]]:
            original_lookup = {idea.idea_id: idea for idea in task['batch_ideas']}
            attr_id_to_name = task['attr_id_to_name']
            assignments: Dict[str, str] = {}

            for assignment in response.assignments:
                # Validate returned idea_id exists in original batch
                original_idea = original_lookup.get(assignment.idea_id)
                if original_idea is None:
                    print(f"    ID DRIFT: LLM returned unexpected idea_id "
                          f"'{assignment.idea_id}' in attr batch {task['batch_idx']} — skipping")
                    continue

                # Content cross-validation
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

                # Map attribute_id to name — single-attribute fallback
                attr_name = attr_id_to_name.get(assignment.assigned_attribute_id)
                if attr_name is None:
                    if len(attr_id_to_name) == 1:
                        attr_name = next(iter(attr_id_to_name.values()))
                    else:
                        print(f"    WARNING: Invalid attribute_id '{assignment.assigned_attribute_id}' "
                              f"for idea '{original_idea.idea_id}' — skipping")
                        continue

                # Detect duplicate assignments
                if original_idea.idea_id in assignments:
                    print(f"    WARNING: Duplicate attr assignment for '{original_idea.idea_id}' — "
                          f"overwriting '{assignments[original_idea.idea_id]}' with '{attr_name}'")

                assignments[original_idea.idea_id] = attr_name

            return assignments
        return parse_fn

    @staticmethod
    def _p6_fallback_fn():
        """Return fallback_fn closure for P6 attribute assignment."""
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
                'extra_kwargs': get_reasoning_params(self._model_p1),
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
                        "max_tokens": self._max_tokens_facet_discovery,
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
                'max_tokens': self._max_tokens_facet_discovery,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p2),
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
    # PHASE 4 (P4): PER-FACET ATTRIBUTE DISCOVERY (SmoothRequester)
    # =========================================================================

    def _p4_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P4 attribute discovery."""
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
                        "model": self._model_p4,
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
                'extra_kwargs': get_reasoning_params(self._model_p4),
            }
        return prepare_fn

    def _p4_parse_fn(self):
        """Return parse_fn closure for P4 attribute discovery."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredAttribute]]:
            return response.attributes if response else []
        return parse_fn

    @staticmethod
    def _p4_fallback_fn():
        """Return fallback_fn closure for P4 attribute discovery."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredAttribute]:
            return []
        return fallback_fn

    # =========================================================================
    # PHASE 5 (P5): ATTRIBUTE CONSOLIDATION (SmoothRequester)
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
        """Return prepare_fn closure for P5 attribute consolidation."""
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
                        "model": self._model_p5,
                        "temperature": 0.0,
                        "max_tokens": self._max_tokens_attribute_discovery,
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
                'max_tokens': self._max_tokens_attribute_discovery,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p5),
            }
        return prepare_fn

    def _p5_parse_fn(self):
        """Return parse_fn closure for P5 attribute consolidation."""
        def parse_fn(task: Dict, response) -> Optional[List[DiscoveredAttribute]]:
            return response.attributes if response else []
        return parse_fn

    @staticmethod
    def _p5_fallback_fn():
        """Return fallback_fn closure for P5 attribute consolidation."""
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredAttribute]:
            return []
        return fallback_fn

    # =========================================================================
    # PHASE 7 (P7): CROSS-FACET ATTRIBUTE CONSOLIDATION (SmoothRequester)
    # =========================================================================

    def _p7_prepare_fn(self, prompt_context: PromptContext):
        """Return prepare_fn closure for P7 cross-facet attribute consolidation."""
        def prepare_fn(task: Dict) -> Dict:
            # Compute attribute frequencies from assignments
            attr_counts: Dict[str, int] = {}
            if task.get('domain_attr_assigns'):
                for attr_name in task['domain_attr_assigns'].values():
                    attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

            # Format facet→attributes block
            lines = []
            for facet_name, attributes in sorted(task['facet_attributes'].items()):
                facet_desc = ""
                for f in task['domain_facets']:
                    if f.facet_name == facet_name:
                        facet_desc = f.facet_description
                        break
                lines.append(f'Facet: "{facet_name}" — {facet_desc}')
                for attr in attributes:
                    examples = "; ".join(attr.example_observations[:2])
                    count = attr_counts.get(attr.attribute_name, 0)
                    freq_tag = f" ({count} ideas)" if task.get('domain_attr_assigns') else ""
                    lines.append(
                        f'  - "{attr.attribute_name}"{freq_tag} — '
                        f'{attr.attribute_description} '
                        f'(examples: {examples})'
                    )
                lines.append("")

            prompt = build_attribute_consolidation_prompt(
                survey_question=prompt_context.survey_question,
                language=prompt_context.language,
                dataset_context_section=prompt_context.dataset_context_section,
                dimension_def=prompt_context.dimension_def,
                dimension_name=prompt_context.dimension_name,
                dimension_description=prompt_context.dimension_description,
                domain_name=task['domain_name'],
                domain_definition=task['part_context'].partition_definition,
                facet_attributes_block="\n".join(lines),
                excluded_domains=task['excluded_domains'],
            )

            # Prompt capture
            gate_key = f"qr_attribute_consolidation_{task['domain_name']}"
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
                        "domain_name": task['domain_name'],
                        "n_facets": len(task['facet_attributes']),
                        "n_attributes_before": sum(
                            len(a) for a in task['facet_attributes'].values()
                        ),
                    }
                )
                self._captured_gates.add(gate_key)

            return {
                'prompt': prompt,
                'response_model': AttributeConsolidatedResponse,
                'temperature': self._temperature,
                'max_tokens': self._max_tokens_attribute_discovery,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(self._model_p7),
            }
        return prepare_fn

    def _p7_parse_fn(self):
        """Return parse_fn closure for P7 cross-facet attribute consolidation."""
        def parse_fn(task: Dict, response) -> Optional[List[ConsolidatedAttribute]]:
            return response.attributes if response else []
        return parse_fn

    @staticmethod
    def _p7_fallback_fn():
        """Return fallback_fn closure for P7 cross-facet attribute consolidation."""
        def fallback_fn(task: Dict, reason: str) -> List[ConsolidatedAttribute]:
            return []
        return fallback_fn

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
