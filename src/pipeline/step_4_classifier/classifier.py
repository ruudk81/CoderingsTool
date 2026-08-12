"""
Taxonomy Classifier: inductive taxonomy discovery, nine phases.

Each level is built the way step 3 builds the domain layer — discover, settle
the inventory, assign, then judge the result against what the buckets really
hold:

  facet_discovery          per (domain, chunk)   propose facets
  facet_consolidation      per domain            settle the inventory
  facet_assignment         per batch of ideas    fill it, with valence
  facet_refinement         per domain            judge it on real contents
  attribute_discovery      per (facet, chunk)    the same four, one level down
  attribute_consolidation  per facet
  attribute_assignment     per batch of ideas
  attribute_refinement     per facet
  valence_merge            see valence_consolidator.py

Phases are named by function, never by number: renumbering cold-started the
perf model and stranded config keys, twice.

There is no axis anywhere in step 4. A dimension organises each of its layers
along one axis, and that axis is what the layer's diagnostic question asks
about — a constant of the dimension, stated in dimension_data.py, not something
a phase discovers.

Two invariants hold across every phase:

  * The scope one level up is FIXED and absent from the response schema. A facet
    cannot be moved to another domain, an attribute cannot be moved to another
    facet. When a group of ideas belongs elsewhere, the IDEAS move (`misfits`)
    and the structure stays put. Per-idea (domain, facet) is DERIVED from where
    the attribute lives, so a structural relocation would drag every idea in the
    bucket along at once.
  * Consolidation runs BEFORE a single idea is assigned, on the observations
    each candidate was built from; refinement runs AFTER, on real counts and
    real response texts. They answer different questions and both are needed.

Tasks within a phase run CONCURRENTLY through SmoothRequester.

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
    attribute_card_text, facet_card_text, group_label_reps, make_batches,
    shortlist_indices, validate_batch_response,
)
from models import DomainSet
from .prompts_facet import (
    DiscoveredFacet, ConsolidatedFacet, RefinedFacet,
    FacetDiscoveryResult, FacetConsolidationResult, FacetRefinementResult,
    build_facet_discovery_prompt,
    build_facet_consolidation_prompt,
    build_facet_menu,
    build_facet_assignment_model,
    build_facet_assignment_prompt,
    build_facet_contents_block,
    build_facet_refinement_prompt,
)
from .prompts_attribute import (
    DiscoveredAttribute, ConsolidatedAttribute, RefinedAttribute,
    AttributeDiscoveryResult, AttributeConsolidationResult,
    AttributeRefinementResult,
    build_attribute_discovery_prompt,
    build_attribute_consolidation_prompt,
    build_attribute_menu,
    build_attribute_assignment_model,
    build_attribute_assignment_prompt,
    build_attribute_contents_block,
    build_attribute_refinement_prompt,
    build_neighbour_block,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


# =============================================================================
# HELPERS
# =============================================================================

def _escalated(task: Dict, rep, reason: str) -> Dict:
    """One rep on its own against the full menu — the escalation task shape.

    Every key of the batch task it came from is kept, so the same prepare/parse
    pair serves both passes and the two cannot drift apart.
    """
    return {**task, "reps": [rep], "menu": task["full_menu"], "reason": reason}


# =============================================================================
# SHARED DATACLASSES
# =============================================================================

@dataclass
class PromptContext:
    """Everything a phase needs to build its prompts and scope its tasks.

    Carries the domains rather than looking them up on the classifier, so every
    `_build_<phase>_tasks` can be a pure function of its arguments — which is
    what makes the task shape (scope, skips, chunking, counts) testable without
    an LLM call.

    `domains` maps a domain label to its four boundary fields, exactly as step 3
    wrote them: label / definition / boundary_test / exclusions. `drain_labels`
    holds the two standing drain domains, which get no facets at all.
    """
    language: str
    survey_question: str
    sector: str = ""
    entity: str = ""
    topic: str = ""
    perspective: str = ""
    intent: str = ""
    dimension: Optional[DimensionDefinition] = None
    dimension_name: str = ""
    dimension_description: str = ""
    domains: Dict[str, Dict] = field(default_factory=dict)
    drain_labels: Set[str] = field(default_factory=set)

    def specifiers(self) -> Dict[str, str]:
        """The five context specifiers, as every prompt builder takes them."""
        return {
            "sector": self.sector, "entity": self.entity, "topic": self.topic,
            "perspective": self.perspective, "intent": self.intent,
        }

    def domain(self, label: str) -> Dict:
        """One domain's boundary fields, empty-but-present when unknown."""
        return self.domains.get(label) or {
            "label": label, "definition": "", "boundary_test": "", "exclusions": [],
        }


@dataclass
class TaxonomyResult:
    """Output of the eight in-classifier phases (valence merge runs after)."""
    partition_n_labels: Dict[str, int]
    partition_n_batches: Dict[str, int]
    partition_facets: Dict[str, List[ConsolidatedFacet]]
    partition_assignments: Dict[str, Dict[str, str]]  # domain -> {idea_id -> facet_name}
    partition_attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]]  # domain -> {facet -> [attrs]}
    attribute_assignments: Dict[str, str]  # idea_id -> attribute_name
    # Discovery snapshots, taken before each level's consolidation settles it.
    # The state before a merge is what makes a bad merge diagnosable afterwards.
    partition_raw_facets: Dict[str, List[DiscoveredFacet]] = field(default_factory=dict)
    raw_partition_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]] = field(default_factory=dict)
    raw_attribute_assignments: Dict[str, str] = field(default_factory=dict)
    # Assignment confidence scores (0.0-1.0)
    facet_confidence: Dict[str, float] = field(default_factory=dict)
    attribute_confidence: Dict[str, float] = field(default_factory=dict)
    # Assignment valence (+, -, 0)
    facet_valence: Dict[str, str] = field(default_factory=dict)
    attribute_valence: Dict[str, str] = field(default_factory=dict)
    # One entry per action taken, so every merge/split/move is auditable after
    # the fact. Written to a JSON file by the runner; deliberately NOT put in
    # the shared cache model, which production also uses.
    consolidation_log: List[Dict] = field(default_factory=list)


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class TaxonomyClassifier:
    """Builds the facet (L3) and attribute (L4) layers of the taxonomy.

    Eight phases run here, four per level, in the same order step 3 uses for the
    domain layer: discovery → consolidation → assignment → refinement. The ninth
    phase, the valence-neutral merge, runs afterwards from the runner (see
    `valence_consolidator.py`).

    Every phase is one method with an explicit signature, plus a pure
    `_build_<phase>_tasks` that decides the task shape. The orchestrator is the
    sequence of those nine calls and nothing else.
    """

    # Phase key → (config attribute, cost-tracker label). One table, so a phase
    # cannot end up with a model in one register and a different one in another.
    PHASES = (
        "facet_discovery", "facet_consolidation", "facet_assignment",
        "facet_refinement", "attribute_discovery", "attribute_consolidation",
        "attribute_assignment", "attribute_refinement", "valence_merge",
    )

    def __init__(self, config: CategoriesConfig, prompt_printer=None, cost_tracker=None):
        self.cost_tracker = cost_tracker
        self._config = config

        self._model: Dict[str, str] = {
            phase: getattr(config, f"model_{phase}") for phase in self.PHASES
        }

        if self.cost_tracker:
            self.cost_tracker.set_step_models(
                "step_4_taxonomy_classifier", dict(self._model))

        self._temperature = config.qr_temperature
        self._max_tokens_facet_discovery = config.qr_max_tokens_facet_discovery
        self._max_tokens_attribute_discovery = config.qr_max_tokens_attribute_discovery
        self._max_tokens_consolidation = config.qr_max_tokens_consolidation
        self._max_tokens_assignment = config.qr_max_tokens_assignment
        self._contents_top_n = config.contents_top_n

        # Chunking — facet discovery input, per domain
        self._batch_size_min = config.batch_size_min
        self._batch_size_max = config.batch_size_max
        self._target_batches = config.target_batches
        self._chunk_overlap = config.chunk_overlap

        # Chunking — attribute discovery input, per facet
        self._attribute_chunk_size_min = config.attribute_chunk_size_min
        self._attribute_chunk_size_max = config.attribute_chunk_size_max
        self._attribute_target_batches = config.attribute_target_batches
        self._attribute_chunk_overlap = config.attribute_chunk_overlap

        # Consolidation — how much fits in one call, and how often to round-trip
        self._consolidation_max_chunks_per_call = config.consolidation_max_chunks_per_call
        self._consolidation_max_items_per_call = config.consolidation_max_items_per_call
        self._consolidation_max_rounds = config.consolidation_max_rounds

        # Label source for observation formatting
        self._label_source = config.label_source
        self._label_prefix = config.label_prefix

        # Assignment — batching and menu shortlisting, both levels
        self._assign_batch_k = config.assignment_batch_k
        self._assign_shortlist_enabled = config.assignment_shortlist_enabled
        self._assign_shortlist_k = config.assignment_shortlist_k

        # Prompt capture (optional)
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()

        self._stop_after_phase = config.stop_after_phase
        if (self._stop_after_phase is not None
                and self._stop_after_phase not in self.PHASES):
            raise ValueError(
                f"stop_after_phase={self._stop_after_phase!r} is not a phase. "
                f"Valid: {', '.join(self.PHASES)}."
            )

        # One entry per action a phase takes, in run order. Lives on the
        # instance rather than being threaded through every signature: the
        # orchestrator reads as the nine phase calls it is, and the task
        # builders stay pure because only the run methods touch this.
        self._action_log: List[Dict] = []
        self._last_stats: Dict = {}

        # Assignment confidence scores and valence (populated by the two
        # assignment phases' parse_fns)
        self._facet_confidence: Dict[str, float] = {}
        self._attribute_confidence: Dict[str, float] = {}
        self._facet_valence: Dict[str, str] = {}
        self._attribute_valence: Dict[str, str] = {}

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
        extraction_metadata=None,
        verbose: bool = False,
    ) -> Tuple[PromptContext, Dict[str, PartitionLabelMapping]]:
        """Shared setup: resolve the dimension, collect the domains, drop empties.

        A dimension that cannot be resolved raises. Every prompt in step 4 asks
        this dimension's own diagnostic question and shows its own four-level
        taxonomy; without it there is no generic version to fall back on, and a
        phase built around the wrong question produces plausible output that is
        wrong all the way down.
        """
        dimension_def = get_dimension(dimension_name) if dimension_name else None
        if dimension_def is None:
            raise ValueError(
                f"No DimensionDefinition for primary_dimension "
                f"{dimension_name!r}. Step 4 builds every prompt around this "
                f"dimension's diagnostics; see dimension_data.py."
            )
        if verbose:
            print(f"  Dimension: {dimension_name}")
            print(f"  Facet diagnostic: {dimension_def.prompt_rules.facet_diagnostic}")
            print(f"  Attribute diagnostic: {dimension_def.prompt_rules.attribute_diagnostic}")

        active_partitions = {
            name: mapping for name, mapping in label_mappings.items()
            if mapping.labels
        }

        context = dataset_context or {}
        prompt_context = PromptContext(
            language=language,
            survey_question=survey_question,
            sector=context.get("sector", ""),
            entity=context.get("entity", ""),
            topic=context.get("topic", ""),
            perspective=context.get("perspective", ""),
            intent=context.get("intent", ""),
            dimension=dimension_def,
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            domains={
                part.partition_name: {
                    "label": part.partition_name,
                    "definition": part.inclusion_definition,
                    "boundary_test": part.boundary_test,
                    "exclusions": part.exclusions,
                    # Discovery input travels with the domain, so the task
                    # builder needs nothing but the context to decide its shape.
                    "observations": list(
                        active_partitions[part.partition_name].labels),
                }
                for part in partition_set.partitions
                if part.partition_name in active_partitions
            },
            drain_labels=drain_domains(extraction_metadata),
        )

        return prompt_context, active_partitions

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
        """Run the eight in-classifier phases: facets, attributes, assignments.

        `extraction_metadata` (models.ExtractionMetadata) identifies the two
        standing drain domains by key (taxonomy_health.drain_domains); those get
        no facets, because step 3 defines them as deliberately broad catch-alls.
        """
        print(f"\n{'='*70}")
        print(f"TAXONOMY DISCOVERY (8 phases)")
        print(f"{'='*70}")

        prompt_context, active_partitions = self._prepare_context(
            label_mappings, partition_set, survey_question, language,
            dataset_context, dimension_name, dimension_description,
            extraction_metadata, verbose,
        )

        if verbose:
            total_labels = sum(m.label_count for m in active_partitions.values())
            total_ideas = sum(len(m.ideas) for m in active_partitions.values())
            n_partitions = len(active_partitions)
            print(f"  Processing {n_partitions} domains concurrently "
                  f"({total_labels} observations, {total_ideas} ideas)")
            print(f"  Per level: discovery → consolidation → assignment → refinement")

        async def _run():
            await self._initialize_async_resources(verbose)
            return await self._process_taxonomy_async(
                active_partitions, prompt_context, verbose)

        return asyncio.run(_run())

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _initialize_async_resources(self, verbose: bool):
        """Fetch rate limits from API — one probe per unique deployment.

        Quota is per deployment, so each phase must read the limits of the
        deployment its own model resolves to — not whatever the first phase
        happens to run on. Probes for distinct deployments run in parallel, so
        wall-clock cost stays that of a single probe.
        """
        phase_models = [self._model[p] for p in self.PHASES if p != "valence_merge"]

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
            print("  Models: " + ", ".join(
                f"{phase}={self._model[phase]}" for phase in self.PHASES))
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
        prompt_context: PromptContext,
        verbose: bool,
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
                label_vectors = await embedder.embed_texts([rep.label or " " for rep in reps])
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
        elif verbose and p3_auto_assigned and not p3_batch_tasks:
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
    # SHARED PHASE PLUMBING
    # =========================================================================

    async def _dispatch(
        self, phase: str, tasks: List[Dict], prepare_fn, parse_fn, fallback_fn,
        verbose: bool, *, quiet: bool = True,
    ) -> List:
        """Run one phase's tasks through SmoothRequester and record its cost.

        One place where a phase meets the requester, so a phase cannot end up
        dispatching on a different model than the one its cost is booked
        against, and the phase key cannot drift from the phase name.
        """
        self._last_stats = {}
        if not tasks:
            return []
        model = self._model[phase]
        snapshot = token_tracker.snapshot() if self.cost_tracker else None
        requester = SmoothRequester(
            model=model,
            phase_key=f"step4_{phase}",
            num_tasks=len(tasks),
            verbose=verbose,
            known_limits=self._limits_by_model[model],
            has_server_headers=self._has_headers_by_model[model],
            show_setup=False,
            quiet=quiet,
        )
        results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
        self._last_stats = requester.stats
        if self.cost_tracker and snapshot is not None:
            self.cost_tracker.record_phase(
                "step_4_taxonomy_classifier", phase, snapshot,
                token_tracker.snapshot(), model)
        return results

    def _capture(self, gate_key: str, prompt: str, prompt_type: str,
                 metadata: Dict) -> None:
        """Capture the first prompt of its kind, once per gate."""
        if self._prompt_printer is None or gate_key in self._captured_gates:
            return
        self._prompt_printer.capture_prompt(
            step_name="qualitative_researcher",
            utility_name="QualitativeResearcher",
            prompt_content=prompt,
            prompt_type=prompt_type,
            metadata=metadata,
        )
        self._captured_gates.add(gate_key)

    def _consolidation_groups(self, items: List) -> List[List]:
        """Split a candidate list into groups that each fit in one call."""
        cap = self._consolidation_max_items_per_call
        if len(items) <= cap:
            return [list(items)]
        return [list(items[i:i + cap]) for i in range(0, len(items), cap)]

    # =========================================================================
    # PHASE — FACET DISCOVERY (per domain, chunked)
    # =========================================================================

    def _build_facet_discovery_tasks(self, ctx: PromptContext) -> List[Dict]:
        """One task per (domain, chunk).

        The two standing drain domains are skipped. Step 3 defines them as
        deliberately broad catch-alls; imposing structure on a catch-all invents
        distinctions the responses do not carry.
        """
        tasks: List[Dict] = []
        for label in sorted(ctx.domains):
            if label in ctx.drain_labels:
                continue
            observations = ctx.domain(label).get("observations") or []
            chunks = self._create_batches(observations)
            for chunk_idx, chunk in enumerate(chunks):
                tasks.append({
                    "domain_label": label,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                    "observations": chunk,
                })
        return tasks

    def _facet_discovery_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            prompt = build_facet_discovery_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                dimension_name=ctx.dimension_name,
                dimension_description=ctx.dimension_description,
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                domain_boundary_test=domain["boundary_test"],
                domain_exclusions=domain["exclusions"],
                observations=task["observations"],
            )
            if task["chunk_idx"] == 0:
                self._capture(
                    f"facet_discovery_{task['domain_label']}", prompt,
                    "facet_discovery",
                    {"model": self._model["facet_discovery"],
                     "temperature": self._temperature,
                     "max_tokens": self._max_tokens_facet_discovery,
                     "language": ctx.language,
                     "domain": task["domain_label"],
                     "total_chunks": task["total_chunks"],
                     "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": FacetDiscoveryResult,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens_facet_discovery,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["facet_discovery"],
                    phase="classifier_facet_discovery"),
            }
        return prepare_fn

    @staticmethod
    def _facet_discovery_parse_fn():
        def parse_fn(task: Dict, response) -> List[DiscoveredFacet]:
            return list(response.facets) if response else []
        return parse_fn

    @staticmethod
    def _facet_discovery_fallback_fn():
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredFacet]:
            return []
        return fallback_fn

    async def _run_facet_discovery(
        self, ctx: PromptContext, verbose: bool,
    ) -> Dict[str, List[DiscoveredFacet]]:
        """Propose facets per domain, then collapse byte-identical re-proposals.

        Every chunk rediscovers largely the same structure, so the flattened
        yield holds exact repeats. Only those are removed here; near-duplicates
        are a judgment and belong to consolidation, which sees each candidate
        together with the observations that produced it.
        """
        if verbose:
            print(f"\n  Facet discovery")
        started = time.time()

        tasks = self._build_facet_discovery_tasks(ctx)
        results = await self._dispatch(
            "facet_discovery", tasks,
            self._facet_discovery_prepare_fn(ctx),
            self._facet_discovery_parse_fn(),
            self._facet_discovery_fallback_fn(),
            verbose,
        )

        flat: Dict[str, List[DiscoveredFacet]] = {}
        for task, result in zip(tasks, results):
            flat.setdefault(task["domain_label"], []).extend(result or [])

        raw: Dict[str, List[DiscoveredFacet]] = {}
        for label in sorted(flat):
            deduped = dedup_exact_facets(flat[label])
            if len(deduped) < len(flat[label]):
                self._action_log.append({
                    "action": "facet_exact_dedup", "domain": label,
                    "before": len(flat[label]), "after": len(deduped)})
            raw[label] = deduped

        if verbose:
            s = self._last_stats
            print(f"    {len(tasks)} tasks, {time.time() - started:.1f}s "
                  f"({s.get('tasks_successful', 0)} ok, "
                  f"{s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries) → "
                  f"{sum(len(f) for f in raw.values())} candidate facets")
            for label in sorted(raw):
                print(f"      {label}: {len(raw[label])}")
        return raw

    # =========================================================================
    # PHASE — FACET CONSOLIDATION (per domain, before any idea is assigned)
    # =========================================================================

    def _build_facet_consolidation_tasks(
        self, ctx: PromptContext, raw: Dict[str, List[DiscoveredFacet]],
    ) -> List[Dict]:
        """One task per domain that has candidates, split when it holds too many.

        A domain whose candidates do not fit one call is consolidated in rounds:
        the groups here are the first round, and their survivors go back in
        together so the groups still get to see each other.
        """
        tasks: List[Dict] = []
        for label in sorted(raw):
            candidates = raw[label]
            if not candidates:
                continue
            for group in self._consolidation_groups(candidates):
                tasks.append({"domain_label": label, "candidates": group})
        return tasks

    def _facet_consolidation_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            prompt = build_facet_consolidation_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                dimension_name=ctx.dimension_name,
                dimension_description=ctx.dimension_description,
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                domain_boundary_test=domain["boundary_test"],
                candidates=task["candidates"],
            )
            self._capture(
                f"facet_consolidation_{task['domain_label']}", prompt,
                "facet_consolidation",
                {"model": self._model["facet_consolidation"],
                 "temperature": 0.0,
                 "max_tokens": self._max_tokens_consolidation,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "n_candidates": len(task["candidates"]),
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": FacetConsolidationResult,
                "temperature": 0.0,
                "max_tokens": self._max_tokens_consolidation,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["facet_consolidation"],
                    phase="classifier_facet_consolidation"),
            }
        return prepare_fn

    @staticmethod
    def _facet_consolidation_parse_fn():
        def parse_fn(task: Dict, response) -> Optional[FacetConsolidationResult]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _facet_consolidation_fallback_fn():
        """On failure the domain keeps its candidates — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    @staticmethod
    def _as_consolidated_facet(facet) -> ConsolidatedFacet:
        """A candidate that survives untouched, in the settled type."""
        if isinstance(facet, ConsolidatedFacet):
            return facet
        return ConsolidatedFacet(**facet.model_dump(),
                                 source_facets=[facet.facet_name])

    def _facet_consolidation_survivors(
        self, task: Dict, result,
    ) -> List[ConsolidatedFacet]:
        """The settled facets of one group, with unclaimed candidates kept.

        A candidate that appears in no `source_facets` is left standing. Dropping
        it silently would remove a facet the model never judged — a fail-safe,
        logged so under-claiming is visible rather than invisible.
        """
        label, candidates = task["domain_label"], task["candidates"]
        if result is None or not result.facets:
            self._action_log.append({
                "action": "facet_consolidation_failed", "domain": label,
                "note": "no result — candidates left as discovered",
                "candidates": [c.facet_name for c in candidates]})
            return [self._as_consolidated_facet(c) for c in candidates]

        survivors: List[ConsolidatedFacet] = list(result.facets)
        claimed = {self._norm_text(s)
                   for f in survivors for s in (f.source_facets or [])}
        returned = {self._norm_text(f.facet_name) for f in survivors}
        for candidate in candidates:
            key = self._norm_text(candidate.facet_name)
            if key in claimed or key in returned:
                continue
            survivors.append(self._as_consolidated_facet(candidate))
            self._action_log.append({
                "action": "facet_kept_unclaimed", "domain": label,
                "facet": candidate.facet_name})

        self._action_log.append({
            "action": "facet_consolidation", "domain": label,
            "before": len(candidates), "after": len(survivors)})
        return survivors

    async def _run_facet_consolidation(
        self, ctx: PromptContext, raw: Dict[str, List[DiscoveredFacet]],
        verbose: bool,
    ) -> Dict[str, List[ConsolidatedFacet]]:
        """Settle each domain's facet inventory before a single idea is assigned.

        Rounds, not one shot: a domain with more candidates than fit in a call is
        consolidated per group, and the survivors go back in together. A domain
        that already fits in one call is settled after one round.
        """
        if verbose:
            print(f"\n  Facet consolidation")
        started = time.time()

        settled: Dict[str, List[ConsolidatedFacet]] = {}
        pending: Dict[str, List[DiscoveredFacet]] = {
            label: list(facets) for label, facets in raw.items() if facets
        }

        for _ in range(self._consolidation_max_rounds):
            # One candidate is nothing to merge: no call, keep the facet.
            for label in [l for l, c in pending.items() if len(c) == 1]:
                settled[label] = [self._as_consolidated_facet(pending.pop(label)[0])]

            tasks = self._build_facet_consolidation_tasks(ctx, pending)
            if not tasks:
                break

            results = await self._dispatch(
                "facet_consolidation", tasks,
                self._facet_consolidation_prepare_fn(ctx),
                self._facet_consolidation_parse_fn(),
                self._facet_consolidation_fallback_fn(),
                verbose,
            )

            groups_per_domain = Counter(t["domain_label"] for t in tasks)
            survivors: Dict[str, List[ConsolidatedFacet]] = {}
            for task, result in zip(tasks, results):
                survivors.setdefault(task["domain_label"], []).extend(
                    self._facet_consolidation_survivors(task, result))

            pending = {}
            for label, facets in survivors.items():
                if groups_per_domain[label] == 1:
                    settled[label] = facets
                else:
                    pending[label] = facets
            if not pending:
                break

        for label, leftover in pending.items():
            settled[label] = [self._as_consolidated_facet(f) for f in leftover]
            self._action_log.append({
                "action": "facet_consolidation_rounds_exhausted",
                "domain": label, "rounds": self._consolidation_max_rounds,
                "remaining": len(leftover)})

        if verbose:
            print(f"    {time.time() - started:.1f}s → "
                  f"{sum(len(f) for f in settled.values())} facets")
            for label in sorted(settled):
                names = ", ".join(f.facet_name for f in settled[label])
                print(f"      {label}: {len(settled[label])} — {names}")
        return settled

    # =========================================================================
    # PHASE — FACET ASSIGNMENT (ideas into the settled inventory)
    # =========================================================================

    async def _apply_shortlist(self, tasks: List[Dict], card_text) -> None:
        """Trim each task's menu to the union of its labels' nearest cards.

        Runs outside the task builder, which has to stay pure and synchronous:
        embedding is I/O. The gate is `len(menu) > shortlist_k` per scope; below
        that the shortlist would return the whole menu anyway.
        """
        if not self._assign_shortlist_enabled or not tasks:
            return
        by_scope: Dict[Tuple, List[Dict]] = {}
        for task in tasks:
            by_scope.setdefault(task["scope"], []).append(task)

        embedder = SharedEmbedder()
        for scope_tasks in by_scope.values():
            menu = scope_tasks[0]["full_menu"]
            if len(menu) <= self._assign_shortlist_k:
                continue
            cards = await embedder.embed_texts(
                [card_text(item.model_dump()) for item in menu])
            reps = [rep for task in scope_tasks for rep in task["reps"]]
            vectors = await embedder.embed_texts([rep.label or " " for rep in reps])
            offset = 0
            for task in scope_tasks:
                n = len(task["reps"])
                keep = shortlist_indices(
                    vectors[offset:offset + n], cards, self._assign_shortlist_k)
                task["menu"] = [menu[i] for i in keep]
                offset += n

    def _build_facet_assignment_tasks(
        self,
        ctx: PromptContext,
        facets: Dict[str, List[ConsolidatedFacet]],
        labels: Dict[str, Dict[str, str]],
    ) -> List[Dict]:
        """One task per batch of unique labels within one domain.

        Ideas carrying the same normalized label become one rep: a single call
        decides for all of them. One call per idea would resend the whole menu
        thousands of times, which is what this level used to cost.

        A domain with fewer than two facets gets no task — there is nothing to
        choose, so `_run_facet_assignment` assigns it without a call.
        """
        tasks: List[Dict] = []
        for label in sorted(facets):
            menu = facets[label]
            if len(menu) < 2:
                continue
            reps = group_label_reps((labels.get(label) or {}).items())
            for group in make_batches(len(reps), self._assign_batch_k):
                tasks.append({
                    "domain_label": label,
                    "scope": label,
                    "reps": [reps[i] for i in group],
                    "menu": list(menu),
                    "full_menu": list(menu),
                })
        return tasks

    def _facet_assignment_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            menu = task["menu"]
            ideas = [(rep.idea_ids[0], rep.label) for rep in task["reps"]]
            prompt = build_facet_assignment_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                facets=menu,
                ideas=ideas,
            )
            self._capture(
                f"facet_assignment_{task['domain_label']}", prompt,
                "facet_assignment",
                {"model": self._model["facet_assignment"],
                 "temperature": self._temperature,
                 "max_tokens": self._max_tokens_assignment,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "n_facets": len(menu),
                 "n_ideas": len(ideas),
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": build_facet_assignment_model(
                    [f"F{i}" for i in range(1, len(menu) + 1)],
                    [idea_id for idea_id, _ in ideas]),
                "temperature": self._temperature,
                "max_tokens": self._max_tokens_assignment,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["facet_assignment"],
                    phase="classifier_facet_assignment"),
            }
        return prepare_fn

    def _facet_assignment_parse_fn(self, pending: Optional[List[Dict]]):
        """Accept validated items, fan them out over the rep's instances.

        `pending` collects the reps this call could not place — missing,
        duplicated, or answered F_NONE. They run a second pass as single calls
        against the full menu. `pending=None` IS that second pass: an item that
        fails there is left to the __UNASSIGNED__ net rather than looping.
        """
        def parse_fn(task: Dict, response) -> Dict[str, str]:
            id_to_name = {f"F{i}": f.facet_name
                          for i, f in enumerate(task["menu"], 1)}
            rep_by_id = {rep.idea_ids[0]: rep for rep in task["reps"]}
            ok, escalate = validate_batch_response(
                list(rep_by_id), response,
                id_field="assigned_facet_id", none_id="F_NONE")

            out: Dict[str, str] = {}
            for rep_id, item in ok.items():
                facet_name = id_to_name[item.assigned_facet_id]
                for idea_id in rep_by_id[rep_id].idea_ids:
                    out[idea_id] = facet_name
                    self._facet_confidence[idea_id] = item.confidence
                    self._facet_valence[idea_id] = item.valence

            if pending is not None:
                for rep_id, reason in escalate.items():
                    pending.append(_escalated(task, rep_by_id[rep_id], reason))
            return out
        return parse_fn

    @staticmethod
    def _facet_assignment_fallback_fn(pending: Optional[List[Dict]]):
        """A definitively failed batch escalates whole."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            if pending is not None:
                for rep in task["reps"]:
                    pending.append(_escalated(task, rep, "batch_failed"))
            return {}
        return fallback_fn

    async def _run_facet_assignment(
        self,
        ctx: PromptContext,
        facets: Dict[str, List[ConsolidatedFacet]],
        labels: Dict[str, Dict[str, str]],
        verbose: bool,
    ) -> Dict[str, Dict[str, str]]:
        """Assign every idea to a facet within its own domain, with valence."""
        if verbose:
            print(f"\n  Facet assignment")
        started = time.time()

        assignments: Dict[str, Dict[str, str]] = {}
        auto_assigned: Dict[str, int] = {}
        for label, menu in facets.items():
            if len(menu) != 1:
                continue
            idea_ids = list(labels.get(label) or {})
            assignments[label] = {iid: menu[0].facet_name for iid in idea_ids}
            for idea_id in idea_ids:
                self._facet_confidence[idea_id] = 1.0
            auto_assigned[label] = len(idea_ids)

        tasks = self._build_facet_assignment_tasks(ctx, facets, labels)
        await self._apply_shortlist(tasks, facet_card_text)

        pending: List[Dict] = []
        results = await self._dispatch(
            "facet_assignment", tasks,
            self._facet_assignment_prepare_fn(ctx),
            self._facet_assignment_parse_fn(pending),
            self._facet_assignment_fallback_fn(pending),
            verbose, quiet=False,
        )
        for task, result in zip(tasks, results):
            if result:
                assignments.setdefault(task["domain_label"], {}).update(result)

        if pending:
            escalated = await self._dispatch(
                "facet_assignment", pending,
                self._facet_assignment_prepare_fn(ctx),
                self._facet_assignment_parse_fn(None),
                self._facet_assignment_fallback_fn(None),
                verbose,
            )
            for task, result in zip(pending, escalated):
                if result:
                    assignments.setdefault(task["domain_label"], {}).update(result)

            reasons: Dict[str, Counter] = defaultdict(Counter)
            for task in pending:
                reasons[task["domain_label"]][task["reason"]] += 1
            for label, counts in sorted(reasons.items()):
                self._action_log.append({
                    "action": "facet_assignment_escalation",
                    "domain": label, "reasons": dict(counts)})

        # The net, after both passes: an idea with no facet still needs a home
        # in the structure, or everything downstream silently loses it.
        for label, menu in facets.items():
            if not menu or label in auto_assigned:
                continue
            expected = set(labels.get(label) or {})
            missing = expected - set(assignments.get(label, {}))
            if missing:
                print(f"    WARNING: {len(missing)}/{len(expected)} ideas received "
                      f"no facet assignment in '{label}'")
                for idea_id in missing:
                    assignments.setdefault(label, {})[idea_id] = "__UNASSIGNED__"
                    self._facet_confidence[idea_id] = 0.0

        if verbose:
            n_reps = sum(len(t["reps"]) for t in tasks)
            print(f"    {len(tasks)} calls for {n_reps} unique labels, "
                  f"{time.time() - started:.1f}s; {len(pending)} escalated; "
                  f"{len(auto_assigned)} domains auto-assigned")
            for label in sorted(assignments):
                tag = " (auto)" if label in auto_assigned else ""
                print(f"      {label}: {len(assignments[label])}"
                      f"/{len(labels.get(label) or {})}{tag}")
        return assignments

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

    # =========================================================================
    # PHASE — FACET REFINEMENT (per domain, after every idea is assigned)
    # =========================================================================

    def _build_facet_refinement_tasks(
        self,
        ctx: PromptContext,
        facets: Dict[str, List[ConsolidatedFacet]],
        assignments: Dict[str, Dict[str, str]],
        labels: Dict[str, Dict[str, str]],
    ) -> List[Dict]:
        """One task per domain that has at least two facets to judge.

        The rows carry what each facet ended up holding: its count, its share of
        the domain, and the response texts themselves. The share is what makes
        granularity judgeable — "thin" and "large" only mean something relative
        to the siblings, never against an absolute number.

        The texts are the same ones assignment showed the model. Judging a bucket
        on text the assigning call never saw would ask about a different object
        than the one that was filled.
        """
        tasks: List[Dict] = []
        for label in sorted(facets):
            menu = facets[label]
            if len(menu) < 2:
                continue
            assigned = assignments.get(label) or {}
            texts = labels.get(label) or {}

            held: Dict[str, List[str]] = {}
            for idea_id, facet_name in assigned.items():
                held.setdefault(facet_name, []).append(texts.get(idea_id, ""))
            total = sum(len(held.get(f.facet_name, [])) for f in menu)

            rows: List[Tuple[str, int, float, List[str]]] = []
            for facet in menu:
                mine = held.get(facet.facet_name, [])
                distinct = Counter(t.strip() for t in mine if t.strip())
                rows.append((
                    facet.facet_name,
                    len(mine),
                    len(mine) / total if total else 0.0,
                    [t for t, _ in distinct.most_common(self._contents_top_n)],
                ))

            tasks.append({
                "domain_label": label,
                "facets": list(menu),
                "rows": rows,
            })
        return tasks

    def _facet_refinement_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            prompt = build_facet_refinement_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                facets_block=build_facet_contents_block(task["rows"]),
            )
            self._capture(
                f"facet_refinement_{task['domain_label']}", prompt,
                "facet_refinement",
                {"model": self._model["facet_refinement"],
                 "temperature": 0.0,
                 "max_tokens": self._max_tokens_consolidation,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "n_facets": len(task["facets"]),
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": FacetRefinementResult,
                "temperature": 0.0,
                "max_tokens": self._max_tokens_consolidation,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["facet_refinement"],
                    phase="classifier_facet_refinement"),
            }
        return prepare_fn

    @staticmethod
    def _facet_refinement_parse_fn():
        def parse_fn(task: Dict, response) -> Optional[FacetRefinementResult]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _facet_refinement_fallback_fn():
        """On failure the domain is left exactly as consolidation settled it."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_facet_refinement(
        self,
        *,
        tasks: List[Dict],
        results: List,
        facets: Dict[str, List[ConsolidatedFacet]],
        assignments: Dict[str, Dict[str, str]],
        labels: Dict[str, Dict[str, str]],
        verbose: bool,
    ) -> Tuple[Dict[str, List[ConsolidatedFacet]], Dict[str, Dict[str, str]]]:
        """Apply every refinement result — structure first, ideas second.

        Order matters and is deliberate:
          1. rebuild each domain's facet list, so move targets resolve against
             the FINAL names rather than the names a concurrent call renamed
          2. splits — route by exact normalised response text to a named child
          3. merges — wholesale rename of a source facet
          4. moves  — per text, within the domain; the structure does not follow,
                      so no other idea is dragged along

        The domain is fixed by the task and absent from the response schema, so
        nothing here can move a facet out of its domain.
        """
        pre_facets: Dict[str, List[ConsolidatedFacet]] = {
            dom: list(items) for dom, items in facets.items()
        }

        # ---- 1. structure ----------------------------------------------------
        remap: Dict[Tuple[str, str], str] = {}            # (dom, src) -> new name
        splits: Dict[Tuple[str, str, str], str] = {}      # (dom, src, text) -> child
        moves: Dict[Tuple[str, str, str], Optional[str]] = {}
        renamed_to: Dict[Tuple[str, str], str] = {}       # (dom, norm old) -> new
        split_children: Dict[Tuple[str, str], List[str]] = {}
        touched: Set[str] = set()

        for task, result in zip(tasks, results):
            dom = task["domain_label"]
            group = task["facets"]
            before = [f.facet_name for f in group]

            if result is None or not result.facets:
                self._action_log.append({
                    "action": "facet_refinement_failed", "domain": dom,
                    "note": "no result — domain left as consolidation settled it",
                    "facets_before": before})
                continue

            touched.add(dom)
            by_norm = {self._norm_text(b): b for b in before}

            def _resolve(src: str) -> Optional[str]:
                return by_norm.get(self._norm_text(src))

            unmatched = sorted({
                s for item in result.facets for s in (item.source_facets or [])
                if _resolve(s) is None
            })
            if unmatched:
                self._action_log.append({
                    "action": "unknown_source_facet", "domain": dom,
                    "sources": unmatched})

            # A source claimed by several returned facets is only routable when
            # the claimants carry instance_texts. Without that the remap would
            # let the last writer win and move the whole bucket.
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
                self._action_log.append({
                    "action": "unroutable_facet_claim", "domain": dom,
                    "sources": sorted(contested),
                    "note": "claimed by several returned facets without "
                            "instance_texts — ideas left on the source"})

            settled: List[ConsolidatedFacet] = []
            consumed: Set[str] = set()
            for item in result.facets:
                settled.append(ConsolidatedFacet(
                    facet_name=item.facet_name,
                    facet_definition=item.facet_definition,
                    boundary_test=item.boundary_test,
                    exclusions=item.exclusions,
                    example_observations=item.example_observations,
                    source_facets=item.source_facets,
                ))

                sources = [r for r in (_resolve(s) for s in (item.source_facets or []))
                           if r is not None]
                consumed.update(sources)
                if item.action == "split" and item.instance_texts:
                    for src in (sources or before):
                        for txt in item.instance_texts:
                            splits[(dom, src, self._norm_text(txt))] = item.facet_name
                        split_children.setdefault(
                            (dom, self._norm_text(src)), []).append(item.facet_name)
                    self._action_log.append({
                        "action": "facet_split", "domain": dom,
                        "into": item.facet_name, "sources": sources,
                        "n_texts": len(item.instance_texts),
                        "texts": item.instance_texts})
                else:
                    for src in sources:
                        if src != item.facet_name and src not in contested:
                            remap[(dom, src)] = item.facet_name
                            renamed_to[(dom, self._norm_text(src))] = item.facet_name
                    if item.action in ("merge", "widen") or (
                            sources and sources != [item.facet_name]):
                        self._action_log.append({
                            "action": f"facet_{item.action}", "domain": dom,
                            "result": item.facet_name, "sources": sources})

            # Sources never claimed by any returned facet: keep them. Dropping a
            # facet silently would orphan every idea assigned to it.
            returned = {self._norm_text(f.facet_name) for f in settled}
            for facet in group:
                if (facet.facet_name in consumed
                        or self._norm_text(facet.facet_name) in returned):
                    continue
                settled.append(facet)
                self._action_log.append({
                    "action": "facet_kept_unclaimed_in_refinement",
                    "domain": dom, "facet": facet.facet_name})

            facets[dom] = settled

            for m in (result.misfits or []):
                real_from = _resolve(m.from_facet) or m.from_facet
                for txt in (m.instance_texts or []):
                    moves[(dom, real_from, self._norm_text(txt))] = (
                        m.target_facet if m.verdict == "move" else None)
                self._action_log.append({
                    "action": f"facet_misfit_{m.verdict}", "domain": dom,
                    "from_facet": real_from, "target": m.target_facet,
                    "n_texts": len(m.instance_texts or []),
                    "texts": m.instance_texts, "reason": m.reason})

        # A source split into exactly one child was renamed, not divided.
        for key, children in split_children.items():
            uniq = sorted(set(children))
            if len(uniq) == 1:
                renamed_to.setdefault(key, uniq[0])

        home: Dict[str, Set[str]] = {
            dom: {f.facet_name for f in items} for dom, items in facets.items()
        }

        # ---- 2-4. ideas ------------------------------------------------------
        n_split = n_remap = n_moved = n_out = n_unresolved = 0
        unresolved_targets: Counter = Counter()
        for dom in touched:
            assigns = assignments.get(dom, {})
            texts = labels.get(dom) or {}
            for idea_id, current in list(assigns.items()):
                txt = self._norm_text(texts.get(idea_id, ""))

                key = (dom, current, txt)
                if key in moves:
                    target = moves[key]
                    if target is None:
                        n_out += 1          # flagged contentless; left in place
                        continue
                    if target not in home.get(dom, set()):
                        target = renamed_to.get((dom, self._norm_text(target)), target)
                    if target in home.get(dom, set()):
                        assigns[idea_id] = target
                        n_moved += 1
                    else:
                        n_unresolved += 1
                        unresolved_targets[target] += 1
                    continue

                if key in splits:
                    assigns[idea_id] = splits[key]
                    n_split += 1
                    continue

                if (dom, current) in remap:
                    assigns[idea_id] = remap[(dom, current)]
                    n_remap += 1

        # ---- 5. self-check: no idea may point at a facet that does not exist --
        orphans: Counter = Counter()
        for dom in touched:
            names = home.get(dom, set())
            for idea_id, facet_name in assignments.get(dom, {}).items():
                if facet_name and facet_name != "__UNASSIGNED__" and facet_name not in names:
                    orphans[(dom, facet_name)] += 1

        restored = 0
        for (dom, facet_name), _count in orphans.items():
            source = next((f for f in pre_facets.get(dom, [])
                           if f.facet_name == facet_name), None)
            if source is None:
                continue
            if all(f.facet_name != facet_name for f in facets[dom]):
                facets[dom].append(source)
            home.setdefault(dom, set()).add(facet_name)
            restored += 1
        if orphans:
            self._action_log.append({
                "action": "orphaned_facet_assignment", "restored_nodes": restored,
                "ideas_affected": sum(orphans.values()),
                "facets": sorted({f for (_, f) in orphans})})
            if verbose:
                print(f"    SELF-CHECK: {sum(orphans.values())} ideas pointed at "
                      f"{len(orphans)} facet(s) missing from the structure — "
                      f"{restored} node(s) restored")

        self._action_log.append({
            "action": "_facet_totals", "ideas_split": n_split,
            "ideas_remapped": n_remap, "ideas_moved": n_moved,
            "flagged_contentless_left_in_place": n_out,
            "moves_with_unresolvable_target": n_unresolved,
            "unresolved_target_names": dict(unresolved_targets.most_common(20))})

        if verbose:
            print(f"    Ideas: {n_remap} remapped, {n_split} split, {n_moved} moved "
                  f"across facets, {n_out} flagged contentless (left in place)")

        return facets, assignments

    async def _run_facet_refinement(
        self,
        ctx: PromptContext,
        facets: Dict[str, List[ConsolidatedFacet]],
        assignments: Dict[str, Dict[str, str]],
        labels: Dict[str, Dict[str, str]],
        verbose: bool,
    ) -> Tuple[Dict[str, List[ConsolidatedFacet]], Dict[str, Dict[str, str]]]:
        """Judge each domain's facets against what they actually ended up holding.

        The one phase that can only run after assignment: real counts and real
        texts do not exist before it.
        """
        if verbose:
            print(f"\n  Facet refinement")
        started = time.time()

        tasks = self._build_facet_refinement_tasks(ctx, facets, assignments, labels)
        results = await self._dispatch(
            "facet_refinement", tasks,
            self._facet_refinement_prepare_fn(ctx),
            self._facet_refinement_parse_fn(),
            self._facet_refinement_fallback_fn(),
            verbose,
        )
        facets, assignments = self._apply_facet_refinement(
            tasks=tasks, results=results, facets=facets,
            assignments=assignments, labels=labels, verbose=verbose,
        )

        if verbose:
            s = self._last_stats
            print(f"    {len(tasks)} tasks, {time.time() - started:.1f}s "
                  f"({s.get('tasks_successful', 0)} ok, "
                  f"{s.get('timeouts', 0)} timeouts) → "
                  f"{sum(len(f) for f in facets.values())} facets")
        return facets, assignments

    # =========================================================================
    # PHASE — ATTRIBUTE DISCOVERY (per facet, chunked)
    # =========================================================================

    def _build_attribute_discovery_tasks(
        self,
        ctx: PromptContext,
        facets: Dict[str, List[ConsolidatedFacet]],
        observations: Dict[Tuple[str, str], List[str]],
    ) -> List[Dict]:
        """One task per (facet, chunk), for every facet that holds observations.

        The mirror of facet discovery one level down. The drain domains are
        already absent here: they never got facets, so they hold nothing to
        chunk.
        """
        tasks: List[Dict] = []
        for domain_label in sorted(facets):
            for facet in facets[domain_label]:
                held = observations.get((domain_label, facet.facet_name)) or []
                if not held:
                    continue
                chunks = self._create_batches(
                    held,
                    size_min=self._attribute_chunk_size_min,
                    size_max=self._attribute_chunk_size_max,
                    target=self._attribute_target_batches,
                    overlap=self._attribute_chunk_overlap,
                )
                for chunk_idx, chunk in enumerate(chunks):
                    tasks.append({
                        "domain_label": domain_label,
                        "facet_name": facet.facet_name,
                        "facet": facet,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                        "observations": chunk,
                    })
        return tasks

    def _attribute_discovery_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            facet = task["facet"]
            prompt = build_attribute_discovery_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                dimension_name=ctx.dimension_name,
                dimension_description=ctx.dimension_description,
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                facet_name=facet.facet_name,
                facet_definition=facet.facet_definition,
                facet_boundary_test=facet.boundary_test,
                facet_exclusions=facet.exclusions,
                observations=task["observations"],
            )
            if task["chunk_idx"] == 0:
                self._capture(
                    f"attribute_discovery_{task['domain_label']}_{task['facet_name']}",
                    prompt, "attribute_discovery",
                    {"model": self._model["attribute_discovery"],
                     "temperature": self._temperature,
                     "max_tokens": self._max_tokens_attribute_discovery,
                     "language": ctx.language,
                     "domain": task["domain_label"],
                     "facet": task["facet_name"],
                     "total_chunks": task["total_chunks"],
                     "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": AttributeDiscoveryResult,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens_attribute_discovery,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["attribute_discovery"],
                    phase="classifier_attribute_discovery"),
            }
        return prepare_fn

    @staticmethod
    def _attribute_discovery_parse_fn():
        def parse_fn(task: Dict, response) -> List[DiscoveredAttribute]:
            return list(response.attributes) if response else []
        return parse_fn

    @staticmethod
    def _attribute_discovery_fallback_fn():
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredAttribute]:
            return []
        return fallback_fn

    async def _run_attribute_discovery(
        self,
        ctx: PromptContext,
        facets: Dict[str, List[ConsolidatedFacet]],
        observations: Dict[Tuple[str, str], List[str]],
        verbose: bool,
    ) -> Dict[str, Dict[str, List[DiscoveredAttribute]]]:
        """Propose attributes per facet, then collapse byte-identical repeats."""
        if verbose:
            print(f"\n  Attribute discovery")
        started = time.time()

        tasks = self._build_attribute_discovery_tasks(ctx, facets, observations)
        results = await self._dispatch(
            "attribute_discovery", tasks,
            self._attribute_discovery_prepare_fn(ctx),
            self._attribute_discovery_parse_fn(),
            self._attribute_discovery_fallback_fn(),
            verbose,
        )

        flat: Dict[Tuple[str, str], List[DiscoveredAttribute]] = {}
        for task, result in zip(tasks, results):
            flat.setdefault(
                (task["domain_label"], task["facet_name"]), []).extend(result or [])

        raw: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        for (domain_label, facet_name), attributes in sorted(flat.items()):
            deduped = dedup_exact_attributes(attributes)
            if len(deduped) < len(attributes):
                self._action_log.append({
                    "action": "attribute_exact_dedup", "domain": domain_label,
                    "facet": facet_name,
                    "before": len(attributes), "after": len(deduped)})
            raw.setdefault(domain_label, {})[facet_name] = deduped

        if verbose:
            s = self._last_stats
            total = sum(len(a) for f in raw.values() for a in f.values())
            print(f"    {len(tasks)} tasks, {time.time() - started:.1f}s "
                  f"({s.get('tasks_successful', 0)} ok, "
                  f"{s.get('timeouts', 0)} timeouts) → "
                  f"{total} candidate attributes across {len(flat)} facets")
        return raw

    # =========================================================================
    # PHASE — ATTRIBUTE CONSOLIDATION (per facet, before any idea is assigned)
    # =========================================================================

    def _build_attribute_consolidation_tasks(
        self,
        ctx: PromptContext,
        raw: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        facets: Optional[Dict[str, List[ConsolidatedFacet]]] = None,
    ) -> List[Dict]:
        """One task per facet that has candidates, split when it holds too many.

        `facets` supplies the facet card the prompt shows; without it the task
        still carries the name, which is what the tests scope on.
        """
        by_name: Dict[Tuple[str, str], ConsolidatedFacet] = {}
        for domain_label, items in (facets or {}).items():
            for facet in items:
                by_name[(domain_label, facet.facet_name)] = facet

        tasks: List[Dict] = []
        for domain_label in sorted(raw):
            for facet_name in sorted(raw[domain_label]):
                candidates = raw[domain_label][facet_name]
                if not candidates:
                    continue
                for group in self._consolidation_groups(candidates):
                    tasks.append({
                        "domain_label": domain_label,
                        "facet_name": facet_name,
                        "facet": by_name.get((domain_label, facet_name)),
                        "candidates": group,
                    })
        return tasks

    def _attribute_consolidation_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            facet = task["facet"]
            prompt = build_attribute_consolidation_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                dimension_name=ctx.dimension_name,
                dimension_description=ctx.dimension_description,
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                facet_name=task["facet_name"],
                facet_definition=facet.facet_definition if facet else "",
                candidates=task["candidates"],
            )
            self._capture(
                f"attribute_consolidation_{task['domain_label']}_{task['facet_name']}",
                prompt, "attribute_consolidation",
                {"model": self._model["attribute_consolidation"],
                 "temperature": 0.0,
                 "max_tokens": self._max_tokens_consolidation,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "facet": task["facet_name"],
                 "n_candidates": len(task["candidates"]),
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": AttributeConsolidationResult,
                "temperature": 0.0,
                "max_tokens": self._max_tokens_consolidation,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["attribute_consolidation"],
                    phase="classifier_attribute_consolidation"),
            }
        return prepare_fn

    @staticmethod
    def _attribute_consolidation_parse_fn():
        def parse_fn(task: Dict, response) -> Optional[AttributeConsolidationResult]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _attribute_consolidation_fallback_fn():
        """On failure the facet keeps its candidates — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    @staticmethod
    def _as_consolidated_attribute(attribute) -> ConsolidatedAttribute:
        """A candidate that survives untouched, in the settled type."""
        if isinstance(attribute, ConsolidatedAttribute):
            return attribute
        return ConsolidatedAttribute(**attribute.model_dump(),
                                     source_attributes=[attribute.attribute_name])

    def _attribute_consolidation_survivors(
        self, task: Dict, result,
    ) -> List[ConsolidatedAttribute]:
        """The settled attributes of one group, with unclaimed candidates kept."""
        candidates = task["candidates"]
        scope = {"domain": task["domain_label"], "facet": task["facet_name"]}
        if result is None or not result.attributes:
            self._action_log.append({
                "action": "attribute_consolidation_failed", **scope,
                "note": "no result — candidates left as discovered",
                "candidates": [c.attribute_name for c in candidates]})
            return [self._as_consolidated_attribute(c) for c in candidates]

        survivors: List[ConsolidatedAttribute] = list(result.attributes)
        claimed = {self._norm_text(s)
                   for a in survivors for s in (a.source_attributes or [])}
        returned = {self._norm_text(a.attribute_name) for a in survivors}
        for candidate in candidates:
            key = self._norm_text(candidate.attribute_name)
            if key in claimed or key in returned:
                continue
            survivors.append(self._as_consolidated_attribute(candidate))
            self._action_log.append({
                "action": "attribute_kept_unclaimed", **scope,
                "attribute": candidate.attribute_name})

        self._action_log.append({
            "action": "attribute_consolidation", **scope,
            "before": len(candidates), "after": len(survivors)})
        return survivors

    async def _run_attribute_consolidation(
        self,
        ctx: PromptContext,
        raw: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        facets: Dict[str, List[ConsolidatedFacet]],
        verbose: bool,
    ) -> Dict[str, Dict[str, List[ConsolidatedAttribute]]]:
        """Settle each facet's attribute inventory before any idea is assigned.

        Same round logic as the facet level: a facet whose candidates do not fit
        one call is consolidated per group, and the survivors go back in
        together so the groups still get to see each other.
        """
        if verbose:
            print(f"\n  Attribute consolidation")
        started = time.time()

        settled: Dict[str, Dict[str, List[ConsolidatedAttribute]]] = {}
        pending: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {
            domain_label: {f: list(a) for f, a in items.items() if a}
            for domain_label, items in raw.items()
        }

        def _settle(domain_label, facet_name, attributes):
            settled.setdefault(domain_label, {})[facet_name] = attributes

        for _ in range(self._consolidation_max_rounds):
            # One candidate is nothing to merge: no call, keep the attribute.
            for domain_label in list(pending):
                for facet_name in [f for f, a in pending[domain_label].items()
                                   if len(a) == 1]:
                    lone = pending[domain_label].pop(facet_name)[0]
                    _settle(domain_label, facet_name,
                            [self._as_consolidated_attribute(lone)])
                if not pending[domain_label]:
                    pending.pop(domain_label)

            tasks = self._build_attribute_consolidation_tasks(ctx, pending, facets)
            if not tasks:
                break

            results = await self._dispatch(
                "attribute_consolidation", tasks,
                self._attribute_consolidation_prepare_fn(ctx),
                self._attribute_consolidation_parse_fn(),
                self._attribute_consolidation_fallback_fn(),
                verbose,
            )

            groups_per_facet = Counter(
                (t["domain_label"], t["facet_name"]) for t in tasks)
            survivors: Dict[Tuple[str, str], List[ConsolidatedAttribute]] = {}
            for task, result in zip(tasks, results):
                survivors.setdefault(
                    (task["domain_label"], task["facet_name"]), []).extend(
                        self._attribute_consolidation_survivors(task, result))

            pending = {}
            for (domain_label, facet_name), attributes in survivors.items():
                if groups_per_facet[(domain_label, facet_name)] == 1:
                    _settle(domain_label, facet_name, attributes)
                else:
                    pending.setdefault(domain_label, {})[facet_name] = attributes
            if not pending:
                break

        for domain_label, items in pending.items():
            for facet_name, leftover in items.items():
                _settle(domain_label, facet_name,
                        [self._as_consolidated_attribute(a) for a in leftover])
                self._action_log.append({
                    "action": "attribute_consolidation_rounds_exhausted",
                    "domain": domain_label, "facet": facet_name,
                    "rounds": self._consolidation_max_rounds,
                    "remaining": len(leftover)})

        if verbose:
            total = sum(len(a) for f in settled.values() for a in f.values())
            print(f"    {time.time() - started:.1f}s → {total} attributes")
        return settled

    # =========================================================================
    # PHASE — ATTRIBUTE ASSIGNMENT (ideas into the settled inventory)
    # =========================================================================

    def _build_attribute_assignment_tasks(
        self,
        ctx: PromptContext,
        attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]],
        ideas: Dict[Tuple[str, str], Dict[str, str]],
        facets: Optional[Dict[str, List[ConsolidatedFacet]]] = None,
    ) -> List[Dict]:
        """One task per batch of unique labels within one facet.

        This level used to run one call per idea, which resent the whole menu
        every time. It now takes the same shape as the facet level: reps on the
        normalized label, batches of K, and the same escalation ladder.

        A facet with fewer than two attributes gets no task — nothing to choose.
        """
        by_name: Dict[Tuple[str, str], ConsolidatedFacet] = {}
        for domain_label, items in (facets or {}).items():
            for facet in items:
                by_name[(domain_label, facet.facet_name)] = facet

        tasks: List[Dict] = []
        for domain_label in sorted(attributes):
            for facet_name in sorted(attributes[domain_label]):
                menu = attributes[domain_label][facet_name]
                if len(menu) < 2:
                    continue
                labels = ideas.get((domain_label, facet_name)) or {}
                reps = group_label_reps(labels.items())
                for group in make_batches(len(reps), self._assign_batch_k):
                    tasks.append({
                        "domain_label": domain_label,
                        "facet_name": facet_name,
                        "facet": by_name.get((domain_label, facet_name)),
                        "scope": (domain_label, facet_name),
                        "reps": [reps[i] for i in group],
                        "menu": list(menu),
                        "full_menu": list(menu),
                    })
        return tasks

    def _attribute_assignment_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            facet = task["facet"]
            menu = task["menu"]
            ideas = [(rep.idea_ids[0], rep.label) for rep in task["reps"]]
            prompt = build_attribute_assignment_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                facet_name=task["facet_name"],
                facet_definition=facet.facet_definition if facet else "",
                attributes=menu,
                ideas=ideas,
            )
            self._capture(
                f"attribute_assignment_{task['domain_label']}_{task['facet_name']}",
                prompt, "attribute_assignment",
                {"model": self._model["attribute_assignment"],
                 "temperature": self._temperature,
                 "max_tokens": self._max_tokens_assignment,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "facet": task["facet_name"],
                 "n_attributes": len(menu),
                 "n_ideas": len(ideas),
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": build_attribute_assignment_model(
                    [f"A{i}" for i in range(1, len(menu) + 1)],
                    [idea_id for idea_id, _ in ideas]),
                "temperature": self._temperature,
                "max_tokens": self._max_tokens_assignment,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["attribute_assignment"],
                    phase="classifier_attribute_assignment"),
            }
        return prepare_fn

    def _attribute_assignment_parse_fn(self, pending: Optional[List[Dict]]):
        """Accept validated items, fan them out over the rep's instances."""
        def parse_fn(task: Dict, response) -> Dict[str, str]:
            id_to_name = {f"A{i}": a.attribute_name
                          for i, a in enumerate(task["menu"], 1)}
            rep_by_id = {rep.idea_ids[0]: rep for rep in task["reps"]}
            ok, escalate = validate_batch_response(
                list(rep_by_id), response,
                id_field="assigned_attribute_id", none_id="A_NONE")

            out: Dict[str, str] = {}
            for rep_id, item in ok.items():
                attribute_name = id_to_name[item.assigned_attribute_id]
                for idea_id in rep_by_id[rep_id].idea_ids:
                    out[idea_id] = attribute_name
                    self._attribute_confidence[idea_id] = item.confidence
                    self._attribute_valence[idea_id] = item.valence

            if pending is not None:
                for rep_id, reason in escalate.items():
                    pending.append(_escalated(task, rep_by_id[rep_id], reason))
            return out
        return parse_fn

    @staticmethod
    def _attribute_assignment_fallback_fn(pending: Optional[List[Dict]]):
        """A definitively failed batch escalates whole."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            if pending is not None:
                for rep in task["reps"]:
                    pending.append(_escalated(task, rep, "batch_failed"))
            return {}
        return fallback_fn

    async def _run_attribute_assignment(
        self,
        ctx: PromptContext,
        attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]],
        ideas: Dict[Tuple[str, str], Dict[str, str]],
        facets: Dict[str, List[ConsolidatedFacet]],
        verbose: bool,
    ) -> Dict[str, str]:
        """Assign every idea to an attribute within its own facet, with valence.

        The valence recorded here is the more precise one and supersedes the
        facet level's: it is judged against the attribute the idea landed on.
        """
        if verbose:
            print(f"\n  Attribute assignment")
        started = time.time()

        assignments: Dict[str, str] = {}
        auto_assigned: Set[Tuple[str, str]] = set()
        for domain_label, items in attributes.items():
            for facet_name, menu in items.items():
                if len(menu) != 1:
                    continue
                for idea_id in (ideas.get((domain_label, facet_name)) or {}):
                    assignments[idea_id] = menu[0].attribute_name
                    self._attribute_confidence[idea_id] = 1.0
                auto_assigned.add((domain_label, facet_name))

        tasks = self._build_attribute_assignment_tasks(
            ctx, attributes, ideas, facets)
        await self._apply_shortlist(tasks, attribute_card_text)

        pending: List[Dict] = []
        results = await self._dispatch(
            "attribute_assignment", tasks,
            self._attribute_assignment_prepare_fn(ctx),
            self._attribute_assignment_parse_fn(pending),
            self._attribute_assignment_fallback_fn(pending),
            verbose, quiet=False,
        )
        for result in results:
            if result:
                assignments.update(result)

        if pending:
            escalated = await self._dispatch(
                "attribute_assignment", pending,
                self._attribute_assignment_prepare_fn(ctx),
                self._attribute_assignment_parse_fn(None),
                self._attribute_assignment_fallback_fn(None),
                verbose,
            )
            for result in escalated:
                if result:
                    assignments.update(result)

            reasons: Dict[str, Counter] = defaultdict(Counter)
            for task in pending:
                reasons[f"{task['domain_label']}::{task['facet_name']}"][
                    task["reason"]] += 1
            for scope, counts in sorted(reasons.items()):
                domain_label, facet_name = scope.split("::", 1)
                self._action_log.append({
                    "action": "attribute_assignment_escalation",
                    "domain": domain_label, "facet": facet_name,
                    "reasons": dict(counts)})

        # The net, after both passes.
        for domain_label, items in attributes.items():
            for facet_name, menu in items.items():
                if not menu or (domain_label, facet_name) in auto_assigned:
                    continue
                expected = set(ideas.get((domain_label, facet_name)) or {})
                missing = expected - set(assignments)
                if missing:
                    print(f"    WARNING: {len(missing)}/{len(expected)} ideas received "
                          f"no attribute assignment in facet '{facet_name}'")
                    for idea_id in missing:
                        assignments[idea_id] = "__UNASSIGNED__"
                        self._attribute_confidence[idea_id] = 0.0

        if verbose:
            n_reps = sum(len(t["reps"]) for t in tasks)
            print(f"    {len(tasks)} calls for {n_reps} unique labels, "
                  f"{time.time() - started:.1f}s; {len(pending)} escalated; "
                  f"{len(auto_assigned)} facets auto-assigned → "
                  f"{len(assignments)} ideas assigned")
        return assignments

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

    def _group_ideas_by_domain_facet(
        self,
        label_mappings: Dict[str, PartitionLabelMapping],
        partition_assignments: Dict[str, Dict[str, str]],
    ) -> Dict[Tuple[str, str], List]:
        """Group idea objects by (domain, facet), using the facet assignments."""
        groups: Dict[Tuple[str, str], List] = {}
        for domain_name, assignments in partition_assignments.items():
            mapping = label_mappings.get(domain_name)
            if not mapping:
                continue
            idea_lookup = {idea.idea_id: idea for idea in mapping.ideas}
            for idea_id, facet_name in assignments.items():
                idea = idea_lookup.get(idea_id)
                if idea is not None:
                    groups.setdefault((domain_name, facet_name), []).append(idea)
        return groups
