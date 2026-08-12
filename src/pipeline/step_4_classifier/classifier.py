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
    DiscoveredFacet, ConsolidatedFacet,
    FacetDiscoveryResult, FacetConsolidationResult, FacetRefinementResult,
    build_facet_discovery_prompt,
    build_facet_consolidation_prompt,
    build_facet_assignment_model,
    build_facet_assignment_prompt,
    build_facet_contents_block,
    build_facet_refinement_prompt,
)
from .prompts_attribute import (
    DiscoveredAttribute, ConsolidatedAttribute,
    AttributeDiscoveryResult, AttributeConsolidationResult,
    AttributeRefinementResult,
    build_attribute_discovery_prompt,
    build_attribute_consolidation_prompt,
    build_attribute_assignment_model,
    build_attribute_assignment_prompt,
    build_attribute_contents_block,
    build_attribute_refinement_prompt,
    build_neighbour_block,
    build_cross_scope_model,
    build_cross_scope_prompt,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


# =============================================================================
# HELPERS
# =============================================================================

def _norm(text: Optional[str]) -> str:
    """Normalise a name or response text for matching.

    Case- and padding-insensitive only — no stemming, no stopwords, nothing
    language-specific, so this stays use-case agnostic and every match is
    checkable by eye. Every name comparison in step 4 goes through here; two
    normalisers would drift, and a drifted one fails silently.
    """
    return (text or "").strip().lower()


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

    def is_drain(self, label: str) -> bool:
        """Whether this is one of the two standing catch-all domains.

        Matched case-insensitively, and that is not cosmetic: step 3 writes the
        label into the metadata with its original capitalisation, while domain
        discovery lowercases the partition name. An exact match finds neither,
        silently, and both catch-alls get a full facet and attribute layer —
        which is exactly what step 3 defines them to be exempt from.
        """
        return _norm(label) in {_norm(d) for d in self.drain_labels}


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
        "attribute_assignment", "attribute_refinement",
        "cross_scope_consolidation", "valence_merge",
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
        # Hoe vaak elke kandidaat door een onafhankelijke chunk is
        # voorgesteld — de enige prevalentie die vóór toewijzing bestaat.
        self._recurrence: Dict[tuple, Dict[str, int]] = {}
        self._passes: Dict[tuple, int] = {}
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
        print("TAXONOMY DISCOVERY (8 phases)")
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
            print("  Per level: discovery → consolidation → assignment → refinement")

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
            print("\n  [RATE LIMITING SETUP]")
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
        """The eight phases, in order. Each one logs its own lines and its own
        cost; this method only decides what runs and what feeds what.

        `stop_after_phase` returns the state as it stands after that phase. A
        partial return is a real result, not an empty one — the phases that did
        run are in it, so the runner can still cache and inspect them.
        """
        started = time.time()
        self._facet_confidence.clear()
        self._attribute_confidence.clear()
        self._facet_valence.clear()
        self._attribute_valence.clear()
        self._action_log.clear()

        ctx = prompt_context

        # One rendered label per idea, per domain. Rendered once, here: every
        # phase downstream shows the model the same text, and the refinement
        # remaps match on it.
        labels: Dict[str, Dict[str, str]] = {
            domain_label: {
                idea.idea_id: format_label(
                    idea, self._label_source, self._label_prefix)
                for idea in mapping.ideas
            }
            for domain_label, mapping in label_mappings.items()
        }

        state: Dict = {
            "partition_n_labels": {
                d: len(m.labels) for d, m in label_mappings.items()},
            "partition_n_batches": {
                d: len(self._create_batches(m.labels))
                for d, m in label_mappings.items()},
            "partition_facets": {},
            "partition_assignments": {},
            "partition_attributes": {},
            "attribute_assignments": {},
            "partition_raw_facets": {},
            "raw_partition_attributes": {},
            "raw_attribute_assignments": {},
        }

        def _stop(phase: str) -> bool:
            if self._stop_after_phase != phase:
                return False
            if verbose:
                print(f"\n  [STOP] after {phase} — remaining phases skipped")
            return True

        # ---- facet level -----------------------------------------------------
        raw_facets = await self._run_facet_discovery(ctx, verbose)
        state["partition_raw_facets"] = raw_facets
        if _stop("facet_discovery"):
            state["partition_facets"] = {
                d: [self._as_consolidated_facet(f) for f in items]
                for d, items in raw_facets.items()}
            return self._taxonomy_result(state, started, verbose)

        facets = await self._run_facet_consolidation(ctx, raw_facets, verbose)
        state["partition_facets"] = facets
        if _stop("facet_consolidation"):
            return self._taxonomy_result(state, started, verbose)

        assignments = await self._run_facet_assignment(ctx, facets, labels, verbose)
        state["partition_assignments"] = assignments
        if _stop("facet_assignment"):
            return self._taxonomy_result(state, started, verbose)

        facets, assignments = await self._run_facet_refinement(
            ctx, facets, assignments, labels, verbose)
        state["partition_facets"] = facets
        state["partition_assignments"] = assignments
        if _stop("facet_refinement"):
            return self._taxonomy_result(state, started, verbose)

        # ---- attribute level -------------------------------------------------
        # Rebuilt from the refined assignments: which ideas a facet holds only
        # settles once refinement has moved what did not belong.
        ideas_per_facet = self._ideas_per_facet(assignments, labels)
        observations = {key: list(texts.values())
                        for key, texts in ideas_per_facet.items()}

        raw_attributes = await self._run_attribute_discovery(
            ctx, facets, observations, verbose)
        state["raw_partition_attributes"] = raw_attributes
        if _stop("attribute_discovery"):
            state["partition_attributes"] = {
                d: {f: [self._as_consolidated_attribute(a) for a in items]
                    for f, items in facet_items.items()}
                for d, facet_items in raw_attributes.items()}
            return self._taxonomy_result(state, started, verbose)

        attributes = await self._run_attribute_consolidation(
            ctx, raw_attributes, facets, verbose)
        state["partition_attributes"] = attributes
        if _stop("attribute_consolidation"):
            return self._taxonomy_result(state, started, verbose)

        attribute_assignments = await self._run_attribute_assignment(
            ctx, attributes, ideas_per_facet, facets, verbose)
        state["attribute_assignments"] = attribute_assignments
        # The state before refinement remaps anything. This is what makes a bad
        # merge diagnosable afterwards, and what the standalone replay runs on.
        state["raw_attribute_assignments"] = dict(attribute_assignments)
        if _stop("attribute_assignment"):
            return self._taxonomy_result(state, started, verbose)

        attributes, attribute_assignments, assignments = (
            await self._run_attribute_refinement(
                ctx, attributes, attribute_assignments, assignments, facets,
                ideas_per_facet, verbose))
        if _stop("attribute_refinement"):
            state["partition_attributes"] = attributes
            state["attribute_assignments"] = attribute_assignments
            state["partition_assignments"] = assignments
            return self._taxonomy_result(state, started, verbose)

        attributes, attribute_assignments, assignments = (
            await self._run_cross_scope_consolidation(
                ctx, attributes, attribute_assignments, assignments, verbose))
        state["partition_attributes"] = attributes
        state["attribute_assignments"] = attribute_assignments
        state["partition_assignments"] = assignments

        return self._taxonomy_result(state, started, verbose)

    @staticmethod
    def _ideas_per_facet(
        assignments: Dict[str, Dict[str, str]],
        labels: Dict[str, Dict[str, str]],
    ) -> Dict[Tuple[str, str], Dict[str, str]]:
        """Regroup the rendered labels by (domain, facet), following assignment."""
        out: Dict[Tuple[str, str], Dict[str, str]] = {}
        for domain_label, assigned in assignments.items():
            texts = labels.get(domain_label) or {}
            for idea_id, facet_name in assigned.items():
                out.setdefault((domain_label, facet_name), {})[idea_id] = \
                    texts.get(idea_id, "")
        return out

    def _taxonomy_result(
        self, state: Dict, started: float, verbose: bool,
    ) -> TaxonomyResult:
        """Assemble the result from whatever the phases that ran produced."""
        if verbose:
            print(f"\n  Taxonomy complete in {time.time() - started:.1f}s")
        return TaxonomyResult(
            **state,
            facet_confidence=self._facet_confidence,
            attribute_confidence=self._attribute_confidence,
            facet_valence=self._facet_valence,
            attribute_valence=self._attribute_valence,
            consolidation_log=list(self._action_log),
        )

    # =========================================================================
    # PHASE — CROSS-SCOPE CONSOLIDATION (every domain and facet at once)
    # =========================================================================

    def _build_cross_scope_task(
        self,
        attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]],
        assignments: Dict[str, str],
    ) -> Optional[Dict]:
        """One task holding the whole inventory, keyed on stable ids.

        Every other phase is scope-locked: a structural merge across a boundary
        would drag every idea in the bucket along. This is the one place where
        that is the intent — forty-odd facets each settled alone, so the same
        concept survives in several of them and nothing else can see it.
        """
        counts = Counter(assignments.values())
        entries: List[Dict] = []
        for domain_label in sorted(attributes):
            for facet_name in sorted(attributes[domain_label]):
                for attribute in attributes[domain_label][facet_name]:
                    entries.append({
                        "id": f"A{len(entries) + 1}",
                        "domain": domain_label,
                        "facet": facet_name,
                        "attribute": attribute,
                        "n": counts.get(attribute.attribute_name, 0),
                    })
        if len(entries) < 2:
            return None

        lines, current = [], None
        for e in entries:
            head = f"{e['domain']} > {e['facet']}"
            if head != current:
                lines.append(f"\n{head}")
                current = head
            lines.append(
                f"  [{e['id']}] {e['attribute'].attribute_name} — {e['n']} responses\n"
                f"        {e['attribute'].attribute_definition}")
        return {"entries": entries, "inventory_block": "\n".join(lines)}

    def _cross_scope_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            ids = [e["id"] for e in task["entries"]]
            prompt = build_cross_scope_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                dimension_name=ctx.dimension_name,
                dimension_description=ctx.dimension_description,
                inventory_block=task["inventory_block"],
            )
            self._capture("cross_scope", prompt, "cross_scope_consolidation",
                          {"model": self._model["cross_scope_consolidation"],
                           "temperature": 0.0,
                           "max_tokens": self._max_tokens_consolidation,
                           "language": ctx.language,
                           "n_attributes": len(ids),
                           "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": build_cross_scope_model(ids),
                "temperature": 0.0,
                "max_tokens": self._max_tokens_consolidation,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["cross_scope_consolidation"],
                    phase="classifier_cross_scope_consolidation"),
            }
        return prepare_fn

    @staticmethod
    def _cross_scope_parse_fn():
        def parse_fn(task: Dict, response):
            return response if response else None
        return parse_fn

    @staticmethod
    def _cross_scope_fallback_fn():
        """On failure the inventory is left exactly as the facets settled it."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    async def _run_cross_scope_consolidation(
        self,
        ctx: PromptContext,
        attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]],
        assignments: Dict[str, str],
        facet_assignments: Dict[str, Dict[str, str]],
        verbose: bool,
    ):
        """Fold duplicate attributes together across facets and domains.

        An id the model does not claim keeps its own attribute where it is — the
        same fail-safe every other consolidation has, and here it is what stops a
        forgotten id from taking its responses with it.
        """
        if verbose:
            print("\n  Cross-scope consolidation")
        started = time.time()

        task = self._build_cross_scope_task(attributes, assignments)
        if task is None:
            return attributes, assignments, facet_assignments

        results = await self._dispatch(
            "cross_scope_consolidation", [task],
            self._cross_scope_prepare_fn(ctx),
            self._cross_scope_parse_fn(),
            self._cross_scope_fallback_fn(),
            verbose,
        )
        result = results[0] if results else None
        by_id = {e["id"]: e for e in task["entries"]}
        before = len(by_id)

        if result is None or not result.attributes:
            self._action_log.append({
                "action": "cross_scope_failed", "n_attributes": before,
                "note": "no result — inventory left as the facets settled it"})
            return attributes, assignments, facet_assignments

        # ---- 1. structure: id -> (new name, home) -------------------------
        rename: Dict[str, str] = {}
        home: Dict[str, Tuple[str, str]] = {}
        settled: Dict[str, Dict[str, List[ConsolidatedAttribute]]] = {}
        claimed: Set[str] = set()

        for item in result.attributes:
            sources = [i for i in (item.source_ids or []) if i in by_id]
            if not sources:
                continue
            anchor = by_id.get(item.home_id) or by_id[sources[0]]
            dom, fac = anchor["domain"], anchor["facet"]
            merged = ConsolidatedAttribute(
                attribute_name=item.attribute_name,
                attribute_definition=item.attribute_definition,
                boundary_test=anchor["attribute"].boundary_test,
                exclusions=anchor["attribute"].exclusions,
                example_observations=anchor["attribute"].example_observations,
                source_attributes=[by_id[i]["attribute"].attribute_name for i in sources],
            )
            settled.setdefault(dom, {}).setdefault(fac, []).append(merged)
            for i in sources:
                claimed.add(i)
                rename[i] = item.attribute_name
                home[i] = (dom, fac)
            if len(sources) > 1:
                self._action_log.append({
                    "action": "cross_scope_merge", "result": item.attribute_name,
                    "home": f"{dom} > {fac}",
                    "sources": [f"{by_id[i]['domain']} > {by_id[i]['facet']} > "
                                f"{by_id[i]['attribute'].attribute_name}" for i in sources]})

        for entry_id, e in by_id.items():
            if entry_id in claimed:
                continue
            settled.setdefault(e["domain"], {}).setdefault(e["facet"], []).append(
                e["attribute"])
            rename[entry_id] = e["attribute"].attribute_name
            home[entry_id] = (e["domain"], e["facet"])
            self._action_log.append({
                "action": "cross_scope_kept_unclaimed",
                "attribute": e["attribute"].attribute_name})

        # ---- 2. ideas follow their attribute ------------------------------
        by_old: Dict[Tuple[str, str, str], str] = {}
        for entry_id, e in by_id.items():
            by_old[(e["domain"], e["facet"], e["attribute"].attribute_name)] = entry_id

        idea_home: Dict[str, Tuple[str, str]] = {}
        for dom, assigns in facet_assignments.items():
            for idea_id, fac in assigns.items():
                idea_home[idea_id] = (dom, fac)

        n_moved = 0
        for idea_id, name in list(assignments.items()):
            place = idea_home.get(idea_id)
            if not place:
                continue
            entry_id = by_old.get((place[0], place[1], name))
            if entry_id is None:
                continue
            assignments[idea_id] = rename[entry_id]
            dom, fac = home[entry_id]
            if (dom, fac) != place:
                facet_assignments.setdefault(dom, {})[idea_id] = fac
                facet_assignments.get(place[0], {}).pop(idea_id, None)
                n_moved += 1

        after = sum(len(a) for f in settled.values() for a in f.values())
        self._action_log.append({
            "action": "_cross_scope_totals", "before": before, "after": after,
            "ideas_rehoused": n_moved})
        if verbose:
            print(f"    {time.time() - started:.1f}s → {before} - {before - after} = "
                  f"{after} attributes, {n_moved} ideas rehoused")
        return settled, assignments, facet_assignments

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
            if ctx.is_drain(label):
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
            print("\n  Facet discovery")
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

        passes = Counter(t["domain_label"] for t in tasks)
        raw: Dict[str, List[DiscoveredFacet]] = {}
        for label in sorted(flat):
            seen = Counter(_norm(f.facet_name) for f in flat[label])
            deduped = dedup_exact_facets(flat[label])
            self._recurrence[(label,)] = {
                f.facet_name: seen[_norm(f.facet_name)] for f in deduped}
            self._passes[(label,)] = passes[label]
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
                tasks.append({
                    "domain_label": label, "candidates": group,
                    "recurrence": self._recurrence.get((label,)),
                    "n_passes": self._passes.get((label,), 0)})
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
                recurrence=task.get("recurrence"),
                n_passes=task.get("n_passes", 0),
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
        claimed = {_norm(s)
                   for f in survivors for s in (f.source_facets or [])}
        returned = {_norm(f.facet_name) for f in survivors}
        for candidate in candidates:
            key = _norm(candidate.facet_name)
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
            print("\n  Facet consolidation")
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
            print("\n  Facet assignment")
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
    # TEXT MATCHING
    # =========================================================================

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
            by_norm = {_norm(b): b for b in before}

            def _resolve(src: str) -> Optional[str]:
                return by_norm.get(_norm(src))

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
                            splits[(dom, src, _norm(txt))] = item.facet_name
                        split_children.setdefault(
                            (dom, _norm(src)), []).append(item.facet_name)
                    self._action_log.append({
                        "action": "facet_split", "domain": dom,
                        "into": item.facet_name, "sources": sources,
                        "n_texts": len(item.instance_texts),
                        "texts": item.instance_texts})
                else:
                    for src in sources:
                        if src != item.facet_name and src not in contested:
                            remap[(dom, src)] = item.facet_name
                            renamed_to[(dom, _norm(src))] = item.facet_name
                    if item.action in ("merge", "widen") or (
                            sources and sources != [item.facet_name]):
                        self._action_log.append({
                            "action": f"facet_{item.action}", "domain": dom,
                            "result": item.facet_name, "sources": sources})

            # Sources never claimed by any returned facet: keep them. Dropping a
            # facet silently would orphan every idea assigned to it.
            returned = {_norm(f.facet_name) for f in settled}
            for facet in group:
                if (facet.facet_name in consumed
                        or _norm(facet.facet_name) in returned):
                    continue
                settled.append(facet)
                self._action_log.append({
                    "action": "facet_kept_unclaimed_in_refinement",
                    "domain": dom, "facet": facet.facet_name})

            facets[dom] = settled

            for m in (result.misfits or []):
                real_from = _resolve(m.from_facet) or m.from_facet
                for txt in (m.instance_texts or []):
                    moves[(dom, real_from, _norm(txt))] = (
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
                txt = _norm(texts.get(idea_id, ""))

                key = (dom, current, txt)
                if key in moves:
                    target = moves[key]
                    if target is None:
                        n_out += 1          # flagged contentless; left in place
                        continue
                    if target not in home.get(dom, set()):
                        target = renamed_to.get((dom, _norm(target)), target)
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
            print("\n  Facet refinement")
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
            print("\n  Attribute discovery")
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

        passes = Counter((t["domain_label"], t["facet_name"]) for t in tasks)
        raw: Dict[str, Dict[str, List[DiscoveredAttribute]]] = {}
        for (domain_label, facet_name), attributes in sorted(flat.items()):
            seen = Counter(_norm(a.attribute_name) for a in attributes)
            deduped = dedup_exact_attributes(attributes)
            key = (domain_label, facet_name)
            self._recurrence[key] = {
                a.attribute_name: seen[_norm(a.attribute_name)] for a in deduped}
            self._passes[key] = passes[key]
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
                        "recurrence": self._recurrence.get((domain_label, facet_name)),
                        "n_passes": self._passes.get((domain_label, facet_name), 0),
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
                recurrence=task.get("recurrence"),
                n_passes=task.get("n_passes", 0),
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
        claimed = {_norm(s)
                   for a in survivors for s in (a.source_attributes or [])}
        returned = {_norm(a.attribute_name) for a in survivors}
        for candidate in candidates:
            key = _norm(candidate.attribute_name)
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
            print("\n  Attribute consolidation")
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
            print("\n  Attribute assignment")
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

    # =========================================================================
    # PHASE — ATTRIBUTE REFINEMENT (per facet, after every idea is assigned)
    # =========================================================================

    def _build_attribute_refinement_tasks(
        self,
        ctx: PromptContext,
        attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]],
        assignments: Dict[str, str],
        labels: Dict[Tuple[str, str], Dict[str, str]],
    ) -> List[Dict]:
        """One task per facet that has at least two attributes to judge.

        Besides its own contents each task carries its neighbouring facets in
        the same domain, with their real sizes. They are context for writing
        boundaries against something real, and a place to name a move target —
        never merge candidates. Without that distinction stated, the model
        starts merging across facets, which is the failure this phase exists to
        prevent.
        """
        tasks: List[Dict] = []
        for domain_label in sorted(attributes):
            items = attributes[domain_label]

            def _held(facet_name: str) -> Dict[str, List[str]]:
                out: Dict[str, List[str]] = {}
                for idea_id, text in (labels.get((domain_label, facet_name)) or {}).items():
                    name = assignments.get(idea_id)
                    if name:
                        out.setdefault(name, []).append(text)
                return out

            for facet_name in sorted(items):
                menu = items[facet_name]
                if len(menu) < 2:
                    continue

                held = _held(facet_name)
                total = sum(len(held.get(a.attribute_name, [])) for a in menu)
                rows: List[Tuple[str, int, float, List[str]]] = []
                for attribute in menu:
                    mine = held.get(attribute.attribute_name, [])
                    distinct = Counter(t.strip() for t in mine if t.strip())
                    rows.append((
                        attribute.attribute_name,
                        len(mine),
                        len(mine) / total if total else 0.0,
                        [t for t, _ in distinct.most_common(self._contents_top_n)],
                    ))

                neighbours = []
                for other_name in sorted(items):
                    if other_name == facet_name or not items[other_name]:
                        continue
                    other_held = _held(other_name)
                    neighbours.append((
                        other_name,
                        [(a.attribute_name, len(other_held.get(a.attribute_name, [])))
                         for a in items[other_name]],
                    ))

                tasks.append({
                    "domain_label": domain_label,
                    "facet_name": facet_name,
                    "attributes": list(menu),
                    "rows": rows,
                    "neighbour_block": build_neighbour_block(neighbours),
                })
        return tasks

    def _attribute_refinement_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            prompt = build_attribute_refinement_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                facet_name=task["facet_name"],
                facet_definition=task["facet_definition"],
                attributes_block=build_attribute_contents_block(task["rows"]),
                neighbour_block=task["neighbour_block"],
            )
            self._capture(
                f"attribute_refinement_{task['domain_label']}_{task['facet_name']}",
                prompt, "attribute_refinement",
                {"model": self._model["attribute_refinement"],
                 "temperature": 0.0,
                 "max_tokens": self._max_tokens_consolidation,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "facet": task["facet_name"],
                 "n_attributes": len(task["attributes"]),
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": AttributeRefinementResult,
                "temperature": 0.0,
                "max_tokens": self._max_tokens_consolidation,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["attribute_refinement"],
                    phase="classifier_attribute_refinement"),
            }
        return prepare_fn

    @staticmethod
    def _attribute_refinement_parse_fn():
        def parse_fn(task: Dict, response) -> Optional[AttributeRefinementResult]:
            return response if response else None
        return parse_fn

    @staticmethod
    def _attribute_refinement_fallback_fn():
        """On failure the facet is left exactly as consolidation settled it."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_attribute_refinement(
        self,
        *,
        tasks: List[Dict],
        results: List,
        attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]],
        assignments: Dict[str, str],
        facet_assignments: Dict[str, Dict[str, str]],
        labels: Dict[Tuple[str, str], Dict[str, str]],
        verbose: bool,
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        """Apply every refinement result, then remap ideas — structure first.

        Order matters and is deliberate:
          1. rebuild the structure for every facet, so move targets resolve
             against the FINAL names rather than the names a concurrent call
             has renamed
          2. splits — route by exact response text to a named child
          3. merges — wholesale rename of a source attribute
          4. moves  — per text, may cross a facet boundary; the structure does
                      not follow, so no other idea is dragged along
        """
        # Keep the pre-refinement attributes reachable, so the self-check below
        # can put back a node the phase dropped without ever naming it.
        pre_attrs: Dict[Tuple[str, str], List[ConsolidatedAttribute]] = {
            (t["domain_label"], t["facet_name"]): list(t["attributes"]) for t in tasks
        }

        # ---- 1. structure ----------------------------------------------------
        # scope key is (domain, facet): attribute names are not unique across facets
        remap: Dict[Tuple[str, str, str], str] = {}
        splits: Dict[Tuple[str, str, str, str], str] = {}
        moves: Dict[Tuple[str, str, str, str], Optional[str]] = {}
        renamed_to: Dict[str, str] = {}             # normalised old name -> new, global
        split_children: Dict[str, List[str]] = {}   # normalised old name -> children

        for task, result in zip(tasks, results):
            dom, fac = task["domain_label"], task["facet_name"]
            before = [a.attribute_name for a in task["attributes"]]

            if result is None or not result.attributes:
                self._action_log.append({
                    "action": "attribute_refinement_failed", "domain": dom,
                    "facet": fac,
                    "note": "no result — facet left as consolidation settled it",
                    "attributes_before": before})
                continue

            # Match the model's source names against the real ones case- and
            # padding-insensitively. A strict equality check silently dropped
            # sources differing only in capitalisation, leaving their ideas on a
            # name no longer present in the structure.
            by_norm = {_norm(b): b for b in before}

            def _resolve(src: str) -> Optional[str]:
                return by_norm.get(_norm(src))

            unmatched = sorted({
                s for item in result.attributes
                for s in (item.source_attributes or [])
                if _resolve(s) is None
            })
            if unmatched:
                self._action_log.append({
                    "action": "unknown_source_name", "domain": dom, "facet": fac,
                    "sources": unmatched,
                    "note": "named as a source but not among this facet's attributes"})

            # A source claimed by more than one returned attribute is only
            # routable when the claimants carry instance_texts (a split).
            # Without that the remap would let the last writer win and move the
            # whole bucket.
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
                self._action_log.append({
                    "action": "unroutable_claim", "domain": dom, "facet": fac,
                    "sources": sorted(contested),
                    "note": "claimed by several returned attributes with no "
                            "instance_texts — ideas left on the source"})

            settled: List[ConsolidatedAttribute] = []
            consumed: Set[str] = set()
            for item in result.attributes:
                settled.append(ConsolidatedAttribute(
                    attribute_name=item.attribute_name,
                    attribute_definition=item.attribute_definition,
                    boundary_test=item.boundary_test,
                    exclusions=item.exclusions,
                    example_observations=item.example_observations,
                    source_attributes=item.source_attributes,
                ))

                sources = [r for r in (_resolve(s) for s in (item.source_attributes or []))
                           if r is not None]
                consumed.update(sources)
                if item.action == "split" and item.instance_texts:
                    for src in (sources or before):
                        for txt in item.instance_texts:
                            splits[(dom, fac, src, _norm(txt))] = item.attribute_name
                        split_children.setdefault(
                            _norm(src), []).append(item.attribute_name)
                    self._action_log.append({
                        "action": "split", "domain": dom, "facet": fac,
                        "into": item.attribute_name, "sources": sources,
                        "n_texts": len(item.instance_texts),
                        "texts": item.instance_texts})
                else:
                    for src in sources:
                        if src != item.attribute_name and src not in contested:
                            remap[(dom, fac, src)] = item.attribute_name
                            # Facets refine concurrently, so a move may name a
                            # target by the name it had in the neighbour block
                            # while its own facet is renaming it. Keep the trail.
                            renamed_to[_norm(src)] = item.attribute_name
                    if item.action in ("merge", "widen") or (
                            sources and sources != [item.attribute_name]):
                        self._action_log.append({
                            "action": item.action, "domain": dom, "facet": fac,
                            "result": item.attribute_name, "sources": sources})

            # Sources never claimed: keep them, or their ideas point at a name
            # absent from the structure.
            returned = {_norm(a.attribute_name) for a in settled}
            for attribute in task["attributes"]:
                if (attribute.attribute_name in consumed
                        or _norm(attribute.attribute_name) in returned):
                    continue
                settled.append(attribute)
                self._action_log.append({
                    "action": "attribute_kept_unclaimed_in_refinement",
                    "domain": dom, "facet": fac,
                    "attribute": attribute.attribute_name})

            attributes.setdefault(dom, {})[fac] = settled

            for m in (result.misfits or []):
                for txt in (m.instance_texts or []):
                    moves[(dom, fac, m.from_attribute, _norm(txt))] = (
                        m.target_attribute if m.verdict == "move" else None)
                self._action_log.append({
                    "action": f"misfit_{m.verdict}", "domain": dom, "facet": fac,
                    "from_attribute": m.from_attribute,
                    "target": m.target_attribute,
                    "n_texts": len(m.instance_texts or []),
                    "texts": m.instance_texts, "reason": m.reason})

        # A source split into exactly ONE child was renamed, not divided — follow
        # it. A source split into several children is genuinely ambiguous as a
        # move target: picking a child would be a guess, so those are reported.
        split_ambiguous: Dict[str, List[str]] = {}
        for src, children in split_children.items():
            uniq = sorted(set(children))
            if len(uniq) == 1:
                renamed_to.setdefault(src, uniq[0])
            else:
                split_ambiguous[src] = uniq

        # Where does every surviving attribute live now?
        home: Dict[str, Tuple[str, str]] = {}
        ambiguous: Set[str] = set()
        for dom, items in attributes.items():
            for fac, attrs in items.items():
                for a in attrs:
                    if a.attribute_name in home and home[a.attribute_name] != (dom, fac):
                        ambiguous.add(a.attribute_name)
                    home[a.attribute_name] = (dom, fac)

        # ---- 2-4. ideas ------------------------------------------------------
        idea_home: Dict[str, Tuple[str, str]] = {}
        text_of: Dict[str, str] = {}
        for (dom, fac), scoped in labels.items():
            for idea_id, text in scoped.items():
                idea_home[idea_id] = (dom, fac)
                text_of[idea_id] = _norm(text)

        n_split = n_remap = n_moved = n_out = n_unresolved = n_target_split = 0
        unresolved_targets: Counter = Counter()
        for idea_id, current in list(assignments.items()):
            place = idea_home.get(idea_id)
            if not place:
                continue
            dom, fac = place
            txt = text_of.get(idea_id, "")

            key = (dom, fac, current, txt)
            if key in moves:
                target = moves[key]
                if target is None:
                    n_out += 1                  # flagged contentless; left in place
                    continue
                if target not in home:
                    target = renamed_to.get(_norm(target), target)
                if target in home and target not in ambiguous:
                    assignments[idea_id] = target
                    t_dom, t_fac = home[target]
                    facet_assignments.setdefault(t_dom, {})[idea_id] = t_fac
                    if t_dom != dom:
                        facet_assignments.get(dom, {}).pop(idea_id, None)
                    n_moved += 1
                elif _norm(target) in split_ambiguous:
                    n_target_split += 1   # target was divided; a child would be a guess
                    unresolved_targets[target] += 1
                else:
                    n_unresolved += 1
                    unresolved_targets[target] += 1
                continue

            if key in splits:
                assignments[idea_id] = splits[key]
                n_split += 1
                continue

            if (dom, fac, current) in remap:
                assignments[idea_id] = remap[(dom, fac, current)]
                n_remap += 1

        # ---- 5. self-check: no idea may point at a node that does not exist ---
        orphans: Counter = Counter()
        for idea_id, name in assignments.items():
            if name and name not in home:
                orphans[(idea_home.get(idea_id, ("?", "?")), name)] += 1

        restored = 0
        for (place, name), _count in orphans.items():
            dom, fac = place
            source = next((a for a in (pre_attrs.get((dom, fac)) or [])
                           if a.attribute_name == name), None)
            if source is None or dom not in attributes:
                continue
            attrs = attributes[dom].setdefault(fac, [])
            if all(a.attribute_name != name for a in attrs):
                attrs.append(source)
            home[name] = (dom, fac)
            restored += 1
        if orphans:
            self._action_log.append({
                "action": "orphaned_assignment", "restored_nodes": restored,
                "ideas_affected": sum(orphans.values()),
                "attributes": sorted({n for (_, n) in orphans}),
                "note": ("refinement returned no attribute claiming these, so "
                         "their ideas kept a name absent from the structure; "
                         "the node was put back to keep the two consistent")})
            if verbose:
                print(f"    SELF-CHECK: {sum(orphans.values())} ideas pointed at "
                      f"{len(orphans)} attribute(s) missing from the structure — "
                      f"{restored} node(s) restored")

        self._action_log.append({
            "action": "_totals", "ideas_split": n_split, "ideas_remapped": n_remap,
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

        return assignments, facet_assignments

    async def _run_attribute_refinement(
        self,
        ctx: PromptContext,
        attributes: Dict[str, Dict[str, List[ConsolidatedAttribute]]],
        assignments: Dict[str, str],
        facet_assignments: Dict[str, Dict[str, str]],
        facets: Dict[str, List[ConsolidatedFacet]],
        labels: Dict[Tuple[str, str], Dict[str, str]],
        verbose: bool,
    ) -> Tuple[Dict[str, Dict[str, List[ConsolidatedAttribute]]], Dict[str, str],
               Dict[str, Dict[str, str]]]:
        """Judge each facet's attributes against what they actually hold."""
        if verbose:
            print("\n  Attribute refinement")
        started = time.time()

        definition_of = {
            (domain_label, facet.facet_name): facet.facet_definition
            for domain_label, items in facets.items() for facet in items
        }
        tasks = self._build_attribute_refinement_tasks(
            ctx, attributes, assignments, labels)
        for task in tasks:
            task["facet_definition"] = definition_of.get(
                (task["domain_label"], task["facet_name"]), "")

        results = await self._dispatch(
            "attribute_refinement", tasks,
            self._attribute_refinement_prepare_fn(ctx),
            self._attribute_refinement_parse_fn(),
            self._attribute_refinement_fallback_fn(),
            verbose,
        )
        assignments, facet_assignments = self._apply_attribute_refinement(
            tasks=tasks, results=results, attributes=attributes,
            assignments=assignments, facet_assignments=facet_assignments,
            labels=labels, verbose=verbose,
        )

        if verbose:
            s = self._last_stats
            total = sum(len(a) for f in attributes.values() for a in f.values())
            print(f"    {len(tasks)} tasks, {time.time() - started:.1f}s "
                  f"({s.get('tasks_successful', 0)} ok, "
                  f"{s.get('timeouts', 0)} timeouts) → {total} attributes")
        return attributes, assignments, facet_assignments

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
