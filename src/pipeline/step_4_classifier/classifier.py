"""Taxonomy Classifier: inductive taxonomy discovery, seven phases.

Facets (L3) and attributes (L4) are FOUND together rather than as two stacked
layers, and SETTLED apart — one call per level, so each merge judgement is made
with everything it compares in view and nothing else:

  discovery               per (domain, chunk)  facets WITH their attributes
  facet_consolidation     per domain           settle the facets, pool the rest
  attribute_consolidation per settled facet    settle that facet's pool
  assignment              per unique label     one attribute; the facet follows
  refinement              per domain           judge it on real contents
  cross_domain            everything at once   fold duplicates across domains
  valence_merge           see valence_consolidator.py

Phases are named by function, never by number: renumbering cold-started the
perf model and stranded config keys, twice.

Three things hold across every phase:

  * **The domain is fixed.** No phase but `cross_domain` may move a concept out
    of the domain it was found in. Per-idea (domain, facet) is DERIVED from
    where the attribute lives, so a structural relocation drags every idea in
    the bucket along at once — which is exactly what `cross_domain` is for, and
    why nothing else is allowed to do it.
  * **The facet is not fixed.** Inside one domain, refinement may move an
    attribute to another facet. The domain stays the same, so the ideas simply
    get relabelled to the facet where their attribute now lives.
  * **Every idea gets an attribute.** The menu always carries a catch-all per
    facet and a catch-all facet per domain, so there is no `__UNASSIGNED__` and
    nothing downstream needs an exception for a name absent from the structure.

Internally the settled structure is plain dicts — a list of facet dicts per
domain, each carrying its `attributes`. The Pydantic models exist only to
validate what an LLM returned; converting once at the phase boundary keeps the
catch-alls (which are built, not proposed) the same kind of thing as everything
else.

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
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
from .drains import (
    is_drain_item, make_drain_attribute, make_drain_facet, strip_empty_drains,
)
from .assignment_batching import group_label_reps
from models import DomainSet
from .prompts_shared import build_cross_scope_model
from .prompts_discovery import (
    DiscoveredAttribute, DiscoveredFacet, DiscoveryResult,
    build_discovery_prompt,
)
from .prompts_consolidation import (
    AttributeConsolidationResult, FacetConsolidationResult, FacetPool,
    build_attribute_candidate_block, build_attribute_candidate_index,
    build_attribute_consolidation_prompt, build_facet_candidate_block,
    build_facet_candidate_index, build_facet_consolidation_prompt,
)
from .prompts_assignment import (
    build_assignment_menu, build_assignment_model, build_assignment_prompt,
)
from .prompts_refinement import (
    RefinementResult, build_contents_block, build_cross_domain_prompt,
    build_refinement_prompt,
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


_ENUMERATION = re.compile(r"^\s*\d+\.\s*")


def _strip_enumeration(text: str) -> str:
    """Remove the list number the observation block put in front of a line.

    Discovery renders its input as `f"{i}. {obs}"` so the scratchpad can point
    at specific observations. The model then copies a whole line into
    `example_observations` — it was told to use the exact observation text, and
    the number is part of what it was shown. That rendering artefact became
    data: `"6. investeert in natuur → …"` travelled through consolidation into
    the codebook.

    Stripped here rather than only asked for in the prompt, because compliance
    is not something to depend on for a purely mechanical repair.
    """
    return _ENUMERATION.sub("", text).strip()


def facet_dicts(nested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The facet cards of a domain, without their nested attributes.

    The cache keeps facets and attributes in two registers, so the nesting that
    carries the structure through the run is unpacked at the end rather than
    stored twice.
    """
    return [{k: v for k, v in facet.items() if k != "attributes"}
            for facet in nested]


def attribute_dicts(
    nested: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """The attributes of a domain, keyed by the facet they sit under."""
    return {facet["facet_name"]: list(facet.get("attributes") or [])
            for facet in nested}


def count_structure(structure: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    """Facets and attributes, with the catch-alls counted separately.

    Every phase prints through this count, because a phase that counts the
    catch-alls next to a phase that does not reads as growth that is not there.
    On 2026-08-13 cross-domain therefore appeared to take 93 attributes in and
    put 120 out, while it was 93 in and 88 out — the rest were the catch-alls
    that same phase reattaches.
    """
    facets = drain_facets = attributes = drain_attributes = 0
    for cards in structure.values():
        for facet in cards:
            if is_drain_item(facet):
                drain_facets += 1
            else:
                facets += 1
            for attribute in facet.get("attributes") or []:
                if is_drain_item(attribute):
                    drain_attributes += 1
                else:
                    attributes += 1
    return {"facets": facets, "drain_facets": drain_facets,
            "attributes": attributes, "drain_attributes": drain_attributes}


def format_counts(structure: Dict[str, List[Dict[str, Any]]]) -> str:
    """One line stating what is there, without hiding the catch-alls."""
    c = count_structure(structure)
    line = f"{c['facets']} facets, {c['attributes']} attributes"
    if c["drain_facets"] or c["drain_attributes"]:
        line += (f" (+{c['drain_facets']} catch-all facets, "
                 f"{c['drain_attributes']} catch-all attributes)")
    return line


def derive_facet_assignments(
    attribute_assignments: Dict[str, str],
    structure: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, str]]:
    """Which facet each idea sits in, read off where its attribute lives.

    One source of truth. Two separately determined assignments could put an
    idea in facet F and in an attribute that hangs under G; here that cannot be
    expressed.
    """
    home: Dict[str, Tuple[str, str]] = {}
    for domain, facets in structure.items():
        for facet in facets:
            for attribute in facet.get("attributes") or []:
                home[_norm(attribute["attribute_name"])] = (
                    domain, facet["facet_name"])

    out: Dict[str, Dict[str, str]] = {}
    for idea_id, attribute_name in attribute_assignments.items():
        found = home.get(_norm(attribute_name))
        if found is None:
            continue
        domain, facet_name = found
        out.setdefault(domain, {})[idea_id] = facet_name
    return out


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
    """Output of the five in-classifier phases (valence merge runs after)."""
    partition_n_labels: Dict[str, int]
    partition_n_batches: Dict[str, int]
    partition_facets: Dict[str, List[Dict[str, Any]]]
    partition_assignments: Dict[str, Dict[str, str]]  # domain -> {idea_id -> facet_name}
    partition_attributes: Dict[str, Dict[str, List[Dict[str, Any]]]]  # domain -> {facet -> [attrs]}
    attribute_assignments: Dict[str, str]  # idea_id -> attribute_name
    # Discovery snapshots, taken before consolidation settles the inventory.
    # The state before a merge is what makes a bad merge diagnosable afterwards.
    partition_raw_facets: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    raw_partition_attributes: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)
    raw_attribute_assignments: Dict[str, str] = field(default_factory=dict)
    # Assignment confidence scores (0.0-1.0)
    facet_confidence: Dict[str, float] = field(default_factory=dict)
    attribute_confidence: Dict[str, float] = field(default_factory=dict)
    # Assignment valence (+, -, 0). Only the attribute level is filled: valence
    # is judged relative to the chosen attribute, and inventing a facet-level
    # one would simulate a judgment the model never gave. The cascade downstream
    # is attribute > facet > step 3, so the attribute value is what wins anyway.
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

    Six phases run here; the seventh, the valence-neutral merge, runs afterwards
    from the runner (see `valence_consolidator.py`).

    Every phase is one method with an explicit signature, plus a pure
    `_build_<phase>_tasks` that decides the task shape. The orchestrator is the
    sequence of those calls and nothing else.
    """

    # Phase key → (config attribute, cost-tracker label). One table, so a phase
    # cannot end up with a model in one register and a different one in another.
    PHASES = (
        "discovery", "facet_consolidation", "attribute_consolidation",
        "assignment", "refinement", "cross_domain", "valence_merge",
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
        self._max_tokens_discovery = config.qr_max_tokens_discovery
        self._max_tokens_consolidation = config.qr_max_tokens_consolidation
        self._max_tokens_assignment = config.qr_max_tokens_assignment
        self._contents_top_n = config.contents_top_n

        # Chunking — discovery input, per domain
        self._batch_size_min = config.batch_size_min
        self._batch_size_max = config.batch_size_max
        self._target_batches = config.target_batches
        self._chunk_overlap = config.chunk_overlap

        # Consolidation — how much fits in one call, and how often to round-trip
        self._consolidation_max_rounds = config.consolidation_max_rounds
        self._facet_consolidation_max_facets_per_call = (
            config.facet_consolidation_max_facets_per_call)
        self._attribute_consolidation_max_attributes_per_call = (
            config.attribute_consolidation_max_attributes_per_call)

        # Label source for observation formatting
        self._label_source = config.label_source
        self._label_prefix = config.label_prefix

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
        # orchestrator reads as the phase calls it is, and the task builders
        # stay pure because only the run methods touch this.
        self._action_log: List[Dict] = []
        # How often an independent chunk proposed each candidate — the only
        # prevalence that exists before assignment. Kept per level, because
        # rule 2 is applied at both: consolidation groups facets on it and
        # step 6 groups attributes on it.
        self._recurrence: Dict[str, Dict[str, int]] = {}
        self._attr_recurrence: Dict[str, Dict[str, int]] = {}
        self._passes: Dict[str, int] = {}
        self._last_stats: Dict = {}

        # Assignment confidence and valence (populated by the assignment parse)
        self._facet_confidence: Dict[str, float] = {}
        self._attribute_confidence: Dict[str, float] = {}
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

        A dimension that cannot be resolved raises. Every prompt in step 4 shows
        this dimension's own four-level taxonomy, worded in its own terms;
        without it there is no generic version to fall back on, and a phase
        built around the wrong wording produces plausible output that is wrong
        all the way down.
        """
        dimension_def = get_dimension(dimension_name) if dimension_name else None
        if dimension_def is None:
            raise ValueError(
                f"No DimensionDefinition for primary_dimension "
                f"{dimension_name!r}. Step 4 builds every prompt around this "
                f"dimension's own wording; see dimension_data.py."
            )
        if verbose:
            print(f"  Dimension: {dimension_name}")

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
        """Run the five in-classifier phases.

        `extraction_metadata` (models.ExtractionMetadata) identifies the two
        standing drain domains by key (taxonomy_health.drain_domains); those get
        no facets, because step 3 defines them as deliberately broad catch-alls.
        """
        print(f"\n{'='*70}")
        print("TAXONOMY DISCOVERY (5 phases)")
        print(f"{'='*70}")

        prompt_context, active_partitions = self._prepare_context(
            label_mappings, partition_set, survey_question, language,
            dataset_context, dimension_name, dimension_description,
            extraction_metadata, verbose,
        )

        if verbose:
            total_labels = sum(m.label_count for m in active_partitions.values())
            total_ideas = sum(len(m.ideas) for m in active_partitions.values())
            print(f"  Processing {len(active_partitions)} domains concurrently "
                  f"({total_labels} observations, {total_ideas} ideas)")
            print("  discovery → consolidation → assignment → refinement "
                  "→ cross-domain")

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

        representatives: Dict[Tuple, str] = {}
        for model in phase_models:
            representatives.setdefault(route_key(model), model)

        if verbose:
            print(f"  Fetching rate limits from API "
                  f"({len(representatives)} deployment(s))...")

        # `fetch_rate_limits` already returns (RateLimits, has_headers), so the
        # probe result is unpacked, never wrapped again.
        probes = await asyncio.gather(
            *(llm_fetch_rate_limits(m) for m in representatives.values()))
        by_route = dict(zip(representatives.keys(), probes))

        for model in set(phase_models):
            limits, has_headers = by_route[route_key(model)]
            if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
                if verbose:
                    print(f"  WARNING: {model}: using fallback rate limits "
                          f"(TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
                limits = RateLimits(
                    tokens_per_minute=FALLBACK_TPM,
                    requests_per_minute=FALLBACK_RPM,
                )
            self._limits_by_model[model] = limits
            self._has_headers_by_model[model] = has_headers

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
        """The six phases that run here, in order. Each one logs its own lines
        and its own cost; this method only decides what runs and what feeds
        what. The seventh, the valence merge, runs from the runner.

        `stop_after_phase` returns the state as it stands after that phase. A
        partial return is a real result, not an empty one — the phases that did
        run are in it, so the runner can still cache and inspect them.
        """
        started = time.time()
        self._facet_confidence.clear()
        self._attribute_confidence.clear()
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

        raw = await self._run_discovery(ctx, verbose)
        raw_nested = {d: [f.model_dump() for f in items] for d, items in raw.items()}
        state["partition_raw_facets"] = {
            d: facet_dicts(items) for d, items in raw_nested.items()}
        state["raw_partition_attributes"] = {
            d: attribute_dicts(items) for d, items in raw_nested.items()}
        if _stop("discovery"):
            return self._taxonomy_result(state, raw_nested, {}, started, verbose)

        settled = await self._run_facet_consolidation(ctx, raw, verbose)
        if _stop("facet_consolidation"):
            structure = self._assemble_structure(settled, {})
            return self._taxonomy_result(state, structure, {}, started, verbose)

        structure = await self._run_attribute_consolidation(ctx, settled, verbose)
        structure = self._add_drains(ctx, structure)
        if _stop("attribute_consolidation"):
            return self._taxonomy_result(state, structure, {}, started, verbose)

        assignments = await self._run_assignment(ctx, structure, labels, verbose)
        # The state before refinement remaps anything. This is what makes a bad
        # merge diagnosable afterwards.
        state["raw_attribute_assignments"] = dict(assignments)
        if _stop("assignment"):
            return self._taxonomy_result(state, structure, assignments, started, verbose)

        structure, assignments = await self._run_refinement(
            ctx, structure, assignments, labels, verbose)
        if _stop("refinement"):
            return self._taxonomy_result(state, structure, assignments, started, verbose)

        structure, assignments = await self._run_cross_domain(
            ctx, structure, assignments, verbose)

        return self._taxonomy_result(state, structure, assignments, started, verbose)

    def _taxonomy_result(
        self,
        state: Dict,
        structure: Dict[str, List[Dict[str, Any]]],
        assignments: Dict[str, str],
        started: float,
        verbose: bool,
    ) -> TaxonomyResult:
        """Unpack the nested structure into the two registers the cache keeps.

        Empty catch-alls are dropped here and nowhere earlier: a catch-all is an
        offer, and whether it caught anything is only known once every phase has
        had its say.
        """
        facets = {d: facet_dicts(items) for d, items in structure.items()}
        attributes = {d: attribute_dicts(items) for d, items in structure.items()}
        facets, attributes, assignments = strip_empty_drains(
            facets, attributes, assignments)

        state["partition_facets"] = facets
        state["partition_attributes"] = attributes
        state["attribute_assignments"] = assignments
        state["partition_assignments"] = derive_facet_assignments(
            assignments, structure)

        if verbose:
            print(f"\n  Taxonomy complete in {time.time() - started:.1f}s")
        return TaxonomyResult(
            **state,
            facet_confidence=self._facet_confidence,
            attribute_confidence=self._attribute_confidence,
            attribute_valence=self._attribute_valence,
            consolidation_log=list(self._action_log),
        )

    # =========================================================================
    # DISPATCH AND SHARED PLUMBING
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

    def _compute_batch_size(self, n_labels: int) -> int:
        """Adaptive chunk size for discovery input."""
        if n_labels <= self._batch_size_min:
            return n_labels
        ideal = max(n_labels // self._target_batches, 1)
        return max(self._batch_size_min, min(ideal, self._batch_size_max))

    def _create_batches(self, labels: List[str]) -> List[List[str]]:
        """Split observations into overlapping chunks.

        Each chunk overlaps the previous by `chunk_overlap * batch_size`, so a
        concept that straddles a boundary is seen whole by at least one call.
        """
        batch_size = self._compute_batch_size(len(labels))
        if len(labels) <= batch_size:
            return [labels]

        step = max(batch_size - int(batch_size * self._chunk_overlap), 1)
        batches = []
        i = 0
        while i < len(labels):
            batches.append(labels[i:i + batch_size])
            i += step
            if i < len(labels) and i + batch_size > len(labels):
                batches.append(labels[-batch_size:])
                break
        return batches

    def _add_drains(
        self, ctx: PromptContext, structure: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Give every domain its catch-alls, once the inventory has settled.

        Added after consolidation rather than before: consolidation judges what
        the passes proposed, and a bucket that exists by construction is not a
        proposal. Added before assignment, because that is the phase that needs
        somewhere to put what fits nowhere.
        """
        out: Dict[str, List[Dict[str, Any]]] = {}
        for domain, facets in structure.items():
            with_drains = []
            for facet in facets:
                facet = dict(facet)
                facet["attributes"] = list(facet.get("attributes") or []) + [
                    make_drain_attribute(facet["facet_name"], ctx.language)]
                with_drains.append(facet)
            with_drains.append(make_drain_facet(domain, ctx.language))
            out[domain] = with_drains
        return out

    # =========================================================================
    # PHASE — DISCOVERY (per domain, chunked; facets with their attributes)
    # =========================================================================

    def _build_discovery_tasks(self, ctx: PromptContext) -> List[Dict]:
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

    def _discovery_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            prompt = build_discovery_prompt(
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
                    f"discovery_{task['domain_label']}", prompt, "discovery",
                    {"model": self._model["discovery"],
                     "temperature": self._temperature,
                     "max_tokens": self._max_tokens_discovery,
                     "language": ctx.language,
                     "domain": task["domain_label"],
                     "total_chunks": task["total_chunks"],
                     "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": DiscoveryResult,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens_discovery,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["discovery"], phase="classifier_discovery"),
            }
        return prepare_fn

    @staticmethod
    def _discovery_parse_fn():
        """Facets as proposed, with the list numbering taken back off the examples.

        The one place where examples enter the step, so the one place this has
        to happen: everything downstream carries them over rather than reading
        the observation block again.
        """
        def parse_fn(task: Dict, response) -> List[DiscoveredFacet]:
            if not response:
                return []
            for facet in response.facets:
                for attribute in facet.attributes:
                    attribute.example_observations = [
                        cleaned for cleaned in (
                            _strip_enumeration(t)
                            for t in attribute.example_observations)
                        if cleaned]
            return list(response.facets)
        return parse_fn

    @staticmethod
    def _discovery_fallback_fn():
        def fallback_fn(task: Dict, reason: str) -> List[DiscoveredFacet]:
            return []
        return fallback_fn

    async def _run_discovery(
        self, ctx: PromptContext, verbose: bool,
    ) -> Dict[str, List[DiscoveredFacet]]:
        """Propose facets with their attributes, then collapse exact repeats.

        Every chunk rediscovers largely the same structure, so the flattened
        yield holds byte-identical re-proposals. Only those are removed here;
        near-duplicates are a judgment and belong to consolidation, which sees
        each candidate together with how many passes proposed it.
        """
        if verbose:
            print("\n  Discovery")
        started = time.time()

        tasks = self._build_discovery_tasks(ctx)
        results = await self._dispatch(
            "discovery", tasks,
            self._discovery_prepare_fn(ctx),
            self._discovery_parse_fn(),
            self._discovery_fallback_fn(),
            verbose,
        )

        flat: Dict[str, List[DiscoveredFacet]] = {}
        for task, result in zip(tasks, results):
            flat.setdefault(task["domain_label"], []).extend(result or [])

        passes = Counter(t["domain_label"] for t in tasks)
        raw: Dict[str, List[DiscoveredFacet]] = {}
        for label in sorted(flat):
            seen = Counter(_norm(f.facet_name) for f in flat[label])
            # Counted domain-wide and before the dedup, exactly as facets are:
            # the same attribute proposed under two different facets is still
            # one concept two passes put forward.
            attr_seen = Counter(_norm(a.attribute_name)
                                for f in flat[label] for a in f.attributes)
            deduped = dedup_exact_facets(flat[label])
            self._recurrence[label] = {
                f.facet_name: seen[_norm(f.facet_name)] for f in deduped}
            self._attr_recurrence[label] = {
                a.attribute_name: attr_seen[_norm(a.attribute_name)]
                for f in deduped for a in f.attributes}
            self._passes[label] = passes[label]
            if len(deduped) < len(flat[label]):
                self._action_log.append({
                    "action": "facet_exact_dedup", "domain": label,
                    "before": len(flat[label]), "after": len(deduped)})
            raw[label] = deduped

        if verbose:
            s = self._last_stats
            n_attrs = sum(len(f.attributes) for v in raw.values() for f in v)
            print(f"    {len(tasks)} tasks, {time.time() - started:.1f}s "
                  f"({s.get('tasks_successful', 0)} ok, "
                  f"{s.get('timeouts', 0)} timeouts, "
                  f"{s.get('recovered', 0)} retries) → "
                  f"{sum(len(f) for f in raw.values())} candidate facets, "
                  f"{n_attrs} candidate attributes")
            for label in sorted(raw):
                inner = sum(len(f.attributes) for f in raw[label])
                print(f"      {label}: {len(raw[label])} facets, {inner} attributes")
        return raw

    # =========================================================================
    # PHASE — FACET CONSOLIDATION (per domain, before any attribute is settled)
    # =========================================================================

    def _facet_consolidation_groups(
        self, candidates: List[FacetPool],
    ) -> List[List[FacetPool]]:
        """Split a domain's facet candidates into groups that fit in one call.

        Counted in FACETS. Its predecessor counted attributes, because the
        candidate block rendered every attribute in full and the prompt grew
        with them; this call lists attribute names only, so what bounds the
        judgement is the number of facets being compared.

        Sorted by normalised name first, so near-identical proposals sit next
        to each other and usually land in the same group instead of missing
        each other for a round. Usually, not always: a fixed-size split can
        still put the boundary between two neighbours. Accepted — the next
        round puts the survivors together anyway.
        """
        cap = self._facet_consolidation_max_facets_per_call
        ordered = sorted(candidates, key=lambda p: _norm(p.facet_name))
        groups: List[List[FacetPool]] = []
        for i in range(0, len(ordered), cap):
            groups.append(ordered[i:i + cap])
        return groups or [[]]

    def _build_facet_consolidation_tasks(
        self, ctx: PromptContext, pending: Dict[str, List[FacetPool]],
    ) -> List[Dict]:
        """One task per domain that has candidates, split when it holds too many."""
        tasks: List[Dict] = []
        for label in sorted(pending):
            candidates = pending[label]
            if not candidates:
                continue
            for group in self._facet_consolidation_groups(candidates):
                if not group:
                    continue
                tasks.append({
                    "domain_label": label, "candidates": group,
                    "recurrence": self._recurrence.get(label) or {},
                    "n_passes": self._passes.get(label, 0)})
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
                domain_exclusions=domain["exclusions"],
                candidate_block=build_facet_candidate_block(
                    task["candidates"], task["recurrence"], task["n_passes"]),
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
        def parse_fn(task: Dict, response):
            return response if response else None
        return parse_fn

    @staticmethod
    def _facet_consolidation_fallback_fn():
        """On failure the domain keeps its candidates — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _facet_consolidation_survivors(
        self, task: Dict, result,
    ) -> List[FacetPool]:
        """The settled facets of one group, with unclaimed candidates kept.

        A candidate that was merged and a candidate that was forgotten look
        identical in the answer: neither appears. `source_facet_ids` is what
        tells them apart, so whatever nobody claims is kept whole.

        Each survivor pools the attributes of the candidates it claimed —
        unconsolidated, exact repeats collapsed. Settling that pool is the next
        phase's job. A candidate claimed by two survivors pools into the first;
        pooling into both would let one attribute survive under two facets.
        """
        candidates = task["candidates"]
        if result is None or not result.facets:
            self._action_log.append({
                "action": "facet_consolidation_failed",
                "domain": task["domain_label"],
                "note": "kept every candidate"})
            return list(candidates)

        cand_facets = build_facet_candidate_index(candidates)
        self._log_unknown_facet_sources(task["domain_label"], result, cand_facets)

        survivors: List[FacetPool] = []
        taken: Set[str] = set()
        claimed_by: Dict[str, List[str]] = {}
        for facet in result.facets:
            pooled: List[DiscoveredAttribute] = []
            for source in (facet.source_facet_ids or []):
                source = source.strip()
                claimed_by.setdefault(source, []).append(facet.facet_name)
                candidate = cand_facets.get(source)
                if candidate is None or source in taken:
                    continue
                taken.add(source)
                pooled.extend(candidate.attributes)
            survivors.append(FacetPool(
                facet_name=facet.facet_name,
                facet_definition=facet.facet_definition,
                facet_question=facet.facet_question,
                attributes=dedup_exact_attributes(pooled)))

        for source, claimants in claimed_by.items():
            if len(claimants) > 1:
                self._action_log.append({
                    "action": "divided_source_facet",
                    "domain": task["domain_label"],
                    "source": source, "claimants": claimants,
                    "note": "split across survivors; attributes pooled into the first"})

        # The name fallback catches a survivor that kept a candidate's name
        # without citing its id. It only applies to a name that identifies one
        # candidate: where two candidates share it, the name says nothing about
        # which was meant, and letting it count would undo what the ids fixed.
        #
        # A candidate covered this way hands over its attributes, exactly as an
        # id-cited one does. `source_facet_ids` is the only channel from this
        # phase to the next, so treating the candidate as absorbed without
        # moving its pool would lose it for good.
        names = Counter(_norm(p.facet_name) for p in cand_facets.values())
        by_name: Dict[str, FacetPool] = {}
        for survivor in survivors:
            by_name.setdefault(_norm(survivor.facet_name), survivor)
        for facet_id, candidate in cand_facets.items():
            if facet_id in taken:
                continue
            name = _norm(candidate.facet_name)
            landing = by_name.get(name) if names[name] == 1 else None
            if landing is not None:
                landing.attributes = dedup_exact_attributes(
                    landing.attributes + candidate.attributes)
                self._action_log.append({
                    "action": "facet_claimed_by_name",
                    "domain": task["domain_label"],
                    "facet": candidate.facet_name, "id": facet_id,
                    "note": "survivor kept the name but cited no id; "
                            "attributes pooled into it"})
                continue
            survivors.append(candidate)
            self._action_log.append({
                "action": "facet_kept_unclaimed",
                "domain": task["domain_label"],
                "facet": candidate.facet_name, "id": facet_id})

        self._log_facet_provenance(task["domain_label"], result)
        self._action_log.append({
            "action": "facet_consolidation",
            "domain": task["domain_label"],
            "facets_before": len(candidates),
            "facets_after": len(survivors)})
        return survivors

    def _log_unknown_facet_sources(
        self, label: str, result, cand_facets: Dict[str, FacetPool],
    ) -> None:
        """Source ids the model cited that were never handed out.

        An invented id claims nothing, which silently turns the candidate it was
        meant to cover into an unclaimed one. On ids this is a clean check: the
        set was handed out in this very prompt.
        """
        unknown = sorted({s.strip() for f in result.facets
                          for s in (f.source_facet_ids or [])
                          if s.strip() not in cand_facets})
        if not unknown:
            return
        self._action_log.append({
            "action": "unknown_source_id", "domain": label, "facets": unknown,
            "note": "cited as a source but never handed out for this call"})

    def _log_facet_provenance(self, label: str, result) -> None:
        """Which candidate went where, and on what question.

        The `source_facet_ids` field dies with the phase — survivors are rebuilt
        as plain pools — so without this the log reports that forty facets
        became twelve but not which absorbed which, and that is exactly what
        separates a changed rule from a changed run.

        Two survivors stating the same question is a visible breach of rule 1.
        It is logged, not repaired: merging them here would overrule a judgement
        the model made with every candidate in view.
        """
        stated: Dict[str, str] = {}
        for facet in result.facets:
            key = _norm(facet.facet_question)
            if key and key in stated:
                self._action_log.append({
                    "action": "duplicate_facet_question", "domain": label,
                    "facets": [stated[key], facet.facet_name],
                    "question": facet.facet_question,
                    "note": "two survivors answer the same question — rule 1"})
            elif key:
                stated[key] = facet.facet_name

        self._action_log.append({
            "action": "facet_provenance", "domain": label,
            "facets": [
                {"facet": f.facet_name,
                 "facet_question": f.facet_question,
                 "source_facet_ids": list(f.source_facet_ids or [])}
                for f in result.facets],
            "decisions": list(result.decision_summary or [])})

    async def _run_facet_consolidation(
        self, ctx: PromptContext, raw: Dict[str, List[DiscoveredFacet]],
        verbose: bool,
    ) -> Dict[str, List[FacetPool]]:
        """Settle each domain's facet inventory before any attribute is settled.

        Rounds, not one shot: round one consolidates what the chunks proposed,
        round two the survivors of round one, until a domain fits one group.
        Each round's survivors carry the attribute pools of what they absorbed,
        so a round-two candidate arrives with everything under it.
        """
        if verbose:
            print("\n  Facet consolidation")
        started = time.time()

        settled: Dict[str, List[FacetPool]] = {}
        pending: Dict[str, List[FacetPool]] = {
            label: [FacetPool(facet_name=f.facet_name,
                              facet_definition=f.facet_definition,
                              facet_question="",
                              attributes=list(f.attributes))
                    for f in facets]
            for label, facets in raw.items() if facets
        }

        for round_no in range(1, self._consolidation_max_rounds + 1):
            # One candidate is nothing to merge: no call, keep the facet.
            for label in [l for l, c in pending.items() if len(c) == 1]:
                settled[label] = [pending.pop(label)[0]]

            tasks = self._build_facet_consolidation_tasks(ctx, pending)
            if not tasks:
                break

            groups_per_domain = Counter(t["domain_label"] for t in tasks)
            if verbose and any(n > 1 for n in groups_per_domain.values()):
                busy = {d: n for d, n in groups_per_domain.items() if n > 1}
                print(f"    round {round_no}: {busy} groups")

            results = await self._dispatch(
                "facet_consolidation", tasks,
                self._facet_consolidation_prepare_fn(ctx),
                self._facet_consolidation_parse_fn(),
                self._facet_consolidation_fallback_fn(),
                verbose,
            )

            survivors: Dict[str, List[FacetPool]] = {}
            for task, result in zip(tasks, results):
                survivors.setdefault(task["domain_label"], []).extend(
                    self._facet_consolidation_survivors(task, result))

            pending = {}
            for label, pools in survivors.items():
                if groups_per_domain[label] == 1:
                    settled[label] = pools
                else:
                    pending[label] = pools
            if not pending:
                break

        for label, leftover in pending.items():
            settled[label] = leftover
            self._action_log.append({
                "action": "consolidation_rounds_exhausted",
                "domain": label, "rounds": self._consolidation_max_rounds,
                "remaining": len(leftover)})

        if verbose:
            n_facets = sum(len(v) for v in settled.values())
            n_attrs = sum(len(p.attributes) for v in settled.values() for p in v)
            print(f"    {time.time() - started:.1f}s → {n_facets} facets, "
                  f"{n_attrs} pooled attributes")
            for label in sorted(settled):
                names = ", ".join(p.facet_name for p in settled[label])
                print(f"      {label}: {len(settled[label])} — {names}")
        return settled

    # =========================================================================
    # PHASE — ATTRIBUTE CONSOLIDATION (per settled facet, one call each)
    # =========================================================================

    def _attribute_consolidation_groups(
        self, attributes: List[DiscoveredAttribute],
    ) -> List[List[DiscoveredAttribute]]:
        """Split one facet's pool into groups that fit in one call.

        Counted in attributes, which is all this call holds. A facet whose pool
        fits — the normal case — gets one group and no rounds.

        Sorted by normalised name first, so near-identical proposals sit next
        to each other and usually land in the same group instead of missing
        each other for a round.
        """
        cap = self._attribute_consolidation_max_attributes_per_call
        ordered = sorted(attributes, key=lambda a: _norm(a.attribute_name))
        return [ordered[i:i + cap] for i in range(0, len(ordered), cap)] or [[]]

    def _build_attribute_consolidation_tasks(
        self, ctx: PromptContext, settled: Dict[str, List[FacetPool]],
    ) -> List[Dict]:
        """One task per (domain, facet) that has something to merge.

        A facet holding one attribute is skipped: there is nothing to fold. It
        is absent from the consolidated map afterwards, which is how the
        assembler knows to keep the pool as it stands.

        `facet_index` is the pool's position in the list this call was handed.
        In round one that list is `settled` and the index is the result key; from
        round two it is the shrunken `pending`, and the runner translates it back
        through `origin`. Either way the key is a position, never a name: two
        facets in one domain can carry the same name — the facet phase keeps
        both when the name is ambiguous — so a name-keyed result would hand one
        card the other's attributes.
        """
        tasks: List[Dict] = []
        for label in sorted(settled):
            for index, pool in enumerate(settled[label]):
                if len(pool.attributes) < 2:
                    continue
                for group in self._attribute_consolidation_groups(pool.attributes):
                    if not group:
                        continue
                    tasks.append({
                        "domain_label": label, "facet_index": index,
                        "facet": pool, "candidates": group,
                        "recurrence": self._attr_recurrence.get(label) or {},
                        "n_passes": self._passes.get(label, 0)})
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
                facet_name=facet.facet_name,
                facet_definition=facet.facet_definition,
                facet_question=facet.facet_question,
                candidate_block=build_attribute_candidate_block(
                    task["candidates"], task["recurrence"], task["n_passes"]),
            )
            self._capture(
                f"attribute_consolidation_{task['domain_label']}", prompt,
                "attribute_consolidation",
                {"model": self._model["attribute_consolidation"],
                 "temperature": 0.0,
                 "max_tokens": self._max_tokens_consolidation,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "facet": facet.facet_name,
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
        def parse_fn(task: Dict, response):
            return response if response else None
        return parse_fn

    @staticmethod
    def _attribute_consolidation_fallback_fn():
        """On failure the facet keeps its pool — never silently emptied."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _attribute_consolidation_survivors(
        self, task: Dict, result,
    ) -> List[DiscoveredAttribute]:
        """The settled attributes of one facet, with unclaimed candidates kept.

        Simpler than the facet net one level up, and structurally so: one call
        is one facet, so an unclaimed candidate has exactly one place to land.
        Its predecessor had to work out which survivor had absorbed the
        candidate's facet, and could fail to find one at all.

        The prompt tells the model never to drop an attribute. This is what
        makes that true when it does anyway, or when the call never returns.
        """
        candidates = task["candidates"]
        label, facet = task["domain_label"], task["facet"]
        if result is None or not result.attributes:
            self._action_log.append({
                "action": "attribute_consolidation_failed", "domain": label,
                "facet": facet.facet_name, "note": "kept every candidate"})
            return list(candidates)

        index = build_attribute_candidate_index(candidates)
        claimed = {s.strip() for a in result.attributes
                   for s in (a.source_attribute_ids or [])}
        unknown = sorted(s for s in claimed if s not in index)
        if unknown:
            self._action_log.append({
                "action": "unknown_source_id", "domain": label,
                "facet": facet.facet_name, "attributes": unknown,
                "note": "cited as a source but never handed out for this call"})

        survivors: List[DiscoveredAttribute] = [
            DiscoveredAttribute(
                attribute_name=a.attribute_name,
                attribute_definition=a.attribute_definition,
                example_observations=list(a.example_observations))
            for a in result.attributes]

        # The name fallback catches a survivor that kept a candidate's name
        # without citing its id. It only applies to a name that identifies one
        # candidate: where two candidates share it, the name says nothing about
        # which was meant, and letting it count would undo what the ids fixed.
        names = Counter(_norm(a.attribute_name) for a in index.values())
        returned = {_norm(a.attribute_name) for a in result.attributes}
        for attribute_id, candidate in index.items():
            name = _norm(candidate.attribute_name)
            if attribute_id in claimed or (
                    names[name] == 1 and name in returned):
                continue
            survivors.append(candidate)
            self._action_log.append({
                "action": "attribute_kept_unclaimed", "domain": label,
                "facet": facet.facet_name,
                "attribute": candidate.attribute_name, "id": attribute_id})

        self._action_log.append({
            "action": "attribute_provenance", "domain": label,
            "facet": facet.facet_name,
            "attributes": [
                {"attribute": a.attribute_name,
                 "source_attribute_ids": list(a.source_attribute_ids or [])}
                for a in result.attributes],
            "decisions": list(result.decision_summary or [])})
        self._action_log.append({
            "action": "attribute_consolidation", "domain": label,
            "facet": facet.facet_name,
            "attributes_before": len(candidates),
            "attributes_after": len(survivors)})
        return survivors

    @staticmethod
    def _assemble_structure(
        settled: Dict[str, List[FacetPool]],
        consolidated: Dict[Tuple[str, int], List[DiscoveredAttribute]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """The two phases folded back into the nested dicts the step carries.

        Keyed on the facet's position in its domain, not on its name: a domain
        can hold two facets with the same name, and a name-keyed lookup gives
        both of them whichever result was stored last — losing the attributes
        of the other.

        A facet with no entry held one attribute and was skipped — that is the
        only way to reach here without one. A failed call does produce an entry:
        the survivor net returns the whole pool.
        """
        out: Dict[str, List[Dict[str, Any]]] = {}
        for label, pools in settled.items():
            cards = []
            for index, pool in enumerate(pools):
                attributes = consolidated.get((label, index), pool.attributes)
                cards.append({
                    "facet_name": pool.facet_name,
                    "facet_definition": pool.facet_definition,
                    "facet_question": pool.facet_question,
                    "attributes": [a.model_dump() for a in attributes]})
            out[label] = cards
        return out

    async def _run_attribute_consolidation(
        self, ctx: PromptContext, settled: Dict[str, List[FacetPool]],
        verbose: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fold each settled facet's pool into its minimal set, one call each.

        Every facet is judged with all of its own attributes in view and none of
        anyone else's — the condition a merge judgement needs, and the one this
        work could not have as step six of the facet call.

        Rounds only where a single facet's pool exceeds the cap, which is rare:
        the pool of one facet is a fraction of a domain's.
        """
        if verbose:
            print("\n  Attribute consolidation")
        started = time.time()

        consolidated: Dict[Tuple[str, int], List[DiscoveredAttribute]] = {}
        pending: Dict[str, List[FacetPool]] = dict(settled)
        # Where each pending pool sits in `settled`, since `pending` shrinks
        # between rounds while the result keys must keep pointing at the
        # original positions. Positions, not names: a domain can hold two
        # facets called the same thing.
        origin: Dict[str, List[Tuple[str, int]]] = {
            label: [(label, i) for i in range(len(pools))]
            for label, pools in settled.items()}

        for round_no in range(1, self._consolidation_max_rounds + 1):
            tasks = self._build_attribute_consolidation_tasks(ctx, pending)
            if not tasks:
                # Nothing needs a call, so nothing is left in flight.
                pending = {}
                break

            groups_per_facet = Counter(
                origin[t["domain_label"]][t["facet_index"]] for t in tasks)
            if verbose and any(n > 1 for n in groups_per_facet.values()):
                busy = {f"{d} › {settled[d][i].facet_name}": n
                        for (d, i), n in groups_per_facet.items() if n > 1}
                print(f"    round {round_no}: {busy} groups")

            results = await self._dispatch(
                "attribute_consolidation", tasks,
                self._attribute_consolidation_prepare_fn(ctx),
                self._attribute_consolidation_parse_fn(),
                self._attribute_consolidation_fallback_fn(),
                verbose,
            )

            merged: Dict[Tuple[str, int], List[DiscoveredAttribute]] = {}
            for task, result in zip(tasks, results):
                key = origin[task["domain_label"]][task["facet_index"]]
                merged.setdefault(key, []).extend(
                    self._attribute_consolidation_survivors(task, result))

            consolidated.update(merged)
            # A facet split over several groups is not settled yet: its groups
            # never saw each other. Put the survivors back in as one pool, and
            # carry its position along so the next round writes the same key.
            cap = self._attribute_consolidation_max_attributes_per_call
            again: Dict[str, List[FacetPool]] = {}
            again_origin: Dict[str, List[Tuple[str, int]]] = {}
            for label, pools in pending.items():
                for index, pool in enumerate(pools):
                    key = origin[label][index]
                    kept = merged.get(key)
                    if kept is None or len(kept) <= cap:
                        continue
                    again.setdefault(label, []).append(FacetPool(
                        facet_name=pool.facet_name,
                        facet_definition=pool.facet_definition,
                        facet_question=pool.facet_question,
                        attributes=kept))
                    again_origin.setdefault(label, []).append(key)
            pending, origin = again, again_origin
            if not pending:
                break

        # A facet still pending here never got a call that brought it under the
        # cap. Its survivors are in `consolidated` and nothing is lost, but a
        # phase that ran out of rounds must say so — the facet half logs this,
        # and a diagnostic that exists on one side of the split only is what
        # misleads the next reader.
        for label, pools in pending.items():
            for pool in pools:
                self._action_log.append({
                    "action": "consolidation_rounds_exhausted",
                    "domain": label, "facet": pool.facet_name,
                    "rounds": self._consolidation_max_rounds,
                    "remaining": len(pool.attributes)})

        structure = self._assemble_structure(settled, consolidated)
        if verbose:
            print(f"    {time.time() - started:.1f}s → {format_counts(structure)}")
        return structure

    # =========================================================================
    # PHASE — ASSIGNMENT (one attribute per unique label; the facet follows)
    # =========================================================================

    def _build_assignment_tasks(
        self,
        ctx: PromptContext,
        structure: Dict[str, List[Dict[str, Any]]],
        labels: Dict[str, Dict[str, str]],
    ) -> List[Dict]:
        """One task per unique normalised label within one domain.

        Ideas carrying the same label become one rep: a single call decides for
        all of them. That is not a batch — the model sees one label and returns
        one attribute — it just does not pay to ask the same question twice.

        A domain whose menu holds a single attribute gets no task: there is
        nothing to choose.
        """
        tasks: List[Dict] = []
        for domain in sorted(structure):
            menu_block, id_map = build_assignment_menu(structure[domain])
            if len(id_map) < 2:
                continue
            for rep in group_label_reps((labels.get(domain) or {}).items()):
                tasks.append({
                    "domain_label": domain,
                    "rep": rep,
                    "menu_block": menu_block,
                    "id_map": id_map,
                })
        return tasks

    def _assignment_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            prompt = build_assignment_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                menu_block=task["menu_block"],
                label=task["rep"].label,
            )
            self._capture(
                f"assignment_{task['domain_label']}", prompt, "assignment",
                {"model": self._model["assignment"],
                 "temperature": self._temperature,
                 "max_tokens": self._max_tokens_assignment,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "n_attributes": len(task["id_map"]),
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": build_assignment_model(list(task["id_map"])),
                "temperature": self._temperature,
                "max_tokens": self._max_tokens_assignment,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["assignment"], phase="classifier_assignment"),
            }
        return prepare_fn

    def _assignment_parse_fn(self):
        """Fan the one judgment out over every idea carrying that label."""
        def parse_fn(task: Dict, response) -> Dict[str, str]:
            if response is None:
                return {}
            choice = task["id_map"][response.assigned_attribute_id]
            out: Dict[str, str] = {}
            for idea_id in task["rep"].idea_ids:
                out[idea_id] = choice["attribute_name"]
                self._attribute_confidence[idea_id] = response.confidence
                self._facet_confidence[idea_id] = response.confidence
                self._attribute_valence[idea_id] = response.valence
            return out
        return parse_fn

    @staticmethod
    def _assignment_fallback_fn():
        """A failed call leaves its ideas unplaced; the net below catches them."""
        def fallback_fn(task: Dict, reason: str) -> Dict[str, str]:
            return {}
        return fallback_fn

    async def _run_assignment(
        self,
        ctx: PromptContext,
        structure: Dict[str, List[Dict[str, Any]]],
        labels: Dict[str, Dict[str, str]],
        verbose: bool,
    ) -> Dict[str, str]:
        """Place every idea on one attribute within its own domain."""
        if verbose:
            print("\n  Assignment")
        started = time.time()

        assignments: Dict[str, str] = {}
        auto: Dict[str, int] = {}

        # A menu of one is not a choice. Assign it without a call.
        for domain, facets in structure.items():
            _, id_map = build_assignment_menu(facets)
            if len(id_map) != 1:
                continue
            only = next(iter(id_map.values()))["attribute_name"]
            for idea_id in labels.get(domain) or {}:
                assignments[idea_id] = only
                self._attribute_confidence[idea_id] = 1.0
                self._facet_confidence[idea_id] = 1.0
            auto[domain] = len(labels.get(domain) or {})

        tasks = self._build_assignment_tasks(ctx, structure, labels)
        results = await self._dispatch(
            "assignment", tasks,
            self._assignment_prepare_fn(ctx),
            self._assignment_parse_fn(),
            self._assignment_fallback_fn(),
            verbose, quiet=False,
        )
        for result in results:
            if result:
                assignments.update(result)

        # The net: only a failed call can leave an idea unplaced now, since the
        # menu always holds a catch-all. Route those to their domain's catch-all
        # rather than to a name absent from the structure.
        for domain, facets in structure.items():
            expected = set(labels.get(domain) or {})
            missing = expected - set(assignments)
            if not missing:
                continue
            drain = next((f for f in facets if is_drain_item(f)), None)
            if drain is None:
                continue
            target = drain["attributes"][0]["attribute_name"]
            print(f"    WARNING: {len(missing)}/{len(expected)} ideas in "
                  f"'{domain}' got no answer — routed to the domain catch-all")
            self._action_log.append({
                "action": "assignment_failed_to_drain", "domain": domain,
                "n_ideas": len(missing), "target": target})
            for idea_id in missing:
                assignments[idea_id] = target
                self._attribute_confidence[idea_id] = 0.0
                self._facet_confidence[idea_id] = 0.0

        if verbose:
            print(f"    {len(tasks)} calls for {len(tasks)} unique labels, "
                  f"{time.time() - started:.1f}s; "
                  f"{len(auto)} domains auto-assigned")
            placed = derive_facet_assignments(assignments, structure)
            for domain in sorted(structure):
                n = len(placed.get(domain) or {})
                tag = " (auto)" if domain in auto else ""
                print(f"      {domain}: {n}/{len(labels.get(domain) or {})}{tag}")
        return assignments

    # =========================================================================
    # PHASE — REFINEMENT (per domain, after every idea is assigned)
    # =========================================================================

    def _build_refinement_tasks(
        self,
        ctx: PromptContext,
        structure: Dict[str, List[Dict[str, Any]]],
        assignments: Dict[str, str],
        labels: Dict[str, Dict[str, str]],
    ) -> List[Dict]:
        """One task per domain, carrying what its attributes really hold.

        This is the first phase with real counts in front of it, so the task
        shape is where those counts get computed: per attribute the number of
        ideas, its share of the domain, and the distinct response texts.
        """
        tasks: List[Dict] = []
        for domain in sorted(structure):
            texts = labels.get(domain) or {}
            counts: Counter = Counter()
            contents: Dict[str, List[str]] = {}
            for idea_id, attribute_name in assignments.items():
                if idea_id not in texts:
                    continue
                counts[attribute_name] += 1
                seen = contents.setdefault(attribute_name, [])
                text = texts[idea_id]
                if text and text not in seen:
                    seen.append(text)
            total = sum(counts.values())
            if total == 0:
                continue
            shares = {name: n / total for name, n in counts.items()}
            tasks.append({
                "domain_label": domain,
                "facets": structure[domain],
                "counts": dict(counts),
                "shares": shares,
                "contents": contents,
            })
        return tasks

    def _refinement_prepare_fn(self, ctx: PromptContext):
        def prepare_fn(task: Dict) -> Dict:
            domain = ctx.domain(task["domain_label"])
            prompt = build_refinement_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                dimension=ctx.dimension,
                dimension_name=ctx.dimension_name,
                dimension_description=ctx.dimension_description,
                domain_label=domain["label"],
                domain_definition=domain["definition"],
                contents_block=build_contents_block(
                    task["facets"], task["contents"], task["shares"],
                    task["counts"], self._contents_top_n),
            )
            self._capture(
                f"refinement_{task['domain_label']}", prompt, "refinement",
                {"model": self._model["refinement"],
                 "temperature": 0.0,
                 "max_tokens": self._max_tokens_consolidation,
                 "language": ctx.language,
                 "domain": task["domain_label"],
                 "dimension_name": ctx.dimension_name})
            return {
                "prompt": prompt,
                "response_model": RefinementResult,
                "temperature": 0.0,
                "max_tokens": self._max_tokens_consolidation,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["refinement"], phase="classifier_refinement"),
            }
        return prepare_fn

    @staticmethod
    def _refinement_parse_fn():
        def parse_fn(task: Dict, response):
            return response if response else None
        return parse_fn

    @staticmethod
    def _refinement_fallback_fn():
        """On failure the domain keeps the inventory assignment settled."""
        def fallback_fn(task: Dict, reason: str) -> None:
            return None
        return fallback_fn

    def _apply_refinement(
        self,
        *,
        tasks: List[Dict],
        results: List,
        structure: Dict[str, List[Dict[str, Any]]],
        assignments: Dict[str, str],
        labels: Dict[str, Dict[str, str]],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
        """Rebuild each domain's structure, then route its ideas.

        Structure first, so a move target resolves against the FINAL names
        rather than the name it had before a merge renamed it. Then the ideas,
        in the order splits → renames → misfits: a split routes by exact text
        and must win over the wholesale rename its source also carries.

        Only one domain is ever in flight per task, so unlike the per-facet
        predecessor there is no concurrent scope renaming this one's targets.
        """
        new_structure = dict(structure)
        remap: Dict[Tuple[str, str], str] = {}          # (domain, old) -> new
        splits: Dict[Tuple[str, str], str] = {}          # (domain, norm text) -> child
        misfits: Dict[Tuple[str, str], Optional[str]] = {}  # (domain, norm text) -> target

        for task, result in zip(tasks, results):
            domain = task["domain_label"]
            before = [a["attribute_name"]
                      for f in task["facets"] for a in (f.get("attributes") or [])]

            if result is None or not result.attributes:
                self._action_log.append({
                    "action": "refinement_failed", "domain": domain,
                    "note": "no result — domain left as assignment settled it",
                    "attributes_before": before})
                continue

            by_norm = {_norm(b): b for b in before}
            drains_by_name = {
                a["attribute_name"]: (f["facet_name"], a)
                for f in task["facets"] for a in (f.get("attributes") or [])
                if is_drain_item(a)}

            unmatched = sorted({
                s for item in result.attributes
                for s in (item.source_attributes or [])
                if _norm(s) not in by_norm})
            if unmatched:
                self._action_log.append({
                    "action": "unknown_source_name", "domain": domain,
                    "sources": unmatched,
                    "note": "named as a source but not among this domain's attributes"})

            # A source claimed by several returned attributes is only routable
            # when the claimants carry instance_texts. Without them the remap
            # would let the last writer win and move the whole bucket.
            claims: Counter = Counter()
            for item in result.attributes:
                for src in (item.source_attributes or []):
                    real = by_norm.get(_norm(src))
                    if real:
                        claims[real] += 1
            split_sources = {
                by_norm.get(_norm(s))
                for item in result.attributes if item.instance_texts
                for s in (item.source_attributes or [])}
            contested = {src for src, n in claims.items()
                         if n > 1 and src not in split_sources}
            if contested:
                self._action_log.append({
                    "action": "unroutable_claim", "domain": domain,
                    "sources": sorted(contested),
                    "note": "claimed by several returned attributes with no "
                            "instance_texts — ideas left on the source"})

            # ---- structure ---------------------------------------------------
            per_facet: Dict[str, List[Dict[str, Any]]] = {}
            consumed: Set[str] = set()
            for item in result.attributes:
                if item.attribute_name in drains_by_name:
                    continue  # a catch-all the model touched anyway: ignore it
                per_facet.setdefault(item.facet_name, []).append({
                    "attribute_name": item.attribute_name,
                    "attribute_definition": item.attribute_definition,
                    "example_observations": list(item.example_observations),
                })
                sources = [by_norm[_norm(s)] for s in (item.source_attributes or [])
                           if _norm(s) in by_norm]
                consumed.update(sources)

                if item.action == "split" and item.instance_texts:
                    for text in item.instance_texts:
                        splits[(domain, _norm(text))] = item.attribute_name
                    self._action_log.append({
                        "action": "split", "domain": domain,
                        "into": item.attribute_name, "sources": sources,
                        "n_texts": len(item.instance_texts)})
                else:
                    for src in sources:
                        if src != item.attribute_name and src not in contested:
                            remap[(domain, src)] = item.attribute_name
                    if item.action in ("merge", "widen") or (
                            sources and sources != [item.attribute_name]):
                        self._action_log.append({
                            "action": item.action, "domain": domain,
                            "result": item.attribute_name, "sources": sources})

            # Sources never claimed keep their place, or their ideas would point
            # at a name no longer in the structure.
            returned = {_norm(a["attribute_name"])
                        for items in per_facet.values() for a in items}
            for facet in task["facets"]:
                for attribute in facet.get("attributes") or []:
                    name = attribute["attribute_name"]
                    if is_drain_item(attribute):
                        per_facet.setdefault(facet["facet_name"], []).append(attribute)
                        continue
                    if name in consumed or _norm(name) in returned:
                        continue
                    per_facet.setdefault(facet["facet_name"], []).append(attribute)
                    self._action_log.append({
                        "action": "attribute_kept_unclaimed_in_refinement",
                        "domain": domain, "attribute": name})

            # Facet cards: keep the ones that still hold attributes, in the order
            # the domain had them, then any facet the model named that is new.
            known = {f["facet_name"]: f for f in task["facets"]}
            rebuilt: List[Dict[str, Any]] = []
            for facet_name, items in per_facet.items():
                card = dict(known.get(facet_name) or {
                    "facet_name": facet_name,
                    "facet_definition": "",
                })
                card["attributes"] = items
                rebuilt.append(card)
            new_structure[domain] = rebuilt

            for misfit in (result.misfits or []):
                target = misfit.target_attribute if misfit.verdict == "move" else None
                for text in (misfit.instance_texts or []):
                    misfits[(domain, _norm(text))] = target
                self._action_log.append({
                    "action": f"misfit_{misfit.verdict}", "domain": domain,
                    "target": misfit.target_attribute,
                    "n_texts": len(misfit.instance_texts or [])})

        # ---- ideas -----------------------------------------------------------
        domain_of: Dict[str, str] = {
            idea_id: domain
            for domain, texts in labels.items() for idea_id in texts}
        live = {_norm(a["attribute_name"])
                for facets in new_structure.values() for f in facets
                for a in (f.get("attributes") or [])}

        moved = 0
        out: Dict[str, str] = {}
        for idea_id, attribute_name in assignments.items():
            domain = domain_of.get(idea_id)
            if domain is None:
                out[idea_id] = attribute_name
                continue
            text = _norm((labels.get(domain) or {}).get(idea_id))

            target = splits.get((domain, text))
            if target is None and (domain, text) in misfits:
                target = misfits[(domain, text)]
                if target is None:
                    # verdict "out": no destination exists, so it stays where it
                    # is and the log carries the finding.
                    target = attribute_name
            if target is None:
                target = remap.get((domain, attribute_name), attribute_name)

            if _norm(target) not in live:
                target = attribute_name
            if target != attribute_name:
                moved += 1
            out[idea_id] = target

        if moved:
            self._action_log.append({"action": "ideas_moved", "n_ideas": moved})
        return new_structure, out

    async def _run_refinement(
        self,
        ctx: PromptContext,
        structure: Dict[str, List[Dict[str, Any]]],
        assignments: Dict[str, str],
        labels: Dict[str, Dict[str, str]],
        verbose: bool,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
        """Judge each domain against what its attributes really hold."""
        if verbose:
            print("\n  Refinement")
        started = time.time()

        tasks = self._build_refinement_tasks(ctx, structure, assignments, labels)
        results = await self._dispatch(
            "refinement", tasks,
            self._refinement_prepare_fn(ctx),
            self._refinement_parse_fn(),
            self._refinement_fallback_fn(),
            verbose,
        )
        structure, assignments = self._apply_refinement(
            tasks=tasks, results=results, structure=structure,
            assignments=assignments, labels=labels)

        if verbose:
            print(f"    {len(tasks)} domains, {time.time() - started:.1f}s → "
                  f"{format_counts(structure)}")
        return structure, assignments

    # =========================================================================
    # PHASE — CROSS-DOMAIN (every domain at once, the one relocation step)
    # =========================================================================

    async def _run_cross_domain(
        self,
        ctx: PromptContext,
        structure: Dict[str, List[Dict[str, Any]]],
        assignments: Dict[str, str],
        verbose: bool,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
        """Fold duplicate attributes together across domains.

        Every domain settled on its own, so the same concept survives in several
        places and no scope-bound phase can see it. This one may relocate
        structure, and it is the only one: it works on ids, and the survivor
        inherits the domain and facet of its `home_id`.
        """
        if verbose:
            print("\n  Cross-domain consolidation")
        started = time.time()

        counts = Counter(assignments.values())
        entries: List[Dict[str, Any]] = []
        for domain in sorted(structure):
            for facet in structure[domain]:
                for attribute in facet.get("attributes") or []:
                    if is_drain_item(attribute):
                        continue
                    entries.append({
                        "id": f"A{len(entries) + 1}",
                        "domain": domain,
                        "facet": facet["facet_name"],
                        "attribute": attribute,
                        "n": counts.get(attribute["attribute_name"], 0)})
        if len(entries) < 2:
            return structure, assignments

        lines, current = [], None
        for entry in entries:
            head = (entry["domain"], entry["facet"])
            if head != current:
                lines.append(f"\n{entry['domain']} › {entry['facet']}")
                current = head
            lines.append(
                f"  [{entry['id']}] {entry['attribute']['attribute_name']} — "
                f"{entry['n']} responses\n"
                f"        {entry['attribute']['attribute_definition']}")

        task = {"entries": entries, "inventory_block": "\n".join(lines)}
        by_id = {e["id"]: e for e in entries}

        def prepare_fn(t: Dict) -> Dict:
            prompt = build_cross_domain_prompt(
                language=ctx.language,
                survey_question=ctx.survey_question,
                **ctx.specifiers(),
                inventory_block=t["inventory_block"],
            )
            self._capture("cross_domain", prompt, "cross_domain",
                          {"model": self._model["cross_domain"],
                           "temperature": 0.0,
                           "max_tokens": self._max_tokens_consolidation,
                           "language": ctx.language,
                           "n_attributes": len(t["entries"])})
            return {
                "prompt": prompt,
                "response_model": build_cross_scope_model(
                    [e["id"] for e in t["entries"]], "attribute"),
                "temperature": 0.0,
                "max_tokens": self._max_tokens_consolidation,
                "max_retries": 3,
                "extra_kwargs": get_reasoning_params(
                    self._model["cross_domain"],
                    phase="classifier_cross_domain"),
            }

        results = await self._dispatch(
            "cross_domain", [task], prepare_fn,
            lambda t, response: response,
            lambda t, reason: None,
            verbose,
        )
        result = results[0] if results else None
        if result is None or not result.items:
            self._action_log.append({
                "action": "cross_domain_failed",
                "note": "no result — every domain left as refinement settled it"})
            return structure, assignments

        # Rebuild from the answer. An id nobody claimed stays exactly where it
        # is: a dropped id must not silently take its ideas with it.
        rebuilt: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        rename: Dict[str, str] = {}
        claimed: Set[str] = set()
        merges = 0

        for item in result.items:
            home = by_id.get(item.home_id)
            sources = [by_id[i] for i in item.source_ids if i in by_id]
            if home is None or not sources:
                continue
            claimed.update(e["id"] for e in sources)
            rebuilt.setdefault(home["domain"], {}).setdefault(
                home["facet"], []).append({
                    "attribute_name": item.name,
                    "attribute_definition": item.definition,
                    "example_observations": list(
                        home["attribute"].get("example_observations") or []),
                })
            for entry in sources:
                rename[_norm(entry["attribute"]["attribute_name"])] = item.name
            if len(sources) > 1:
                merges += 1
                self._action_log.append({
                    "action": "cross_domain_merge", "result": item.name,
                    "sources": [e["attribute"]["attribute_name"] for e in sources],
                    "home": f"{home['domain']} › {home['facet']}"})

        for entry in entries:
            if entry["id"] in claimed:
                continue
            rebuilt.setdefault(entry["domain"], {}).setdefault(
                entry["facet"], []).append(entry["attribute"])
            self._action_log.append({
                "action": "attribute_kept_unclaimed_cross_domain",
                "attribute": entry["attribute"]["attribute_name"]})

        # Put the catch-alls back — they were held out of the inventory.
        drains = {
            (d, f["facet_name"]): [a for a in (f.get("attributes") or [])
                                   if is_drain_item(a)]
            for d, facets in structure.items() for f in facets}
        for (domain, facet_name), items in drains.items():
            if items:
                rebuilt.setdefault(domain, {}).setdefault(
                    facet_name, []).extend(items)

        # Rebuild the facet cards. Definition and drain flag are carried over
        # from the card the facet already had rather than reconstructed: this
        # phase judges attributes, and a facet it never looked at should come
        # out exactly as it went in.
        previous = {(d, f["facet_name"]): f
                    for d, facets in structure.items() for f in facets}
        new_structure: Dict[str, List[Dict[str, Any]]] = {}
        for domain, per_facet in rebuilt.items():
            cards = []
            for facet_name, items in per_facet.items():
                old = previous.get((domain, facet_name)) or {}
                card = {k: v for k, v in old.items() if k != "attributes"}
                card.setdefault("facet_name", facet_name)
                card.setdefault("facet_definition", "")
                card["attributes"] = items
                cards.append(card)
            new_structure[domain] = cards

        moved = 0
        out: Dict[str, str] = {}
        for idea_id, attribute_name in assignments.items():
            target = rename.get(_norm(attribute_name), attribute_name)
            if target != attribute_name:
                moved += 1
            out[idea_id] = target

        if verbose:
            after = count_structure(new_structure)["attributes"]
            print(f"    {len(entries)} attributes in, {after} out "
                  f"({merges} merged, {moved} ideas moved), "
                  f"{time.time() - started:.1f}s")
            print(f"    → {format_counts(new_structure)}")
        return new_structure, out
