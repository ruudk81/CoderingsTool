"""Deterministic taxonomy hygiene and health metrics. No LLM.

Four things, all dataset-independent and all cheap enough to run on every build:

  attr_structure_home() Maps attribute_name -> (domain, facet) from the structure, so
                        per-idea (domain, facet) stays a DERIVED projection of the
                        taxonomy instead of an independently-maintained copy that
                        drifts from it.

  prune_empty_nodes()   Consolidation moves ideas out of an attribute but leaves the
                        structure node behind. The structure-projection fix cannot
                        clear these — with no ideas to project there is nothing to
                        correct — so they survive into the export as n=0 rows.
                        This drops them.

  drain_domains()       Labels of the standing drain domains (other, not_known),
                        identified by their metadata key rather than their (possibly
                        translated/re-described) label. Used in classifier.py to skip
                        the two catch-all domains during discovery: imposing structure
                        on a deliberately broad catch-all invents distinctions the
                        responses do not carry. Warns (never fails silently) when it
                        finds fewer than the two it expects.

  measure()             Turns "the taxonomy feels flat" into numbers you can compare
                        across datasets. Every metric below marks a label layer that
                        carries no discriminating information:

                          facet == attribute   the facet name says nothing the
                                               attribute does not already say
                          solo facets          a facet with one attribute is a level
                                               that adds no split
                          duplicate names      the same name under two parents; since
                                               name IS identity here, that is a
                                               genuine ambiguity, not just cosmetics

                        None of these are errors on their own — a solo facet can be
                        a deliberate placeholder for later waves. A high SHARE of
                        them is the signal.

Scope: this module operates PRE-finalization, where name still is identity —
stable ids (D#/F#/A#, src/identity.py) are minted at cache-save, after these
checks ran. It therefore deliberately stays name-based.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from models import TaxonomyResultsCache

SENTINELS = {"__UNASSIGNED__", "(no attribute)", "(geen attribuut)"}

DRAIN_KEYS = frozenset({"other", "not_known", "no_subject"})


# =============================================================================
# DRAIN DOMAINS
# =============================================================================

def drain_domains(extraction_metadata) -> Set[str]:
    """Labels of the standing drain domains, identified by their metadata key
    rather than their (possibly translated/re-described) label.

    Always two by construction (`DRAIN_KEYS`) whenever `extraction_metadata`
    carries domains at all. A cache written under a stale key — this module's
    own `DRAIN_KEYS`, `measure_stability.DRAIN_KEYS` and
    `prompts_ideaExtractor.STANDING_NOT_KNOWN_KEY`/`STANDING_OTHER_KEY` can
    drift apart on a rename that touches only one of them — matches on
    whichever key survived and returns fewer than two. That result is
    non-empty, so nothing downstream would otherwise notice: prints a warning
    naming what it did find whenever the count is short. A run with domain
    discovery off carries no domains at all, and that stays a silent,
    legitimate zero.
    """
    domains = getattr(extraction_metadata, "domains", None) or []
    found = {d.get("label", "") for d in domains if d.get("key") in DRAIN_KEYS}
    if domains and len(found) < len(DRAIN_KEYS):
        print(
            f"  WARNING: drain_domains found {len(found)}/{len(DRAIN_KEYS)} "
            f"standing domains ({sorted(found)}) — a cache key may be stale "
            f"against DRAIN_KEYS={sorted(DRAIN_KEYS)}."
        )
    return found


# =============================================================================
# STRUCTURE PROJECTION
# =============================================================================

def attr_structure_home(
    taxonomy_cache: TaxonomyResultsCache,
) -> Dict[str, Tuple[str, str]]:
    """Map attribute_name -> (domain, facet) from the taxonomy STRUCTURE
    (partition_results[*].attributes).

    Only UNAMBIGUOUS names (present under exactly one (domain, facet)) are
    returned; ambiguous names are omitted so callers fall back to the existing
    per-idea assignment. This lets per-idea (domain, facet) be a derived
    projection of the structure — one source of truth — instead of an
    independently-maintained copy that can drift from it.
    """
    places: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for dom, res in taxonomy_cache.partition_results.items():
        for fac, lst in (getattr(res, "attributes", {}) or {}).items():
            for a in lst:
                name = (a.get("attribute_name") if isinstance(a, dict)
                        else getattr(a, "attribute_name", None))
                if name:
                    places[name].add((dom, fac))
    return {n: next(iter(p)) for n, p in places.items() if len(p) == 1}


# =============================================================================
# PRUNE
# =============================================================================

@dataclass
class PruneReport:
    attributes: List[Tuple[str, str, str]] = field(default_factory=list)  # domain, facet, attr
    facets: List[Tuple[str, str]] = field(default_factory=list)           # domain, facet

    @property
    def total(self) -> int:
        return len(self.attributes) + len(self.facets)

    def lines(self) -> List[str]:
        out = []
        for d, f, a in self.attributes:
            out.append(f"    attribuut  {a!r}  ({d} / {f})")
        for d, f in self.facets:
            out.append(f"    facet      {f!r}  ({d}) — had geen attributen meer")
        return out


def prune_empty_nodes(tax: TaxonomyResultsCache) -> PruneReport:
    """Drop structure nodes that carry zero ideas. Mutates `tax` in place.

    An attribute goes when no idea is assigned to it. A facet goes only when it has
    no attributes left AND no ideas of its own — never strand an idea.

    Does nothing when the run produced no assignments at all. "No ideas here" then
    does not mean the node is empty, only that nothing has been assigned yet — and
    pruning on that reading deletes the entire taxonomy. A `stop_after_phase`
    before `assignment` hit exactly that: 55 facets and 179 attributes discovered,
    all 234 pruned, an empty taxonomy written over the complete one in the cache.

    The guard is on the data rather than on the phase name, so it also covers an
    assignment phase that failed outright, and needs no knowledge of phase order.
    """
    report = PruneReport()

    if not any(dr.attribute_assignments for dr in tax.partition_results.values()):
        return report

    for dname, dr in tax.partition_results.items():
        attr_counts = Counter((dr.attribute_assignments or {}).values())
        facet_counts = Counter((dr.facet_assignments or {}).values())

        for fname, attrs in list((dr.attributes or {}).items()):
            keep = [a for a in attrs if attr_counts.get(a.get("attribute_name"), 0) > 0]
            for a in attrs:
                if a not in keep:
                    report.attributes.append((dname, fname, a.get("attribute_name")))
            if keep:
                dr.attributes[fname] = keep
                continue

            # facet has no populated attribute left
            if facet_counts.get(fname, 0) > 0:
                dr.attributes[fname] = keep          # keep the empty facet; it holds ideas
                continue
            del dr.attributes[fname]
            dr.facets = [f for f in (dr.facets or []) if f.get("facet_name") != fname]
            report.facets.append((dname, fname))

    return report


# =============================================================================
# MEASURE
# =============================================================================

@dataclass
class HealthReport:
    n_domains: int = 0
    n_facets: int = 0
    n_attributes: int = 0
    n_ideas: int = 0
    empty_attributes: List[str] = field(default_factory=list)
    facet_equals_attribute: List[str] = field(default_factory=list)
    solo_facets: List[str] = field(default_factory=list)
    duplicate_attributes: Dict[str, List[str]] = field(default_factory=dict)
    duplicate_facets: Dict[str, List[str]] = field(default_factory=dict)
    n_drain_ideas: int = 0

    @property
    def solo_facet_share(self) -> float:
        return 100.0 * len(self.solo_facets) / self.n_facets if self.n_facets else 0.0

    @property
    def drain_share(self) -> float:
        """How much ended up in a catch-all.

        The counter-metric to coarser grouping: every merge that goes too far
        pushes responses into a catch-all, and that is the only signal that does
        not move along with "fewer attributes is better".
        """
        return 100.0 * self.n_drain_ideas / self.n_ideas if self.n_ideas else 0.0

    def lines(self) -> List[str]:
        out = [
            f"  {self.n_domains} domeinen / {self.n_facets} facetten / "
            f"{self.n_attributes} attributen / {self.n_ideas} ideeën",
            f"  facet == attribuut : {len(self.facet_equals_attribute)}",
            f"  1:1-facetten       : {len(self.solo_facets)} "
            f"({self.solo_facet_share:.0f}% van de facetten)",
            f"  lege attributen    : {len(self.empty_attributes)}",
            f"  dubbele namen      : {len(self.duplicate_attributes)} attribuut, "
            f"{len(self.duplicate_facets)} facet",
            f"  in een vangnet     : {self.n_drain_ideas} ideeën "
            f"({self.drain_share:.1f}% van de toegewezen ideeën)",
        ]
        for label, items in (("facet == attribuut", self.facet_equals_attribute),
                             ("lege attributen", self.empty_attributes)):
            for it in items:
                out.append(f"    [{label}] {it}")
        for name, places in self.duplicate_attributes.items():
            out.append(f"    [dubbel attribuut] {name!r}: {', '.join(places)}")
        for name, places in self.duplicate_facets.items():
            out.append(f"    [dubbel facet] {name!r}: {', '.join(places)}")
        return out


def measure(tax: TaxonomyResultsCache) -> HealthReport:
    """Read-only structural metrics over the taxonomy."""
    rep = HealthReport(n_domains=len(tax.partition_results))
    attr_where: Dict[str, List[str]] = defaultdict(list)
    facet_where: Dict[str, List[str]] = defaultdict(list)
    ideas = set()

    for dname, dr in tax.partition_results.items():
        counts = Counter(v for v in (dr.attribute_assignments or {}).values()
                         if v not in SENTINELS)
        ideas.update(k for k, v in (dr.attribute_assignments or {}).items()
                     if v not in SENTINELS)

        drain_names = {a.get("attribute_name")
                       for attrs in (dr.attributes or {}).values()
                       for a in attrs if a.get("drain_key")}
        rep.n_drain_ideas += sum(n for name, n in counts.items()
                                 if name in drain_names)

        for fname, attrs in (dr.attributes or {}).items():
            rep.n_facets += 1
            facet_where[fname].append(dname)
            names = [a.get("attribute_name") for a in attrs]
            rep.n_attributes += len(names)

            for an in names:
                attr_where[an].append(f"{dname}/{fname}")
                if counts.get(an, 0) == 0:
                    rep.empty_attributes.append(f"{dname} / {fname} / {an}")
                if (an or "").strip().lower() == fname.strip().lower():
                    rep.facet_equals_attribute.append(f"{dname} / {fname}")
            if len(names) == 1:
                rep.solo_facets.append(f"{dname} / {fname} -> {names[0]}")

    rep.n_ideas = len(ideas)
    rep.duplicate_attributes = {k: v for k, v in attr_where.items() if len(v) > 1}
    rep.duplicate_facets = {k: v for k, v in facet_where.items() if len(set(v)) > 1}
    return rep


def print_health(tax: TaxonomyResultsCache, prune: PruneReport = None) -> None:
    """Verbose block for the step-4 runner."""
    print("\nTAXONOMIE-GEZONDHEID")
    if prune is not None and prune.total:
        print(f"  opgeruimd: {prune.total} lege knopen")
        for line in prune.lines():
            print(line)
    for line in measure(tax).lines():
        print(line)
