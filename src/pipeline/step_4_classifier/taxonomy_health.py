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

  drain_domains()       Labels of the standing drain domains (other, bare_evaluation),
                        identified by their metadata key rather than their (possibly
                        translated/re-described) label. Used to keep the P3/P7
                        upstream MECE reviews off the two domains that exist to
                        catch everything else, not to be internally orthogonal.

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

DRAIN_KEYS = frozenset({"other", "bare_evaluation"})


# =============================================================================
# DRAIN DOMAINS
# =============================================================================

def drain_domains(extraction_metadata) -> Set[str]:
    """Labels of the standing drain domains, identified by their metadata key.

    Legacy caches persisted key == label; then no domain qualifies and callers
    treat every domain as reviewable (and say so once in the verbose output).
    """
    domains = getattr(extraction_metadata, "domains", None) or []
    return {d.get("label", "") for d in domains if d.get("key") in DRAIN_KEYS}


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
    """
    report = PruneReport()

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
class AxisPurityResult:
    """Axis-zuiverheid over every sibling set: a domain's facets (shared
    axis, distinct segments) or a facet's attributes (distinct positions).
    Counting on tags alone, no LLM — see `_classify_facet_siblings` /
    `_classify_attribute_siblings`. UNTAGGED (no member carries a tag, e.g.
    a legacy cache or a domain outside the axis-first path) is reported
    separately and is never a violation."""
    pure: int = 0
    untagged: int = 0
    violations: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.pure + self.untagged + len(self.violations)


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
    axis_purity: AxisPurityResult = field(default_factory=AxisPurityResult)

    @property
    def solo_facet_share(self) -> float:
        return 100.0 * len(self.solo_facets) / self.n_facets if self.n_facets else 0.0

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
            f"  as-zuiverheid      : {self.axis_purity.pure}/{self.axis_purity.total} sets puur, "
            f"{self.axis_purity.untagged} untagged, "
            f"{len(self.axis_purity.violations)} violations",
        ]
        for label, items in (("facet == attribuut", self.facet_equals_attribute),
                             ("lege attributen", self.empty_attributes)):
            for it in items:
                out.append(f"    [{label}] {it}")
        for name, places in self.duplicate_attributes.items():
            out.append(f"    [dubbel attribuut] {name!r}: {', '.join(places)}")
        for name, places in self.duplicate_facets.items():
            out.append(f"    [dubbel facet] {name!r}: {', '.join(places)}")
        for v in self.axis_purity.violations:
            out.append(f"    [as-schending] {v}")
        return out


def _norm_text(text) -> str:
    """Normalise a tag value for distinctness comparison. Case- and
    padding-insensitive only, mirroring `TaxonomyClassifier._norm_text`
    (classifier.py) so this metric is exactly as lax as the enforcement it
    audits — a case- or whitespace-variant echo from a review pass (e.g. P7
    re-emitting a position name) must not count as a distinct value here."""
    return (text or "").strip().lower()


def _classify_siblings(
    items: List[dict], tag_key: str, name_key: str
) -> Tuple[str, List[str]]:
    """Classify one sibling set (all items must share ONE tag value each,
    pairwise distinct after normalisation) for axis-purity. Returns
    (status, offending names):

      "untagged"  — no member carries `tag_key` (legacy / non-axis-first)
      "violation" — some but not all members tagged, or two members share
                    the same (normalised) tag value; offenders are the
                    names at fault
      "pure"      — every member tagged, every tag value distinct

    Used for attribute sibling sets (tag_key="position"). Facet sibling sets
    are classified by `_classify_facet_siblings` instead — a domain's facets
    do not share a single tag_key/value pair the way one facet's attributes
    do, since P1a puts 1-4 independent axes on one domain (see there). An
    empty `items` list is the caller's concern — this function assumes at
    least one item.
    """
    tagged = [it for it in items if it.get(tag_key)]
    if not tagged:
        return "untagged", []
    if len(tagged) != len(items):
        offenders = [it.get(name_key, "?") for it in items if not it.get(tag_key)]
        return "violation", offenders
    values = [_norm_text(it.get(tag_key)) for it in items]
    dupes = {v for v, c in Counter(values).items() if c > 1}
    if dupes:
        offenders = [it.get(name_key, "?") for it in items if _norm_text(it.get(tag_key)) in dupes]
        return "violation", offenders
    return "pure", []


def _classify_axis_group(axis_facets: List[dict]) -> Tuple[str, List[str]]:
    """Classify one (domain, axis) facet group — every member already
    carries THIS axis tag (that is the grouping precondition in
    `_classify_facet_siblings`). PURE requires every member to also carry a
    non-empty segment, pairwise distinct after normalisation. A member with
    the axis set but no segment is a violation, not "untagged" — the axis
    tag alone is not enforceable without a segment, so reporting it as a
    clean/untagged set would hide a broken proposal."""
    segmentless = [f for f in axis_facets if not f.get("segment")]
    if segmentless:
        offenders = [f.get("facet_name", "?") for f in segmentless]
        return "violation", offenders
    segments = [_norm_text(f.get("segment")) for f in axis_facets]
    dupes = {s for s, c in Counter(segments).items() if c > 1}
    if dupes:
        offenders = [f.get("facet_name", "?") for f in axis_facets if _norm_text(f.get("segment")) in dupes]
        return "violation", offenders
    return "pure", []


def _classify_facet_siblings(facets: List[dict]) -> List[Tuple[str, str, List[str]]]:
    """Classify a domain's facet sibling set(s) for axis-purity.

    P1a discovers 1-4 axes PER DOMAIN (spec §Fasering), and P2 consolidates
    one facet per populated (axis, segment) across ALL of them — a domain's
    facets do not share a single axis by design. Treating "the domain's
    facets" as one sibling set (an earlier version of this function did)
    flags every legitimate multi-axis domain as a violation, and checks
    segment distinctness across axes that may legitimately reuse a segment
    name (e.g. the injected residual segment, named identically on every
    axis that lacked one).

    Facets are grouped BY AXIS first; each (domain, axis) group is its own
    sibling set requiring pairwise-distinct segments (`_classify_axis_group`).
    Returns one (label_suffix, status, offenders) tuple per sibling set:
    a single ("", "untagged" | "violation", …) for a wholly-untagged or
    mixed-tagged domain (some facets carry an axis, some do not — still one
    violation, not decomposable into axis groups), or one
    (" · as={axis}", …) per axis group once every facet in the domain
    carries an axis tag.
    """
    tagged = [f for f in facets if f.get("axis")]
    if not tagged:
        return [("", "untagged", [])]
    if len(tagged) != len(facets):
        offenders = [f.get("facet_name", "?") for f in facets if not f.get("axis")]
        return [("", "violation", offenders)]

    by_axis: Dict[str, List[dict]] = {}
    for f in facets:
        by_axis.setdefault(f.get("axis"), []).append(f)

    return [
        (f" · as={axis_name}", *_classify_axis_group(axis_facets))
        for axis_name, axis_facets in by_axis.items()
    ]


def _classify_attribute_siblings(attrs: List[dict]) -> Tuple[str, List[str]]:
    """Classify a facet's attribute sibling set: PURE requires every
    attribute tagged with a distinct (normalised) position on the facet's
    refinement axis — a single axis per facet, so (unlike facets) no
    per-axis grouping is needed here."""
    return _classify_siblings(attrs, "position", "attribute_name")


def _record_purity(report: "AxisPurityResult", label: str, status: str, offenders: List[str]) -> None:
    if status == "pure":
        report.pure += 1
    elif status == "untagged":
        report.untagged += 1
    else:
        report.violations.append(f"{label}: {', '.join(offenders) or '?'}")


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

        facets = dr.facets or []
        if facets:
            for suffix, status, offenders in _classify_facet_siblings(facets):
                _record_purity(rep.axis_purity, f"domein {dname}{suffix} (facetten)", status, offenders)

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

            if attrs:
                status, offenders = _classify_attribute_siblings(attrs)
                _record_purity(rep.axis_purity, f"{dname} / {fname} (attributen)", status, offenders)

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
