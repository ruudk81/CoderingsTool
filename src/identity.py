"""
Stable identity for taxonomy and codebook artifacts (naam-als-identiteit).

Every domain, facet, attribute, and code carries an immutable id (D#/F#/A#/K#)
minted once when its artifact is finalized for caching. Names are display-only;
cross-artifact joins use ids. `K` (kode) is deliberately distinct from the
ephemeral per-prompt `C#` option labels in step 6 — a persisted id can never be
confused with a prompt index.

Lazy migration: artifacts written before ids existed are normalized IN MEMORY at
load time (`ensure_ids`, hooked into CacheManager.load_metadata_from_cache).
Minting is deterministic — it follows the artifact's own stored order (partition
set order, list order, dict insertion order), so the same frozen bytes yield the
same ids on every load. Disk is never mutated by normalization; ids persist only
when a step re-saves its artifact. This is the single documented exception to
the no-legacy-tolerance contract.

All functions are idempotent: entities that already carry an id are skipped and
counters continue from the highest existing id per level.
"""

import re
from typing import Dict, List, Optional, Tuple

from models import CodingResultsCache, TaxonomyResultsCache

UNASSIGNED = "__UNASSIGNED__"


def _next_counter(prefix: str, existing_ids) -> int:
    """Highest existing sequential id for `prefix`, so minting continues after it."""
    pat = re.compile(rf"^{prefix}(\d+)$")
    return max((int(m.group(1)) for i in existing_ids if i and (m := pat.match(i))), default=0)


def _domain_order(tax: TaxonomyResultsCache) -> List[str]:
    """Canonical domain iteration order: partition_set order (same ordering the
    step-7 catalog uses), then any partition_results-only keys in insertion order."""
    ordered = [p.partition_name for p in tax.partition_set.partitions
               if p.partition_name in tax.partition_results]
    ordered += [k for k in tax.partition_results if k not in ordered]
    return ordered


def ensure_taxonomy_ids(tax: TaxonomyResultsCache) -> TaxonomyResultsCache:
    """Mint D#/F#/A# on domains, facet dicts, and attribute dicts (in place).

    Corrected attributes (over-merge split copies) inherit the id of the main
    attribute with the same (domain, attribute_name); split-out children that
    have no main counterpart mint fresh ids. Raw (fijn) attributes deliberately
    get no ids: display-only, no cross-artifact joins.
    """
    domains = _domain_order(tax)
    dnum = _next_counter("D", (tax.partition_results[d].domain_id for d in domains))
    fnum = _next_counter("F", (f.get("facet_id") for d in domains
                               for f in tax.partition_results[d].facets))
    anum = _next_counter("A", (a.get("attribute_id") for d in domains
                               for dr in [tax.partition_results[d]]
                               for attrs in list(dr.attributes.values()) + list(dr.corrected_attributes.values())
                               for a in attrs))

    for d in domains:
        dr = tax.partition_results[d]
        if not dr.domain_id:
            dnum += 1
            dr.domain_id = f"D{dnum}"
        for f in dr.facets:
            if not f.get("facet_id"):
                fnum += 1
                f["facet_id"] = f"F{fnum}"
        for attrs in dr.attributes.values():
            for a in attrs:
                if not a.get("attribute_id"):
                    anum += 1
                    a["attribute_id"] = f"A{anum}"
        # corrected copies: same (domain, name) -> same id; split children mint new
        main_by_name = {a["attribute_name"]: a["attribute_id"]
                        for attrs in dr.attributes.values() for a in attrs
                        if a.get("attribute_name")}
        for attrs in dr.corrected_attributes.values():
            for a in attrs:
                if not a.get("attribute_id"):
                    inherited = main_by_name.get(a.get("attribute_name"))
                    if inherited:
                        a["attribute_id"] = inherited
                    else:
                        anum += 1
                        a["attribute_id"] = f"A{anum}"
    return tax


def _effective_attributes(dr) -> Dict[str, List[dict]]:
    """The attribute structure a consumer actually sees: corrected if the
    over-merge corrector populated it, else the consolidated attributes."""
    return dr.corrected_attributes if dr.corrected_attributes else dr.attributes


def ensure_codebook_ids(cache: CodingResultsCache) -> CodingResultsCache:
    """Mint taxonomy ids plus K# on raw_codes (in place).

    Legacy codebooks carry bare attribute names in source_attributes; those are
    resolved to source_attribute_ids against the cache's own effective structure.
    A bare name that exists in several domains resolves to ALL matching ids —
    the documented legacy tolerance (new codebooks resolve per domain, Phase 3).
    """
    ensure_taxonomy_ids(cache)
    knum = _next_counter("K", (c.get("code_id") for c in cache.raw_codes))
    name_to_ids: Dict[str, List[str]] = {}
    for d in _domain_order(cache):
        for attrs in _effective_attributes(cache.partition_results[d]).values():
            for a in attrs:
                if a.get("attribute_name") and a.get("attribute_id"):
                    ids = name_to_ids.setdefault(a["attribute_name"], [])
                    if a["attribute_id"] not in ids:
                        ids.append(a["attribute_id"])
    for c in cache.raw_codes:
        if not c.get("code_id"):
            knum += 1
            c["code_id"] = f"K{knum}"
        if not c.get("source_attribute_ids"):
            c["source_attribute_ids"] = [i for name in c.get("source_attributes", [])
                                         for i in name_to_ids.get(name, [])]
    return cache


def ensure_assignment_ids(responses, structure,
                          codes: Optional[List[dict]] = None):
    """Stamp per-idea domain_id/facet_id/attribute_id (and assigned_code_id when
    `codes` is given) from the idea's effective names, in place.

    `structure` is an id-bearing TaxonomyResultsCache or CodingResultsCache;
    `codes` is its raw_codes list. The __UNASSIGNED__ sentinel passes through as
    assigned_code_id verbatim — it is never a K-id.

    Resolution is (domain, name) against the effective (corrected-first)
    structure, with two legacy tolerances for pre-id artifacts:
    - a stale per-idea domain (ideas predating the 621c0a0 remap fix whose
      attribute was moved cross-domain by P8) resolves via a structure-wide
      unique name — the structure owns placement, the idea's domain field is
      the stale copy;
    - an idea still naming a pre-split attribute (the over-merge corrector
      replaced it in the corrected structure) resolves against the main
      attributes, where the original node still exists.
    Ambiguous or unknown names stay None; nothing is invented.
    """
    domain_ids: Dict[str, str] = {}
    facet_ids: Dict[Tuple[str, str], str] = {}
    attr_ids: Dict[Tuple[str, str], str] = {}
    facet_by_name: Dict[str, List[str]] = {}
    attr_by_name: Dict[str, List[str]] = {}
    for d in _domain_order(structure):
        dr = structure.partition_results[d]
        if dr.domain_id:
            domain_ids[d] = dr.domain_id
        for f in dr.facets:
            if f.get("facet_name") and f.get("facet_id"):
                facet_ids[(d, f["facet_name"])] = f["facet_id"]
                ids = facet_by_name.setdefault(f["facet_name"], [])
                if f["facet_id"] not in ids:
                    ids.append(f["facet_id"])
        # corrected first (effective placement wins), then main as fallback for
        # ideas that predate the over-merge correction
        for attrs_map in (dr.corrected_attributes, dr.attributes):
            for attrs in attrs_map.values():
                for a in attrs:
                    if a.get("attribute_name") and a.get("attribute_id"):
                        attr_ids.setdefault((d, a["attribute_name"]), a["attribute_id"])
                        ids = attr_by_name.setdefault(a["attribute_name"], [])
                        if a["attribute_id"] not in ids:
                            ids.append(a["attribute_id"])

    def _resolve(scoped: Dict[Tuple[str, str], str], by_name: Dict[str, List[str]],
                 domain: str, name: str) -> Optional[str]:
        found = scoped.get((domain, name))
        if found:
            return found
        unique = by_name.get(name)
        return unique[0] if unique and len(unique) == 1 else None
    code_ids = {c["code_name"]: c.get("code_id")
                for c in (codes or []) if c.get("code_name")}

    for resp in responses:
        for idea in (resp.response_ideas or []):
            domain = idea.partition_name or idea.domain or ""
            facet = idea.corrected_facet or idea.facet or ""
            attribute = idea.corrected_attribute or idea.attribute or ""
            if idea.domain_id is None and domain:
                idea.domain_id = domain_ids.get(domain)
            if idea.facet_id is None and facet:
                idea.facet_id = _resolve(facet_ids, facet_by_name, domain, facet)
            if idea.attribute_id is None and attribute:
                idea.attribute_id = _resolve(attr_ids, attr_by_name, domain, attribute)
            assigned = getattr(idea, "assigned_code", None)
            if codes is not None and assigned and getattr(idea, "assigned_code_id", None) is None:
                idea.assigned_code_id = assigned if assigned == UNASSIGNED else code_ids.get(assigned)
    return responses


def restamp_assignment_ids(responses, structure,
                           codes: Optional[List[dict]] = None):
    """Writer-side stamping: clear per-idea placement ids (and assigned_code_id
    when `codes` is given) and re-derive them from the current structure.

    Writers must re-derive rather than fill: a re-consolidated artifact (P7.5/P8/
    corrector rerun) starts from cache-loaded ideas that may carry ids minted
    against the PREVIOUS structure — `ensure_assignment_ids` alone would keep
    those stale ids because it only fills None."""
    for resp in responses:
        for idea in (resp.response_ideas or []):
            idea.domain_id = None
            idea.facet_id = None
            idea.attribute_id = None
            if codes is not None and hasattr(idea, "assigned_code_id"):
                idea.assigned_code_id = None
    return ensure_assignment_ids(responses, structure, codes)


def ensure_ids(obj):
    """Type-dispatching normalizer, hooked into CacheManager metadata loads.
    Structure artifacts get ids minted; anything else passes through untouched."""
    if isinstance(obj, CodingResultsCache):
        return ensure_codebook_ids(obj)
    if isinstance(obj, TaxonomyResultsCache):
        return ensure_taxonomy_ids(obj)
    return obj
