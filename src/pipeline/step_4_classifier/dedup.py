"""Exact-dedup of chunk discovery output (step 4).

P2/P3 (facet discovery) and P6 (attribute discovery) run one independent LLM
call per chunk; every chunk rediscovers largely the same structure, so the
flattened raw yield contains byte-identical re-proposals (measured on ASN Qd1:
115 raw facets -> 60 exact-unique in the largest domain). Assignment (P4/P7)
renders the full raw yield as its menu, so every duplicate is paid for in every
assignment call and splits that concept's ideas over interchangeable ids.

This module removes ONLY byte-identical duplicates: same normalized exact name
within the same scope (axis for facets; the per-facet list for attributes).
Near-duplicate names ("Warm en huiselijk" vs "Warm en menselijk") are a
judgment call, and judgments belong to the post-assignment consolidation
(P5/P8), which sees real counts and real texts.
"""
from typing import Dict, List, Tuple

from pipeline.step_4_classifier.prompts_classifier import (
    DiscoveredAttribute,
    DiscoveredFacet,
)


def _norm(name: str) -> str:
    return name.strip().lower()


def dedup_exact_facets(facets: List[DiscoveredFacet]) -> List[DiscoveredFacet]:
    """Collapse facets with the same normalized name on the same axis.

    Keeps the first card (deep copy — input untouched); unions
    example_observations order-preserving; fills empty inclusion/exclusion
    rules from later duplicates. Returns cards in first-seen order.
    """
    merged: Dict[Tuple[str, str], DiscoveredFacet] = {}
    for facet in facets:
        key = (_norm(facet.facet_name), facet.axis)
        kept = merged.get(key)
        if kept is None:
            merged[key] = facet.model_copy(deep=True)
            continue
        for example in facet.example_observations:
            if example not in kept.example_observations:
                kept.example_observations.append(example)
        if not kept.inclusion_rule and facet.inclusion_rule:
            kept.inclusion_rule = facet.inclusion_rule
        if not kept.exclusion_rule and facet.exclusion_rule:
            kept.exclusion_rule = facet.exclusion_rule
    return list(merged.values())


def dedup_exact_attributes(
    attributes: List[DiscoveredAttribute],
) -> List[DiscoveredAttribute]:
    """Same collapse for one facet's raw attribute yield (P6 -> P7)."""
    merged: Dict[str, DiscoveredAttribute] = {}
    for attribute in attributes:
        key = _norm(attribute.attribute_name)
        kept = merged.get(key)
        if kept is None:
            merged[key] = attribute.model_copy(deep=True)
            continue
        for example in attribute.example_observations:
            if example not in kept.example_observations:
                kept.example_observations.append(example)
    return list(merged.values())
