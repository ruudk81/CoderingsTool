"""Exact-dedup of chunk discovery output (step 4).

Facet and attribute discovery each run one independent LLM call per chunk, and
every chunk rediscovers largely the same structure, so the flattened raw yield
contains byte-identical re-proposals.

This module removes ONLY those: the same normalized exact name within the same
scope — the domain for facets, the facet for attributes. The scope is set by the
caller, which passes one domain's or one facet's yield at a time.

Everything that takes a judgment stays out of here. Near-duplicate names are for
the consolidation phase, which sees each candidate together with the observations
that produced it and can tell a rewording from a real distinction.
"""
from typing import Dict, List

from pipeline.step_4_classifier.prompts_attribute import DiscoveredAttribute
from pipeline.step_4_classifier.prompts_facet import DiscoveredFacet


def _norm(name: str) -> str:
    return name.strip().lower()


def dedup_exact_facets(facets: List[DiscoveredFacet]) -> List[DiscoveredFacet]:
    """Collapse facets with the same normalized name within one domain.

    Keeps the first card (deep copy — input untouched) and unions
    example_observations order-preserving. Returns cards in first-seen order.
    """
    merged: Dict[str, DiscoveredFacet] = {}
    for facet in facets:
        key = _norm(facet.facet_name)
        kept = merged.get(key)
        if kept is None:
            merged[key] = facet.model_copy(deep=True)
            continue
        for example in facet.example_observations:
            if example not in kept.example_observations:
                kept.example_observations.append(example)
    return list(merged.values())


def dedup_exact_attributes(
    attributes: List[DiscoveredAttribute],
) -> List[DiscoveredAttribute]:
    """Same collapse for one facet's raw attribute yield."""
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
