"""Exact-dedup of chunk discovery output (step 4).

Discovery runs one independent LLM call per chunk, and every chunk rediscovers
largely the same structure, so the flattened raw yield contains byte-identical
re-proposals — at both levels, since a chunk proposes facets with their
attributes in one go.

This module removes ONLY those: the same normalized exact name within the same
scope. Two chunks proposing "Snelheid" become one facet holding the union of
their attributes, and two proposals of "Wachttijd" inside it become one
attribute holding the union of their examples.

Everything that takes a judgment stays out of here. Near-duplicate names are for
the two consolidation phases, each of which sees its candidates together with
how many passes proposed them and can tell a rewording from a real distinction.
They divide the work by scope, and so does this module: the facet phase compares
every facet of one domain, so `dedup_exact_facets` runs over a domain's whole
raw yield before it; the attribute phase compares the pool of one settled facet,
so `dedup_exact_attributes` runs over each pool as the facet phase assembles it.
"""
from typing import Dict, List

from pipeline.step_4_classifier.prompts_discovery import (
    DiscoveredAttribute, DiscoveredFacet,
)


def _norm(name: str) -> str:
    return name.strip().lower()


def dedup_exact_attributes(
    attributes: List[DiscoveredAttribute],
) -> List[DiscoveredAttribute]:
    """Collapse attributes with the same normalized name within one facet.

    Keeps the first card (deep copy — input untouched) and unions
    example_observations order-preserving. Returns cards in first-seen order.
    """
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


def dedup_exact_facets(facets: List[DiscoveredFacet]) -> List[DiscoveredFacet]:
    """Collapse facets with the same normalized name within one domain.

    A facet carries its attributes, so merging two proposals of the same facet
    means pooling what each of them saw inside it — and that pool gets the same
    exact-dedup one level down. Dropping the second proposal's attributes
    instead would throw away half of what a chunk observed purely because it
    named the container the same way.
    """
    merged: Dict[str, DiscoveredFacet] = {}
    for facet in facets:
        key = _norm(facet.facet_name)
        kept = merged.get(key)
        if kept is None:
            merged[key] = facet.model_copy(deep=True)
            continue
        kept.attributes = dedup_exact_attributes(
            list(kept.attributes) + list(facet.attributes))
    return list(merged.values())
