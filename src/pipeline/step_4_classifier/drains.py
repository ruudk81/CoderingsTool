"""The two other-catch-alls of step 4, built deterministically.

Assignment picks one attribute per label from a domain-wide menu. Two things can
go wrong there: the idea belongs to a facet but to none of its attributes, or it
belongs to the domain but to none of its facets. The menu has an exit for both —
an `other` attribute under every facet, and an `other` facet under every domain
with an `other` attribute of its own.

That guarantees a valid answer always exists and `__UNASSIGNED__` disappears: an
idea that fitted nowhere used to get a name that was not in the structure, and
everything downstream had to make an exception for it.

**Deterministic, not discovered.** They are built by this module and never
proposed by a model. Step 3 already learned that with its two catch-all domains:
a model asked to invent the residual category sometimes does not, and the
responses then get pushed into a substantive category instead.

**Recognition runs on the key, never on the name.** The name is in the survey
language and may be rewritten by refinement; `drain_key` may not. Same agreement
as in step 3, where `other` and `not_known` are the keys and the label is
translated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Stable across runs, languages and renames. Whoever wants to recognise a
# catch-all reads this key; whoever matches on the name has it wrong at the first
# translation.
DRAIN_ATTRIBUTE_KEY = "other_in_facet"
DRAIN_FACET_KEY = "other_in_domain"


# Just the residual wording, per language. This is format, not use-case content:
# it says nothing about the subject of the survey. An unknown language falls back
# to English — visibly wrong rather than silently empty.
_WORDING: Dict[str, Dict[str, str]] = {
    "english": {
        "other": "Other",
        "attribute_definition": (
            "Responses that belong to this facet but match none of the "
            "attributes listed under it."
        ),
        "facet_definition": (
            "Responses that belong to this domain but match none of the facets "
            "listed under it."
        ),
    },
    "dutch": {
        "other": "Overig",
        "attribute_definition": (
            "Responsen die bij dit facet horen maar bij geen van de attributen "
            "die eronder staan."
        ),
        "facet_definition": (
            "Responsen die bij dit domein horen maar bij geen van de facetten "
            "die eronder staan."
        ),
    },
}

_LANGUAGE_ALIASES = {"nederlands": "dutch", "nl": "dutch", "nl-nl": "dutch",
                     "en": "english", "en-gb": "english", "en-us": "english"}


def _wording(language: str) -> Dict[str, str]:
    key = (language or "").strip().lower()
    key = _LANGUAGE_ALIASES.get(key, key)
    return _WORDING.get(key, _WORDING["english"])


# =============================================================================
# CONSTRUCTIE
# =============================================================================

def _drain_attribute(name: str, definition: str) -> Dict[str, Any]:
    """The shape of a catch-all attribute, in one place."""
    return {
        "attribute_name": name,
        "attribute_definition": definition,
        "example_observations": [],
        "is_drain": True,
        "drain_key": DRAIN_ATTRIBUTE_KEY,
    }


def make_drain_attribute(facet_name: str, language: str) -> Dict[str, Any]:
    """The `other` attribute under one facet.

    The name carries the facet, because a codebook with ten bare residual codes
    is unreadable and step 5 has to be able to tell them apart.
    """
    words = _wording(language)
    return _drain_attribute(f"{words['other']} — {facet_name}",
                            words["attribute_definition"])


def make_drain_facet(domain_label: str, language: str) -> Dict[str, Any]:
    """The `other` facet under one domain, with an `other` attribute of its own.

    Without that attribute the domain catch-all would itself be a hole:
    assignment picks an attribute, so a facet without attributes is unreachable.

    Facet and attribute share name and definition, and that is not sloppiness:
    it IS one bucket. The structure forces it to be expressed at two levels, and
    then saying the same thing twice is more honest than an invented distinction
    — a doubly-residual name promised a refinement that is not there.
    """
    words = _wording(language)
    name = f"{words['other']} — {domain_label}"
    return {
        "facet_name": name,
        "facet_definition": words["facet_definition"],
        "is_drain": True,
        "drain_key": DRAIN_FACET_KEY,
        "attributes": [_drain_attribute(name, words["facet_definition"])],
    }


# =============================================================================
# HERKENNING
# =============================================================================

def is_drain_item(item: Dict[str, Any]) -> bool:
    """Whether this item is a catch-all — by key, not by name."""
    return bool(item.get("drain_key"))


# =============================================================================
# OPRUIMEN
# =============================================================================

def strip_empty_drains(
    facets: Dict[str, List[Dict[str, Any]]],
    attributes: Dict[str, Dict[str, List[Dict[str, Any]]]],
    attribute_assignments: Dict[str, str],
) -> Tuple[Dict[str, List[Dict[str, Any]]],
           Dict[str, Dict[str, List[Dict[str, Any]]]],
           Dict[str, str]]:
    """Remove the catch-alls that caught nothing.

    A catch-all is an offer, not a category: if it stayed empty it does not
    belong in the delivered taxonomy. Whatever did fill up stays — that is a real
    finding about coverage.

    Does not mutate the input; the caller keeps its own state.
    """
    used = set(attribute_assignments.values())

    out_attributes: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for domain, per_facet in attributes.items():
        kept_per_facet: Dict[str, List[Dict[str, Any]]] = {}
        for facet_name, items in per_facet.items():
            kept = [dict(a) for a in items
                    if not is_drain_item(a) or a.get("attribute_name") in used]
            if kept:
                kept_per_facet[facet_name] = kept
        out_attributes[domain] = kept_per_facet

    out_facets: Dict[str, List[Dict[str, Any]]] = {}
    for domain, items in facets.items():
        kept = []
        for facet in items:
            if not is_drain_item(facet):
                kept.append(dict(facet))
                continue
            # A catch-all facet survives only while it still holds attributes.
            if out_attributes.get(domain, {}).get(facet.get("facet_name")):
                survivor = dict(facet)
                survivor["attributes"] = [
                    dict(a) for a in out_attributes[domain][facet["facet_name"]]]
                kept.append(survivor)
        out_facets[domain] = kept

    return out_facets, out_attributes, attribute_assignments
