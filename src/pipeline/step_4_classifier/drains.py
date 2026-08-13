"""De twee other-vangnetten van step 4, deterministisch gebouwd.

Toewijzing kiest per label één attribuut uit een domeinbreed menu. Twee dingen
kunnen dan misgaan: het idee hoort bij een facet maar bij geen van zijn
attributen, of het hoort bij het domein maar bij geen van zijn facetten. Voor
allebei staat er een uitgang in het menu — een `other`-attribuut onder elk
facet, en een `other`-facet onder elk domein met zijn eigen `other`-attribuut.

Daarmee is er altijd een geldig antwoord en verdwijnt `__UNASSIGNED__`: een idee
dat nergens paste kreeg een naam die niet in de structuur stond, en alles
stroomafwaarts moest daar een uitzondering voor maken.

**Deterministisch, niet ontdekt.** Ze worden door deze module gebouwd en nooit
door een model voorgesteld. Step 3 leerde dat al met zijn twee vangnetdomeinen:
een model dat de restcategorie zelf moet bedenken doet dat soms niet, en de
antwoorden worden dan een inhoudelijke categorie in geduwd.

**Herkenning loopt op de sleutel, nooit op de naam.** De naam staat in de
enquêtetaal en mag door naslijpen herschreven worden; `drain_key` niet. Dat is
dezelfde afspraak als bij step 3, waar `other` en `not_known` de sleutels zijn
en het label vertaald is.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Stabiel over runs, talen en hernoemingen heen. Wie een vangnet wil herkennen
# leest deze sleutel; wie op de naam matcht heeft het bij de eerste vertaling
# mis.
DRAIN_ATTRIBUTE_KEY = "other_in_facet"
DRAIN_FACET_KEY = "other_in_domain"


# Alleen de restwoorden, per taal. Dit is opmaak, geen use-case-inhoud: het zegt
# niets over het onderwerp van de enquête. Onbekende taal valt terug op Engels,
# zichtbaar fout in plaats van stil leeg.
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

def make_drain_attribute(facet_name: str, language: str) -> Dict[str, Any]:
    """Het `other`-attribuut onder één facet.

    De naam draagt het facet, want een codeboek met tien kale "Overig"-codes is
    onleesbaar en step 5 moet ze uit elkaar kunnen houden.
    """
    words = _wording(language)
    return {
        "attribute_name": f"{words['other']} — {facet_name}",
        "attribute_definition": words["attribute_definition"],
        "example_observations": [],
        "is_drain": True,
        "drain_key": DRAIN_ATTRIBUTE_KEY,
    }


def make_drain_facet(domain_label: str, language: str) -> Dict[str, Any]:
    """Het `other`-facet onder één domein, mét zijn eigen `other`-attribuut.

    Zonder dat attribuut zou het domein-vangnet zelf een gat zijn: toewijzing
    kiest een attribuut, dus een facet zonder attributen is onbereikbaar.
    """
    words = _wording(language)
    facet_name = f"{words['other']} — {domain_label}"
    return {
        "facet_name": facet_name,
        "facet_definition": words["facet_definition"],
        "is_drain": True,
        "drain_key": DRAIN_FACET_KEY,
        "attributes": [make_drain_attribute(facet_name, language)],
    }


# =============================================================================
# HERKENNING
# =============================================================================

def is_drain_item(item: Dict[str, Any]) -> bool:
    """Of dit item een vangnet is — op sleutel, niet op naam."""
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
    """Haal de vangnetten weg die niets hebben opgevangen.

    Een vangnet is een aanbod, geen categorie: bleef het leeg, dan hoort het
    niet in de opgeleverde taxonomie. Wat wél gevuld raakte blijft staan — dat
    is een echte bevinding over de dekking.

    Muteert de invoer niet; de aanroeper houdt zijn eigen stand.
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
            # Een vangnetfacet overleeft alleen zolang het nog attributen heeft.
            if out_attributes.get(domain, {}).get(facet.get("facet_name")):
                survivor = dict(facet)
                survivor["attributes"] = [
                    dict(a) for a in out_attributes[domain][facet["facet_name"]]]
                kept.append(survivor)
        out_facets[domain] = kept

    return out_facets, out_attributes, attribute_assignments
