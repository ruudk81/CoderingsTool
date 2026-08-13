"""De enige plek die step 4's vorm kent.

Step 4 wordt herschreven; alles wat daar van naam of structuur verandert wordt
hier opgevangen, zodat de rest van step 5 er niet van afweet.

Opgevangen betekent niet weggedrukt. Een veld dat leeg is, is een geldige
waarde en wordt hier tot "" gemaakt; een veld dat niet bestáát is een breuk in
het contract met step 4 en moet gooien. Vandaar gewone attribuuttoegang in
plaats van `getattr(..., default)`: die twee gevallen zijn niet uit elkaar te
houden zodra je een default meegeeft, en het tweede geval werd dan stil het
eerste — een hernoemde `valence` maakte élk idee neutraal, waarna de hele
richtingsbepaling in `consolidator.py` verdween zonder één foutmelding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class IdeaUnit:
    idea_id: str
    respondent_id: str
    attribute_id: str
    valence: str          # "+", "-", "0" of ""
    instance: str
    interpretation: str


@dataclass(frozen=True)
class AttributeRef:
    attribute_id: str
    name: str
    definition: str
    domain: str
    facet: str
    boundary_test: str = ""
    exclusions: tuple = ()


def build_idea_units(classified: List[Any]) -> List[IdeaUnit]:
    """Vlak het step-4-groeimodel af tot idea units met respondent-id."""
    units: List[IdeaUnit] = []
    for response in classified:
        for idea in response.response_ideas or []:
            attribute_id = idea.attribute_id or ""
            if not attribute_id:
                continue
            units.append(IdeaUnit(
                idea_id=idea.idea_id,
                respondent_id=str(response.respondent_id),
                attribute_id=attribute_id,
                valence=idea.valence or "",
                instance=idea.instance or "",
                interpretation=idea.interpretation or "",
            ))
    return units


def build_attribute_refs(partition_results: Dict[str, Any]) -> Dict[str, AttributeRef]:
    """Eén AttributeRef per attribuut-id, over alle domeinen en facetten heen."""
    refs: Dict[str, AttributeRef] = {}
    for domain_name, domain in partition_results.items():
        attributes = domain["attributes"] if isinstance(domain, dict) else domain.attributes
        for facet_name, attribute_list in (attributes or {}).items():
            for attribute in attribute_list:
                attribute_id = attribute.get("attribute_id", "")
                if not attribute_id:
                    continue
                refs[attribute_id] = AttributeRef(
                    attribute_id=attribute_id,
                    name=attribute["attribute_name"],
                    definition=(attribute.get("attribute_definition")
                                or attribute.get("attribute_description", "")),
                    domain=domain_name,
                    facet=facet_name,
                    boundary_test=attribute.get("boundary_test", ""),
                    exclusions=tuple(attribute.get("exclusions", []) or []),
                )
    return refs
