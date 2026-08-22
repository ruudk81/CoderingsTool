"""The only place that knows step 4's shape.

Step 4 is being rewritten; everything that changes name or structure there is
absorbed here, so the rest of step 5 knows nothing about it.

Absorbed does not mean suppressed. A field that is empty is a valid value and is
turned into "" here; a field that does not EXIST is a break in the contract with
step 4 and must throw. Hence plain attribute access rather than
`getattr(..., default)`: those two cases become indistinguishable the moment you
supply a default, and the second then silently became the first — a renamed
`valence` made every idea neutral, after which the whole direction assignment in
`consolidator.py` disappeared without a single error.
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
    # Step 4 bouwt onder elk facet een vangnet-attribuut en markeert dat met
    # `drain_key`. Herkennen gebeurt op die sleutel en nooit op de naam: die
    # staat in de enquetetaal en mag hernoemd worden (step 4's drains.py legt
    # die afspraak vast). Afwezigheid van de sleutel IS hier de betekenis, dus
    # `.get` is juist — dezelfde uitzondering als bij de definitie hieronder.
    is_drain: bool = False


def build_idea_units(classified: List[Any]) -> List[IdeaUnit]:
    """Flatten the step-4 growing model into idea units with a respondent id."""
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
                    is_drain=bool(attribute.get("drain_key")),
                )
    return refs
