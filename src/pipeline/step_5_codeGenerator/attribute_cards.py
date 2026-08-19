"""Wat fase 1 van het model te zien krijgt: één kaart per attribuut.

Bewust drie dingen erop die v1's groeperingsfase NIET toont — het respondent-
aantal, de letterlijke antwoorden en (via run_codebook) de onderzoeksvraag.
Dat weglaten was de reden dat v1 een taak stelde die ook een mens niet kan doen:
groeperen zonder te weten waarvoor of hoe groot iets is.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .concept_inventory import Concept
from .taxonomy_input import IdeaUnit

TOP_ANSWERS = 5


@dataclass(frozen=True)
class AttributeCard:
    attribute_id: str
    name: str
    definition: str
    domain: str
    facet: str
    n_resp: int
    top_answers: Tuple[Tuple[str, int], ...]

    @property
    def tag(self) -> str:
        """Id + naam, zoals overal in step 5. Het id maakt dubbele namen
        onderscheidbaar en draagt zelf geen groepeerbare inhoud."""
        return f"[{self.attribute_id}] {self.name}"


def build_cards(
    concepts: List[Concept],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    top_n: int = TOP_ANSWERS,
) -> List[AttributeCard]:
    """Eén kaart per Concept, in de volgorde waarin de concepten binnenkomen."""
    cards = []
    for concept in concepts:
        answers = Counter(
            unit.instance
            for unit in idea_units_by_attribute.get(concept.attribute_id, [])
            if unit.instance
        )
        cards.append(AttributeCard(
            attribute_id=concept.attribute_id,
            name=concept.name,
            definition=concept.definition,
            domain=concept.domain,
            facet=concept.facet,
            n_resp=concept.n_resp,
            top_answers=tuple(answers.most_common(top_n)),
        ))
    return cards
