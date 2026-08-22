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
    exclude_drains: bool = False,
) -> List[AttributeCard]:
    """Eén kaart per Concept, in de volgorde waarin de concepten binnenkomen.

    `exclude_drains` laat step 4's vangnetten weg. Niet omdat ze onderwerploos
    zijn — dat is een weerlegde motivering die hier tot 2026-08-21 stond. Een
    vangnet is FACET-GEBONDEN: "de rest binnen Politieke richting" heeft een
    onderwerp, alleen niet gespecificeerd, en het bij de hoofdcode van dat facet
    zetten is verdedigbaar.

    De reden is wat zo'n merge met een METING doet. Op de ASN-set kwam elk
    vangnet 28-29 van 30 runs samen met zijn naamgenoot-attribuut — een facet-
    en naamovereenkomst, dus bijna automatisch — en stond daarmee bovenaan de
    co-associatiematrix over bakjes met een of twee respondenten. Dat is
    recurrentie zonder prevalentie, niet een eigenschap van vangnetten.

    Let op de prijs: `apply_overig_sweep` maakt EEN globale `Overig` voor de hele
    dataset, dus met deze vlag aan verliezen die respondenten hun facetcontext.
    Er is een derde weg (van de kaarten af, maar deterministisch naar de code van
    het eigen facet, zoals `pool_thin_within_facet` het al doet) — zie WORK.md.

    **Staat standaard UIT tot het promotiebesluit over het consensus-experiment
    valt.** Zonder de vlag is het gedrag identiek aan voor 2026-08-20. Het
    experiment zet hem aan; `run_codebook.py` niet.
    """
    cards = []
    for concept in concepts:
        if exclude_drains and concept.is_drain:
            continue
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
