"""Fase 2 en 3 — deterministisch, geen LLM.

Hier zit de garantie die op elke dataset werkt, ook wanneer het model een slechte
dag heeft. Het model levert betekenis; deze module levert vorm: een hele partitie,
zuivere valentie, geen code onder de drempel, en een melding zodra het voorstel
degenereert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .attribute_cards import AttributeCard


@dataclass(frozen=True)
class Group:
    """Eén voorgestelde code, vóór valentiesplitsing."""
    member_ids: Tuple[str, ...]
    proposed_name: str
    explanation: str


def _id_by_tag(cards: List[AttributeCard]) -> Dict[str, str]:
    return {card.tag: card.attribute_id for card in cards}


def repair_partition(result, cards: List[AttributeCard], log=None) -> List[Group]:
    """Maakt van het voorstel een echte partitie: elk attribuut precies één keer.

    De enum in het responsemodel bewaakt het vocabulaire — het model kan geen
    attribuut verzinnen — maar niet de volledigheid: vergeten en dubbel plaatsen
    blijven mogelijk en worden hier deterministisch rechtgezet. Beide altijd
    gelogd, nooit stil.
    """
    tag_to_id = _id_by_tag(cards)
    card_by_id = {card.attribute_id: card for card in cards}

    # Voorstel omzetten naar ids, in de volgorde waarin het model ze gaf.
    proposed: List[Tuple[str, str, List[str]]] = []
    for code in result.codes:
        ids = [tag_to_id[tag] for tag in code.topics if tag in tag_to_id]
        proposed.append((code.code_name, code.explanation, ids))

    # Dubbel geplaatst: toewijzen aan de groep met de meeste respondenten.
    # Gelijkspel: meeste leden, dan alfabetisch op codenaam — reproduceerbaar.
    def weight(index: int) -> tuple:
        name, _explanation, ids = proposed[index]
        return (-sum(card_by_id[i].n_resp for i in ids), -len(ids), name)

    owner: Dict[str, int] = {}
    for index, (_name, _explanation, ids) in enumerate(proposed):
        for attribute_id in ids:
            if attribute_id not in owner or weight(index) < weight(owner[attribute_id]):
                owner[attribute_id] = index

    for attribute_id, winner in owner.items():
        losers = [i for i, (_n, _e, ids) in enumerate(proposed)
                  if attribute_id in ids and i != winner]
        if losers and log is not None:
            log.add(action="PARTITION_DOUBLE", attribute_id=attribute_id,
                    kept_in=proposed[winner][0],
                    removed_from=[proposed[i][0] for i in losers])

    groups = []
    for index, (name, explanation, ids) in enumerate(proposed):
        kept = tuple(i for i in ids if owner.get(i) == index)
        if kept:
            groups.append(Group(member_ids=kept, proposed_name=name,
                                explanation=explanation))

    # Vergeten: elk attribuut dat nergens landde wordt een eigen groep. Of het
    # een code wordt beslist `build_shapes` op de drempel, net als elke andere.
    placed = {i for group in groups for i in group.member_ids}
    for card in cards:
        if card.attribute_id in placed:
            continue
        if log is not None:
            log.add(action="PARTITION_MISSING", attribute_id=card.attribute_id,
                    name=card.name)
        groups.append(Group(member_ids=(card.attribute_id,),
                            proposed_name=card.name, explanation=card.definition))
    return groups
