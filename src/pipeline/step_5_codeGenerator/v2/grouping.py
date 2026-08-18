"""Fase 2 en 3 — deterministisch, geen LLM.

Hier zit de garantie die op elke dataset werkt, ook wanneer het model een slechte
dag heeft. Het model levert betekenis; deze module levert vorm: een hele partitie,
zuivere valentie, geen code onder de drempel, en een melding zodra het voorstel
degenereert.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .attribute_cards import AttributeCard
from ..concept_inventory import Concept
from ..consolidator import CodeShape


@dataclass(frozen=True)
class Group:
    """Eén voorgestelde code, vóór valentiesplitsing."""
    member_ids: Tuple[str, ...]
    proposed_name: str
    explanation: str


def _id_by_tag(cards: List[AttributeCard]) -> Dict[str, str]:
    return {card.tag: card.attribute_id for card in cards}


def repair_partition(result, cards: List[AttributeCard],
                     concepts: List[Concept], log=None) -> List[Group]:
    """Maakt van het voorstel een echte partitie: elk attribuut precies één keer.

    De enum in het responsemodel bewaakt het vocabulaire — het model kan geen
    attribuut verzinnen — maar niet de volledigheid: vergeten, dubbel geplaatst
    en tweemaal in dezelfde groep genoemd blijven mogelijk en worden hier
    deterministisch rechtgezet. Elke reparatie wordt gelogd, nooit stil.

    `concepts` levert de respondent-sets voor de dubbel-plaatsing-afweging. Dat
    zit niet op `AttributeCard` — die kaart is bewust precies wat het model te
    zien krijgt, en de bewakingsfase mag meer weten dan het model.
    """
    tag_to_id = _id_by_tag(cards)
    resp_by_id = {concept.attribute_id: concept.resp_ids for concept in concepts}

    # Voorstel omzetten naar ids, in de volgorde waarin het model ze gaf. Eén
    # tag kan tweemaal in dezelfde groep staan — List[Literal[...]] verbiedt
    # dat niet — dus dedupliceren met behoud van volgorde, en elke inkorting
    # loggen.
    proposed: List[Tuple[str, str, List[str]]] = []
    for code in result.codes:
        ids = [tag_to_id[tag] for tag in code.topics if tag in tag_to_id]
        counts = Counter(ids)
        deduped = list(dict.fromkeys(ids))
        if log is not None:
            for attribute_id in deduped:
                if counts[attribute_id] > 1:
                    log.add(action="PARTITION_DUPLICATE_IN_GROUP",
                            attribute_id=attribute_id, group=code.code_name)
        proposed.append((code.code_name, code.explanation, deduped))

    # Dubbel geplaatst: toewijzen aan de groep met de meeste respondenten — de
    # unie van de leden, nooit de som, anders telt een respondent die in twee
    # attributen van dezelfde groep zit dubbel mee (zie concept_inventory.py).
    # Gelijkspel: meeste leden, dan alfabetisch op codenaam — reproduceerbaar.
    def weight(index: int) -> tuple:
        name, _explanation, ids = proposed[index]
        respondents = (frozenset().union(*(resp_by_id[i] for i in ids))
                       if ids else frozenset())
        return (-len(respondents), -len(ids), name)

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


@dataclass(frozen=True)
class ShapingResult:
    shapes: List[CodeShape]
    overig_ids: List[str]
    direction_loss: int


def build_shapes(
    groups: List[Group], concepts: List[Concept], threshold: int,
) -> ShapingResult:
    """Elke groep wordt gesplitst in zijn valentiepolen; elke pool die de drempel
    zelfstandig haalt wordt één code.

    Daarmee is 'geen mix van + en −' een eigenschap van de constructie: een code
    ÍS een pool. In v1 was dit een fallback die de tegengestelde respondenten
    meedroeg zodra maar één pool de drempel haalde — het gat waardoor 17 codes
    een richting claimden die hun inhoud niet had.

    Een pool die de drempel niet haalt gaat naar Overig en telt mee in
    `direction_loss`. Haalt geen enkele pool van een groep de drempel, dan gaan
    de attributen zelf naar Overig.
    """
    concept_by_id = {c.attribute_id: c for c in concepts}
    shapes: List[CodeShape] = []
    overig_ids: List[str] = []
    direction_loss = 0

    for group in groups:
        members = [concept_by_id[i] for i in group.member_ids if i in concept_by_id]
        if not members:
            continue
        poles = {
            "positive": frozenset().union(*(m.resp_pos for m in members)),
            "negative": frozenset().union(*(m.resp_neg for m in members)),
            "neutral": frozenset().union(*(m.resp_neu for m in members)),
        }
        kept = {v: r for v, r in poles.items() if len(r) >= threshold}
        if not kept:
            overig_ids.extend(group.member_ids)
            direction_loss += len(frozenset().union(*poles.values()))
            continue

        # Unie, niet som: een respondent kan zowel een positief als een
        # negatief idee bij hetzelfde attribuut hebben, en zou anders dubbel
        # meetellen als beide polen onder de drempel blijven.
        dropped = (r for v, r in poles.items() if v not in kept)
        direction_loss += len(frozenset().union(*dropped))
        for valence in ("positive", "negative", "neutral"):
            if valence not in kept:
                continue
            resp = kept[valence]
            shapes.append(CodeShape(
                key=f"V{len(shapes) + 1}",
                members=group.member_ids,
                valence=valence,
                umbrella=group.proposed_name,
                resp_ids=resp,
                resp_pos=resp if valence == "positive" else frozenset(),
                resp_neg=resp if valence == "negative" else frozenset(),
                resp_neu=resp if valence == "neutral" else frozenset(),
                origin="pooled" if len(group.member_ids) > 1 else "solo",
            ))
    return ShapingResult(shapes=shapes, overig_ids=overig_ids,
                         direction_loss=direction_loss)
