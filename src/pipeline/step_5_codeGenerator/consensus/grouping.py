"""Fase 2 en 3 — deterministisch, geen LLM.

Hier zit de garantie die op elke dataset werkt, ook wanneer het model een slechte
dag heeft. Het model levert betekenis; deze module levert vorm: een hele partitie,
zuivere valentie, geen code onder de drempel, en een melding zodra het voorstel
degenereert.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .attribute_cards import AttributeCard
from .prompts_consolidation import ConsolidationResult
from .concept_inventory import Concept
from .code_shape import CodeShape


@dataclass(frozen=True)
class Group:
    """Eén voorgestelde code, vóór valentiesplitsing."""
    member_ids: Tuple[str, ...]
    proposed_name: str
    explanation: str


def _id_by_tag(cards: List[AttributeCard]) -> Dict[str, str]:
    return {card.tag: card.attribute_id for card in cards}


def repair_partition(result: ConsolidationResult, cards: List[AttributeCard],
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


def valence_poles(members: List[Concept], two_pole: bool) -> Dict[str, frozenset]:
    """De valentiepolen van een groep leden — de eenheid waarin een code bestaat.

    Staat apart omdat TWEE fasen hem nodig hebben en ze anders uit elkaar
    lopen: `build_shapes` laat een code alleen bestaan als een POOL de drempel
    haalt, dus `pool_thin_within_facet` moet op diezelfde grootheid oordelen.
    Toetste die op het groepstotaal, dan poolde hij groepen die daarna alsnog
    omvielen — gemeten, en precies waarom deze functie bestaat.
    """
    if two_pole:
        return {
            "non_negative": frozenset().union(
                *(m.resp_pos for m in members), *(m.resp_neu for m in members)),
            "negative": frozenset().union(*(m.resp_neg for m in members)),
        }
    return {
        "positive": frozenset().union(*(m.resp_pos for m in members)),
        "negative": frozenset().union(*(m.resp_neg for m in members)),
        "neutral": frozenset().union(*(m.resp_neu for m in members)),
    }


def pool_thin_within_facet(
    groups: List[Group], concepts: List[Concept], threshold: int,
    two_pole: bool = False,
) -> Tuple[List[Group], List[dict]]:
    """Groepen die de drempel niet halen gaan samen met hun facetgenoten.

    Een attribuut dat te dun is voor een eigen code, en dat de consolidatie bij
    niemand heeft ondergebracht, verdwijnt vandaag in Overig — ook wanneer het
    facet ernaast nog twee even dunne buren heeft die er samen wél komen. Deze
    fase pakt dat op, en doet dat op step 4's eigen structuur in plaats van er
    een nieuwe te verzinnen.

    Drie grenzen, alle drie per constructie en niet per instructie:

    - **Alleen materiaal onder de drempel.** Een groep die het op eigen kracht
      haalt wordt niet aangeraakt, dus deze fase kan nooit een dikke code maken.
    - **Nooit over een facetgrens.** Groeperen over facetten heen is het werk
      van de consolidatiecall; hier wordt alleen samengeraapt wat step 4 al
      onder één noemer had staan. Een groep die zelf al twee facetten omvat
      heeft geen eenduidig facet en blijft daarom zoals hij is.
    - **Geen eigen knop.** De enige grens is `threshold`, dezelfde `t_keep` die
      de rest van de fase gebruikt en die met de steekproef meeschaalt. Had
      deze fase een eigen getal nodig, dan zou dat een getal zijn dat op één
      dataset is afgesteld.

    Haalt een facetpool het samen ook niet, dan gebeurt er niets: de groepen
    blijven los en `build_shapes` stuurt ze naar Overig, zoals nu.

    Respondenten worden VERENIGD, nooit opgeteld — wie op twee attributen van
    hetzelfde facet antwoordde telt één keer.
    """
    concept_by_id = {concept.attribute_id: concept for concept in concepts}
    facet_by_id = {concept.attribute_id: concept.facet for concept in concepts}

    def best_pole(member_ids) -> int:
        """De sterkste valentiepool — exact wat `build_shapes` straks eist."""
        members = [concept_by_id[m] for m in member_ids if m in concept_by_id]
        if not members:
            return 0
        return max(len(r) for r in valence_poles(members, two_pole).values())

    def reach(member_ids) -> frozenset:
        sets = [concept_by_id[m].resp_ids for m in member_ids if m in concept_by_id]
        return frozenset().union(*sets) if sets else frozenset()

    def sole_facet(member_ids) -> Optional[str]:
        facets = {facet_by_id[m] for m in member_ids if m in facet_by_id}
        return facets.pop() if len(facets) == 1 else None

    kept: List[Group] = []
    thin: Dict[str, List[Group]] = {}
    for group in groups:
        facet = sole_facet(group.member_ids)
        if facet is None or best_pole(group.member_ids) >= threshold:
            kept.append(group)
        else:
            thin.setdefault(facet, []).append(group)

    log: List[dict] = []
    for facet, members in sorted(thin.items()):
        ids = tuple(m for group in members for m in group.member_ids)
        if len(members) < 2 or best_pole(ids) < threshold:
            kept.extend(members)
            continue
        kept.append(Group(member_ids=tuple(sorted(ids)),
                          proposed_name=facet, explanation=""))
        log.append({"action": "THIN_POOLED_IN_FACET", "facet": facet,
                    "members": sorted(ids), "n_resp": len(reach(ids))})
    return kept, log


@dataclass(frozen=True)
class ShapingResult:
    shapes: List[CodeShape]
    overig_ids: List[str]
    direction_loss: int


def build_shapes(
    groups: List[Group], concepts: List[Concept], threshold: int,
    two_pole: bool = False,
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

    `two_pole` vervangt de driedeling door niet-negatief (positief ∪ neutraal)
    tegenover negatief. De +/0-grens is gemeten ruis bij kale associaties, en
    een samengevoegde pool haalt `t_keep` vaker, dus `direction_loss` daalt.
    Experimenteel — de productieketen draait op de driedeling.
    """
    concept_by_id = {c.attribute_id: c for c in concepts}
    shapes: List[CodeShape] = []
    overig_ids: List[str] = []
    direction_loss = 0

    for group in groups:
        members = [concept_by_id[i] for i in group.member_ids if i in concept_by_id]
        if not members:
            # Onbereikbaar vandaag (elk lid komt uit `repair_partition`, dat
            # zelf uit `cards`/`concepts` put), maar als de aanname ooit breekt
            # moet de boekhouding heel blijven: naar Overig, niet stil weg.
            overig_ids.extend(group.member_ids)
            continue
        poles = valence_poles(members, two_pole)
        order = (("non_negative", "negative") if two_pole
                 else ("positive", "negative", "neutral"))
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
        for valence in order:
            if valence not in kept:
                continue
            resp = kept[valence]
            if valence == "non_negative":
                resp_pos = frozenset().union(*(m.resp_pos for m in members))
                resp_neu = frozenset().union(*(m.resp_neu for m in members))
                resp_neg = frozenset()
            else:
                resp_pos = resp if valence == "positive" else frozenset()
                resp_neg = resp if valence == "negative" else frozenset()
                resp_neu = resp if valence == "neutral" else frozenset()
            shapes.append(CodeShape(
                key=f"V{len(shapes) + 1}",
                members=group.member_ids,
                valence=valence,
                umbrella=group.proposed_name,
                resp_ids=resp,
                resp_pos=resp_pos,
                resp_neg=resp_neg,
                resp_neu=resp_neu,
                origin="pooled" if len(group.member_ids) > 1 else "solo",
            ))
    return ShapingResult(shapes=shapes, overig_ids=overig_ids,
                         direction_loss=direction_loss)


DEGENERATION_FLOOR = 0.05
DEGENERATION_CEILING = 0.90


def check_degeneration(n_groups: int, n_attributes: int) -> Optional[str]:
    """Is er überhaupt geconsolideerd, en is niet alles op één hoop gegooid?

    Beide grenzen zijn RELATIEF aan de input, nooit absoluut: een vaste
    ondergrens zou op een dataset met twintig attributen een correcte uitkomst
    afkeuren. De vraag is niet 'zijn het er genoeg' — dat is een oordeel dat
    geen deterministische toets kan vellen — maar of het voorstel is ontaard.

    De twee factoren zijn beredeneerd, niet gemeten; ze horen bijgesteld te
    worden zodra er runs op meer dan één dataset zijn — wie ze verschuift moet
    weten welke kant gezond is: de vergelijkingen zijn strikt (`>` / `<`), dus
    het interval is GESLOTEN — exact op `DEGENERATION_FLOOR * n_attributes` of
    exact op `DEGENERATION_CEILING * n_attributes` telt als gezond, niet als
    ontaard.
    """
    if n_attributes == 0:
        return None
    if n_groups > DEGENERATION_CEILING * n_attributes:
        return (f"geen consolidatie: {n_groups} groepen op {n_attributes} attributen "
                f"(grens {DEGENERATION_CEILING:.0%})")
    if n_groups < DEGENERATION_FLOOR * n_attributes:
        return (f"alles op één hoop: {n_groups} groepen op {n_attributes} attributen "
                f"(grens {DEGENERATION_FLOOR:.0%})")
    return None
