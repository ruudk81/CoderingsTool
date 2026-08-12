"""Stap 3 — samenvoegen en richting bepalen. Tellen en vergelijken, geen LLM.

Prevalentie bepaalt OF iets samenvoegt; semantiek bepaalt alleen WAARHEEN. Een
concept voegt niet samen omdat het op iets anders lijkt — het voegt samen omdat
het te klein is om op zichzelf te staan.

Volgorde: ontdubbelen (synoniemen, ongeacht grootte) -> klimmen (per koepel, en
zo nodig één niveau hoger naar het taxonomiedomein) -> richting (per resulterende
vorm, op basis van de samengevoegde respondentverzamelingen).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .concept_inventory import Concept
from .prompts_relations import tagged


@dataclass(frozen=True)
class RelationMap:
    """Genormaliseerde LLM-relaties, gekeyd op attribuut-id."""
    umbrella: Dict[str, str]
    synonym_of: Dict[str, str]


@dataclass(frozen=True)
class CodeShape:
    """Eén uitkomst van het consolideren: hoeveel codes, welke leden, welke
    richting. `key` is een run-lokale volgordesleutel, geen codeboek-id."""
    key: str
    members: Tuple[str, ...]
    valence: str
    umbrella: str
    resp_ids: frozenset[str]
    resp_pos: frozenset[str]
    resp_neg: frozenset[str]
    resp_neu: frozenset[str]
    origin: str


@dataclass(frozen=True)
class _Unit:
    """Eén of meer attributen die samen als één geheel klimmen: een los concept,
    een ontdubbelde synoniemgroep, of een gepoolde groep."""
    ids: Tuple[str, ...]
    umbrella: str
    domain: str
    origin: str
    resp_pos: frozenset[str]
    resp_neg: frozenset[str]
    resp_neu: frozenset[str]

    @property
    def resp_ids(self) -> frozenset[str]:
        return self.resp_pos | self.resp_neg | self.resp_neu


def normalize_relations(result, concepts: List[Concept]) -> RelationMap:
    """Vertaalt `RelationsResult`'s gekwalificeerde namen (`tagged()`-formaat)
    terug naar attribuut-ids. Een `synonym_of` die niet in de lijst staat wordt
    genegeerd; kettingen (A->B->C) worden hier niet ontward, dat doet `consolidate`."""
    id_by_tag = {tagged(concept): concept.attribute_id for concept in concepts}
    umbrella: Dict[str, str] = {}
    synonym_of: Dict[str, str] = {}
    for relation in result.relations:
        attribute_id = id_by_tag.get(relation.attribute)
        if attribute_id is None:
            continue
        umbrella[attribute_id] = relation.umbrella_name
        if relation.synonym_of is not None:
            target_id = id_by_tag.get(relation.synonym_of)
            if target_id is not None:
                synonym_of[attribute_id] = target_id
    return RelationMap(umbrella=umbrella, synonym_of=synonym_of)


def _synonym_groups(concepts: List[Concept], synonym_of: Dict[str, str]) -> List[List[Concept]]:
    """Samenhangende componenten over `synonym_of`: een keten A->B->C wordt één
    groep ongeacht de volgorde waarin de paren zijn opgegeven."""
    parent: Dict[str, str] = {concept.attribute_id: concept.attribute_id for concept in concepts}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for member_id, target_id in synonym_of.items():
        if member_id in parent and target_id in parent:
            union(member_id, target_id)

    groups: Dict[str, List[Concept]] = defaultdict(list)
    for concept in concepts:
        groups[find(concept.attribute_id)].append(concept)
    return list(groups.values())


def _build_units(concepts: List[Concept], relations: RelationMap) -> List[_Unit]:
    """Ontdubbelt synoniemgroepen tot units met de vereniging van hun
    respondentverzamelingen. De koepel komt van de meest prevalente naam in de
    groep — bij een synoniempaar wijzen beide leden doorgaans naar dezelfde
    koepel, maar bij afwijking wint het meest prevalente lid."""
    units = []
    for group in _synonym_groups(concepts, relations.synonym_of):
        representative = min(group, key=lambda c: (-c.n_resp, c.attribute_id))
        units.append(_Unit(
            ids=tuple(sorted(c.attribute_id for c in group)),
            umbrella=relations.umbrella.get(representative.attribute_id, representative.domain),
            domain=representative.domain,
            origin="synonym" if len(group) > 1 else "solo",
            resp_pos=frozenset().union(*(c.resp_pos for c in group)),
            resp_neg=frozenset().union(*(c.resp_neg for c in group)),
            resp_neu=frozenset().union(*(c.resp_neu for c in group)),
        ))
    return units


def _pool(units: List[_Unit], umbrella: str) -> _Unit:
    return _Unit(
        ids=tuple(sorted(id_ for unit in units for id_ in unit.ids)),
        umbrella=umbrella,
        domain=units[0].domain,
        origin="pooled",
        resp_pos=frozenset().union(*(unit.resp_pos for unit in units)),
        resp_neg=frozenset().union(*(unit.resp_neg for unit in units)),
        resp_neu=frozenset().union(*(unit.resp_neu for unit in units)),
    )


def _group_leftovers(units: List[_Unit], key) -> Tuple[List[str], Dict[str, List[_Unit]]]:
    order: List[str] = []
    grouped: Dict[str, List[_Unit]] = defaultdict(list)
    for unit in units:
        group_key = key(unit)
        if group_key not in grouped:
            order.append(group_key)
        grouped[group_key].append(unit)
    return order, grouped


def _climb(units: List[_Unit], threshold: int) -> Tuple[List[_Unit], List[str]]:
    """Leden boven de drempel staan op zichzelf en klimmen niet. De rest wordt
    per koepel gepoold; haalt die pool de drempel niet, dan wordt één niveau
    hoger geprobeerd — het taxonomiedomein, de enige bredere groepering die een
    Concept nog draagt. Haalt ook dat niet, dan is er geen niveau meer: Overig."""
    kept: List[_Unit] = []
    short: List[_Unit] = []
    for unit in units:
        (kept if len(unit.resp_ids) >= threshold else short).append(unit)

    domain_candidates: List[_Unit] = []
    umbrella_order, by_umbrella = _group_leftovers(short, lambda unit: unit.umbrella)
    for umbrella in umbrella_order:
        pooled = _pool(by_umbrella[umbrella], umbrella)
        if len(pooled.resp_ids) >= threshold:
            kept.append(pooled)
        else:
            domain_candidates.extend(by_umbrella[umbrella])

    overig: List[str] = []
    domain_order, by_domain = _group_leftovers(domain_candidates, lambda unit: unit.domain)
    for domain in domain_order:
        pooled = _pool(by_domain[domain], domain)
        if len(pooled.resp_ids) >= threshold:
            kept.append(pooled)
        else:
            for unit in by_domain[domain]:
                overig.extend(unit.ids)

    return kept, overig


def _directions(
    resp_pos: frozenset[str], resp_neg: frozenset[str], resp_neu: frozenset[str], threshold: int
) -> List[Tuple[str, frozenset[str], frozenset[str], frozenset[str]]]:
    """Eén vorm wordt er twee (of drie) zodra beide polen de drempel apart
    halen; een enkele pool over de drempel geeft één gerichte vorm die de héle
    groep draagt (er is geen tweede vorm om de rest in te vangen); haalt geen
    enkele pool de drempel, dan blijft het één neutrale vorm over de hele groep."""
    p, g, u = len(resp_pos), len(resp_neg), len(resp_neu)
    if p >= threshold and g >= threshold:
        if u >= threshold:
            return [
                ("positive", resp_pos, frozenset(), frozenset()),
                ("negative", frozenset(), resp_neg, frozenset()),
                ("neutral", frozenset(), frozenset(), resp_neu),
            ]
        if p >= g:
            return [
                ("positive", resp_pos, frozenset(), resp_neu),
                ("negative", frozenset(), resp_neg, frozenset()),
            ]
        return [
            ("positive", resp_pos, frozenset(), frozenset()),
            ("negative", frozenset(), resp_neg, resp_neu),
        ]
    if p >= threshold:
        return [("positive", resp_pos, resp_neg, resp_neu)]
    if g >= threshold:
        return [("negative", resp_pos, resp_neg, resp_neu)]
    return [("neutral", resp_pos, resp_neg, resp_neu)]


def consolidate(
    concepts: List[Concept], relations: RelationMap, threshold: int, log: Optional[object] = None
) -> Tuple[List[CodeShape], List[str]]:
    """Ontdubbelen -> klimmen -> richting. `log` is gereserveerd voor het
    beslislog (een volgende stap); deze functie vult het nog niet."""
    units = _build_units(concepts, relations)
    kept, overig = _climb(units, threshold)

    shapes: List[CodeShape] = []
    for unit in kept:
        for valence, resp_pos, resp_neg, resp_neu in _directions(
            unit.resp_pos, unit.resp_neg, unit.resp_neu, threshold
        ):
            shapes.append(CodeShape(
                key=f"S{len(shapes) + 1}",
                members=unit.ids,
                valence=valence,
                umbrella=unit.umbrella,
                resp_ids=resp_pos | resp_neg | resp_neu,
                resp_pos=resp_pos,
                resp_neg=resp_neg,
                resp_neu=resp_neu,
                origin=unit.origin,
            ))
    return shapes, overig
