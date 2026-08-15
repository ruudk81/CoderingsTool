"""Step 3 — merging and settling direction. Counting and comparing, no LLM.

Prevalence decides WHETHER something merges; semantics only decides WHERE TO. A
concept does not merge because it resembles something else — it merges because it
is too small to stand on its own.

Order: deduplicate (synonyms, regardless of size) -> climb (per umbrella, and one
level higher to the taxonomy domain when needed) -> direction (per resulting
shape, based on the merged respondent sets).
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
    """One outcome of consolidation: how many codes, which members, which
    direction. `key` is a run-local ordering key, not a codebook id."""
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
    """One or more attributes that climb together as a whole: a single concept, a
    deduplicated synonym group, or a pooled group."""
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
    """Translates `RelationsResult`'s qualified names (`tagged()` format) back
    into attribute ids. A `synonym_of` not present in the list is ignored; chains
    (A->B->C) are not untangled here, `consolidate` does that."""
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
    """Connected components over `synonym_of`: a chain A->B->C becomes one group
    regardless of the order in which the pairs were given."""
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
    """Deduplicates synonym groups into units carrying the union of their
    respondent sets. The umbrella comes from the most prevalent name in the group
    — in a synonym pair both members usually point at the same umbrella, but on a
    disagreement the most prevalent member wins."""
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
    """Members above the threshold stand on their own and do not climb. The rest
    is pooled per umbrella; if that pool does not clear the threshold, one level
    higher is tried — the taxonomy domain, the only broader grouping a Concept
    still carries. If that does not clear it either, there is no level left:
    residual."""
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
    """One shape becomes two (or three) as soon as both poles clear the threshold
    separately; a single pole over the threshold gives one directed shape carrying
    the WHOLE group (there is no second shape to catch the rest); if no pole
    clears it, one neutral shape covers the whole group."""
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
    """Deduplicate -> climb -> direction. `log` is reserved for the decision log
    (a later step); this function does not fill it yet."""
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
