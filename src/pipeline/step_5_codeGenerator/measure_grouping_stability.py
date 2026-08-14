"""Meet hoe reproduceerbaar stadium 2 zijn groepering maakt. Read-only.

Draait `resolve_relations` een aantal keer op dezelfde invoer en rapporteert vier
dingen die nergens in de keten zelf geteld worden:

  volledigheid   beantwoordt het model élk getoond attribuut (het `.get`-gat:
                 een vergeten attribuut krijgt in `_climb` stil zijn domein als
                 koepel, en dat is stroomafwaarts niet te onderscheiden van een
                 echte koepel)
  synoniemen     welke paren het vindt, en of ze over runs heen terugkomen
  vangnetten     of het drains aan elkaar knoopt — `_build_units` voegt
                 synoniemen onvoorwaardelijk en onomkeerbaar samen, dus een
                 fusie tussen twee restbakken is niet meer te repareren
  ARI            hoe sterk de koepelindeling zelf over twee runs verschilt

Basislijn op ASN (99 attributen / 92 concepten, 2026-08-14): 92/92 volledig,
0 synoniemen, 0 drainfusies, 20 vs 28 koepels, ARI 0,648, en 0 van 92
attributen kreeg twee keer dezelfde koepelnaam — bij `temperature=0.0`.

Kosten: één LLM-call per run. Schrijft niets naar de cache.

    python -m pipeline.step_5_codeGenerator.measure_grouping_stability [runs]
"""
from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set

from .concept_inventory import build_inventory
from .config_codeGenerator import CodebookConfig
from .prompts_relations import tagged
from .relations import resolve_relations
from .run_codeGenerator import (
    load_classified_ideas, load_extraction_metadata, load_taxonomy_cache,
)
from .taxonomy_input import build_attribute_refs, build_idea_units


def drain_ids(partition_results) -> Set[str]:
    """Attribuut-ids van de vangnetten, structureel via `is_drain` — nooit op
    naam: die staat in de enquêtetaal en mag door step 4 herschreven worden."""
    ids = set()
    for domain in partition_results.values():
        attributes = domain["attributes"] if isinstance(domain, dict) else domain.attributes
        for attribute_list in (attributes or {}).values():
            for attribute in attribute_list:
                if attribute.get("is_drain") and attribute.get("attribute_id"):
                    ids.add(attribute["attribute_id"])
    return ids


def synonym_pairs(result, concepts) -> Set[frozenset]:
    id_by_tag = {tagged(c): c.attribute_id for c in concepts}
    pairs = set()
    for relation in result.relations:
        left = id_by_tag.get(relation.attribute)
        right = id_by_tag.get(relation.synonym_of) if relation.synonym_of else None
        if left and right and left != right:
            pairs.add(frozenset((left, right)))
    return pairs


def umbrella_map(result, concepts) -> Dict[str, str]:
    id_by_tag = {tagged(c): c.attribute_id for c in concepts}
    return {id_by_tag[r.attribute]: r.umbrella_name
            for r in result.relations if r.attribute in id_by_tag}


def adjusted_rand(a: Dict[str, str], b: Dict[str, str]) -> float:
    """ARI over twee indelingen van dezelfde ids, via paartelling. Handmatig,
    zodat dit geen sklearn-afhankelijkheid in de pijplijn trekt."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return float("nan")
    pairs = list(combinations(ids, 2))
    same_a = {p for p in pairs if a[p[0]] == a[p[1]]}
    same_b = {p for p in pairs if b[p[0]] == b[p[1]]}
    total = len(pairs)
    expected = len(same_a) * len(same_b) / total
    maximum = (len(same_a) + len(same_b)) / 2
    if maximum == expected:
        return 1.0
    return (len(same_a & same_b) - expected) / (maximum - expected)


def _components(pairs: Set[frozenset]) -> List[Set[str]]:
    parent: Dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            node = parent[node]
        return node

    for pair in pairs:
        left, right = tuple(pair)
        parent[find(left)] = find(right)
    groups: Dict[str, Set[str]] = defaultdict(set)
    for node in list(parent):
        groups[find(node)].add(node)
    return [g for g in groups.values() if len(g) > 1]


async def measure(n_runs: int = 2) -> None:
    metadata = load_extraction_metadata()
    classified = load_classified_ideas()
    taxonomy = load_taxonomy_cache()
    if taxonomy is None or not classified:
        print("Geen cache — draai eerst step 4.")
        return

    structure = taxonomy.partition_results
    drains = drain_ids(structure)
    concepts = build_inventory(build_idea_units(classified),
                               build_attribute_refs(structure))
    language = getattr(metadata, "lang", "") or "Dutch"
    name_of = {c.attribute_id: c.name for c in concepts}
    config = CodebookConfig()

    print(f"\n{len(concepts)} concepten, {len(drains)} vangnetten, "
          f"{n_runs} runs bij temperature={config.temperature_relations}\n")

    umbrellas: List[Dict[str, str]] = []
    all_pairs: List[Set[frozenset]] = []
    for run in range(1, n_runs + 1):
        result = await resolve_relations(concepts, config, language, verbose=False)
        pairs = synonym_pairs(result, concepts)
        umbrella = umbrella_map(result, concepts)
        umbrellas.append(umbrella)
        all_pairs.append(pairs)

        answered = {r.attribute for r in result.relations}
        gap = len(concepts) - len(answered)
        drain_pairs = [p for p in pairs if p <= drains]
        print(f"run {run}: {len(answered)}/{len(concepts)} beantwoord"
              f"{f'  <-- {gap} ONBEANTWOORD' if gap else ''}"
              f" | {len(pairs)} synoniemparen"
              f"{f'  <-- {len(drain_pairs)} DRAINFUSIE' if drain_pairs else ''}"
              f" | {len(set(umbrella.values()))} koepels"
              f" | {len(_components(pairs))} synoniemgroepen")
        for pair in sorted(pairs, key=sorted):
            left, right = sorted(pair)
            flag = "  [DRAIN]" if pair & drains else ""
            print(f"        {name_of.get(left, left)}  ==  {name_of.get(right, right)}{flag}")

    if n_runs < 2:
        return
    print("\nover de runs heen")
    stable = set.intersection(*all_pairs) if all_pairs else set()
    print(f"  synoniemen in élke run : {len(stable)} van "
          f"{len(set.union(*all_pairs)) if all_pairs else 0} gevonden")
    for first, second in combinations(range(n_runs), 2):
        identical = sum(1 for k in umbrellas[first]
                        if umbrellas[second].get(k) == umbrellas[first][k])
        print(f"  run {first+1} vs {second+1}: ARI {adjusted_rand(umbrellas[first], umbrellas[second]):.3f}"
              f" | identieke koepelnaam {identical}/{len(umbrellas[first])}")


if __name__ == "__main__":
    asyncio.run(measure(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
