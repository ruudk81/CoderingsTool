"""Measure how reproducibly stage 2 produces its grouping. Read-only.

Runs `resolve_relations` a number of times on the same input and reports four
things that are counted nowhere in the chain itself:

  completeness   does the model answer for EVERY attribute shown (the `.get`
                 hole: a forgotten attribute silently gets its domain as its
                 umbrella in `_climb`, and downstream that is indistinguishable
                 from a real umbrella)
  synonyms       which pairs it finds, and whether they recur across runs
  catch-alls     whether it ties drains together — `_build_units` merges
                 synonyms unconditionally and irreversibly, so a fusion between
                 two residual buckets cannot be repaired
  ARI            how far the umbrella partition itself differs across two runs

Baseline on ASN (99 attributes / 92 concepts, 2026-08-14): 92/92 complete,
0 synonyms, 0 drain fusions, 20 vs 28 umbrellas, ARI 0.648, and 0 of 92
attributes got the same umbrella name twice — at `temperature=0.0`.

Cost: one LLM call per run. Writes nothing to the cache.

    python -m pipeline.step_5_codeGenerator.measure_grouping_stability [runs]
"""
from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set

from ..concept_inventory import build_inventory
from ..config_codeGenerator import CodebookConfig
from .prompts_relations import tagged
from .relations import resolve_relations
from .run_codeGenerator import (
    load_classified_ideas, load_extraction_metadata, load_taxonomy_cache,
)
from ..taxonomy_input import build_attribute_refs, build_idea_units


def drain_ids(partition_results) -> Set[str]:
    """Attribute ids of the catch-alls, structurally via `is_drain` — never by
    name: that is in the survey language and step 4 may rewrite it."""
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
    """ARI over two partitions of the same ids, via pair counting. Done by hand,
    so this pulls no sklearn dependency into the pipeline."""
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
