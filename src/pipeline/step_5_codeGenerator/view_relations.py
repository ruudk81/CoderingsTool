#%%

"""Toon de groepering uit stap 2 — de poort voor de rest van de herbouw.

Draait op de bestaande step-4-cache: één LLM-call, geen volledige pijplijnrun.

    cd src && python -m pipeline.step_5_codeGenerator.view_relations
    cd src && python -m pipeline.step_5_codeGenerator.view_relations --cache-dir <pad>

Dit is een STOP-taak: na dit overzicht wordt niets uit taak 5 en verder gebouwd
tot de gebruiker de groepering heeft gezien en goedgekeurd.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import CacheConfig
from utils.cacheManager import CacheManager
from utils.llm import token_tracker

from pipeline.step_5_codeGenerator import run_codeGenerator
from pipeline.step_5_codeGenerator.concept_inventory import Concept, build_inventory, t_keep
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.prompts_relations import RelationsResult, tagged
from pipeline.step_5_codeGenerator.relations import resolve_relations
from pipeline.step_5_codeGenerator.taxonomy_input import build_attribute_refs, build_idea_units

CLEAR, SMALL = "✓ eigen code", "· te klein"


class _CacheDirOverride:
    """Wijst run_codeGenerator's laad-helpers tijdelijk naar een andere cache_dir,
    zonder de default in config.py aan te raken."""

    def __init__(self, cache_dir: Path):
        self._config = CacheConfig(cache_dir=cache_dir)
        self._original = None

    def __enter__(self):
        self._original = run_codeGenerator.CacheManager
        run_codeGenerator.CacheManager = lambda: CacheManager(self._config)
        return self

    def __exit__(self, *exc_info):
        run_codeGenerator.CacheManager = self._original


def load_cache(cache_dir: Path):
    """Laad extraction metadata, classified ideas en taxonomy cache via de
    bestaande run_codeGenerator-helpers, gericht op `cache_dir`."""
    with _CacheDirOverride(cache_dir):
        extraction_metadata = run_codeGenerator.load_extraction_metadata()
        classified_ideas = run_codeGenerator.load_classified_ideas()
        taxonomy_cache = run_codeGenerator.load_taxonomy_cache()
    return extraction_metadata, classified_ideas, taxonomy_cache


def tagged_lookup(concepts: List[Concept]) -> Dict[str, Concept]:
    return {tagged(c): c for c in concepts}


def group_by_umbrella(
    concepts: List[Concept], relations: RelationsResult,
) -> Tuple[Dict[str, List[Concept]], Dict[str, str], List[Tuple[Concept, Concept]]]:
    """Groepeer concepten per koepel en verzamel unieke synoniemparen, zoals de
    relatiecall ze teruggaf — geen aantallen zijn hierbij betrokken."""
    lookup = tagged_lookup(concepts)
    umbrellas: Dict[str, List[Concept]] = {}
    umbrella_defs: Dict[str, str] = {}
    synonym_pairs: List[Tuple[Concept, Concept]] = []
    seen_pairs = set()

    for relation in relations.relations:
        concept = lookup.get(relation.attribute)
        if concept is None:
            continue
        umbrellas.setdefault(relation.umbrella_name, []).append(concept)
        umbrella_defs.setdefault(relation.umbrella_name, relation.umbrella_definition)

        if relation.synonym_of and relation.synonym_of != relation.attribute:
            pair_key = tuple(sorted((relation.attribute, relation.synonym_of)))
            if pair_key not in seen_pairs:
                other = lookup.get(relation.synonym_of)
                if other is not None:
                    seen_pairs.add(pair_key)
                    synonym_pairs.append((concept, other))

    return umbrellas, umbrella_defs, synonym_pairs


def format_report(
    concepts: List[Concept],
    relations: RelationsResult,
    t: int,
    n_resp_total: int,
    t_keep_share: float,
) -> str:
    umbrellas, umbrella_defs, synonym_pairs = group_by_umbrella(concepts, relations)
    name_width = max((len(c.name) for c in concepts), default=0) + 2

    def union_size(items: List[Concept]) -> int:
        ids = set()
        for c in items:
            ids |= c.resp_ids
        return len(ids)

    ordered_umbrellas = sorted(
        umbrellas.items(), key=lambda kv: (-union_size(kv[1]), kv[0])
    )

    lines = [f"T_keep = {t} ({t_keep_share:.0%} van {n_resp_total} respondenten)", ""]

    n_clear = 0
    n_pooled_attrs = 0
    n_empty_umbrellas = 0

    for name, items in ordered_umbrellas:
        lines.append(f"KOEPEL: {name}")
        definition = umbrella_defs.get(name, "")
        if definition:
            lines.append(f"  ({definition})")

        items_sorted = sorted(items, key=lambda c: (-c.n_resp, c.name))
        below = [c for c in items_sorted if c.n_resp < t]
        above = [c for c in items_sorted if c.n_resp >= t]
        n_clear += len(above)
        n_pooled_attrs += len(below)

        for c in items_sorted:
            marker = CLEAR if c.n_resp >= t else SMALL
            lines.append(f"  {c.name:<{name_width}}{c.n_resp:>6}  {marker}")

        pooled_n = union_size(below) if below else 0
        if not above and pooled_n < t:
            n_empty_umbrellas += 1
        if len(below) >= 2:
            mark = "✓" if pooled_n >= t else "✗ blijft te klein"
            lines.append(f"  {'':<{name_width}}── gepoold: {pooled_n} {mark}")
        lines.append("")

    lines.append("SYNONIEMEN")
    if not synonym_pairs:
        lines.append("  (geen)")
    else:
        for c1, c2 in sorted(synonym_pairs, key=lambda p: -(p[0].n_resp + p[1].n_resp)):
            line = f"  {c1.name} ({c1.n_resp}) = {c2.name} ({c2.n_resp})"
            if c1.n_resp >= t and c2.n_resp >= t:
                line += "   ⚠ beide boven de drempel"
            lines.append(line)
    lines.append("")

    if relations.scratchpad:
        lines.append("MODEL-TOELICHTING")
        lines.append(f"  {relations.scratchpad}")
        lines.append("")

    lines.append("SAMENVATTING")
    lines.append(f"  Attributen in:                 {len(concepts)}")
    lines.append(f"  Koepels:                        {len(umbrellas)}")
    lines.append(f"  Eigen code (boven T_keep):      {n_clear}")
    lines.append(f"  Gepooled (onder T_keep):        {n_pooled_attrs}")
    lines.append(f"  Koepels zonder code (pool blijft onder T_keep): {n_empty_umbrellas}")
    lines.append(f"  Synoniemparen:                  {len(synonym_pairs)}")

    return "\n".join(lines)


def format_facet_comparison(concepts: List[Concept], relations: RelationsResult) -> str:
    """Vergelijk de koepels met step 4's facetten — de laag die al bestond
    tussen domein en attribuut. Evidence, geen aanbeveling: de gebruiker
    beslist of de koepelvraag iets toevoegt boven de facetten."""
    umbrellas, _, _ = group_by_umbrella(concepts, relations)

    def facet_key(c: Concept) -> Tuple[str, str]:
        return (c.domain, c.facet)

    all_facets = {facet_key(c) for c in concepts}
    lines = [
        "FACET-VERGELIJKING (step-4-facetten vs. koepels van het model)", "",
        f"  Facetten (step 4, met >=1 idee): {len(all_facets)}",
        f"  Koepels (dit model): {len(umbrellas)}", "",
        "  Koepels naar aantal facetten waaruit ze putten:",
    ]
    name_width = max((len(n) for n in umbrellas), default=0) + 2
    for name, items in sorted(
        umbrellas.items(), key=lambda kv: -len({facet_key(c) for c in kv[1]})
    ):
        n_facets_here = len({facet_key(c) for c in items})
        lines.append(f"    {name:<{name_width}}{len(items):>4} leden  {n_facets_here:>3} facetten")
    lines.append("")

    facet_to_umbrellas: Dict[Tuple[str, str], Dict[str, List[Concept]]] = {}
    for name, items in umbrellas.items():
        for c in items:
            facet_to_umbrellas.setdefault(facet_key(c), {}).setdefault(name, []).append(c)

    split = {k: v for k, v in facet_to_umbrellas.items() if len(v) > 1}
    lines.append(f"  Facetten gesplitst over meerdere koepels: {len(split)} van {len(all_facets)}")
    for (domain, facet), by_umbrella in sorted(split.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"    {facet} ({domain}):")
        for umbrella_name, members in by_umbrella.items():
            member_names = ", ".join(c.name for c in members)
            lines.append(f"      -> {umbrella_name}: {member_names}")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override CacheConfig.cache_dir (default: de normale cache-map)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir or CacheConfig().cache_dir
    print(f"Cache-map: {cache_dir}")

    token_tracker.reset()

    extraction_metadata, classified_ideas, taxonomy_cache = load_cache(cache_dir)
    if taxonomy_cache is None:
        print("\nERROR: geen taxonomie-cache gevonden — is de cache-map correct?")
        return
    if not classified_ideas:
        print("\nERROR: geen classified ideas gevonden — is de cache-map correct?")
        return

    language = getattr(extraction_metadata, "lang", "") or "Dutch"

    units = build_idea_units(classified_ideas)
    refs = build_attribute_refs(taxonomy_cache.partition_results)
    concepts = build_inventory(units, refs)

    config = CodebookConfig()
    n_resp_total = len(classified_ideas)
    t = t_keep(n_resp_total, config)

    print(f"Concepten in inventaris: {len(concepts)}")
    print("Relatiecall wordt uitgevoerd (1 LLM-call)...\n")

    relations = asyncio.run(resolve_relations(concepts, config, language, verbose=True))

    print(format_report(concepts, relations, t, n_resp_total, config.t_keep_share))
    print(format_facet_comparison(concepts, relations))

    if token_tracker.call_count > 0:
        print("\n" + token_tracker.get_summary())

    print("\n" + "=" * 70)
    print("STOP: dit is een beoordelingspoort. Bouw niets uit taak 5 en verder")
    print("totdat de gebruiker deze groepering heeft gezien en goedgekeurd.")
    print("=" * 70)


if __name__ == "__main__":
    main()

# %%
