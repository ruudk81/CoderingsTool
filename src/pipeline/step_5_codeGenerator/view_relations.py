#%%

"""Show the grouping from step 2 — the gate for the rest of the rebuild.

Runs on the existing step-4 cache: two LLM calls (relations + umbrella-name
consolidation), not a full pipeline run.

    cd src && python -m pipeline.step_5_codeGenerator.view_relations
    cd src && python -m pipeline.step_5_codeGenerator.view_relations --cache-dir <path>

This is a STOP task: after this overview nothing from task 5 onwards is built
until the user has seen and approved the grouping.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import CacheConfig
from utils.cacheManager import CacheManager
from utils.llm import token_tracker

from pipeline.step_5_codeGenerator import run_codeGenerator
from pipeline.step_5_codeGenerator.concept_inventory import Concept, build_inventory, t_keep
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.prompts_relations import RelationsResult, tagged
from pipeline.step_5_codeGenerator.prompts_umbrella_merge import Umbrella, umbrellas_from_relations
from pipeline.step_5_codeGenerator.relations import apply_umbrella_merge, resolve_relations, resolve_umbrella_merge
from pipeline.step_5_codeGenerator.taxonomy_input import build_attribute_refs, build_idea_units

CLEAR, SMALL = "✓ eigen code", "· te klein"

# The ordinary cache. If you work on step 5 while step 4 is still moving, point
# --cache-dir at a frozen copy: three gate runs in one session each saw a
# different taxonomy (182 -> 141 -> 189 attributes), and then you are measuring
# two changes at once.
DEFAULT_CACHE_DIR = project_root / "data" / "cache"


class _CacheDirOverride:
    """Points run_codeGenerator's loading helpers at a different cache_dir for
    the duration, without touching the default in config.py."""

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
    """Group concepts per umbrella and collect unique synonym pairs, as the
    relations call returned them — no counts are involved here."""
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


def format_consolidation(
    umbrellas_before: List[Umbrella], relations_before: RelationsResult,
    relations_after: Optional[RelationsResult],
) -> str:
    """Show what step 2b (consolidating umbrella names) did — or, on a failed
    call, that it continued unconsolidated. `relations_after` is the result of
    `apply_umbrella_merge`, or None when the call failed."""
    lines = ["VERZAMELNAMEN OPGESCHOOND"]
    if relations_after is None:
        lines.append(
            "  Consolidatie is mislukt (2e LLM-call zonder resultaat) — "
            "doorgegaan met de ongeconsolideerde koepelnamen hieronder."
        )
        lines.append("")
        return "\n".join(lines)

    before_by_attribute = {r.attribute: r.umbrella_name for r in relations_before.relations}
    after_by_attribute = {r.attribute: r.umbrella_name for r in relations_after.relations}

    n_before = len(umbrellas_before)
    n_after = len({r.umbrella_name for r in relations_after.relations})
    lines.append(f"  {n_before} namen  →  {n_after} namen")

    renames = sorted({
        (before_by_attribute[attribute], after_by_attribute[attribute])
        for attribute in before_by_attribute
        if before_by_attribute[attribute] != after_by_attribute[attribute]
    })
    if renames:
        lines.append("  Samengevoegd:")
        width = max(len(old) for old, _ in renames) + 2
        for old, new in renames:
            lines.append(f"    {old:<{width}}→  {new}")
    lines.append("")
    return "\n".join(lines)


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
    """Compare the umbrellas with step 4's facets — the layer that already
    existed between domain and attribute. Evidence, not a recommendation: the
    user decides whether the umbrella question adds anything over the facets."""
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
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Override CacheConfig.cache_dir (default: data/cache)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
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
    print("Relatiecall + consolidatiecall worden uitgevoerd (2 LLM-calls)...\n")

    async def _run():
        relations_result = await resolve_relations(concepts, config, language, verbose=True)
        umbrellas_before = umbrellas_from_relations(relations_result)
        merge_result = await resolve_umbrella_merge(umbrellas_before, config, verbose=True)
        return relations_result, umbrellas_before, merge_result

    relations_before, umbrellas_before, merge_result = asyncio.run(_run())
    relations_after = (
        apply_umbrella_merge(relations_before, merge_result) if merge_result is not None else None
    )
    final_relations = relations_after if relations_after is not None else relations_before

    print(format_consolidation(umbrellas_before, relations_before, relations_after))
    print(format_report(concepts, final_relations, t, n_resp_total, config.t_keep_share))
    print(format_facet_comparison(concepts, final_relations))

    if token_tracker.call_count > 0:
        print("\n" + token_tracker.get_summary())

    print("\n" + "=" * 70)
    print("STOP: dit is een beoordelingspoort. Bouw niets uit taak 5 en verder")
    print("totdat de gebruiker deze groepering heeft gezien en goedgekeurd.")
    print("=" * 70)


if __name__ == "__main__":
    main()

# %%
