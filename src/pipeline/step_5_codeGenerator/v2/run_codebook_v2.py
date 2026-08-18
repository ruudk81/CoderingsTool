"""De v2-keten: twee LLM-calls, met Python ertussen en eromheen.

fase 1  consolidatie   LLM    welke attributen vormen samen één code
fase 2  richting       Python elke groep gesplitst in zuivere valentiepolen
fase 3  bewaking       Python partitie heel, degeneratie gemeld
fase 4  schrijven      LLM    naam, definitie, diagnostiek, indicatoren

Output is dezelfde `List[ConsolidatedCode]` als v1, onder een eigen cachesleutel,
zodat beide codeboeken op dezelfde taxonomie naast elkaar te leggen zijn.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.cacheManager import CacheManager, generate_enhanced_variable_key

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension

from ..codebook_writer import (
    find_duplicate_definitions, find_naming_mismatches, resolve_duplicate_names,
    write_codebook,
)
from ..concept_inventory import Concept, build_inventory, t_keep
from ..config_codeGenerator import CodebookConfig
from ..consolidator import CodeShape
from ..prompts_codeGenerator import ConsolidatedCode
from ..taxonomy_input import IdeaUnit, build_attribute_refs, build_idea_units
from .attribute_cards import build_cards
from .consolidation import resolve_consolidation
from .grouping import build_shapes, check_degeneration, repair_partition
from .prompts_writer_v2 import build_writer_prompt_v2

CACHE_STEP = "mece_codes_v2"


class _RepairLog:
    """Duck-typed log, zoals `_RoundLog` en `_CollisionLog` in v1."""
    def __init__(self):
        self.entries: List[dict] = []

    def add(self, **kwargs):
        self.entries.append(kwargs)


@dataclass
class GeneratedCodebookV2:
    shapes: List[CodeShape]
    overig_ids: List[str]
    codes: List[ConsolidatedCode]
    direction_loss: int
    degeneration: Optional[str]
    partition_repairs: List[dict]
    collisions: List[dict]
    naming_mismatches: List[dict]
    duplicate_definitions: List[dict]
    concept_by_id: Dict[str, Concept] = field(repr=False)


def _shape_lookup(
    shapes: List[CodeShape], concept_by_id: Dict[str, Concept],
) -> Dict[tuple, CodeShape]:
    """Key shapes by (their source-attribute names, valence) — the same two
    things `write_codebook` echoes back on each `ConsolidatedCode` (see
    `_to_consolidated_code` in `codebook_writer.py`). `write_codebook` can
    veto a `pooled` shape and simply omit it from what it returns, so `codes`
    can come back shorter than the `shapes` list that went in; this lookup is
    how `_generate_async` recovers, for each code actually written, which
    shape it belongs to — the same technique v1's `_generate_codebook_async`
    uses for the same reason."""
    lookup = {}
    for shape in shapes:
        names = frozenset(concept_by_id[m].name for m in shape.members if m in concept_by_id)
        lookup[(names, shape.valence)] = shape
    return lookup


async def _generate_async(
    concepts, idea_units_by_attribute, threshold, survey_question, n_respondents,
    dimension_diagnostic, language, config, verbose, prompt_printer,
) -> GeneratedCodebookV2:
    cards = build_cards(concepts, idea_units_by_attribute)
    proposal = await resolve_consolidation(
        cards, survey_question, n_respondents, language, config,
        verbose=verbose, prompt_printer=prompt_printer,
    )

    repair_log = _RepairLog()
    groups = repair_partition(proposal, cards, concepts, log=repair_log)
    degeneration = check_degeneration(len(groups), len(cards))
    shaped = build_shapes(groups, concepts, threshold)

    codes = await write_codebook(
        shaped.shapes, concepts, dimension_diagnostic, language, config,
        verbose=verbose, prompt_printer=prompt_printer,
        prompt_builder=build_writer_prompt_v2,
    )

    # `codes[i]` must line up with `shapes[i]` for resolve_duplicate_names and
    # the two finders below (their own docstrings require it) — but a veto in
    # write_codebook means `codes` can be shorter than `shaped.shapes`. Match
    # each written code back to its own shape rather than assuming the two
    # lists still walk in lockstep.
    concept_by_id = {c.attribute_id: c for c in concepts}
    shape_lookup = _shape_lookup(shaped.shapes, concept_by_id)
    shapes = [shape_lookup[(frozenset(code.source_attributes), code.valence)]
             for code in codes]

    collision_log = _RepairLog()
    codes = resolve_duplicate_names(codes, shapes, log=collision_log)
    return GeneratedCodebookV2(
        shapes=shapes, overig_ids=shaped.overig_ids, codes=codes,
        direction_loss=shaped.direction_loss, degeneration=degeneration,
        partition_repairs=repair_log.entries, collisions=collision_log.entries,
        naming_mismatches=find_naming_mismatches(codes, shapes, concept_by_id),
        duplicate_definitions=find_duplicate_definitions(codes, shapes),
        concept_by_id=concept_by_id,
    )


def generate_codebook_v2(
    concepts, idea_units_by_attribute, threshold, survey_question, n_respondents,
    dimension_diagnostic, language, config, verbose=True, prompt_printer=None,
) -> GeneratedCodebookV2:
    """Sync wrapper — één orchestratie-ingang, zoals `generate_codebook` in v1."""
    return asyncio.run(_generate_async(
        concepts, idea_units_by_attribute, threshold, survey_question, n_respondents,
        dimension_diagnostic, language, config, verbose, prompt_printer,
    ))


def report_codebook_build_v2(result: GeneratedCodebookV2) -> None:
    """Wat een run zichtbaar moet maken. Degeneratie en richtingsverlies staan
    bovenaan: dat zijn de twee dingen die geen enkele bestaande check meldt."""
    if result.degeneration:
        print(f"DEGENERATIE (harde FAIL): {result.degeneration}")

    if result.direction_loss:
        print(f"RICHTINGSVERLIES: {result.direction_loss} respondent(en) in een "
              f"minderheidspool die de drempel niet haalde — naar Overig.")

    for entry in result.partition_repairs:
        if entry["action"] == "PARTITION_MISSING":
            print(f"  PARTITIE: '{entry['name']}' ({entry['attribute_id']}) was "
                  f"vergeten — eigen groep gemaakt")
        else:
            print(f"  PARTITIE: {entry['attribute_id']} stond in meerdere groepen — "
                  f"gehouden in '{entry['kept_in']}', verwijderd uit "
                  f"{', '.join(entry['removed_from'])}")

    if result.collisions:
        print(f"WAARSCHUWING: {len(result.collisions)} dubbele codenaam/namen opgelost:")
        for c in result.collisions:
            print(f"  '{c['name']}' behouden; kleinere hernoemd naar '{c['renamed_to']}'")

    if result.naming_mismatches:
        print(f"WAARSCHUWING: {len(result.naming_mismatches)} code(s) waarvan de naam "
              f"geen woord deelt met een van zijn bronattributen:")
        for m in result.naming_mismatches:
            print(f"  '{m['code_name']}' ({m['n_resp']} resp.) — leden: "
                  f"{', '.join(m['members'])}")

    if result.duplicate_definitions:
        print(f"WAARSCHUWING: {len(result.duplicate_definitions)} groep(en) codes "
              f"met identieke definitie")


def run_codebook_v2(filename: str = None, var_name: str = None,
                    sample_size: Optional[int] = None,
                    force_recalc: bool = False) -> None:
    """Productie-ingang van v2. Leest dezelfde cache als v1 en schrijft onder
    CACHE_STEP, zodat beide codeboeken naast elkaar blijven bestaan."""
    from ..run_codeGenerator import (
        FALLBACK_DIAGNOSTIC, FILENAME, SAMPLE_SIZE, VARIABLE, apply_overig_sweep,
        cache_mece_results, load_classified_ideas, load_extraction_metadata,
        load_taxonomy_cache, print_codebook_results, run_scorecard,
    )

    filename = FILENAME if filename is None else filename
    var_name = VARIABLE if var_name is None else var_name
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size

    print("=" * 70)
    print("CODE GENERATOR v2 (loading taxonomy from cache)")
    print("=" * 70)

    variable_key = generate_enhanced_variable_key(
        selected_variables=[var_name], is_merged=False, sample_size=sample_size)

    cache_manager = CacheManager()
    if not force_recalc and cache_manager.is_metadata_cache_valid(
            filename, CACHE_STEP, variable_key):
        print("v2-codeboek cache geldig — overgeslagen (force_recalc=True om te herdraaien).\n")
        return

    metadata = load_extraction_metadata(filename, var_name, sample_size, variable_key)
    classified = load_classified_ideas(filename, var_name, sample_size, variable_key)
    taxonomy = load_taxonomy_cache(filename, var_name, sample_size, variable_key)
    if taxonomy is None:
        print("\nERROR: geen taxonomie in cache. Draai eerst step 4.")
        return

    refs = build_attribute_refs(taxonomy.partition_results)
    units = [u for u in build_idea_units(classified) if u.attribute_id in refs]
    concepts = build_inventory(units, refs)

    by_attribute: Dict[str, List[IdeaUnit]] = {}
    for unit in units:
        by_attribute.setdefault(unit.attribute_id, []).append(unit)

    config = CodebookConfig()
    # Dezelfde drempelbasis als v1 (`run_codeGenerator.py`): het totale aantal
    # responses, niet het aantal respondenten mét een idee. Wijkt v2 hiervan af,
    # dan vergelijkt compare_v1_v2 twee codeboeken op verschillende drempels.
    n_resp_total = len(classified)
    threshold = t_keep(n_resp_total, config)

    # Exact de afleiding uit v1 — `lang`, niet `language`, en het diagnostisch
    # stramien komt uit de dimensietabel, niet uit de metadata.
    language = getattr(metadata, "lang", "") or "Dutch"
    survey_question = (getattr(metadata, "var_lab", "") or "").strip()
    dimension_name = getattr(metadata, "primary_dimension", "") or ""
    dimension_diagnostic = (
        get_dimension(dimension_name).criterion if dimension_name else FALLBACK_DIAGNOSTIC
    )
    n_respondents = len({u.respondent_id for u in units})

    print(f"  {len(concepts)} attributen, {n_respondents} met een idee, "
          f"T_keep = {threshold} over {n_resp_total} responses")

    result = generate_codebook_v2(
        concepts, by_attribute, threshold, survey_question, n_respondents,
        dimension_diagnostic, language, config, verbose=config.verbose,
    )
    report_codebook_build_v2(result)

    overig_name = apply_overig_sweep(result.codes, taxonomy.partition_results, language)
    print_codebook_results(result.codes)
    run_scorecard(result.codes, taxonomy.partition_results, overig_name)

    cache_mece_results(
        taxonomy.partition_set, taxonomy.partition_results, result.codes,
        filename=filename, variable=var_name, sample_size=sample_size,
        variable_key=variable_key, step=CACHE_STEP,
    )
    print(f"v2-codeboek gecached ({len(result.codes)} codes) onder '{CACHE_STEP}'")


if __name__ == "__main__":
    run_codebook_v2(force_recalc=True)
