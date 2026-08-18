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
from ..run_codeGenerator import _match_shape, _shape_lookup
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
    vetoes: List[dict]
    concept_by_id: Dict[str, Concept] = field(repr=False)


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

    # `write_codebook` can veto a `pooled` shape (`nameable: false`) — every
    # multi-attribute group v2 builds is `pooled` (`grouping.build_shapes`),
    # so this is the normal route, not an edge case. The attributes don't get
    # lost (the Overig sweep in `run_codebook_v2` still routes them), but a
    # silent veto would make a v1-vs-v2 comparison see a smaller v2 codebook
    # with a bigger Overig and be unable to tell consolidation quality apart
    # from writer vetoes — the same reason degeneration is reported, not
    # absorbed. `veto_log` makes it visible.
    veto_log = _RepairLog()
    codes = await write_codebook(
        shaped.shapes, concepts, dimension_diagnostic, language, config,
        log=veto_log, verbose=verbose, prompt_printer=prompt_printer,
        prompt_builder=build_writer_prompt_v2,
    )

    # `codes[i]` must line up with `shapes[i]` for resolve_duplicate_names and
    # the two finders below (their own docstrings require it) — but a veto in
    # write_codebook means `codes` can be shorter than `shaped.shapes`. Match
    # each written code back to its own shape rather than assuming the two
    # lists still walk in lockstep — the same technique v1's
    # `_generate_codebook_async` uses for the same reason.
    concept_by_id = {c.attribute_id: c for c in concepts}
    shape_lookup = _shape_lookup(shaped.shapes, concept_by_id)
    shapes = [_match_shape(code, shape_lookup) for code in codes]

    collision_log = _RepairLog()
    codes = resolve_duplicate_names(codes, shapes, log=collision_log)
    return GeneratedCodebookV2(
        shapes=shapes, overig_ids=shaped.overig_ids, codes=codes,
        direction_loss=shaped.direction_loss, degeneration=degeneration,
        partition_repairs=repair_log.entries, collisions=collision_log.entries,
        naming_mismatches=find_naming_mismatches(codes, shapes, concept_by_id),
        duplicate_definitions=find_duplicate_definitions(codes, shapes),
        vetoes=veto_log.entries,
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
    """Wat een run zichtbaar moet maken. Degeneratie, richtingsverlies en
    vetoes staan bovenaan: dat zijn de drie dingen die geen enkele bestaande
    check meldt — melden, nooit stil absorberen."""
    if result.degeneration:
        print(f"DEGENERATIE (harde FAIL): {result.degeneration}")

    if result.direction_loss:
        # Groepstelling, geen respondent-uniek totaal: build_shapes telt per
        # groep op, dus een respondent die in twee groepen een minderheidspool
        # mist telt twee keer mee.
        print(f"RICHTINGSVERLIES: {result.direction_loss} verloren pool-plaatsing(en) "
              f"onder de drempel — naar Overig.")

    if result.vetoes:
        print(f"WAARSCHUWING: {len(result.vetoes)} pooled code(s) geveto'd "
              f"(niet noembaar) — leden gaan naar Overig:")
        for v in result.vetoes:
            print(f"  '{v['umbrella']}' — leden: {', '.join(v['members'])}")

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

    # v1's legacy leespad (P9-era over-merge-correctie, van vóór de
    # P7-promotie) vervangt zowel partition_results als classified_ideas door
    # een gecorrigeerde variant wanneer die cache bestaat. v2 port dat pad
    # niet — het zou een uitstervend legacy-mechanisme in een experiment
    # repliceren — maar mag er ook niet stilzwijgend aan voorbijgaan: zonder
    # deze check zou v2 op een andere taxonomie bouwen dan v1 (en daarmee ook
    # een andere drempelbasis), en de vergelijking met v1 zou niet meer
    # opgaan.
    if cache_manager.is_metadata_cache_valid(filename, "taxonomy_corrected", variable_key):
        print("\nERROR: geldige 'taxonomy_corrected'-cache gevonden voor deze dataset. "
              "v2 ondersteunt dat legacy leespad niet — draai v2 hier niet op, de "
              "vergelijking met v1 zou anders op een andere taxonomie gebeuren.")
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
