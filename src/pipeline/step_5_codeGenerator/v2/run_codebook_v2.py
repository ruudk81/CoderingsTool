"""De v2-keten: twee LLM-calls, met Python ertussen en eromheen.

fase 1  consolidatie   LLM    welke attributen vormen samen één code
fase 2  richting       Python elke groep gesplitst in zuivere valentiepolen
fase 3  bewaking       Python partitie heel, degeneratie gemeld
fase 4  schrijven      LLM    naam, definitie, diagnostiek, indicatoren

Output is een `List[ConsolidatedCode]` onder de productiesleutel `mece_codes`,
waar step 6 en step 7 hem lezen.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.costTracker import CostTracker
from utils.llm import token_tracker
from utils.promptPrinter import PromptPrinter
from utils.saveVerbose import VerboseCapture

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension

from ..codebook_writer import (
    find_duplicate_definitions, find_naming_mismatches, resolve_duplicate_names,
    write_codebook,
)
from ..concept_inventory import Concept, build_inventory, t_keep
from ..config_codeGenerator import CodebookConfig
from ..code_shape import CodeShape, _match_shape, _shape_lookup
from ..codebook_io import (
    FALLBACK_DIAGNOSTIC, FILENAME, SAMPLE_SIZE, VARIABLE, apply_overig_sweep,
    cache_mece_results, load_classified_ideas, load_extraction_metadata,
    load_taxonomy_cache, print_codebook_results, run_scorecard,
    save_prompts_to_json,
)
from models import ConsolidatedCode
from ..taxonomy_input import IdeaUnit, build_attribute_refs, build_idea_units
from .attribute_cards import build_cards
from .consolidation import resolve_consolidation
from .grouping import Group, build_shapes, check_degeneration, repair_partition
from .postmortem import (
    apply_splits, format_postmortem, resolve_postmortem, select_candidates,
)
from .stability import (
    StabilityReport, format_stability, run_consolidation_repeatedly,
)
from .prompts_writer_v2 import build_writer_prompt_v2

CACHE_STEP = "mece_codes"

PRINT_PROMPTS = False  # True zet de prompts realtime op de console


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
    # Alleen gevuld wanneer `stability_runs` >= 2; zonder die meting draait de
    # post-mortem niet en is "niets gemeten, niets gesplitst" de juiste toestand.
    stability: Optional[StabilityReport] = None
    postmortem_candidates: int = 0
    postmortem_log: List[dict] = field(default_factory=list)


async def _generate_async(
    concepts: List[Concept],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    threshold: int,
    survey_question: str,
    n_respondents: int,
    dimension_diagnostic: str,
    language: str,
    config: CodebookConfig,
    verbose: bool,
    prompt_printer,
    stability_runs: int = 0,
) -> GeneratedCodebookV2:
    cards = build_cards(concepts, idea_units_by_attribute)
    repair_log = _RepairLog()

    # Met `stability_runs` draait fase 1 meerdere keren. De EERSTE run wordt het
    # codeboek — er wordt niets gemiddeld of samengevoegd, want dat zou een
    # indeling opleveren die geen enkele run heeft voorgesteld. De overige runs
    # dienen alleen om te zien welke groeperingen wisselden, en die lijst stuurt
    # de post-mortem naar de plekken waar het model zelf geen vast oordeel had.
    report: Optional[StabilityReport] = None
    if stability_runs >= 2:
        report, runs_groups = await run_consolidation_repeatedly(
            cards, concepts, survey_question, n_respondents, language, config,
            runs=stability_runs, verbose=verbose, first_run_log=repair_log,
        )
        groups = runs_groups[0]
    else:
        proposal = await resolve_consolidation(
            cards, survey_question, n_respondents, language, config,
            verbose=verbose, prompt_printer=prompt_printer,
        )
        groups = repair_partition(proposal, cards, concepts, log=repair_log)

    # Post-mortem: alleen groepen die te breed uitvielen of tussen runs wisselden.
    # Een afgewezen of mislukt oordeel laat het codeboek staan zoals het was —
    # dit is bijstelling, geen fundament.
    candidates: List[Group] = []
    postmortem_log: List[dict] = []
    if report is not None:
        candidates = select_candidates(groups, concepts, report, n_respondents)
        if candidates:
            verdicts = await resolve_postmortem(
                candidates, cards, survey_question, n_respondents, language,
                config, verbose=verbose,
            )
            groups, postmortem_log = apply_splits(groups, verdicts)

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
        stability=report, postmortem_candidates=len(candidates),
        postmortem_log=postmortem_log,
        partition_repairs=repair_log.entries, collisions=collision_log.entries,
        naming_mismatches=find_naming_mismatches(codes, shapes, concept_by_id),
        duplicate_definitions=find_duplicate_definitions(codes, shapes),
        vetoes=veto_log.entries,
        concept_by_id=concept_by_id,
    )


def generate_codebook_v2(
    concepts: List[Concept],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    threshold: int,
    survey_question: str,
    n_respondents: int,
    dimension_diagnostic: str,
    language: str,
    config: CodebookConfig,
    verbose: bool = True,
    prompt_printer=None,
    stability_runs: int = 0,
) -> GeneratedCodebookV2:
    """Sync wrapper — één orchestratie-ingang, zoals `generate_codebook` in v1.

    `stability_runs` >= 2 herhaalt fase 1 en zet de post-mortem aan; 0 laat de
    keten precies zoals hij was."""
    return asyncio.run(_generate_async(
        concepts, idea_units_by_attribute, threshold, survey_question, n_respondents,
        dimension_diagnostic, language, config, verbose, prompt_printer,
        stability_runs,
    ))


def report_codebook_build_v2(result: GeneratedCodebookV2) -> None:
    """Wat een run zichtbaar moet maken. Degeneratie, richtingsverlies en
    vetoes staan bovenaan: dat zijn de drie dingen die geen enkele bestaande
    check meldt — melden, nooit stil absorberen."""
    if result.stability is not None:
        names = {i: c.name for i, c in result.concept_by_id.items()}
        print(format_stability(result.stability, names))
    if result.postmortem_candidates or result.postmortem_log:
        print(format_postmortem(result.postmortem_candidates, result.postmortem_log))

    if result.degeneration:
        print(f"DEGENERATIE (harde FAIL): {result.degeneration}")

    if result.direction_loss:
        # Groepstelling, geen respondent-uniek totaal: build_shapes telt per
        # groep op, dus een respondent die in twee groepen een minderheidspool
        # mist telt twee keer mee.
        #
        # Niet "naar Overig": dat klopt alleen wanneer GEEN enkele pool van de
        # groep de drempel haalt (grouping.py:145-148). Haalt de andere pool
        # wél de drempel (:153-154), dan blijft het bronattribuut een source
        # van die overblijvende code — apply_overig_sweep ziet het dus niet als
        # wees — en komen deze respondenten zonder eigen code terecht bij de
        # overblijvende, tegengesteld gerichte code.
        print(f"RICHTINGSVERLIES: {result.direction_loss} verloren pool-plaatsing(en) "
              f"onder de drempel — geen eigen code. Haalt de andere pool van "
              f"dezelfde groep wél de drempel, dan belanden deze respondenten bij "
              f"die overblijvende (tegengesteld gerichte) code; haalt geen enkele "
              f"pool de drempel, dan gaat de hele groep naar Overig.")

    if result.vetoes:
        print(f"WAARSCHUWING: {len(result.vetoes)} pooled code(s) geveto'd "
              f"(niet noembaar) — leden gaan naar Overig:")
        for v in result.vetoes:
            print(f"  '{v['umbrella']}' — leden: {', '.join(v['members'])}")

    for entry in result.partition_repairs:
        if entry["action"] == "PARTITION_MISSING":
            print(f"  PARTITIE: '{entry['name']}' ({entry['attribute_id']}) was "
                  f"vergeten — eigen groep gemaakt")
        elif entry["action"] == "PARTITION_DOUBLE":
            print(f"  PARTITIE: {entry['attribute_id']} stond in meerdere groepen — "
                  f"gehouden in '{entry['kept_in']}', verwijderd uit "
                  f"{', '.join(entry['removed_from'])}")
        else:  # PARTITION_DUPLICATE_IN_GROUP
            print(f"  PARTITIE: {entry['attribute_id']} stond dubbel in dezelfde "
                  f"groep '{entry['group']}' — eenmaal geteld")

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
                    force_recalc: bool = False,
                    stability_runs: int = 0) -> None:
    """Productie-ingang van step 5. Leest de taxonomie uit de step-4-cache en
    schrijft het codeboek onder CACHE_STEP, waar step 6 het opent.

    `stability_runs` >= 2 herhaalt fase 1, meet welke groeperingen wisselen en
    zet daarmee de post-mortem aan. De eerste run wordt het codeboek; de rest
    dient alleen om te zien waar het model geen vast oordeel had."""
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

    cost_tracker = CostTracker(filename=filename, var_name=var_name,
                               sample_size=sample_size)
    snapshot_before = token_tracker.snapshot()
    prompt_printer = PromptPrinter(enabled=True, print_realtime=PRINT_PROMPTS)

    result = generate_codebook_v2(
        concepts, by_attribute, threshold, survey_question, n_respondents,
        dimension_diagnostic, language, config, verbose=config.verbose,
        stability_runs=stability_runs, prompt_printer=prompt_printer,
    )
    report_codebook_build_v2(result)

    # Eén fase voor de hele keten: consolidatie en schrijven staan op dezelfde
    # sport van STEP_MODEL, dus fijner uitsplitsen levert geen ander getal op.
    cost_tracker.record_phase(
        "step_5_code_generator", "codebook_generation",
        snapshot_before, token_tracker.snapshot(), model=config.model_writer,
    )
    cost_tracker.finalize_step("step_5_code_generator")

    save_prompts_to_json(prompt_printer)

    overig_name = apply_overig_sweep(result.codes, taxonomy.partition_results, language)
    print_codebook_results(result.codes)
    scorecard = run_scorecard(result.codes, taxonomy.partition_results, overig_name)

    if result.direction_loss:
        # De maat die RICHTINGSVERLIES's effect op déze run zichtbaar maakt:
        # een homeless tegenpool zonder counter-valence code is precies wat
        # under_split_codes telt.
        print(f"  (RICHTINGSVERLIES-effect in de scorecard: "
              f"{len(scorecard.under_split_codes)} under-split code(s))")

    # Degeneratie is een harde FAIL (spec): melden, niet repareren — de codebook-
    # print en scorecard hierboven blijven dus draaien, alleen de cache-write
    # niet. Zonder deze afslag zou een ontaard voorstel onder CACHE_STEP landen
    # en step 6 het stilzwijgend inlezen, terwijl de DEGENERATIE-regel hierboven
    # al meldde dat het niet deugt.
    if result.degeneration:
        print(f"v2-codeboek NIET gecached — degeneratie: {result.degeneration}")
        return

    print(f"v2-codeboek cachen onder '{CACHE_STEP}' ({len(result.codes)} codes)...")
    cache_mece_results(
        taxonomy.partition_set, taxonomy.partition_results, result.codes,
        filename=filename, variable=var_name, sample_size=sample_size,
        variable_key=variable_key, step=CACHE_STEP,
    )


if __name__ == "__main__":
    # stability_runs blijft 0: de post-mortem-splitser staat uit tot zijn
    # vraagvorm herzien is — zie dev/WORK.md, sectie "v2 post-mortem".
    with VerboseCapture(filename=FILENAME, var_name=VARIABLE,
                        sample_size=SAMPLE_SIZE, step=5):
        token_tracker.reset()
        run_codebook_v2(force_recalc=True)
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())
