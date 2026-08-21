"""De keten van de consensuskandidaat.

fase 1  consolidatie   LLM     N runs parallel over dezelfde inventaris
fase 2  consensus      Python  co-associatiematrix, volledige koppeling, tau
fase 3  richting       Python  elke groep gesplitst in valentiepolen
fase 4  bewaking       Python  partitie heel, degeneratie gemeld
fase 5  schrijven      LLM     naam, definitie, diagnostiek, indicatoren

Output gaat onder `mece_codes` — dezelfde sleutel als de productieketen, zodat
step 6 en 7 op deze codes kunnen draaien. Gevolg: de cache houdt één codeboek
tegelijk en de laatst gedraaide keten wint.
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

from ..attribute_cards import build_cards
from ..code_shape import CodeShape, _match_shape, _shape_lookup
from ..codebook_io import (
    FALLBACK_DIAGNOSTIC, FILENAME, SAMPLE_SIZE, VARIABLE, apply_overig_sweep,
    cache_mece_results, load_classified_ideas, load_extraction_metadata,
    load_taxonomy_cache, print_codebook_results, run_scorecard,
    save_prompts_to_json,
)
from ..codebook_writer import (
    find_duplicate_definitions, find_naming_mismatches, resolve_duplicate_names,
    write_codebook,
)
from ..concept_inventory import Concept, build_inventory, t_keep
from ..grouping import (
    Group, build_shapes, check_degeneration, pool_thin_within_facet,
    repair_partition,
)
from ..taxonomy_input import IdeaUnit, build_attribute_refs, build_idea_units
from models import ConsolidatedCode

from .config_consensus import ConsensusConfig
from .consensus import consensus_partition, dominant_member, together_from_runs
from .consolidation import resolve_consolidations

CACHE_STEP = "mece_codes"
# Eigen stapnaam in het kostenregister, zodat je kunt zien wat een route kost.
# De cachesleutel is gedeeld, de rekening niet.
COST_STEP = "step_5_consensus"

PRINT_PROMPTS = False  # True zet de prompts realtime op de console


class _RepairLog:
    """Duck-typed log, zoals `_RoundLog` en `_CollisionLog` in v1."""
    def __init__(self):
        self.entries: List[dict] = []

    def add(self, **kwargs):
        self.entries.append(kwargs)


@dataclass
class GeneratedCodebook:
    shapes: List[CodeShape]
    overig_ids: List[str]
    codes: List[ConsolidatedCode]
    direction_loss: int
    degeneration: Optional[str]
    # Aantal, niet de lijst: bij N=30 runs is elke reparatie een normaal
    # onderdeel van één los voorstel, en de reparatielijst van 30 runs samen
    # zou het signaal verdrinken. Zie `report_codebook_build`.
    partition_repairs: int
    collisions: List[dict]
    naming_mismatches: List[dict]
    duplicate_definitions: List[dict]
    vetoes: List[dict]
    concept_by_id: Dict[str, Concept] = field(repr=False)
    runs_used: int = 0
    runs_failed: int = 0
    pool_log: List[dict] = field(default_factory=list)


def groups_from_clusters(clusters, concepts: List[Concept]) -> List[Group]:
    """Consensusclusters worden `Group`s voor de bestaande keten.

    `proposed_name` vult `CodeShape.umbrella`, dat op twee plaatsen gebruikt
    wordt: als noodnaam wanneer de schrijfcall een vorm overslaat, en als
    hernoemkandidaat wanneer twee codes dezelfde naam krijgen. Een
    consensusgroep is door geen enkele call voorgesteld en heeft er geen, dus
    hier staat het zwaarste lid — deterministisch, met gelijkspel naar het
    lexicografisch kleinste id.
    """
    concept_by_id = {c.attribute_id: c for c in concepts}
    weight_by_id = {c.attribute_id: c.n_resp for c in concepts}
    groups = []
    for cluster in clusters:
        bekend = [m for m in cluster if m in concept_by_id]
        umbrella = (concept_by_id[dominant_member(bekend, weight_by_id)].name
                    if bekend else "")
        groups.append(Group(member_ids=tuple(cluster), proposed_name=umbrella,
                            explanation=""))
    return groups


async def generate_codebook(
    concepts: List[Concept],
    idea_units_by_attribute: Dict[str, List[IdeaUnit]],
    threshold: int,
    survey_question: str,
    n_respondents: int,
    dimension_diagnostic: str,
    language: str,
    config: ConsensusConfig,
    verbose: bool = True,
    prompt_printer=None,
) -> GeneratedCodebook:
    cards = build_cards(concepts, idea_units_by_attribute,
                        exclude_drains=config.exclude_drains)
    salts = [f"run{i}" for i in range(config.runs)]
    proposals, runs_failed = await resolve_consolidations(
        cards, survey_question, n_respondents, language, config, salts,
        verbose=verbose, prompt_printer=prompt_printer,
    )
    # Eén log over alle N runs heen, niet één per run: een repareerbeurt per
    # run is hier de normale gang van zaken (elke run is een los voorstel),
    # dus wat aggregeert is bruikbaar diagnostiek — een model dat structureel
    # attributen vergeet — en een lijst van 30 runs' individuele reparaties
    # zou dat juist verdrinken.
    repair_log = _RepairLog()
    partities = [
        [tuple(sorted(group.member_ids))
         for group in repair_partition(proposal, cards, concepts, log=repair_log)]
        for proposal in proposals
    ]

    ids = [card.attribute_id for card in cards]
    together = together_from_runs(partities, ids)
    clusters = consensus_partition(together, ids, len(partities), config.tau)
    groups = groups_from_clusters(clusters, concepts)

    # Degeneratie wordt op de CONSENSUSINDELING gemeten, vóór de facetpool —
    # dat is de indeling die deze keten voorstelt. De pool erna kan het aantal
    # groepen alleen verlagen, en dat is geen ander voorstel maar een
    # afronding ervan.
    degeneration = check_degeneration(len(groups), len(ids))
    groups, pool_log = pool_thin_within_facet(groups, concepts, threshold,
                                              two_pole=config.two_pole)
    shaped = build_shapes(groups, concepts, threshold, two_pole=config.two_pole)

    # `write_codebook` can veto a `pooled` shape (`nameable: false`) — every
    # multi-attribute group this chain builds is `pooled` (`grouping.build_shapes`),
    # so this is the normal route, not an edge case. The attributes don't get
    # lost (the Overig sweep in `run_codebook` still routes them), but a
    # silent veto would make a comparison against the production chain see
    # a smaller codebook with a bigger Overig and be unable to tell
    # consolidation quality apart from writer vetoes — the same reason
    # degeneration is reported, not absorbed. `veto_log` makes it visible.
    veto_log = _RepairLog()
    codes = await write_codebook(
        shaped.shapes, concepts, dimension_diagnostic, language, config,
        log=veto_log, verbose=verbose, prompt_printer=prompt_printer,
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
    return GeneratedCodebook(
        shapes=shapes, overig_ids=shaped.overig_ids, codes=codes,
        direction_loss=shaped.direction_loss, degeneration=degeneration,
        partition_repairs=len(repair_log.entries), collisions=collision_log.entries,
        naming_mismatches=find_naming_mismatches(codes, shapes, concept_by_id),
        duplicate_definitions=find_duplicate_definitions(codes, shapes),
        vetoes=veto_log.entries,
        concept_by_id=concept_by_id,
        runs_used=len(partities), runs_failed=runs_failed, pool_log=pool_log,
    )


def report_codebook_build(result: GeneratedCodebook, config: ConsensusConfig) -> None:
    """Wat een run zichtbaar moet maken. Eerst wat geen enkele bestaande check
    meldt — hoeveel runs meetelden en wat de facetpool samenvoegde — daarna
    dezelfde diagnostiek als productie: degeneratie, richtingsverlies, vetoes,
    partitiereparaties, botsingen en naam-/definitieafwijkingen."""
    print(f"CONSENSUS: {result.runs_used} runs gebruikt, tau={config.tau}")
    if result.runs_failed:
        print(f"  LET OP: {result.runs_failed} run(s) kwamen niet terug — de "
              f"drempel rekent over {result.runs_used}, niet over het "
              f"gevraagde aantal")
    for entry in result.pool_log:
        leden = ", ".join(result.concept_by_id[m].name for m in entry["members"])
        print(f"  FACETPOOL {entry['facet']}: {leden}  ({entry['n_resp']} resp)")

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

    if result.partition_repairs:
        # Geaggregeerd, niet per entry: bij N runs is elke reparatie een
        # normaal onderdeel van één los voorstel — de volledige lijst van
        # productie's report_codebook_build zou hier drukken wat gewoon is.
        # Wat wél diagnostisch is: HOEVEEL reparaties er over alle runs samen
        # nodig waren. Zonder deze telling duikt een model dat structureel
        # attributen vergeet alleen indirect op, als extra solo's.
        print(f"  PARTITIE: {result.partition_repairs} reparatie(s) over "
              f"{result.runs_used} runs — vergeten, dubbel geplaatste of "
              f"dubbel genoemde attributen die repair_partition per run "
              f"rechttrok")

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


def run_codebook(filename: str = None, var_name: str = None,
                    sample_size: Optional[int] = None,
                    force_recalc: bool = False,
                    config: ConsensusConfig = None) -> None:
    """Ingang van de consensuskandidaat. Leest de taxonomie uit de step-4-cache
    — dezelfde als productie — en schrijft het codeboek onder CACHE_STEP, waar
    step 6 het opent."""
    filename = FILENAME if filename is None else filename
    var_name = VARIABLE if var_name is None else var_name
    sample_size = SAMPLE_SIZE if sample_size is None else sample_size
    config = ConsensusConfig() if config is None else config

    print("=" * 70)
    print("CODE GENERATOR — CONSENSUSKANDIDAAT (loading taxonomy from cache)")
    print("=" * 70)

    variable_key = generate_enhanced_variable_key(
        selected_variables=[var_name], is_merged=False, sample_size=sample_size)

    cache_manager = CacheManager()
    if not force_recalc and cache_manager.is_metadata_cache_valid(
            filename, CACHE_STEP, variable_key):
        print("codeboek-cache geldig — overgeslagen (force_recalc=True om te herdraaien).\n")
        return

    metadata = load_extraction_metadata(filename, var_name, sample_size, variable_key)
    classified = load_classified_ideas(filename, var_name, sample_size, variable_key)
    taxonomy = load_taxonomy_cache(filename, var_name, sample_size, variable_key)
    if taxonomy is None:
        print("\nERROR: geen taxonomie in cache. Draai eerst step 4.")
        return

    # Zelfde afslag als productie, om dezelfde reden: zonder deze check zou
    # deze keten op een andere taxonomie bouwen dan de productieketen (en
    # daarmee ook een andere drempelbasis), en zou een codeboek-vergelijking
    # tussen de twee ketens niet meer opgaan.
    if cache_manager.is_metadata_cache_valid(filename, "taxonomy_corrected", variable_key):
        print("\nERROR: geldige 'taxonomy_corrected'-cache gevonden voor deze dataset. "
              "De consensuskandidaat ondersteunt dat legacy leespad niet — draai hem hier "
              "niet op, de vergelijking met de productieketen zou anders op een andere "
              "taxonomie gebeuren.")
        return

    refs = build_attribute_refs(taxonomy.partition_results)
    units = [u for u in build_idea_units(classified) if u.attribute_id in refs]
    concepts = build_inventory(units, refs)

    by_attribute: Dict[str, List[IdeaUnit]] = {}
    for unit in units:
        by_attribute.setdefault(unit.attribute_id, []).append(unit)

    # Dezelfde drempelbasis als productie: het totale aantal responses, niet
    # het aantal respondenten mét een idee. Wijkt deze keten hiervan af, dan
    # vergelijkt een codeboek-vergelijking twee codeboeken op verschillende
    # drempels.
    n_resp_total = len(classified)
    threshold = t_keep(n_resp_total, config)

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

    result = asyncio.run(generate_codebook(
        concepts, by_attribute, threshold, survey_question, n_respondents,
        dimension_diagnostic, language, config, verbose=config.verbose,
        prompt_printer=prompt_printer,
    ))
    report_codebook_build(result, config)

    cost_tracker.record_phase(
        COST_STEP, "codebook_generation",
        snapshot_before, token_tracker.snapshot(), model=config.model_writer,
    )
    cost_tracker.finalize_step(COST_STEP)

    # Eigen doctype, anders overschrijft deze run stil het promptexport van de
    # productieketen (`.save_prompts` opent in 'w'-modus, geen merge) — en de
    # vergelijking tussen de twee ketens' promptcapture is precies wat dit
    # codeboek moet mogelijk maken.
    save_prompts_to_json(prompt_printer, doctype="prompts_step5c")

    overig_name = apply_overig_sweep(result.codes, taxonomy.partition_results, language)
    print_codebook_results(result.codes)
    scorecard = run_scorecard(result.codes, taxonomy.partition_results, overig_name)

    if result.direction_loss:
        # De maat die RICHTINGSVERLIES's effect op déze run zichtbaar maakt:
        # een homeless tegenpool zonder counter-valence code is precies wat
        # under_split_codes telt.
        print(f"  (RICHTINGSVERLIES-effect in de scorecard: "
              f"{len(scorecard.under_split_codes)} under-split code(s))")

    # Degeneratie is een harde FAIL: melden, niet repareren — de codebook-
    # print en scorecard hierboven blijven dus draaien, alleen de cache-write
    # niet. Zonder deze afslag zou een ontaard voorstel onder CACHE_STEP landen
    # en step 6 het stilzwijgend inlezen, terwijl de DEGENERATIE-regel hierboven
    # al meldde dat het niet deugt.
    if result.degeneration:
        print(f"codeboek NIET gecached — degeneratie: {result.degeneration}")
        return

    print(f"codeboek cachen onder '{CACHE_STEP}' ({len(result.codes)} codes)...")
    cache_mece_results(
        taxonomy.partition_set, taxonomy.partition_results, result.codes,
        filename=filename, variable=var_name, sample_size=sample_size,
        variable_key=variable_key, step=CACHE_STEP,
    )


if __name__ == "__main__":
    with VerboseCapture(filename=FILENAME, var_name=VARIABLE,
                        sample_size=SAMPLE_SIZE, step="5c"):
        token_tracker.reset()
        run_codebook(force_recalc=True)
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())
