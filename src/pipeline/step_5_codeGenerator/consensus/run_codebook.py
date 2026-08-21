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
import contextlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

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

from .analysis import (
    consensus_ari, histogram, merge_recurrence, pairwise_ari, tau_sweep,
)
from .config_consensus import ConsensusConfig, effort_van
from .consensus import consensus_partition, dominant_member, together_from_runs
from .consolidation import resolve_consolidations
from .storage import RunSet, load_runset, save_runset

CACHE_STEP = "mece_codes"
# Eigen stapnaam in het kostenregister, zodat je kunt zien wat een route kost.
# De cachesleutel is gedeeld, de rekening niet.
COST_STEP = "step_5_consensus"

PRINT_PROMPTS = False  # True zet de prompts realtime op de console

# `run_codebook.py` zit een map dieper dan `run_classifier.py` (in
# `consensus/`, niet direct in de stap), dus één `parents`-index meer om
# hetzelfde projectroot te bereiken.
OUT_DIR = Path(__file__).resolve().parents[4] / "exports" / "experiment_logs"

# Alleen gebruikt door `analyse`'s tau-sweep — hoe vaak twee attributen samen
# moeten hebben gezeten om te mogen koppelen, over het hele bereik heen zodat
# je in één oogopslag ziet welke drempel een bruikbare indeling oplevert.
TAUS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def runset_path(config_name: str, set_index: int) -> Path:
    """De bestandsnaam draagt de configuratie en het setnummer, zodat twee
    configuraties nooit op hetzelfde bestand landen."""
    return OUT_DIR / f"consensus_{config_name}_set{set_index}.json"


def analyse(config: ConsensusConfig, set_index: int) -> None:
    """Fase 2 t/m 4 — kost geen enkele LLM-call."""
    path = runset_path(config.config_name, set_index)
    runset = load_runset(path)
    n_runs = len(runset.runs)

    print(f"\n{'=' * 78}\n{runset.model} / {runset.effort} — {n_runs} runs, "
          f"{len(runset.attribute_ids)} attributen, salted={runset.salted}"
          + (f", {runset.n_failed} mislukt" if runset.n_failed else "")
          + f"\n{'=' * 78}")

    print("\nAantal groepen per run:")
    print("  " + ", ".join(str(len(run)) for run in runset.runs))

    if n_runs < 2:
        # Zowel `pairwise_ari` als `measure_stability` hebben twee runs nodig;
        # bij één is er geen paar om te vergelijken en geen matrix om te vullen.
        print("\nMinstens twee runs nodig voor een ARI-vergelijking en een "
              "co-associatiematrix — verzamel er meer met `collect`.")
        return

    aris = pairwise_ari(runset.runs)
    print(f"\nFASE 2 — ARI tussen de runs ({len(aris)} vergelijkingen)")
    print(f"  laagste {min(aris):.3f}   mediaan {median(aris):.3f}   "
          f"hoogste {max(aris):.3f}")

    together = together_from_runs(runset.runs, runset.attribute_ids)
    counts = histogram(together, n_runs)
    total = sum(counts)
    print(f"\nFASE 2 — vorm van de matrix ({total} paren)")
    for n, aantal in enumerate(counts):
        if aantal:
            print(f"  {n:2d}/{n_runs} samen: {aantal:5d}  ({aantal / total:5.1%})")
    kern = counts[n_runs]
    schil = total - counts[0] - kern
    print(f"  kern (altijd samen): {kern}   schil (wisselend): {schil}")

    print("\nFASE 4 — tau-sweep")
    print(f"  {'tau':>5}  {'groepen':>8}  {'grootste':>9}  {'solo':>5}")
    for row in tau_sweep(together, runset.attribute_ids, n_runs, TAUS):
        print(f"  {row['tau']:>5.2f}  {row['n_groups']:>8d}  "
              f"{row['largest']:>9d}  {row['n_solo']:>5d}")


def vergelijk(config: ConsensusConfig, set_a: int = 1, set_b: int = 2) -> None:
    """Fase 5 — de hoofdmaat: ARI tussen twee onafhankelijke consensusindelingen.
    Kost geen enkele LLM-call — beide sets staan al op schijf.

    Welke twee sets is een argument en geen aanname: de sets op schijf zijn niet
    altijd 1 en 2 (de 30-runsmeting draaide op 3 en 4, en luna heeft geen set 2).
    """
    a = load_runset(runset_path(config.config_name, set_a))
    b = load_runset(runset_path(config.config_name, set_b))

    # `adjusted_rand_index` beperkt zich stil tot de doorsnede van de twee
    # eenhedenverzamelingen. Een step-4-herberekening tussen de twee sets zou de
    # hoofdmaat dan op een deelverzameling berekenen zonder waarschuwing — hier
    # weigeren in plaats van dat risico te lopen.
    if a.attribute_ids != b.attribute_ids:
        raise SystemExit(
            f"de attribuutuniversa van set {set_a} en set {set_b} verschillen — "
            "ARI zou stilzwijgend op de doorsnede berekend worden. Verzamel "
            "beide sets opnieuw tegen dezelfde step-4-cache.")

    together_a = together_from_runs(a.runs, a.attribute_ids)
    together_b = together_from_runs(b.runs, b.attribute_ids)
    clusters_a = consensus_partition(together_a, a.attribute_ids, len(a.runs), config.tau)
    clusters_b = consensus_partition(together_b, b.attribute_ids, len(b.runs), config.tau)

    print(f"\n{'=' * 78}\nFASE 5 — set {set_a} vs set {set_b}  "
          f"({config.config_name}, {len(a.runs)}+{len(b.runs)} runs, tau={config.tau})"
          f"\n{'=' * 78}")
    for index, runset, clusters in ((set_a, a, clusters_a), (set_b, b, clusters_b)):
        n_solo = sum(1 for cluster in clusters if len(cluster) == 1)
        degeneration = check_degeneration(len(clusters), len(runset.attribute_ids))
        print(f"  set {index}: {len(clusters)} groepen, {n_solo} solo's"
              f"  — {degeneration or 'geen degeneratie'}")

    # Louter solo's scoort hier 1.0, niet NaN — een vals perfecte score omdat
    # maximum en kansverwachting samenvallen (zie `consensus_ari`). Daarom staat
    # de degeneratieverdict hierboven altijd naast dit getal, nooit zonder.
    ari = consensus_ari(clusters_a, clusters_b)
    print(f"\n  ARI(set {set_a}, set {set_b}) = {ari:.3f}")

    # ARI weegt élke paarbeslissing even zwaar en gaat op een dunne indeling dus
    # vooral over attributen die toch alleen blijven. De samenvoegingen zijn wat
    # het codeboek bepaalt, dus die staan er als aparte maat naast.
    merges = merge_recurrence(clusters_a, clusters_b)
    overeenstemming = merges["pair_agreement"]
    print(f"  samenvoegingen: {merges['identical']} identiek "
          f"(van {merges['merges_a']} in set {set_a}, "
          f"{merges['merges_b']} in set {set_b})")
    print("  paarovereenstemming over samengevoegd materiaal: "
          + ("n.v.t. — geen van beide indelingen voegt iets samen"
             if overeenstemming is None else f"{overeenstemming:.1%}"))


@dataclass
class _PijplijnMateriaal:
    """Wat beide ingangen — `load_material()` en `run_codebook()` — uit de
    step-3/4-cache nodig hebben, vóórdat elk zijn eigen laatste stap zet:
    `load_material()` de kaarten, `run_codebook()` de cachegeldigheid en de
    drempel.

    `taxonomy` kan `None` zijn (geen step-4-cache in de cache). Welke melding
    daarbij hoort — een `SystemExit` of een `print` en een stille `return` —
    is aan de aanroeper, dus deze klasse beslist dat niet zelf; de overige
    velden zijn dan leeg/`0` in plaats van een keuze te forceren.
    """
    variable_key: str
    metadata: Any
    classified: List[Any]
    taxonomy: Any
    refs: Dict[str, Any]
    units: List[IdeaUnit]
    concepts: List[Concept]
    by_attribute: Dict[str, List[IdeaUnit]]
    language: str
    dimension_diagnostic: str
    n_respondents: int


def _laad_pijplijnmateriaal(filename: str, var_name: str, sample_size: int) -> _PijplijnMateriaal:
    """Inlezen en de conceptinventaris bouwen — het stuk dat `load_material()`
    en `run_codebook()` vóór deze samenvoeging allebei apart deden, twintig
    regels voor twintig regel identiek maar een paar honderd regels uit elkaar
    in hetzelfde bestand. Nu op één plek; elke aanroeper voegt zijn eigen
    laatste stap toe.
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[var_name], is_merged=False, sample_size=sample_size)

    metadata = load_extraction_metadata(filename, var_name, sample_size, variable_key)
    classified = load_classified_ideas(filename, var_name, sample_size, variable_key)
    taxonomy = load_taxonomy_cache(filename, var_name, sample_size, variable_key)
    if taxonomy is None:
        return _PijplijnMateriaal(
            variable_key=variable_key, metadata=metadata, classified=classified,
            taxonomy=None, refs={}, units=[], concepts=[], by_attribute={},
            language="", dimension_diagnostic="", n_respondents=0)

    refs = build_attribute_refs(taxonomy.partition_results)
    units = [u for u in build_idea_units(classified) if u.attribute_id in refs]
    concepts = build_inventory(units, refs)

    by_attribute: Dict[str, List[IdeaUnit]] = defaultdict(list)
    for unit in units:
        by_attribute[unit.attribute_id].append(unit)

    language = getattr(metadata, "lang", "") or "Dutch"
    dimension_name = getattr(metadata, "primary_dimension", "") or ""
    dimension_diagnostic = (get_dimension(dimension_name).criterion
                            if dimension_name else FALLBACK_DIAGNOSTIC)
    n_respondents = len({u.respondent_id for u in units})

    return _PijplijnMateriaal(
        variable_key=variable_key, metadata=metadata, classified=classified,
        taxonomy=taxonomy, refs=refs, units=units, concepts=concepts,
        by_attribute=by_attribute, language=language,
        dimension_diagnostic=dimension_diagnostic, n_respondents=n_respondents)


def load_material(config: ConsensusConfig) -> Dict:
    """Kaarten en concepten uit de step-4-cache. Zelfde route als de
    productiestap hieronder, alleen zonder iets weg te schrijven — dit voedt
    alleen `verzamelen`."""
    m = _laad_pijplijnmateriaal(FILENAME, VARIABLE, SAMPLE_SIZE)
    if m.taxonomy is None:
        raise SystemExit("geen taxonomie in cache — draai eerst step 4")

    return {
        # Komt uit de config, niet hardgecodeerd: zo verzamelt deze actie op
        # dezelfde uitsluiting als waarmee het codeboek er straks uit gebouwd
        # wordt, in plaats van een vaste aanname die van de configuratie kan
        # afwijken.
        "cards": build_cards(m.concepts, m.by_attribute, exclude_drains=config.exclude_drains),
        "concepts": m.concepts,
        "question": (getattr(m.metadata, "var_lab", "") or "").strip(),
        "language": m.language,
        "n_respondents": m.n_respondents,
    }


def verzamelen(config: ConsensusConfig, set_index: int) -> Path:
    """N keer deel 1, elke run met een eigen salt — of, met `config.salted=False`,
    N identieke aanroepen die de kale servervariatie blootleggen. Schrijft de
    partities weg.

    Synchroon, met een eigen `asyncio.run()` — net als `run_codebook()`
    verderop — zodat elke actie in dit bestand dezelfde vorm heeft en
    `__main__` geen losse async-tak nodig heeft.

    Eén `resolve_consolidations`-aanroep met alle salts, in plaats van een
    `for`-lus met een `await` per run: N losse aanroepen bouwen N losse
    requesters die elk één taak zien, en dan heeft de adaptieve
    doorvoerregeling waar die component voor bestaat niets te regelen (zie
    `consolidation.py`).
    """
    material = load_material(config)

    with effort_van(config):
        salts = [f"set{set_index}run{i}" if config.salted else ""
                 for i in range(config.runs)]
        proposals, mislukt = asyncio.run(resolve_consolidations(
            material["cards"], material["question"], material["n_respondents"],
            material["language"], config, salts,
        ))
        partitions = [[tuple(sorted(g.member_ids))
                       for g in repair_partition(p, material["cards"],
                                                 material["concepts"])]
                      for p in proposals]
        if mislukt:
            print(f"  LET OP: {mislukt} van de {config.runs} runs kwam niet terug")

    runset = RunSet(
        model=config.model_relations,
        effort=config.effort,
        attribute_ids=[c.attribute_id for c in material["cards"]],
        attribute_names={c.attribute_id: c.name for c in material["cards"]},
        n_respondents=material["n_respondents"],
        runs=partitions,
        salted=config.salted,
        n_failed=mislukt,
    )
    path = runset_path(config.config_name, set_index)
    save_runset(runset, path)
    print(f"\n{len(partitions)} van de {config.runs} runs weggeschreven naar {path}")
    return path


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
    partitions: Optional[List[List[Tuple[str, ...]]]] = None,
) -> GeneratedCodebook:
    cards = build_cards(concepts, idea_units_by_attribute,
                        exclude_drains=config.exclude_drains)

    # Eén log over alle N runs heen, niet één per run: een repareerbeurt per
    # run is hier de normale gang van zaken (elke run is een los voorstel),
    # dus wat aggregeert is bruikbaar diagnostiek — een model dat structureel
    # attributen vergeet — en een lijst van 30 runs' individuele reparaties
    # zou dat juist verdrinken.
    repair_log = _RepairLog()

    if partitions is None:
        # Productieroute: draai deel 1 zelf.
        salts = [f"run{i}" for i in range(config.runs)] if config.salted else [""] * config.runs
        proposals, runs_failed = await resolve_consolidations(
            cards, survey_question, n_respondents, language, config, salts,
            verbose=verbose, prompt_printer=prompt_printer,
        )
        partities = [
            [tuple(sorted(group.member_ids))
             for group in repair_partition(proposal, cards, concepts, log=repair_log)]
            for proposal in proposals
        ]
    else:
        # Meetroute: de runs zijn al betaald en staan op schijf. `repair_partition`
        # is er al overheen geweest toen ze werden weggeschreven, dus hier niet
        # nog eens — dat zou een tweede reparatie op gerepareerd materiaal zijn.
        partities, runs_failed = list(partitions), 0

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


class _Tee:
    """Schrijft tegelijk naar het scherm en naar het verslagbestand.

    Niet achteraf opvangen en dan wegschrijven: de schrijfcall duurt minuten en
    je wilt ondertussen zien dat er iets gebeurt.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        self._handle.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()


def report_path(config_name: str, set_index: int, source: str,
                tau: float, poles: str) -> Path:
    """De naam draagt de instellingen, zodat twee varianten nooit op elkaar
    landen en je achteraf weet waar een codeboek vandaan komt."""
    stem = f"codeboek_{config_name}_set{set_index}_{source}"
    if source == "consensus":
        stem += "_tau" + f"{tau:g}".replace(".", "")
    return OUT_DIR / f"{stem}_{poles}polen.txt"


def bezette_sets(config_name: str, *indices: int) -> List[int]:
    """Welke van deze setnummers al op schijf staan.

    `verzamelen` overschrijft zonder waarschuwing, en een set is RUNS LLM-calls
    die je niet terugkrijgt. Een ronde die per ongeluk op een bezet nummer
    landt wist dus het materiaal van een eerdere meting.
    """
    return [index for index in indices if runset_path(config_name, index).exists()]


def codeboek(config: ConsensusConfig, set_index: int, source: str) -> None:
    """Codeboek uit de partities die al op schijf staan — geen nieuwe deel-1-calls.

    Gaat door `run_codebook()`, niet door `generate_codebook()`: de cache-write,
    de kostenpost, de promptexport en de degeneratiepoort zitten in de eerste.
    Dat is het hele punt van deze taak — er stond hier een tweede kopie van de
    keten die dertig regels uit elkaar was gelopen en geen van de vijf
    deliverables schreef.

    Schrijft daarnaast het volledige verslag naar `exports/experiment_logs/`
    (via `_Tee`): `log_step5c.txt` (een van de vijf deliverables) wordt bij
    elke klik overschreven, terwijl deze bestandsnaam de instellingen draagt
    en dus achteraf nog zegt welke configuratie dit codeboek maakte. Ze
    vervangen elkaar niet, ze documenteren allebei iets anders.
    """
    runset = load_runset(runset_path(config.config_name, set_index))
    if not runset.runs:
        # `partitions=[]` zou hier ongemerkt doorstromen (zie taak 2's review)
        # tot diep in de consensusstap — hier weigeren, vóór de keten, met een
        # duidelijke reden, in plaats van daar op een onherkenbaar mankement
        # te stuiten.
        raise SystemExit(
            f"set {set_index} ({config.config_name}) bevat geen partities — "
            "verzamel eerst met `verzamelen`.")

    # Bij 'baseline' gaat er ÉÉN partitie in. Er valt dan niets te middelen en
    # de consensusstap geeft precies die ene indeling terug — dat is gewenst,
    # want dit is de referentie waar de consensusversie tegen afgezet wordt.
    # Zonder deze regel leest het als een bug.
    partities = runset.runs if source == "consensus" else [runset.runs[0]]

    poles = "two" if config.two_pole else "three"
    path = report_path(config.config_name, set_index, source, config.tau, poles)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(_Tee(sys.stdout, handle)):
            run_codebook(force_recalc=True, config=config, partitions=partities)
    print(f"\nCodeboek weggeschreven naar {path}")


def alles(config: ConsensusConfig, set_index: int, set_b: int) -> None:
    """De hele ronde achter elkaar: twee sets verzamelen, meten, en het
    consensuscodeboek bouwen.

    De losse acties bestaan om een ronde te kunnen onderbreken of om een
    opgeslagen set later opnieuw te bevragen. Wie de meting gewoon wil
    uitvoeren hoort dat niet in vijf losse stappen te hoeven doen.
    """
    bezet = bezette_sets(config.config_name, set_index, set_b)
    if bezet:
        raise SystemExit(
            f"set {' en '.join(map(str, bezet))} bestaat al voor "
            f"{config.config_name} en zou overschreven worden — dat is "
            f"{config.runs} LLM-calls per set die je kwijt bent. Kies vrije "
            f"setnummers.")

    verzamelen(config, set_index)
    verzamelen(config, set_b)
    analyse(config, set_index)
    vergelijk(config, set_index, set_b)
    codeboek(config, set_index, "consensus")


def run_codebook(filename: str = None, var_name: str = None,
                    sample_size: Optional[int] = None,
                    force_recalc: bool = False,
                    config: ConsensusConfig = None,
                    partitions: Optional[List[List[Tuple[str, ...]]]] = None) -> None:
    """Ingang van de consensuskandidaat. Leest de taxonomie uit de step-4-cache
    — dezelfde als productie — en schrijft het codeboek onder CACHE_STEP, waar
    step 6 het opent.

    `partitions` geeft door aan `generate_codebook`: gevuld slaat deel 1 over
    (de meetroute — de runs staan al op schijf), zodat de meetkant hier
    binnenkomt en de vijf leveringen (cache, kosten, prompts, perf, verbose)
    alsnog krijgt in plaats van zijn eigen kopie van de keten te onderhouden."""
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

    m = _laad_pijplijnmateriaal(filename, var_name, sample_size)
    if m.taxonomy is None:
        print("\nERROR: geen taxonomie in cache. Draai eerst step 4.")
        return
    metadata, classified, taxonomy = m.metadata, m.classified, m.taxonomy

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

    # Dezelfde drempelbasis als productie: het totale aantal responses, niet
    # het aantal respondenten mét een idee. Wijkt deze keten hiervan af, dan
    # vergelijkt een codeboek-vergelijking twee codeboeken op verschillende
    # drempels.
    n_resp_total = len(classified)
    threshold = t_keep(n_resp_total, config)
    survey_question = (getattr(metadata, "var_lab", "") or "").strip()

    print(f"  {len(m.concepts)} attributen, {m.n_respondents} met een idee, "
          f"T_keep = {threshold} over {n_resp_total} responses")

    cost_tracker = CostTracker(filename=filename, var_name=var_name,
                               sample_size=sample_size)
    snapshot_before = token_tracker.snapshot()
    prompt_printer = PromptPrinter(enabled=True, print_realtime=PRINT_PROMPTS)

    result = asyncio.run(generate_codebook(
        m.concepts, m.by_attribute, threshold, survey_question, m.n_respondents,
        m.dimension_diagnostic, m.language, config, verbose=config.verbose,
        prompt_printer=prompt_printer, partitions=partitions,
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

    overig_name = apply_overig_sweep(result.codes, taxonomy.partition_results, m.language)
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
