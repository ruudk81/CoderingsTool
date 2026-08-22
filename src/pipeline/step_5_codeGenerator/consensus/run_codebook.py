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

# Absolute imports, geen relatieve — dit bestand moet met een enkele klik (VS
# Code Code Runner, `python run_codebook.py`) blijven draaien, en dan is
# `__package__` leeg: `from ..attribute_cards import ...` zou stuklopen op
# "attempted relative import with no known parent package".
SRC = Path(__file__).resolve().parents[3]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key  # noqa: E402
from utils.costTracker import CostTracker  # noqa: E402
from utils.llm import token_tracker  # noqa: E402
from utils.promptPrinter import PromptPrinter  # noqa: E402
from utils.saveVerbose import VerboseCapture  # noqa: E402

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension  # noqa: E402

from pipeline.step_5_codeGenerator.consensus.attribute_cards import build_cards  # noqa: E402
from pipeline.step_5_codeGenerator.consensus.code_shape import (  # noqa: E402
    CodeShape, _match_shape, _shape_lookup,
)
from pipeline.step_5_codeGenerator.consensus.codebook_io import (  # noqa: E402
    FALLBACK_DIAGNOSTIC, FILENAME, SAMPLE_SIZE, VARIABLE, apply_overig_sweep,
    cache_mece_results, load_classified_ideas, load_extraction_metadata,
    load_taxonomy_cache, print_codebook_results, run_scorecard,
    save_prompts_to_json,
)
from pipeline.step_5_codeGenerator.consensus.codebook_writer import (  # noqa: E402
    find_duplicate_definitions, find_naming_mismatches, resolve_duplicate_names,
    write_codebook, write_miscellaneous,
)
from pipeline.step_5_codeGenerator.consensus.concept_inventory import (  # noqa: E402
    Concept, build_inventory, t_keep,
)
from pipeline.step_5_codeGenerator.consensus.grouping import (  # noqa: E402
    Group, build_shapes, check_degeneration, pool_thin_within_facet,
    repair_partition,
)
from pipeline.step_5_codeGenerator.consensus.taxonomy_input import (  # noqa: E402
    IdeaUnit, build_attribute_refs, build_idea_units,
)
from models import ConsolidatedCode  # noqa: E402

from pipeline.step_5_codeGenerator.consensus.analysis import (  # noqa: E402
    consensus_ari, histogram, merge_recurrence, pairwise_ari, tau_sweep,
)
from pipeline.step_5_codeGenerator.consensus.config_consensus import (  # noqa: E402
    ConsensusConfig, effort_van,
)
from pipeline.step_5_codeGenerator.consensus.consensus import (  # noqa: E402
    consensus_partition, dominant_member, together_from_runs,
)
from pipeline.step_5_codeGenerator.consensus.consolidation import (  # noqa: E402
    resolve_consolidations,
)
from pipeline.step_5_codeGenerator.consensus.storage import (  # noqa: E402
    RunSet, load_runset, save_runset,
)

# =============================================================================
# INSTELLINGEN — dit is wat een klik op Run doet
# =============================================================================
# Verander deze regels en druk op Run. `run_pipeline.py` roept de functie
# `run_codebook()` aan en ziet ACTIE nooit; die krijgt de standaardwaarden
# hieronder via ConsensusConfig.

AUTO = "auto"   # SET-waarde die zegt: kies zelf het volgende vrije nummer
MAX_SET = 999   # bovengrens van de setnummerscan; een ronde is 2 sets, dus ruim

ACTIE   = "alles"      # alles | verzamelen | codeboek | analyse | vergelijk
CONFIG  = "luna"       # luna (goedkoop) | gpt54 (12,5x duurder)
RUNS    = 30           # hoe vaak deel 1 draait
TAU     = 0.7          # hoe vaak twee attributen samen moeten hebben gezeten
SET     = AUTO         # nummer om naar te schrijven, of AUTO (= doortellen).
                       #   Bij 'analyse', 'vergelijk' en 'codeboek' MOET je een
                       #   nummer invullen: die lezen een bestaande set.
SET_B   = AUTO         # tweede set bij 'vergelijk' en 'alles'
SOURCE  = "consensus"  # alleen bij 'codeboek': consensus | baseline
POLES   = "two"        # two (niet-negatief/negatief) | three (pos/neu/neg)
DRAINS  = "uit"        # vangnetten op de kaarten: uit | aan
SALT    = "aan"        # aan = volgorde varieert per run; uit = kale servervariatie

#   alles        verzamelen x2 -> analyse -> vergelijk -> codeboek   RUNS*2+2 calls
#   verzamelen   alleen deel 1, partities naar schijf                RUNS calls
#   codeboek     uit de partities van SET                            2 calls
#   analyse      een set lezen: ARI, matrixvorm, tau-sweep           0 calls
#   vergelijk    SET tegen SET_B: hoofdmaat en merge-recurrentie     0 calls

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
              "co-associatiematrix — verzamel er meer met `verzamelen`.")
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
    metadata: Any
    classified: List[Any]
    taxonomy: Any
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
            metadata=metadata, classified=classified,
            taxonomy=None, concepts=[], by_attribute={},
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
        metadata=metadata, classified=classified,
        taxonomy=taxonomy, concepts=concepts,
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


def verzamelen(config: ConsensusConfig, set_index, prompt_printer=None,
               cost_tracker=None) -> Path:
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

    Dit is de actie die betaalt — RUNS calls, de dure kant van de keten — dus
    hij krijgt zijn eigen kostenregistratie en promptexport, net als
    `run_codebook()` verderop doet voor de schrijfcall. Zonder deze twee
    boekte een volledige `alles`-ronde alleen de ene schrijfcall onder
    `COST_STEP`, terwijl de consolidatiecalls — de meerderheid van de kosten —
    nergens stonden.
    """
    if set_index == AUTO:
        set_index = vrije_sets(config.config_name, 1)[0]
        print(f"SET = {AUTO!r}: deze run schrijft naar set {set_index}")

    material = load_material(config)

    # Een aangereikte printer/teller betekent: je bent onderdeel van een ronde,
    # en die schrijft aan het eind één keer weg. Zelf wegschrijven zou de
    # export van de vorige stap overschrijven — `save_prompts.py` opent in
    # 'w'-modus en `record_phase` WIJST TOE op (stap, fase). Op de ronde van
    # 2026-08-22 kostte dat 60 van de 60 consolidatieprompts en de helft van de
    # geboekte calls.
    eigen_boekhouding = cost_tracker is None
    if prompt_printer is None:
        prompt_printer = PromptPrinter(enabled=True, print_realtime=PRINT_PROMPTS)
    if eigen_boekhouding:
        cost_tracker = CostTracker(filename=FILENAME, var_name=VARIABLE,
                               sample_size=SAMPLE_SIZE)
    snapshot_before = token_tracker.snapshot()

    with effort_van(config):
        salts = [f"set{set_index}run{i}" if config.salted else ""
                 for i in range(config.runs)]
        proposals, mislukt = asyncio.run(resolve_consolidations(
            material["cards"], material["question"], material["n_respondents"],
            material["language"], config, salts,
            verbose=config.verbose, prompt_printer=prompt_printer,
        ))
        partitions = [[tuple(sorted(g.member_ids))
                       for g in repair_partition(p, material["cards"],
                                                 material["concepts"])]
                      for p in proposals]
        if mislukt:
            print(f"  LET OP: {mislukt} van de {config.runs} runs kwam niet terug")

    if eigen_boekhouding:
        cost_tracker.record_phase(
            COST_STEP, "consolidation",
            snapshot_before, token_tracker.snapshot(),
            model=config.model_relations,
        )
        cost_tracker.finalize_step(COST_STEP)
        save_prompts_to_json(prompt_printer, doctype="prompts_step5c")

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
    coverage_recovered: int
    first_time_covered: int
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
    """De herbruikbare kern van de keten (fase 1 t/m 5), zonder cache, kosten,
    prompts of verbose-log — die zitten in `run_codebook()` eromheen.

    `partitions=None` is de productieroute: deel 1 draait hier zelf, N verse
    consolidatiecalls. Een gevulde `partitions` is de meetroute: de runs staan
    al op schijf (`verzamelen` heeft ze al door `repair_partition` gehaald),
    dus deel 1 en de reparatie slaan hier over — dat voorkomt zowel een tweede
    reparatie op al gerepareerd materiaal als N overbodige herhaalde calls.
    """
    cards = build_cards(concepts, idea_units_by_attribute,
                        exclude_drains=config.exclude_drains)

    # Eén log over alle N runs heen, niet één per run: een repareerbeurt per
    # run is hier de normale gang van zaken (elke run is een los voorstel),
    # dus wat aggregeert is bruikbaar diagnostiek — een model dat structureel
    # attributen vergeet — en een lijst van 30 runs' individuele reparaties
    # zou dat juist verdrinken.
    repair_log = _RepairLog()

    if partitions is None:
        # Productieroute: draai deel 1 zelf. Zelfde `effort_van`-omhulling als
        # `verzamelen()` gebruikt voor dezelfde call — zonder deze wikkel liep
        # deze route altijd op `STEP_EFFORT`'s ongewijzigde waarde (`high`),
        # ongeacht welke `CONFIG` (bijv. luna/`medium`) was gekozen: twee
        # deuren naar dezelfde fase die een ander antwoord gaven.
        with effort_van(config):
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
    shaped = build_shapes(groups, concepts, threshold, two_pole=config.two_pole,
                          floor=config.t_keep_min_respondents)

    # Twee schrijfcalls, gesplitst op HERKOMST. Een kind is een restcategorie
    # binnen één onderwerp en mag geen eigen kop claimen; de hoofdcodeprompt
    # vraagt juist wél om een kop. Zonder deze splitsing zou `write_codebook`
    # álle vormen krijgen — kinderen inbegrepen — en dan zouden ze twee keer in
    # het boek staan zodra de tweede call erbij komt: één keer met de verkeerde
    # instructie geschreven.
    hoofdvormen = [s for s in shaped.shapes if s.origin != "child"]
    kindvormen = [s for s in shaped.shapes if s.origin == "child"]

    # `write_codebook` can veto a `pooled` shape (`nameable: false`) — that is
    # the shape a consolidation call proposed, and this chain proposes them by
    # the dozen (`grouping.build_shapes`), so this is the normal route, not an
    # edge case. `recovered` en `child` blijven er buiten: die zijn step 4's
    # eigen structuur, geen modelvoorstel, en een veto zou hun respondenten in
    # de regel terugzetten onder de zusterpool die het tegenovergestelde beweert
    # — en anders ononderscheiden in Overig (zie de vetotoets in
    # `codebook_writer`). De attributen van een geveto'de `pooled`
    # vorm gaan niet verloren (de Overig-sweep in `run_codebook` routeert ze),
    # maar een stil veto zou een vergelijking met de productieketen een kleiner
    # codeboek met een grotere Overig laten zien zonder consolidatiekwaliteit
    # van schrijversvetos te kunnen scheiden — dezelfde reden waarom degeneratie
    # wordt gemeld en niet geabsorbeerd. `veto_log` maakt het zichtbaar.
    veto_log = _RepairLog()
    codes = await write_codebook(
        hoofdvormen, concepts, dimension_diagnostic, language, config,
        log=veto_log, verbose=verbose, prompt_printer=prompt_printer,
    )
    # `taken_names` zijn de namen die de eerste call zojuist vastlegde. De
    # kinderen zien die vormen niet, dus zonder deze lijst kan een kind precies
    # de naam kiezen die een hoofdcode al draagt. Een promptregel is hier de
    # vraag, niet de garantie: `resolve_duplicate_names` draait hieronder over
    # de HERENIGDE lijst en is de deterministische achtervang.
    kindcodes = await write_miscellaneous(
        kindvormen, concepts, dimension_diagnostic, language, config,
        verbose=verbose, prompt_printer=prompt_printer,
        taken_names=[code.code_name for code in codes],
    )

    # `codes[i]` must line up with `shapes[i]` for resolve_duplicate_names and
    # the two finders below (their own docstrings require it) — but a veto in
    # write_codebook means `codes` can be shorter than `shaped.shapes`. Match
    # each written code back to its own shape rather than assuming the two
    # lists still walk in lockstep — the same technique v1's
    # `_generate_codebook_async` uses for the same reason.
    #
    # ÉÉN lookup over ALLE vormen, en de match op (bronnamen, valentie) — niet
    # twee lookups en niet zippen. De twee calls geven hun codes in hun eigen
    # volgorde terug, en volgorde is geen identiteit.
    concept_by_id = {c.attribute_id: c for c in concepts}
    shape_lookup = _shape_lookup(shaped.shapes, concept_by_id)
    codes = codes + kindcodes
    shapes = [_match_shape(code, shape_lookup) for code in codes]

    collision_log = _RepairLog()
    codes = resolve_duplicate_names(codes, shapes, log=collision_log,
                                    language=language)
    return GeneratedCodebook(
        shapes=shapes, overig_ids=shaped.overig_ids, codes=codes,
        coverage_recovered=shaped.coverage_recovered,
        first_time_covered=shaped.first_time_covered, degeneration=degeneration,
        partition_repairs=len(repair_log.entries), collisions=collision_log.entries,
        naming_mismatches=find_naming_mismatches(codes, shapes, concept_by_id),
        duplicate_definitions=find_duplicate_definitions(codes, shapes),
        vetoes=veto_log.entries,
        concept_by_id=concept_by_id,
        runs_used=len(partities), runs_failed=runs_failed, pool_log=pool_log,
    )


def link_children_to_overig(codes: List[ConsolidatedCode], shapes: List[CodeShape],
                            parent: ConsolidatedCode) -> List[str]:
    """Hang elk kind aan de Overig-code — in een VELD, nooit in een naam.

    Geeft de `K#`'s terug van de codes die volgens hun VORM een kind zijn. Dat
    is de bedoeling, en die gaat naar de scorecard, die hem tegen het
    `parent_code_id`-veld legt. Die toets is een struikeldraad en geen dekking:
    dezelfde lus zet hier het veld, dus hij kan alleen vuren als een kind zijn
    ouder daarna kwijtraakt. Zie `build_scorecard`'s `child_code_ids` voor
    waarom er geen afleiding bestaat die het er wél mee oneens kan zijn.

    Kan niet eerder dan hier. De ouder wordt door `apply_overig_sweep` gemaakt
    en krijgt daar zijn `K#`; vóór die sweep bestaat er geen id om naar te
    wijzen. `codes` draagt op dit moment precies één element méér dan `shapes` —
    Overig zelf, achteraan aangeplakt — en de rest is nog steeds de positionele
    afspraak die `resolve_duplicate_names` en de twee vinders ook hanteren:
    `codes[i]` is de tekst van `shapes[i]`. Die afspraak wordt hier getoetst in
    plaats van aangenomen, want een verschoven index zou de verkeerde codes
    onder Overig hangen zonder ergens te falen.
    """
    if len(codes) != len(shapes) + 1:
        raise ValueError(
            "codes moet shapes plus precies de Overig-code lang zijn — "
            f"{len(codes)} tegen {len(shapes)} + 1")

    kind_ids: List[str] = []
    for code, shape in zip(codes, shapes):
        if shape is not None and shape.origin == "child":
            code.parent_code_id = parent.code_id
            kind_ids.append(code.code_id)
    return kind_ids


def report_true_overig(result: GeneratedCodebook, overig: ConsolidatedCode) -> None:
    """Wat "echt-overig" vandaag écht betekent, in twee getallen.

    `build_shapes` verzamelt in `overig_ids` de attributen wier facetunie onder
    de bodem bleef. `apply_overig_sweep` leidt Overig daar NIET uit af: die
    neemt de taxonomie-attributen die geen enkele code noemt. Voor een attribuut
    wiens ene pool in een code landde en wiens andere pool door de bodem zakte
    is dat verschil beslissend — die code noemt het attribuut, dus het is geen
    wees, dus die respondenten worden nergens geteld.

    Sinds de verbreding van 2026-08-22 wordt dat getal GROTER in plaats van
    kleiner, en dat is geen regressie maar hetzelfde gat op een breder vlak:
    doordat élke groep nu polen aanlevert, houdt een attribuut vaker érgens een
    code over. Op set 7 (luna, tau=0,7, drempel 23) bleven 9 attributen onder de
    bodem en werden ze alle 9 nog door een code genoemd — onder de smalle regel
    waren dat er 14, waarvan 5.

    Gemeld en niet gerepareerd, en dat is een besluit met een reden. Overig die
    namen laten claimen zet hetzelfde attribuut TWEE keer in het codeboek: één
    keer onder zijn eigen hoofdcode en één keer onder Overig. Wat er nodig is,
    is een Overig die één VALENTIE van een attribuut opneemt, en dat kan
    `ConsolidatedCode` niet uitdrukken — één code draagt één valentie over al
    zijn bronattributen. Dat is een wijziging aan het gedeelde contract in
    `models.py` en aan step 6's toewijzing, niet aan de bedrading van deze
    keten. Zolang dat er niet is, is dit getal het verschil tussen wat het plan
    belooft en wat de keten levert — en een getal dat op de console staat is
    geen stille aanname meer.

    Dat verschil wordt in RESPONDENTEN gemeld en niet alleen in attributen. Een
    attribuuttelling zegt niets over de omvang: op set 7 staan 9 attributen
    onder de bodem terwijl er 5 respondenten van 2317 werkelijk in geen enkele
    code voorkomen. Twee keer in dit plan is een besluit bijna genomen op een
    getal dat iets anders telde dan zijn naam beloofde; dit is de eenheid
    waarin het besluit valt.
    """
    # Nergens = in geen enkele vorm, en ook niet via een bronattribuut van
    # Overig. Vormen, niet codes: een geveto'de vorm heeft geen code, dus zijn
    # respondenten staan er terecht niet in.
    alle: set = set()
    resp_per_naam: Dict[str, set] = {}
    for concept in result.concept_by_id.values():
        alle |= concept.resp_ids
        resp_per_naam.setdefault(concept.name, set()).update(concept.resp_ids)
    in_vorm = frozenset().union(*(s.resp_ids for s in result.shapes if s is not None))
    in_overig: set = set()
    for naam in (overig.source_attributes or []):
        in_overig |= resp_per_naam.get(naam, set())
    nergens = alle - in_vorm - in_overig
    if not nergens and not result.overig_ids:
        # Geen gat in respondenten en geen attribuut onder de bodem: er is
        # niets te melden, en een regel die dat toch afdrukt is ruis.
        return
    print(f"ECHT-OVERIG: {len(nergens)} van {len(alle)} respondent(en) komen in geen "
          f"enkele code voor — geen vorm, en ook niet via een bronattribuut van "
          f"'{overig.code_name}'.")

    if not result.overig_ids:
        return

    namen = [result.concept_by_id[i].name for i in result.overig_ids
             if i in result.concept_by_id]
    bronnen = set(overig.source_attributes or [])
    zwevend = [n for n in namen if n not in bronnen]
    print(f"  IN ATTRIBUTEN: {len(namen)} attribuut(en) bleven onder de bodem; "
          f"{len(namen) - len(zwevend)} daarvan staan in '{overig.code_name}'.")
    if zwevend:
        print(f"  LET OP: {len(zwevend)} niet, omdat een overlevende code ze nog "
              f"noemt — hun afgevallen pool wordt nergens geteld: "
              f"{', '.join(sorted(zwevend))}")


def report_codebook_build(result: GeneratedCodebook, config: ConsensusConfig) -> None:
    """Wat een run zichtbaar moet maken. Eerst wat geen enkele bestaande check
    meldt — hoeveel runs meetelden en wat de facetpool samenvoegde — daarna
    dezelfde diagnostiek als productie: degeneratie, herstelde dekking, vetoes,
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

    kinderen = [s for s in result.shapes if s is not None and s.origin == "child"]
    kind_resp = frozenset().union(*(s.resp_ids for s in kinderen)) if kinderen else frozenset()
    # De hoofdmaat van deze bouw, en de enige die zonder rekenwerk te vergelijken
    # is met een codeboek van vóór de kinderen: hoeveel codes een eigen kop
    # dragen, en hoeveel eronder hangen. Overig zelf zit hier nog niet bij — die
    # maakt `apply_overig_sweep` pas na deze rapportage.
    print(f"CODES: {len(result.codes) - len(kinderen)} hoofdcode(s) + "
          f"{len(kinderen)} kind(eren) onder Overig, samen {len(kind_resp)} "
          f"respondent(en) in de kinderen")

    if result.coverage_recovered:
        # Respondent-uniek, geen groepstelling: wie in twee groepen van hetzelfde
        # facet een afgevallen pool had telt één keer. `coverage_recovered` is de
        # OMVANG van die vormen, geen reddingstelling — een respondent erin kan
        # al eerder een code hebben gehad via een solo/pooled vorm van hetzelfde
        # of een ander attribuut in het facet. `first_time_covered` is het
        # verzamelingsverschil dat dát wél meet. De voorganger (RICHTINGSVERLIES)
        # telde het omgekeerde — verloren pool-plaatsingen — en dat getal zakt
        # sinds `pool_minority_poles` naar bijna nul omdat er niets meer wegvalt,
        # niet omdat het codeboek beter werd.
        print(f"DEKKING HERSTELD: {result.coverage_recovered} respondent(en) in "
              f"een hoofdcode of kind uit de facetpool van afgevallen polen — "
              f"polen uit élke groep, ook uit groepen waar geen enkele pool de "
              f"drempel haalde. Daarvan kregen {result.first_time_covered} voor "
              f"het eerst een code; de rest stond al ergens via een solo/pooled "
              f"vorm. Wat ook samengenomen onder de bodem bleef is echt-overig.")

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


def vrije_sets(config_name: str, hoeveel: int) -> List[int]:
    """De volgende `hoeveel` setnummers voor deze configuratie, doortellend
    boven het hoogste dat al bestaat.

    Bestaat omdat `SET` de enige knop in het blok was die bij ELKE ronde moest
    veranderen, met een weigering als faalmodus — dat is geen instelling maar
    een teller die de gebruiker zelf bijhield.

    Doortellen, en nadrukkelijk NIET het laagste gat vullen. Een gat betekent
    meestal dat daar een set is weggegooid; hergebruik je dat nummer, dan wijst
    "set 2" in aantekeningen van vorige week naar ander materiaal dan "set 2"
    van vandaag. Een setnummer moet één ding blijven aanwijzen.

    Per configuratie geteld: luna en gpt54 hebben hun eigen reeks.
    """
    hoogste = -1
    for nummer in range(MAX_SET + 1):
        if runset_path(config_name, nummer).exists():
            hoogste = nummer
    return list(range(hoogste + 1, hoogste + 1 + hoeveel))


def _eis_bestaand_setnummer(waarde, actie: str) -> int:
    """`"auto"` betekent "kies vrije nummers om NAAR te schrijven", en dat is
    betekenisloos voor een actie die van een set LEEST — die zou dan een set
    aanwijzen die per definitie niet bestaat. Eén woord met twee betekenissen
    is precies wat later bijt, dus hier stopt het."""
    if waarde == AUTO:
        raise SystemExit(
            f"SET = {AUTO!r} kan niet bij '{actie}': die leest een bestaande "
            f"set, en 'auto' kiest juist een nog ONgebruikt nummer. Vul het "
            f"nummer in van de set die je wilt bekijken.")
    return waarde


def codeboek(config: ConsensusConfig, set_index: int, source: str,
             prompt_printer=None, cost_tracker=None) -> None:
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

    # `generate_codebook` telt paren alleen over `ids` uit de HUIDIGE
    # step-4-cache. Is die sinds het verzamelen van deze set veranderd, dan
    # verdwijnen attributen die niet meer bestaan stilzwijgend uit de telling
    # en worden nieuwe attributen automatisch solo — en dit is de actie die
    # het resultaat onder `mece_codes` wegschrijft, waar step 6 en 7 het
    # zonder waarschuwing overnemen. `vergelijk` weigert precies dit tussen
    # twee sets; hier geldt dezelfde weigering tussen de set en de cache van nu.
    huidige_ids = [c.attribute_id for c in load_material(config)["cards"]]
    if huidige_ids != runset.attribute_ids:
        raise SystemExit(
            f"het attribuutuniversum van set {set_index} ({config.config_name}) "
            "wijkt af van de huidige step-4-cache — het codeboek zou stilzwijgend "
            "op de doorsnede gebouwd worden, met verdwenen attributen en nieuwe "
            "solo's als gevolg. Verzamel de set opnieuw tegen de huidige "
            "step-4-cache.")

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
            run_codebook(force_recalc=True, config=config, partitions=partities,
                 prompt_printer=prompt_printer, cost_tracker=cost_tracker)
    print(f"\nCodeboek weggeschreven naar {path}")


def alles(config: ConsensusConfig, set_index: int, set_b: int) -> None:
    """De hele ronde achter elkaar: twee sets verzamelen, meten, en het
    consensuscodeboek bouwen.

    De losse acties bestaan om een ronde te kunnen onderbreken of om een
    opgeslagen set later opnieuw te bevragen. Wie de meting gewoon wil
    uitvoeren hoort dat niet in vijf losse stappen te hoeven doen.
    """
    if set_index == AUTO or set_b == AUTO:
        set_index, set_b = vrije_sets(config.config_name, 2)
        print(f"SET = {AUTO!r}: deze ronde schrijft naar set {set_index} en {set_b}")

    bezet = bezette_sets(config.config_name, set_index, set_b)
    if bezet:
        vrij = vrije_sets(config.config_name, 2)
        raise SystemExit(
            f"set {' en '.join(map(str, bezet))} bestaat al voor "
            f"{config.config_name} en zou overschreven worden — dat is "
            f"{config.runs} LLM-calls per set die je kwijt bent. Vrij zijn "
            f"{vrij[0]} en {vrij[1]}, of zet SET = {AUTO!r}.")

    # Eén printer en één teller voor de hele ronde. Lieten de acties dat zelf
    # doen, dan overschrijft elke volgende de vorige: `save_prompts` opent in
    # 'w'-modus en `record_phase` wijst toe op (stap, fase). Gemeten op de ronde
    # van 2026-08-22: 0 van de 60 consolidatieprompts bewaard, 30 van de 60
    # calls geboekt.
    printer = PromptPrinter(enabled=True, print_realtime=PRINT_PROMPTS)
    teller = CostTracker(filename=FILENAME, var_name=VARIABLE,
                         sample_size=SAMPLE_SIZE)
    voor_consolidatie = token_tracker.snapshot()

    verzamelen(config, set_index, prompt_printer=printer, cost_tracker=teller)
    verzamelen(config, set_b, prompt_printer=printer, cost_tracker=teller)

    # Beide verzamelrondes in één post: de fasenaam is de sleutel, dus twee
    # keer boeken zou de eerste wissen in plaats van optellen.
    na_consolidatie = token_tracker.snapshot()
    teller.record_phase(COST_STEP, "consolidation", voor_consolidatie,
                        na_consolidatie, model=config.model_relations)

    analyse(config, set_index)
    vergelijk(config, set_index, set_b)
    codeboek(config, set_index, "consensus",
             prompt_printer=printer, cost_tracker=teller)

    teller.record_phase(COST_STEP, "codebook_generation", na_consolidatie,
                        token_tracker.snapshot(), model=config.model_writer)
    teller.finalize_step(COST_STEP)
    save_prompts_to_json(printer, doctype="prompts_step5c")


def run_codebook(filename: str = None, var_name: str = None,
                    sample_size: Optional[int] = None,
                    force_recalc: bool = False,
                    config: ConsensusConfig = None,
                    partitions: Optional[List[List[Tuple[str, ...]]]] = None,
                    prompt_printer=None, cost_tracker=None) -> None:
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

    # De drempel telt in RESPONDENTEN, niet in responses. `build_shapes` toetst
    # hem tegen `len(pool.resp_ids)` — een verzameling unieke respondenten — dus
    # een noemer in responses zet teller en noemer in verschillende eenheden.
    # Deze keten week tot 2026-08-22 af: hij rekende met `len(classified)` om
    # dezelfde drempelbasis te houden als productie (en daarvóór als v1), zodat
    # twee codeboeken op één lat lagen. Op de ASN-set scheelde dat 27 tegen 23,
    # dus poolen met 23 t/m 26 respondenten vielen af op een grens die over iets
    # anders ging dan wat er geteld werd.
    #
    # LET OP bij vergelijken: productie's `run_codebook.py` rekent nog met
    # responses, dus een codeboek van deze keten en een van productie liggen nu
    # NIET meer op dezelfde drempel.
    n_resp_total = m.n_respondents
    threshold = t_keep(n_resp_total, config)
    survey_question = (getattr(metadata, "var_lab", "") or "").strip()

    print(f"  {len(m.concepts)} attributen, {len(classified)} responses, "
          f"T_keep = {threshold} over {n_resp_total} respondenten")

    # Aangereikt betekent: je bent onderdeel van een ronde die zelf afsluit.
    # Zie de toelichting in `verzamelen`.
    eigen_boekhouding = cost_tracker is None
    if prompt_printer is None:
        prompt_printer = PromptPrinter(enabled=True, print_realtime=PRINT_PROMPTS)
    if eigen_boekhouding:
        cost_tracker = CostTracker(filename=filename, var_name=var_name,
                               sample_size=sample_size)
    snapshot_before = token_tracker.snapshot()

    result = asyncio.run(generate_codebook(
        m.concepts, m.by_attribute, threshold, survey_question, m.n_respondents,
        m.dimension_diagnostic, m.language, config, verbose=config.verbose,
        prompt_printer=prompt_printer, partitions=partitions,
    ))
    report_codebook_build(result, config)

    if eigen_boekhouding:
        cost_tracker.record_phase(
            COST_STEP, "codebook_generation",
            snapshot_before, token_tracker.snapshot(),
            model=config.model_writer,
        )
        cost_tracker.finalize_step(COST_STEP)

    # Eigen doctype, anders overschrijft deze run stil het promptexport van de
    # productieketen (`.save_prompts` opent in 'w'-modus, geen merge) — en de
    # vergelijking tussen de twee ketens' promptcapture is precies wat dit
    # codeboek moet mogelijk maken.
    if eigen_boekhouding:
        save_prompts_to_json(prompt_printer, doctype="prompts_step5c")

    # De sweep maakt de ouder en mint de K#'s; pas dáárna kunnen de kinderen
    # eraan hangen. Andersom zou een kind naar een lege id wijzen.
    overig = apply_overig_sweep(result.codes, taxonomy.partition_results, m.language)
    kind_ids = link_children_to_overig(result.codes, result.shapes, overig)
    if kind_ids:
        print(f"HIËRARCHIE: {len(kind_ids)} kind(eren) hangen onder "
              f"'{overig.code_name}' ({overig.code_id})")
    report_true_overig(result, overig)
    print_codebook_results(result.codes)
    scorecard = run_scorecard(result.codes, taxonomy.partition_results,
                              overig.code_name, child_code_ids=set(kind_ids))

    if result.coverage_recovered:
        # De tegenmetriek, niet de bevestiging: `under_split_codes` telt een
        # dakloze tegenpool zonder counter-valence code. Werkt de facetpool,
        # dan hoort dit getal te DALEN — stijgt het, dan zijn er polen
        # bijgekomen zonder hun tegenhanger.
        print(f"  (tegenmetriek in de scorecard: "
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
        # `result.runs_used` (niet `config.runs`): op de meetroute kunnen runs
        # mislukken, en de herkomstregel moet zeggen hoeveel er echt in de
        # matrix zaten, niet hoeveel er aangevraagd waren.
        narrative=provenance(config, result.runs_used),
    )


def provenance(config: ConsensusConfig, runs_used: int) -> str:
    """Eén regel die zegt waar dit codeboek vandaan komt.

    Gaat mee in `codebook_narrative`, een veld dat deze keten verder niet
    gebruikt. Licht misbruik van een legacy-veld, en dat is de ruil: het
    alternatief is `models.py` verruimen, en dat raakt step 6 en 7 voor iets
    wat alleen documentatie is.
    """
    polen = "twee polen" if config.two_pole else "drie polen"
    return (f"consensus over {runs_used} runs, tau={config.tau}, {polen}, "
            f"vangnetten {'uitgesloten' if config.exclude_drains else 'inbegrepen'}")


def config_uit_instellingen() -> ConsensusConfig:
    """Vertaalt het blok naar de ene knoppentabel. POLES/DRAINS/SALT lezen als
    tekst prettiger bij een klik op Run; `ConsensusConfig` zelf kent alleen
    booleans — die vertaling gebeurt hier en nergens anders, anders ontstaat
    er een tweede plek waar knoppen staan."""
    return ConsensusConfig(
        config_name=CONFIG,
        runs=RUNS,
        tau=TAU,
        two_pole=(POLES == "two"),
        exclude_drains=(DRAINS == "uit"),
        salted=(SALT == "aan"),
    )


ACTIES = {"alles", "verzamelen", "codeboek", "analyse", "vergelijk"}
SOURCES = {"consensus", "baseline"}


def _draai_actie(actie: str) -> None:
    """Dispatcht op ACTIE. Een tikfout hoort een nette melding te geven —
    geen stille no-op — dus een onbekende actie stopt hier hard, met de vijf
    geldige namen erbij. SOURCE geldt alleen bij `codeboek` en krijgt daar
    dezelfde behandeling: zonder deze wacht betekent elke waarde die niet
    letterlijk `"consensus"` is stilzwijgend `baseline`, en draagt
    `report_path` die tikfout dan door in de bestandsnaam."""
    if actie not in ACTIES:
        raise SystemExit(
            f"onbekende ACTIE {actie!r} — kies uit {', '.join(sorted(ACTIES))}")
    if actie == "codeboek" and SOURCE not in SOURCES:
        raise SystemExit(
            f"onbekende SOURCE {SOURCE!r} — kies uit {', '.join(sorted(SOURCES))}")

    config = config_uit_instellingen()
    if actie == "alles":
        alles(config, SET, SET_B)
    elif actie == "verzamelen":
        verzamelen(config, SET)
    elif actie == "codeboek":
        codeboek(config, _eis_bestaand_setnummer(SET, actie), SOURCE)
    elif actie == "analyse":
        analyse(config, _eis_bestaand_setnummer(SET, actie))
    elif actie == "vergelijk":
        vergelijk(config, _eis_bestaand_setnummer(SET, actie),
                  _eis_bestaand_setnummer(SET_B, actie))


if __name__ == "__main__":
    with VerboseCapture(filename=FILENAME, var_name=VARIABLE,
                        sample_size=SAMPLE_SIZE, step="5c"):
        token_tracker.reset()
        _draai_actie(ACTIE)
        if token_tracker.call_count > 0:
            print(token_tracker.get_summary())
