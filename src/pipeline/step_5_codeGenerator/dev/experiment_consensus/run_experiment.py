#%%
"""Consensus over N consolidatieruns — het experiment.

Draait deel 1 van step 5 meerdere keren met een wisselende volgordesalt, telt
per attribuutpaar in hoeveel runs de twee samen zaten, en snijdt die matrix
deterministisch tot één indeling.

    python run_experiment.py collect --config luna  --runs 10 --set 1
    python run_experiment.py collect --config gpt54 --runs 10 --set 1
    python run_experiment.py analyse --config luna  --set 1
    python run_experiment.py compare --config luna  --tau 0.8

Of, zonder commandoregel: zet de INSTELLINGEN hieronder goed en druk op Run.
Een aanroep met argumenten wint altijd van die instellingen.

Raakt de productiecache niet aan: er wordt niets weggeschreven onder
`mece_codes`.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

SRC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

import config as project_config  # noqa: E402
from utils.cacheManager import generate_enhanced_variable_key  # noqa: E402

from pipeline.step_3_ideaExtractor.dimension_data import get_dimension  # noqa: E402
from pipeline.step_5_codeGenerator.attribute_cards import build_cards  # noqa: E402
from pipeline.step_5_codeGenerator.codebook_io import (  # noqa: E402
    FALLBACK_DIAGNOSTIC, FILENAME, SAMPLE_SIZE, VARIABLE,
    apply_overig_sweep, load_classified_ideas, load_extraction_metadata,
    load_taxonomy_cache, print_codebook_results, run_scorecard,
)
from pipeline.step_5_codeGenerator.code_shape import _match_shape, _shape_lookup  # noqa: E402
from pipeline.step_5_codeGenerator.codebook_writer import (  # noqa: E402
    resolve_duplicate_names, write_codebook,
)
from pipeline.step_5_codeGenerator.concept_inventory import build_inventory, t_keep  # noqa: E402
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig  # noqa: E402
from pipeline.step_5_codeGenerator.consolidation import resolve_consolidation  # noqa: E402
from pipeline.step_5_codeGenerator.grouping import (  # noqa: E402
    Group, build_shapes, check_degeneration, pool_thin_within_facet,
    repair_partition,
)
from pipeline.step_5_codeGenerator.taxonomy_input import (  # noqa: E402
    build_attribute_refs, build_idea_units,
)

from analysis import (  # noqa: E402
    consensus_ari, histogram, merge_recurrence, pairwise_ari, tau_sweep,
)
from consensus import consensus_partition, dominant_member  # noqa: E402
from stability_bridge import together_from_runs  # noqa: E402
from storage import RunSet, load_runset, save_runset  # noqa: E402

OUT_DIR = SRC.parent / "exports" / "experiment_logs"

# =============================================================================
# INSTELLINGEN — dit is wat een klik op Run doet
# =============================================================================
# Verander deze regels, druk op Run, klaar. Wie liever de commandoregel
# gebruikt: argumenten winnen hiervan, dus de commando's in het verslag blijven
# werken.
#
# Wat de vier acties doen, en wat ze kosten:
#   alles     de hele ronde achter elkaar               2xRUNS + 2 calls  <-- kost geld
#   analyse   lees een opgeslagen set: hoe onstabiel is deel 1?     0 calls
#   compare   twee sets tegen elkaar: reproduceert de consensus?    0 calls
#   codebook  bouw een codeboek uit een set                         1 call
#   collect   draai deel 1 opnieuw, N keer                          N calls  <-- kost geld

ACTIE = "alles"       # alles | analyse | compare | codebook | collect
CONFIG = "luna"         # luna (goedkoop) | gpt54 (12,5x duurder)
SET = 5                 # welke opgeslagen set. luna: 0,1,3,4  |  gpt54: 1,2
SET_B = 6               # bij 'compare' en 'alles': de tweede set
TAU = 0.7               # alleen bij compare/codebook: hoe vaak twee attributen
                        # samen moeten hebben gezeten om te mogen koppelen
RUNS = 30               # alleen bij 'collect': hoeveel keer deel 1 draait
SOURCE = "consensus"    # alleen bij 'codebook': consensus | baseline (1 losse run)
POLES = "two"         # alleen bij 'codebook': three (pos/neu/neg) | two (niet-negatief/neg)


def settings_argv() -> List[str]:
    """De instellingen hierboven als commandoregel.

    Zo loopt een klik op Run door exact dezelfde parser als een aanroep uit de
    terminal — één code-pad, dus de knop kan nooit iets anders doen dan het
    commando dat ernaast in het verslag staat.
    """
    argv = [ACTIE, "--config", CONFIG]
    if ACTIE == "alles":
        argv += ["--runs", str(RUNS), "--set-a", str(SET), "--set-b", str(SET_B),
                 "--tau", str(TAU), "--poles", POLES]
    elif ACTIE == "collect":
        argv += ["--runs", str(RUNS), "--set", str(SET)]
    elif ACTIE == "analyse":
        argv += ["--set", str(SET)]
    elif ACTIE == "compare":
        argv += ["--tau", str(TAU), "--set-a", str(SET), "--set-b", str(SET_B)]
    elif ACTIE == "codebook":
        argv += ["--set", str(SET), "--tau", str(TAU),
                 "--source", SOURCE, "--poles", POLES]
    return argv


def gekozen_argv(argv: List[str]) -> List[str]:
    """Argumenten van de commandoregel, of anders de instellingen hierboven."""
    return argv if argv else settings_argv()

# De twee configuraties die fase 1 tegen elkaar zet. De sleutel van STEP_EFFORT
# is de fase, niet het model, dus de effort wordt tijdelijk omgezet en in een
# `finally` teruggezet.
CONFIGS = {
    "luna": {"model": project_config.MODELS[("5.6", 3)].name, "effort": "medium"},
    "gpt54": {"model": project_config.MODELS[("5.4", 5)].name, "effort": "high"},
}


def load_material():
    """Kaarten en concepten uit de step-4-cache. Zelfde route als
    `run_codebook.py`, alleen zonder iets weg te schrijven."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE)

    metadata = load_extraction_metadata(FILENAME, VARIABLE, SAMPLE_SIZE, variable_key)
    classified = load_classified_ideas(FILENAME, VARIABLE, SAMPLE_SIZE, variable_key)
    taxonomy = load_taxonomy_cache(FILENAME, VARIABLE, SAMPLE_SIZE, variable_key)
    if taxonomy is None:
        raise SystemExit("geen taxonomie in cache — draai eerst step 4")

    refs = build_attribute_refs(taxonomy.partition_results)
    units = [u for u in build_idea_units(classified) if u.attribute_id in refs]
    concepts = build_inventory(units, refs)

    by_attribute: Dict[str, list] = defaultdict(list)
    for unit in units:
        by_attribute[unit.attribute_id].append(unit)

    dimension_name = getattr(metadata, "primary_dimension", "") or ""
    return {
        # Het experiment zet de vangnetuitsluiting AAN; productie niet,
        # tot het promotiebesluit valt.
        "cards": build_cards(concepts, by_attribute, exclude_drains=True),
        "concepts": concepts,
        "question": (getattr(metadata, "var_lab", "") or "").strip(),
        "language": getattr(metadata, "lang", "") or "Dutch",
        "dimension_diagnostic": (get_dimension(dimension_name).criterion
                                 if dimension_name else FALLBACK_DIAGNOSTIC),
        "n_respondents": len({u.respondent_id for u in units}),
        # `t_keep` gebruikt in productie het aantal RESPONSES, niet het aantal
        # respondenten mét een idee. Zelfde basis aanhouden, anders draait het
        # experiment op een andere drempel dan de keten waarmee het vergeleken
        # wordt.
        "n_classified": len(classified),
        # `apply_overig_sweep` en `run_scorecard` werken op de taxonomiestructuur,
        # niet op de concepten.
        "partition_results": taxonomy.partition_results,
    }


async def collect(config_name: str, runs: int, set_index: int, salted: bool = True) -> Path:
    """N keer deel 1, elke run met een eigen salt — of, met `salted=False`,
    N identieke aanroepen die de kale servervariatie blootleggen. Schrijft de
    partities weg."""
    spec = CONFIGS[config_name]
    material = load_material()
    codebook_config = CodebookConfig(model_relations=spec["model"])

    original_effort = project_config.STEP_EFFORT.get("codegen_relations")
    project_config.STEP_EFFORT["codegen_relations"] = spec["effort"]
    try:
        partitions: List[List[Tuple[str, ...]]] = []
        for index in range(runs):
            salt = f"set{set_index}run{index}" if salted else ""
            print(f"  run {index + 1}/{runs}  (salt={salt!r})")
            proposal = await resolve_consolidation(
                material["cards"], material["question"], material["n_respondents"],
                material["language"], codebook_config, salt=salt,
            )
            groups = repair_partition(proposal, material["cards"], material["concepts"])
            partitions.append([tuple(sorted(g.member_ids)) for g in groups])
            print(f"      {len(groups)} groepen")
    finally:
        project_config.STEP_EFFORT["codegen_relations"] = original_effort

    runset = RunSet(
        model=spec["model"],
        effort=spec["effort"],
        attribute_ids=[c.attribute_id for c in material["cards"]],
        attribute_names={c.attribute_id: c.name for c in material["cards"]},
        n_respondents=material["n_respondents"],
        runs=partitions,
        salted=salted,
    )
    path = OUT_DIR / f"consensus_{config_name}_set{set_index}.json"
    save_runset(runset, path)
    print(f"\n{runs} runs weggeschreven naar {path}")
    return path


TAUS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


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


def analyse(config_name: str, set_index: int) -> None:
    """Fase 2 t/m 4 — kost geen enkele LLM-call."""
    path = OUT_DIR / f"consensus_{config_name}_set{set_index}.json"
    runset = load_runset(path)
    n_runs = len(runset.runs)

    print(f"\n{'=' * 78}\n{runset.model} / {runset.effort} — {n_runs} runs, "
          f"{len(runset.attribute_ids)} attributen, salted={runset.salted}"
          f"\n{'=' * 78}")

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


def compare(config_name: str, tau: float, set_a: int = 1, set_b: int = 2) -> None:
    """Fase 5 — de hoofdmaat: ARI tussen twee onafhankelijke consensusindelingen.
    Kost geen enkele LLM-call — beide sets staan al op schijf.

    Welke twee sets is een argument en geen aanname: de sets op schijf zijn niet
    altijd 1 en 2 (de 30-runsmeting draaide op 3 en 4, en luna heeft geen set 2).
    """
    a = load_runset(OUT_DIR / f"consensus_{config_name}_set{set_a}.json")
    b = load_runset(OUT_DIR / f"consensus_{config_name}_set{set_b}.json")

    # `adjusted_rand_index` beperkt zich stil tot de doorsnede van de twee
    # eenhedenverzamelingen. Een step-4-herberekening tussen de twee sets zou de
    # hoofdmaat dan op een deelverzameling berekenen zonder waarschuwing — hier
    # weigeren in plaats van dat risico te lopen.
    if a.attribute_ids != b.attribute_ids:
        raise SystemExit(
            f"de attribuutuniversa van set {set_a} en set {set_b} verschillen — "
            "ARI zou stilzwijgend op de doorsnede berekend worden. Verzamel "
            "beide sets opnieuw tegen dezelfde step-4-cache."
        )

    together_a = together_from_runs(a.runs, a.attribute_ids)
    together_b = together_from_runs(b.runs, b.attribute_ids)
    clusters_a = consensus_partition(together_a, a.attribute_ids, len(a.runs), tau)
    clusters_b = consensus_partition(together_b, b.attribute_ids, len(b.runs), tau)

    print(f"\n{'=' * 78}\nFASE 5 — set {set_a} vs set {set_b}  "
          f"({config_name}, {len(a.runs)}+{len(b.runs)} runs, tau={tau})"
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


def bezette_sets(config_name: str, *indices: int) -> List[int]:
    """Welke van deze setnummers al op schijf staan.

    `collect` overschrijft zonder waarschuwing, en een set is RUNS LLM-calls
    die je niet terugkrijgt. Een ronde die per ongeluk op een bezet nummer
    landt wist dus het materiaal van een eerdere meting.
    """
    return [index for index in indices
            if (OUT_DIR / f"consensus_{config_name}_set{index}.json").exists()]


async def alles(config_name: str, runs: int, set_a: int, set_b: int,
                tau: float, poles: str) -> None:
    """De hele ronde achter elkaar: verzamelen, meten, en twee codeboeken.

    De losse acties bestaan om een ronde te kunnen onderbreken of om een
    opgeslagen set later opnieuw te bevragen. Wie de meting gewoon wil
    uitvoeren hoort dat niet in zes stappen te hoeven doen.

    De twee codeboeken zijn er allebei omdat een codeboek op zichzelf niets
    zegt: de basislijn uit een losse run is waar de consensusversie tegen
    afgezet wordt.
    """
    bezet = bezette_sets(config_name, set_a, set_b)
    if bezet:
        raise SystemExit(
            f"set {' en '.join(map(str, bezet))} bestaat al voor {config_name} "
            f"en zou overschreven worden — dat is {runs} LLM-calls per set die "
            f"je kwijt bent. Kies vrije nummers via SET / SET_B.")

    print(f"\n{'=' * 78}\nVOLLEDIGE RONDE — {config_name}, {runs} runs per set, "
          f"tau={tau}, {poles} polen\n  kosten: {2 * runs} + 2 = "
          f"{2 * runs + 2} LLM-calls\n{'=' * 78}")

    print(f"\n[1/6] set {set_a} verzamelen ({runs} calls)")
    await collect(config_name, runs, set_a)
    print(f"\n[2/6] set {set_b} verzamelen ({runs} calls)")
    await collect(config_name, runs, set_b)
    print(f"\n[3/6] hoe onstabiel is deel 1 (gratis)")
    analyse(config_name, set_a)
    print(f"\n[4/6] de hoofdmaat: reproduceert de consensus (gratis)")
    compare(config_name, tau, set_a, set_b)
    print(f"\n[5/6] basislijncodeboek uit een losse run (1 call)")
    await codebook(config_name, set_a, tau, "baseline", poles)
    print(f"\n[6/6] consensuscodeboek (1 call)")
    await codebook(config_name, set_a, tau, "consensus", poles)

    print(f"\n{'=' * 78}\nRONDE KLAAR — beide codeboeken staan in {OUT_DIR}"
          f"\n{'=' * 78}")


class _Log:
    """Duck-typed log, zoals `_RepairLog` in run_codebook.py."""
    def __init__(self):
        self.entries: List[dict] = []

    def add(self, **kwargs):
        self.entries.append(kwargs)


async def codebook(config_name: str, set_index: int, tau: float,
                   source: str, poles: str) -> None:
    """Fase 6 (basislijn uit één losse run) en fase 7 (consensus), elk in
    driedeling of tweedeling. Schrijft NIETS naar de cache — wel het volledige
    codeboek naar `exports/experiment_logs/`, want een codeboek dat alleen in
    de terminal staat is er geen."""
    path = report_path(config_name, set_index, source, tau, poles)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(_Tee(sys.stdout, handle)):
            await _codebook_body(config_name, set_index, tau, source, poles)
    print(f"\nCodeboek weggeschreven naar {path}")


async def _codebook_body(config_name: str, set_index: int, tau: float,
                         source: str, poles: str) -> None:
    runset = load_runset(OUT_DIR / f"consensus_{config_name}_set{set_index}.json")
    material = load_material()
    codebook_config = CodebookConfig(model_relations=CONFIGS[config_name]["model"])
    two_pole = poles == "two"
    concepts = material["concepts"]
    concept_by_id = {c.attribute_id: c for c in concepts}

    # Vangnetten horen niet in de groepering. `build_cards` sluit ze sinds
    # 2026-08-20 uit, dus een runset die er nog wel op is verzameld moet hier
    # bijgetrokken worden — anders bouwt dit commando een codeboek op een
    # indeling die de productieketen niet meer maakt.
    drains = {c.attribute_id for c in concepts if c.is_drain}
    ids = [i for i in runset.attribute_ids if i not in drains]

    def _schoon(runs):
        return [[g for g in ([tuple(x for x in c if x not in drains) for c in r]) if g]
                for r in runs]

    if source == "consensus":
        together = together_from_runs(_schoon(runset.runs), ids)
        clusters = consensus_partition(together, ids, len(runset.runs), tau)
        label = f"consensus tau={tau}"
    else:
        clusters = _schoon([runset.runs[0]])[0]
        label = "basislijn (run 1)"

    # `proposed_name` vult `CodeShape.umbrella`, de hernoemkandidaat bij een
    # naambotsing. Een consensusgroep heeft er geen, dus: het zwaarste lid.
    weight_by_id = {c.attribute_id: c.n_resp for c in concepts}
    groups = []
    for cluster in clusters:
        known = [m for m in cluster if m in concept_by_id]
        umbrella = (concept_by_id[dominant_member(known, weight_by_id)].name
                    if known else "")
        groups.append(Group(member_ids=tuple(cluster), proposed_name=umbrella,
                            explanation=""))

    degeneration = check_degeneration(len(groups), len(runset.attribute_ids))
    if degeneration:
        print(f"DEGENERATIE: {degeneration}")

    threshold = t_keep(material["n_classified"], codebook_config)
    groups, pool_log = pool_thin_within_facet(groups, concepts, threshold,
                                              two_pole=two_pole)
    for entry in pool_log:
        leden = ", ".join(concept_by_id[m].name for m in entry["members"])
        print(f"  FACETPOOL {entry['facet']}: {leden}  ({entry['n_resp']} resp)")
    shaping = build_shapes(groups, concepts, threshold, two_pole=two_pole)

    veto_log = _Log()
    codes = await write_codebook(
        shaping.shapes, concepts, material["dimension_diagnostic"],
        material["language"], codebook_config, log=veto_log,
    )

    # `codes` kan korter zijn dan `shaping.shapes` (een veto laat een vorm
    # vallen), dus matchen op vorm, nooit zippen — zie code_shape.py.
    lookup = _shape_lookup(shaping.shapes, concept_by_id)
    shapes = [_match_shape(code, lookup) for code in codes]
    codes = resolve_duplicate_names(codes, shapes, log=_Log())

    overig_name = apply_overig_sweep(codes, material["partition_results"],
                                     material["language"])
    print_codebook_results(codes)
    scorecard = run_scorecard(codes, material["partition_results"], overig_name)

    # Alleen een `pooled` vorm is vetobaar (zie `write_codebook`'s docstring) —
    # `solo`/`synonym` niet, want die zijn per definitie nameable. Bij een hogere
    # tau zijn er minder gepoolde vormen en dus minder om te vetoën; het absolute
    # aantal veto's beweegt dan mee met precies de faalvorm die het moet
    # betrappen. De noemer maakt het getal normaliseerbaar over tau's.
    n_pooled = sum(1 for shape in shaping.shapes if shape.origin == "pooled")

    print(f"\n{'=' * 78}\n{label} — {poles}deling — {runset.model}/{runset.effort}"
          f"\n{'=' * 78}")
    print(f"  groepen:          {len(groups)}")
    print(f"  vormen:           {len(shaping.shapes)} (waarvan {n_pooled} gepoold)")
    print(f"  direction_loss:   {shaping.direction_loss}")
    print(f"  codes geschreven: {len(codes)}")
    print(f"  nameable-veto's:  {len(veto_log.entries)} van {n_pooled} gepoolde vormen")
    print(f"  degeneratie:      {degeneration or 'nee'}")
    print(f"  scorecard:        {'PASS' if scorecard.passed else 'FAIL'}")


def build_parser() -> argparse.ArgumentParser:
    """Apart van `main()` zodat de CLI zonder uitvoering te toetsen is."""
    parser = argparse.ArgumentParser(description="consensus over N consolidatieruns")
    sub = parser.add_subparsers(dest="command", required=True)

    alles_parser = sub.add_parser(
        "alles", help="de hele ronde: verzamelen, meten, twee codeboeken")
    alles_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    alles_parser.add_argument("--runs", type=int, default=30)
    alles_parser.add_argument("--set-a", type=int, default=5, dest="set_a")
    alles_parser.add_argument("--set-b", type=int, default=6, dest="set_b")
    alles_parser.add_argument("--tau", type=float, default=0.7)
    alles_parser.add_argument("--poles", choices=("three", "two"), default="three")

    collect_parser = sub.add_parser("collect", help="draai deel 1 N keer")
    collect_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    collect_parser.add_argument("--runs", type=int, default=10)
    collect_parser.add_argument("--set", type=int, default=1, dest="set_index")
    collect_parser.add_argument(
        "--no-salt", action="store_true", dest="no_salt",
        help="draai alle N runs met salt=\"\" — identieke aanroepen. Dat is de "
             "bedoeling, geen bug: de gemeten spreiding is dan kale "
             "servervariatie, ontdaan van volgordegevoeligheid.")

    analyse_parser = sub.add_parser("analyse", help="analyseer opgeslagen runs")
    analyse_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    analyse_parser.add_argument("--set", type=int, default=1, dest="set_index")

    compare_parser = sub.add_parser(
        "compare", help="ARI tussen de consensusindeling van set 1 en set 2")
    compare_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    compare_parser.add_argument("--tau", type=float, default=0.8)
    compare_parser.add_argument("--set-a", type=int, default=1, dest="set_a")
    compare_parser.add_argument("--set-b", type=int, default=2, dest="set_b")

    codebook_parser = sub.add_parser("codebook", help="bouw een codeboek")
    codebook_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    codebook_parser.add_argument("--set", type=int, default=1, dest="set_index")
    codebook_parser.add_argument("--tau", type=float, default=0.8)
    codebook_parser.add_argument("--source", choices=("consensus", "baseline"),
                                 default="consensus")
    codebook_parser.add_argument("--poles", choices=("three", "two"), default="three")

    return parser


def main() -> None:
    args = build_parser().parse_args(gekozen_argv(sys.argv[1:]))
    if args.command == "alles":
        asyncio.run(alles(args.config, args.runs, args.set_a, args.set_b,
                          args.tau, args.poles))
    elif args.command == "collect":
        asyncio.run(collect(args.config, args.runs, args.set_index,
                            salted=not args.no_salt))
    elif args.command == "analyse":
        analyse(args.config, args.set_index)
    elif args.command == "compare":
        compare(args.config, args.tau, args.set_a, args.set_b)
    elif args.command == "codebook":
        asyncio.run(codebook(args.config, args.set_index, args.tau,
                             args.source, args.poles))


if __name__ == "__main__":
    main()
