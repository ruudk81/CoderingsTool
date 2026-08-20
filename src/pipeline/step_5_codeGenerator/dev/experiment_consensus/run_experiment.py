#%%
"""Consensus over N consolidatieruns — het experiment.

Draait deel 1 van step 5 meerdere keren met een wisselende volgordesalt, telt
per attribuutpaar in hoeveel runs de twee samen zaten, en snijdt die matrix
deterministisch tot één indeling.

    python run_experiment.py collect --config luna  --runs 10 --set 1
    python run_experiment.py collect --config gpt54 --runs 10 --set 1
    python run_experiment.py analyse --config luna  --set 1

Raakt de productiecache niet aan: er wordt niets weggeschreven onder
`mece_codes`.
"""
from __future__ import annotations

import argparse
import asyncio
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
    load_classified_ideas, load_extraction_metadata, load_taxonomy_cache,
)
from pipeline.step_5_codeGenerator.code_shape import _match_shape, _shape_lookup  # noqa: E402
from pipeline.step_5_codeGenerator.codebook_io import (  # noqa: E402
    apply_overig_sweep, print_codebook_results, run_scorecard,
)
from pipeline.step_5_codeGenerator.codebook_writer import (  # noqa: E402
    resolve_duplicate_names, write_codebook,
)
from pipeline.step_5_codeGenerator.concept_inventory import build_inventory, t_keep  # noqa: E402
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig  # noqa: E402
from pipeline.step_5_codeGenerator.consolidation import resolve_consolidation  # noqa: E402
from pipeline.step_5_codeGenerator.grouping import (  # noqa: E402
    Group, build_shapes, check_degeneration, repair_partition,
)
from pipeline.step_5_codeGenerator.taxonomy_input import (  # noqa: E402
    build_attribute_refs, build_idea_units,
)

from analysis import histogram, pairwise_ari, tau_sweep  # noqa: E402
from consensus import consensus_partition, dominant_member  # noqa: E402
from stability_bridge import together_from_runs  # noqa: E402
from storage import RunSet, load_runset, save_runset  # noqa: E402

OUT_DIR = SRC.parent / "exports" / "experiment_logs"

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
        "cards": build_cards(concepts, by_attribute),
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


async def collect(config_name: str, runs: int, set_index: int) -> Path:
    """N keer deel 1, elke run met een eigen salt. Schrijft de partities weg."""
    spec = CONFIGS[config_name]
    material = load_material()
    codebook_config = CodebookConfig(model_relations=spec["model"])

    original_effort = project_config.STEP_EFFORT.get("codegen_relations")
    project_config.STEP_EFFORT["codegen_relations"] = spec["effort"]
    try:
        partitions: List[List[Tuple[str, ...]]] = []
        for index in range(runs):
            salt = f"set{set_index}run{index}"
            print(f"  run {index + 1}/{runs}  (salt={salt})")
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
    )
    path = OUT_DIR / f"consensus_{config_name}_set{set_index}.json"
    save_runset(runset, path)
    print(f"\n{runs} runs weggeschreven naar {path}")
    return path


TAUS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def analyse(config_name: str, set_index: int) -> None:
    """Fase 2 t/m 4 — kost geen enkele LLM-call."""
    path = OUT_DIR / f"consensus_{config_name}_set{set_index}.json"
    runset = load_runset(path)
    n_runs = len(runset.runs)

    print(f"\n{'=' * 78}\n{runset.model} / {runset.effort} — {n_runs} runs, "
          f"{len(runset.attribute_ids)} attributen\n{'=' * 78}")

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


class _Log:
    """Duck-typed log, zoals `_RepairLog` in run_codebook.py."""
    def __init__(self):
        self.entries: List[dict] = []

    def add(self, **kwargs):
        self.entries.append(kwargs)


async def codebook(config_name: str, set_index: int, tau: float,
                   source: str, poles: str) -> None:
    """Fase 6 (basislijn uit één losse run) en fase 7 (consensus), elk in
    driedeling of tweedeling. Schrijft NIETS naar de cache."""
    runset = load_runset(OUT_DIR / f"consensus_{config_name}_set{set_index}.json")
    material = load_material()
    codebook_config = CodebookConfig(model_relations=CONFIGS[config_name]["model"])
    two_pole = poles == "two"
    concepts = material["concepts"]
    concept_by_id = {c.attribute_id: c for c in concepts}

    if source == "consensus":
        together = together_from_runs(runset.runs, runset.attribute_ids)
        clusters = consensus_partition(together, runset.attribute_ids,
                                       len(runset.runs), tau)
        label = f"consensus tau={tau}"
    else:
        clusters = runset.runs[0]
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

    print(f"\n{'=' * 78}\n{label} — {poles}deling — {runset.model}/{runset.effort}"
          f"\n{'=' * 78}")
    print(f"  groepen:          {len(groups)}")
    print(f"  vormen:           {len(shaping.shapes)}")
    print(f"  direction_loss:   {shaping.direction_loss}")
    print(f"  codes geschreven: {len(codes)}")
    print(f"  nameable-veto's:  {len(veto_log.entries)}")
    print(f"  degeneratie:      {degeneration or 'nee'}")
    print(f"  scorecard:        {'PASS' if scorecard.passed else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="consensus over N consolidatieruns")
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect", help="draai deel 1 N keer")
    collect_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    collect_parser.add_argument("--runs", type=int, default=10)
    collect_parser.add_argument("--set", type=int, default=1, dest="set_index")

    analyse_parser = sub.add_parser("analyse", help="analyseer opgeslagen runs")
    analyse_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    analyse_parser.add_argument("--set", type=int, default=1, dest="set_index")

    codebook_parser = sub.add_parser("codebook", help="bouw een codeboek")
    codebook_parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    codebook_parser.add_argument("--set", type=int, default=1, dest="set_index")
    codebook_parser.add_argument("--tau", type=float, default=0.8)
    codebook_parser.add_argument("--source", choices=("consensus", "baseline"),
                                 default="consensus")
    codebook_parser.add_argument("--poles", choices=("three", "two"), default="three")

    args = parser.parse_args()
    if args.command == "collect":
        asyncio.run(collect(args.config, args.runs, args.set_index))
    elif args.command == "analyse":
        analyse(args.config, args.set_index)
    elif args.command == "codebook":
        asyncio.run(codebook(args.config, args.set_index, args.tau,
                             args.source, args.poles))


if __name__ == "__main__":
    main()
