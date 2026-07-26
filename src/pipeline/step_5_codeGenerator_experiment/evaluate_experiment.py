"""Task 7 — evaluation and reproducibility harness for the step-5 experiment.

Deterministic (no LLM calls): reads already-produced caches/exports and
reports on them. Two things live here:

- `dump_run_artifacts` — after one real experiment run, copy its cache
  (`mece_codes_exp`) and decision-log exports into a per-run directory so
  the run survives the NEXT run overwriting the cache.
- `richting_dekking` / `compare` — deterministic scoring of a codebook
  (baseline or a dumped run) against the taxonomy, and a side-by-side
  comparison table across baseline + N runs plus a reproducibility block.

Baseline is read from the `mece_codes` cache (production step 5) —
READ-ONLY, never written by this module. Experiment runs are read from the
per-run directories `dump_run_artifacts` wrote to, never from the live
`mece_codes_exp` cache (which the next run would have already overwritten).
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from config import MISCELLANEOUS_CODE_LABELS
from models import CodingResultsCache
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from pipeline.step_5_codeGenerator.codebook_verifier import (
    build_scorecard, collect_attribute_valence,
)

# src/pipeline/step_5_codeGenerator_experiment/evaluate_experiment.py -> repo
# root is 3 parents up (same depth as assembler.py's _PROJECT_ROOT).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_OVERIG_NAMES = set(MISCELLANEOUS_CODE_LABELS.values())
_VALENCE_BUCKETS = ("positive", "neutral", "negative")


def _field(code: Any, name: str) -> Any:
    """Read a field from a dict or a Pydantic-model-like object."""
    if isinstance(code, dict):
        return code.get(name)
    return getattr(code, name, None)


# =============================================================================
# Run artifacts
# =============================================================================
def dump_run_artifacts(run_dir: Path, filename: str, var_name: str, sample_size) -> None:
    """Dump the CURRENT `mece_codes_exp` cache + its exports into `run_dir`.

    Reads `CodingResultsCache` via `CacheManager.load_metadata_from_cache`
    for (filename, var_name, sample_size) and writes into `run_dir`
    (created if missing):

    - `raw_codes.json`  — `cache.raw_codes` (the list of ConsolidatedCode dicts).
    - `narrative.txt`   — `cache.codebook_narrative`.
    - a copy of the matching `..._EXP_decisions.json` and
      `..._EXP_grensgevallen.txt` exports from `exports/codebook/`, using
      the same basename convention as `assembler.save_experiment`
      (`codebook_{filename-stem}_{variable_key}_EXP_...`). Copied only if
      present — a run that produced no decisions still dumps the cache.

    Must be called right after the matching `run_experiment` call and
    before the NEXT run starts (the next run's `save_experiment` overwrites
    both the `mece_codes_exp` cache and these exports in place).

    Raises `RuntimeError` if no `mece_codes_exp` cache is found.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    vk = generate_enhanced_variable_key(
        selected_variables=[var_name], is_merged=False, sample_size=sample_size)
    cm = CacheManager()
    cache = cm.load_metadata_from_cache(filename, "mece_codes_exp", vk, CodingResultsCache)
    if cache is None:
        raise RuntimeError(
            f"no mece_codes_exp cache for {filename!r}/{var_name!r} "
            f"sample={sample_size!r} — run the experiment before dumping artifacts"
        )

    (run_dir / "raw_codes.json").write_text(
        json.dumps(cache.raw_codes, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (run_dir / "narrative.txt").write_text(cache.codebook_narrative or "", encoding="utf-8")

    export_dir = _PROJECT_ROOT / "exports" / "codebook"
    base = Path(filename).stem.replace(" ", "_")
    prefix = f"codebook_{base}_{vk}"
    for suffix in ("_EXP_decisions.json", "_EXP_grensgevallen.txt"):
        src = export_dir / f"{prefix}{suffix}"
        if src.exists():
            shutil.copy2(src, run_dir / src.name)


def _load_run_codes(run_dir: Path) -> List[dict]:
    """Load `raw_codes.json` written by `dump_run_artifacts` for one run."""
    return json.loads((Path(run_dir) / "raw_codes.json").read_text(encoding="utf-8"))


# =============================================================================
# richting_dekking
# =============================================================================
def richting_dekking(codes: list, partition_results) -> float:
    """Share of substantial valence poles that have their own code, per phenomenon.

    A "fenomeen" (phenomenon) is the exact `source_attributes` set shared by
    a group of codes — this is precisely how a direction split is modelled
    by `assembler.assemble_codebook`: a split pair shares the SAME source
    attributes at different valence. Overig is excluded (it is the
    catch-all, not a phenomenon).

    For each fenomeen, the three valence buckets (positive/neutral/negative)
    are summed from `collect_attribute_valence(partition_results)` across
    every attribute in the group. A pole is SUBSTANTIAL when its count is
    >= max(2, int(log(fenomeen_total))) — the same population-scaled floor
    codebook_verifier's under-split/mini-code gates use, applied here to the
    fenomeen's own total rather than the global idea total.

    dekking = (number of substantial poles that have >= 1 code of exactly
    that valence in the same fenomeen group) / (total number of substantial
    poles), aggregated (summed, not averaged) across all fenomena. A
    fenomeen with no substantial poles (e.g. too little data) contributes
    nothing to numerator or denominator. If NO fenomeen anywhere has a
    substantial pole, dekking is vacuously 1.0.
    """
    attr_valence = collect_attribute_valence(partition_results)

    groups: Dict[frozenset, List[Any]] = defaultdict(list)
    for code in codes:
        name = _field(code, "code_name") or ""
        if name in _OVERIG_NAMES:
            continue
        sources = _field(code, "source_attributes") or []
        groups[frozenset(sources)].append(code)

    substantial_total = 0
    covered_total = 0
    for sources, group_codes in groups.items():
        counts = {b: 0 for b in _VALENCE_BUCKETS}
        for attr in sources:
            c = attr_valence.get(attr, {})
            for b in _VALENCE_BUCKETS:
                counts[b] += c.get(b, 0)
        total = sum(counts.values())
        if total <= 0:
            continue
        floor = max(2, int(math.log(total)))
        substantial = [b for b in _VALENCE_BUCKETS if counts[b] >= floor]
        if not substantial:
            continue

        present_valences = {_field(code, "valence") for code in group_codes}
        substantial_total += len(substantial)
        covered_total += sum(1 for b in substantial if b in present_valences)

    if substantial_total == 0:
        return 1.0
    return covered_total / substantial_total


# =============================================================================
# Reproducibility — Jaccard overlap of code-name sets
# =============================================================================
def jaccard(names_a: set, names_b: set) -> float:
    """Case-insensitive Jaccard overlap of two code-name sets.

    Empty/empty is defined as 1.0 (nothing differs)."""
    a = {n.strip().lower() for n in names_a}
    b = {n.strip().lower() for n in names_b}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _non_overig_names(codes: List[dict]) -> set:
    return {_field(c, "code_name") or "" for c in codes if (_field(c, "code_name") or "") not in _OVERIG_NAMES}


# =============================================================================
# compare — report table + reproducibility block
# =============================================================================
def compare(
    baseline_cache: CodingResultsCache,
    exp_run_dirs: List[Path],
    partition_results: Dict[str, Any],
) -> str:
    """Readable comparison table: baseline + each experiment run, one column each.

    Rows: code count, under-split advisories (`build_scorecard`), mini-codes,
    Overig share, `richting_dekking`. `partition_results` is used for every
    column — baseline and all runs are evaluated against the SAME taxonomy
    (the point of the experiment is a different codebook over the same
    step-4 structure, not a different taxonomy).

    Baseline codes come from `baseline_cache.raw_codes` (caller loads this
    from the `mece_codes` cache — read-only, this function never touches
    the cache). Each run's codes come from `raw_codes.json` in its
    `exp_run_dirs` entry, as written by `dump_run_artifacts`.

    Appends a reproducibility block over the experiment runs only (baseline
    excluded, it is not a repeated run): the spread (min/max/range) of code
    counts, and the pairwise case-insensitive Jaccard overlap of code-name
    sets (Overig excluded from the sets — it is constant by construction
    and would inflate overlap without signalling reproducibility).
    """
    labels = ["baseline"] + [f"run{i + 1}" for i in range(len(exp_run_dirs))]
    all_codes: List[List[dict]] = [baseline_cache.raw_codes] + [
        _load_run_codes(d) for d in exp_run_dirs
    ]

    metrics: Dict[str, Dict[str, Any]] = {}
    for label, codes in zip(labels, all_codes):
        overig_name = _field(codes[-1], "code_name") if codes else None
        sc = build_scorecard(codes, partition_results, overig_name)
        metrics[label] = {
            "codes": len(codes),
            "under_split": len(sc.under_split_codes),
            "mini_codes": len(sc.mini_codes),
            "overig_share": sc.overig_idea_share_pct,
            "richting_dekking": round(richting_dekking(codes, partition_results), 3),
        }

    rows = [
        ("Codes", "codes"),
        ("Onder-split adviezen", "under_split"),
        ("Mini-codes", "mini_codes"),
        ("Overig-share (%)", "overig_share"),
        ("Richting-dekking", "richting_dekking"),
    ]
    label_col = 24
    col_width = max(10, max(len(l) for l in labels) + 2)

    lines = ["Metric".ljust(label_col) + "".join(l.rjust(col_width) for l in labels)]
    lines.append("-" * (label_col + col_width * len(labels)))
    for title, key in rows:
        lines.append(
            title.ljust(label_col) + "".join(str(metrics[l][key]).rjust(col_width) for l in labels)
        )

    run_labels = labels[1:]
    run_codes = all_codes[1:]
    lines.append("")
    lines.append("Reproduceerbaarheid (experimentruns onderling):")
    if not run_labels:
        lines.append("  (geen experimentruns opgegeven)")
        return "\n".join(lines)

    run_counts = [metrics[l]["codes"] for l in run_labels]
    lines.append(
        f"  codeaantal spreiding: min={min(run_counts)} max={max(run_counts)} "
        f"range={max(run_counts) - min(run_counts)}"
    )

    name_sets = [_non_overig_names(c) for c in run_codes]
    if len(name_sets) < 2:
        lines.append("  Jaccard: n.v.t. (minder dan 2 runs)")
    else:
        for i in range(len(name_sets)):
            for j in range(i + 1, len(name_sets)):
                jac = jaccard(name_sets[i], name_sets[j])
                lines.append(f"  Jaccard({run_labels[i]}, {run_labels[j]}) = {jac:.3f}")

    return "\n".join(lines)


# =============================================================================
# CLI — python -m pipeline.step_5_codeGenerator_experiment.evaluate_experiment run_dir1 ...
# =============================================================================
if __name__ == "__main__":
    from test_data import TEST_DATA

    run_dirs = [Path(a) for a in sys.argv[1:]]
    if not run_dirs:
        raise SystemExit(
            "usage: python -m pipeline.step_5_codeGenerator_experiment.evaluate_experiment "
            "run_dir1 [run_dir2 ...]"
        )

    _vk = generate_enhanced_variable_key(
        selected_variables=[TEST_DATA.var_name], is_merged=False, sample_size=TEST_DATA.sample_size)
    _cm = CacheManager()
    _baseline = _cm.load_metadata_from_cache(TEST_DATA.filename, "mece_codes", _vk, CodingResultsCache)
    if _baseline is None:
        raise SystemExit("no baseline mece_codes cache found — restore it before comparing")

    print(compare(_baseline, run_dirs, _baseline.partition_results))
