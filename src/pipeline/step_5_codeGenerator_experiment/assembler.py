"""Fase 5 — assemblage, beslislog, cache-save en scorecard.

Consumes the outputs of the earlier experiment phases (`ClusterResult` from
phenomenon_clusterer, per-cluster code plans from direction_rules.codes_for,
`CodeNaming` results from judgments, and the accumulated `Decision` log) and
assembles the final `CodingResultsCache`, mirroring the baseline's
`cache_mece_results` construction (run_codeGenerator.py ~L276) so step 6 and
the app can consume the experimental codebook exactly like the production one.

Cache-save is EXCLUSIVELY under step name "mece_codes_exp" — never
"mece_codes"/"taxonomy_codes", so the experiment never collides with or
silently replaces the production codebook cache.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from models import CodingResultsCache, DomainSet
from identity import ensure_codebook_ids
from utils.cacheManager import CacheManager
from config import MISCELLANEOUS_CODE_LABELS
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.codebook_verifier import (
    build_scorecard, format_scorecard, collect_taxonomy_attributes,
)

from pipeline.step_5_codeGenerator_experiment.data_io import ExperimentInputs
from pipeline.step_5_codeGenerator_experiment.phenomenon_clusterer import ClusterResult
from pipeline.step_5_codeGenerator_experiment.judgments import CodeNaming

# src/pipeline/step_5_codeGenerator_experiment/assembler.py -> repo root is 4
# parents up, same depth as the baseline's run_codeGenerator.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

OVERIG_DEFINITION = (
    "Catch-all voor antwoorden die geen specifieke code kregen "
    "(o.a. diffuus of algemeen oordeel zonder concreet onderwerp)."
)
OVERIG_DIAGNOSTIC = "valt buiten alle specifieke codes"


# =============================================================================
# Decision log
# =============================================================================
class Decision(BaseModel):
    """A single beslisrecord in the experiment's decision log.

    `phase` groups records in the rendered narrative (e.g. "clustering",
    "direction", "naming"); `subject` names what was decided about (an
    attribute, a phenomenon cluster, a code); `outcome` is the decision
    itself; `evidence` carries whatever numbers/text back it; `votes` holds
    a raw vote tally when the decision came from `judgments.vote()`;
    `is_borderline` flags records worth a human's second look (routed to the
    grensgevallen export).
    """

    phase: str
    subject: str
    outcome: str
    evidence: dict = Field(default_factory=dict)
    votes: Optional[dict] = None
    is_borderline: bool = False


def render_decision_log(decisions: List[Decision]) -> str:
    """Readable rendering of a decision log, grouped by phase (first-seen order).

    Each record renders as "- subject -> outcome" with its evidence appended
    and a "[borderline]" tag when flagged. Used both for `codebook_narrative`
    (full log) and the grensgevallen export (borderline subset only).
    """
    if not decisions:
        return ""
    phases: Dict[str, List[Decision]] = {}
    for d in decisions:
        phases.setdefault(d.phase, []).append(d)

    lines: List[str] = []
    for phase, records in phases.items():
        lines.append(f"=== {phase} ===")
        for d in records:
            evidence_str = ", ".join(f"{k}={v}" for k, v in (d.evidence or {}).items())
            tag = " [borderline]" if d.is_borderline else ""
            line = f"- {d.subject} -> {d.outcome}{tag}"
            if evidence_str:
                line += f" ({evidence_str})"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# =============================================================================
# Overig — dangling idea-assigned attributes outside the taxonomy
# =============================================================================
def _dangling_attributes(idea_assignments: Dict[str, str], taxonomy_attrs: List[str]) -> List[str]:
    """Names ideas were assigned to that are not in the taxonomy's attribute list.

    First-seen order, deduplicated. May be empty — Overig is always emitted
    regardless (it must exist as a step-6 routing target)."""
    known = set(taxonomy_attrs)
    seen = set()
    out: List[str] = []
    for name in idea_assignments.values():
        if name and name not in known and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _overig_code(language: str, idea_assignments: Dict[str, str], taxonomy_attrs: List[str]) -> ConsolidatedCode:
    label = MISCELLANEOUS_CODE_LABELS.get(language, "Overig")
    return ConsolidatedCode(
        code_name=label,
        definition=OVERIG_DEFINITION,
        diagnostic_test=OVERIG_DIAGNOSTIC,
        valence="neutral",
        typical_indicators=[],
        source_attributes=_dangling_attributes(idea_assignments, taxonomy_attrs),
    )


# =============================================================================
# Assembly
# =============================================================================
def assemble_codebook(
    inputs: ExperimentInputs,
    cluster_result: ClusterResult,
    code_plans: Dict[int, List[dict]],
    namings: Dict[Tuple[int, str], CodeNaming],
    decisions: List[Decision],
    partition_set: DomainSet,
) -> CodingResultsCache:
    """Build the final `CodingResultsCache` from the experiment's phase outputs.

    `code_plans` maps a phenomenon-cluster label (as in `cluster_result.clusters`)
    to the `direction_rules.codes_for()` output for that cluster — a list of
    `{"valence": str, "expected": int}` entries. Every entry for a given
    cluster shares that cluster's full member-attribute list as its
    `source_attributes`: a split pair (positive/negative[/neutral]) is the
    SAME phenomenon coded at different valence, not a different attribute
    set — so a pair sharing sources is a property of this construction, not
    something asserted after the fact.

    `namings` supplies the `CodeNaming` for each (cluster_label, valence)
    pair produced by `code_plans`. `partition_set` is passed through
    unchanged from the taxonomy cache — `ExperimentInputs` deliberately does
    not carry it (data_io only reads `partition_results`).

    Always appends a neutral "Overig" (per-language) code last, with
    `source_attributes` = dangling idea-assigned attribute names (outside the
    taxonomy) — may be empty. `codebook_narrative` renders `decisions` in
    full. Calls `ensure_codebook_ids` before returning so the cache is
    id-bearing on construction, not just at load.
    """
    codes: List[ConsolidatedCode] = []
    for cluster_label in sorted(code_plans):
        members = list(cluster_result.clusters.get(cluster_label, []))
        for entry in code_plans[cluster_label]:
            valence = entry["valence"]
            naming = namings.get((cluster_label, valence))
            if naming is None:
                raise KeyError(
                    f"no CodeNaming for cluster {cluster_label!r} valence {valence!r}"
                )
            codes.append(ConsolidatedCode(
                code_name=naming.code_name,
                definition=naming.definition,
                diagnostic_test=naming.diagnostic_test,
                valence=valence,
                typical_indicators=naming.typical_indicators,
                source_attributes=members,
            ))

    taxonomy_attrs = collect_taxonomy_attributes(inputs.partition_results)
    codes.append(_overig_code(inputs.language, inputs.idea_assignments, taxonomy_attrs))

    cache = CodingResultsCache(
        partition_set=partition_set,
        partition_results=inputs.partition_results,
        label_counts={name: r.n_labels for name, r in inputs.partition_results.items()},
        total_categories=len(codes),
        raw_codes=[c.model_dump() for c in codes],
        codebook_narrative=render_decision_log(decisions),
        idea_embeddings=inputs.idea_embeddings or None,
        embedding_code_source="",
        embedding_model="",
    )
    ensure_codebook_ids(cache)
    return cache


# =============================================================================
# Cache-save + exports
# =============================================================================
def save_experiment(
    cache: CodingResultsCache,
    filename: str,
    variable_key: str,
    decisions: List[Decision],
    project_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Save `cache` under step "mece_codes_exp" and export the decision log.

    Exports land in `exports/codebook/` with the baseline's basename
    convention (`codebook_{filename-stem}_{variable_key}_...`):
    - `..._EXP_decisions.json` — every Decision record.
    - `..._EXP_grensgevallen.txt` — the `is_borderline` subset, same rendering
      as `codebook_narrative`.

    `project_root` defaults to the repo root; tests pass `tmp_path` so no
    real export lands in the project tree. `CacheManager` is imported at
    module scope so tests can monkeypatch `assembler.CacheManager` and avoid
    a real cache write.

    Returns (decisions_path, grensgevallen_path).
    """
    cache_manager = CacheManager()
    cache_manager.save_metadata_to_cache(
        metadata=cache,
        filename=filename,
        step="mece_codes_exp",
        variable_key=variable_key,
    )

    root = project_root if project_root is not None else _PROJECT_ROOT
    export_dir = root / "exports" / "codebook"
    export_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename).stem.replace(" ", "_")
    prefix = f"codebook_{base}_{variable_key}"

    decisions_path = export_dir / f"{prefix}_EXP_decisions.json"
    decisions_path.write_text(
        json.dumps([d.model_dump() for d in decisions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    borderline = [d for d in decisions if d.is_borderline]
    grensgevallen_path = export_dir / f"{prefix}_EXP_grensgevallen.txt"
    grensgevallen_path.write_text(render_decision_log(borderline), encoding="utf-8")

    return decisions_path, grensgevallen_path


# =============================================================================
# Scorecard
# =============================================================================
def run_scorecard_on(cache: CodingResultsCache, partition_results: Dict[str, Any]):
    """Build and print the post-assembly scorecard for `cache`, reusing the
    baseline's `build_scorecard`/`format_scorecard` — no reimplementation.

    The Overig code is always last in `cache.raw_codes` (assemble_codebook's
    invariant), so its name is read positionally rather than re-derived.
    """
    overig_name = cache.raw_codes[-1]["code_name"] if cache.raw_codes else None
    scorecard = build_scorecard(cache.raw_codes, partition_results, overig_name)
    print("\n" + format_scorecard(scorecard))
    return scorecard
