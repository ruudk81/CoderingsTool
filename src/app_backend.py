"""
app_backend.py — Non-UI backend for the CoderingsTool Streamlit app.

Design intent (carried over from the old app, see app_old.py):
    The app is a thin orchestrator over a cache-backed pipeline; the
    CacheManager is the source of truth. The UI never owns the canonical
    results — it asks the cache. This module owns everything that is NOT
    Streamlit, so it stays importable and testable on its own:

      - DatasetSpec      : the identity of a run (filename, var, sample) -> variable_key
      - list_cached_datasets() : resumable datasets, read straight from the cache DB
      - step_status() / max_completed_step() : which step is done / can be (re)run
      - run_step()       : dispatch a step to its pipeline runner (+ verbose capture)
      - invalidate_from() : cascade invalidation (force-recalc from step N onwards)
      - load helpers + export_path() : feed the results view

The cache IS the state: "done" is probed live from the cache each call, exactly
the way each runner probes it before deciding to skip. There is no parallel
"completed steps" bookkeeping to drift out of sync.
"""

from __future__ import annotations

import os
import sys
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any

# Ensure src/ is importable (so `streamlit run app.py` resolves utils/, pipeline/, ...)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from test_data import TEST_DATA
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.saveVerbose import VerboseCapture

PROJECT_ROOT = Path(_SRC_DIR).parent

# =============================================================================
# STEP DEFINITIONS — single source of truth for the 8-step pipeline (0..7)
# =============================================================================

LAST_STEP = 7  # step 7 = Export

# Neutral English labels (the UI overlays bilingual labels via ui_text.STEP_NAMES)
STEP_LABELS = {
    0: "Upload / Select data",
    1: "Preprocessing",
    2: "Quality Filter",
    3: "Idea Extraction",
    4: "Taxonomy",
    5: "Codebook",
    6: "Code Assignment",
    7: "Export",
}

# Single source of truth for each step's cache contract. Both is_step_done (probe)
# and invalidate_from (cascade) derive from THIS one table, so they cannot drift
# apart (the old bug class). Each entry is (base_step_name, is_metadata); the
# on-disk DB step_name for a metadata entry is f"{base}_metadata" (cacheManager
# convention). Step 7 has no cache row — "done" = the export file exists.
STEP_CACHE: Dict[int, tuple] = {
    0: (("data", False),),
    1: (("preprocessed", False),),
    2: (("quality_filter", False),),
    3: (("extracted_ideas", False), ("extracted_ideas", True)),
    4: (("taxonomy", True), ("taxonomy_classified", False)),
    5: (("mece_codes", True),),
    6: (("taxonomy_codes", False),),
    7: (),
}


def _db_step_name(base: str, is_metadata: bool) -> str:
    return f"{base}_metadata" if is_metadata else base


# =============================================================================
# DATASET IDENTITY
# =============================================================================

@dataclass
class DatasetSpec:
    """Identity of a pipeline run. variable_key is derived exactly as the runners do."""
    filename: str
    var_name: str
    sample_size: Optional[int]
    id_column: str = TEST_DATA.id_column   # only needed if step 0 reloads from SPSS
    var_lab: Optional[str] = None          # survey question (cached or from SPSS)

    @property
    def variable_key(self) -> str:
        return generate_enhanced_variable_key([self.var_name], is_merged=False,
                                              sample_size=self.sample_size)

    @property
    def display_name(self) -> str:
        stem = Path(self.filename).stem
        size = self.sample_size if self.sample_size is not None else "full"
        return f"{stem} · {self.var_name} · {size}"


def _split_variable_key(variable_key: str) -> tuple[str, Optional[int]]:
    """'Qd1_combined_2000' -> ('Qd1_combined', 2000); 'Q18_full' -> ('Q18', None)."""
    head, _, tail = variable_key.rpartition("_")
    if not head:
        return variable_key, None
    if tail.isdigit():
        return head, int(tail)
    if tail == "full":
        return head, None
    return variable_key, None


# =============================================================================
# DATASET DISCOVERY (read-only, straight from the cache DB)
# =============================================================================

def _db_path() -> Path:
    return CacheConfig().db_path


def list_cached_datasets() -> List[DatasetSpec]:
    """Every resumable dataset = a valid 'data' row in cache_metadata.

    Read-only SQL (no CacheManager construction) so listing has zero side effects.
    var_lab is pulled from the 'preprocessed' row when present (it carries the
    survey question).
    """
    db = _db_path()
    if not db.exists():
        return []

    specs: List[DatasetSpec] = []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        data_rows = conn.execute(
            "SELECT DISTINCT filename, variable_key FROM cache_metadata "
            "WHERE step_name = 'data' AND status = 'valid' ORDER BY filename, variable_key"
        ).fetchall()
        for row in data_rows:
            filename, variable_key = row["filename"], row["variable_key"]
            var_name, sample_size = _split_variable_key(variable_key)
            lab_row = conn.execute(
                "SELECT var_lab FROM cache_metadata "
                "WHERE filename = ? AND variable_key = ? AND var_lab IS NOT NULL "
                "AND status = 'valid' LIMIT 1",
                (filename, variable_key),
            ).fetchone()
            specs.append(DatasetSpec(
                filename=filename,
                var_name=var_name,
                sample_size=sample_size,
                var_lab=lab_row["var_lab"] if lab_row else None,
            ))
    finally:
        conn.close()
    return specs


# =============================================================================
# STEP STATUS — the cache IS the state
# =============================================================================

def is_step_done(step: int, spec: DatasetSpec, cm: CacheManager) -> bool:
    """Probe the cache the same way each runner does before skipping. Derives the
    cache step-names from STEP_CACHE so it can't drift from invalidate_from."""
    if step == LAST_STEP:
        return export_path(spec).exists()
    entries = STEP_CACHE.get(step, ())
    if not entries:
        return False
    f, vk = spec.filename, spec.variable_key
    return all(
        (cm.is_metadata_cache_valid(f, base, vk) if is_meta
         else cm.is_cache_valid(f, base, vk))
        for base, is_meta in entries
    )


def step_status(spec: DatasetSpec, cm: CacheManager) -> Dict[int, bool]:
    """{step: done} for steps 0..LAST_STEP."""
    return {s: is_step_done(s, spec, cm) for s in range(LAST_STEP + 1)}


def max_completed_step(spec: DatasetSpec, cm: CacheManager) -> int:
    """Highest step with a valid cached result (-1 if nothing cached)."""
    done = [s for s, ok in step_status(spec, cm).items() if ok]
    return max(done) if done else -1


# =============================================================================
# CASCADE INVALIDATION (force-recalc from step N onwards)
# =============================================================================

def invalidate_from(step: int, spec: DatasetSpec, cm: CacheManager) -> List[str]:
    """Mark step..LAST_STEP caches invalid + delete the export file.

    Output of step N depends on every upstream step, so re-running N must
    invalidate N..end. Returns the DB step_names that were invalidated.
    """
    invalidated: List[str] = []
    for s in range(step, LAST_STEP + 1):
        for base, is_meta in STEP_CACHE.get(s, ()):
            name = _db_step_name(base, is_meta)
            cm.db.invalidate_cache(spec.filename, name, spec.variable_key)
            invalidated.append(name)
    # Step 7 (export) is a file on disk, not a cache row
    xlsx = export_path(spec)
    if xlsx.exists():
        try:
            xlsx.unlink()
            invalidated.append(xlsx.name)
        except OSError:
            pass
    return invalidated


# =============================================================================
# RUNNING A STEP — dispatch to the pipeline runner, with verbose capture
# =============================================================================

def _verbose(spec: DatasetSpec, step: int) -> VerboseCapture:
    return VerboseCapture(
        filename=spec.filename,
        variable_key=spec.variable_key,
        sample_size=spec.sample_size,
        run_until_step=step,
        append_mode=True,
    )


def run_step(step: int, spec: DatasetSpec, force_recalc: bool = False) -> str:
    """Run one pipeline step and return a short human-readable summary.

    Steps 0-3,7 take a StepConfig; steps 4-6 take explicit dataset params.
    All console output is tee'd to exports/verbose_logs/ via VerboseCapture.
    """
    with _verbose(spec, step):
        summary = _dispatch(step, spec, force_recalc)
    # Safety net: a successful run MUST leave a probeable result under THIS
    # dataset's keys. If not, the runner wrote somewhere else (e.g. the old
    # default-arg capture bug) — surface it loudly instead of silently passing.
    if not is_step_done(step, spec, CacheManager()):
        summary += (" ⚠️ WAARSCHUWING: geen cache-resultaat gevonden onder "
                    f"{spec.filename}:{spec.variable_key} — runner schreef mogelijk "
                    "naar de verkeerde dataset-key.")
    return summary


@dataclass
class StepResult:
    """One step's outcome during a full run."""
    step: int
    ok: bool
    summary: str


def run_all_steps(spec: DatasetSpec, cm: CacheManager,
                  first_step: int = 1, last_step: int = LAST_STEP):
    """Force-recompute steps first_step..last_step sequentially.

    Generator: yields a StepResult after each step so the caller (UI) can stream
    progress between blocking steps. Stops on the first failure (exception OR the
    safety-guard warning string). Cascade-invalidates first_step..LAST_STEP ONCE
    up front, so a mid-run failure leaves every downstream step correctly
    "not done" in the cache.

    Each step runs via run_step() — same per-step VerboseCapture log + cache save
    as a single-step run. The runs are strictly sequential and blocking, so no
    step's verbose output can interleave with the next; sys.stdout.flush() drains
    the buffer between steps as belt-and-suspenders.
    """
    invalidate_from(first_step, spec, cm)
    for step in range(first_step, last_step + 1):
        try:
            summary = run_step(step, spec, force_recalc=True)
            sys.stdout.flush()
            ok = "⚠️ WAARSCHUWING" not in summary
            yield StepResult(step, ok, summary)
            if not ok:
                return
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the loop
            sys.stdout.flush()
            yield StepResult(step, False, f"__ERROR__ {exc}")
            return


def _dispatch(step: int, spec: DatasetSpec, force_recalc: bool) -> str:
    f, idc, vn, ss, vl = (spec.filename, spec.id_column, spec.var_name,
                          spec.sample_size, spec.var_lab)

    if step == 0:
        from pipeline.step_0_dataLoader.run_dataLoader import run_step as r, StepConfig as C
        data = r(C(filename=f, id_column=idc, var_name=vn, sample_size=ss, force_recalc=force_recalc))
        return f"{len(data)} responses loaded"

    if step == 1:
        from pipeline.step_1_preProcessor.run_preProcessor import run_step as r, StepConfig as C
        data = r(C(filename=f, id_column=idc, var_name=vn, sample_size=ss, var_lab=vl, force_recalc=force_recalc))
        return f"{len(data)} responses preprocessed"

    if step == 2:
        from pipeline.step_2_qualityFilter.run_qualityFilter import run_step as r, StepConfig as C
        data = r(C(filename=f, id_column=idc, var_name=vn, sample_size=ss, force_recalc=force_recalc))
        kept = sum(1 for d in data if getattr(d, "quality_filter", None) is False)
        return f"{kept} of {len(data)} responses kept (meaningful)"

    if step == 3:
        from pipeline.step_3_ideaExtractor.run_ideaExtractor import run_step as r, StepConfig as C
        ideas, _, _ = r(C(filename=f, id_column=idc, var_name=vn, sample_size=ss, var_lab=vl, force_recalc=force_recalc))
        total = sum(len(m.response_ideas) for m in ideas
                    if getattr(m, "response_ideas", None))
        return f"{total} ideas extracted from {len(ideas)} responses"

    if step == 4:
        from pipeline.step_4_classifier.run_classifier import run_taxonomy
        run_taxonomy(filename=f, var_name=vn, sample_size=ss, force_recalc=force_recalc)
        return "Taxonomy built (facets, attributes, valence)"

    if step == 5:
        from pipeline.step_5_codeGenerator.run_codeGenerator import run_codebook
        run_codebook(filename=f, var_name=vn, sample_size=ss, force_recalc=force_recalc)
        codes = load_codebook(spec)
        n = len(codes.raw_codes) if codes else 0
        return f"Codebook generated ({n} codes)"

    if step == 6:
        from pipeline.step_6_codeAssigner.run_codeAssigner import run_assignment
        run_assignment(filename=f, var_name=vn, sample_size=ss, force_recalc=force_recalc)
        return "Codes assigned to ideas"

    if step == 7:
        # Two exports at the Export step: (1) the results workbook + .sav (run_export),
        # (2) the codebook/taxonomy readouts CSV+XLSX (export_codebook, from step 6 data).
        from pipeline.step_7_export.run_export import run_step as r, StepConfig as C
        paths = r(C(filename=f, id_column=idc, var_name=vn, sample_size=ss, var_lab=vl, force_recalc=force_recalc))
        from pipeline.step_6_codeAssigner.view_codebook import export_codebook
        cb_path = export_codebook(filename=f, var_name=vn, sample_size=ss)
        # run_export returns a dict {"excel": ..., <sav suffixes>...}; the codebook
        # export returns its xlsx path. Summarize both deliverables.
        results_xlsx = paths.get("excel") if isinstance(paths, dict) else paths
        n_sav = sum(1 for k in paths if k != "excel") if isinstance(paths, dict) else 0
        bits = []
        if results_xlsx:
            bits.append(f"results: {Path(results_xlsx).name} (+{n_sav} .sav)")
        if cb_path:
            bits.append(f"codebook: {Path(cb_path).name}")
        return "Exported — " + (", ".join(bits) if bits else "complete")

    raise ValueError(f"Unknown step {step}")


# =============================================================================
# LOADERS FOR THE RESULTS VIEW
# =============================================================================

def export_path(spec: DatasetSpec) -> Path:
    """Canonical results-workbook path — imported from resultsExporter so it can't
    drift from where step 7 actually writes (the old export_path bug class)."""
    from pipeline.step_7_export.resultsExporter import results_xlsx_path
    return results_xlsx_path(spec.filename, spec.var_name)


def codebook_path(spec: DatasetSpec) -> Path:
    """Canonical codebook-workbook path — imported from view_codebook (single source)."""
    from pipeline.step_6_codeAssigner.view_codebook import codebook_xlsx_path
    return codebook_xlsx_path(spec.filename, spec.var_name, spec.sample_size)


def load_codebook(spec: DatasetSpec) -> Optional[Any]:
    """Step 5 codebook (CodingResultsCache) — has .raw_codes."""
    from models import CodingResultsCache
    cm = CacheManager()
    return cm.load_metadata_from_cache(spec.filename, "mece_codes", spec.variable_key,
                                       CodingResultsCache)


def load_assignments(spec: DatasetSpec) -> Optional[List[Any]]:
    """Step 6 per-response code-assigned models."""
    from models import CodeAssignedModel
    cm = CacheManager()
    return cm.load_from_cache(spec.filename, "taxonomy_codes", spec.variable_key,
                              CodeAssignedModel)


def find_verbose_log(spec: DatasetSpec, step: int) -> Optional[str]:
    """Latest captured console log for a step, or None."""
    path = VerboseCapture.find_latest_log(spec.filename, spec.variable_key, step)
    return VerboseCapture.load_log_content(path) if path else None


# Quick self-test: `python app_backend.py`
if __name__ == "__main__":
    cm = CacheManager()
    specs = list_cached_datasets()
    print(f"Cached datasets: {len(specs)}")
    for sp in specs:
        status = step_status(sp, cm)
        done = "".join(str(s) if ok else "·" for s, ok in status.items())
        print(f"  {sp.display_name}")
        print(f"    steps done [0-7]: {done}   max={max_completed_step(sp, cm)}")
        print(f"    export: {'yes' if export_path(sp).exists() else 'no'}")
