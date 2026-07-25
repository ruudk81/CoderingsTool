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
import re
import sys
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any

# Ensure src/ (utils/, pipeline/, config, models) AND src/app/ (ui_text,
# app_views, …) are importable, wherever this module is run from.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
# SELECTION-PHASE INSPECTION (plan §3.7) — "kiezen is licht, vastleggen is
# zwaar". A bounded read (row_limit rows, ~9ms on a 71MB .sav) feeds the whole
# select page: names, labels, types, datetime filter, slot groups AND preview
# data. Full reads and disk writes are reserved for the commit moment.
# =============================================================================

# Encoding fallback order (mirrors utils/dataLoader.load_sav). With a bounded
# read a failed attempt only parses `rows` rows, so the carousel is harmless.
_SAV_ENCODINGS = ("utf-8", "windows-1252", "iso-8859-1", "cp1252",
                  "iso-8859-15", "windows-1250", None)


@dataclass
class SavInspection:
    """Bounded view of a .sav: variable info + the first `rows` rows of data."""
    variables: Dict[str, Dict[str, Any]]   # {name: {"label": str, "is_string": bool}}
    frame: Any                             # pandas DataFrame (first `rows` rows)

    @property
    def string_vars(self) -> List[str]:
        return [v for v, i in self.variables.items() if i["is_string"]]


def _looks_datetime(series, sample_size: int = 100) -> bool:
    """Mirror of dataLoader._is_datetime_string_column: >80% of a sample parses."""
    import warnings
    import pandas as pd
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.head(min(sample_size, len(non_null)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(sample, errors="coerce")
        return converted.notna().sum() / len(sample) > 0.8
    except Exception:
        return False


def inspect_sav(fname: str, rows: int = 200) -> SavInspection:
    """Read names, labels, types and `rows` rows from data/<fname> — bounded."""
    import pyreadstat
    path = str(PROJECT_ROOT / "data" / fname)
    last_error = None
    for enc in _SAV_ENCODINGS:
        try:
            df, meta = pyreadstat.read_sav(path, row_limit=rows,
                                           apply_value_formats=True, encoding=enc)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Kon {fname} niet lezen: {last_error}")

    variables: Dict[str, Dict[str, Any]] = {}
    for name in meta.column_names:
        is_string = meta.readstat_variable_types.get(name) == "string"
        if is_string and _looks_datetime(df[name]):
            is_string = False
        variables[name] = {"label": meta.column_names_to_labels.get(name) or "",
                           "is_string": is_string}
    return SavInspection(variables=variables, frame=df)


def clean_question(label: str, var_name: str) -> str:
    """The survey question inside an SPSS label: strip the '[...]' prefix and
    the variable's own name, collapse whitespace."""
    q = (label or "")
    if "]" in q:
        q = q[q.rfind("]") + 1:]
    q = q.strip()
    if var_name and q.lower().startswith(var_name.lower()):
        rest = q[len(var_name):]
        # only strip when a separator follows — never eat 'Qd10…' when var is Qd1
        if rest[:1] in ("", " ", ":", "-", ".", "_"):
            q = rest.lstrip(" :-._")
    return " ".join(q.split())


def series_question(insp: SavInspection, cols: List[str]) -> tuple[str, Dict[str, str]]:
    """Same-question gate for a slot series (plan §3.7 merge-integrity rule).

    Returns (question, mismatches): question = the FIRST member's cleaned label
    (inherited by the merged variable); mismatches = {col: cleaned_label} for
    members whose cleaned label differs — non-empty means merging is blocked.
    """
    cleaned = {c: clean_question(insp.variables[c]["label"], c) for c in cols}
    first = cleaned[cols[0]]
    return first, {c: q for c, q in cleaned.items() if q != first}


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
# SCREEN MODEL (app_v2) — the one explicit decision each step page makes
# =============================================================================

class Screen(str, Enum):
    """What a step page shows. app_old had these as emergent per-block gating;
    here it is one explicit decision (see utils/dev/app_development_plan.md §3.2).
    REVIEW is the HITL screen type: designed now, rendered in a later phase."""
    LOCKED = "locked"    # previous step not done — cannot run yet
    RUN = "run"          # ready: explain the step, offer the run button
    OUTPUT = "output"    # done: show evidence (stats, samples, log) + continue
    REVIEW = "review"    # done + editable artifact awaiting human review (Phase D)


def screen_for(step: int, status: Dict[int, bool]) -> Screen:
    """Resolve the screen for a step from the live cache status ({step: done})."""
    if status.get(step, False):
        return Screen.OUTPUT
    prev_done = (step == 0) or status.get(step - 1, False)
    return Screen.RUN if prev_done else Screen.LOCKED


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
        # run_export delivers everything: the results workbook + .sav (exports/
        # coderingen/) and the codebook workbook (exports/codebook/).
        from pipeline.step_7_export.run_export import run_step as r, StepConfig as C
        paths = r(C(filename=f, id_column=idc, var_name=vn, sample_size=ss, var_lab=vl, force_recalc=force_recalc))
        results_xlsx = paths.get("excel")
        cb_path = paths.get("codeboek_workbook")
        n_sav = sum(1 for k in paths if k not in ("excel", "codeboek_workbook"))
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


def load_quality_filtered(spec: DatasetSpec) -> Optional[List[Any]]:
    """Step 2 per-response models (carry quality_filter_code)."""
    from models import QualityFilteredModel
    cm = CacheManager()
    return cm.load_from_cache(spec.filename, "quality_filter", spec.variable_key,
                              QualityFilteredModel)


def load_raw(spec: DatasetSpec) -> Optional[List[Any]]:
    """Step 0 raw responses (before any correction)."""
    from models import ResponseModel
    cm = CacheManager()
    return cm.load_from_cache(spec.filename, "data", spec.variable_key, ResponseModel)


def load_preprocessed(spec: DatasetSpec) -> Optional[List[Any]]:
    """Step 1 spell-checked responses."""
    from models import PreprocessedModel
    cm = CacheManager()
    return cm.load_from_cache(spec.filename, "preprocessed", spec.variable_key,
                              PreprocessedModel)


def load_extracted(spec: DatasetSpec) -> Optional[List[Any]]:
    """Step 3 per-response models with response_ideas (abstraction ladder)."""
    from models import IdeasExtractedModel
    cm = CacheManager()
    return cm.load_from_cache(spec.filename, "extracted_ideas", spec.variable_key,
                              IdeasExtractedModel)


def load_extraction_metadata(spec: DatasetSpec) -> Optional[Any]:
    """Step 3 dataset-level ExtractionMetadata (context lens + domains)."""
    from models import ExtractionMetadata
    cm = CacheManager()
    return cm.load_metadata_from_cache(spec.filename, "extracted_ideas",
                                       spec.variable_key, ExtractionMetadata)


def load_taxonomy(spec: DatasetSpec) -> Optional[Any]:
    """Step 4 taxonomy structure (TaxonomyResultsCache) — feeds taxonomy_health."""
    from models import TaxonomyResultsCache
    cm = CacheManager()
    return cm.load_metadata_from_cache(spec.filename, "taxonomy", spec.variable_key,
                                       TaxonomyResultsCache)


# =============================================================================
# COSTS (Phase B6) — read-only view on the cumulative costs JSON written by
# utils/costTracker.py. Contract: app_development_plan.md §3.6c. The file is
# cumulative per dataset; each step entry carries a `date` so a stale entry
# (from an older run) stays recognizable.
# =============================================================================

# App step number → step key used by the pipeline when recording costs.
# Steps 0/1/7 record no LLM costs and have no key.
STEP_COSTS_KEY: Dict[int, str] = {
    2: "step_2_quality_filter",
    3: "step_3_idea_extraction",
    4: "step_4_taxonomy_classifier",
    5: "step_5_code_generator",
    6: "step_6_code_assigner",
}


def costs_path(spec: DatasetSpec) -> Path:
    stem = Path(spec.filename).stem
    return PROJECT_ROOT / "exports" / "costs" / f"{stem}_{spec.variable_key}_costs.json"


def load_costs(spec: DatasetSpec) -> Optional[Dict[str, Any]]:
    """The dataset's full costs JSON, or None when no costs were recorded yet."""
    import json
    path = costs_path(spec)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def step_costs(spec: DatasetSpec, step: int) -> Optional[Dict[str, Any]]:
    """One step's cost entry ({phases, model_config, total, date}), or None."""
    key = STEP_COSTS_KEY.get(step)
    if not key:
        return None
    data = load_costs(spec)
    return (data or {}).get("steps", {}).get(key)


def find_verbose_log(spec: DatasetSpec, step: int) -> Optional[str]:
    """Latest captured console log for a step, or None."""
    path = VerboseCapture.find_latest_log(spec.filename, spec.variable_key, step)
    return VerboseCapture.load_log_content(path) if path else None


# =============================================================================
# VERBOSE LOG → REPORT (Phase B0) — turn the raw console capture into a
# structured report. Contract: app_development_plan.md §3.6a. The parser is
# LENIENT by design: unknown lines fall through to the body, dividers are
# dropped, and nothing ever raises — every log in exports/verbose_logs/ must
# parse (test_app_v2.py verifies line accounting on all of them).
# =============================================================================

@dataclass
class VerboseSection:
    """One report section: readable body, highlighted summary, collapsed noise."""
    title: str
    body: List[str] = field(default_factory=list)
    summary: List[str] = field(default_factory=list)
    noise: List[str] = field(default_factory=list)


@dataclass
class VerboseReport:
    meta: Dict[str, str]                 # Dataset, Variable, Sample size, times
    sections: List[VerboseSection]

    @property
    def noise_count(self) -> int:
        return sum(len(s.noise) for s in self.sections)


# Telemetry the report collapses under "technical details". This table is the
# contract with utils/verboseReporter.py & utils/smoothRequester.py output —
# when new streaming/rate-limit line types appear there, extend it HERE.
_NOISE_LINE = re.compile(r"""
      \|\ inflight:                       # [step…] 9/82 | inflight:50 | tok:…
    | ^⏱                                  # ⏱ T+0.0s: …
    | ^\[WARM-UP\]
    | ^Workers:\ \d
    | ^RATE\ LIMITING\ SETUP
    | ^Processing\ individual\ tasks\.\.\.
    | ^-\ (RPM|TPM)\ limit:
    | ^-\ Initial\ avg_tokens
    | ^-\ Target\ concurrency:
    | ^-\ Concurrent\ (subroutines|ceiling)
    | ^-\ Rate\ limit\ concurrency:
    | ^-\ Optimal\ by\ Little
    | ^-\ Start:\ (cold|warm)
    | ^-\ System:\ [AB]\b
    | ^-\ Final\ concurrency:
    | ^\ *Final\ concurrency:
""", re.X)

_DIVIDER = re.compile(r"^[=─-]{6,}$")
_META_KEYS = ("Dataset", "Variable", "Sample size", "Run until step",
              "Start time", "End time")
_SUMMARY_HEAD = re.compile(r"^(SUMMARY\b|\[STATS\]|\[SUMMARY\])")


def parse_verbose_log(text: str) -> VerboseReport:
    """Split a captured console log into meta + sections(body/summary/noise)."""
    meta: Dict[str, str] = {}
    sections: List[VerboseSection] = []
    cur = VerboseSection(title="")
    sections.append(cur)
    in_summary = False
    summary_content_seen = False

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        s = line.strip()

        if not s:                                   # blank ends a summary block
            if in_summary and summary_content_seen:
                in_summary = False
            continue
        if _DIVIDER.match(s):                       # pure formatting — drop
            continue

        # File header metadata (Dataset: …) — regardless of position
        head_key = s.split(":", 1)[0]
        if head_key in _META_KEYS and ":" in s and head_key not in meta:
            meta[head_key] = s.split(":", 1)[1].strip()
            continue
        if s == "PIPELINE VERBOSE OUTPUT LOG":
            continue

        # New section
        if s.startswith("[SECTION]"):
            in_summary = False
            cur = VerboseSection(title=s[len("[SECTION]"):].strip())
            sections.append(cur)
            continue

        # Summary block: head starts collection; content until blank line
        if _SUMMARY_HEAD.match(s):
            in_summary = True
            summary_content_seen = False
            cur.summary.append(s)
            continue
        if in_summary:
            summary_content_seen = True
            cur.summary.append(line)                # keep indentation (alignment)
            continue

        # Telemetry noise → collapsed
        if _NOISE_LINE.search(s):
            cur.noise.append(line)
            continue

        # Everything else is body — lenient default
        cur.body.append(line)

    # Drop a completely empty preamble section
    sections = [sec for sec in sections
                if sec.title or sec.body or sec.summary or sec.noise]
    return VerboseReport(meta=meta, sections=sections)


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
