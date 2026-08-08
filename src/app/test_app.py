"""
test_app.py — verification for the Streamlit app.

Covers: every step shows the right screen for every cache state; sticky errors
survive a rerun. Pure logic (screen_for, registry) is unit-tested exhaustively;
the page layer is smoke-tested with streamlit.testing.v1.AppTest against the
real cache (dataset-dependent tests skip cleanly when the cache is empty).

Run:  cd src && python app/test_app.py     (or: pytest app/test_app.py)
NOTE: tests never click a 🚀 run button — that would start a real LLM run.
"""

import glob
import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import app_backend as be
from app_backend import Screen, screen_for, LAST_STEP
from utils import concatOpenEnds as co

APP = os.path.join(_APP_DIR, "app.py")


# =============================================================================
# Unit: screen_for — exhaustive over the state space that matters
# =============================================================================

def test_screen_for_output_when_done():
    for s in range(LAST_STEP + 1):
        status = {i: True for i in range(LAST_STEP + 1)}
        assert screen_for(s, status) is Screen.OUTPUT

def test_screen_for_run_when_prev_done():
    # exactly steps 0..k done → step k+1 is RUN, k+2.. are LOCKED
    for k in range(-1, LAST_STEP):
        status = {i: i <= k for i in range(LAST_STEP + 1)}
        nxt = k + 1
        assert screen_for(nxt, status) is Screen.RUN, f"step {nxt} after done-through-{k}"
        for s in range(nxt + 1, LAST_STEP + 1):
            assert screen_for(s, status) is Screen.LOCKED, f"step {s} after done-through-{k}"

def test_screen_for_step0_never_locked():
    assert screen_for(0, {i: False for i in range(LAST_STEP + 1)}) is Screen.RUN

def test_screen_for_hole_in_the_middle():
    # done={0,1,3}: step 2 runnable, step 3 shows its (stale-input) output,
    # step 4 runnable on top of 3 — mirrors is_step_done semantics exactly.
    status = {0: True, 1: True, 2: False, 3: True, 4: False, 5: False, 6: False, 7: False}
    assert screen_for(2, status) is Screen.RUN
    assert screen_for(3, status) is Screen.OUTPUT
    assert screen_for(4, status) is Screen.RUN
    assert screen_for(5, status) is Screen.LOCKED


# =============================================================================
# Unit: the view registry
# =============================================================================

def test_registry_covers_all_steps():
    import app_views as av
    assert set(av.STEP_VIEWS.keys()) == set(range(LAST_STEP + 1))

def test_registry_phases_exist_in_config():
    import app_views as av
    from config import STEP_MODEL
    for step, view in av.STEP_VIEWS.items():
        for phase in view.phases:
            assert phase in STEP_MODEL, f"step {step}: unknown phase {phase!r}"

def test_models_line_resolves_for_all_steps():
    import app_views as av
    for step in range(LAST_STEP + 1):
        line = av.models_line(step)
        if av.STEP_VIEWS[step].phases:
            assert line, f"step {step} has phases but no model line"
        else:
            assert line is None


# =============================================================================
# Unit: variable merging (upload pre-step, utils/concatOpenEnds.py)
# =============================================================================

def test_concat_slot_group_detection():
    cols = ["resp_id", "xQd1_1", "xQd1_2", "xQd1_10", "Q5", "opm_1"]
    groups = co.find_slot_groups(cols)
    # Numeric order (2 before 10), singles like opm_1 excluded
    assert groups == {"xQd1_": ["xQd1_1", "xQd1_2", "xQd1_10"]}


def test_concat_combine_row_skips_empty():
    import pandas as pd
    assert co.combine_row(["goed", "", None, " duur "], ", ") == "goed, duur"
    assert co.combine_row(["", None], ", ") is pd.NA


def test_concat_variables_roundtrip(tmp_dir=None):
    """End-to-end on a tiny synthetic .sav in a temp dir (never touches data/)."""
    import tempfile
    import pandas as pd
    import pyreadstat
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "mini.sav")
        pyreadstat.write_sav(pd.DataFrame({
            "id": [1.0, 2.0, 3.0],
            "xQ1_1": ["goed", "", "duur"],
            "xQ1_2": ["snel", "", ""],
        }), src)
        res = co.concat_variables(src, "Q1", prefix="xQ1_", sep=", ")
        assert res["columns"] == ["xQ1_1", "xQ1_2"]
        assert res["rows"] == 3 and res["filled"] == 2
        chk, _ = pyreadstat.read_sav(res["outfile"])
        assert list(chk["Q1"].fillna("")) == ["goed, snel", "", "duur"]
        # Existing variable name must refuse, not overwrite
        try:
            co.concat_variables(src, "xQ1_1", prefix="xQ1_")
            assert False, "expected ValueError for existing variable"
        except ValueError:
            pass


# =============================================================================
# Unit: formatting vs real correction (step-1 view)
# =============================================================================

def test_formatting_key_separates_layout_from_correction():
    import app_views as av
    # Maintainer definition: capitals, punctuation AND whitespace are layout
    assert av._formatting_key("ik vind het mooi") == av._formatting_key("Ik vind het mooi.")
    assert av._formatting_key("goed,  betrouwbaar") == av._formatting_key("Goed betrouwbaar.")
    assert av._formatting_key("Nvt") == av._formatting_key("N. v. t.")
    # (consequence, accepted: word joins like 'spaar bank'→'spaarbank' count as layout)
    assert av._formatting_key("spaar bank") == av._formatting_key("spaarbank")
    # A changed word is a real correction
    assert av._formatting_key("betrouwbar") != av._formatting_key("betrouwbaar")


# =============================================================================
# Unit: two-phase selection (plan §3.7) — inspection, question cleaning, gate
# =============================================================================

def test_clean_question_strips_prefix_and_own_name():
    assert be.clean_question("[xQd1_1] xQd1_1 Wat vind je?", "xQd1_1") == "Wat vind je?"
    assert be.clean_question("Qd1: Wat is je eerste gevoel", "Qd1") == "Wat is je eerste gevoel"
    # never eat 'Qd10…' when the variable is Qd1
    assert be.clean_question("Qd10 iets anders", "Qd1") == "Qd10 iets anders"
    assert be.clean_question("", "Q1") == ""


def test_series_question_gate():
    insp = be.SavInspection(variables={
        "xQ1_1": {"label": "xQ1_1 Wat vind je?", "is_string": True},
        "xQ1_2": {"label": "xQ1_2 Wat  vind je?", "is_string": True},   # whitespace-insensitive
        "xQ1_3": {"label": "xQ1_3 Iets heel anders", "is_string": True},
    }, frame=None)
    q, mm = be.series_question(insp, ["xQ1_1", "xQ1_2"])
    assert q == "Wat vind je?" and not mm, "same question must pass the gate"
    q, mm = be.series_question(insp, ["xQ1_1", "xQ1_3"])
    assert q == "Wat vind je?" and list(mm) == ["xQ1_3"], "mismatch must be flagged"


def test_inspect_sav_bounded_read():
    """Synthetic .sav in a temp data/: labels, string/datetime typing, bounded frame."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    import pyreadstat
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "data").mkdir()
        df = pd.DataFrame({
            "id": [1.0, 2.0, 3.0],
            "xQ1_1": ["goed", "", "duur"],
            "datum": ["2024-01-01", "2024-02-01", "2024-03-01"],
        })
        pyreadstat.write_sav(df, str(Path(td) / "data" / "mini.sav"),
                             column_labels=["Respondent", "xQ1_1 Wat vind je?", "Datum"])
        old_root = be.PROJECT_ROOT
        be.PROJECT_ROOT = Path(td)
        try:
            insp = be.inspect_sav("mini.sav")
        finally:
            be.PROJECT_ROOT = old_root
        assert insp.variables["xQ1_1"] == {"label": "xQ1_1 Wat vind je?", "is_string": True}
        assert not insp.variables["id"]["is_string"]
        assert not insp.variables["datum"]["is_string"], "datetime strings must be filtered"
        assert insp.string_vars == ["xQ1_1"]
        assert len(insp.frame) == 3


def test_selection_phase_writes_nothing():
    """§3.7: picking a server file must not create or touch anything in data/."""
    data_dir = os.path.join(be.PROJECT_ROOT, "data")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".sav"))
    if not files:
        print("  (skipped: no .sav files in data/)")
        return
    before = {f: os.path.getmtime(os.path.join(data_dir, f)) for f in files}
    at = _apptest()
    at.run()
    sb = next(s for s in at.selectbox if s.key == "server_pick")
    sb.set_value(files[0])
    at.run()
    assert not at.exception
    after = {f: os.path.getmtime(os.path.join(data_dir, f))
             for f in sorted(os.listdir(data_dir)) if f.endswith(".sav")}
    assert after == before, "selection phase wrote to data/"


def test_question_survives_before_step1():
    """The committed question must be readable from the 'data' row alone, and a
    resume-edit (set_question) must stick; 'preprocessed' wins when present."""
    import sqlite3
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "cache.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE cache_metadata "
                     "(filename TEXT, step_name TEXT, variable_key TEXT, "
                     "status TEXT, var_lab TEXT)")
        fresh = be.DatasetSpec(filename="a.sav", var_name="Q1", sample_size=200)
        ran = be.DatasetSpec(filename="b.sav", var_name="Q2", sample_size=None)
        rows = [
            # freshly committed: only step 0 done, question on the data row
            ("a.sav", "data", fresh.variable_key, "valid", "Vraag bij commit?"),
            # step 1 ran: preprocessed question wins over the data row's
            ("b.sav", "data", ran.variable_key, "valid", "Oude vraag?"),
            ("b.sav", "preprocessed", ran.variable_key, "valid", "Gebruikte vraag?"),
        ]
        conn.executemany("INSERT INTO cache_metadata VALUES (?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        old_db_path = be._db_path
        be._db_path = lambda: db
        try:
            labs = {(s.filename, s.variable_key): s.var_lab
                    for s in be.list_cached_datasets()}
            assert labs[("a.sav", fresh.variable_key)] == "Vraag bij commit?"
            assert labs[("b.sav", ran.variable_key)] == "Gebruikte vraag?"
            fresh.var_lab = "Aangepaste vraag?"
            be.set_question(fresh)
            labs = {(s.filename, s.variable_key): s.var_lab
                    for s in be.list_cached_datasets()}
            assert labs[("a.sav", fresh.variable_key)] == "Aangepaste vraag?"
        finally:
            be._db_path = old_db_path


def test_run_step_translates_cache_corruption():
    """A cache going valid->invalid during a failed run yields the clean
    'reset, press Opnieuw' message; an unrelated failure re-raises as-is."""
    import contextlib
    spec = be.DatasetSpec(filename="x.sav", var_name="Q1", sample_size=100)
    orig_dispatch, orig_valid, orig_verbose = (
        be._dispatch, be._valid_data_caches, be._verbose)
    be._verbose = lambda spec, step: contextlib.nullcontext()

    def boom(*a, **k):
        raise TypeError("object of type 'NoneType' has no len()")
    be._dispatch = boom
    try:
        # Corruption signature: a valid cache disappears across the failed run
        seen = {"n": 0}
        def losing(step, s, cm):
            seen["n"] += 1
            return {"preprocessed"} if seen["n"] == 1 else set()
        be._valid_data_caches = losing
        try:
            be.run_step(1, spec)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "beschadigd" in str(e) and "Opnieuw" in str(e)

        # No cache lost: the original error must propagate unchanged
        be._valid_data_caches = lambda step, s, cm: {"preprocessed"}
        try:
            be.run_step(1, spec)
            assert False, "expected TypeError"
        except TypeError as e:
            assert "NoneType" in str(e)
    finally:
        be._dispatch, be._valid_data_caches, be._verbose = (
            orig_dispatch, orig_valid, orig_verbose)


# =============================================================================
# Unit: costs (Phase B6)
# =============================================================================

def test_costs_key_mapping_matches_real_files():
    """Plan §3.6c: every step key the pipeline ever wrote must be in our mapping."""
    import glob
    import json
    known = set(be.STEP_COSTS_KEY.values())
    files = glob.glob(os.path.join(be.PROJECT_ROOT, "exports", "costs", "*_kosten.json"))
    assert files, "no costs files found to verify against"
    seen = set()
    for f in files:
        seen.update(json.load(open(f, encoding="utf-8")).get("steps", {}).keys())
    assert seen, "costs files carry no step entries"
    assert seen <= known, f"unmapped cost step keys: {seen - known}"


def test_step_costs_reads_existing_file():
    """For a cached dataset with a costs file, step_costs yields total + date."""
    specs = [s for s in be.list_cached_datasets() if be.costs_path(s).exists()]
    if not specs:
        print("  (skipped: no cached dataset with a costs file)")
        return
    spec = specs[0]
    data = be.load_costs(spec)
    assert data and data.get("steps"), f"{be.costs_path(spec)}: empty costs JSON"
    hit = False
    for step in be.STEP_COSTS_KEY:
        entry = be.step_costs(spec, step)
        if entry is None:
            continue
        hit = True
        assert "cost_usd" in (entry.get("total") or {}), f"step {step}: no total.cost_usd"
        assert entry.get("date"), f"step {step}: no date (staleness must be visible)"
    assert hit, "costs file present but no step matched STEP_COSTS_KEY"
    # Steps without LLM costs must yield None, not crash
    assert be.step_costs(spec, 0) is None and be.step_costs(spec, 7) is None


# =============================================================================
# AppTest: page layer
# =============================================================================

def _apptest():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=30)

def _first_cached_spec():
    specs = be.list_cached_datasets()
    return specs[0] if specs else None


def test_boot_dataset_select():
    at = _apptest()
    at.run()
    assert not at.exception
    # No dataset loaded → the select page (header present, no step nav buttons)
    assert any("CoderingsTool" in h.value for h in at.header)


def test_step_screens_match_cache_state():
    spec = _first_cached_spec()
    if spec is None:
        print("  (skipped: no cached datasets)")
        return
    status = be.step_status(spec, be.CacheManager())
    for step in range(1, LAST_STEP + 1):
        at = _apptest()
        at.session_state["spec"] = spec
        at.session_state["step"] = step
        at.run()
        assert not at.exception, f"step {step}: exception {at.exception}"
        keys = {b.key for b in at.button}
        expected = screen_for(step, status)
        if expected is Screen.OUTPUT:
            assert f"rerun_{step}" in keys, f"step {step}: OUTPUT lacks re-run"
            assert f"run_{step}" not in keys, f"step {step}: OUTPUT shows run button"
            if step < LAST_STEP:
                assert f"continue_{step}" in keys, f"step {step}: OUTPUT lacks continue"
        elif expected is Screen.RUN:
            assert f"run_{step}" in keys, f"step {step}: RUN lacks run button"
            # RUN screen explains the step before spending credits
            assert at.info, f"step {step}: RUN shows no step info"
        else:  # LOCKED
            assert f"run_{step}" not in keys, f"step {step}: LOCKED shows run button"
            assert at.warning, f"step {step}: LOCKED shows no lock warning"


def test_error_is_sticky_across_rerun():
    spec = _first_cached_spec()
    if spec is None:
        print("  (skipped: no cached datasets)")
        return
    at = _apptest()
    at.session_state["spec"] = spec
    at.session_state["step"] = 1
    at.session_state["last_error"] = (1, "Simulated failure")
    at.run()
    assert any("Simulated failure" in e.value for e in at.error)
    # An unrelated interaction (navigate away and back) must NOT clear it
    at.session_state["step"] = 2
    at.run()
    at.session_state["step"] = 1
    at.run()
    assert any("Simulated failure" in e.value for e in at.error), "error vanished on rerun"
    # Dismissing clears it
    at.button(key="dismiss_err_1").click().run()
    assert not any("Simulated failure" in e.value for e in at.error)


def test_continue_button_advances():
    spec = _first_cached_spec()
    if spec is None:
        print("  (skipped: no cached datasets)")
        return
    status = be.step_status(spec, be.CacheManager())
    done_steps = [s for s in range(1, LAST_STEP) if status[s]]
    if not done_steps:
        print("  (skipped: no completed mid-pipeline steps)")
        return
    step = done_steps[0]
    at = _apptest()
    at.session_state["spec"] = spec
    at.session_state["step"] = step
    at.run()
    at.button(key=f"continue_{step}").click().run()
    assert at.session_state["step"] == step + 1


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
