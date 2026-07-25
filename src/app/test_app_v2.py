"""
test_app_v2.py — Phase A verification for the v2 Streamlit app.

Covers (plan §3.5 Phase A): every step shows the right screen for every cache
state; sticky errors survive a rerun. Pure logic (screen_for, registry) is
unit-tested exhaustively; the page layer is smoke-tested with
streamlit.testing.v1.AppTest against the real cache (dataset-dependent tests
skip cleanly when the cache is empty).

Run:  cd src && python test_app_v2.py     (or: pytest test_app_v2.py)
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

APP = os.path.join(_APP_DIR, "app_v2.py")


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
    from config import STEP_MODEL_TIERS
    for step, view in av.STEP_VIEWS.items():
        for phase in view.phases:
            assert phase in STEP_MODEL_TIERS, f"step {step}: unknown phase {phase!r}"

def test_models_line_resolves_for_all_steps():
    import app_views as av
    for step in range(LAST_STEP + 1):
        line = av.models_line(step)
        if av.STEP_VIEWS[step].phases:
            assert line, f"step {step} has phases but no model line"
        else:
            assert line is None


# =============================================================================
# Unit: verbose-log → report parser (Phase B0)
# =============================================================================

def test_parser_handles_all_existing_logs():
    """Plan §3.6a: lenient — every log in exports/verbose_logs/ must parse."""
    import glob
    files = glob.glob(os.path.join(be.PROJECT_ROOT, "exports", "verbose_logs", "*.txt"))
    assert files, "no logs found to verify against"
    for f in files:
        text = open(f, encoding="utf-8", errors="replace").read()
        rep = be.parse_verbose_log(text)   # must never raise
        # nothing invented: every kept line exists verbatim in the source
        for sec in rep.sections:
            for ln in sec.body[:3] + sec.summary[:3] + sec.noise[:3]:
                assert ln.strip() == "" or ln.strip() in text, f"{f}: fabricated line {ln!r}"
        # logs with an explicit section marker yield a titled section
        if "[SECTION]" in text:
            assert any(s.title for s in rep.sections), f"{f}: [SECTION] present but no titled section"


def test_parser_step2_example():
    """Known log from the 2026-07-23 run: meta, summary and noise land correctly."""
    # The known run is identified by its timestamp; the dataset part of the
    # filename is local-only, so the log is looked up by glob.
    matches = glob.glob(os.path.join(be.PROJECT_ROOT, "exports", "verbose_logs",
                                     "*_Qd1_100_100_step2_20260723_042437.txt"))
    if not matches:
        print("  (skipped: example log not present)")
        return
    rep = be.parse_verbose_log(open(matches[0], encoding="utf-8").read())
    assert rep.meta.get("Sample size") == "100"
    sec = next(s for s in rep.sections if s.title == "QUALITY FILTERING")
    assert any("Total meaningful" in ln for ln in sec.summary), "summary block not captured"
    assert any("RATE LIMITING SETUP" in ln for ln in sec.noise), "telemetry not collapsed"
    assert not any("inflight" in ln for ln in sec.body), "telemetry leaked into body"


# =============================================================================
# Unit: variable merging (upload pre-step, concat_open_ends)
# =============================================================================

_CONCAT_DIR = os.path.join(be.PROJECT_ROOT, "concatenate")
if _CONCAT_DIR not in sys.path:
    sys.path.insert(0, _CONCAT_DIR)


def test_concat_slot_group_detection():
    import concat_open_ends as co
    cols = ["resp_id", "xQd1_1", "xQd1_2", "xQd1_10", "Q5", "opm_1"]
    groups = co.find_slot_groups(cols)
    # Numeric order (2 before 10), singles like opm_1 excluded
    assert groups == {"xQd1_": ["xQd1_1", "xQd1_2", "xQd1_10"]}


def test_concat_combine_row_skips_empty():
    import concat_open_ends as co
    import pandas as pd
    assert co.combine_row(["goed", "", None, " duur "], ", ") == "goed, duur"
    assert co.combine_row(["", None], ", ") is pd.NA


def test_concat_variables_roundtrip(tmp_dir=None):
    """End-to-end on a tiny synthetic .sav in a temp dir (never touches data/)."""
    import tempfile
    import pandas as pd
    import pyreadstat
    import concat_open_ends as co
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
    # Capitals + closing period = formatting, not a correction
    assert av._formatting_key("ik vind het mooi") == av._formatting_key("Ik vind het mooi.")
    # Extra whitespace and commas are formatting too
    assert av._formatting_key("goed,  betrouwbaar") == av._formatting_key("Goed betrouwbaar.")
    # A changed word is a real correction
    assert av._formatting_key("betrouwbar") != av._formatting_key("betrouwbaar")


# =============================================================================
# Unit: costs (Phase B6)
# =============================================================================

def test_costs_key_mapping_matches_real_files():
    """Plan §3.6c: every step key the pipeline ever wrote must be in our mapping."""
    import glob
    import json
    known = set(be.STEP_COSTS_KEY.values())
    files = glob.glob(os.path.join(be.PROJECT_ROOT, "exports", "costs", "*_costs.json"))
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
