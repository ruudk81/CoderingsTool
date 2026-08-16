"""What config.py cannot check about itself.

_validate() runs at import and catches every internal contradiction: a phase
pointing at a rung that was never deployed, a model on two rungs, an effort or
verbosity value the API rejects, a STEP_EFFORT key that is not a phase. None of
those need a test — they cannot survive an import, and this file imports config.

Three things are left over, and they are the reason this file exists:

  - Does that guard actually fire? A module cannot assert that it crashed on
    import and go on running. Only an outsider can provoke the failure.
  - Is the setting ever passed? STEP_EFFORT only does something if a call site
    hands get_reasoning_params() a phase=. Checking that means reading
    pipeline/, which config.py must never do at import.
  - Bug or preference? Keeping the bulk phases on the default is a cost policy,
    not a contradiction. That deserves a red test you can consciously change,
    not a hard crash that blocks an experiment.
"""

from pathlib import Path

import pytest

from config import (
    REASONING_EFFORT,
    STEP_EFFORT,
    STEP_MODEL,
    _validate,
    get_azure_route,
    get_model_for_api,
    get_reasoning_params,
    get_step_effort,
    get_step_model,
    get_step_verbosity,
)


# =============================================================================
# THE IMPORT-TIME GUARD FIRES
# =============================================================================

def test_unknown_phase_raises():
    with pytest.raises(RuntimeError, match="unknown phase"):
        get_step_model("classifier_p42")


def test_unknown_rung_raises(monkeypatch):
    """Sol ("5.6", 5) is a real model but is not deployed in this tenant.

    Pointing a phase at it must fail loudly. The old config synthesised a name
    for exactly this case and a fallback routed it elsewhere.
    """
    monkeypatch.setitem(STEP_MODEL, "classifier_p1", ("5.6", 5))
    with pytest.raises(RuntimeError, match="no model deployed at rung"):
        get_step_model("classifier_p1")


def test_unknown_model_is_not_silently_routed():
    """Routing helpers must reject a name they do not know, never guess."""
    with pytest.raises(RuntimeError, match="unknown model"):
        get_azure_route("gpt-5.6-luna-mini")
    with pytest.raises(RuntimeError, match="unknown model"):
        get_model_for_api("gpt-5.6-luna-mini")


def test_renamed_effort_phase_is_caught(monkeypatch):
    """get_step_effort() uses .get(), so a stale key cannot raise on its own."""
    monkeypatch.setitem(STEP_EFFORT, "classifier_p3", "low")
    with pytest.raises(RuntimeError, match="not a phase in STEP_MODEL"):
        _validate()


def test_effort_the_api_rejects_is_caught(monkeypatch):
    """Without this the value survives import and 400s on a live call, mid-run."""
    monkeypatch.setitem(STEP_EFFORT, "classifier_facet_consolidation", "minimal")
    with pytest.raises(RuntimeError, match="rejected by the API"):
        _validate()


# =============================================================================
# THE SETTING REACHES THE API
# =============================================================================

def test_reasoning_params_carry_the_per_phase_effort():
    """A judgement phase must actually reach the API with its own effort.

    Resolved through get_step_model rather than a hardcoded model name: with a
    literal here the test passes even when the phase resolves to something else
    entirely, which is precisely how the 2026-08-08 misroute stayed invisible.
    """
    phase = "classifier_facet_consolidation"
    judgement = get_reasoning_params(get_step_model(phase), phase=phase)
    bulk = get_reasoning_params(get_step_model("code_assignment"), phase="code_assignment")

    assert judgement["reasoning"]["effort"] == STEP_EFFORT[phase]
    assert bulk["reasoning"]["effort"] == REASONING_EFFORT
    # Nested shape — both providers are on the Responses API since 2026-08-01.
    assert set(judgement) == {"reasoning", "text"}
    assert judgement["text"]["verbosity"] == get_step_verbosity(phase)


def test_chat_models_get_no_reasoning_params():
    assert get_reasoning_params("gpt-4.1", phase="classifier_facet_consolidation") == {}


@pytest.mark.parametrize("phase", sorted(STEP_EFFORT))
def test_effort_phase_is_actually_passed(phase):
    """A STEP_EFFORT entry only does something if a call site passes phase=.

    get_reasoning_params(model) without phase silently returns the global default,
    so a raised phase can sit in config doing nothing. That is what happened to
    idea_extraction_context/-taxonomy: step 3 passed no phase anywhere, so both
    entries were inert until 2026-08-01.
    """
    pipeline_dir = Path(__file__).parent / "pipeline"
    needle = f'phase="{phase}"'
    hits = [
        py for py in pipeline_dir.rglob("*.py")
        if "_experiment" not in str(py) and needle in py.read_text(encoding="utf-8")
    ]
    assert hits, (
        f"STEP_EFFORT raises {phase!r} to {STEP_EFFORT[phase]!r}, but no call site in "
        f"pipeline/ passes phase={phase!r} to get_reasoning_params — the setting is inert"
    )


# =============================================================================
# COST POLICY
# =============================================================================

def test_bulk_phases_stay_at_the_default():
    """The high-volume phases are the reason the default exists — keep them on it."""
    for phase in ("spell_check", "quality_filter", "idea_extraction_abstraction_ladder",
                  "code_assignment"):
        assert phase in STEP_MODEL
        assert get_step_effort(phase) == REASONING_EFFORT, (
            f"{phase} is a bulk phase; giving it its own effort multiplies cost "
            f"across ~98% of all calls"
        )


def test_step4_judgement_phases_run_low_verbosity():
    """Every step-4 phase raised to STEP_EFFORT "medium" reasons with a
    scratchpad, and low verbosity is what saves tokens on it (see the comment
    on STEP_VERBOSITY). A classifier_ phase found reasoning at "medium"
    without "low" here means the two tables have started disagreeing about
    what kind of phase it is — exactly what happened to classifier_facet_settle
    before this test existed.
    """
    for phase, effort in STEP_EFFORT.items():
        if phase.startswith("classifier_") and effort == "medium":
            assert get_step_verbosity(phase) == "low", phase
