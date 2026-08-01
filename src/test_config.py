"""Guards on config.py's per-phase tables.

STEP_EFFORT and STEP_VERBOSITY are keyed by phase name, and so is
STEP_MODEL_TIERS — three hand-maintained lists that must agree. A renamed phase
does not crash: the .get() falls back to the global default, so the phase
silently loses its setting. These tests make that drift loud instead.
"""

from pathlib import Path

import pytest

from config import (
    REASONING_EFFORT,
    STEP_EFFORT,
    STEP_MODEL_TIERS,
    STEP_VERBOSITY,
    TEXT_VERBOSITY,
    get_reasoning_params,
    get_step_effort,
    get_step_verbosity,
)

# "minimal" is absent on purpose: both gpt-5.4 and gpt-5.6-luna reject it.
VALID_EFFORTS = {"none", "low", "medium", "high"}
VALID_VERBOSITIES = {"low", "medium", "high"}


@pytest.mark.parametrize("phase", sorted(STEP_EFFORT))
def test_effort_phase_exists(phase):
    assert phase in STEP_MODEL_TIERS, (
        f"STEP_EFFORT has {phase!r}, which is not a phase in STEP_MODEL_TIERS — "
        f"renamed? the phase silently falls back to {REASONING_EFFORT!r}"
    )


@pytest.mark.parametrize("phase", sorted(STEP_VERBOSITY))
def test_verbosity_phase_exists(phase):
    assert phase in STEP_MODEL_TIERS, (
        f"STEP_VERBOSITY has {phase!r}, which is not a phase in STEP_MODEL_TIERS — "
        f"renamed? the phase silently falls back to {TEXT_VERBOSITY!r}"
    )


def test_effort_values_are_accepted_by_the_api():
    assert REASONING_EFFORT in VALID_EFFORTS
    bad = {p: e for p, e in STEP_EFFORT.items() if e not in VALID_EFFORTS}
    assert not bad, f"unsupported effort values (the API 400s on these): {bad}"


def test_verbosity_values_are_valid():
    assert TEXT_VERBOSITY in VALID_VERBOSITIES
    bad = {p: v for p, v in STEP_VERBOSITY.items() if v not in VALID_VERBOSITIES}
    assert not bad, f"unsupported verbosity values: {bad}"


def test_bulk_phases_stay_at_the_default():
    """The high-volume phases are the reason the default exists — keep them on it."""
    for phase in ("spell_check", "quality_filter", "idea_extraction_abstraction_ladder",
                  "code_assignment"):
        assert phase in STEP_MODEL_TIERS
        assert get_step_effort(phase) == REASONING_EFFORT, (
            f"{phase} is a bulk phase; giving it its own effort multiplies cost "
            f"across ~98% of all calls"
        )


def test_reasoning_params_carry_the_per_phase_effort():
    """A judgement phase must actually reach the API with its own effort."""
    judgement = get_reasoning_params("gpt-5.4", phase="classifier_p2")
    bulk = get_reasoning_params("gpt-5.4", phase="code_assignment")

    assert judgement["reasoning"]["effort"] == STEP_EFFORT["classifier_p2"]
    assert bulk["reasoning"]["effort"] == REASONING_EFFORT
    # Nested shape — both providers are on the Responses API since 2026-08-01.
    assert set(judgement) == {"reasoning", "text"}
    assert judgement["text"]["verbosity"] == get_step_verbosity("classifier_p2")


def test_chat_models_get_no_reasoning_params():
    assert get_reasoning_params("gpt-4.1", phase="classifier_p2") == {}


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
