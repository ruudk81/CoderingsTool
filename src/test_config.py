"""Guards on config.py's per-phase tables.

Two kinds of drift, and both used to be silent.

KEYS: STEP_EFFORT and STEP_VERBOSITY are keyed by phase name, and so is
STEP_MODEL — three hand-maintained lists that must agree. A renamed phase does
not crash: the .get() falls back to the global default, so the phase silently
loses its setting.

VALUES: a phase can also point at a rung of the model ladder that was never
deployed. That is how, on 2026-08-08, seven phases came to resolve to a model
that did not exist and got quietly routed to a different deployment — with the
whole suite green, because every test checked keys and none checked values.
"""

from pathlib import Path

import pytest

from config import (
    MODEL_PRICING,
    MODELS,
    OPENAI_MODEL_LIMITS,
    REASONING_EFFORT,
    STEP_EFFORT,
    STEP_MODEL,
    STEP_VERBOSITY,
    TEXT_VERBOSITY,
    ModelConfig,
    get_azure_route,
    get_model_for_api,
    get_reasoning_params,
    get_step_effort,
    get_step_model,
    get_step_verbosity,
)

# "minimal" is absent on purpose: both gpt-5.4 and gpt-5.6-luna reject it.
VALID_EFFORTS = {"none", "low", "medium", "high"}
VALID_VERBOSITIES = {"low", "medium", "high"}


# =============================================================================
# KEYS — the three per-phase tables agree
# =============================================================================

@pytest.mark.parametrize("phase", sorted(STEP_EFFORT))
def test_effort_phase_exists(phase):
    assert phase in STEP_MODEL, (
        f"STEP_EFFORT has {phase!r}, which is not a phase in STEP_MODEL — "
        f"renamed? the phase silently falls back to {REASONING_EFFORT!r}"
    )


@pytest.mark.parametrize("phase", sorted(STEP_VERBOSITY))
def test_verbosity_phase_exists(phase):
    assert phase in STEP_MODEL, (
        f"STEP_VERBOSITY has {phase!r}, which is not a phase in STEP_MODEL — "
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
        assert phase in STEP_MODEL
        assert get_step_effort(phase) == REASONING_EFFORT, (
            f"{phase} is a bulk phase; giving it its own effort multiplies cost "
            f"across ~98% of all calls"
        )


# =============================================================================
# VALUES — every phase resolves to a model that is fully described
# =============================================================================

@pytest.mark.parametrize("phase", sorted(STEP_MODEL))
def test_every_phase_resolves_to_a_deployed_model(phase):
    """A phase may only point at a rung that MODELS actually carries."""
    model = get_step_model(phase)
    assert model in {m.name for m in MODELS.values()}


@pytest.mark.parametrize("phase", sorted(STEP_MODEL))
def test_every_phase_model_is_completely_described(phase):
    """Present in one register and missing from another is the failure mode here.

    A model absent from MODEL_PRICING costs DEFAULT_PRICING (wrong numbers, no
    warning); absent from MODEL_TYPES it loses its reasoning params; absent from
    the limits it reports no context window.
    """
    model = get_step_model(phase)
    assert model in MODEL_PRICING, f"{phase}: {model} would fall back to DEFAULT_PRICING"
    assert model in OPENAI_MODEL_LIMITS, f"{phase}: {model} has no context/max_output"
    assert model in ModelConfig.MODEL_TYPES, f"{phase}: {model} has no reasoning/chat type"


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


def test_no_model_sits_on_two_rungs():
    names = [m.name for m in MODELS.values()]
    assert len(names) == len(set(names)), f"duplicate model in MODELS: {names}"


# =============================================================================
# REASONING PARAMS — the per-phase effort actually reaches the API
# =============================================================================

def test_reasoning_params_carry_the_per_phase_effort():
    """A judgement phase must actually reach the API with its own effort.

    Resolved through get_step_model rather than a hardcoded model name: with a
    literal here the test passes even when the phase resolves to something else
    entirely, which is precisely how the 2026-08-08 misroute stayed invisible.
    """
    judgement = get_reasoning_params(get_step_model("classifier_p2"), phase="classifier_p2")
    bulk = get_reasoning_params(get_step_model("code_assignment"), phase="code_assignment")

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
