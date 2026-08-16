"""Tests for SmoothRequester's dispatch spread.

The spread exists to prevent one thing: the first wave of heavy requests arriving
at the server as a single wall. Two things went wrong there and are pinned here —
it was switched on by a borrowed estimate, and it never stopped.
"""
import asyncio

import pytest

from utils.llm import RateLimits
from utils.perfModel import Prediction
from utils.smoothRequester import (
    DISPATCH_DELAY_P50_THRESHOLD,
    DISPATCH_DELAY_SPREAD_FACTOR,
    SmoothRequester,
)


def _requester(monkeypatch, *, p50, origin, num_tasks=1451):
    """A requester with a prescribed warm start, without a network."""
    pred = Prediction()
    pred.p50_latency_s = p50
    pred.avg_tokens = 1500
    pred.expected_input_tokens = 1200
    pred.expected_output_tokens = 300
    pred.origins = {"avg_tokens": origin, "p50_latency_s": "curve"}

    import utils.smoothRequester as mod
    monkeypatch.setattr(mod.perf_model, "predict", lambda model, phase: pred)

    return SmoothRequester(
        model="gpt-5.6-luna", phase_key="step4_assignment", num_tasks=num_tasks,
        verbose=False, known_limits=RateLimits(105_000, 1_000),
        has_server_headers=True, show_setup=False, quiet=True,
    )


# =============================================================================
# FIX 2 — do not spread on a borrowed estimate
# =============================================================================

def test_no_spread_when_the_estimate_comes_from_the_pool():
    """Without its own history the token estimate is an average over phases doing
    1.5k and 5.5k tokens. A light phase then inherits a heavy one's caution — that
    cost six minutes on 8% of the token budget on 2026-08-13."""
    def check(monkeypatch):
        r = _requester(monkeypatch, p50=7.85, origin="pool")
        assert r._dispatch_delay == 0.0
    with pytest.MonkeyPatch.context() as mp:
        check(mp)


def test_spread_when_the_phase_has_shown_itself_heavy():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        verwacht = (17.0 - DISPATCH_DELAY_P50_THRESHOLD) / DISPATCH_DELAY_SPREAD_FACTOR
        assert r._dispatch_delay == pytest.approx(verwacht)


def test_a_light_phase_with_its_own_history_does_not_spread():
    """Onder de drempel is er niets te ontzien."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=2.8, origin="phase")
        assert r._dispatch_delay == 0.0


def test_without_a_p50_there_is_no_spread():
    """A cold phase does not guess that it is heavy."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=None, origin="phase")
        assert r._dispatch_delay == 0.0


# =============================================================================
# FIX 1 — spreiden stopt zodra de pijplijn vol is
# =============================================================================

def test_the_spread_applies_while_filling():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(1) == pytest.approx(1000.0 + r._dispatch_delay)
        assert r._stagger_target(199) == pytest.approx(1000.0 + 199 * r._dispatch_delay)


def test_the_spread_stops_once_the_pipeline_is_full():
    """The counter ran on across the whole phase and thereby became a permanent
    throughput ceiling of 1/delay, which knows nothing of RPM or TPM."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(200) is None
        assert r._stagger_target(1450) is None


def test_the_first_dispatch_never_waits():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(0) is None


def test_zonder_vertraging_wacht_niemand():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=2.8, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(5) is None


def test_concurrency_nul_blokkeert_niet():
    """optimal_concurrency is 0 until _probe_and_setup runs; a division by zero
    or an eternal queue is not an acceptable outcome there."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 0
        r._dispatch_start = 1000.0
        assert r._stagger_target(1) is None


def test_growing_concurrency_spreads_the_new_slots():
    """Workers are added when the controller raises concurrency; that new wave is
    a real wave and may be spread."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r._dispatch_start = 1000.0
        r.optimal_concurrency = 200
        assert r._stagger_target(250) is None
        r.optimal_concurrency = 330
        assert r._stagger_target(250) is not None


# =============================================================================
# An unknown limit does not clamp the measured capacity
# =============================================================================

def _setup(monkeypatch, *, probe_limits, num_tasks=1236, knee=267):
    """A requester that has been through _probe_and_setup, without a network."""
    pred = Prediction()
    pred.p50_latency_s = 2.4
    pred.avg_tokens = 1500
    pred.expected_input_tokens, pred.expected_output_tokens = 1200, 300
    pred.concurrency = knee
    pred.origins = {"avg_tokens": "phase", "p50_latency_s": "curve"}

    import utils.smoothRequester as mod
    monkeypatch.setattr(mod.perf_model, "predict", lambda model, phase: pred)

    async def fake_fetch(model):
        return probe_limits, False

    monkeypatch.setattr(mod, "llm_fetch_rate_limits", fake_fetch)
    r = mod.SmoothRequester(
        model="gpt-5.6-luna", phase_key="step3_idea_extraction",
        num_tasks=num_tasks, verbose=False, show_setup=False, quiet=True)
    import asyncio
    asyncio.run(r._probe_and_setup(num_tasks))
    return r


def test_echte_limieten_klemmen_wel():
    """With a limit from the API itself, min(rate, server, tasks) is exactly right."""
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(7_000_000, 7_000))
        assert r._limits_are_known is True
        assert r.optimal_concurrency == min(
            r._rate_limit_concurrency, r._server_concurrency, 1236)


def test_an_unknown_limit_does_not_clamp_the_measured_knee():
    """A guessed limit is not evidence. Measured 2026-08-14: the quota headers
    dropped out, FALLBACK_RPM=100 gave rate_concurrency=3 against a measured
    server_concurrency of 267, and a phase of 41 seconds took fourteen minutes."""
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(0, 0), knee=267)
        assert r._limits_are_known is False
        assert r._rate_limit_concurrency < 10       # de geraden limiet is krap
        assert r.optimal_concurrency == 267         # en telt niet mee


def test_an_unknown_limit_stays_bounded_by_the_task_count():
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(0, 0), knee=267, num_tasks=12)
        assert r.optimal_concurrency == 12


def test_without_a_measured_knee_it_falls_back_to_the_cold_ceiling():
    """No limits and no history: then the cold ceiling is the only number there
    is — still better than an invented RPM."""
    import utils.smoothRequester as mod
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(0, 0), knee=None)
        assert r.optimal_concurrency == min(mod.COLD_START_CAP, 1236)


def test_the_token_bucket_does_use_the_fallback():
    """Something has to feed the bucket; only the concurrency conclusion lapses."""
    with pytest.MonkeyPatch.context() as mp:
        from config import FALLBACK_TPM
        r = _setup(mp, probe_limits=RateLimits(0, 0))
        assert r.rate_limits.tokens_per_minute == FALLBACK_TPM


# =============================================================================
# The retry pass has to reach a task that carries no respondent
# =============================================================================

def _run_worker(requester, tasks):
    """Drive one worker over `tasks`, with every call raising.

    `_execute_task` is replaced rather than mocked at the network boundary: what
    is under test is what the worker records about a failure, not how the
    failure came about.
    """
    async def _boom(task_data, prepare_fn, parse_fn):
        raise RuntimeError("collapsed")

    async def _drive():
        requester._execute_task = _boom
        queue = asyncio.Queue()
        for i, task in enumerate(tasks):
            await queue.put((i, task))
        await queue.put(None)
        results = [None] * len(tasks)
        await requester._worker(queue, results, [], lambda t: {}, lambda t, r: r, None)
        return results

    return asyncio.run(_drive())


def test_a_failing_task_is_registered_by_position():
    """Steps 4, 5 and 6 build tasks per domain, facet or label — no respondent
    anywhere. Keying the retry on `respondent_id` meant a failure there was
    recorded as `'?'` and looked up as `''`, so those three steps never retried
    a failed task at all."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=1.0, origin="curve", num_tasks=2)
        _run_worker(r, [{"domain_label": "a"}, {"domain_label": "b"}])
        assert r.failed_task_indices == {0, 1}
        assert r.failed_task_ids == set()


def test_a_respondent_is_still_reported_where_there_is_one():
    """Steps 1-3 do carry one, and their failure reporting reads it."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=1.0, origin="curve", num_tasks=1)
        _run_worker(r, [{"respondent_id": 42, "response": "x"}])
        assert r.failed_task_indices == {0}
        assert r.failed_task_ids == {"42"}
        assert r.failure_log[0]["task_id"] == 42
