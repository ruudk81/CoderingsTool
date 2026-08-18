"""Tests for SmoothRequester's dispatch spread.

The spread exists to prevent one thing: the first wave of heavy requests arriving
at the server as a single wall. Two things went wrong there and are pinned here —
it was switched on by a borrowed estimate, and it never stopped.
"""
import asyncio
import time

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


# =============================================================================
# The tick line names the brake that is actually holding the phase back
#
# Measured 2026-08-17 on step4_assignment: 132 workers queued behind the
# admission gate at 1% of the token budget, for 158 of the phase's 209 seconds,
# while the tick line read `inflight:81 ... RATE-CAPPED`. Every number in that
# line was true and not one of them was the brake — `inflight` counted workers
# standing in a queue, and `RATE-CAPPED` compares two concurrency arms that were
# both slack.
# =============================================================================

def _at_work(monkeypatch, *, num_tasks=1375, avg_tokens=2277):
    """A requester with both gates built and past warm-up, without a network."""
    r = _requester(monkeypatch, p50=2.88, origin="phase", num_tasks=num_tasks)
    limits = RateLimits(6_300_000, 63_000)
    r.rate_limits = limits
    r._limits_are_known = True
    r.avg_tokens = avg_tokens
    r._setup_rate_pacing(limits)
    r._setup_concurrency(limits, num_tasks, has_server_headers=False)
    r._start_time = time.time() - 60
    return r


def _tick_line(r, *, active, concurrency, tpm_pct=50.0):
    effective_tpm = r.rate_limits.tokens_per_minute * r._headroom
    effective_rpm = r.rate_limits.requests_per_minute * r._headroom
    return r._tick(
        completed=1013, total=1375, tick_rate=9.0, p50=1.85, throughput=40.0,
        active=active, concurrency=concurrency,
        current_tpm=effective_tpm * tpm_pct / 100, effective_tpm=effective_tpm,
        current_rpm=effective_rpm * tpm_pct / 100, effective_rpm=effective_rpm,
        num_tasks=1375)


def test_the_tick_separates_calls_at_the_api_from_workers_in_the_queue():
    """`inflight` was `semaphore.active`, and the semaphore is held while a worker
    waits at either gate. So `inflight:81 | tok:1%` was not a contradiction to be
    explained away — it was 81 workers in a queue, reported as work."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r._inflight_starts = {i: time.perf_counter() for i in range(3)}
        r._waiting_admit = 127
        line = _tick_line(r, active=130, concurrency=163, tpm_pct=1.0)
        assert "api:3" in line
        assert "queued:127" in line


def test_the_tick_shows_the_admission_rate():
    """The one control variable that was binding is the one the line never
    printed."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r.current_arrival_rate = 0.5
        line = _tick_line(r, active=130, concurrency=163)
        assert "admit:0.5/s" in line


def test_the_admission_gate_is_named_when_it_holds_most_of_the_fleet():
    """The regression: token budget almost untouched, fleet parked at the
    admission gate, and the label has to say so."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r._inflight_starts = {i: time.perf_counter() for i in range(3)}
        r._waiting_admit = 127
        line = _tick_line(r, active=130, concurrency=163, tpm_pct=1.0)
        assert line.rstrip().endswith("ADMIT")


def test_the_token_bucket_is_named_when_it_holds_most_of_the_fleet():
    """Genuinely TPM-bound looks different from throttled-by-the-controller, and
    the difference is which queue the workers are standing in."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r._inflight_starts = {i: time.perf_counter() for i in range(3)}
        r._waiting_tokens = 127
        line = _tick_line(r, active=130, concurrency=163, tpm_pct=99.0)
        assert line.rstrip().endswith("TPM")


def test_a_full_semaphore_is_named_when_no_gate_is_queueing():
    """Nobody waiting at a gate but no free slot either: then concurrency is the
    brake, and which arm set it is the useful half of the old RATE-CAPPED."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r._inflight_starts = {i: time.perf_counter() for i in range(163)}
        r._concurrency_controller.current = 10_000      # server arm slack
        line = _tick_line(r, active=163, concurrency=163)
        assert line.rstrip().endswith("CONC(rate)")


def test_a_phase_that_is_not_held_up_reports_its_controller_state():
    """No queue, slots to spare — then there is nothing braking and the state
    machine is the interesting thing again."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r._inflight_starts = {i: time.perf_counter() for i in range(60)}
        line = _tick_line(r, active=60, concurrency=163)
        assert line.rstrip().endswith("RAMP-UP")


# =============================================================================
# The summary reports being held up, because the tick lines scroll away
# =============================================================================

def test_time_at_the_admission_floor_is_booked():
    """At the floor the phase admits one call every two seconds regardless of
    what any controller decides, so how long it sat there is the whole story."""
    import utils.smoothRequester as mod
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r.current_arrival_rate = mod.ADMIT_RATE_FLOOR
        r._account_gate_time(0.1)
        assert r._admit_floor_seconds == pytest.approx(0.1)


def test_throttling_while_the_budget_is_idle_is_booked():
    """The number that would have made 2026-08-17 self-evident: held up with
    room to spare."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r._waiting_admit = 127
        r._last_tpm_pct = 1.0
        r._account_gate_time(0.1)
        assert r._starved_seconds == pytest.approx(0.1)


def test_queueing_against_a_spent_budget_is_not_starvation():
    """Waiting because the tokens really are gone is the system working."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        r._waiting_tokens = 127
        r._last_tpm_pct = 98.0
        r._account_gate_time(0.1)
        assert r._starved_seconds == 0.0


# =============================================================================
# The admission gate is adjusted, never replaced
#
# The 158-second tail of 2026-08-17. The PID recovered from its floor to the RPM
# ceiling within five seconds — and the 132 workers already inside the gate
# never heard it, because `self.rate_limiter = AsyncLimiter(...)` rebinds an
# attribute while they are awaiting a future in the old object's queue. That
# object kept releasing one admission every two seconds until it ran dry.
# ConcurrencyGate solved exactly this for the other gate, with set_limit().
# =============================================================================

def _admissions(pacer, n, *, after=None, delay=0.2, timeout=5.0):
    """Queue `n` workers on `pacer`, optionally run `after` once they are all
    waiting, and return each admission time relative to the start."""
    async def run():
        t0 = time.perf_counter()
        seen = []

        async def worker():
            await pacer.acquire()
            seen.append(time.perf_counter() - t0)

        tasks = [asyncio.create_task(worker()) for _ in range(n)]
        await asyncio.sleep(delay)
        if after is not None:
            after()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        return seen

    return asyncio.run(run())


def test_raising_the_rate_releases_workers_that_are_already_waiting():
    """The regression. Queued at one admission per two seconds, then the rate
    recovers — everyone still waiting must feel it."""
    from utils.smoothRequester import ArrivalPacer
    pacer = ArrivalPacer(0.5)
    seen = _admissions(pacer, 8, after=lambda: pacer.set_rate(900.0))
    assert max(seen) < 1.0, f"still draining at the old rate: {seen}"


def test_the_pacer_holds_the_rate_it_is_given():
    """It is a pacer, not a pass-through: without a rate change the spacing has
    to be real, or the TPM budget is defended by nothing."""
    from utils.smoothRequester import ArrivalPacer
    pacer = ArrivalPacer(20.0)
    seen = _admissions(pacer, 10, delay=0.0)
    assert max(seen) >= 0.4       # 10 admissions at 20/s cannot fit in less


def test_the_pacer_will_not_pace_below_the_floor():
    from utils.smoothRequester import ArrivalPacer, ADMIT_RATE_FLOOR
    pacer = ArrivalPacer(0.001)
    assert pacer.rate == ADMIT_RATE_FLOOR
    pacer.set_rate(0.001)
    assert pacer.rate == ADMIT_RATE_FLOOR


def test_the_requester_adjusts_its_gate_in_place():
    """Identity is the property that has to hold: a new object is a new queue,
    and the old queue is where the fleet was standing."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        gate = r.arrival_pacer
        asyncio.run(r._apply_pid())
        assert r.arrival_pacer is gate


def test_warm_up_calibration_also_adjusts_in_place():
    """The other caller that used to swap the object."""
    with pytest.MonkeyPatch.context() as mp:
        r = _at_work(mp)
        gate = r.arrival_pacer
        r.actual_total_tokens.extend([2000, 2100, 2200])
        r.latency_tracker.add(1.8)
        r.latency_tracker.add(1.9)
        r._calibrate_tokens()
        assert r.arrival_pacer is gate


# =============================================================================
# The token bucket settles both ways
#
# It debits an estimate up front and settles afterwards, but only ever
# downwards: a call that cost more than estimated was never charged for the
# difference. So the one mechanism that sees real per-call token counts — the
# only one that can hold TPM exactly — could not hold it, and measured usage sat
# at 106-113% of the headroom budget while the bucket reported no waiting at
# all. That left the PID as the sole regulator, which is where 2026-08-17 began.
# =============================================================================

def test_an_under_estimate_is_charged_back():
    from utils.smoothRequester import TokenBucket

    async def run():
        bucket = TokenBucket(60_000)
        await bucket.wait_and_acquire(1_000)     # debit the estimate
        before = bucket.available
        await bucket.reconcile(500)              # it actually cost 1,500
        return before, bucket.available

    before, after = asyncio.run(run())
    assert after == pytest.approx(before - 500)


def test_an_over_estimate_is_still_refunded():
    from utils.smoothRequester import TokenBucket

    async def run():
        bucket = TokenBucket(60_000)
        await bucket.wait_and_acquire(1_000)
        before = bucket.available
        await bucket.reconcile(-400)             # it only cost 600
        return before, bucket.available

    before, after = asyncio.run(run())
    assert after == pytest.approx(before + 400)


def test_overspend_is_repaid_before_the_next_call_goes_out():
    """Backpressure is the point: an overspent budget has to be felt by the next
    acquisition, not carried silently as headroom that was never there."""
    from utils.smoothRequester import TokenBucket

    async def run():
        bucket = TokenBucket(60_000)
        await bucket.wait_and_acquire(30_000)    # half the budget, on estimate
        await bucket.reconcile(40_000)           # it actually cost 70,000
        return await bucket.acquire(1_000)       # so there is nothing left

    assert asyncio.run(run()) is not True        # a wait in seconds, not a pass


# =============================================================================
# The controllers act no more often than their measurement settles
#
# `_apply_pid` moves the dispatch rate and `_adjust_throughput_if_needed` moves
# avg_tokens, which sizes the fleet through Little's law. Both read a 60-second
# sliding window, and both ran on every pass of a loop that sleeps 0.1s — so
# roughly 200 corrections landed before the first one could show up in the
# signal. Measured 2026-08-17: a steady 6% overshoot took the arrival rate from
# 46 req/s to its 0.5 req/s floor in fifteen seconds, and avg_tokens swung -54%
# then +96% inside two seconds. `ADJUSTMENT_INTERVAL` was in the file all along,
# used nowhere.
# =============================================================================

def _fake_run(monkeypatch, *, num_tasks, latency, interval):
    """Run process_all against a fake LLM. Returns (adjustments, elapsed)."""
    import types
    import utils.smoothRequester as mod

    pred = Prediction()
    pred.p50_latency_s, pred.avg_tokens = latency, 1500
    pred.expected_input_tokens, pred.expected_output_tokens = 1200, 300
    pred.concurrency, pred.timeout_s = 40, 5.0
    pred.origins = {"avg_tokens": "phase", "p50_latency_s": "curve",
                    "concurrency": "deployment", "timeout_s": "curve",
                    "tiktoken_offset": "phase"}

    class Usage:
        input_tokens, output_tokens, total_tokens = 1200, 300, 1500

    class Response:
        usage = Usage()

    async def fake_call(**kwargs):
        await asyncio.sleep(latency)
        return Response()

    monkeypatch.setattr(mod.perf_model, "predict", lambda m, p: pred)
    monkeypatch.setattr(mod.perf_model, "observe", lambda *a, **k: None)
    monkeypatch.setattr(mod.perf_model, "save", lambda: None)
    monkeypatch.setattr(mod, "create_client",
                        lambda *a, **k: types.SimpleNamespace(_header_transport=None))
    monkeypatch.setattr(mod, "llm_create_async", fake_call)
    monkeypatch.setattr(mod, "ADJUSTMENT_INTERVAL", interval)

    r = mod.SmoothRequester(
        model="gpt-5.6-luna", phase_key="step4_assignment", num_tasks=num_tasks,
        known_limits=RateLimits(6_300_000, 63_000), has_server_headers=False,
        show_setup=False, quiet=True)

    calls = []
    real_pid = r._apply_pid

    async def counting_pid():
        calls.append(time.perf_counter())
        await real_pid()

    r._apply_pid = counting_pid

    started = time.perf_counter()
    asyncio.run(r.process_all(
        [{"task_index": i} for i in range(num_tasks)],
        lambda t: {"prompt": "word " * 200},
        lambda t, resp: {"ok": True}, None))
    return len(calls), time.perf_counter() - started


def test_the_controllers_act_no_more_often_than_the_interval_allows():
    """The invariant, not a tuned count: over an elapsed phase you cannot have
    intervened more often than the interval permits."""
    with pytest.MonkeyPatch.context() as mp:
        adjustments, elapsed = _fake_run(mp, num_tasks=60, latency=0.05, interval=0.5)
        assert adjustments <= elapsed / 0.5 + 1, (
            f"{adjustments} adjustments in {elapsed:.1f}s at a 0.5s interval")


def test_a_short_phase_is_not_retuned_at_all():
    """At the real interval a brief phase gets no PID trim — the token bucket is
    the thing enforcing TPM, per call and exactly, and the arrival rate is a
    coarse trim on top of it that has nothing to add in two seconds."""
    import utils.smoothRequester as mod
    with pytest.MonkeyPatch.context() as mp:
        adjustments, elapsed = _fake_run(
            mp, num_tasks=40, latency=0.05, interval=mod.ADJUSTMENT_INTERVAL)
        assert elapsed < mod.ADJUSTMENT_INTERVAL
        assert adjustments == 0
