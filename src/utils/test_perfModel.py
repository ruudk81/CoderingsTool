"""Tests for perfModel. Run: python -m utils.test_perfModel (from src/)."""
import json
import math
import tempfile
from datetime import date, timedelta
from pathlib import Path

from utils.perfModel import PerfModel, RING_SIZE, PRUNE_DAYS, _model_key, phase_expectation, pool_expectation, phase_offset, fit_curve, curve_p50, capacity_knee


def _tmp_store():
    return Path(tempfile.mkdtemp()) / "perf_model.json"


def test_observe_ring_and_roundtrip():
    path = _tmp_store()
    pm = PerfModel(path)
    for i in range(RING_SIZE + 10):
        pm.observe("gpt-5.4", "p_test", 800 + i, 100, 1.5, 50, False, 780)
    buf = pm._buffers[_model_key("gpt-5.4")]["p_test"]
    assert len(buf) == RING_SIZE
    assert buf[-1][0] == 800 + RING_SIZE + 9   # newest kept
    pm.save()
    pm2 = PerfModel(path)
    assert len(pm2._buffers[_model_key("gpt-5.4")]["p_test"]) == RING_SIZE


def test_corrupt_file_starts_fresh():
    path = _tmp_store()
    path.write_text("{not json")
    pm = PerfModel(path)           # must not raise
    assert pm._buffers == {}


def test_prune_old_buffers_on_save():
    path = _tmp_store()
    pm = PerfModel(path)
    pm.observe("gpt-5.4", "p_old", 800, 100, 1.5, 50)
    old = (date.today() - timedelta(days=PRUNE_DAYS + 1)).isoformat()
    pm._buffers[_model_key("gpt-5.4")]["p_old"][-1][6] = old
    pm.observe("gpt-5.4", "p_new", 800, 100, 1.5, 50)
    pm.save()
    pm2 = PerfModel(path)
    phases = pm2._buffers[_model_key("gpt-5.4")]
    assert "p_new" in phases and "p_old" not in phases


def test_zero_token_observation_skipped():
    pm = PerfModel(_tmp_store())
    pm.observe("gpt-5.4", "p_test", 0, 0, 1.5, 50)
    assert pm._buffers == {}


def _obs(i, o, lat=1.0, conc=50, to=False, est=None):
    return [i, o, lat, conc, to, est, date.today().isoformat()]


def test_phase_expectation():
    buf = [_obs(800, 100) for _ in range(5)]
    assert phase_expectation(buf) == (800, 100)
    assert phase_expectation(buf[:4]) is None                       # below threshold
    buf_to = buf + [_obs(800, 0, lat=90.0, to=True)] * 10
    assert phase_expectation(buf_to) == (800, 100)                  # timeouts excluded


def test_pool_expectation():
    phases = {
        "a": [_obs(1000, 100) for _ in range(5)],   # ratio 0.10
        "b": [_obs(2000, 400) for _ in range(5)],   # ratio 0.20
        "c": [_obs(3000, 900) for _ in range(5)],   # ratio 0.30
    }
    in_e, out_e = pool_expectation(phases)
    assert in_e == 2000                              # median input
    assert out_e == 400                              # median ratio 0.20 × 2000
    assert pool_expectation({}) is None


def test_phase_offset():
    buf = [_obs(820, 100, est=800) for _ in range(5)]
    assert phase_offset(buf) == 20
    assert phase_offset([_obs(820, 100, est=None)] * 5) is None


def test_fit_curve_recovers_planted_coeffs():
    a, b_in, b_out = 0.5, 0.3, 0.9
    phases = {"p": []}
    for i, (tin, tout) in enumerate([(500, 50), (1000, 100), (2000, 400),
                                     (4000, 200), (800, 300), (1500, 700),
                                     (3000, 900), (600, 60), (2500, 150), (1200, 500)]):
        lat = math.exp(a + b_in * math.log(tin) + b_out * math.log(tout))
        phases["p"].append(_obs(tin, tout, lat=lat))
    coeffs = fit_curve(phases)
    assert coeffs is not None
    fa, fb_in, fb_out = coeffs
    assert abs(fa - a) < 0.01 and abs(fb_in - b_in) < 0.01 and abs(fb_out - b_out) < 0.01
    assert abs(curve_p50(coeffs, 1000, 100) -
               math.exp(a + b_in * math.log(1000) + b_out * math.log(100))) < 0.01


def test_fit_curve_guards():
    assert fit_curve({"p": [_obs(1000, 100, lat=2.0)] * 5}) is None      # no spread
    assert fit_curve({"p": [_obs(1000, 100, lat=2.0),
                            _obs(2000, 200, lat=3.0)]}) is None          # too few
    to = {"p": [_obs(500 * (i + 1), 100, lat=90.0, to=True) for i in range(12)]}
    assert fit_curve(to) is None                                          # timeouts excluded


def test_capacity_knee_finds_planted_knee():
    bufs = [[]]
    for conc in (20, 40, 60, 80, 100):
        for _ in range(5):
            bufs[0].append(_obs(800, 100, lat=1.0 + conc / 100, conc=conc))
    for _ in range(5):                                   # concurrency 120: sick
        bufs[0].append(_obs(800, 100, lat=60.0, conc=120, to=True))
    assert capacity_knee(bufs) == 100


def test_capacity_knee_no_pressure_no_claim():
    assert capacity_knee([[_obs(800, 100, conc=50)] * 2]) is None    # n < MIN_BUCKET_N
    assert capacity_knee([]) is None


def test_capacity_knee_never_extrapolates():
    bufs = [[_obs(800, 100, conc=73) for _ in range(10)]]
    assert capacity_knee(bufs) == 73                      # observed max, not a rounded-up bucket edge


if __name__ == "__main__":
    test_observe_ring_and_roundtrip()
    test_corrupt_file_starts_fresh()
    test_prune_old_buffers_on_save()
    test_zero_token_observation_skipped()
    test_phase_expectation()
    test_pool_expectation()
    test_phase_offset()
    test_fit_curve_recovers_planted_coeffs()
    test_fit_curve_guards()
    test_capacity_knee_finds_planted_knee()
    test_capacity_knee_no_pressure_no_claim()
    test_capacity_knee_never_extrapolates()
    print("test_perfModel: OK")
