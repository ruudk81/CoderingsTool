"""Tests for perfModel. Run: python -m utils.test_perfModel (from src/)."""
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

from utils.perfModel import PerfModel, RING_SIZE, PRUNE_DAYS, _model_key


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


if __name__ == "__main__":
    test_observe_ring_and_roundtrip()
    test_corrupt_file_starts_fresh()
    test_prune_old_buffers_on_save()
    test_zero_token_observation_skipped()
    print("test_perfModel: task 1 OK")
