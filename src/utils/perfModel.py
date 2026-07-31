"""Parametric performance model for LLM call pacing.

One storage primitive: per (provider:model, phase) a ring buffer of recent
observations. Token expectations, the latency curve and deployment capacity
are derived from it at read time.
Design: .superpowers/specs/2026-07-31-perf-model-rekey-design.md
"""
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from config import API_PROVIDER, get_model_for_api

STORE_FILE = Path(__file__).parent.parent.parent / "data" / "perf_model.json"
RING_SIZE = 50
MIN_PHASE_N = 5          # tier-1 threshold: a phase buffer speaks for itself
MIN_FIT_N = 10           # curve fit needs at least this many points
MIN_FIT_SPREAD = 1.5     # ... spanning at least this input-size ratio
BUCKET_WIDTH = 10        # concurrency bucket width for the capacity knee
MIN_BUCKET_N = 3         # a bucket needs this many observations to count
MAX_TIMEOUT_RATE = 0.05  # bucket health threshold
PRUNE_DAYS = 60
DEFAULT_TIKTOKEN_OFFSET = 300

# Observation layout: [in, out, latency_s, concurrency, timed_out, est_in, "YYYY-MM-DD"]
IN, OUT, LAT, CONC, TIMED_OUT, EST_IN, DAY = range(7)


def _model_key(model: str) -> str:
    return f"{API_PROVIDER}:{model}"


class PerfModel:
    """Self-learning, constant-size performance store. Never raises into a run."""

    def __init__(self, path: Path = STORE_FILE):
        self._path = Path(path)
        self._lock = Lock()
        self._buffers: Dict[str, Dict[str, List[list]]] = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text())
            buffers = data.get("buffers", {})
            if isinstance(buffers, dict):
                return buffers
        except FileNotFoundError:
            return {}
        except Exception as exc:
            print(f"[perfModel] WARNING: could not read {self._path}: {exc} — starting fresh")
        return {}

    def observe(self, model: str, phase: str, input_tokens: int, output_tokens: int,
                latency_s: float, concurrency: int, timed_out: bool = False,
                est_input_tokens: Optional[int] = None) -> None:
        if input_tokens <= 0 and output_tokens <= 0:
            return
        obs = [int(input_tokens), int(output_tokens), round(float(latency_s), 3),
               int(concurrency), bool(timed_out),
               int(est_input_tokens) if est_input_tokens else None,
               date.today().isoformat()]
        with self._lock:
            buf = self._buffers.setdefault(_model_key(model), {}).setdefault(phase, [])
            buf.append(obs)
            del buf[:-RING_SIZE]

    def save(self) -> None:
        try:
            cutoff = (date.today() - timedelta(days=PRUNE_DAYS)).isoformat()
            with self._lock:
                for mk in list(self._buffers):
                    phases = self._buffers[mk]
                    for ph in list(phases):
                        if not phases[ph] or phases[ph][-1][DAY] < cutoff:
                            del phases[ph]
                    if not phases:
                        del self._buffers[mk]
                payload = json.dumps({"version": 1, "buffers": self._buffers}, indent=1)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(payload)
            tmp.replace(self._path)
        except Exception as exc:
            print(f"[perfModel] WARNING: could not save stats: {exc}")


# Shared instance: one store per process, like token_tracker in llm.py.
perf_model = PerfModel()
