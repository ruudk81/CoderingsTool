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
TIMEOUT_FACTOR = 6.0

# Observation layout: [in, out, latency_s, concurrency, timed_out, est_in, "YYYY-MM-DD"]
IN, OUT, LAT, CONC, TIMED_OUT, EST_IN, DAY = range(7)


def _model_key(model: str) -> str:
    return f"{API_PROVIDER}:{model}"


@dataclass
class Prediction:
    expected_input_tokens: Optional[int] = None
    expected_output_tokens: Optional[int] = None
    avg_tokens: Optional[int] = None
    p50_latency_s: Optional[float] = None
    concurrency: Optional[int] = None
    timeout_s: Optional[float] = None
    tiktoken_offset: int = DEFAULT_TIKTOKEN_OFFSET
    origins: Dict[str, str] = field(default_factory=dict)

    def origin_line(self) -> str:
        if set(self.origins.values()) <= {"default"}:
            return "all cold (default)"
        avg = f"{self.avg_tokens:,}" if self.avg_tokens is not None else "—"
        timeout = f"{self.timeout_s:.0f}s" if self.timeout_s is not None else "—"
        conc = f"{self.concurrency}" if self.concurrency is not None else "—"
        return (f"avg_tokens: {avg} ({self.origins['avg_tokens']}) | "
                f"timeout: {timeout} ({self.origins['p50_latency_s']}) | "
                f"concurrency: {conc} ({self.origins['concurrency']})")


def _cold() -> Prediction:
    return Prediction(origins={k: "default" for k in
                               ("avg_tokens", "p50_latency_s", "concurrency", "tiktoken_offset")})


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

    def predict(self, model: str, phase: str) -> Prediction:
        try:
            with self._lock:
                phases = {p: list(b) for p, b in self._buffers.get(_model_key(model), {}).items()}
                deployment = get_model_for_api(model)
                dep_buffers = [list(b)
                               for mk, phs in self._buffers.items()
                               if mk.startswith(f"{API_PROVIDER}:") and get_model_for_api(mk.split(":", 1)[1]) == deployment
                               for b in phs.values()]
            pred = _cold()
            buf = phases.get(phase, [])

            exp = phase_expectation(buf)
            if exp:
                pred.origins["avg_tokens"] = "phase"
            else:
                exp = pool_expectation(phases)
                if exp:
                    pred.origins["avg_tokens"] = "pool"
            if exp:
                pred.expected_input_tokens, pred.expected_output_tokens = exp
                pred.avg_tokens = exp[0] + exp[1]

            coeffs = fit_curve(phases)
            if coeffs and exp:
                pred.p50_latency_s = curve_p50(coeffs, *exp)
                pred.timeout_s = pred.p50_latency_s * TIMEOUT_FACTOR
                pred.origins["p50_latency_s"] = "curve"

            knee = capacity_knee(dep_buffers)
            if knee:
                pred.concurrency = knee
                pred.origins["concurrency"] = "deployment"

            off = phase_offset(buf)
            if off is not None:
                pred.tiktoken_offset = off
                pred.origins["tiktoken_offset"] = "phase"
            return pred
        except Exception as exc:
            print(f"[perfModel] WARNING: predict failed ({exc}) — running cold")
            return _cold()


def _live_rows(buf: List[list]) -> List[list]:
    return [r for r in buf if not r[TIMED_OUT]]


def phase_expectation(buf: List[list]) -> Optional[tuple]:
    rows = _live_rows(buf)
    if len(rows) < MIN_PHASE_N:
        return None
    return (round(statistics.mean(r[IN] for r in rows)),
            round(statistics.mean(r[OUT] for r in rows)))


def pool_expectation(phases: Dict[str, List[list]]) -> Optional[tuple]:
    per_phase = [e for e in (phase_expectation(b) for b in phases.values()) if e]
    per_phase = [(i, o) for i, o in per_phase if i > 0]
    if not per_phase:
        return None
    in_med = statistics.median(i for i, _ in per_phase)
    ratio_med = statistics.median(o / i for i, o in per_phase)
    return (round(in_med), round(in_med * ratio_med))


def phase_offset(buf: List[list]) -> Optional[int]:
    deltas = [r[IN] - r[EST_IN] for r in _live_rows(buf) if r[EST_IN]]
    if len(deltas) < MIN_PHASE_N:
        return None
    return round(statistics.median(deltas))


def fit_curve(phases: Dict[str, List[list]]) -> Optional[tuple]:
    pts = [(r[IN], r[OUT], r[LAT]) for b in phases.values() for r in _live_rows(b)
           if r[IN] > 0 and r[OUT] > 0 and r[LAT] > 0]
    if len(pts) < MIN_FIT_N:
        return None
    ins = [p[0] for p in pts]
    if max(ins) / min(ins) < MIN_FIT_SPREAD:
        return None
    # OLS on ln(lat) = a + b_in·ln(in) + b_out·ln(out), via 3×3 normal equations.
    rows = [(1.0, math.log(i), math.log(o)) for i, o, _ in pts]
    ys = [math.log(l) for _, _, l in pts]
    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    xty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(3)]
    # Gaussian elimination with partial pivoting.
    m = [xtx[i] + [xty[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return tuple(m[i][3] / m[i][i] for i in range(3))


def curve_p50(coeffs: tuple, in_e: int, out_e: int) -> float:
    a, b_in, b_out = coeffs
    return math.exp(a + b_in * math.log(max(in_e, 1)) + b_out * math.log(max(out_e, 1)))


def capacity_knee(buffers: List[List[list]]) -> Optional[int]:
    """Deployment capacity knee from concurrency buckets.

    Takes a list of buffers (all phases, all models resolving to one deployment),
    buckets concurrency by BUCKET_WIDTH, and returns the highest observed concurrency
    in the highest healthy bucket. Saturation runs upward: a sick bucket disqualifies
    itself and anything the walk has not yet reached below it stays claimable, but
    timeouts at LOW inflight (drain tails, single slow calls) say nothing about
    capacity and must not poison healthy evidence above them. Never extrapolates
    above observed values.
    """
    rows = [r for b in buffers for r in b if r[CONC] > 0]
    buckets: Dict[int, list] = {}
    for r in rows:
        buckets.setdefault(r[CONC] // BUCKET_WIDTH, []).append(r)
    for key in sorted(buckets, reverse=True):
        grp = buckets[key]
        if len(grp) < MIN_BUCKET_N:
            continue
        if sum(1 for r in grp if r[TIMED_OUT]) / len(grp) <= MAX_TIMEOUT_RATE:
            return max(r[CONC] for r in grp)
    return None


# Shared instance: one store per process, like token_tracker in llm.py.
perf_model = PerfModel()
