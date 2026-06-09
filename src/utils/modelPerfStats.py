"""
Persistent model performance stats — cross-run cold-start calibration.

Stats are stored in data/model_perf_stats.json (gitignored, machine-specific).
Each entry records empirical latency and token measurements per (model, phase).

On cold start: load stored stats and override config defaults (estimated_latency,
estimated_avg_tokens, timeout_floor) so each run starts from measured reality
instead of hardcoded guesses.

After each run: update stats with an EMA so values drift toward recent experience
without discarding history. New models get a fresh entry; unknown phases use
hardcoded config defaults unchanged.

Schema:
{
  "version": 1,
  "stats": {
    "<model_name>": {
      "<phase_key>": {
        "p50_latency_s": float,
        "p95_latency_s": float,
        "p99_latency_s": float,
        "avg_tokens": float,
        "tiktoken_offset": float,   # optional — only written by steps that learn it
        "timeout_rate": float,      # optional — fraction of tasks that timed out (0.0–1.0)
        "had_timeouts": bool,       # optional — whether previous run had any timeouts
        "empirical_capacity": float, # optional — measured server concurrency (throughput × P50)
        "sample_count": int,
        "last_updated": "YYYY-MM-DD"
      }
    }
  }
}

Phase keys:
  step2_quality_filter
  step3_idea_extraction
  step4_p1_facet_discovery
  step4_p3_facet_assignment
  step4_p4_attribute_discovery
  step4_p6_attribute_assignment
  step4_p7_consolidation
  step5_p8_codebook_generation
  step5_p9_consolidation
  step6_code_assignment
"""

import json
import os
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from config import API_PROVIDER

# Absolute path derived from this file's location (src/utils/ → project root → data/)
STATS_FILE = Path(__file__).parent.parent.parent / "data" / "model_perf_stats.json"

# Only apply stored stats when we have at least this many samples.
# Below this threshold the estimate may not be reliable.
MIN_SAMPLES = 10

# EMA stabilises at this weight once sample_count reaches 20.
_EMA_FLOOR_ALPHA = 0.05

# Empirical timeout = max(P99_MULTIPLIER × P99, TIMEOUT_FACTOR × P99)
# P99_MULTIPLIER: baseline multiplier, always applied
# TIMEOUT_FACTOR: applied only when previous run had timeouts (0 otherwise)
EMPIRICAL_P99_MULTIPLIER = 1.2
EMPIRICAL_TIMEOUT_FACTOR = 2.0

# Legacy: kept for backward compatibility with steps not yet migrated to P99
EMPIRICAL_P95_MULTIPLIER = 2.0

# Fields that may appear in a measurement dict (EMA-smoothed).
_NUMERIC_FIELDS = ("p50_latency_s", "p95_latency_s", "p99_latency_s", "avg_tokens",
                   "tiktoken_offset", "timeout_rate", "empirical_capacity")


def _model_key(model: str) -> str:
    """Build a provider-scoped model key — e.g. 'azure:gpt-4.1-mini'.

    Azure and OpenAI share model names but have completely different
    rate limits, latency profiles, and throughput characteristics.
    Stats must never be mixed across providers.
    """
    return f"{API_PROVIDER}:{model}"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_stats(filepath: Path = STATS_FILE) -> Dict[str, Any]:
    """Load stats from JSON. Returns empty dict on missing file or parse error."""
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "stats" not in data:
            print(f"[modelPerfStats] Unexpected format in {filepath} — starting fresh")
            return {"version": 1, "stats": {}}
        return data
    except FileNotFoundError:
        return {"version": 1, "stats": {}}
    except Exception as exc:
        print(f"[modelPerfStats] WARNING: could not read {filepath}: {exc} — starting fresh")
        return {"version": 1, "stats": {}}


def save_stats(stats: Dict[str, Any], filepath: Path = STATS_FILE) -> None:
    """Atomically write stats to JSON (write tmp → rename)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, filepath)
    except Exception as exc:
        print(f"[modelPerfStats] WARNING: could not save stats to {filepath}: {exc}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_phase_stats(
    stats: Dict[str, Any],
    model: str,
    phase_key: str,
) -> Optional[Dict[str, Any]]:
    """Return stored stats for (provider:model, phase_key), or None if not present."""
    return stats.get("stats", {}).get(_model_key(model), {}).get(phase_key)


def get_dataset_phase_stats(
    stats: Dict[str, Any],
    model: str,
    phase_key: str,
    dataset_key: str,
) -> Optional[Dict[str, Any]]:
    """Return stored stats for (provider:model, phase_key, dataset_key).

    Used by step 3 where stats are scoped per dataset (filename:variable_key).
    Returns None if the phase or dataset entry doesn't exist.
    """
    phase = get_phase_stats(stats, model, phase_key)
    if phase and isinstance(phase, dict) and dataset_key in phase:
        entry = phase[dataset_key]
        if isinstance(entry, dict) and "sample_count" in entry:
            return entry
    return None


def get_phase_prior(
    stats: Dict[str, Any],
    model: str,
    phase_key: str,
    exclude_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Synthesize a model+phase 'warm' prior from sibling dataset cases.

    Aggregates (median) the infra/timing fields across all other dataset entries
    under the same (provider:model, phase) that have >= MIN_SAMPLES samples.
    Returns a synthetic stats dict, or None if no qualifying siblings exist.

    In-memory only — never persisted. Lets a new case reuse the measured
    concurrency ceiling + latency instead of cold-starting from scratch.
    """
    phase = get_phase_stats(stats, model, phase_key)
    if not isinstance(phase, dict):
        return None
    siblings = [
        e for k, e in phase.items()
        if k != exclude_key and isinstance(e, dict)
        and e.get("sample_count", 0) >= MIN_SAMPLES
    ]
    if not siblings:
        return None

    prior: Dict[str, Any] = {}
    for field in ("empirical_capacity", "p50_latency_s", "avg_tokens", "tiktoken_offset"):
        vals = [e[field] for e in siblings if field in e]
        if vals:
            prior[field] = statistics.median(vals)

    # has_server_headers is a model property: True if any sibling saw headers.
    headers = [e["has_server_headers"] for e in siblings if "has_server_headers" in e]
    if headers:
        prior["has_server_headers"] = any(headers)

    if not prior:
        return None
    # sample_count set to MIN_SAMPLES so downstream >= MIN_SAMPLES gates pass.
    prior["sample_count"] = MIN_SAMPLES
    return prior


def get_dataset_phase_stats_or_prior(
    stats: Dict[str, Any],
    model: str,
    phase_key: str,
    dataset_key: str,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Tiered lookup: exact case (hot) -> model+phase prior (warm) -> None (cold).

    Returns (entry, origin) with origin in {'hot', 'warm', 'cold'}.
    """
    exact = get_dataset_phase_stats(stats, model, phase_key, dataset_key)
    if exact is not None:
        return exact, "hot"
    prior = get_phase_prior(stats, model, phase_key, exclude_key=dataset_key)
    if prior is not None:
        return prior, "warm"
    return None, "cold"


# ---------------------------------------------------------------------------
# Update (EMA)
# ---------------------------------------------------------------------------

def update_phase_stats(
    stats: Dict[str, Any],
    model: str,
    phase_key: str,
    measurements: Dict[str, float],
    n_new_samples: int,
) -> None:
    """Update stored stats with new measurements using an EMA.

    measurements: dict with any of: p50_latency_s, p95_latency_s, avg_tokens,
                  tiktoken_offset.
    n_new_samples: number of completions this measurement is based on.
    """
    if n_new_samples <= 0:
        return

    model_stats = stats.setdefault("stats", {}).setdefault(_model_key(model), {})
    entry = model_stats.setdefault(phase_key, {"sample_count": 0})

    old_count = entry.get("sample_count", 0)
    # Alpha converges quickly when data is sparse, stabilises at floor once mature.
    alpha = max(_EMA_FLOOR_ALPHA, 1.0 / min(old_count + n_new_samples, 20))

    for field in _NUMERIC_FIELDS:
        if field not in measurements:
            continue
        measured = float(measurements[field])
        if field in entry:
            entry[field] = round(alpha * measured + (1.0 - alpha) * entry[field], 4)
        else:
            entry[field] = round(measured, 4)

    # Boolean fields: overwrite (not EMA'd)
    if "had_timeouts" in measurements:
        entry["had_timeouts"] = bool(measurements["had_timeouts"])

    entry["sample_count"] = old_count + n_new_samples
    entry["last_updated"] = date.today().isoformat()


def update_dataset_phase_stats(
    stats: Dict[str, Any],
    model: str,
    phase_key: str,
    dataset_key: str,
    measurements: Dict[str, Any],
    n_new_samples: int,
    overwrite_fields: Optional[list] = None,
) -> None:
    """Update stats for a specific dataset within a phase.

    Used by step 3 where stats are scoped per dataset (filename:variable_key).
    Fields in `overwrite_fields` are written directly (not EMA'd) — used for
    empirical_capacity (last-run is best evidence) and bottleneck type.

    All other numeric fields are EMA'd as usual.
    """
    if n_new_samples <= 0:
        return

    overwrite = set(overwrite_fields or [])
    _ema_fields = ("p50_latency_s", "avg_tokens", "tiktoken_offset")

    model_stats = stats.setdefault("stats", {}).setdefault(_model_key(model), {})
    phase = model_stats.setdefault(phase_key, {})
    entry = phase.setdefault(dataset_key, {"sample_count": 0})

    old_count = entry.get("sample_count", 0)
    alpha = max(_EMA_FLOOR_ALPHA, 1.0 / min(old_count + n_new_samples, 20))

    for field in _ema_fields:
        if field not in measurements:
            continue
        measured = float(measurements[field])
        if field in entry:
            entry[field] = round(alpha * measured + (1.0 - alpha) * entry[field], 4)
        else:
            entry[field] = round(measured, 4)

    # Overwrite fields: written directly (e.g., empirical_capacity, bottleneck)
    for field in overwrite:
        if field in measurements:
            entry[field] = measurements[field]

    entry["sample_count"] = old_count + n_new_samples
    entry["last_updated"] = date.today().isoformat()

    # Bump version to 2 (dataset-scoped entries)
    stats["version"] = 2


# ---------------------------------------------------------------------------
# Apply to config (cold-start override)
# ---------------------------------------------------------------------------

def apply_to_ramp_config(
    stats: Dict[str, Any],
    model: str,
    phase_key: str,
    ramp_config: Any,
) -> None:
    """Override ClassifierRampConfig cold-start fields from stored stats.

    Only applied when sample_count >= MIN_SAMPLES. Silently skips missing
    entries so hardcoded defaults remain in effect for new models/phases.

    Fields overridden (if present in stored stats):
      ramp_config.estimated_latency_seconds  ← p50_latency_s
      ramp_config.timeout_floor_seconds      ← p95_latency_s (P95 is the natural floor)
      ramp_config.default_timeout_seconds    ← p95_latency_s
      ramp_config.estimated_avg_tokens       ← avg_tokens (cast to int)
    """
    entry = get_phase_stats(stats, model, phase_key)
    if entry is None or entry.get("sample_count", 0) < MIN_SAMPLES:
        return

    if "p50_latency_s" in entry and hasattr(ramp_config, "estimated_latency_seconds"):
        ramp_config.estimated_latency_seconds = entry["p50_latency_s"]

    if "p99_latency_s" in entry:
        # Timeout = max(1.2 × P99, timeoutFactor × P99)
        # timeoutFactor = 2 if previous run had timeouts, 0 otherwise
        p99 = entry["p99_latency_s"]
        timeout_factor = EMPIRICAL_TIMEOUT_FACTOR if entry.get("had_timeouts") else 0
        floor = max(EMPIRICAL_P99_MULTIPLIER * p99, timeout_factor * p99)
        if hasattr(ramp_config, "timeout_floor_seconds"):
            ramp_config.timeout_floor_seconds = floor
        if hasattr(ramp_config, "default_timeout_seconds"):
            ramp_config.default_timeout_seconds = floor
    elif "p95_latency_s" in entry:
        # Fallback for phases that don't yet persist P99
        floor = entry["p95_latency_s"] * EMPIRICAL_P95_MULTIPLIER
        if hasattr(ramp_config, "timeout_floor_seconds"):
            ramp_config.timeout_floor_seconds = floor
        if hasattr(ramp_config, "default_timeout_seconds"):
            ramp_config.default_timeout_seconds = floor

    if "avg_tokens" in entry and hasattr(ramp_config, "estimated_avg_tokens"):
        ramp_config.estimated_avg_tokens = int(entry["avg_tokens"])
