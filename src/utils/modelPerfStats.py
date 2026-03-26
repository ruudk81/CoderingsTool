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
        "avg_tokens": float,
        "tiktoken_offset": float,   # optional — only written by steps that learn it
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

# Fields that may appear in a measurement dict.
_NUMERIC_FIELDS = ("p50_latency_s", "p95_latency_s", "avg_tokens", "tiktoken_offset")


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

    entry["sample_count"] = old_count + n_new_samples
    entry["last_updated"] = date.today().isoformat()


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

    if "p95_latency_s" in entry:
        if hasattr(ramp_config, "timeout_floor_seconds"):
            ramp_config.timeout_floor_seconds = entry["p95_latency_s"]
        if hasattr(ramp_config, "default_timeout_seconds"):
            ramp_config.default_timeout_seconds = entry["p95_latency_s"]

    if "avg_tokens" in entry and hasattr(ramp_config, "estimated_avg_tokens"):
        ramp_config.estimated_avg_tokens = int(entry["avg_tokens"])
