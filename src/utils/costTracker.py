"""
Cost Tracker — persistent per-dataset JSON ledger for LLM costs.

Accumulates per-step, per-phase token usage and cost data in
exports/costs/{dataset}_{var}_{sample}_kosten.json.

Usage:
    from utils.costTracker import CostTracker
    from utils.llm import token_tracker

    tracker = CostTracker(filename="data.sav", var_name="Q1", sample_size=500)

    # Before a phase
    snap_before = token_tracker.snapshot()
    # ... run LLM calls ...
    snap_after = token_tracker.snapshot()

    tracker.record_phase("step_3_idea_extraction", "bulk_extraction",
                         snap_before, snap_after, model="gpt-5.4-nano")
    tracker.finalize_step("step_3_idea_extraction")

Wiring a step
-------------
The runner creates the tracker and finalizes; the worker class records phases.

    # run_<step>.py
    tracker = CostTracker(filename=config.filename, var_name=config.var_name,
                          sample_size=config.sample_size)
    worker = Worker(..., cost_tracker=tracker)
    ...
    tracker.finalize_step("step_2_quality_filter")

    # <step>.py — accept cost_tracker=None, register models, snapshot per phase
    self.cost_tracker.set_step_models("step_2_quality_filter", {"grading": self.model})

Conventions
-----------
- Snapshot placement: take `snap_before` immediately before the LLM calls, not at
  the top of the method — otherwise setup and pre-filtering token usage is counted
  as part of the phase.
- Guard every call with `if self.cost_tracker`. The parameter is optional, so the
  worker stays usable in tests and standalone runs that do not persist cost.
- One phase per logical LLM batch. A step with several passes (step 3: context,
  taxonomy, bulk extraction) records each separately; `finalize_step()` sums them.
- Naming: step name matches the pipeline step identity (`step_2_quality_filter`),
  phase name says what the LLM is doing (`grading`). Both become JSON keys, so keep
  them stable across runs or the history fragments.
- CostTracker owns the file lifecycle: creates the file and the step/phase entries,
  overwrites phase data on re-run (idempotent), and writes atomically via `.tmp`
  rename. Several steps accumulate in one file, keyed by step name.
- Do NOT thread one tracker through `run_pipeline.py`. Each step runner constructs
  its own with the same filename + var_name + sample_size; the constructor loads
  the existing JSON, so steps accumulate without a shared instance.

Output format
-------------
    {
      "dataset": "data.sav",
      "var_name": "Q20",
      "sample_size": 500,
      "deployment": {"provider": "azure", "generations": "5.4+5.6"},
      "steps": {
        "step_2_quality_filter": {
          "model_config": {"grading": "gpt-5.4-nano"},
          "phases": {
            "grading": {"model": "gpt-5.4-nano", "input_tokens": 12345,
                        "output_tokens": 678, "cost_usd": 0.0042, "calls": 500}
          },
          "total": {"input_tokens": 12345, "output_tokens": 678,
                    "cost_usd": 0.0042, "calls": 500},
          "date": "2026-04-05"
        }
      }
    }
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional


class CostTracker:
    """Manages a JSON cost ledger for a dataset+variable combination."""

    def __init__(
        self,
        filename: str,
        var_name: str,
        sample_size,
        exports_dir: Optional[Path] = None,
    ):
        from config import ACTIVE_GENERATIONS, API_PROVIDER
        from utils.exportNaming import export_filename

        self._filename = filename
        self._var_name = var_name
        self._sample_size = sample_size
        self._provider = API_PROVIDER
        self._generations = ACTIVE_GENERATIONS

        if exports_dir is None:
            exports_dir = Path(__file__).parent.parent.parent / "exports" / "costs"
        exports_dir.mkdir(parents=True, exist_ok=True)

        self._json_path = exports_dir / export_filename(
            filename, var_name, sample_size, "kosten", "json")
        self._data = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_step_models(self, step: str, models: dict) -> None:
        """Store model config for a step (e.g. context, taxonomy, abstraction_ladder)."""
        step_data = self._ensure_step(step)
        step_data["model_config"] = models

    def record_phase(
        self,
        step: str,
        phase: str,
        snapshot_before: dict,
        snapshot_after: dict,
        model: str,
    ) -> None:
        """Record cost delta for a phase. Overwrites existing phase data on re-run."""
        delta = {
            "model": model,
            "input_tokens": snapshot_after["input_tokens"] - snapshot_before["input_tokens"],
            "output_tokens": snapshot_after["output_tokens"] - snapshot_before["output_tokens"],
            "cost_usd": round(snapshot_after["cost_usd"] - snapshot_before["cost_usd"], 6),
            "calls": snapshot_after["calls"] - snapshot_before["calls"],
        }
        step_data = self._ensure_step(step)
        step_data["phases"][phase] = delta

    def finalize_step(self, step: str) -> None:
        """Sum all phases into step total, set date, and write to disk."""
        step_data = self._ensure_step(step)
        phases = step_data.get("phases", {})

        total_input = sum(p["input_tokens"] for p in phases.values())
        total_output = sum(p["output_tokens"] for p in phases.values())
        total_cost = sum(p["cost_usd"] for p in phases.values())
        total_calls = sum(p["calls"] for p in phases.values())

        step_data["total"] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(total_cost, 6),
            "calls": total_calls,
        }
        step_data["date"] = date.today().isoformat()
        self._save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_step(self, step: str) -> dict:
        """Get or create a step entry in the data."""
        steps = self._data["steps"]
        if step not in steps:
            steps[step] = {"phases": {}}
        return steps[step]

    def _load(self) -> dict:
        """Load existing JSON or return a new template."""
        if self._json_path.exists():
            try:
                with open(self._json_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "dataset": self._filename,
            "var_name": self._var_name,
            "sample_size": self._sample_size,
            "deployment": {
                "provider": self._provider,
                "generations": self._generations,
            },
            "steps": {},
        }

    def _save(self) -> None:
        """Atomic write: write to .tmp then rename."""
        tmp_path = self._json_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp_path.replace(self._json_path)
