"""
Cost Tracker — persistent per-dataset JSON ledger for LLM costs.

Accumulates per-step, per-phase token usage and cost data in
exports/costs/{dataset}_{variable_key}_costs.json.

Usage:
    from utils.costTracker import CostTracker
    from utils.llm import token_tracker

    tracker = CostTracker(filename="data.sav", variable_key="Q1_500")

    # Before a phase
    snap_before = token_tracker.snapshot()
    # ... run LLM calls ...
    snap_after = token_tracker.snapshot()

    tracker.record_phase("step_3_idea_extraction", "bulk_extraction",
                         snap_before, snap_after, model="gpt-5.4-nano")
    tracker.finalize_step("step_3_idea_extraction")
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
        variable_key: str,
        exports_dir: Optional[Path] = None,
    ):
        from config import API_PROVIDER, MODEL_FAMILY

        self._filename = filename
        self._variable_key = variable_key
        self._provider = API_PROVIDER
        self._model_family = MODEL_FAMILY

        if exports_dir is None:
            exports_dir = Path(__file__).parent.parent.parent / "exports" / "costs"
        exports_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        self._json_path = exports_dir / f"{stem}_{variable_key}_costs.json"
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
            "variable_key": self._variable_key,
            "deployment": {
                "provider": self._provider,
                "model_family": self._model_family,
            },
            "steps": {},
        }

    def _save(self) -> None:
        """Atomic write: write to .tmp then rename."""
        tmp_path = self._json_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp_path.replace(self._json_path)
