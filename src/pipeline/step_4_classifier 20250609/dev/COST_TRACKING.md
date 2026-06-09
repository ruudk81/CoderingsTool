# Step 4 — Cost Tracking

## Overview

Step 4 persists per-phase LLM costs to disk using `CostTracker` (`utils/costTracker.py`). Costs are saved to `exports/costs/{dataset_stem}_{variable_key}_costs.json`.

## Implementation Pattern

### Runner (`run_classifier.py`)

1. Create `CostTracker(filename=FILENAME, variable_key=variable_key)`
2. Pass `cost_tracker` to `TaxonomyClassifier()`
3. After processing: `cost_tracker.finalize_step("step_4_taxonomy_classifier")`

### Classifier (`classifier.py`)

1. Accept `cost_tracker=None` in `__init__()`
2. Register all 9 models (P1-P8 + P7.5): `cost_tracker.set_step_models("step_4_taxonomy_classifier", {"p1_facet_discovery": ..., ..., "p7_5_valence_merge": ..., "p8_cross_domain_consolidation": ...})`
3. Snapshot before/after each phase:
   - P1: facet discovery
   - P2: facet consolidation
   - P3: facet assignment
   - P4: attribute discovery
   - P5: attribute consolidation
   - P6: attribute assignment
   - P7: cross-facet attribute consolidation (in `_process_taxonomy_async()`)
   - P7.5: valence-neutral attribute merge (recorded in `ValenceConsolidator.consolidate()`; often 0 cost when there are no valence-split pairs)
   - P8: cross-domain attribute consolidation (recorded in `CrossDomainConsolidator.consolidate()`)

## Best Practices

### Snapshot placement

- Take `snap_before` immediately before the LLM calls, not at the start of `_process_taxonomy_async()`. This avoids capturing unrelated token usage from setup.
- Take `snap_after` via `token_tracker.snapshot()` immediately after the phase completes.

### Guard all cost tracking with `if self.cost_tracker`

The `cost_tracker` parameter is optional (`None` by default). Every call to `set_step_models()`, `snapshot()`, and `record_phase()` must be guarded. This keeps the TaxonomyClassifier usable without cost tracking (e.g., in tests or standalone runs that don't care about cost persistence).

### One phase per pipeline stage

Step 4 records 9 phase entries (the 8 internal phases P1-P8 plus the P7.5 valence merge). Each has its own cost snapshot; P7.5 reuses the `classifier_p7` model.

### Step and phase naming

- Step name: `"step_4_taxonomy_classifier"` — matches the pipeline step identity
- Phase names: `"p1_facet_discovery"`, `"p2_facet_consolidation"`, `"p3_facet_assignment"`, `"p4_attribute_discovery"`, `"p5_attribute_consolidation"`, `"p6_attribute_assignment"`, `"p7_cross_facet_consolidation"`, `"p7_5_valence_merge"`, `"p8_cross_domain_consolidation"`
- These become keys in the JSON output, so keep them stable across runs

### CostTracker owns the file lifecycle

- Creates the JSON file if it doesn't exist
- Creates step/phase entries on first write
- Overwrites phase data on re-run (idempotent)
- `finalize_step()` sums phases and writes to disk atomically (via `.tmp` rename)
- Multiple steps accumulate in the same JSON file (keyed by step name)

### Don't pass CostTracker from the pipeline runner

Each step runner creates its own `CostTracker` instance with the same `filename` + `variable_key`. Since `CostTracker` loads the existing JSON on init, steps naturally accumulate in the same file without needing a shared instance threaded through `run_pipeline.py`.

## Output Format

```json
{
  "dataset": "data.sav",
  "variable_key": "Q20_500",
  "deployment": { "provider": "openai", "model_family": "gpt-5.4" },
  "steps": {
    "step_4_taxonomy_classifier": {
      "model_config": {
        "p1_facet_discovery": "gpt-5.4",
        "p2_facet_consolidation": "gpt-5.4",
        "p3_facet_assignment": "gpt-5.4-nano",
        "p4_attribute_discovery": "gpt-5.4",
        "p5_attribute_consolidation": "gpt-5.4",
        "p6_attribute_assignment": "gpt-5.4-nano",
        "p7_cross_facet_consolidation": "gpt-5.4",
        "p7_5_valence_merge": "gpt-5.4",
        "p8_cross_domain_consolidation": "gpt-5.4"
      },
      "phases": {
        "p1_facet_discovery": {
          "model": "gpt-5.4",
          "input_tokens": 20000,
          "output_tokens": 4000,
          "cost_usd": 0.009,
          "calls": 14
        },
        "p2_facet_consolidation": {
          "model": "gpt-5.4",
          "input_tokens": 5000,
          "output_tokens": 1000,
          "cost_usd": 0.003,
          "calls": 4
        },
        "p3_facet_assignment": {
          "model": "gpt-5.4-nano",
          "input_tokens": 30000,
          "output_tokens": 4000,
          "cost_usd": 0.005,
          "calls": 50
        },
        "p4_attribute_discovery": {
          "model": "gpt-5.4",
          "input_tokens": 28000,
          "output_tokens": 6000,
          "cost_usd": 0.014,
          "calls": 18
        },
        "p5_attribute_consolidation": {
          "model": "gpt-5.4",
          "input_tokens": 7000,
          "output_tokens": 2000,
          "cost_usd": 0.004,
          "calls": 6
        },
        "p6_attribute_assignment": {
          "model": "gpt-5.4-nano",
          "input_tokens": 40000,
          "output_tokens": 5000,
          "cost_usd": 0.006,
          "calls": 60
        },
        "p7_cross_facet_consolidation": {
          "model": "gpt-5.4",
          "input_tokens": 8000,
          "output_tokens": 3000,
          "cost_usd": 0.004,
          "calls": 4
        },
        "p7_5_valence_merge": {
          "model": "gpt-5.4",
          "input_tokens": 0,
          "output_tokens": 0,
          "cost_usd": 0.0,
          "calls": 0
        },
        "p8_cross_domain_consolidation": {
          "model": "gpt-5.4",
          "input_tokens": 12000,
          "output_tokens": 4000,
          "cost_usd": 0.006,
          "calls": 4
        }
      },
      "total": {
        "input_tokens": 150000,
        "output_tokens": 29000,
        "cost_usd": 0.051,
        "calls": 160
      },
      "date": "2026-04-05"
    }
  }
}
```
