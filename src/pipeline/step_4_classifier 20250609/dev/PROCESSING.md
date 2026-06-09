# Step 4 — Processing

Source of truth: the code in `classifier.py` (P1-P7) and `cross_domain_consolidator.py` (P8).
Last verified against code: 2026-06-07

## Contract

Principles for how processing works in this step. Updating code = updating this doc.

### 1. Goal

Optimize the processing of prompts across providers (OpenAI, Azure), models (nano, default), and constraints (RPM, TPM, server-side queue, latency), where the binding bottleneck varies by deployment.

### 2. SmoothRequester for all LLM calls

Every phase (P1-P8) dispatches LLM calls through `SmoothRequester.process_all()` with `prepare_fn`/`parse_fn`/`fallback_fn` callbacks. No direct `_llm_call`, no hand-rolled rate limiting. SmoothRequester handles rate pacing (TokenBucket + AsyncLimiter), concurrency control (header-aware System A or client-side System B), retry pass, and empirical stats caching.

### 3. Empirical cold-start calibration

SmoothRequester persists P50 latency and avg tokens per phase_key to `model_perf_stats.json`. On subsequent runs, stored data seeds timeout calculations. First-ever run uses `default_timeout` (60s for discovery/consolidation, 10s for assignment).

Empirical server capacity (concurrency limit) is only saved when the concurrency controller's state machine actually found a ceiling (exited `RAMP_UP`). For small-task phases that never detect pressure, no capacity is saved and the next run cold-starts at `COLD_START_CAP` (50).

### 4. Adaptive timeout multiplier

`min(6, round(ln(num_tasks) + 1))` — few tasks get tight timeouts (P50 is reliable), many tasks get generous headroom for the outlier tail.

### 5. Retry for failures

SmoothRequester provides a built-in retry pass with reduced concurrency. Timed-out and failed tasks are re-queued automatically. Permanently failed tasks get `__UNASSIGNED__` fallback.

### 6. Auto-assignment for trivial cases

Single-facet domains skip P3 (facet assignment). Single-attribute facets skip P6 (attribute assignment). Deterministic assignments don't need LLM calls.

### 7. Model-tier-aware output handling

Uses instructor + Pydantic validation. Multiple models per pipeline — 8 model config keys across 8 internal phases (`qr_model_p1`–`qr_model_p8`).

### 8. Documentation tracks implementation

This PROCESSING.md reflects what the code does now, not what we plan to do. When we fix a gap, we update the code and this doc in the same commit.

### 9. Development code stays clean

This is development, not production. No legacy references, no backward compatibility shims, no dead or redundant code. If something is replaced, the old version is deleted.

## Processing

### Overview

Step 4 builds a hierarchical taxonomy (facets + attributes) from extracted ideas and assigns them to all responses. 6-phase pipeline with SmoothRequester dispatch. Each phase gets its own SR instance.

- **Input:** `List[IdeasExtractedModel]` (from step 3)
- **Output:** Taxonomy with facets, attributes, and per-idea assignments
- **Models:** 8 config keys: `_model_p1` … `_model_p8`
- **Provider:** OpenAI or Azure, abstracted by `llm.py` + SmoothRequester
- **Dispatch:** `SmoothRequester.process_all()` with prepare/parse/fallback callbacks

---

### Processing Strategy

Six display phases (internal P1-P8), alternating between concurrent discovery and concurrent consolidation:

### Phase 1: Facet Discovery + Consolidation (P1 + P2)

**P1**: Flatten all (domain, chunk) pairs into a task list. SmoothRequester dispatches all concurrently. Each task sends a chunk of observations to discover facet candidates. The domain's `boundary_test` + `exclusions` (persisted by step 3) are injected so facets stay within the domain boundary.

**P2**: Flatten all domain consolidation calls into a task list. SmoothRequester dispatches concurrently. Multi-round: if chunks exceed capacity thresholds, pre-group into round-1 tasks, then check if round 2 needed.

### Phase 2: Facet Assignment + Valence (P3)

**Auto-assign**: domains with 1 facet → all ideas assigned directly, no LLM call. Valence unchanged (keeps step 3 value).

**Multi-facet**: flatten all (domain, batch) pairs. SmoothRequester dispatches with `parse_fn` validation (ID drift, content similarity, facet ID mapping). Returns facet_id, confidence, and valence (+, -, 0). 10 ideas per batch.

### Phase 3: Attribute Discovery + Consolidation (P4 + P5)

Same pattern as Phase 1 but per-facet instead of per-domain. P4 discovers attribute candidates from chunks. P5 consolidates per-facet with multi-round support.

### Phase 4: Attribute Assignment + Valence (P6)

**Auto-assign**: facets with 1 attribute → all ideas assigned directly. Valence unchanged (keeps P3 value).

**Multi-attribute**: same pattern as Phase 2. Returns attribute_id, confidence, and valence (+, -, 0). `parse_fn` includes single-attribute fallback for invalid attribute IDs. P6 valence overwrites P3 valence in the growing model.

### Phase 5: Cross-facet Attribute Consolidation (P7)

One task per domain. SmoothRequester dispatches all domains concurrently. Post-processing rebuilds facet→attribute structure and remaps assignments.

### Phase 5.5: Valence-neutral Attribute Merge (P7.5)

Runs in `valence_consolidator.py` after P7, before P8. Deterministically detects attribute pairs within a facet that differ only in valence (near-identical labels + opposite valence skew) and merges the safe ones (auto-safe + single-token diff) into one descriptive attribute. The neutral merged name/description comes from a single direct LLM call (`classifier_p7` model — not SmoothRequester, only 0-N tiny tasks) with a deterministic single-token fallback; idea reassignment preserves valence/confidence. Overwrites the cache; cost-tracked as `p7_5_valence_merge`. Typically a no-op once the P4/P5/P7 prompts prevent the split at the source.

### Phase 6: Cross-domain Attribute Consolidation (P8)

Runs in `cross_domain_consolidator.py` after P7, on the just-cached `taxonomy` / `taxonomy_classified`. Embeds all ideas → attribute centroids → seriates into a 1D order → sliding windows of ~10 attributes → one LLM merge task per window (SmoothRequester, concurrent). The merge map (keyed by `(domain, attribute_name)`) is applied to both the cache and the growing model, carrying valence/confidence and keeping `partition_name`/`domain` in sync. `_verify_consistency()` then checks idea count, valence/confidence preservation, and orphan assignments, printing `P8 consistency: OK` or warnings. P8 overwrites the P7 cache in place and is skipped when fewer than 2 attributes exist. See [CONSOLIDATION_LOGIC.md](CONSOLIDATION_LOGIC.md).

---

## SmoothRequester Integration

### Shared pattern (all phases)

```python
requester = SmoothRequester(
    model=self._model_XX,
    dataset_key=self._dataset_key,
    phase_key="step4_pN_description",
    num_tasks=len(tasks),
    verbose=verbose,
    known_limits=self._fetched_limits,  # skip probe
    show_setup=False,                    # caller prints setup once
    quiet=True,                          # caller builds verbose report
    default_timeout=60.0,               # cold-start (discovery/consolidation)
)
results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
```

### Rate limit fetching

`_initialize_async_resources()` calls `llm.fetch_rate_limits(model)` once at startup. The result is stored as `self._fetched_limits` and passed to all SmoothRequester instances via `known_limits`, eliminating per-phase probe calls.

### Phase keys for `model_perf_stats.json`

`step4_p1_facet_discovery`, `step4_p2_facet_consolidation`, `step4_p3_facet_assignment`, `step4_p4_attribute_discovery`, `step4_p5_attribute_consolidation`, `step4_p6_attribute_assignment`, `step4_p7_attribute_consolidation`, `step4_p8_cross_domain_consolidation`

### Per-step verbosity

All 8 phases pass the `phase` key to `get_reasoning_params()`, enabling `STEP_VERBOSITY` overrides from `config.py`. P1/P2/P4/P5/P7/P8 (discovery + consolidation) use `"low"` verbosity to save tokens alongside scratchpad-driven chain-of-thought. P3/P6 (assignment) fall back to the default `TEXT_VERBOSITY` (`"medium"`).

---

## Divergent Paths

### Models per phase

| Phase | Model config | Typical use |
|---|---|---|
| P1 | `_model_p1` | Facet discovery (mini tier) |
| P2 | `_model_p2` | Facet consolidation (default tier) |
| P3 | `_model_p3` | Facet assignment (nano for speed) |
| P4 | `_model_p4` | Attribute discovery (mini tier) |
| P5 | `_model_p5` | Attribute consolidation (default tier) |
| P6 | `_model_p6` | Attribute assignment (nano for speed) |
| P7 | `_model_p7` | Cross-facet consolidation (default tier) |
| P8 | `_model_p8` | Cross-domain consolidation (default tier) |

All use instructor + Pydantic (Pattern B).

### OpenAI vs Azure

Provider abstraction handled by `llm.py`. Rate limit fetching via `llm.fetch_rate_limits()`. SmoothRequester uses `create_client()` internally.

---

## Configuration Reference

### SmoothRequester parameters per phase

| Phase | default_timeout | Model tier | Typical tasks |
|---|---|---|---|
| P1 | 60.0s | default | 6-40 (chunks × domains) |
| P2 | 60.0s | default | 5-10 (1 per domain) |
| P3 | 10.0s | nano | 20-100 (batches × domains) |
| P4 | 60.0s | default | 8-40 (chunks × facets) |
| P5 | 60.0s | default | 8-20 (1 per facet) |
| P6 | 10.0s | nano | 20-100 (batches × facets) |
| P7 | 60.0s | default | 5-10 (1 per domain) |
| P8 | 60.0s | default | ~4 (1 per sliding window) |
