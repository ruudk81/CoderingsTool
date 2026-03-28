# Step 4: Taxonomy Classifier — Processing Reference

Reference for debugging and improvement. Source of truth: the code in `classifier.py`.

Last verified against code: 2026-03-28

---

# I. Contract

Principles we agreed on for how prompt processing works in this step. Updating code = updating this doc. Commit both together.

### 1. Goal

Optimize the processing of prompts across providers (OpenAI, Azure), models (nano, mini, default), and constraints (RPM, TPM, server-side queue, latency), where the binding bottleneck varies by deployment.

### 2. Four-layer rate-limiting stack

Every API call passes through: ConcurrencyGate -> TokenBucket -> AsyncLimiter -> Timeout. No bypasses, no shortcuts. Small phases gracefully skip layers 2-4 when `PhaseRampState` layers are None.

### 3. Self-tuning cold start

After each phase, persist empirical stats (P50/P95 latency, avg tokens) to `model_perf_stats.json`. On next cold start, load stored stats per phase key. First-ever run falls back to model-tier defaults.

### 4. Warm-up then ramp

Per-phase warm-up. Start conservatively, calibrate from real completions, then ramp toward optimal. Completion-based ramp with congestion detection governs concurrency per phase.

### 5. Generous timeouts, retry for true failures

Timeouts are a safety net. Timed-out tasks trigger circuit breaker tracking. No dedicated retry pass — timeouts are permanent fallback. Inline 2-attempt retry for exceptions in some phases.

### 6. Model-tier-aware output handling

Uses instructor + Pydantic validation (Pattern B). Multiple models per pipeline — up to 7 different model assignments across phases.

### 7. Documentation tracks implementation

This PROCESSING.md reflects what the code does now, not what we plan to do. Known gaps go in section E. When we fix a gap, we update the code and this doc in the same commit.

### 8. Development code stays clean

This is development, not production. No legacy references, no backward compatibility shims, no dead or redundant code. If something is replaced, the old version is deleted. Clean code now is easier to promote to production later.

---

# II. Processing

## A. Overview

Step 4 builds a hierarchical taxonomy (facets + attributes) from extracted ideas and assigns them to all responses. Multi-phase pipeline with gather-based dispatch. Each phase gets independent rate-limiting state.

- **Input:** `List[IdeasExtractedModel]` (from step 3)
- **Output:** Taxonomy with facets, attributes, and per-idea assignments
- **Models:** Up to 7 models (`_model_p1` through `_model_p7`), configured per phase
- **Provider:** OpenAI or Azure, switched via `API_PROVIDER`
- **Dispatch:** `asyncio.gather` with background `_phase_monitor()` (not worker/queue)

---

## B. Processing strategy

Seven phases, alternating between concurrent discovery/assignment and sequential consolidation:

### P1: Per-domain Facet Discovery (concurrent)

Discover facets within each domain by processing chunks of ideas via gather. Each domain's ideas split into chunks → concurrent LLM calls → collect facet candidates.

### P2: Facet Consolidation (sequential)

Hierarchical merge of facet candidates per domain. Single-threaded — not dispatched via `_run_with_ramp`. Deduplicates and normalizes discovered facets.

### P3: Per-domain Facet Assignment (concurrent)

Assign facets to all ideas in batches (10 ideas per API call). Gather-based dispatch with full 4-layer stack. Batch-level granularity.

### P4: Per-facet Attribute Discovery (concurrent)

Discover attributes within each facet by processing chunks of assigned ideas. Same gather pattern as P1.

### P5: Attribute Chunk Consolidation (sequential)

Hierarchical merge of attribute candidates per facet. Single-threaded.

### P6: Per-facet Attribute Assignment (concurrent)

Assign attributes to all ideas in batches. Same gather pattern as P3.

### P7: Cross-facet Attribute Consolidation (sequential)

Final consolidation per domain. Single-threaded.

**Only P1, P3, P4, P6 use `_run_with_ramp()`** with full rate limiting. P2, P5, P7 are sequential single-call phases.

---

## C. Rate-limiting machinery

### Per-phase state

Each concurrent phase gets a `PhaseRampState` dataclass containing:
- `gate` (ConcurrencyGate), `ramp` (ConcurrencyRamp), `rpm_tracker`, `tpm_tracker`
- `token_bucket`, `latency_tracker`, `circuit_breaker` (None for small phases in "light" mode)
- `actual_total_tokens` deque, warm-up tracking

Created via `_create_phase_ramp(phase_name, num_tasks, model, phase_key)`. Phases run sequentially so there is no cross-phase contention.

### Dispatch: `_run_with_ramp()`

```python
async def _run_with_ramp(coros, state):
    async def _work():
        results = await asyncio.gather(*coros, return_exceptions=True)
        state.done = True
        return results
    results, _ = await asyncio.gather(_work(), _phase_monitor(state))
    return results
```

All coroutines dispatched at once. `_phase_monitor()` runs in parallel, evaluating ramp/PID/circuit-breaker every 0.5s until `state.done`.

### Request flow (per LLM call via `_llm_call`)

```
async with concurrency_gate:                    # Layer 1: ConcurrencyGate
    effective_timeout = latency_tracker.get_timeout()  # Adaptive timeout
    await token_bucket.wait_and_acquire(est)     # Layer 2: TokenBucket (if full mode)
    async with rate_limiter:                     # Layer 3: AsyncLimiter (RPM)
        result = await asyncio.wait_for(         # Safety net: Timeout
            api_call, timeout=effective_timeout
        )
```

Small phases skip layers 2-4 gracefully (None checks).

### Background monitor (`_phase_monitor`)

Runs at 2 Hz (0.5s interval) while gather is in progress:

| Check | What it does |
|---|---|
| Circuit breaker | `check_and_adjust()` — evaluates timeout rate |
| Completion-based ramp | `ConcurrencyRamp.record_measurement()` — advances with progress |
| Warm-up calibration | One-shot after enough samples — recalculates Little's Law |
| PID adjustment | `_apply_pid_adjustment()` — continuous TPM-based arrival rate tuning |
| Progress report | Logs completions, concurrency, timeouts |

### Tiktoken offset learning

`TiktokenOffsetLearner` tracks delta between tiktoken and actual API token counts. Applies learned offset to improve token bucket pre-acquisition accuracy. Shared across phases.

---

## D. Divergent paths

### Models per phase

| Phase | Model config | Typical use |
|---|---|---|
| P1 | `_model_p1` | Facet discovery (mini/default) |
| P2 | `_model_p2` | Consolidation (mini/default) |
| P3 | `_model_p3` | Facet assignment (nano for speed) |
| P4 | `_model_p4` | Attribute discovery (mini/default) |
| P5 | `_model_p5` | Consolidation (mini/default) |
| P6 | `_model_p6` | Attribute assignment (nano for speed) |
| P7 | `_model_p7` | Final consolidation (mini/default) |

All use instructor + Pydantic (Pattern B).

### OpenAI vs Azure

Provider abstraction handled by `llm.py` via `llm_create_async()`. Rate limit fetching is provider-aware (same pattern as step 2).

---

## E. Known issues and divergences

### 1. No dedicated retry pass

Unlike steps 2/3/6, step 4 has no post-batch retry pass. Timed-out tasks get permanent fallback. Some phases have inline 2-attempt retry for exceptions, but not systematic.

### 2. Shared rate limiter across phases

`self._rate_limiter` (AsyncLimiter) is shared across all phases. Per-phase state manages gate/bucket/tracker independently, but RPM spacing is global. This means PID adjustments in one phase affect the next phase's starting arrival rate.

---

## F. Configuration reference

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| Timeout floor | 45s | Chunk processing default |
| `COLD_START_P95_MULTIPLIER` | 2.0 | `modelPerfStats.py` |
| Phase monitor interval | 0.5s | `_phase_monitor()` |
| Circuit breaker min tasks | Configurable | `cfg.circuit_breaker_min_tasks` |

### Per-phase state (`PhaseRampState`)

| Field | Purpose |
|---|---|
| `gate` | ConcurrencyGate for this phase |
| `ramp` | ConcurrencyRamp (50% → 90% of Little's Law) |
| `token_bucket` | TPM safety rail (None for light mode) |
| `latency_tracker` | Adaptive timeout (None for light mode) |
| `circuit_breaker` | Timeout rate monitoring (None for light mode) |
| `actual_total_tokens` | deque(maxlen=100) for warm-up calibration |

### Phase keys for `model_perf_stats.json`

`step4_p1_facet_discovery`, `step4_p3_facet_assignment`, `step4_p4_attribute_discovery`, `step4_p6_attribute_assignment`, `step4_p7_consolidation`

### Shared dataclasses

Same `RampUpConfig`, `CircuitBreakerConfig`, `PIDControllerConfig`, `TPMTrackingConfig` as steps 2/3.
