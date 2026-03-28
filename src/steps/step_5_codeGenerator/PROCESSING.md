# Step 5: Codebook Generator — Processing Reference

Reference for debugging and improvement. Source of truth: the code in `codebook_generator.py`.

Last verified against code: 2026-03-28

---

# I. Contract

Principles we agreed on for how prompt processing works in this step. Updating code = updating this doc. Commit both together.

### 1. Goal

Optimize the processing of prompts across providers (OpenAI, Azure), models (nano, mini, default), and constraints (RPM, TPM, server-side queue, latency), where the binding bottleneck varies by deployment.

### 2. Four-layer rate-limiting stack

Every API call passes through: ConcurrencyGate -> TokenBucket -> AsyncLimiter -> Timeout. Same `_llm_call` pattern as step 4. Small phases skip layers gracefully.

### 3. Self-tuning cold start

Per-phase stats persisted to `model_perf_stats.json`. Loaded on next cold start.

### 4. Warm-up then ramp

Per-phase warm-up. Completion-based ramp with congestion detection. Only applies to P8 (concurrent); P9 is a single call.

### 5. Generous timeouts, retry for true failures

Inline 2-attempt retry per phase. No dedicated retry pass. Second failure returns empty/fallback result.

### 6. Model-tier-aware output handling

Uses instructor + Pydantic validation (Pattern B). Two models (P8, P9).

### 7. Documentation tracks implementation

This PROCESSING.md reflects what the code does now, not what we plan to do. Known gaps go in section E.

### 8. Development code stays clean

This is development, not production. No legacy references, no backward compatibility shims, no dead or redundant code.

---

# II. Processing

## A. Overview

Step 5 generates a codebook from the taxonomy built in step 4. Per-domain code generation (concurrent) followed by cross-domain consolidation (single call).

- **Input:** Taxonomy with facets and attributes (from step 4)
- **Output:** Consolidated codebook with codes, descriptions, and hierarchical structure
- **Models:** `_model_p8` (code generation), `_model_p9` (consolidation)
- **Provider:** OpenAI or Azure
- **Dispatch:** `asyncio.gather` with background `_phase_monitor()` (reuses step 4's pattern)

---

## B. Processing strategy

Two phases:

### P8: Per-domain Code Generation (concurrent)

Generate codebook entries from attributes, per domain. All domains dispatched via `_run_with_ramp()` with full 4-layer stack. Each domain is one LLM call.

Inline 2-attempt retry: if the first call fails, logs error and retries once. Second failure returns empty result for that domain.

### P9: Cross-domain Codebook Consolidation (single call)

Single LLM call to merge and deduplicate codes across all domains. Not dispatched via gather — direct `_llm_call` with timeout.

Inline 2-attempt retry: same pattern as P8.

### Stats persistence

After both phases, persists measured latency and token stats per phase key to `model_perf_stats.json`.

---

## C. Rate-limiting machinery

### Reuses step 4's infrastructure

Imports `PhaseRampState` from step 4's `classifier.py`. Own implementations of `_create_phase_ramp`, `_phase_monitor`, `_run_with_ramp`, `_llm_call` — same patterns, adapted for 2 phases.

### Request flow

Same as step 4:

```
async with concurrency_gate:                    # Layer 1
    effective_timeout = latency_tracker.get_timeout()
    await token_bucket.wait_and_acquire(est)     # Layer 2 (if full mode)
    async with rate_limiter:                     # Layer 3
        result = await asyncio.wait_for(api_call, timeout)
```

### Background monitor

Same 0.5s interval monitor as step 4: circuit breaker, completion-based ramp, warm-up calibration, progress reporting.

### No PID controller

Step 5 does not have PID arrival rate adjustment. The arrival rate is set at initialization and only changes via warm-up calibration.

---

## D. Divergent paths

### Models

| Phase | Model | Use |
|---|---|---|
| P8 | `_model_p8` | Per-domain code generation |
| P9 | `_model_p9` | Cross-domain consolidation |

Both use instructor + Pydantic (Pattern B).

---

## E. Known issues and divergences

### 1. No PID controller

Unlike steps 2/3/4/6, step 5 has no PID arrival rate adjustment. For typical workloads (few domains = few tasks in P8), this is unlikely to matter — but for datasets with many domains it could lead to suboptimal throughput.

### 2. No dedicated retry pass

Same as step 4. Inline 2-attempt retry only. No post-batch systematic retry.

---

## F. Configuration reference

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| Default timeout | 180s | `_llm_call` default |
| Phase monitor interval | 0.5s | `_phase_monitor()` |

### Phase keys for `model_perf_stats.json`

`step5_p8_codebook_generation`, `step5_p9_consolidation`

### Shared infrastructure

`PhaseRampState` imported from step 4. Same `ConcurrencyGate`, `ConcurrencyRamp`, `TokenBucket`, `LatencyTracker`, `ConcurrencyCircuitBreaker` classes.
