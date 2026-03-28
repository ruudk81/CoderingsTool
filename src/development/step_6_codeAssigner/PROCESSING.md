# Step 6: Code Assigner — Processing Reference

Reference for debugging and improvement. Source of truth: the code in `code_assignment.py`.

Last verified against code: 2026-03-28

---

# I. Contract

Principles we agreed on for how prompt processing works in this step. Updating code = updating this doc. Commit both together.

### 1. Goal

Optimize the processing of prompts across providers (OpenAI, Azure), models (nano, mini, default), and constraints (RPM, TPM, server-side queue, latency), where the binding bottleneck varies by deployment.

### 2. Four-layer rate-limiting stack

Every API call passes through: ConcurrencyGate -> TokenBucket -> AsyncLimiter -> Timeout. No bypasses, no shortcuts. Retry pass uses the same layers.

### 3. Self-tuning cold start

After each run, persist empirical stats to `model_perf_stats.json`. On next cold start, load stored stats for timeout floor and token estimation.

### 4. Warm-up then ramp

Start conservatively, calibrate from real completions, then ramp toward optimal. Completion-based ramp with congestion detection governs concurrency after warm-up. Dynamic worker spawning post-calibration.

### 5. Generous timeouts, retry for true failures

Timeouts are a safety net. Timed-out tasks collected for fallback retry pass. Tenacity @retry (5 attempts, exponential backoff) for transient API errors.

### 6. Model-tier-aware output handling

Uses instructor + Pydantic validation (Pattern B). Single assignment model, with tier-aware field validators for nano vs mini/default.

### 7. Documentation tracks implementation

This PROCESSING.md reflects what the code does now, not what we plan to do. Known gaps go in section E.

### 8. Development code stays clean

This is development, not production. No legacy references, no backward compatibility shims, no dead or redundant code.

---

# II. Processing

## A. Overview

Step 6 assigns codes from the codebook to all extracted ideas. Batch-level processing (multiple ideas per API call). Worker/queue dispatch with a single phase.

- **Input:** Ideas from step 3, codebook from step 5, taxonomy from step 4
- **Output:** Per-idea code assignments (dual assignment: facet + attribute codes)
- **Model:** `config.assignment_model` — typically nano for speed, supports mini/default
- **Provider:** OpenAI or Azure
- **Dispatch:** `asyncio.Queue` + worker pool (same pattern as steps 2/3)

---

## B. Processing strategy

Single processing phase with warm-up:

### Phase 1: Fetch API rate limits

Minimal API call to read rate limit headers.

### Phase 2: Initialize rate limiters

Full 4-layer stack initialization:
- Arrival rate from `min(RPM/60, TPM/60/avg_tokens)` with headroom
- Cold-start concurrency from `min(RPM/60, 50)`, capped at task count
- Little's Law target from latency estimate + token estimate
- ConcurrencyRamp (50% → 90% of Little's Law)
- Circuit breaker, PID controller, TPM/RPM trackers

Cold-start inputs from persistent stats when available.

### Phase 3: Main batch processing

Queue + workers dispatch. Main loop at 10 Hz:

| Check | Frequency | What it does |
|---|---|---|
| Circuit breaker | Every tick | Evaluates timeout rate |
| Completion-based ramp | Every tick | ConcurrencyRamp advances with progress |
| Warm-up calibration | Once (after 15-30 completions) | Recalibrates, spawns extra workers |
| PID adjustment | Periodic | TPM-based arrival rate tuning |
| Progress report | Every 2s | Logs rate, concurrency, latency |

### Phase 4: Retry pass

After main batch, timed-out tasks requeued with:
- Reduced concurrency (10% of workers, min 5)
- Dedicated retry queue and workers
- Permanent fallback if still failed

### Phase 5: Stats persistence

Persists measured stats to `model_perf_stats.json`.

---

## C. Rate-limiting machinery

### Request flow (per batch)

```
async with self._gate:                          # Layer 1: ConcurrencyGate
    timeout = self._latency_tracker.get_timeout()
    await self._tpm_bucket.wait_and_acquire()    # Layer 2: TokenBucket
    async with self._rate_limiter:               # Layer 3: AsyncLimiter (RPM)
        result = await asyncio.wait_for(         # Safety net: Timeout
            api_call, timeout=timeout
        )
```

### Layers

Same 4-layer stack as steps 2/3. ConcurrencyGate, TokenBucket, AsyncLimiter, LatencyTracker, ConcurrencyCircuitBreaker.

### Ramp

`ConcurrencyRamp` — completion-based, 50% → 90% of Little's Law. Same congestion detection as step 3.

### PID controller

`PIDThroughputController` — same asymmetric gains, targets 80% TPM utilization.

### Retry

Tenacity `@retry` decorator on the core processing method: 5 attempts, exponential backoff with jitter (2-60s). Catches RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, InstructorRetryException.

---

## D. Divergent paths

### Model tier handling

Single model (`assignment_model`). Tier-aware field validators in Pydantic response model:
- **Mini/default:** Strict validation (raise on invalid)
- **Nano:** Coerce rather than reject

### Batch processing

Unlike steps 2/3 (one response per API call), step 6 processes batches of ideas per call. The unit of failure/retry is the batch, not the individual idea.

---

## E. Known issues and divergences

### 1. ConcurrencyRamp instead of signal-based ramp

Same as step 3 — uses completion-based `ConcurrencyRamp`, not the 4-signal ramp that step 2 was migrated to.

---

## F. Configuration reference

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| Timeout floor | From stored P95 x 2, or model default | Cold-start stats |
| `COLD_START_P95_MULTIPLIER` | 2.0 | `modelPerfStats.py` |
| Phase key | `step6_code_assignment` | `model_perf_stats.json` lookup |
| Tenacity retries | 5 attempts | `@retry` decorator |
| Retry backoff | 2-60s exponential with jitter | `wait_exponential_jitter` |

### Shared dataclasses

Same `RampUpConfig`, `CircuitBreakerConfig`, `PIDControllerConfig`, `TPMTrackingConfig` as steps 2/3.
