# Step 3: Idea Extractor — Processing Reference

Reference for debugging and improvement. Source of truth: the code in `ideaExtractor_exp.py`.

Last verified against code: 2026-03-28

---

# I. Contract

Principles we agreed on for how prompt processing works in this step. Updating code = updating this doc. Commit both together.

### 1. Goal

Optimize the processing of prompts across providers (OpenAI, Azure), models (nano, mini, default), and constraints (RPM, TPM, server-side queue, latency), where the binding bottleneck varies by deployment.

### 2. Four-layer rate-limiting stack

Every API call passes through: ConcurrencyGate -> TokenBucket -> AsyncLimiter -> Timeout. No bypasses, no shortcuts. Retry pass uses the same layers.

### 3. Self-tuning cold start

After each run, persist empirical stats (P50/P95 latency, avg tokens) to `model_perf_stats.json`. On next cold start, load stored stats and use them for Little's Law (P50), timeout floor (P95 x 2), and token estimation (avg tokens) — so each run starts from measured reality, not guesses. First-ever run (no stored stats) falls back to model-tier defaults.

### 4. Warm-up then ramp

Start conservatively, calibrate from real completions, then ramp toward optimal. Completion-based ramp with congestion detection (throughput drop, timeout rate) governs concurrency after warm-up.

### 5. Generous timeouts, retry for true failures

Timeouts are a safety net (P95 x margin), not an optimization lever. Timed-out tasks get fallback. True failures (429, connection error, 500) get one retry pass at reduced concurrency.

### 6. Model-tier-aware output handling

Uses instructor + Pydantic validation (Pattern B). Field validators adapt strictness based on model tier.

### 7. Documentation tracks implementation

This PROCESSING.md reflects what the code does now, not what we plan to do. Known gaps go in section E. When we fix a gap, we update the code and this doc in the same commit.

### 8. Development code stays clean

This is development, not production. No legacy references, no backward compatibility shims, no dead or redundant code. If something is replaced, the old version is deleted — not commented out, not kept "for backward compat." Clean code now is easier to promote to production later.

---

# II. Processing

## A. Overview

Step 3 extracts individual ideas/concepts from each survey response. One API call per response. Worker/queue dispatch with context extraction phases before the main batch.

- **Input:** `List[QualityFilteredModel]` (from step 2, meaningful responses only)
- **Output:** `List[IdeasExtractedModel]` with extracted ideas per response
- **Model:** Configured via `get_step_model("segmentation")` — typically mini/default
- **Provider:** OpenAI (Responses API) or Azure (Chat Completions API), switched via `API_PROVIDER`

---

## B. Processing strategy

Six lifecycle phases + retry pass:

### Phase 1: Fetch API rate limits

Minimal API call to read rate limit headers. Falls back to `FALLBACK_TPM` / `FALLBACK_RPM` if missing.

### Phase 2: Conservative rate limiters

Initializes rate limiters with conservative settings (50% headroom) for context extraction phases. Concurrency capped at max(tasks, 10). These are temporary — replaced in phase 5 with calibrated limiters.

### Phase 3: Context extraction (3 sub-phases)

Run before the main batch to build prompt context:
- **3a: Generic specifiers** — Extract Group 1 + Group 2 context specifiers from response samples in parallel
- **3b: Taxonomy scoring** — Determine primary dimension (the analytical axis for idea extraction)
- **3c: Domain discovery** — Discover content domains from response samples (optional, toggle: `discover_domains`)

Output: `generic_specifiers`, `primary_dimension`, `domains` — used to build per-response prompts in phase 6.

### Phase 4: Token recalibration

Recalculate `avg_tokens` using tiktoken on real prompts (with primary dimension description injected). Replaces the initial estimate which lacked context.

### Phase 5: Rate limiting initialization

Full initialization with calibrated token estimates:
- Compute Little's Law target from API limits + model-tier latency + recalibrated tokens
- Initialize ConcurrencyRamp (50% → 90% of Little's Law)
- Initialize circuit breaker, PID controller, TPM/RPM trackers
- Load persistent cold-start stats if available

### Phase 6: Main batch processing

Worker/queue dispatch. Main loop runs at 10 Hz:

| Check | Frequency | What it does |
|---|---|---|
| Circuit breaker | Every tick (~0.1s) | Evaluates timeout rate, trips/recovers if needed |
| Completion-based ramp | Every tick | ConcurrencyRamp advances proportional to progress |
| Warm-up calibration | Once (after 15-30 completions) | Recalibrates Little's Law with measured data, spawns extra workers |
| PID / throughput adjustment | Every 20s | Threshold-based token correction, then PID if threshold didn't fire |
| Progress report | Every 2s or N completions | Logs rate, concurrency, TPM/RPM%, latency |

### Phase 7: Retry pass

After main batch:
- Timed-out tasks + failed tasks requeued with reduced concurrency (10% of workers, min 5)
- Generous timeout (P95×3 or 180s)
- Permanent fallback with `PROCESSING_ERROR` if still failed
- Persists measured stats to `model_perf_stats.json`

---

## C. Rate-limiting machinery

### Request flow (per task)

```
async with self.semaphore:                    # Layer 1: ConcurrencyGate
    await self.tpm_bucket.wait_and_acquire()  # Layer 2: TokenBucket
    async with self.rate_limiter:             # Layer 3: AsyncLimiter (RPM)
        response = await asyncio.wait_for(    # Safety net: Timeout
            api_call, timeout=timeout
        )
```

### Layer 1: ConcurrencyGate

Same implementation as step 2. Runtime limit changes supported.

### Layer 2: TokenBucket

TPM safety rail with reconciliation. Additionally learns tiktoken-to-API offset (~300 tokens for instructor/system overhead) via `TiktokenOffsetLearner` after 5+ samples.

### Layer 3: AsyncLimiter (RPM)

`aiolimiter.AsyncLimiter` for request spacing. Adjusted by PID controller.

### Safety net: Timeout

| Condition | Timeout |
|---|---|
| Cold start | 45s (complex multi-field extraction) |
| After warm-up | `max(45s, min(P95 * 3, 180s))` |
| Retry mode | Very generous (P95×3 or 180s) |

### Layer 4: Circuit breaker

Same state machine as step 2: CLOSED -> OPEN -> RECOVERING -> CLOSED. Trips at > 5% timeout rate.

### Completion-based ramp

`ConcurrencyRamp` advances linearly from 50% to 90% of Little's Law, proportional to completion progress. Stops early on throughput drop (>10% decline for 2 consecutive windows) or queue congestion (>5% timeout rate).

### PID arrival rate adjustment

`_apply_pid_adjustment()` runs every 20s. Same `PIDThroughputController` as step 2 — asymmetric gains, targets 80% TPM utilization. Only fires if threshold-based `_adjust_throughput_if_needed()` didn't trigger first.

### Token offset learning

`TiktokenOffsetLearner` tracks the delta between tiktoken estimates and actual API token counts. After 5+ samples, applies learned offset to improve token estimation for bucket allocation.

---

## D. Divergent paths

### Model tier handling

Step 3 uses instructor (Pattern B) for all model tiers, with tier-aware field validators in the Pydantic response model:

- **Mini/default (`_strict_mode = True`):** Field validators raise `ValueError` on empty/invalid fields
- **Nano (`_strict_mode = False`):** Field validators coerce rather than reject (e.g., empty string instead of error)

`_strict_mode` is set once at init via `configure_validation_mode(model)`.

### OpenAI vs Azure (provider)

Provider abstraction handled by `llm.py` via `llm_create_async()`. Same as step 2.

---

## E. Known issues and divergences

### 1. ConcurrencyRamp instead of signal-based ramp

Step 3 uses `ConcurrencyRamp` (completion-based linear ramp with 2 stop signals: throughput drop, timeout rate). Step 2 was migrated to signal-based ramp (4-signal: queue, RPM%, TPM%, latency trend). The signal-based ramp is more sophisticated but step 3 hasn't been migrated yet.

### 2. Dual adjustment: threshold + PID coexist

The main loop runs `_adjust_throughput_if_needed()` first; PID only fires if the threshold didn't trigger. This means PID is partially bypassed whenever tokens diverge >5% from estimate. Step 2 was cleaned up to use PID only.

### 3. v3_stats prefix

Stats tracked with `v3_` prefix throughout the code, suggesting previous versions. Not harmful but adds naming clutter.

### 4. Legacy debug captures

`_captured_consolidate1`, `_captured_consolidate2` flags for one-shot prompt capture. Debug scaffolding that could be removed.

---

## F. Configuration reference

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| Timeout floor | 45s | `config_ideaExtractor.py` TimeoutConfig |
| Default timeout | 45s | `config_ideaExtractor.py` TimeoutConfig |
| `COLD_START_CAP` | 50 | Shared constant |
| `ADJUSTMENT_INTERVAL` | 20s | `config_ideaExtractor.py` ReportingConfig |
| `COLD_START_P95_MULTIPLIER` | 2.0 | `modelPerfStats.py` |
| Phase key | `step3_idea_extraction` | `model_perf_stats.json` lookup |

### Shared dataclasses (from `config_ideaExtractor.py`)

| Dataclass | Key values |
|---|---|
| `SegmentationConfig` | max_tokens=16000, temperature=0.0, model from config |
| `RampUpConfig` | start=50%, target=90%, min_initial=5 |
| `CircuitBreakerConfig` | window=30s, trip=5%, reduction=15%, cooldown=60s |
| `WarmUpConfig` | sample_min=15, sample_max=30 |
| `PIDControllerConfig` | kp_up=0.4, kp_down=0.2, ki=0.05, kd=0.1, min=2%, max=15% |
| `TPMTrackingConfig` | window=60s, target_utilization=80% |
| `TiktokenOffsetConfig` | default_offset=300, min_samples=5 |
