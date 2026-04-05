# Step 2 — Processing

Source of truth: the code in `qualityFilter.py`.
Last verified against code: 2026-04-05

## Contract

Principles for how processing works in this step. Updating code = updating this doc.

### 1. Goal

Optimize the processing of prompts across providers (OpenAI, Azure), models (nano, mini, default), and constraints (RPM, TPM, server-side queue, latency), where the binding bottleneck varies by deployment.

### 2. Four-layer rate-limiting stack

Every API call passes through: ConcurrencyGate → TokenBucket → AsyncLimiter → Timeout. No bypasses, no shortcuts. Retry pass uses the same layers.

### 3. Self-tuning cold start

After each run, persist empirical stats (P50/P95 latency, avg tokens) to `model_perf_stats.json`. On next cold start, load stored stats and use them for Little's Law (P50), timeout floor (P95 x 2), and token estimation (avg tokens). First-ever run (no stored stats) falls back to model-tier defaults.

### 4. Warm-up then ramp

Start conservatively, calibrate from real completions, then ramp toward optimal. Signal-based ramp (4 signals: queue health, RPM%, TPM%, latency trend) governs concurrency after warm-up.

### 5. Generous timeouts, retry for true failures

Timeouts are a safety net (P95 x margin), not an optimization lever. Timed-out tasks get fallback. True failures (429, connection error, 500) get one retry pass at reduced concurrency.

### 6. Model-tier-aware output handling

Nano: raw API + regex/tag parsing (Pattern A). Mini/default: instructor + Pydantic validation (Pattern B). The tier determines the path at init (`_is_nano` flag).

### 7. Documentation tracks implementation

This PROCESSING.md reflects what the code does now. Known gaps go in the Known Issues section.

## Processing

### Overview

Step 2 evaluates each survey response individually and flags low-quality items. One API call per response. Queue + worker dispatch with full rate-limiting stack.

- **Input:** `List[PreprocessedModel]` (from step 1)
- **Output:** `List[QualityFilteredModel]` with `quality_filter_code` set
- **Model:** Configured via `get_step_model("quality_filter")` — typically nano
- **Provider:** OpenAI (Responses API) or Azure (Chat Completions API), switched via `API_PROVIDER`

### Pre-Processing Filters

Before LLM dispatch, `grade()` applies:
- **Step 1 passthrough**: `quality_filter_code is not None` → skip (already flagged by step 1)
- **Empty value pre-filter**: catches `None`, `NaN`, `<NA>`, empty strings → mark `99999998`

### Processing Strategy

Six lifecycle phases, executed sequentially:

**Phase 1: Fetch API rate limits**
Minimal API call (`"Hi"`) to read `x-ratelimit-limit-tokens` and `x-ratelimit-limit-requests` from response headers. Falls back to `FALLBACK_TPM` / `FALLBACK_RPM` if headers missing. Provider-aware: uses `responses.create()` for OpenAI, `chat.completions.create()` for Azure.

**Phase 2: Initialize rate limiters**
`_initialize_rate_limiters()` sets up all control layers:
1. Arrival rate from `min(RPM/60, TPM/60/avg_tokens)` with headroom
2. Token bucket initialized at full TPM (with headroom)
3. Cold-start concurrency = `min(RPM/60, 50)`, floored at 5, capped at task count
4. Little's Law target computed from latency estimate + token estimate
5. Circuit breaker initialized with cold-start as baseline
6. PID controller + RealTimeTPMTracker for continuous arrival rate tuning

Cold-start inputs come from persistent stats when available (>= 10 samples), otherwise model-tier defaults:

| Input | With stored stats | Without stored stats |
|---|---|---|
| Latency estimate | Stored P50 latency | `get_model_tier_latency()` (e.g. 2s for nano) |
| Token estimate | Stored avg_tokens | tiktoken sampling of 10 prompts x output ratio |
| Timeout floor | Stored P95 x 2.0 | `TIMEOUT_FLOOR_SECONDS` (60s) |

Workers spawned upfront: `min(target_semaphore, num_tasks)`, floor of 5.

**Phase 3: Warm-up calibration**
Triggers after `WARM_UP_WINDOW_SECONDS` (10s) AND `WARM_UP_MIN_COMPLETIONS` (3) completions.
- Measures actual avg tokens and median latency
- Recalculates Little's Law with measured values
- Stress detection: checks for timeouts or increasing latency
- If no stress: jumps concurrency to `min(100, Little's Law)`; if stress: holds at cold start
- Activates signal-based ramp toward 90% of Little's Law
- Recalculates arrival rate with measured token counts
- Spawns extra workers if target increased
- Resets PID state

**Phase 4: Main processing loop**
Runs every 0.1s while queue is not empty:

| Check | Frequency | What it does |
|---|---|---|
| Circuit breaker | Every tick (0.1s) | Evaluates timeout rate, trips/recovers if needed |
| Warm-up calibration | Once (after 10s + 3 completions) | Phase 3 above |
| Signal-based ramp | Every 5s | 4-signal evaluation, adjusts concurrency |
| PID arrival rate adjustment | Every 15s | Continuous TPM-based arrival rate tuning |
| Progress report | Every 5s | Logs completion rate, utilization, latency |

**Phase 5: Persist stats**
After all tasks complete, persists measured P50/P95 latency and avg tokens to `data/model_perf_stats.json` via EMA update (requires >= 5 latency samples). Phase key: `step2_quality_filter`.

**Phase 6: Retry pass for true failures**
If any tasks failed (timeout, 429, connection error, 500):
- Requeues failed tasks with reduced concurrency (10% of workers, min 5)
- Sets `retry_mode = True` on latency tracker (180s timeout)
- One retry only — second failure gets permanent fallback (`quality_filter_code = -1`)
- Merges recovered results back

## Rate-Limiting Machinery

### Request flow (per task)

```
async with self.semaphore:                    # Layer 1: ConcurrencyGate
    timeout = self.latency_tracker.get_timeout()  # Computed AFTER semaphore
    await self.tpm_bucket.wait_and_acquire()  # Layer 2: TokenBucket
    async with self.rate_limiter:             # Layer 3: AsyncLimiter (RPM)
        response = await asyncio.wait_for(    # Safety net: Timeout
            api_call, timeout=timeout
        )
```

### Layer 1: ConcurrencyGate

Custom semaphore with runtime limit changes. Future-based waiting (no polling). Limit decrease: in-flight drains naturally. Limit increase: blocked waiters wake immediately.

### Layer 2: TokenBucket

TPM safety rail. Pre-acquires estimated tokens; reconciles estimate vs actual after each call.

### Layer 3: AsyncLimiter (RPM)

`aiolimiter.AsyncLimiter` for request spacing. Reinstalled by PID controller when arrival rate adjusts.

### Safety net: Timeout

| Condition | Timeout |
|---|---|
| Retry mode | 180s |
| Cold start (no latency data) | `max(timeout_floor, default_timeout)` |
| After warm-up | `max(timeout_floor, min(P95 * margin, ceiling))` |

Computed **after** semaphore acquisition to prevent stale cold-start values.

### Layer 4: Circuit breaker

CLOSED → OPEN → RECOVERING → CLOSED. Monitors timeout rate in 30s sliding window. Trips at > 5% (>= 10 events). Reduces concurrency 15%, 60s cooldown, recovers +10%/30s.

### Signal-based ramp (post-warm-up)

Four signals evaluated every 5s:

| Signal | Green | Yellow | Red |
|---|---|---|---|
| Queue health | Shrinking/stable | Growing slowly | Growing rapidly |
| RPM utilization | < 80% | 80-90% | > 90% |
| TPM utilization | < 80% | 80-90% | > 90% |
| Latency trend (P95) | Stable/decreasing | Increased > 10% | Increased > 25% |

All green → +10%; any yellow → hold; any red → -20%.

### PID arrival rate adjustment

`_apply_pid_adjustment()` runs every 15s. Tunes AsyncLimiter arrival rate based on real-time TPM utilization.

- `RealTimeTPMTracker` — 60s sliding window of actual token consumption
- `PIDThroughputController` — asymmetric gains (kp_up=0.4, kp_down=0.2), targets 80% TPM utilization
- Adjustment clamped to +/-15% per step, ignored if <2% change
- PID state reset after warm-up calibration

## Divergent Paths

### Nano vs mini/default (model tier)

Determined at init: `self._is_nano = "nano" in self.model.lower()`

| Aspect | Nano (Pattern A) | Mini/Default (Pattern B) |
|---|---|---|
| Client | Raw `AsyncOpenAI` | `create_client()` (instructor-wrapped) |
| API call | `client.responses.create()` | `llm_create_async()` with Pydantic `response_model` |
| Prompt | `GRADER_INSTRUCTIONS_NANO` | `GRADER_INSTRUCTIONS_STRUCTURED` |
| Output parsing | `parse_quality_code()` — regex on `<category>` tag | Instructor validates into `QualityFilterStructuredResponse` |
| Parse failure | `None` (conservative: keep response) | Instructor retries up to 3 times |

Both share: same prompt body, same rate limiting, same result model.

### OpenAI vs Azure (provider)

Divergence limited to `_fetch_rate_limits_from_api()`:
- OpenAI: `responses.with_raw_response.create()`
- Azure: `chat.completions.with_raw_response.create()` with deployment name

Main `process_task()` uses `llm.py` for provider abstraction (Pattern B only).

## Known Issues and Divergences

### 1. Timeout floor higher than strategy doc

Code: `TIMEOUT_FLOOR_SECONDS = 60.0` (hardcoded in `qualityFilter.py` line 169). On first-ever run with no stored stats, cold-start timeout is 180s (`DEFAULT_TIMEOUT_SECONDS`).

### 2. Backward-compat alias in prompts

`GRADER_INSTRUCTIONS = GRADER_INSTRUCTIONS_NANO` (line 108 in `prompts_qualityFilter.py`). Should be removed — the code should reference `GRADER_INSTRUCTIONS_NANO` directly.

## Configuration Reference

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| `COLD_START_CAP` | 50 | `config_qualityFilter.py` |
| `WARM_UP_WINDOW_SECONDS` | 10.0 | `config_qualityFilter.py` |
| `WARM_UP_MIN_COMPLETIONS` | 3 | `config_qualityFilter.py` |
| `RAMP_INCREASE_FACTOR` | 1.10 (+10%) | `config_qualityFilter.py` |
| `RAMP_DECREASE_FACTOR` | 0.80 (-20%) | `config_qualityFilter.py` |
| `SIGNAL_GREEN_THRESHOLD` | 0.80 | `config_qualityFilter.py` |
| `SIGNAL_YELLOW_THRESHOLD` | 0.90 | `config_qualityFilter.py` |
| `TIMEOUT_FLOOR_SECONDS` | 60.0 | `qualityFilter.py` hardcoded |
| `COLD_START_P95_MULTIPLIER` | 2.0 | `modelPerfStats.py` |
| Post-warm-up jump cap | 100 | `qualityFilter.py` hardcoded |
| Retry worker fraction | 10% (min 5) | `qualityFilter.py` hardcoded |

### Persistent stats (`model_perf_stats.json`)

Phase key: `step2_quality_filter`. Activates at >= 10 samples. Stores and applies: P50 latency (Little's Law), P95 latency (timeout floor), avg tokens (token estimation).
