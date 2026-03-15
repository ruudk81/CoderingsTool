# Prompt Processing Strategy

Reference document for the rate limiting system in `ideaExtractor_exp.py`.

Last updated: 2026-03-15

---

## Architecture overview

Four control layers + safety-net timeout:

```
Request flow:

  worker coroutine (N = ramp target, scaled up after warm-up)
       │
       ▼
  ┌─────────────┐
  │  Concurrency │  Layer 1: ConcurrencyGate
  │  Gate        │  Completion-based ramp: 50% → 90% of Little's Law
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Token Bucket │  Layer 2: TPM safety rail (self-regulating)
  │              │  Prevents tokens-per-minute quota violations
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ AsyncLimiter │  Layer 3: RPM safety rail (PID-adjusted arrival rate)
  │              │  Prevents requests-per-minute quota violations
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Timeout     │  Safety net: 60s floor, P95×3 adaptive
  │  (generous)  │  Only catches truly stuck requests
  └──────┬──────┘
         │
         ▼
     API call

  ┌─────────────┐
  │   Circuit    │  Layer 4: Monitors timeout RATE in sliding window
  │   Breaker    │  Adjusts Layer 1 on sustained pressure only
  └─────────────┘
```

### Four constraints model

The system manages four independent constraints:

| Constraint | What it is | How we handle it |
|-----------|-----------|-----------------|
| **RPM** | Requests per minute (API quota) | AsyncLimiter + PID controller |
| **TPM** | Tokens per minute (API quota) | TokenBucket (self-regulating) |
| **Server-side queue** | OpenAI's internal request queue | ConcurrencyGate (Little's Law cap) |
| **Latency variance** | Natural response time variation (1-60s) | Generous timeout (NOT aggressive) |

**Key insight:** Latency variance is the dominant constraint for most workloads. ~5% of API calls take 10-30s due to complex prompts, longer responses, or server batching. This is normal — NOT a sign of congestion. Aggressive timeouts are counterproductive: they defer legitimate slow tasks, double API costs on retry, and trigger the circuit breaker unnecessarily.

### Design principles

1. **Generous timeouts** — 60s floor, P95×3 adaptive. Only catches truly stuck requests. Timed-out tasks get fallback (no retry) since they're genuine outliers.

2. **Completion-based ramp** — concurrency scales with progress, not wall-clock time. Works whether processing takes 5 seconds or 5 minutes.

3. **Constraint visibility** — every progress report shows TPM%, RPM%, and Concurrency% utilization so you can see which constraint is the bottleneck.

4. **Workers match capacity** — worker count equals ramp target, scaled up after warm-up calibration. No idle workers queued at the semaphore.

5. **Timeout computed after semaphore** — prevents stale cold-start values when all workers acquire slots simultaneously before any latency data exists.

---

## Concurrency management

### Little's Law as upper bound

Little's Law (`N = λ × W`) computes the optimal concurrency from arrival rate and latency. We use it as an **upper bound**, not a target, because it assumes constant latency — which breaks under server-side queuing.

### Completion-based ramp (ConcurrencyRamp)

Concurrency scales linearly with completion progress:

```
  0% complete → 50% of Little's Law (start_fraction)
 50% complete → 70% of Little's Law
100% complete → 90% of Little's Law (target_fraction)
```

This replaces time-based ramping which failed when processing completed faster than the ramp duration.

**Two stop signals:**
1. **Throughput drop** — completion rate declining >10% for 2 consecutive measurement windows
2. **Queue congestion** — timeout rate >5% in a measurement window

After warm-up calibration, Little's Law is recalculated with measured latency (P10, to avoid queuing-inflated values). The ramp adjusts start/target but **preserves congestion detection state** so ongoing throughput decline isn't forgotten.

### Worker scaling

- **Initial:** workers = ramp target (90% of Little's Law)
- **After warm-up:** if recalculated ramp target is higher, extra workers are spawned
- Workers cycle through the task queue — a finished worker picks the next task

### Circuit breaker (safety net)

State machine: CLOSED → OPEN → RECOVERING → CLOSED

```
CLOSED:     Monitoring. Timeout rate < 5% → no action.
OPEN:       Tripped. Concurrency reduced 15%. Cooldown 60s.
RECOVERING: Cooldown expired, rate OK. Ramp +10% per 30s toward baseline.
```

With 60s timeouts, the circuit breaker should rarely trip. If it does, something is genuinely wrong (network issues, API degradation).

---

## Timeout strategy

### Why generous timeouts

Evidence from 5+ test runs with 1375 tasks:

| Timeout strategy | Deferrals | Total time | API cost |
|-----------------|-----------|------------|----------|
| P95×1.3, floor 5s | 150-174 | 130-170s | High (double calls for deferred) |
| P95×1.3, floor 10s | 55-156 | 70-97s | Medium |
| avg×1.3, floor 30s | 2-68 | 64-84s | Low-medium |
| **P95×3, floor 60s** | **0-1** | **64s** | **Lowest** |

Aggressive timeouts create a vicious cycle:
1. Tight timeout → defer slow (but legitimate) tasks
2. Deferred tasks retry later → double API cost
3. Timeouts trigger circuit breaker → reduce concurrency → slower overall
4. Retry phase adds 35-60s of wall-clock time

**Solution:** Accept that ~5% of API calls take 10-30s. Set timeout high enough that only truly stuck requests (>60s) are caught. Those get fallback — no retry, since they're genuine outliers.

### Timeout computation

```python
def get_timeout(est_tokens):
    if retry_mode:    return 120.0   # Very generous
    if no data yet:   return 60.0    # Cold start
    else:             return max(60.0, min(P95 * 3.0, 120.0))  # Adaptive safety net
```

**Critical:** timeout is computed AFTER semaphore acquisition, not before. This prevents all workers from getting stale cold-start values when they acquire slots simultaneously at T+0.

### Timed-out tasks

Tasks that exceed the timeout get a `PROCESSING_ERROR` fallback — no retry. At 60s floor, this happens to <0.1% of tasks. These are genuine outliers (extremely complex responses, network hiccups).

---

## PID arrival rate optimization

The PID controller adjusts the AsyncLimiter's arrival rate based on real-time TPM utilization:

1. `RealTimeTPMTracker`: sliding window (60s) of actual token consumption → current TPM
2. `PIDThroughputController`: asymmetric gains (kp_up=0.4, kp_down=0.2)
3. Every 20s: compute utilization = current_TPM / TPM_limit → PID output → adjust arrival rate
4. Only applies if >2% relative change (avoids chatter)

**Asymmetric gains**: Aggressive when under-utilizing (speed up 2x faster), gentle when over-utilizing (avoid 429s).

---

## Lifecycle phases

### Phase 1: Fetch API rate limits
Minimal API call → read `x-ratelimit-limit-tokens` / `x-ratelimit-limit-requests` from headers.

### Phase 2: Conservative initialization
Conservative rate limiters for context extraction (concurrency=10, 50% headroom).

### Phase 3: Context extraction
Discovers specifiers, primary dimension, domains with conservative limiters.

### Phase 4: Recalculate token estimates
Tiktoken-based avg_tokens with real context (local calculation, no API calls).

### Phase 5: Production rate limiting initialization
- Arrival rate from `min(RPM, TPM)` with headroom (PID adjusts this)
- Little's Law as upper bound
- ConcurrencyGate starts at 50% of Little's Law
- Workers = ramp target (90% of Little's Law)
- Circuit breaker, PID controller, TPM/RPM trackers initialized

### Phase 6: Main processing loop

```
Every 0.1s:
  circuit_breaker.check_and_adjust()   ← evaluates timeout RATE
  await _check_ramp_up()               ← completion-based concurrency ramp

Every 68 completions OR 2s:
  progress report (TPM%, RPM%, Concurrency%, ramp status, deferred count)

After 30 completions:
  _calibrate_from_warm_up()            ← update tokens + arrival rate
                                       ← recalibrate ramp (new Little's Law with P10 latency)
                                       ← spawn extra workers if ramp target increased

Every 20s:
  _adjust_throughput_if_needed()       ← threshold token correction
  OR _apply_pid_adjustment()           ← PID arrival rate fine-tuning
```

### Phase 7: Cleanup
Timed-out tasks (if any) get fallback responses. No retry phase.

---

## Configuration reference

### Ramp-up (RampUpConfig)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `start_fraction` | 0.50 | Start at 50% of Little's Law |
| `target_fraction` | 0.90 | Ramp toward 90% of Little's Law |
| `min_initial` | 5 | Never start below 5 |
| `measurement_window_seconds` | 0.5 | Check every 0.5s |
| `min_completions_per_step` | 3 | Need 3 completions to evaluate |

### Circuit breaker (CircuitBreakerConfig)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `window_seconds` | 30.0 | Sliding window for timeout rate |
| `trip_threshold` | 0.05 | Trip at >5% timeout rate |
| `min_events_to_trip` | 10 | Need 10+ events to evaluate |
| `reduction_factor` | 0.85 | Reduce 15% when tripped |
| `cooldown_seconds` | 60.0 | No further reductions during cooldown |
| `recovery_step_pct` | 0.10 | Recover 10% per interval |
| `recovery_interval_seconds` | 30.0 | Recovery check every 30s |
| `min_concurrency` | 10 | Hard floor |

### PID controller (PIDControllerConfig)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `kp_up` | 0.4 | Aggressive when under-utilizing |
| `kp_down` | 0.2 | Gentle when over-utilizing |
| `ki` | 0.05 | Integral (persistent bias) |
| `kd` | 0.1 | Derivative (dampen oscillation) |
| `min_adjustment` | 0.02 | Ignore <2% changes |
| `max_adjustment` | 0.15 | Cap at 15% per step |

### Timeout (LatencyTracker)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Cold start | 60s | No latency data yet |
| Floor | 60s | Minimum timeout |
| Ceiling | 120s | Maximum timeout |
| Multiplier | P95 × 3.0 | Adaptive safety net |
| Retry mode | 120s | Very generous (unused — no retry phase) |

### TPM tracking (TPMTrackingConfig)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `sliding_window_seconds` | 60.0 | Track over last 60s |
| `target_utilization` | 0.80 | Target 80% TPM utilization |

### Warm-up calibration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `sample_min` | 15 | Min completions before calibration |
| `sample_max` | 30 | Max completions before forced calibration |
| Latency metric | P10 | Avoids queuing-inflated values |

---

## Example output

```
RATE LIMITING SETUP - Completion-Based Ramp + Congestion Detection + PID
- Model: gpt-4.1-mini
- RPM limit: 30,000 (27,000 with headroom)
- TPM limit: 150,000,000 (135,000,000 with headroom)
- Little's Law: 900
- Ramp: 450 (50%) → 810 (90%) proportional to completions
- Starting concurrency: 450
- Timeout: 60s safety net (no retry, fallback on timeout)
- Processing 1,375 tasks

Workers: 810, Starting: 450, Target: 810

⏱ T+0.0s: Starting task processing
Progress: 810/1375 (58.9%) Rate: 470/s | TPM:0% RPM:0% Conc:450/450(100%) CB:CLOSED ramp:450→810
RAMP RECALIBRATED: 687 → 1237 (Little's Law: 1375)
Workers: 1237 (+427 after calibration)
Progress: 887/1375 (64.5%) Rate: 163/s | TPM:3% RPM:8% Conc:715/715(100%) CB:CLOSED ramp:715→1237
Progress: 1041/1375 (75.7%) Rate: 150/s | TPM:5% RPM:13% Conc:779/779(100%) CB:CLOSED
Progress: 1353/1375 (98.4%) Rate: 147/s | TPM:7% RPM:19% Conc:810/904(90%) CB:CLOSED
⏱ T+64.4s: Main batch done — 1374 succeeded, 1 deferred

Completed 1375 tasks in 64.4s
- Successful: 1374
- Failed: 1
- Timeouts: 1 (fallback, no retry)
```

---

## Lessons learned

### Aggressive timeouts are counterproductive
The biggest performance win came from raising the timeout floor from 10s to 60s. Tight timeouts don't make processing faster — they create a retry tail that doubles both cost and wall-clock time.

### Latency variance is the real bottleneck
With high-tier API limits (150M TPM, 30K RPM), RPM and TPM are rarely the constraint. The actual bottleneck is latency variance: ~5% of LLM API calls naturally take 10-30s. This is server-side behavior, not something our rate limiting can fix.

### Time-based ramps don't work at high speed
When 1375 tasks complete in seconds, a "ramp over 60 seconds" barely moves. Completion-based ramps (concurrency proportional to progress) work at any speed.

### Workers must match concurrency capacity
Having 1000 workers for 450 concurrency slots means 550 workers queue at the semaphore. When they finally get through, they compute timeout with stale data. Workers should equal the ramp target, scaled up after warm-up.

### Cold-start timeout must be generous
All workers acquire semaphore slots simultaneously at T+0. They all compute timeout before any latency data exists. A 10s cold-start timeout caused ~5% of the first batch to be deferred. Solution: compute timeout AFTER semaphore, and use 60s cold start.

### Measure latency at the right scope
Latency fed into the tracker must measure only API response time (after semaphore + token bucket), not total task time including queue wait. Otherwise, queuing time inflates the latency estimate, inflating Little's Law, creating a positive feedback loop.

---

## Changelog

### 2026-03-15: Generous timeouts + no retry + worker scaling

**Problem:** 55-156 tasks deferred per run due to aggressive timeouts (10s cold start, P95×1.3 floor). Retry phase added 35-60s. Root cause: (1) all workers computed timeout before any latency data existed (cold start = 10s), (2) tight timeouts deferred legitimate slow tasks.

**Fix:**
1. **Timeout floor 60s** — P95×3.0, floor 60s, ceiling 120s. Only catches truly stuck requests.
2. **No retry** — timed-out tasks get fallback. At 60s floor, <0.1% of tasks timeout.
3. **Timeout after semaphore** — computed with current latency data, not stale cold-start values.
4. **Workers = ramp target** — scaled up after warm-up calibration when ramp target increases.
5. **Latency from API start** — measures only API time, not semaphore queue wait.

**Result:** 1374/1375 succeeded, 1 deferred (genuine 63s outlier), 64s total. Down from 70-97s.

### 2026-03-15: Completion-based ramp

**Problem:** Time-based ramp (50%→90% over 60s) didn't work when processing completed in seconds. Linear interpolation `elapsed/60s` barely moved. Concurrency stayed at starting value.

**Fix:** Ramp proportional to completions: `ramp_fraction = completions / total_tasks`. Works at any processing speed.

### 2026-03-15: Linear ramp replacing TCP slow-start

**Problem:** Exponential doubling (5→10→20→40→80→160→320) too aggressive at high concurrency. Server-side queuing caused throughput to drop as concurrency increased.

**Fix:** Linear ramp from 50% to 90% of Little's Law with two congestion stop signals (throughput drop, timeout rate). P10 latency for Little's Law recalculation to avoid queuing-inflated values.

### 2026-03-15: TCP slow-start empirical concurrency ramp

Replaced blind Little's Law ramp with empirical throughput-plateau detection. Later replaced by linear ramp.

### 2026-03-15: Hybrid strategy — Little's Law + PID + Circuit Breaker

Combined Little's Law, PID arrival rate adjustment, and circuit breaker. Replaced timeout-based ceiling discovery which caused death spiral.

### 2026-03-15: Initial rate limiting architecture

Four-layer design: ConcurrencyGate, TokenBucket, AsyncLimiter, Circuit Breaker. Batch retry for timeouts. Constraint visibility in progress reports.
