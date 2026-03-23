# Prompt Processing Strategy

Reference document for the rate limiting system across all pipeline steps — including multi-phase pipelines (step 4 classifier, step 5 codeGenerator, step 6 codeAssigner) using `asyncio.gather` dispatch, and single-phase steps (step 2 qualityFilter, step 3 ideaExtractor) using worker/queue dispatch.

Last updated: 2026-03-23

---

## Architecture overview

Four control layers + safety-net timeout + failure recovery:

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

  ┌─────────────┐
  │  Retry Pass  │  After main batch: requeue true failures
  │              │  10% concurrency, one attempt, then permanent fallback
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

1. **Generous timeouts** — 60s floor, 180s ceiling, P95×3 adaptive. Cold-start 180s. Only catches truly stuck requests. Timed-out tasks get fallback (no retry) since they're genuine outliers. Ceiling must accommodate the heaviest prompt type in the pipeline.

2. **Completion-based ramp** — concurrency scales with progress, not wall-clock time. Works whether processing takes 5 seconds or 5 minutes.

3. **Constraint visibility** — every progress report shows TPM%, RPM%, and Concurrency% utilization so you can see which constraint is the bottleneck.

4. **Workers match capacity** — worker count equals ramp target, scaled up after warm-up calibration. No idle workers queued at the semaphore.

5. **Timeout computed after semaphore** — prevents stale cold-start values when all workers acquire slots simultaneously before any latency data exists.

---

## Concurrency management

### Little's Law as upper bound

Little's Law (`N = λ × W`) computes the optimal concurrency from arrival rate and latency. We use it as an **upper bound**, not a target, because it assumes constant latency — which breaks under server-side queuing.

### Concurrency initialization: two options

There are two mutually exclusive strategies for initializing concurrency. Choose one, not both.

#### Option A: Bootstrap

Make 3 probe calls upfront, measure latency and tokens, compute Little's Law, start at full computed concurrency.

```
Phase 1: 3 serial probe calls → measure avg latency + avg tokens
Phase 2: Compute Little's Law → set concurrency, workers, rate limiters
Phase 3: Start processing at full computed concurrency
```

**Strengths:**
- Fast startup — no warm-up phase, peak throughput from the first real task
- Simple to implement — one-shot calculation, no recalibration logic

**Risks:**
- 3 probe calls may not be representative of real workload (different prompt lengths, response complexity)
- If probes underestimate latency → concurrency too high → server-side queuing → throughput drop
- If probes overestimate latency → concurrency too low → underutilization for the entire run
- No self-correction — bootstrap values are used for the entire batch

**Best for:** Small batches (<100 tasks) where warm-up overhead is disproportionate, or deployments with highly predictable latency.

#### Option B: Warm-up with conservative ramp (preferred)

Skip probes. Start conservatively, let real production completions calibrate the system, then ramp toward optimal.

```
Phase 1: Theoretical Little's Law estimate (from model limits + tiktoken token estimates)
Phase 2: Start at 50% of estimate (conservative)
Phase 3: Process real tasks, collect latency data
Phase 4: After 15-30 completions, recalibrate with measured P10 latency
Phase 5: Ramp linearly toward 90% of recalibrated Little's Law
```

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

**Strengths:**
- Self-correcting — real production data replaces theoretical estimates
- Conservative start protects against overloading the API
- Ramp ensures concurrency tracks actual capacity, not assumptions
- Works regardless of how representative initial estimates are

**Risks:**
- Slower to reach peak throughput — first 15-30 completions run at reduced concurrency
- On slow deployments (low TPM/RPM quotas), the conservative 50% start could mean a very slow first phase, and the 15-30 completion recalibration window could take significantly longer to reach

**Best for:** Large batches (100+ tasks) where the warm-up cost is negligible relative to total processing time, and where robust self-correction matters more than fast startup.

**Current preference: Option B.** Validated on high-throughput deployments (150M TPM, 30K RPM). Not yet tested on slow deployments with significantly lower quotas — this is an open area for future validation. On such deployments, the conservative 50% start may need a higher `start_fraction` or a smaller recalibration window to avoid excessive warm-up time.

#### Capacity-relative starting (supersedes old "small batches" rule)

The old rule (<30 tasks → static semaphore) was too coarse. With high-tier deployments (Little's Law = 4250), even 100 tasks should start at full concurrency. The new rule:

```
initial = max(50% of Little's Law, num_tasks)
initial = min(initial, num_tasks)  # never exceed task count
```

When `num_tasks ≤ 50% of Little's Law`, all tasks start immediately — the rate limiter + token bucket govern throughput. The 50% ramp only applies when task count exceeds half of Little's Law.

**Evidence:** P1 with 26 tasks and Little's Law = 4250. Starting at 13 (old rule: 50% of 26) spent the entire phase ramping while API had massive spare capacity. RPM used: 45 out of 27,000. TPM used: 210K out of 135M. Starting at 26 would have no RPM/TPM impact but would halve wall-clock time.

**Strategy by phase size:**

| Phase size relative to Little's Law | Strategy | Rationale |
|--------------------------------------|----------|-----------|
| `tasks ≤ 50% of Little's Law` | Full stack, start at `num_tasks` | No ramp needed — rate limiter governs. All tasks run at once |
| `tasks > 50% of Little's Law`, 30-100 | Full stack, start at 50%, warm-up if ≥30 | Enough tasks for calibration, ramp advances with completions |
| `tasks > 50% of Little's Law`, 100+ | Full stack, start at 50%, warm-up + ramp to 90% | Full self-correction |

Multi-phase pipelines must use **per-phase state** — never share bootstrap/semaphore/ramp across phases with different models or prompt types.

### Worker scaling

- **Option A:** workers = computed optimal concurrency (static for entire run)
- **Option B:** workers = ramp target (90% of Little's Law); after warm-up recalibration, if ramp target increases, extra workers are spawned
- In both options, workers cycle through the task queue — a finished worker picks the next task

### Circuit breaker (safety net)

State machine: CLOSED → OPEN → RECOVERING → CLOSED

```
CLOSED:     Monitoring. Timeout rate < 5% → no action.
OPEN:       Tripped. Concurrency reduced 15%. Cooldown 60s.
RECOVERING: Cooldown expired, rate OK. Ramp +10% per 30s toward baseline.
```

With 60s timeouts, the circuit breaker should rarely trip. If it does, something is genuinely wrong (network issues, API degradation).

Applicable to both Option A and Option B, though more valuable with Option B where concurrency changes over time.

### Multi-phase pipeline adaptation

The architecture above was designed for single-model, single-phase worker/queue pipelines (steps 2/3). Multi-phase pipelines (step 4 classifier: 7 phases, 4-5 models) require adaptations:

**Dispatch pattern agnostic.** The 4-layer stack operates at the `_llm_call()` level. Whether tasks are dispatched via worker/queue (`process_task()`) or `asyncio.gather()`, every API call passes through the same layers in the same order: ConcurrencyGate → TokenBucket → AsyncLimiter → Timeout. The dispatch pattern is irrelevant to the layers.

**Per-phase rate limiting state.** Each phase gets its own `PhaseRampState` containing: gate, ramp, token bucket, latency tracker, circuit breaker. Phases run sequentially so there is no cross-phase contention. This avoids the "shared bootstrap" anti-pattern where one phase's calibration poisons another's concurrency settings.

**Per-phase warm-up.** Different phases use different models with radically different token/latency profiles. For example, step 4's P3 (gpt-4.1-nano, batch assignment: P50=4s, 1843 avg tokens) vs P1 (gpt-4.1-mini, discovery: P50=24s, 4795 avg tokens). Cross-phase calibration would be meaningless — each phase must calibrate independently.

**Capacity-relative starting.** When `num_tasks ≤ 50% of Little's Law`, skip the ramp and start at `num_tasks`. The rate limiter + token bucket prevent RPM/TPM violations; there is no benefit to throttling concurrency when the API has massive spare capacity. See "Concurrency initialization" below.

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
    if retry_mode:    return 180.0   # Very generous
    if no data yet:   return 180.0   # Cold start — no latency data, be generous
    else:             return max(60.0, min(P95 * 3.0, 180.0))  # Adaptive safety net
```

Cold-start must be generous because complex prompts (100+ observations) can take 30-45s. A 60s cold-start causes unnecessary timeouts before the latency tracker has data.

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

### Phase 2: Concurrency initialization

**If using Option A (Bootstrap):**
- 3 serial probe calls → measure avg latency + avg tokens
- Compute Little's Law → set concurrency, workers, rate limiters at full computed value
- Proceed to Phase 3

**If using Option B (Warm-up with conservative ramp):**
- Theoretical Little's Law estimate from model limits + tiktoken token estimates
- Arrival rate from `min(RPM, TPM)` with headroom (PID adjusts this)
- ConcurrencyGate starts at 50% of Little's Law
- Workers = ramp target (90% of Little's Law)
- Circuit breaker, PID controller, TPM/RPM trackers initialized
- Proceed to Phase 3

### Phase 3: Context extraction (if applicable)
Discovers specifiers, primary dimension, domains with conservative limiters. (Step-specific — not all utilities need this phase.)

### Phase 4: Main processing loop

**Option A (Bootstrap):** Straightforward — all tasks processed at the static concurrency computed during bootstrap. No ramp, no recalibration.

**Option B (Warm-up with conservative ramp):**

```
Every 0.1s:
  circuit_breaker.check_and_adjust()   ← evaluates timeout RATE
  await _check_ramp_up()               ← completion-based concurrency ramp

Every 68 completions OR 2s:
  progress report (TPM%, RPM%, Concurrency%, ramp status, deferred count)

After 15-30 completions:
  _calibrate_from_warm_up()            ← update tokens + arrival rate
                                       ← recalibrate ramp (new Little's Law with P10 latency)
                                       ← spawn extra workers if ramp target increased

Every 20s:
  _adjust_throughput_if_needed()       ← threshold token correction
  OR _apply_pid_adjustment()           ← PID arrival rate fine-tuning
```

### Phase 5: Cleanup
Timed-out tasks (if any) get fallback responses.

### Phase 6: Retry pass for true failures

After the main batch completes, any tasks that experienced a true process failure (API timeout, 429, connection error, internal server error) are retried once with reduced concurrency. This is distinct from the earlier "batch retry" approach that was removed — that retried aggressively-timed-out tasks that were likely just slow. This phase only targets tasks where the API call genuinely did not return a usable response.

See [Failure tracking and retry](#failure-tracking-and-retry) below for the full design.

---

## Failure tracking and retry

### Why

The generous timeout strategy (60s floor, P95×3) reduces failures to <0.1% of tasks. But <0.1% is not zero. At scale (10k+ tasks), that's 10+ data points permanently lost to transient errors — rate limit spikes, momentary connection drops, server-side 500s. These are recoverable failures: the same task sent again a few seconds later will almost certainly succeed, because the root cause (transient API pressure) has passed by the time the main batch finishes.

Accepting permanent data loss from transient errors is an unnecessary trade-off. A single retry pass after the main batch costs near-zero (a handful of API calls at reduced concurrency) and recovers the majority of failed tasks.

### What

Three components:

1. **Explicit failure tracking** — a `failed_task_ids` set that records which tasks experienced a true process failure (API did not return a usable response). This is distinct from tasks where the LLM intentionally returned the original text unchanged or returned a `[NO RESPONSE]` / `PROCESSING_ERROR` fallback.

2. **Retry pass** — after the main batch completes, failed tasks are requeued with reduced concurrency. One retry only — if a task fails twice, it gets permanent fallback.

3. **Recovery reporting** — explicit output showing how many tasks were recovered vs permanently failed, including the IDs of permanently failed tasks for auditability.

### What counts as a true failure

| Error type | Tracked as failure | Rationale |
|-----------|-------------------|-----------|
| `asyncio.TimeoutError` | Yes | Request exceeded generous timeout — likely stuck |
| `RateLimitError` (429) | Yes | Transient API quota pressure — likely succeeds on retry |
| `APITimeoutError` | Yes | Server-side timeout — transient |
| `APIConnectionError` | Yes | Network interruption — transient |
| `InternalServerError` (500) | Yes | Server-side error — transient |
| Unhandled exception in worker | Yes | Catch-all for unexpected failures |
| LLM returns original text unchanged | No | Intentional — LLM decided no correction needed |
| LLM returns `[NO RESPONSE]` / `PROCESSING_ERROR` | No | LLM processed the task and gave a deliberate answer |

### How

#### Tracking

Each utility class maintains a `failed_task_ids: set` attribute, cleared at the start of each processing run. Every error handler adds the task's unique identifier (e.g., `respondent_id`) to this set alongside incrementing `llm_calls_failed`.

```python
# In __init__
self.failed_task_ids = set()

# At start of processing run
self.failed_task_ids.clear()

# In every error handler (timeout, 429, connection, 500, unknown)
self.stats['llm_calls_failed'] += 1
self.failed_task_ids.add(task_id)
```

#### Retry pass

After the main batch returns, collect tasks whose ID is in `failed_task_ids` and reprocess with conservative settings:

```python
if self.failed_task_ids:
    failed_task_list = [t for t in tasks if t['task_id'] in self.failed_task_ids]

    # Reset tracking for retry pass
    retry_failed_ids = set(self.failed_task_ids)
    self.failed_task_ids.clear()

    # Reduced concurrency: 10% of original, floor of 5
    retry_workers = max(5, min(len(failed_task_list), num_workers // 10))

    retry_results = await self._process_all_tasks_async(
        failed_task_list, ..., retry_workers
    )

    results.update(retry_results)
```

Key design choices:
- **One retry only** — if a task fails on retry, it gets permanent fallback. Repeated retries risk masking a systemic issue.
- **Reduced concurrency** — 10% of original worker count (minimum 5). The main batch just finished, so API pressure is at its lowest. Conservative concurrency avoids re-triggering the conditions that caused the original failures.
- **Same admission controls** — retry tasks go through the same semaphore, token bucket, and rate limiter. No bypassing safety layers.
- **Reuses existing infrastructure** — the retry pass calls the same `_process_all_tasks_async` method. No separate code path to maintain.

#### Reporting

```
[RETRY PASS] Retrying 7 failed tasks with reduced concurrency...
[RETRY PASS] Recovered: 6, Still failed: 1
[RETRY PASS] Permanently failed task IDs: [4821]
```

The permanently failed IDs are logged so that downstream consumers can identify which data points have fallback values rather than real corrections.

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `retry_concurrency_fraction` | 0.10 | Use 10% of main-pass worker count |
| `retry_min_workers` | 5 | Floor for retry worker count |
| `retry_max_attempts` | 1 | Single retry pass only |

### Wall-clock cost warning

The retry pass reuses the same generous timeout (60s floor, P95×3 adaptive). Since it runs sequentially after the main batch — not in parallel with it — every retried task that times out again adds up to 60s of pure wall-clock overhead to the total processing time. With reduced concurrency (e.g., 5 workers), `n` failed tasks that all time out again cost roughly `ceil(n / workers) × 60s` in the worst case.

In practice this is rare: most transient failures (429s, connection drops) succeed immediately on retry, adding only seconds. But if the failures are not transient (e.g., a sustained API outage), the retry pass becomes an expensive no-op. Implementers should be aware of this tail-risk and may want to:

- **Cap total retry duration** — e.g., abort the retry pass if it exceeds 120s regardless of remaining tasks
- **Use a shorter timeout for retries** — e.g., 30s instead of 60s, since the latency tracker has a full history and genuinely stuck requests don't need a second generous window
- **Skip retry entirely if failure count exceeds a threshold** — e.g., >5% of tasks failed suggests a systemic issue, not transient errors

These are implementation choices — the default (same timeout, one pass, reduced concurrency) is the safe starting point.

### Interaction with other layers

- **Generous timeouts remain the primary defense.** The retry pass is a safety net for the safety net — it handles the <0.1% that generous timeouts cannot prevent (true transient errors vs slow-but-legitimate responses).
- **Circuit breaker state is preserved** across the retry pass. If the circuit breaker tripped during the main batch, the retry pass operates under those reduced limits.
- **The retry pass does not change the timeout strategy.** Retry tasks use the same adaptive timeout (P95×3, 60s floor). No special "retry mode" timeout is needed because the latency tracker has a full history from the main batch.

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
| Cold start | 180s | No latency data yet — generous for complex prompts |
| Floor | 60s | Minimum timeout |
| Ceiling | 180s | Maximum timeout (accommodates heavy discovery prompts) |
| Multiplier | P95 × 3.0 | Adaptive safety net |
| Retry mode | 180s | Very generous for retry pass |

### TPM tracking (TPMTrackingConfig)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `sliding_window_seconds` | 60.0 | Track over last 60s |
| `target_utilization` | 0.80 | Target 80% TPM utilization |

### Warm-up calibration (Option B only)
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
⏱ T+64.4s: Main batch done — 1374 succeeded, 1 failed

[RETRY PASS] Retrying 1 failed tasks with reduced concurrency...
[RETRY PASS] Recovered: 1, Still failed: 0

Completed 1375 tasks in 65.8s
- Successful: 1375 (1374 main + 1 recovered)
- Permanently failed: 0
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

### Match calibration strategy to batch size
Bootstrap and warm-up both have overhead. For small batches (<30 tasks), that overhead is disproportionate — you spend more time calibrating than processing. A static semaphore with a rate limiter is sufficient. This is especially important in multi-phase pipelines: don't run a shared bootstrap that calibrates against one phase's prompts and then applies that calibration to a different phase with entirely different latency characteristics.

### Measure latency at the right scope
Latency fed into the tracker must measure only API response time (after semaphore + token bucket), not total task time including queue wait. Otherwise, queuing time inflates the latency estimate, inflating Little's Law, creating a positive feedback loop.

### Complex prompts have fundamentally different latency profiles
Steps 2/3 process single responses with small prompts (P50: 3-5s). Step 4 discovery phases send 100-150 observations per prompt (P50: 16-24s, P95: 31-41s). A 90s timeout that works for small prompts causes unnecessary timeouts on discovery prompts. Timeout strategy must accommodate the heaviest prompt type in the pipeline, not the average.

### Don't throttle concurrency when capacity dwarfs demand
With high-tier API limits (Little's Law = 4250) and 26 tasks, starting at 50% of task count (13) wastes half the phase ramping up while the API is idle. The rate limiter and token bucket already prevent quota violations — the concurrency gate should only throttle when demand approaches capacity. Rule: `initial = max(50% of Little's Law, num_tasks)`.

---

## Changelog

### 2026-03-23: Multi-phase pipeline adoption + capacity-relative starting

**Problem:** Step 4 classifier (P1-P7) used a static semaphore of 15 with no rate limiting layers. Brought the full 4-layer stack from steps 2/3, but discovered that the 50% start fraction and 90s timeout were wrong for this workload. With 26 P1 tasks and Little's Law = 4250, starting at 13 (50% of 26) throttled throughput when the API had massive spare capacity. The 90s timeout tripped on P4 discovery calls with P95 latency of 41s.

**Fixes:**
1. **Capacity-relative starting:** `initial = max(50% of Little's Law, num_tasks)`. Small phases relative to capacity start at full concurrency.
2. **180s timeout ceiling:** Cold-start 180s, floor 60s. Accommodates heavy discovery prompts.
3. **Per-phase state:** Each phase gets independent gate, ramp, token bucket, latency tracker, and circuit breaker. No cross-phase contamination.
4. **Gather-pattern compatible:** The 4-layer stack works with `asyncio.gather` dispatch, not just worker/queue. Layers operate at `_llm_call()` level regardless of dispatch pattern.
5. **Phase latency summary:** Per-phase P10/P50/P95 latency + avg tokens printed at phase end for diagnostics.

**Key insight:** The strategy doc assumed a single-model, single-phase worker/queue pipeline. Multi-phase pipelines with varying models and prompt sizes need per-phase calibration and capacity-aware starting — the ramp is only useful when task count approaches or exceeds API capacity.

### 2026-03-22: Batch size guidance — when calibration is overhead

**Problem:** In multi-phase pipelines (e.g., qualitative_researcher), a single bootstrap was run once and shared across all phases. The bootstrap was calibrated against Phase 1 (facet discovery, ~9 tasks with 7.67s avg latency), but the same semaphore (58) was used for Phase 3 (facet assignment, 844 tasks) and Phase 6 (attribute assignment, 844 tasks) which have completely different prompt types and latency profiles. The 23s bootstrap overhead was also disproportionate for the small phases it was supposedly calibrating.

**Fix:** Added batch-size-based strategy guidance:
- <30 tasks: static semaphore + rate limiter (no calibration)
- 30-100 tasks: Option A (bootstrap — enough to amortize 3 probes)
- 100+ tasks: Option B (warm-up ramp — enough for recalibration to pay off)

**Key insight:** Calibration has a cost. 3 probe calls at ~8s each = 24s overhead. For a 9-task phase completing in 33s, that's 73% overhead. For an 844-task phase, it's <3%. Match the strategy to the workload, not the pipeline.

### 2026-03-22: Failure tracking and retry pass

**Problem:** With generous timeouts, <0.1% of tasks fail due to genuine transient errors (429 rate limits, connection drops, server 500s). These tasks received permanent fallback values — the original text passed through uncorrected. At scale (10k+ tasks), this means 10+ data points silently lost. The failures are transient and would succeed on a second attempt, but no retry mechanism existed.

**Fix:**
1. **Explicit failure tracking** — `failed_task_ids` set distinguishes true process failures from intentional LLM responses.
2. **Retry pass** — failed tasks requeued after main batch with 10% concurrency (min 5 workers). One retry only.
3. **Recovery reporting** — logs recovered count, permanently failed count, and failed IDs for auditability.

**Design rationale:** This complements rather than replaces the generous timeout strategy. Generous timeouts remain the primary defense against unnecessary failures. The retry pass handles the irreducible residual — true transient errors that no timeout strategy can prevent. Cost is near-zero (handful of API calls when API pressure is lowest). Single retry avoids masking systemic issues.

**Applies to:** All utilities using the prompt processing strategy (spellChecker, ideaExtractor, qualityFilter, codeAssigner, etc.)

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
