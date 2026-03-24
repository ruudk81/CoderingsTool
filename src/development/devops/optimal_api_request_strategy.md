# Optimal API Request Strategy

Reference implementation: Step 2 Quality Filter (single phase, single model, homogeneous tasks).

---

## Phase 1: Cold Start (t=0)

**Use provider-given constraints + model defaults to calculate a safe starting point.**

### Given (fetched from API headers)

- RPM (requests per minute)
- TPM (tokens per minute)

### Estimated (model-tier defaults, replaced at warm-up)

| Parameter | Method | Values |
|---|---|---|
| Latency | Model-tier lookup | nano=2s, mini=5s, default=10s, reasoning=20s |
| Output token ratio | Model type | chat=1.15x, reasoning=2.5x |
| avg_tokens | tiktoken on 10 sample prompts x output ratio | Varies per workload |

### Calculated

```
throughput       = min(RPM / 60, TPM / 60 / avg_tokens)    # requests/second
binding          = whichever is lower (RPM or TPM)
little_law       = throughput x latency_estimate            # theoretical concurrency budget
cold_semaphore   = min(RPM / 60 , 50)                   # safe starting concurrency
rate_limit       = 1 / throughput                           # seconds between admissions
```

### Why `min(RPM / 60 , 50)`?

- Capped at 50 because OpenAI has an undocumented concurrent connection ceiling (~50-200 for most tiers, per community reports). The cap is a safety net — the warm-up discovers the real ceiling.
- For constrained deployments (Azure 600 RPM): `min(10 x 2, 50) = 20` — naturally conservative.
- For high-tier deployments (OpenAI 30K RPM): `min(500 x 2, 50) = 50` — capped, safe.

### Deploy

Start `cold_semaphore` workers pulling from an asyncio.Queue. Rate limiter spaces admissions. TokenBucket guards TPM. All three layers active from t=0.

---

## Phase 2: Warm-Up (t=0 to t=10s)

**Measure reality and recalibrate.**

After 10 seconds of real work, we have:

- **Measured latency** (median of completed requests)
- **Measured avg_tokens** (mean of actual input + output tokens)
- **Measured throughput** (completions per second)

### Recalibrate

```
new_throughput   = min(RPM / 60, TPM / 60 / measured_avg_tokens)
new_little_law   = new_throughput x measured_latency
target_semaphore = min(new_little_law x 0.90, num_tasks)    # 90% of capacity
```

### Post-warm-up jump

The jump size depends on whether the warm-up showed stress:

- **No stress (latency stable, no timeouts):** Jump to `min(100, Little's Law)`
- **Stress detected (latency increasing or timeouts):** Hold at cold_start, ramp gently

The 100 cap is double the cold-start ceiling (50) — a known-safe doubling from a proven baseline.

Examples:

```
Azure  (cold_start=20, little_law=60):   no stress → jump to min(100, 60)   = 60  → ramp from 60
OpenAI (cold_start=50, little_law=1651): no stress → jump to min(100, 1651) = 100 → ramp from 100
```

### Ramp from post-warm-up (+10% every 5s)

After the jump, ramp gently toward `target_semaphore`, guided by monitoring signals:

```
OpenAI example: 100 → 125 → 156 → 195 → 244 → ... → signal says stop
Azure example:   18 →  23 →  29 →  36 →  45 → ... → signal says stop
```

At each 5s interval, check all four signals (Phase 3). Only ramp if ALL are green.

The ramp **stops** when either:
- `target_semaphore` is reached (full utilization), or
- Any signal goes yellow/red (practical ceiling discovered)

Whichever comes first becomes the **operating point**.

---

## Phase 3: Continuous Monitoring & Adaptation (t=10s onwards)

**Monitor four signals. Ramp up only when ALL are green. Throttle down when ANY is yellow/red.**

### Primary signals

| Signal | Green (ramp up) | Yellow (hold) | Red (throttle down) |
|---|---|---|---|
| **Queue health** | Queue shrinking or stable (outflow >= inflow) | Queue slowly growing | Queue rapidly growing |
| **RPM utilization** | < 80% of limit | 80-90% of limit | > 90% of limit |
| **TPM utilization** | < 80% of limit | 80-90% of limit | > 90% of limit |
| **Latency trend** | P95 stable or decreasing | P95 increased >10% vs previous check | P95 increased >25% vs previous check |

Limit = 90% of RPM/TPM constraint, leaving 10% headroom. So "90% of limit" = ~81% of raw RPM/TPM.

Latency trend detects API-side pressure that doesn't show up in RPM/TPM utilization — the API accepts requests within limits but queues them server-side, causing latency spikes.

### Defensive signals

| Signal | Action |
|---|---|
| Timeout rate > 5% | Circuit breaker: reduce concurrency by 20%, cooldown 60s |
| 429 rate limit error | Exponential backoff on affected request, reduce admission rate |

### Ramp logic

- All four green → increase semaphore by +10% (add workers) toward `target_semaphore`
- Any yellow → hold current semaphore
- Any red → reduce semaphore by 20% (remove workers), reduce admission rate

### Monitoring interval

Every 5 seconds, evaluate all three signals. Monitor internally on every completion (for latency/token tracking), but only adjust concurrency every 5s to avoid oscillation.

---

## Architecture (per request flow)

```
Task Queue → Worker picks task → Rate Limiter (spacing) → Token Bucket (TPM guard) → API call
                                                                                        |
                                                                              Latency tracker records
                                                                              Token usage recorded
                                                                              Queue health updated
                                                                                        |
                                                                              Monitor evaluates (every 5s):
                                                                                queue + RPM% + TPM%
                                                                                        |
                                                                              Ramp / Hold / Throttle
```

**Worker count = semaphore.** Adding a worker = increasing concurrency. Removing a worker = decreasing concurrency. Queue depth = `queue.qsize()` = directly observable.

---

## What This Strategy Achieves

- **No hardcoded concurrency numbers** — cold start derived from RPM, warm-up replaces estimates with measurements
- **Self-tuning across providers** — Azure (600 RPM -> start at 20) and OpenAI (30K RPM -> start at 50, capped) use the same logic
- **Self-tuning across models** — nano (1.5s latency) and reasoning (20s latency) discover their own operating point
- **The binding constraint is discovered, not assumed** — whether it's RPM, TPM, or the undocumented concurrency ceiling, the system finds it through monitoring and stops ramping
- **Graceful under pressure** — any single signal going yellow/red triggers a defensive response before failures cascade

---

## Timeout Strategy

| Phase | Timeout | Rationale |
|---|---|---|
| Cold start | `max(latency_estimate x 3, 60s)` | Generous floor, unknown conditions |
| After warm-up | `max(P95_latency x 3, 60s)` | Data-driven, adaptive |
| Hard ceiling | 180s | Prevents infinite waits |

Timeout is applied **inside** the semaphore/worker — only counts API response time, not queue wait time.

---

## Retry Strategy

| Error type | Retry? | Strategy |
|---|---|---|
| Timeout | Yes, 1 retry | Same request, fresh timeout |
| 429 (rate limit) | Yes, with backoff | Exponential backoff: 2s, 4s, 8s |
| 500/502/503 (server) | Yes, 1 retry | Immediate retry |
| Validation error (Pydantic) | Yes, via instructor | instructor's built-in retry (max_retries=2) |
| 400 (bad request) | No | Log and skip — request is malformed |

---

## Monitoring Output

Per-phase progress line every 5 seconds:

```
[STEP2] 150/2000 (12.3/s) | TPM:45% RPM:62% Conc:35/50 Queue:12 | P50:1.2s P95:2.1s
```

Fields:
- Completion count and throughput
- TPM/RPM utilization as percentage of limit
- Current / target concurrency
- Queue depth (tasks waiting for a worker)
- Latency percentiles (P50, P95)

Suppress output when no completions in last interval.

---

## Scope

This document covers **single-phase, single-model** workloads (steps 1-3). Multi-phase coordination (steps 4-6, where phases share rate limits and use different models) requires an extension document.
