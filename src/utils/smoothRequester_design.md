# smoothRequester.py — Design Document

## 1. What We Agreed On

### Purpose

The smoothRequester.py is the **orchestrator, assembler and executor** of processing API requests. It is not a bag of tools — it owns the entire processing loop: workers, dispatch, pacing, throttling, monitoring, retry, and reporting.

The caller only provides information/params required for getting the work done: model, provider, dataset key, task list, and a per-task processing function. Everything about *how* and *when* tasks are dispatched is internal to the smoothRequester.

### Constraints We Handle

- **RPM** (requests per minute) — provider/model specific
- **TPM** (tokens per minute) — provider/model specific
- **Throughput/concurrency** — server-side limit, not exposed by API, must be discovered

### How We Handle Them

- Async processing with semaphore-based worker slots
- Rate pacing: PID-controlled arrival rate, token bucket, request limiter
- Concurrency control: state machine that adjusts semaphore based on server feedback
- Both controls always active simultaneously — whichever binds first gates naturally

### Two Systems (cleanly separated, not interwoven)

Depending on whether we have credentials to access server-side info:

**System A (server-side data — OpenAI with headers):**
- Rate pacing: TokenBucket + AsyncLimiter (passive rails at headroom limits)
- Concurrency: HeaderAwareConcurrencyController driven by residual latency drift (observed - openai-processing-ms)
- Signal source: response headers (openai-processing-ms, remaining-requests/tokens)

**System B (client-side data — Azure, no headers):**
- Rate pacing: TokenBucket + AsyncLimiter, PID-adjusted based on inferred TPM utilization
- Concurrency: P50-drift state machine + Little's Law recalculation
- Signal source: observed latency, inferred token/request rates

### Core Contract

**The smoothRequester is the orchestrator, assembler and executor of processing API requests.**

It is NOT a bag of tools the caller wires together. It owns the entire processing loop.
The caller only provides information/params required for getting the work done.

### What smoothRequester.py Owns

1. **Probe call** — discovers rate limits and header availability
2. **Cache loading** — loads empirical data from modelPerfStats
3. **Setup** — creates all rate pacing and concurrency control components
4. **System selection** — activates System A or B based on probe result
5. **Worker management** — creates async workers, manages queue, spawns extra on concurrency increase
6. **Per-request dispatch** — gate (semaphore + token bucket + rate limiter), inflight tracking
7. **Per-request outcome recording** — latency, token reconciliation, header reading, circuit breaker
8. **Token estimation** — estimates tokens per prompt (tiktoken + learned offset)
9. **Tick evaluation** — evaluates both controls, adjusts concurrency and arrival rate
10. **Warm-up calibration** — one-shot recalibration from production data
11. **Monitoring and reporting** — progress lines, diagnostics, verbose output
12. **Retry pass** — retries timed-out and failed tasks with reduced concurrency
13. **End-of-run stats** — assembles and returns stats for cache storage
14. **Cost tracking integration** — tracks token usage for cost reporting

### What the Caller Provides

The caller provides ONLY:
1. **Task list** — the items to process
2. **Process function** — step-specific logic per task (build prompt, call LLM, parse response)
3. **Config params** — model, provider, dataset key
4. **VerboseReporter** — for diagnostic output

The caller calls:
```python
results = await limiter.process_all(tasks, process_fn)
stats = limiter.get_stats_for_cache()
```

The caller does NOT manage workers, gates, ticks, warm-up, PID, retry, inflight tracking,
token estimation, or reporting. Those are all internal to smoothRequester.

### Design Decisions

- **Probe call**: smoothRequester makes this itself (needs API client)
- **Config**: smoothRequester imports directly from config.py (provider, model, rate limits)
- **Cache**: smoothRequester loads empirical data itself, returns stats dict at end for caller to save
- **Cost tracking**: returns data for caller to pass to CostTracker
- **Verbose reporting**: integrates with VerboseReporter passed by caller
- **Token estimation**: smoothRequester has its own tiktoken estimator
- **Worker management**: smoothRequester owns workers, queue, spawning
- **Inflight tracking**: smoothRequester owns _inflight_starts — internal state
- **Retry pass**: smoothRequester owns retry logic for timed-out and failed tasks

---

## 2. Code Investigation — What Exists and Where

All line numbers from `src/pipeline/step_3_ideaExtractor/ideaExtractor.py` unless noted.

### 2.1 Probe Call (lines 1678-1714)

`_fetch_rate_limits_from_api()` — makes minimal API call, returns `(RateLimits, has_server_headers)`.

- Creates `AsyncOpenAI` client for Azure or OpenAI (lines 1685-1700)
- Uses `.with_raw_response.create()` to access headers (lines 1702-1710)
- Extracts rate limits via `extract_rate_limits_from_response()` from `utils/llm.py` (line 1712)
- Detects headers: `has_server_headers = 'openai-processing-ms' in response.headers` (line 1713)

**Moves to:** `RateLimiter.__init__()` or `RateLimiter.probe()`

### 2.2 Loading Empirical Data (lines 274-305)

In `IdeaExtractor.__init__()`:

- `load_stats()` from modelPerfStats (line 277)
- `get_dataset_phase_stats()` with dataset_key (lines 278-280)
- Reads: `p50_latency_s`, `tiktoken_offset`, `empirical_capacity`, `avg_tokens` (lines 282-305)
- All gated by `sample_count >= 10`

**Moves to:** `RateLimiter.__init__()` — loads its own cache

### 2.3 Rate Pacing Setup (lines 1734-1742)

In `_initialize_rate_limiters()`:

- Arrival rate = `min(RPM * headroom / 60, TPM * headroom / avg_tokens / 60)` (lines 1734-1737)
- `AsyncLimiter(1, time_period=1.0 / arrival_rate)` (line 1738)
- `TokenBucket(int(TPM * headroom))` (line 1742)

**Config used:** `processing_config.rate_limit_headroom` (from config.py, default 0.9)

**Moves to:** `RateLimiter.__init__()` — always created, both systems

### 2.4 Concurrency Setup (lines 1744-1791)

In `_initialize_rate_limiters()`:

**Starting concurrency (lines 1744-1756):**
- Little's Law cap: `compute_optimal_concurrency(api_limits, latency, avg_tokens, headroom)` (lines 1745-1748)
- Empirical capacity from cache or COLD_START_CAP (50) (lines 1750-1754)
- `ConcurrencyGate(target)` (line 1757)

**System A (lines 1762-1765):**
- `ResidualLatencyTracker()` 
- `HeaderAwareConcurrencyController(starting=target)`
- `SimplifiedCircuitBreaker()`

**System B (lines 1767-1791):**
- `ArchivedP50StateMachine(starting=target, bottleneck="throughput", config=...)`
- `ArchivedCircuitBreaker(config=...)`
- `PIDThroughputController(target_utilization=..., kp_up=..., etc.)`
- `RealTimeTPMTracker(window_seconds=60.0)`
- `RealTimeRPMTracker(window_seconds=60.0)`

**Moves to:** `RateLimiter.__init__()` — System A or B based on probe result

### 2.5 Per-Request Gate (lines 1324-1343)

In `process_task()`:

```python
async with self.semaphore:                              # 1. ConcurrencyGate
    timeout = self.latency_tracker.get_timeout(est_tokens)
    await self.tpm_bucket.wait_and_acquire(est_tokens)   # 2. TokenBucket
    api_start = time.perf_counter()
    self._inflight_starts[task_id] = api_start
    async with self.rate_limiter:                        # 3. AsyncLimiter
        response = await asyncio.wait_for(
            llm_create_async(...),
            timeout=timeout
        )
```

**Moves to:** `RateLimiter.gate()` context manager — acquires all three layers

### 2.6 Recording Outcomes (lines 1345-1410)

In `process_task()` after successful API call:

- **Latency:** `self.latency_tracker.add(latency)` (line 1347)
- **Circuit breaker:** `self.circuit_breaker.record_completion()` (line 1351)
- **Header reading (System A):** Lines 1354-1373
  - Look up headers via `self._header_transport.get(client_id)`
  - Feed `self._residual_tracker.add(latency, processing_ms)`
  - Track `remaining_requests` and `limit_requests` for budget pressure
  - Header detection majority vote (first 10 responses)
- **Token reconciliation:** Lines 1375-1410
  - Extract actual tokens from response usage
  - `await self.tpm_bucket.reconcile(delta)` (line 1400)
  - Feed TPM/RPM trackers (lines 1407-1410)
  - Learn tiktoken offset (line 1403)

**On timeout:** `self.circuit_breaker.record_timeout()` (line 1506)

**Moves to:** `RateLimiter.record_success()` and `RateLimiter.record_timeout()`

### 2.7 Tick Evaluation (lines 1911-2089)

Four methods:

**`_tick_throughput()` (lines 1911-1930):** Dispatcher — routes to header_aware or p50_drift_fallback

**`_tick_header_aware()` (lines 1932-2000):**
- Reads residual tracker: median, normalized, trend
- Computes header pressure from remaining/limit requests
- Calls `sm.evaluate(median_residual_ms, normalized_residual, trend, header_pressure, throughput, p50)`
- Updates semaphore if concurrency changed
- Logs BACKOFF events
- Formats progress line with: inflight, arrival, completing/s, proc, residual, drift%, baseline, state

**`_tick_p50_drift_fallback()` (lines 2002-2056):**
- Computes in-flight P95/P100 (filtered by signal_cutoff)
- Checks circuit breaker
- Calls `sm.evaluate(p50, inflight_p95, inflight_p100, throughput, inflight)`
- Formats progress line with: inflight, completing/s, P50, drift%, p95/p100 ratios, state

**`_tick_rate_limited()` (lines 2058-2089):**
- Smooths throughput over 5-tick window
- Every 10s: recalculates Little's Law concurrency from measured data
- Formats progress line with: constraint (tok or req), pace%, throughput

**Moves to:** `RateLimiter.tick()` — one method, internally routes based on system

### 2.8 Warm-Up Calibration (lines 1826-1888)

**`_calibrate_tokens_from_warm_up()` (lines 1826-1857):**
- Triggered after 15-30 completions
- Updates `avg_tokens` from measured data
- Recalculates arrival rate
- Resets PID if active
- Shared by both systems

**`_calibrate_concurrency_from_warm_up()` (lines 1859-1888):**
- System B only
- Calls `_calibrate_tokens_from_warm_up()` first
- Recalculates Little's Law concurrency with measured latency
- Updates semaphore

**Triggered from monitoring loop:** Lines 2483-2492

**Moves to:** `RateLimiter.calibrate_warm_up()` — handles both systems internally

### 2.9 PID + Token Correction (lines 2093-2160+)

**`_adjust_throughput_if_needed()` (lines 2093-2125):**
- Fires when actual_avg / estimate > 1.05
- Updates `avg_tokens`
- Both systems

**`_apply_pid_adjustment()` (lines 2127-2160+):**
- Reads current TPM from tracker
- PID computes adjustment
- Updates arrival rate (AsyncLimiter)
- System B only

**Triggered from monitoring loop:** Lines 2494-2498

**Moves to:** Internal to `RateLimiter.tick()` 

### 2.10 End-of-Run Stats (lines 2760-2776)

Assembles measurements dict:
- `p50_latency_s` (from latency tracker)
- `avg_tokens` (from actual_total_tokens)
- `empirical_capacity` (95% of last_healthy_concurrency)
- `has_server_headers`
- `tiktoken_offset` (if learned)

Saves via `update_dataset_phase_stats()` with overwrite for empirical_capacity.

**Moves to:** `RateLimiter.get_stats_for_cache()` — returns dict, caller saves

### 2.11 Monitoring Loop Logic (lines 2370-2507)

Currently ~137 lines in `process_all_tasks_async()`:
- Healthy throughput tracking (lines 2381-2435)
- TPM/RPM utilization computation (lines 2400-2412)
- Tick context building + dispatch (lines 2446-2477)
- Warm-up calibration trigger (lines 2483-2492)
- PID adjustment interval (lines 2494-2498)
- Worker spawning on concurrency increase (lines 2501-2507)

After move, the loop becomes:
- `limiter.tick()` for progress + adjustments
- `limiter.calibrate_warm_up()` check
- Worker spawning based on `limiter.concurrency`

### 2.12 Config Values Used

**From config.py:**
- `API_PROVIDER` — probe call routing
- `OPENAI_API_KEY`, `AZURE_OPENAI_*` — probe call credentials
- `FALLBACK_TPM` (150000), `FALLBACK_RPM` (30000) — when probe fails
- `rate_limit_headroom` (0.9) — from ProcessingConfig

**From config_ideaExtractor.py:**
- `TIMEOUT_FLOOR_SECONDS` (10.0), `DEFAULT_TIMEOUT_SECONDS` (10.0)
- `DEFAULT_LATENCY_SECONDS` (2.0)
- `COLD_START_CAP` (50)
- `PROGRESS_REPORT_INTERVAL` (5), `DIAGNOSTIC_INTERVAL` (10), `ADJUSTMENT_INTERVAL` (20)
- `DEFAULT_AVG_TOKENS` (1500)
- `THROUGHPUT_ADJUSTMENT_MIN_SAMPLES` (10), `THROUGHPUT_ADJUSTMENT_THRESHOLD` (1.05)
- All PID config values (kp_up=0.4, kp_down=0.2, etc.)
- All header-aware config values (drift thresholds, etc.)
- All concurrency control config values (ramp_step_pct=0.025, backoff_pct=0.9, etc.)
- Warm-up config (sample_min=15, sample_max=30)
- Specifier config (not rate-limiting related — stays in ideaExtractor)

**Note:** Most of these are currently step-3-specific config. For smoothRequester.py to be generic, these need to become constructor parameters or a config dataclass that any step can provide.

### 2.13 Dependencies on Other Utils

- `utils/llm.py`: `create_client`, `llm_create_async`, `RateLimits`, `extract_rate_limits_from_response`, `HeaderCaptureTransport`
- `utils/modelPerfStats.py`: `load_stats`, `save_stats`, `get_dataset_phase_stats`, `update_dataset_phase_stats`
- `utils/verboseReporter.py`: `VerboseReporter` (diagnostic output)
- `aiolimiter`: `AsyncLimiter` (external package)
- `numpy`: percentile calculations

---

## 3. What Does NOT Move

These stay in ideaExtractor.py (step-3 business logic):

- `process_task()` — prompt building, LLM call, response parsing, idea extraction, empty-idea retry
- `worker()` — queue management, exception handling, failure logging, fallback creation
- `process_all_tasks_async()` — phase orchestration (context extraction, dimension selection, domain discovery, bootstrap), retry pass
- `extract()` — top-level orchestration
- All prompt/taxonomy/domain logic
- Cost tracker phase snapshots
- Encoding/tiktoken setup (step uses specific model)
- Token estimation (`estimate_tokens()`, `_estimate_preprocessed_tokens()`)

---

## 4. Resolved Questions

1. **Token estimation** — smoothRequester has its own tiktoken estimator. Internal.

2. **Worker spawning** — smoothRequester owns workers, queue, and spawning. The caller provides a task list and a process function. `process_all()` handles everything.

3. **Inflight tracking** — smoothRequester owns `_inflight_starts`. Internal state used by tick evaluation and signal filtering.
