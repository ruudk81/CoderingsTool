# smoothRequester.py — Design Document

## Purpose

The smoothRequester is the **orchestrator, assembler and executor** of concurrent LLM API request processing. It owns the entire processing loop: workers, dispatch, pacing, throttling, monitoring, retry, and reporting.

The caller only provides: task list, prepare function, parse function, config params.

## How It Works

Rate pacing (PID + TokenBucket + AsyncLimiter) and concurrency control (state machine + ConcurrencyGate) run every tick simultaneously. Neither interferes because latency is measured after all gates — the state machine's signal (residual drift or P50 drift) is clean.

Effective concurrency = `min(rate_limit_concurrency, server_concurrency, num_tasks)`.

Whichever is smaller naturally caps the system. No mode switching, no oscillation.

## Architecture:
Role
Generic orchestrator for bulk concurrent LLM API calls. Step 3's IdeaExtractor delegates its heavy-lifting phase (per-response idea extraction) entirely to it.

### Interface (3 callbacks)
The caller provides three step-specific functions:

- `prepare_fn(task)` → `{prompt, response_model, temperature, max_tokens, ...}` — builds the LLM call
- `parse_fn(task, response)` → parsed result or None — converts raw LLM output
- `fallback_fn(task, reason)` → fallback result — handles permanent failures
SmoothRequester owns the actual LLM call so it can read response headers and reconcile tokens.

### Lifecycle (`process_all`)
1. Probe — single API call to discover rate limits (TPM/RPM) and check for openai-processing-ms header
2. System selection — header present → System A, absent → System B
3. Setup — rate pacing (TokenBucket + AsyncLimiter) + concurrency (ConcurrencyGate + state machine)
4. Queue + workers — tasks queued, N workers process concurrently
5. Monitoring loop (every 2s or N completions):
    - Warm-up calibration (after 15-30 completions): recalibrate avg_tokens + concurrency from measured data
    - Token correction: adjust if actual > estimate by >5%
    - PID rate adjustment (System B)
    - State machine concurrency adjustment
    - Spawn extra workers if concurrency increased
6. Retry pass — timed-out and failed tasks retried with reduced concurrency + extended timeouts
7. Save stats — P50, avg_tokens, empirical_capacity, tiktoken_offset persisted for next run's cold start

### Dual Control Systems
#### Rate pacing (always active):

- `TokenBucket` — TPM limiter
- `AsyncLimiter` — RPM limiter
- `PIDThroughputController` — adjusts arrival rate based on TPM utilization
- Effective concurrency = `min(rate_limit_concurrency, server_concurrency)`

#### Concurrency control (system-dependent):

- System A (header-aware): `ResidualLatencyTracker` computes `observed - openai-processing-ms`, `HeaderAwareConcurrencyController` state machine (RAMP-UP → STEADY ↔ BACKOFF → RECOVER) based on residual drift from baseline
- System B (client-side): Archived P50-drift state machine with in-flight P95/P100 monitoring

#### Concurrency control (system-dependent):Building Blocks (also exported for IdeaExtractor's conservative context phases)
- `TokenBucket` — leaky bucket for TPM
- `ConcurrencyGate` — dynamic semaphore with set_limit()
- `LatencyTracker` — EMA + adaptive timeout (P50 × 6, clamped to 180s)
- `TiktokenOffsetLearner` — learns API overhead vs tiktoken estimate (~300 token offset)
- `SimplifiedCircuitBreaker` — trips on >5% timeout rate, defense-in-depth

### Cold Start / Warm Start
Loads `modelPerfStats` cache keyed by `model:phase:dataset`. If ≥10 prior samples exist, uses stored P50, avg_tokens, empirical_capacity, and tiktoken_offset. Otherwise cold-starts with conservative defaults and calibrates during warm-up.


## Caller Interface

```python
requester = SmoothRequester(
    model="gpt-5.4-nano",
    dataset_key="M000000:Qd1_combined_full",
    phase_key="step3_idea_extraction",
)

def prepare_fn(task):
    """Return {prompt, response_model, temperature, max_tokens, ...}"""

def parse_fn(task, response):
    """Parse LLM response into step-specific result. Return None for empty."""

def fallback_fn(task, reason):
    """Create fallback for permanently failed task."""

results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
```

## Caching

- Empirical capacity: only saved when server was the binding constraint (not rate-capped). Prevents phantom measurements from rate-limited runs.
- Per provider:model → phase → dataset_key in `model_perf_stats.json`
- `was_rate_limited` not needed — determined by `min()` at save time
- Other fields (p50, avg_tokens, tiktoken_offset) always saved, EMA'd across runs

## Cost Tracking

smoothRequester does NOT calculate costs. The caller takes token snapshots before/after `process_all()` and passes them to CostTracker.

## Files

- `src/utils/smoothRequester.py` — the orchestrator + all building block classes
- `src/utils/llm.py` — HeaderCaptureTransport for response header capture
- `src/utils/modelPerfStats.py` — empirical data cache
- `src/pipeline/step_3_ideaExtractor/dev/_archived_p50_drift_state_machine.py` — P50-drift fallback (no headers)



