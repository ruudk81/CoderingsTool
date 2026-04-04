# smoothRequester.py — Design Document

## Purpose

The smoothRequester is the **orchestrator, assembler and executor** of concurrent LLM API request processing. It owns the entire processing loop: workers, dispatch, pacing, throttling, monitoring, retry, and reporting.

The caller only provides: task list, prepare function, parse function, config params.

## How It Works

### Integrated Rate + Concurrency Control

Both rate pacing (RPM/TPM) and concurrency control are always active simultaneously. The binding constraint is determined continuously by:

```
rate_limit_concurrency = Little's Law from RPM/TPM limits
server_concurrency = discovered via state machine (empirical or cold start)
effective = min(rate_limit_concurrency * 0.95, server_concurrency)
```

The 0.95 bias favors rate limiting — exceeding rate limits causes 429 hard failures, while under-utilizing server capacity only costs throughput.

### Binding Constraint Determines Active Controller

Each tick:
- If rate-limited: PID adjusts arrival rate, state machine is idle
- If throughput-bound: state machine adjusts concurrency, PID is idle
- Binding can shift mid-run (logged as `[BINDING]`)

### Two Signal Sources (cleanly separated)

Depending on header availability (one probe call at startup):
- **With headers (OpenAI):** Concurrency via residual latency drift (observed - openai-processing-ms)
- **Without headers (Azure):** Concurrency via P50 latency drift

Both use the same integrated framework — only the concurrency signal source differs.

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

The caller does NOT manage workers, gates, ticks, warm-up, PID, retry, inflight tracking, token estimation, or reporting.

## What smoothRequester Owns

1. Probe call (rate limits + header detection)
2. Cache loading (empirical data from modelPerfStats)
3. Rate pacing (TokenBucket + AsyncLimiter, PID-adjusted when rate-limited)
4. Concurrency control (ConcurrencyGate + state machine)
5. Worker management (queue, spawning, dynamic concurrency increase)
6. Per-request gate (semaphore → token bucket → rate limiter → timeout)
7. Per-request outcome recording (latency, tokens, headers, circuit breaker)
8. Token estimation (tiktoken + learned offset)
9. Tick evaluation (binding constraint → right controller → progress line)
10. Warm-up calibration (one-shot token + concurrency recalibration)
11. Monitoring and reporting (progress lines, BACKOFF events, BINDING shifts)
12. Retry pass (timed-out + failed tasks, reduced concurrency)
13. End-of-run stats (saved to cache for next run)

## Cost Tracking

smoothRequester does NOT calculate costs directly. It tracks token usage internally for estimation and reconciliation. The caller takes token snapshots before/after `process_all()` and passes them to CostTracker.

## Verbose Reporting

Progress lines are printed directly by smoothRequester. Format depends on binding constraint:
- Throughput-bound: `inflight, arrival, completing/s, proc, residual, drift%, baseline, state`
- Rate-limited: `tok or req rate, limit, pace%, completing/s, thru`

## Files

- `src/utils/smoothRequester.py` — the orchestrator + all building block classes
- `src/utils/llm.py` — HeaderCaptureTransport for response header capture
- `src/utils/modelPerfStats.py` — empirical data cache (load/save)
- `src/pipeline/step_3_ideaExtractor/dev/_archived_p50_drift_state_machine.py` — P50-drift fallback (no headers)
