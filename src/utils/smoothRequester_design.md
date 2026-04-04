# smoothRequester.py — Design Document

## Purpose

The smoothRequester is the **orchestrator, assembler and executor** of concurrent LLM API request processing. It owns the entire processing loop: workers, dispatch, pacing, throttling, monitoring, retry, and reporting.

The caller only provides: task list, prepare function, parse function, config params.

## How It Works

### Both Controllers Always Active

Rate pacing (PID + TokenBucket + AsyncLimiter) and concurrency control (state machine + ConcurrencyGate) run every tick simultaneously. Neither interferes because latency is measured after all gates — the state machine's signal (residual drift or P50 drift) is clean.

Effective concurrency = `min(rate_limit_concurrency, server_concurrency, num_tasks)`.

Whichever is smaller naturally caps the system. No mode switching, no oscillation.

### Dynamic State Display

Each tick, the `min()` determines what's reported:

- `RATE-CAPPED` — rate limits are the binding constraint
- State machine state (`RAMP-UP`, `STEADY`, `BACKOFF (N→M)`, `RECOVER`) — server is the binding constraint
- `WARM-UP` — first 5 seconds, building baseline data

### Two Signal Sources

Depending on header availability (one probe call at startup):
- **With headers (OpenAI):** `thru` = residual latency (observed - openai-processing-ms)
- **Without headers (Azure):** `thru` = P50 observed latency in ms

Same display format, same label, different data source.

### Report Line Format

```
[PHASE6] 500/1075 | inflight:92 | req:57/450 (13%) | thru:207ms/193ms (+7%) | completing:50/s | RAMP-UP
[PHASE6] 693/1075 | inflight:106 | req:58/450 (13%) | thru:292ms/193ms (+50%) | completing:52/s | BACKOFF (106→95) | deferred:1
[PHASE6] 20/50 | inflight:3 | tok:2.1k/2.2k (94%) | thru:2100ms/2100ms (+0%) | completing:1/s | RATE-CAPPED
```

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
