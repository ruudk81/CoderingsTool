# Audit & Plan: code_assignment.py (Step 4, Instance 1)

## Context

Auditing `src/development/step_4_classNcoder/code_assignment.py` against the prompt processing strategy best practices (`src/development/devops_best practices/prompt processing strategy.md`). This is Instance 1 of 4 full-dataset prompt processing instances in Step 4. The audit findings and plan will serve as a reusable playbook for Instances 2-4.

## File Under Audit

`src/development/step_4_classNcoder/code_assignment.py`

- **Main entry**: `assign_all()` → `_assign_all_async()`
- **Worker**: `_worker()` (L572)
- **Process task**: `_process_task_with_retry()` (L651)
- **1 LLM call per idea** (full dataset, per-idea processing)

---

## Audit Results

### Aligned

| Aspect | Strategy Doc | code_assignment.py | Status |
|--------|-------------|-------------------|--------|
| **Concurrency init** | Option B (warm-up ramp, no probes) | Option B — theoretical Little's Law + 50% start, no probe calls (L837-847) | Aligned |
| **ConcurrencyGate** | Dynamic limits, replaces Semaphore | ConcurrencyGate imported from ideaExtractor (L65, L844) | Aligned |
| **ConcurrencyRamp** | Completion-based 50%→90% | Present, completion-based (L845-847, L931-945) | Aligned |
| **Circuit breaker** | CLOSED/OPEN/RECOVERING | Present (L850-852, L418-420) | Aligned |
| **PID controller** | Asymmetric gains, adjusts arrival rate | Present, every 20s (L859-867, L444-447) | Aligned |
| **Timeout floor** | 60s, P95×3 | 60s floor (L12, L693-694) | Aligned |
| **Timeout after semaphore** | Compute after gate | After gate (L693-694) | Aligned |
| **Warm-up calibration** | 15-30 completions, P10 latency | Present, adaptive target, P10 latency (L363-367, L875-929) | Aligned |
| **Worker scaling** | Dynamic after warm-up | Dynamic — spawns extra workers after calibration (L435-442) | Aligned |
| **Rate limiting layers** | 4 layers | All 4: ConcurrencyGate → TokenBucket → AsyncLimiter → Timeout (L691-700) | Aligned |
| **Response mapping** | By unique ID | By idea_id from original task, not LLM response (L774-781) | Aligned |
| **Data points per call** | 1 | 1 idea per call | Aligned |
| **Failure tracking** | Explicit tracking | `_failure_log` list (L186, L627-631) | Aligned |

### Gaps

| # | Aspect | Strategy Doc | code_assignment.py | Severity |
|---|--------|-------------|-------------------|----------|
| 1 | **Retry pass** | Single retry with reduced concurrency, then permanent fallback | Has retry (L457-486) but it's **serial** — processes timed-out tasks one by one, not through reduced-concurrency queue+workers | **Low** — works but slower than strategy pattern |
| 2 | **Retry scope** | Retry both timeouts AND exception failures | Only retries timed-out tasks (L457: `if timed_out`); exception failures from `_failure_log` are NOT retried | **Medium** — exception failures (429s, connection errors) are recoverable but permanently lost |

---

## Findings Detail

### Gap 1: Serial retry vs queue-based retry

Current retry (L457-486):
```python
if timed_out:
    for idx, task in timed_out:
        result = await self._process_task_with_retry(task)  # Serial, one at a time
```

Strategy doc recommends: queue + reduced workers (10% of original, min 5) for parallel retry. The serial approach works for small numbers of timeouts (<5) but becomes a bottleneck if there are 20+ failures.

### Gap 2: Exception failures not retried

The worker (L572-637) catches exceptions after Tenacity exhausts 5 retries and adds to `_failure_log`. These tasks get permanent fallback. But Tenacity retries happen under the same rate pressure as the main batch — a retry pass after the batch (when API pressure is lowest) would likely recover many of these.

---

## Plan

### Change 1: Expand retry to include exception failures

Collect both `timed_out` tasks (returned `None` from `_process_task_with_retry`) AND exception failures (from `_failure_log`) into a single retry list.

**Where:** After main batch completes (L457), before final stats.

**What:**
- Build retry list from `timed_out` + tasks whose `respondent_id` is in `_failure_log`
- Clear `_failure_log` before retry pass

### Change 2: Queue-based retry with reduced concurrency

Replace the serial retry loop with queue+workers at reduced concurrency.

**Where:** Replace L457-486.

**What:**
- Create retry queue
- Launch `max(5, min(len(retry_tasks), num_workers // 10))` retry workers
- Set `self._latency_tracker.retry_mode = True` (need to add `retry_mode` to LatencyTracker if not present)
- Process retry queue
- Merge results back into main results array
- Report recovery stats

### Pre-check: LatencyTracker retry_mode

The `LatencyTracker` used in code_assignment.py is imported from ideaExtractor_exp.py. Check if it already has `retry_mode`. If so, no change needed. If not, add it (same pattern as qualityFilter).

### What stays unchanged

Everything else is already aligned. No changes to:
- ConcurrencyGate, ConcurrencyRamp, CircuitBreaker
- PID controller
- Timeout strategy (60s floor, after gate)
- Warm-up calibration
- Worker scaling
- Response mapping
- Rate limiting layers

---

## Reusable Patterns for Instances 2-4

When auditing `qualitative_researcher.py` (Instances 2-4), apply these checks:

1. **Does it use ConcurrencyGate or asyncio.Semaphore?** → Should be ConcurrencyGate
2. **Does it compute timeout after gate?** → Should be after, not before
3. **Does it have a retry pass for both timeouts and exceptions?** → Should have queue-based retry
4. **Does it have warm-up calibration?** → Should have if processing >50 tasks
5. **What's the timeout floor?** → 60s for multi-prompt steps, 20s for single-prompt steps
6. **Is there a circuit breaker?** → Should have for high-volume processing

Note: Instances 2-3 use **batched** LLM calls (multiple ideas per prompt), so the retry pattern needs to handle batch-level failures, not per-idea failures. Instance 4 (code generation) is domain-level (3-5 calls) — strategy doc patterns are overkill.

---

## Verification

1. **Syntax check:** `python -c "import ast; ast.parse(open('code_assignment.py').read())"`
2. **Run with small sample:** Execute pipeline with `sample_size=20` and verify:
   - Retry pass appears in output (even if 0 failures to retry)
   - Stats show recovered count
3. **Compare with backup:** Diff output stats between old and new
