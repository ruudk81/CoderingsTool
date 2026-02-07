# Rate Limiting Audit: ideaExtractor_exp.py

Audit performed after fixing the same class of bugs in step_2 `qualityFilter_exp.py`.
Lessons learned there applied systematically to step_3.

## Architecture Overview

The ideaExtractor uses a 3-layer rate limiting stack:
1. **AsyncLimiter** (pacing) - controls request arrival rate
2. **TokenBucket** (TPM) - tracks token budget per minute
3. **Semaphore** (concurrency) - caps in-flight API calls

Plus two feedback mechanisms:
- **Threshold adjustment** - step correction when actual tokens >> bootstrap estimate (>5%)
- **PID controller** - continuous fine-tuning based on real-time TPM utilization

---

## Finding 1: `int()` Truncation in AsyncLimiter (HIGH)

**The same bug we fixed in step_2.** Four locations use `int()` on the arrival rate:

| Location | Line | Context |
|----------|------|---------|
| `_initialize_rate_limiters()` | 2005 | Initial setup after bootstrap |
| `_initialize_conservative_rate_limiters()` | 2039 | Conservative setup for context phase |
| `_apply_pid_adjustment()` | 2090 | PID reinstalls rate limiter |
| `_adjust_throughput_if_needed()` | 2132 | Threshold adjustment reinstalls |

**Pattern (buggy):**
```python
if arrival_rate < 1:
    self.rate_limiter = AsyncLimiter(1, time_period=1/arrival_rate)
else:
    self.rate_limiter = AsyncLimiter(int(arrival_rate), time_period=1.0)  # BUG
```

**Impact:** `int(1.88)` = 1 -> 47% throughput loss. `int(3.7)` = 3 -> 19% loss.
The PID tries to compensate by increasing rate, but each reinstall re-truncates.

**Fix:** Replace all 4 with:
```python
self.rate_limiter = AsyncLimiter(1, time_period=1.0/arrival_rate)
```

Works for all rates. Provides smooth, non-bursty pacing.

---

## Finding 2: PID Tracking Mismatch (MEDIUM)

`self.current_arrival_rate` stores the un-truncated float, but the actual
`AsyncLimiter` enforces `int(rate)`. The PID reads `self.current_arrival_rate`
as `old_rate` (line 2075), so its adjustment math is based on a rate the limiter
isn't actually enforcing.

**Example:** Rate calculated as 3.7/s -> limiter enforces 3/s -> PID thinks rate
is 3.7/s -> PID's corrections are systematically biased.

**Fix:** Resolves automatically when Finding 1 is fixed (limiter matches the float).

---

## Finding 3: Sparse Mid-Run Diagnostics (MEDIUM)

The monitoring loop (lines 2393-2401) only reports when the token bucket is low:
```python
if bucket_status['low_tokens']:
    self.verbose_reporter.stat_line(f"Token bucket low: ...")
```

**Missing:**
- Token drift vs bootstrap (actual vs estimate, % difference)
- Whether threshold adjustment is expected ("below/exceeds 5% threshold")
- Current pacing rate and whether PID is active
- Adjustment count so far

Compare with step_2's diagnostics which show all of this every `DIAGNOSTIC_INTERVAL`.

**Fix:** Add the same diagnostic pattern we built for step_2, adapted for PID.

---

## Finding 4: Final Summary Lacks Context (MEDIUM)

The final stats (lines 2456-2458) show:
```
Token usage summary: Initial X -> Actual Y (+Z tokens, +N%)
Throughput analysis: Expected X/s -> Optimal Y/s with perfect estimation
```

**Missing:**
- Current (adjusted) avg_tokens vs actual (residual drift)
- Whether residual drift is within threshold
- "Bootstrap X/s -> Optimal Y/s" labeling (not "Expected")

**Fix:** Port step_2's final diagnostic pattern with threshold context.

---

## Finding 5: Double Headroom Application (LOW)

`rate_limit_headroom` (e.g., 0.85) is applied to BOTH:
1. Arrival rate calculation (line 1999): `limits.rpm * headroom / 60`
2. Token bucket capacity (line 660): `limits.tpm * headroom`

Net effect: system operates at `headroom^2` = 0.85 * 0.85 = 72% of true capacity.

**Assessment:** This is conservative but not necessarily wrong - the arrival rate
headroom prevents hitting limits, the bucket headroom prevents burst overruns.
However, the PID target of 85% utilization then means 85% of the headroom-reduced
bucket, which is 85% * 85% = ~72% of true TPM. Worth noting but LOW priority.

**Fix:** Not recommended to change now - would require careful testing. Document only.

---

## Finding 6: One-Directional Threshold (LOW)

The threshold check (line 2118) only triggers when actual > estimate:
```python
if ratio <= THROUGHPUT_ADJUSTMENT_THRESHOLD:
    return False
```

If bootstrap OVERESTIMATES tokens, there's no threshold correction to speed up.
The PID handles this case, but convergence is gradual (20s intervals, 60s window).

**Assessment:** The PID compensates, so this is LOW priority. The asymmetry is
intentional: underestimation causes 429s (urgent), overestimation wastes time
(gradual PID correction is acceptable).

---

## Summary: What to Fix

### Must fix (HIGH impact):
1. **int() truncation** - 4 locations, one-line fix each

### Should fix (MEDIUM impact):
2. ~~PID tracking mismatch~~ - auto-resolves with fix 1
3. **Mid-run diagnostics** - add threshold context, token drift, pacing info
4. **Final summary** - add adjusted/residual tracking, threshold context

### Document only (LOW impact):
5. Double headroom - conservative, not a bug
6. One-directional threshold - PID compensates
