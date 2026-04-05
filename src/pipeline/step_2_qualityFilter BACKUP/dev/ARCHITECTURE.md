# Step 2 — Architecture

## Design Intent

LLM-based quality assessment of preprocessed survey responses. Each response is evaluated individually and flagged as noise or kept for downstream analysis.

Key design choices:
- **Per-response LLM grading**: each response evaluated independently with full survey context
- **Dual prompt strategy**: nano uses raw text + XML tags (Pattern A); mini/default uses instructor + Pydantic (Pattern B)
- **5-category noise taxonomy**: don't know, not applicable, absence of answer, no text/empty, invalid/nonsense — plus null (keep)
- **Pre-filtered passthrough**: responses already flagged by step 1 skip LLM grading
- **Pre-filtering of empties**: catches empty/None/NaN values that step 1 may have missed
- **Full rate-limiting stack**: ConcurrencyGate, TokenBucket, AsyncLimiter, CircuitBreaker, PID controller, signal-based ramp
- **Persistent cold-start stats**: saves P50/P95 latency and avg tokens to `model_perf_stats.json` for next run

## Pipeline Overview

```
Input: List[PreprocessedModel] (from step 1 cache, prefix 002)

Stage 1 — Pre-Filter:
  a. Responses with quality_filter_code set → passthrough
  b. Empty/None/NaN responses → mark 99999998, passthrough

Stage 2 — LLM Quality Grading (Grader):
  Per response (async, rate-limited):
    Response text + survey question + language
      → [GRADER_INSTRUCTIONS prompt] 3-step evaluation
      → category (1-5 or null)
      → map to quality_filter_code via CATEGORY_TO_CODE

Stage 3 — Output Assembly:
  Merge graded results with pre-filtered passthrough
  → List[QualityFilteredModel] → cache

Output: List[QualityFilteredModel] (step="quality_filter", prefix 003)
```

## Prompt Variants & Response Models

In `prompts_qualityFilter.py`:

| Variant | Prompt | Response Model | Model tier |
|---------|--------|---------------|------------|
| Pattern A | `GRADER_INSTRUCTIONS_NANO` | Raw text: `<scratchpad>` + `<category>` tags | Nano |
| Pattern B | `GRADER_INSTRUCTIONS_STRUCTURED` | `QualityFilterStructuredResponse` (via instructor) | Mini/default |

### Shared Prompt Structure

Both variants share the same building blocks:
1. **Context block** (`_CONTEXT_BLOCK`): language, survey question, response text
2. **Categories block** (`_CATEGORIES_BLOCK`): 5 noise categories with examples
3. **Decision rule** (`_DECISION_RULE`): 3-step evaluation (interpret → assess → categorize)

### Category → Code Mapping

| Category | Meaning | quality_filter_code |
|----------|---------|-------------------|
| 1 | Don't know / Not knowing | 99999997 |
| 2 | Not applicable / Not having | 99999997 |
| 3 | Absence of answer / Not addressing | 99999998 |
| 4 | No text / Empty | 99999998 |
| 5 | Invalid text / Nonsense | 99999999 |
| null | Keep (meaningful response) | None |

Special codes (not from LLM):
- `-1` — permanent processing failure (after retry)
- `0` — unprocessed fallback (shouldn't occur)
- `99999998` — pre-filtered empty/None (set by `grade()` before LLM)

### Response Models

**Pattern A (nano)**: raw text parsed via regex — extract `<category>` tag, map to code via `parse_quality_code()`. Parse failure → `None` (conservative: keep response).

**Pattern B (mini/default)**:
```
QualityFilterStructuredResponse:
  scratchpad: str      (evaluation reasoning)
  category: Optional[Literal[1, 2, 3, 4, 5]]   (null = keep)
```

## Concurrency & Rate Limiting

### Rate Limiting Stack

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | ConcurrencyGate | Dynamic concurrent request limit (supports runtime changes) |
| 2 | TokenBucket | TPM enforcement with wait-and-acquire + reconciliation |
| 3 | AsyncLimiter | RPM pacing (reinstalled by PID when arrival rate adjusts) |
| Safety | LatencyTracker | EMA-based adaptive timeout (computed after semaphore) |
| 4 | CircuitBreaker | Timeout rate monitoring → concurrency reduction on sustained pressure |
| 5 | PIDController | Asymmetric PID (kp_up=0.4, kp_down=0.2) tunes arrival rate from real-time TPM |
| 6 | Signal ramp | 4-signal evaluation (queue, RPM%, TPM%, latency trend) adjusts concurrency |

### Error Handling

- Per-response retry with tenacity (5 attempts, exponential jitter backoff)
- Failed tasks tracked in `failure_log`
- Retry pass at reduced concurrency (Phase 6)
- Permanent failures get fallback code `-1`

## Configuration

**`QualityFilterConfig`** (dataclass in `config_qualityFilter.py`):

| Field | Default | Purpose |
|-------|---------|---------|
| `model` | `get_step_model("quality_filter")` | LLM model for grading |
| `temperature` | 0.0 | Deterministic grading |
| `max_tokens` | 4000 | Token budget per call |
| `retries` / `instructor_retries` | 3 / 3 | Retry attempts |
| `max_concurrent_requests` | 5 | Initial concurrency cap |
| `minimum_timeout_seconds` | 15.0 | Timeout floor |
| `maximum_timeout_seconds` | 60.0 | Timeout ceiling |

**Processing constants** (in `config_qualityFilter.py`):

| Constant | Value | Purpose |
|----------|-------|---------|
| `INPUT_HISTORY_MAXLEN` | 3 | EMA input token window |
| `OUTPUT_HISTORY_MAXLEN` | 5 | EMA output token window |
| `DEFAULT_TIMEOUT_SECONDS` | 180.0 | Cold-start timeout |
| `COLD_START_CAP` | 50 | Max initial concurrency |
| `WARM_UP_WINDOW_SECONDS` | 10.0 | Time before calibration |
| `WARM_UP_MIN_COMPLETIONS` | 3 | Min completions for calibration |
| `RAMP_INCREASE_FACTOR` | 1.10 | +10% on all-green signals |
| `RAMP_DECREASE_FACTOR` | 0.80 | -20% on any-red signal |
| `SIGNAL_GREEN_THRESHOLD` | 0.80 | < 80% utilization = green |
| `SIGNAL_YELLOW_THRESHOLD` | 0.90 | 80-90% = yellow, > 90% = red |
| `ADJUSTMENT_INTERVAL` | 15 | Seconds between PID adjustments |

**Shared dataclasses** (from `step_3_ideaExtractor/config_ideaExtractor.py`):

| Dataclass | Key values |
|---|---|
| `RampUpConfig` | start=50%, target=90%, min_initial=5 |
| `CircuitBreakerConfig` | window=30s, trip=5%, reduction=15%, cooldown=60s |
| `WarmUpConfig` | sample_min=15, sample_max=30 |
| `PIDControllerConfig` | kp_up=0.4, kp_down=0.2, ki=0.05, kd=0.1, max=15% |
| `TPMTrackingConfig` | window=60s, target_utilization=80% |

**Hardcoded in `qualityFilter.py`**:

| Value | Location | Purpose |
|-------|----------|---------|
| `TIMEOUT_FLOOR_SECONDS = 60.0` | line 169 | Per-request timeout floor |
| Post-warm-up jump cap = 100 | `_calibrate_from_warm_up` | Max concurrency after calibration |
| Retry worker fraction = 10% (min 5) | `process_all_tasks_async` | Retry pass concurrency |
