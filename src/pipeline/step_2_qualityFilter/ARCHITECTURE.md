# Step 2 Quality Filter — Architecture & Data Flow

## Design Intent

LLM-based quality assessment of preprocessed survey responses. Step 2 evaluates each response individually and flags noise (don't-know, empty, gibberish, off-topic) while preserving all meaningful responses for downstream analysis.

Key design choices:
- **Per-response LLM grading**: each response evaluated independently with full survey context
- **Dual prompt strategy**: nano models use raw text + XML tags (Pattern A); mini/default models use instructor + Pydantic (Pattern B)
- **5-category noise taxonomy**: don't know, not applicable, absence of answer, no text/empty, invalid/nonsense — plus null (keep)
- **Pre-filtered passthrough**: responses already flagged by step 1 (quality_filter=True) skip LLM grading
- **Adaptive rate limiting**: PID-controlled arrival rate with circuit breaker for sustained timeout pressure

## Pipeline Overview

```
Input: List[PreprocessedModel] (from step 1 cache, prefix 002)

Stage 1 — Filter Pre-Flagged:
  Responses with quality_filter=True → passthrough (already filtered by step 1)
  Responses with quality_filter=False/None → proceed to grading

Stage 2 — LLM Quality Grading (Grader):
  Per response (async, rate-limited):
    Response text + survey question + language
      → [GRADER_INSTRUCTIONS prompt] 3-step evaluation
      → category (1-5 or null)
      → map to quality_filter_code

Stage 3 — Output Assembly:
  Merge graded results with pre-filtered passthrough
  → List[QualityFilteredModel] → cache

Output: List[QualityFilteredModel] (step="quality_filter", prefix 003)
```

## Prompt Variants & Response Models

In `prompts_exp.py`:

| Variant | Prompt | Response Model | Model tier |
|---------|--------|---------------|------------|
| Pattern A | `GRADER_INSTRUCTIONS_NANO` | Raw text: `<scratchpad>` + `<category>` tags | Nano models |
| Pattern B | `GRADER_INSTRUCTIONS_STRUCTURED` | `QualityFilterStructuredResponse` (via instructor) | Mini/default models |

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

### Response Models

**Pattern A (nano)**: raw text parsed via regex — extract `<category>` tag, map to code via `parse_quality_code()`.

**Pattern B (mini/default)**:
```
QualityFilterStructuredResponse:
  scratchpad: str      (evaluation reasoning)
  category: Optional[Literal[1, 2, 3, 4, 5]]   (null = keep)
```

## Data Flow

### From Step 1 (input)

- **`List[PreprocessedModel]`**: cleaned responses with quality_filter flags from step 1

### Step 2 Output (cached)

- **Growing model** (`List[QualityFilteredModel]`, step="quality_filter", prefix 003):
  - Same fields as PreprocessedModel (QualityFilteredModel extends it without adding fields)
  - `quality_filter_code` updated with LLM grading results

### To Step 3 (output contract)

Step 3 loads `List[QualityFilteredModel]` and filters to meaningful responses (`quality_filter=False`) for idea extraction.

## Key Data Structures

**`PreprocessedModel`** (Pydantic, input):
```
respondent_id, response, response_type, quality_filter (bool), quality_filter_code (int|None)
```

**`QualityFilteredModel`** (Pydantic, extends PreprocessedModel):
```
(same fields — no new fields added; quality_filter_code updated by grading)
```

## Concurrency & Rate Limiting

### Rate Limiting Stack

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | ConcurrencyGate | Limits active concurrent requests with dynamic cap |
| 2 | TokenBucket | TPM enforcement with wait-and-acquire |
| 3 | AsyncLimiter | RPM pacing |
| 4 | LatencyTracker | EMA-based adaptive timeout (tier-aware: cold-start vs warm) |
| 5 | CircuitBreaker | Monitors timeout rate in sliding window; reduces concurrency on sustained pressure |
| 6 | PIDController | Asymmetric PID (kp_up=0.4, kp_down=0.2) adjusts arrival rate from real-time TPM utilization |

### Error Handling

- Per-response retry with tenacity (configurable retries)
- Failed tasks tracked in `failed_task_ids`
- Timeout responses collected and retried

## Configuration

**`QualityFilterConfig`** (dataclass in `config_steps/config_qualityFilter.py`):

| Field | Default | Purpose |
|-------|---------|---------|
| `model` | `get_step_model("quality_filter")` | LLM model for grading |
| `batch_size` | 20 | (unused in per-response mode) |
| `temperature` | 0.0 | Deterministic grading |
| `max_tokens` | 4000 | Token budget per call |
| `retries` / `instructor_retries` | 3 / 3 | Retry attempts |
| `max_concurrent_requests` | 5 | Initial concurrency cap |
| `minimum_timeout_seconds` | 15.0 | Timeout floor |
| `maximum_timeout_seconds` | 60.0 | Timeout ceiling |

**Processing constants** (module-level in `config_steps/config_qualityFilter.py`):

| Constant | Value | Purpose |
|----------|-------|---------|
| `INPUT_HISTORY_MAXLEN` | 3 | EMA input token window |
| `OUTPUT_HISTORY_MAXLEN` | 5 | EMA output token window |
| `DEFAULT_TIMEOUT_SECONDS` | 30.0 | Cold-start timeout |
| `PROGRESS_REPORT_INTERVAL` | 5 | Seconds between reports |
| `ADJUSTMENT_INTERVAL` | 15 | Seconds between PID adjustments |

## Files

- **`run_experiment.py`** — orchestrator: loads step 1 cache, runs Grader, saves results
- **`qualityFilter_exp.py`** — Grader: per-response LLM grading with adaptive rate limiting
- **`prompts_exp.py`** — dual prompt variants (nano/structured) + shared building blocks + response models
- **`config_exp.py`** — experimental config overrides (advanced rate limiting settings)
- **`debug_quality_prompts.py`** — inspect captured quality filter prompts
- **`view_quality_results.py`** — view filtered/meaningful response distribution and examples
