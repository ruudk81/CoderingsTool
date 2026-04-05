# Step 2 — Quality Filter

## Purpose
Per-response LLM grading to identify and flag noise (don't-know, empty, gibberish, off-topic) while preserving meaningful responses. Responses already flagged by step 1 (non-strings) skip LLM evaluation. Step 2 also pre-filters empty/None values that step 1 may have missed.

## Key Files
- `run_qualityFilter.py` — step runner / orchestrator
- `qualityFilter.py` — `Grader` class: per-response async LLM grading with full rate-limiting stack
- `prompts_qualityFilter.py` — dual prompt variants + `QualityFilterStructuredResponse` model
- `config_qualityFilter.py` — `QualityFilterConfig` + processing constants + strategy constants

## Input / Output Contract
- **Input**: `List[PreprocessedModel]` from cache key `preprocessed/{variable_key}`
- **Output**: `List[QualityFilteredModel]` cached at `quality_filter/{variable_key}`
  - Updates `quality_filter` and `quality_filter_code` based on LLM grading
  - Codes: `99999997` (don't know / not applicable), `99999998` (absence / empty / NaN), `99999999` (gibberish), `None` (keep), `-1` (processing error), `0` (unprocessed fallback)

## LLM Usage
- **Model**: `get_step_model("quality_filter")` (nano tier)
- **Prompt file**: `prompts_qualityFilter.py`
- **Response model**: `QualityFilterStructuredResponse` (scratchpad + category) for Pattern B; raw XML tags for Pattern A
- **Pattern**: per-response evaluation (one LLM call per response)
- **Dual prompts**: nano uses Pattern A (raw text + `<category>` tags), mini/default uses Pattern B (instructor + Pydantic)

## Shared Utils
- `utils/llm.py` — `create_client()`, `llm_create_async()`, `extract_rate_limits_from_response()`
- `utils/cacheManager.py` — cache load/save
- `utils/modelPerfStats.py` — persistent cold-start stats (`load_stats`, `save_stats`)
- `utils/costTracker.py` — token & cost tracking
- `utils/verboseReporter.py` — step logging
- `utils/promptPrinter.py` — prompt capture for debugging

## Gotchas
- **Pre-filtered passthrough**: responses with `quality_filter_code is not None` from step 1 skip LLM entirely.
- **Pre-filtering of empty values**: `grade()` catches empty/None/NaN responses that step 1 missed, marking them `99999998` before LLM dispatch.
- **Dual prompt strategy**: model tier determines the path at init (`_is_nano` flag). Nano uses raw `AsyncOpenAI` + regex parsing; mini/default uses instructor client.
- **Nano imports openai directly**: Pattern A creates a raw `AsyncOpenAI` client (line 564), bypassing `llm.py`. This is intentional — instructor adds overhead for nano's simple tag parsing.
- **Fallback codes**: `-1` = processing error (permanent failure after retry), `0` = unprocessed (shouldn't happen in practice).
- **Shared config dataclasses**: imports `RampUpConfig`, `CircuitBreakerConfig`, `PIDControllerConfig`, etc. from `step_3_ideaExtractor.config_ideaExtractor`. These are shared infrastructure, not step-3-specific.
- **`force_recalc=True` by default**: cache-hit path never exercised in development.

## Processing Phases
1. **Fetch API rate limits** — minimal probe call to read TPM/RPM from headers
2. **Initialize rate limiters** — 4-layer stack: ConcurrencyGate, TokenBucket, AsyncLimiter, CircuitBreaker + PID controller
3. **Warm-up calibration** — after 10s + 3 completions, recalibrate with measured latency/tokens, jump concurrency
4. **Main processing loop** — queue + workers with signal-based ramp, PID adjustments, circuit breaker checks
5. **Persist stats** — save P50/P95 latency and avg tokens for next cold start
6. **Retry pass** — 10% concurrency for failed tasks, one retry, permanent fallback on second failure

## Dev Docs
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [CACHE_LOGIC.md](CACHE_LOGIC.md) — caching contracts
- [PROCESSING.md](PROCESSING.md) — processing flow
