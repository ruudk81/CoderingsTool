# Step 1 PreProcessor — Architecture & Data Flow

## Design Intent

Text normalization and spell correction of raw survey responses. Step 1 transforms raw text into clean, consistently formatted responses via a 3-stage linear pipeline:

- **TextNormalizer**: deterministic cleanup (whitespace, encoding, special characters, NA handling)
- **SpellChecker**: Hunspell-based OOV detection + LLM batch correction
- **TextFinalizer**: post-correction cleanup (repeated characters, trailing punctuation, final formatting)

Key design choices:
- **Hunspell + LLM hybrid**: Hunspell identifies misspelled words and generates candidate corrections; LLM selects the best correction using survey context
- **Batch correction**: multiple responses corrected per LLM call for efficiency
- **Persistent Hunspell pool**: reusable subprocess pool avoids startup overhead
- **Streaming OOV detection**: Hunspell word checking runs through a queue-based streaming pipeline
- **Non-string passthrough**: numeric, NaN, and code responses (99999997/98/99) are passed through with appropriate quality_filter flags

## Pipeline Overview

```
Input: List[ResponseModel] (from step 0 cache, prefix 001)

Stage 0 — Type Split:
  Separate string responses from non-string (numeric, NaN, codes)
  Non-string → passthrough with quality_filter flags

Stage 1 — Text Normalization (TextNormalizer):
  For each string response:
    Strip whitespace, normalize encoding, handle special characters
    Mark empty/NA responses
  → List[PreprocessedModel]

Stage 2 — Spell Checking (SpellChecker):
  2a. OOV Detection (Hunspell):
    Pool of persistent Hunspell processes
    → identify misspelled words per response

  2b. Suggestion Generation (Hunspell):
    Batch misspelled words → candidate corrections
    Optional: SpaCy pre-validation of suggestions

  2c. LLM Batch Correction:
    Group responses with OOV words into batches
    → [SPELLCHECK_INSTRUCTIONS prompt] context-aware correction
    → apply corrections back to responses
  → List[PreprocessedModel]

Stage 3 — Text Finalization (TextFinalizer):
  Strip repeated characters (5+), fix punctuation, final trim
  → List[PreprocessedModel]

Stage 4 — Output Assembly:
  Merge finalized strings + non-string passthrough
  Set quality_filter flags for NaN/code values
  → List[PreprocessedModel] → cache

Output: List[PreprocessedModel] (step="preprocessed", prefix 002)
```

## Prompt Builder & Response Model

In `prompts_exp.py`:

| Stage | Prompt | Response Model | Notes |
|-------|--------|---------------|-------|
| 2c | `SPELLCHECK_INSTRUCTIONS` | `LLMCorrectionResponse` (List[CorrectionItem]) | Batched: multiple responses per call |

### Spell Check Prompt

**Input per batch**: survey question (`var_lab`), language, correction tasks (each with: sentence with `<oov_word>` placeholders, misspelled words, suggested corrections).

**Response model**:
```
CorrectionItem:
  respondent_id (Any), corrected_response (str)

LLMCorrectionResponse:
  corrections: List[CorrectionItem]
```

**Rules**: use context to select best correction; may split words if grammatically valid; use `[NO RESPONSE]` if no suitable correction exists.

## Data Flow

### From Step 0 (input)

- **`List[ResponseModel]`**: raw responses with respondent_id, response (str/float/int/None), response_type

### Step 1 Output (cached)

- **Growing model** (`List[PreprocessedModel]`, step="preprocessed", prefix 002):
  - Inherits: respondent_id, response, response_type
  - Adds: `quality_filter` (bool), `quality_filter_code` (int or None)

### To Step 2 (output contract)

Step 2 loads `List[PreprocessedModel]`. It needs:
- `response` — cleaned text for quality assessment
- `quality_filter` / `quality_filter_code` — pre-filtered items (NaN, codes) already flagged

## Key Data Structures

**`ResponseModel`** (Pydantic, input):
```
respondent_id (Any), response (str|float|int|None), response_type (str|None)
```

**`PreprocessedModel`** (Pydantic, extends ResponseModel):
```
quality_filter (bool|None), quality_filter_code (int|None)
```

Quality filter codes (set during assembly):
- `None` — meaningful string response
- `99999997` — "don't know" code (passed through from raw data)
- `99999998` — NaN / no response / empty
- `99999999` — gibberish code (passed through from raw data)

## Concurrency & Rate Limiting

### Hunspell Processing

- **HunspellPool**: pool of `hunspell_pool_size` (default 20) persistent subprocesses
- **Streaming OOV detection**: queue-based pipeline with `hunspell_batch_size` (1000) words per batch
- **Suggestion batching**: `SUGGESTION_BATCH_SIZE` (50) words per Hunspell query, `MAX_CONCURRENT_SUGGESTION_BATCHES` (6) concurrent

### LLM Spell Correction

- **TokenBucket**: TPM enforcement with regeneration based on elapsed time
- **AsyncLimiter**: RPM pacing
- **LatencyTracker**: EMA-based adaptive timeout
- **Batch size**: `max_batch_size` (5) responses per LLM call

## Configuration

**`SpellCheckConfig`** (dataclass in `config_steps/config_preProcessor.py`):

| Group | Fields | Defaults |
|-------|--------|----------|
| Model | `model` | `get_step_model("spell_check")` |
| LLM params | `temperature`, `max_tokens`, `retries`, `seed` | 0.0, 4000, 3, 42 |
| Batching | `batch_size`, `max_batch_size`, `completion_reserve` | 20, 5, 1000 |
| Timeout | `minimum_timeout_seconds`, `maximum_timeout_seconds` | 15.0, 60.0 |
| Hunspell | `hunspell_pool_size`, `hunspell_batch_size`, `hunspell_concurrent_sessions` | 20, 1000, 20 |
| Suggestions | `max_concurrent_suggestion_chunks`, `max_words_per_chunk` | 20, 1200 |
| Performance | `max_words_to_check`, `max_unique_oov_words`, `enable_early_termination` | 100000, 5000, True |
| Rate limiting | `rate_limit_safety_factor`, `rate_limit_utilization` | 0.95, 0.98 |

**Processing constants** (module-level in `config_steps/config_preProcessor.py`):

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_HUNSPELL_PROCESSES` | 20 | Subprocess pool cap |
| `MAX_SAFE_BATCH_SIZE` | 1000 | Hunspell word check batch limit |
| `SUGGESTION_BATCH_SIZE` | 50 | Words per suggestion query |
| `OUTPUT_TOKEN_RATIO` | 0.15 | Estimated output/input ratio |

## Files

- **`run_experiment.py`** — orchestrator: loads step 0 cache, runs 3-stage pipeline, saves results
- **`spellChecker_exp.py`** — SpellChecker: Hunspell OOV detection + LLM batch correction with rate limiting
- **`textNormalizer_exp.py`** — TextNormalizer: deterministic text cleanup
- **`textFinalizer_exp.py`** — TextFinalizer: post-correction formatting
- **`prompts_exp.py`** — SPELLCHECK_INSTRUCTIONS prompt + CorrectionItem/LLMCorrectionResponse models
- **`debug_samples.py`** — inspect cached preprocessing results (before/after comparison)
