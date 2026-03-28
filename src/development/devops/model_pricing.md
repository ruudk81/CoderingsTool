# Model Pricing Reference

Pricing per 1M tokens. Source: OpenAI pricing page, verified against `src/config.py:MODEL_PRICING`.

Last updated: 2026-03-24

## GPT-4.1 Family (Chat Models)

| Model | Input ($/1M) | Output ($/1M) | Notes |
|-------|-------------|---------------|-------|
| gpt-4.1 | $2.00 | $8.00 | Full model — consolidation, code generation |
| gpt-4.1-mini | $0.40 | $1.60 | Default for most tasks — discovery, filtering |
| gpt-4.1-nano | $0.10 | $0.40 | Classification/assignment — high volume, simple tasks |

## GPT-5 Family (Reasoning Models)

| Model | Input ($/1M) | Output ($/1M) | Notes |
|-------|-------------|---------------|-------|
| gpt-5 | $1.25 | $10.00 | Full reasoning model |
| gpt-5.1 | $1.25 | $10.00 | Full reasoning model (v5.1) |
| gpt-5.2 | $1.25 | $10.00 | Full reasoning model (v5.2) |
| gpt-5-mini | $0.25 | $2.00 | Default for reasoning tasks |
| gpt-5-nano | $0.05 | $0.40 | Lightweight reasoning — classification |
| gpt-5-chat-latest | $1.25 | $10.00 | Chat variant of gpt-5 (no reasoning params) |

## Embedding Models

| Model | Input ($/1M) | Output | Notes |
|-------|-------------|--------|-------|
| text-embedding-3-large | $0.13 | N/A | 3072 dimensions |
| text-embedding-3-small | $0.02 | N/A | 1536 dimensions |

## Legacy Models

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|---------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |

## Cost Calculation

```
cost = (input_tokens / 1,000,000) * input_price + (output_tokens / 1,000,000) * output_price
```

### Example: Step 2 Quality Filter with gpt-5-mini

```
Input:  2,332,206 tokens * $0.25/1M = $0.583
Output:   762,514 tokens * $2.00/1M = $1.525
Total:                                 $2.108
```

## Where Pricing Lives in Code

`src/config.py` → `MODEL_PRICING` dict + `DEFAULT_PRICING` fallback. Consumed by `TokenTracker.record()` in `src/utils/llm.py`.

The `TokenTracker` is a global singleton (`token_tracker`) that records every LLM call automatically via `_extract_and_track_usage()`. Access summary via `token_tracker.get_summary()`.

## How Token Counts Are Obtained

Token counts come directly from the OpenAI API response — they are **not** calculated locally. Every API response includes a `usage` object with server-side token counts. This means the counts include everything: the prompt, the schema injection that `instructor` adds, and the structured output. What we track matches exactly what we're billed for.

The flow in `src/utils/llm.py`:

1. OpenAI returns token counts in every API response's `usage` object
2. `_extract_and_track_usage()` (line ~540) reads them from the response:
   - **Azure** (Chat Completions API): `usage.prompt_tokens` / `usage.completion_tokens`
   - **OpenAI** (Responses API): `usage.input_tokens` / `usage.output_tokens`
3. Those raw counts are passed to `token_tracker.record()` which multiplies by the `MODEL_PRICING` rates

## Token Tracker Usage in Experiment Runners

Cost tracking is orchestrated from each step's `run_experiment.py`. The pattern:

```python
from utils.llm import token_tracker

# Before the run — reset counters so we only measure this step
token_tracker.reset()

# ... run the step ...

# After the run — print the LLM usage summary
if token_tracker.call_count > 0:
    print(token_tracker.get_summary())
```

- **`token_tracker.reset()`** — zeroes all counters (tokens, cost, call count) so each experiment run starts clean and only reports its own usage.
- **`token_tracker.get_summary()`** — returns a formatted summary string with provider, total API calls, token breakdown (input/output), and total cost in USD.

The summary is printed inside the `TeeOutput` capture block, so it appears both on the console and in the saved verbose log file (`exports/verbose_logs/`).

### Steps with token tracking in `run_experiment.py`

| Step | Runner | Token tracking |
|------|--------|---------------|
| Step 1 (preProcessor) | `src/development/step_1_preProcessor/run_experiment.py` | Yes |
| Step 2 (qualityFilter) | `src/development/step_2_qualityFilter/run_experiment.py` | Yes |
| Step 3 (ideaExtractor) | `src/development/step_3_ideaExtractor/run_experiment.py` | Yes |
| Step 4 (classifier) | `src/development/step_4_classifier/run_experiment.py` | Yes |
| Step 5 (codeGenerator) | `src/development/step_5_codeGenerator/run_experiment.py` | Yes |
| Step 6 (codeAssigner) | `src/development/step_6_codeAssigner/run_experiment.py` | Yes |
