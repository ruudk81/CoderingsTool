# Model Pricing Reference

Pricing per 1M tokens. Source: OpenAI pricing page, verified against `src/utils/llm.py:MODEL_PRICING`.

Last updated: 2026-03-23

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

`src/utils/llm.py` → `MODEL_PRICING` dict (line ~387) + `TokenTracker.record()` method.

The `TokenTracker` is a global singleton (`token_tracker`) that records every LLM call automatically via `_extract_and_track_usage()`. Access summary via `token_tracker.get_summary()`.
