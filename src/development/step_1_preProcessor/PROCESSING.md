# Step 1: Pre-Processor (Spell Checker) — Processing Reference

Reference for debugging and improvement. Source of truth: the code in `spellChecker_exp.py`.

Last verified against code: 2026-03-28

---

# I. Contract

Principles we agreed on for how prompt processing works in this step. Updating code = updating this doc. Commit both together.

### 1. Goal

Optimize the processing of prompts across providers (OpenAI, Azure), models (nano, mini, default), and constraints (RPM, TPM, server-side queue, latency), where the binding bottleneck varies by deployment.

### 2. Three-layer rate-limiting stack

Every LLM API call passes through: Semaphore -> TokenBucket -> AsyncLimiter -> Timeout. Uses `asyncio.Semaphore` (not ConcurrencyGate) with static concurrency after bootstrap.

**Note:** This step does not implement the full 4-layer stack (no ConcurrencyGate, no circuit breaker, no PID, no ramp). This is a known gap — see section E.

### 3. Bootstrap-based cold start (Option A)

Uses 3 serial probe calls to measure avg latency and avg tokens, then computes Little's Law concurrency for the entire run. No warm-up ramp — static concurrency from bootstrap. No persistent stats across runs.

### 4. Generous timeouts, retry for true failures

Timeouts are adaptive (EMA-based P95 with margin). True failures get one retry pass at 10% concurrency.

### 5. Model-tier-aware output handling

Uses instructor + Pydantic validation (Pattern B) for the main processing. Bootstrap probes use raw API calls for speed.

### 6. Hybrid processing: Hunspell + LLM

Step 1 is unique — it combines local Hunspell spell-checking (no rate limiting) with LLM-based correction (rate-limited). Only the LLM phase uses the rate-limiting stack.

### 7. Documentation tracks implementation

This PROCESSING.md reflects what the code does now, not what we plan to do. Known gaps go in section E.

### 8. Development code stays clean

This is development, not production. No legacy references, no backward compatibility shims, no dead or redundant code.

---

# II. Processing

## A. Overview

Step 1 spell-checks survey responses using a three-phase approach: identify misspelled words (Hunspell), generate suggestions (Hunspell), then correct via LLM.

- **Input:** `List[ResponseModel]` (raw survey responses)
- **Output:** `List[PreprocessedModel]` with corrected text
- **Model:** Configured via config — typically mini/default
- **Provider:** OpenAI or Azure
- **Dispatch:** `asyncio.Queue` + worker pool (LLM phase only)

---

## B. Processing strategy

Three phases, only the last is LLM-rate-limited:

### Phase 1: OOV Word Identification (local, no rate limiting)

SpaCy NLP pipeline processes sentences in batches. Words filtered by `is_alpha=True`, `len > 2`. HunspellPool checks all words in parallel batches (10K words per batch). Output: `oov_words` list + `word_to_responses` mapping.

### Phase 2: Hunspell Suggestions (local, no rate limiting)

Batched suggestion generation via `find_best_suggestions_batch_async()`. Produces `best_suggestions_dict: Dict[str, List[suggestions]]`.

### Phase 3: AI Corrections (LLM, rate-limited)

Three sub-phases:

**3a: Bootstrap (3 probe calls)**
- Fetch rate limits from API response headers
- Run 3 serial probe calls to measure avg latency and avg tokens
- Compute Little's Law → set static concurrency, workers, rate limiters

**3b: Main batch processing**
- Queue + workers dispatch at bootstrap-computed concurrency
- Progress monitoring every 5s
- Tracks `llm_calls_successful`, `llm_calls_failed`, `failed_task_ids`

**3c: Retry pass**
- Reduced concurrency (10% of workers, min 5)
- One retry for tasks that failed in main pass
- Merges successful results back

---

## C. Rate-limiting machinery

### Request flow (per task)

```
# Timeout computed BEFORE semaphore (diverges from steps 2-6)
timeout = self.latency_tracker.get_timeout(tokens)

async with self.semaphore:                    # Layer 1: asyncio.Semaphore
    await self.tpm_bucket.wait_and_acquire()  # Layer 2: TokenBucket
    async with self.rate_limiter:             # Layer 3: AsyncLimiter (RPM)
        response = await asyncio.wait_for(api_call, timeout=timeout)
```

### Layer 1: asyncio.Semaphore

Standard `asyncio.Semaphore` — no runtime limit changes. Static concurrency set once during bootstrap and used for the entire run.

### Layer 2: TokenBucket

Same implementation as steps 2-6. Pre-acquires estimated tokens, reconciles after each call.

### Layer 3: AsyncLimiter (RPM)

`aiolimiter.AsyncLimiter` for request spacing. Rate computed once during bootstrap.

### Safety net: Timeout

`LatencyTracker` with EMA-based adaptive timeout:
- Cold start: generous (180s)
- After bootstrap: `max(floor, min(P95 * margin, ceiling))`
- Floor: 15s (configurable), ceiling: 120s (configurable)

### Token estimation

Adaptive: first prompt actual + 15%, then average of first 3 inputs. Output estimated at 15% of input, then average of first 5 actuals. Same pattern as step 2.

---

## D. Divergent paths

### OpenAI vs Azure

Provider abstraction handled by `llm.py`. Bootstrap probe is provider-aware (same pattern as step 2).

### Bootstrap probes vs main processing

Bootstrap uses raw API calls (no instructor) for speed. Main processing uses instructor + Pydantic (Pattern B).

---

## E. Known issues and divergences

### 1. No ConcurrencyGate — uses asyncio.Semaphore

Steps 2-6 use `ConcurrencyGate` which supports runtime limit changes. Step 1 uses standard `asyncio.Semaphore` with static concurrency. No dynamic scaling during the run.

### 2. No circuit breaker

No timeout rate monitoring. If API degrades, concurrency stays at bootstrap level.

### 3. No PID controller

No continuous arrival rate adjustment. Rate is static after bootstrap.

### 4. No warm-up ramp

Uses bootstrap (Option A: 3 probe calls) instead of warm-up with conservative ramp (Option B). Static concurrency for the entire run — no completion-based scaling.

### 5. No persistent cold-start stats

Does not save/load stats from `model_perf_stats.json`. Every run re-bootstraps from scratch.

### 6. Timeout computed before semaphore

Timeout is calculated before acquiring the semaphore, meaning all workers compute timeout at T+0 with the same (potentially stale) cold-start value. Steps 2-6 compute timeout after semaphore acquisition.

---

## F. Configuration reference

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| Bootstrap probes | 3 | Hardcoded in `bootstrap_measure_async` |
| Default timeout | 180s | `LatencyTracker` cold start |
| Timeout floor | 15s | `ProcessingConfig.adaptive_timeout_min_seconds` |
| Timeout ceiling | 120s | `ProcessingConfig.adaptive_timeout_max_seconds` |
| Timeout margin | 1.5x | `ProcessingConfig.adaptive_timeout_margin` |
| Min workers | 50 | Fallback in worker calculation |
| Max workers | 200 | Fallback in worker calculation |
| Retry worker fraction | 10% (min 5) | Hardcoded in retry pass |

### From `config_preProcessor.py`

| Parameter | Value | Purpose |
|---|---|---|
| `MAX_HUNSPELL_PROCESSES` | 20 | Parallel Hunspell processes |
| `SUGGESTION_BATCH_SIZE` | 50 | Words per suggestion batch |
| `OUTPUT_TOKEN_RATIO` | 0.15 | For token estimation |
