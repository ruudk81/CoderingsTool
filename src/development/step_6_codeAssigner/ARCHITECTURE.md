# Step 6 Code Assigner — Architecture & Data Flow

## Design Intent

Assign exactly one MECE code to each idea. Step 6 (P10) takes the codebook from step 5 and ideas from step 4 (or step 3 fallback), then assigns codes via LLM with optional embedding pre-filtering.

Key design choices:
- **Embedding pre-filter**: cosine similarity narrows the codebook to top-N candidates per idea, reducing prompt size and improving accuracy
- **Per-task ID scoping**: when pre-filter is active, each idea gets its own C1..CN → code_name mapping to avoid global ID ambiguity
- **4-layer rate limiting**: same completion-based ramp as steps 4/5, with worker pool and queue-based task distribution
- **Tier-aware validation**: strict field validation for mini/default models, lenient coercion for nano models
- **Fallback retry pass**: timeouts collected separately and retried with generous timeout and reduced concurrency

## Pipeline Overview

```
Input: List[TaxonomyClassifiedModel] (from step 4, prefix 005; fallback: step 3, prefix 004)
       + CodingResultsCache (from step 5, prefix 006)
       + ExtractionMetadata (from step 3, prefix 004)

Phase 1: Setup
  Create async client, build facet lookup, group ideas by partition

Phase 2: Fetch Rate Limits
  Probe call to API → extract TPM/RPM from response headers

Phase 3: Token Estimation
  Sample ~20 ideas → tiktoken estimate → avg tokens per request

Phase 3b: Embedding Pre-Filtering (optional)
  Embed all ideas + all codes → cosine similarity → top-N candidates per idea

Phase 4: Build Task List
  For each idea: determine candidate codes, build per-task ID map

Phase 5: Initialize Rate Limiting
  Little's Law → 4-layer stack (gate, token bucket, rate limiter, circuit breaker)

Phase 6: Queue + Workers
  Enqueue all tasks → N workers drain queue with rate-limited LLM calls
  Background: ramp-up, PID adjustment, circuit breaker, progress reporting

Phase 7: Retry Pass
  Collect timeouts → re-queue with reduced concurrency + generous timeout

Phase 8: Output Assembly
  Resolve code IDs → labels, build List[CodeAssignedModel]

Output: List[CodeAssignedModel] (growing model, step="taxonomy_codes", prefix 007)
```

## Input Contract

### From Step 4 (preferred) or Step 3 (fallback)

**Step 4 growing model** (`taxonomy_classified`, prefix 005):
- `List[TaxonomyClassifiedModel]` — per-idea with facet, attribute, partition_name already populated
- Loaded via `load_step4_enriched()` in run_experiment.py

**Step 3 fallback** (`extracted_ideas`, prefix 004):
- `List[IdeasExtractedModel]` — per-idea with domain, partial facet, abstraction ladder
- Loaded via `load_step3_ideas()` if step 4 cache missing

### From Step 5

**`CodingResultsCache`** (metadata, `mece_codes`, prefix 006):
- `partition_set: DomainSet` — domain definitions
- `partition_results: Dict[str, DomainResultModel]` — per-domain taxonomy results (for facet/attribute lookups)
- `raw_codes: List[Dict]` — ConsolidatedCode dicts, reconstructed via `ConsolidatedCode(**d)`

## Embedding Pre-Filtering

**Class**: `EmbeddingMatcher` in `embedding_matcher.py`

When `config.use_embedding_prefilter = True`:

1. **Build idea texts**: `"{domain} | {facet} | {interpretation} | {abstraction}"` per idea
2. **Build code texts**: `"{code_name} | {definition} | {top 5 indicators}"` per code
3. **Embed both**: batched embedding calls (semaphore-controlled, retry with exponential backoff)
4. **Compute top-N**: cosine similarity matrix → top `embedding_top_n` (default 5) code indices per idea
5. **Store**: `Dict[idea_id → List[code_indices]]` for per-task codebook scoping

This means each idea's assignment prompt only includes its top-N candidate codes, not the full codebook.

## Prompt Builder & Response Model

In `prompts_codeAssigner.py`:

**Builder**: `build_code_assignment_prompt(survey_question, language, dataset_context_section, codes, other_label, idea, facet_lookup)`

**Prompt structure**:
1. Survey context section
2. Codebook block: each code as `[C1]`, `[C2]`, ... with name, definition, diagnostic, top 5 indicators
3. Idea block: text, domain, facet, valence
4. Output guidance

**Response model**: `CodeAssignmentResponse`
```
assigned_code_id: str   ("C1", "C7", etc.)
confidence: float       (0.0 to 1.0)
rationale: str          (brief explanation)
```

**Tier-aware validation** (`configure_validation_mode(model)`):
- Nano models: lenient coercion (missing fields → defaults)
- Mini/default models: strict validation (missing fields → error)

## Code Assignment Flow

**`_process_task_with_retry()`** — per-idea, with tenacity retry (5 attempts, exponential backoff):

1. Build prompt with scoped candidate codes (or full codebook if no pre-filter)
2. 4-layer rate-limited LLM call:
   ```
   ConcurrencyGate → TokenBucket.wait_and_acquire → AsyncLimiter → asyncio.wait_for(timeout)
   ```
3. Record latency, tokens, completion/timeout to all tracking layers
4. Wrap result in `CodeAssignmentBatch`

**Workers** (`_worker()`): pull tasks from `asyncio.Queue`, process with retry, store results by index. Per-task ID resolution when pre-filter is active.

## Key Data Structures

**`CodeAssignment`** (Pydantic, internal wrapper):
```
idea_id, assigned_code_id, confidence, rationale
```

**`CodeAssignmentBatch`** (Pydantic):
```
assignments: List[CodeAssignment]
```

**`CodeAssignedSubmodel`** (extends TaxonomyClassifiedSubmodel):
```
assigned_code, assigned_attribute, confidence, rationale
(inherits: partition_name, facet, attribute, domain, idea, abstraction ladder)
```

**`CodeAssignedModel`** (extends TaxonomyClassifiedModel):
```
response_ideas: List[CodeAssignedSubmodel]
assignment_metadata: Dict[str, Any]
```

## Concurrency & Rate Limiting

### 4-Layer Stack

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | AsyncLimiter | RPM pacing (arrival rate from Little's Law) |
| 2 | TokenBucket | TPM enforcement (wait-and-acquire with reconciliation) |
| 3 | ConcurrencyGate + ConcurrencyRamp | Completion-based concurrency; 50% → 90% of Little's Law ceiling |
| 4 | LatencyTracker + ConcurrencyCircuitBreaker | Adaptive timeout (P95 × margin); backs off on sustained pressure |

### Supporting Components

- **PIDThroughputController**: every 20s, adjusts arrival rate based on real-time TPM utilization (asymmetric: aggressive speed-up, gentle slow-down, ±20% clamp)
- **RealTimeTPMTracker / RealTimeRPMTracker**: 60s sliding window for utilization measurement
- **TiktokenOffsetLearner**: learns tiktoken → API token offset for accurate estimation
- **modelPerfStats**: persistent P50/P95 latency + avg tokens for cold-start timeout floor calibration

### Warm-Up Calibration

After 15-30 completions: recalculates Little's Law from measured P10 latency and actual avg tokens, recalibrates ramp target, recomputes arrival rate, spawns extra workers if concurrency cap increased.

### Queue + Worker Model

```
asyncio.Queue with all tasks → N workers (= initial concurrency)
Main loop (100ms poll):
  - Circuit breaker check (1s)
  - Ramp-up feed (1s)
  - Progress report (2s)
  - Warm-up calibration (one-shot)
  - PID adjustment (20s)
```

### Retry Pass (Phase 7)

Timed-out tasks (soft failures) collected in Phase 6, re-queued in Phase 7 with:
- Reduced concurrency (`num_workers // 10`)
- Generous timeout
- Results merged back into assignment lookup

## ID Resolution

**Normalization** (`_normalize_id()`): strip whitespace, uppercase, prepend 'C' if numeric.

**Two strategies**:
1. **Per-task (priority, when pre-filter active)**: each task has scoped `task_id_to_label` map (C1-C5 → code names for that idea's candidates)
2. **Global (fallback, when no pre-filter)**: global `_id_to_label` map (C1-CN → code names for full codebook)

**Other category**: if `include_other_category = True`, adds `C(N+1)` → language-specific "Other" label.

## Output Assembly (`_build_output_models()`)

Iterates ALL original ideas (including unassigned):
1. Look up assignment from `assignment_lookup`
2. Resolve code ID to label via ID resolution
3. Build `CodeAssignedSubmodel` with all inherited fields + assignment fields
4. Wrap in `CodeAssignedModel` preserving response structure

Unassigned ideas marked with `"__UNASSIGNED__"` label.

## Configuration

**`AssignmentConfig`** (dataclass in `config_steps/config_codeAssigner.py`):

| Field | Default | Purpose |
|-------|---------|---------|
| `assignment_model` | `get_step_model("code_assignment")` | LLM model for assignment |
| `assignment_temperature` | 0.1 | Low for consistency |
| `assignment_max_tokens` | 4000 | Token budget per call |
| `include_other_category` | True | Add fallback "Other" code |
| `use_embedding_prefilter` | True | Enable cosine pre-filter |
| `embedding_top_n` | 5 | Top-N candidates per idea |
| `embedding_model` | `"text-embedding-3-large"` | Embedding model |
| `embedding_batch_size` | 100 | Batch size for embedding calls |
| `embedding_max_concurrent` | 5 | Max concurrent embedding batches |

## Data Flow Summary

```
Step 3 cache (prefix 004)
  └─ ExtractionMetadata → survey_question, language, specifiers
  └─ List[IdeasExtractedModel] → fallback ideas (if step 4 cache missing)

Step 4 cache (prefix 005)
  └─ List[TaxonomyClassifiedModel] → ideas with facet/attribute/partition (preferred)

Step 5 cache (prefix 006)
  └─ CodingResultsCache → partition_set, partition_results, raw_codes (codebook)

        ↓ CodeAssigner.assign_all()

Step 6 cache (prefix 007)
  └─ List[CodeAssignedModel] → per-idea code assignments (growing model)

        ↓ consumed by Step 7 (export)
```

## Files

- **`code_assignment.py`** — CodeAssigner: queue-based assignment with 4-layer rate limiting, embedding pre-filter, output assembly
- **`embedding_matcher.py`** — EmbeddingMatcher: batch embedding + cosine similarity top-N
- **`prompts_codeAssigner.py`** — prompt builder + response model, tier-aware validation
- **`models_codeAssigner.py`** — CodeAssignedSubmodel/Model, CodeAssignment/Batch
- **`run_experiment.py`** — orchestrator: loads step 3/4/5 cache, runs assigner, saves results
- **`debug_assignment_prompt.py`** — inspect captured assignment prompts
- **`view_assignments_codes.py`** — view per-idea code assignments with confidence
- **`view_ideas.py`** — view ideas grouped by code
