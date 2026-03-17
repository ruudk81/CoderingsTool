# Step 4 classNcoder — Work To Be Done

### Prerequisites (in step 3)

Jobs 2 and 3 below depend on step 3's Job 1 (audit dimension_data.py + MECE enforcement in domain prompts). See `step_3_ideaExtractor/work_to_be_done.md`.

---

## Job 1: Align prompt processing strategy with step 3 best practices

### Context

The `prompt processing strategy.md` in step 3 documents hard-won lessons from processing 1375 tasks at scale. It was written as a reusable reference for downstream steps. The current `code_assignment.py` in step 4 does not follow it — it uses a simpler bootstrap-based approach ported from the old `qualityFilter.py`.

### Current approach (code_assignment.py)

| Component | What it does |
|-----------|-------------|
| **Rate limit fetch** | 1 API call to read `x-ratelimit-*` headers |
| **Bootstrap** | 3 dedicated probe calls upfront to measure avg latency + avg tokens |
| **Concurrency** | `asyncio.Semaphore` — static, set once from Little's Law, never changes |
| **TPM** | `TokenBucket` — self-regulating, with reconcile |
| **RPM** | `AsyncLimiter` — fixed arrival rate, never adjusted |
| **Timeout** | None — no `asyncio.wait_for`, workers run until done or error |
| **Circuit breaker** | None |
| **PID controller** | None |
| **Concurrency ramp** | None — full concurrency from the start |
| **Warm-up calibration** | None — bootstrap tokens are used throughout |
| **Throughput adjustment** | Threshold-based: if actual avg tokens > bootstrap × 1.3, recalculate arrival rate. No PID feedback. |
| **Workers** | `min(200, max(10, optimal * 2))` — static, never scaled |
| **Progress** | Basic: completed/total, rate, ETA |
| **Constraint visibility** | None — no TPM%, RPM%, concurrency utilization |

### Intended approach (from step 3's prompt processing strategy)

| Component | What it does |
|-----------|-------------|
| **Rate limit fetch** | Same — 1 API call |
| **Bootstrap** | None — no dedicated probes. Token estimate from tiktoken on sample prompts (local, no API). Latency learned from first N completions (warm-up). |
| **Concurrency** | `ConcurrencyGate` — supports dynamic adjustment (up/down). Ramp scales with completion progress. |
| **TPM** | `TokenBucket` — same pattern, with reconcile |
| **RPM** | `AsyncLimiter` — PID-adjusted arrival rate every 20s based on real-time TPM utilization |
| **Timeout** | `asyncio.wait_for` with generous safety net: 60s floor, P95×3 adaptive, 120s ceiling. Computed AFTER semaphore. |
| **Circuit breaker** | `ConcurrencyCircuitBreaker` — monitors timeout RATE in sliding window. CLOSED → OPEN (reduce 15%) → RECOVERING (+10% per 30s) → CLOSED. |
| **PID controller** | `PIDThroughputController` — asymmetric gains (aggressive up, gentle down). Adjusts AsyncLimiter arrival rate every 20s. |
| **Concurrency ramp** | `ConcurrencyRamp` — completion-based: 50% → 90% of Little's Law proportional to progress. Two stop signals: throughput drop (>10% decline × 2), queue congestion (>5% timeout rate). |
| **Warm-up calibration** | After 15-30 completions: measure actual tokens + P10 latency → recalculate Little's Law → recalibrate ramp → spawn extra workers if needed. One-shot. |
| **Throughput adjustment** | Threshold check (same as current) PLUS PID fine-tuning every 20s |
| **Workers** | Initial = ramp target (90% of Little's Law). After warm-up: extra workers spawned if ramp target increased. Workers match capacity — no idle workers queued at semaphore. |
| **Progress** | Rich: completed/total, rate, TPM%, RPM%, Concurrency% utilization, circuit breaker state, ramp status |
| **Constraint visibility** | Every progress report shows which constraint is the bottleneck |

### Key differences (what needs to change)

| # | Difference | Impact |
|---|-----------|--------|
| 1 | **Bootstrap probes → warm-up calibration**. Current: 3 dedicated API calls wasted before work starts. Intended: first 15-30 real tasks serve as warm-up. Token estimate starts from tiktoken (local). | Saves 2-5s startup time; more accurate calibration from real data. |
| 2 | **Static Semaphore → ConcurrencyGate + ConcurrencyRamp**. Current: full concurrency from T+0. Intended: ramp from 50% → 90% proportional to completion progress, with congestion detection. | Prevents burst throttling; adapts to server-side queuing. |
| 3 | **No timeout → generous safety-net timeout**. Current: workers never timeout. Intended: 60s floor, P95×3, computed after semaphore. Timed-out tasks get fallback (no retry). | Catches truly stuck requests; prevents infinite hangs. |
| 4 | **No circuit breaker → ConcurrencyCircuitBreaker**. Current: if server is overwhelmed, nothing changes. Intended: monitor timeout rate, reduce concurrency on sustained pressure. | Graceful degradation under server stress. |
| 5 | **Fixed arrival rate → PID-adjusted**. Current: AsyncLimiter never changes after init. Intended: PID adjusts every 20s based on real-time TPM utilization. | Better utilization; avoids both under- and over-shooting. |
| 6 | **Static workers → scaled after warm-up**. Current: 200 workers fixed. Intended: workers = ramp target, extra spawned after calibration. | No idle workers queued at semaphore; stale timeout values avoided. |
| 7 | **No constraint visibility → TPM%, RPM%, Conc% in every progress report**. | Understand bottlenecks at a glance. |
| 8 | **No tiktoken offset learning**. Current: fixed avg_tokens from bootstrap. Intended: `TiktokenOffsetLearner` tracks tiktoken vs API token gap. | Tighter token estimates → better bucket management. |
| 9 | **No latency tracking**. Current: latency only from bootstrap probes. Intended: `LatencyTracker` with EMA, feeds timeout calculation and Little's Law recalibration. | Adaptive timeouts; accurate concurrency sizing. |

### Implementation approach

Import the rate limiting classes directly from `ideaExtractor_exp.py` (they are fully decoupled — no step 3 dependencies):

- `TokenBucket` (replace local copy)
- `ConcurrencyGate` (replace `asyncio.Semaphore`)
- `LatencyTracker`
- `TiktokenOffsetLearner`
- `ConcurrencyRamp`
- `RealTimeTPMTracker`
- `RealTimeRPMTracker`
- `PIDThroughputController`
- `ConcurrencyCircuitBreaker`
- `ApiLimits`, `compute_optimal_concurrency`

Then restructure `_assign_all_async()` to follow the lifecycle phases from the strategy doc:
1. Fetch API rate limits
2. Conservative initialization (for any pre-processing)
3. Token estimation via tiktoken (local)
4. Production rate limiting initialization (ConcurrencyGate, ramp, PID, circuit breaker)
5. Main processing loop with warm-up calibration trigger
6. Cleanup (fallback for timed-out tasks)

### NOT in scope

- Retries (step 3 uses tenacity for rate limit errors; step 4 can keep its current retry logic)
- Context extraction / specifier discovery (step 3 specific)
- Adaptive token estimation with output ratio learning (overkill for classification — input/output is predictable)

---

## Job 2: Improve P1.5 — Stronger MECE enforcement for facet consolidation

### Problem

The current P1.5 consolidation prompt merges near-duplicate facets across chunks (e.g., two chunks both discovering "sfeer en ambiance") but allows conceptual neighbors to survive as separate facets.

Example from "financiële aspecten" domain — P1.5 produces 6 facets:
- "Financiële producten en diensten" (what ASN offers)
- "Financiële kenmerken, voorwaarden en kosten" (properties of what ASN offers)
- "Financiële functionaliteit en transacties" (using what ASN offers)
- "Financiële gezondheid, betrouwbaarheid en prestaties" (how ASN performs)
- "Financiële verantwoordelijkheid, ethiek en duurzaamheid" (how ASN invests)
- "Financiële gedragingen en doelstellingen" (what customers do / ASN's goals)

These are not MECE — "kosten" is a property of "producten", "functionaliteit" is a usage aspect of "producten", and an observation like "sparen" could plausibly belong to 3 of these facets.

### Root cause

The P1.5 prompt asks to "MERGE facets that have conceptual overlap" but doesn't enforce a pairwise check. The LLM sees the facets as genuinely different analytical lenses (which they are, abstractly) but doesn't test whether real observations would be ambiguous between them.

### Fix

Prompt-level improvement: add an explicit pairwise MECE check step where the LLM asks for each pair of surviving facets: "Could an observation plausibly belong to both?" If yes → merge.

### Prompt spec

To be defined in `new_1dot5_and_3dot5_prompts.md` (user writes prompt content, we implement from it).

---

## Job 3: Add P3.5 — Attribute consolidation within domain (MECE enforcement)

### Problem

P3 discovers attributes per facet independently. When P1.5 allows overlapping facets to survive, P3 compounds the problem by discovering the same concept as an attribute in multiple facets. Example: "sparen" appears as an attribute in "Financiële functionaliteit", "Financiële gedragingen", and "Financiële producten".

Even if P1.5 is improved (Job 2), some attribute-level overlap will remain because different facets can legitimately share border concepts.

### What P3.5 does

After P3 completes for a domain, consolidate all attributes within that domain into a MECE set:
- Input: all facets + their attributes for one domain (output of P3)
- Task: merge duplicate/overlapping attributes across facets, assign each surviving attribute to its best-fitting facet
- Output: deduplicated attribute inventory for the domain
- Does NOT restructure facets — that's P1.5's responsibility

### Design

- One LLM call per domain, concurrent across domains
- Model: 4.1 (synthesis/judgment task, not classification)
- Runs after P3, before P4
- Response model: consolidated list of attributes, each with parent facet and merged-from provenance

### Pipeline position

```
P1 → P1.5 → P2 → P3 → P3.5 → P4 → P4.5 → P5
       ↑                  ↑              ↑
  facet MECE        attribute MECE   code MECE
```

Each consolidation step is responsible for MECE at its own level only. P3.5 does not fix facet-level problems — it handles attribute-level overlap that legitimately occurs when different facets share border concepts.

### Prompt spec

To be defined in `new_1dot5_and_3dot5_prompts.md` (user writes prompt content, we implement from it).

---

## Job 4: Enrich P4.5 codebook consolidation with boundary guidance

### Problem

The current P5 assignment prompt gives the LLM a code definition, indicators, and attributes — but no guidance on how to handle ambiguous cases. When two codes cover similar territory (e.g., "Duurzaamheid en ethisch bankieren (positief)" vs "Maatschappelijke betrokkenheid en progressieve positionering (positief)"), the LLM has no boundary information to decide between them. The result: inconsistent assignments for edge cases.

### Why P4.5

P4.5 already reviews and consolidates codes across domains. It sees the full codebook at once — making it the natural place to generate boundary guidance, because it can compare neighboring codes directly. P4 (per-domain) can't do this because it doesn't see cross-domain codes.

### What changes in P4.5

The consolidation prompt gets an additional task: for each surviving code, generate:
- **`boundary_test`**: a short decision rule that distinguishes this code from its closest neighbor(s). E.g., "Assign here when the idea describes a financial *product or service*; assign to C4 when the idea describes the *cost or pricing* of that product."
- **`negative_indicators`**: words/phrases that signal this code is NOT the right fit, even if it looks close. E.g., "Not for ideas about the *ethical policy* behind investments — that's C9."

### What changes in the response model

`CodeFromAttributes` (or the P4.5 consolidation output) gets two new fields:
- `boundary_test: str`
- `negative_indicators: List[str]`

### What changes downstream in P5

The codebook block in the assignment prompt includes boundary test and negative indicators per code. The assignment instructions reference them for disambiguation.

---

## Job 5: Embedding-based code pre-filtering for P5 assignment

### Problem

The current P5 assignment prompt presents all 17 codes with all ~250 attributes to nano for every idea. This is wasteful (~3000 tokens per prompt on the codebook alone) and forces the LLM to scan through irrelevant codes. For a classification task on a small model, a smaller candidate set produces better accuracy and lower cost.

### Goal

Use embedding similarity to pre-filter the codebook to the top-5 most relevant codes per idea. Combine with facet-based attribute filtering so each idea only sees attributes from codes that share its facet.

### Approach overview

```
Pre-computation (once per run):
  1. Embed all codes → code_embeddings matrix [N_codes × dim]
  2. Embed all ideas → idea_embeddings matrix [N_ideas × dim]
  3. Compute cosine similarity → [N_ideas × N_codes]
  4. For each idea: top-5 code indices

Per-idea prompt (P5):
  5. Build scoped codebook: 5 codes with full assignment info
  6. Filter attributes: only attributes from codes whose source facets match the idea's facet
  7. LLM assigns code + attribute
```

### Step 1: Standalone util — `embedding_matcher.py`

A single file in `step_4_classNcoder/`. No dependency on step 4 embedder or clusterer. Contains:

```python
class EmbeddingMatcher:
    """Embed texts via OpenAI API, compute cosine similarity, return top-N matches."""

    def __init__(self, model: str = "text-embedding-3-large"):
        ...

    async def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts. Returns [N × dim] numpy array.
        Batches API calls to stay within rate limits."""
        ...

    def compute_similarity(self, query_embeddings: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix. Returns [N_queries × N_corpus]."""
        ...

    def top_n_matches(self, similarity_matrix: np.ndarray, n: int = 5) -> List[List[int]]:
        """For each query, return indices of top-N corpus items."""
        ...
```

Copy from the existing embedder util: the OpenAI embedding API call (supporting both OpenAI and Azure via `config.py`) and batching logic. Nothing else (no UMAP, no clustering, no caching).

### Step 2: Embedding content

**Idea embedding** string per idea:
```python
f"{idea.domain} | {facet_lookup[idea.idea_id]} | {idea.interpretation} | {idea.abstraction}"
```

**Code embedding** string per code:
```python
f"{code.code_name} | {code.definition} | {', '.join(code.typical_indicators)}"
```

Rationale:
- Drop dimension (constant across all items — no discriminative signal)
- Drop instance for ideas (noisy subset of interpretation)
- Codes use definition + indicators rather than domain/facet because P4.5 consolidation merges codes across domains — a code may not have a single "home" domain

### Step 3: Pre-computation in the assignment orchestrator

In `CodeAssigner._assign_all_async()`, before the worker loop:

```python
matcher = EmbeddingMatcher()
idea_embeddings = await matcher.embed_texts(idea_texts)   # ~1850 texts
code_embeddings = await matcher.embed_texts(code_texts)    # ~17 texts
similarity = matcher.compute_similarity(idea_embeddings, code_embeddings)
top5_per_idea = matcher.top_n_matches(similarity, n=5)

self._idea_code_candidates = {
    idea.idea_id: [self._codes[idx] for idx in top5_per_idea[i]]
    for i, idea in enumerate(all_ideas)
}
```

Runs once before workers start. Embedding ~1850 ideas + 17 codes takes ~5-10 seconds with `text-embedding-3-large`.

### Step 4: Attribute filtering by facet

For each idea, after selecting the top-5 codes, filter attributes:
- Each code has `source_attributes` (list of attribute names)
- Each attribute was discovered under a specific facet (from P3 output)
- The idea has an assigned facet (from P2/facet_lookup)
- Show only attributes whose parent facet matches the idea's facet

Requires a mapping `attribute_name → parent_facet`. This is implicit in the cached `attributes` dict (`Dict[facet_name, List[DiscoveredAttribute]]`) — invert it:
```python
{attr.attribute_name: facet_name for facet_name, attrs in attributes.items() for attr in attrs}
```

If no attributes match the idea's facet for a given code, show all attributes for that code as fallback.

### Step 5: Modified prompt builder

New or modified `build_scoped_dual_assignment_prompt()`:

```
<codebook>
[C1] Duurzaamheid en ethisch bankieren (positief)
    Definition: ...
    Indicators: ...
    Boundary test: ...              ← from Job 4
    Negative indicators: ...        ← from Job 4
    Attributes (matching your facet):
      - duurzaam beleggen
      - groen sparen

[C3] Functionele financiële dienstverlening (positief)
    Definition: ...
    ...
</codebook>
```

Only 5 codes shown. Attributes filtered to facet-relevant ones.

### Step 6: Updated worker flow

```python
candidate_codes = self._idea_code_candidates[idea.idea_id]
prompt = build_scoped_dual_assignment_prompt(
    ...,
    codes=candidate_codes,          # top-5 instead of all
    idea=idea,
    facet_lookup=self._facet_lookup,
    attribute_facet_map=self._attribute_facet_map,
)
```

### Step 7: Data that needs to be cached/passed

| Data | Source | Needed by |
|------|--------|-----------|
| `codes` (List[CodeFromAttributes]) | P4.5, cached in `CodingResultsCache.raw_codes` | Embedding + prompt |
| `facet_lookup` (idea_id → facet name) | P2, cached in `DomainResultModel.facet_assignments` | Attribute filtering |
| `attribute_facet_map` (attr_name → parent_facet) | P3, reconstruct from cached `attributes` dict | Attribute filtering |

### Dependencies on other jobs

- **Job 4** (boundary guidance): scoped codebook includes `boundary_test` and `negative_indicators`. Job 5 works without Job 4 but the combination gives the real quality gain.
- **Jobs 2+3** (MECE enforcement): cleaner facets/attributes → better facet-based filtering. Not a blocker.

### Estimated token savings

| Component | Current | After Job 5 |
|-----------|---------|-------------|
| Codes shown | 17 | 5 |
| Attributes shown | ~250 | ~15-30 (facet-filtered) |
| Codebook tokens | ~2500 | ~500-800 |
| Total prompt tokens | ~3300 | ~1300-1600 |

~50-60% token reduction per call × 1847 ideas.

### Validation

Run assignment with both approaches on the same dataset:
- Full codebook (current) vs. top-5 scoped
- Compare: agreement rate, confidence distribution, assigned code distribution
- Check miss rate: how often is the full-codebook assignment NOT in the top-5 candidates? (Target: <2%)

---

## Jobs completed

- **Job 1**: Aligned code_assignment.py with step 3 rate limiting strategy (commit `ea13df8`, 2026-03-17)