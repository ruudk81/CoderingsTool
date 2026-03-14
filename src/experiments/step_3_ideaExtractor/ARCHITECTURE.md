# Step 3 IdeaExtractor v5 — Architecture & Data Flow

## Taxonomy

Dimension > Domain > Facet > Attribute (progressive narrowing).
Step 3 operates at the **Dimension** (L1) and **Domain** (L2) levels.

## Design Intent

Dimension-based idea extraction from open-ended survey responses. Key design choices:
- **10 MECE dimensions** with decision-tree ordering as the organizing principle
- **Data-driven context discovery** (language, sector, perspective, etc.) before extraction
- **Domain discovery** producing 5–15 MECE thematic domains per dimension
- **Taxonomy fields** per idea: instance (L4 Attribute) → facet (L3) + domain (L2)
- **Canonical phrasing** via dimension template patterns for normalized idea statements
- **PID-controlled rate limiting** with learned tiktoken offset for zero 429 errors

## Pipeline Overview

```
Data loading:
  Step 2 cache --> QualityFilteredModel[] (meaningful responses only)

Phase 1 — Context Specifier Extraction:
  Sampled responses --> chunk into groups of 10
    --> [Prompt Group1] lang, perspective, intent    (concurrent per chunk)
    --> [Prompt Group2] sector, topic, entity         (concurrent per chunk)
    --> [Consolidation] single set of 6 specifiers    (LLM if >1 chunk)

Phase 2 — Primary Dimension Selection:
  Sampled responses + specifiers --> chunk
    --> [Decision Tree Prompt] binary walk through 10 dimensions per chunk
    --> majority rule (>50%) or [Consolidation Prompt] with response samples
    --> single primary dimension + description

Phase 3 — Domain Discovery (optional):
  Sampled responses + dimension + specifiers --> chunk
    --> [Discovery Prompt] 5–15 MECE domains per chunk
    --> [Consolidation Prompt] merge into final MECE set

Phase 4 — Bootstrap Measurement:
  5 serial probe calls --> avg latency + avg tokens
    --> initialize PID controller, token bucket, concurrency ceiling

Phase 5 — Idea Extraction (bulk):
  Per response (async queue, N workers):
    Response + dimension + domains + specifiers
      --> build canonical phrasing from dimension template
      --> [Extraction Prompt] split → reformulate → classify → ladder
      --> List[DimensionTaxonomy] per response
      --> validate, retry (up to 2x), or fallback PROCESSING_ERROR

Phase 6 — Result Assembly:
  All extracted ideas --> IdeasExtractedModel[] + ExtractionMetadata --> cache
```

## Dimension System (dimension_data.py)

10 dimensions in decision-tree priority order:

| # | Dimension | What it captures |
|---|-----------|-----------------|
| 1 | PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS | Proposed actions, improvements, suggestions |
| 2 | IDENTITY_DEFINITION | What an entity IS or means |
| 3 | ACTORS_TARGETS | Who is involved or impacted |
| 4 | CONTEXT_CONDITIONS | When, where, under what conditions |
| 5 | MOTIVATIONS_DRIVERS | Why people care, want, or act |
| 6 | EXPERIENCE_PERCEPTION | How something was experienced |
| 7 | EVALUATION_PRIORITIZATION | Opinions, judgments, preferences |
| 8 | BEHAVIOR_FUNCTION | What happens, how it works |
| 9 | ATTRIBUTES_ASSOCIATIONS | Qualities, traits, images |
| 10 | RELATIONS_DEPENDENCIES | Relationships, trade-offs |

Each `DimensionDefinition` (frozen dataclass) contains:
- `criterion` + `criterion_signals` — diagnostic question and signals for decision tree
- `exclusions` — what this dimension is NOT
- `pattern` — template with `[ANCHOR_SUBJECT]` and domain slot (e.g., `[PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER]`)
- `prompt_rules` — dimension-specific extraction instructions for instance, facet, domain (with diagnostic questions)
- `examples` — worked examples with taxonomy (survey context → response → instance → domain → facet → valence)
- `anchor_slot` + `domain_slot` — typed slot definitions (`SlotDefinition` with type and guidance)

## Prompts

### Phase 1: Context Specifier Extraction

**Group 1** (speaker characteristics):
- **Builder**: `build_context_specifier_group1_prompt()`
- **Response model**: `GenericSpecifierGroup1Response`
- **Output**: lang (ISO code), perspective (stakeholder viewpoint), intent (cognitive task)
- **Consolidation**: `build_consolidate_specifiers_group1_prompt()` → same model

**Group 2** (subject matter):
- **Builder**: `build_context_specifier_group2_prompt()`
- **Response model**: `GenericSpecifierGroup2Response`
- **Output**: sector (industry/sector), topic (specific subject), entity (main entity)
- **Consolidation**: `build_consolidate_specifiers_group2_prompt()` → same model

### Phase 2: Primary Dimension Selection

**Decision Tree** (per chunk):
- **Builder**: `build_primary_dimension_decision_tree_prompt()`
- **Response model**: `PrimaryDimensionChunkResponse`
- **Input**: chunked responses + all 10 dimension criteria
- **Output**: primary_dimension, decision_tree_stop_position, evidence (verbatim snippets), clarification, description

**Consolidation** (if no majority):
- **Builder**: `build_primary_dimension_consolidation_prompt()`
- **Response model**: `PrimaryDimensionConsolidatedResponse`
- **Fast path**: `consolidate_primary_dimension_by_majority()` — no LLM call if >50% agree

### Phase 3: Domain Discovery

**Discovery** (per chunk):
- **Builder**: `build_domain_discovery_prompt()`
- **Response model**: `DomainChunkResponse` (list of `DomainItem`: key, label, definition)
- **Constraint**: 5–15 MECE domains; sharp boundaries, no overlap

**Consolidation**:
- **Builder**: `build_domain_consolidation_prompt()`
- **Response model**: `DomainConsolidatedResponse`
- **Merges**: all chunk-level domains into single MECE set, accounting for every input domain

### Phase 5: Idea Extraction

**Extraction** (per response):
- **Builder**: `build_taxonomy_enriched_extraction_prompt()`
- **Response model**: dynamically built via `create_extraction_model()`
  - `DimensionTaxonomy` — per-idea: instance, facet, domain (Literal with fuzzy matching), valence
  - `DimensionExtractionModel` — wraps list of ideas + template prefix + marker validation
- **3-step extraction**: (1) split into atomic ideas, (2) reformulate using canonical phrasing, (3) classify + valence
- **Runs**: once per response, all responses concurrent via async queue

## Taxonomy Fields

Each idea produces taxonomy-aligned fields:

```
Instance:  "more bike lanes"               Attribute (L4): verbatim span
Facet:     "infrastructure expansion"      Facet (L3): dimension-specific aspect
Domain:    "infrastructure and mobility"   Domain (L2): thematic domain
Valence:   "+"                             direction of effect
```

- **Instance** (L4 Attribute): close to verbatim, minimal reformulation
- **Facet** (L3): dimension-specific aspect — meaning shifts per dimension (e.g., "evaluation criterion" for EVALUATION, "type of change" for PRESCRIPTIVE)
- **Domain** (L2): reusable thematic area, not a per-idea descriptor
- **Valence**: +, -, or 0 — direction of effect on the domain, not sentiment

Each dimension defines its own diagnostic questions for Domain and Facet via `PromptRules.domain_diagnostic` and `PromptRules.facet_diagnostic`.

## Rate Limiting (3-tier system)

**TokenBucket**: simple bucket with regeneration based on elapsed time; enforces wait-and-acquire for token consumption.

**TiktokenOffsetLearner**: learns the gap between tiktoken estimates and actual API token counts (~300 tokens for system overhead). Maintains a deque of offsets and applies learned correction once enough samples collected.

**PIDThroughputController**: asymmetric PID gains — aggressive speed-up (kp=0.4) when under-utilizing, gentle slow-down (kp=0.2) when over-utilizing. Clamps adjustments to ±20%.

**Bootstrap** (Phase 4): 5 serial probe calls measure avg latency + tokens → initializes concurrency ceiling (Little's Law), token bucket capacity, and PID setpoint.

**Adaptive adjustment** (continuous during Phase 5):
- Threshold-based: step correction when actual tokens significantly exceed bootstrap estimate
- PID-based: continuous fine-tuning from real-time TPM utilization

## Concurrency Model

- `asyncio.Queue` with all response tasks
- N workers (2x optimal concurrency, min 10, max 200)
- `asyncio.Semaphore` enforces API concurrency ceiling
- Rate limiter controls arrival rate
- TPM bucket enforces token consumption limits
- Per-task retry (up to 2x) with PROCESSING_ERROR fallback

## Files

- `prompts_exp.py` — prompt builders + Pydantic response models (all phases)
- `ideaExtractor_exp.py` — main utility: 6-phase async pipeline with rate limiting (~2400 lines)
- `dimension_data.py` — 10 dimension definitions (frozen dataclasses) + type system + fallback tables
- `models_exp.py` — data models (ExtractionMetadata, IdeasExtractedModel, EmbeddingsModel, etc.)
- `run_experiment.py` — orchestrator: loads step 2 cache, runs extraction, saves results + metadata
- `debug_samples.py` — inspect cached extraction results with full idea details
- `debug_full_prompts.py` — inspect captured prompts
- `view_by_cluster.py` — view ideas grouped by domain
