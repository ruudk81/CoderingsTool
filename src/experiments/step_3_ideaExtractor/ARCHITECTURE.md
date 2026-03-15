# Step 3 IdeaExtractor v5 — Architecture & Data Flow

## Taxonomy

Dimension → Domain → Facet → Attribute.

Conceptual progression: `information type → subject → analytical lens → observable property`

| Level | Name | Question it answers | Key idea |
|-------|------|-------------------|----------|
| L1 | Dimension | What type of information does this statement provide? | The informational role — not the topic |
| L2 | Domain | What is this statement about? | The subject the statement refers to |
| L3 | Facet | Through what analytical lens is the subject being examined? | An independently analyzable interpretive dimension |
| L4 | Attribute | What specific characteristic is being described? | A named observable property (not a verbatim span) |

Each level represents a qualitatively different analytical layer: Dimension identifies the information type, Domain the subject, Facet the analytical lens, and Attribute the specific observable property. Each dimension defines its own diagnostic questions for Domain, Facet, and Attribute (via `prompt_rules`), adapting their meaning to the type of information being analyzed.

**Step 3** operates at L1 (Dimension) and L2 (Domain).
**Step 5** completes the taxonomy by discovering L3 (Facet) and L4 (Attribute), then derives codebook codes.

## Progressive Classification

The pipeline classifies progressively: Dimension → Domain → Facet → Attribute. Each level is discovered or assigned in sequence, and once a level is resolved, it shapes all downstream processing.

The core mechanism: when step 3 selects a dimension, its `DimensionDefinition` from `dimension_data.py` becomes the source of truth for all subsequent prompts. The definition provides:
- **Diagnostic questions** (`domain_diagnostic`, `facet_diagnostic`) that adapt the meaning of Domain and Facet to the selected dimension
- **Extraction rules** (`instance_instruction`, `interpretation_instruction`, etc.) that guide per-idea extraction
- **Worked examples** showing the taxonomy in action for this dimension type

These fields are injected into prompt builders at runtime, so every LLM call — domain discovery, idea extraction (step 3), facet discovery, attribute discovery, code generation (step 5) — is dynamically shaped by the selected dimension's semantics.

## Design Intent

Dimension-based idea extraction from open-ended survey responses. Key design choices:
- **11 MECE dimensions** (10 substantive + GENERAL_OTHER fallback) with decision-tree ordering as the organizing principle
- **Data-driven context discovery** (language, sector, perspective, etc.) before extraction
- **Domain discovery** producing 5–15 MECE thematic domains per dimension
- **Per-idea extraction**: domain (L2) assignment + abstraction ladder (instance → interpretation → abstraction) as input for downstream taxonomy completion
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
    --> [Decision Tree Prompt] binary walk through 11 dimensions per chunk
    --> majority rule (>50%) or [Consolidation Prompt] with response samples
    --> single primary dimension + description

Phase 3 — Domain Discovery (optional):
  Sampled responses + dimension + specifiers --> chunk
    --> [Discovery Prompt] MECE domains per chunk
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

11 dimensions in decision-tree priority order:

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
| 11 | GENERAL_OTHER | Fallback for responses that don't fit a specific dimension |

Each `DimensionDefinition` (frozen dataclass) contains:
- `criterion` + `criterion_signals` — diagnostic question and signals for decision tree
- `exclusions` — what this dimension is NOT
- `pattern` — template with `[ANCHOR_SUBJECT]` and domain slot (e.g., `[PRESCRIPTIVE_CHANGE_OUTCOME_ENABLER]`)
- `prompt_rules` — dimension-specific instructions injected into prompt builders at runtime (see `PromptRules` fields: `instance_instruction`, `interpretation_instruction`, `abstraction_instruction`, `facet_instruction`, `domain_instruction`, `domain_diagnostic`, `facet_diagnostic`)
- `examples` — worked examples showing the full extraction output for this dimension
- `anchor_slot` + `domain_slot` — typed slot definitions (`SlotDefinition` with type and guidance)

The `prompt_rules` are the mechanism by which dimension-specific semantics flow into LLM prompts. Step 5 loads the same `DimensionDefinition` (via the cached dimension key) to parameterize its facet and attribute discovery prompts.

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
- **Input**: chunked responses + all 11 dimension criteria
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

## Step 3's Taxonomy Contribution

Step 3 establishes the first two taxonomy levels for the dataset:

- **Dimension (L1)**: selected once for the entire dataset — identifies the dominant type of variation across responses
- **Domain (L2)**: assigned per idea from the discovered domain set — identifies the subject area each idea refers to

## Per-Idea Extraction Metadata

In addition to the L2 domain assignment, step 3 extracts an **abstraction ladder** per idea — graded meaning representations that are *not* taxonomy levels themselves:

```
Instance:       "more bike lanes"             verbatim span from the response
Interpretation: "cycling infrastructure"      concrete meaning in survey language
Abstraction:    "sustainable urban mobility"  broader significance / theme
Domain:         "infrastructure and mobility" L2 taxonomy assignment
Valence:        "+"                           directional effect (+, -, 0)
```

The abstraction ladder is extraction metadata. Step 5 uses it as input to discover and assign Facets (L3) and Attributes (L4). Instance is NOT the same as Attribute — instance is a verbatim span; Attribute (L4) is a named observable property discovered in step 5.

## Step 3 → Step 5 Handoff (Cached Contract)

Step 3 persists its results in `ExtractionMetadata` (cached alongside the extracted ideas). This cache is the contract step 5 depends on:

- **`primary_dimension`** — dimension key (e.g., `"PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS"`). Step 5 uses this to load the full `DimensionDefinition` from `dimension_data.py`, which provides the dimension-specific semantics for facet/attribute discovery prompts.
- **`primary_dimension_description`** — context-specific description of the dimension
- **`domains`** — list of `{key, label, definition}` for all discovered domains. Step 5 uses these as the partition structure for facet discovery.
- **`lang`, `sector`, `entity`, `topic`, `perspective`, `intent`** — survey context specifiers, injected into step 5 prompts

Per idea (in `IdeasExtractedModel`):
- **`domain`** (L2), **`instance`**, **`interpretation`**, **`abstraction`**, **`valence`**, **`idea`** (canonical statement)

Step 5 uses this to: discover Facets (L3) within domains, assign Facets to ideas, discover Attributes (L4) within Facets, and derive codebook codes from Attributes.

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
- `dimension_data.py` — 11 dimension definitions (frozen dataclasses) + type system + fallback tables
- `models_exp.py` — data models (ExtractionMetadata, IdeasExtractedModel, EmbeddingsModel, etc.)
- `run_experiment.py` — orchestrator: loads step 2 cache, runs extraction, saves results + metadata
- `debug_samples.py` — inspect cached extraction results with full idea details
- `debug_full_prompts.py` — inspect captured prompts
- `view_by_cluster.py` — view ideas grouped by domain
