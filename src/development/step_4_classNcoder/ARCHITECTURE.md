# Step 5 Categories v3 — Architecture & Data Flow

## Design Intent

Taxonomy-driven inductive code generation. Step 3 delivers **Dimension (L1)** and **Domain (L2)** per idea. Step 5 completes the taxonomy by discovering **Facets (L3)** and **Attributes (L4)**, then derives codebook codes from attributes.

Key principles:
- **Dimension-specific semantics** — facet/attribute meaning adapts per dimension using `dimension_data.py` as source of truth
- **Two-stage induction** — first discover and assign facets, then discover attributes within facets
- **Codes from attributes** — codebook entries are grounded in concrete observable attributes (L4), not abstract facets

## Taxonomy Structure

```
Level   | Name      | Source  | Semantic question (dimension-dependent)
--------|-----------|---------|----------------------------------------
L1      | Dimension | Step 3  | What type of variation? (fixed per dataset)
L2      | Domain    | Step 3  | [dimension-specific] e.g., "What part of the system?"
L3      | Facet     | Step 5  | [dimension-specific] e.g., "What type of change?"
L4      | Attribute | Step 5  | [dimension-specific] e.g., "What exactly is proposed?"
Code    | Code      | Step 5  | Derived from attributes
```

The dimension determines what facet and attribute _mean_. These semantics are defined in
`step_3_ideaExtractor/dimension_data.py` per `DimensionDefinition.prompt_rules`:

| Dimension | Facet means | Facet diagnostic | Attribute means |
|-----------|------------|------------------|-----------------|
| PRESCRIPTIVE | Type of change proposed | "How should it change?" | Concrete suggestion |
| EVALUATION | Evaluation criterion | "Speed? cost? quality?" | Specific evaluation signal |
| EXPERIENCE | Experiential dimension | "Flow? atmosphere? interaction?" | Observed experience feature |
| ATTRIBUTES | Attribute category | "Visual? emotional? functional?" | Trait mentioned |
| IDENTITY | Aspect of identity | "Which aspect of identity?" | Defining feature |
| ACTORS | Role of actor | "What role do they play?" | Actor mentioned |
| CONTEXT | Context dimension | "Time? location? constraint?" | Concrete condition |
| MOTIVATIONS | Type of motivation | "Need? goal? fear? value?" | Expressed reason |
| BEHAVIOR | Functional stage | "Which step or function?" | Described action |
| RELATIONS | Relationship type | "Dependency? trade-off? influence?" | Relationship statement |

## Pipeline Overview

```
Data loading:
  Step 3 cache --> IdeasExtractedModel[] + ExtractionMetadata
  dimension_data.py --> DimensionDefinition (prompt_rules, examples, etc.)

Partition discovery:
  Ideas --> group by domain (L2) --> PartitionSet + label mappings

Phase 1: Facet Discovery (per domain, chunked, concurrent)
  Observations --> [P1] Discover facets using dimension-specific facet semantics
                --> programmatic dedup
                --> consolidated facet set per domain

Phase 2: Facet Assignment (per domain, concurrent)
  Ideas + Facet set --> [P2] Assign each idea to a facet (L3)
                    --> ideas now have domain (L2) + facet (L3)

Step 3: Attribute Discovery (per facet within domain, concurrent)
  Ideas grouped by facet --> [P3] Discover attributes (L4) within each facet
                         --> attribute set per facet

Step 4a: Attribute Assignment (per facet, concurrent)
  Ideas + Attributes per facet --> [P4a] Assign each idea to an attribute (L4)
                               --> ideas now have domain + facet + attribute

Step 4b: Attribute Consolidation (cross-facet within domain)
  Attributes + assignment frequencies --> [P3.5] Deduplicate across facets
                                      --> consolidated attribute inventory with frequency data
                                      --> remap idea assignments to consolidated names

Step 5: Code Generation (per domain, valence-split)
  Frequency-weighted attribute inventory --> [P4] Derive codes from attributes
                                         --> codebook (codes grounded in L4 attributes)
  P4.5: Cross-domain codebook consolidation --> final MECE codebook

Step 6: Code Assignment (with embedding pre-filter)
  Ideas + Codebook --> [P5] Assign codes to ideas (code-only, attribute already assigned)
                   --> embedding pre-filter selects top-5 codes per idea
                   --> assigned ideas (code + attribute + confidence)
```

## Dimension-Specific Prompt Injection

All prompts dynamically adapt to the active dimension by loading the `DimensionDefinition`
from `dimension_data.py` via `get_dimension(primary_dimension)`. This provides:

- **`prompt_rules.facet_instruction`** — full guidance for facet-level extraction
  (e.g., "What type of change is proposed? Name the change approach or intervention type.")
- **`prompt_rules.facet_diagnostic`** — short-form facet question for prompt headers
  (e.g., "How should it change?")
- **`prompt_rules.attribute_instruction`** — full guidance for attribute-level extraction
- **`prompt_rules.attribute_diagnostic`** — short-form attribute question for prompt headers
- **`prompt_rules.domain_diagnostic`** — short-form domain question
  (e.g., "What part of the system should change?")
- **`prompt_rules.domain_instruction`** — full domain classification guidance
- **`dimension_description`** — what kind of variation this dimension captures
- **`examples`** — worked examples showing domain/facet/attribute for this dimension

These replace the generic taxonomy explanation blocks from v2.

## Prompts

### P1: Facet Discovery (per domain, chunked)
- **Input**: observations (abstraction ladder labels) within one domain
- **Dimension-specific**: uses `facet_instruction` and `facet_diagnostic` to define what a facet means for this dimension
- **Output**: candidate facets with descriptions and example observations
- **Runs**: N chunks per domain (overlapping), all concurrent
- **Post-processing**: programmatic dedup + optional LLM consolidation

### P2: Facet Assignment (per domain)
- **Input**: ideas within one domain + discovered facet set
- **Dimension-specific**: uses `facet_diagnostic` to frame the assignment question
- **Output**: each idea assigned to exactly one facet
- **Runs**: batched per domain, concurrent
- **Purpose**: every idea gets a facet (L3) assignment before attribute discovery

### P3: Attribute Discovery (per facet)
- **Input**: ideas within one facet, grouped by domain
- **Dimension-specific**: uses attribute-level semantics from the dimension
- **Output**: concrete attributes (L4) within the facet
- **Runs**: per facet within domain, concurrent

### P4: Code Generation (cross-domain)
- **Input**: all attributes across all domains and facets
- **Output**: codebook codes derived from attributes, with definitions and indicators
- **Runs**: single call (or batched if attribute count is large)
- **Purpose**: synthesize concrete attributes into operational codes

### P5: Code Assignment
- **Builder**: `build_category_assignment_prompt()` (batch) or `build_single_idea_assignment_prompt()` (single)
- **Input**: ideas + codebook codes
- **Runs**: batched per domain, concurrent
- **Modes**: "batch" (N ideas per call) or "single" (one idea per call)

## Data Flow: What Step 3 Provides

`ExtractionMetadata` from step 3 cache:
- `primary_dimension` — dimension key (e.g., `"PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS"`)
- `primary_dimension_description` — context-specific description
- `domains` — list of `{key, label, definition}` for discovered domains
- `lang`, `sector`, `entity`, `topic`, `perspective`, `intent` — survey context

Per idea (`IdeasExtractedSubmodel`):
- `domain` — Domain (L2) assignment
- `facet` — Facet (L3): step 3 hint; completed by step 5 P2
- `attribute` — Attribute (L4): named observable property (assigned by step 5)
- `instance` — abstraction ladder rung 1: verbatim span
- `interpretation` — abstraction ladder rung 2: concrete meaning
- `abstraction` — abstraction ladder rung 3: broader significance
- `idea` — full idea text with template prefix
- `valence` — directional effect (+, -, 0)

## Concurrency & Rate Limiting

Same bootstrap pattern as v2:
1. Fetch real rate limits from API response headers
2. Run 3 probe calls to measure avg latency and token usage
3. Compute optimal concurrency via Little's Law
4. Shared `asyncio.Semaphore` + `AsyncLimiter` for all LLM calls

## Key Differences from v2

| Aspect | v2 | v3 |
|--------|----|----|
| Taxonomy depth | Stops at facet-level clusters | Full L3 (facet) + L4 (attribute) |
| Dimension awareness | Generic taxonomy explanation | Dimension-specific semantics from `dimension_data.py` |
| Facet handling | Discovered as clusters, not assigned back | Discovered then assigned to each idea |
| Code derivation | Codes = refined facet-level clusters | Codes derived from L4 attributes |
| Prompt semantics | One-size-fits-all | Adapts facet/attribute meaning per dimension |
| Pipeline stages | 4 prompts + assignment | 5 phases + assignment |

## Files

- `prompts_exp.py` — prompt builders + Pydantic response models
- `qualitative_researcher.py` — main orchestrator (P1-P4) with async concurrency
- `category_assignment.py` — assignment orchestrator (P5) with queue-based workers + retry
- `config_categories_exp.py` — configuration (`CategoriesConfig`, `AssignmentConfig`)
- `models_exp.py` — data models (`PartitionSet`, `MECEResultsCache`, `CategoryAssignedModel`)
- `partition_discoverer.py` — partition ideas by domain, collect unique labels
- `partition_labels.py` — label extraction/formatting (ladder composites, prefixes)
- `run_experiment.py` — experiment runner (full pipeline + assignment-only mode)
- `debug_assignment_prompt.py` — debug helper for inspecting assignment prompts
- `debug_full_prompts.py` — debug helper for inspecting all pipeline prompts
- `view_assignments.py` — view/analyze assignment results

## Source of Truth

- **Dimension definitions**: `step_3_ideaExtractor/dimension_data.py` — `DIMENSIONS` registry
- **Taxonomy logic**: `step_3_ideaExtractor/taxonomy_logic.md` — conceptual mapping per dimension
- **Step 3 output**: cached `ExtractionMetadata` + `IdeasExtractedModel[]`
