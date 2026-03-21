# Step 5 Categories v3 — Architecture & Data Flow

## Design Intent

Taxonomy-driven inductive code generation. Step 3 delivers **Dimension (L1)** and **Domain (L2)** per idea. Step 5 completes the taxonomy by discovering **Facets (L3)** and **Attributes (L4)**, then derives codebook codes from attributes.

Key principles:
- **Dimension-specific semantics** — facet/attribute meaning adapts per dimension using `dimension_data.py` as source of truth
- **Two-stage induction** — first discover and assign facets, then discover and assign attributes within facets
- **Codes from attributes** — codebook entries are grounded in concrete observable attributes (L4), not abstract facets
- **Frequency-weighted code generation** — attribute assignment frequencies inform code derivation and consolidation

## Taxonomy Structure

```
Level   | Name      | Source  | Semantic question (dimension-dependent)
--------|-----------|---------|----------------------------------------
L1      | Dimension | Step 3  | What type of variation? (fixed per dataset)
L2      | Domain    | Step 3  | [dimension-specific] e.g., "What part of the system?"
L3      | Facet     | Step 5  | [dimension-specific] e.g., "What type of change?"
L4      | Attribute | Step 5  | [dimension-specific] e.g., "What exactly is proposed?"
Code    | Code      | Step 5  | Derived from frequency-weighted attributes
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

The pipeline has three stages: **partition discovery**, the **QualitativeResearcher pipeline** (P1–P4.5), and **code assignment**.

```
Stage 0: Partition Discovery
  Step 3 cache --> IdeasExtractedModel[] + ExtractionMetadata
  dimension_data.py --> DimensionDefinition (prompt_rules, examples, etc.)
  Ideas --> group by domain (L2) --> DomainSet + PartitionLabelMapping[]

Stage 1: QualitativeResearcher Pipeline (P1–P4.5)

  P1: Facet Discovery (per domain, chunked, concurrent)
    Observations --> discover facets using dimension-specific semantics
                 --> programmatic dedup

  P1.5: Facet Consolidation (per domain, hierarchical)
    Chunk facets --> LLM consolidation (recursive if >6 chunks or >150 items)
                 --> consolidated facet set per domain

  P2: Facet Assignment (per domain, batched, concurrent)
    Ideas + facet set --> assign each idea to a facet (L3)
                      --> ideas now have domain (L2) + facet (L3)

  P3: Attribute Discovery (per facet within domain, chunked, concurrent)
    Ideas grouped by facet --> discover attributes (L4) within each facet
                           --> chunk consolidation within facet if >100 observations

  P4a: Attribute Assignment (per facet, concurrent)
    Ideas + attributes per facet --> assign each idea to an attribute (L4)
                                 --> ideas now have domain + facet + attribute

  P3.5: Cross-facet Attribute Consolidation (per domain)
    Attributes per facet + assignment frequencies
      --> deduplicate across facets within same domain
      --> remap idea assignments to consolidated names
      --> consolidated attribute inventory with frequency data

  P4: Code Generation (per domain, sequential)
    Frequency-weighted attribute inventory --> derive codes from attributes
                                           --> codes grounded in L4 attributes

  P4.5: Codebook Consolidation (cross-domain, hierarchical)
    All codes + code frequencies --> merge across domains
                                 --> final MECE codebook (ConsolidatedCode list)

Stage 2: Code Assignment (separate, optional)

  Ideas + codebook --> embedding pre-filter selects top-N codes per idea
                   --> LLM assigns one code per idea
                   --> CodeAssignedModel (code + confidence + rationale)
```

## Dimension-Specific Prompt Injection

All prompts dynamically adapt to the active dimension by loading the `DimensionDefinition`
from `dimension_data.py` via `get_dimension(primary_dimension)`. This provides:

- **`prompt_rules.facet_instruction`** — full guidance for facet-level extraction
- **`prompt_rules.facet_diagnostic`** — short-form facet question for prompt headers
- **`prompt_rules.attribute_instruction`** — full guidance for attribute-level extraction
- **`prompt_rules.attribute_diagnostic`** — short-form attribute question for prompt headers
- **`prompt_rules.domain_diagnostic`** — short-form domain question
- **`prompt_rules.domain_instruction`** — full domain classification guidance
- **`dimension_description`** — what kind of variation this dimension captures
- **`examples`** — worked examples showing domain/facet/attribute for this dimension

## Prompts

All prompt builders live in `prompts_exp.py`.

### P1: Facet Discovery — `build_facet_discovery_prompt()`
- **Input**: observations (formatted labels) within one domain
- **Dimension-specific**: uses `facet_instruction` and `facet_diagnostic`
- **Output**: candidate facets with descriptions and example observations
- **Runs**: N chunks per domain (overlapping), all concurrent

### P1.5: Facet Consolidation — `build_facet_consolidation_prompt()`
- **Input**: facets from multiple chunks within one domain
- **Output**: deduplicated, consolidated facet set
- **Runs**: hierarchical (max 6 chunks, max 150 items per consolidation call)

### P2: Facet Assignment — `build_facet_assignment_prompt()`
- **Input**: ideas within one domain + discovered facet set
- **Dimension-specific**: uses `facet_diagnostic` to frame the assignment question
- **Output**: each idea assigned to exactly one facet
- **Runs**: batched per domain (configurable batch size), concurrent

### P3: Attribute Discovery — `build_attribute_discovery_prompt()`
- **Input**: ideas within one facet, formatted as labels
- **Dimension-specific**: uses attribute-level semantics from the dimension
- **Output**: `DiscoveredAttribute` list per facet
- **Runs**: per facet within domain, concurrent; chunked with `build_attribute_chunk_consolidation_prompt()` if >100 observations

### P4a: Attribute Assignment — `build_attribute_assignment_prompt()`
- **Input**: ideas assigned to a facet + discovered attributes for that facet
- **Output**: each idea assigned to exactly one attribute
- **Runs**: per facet, concurrent; small candidate set (~5–15 attributes) so no embedding pre-filter needed

### P3.5: Cross-facet Attribute Consolidation — `build_attribute_consolidation_prompt()`
- **Input**: attributes per facet + assignment frequency counts
- **Output**: consolidated attribute inventory; remap dict for merged names
- **Runs**: per domain (only if 2+ facets with attributes)
- **Purpose**: deduplicate semantically similar attributes across facets, informed by frequency

### P4: Code Generation — `build_code_from_attributes_prompt()`
- **Input**: consolidated attribute inventory per domain with frequencies
- **Output**: `CodeFromAttributes` list — codes derived from attributes with `source_attributes`
- **Runs**: per domain, sequential

### P4.5: Codebook Consolidation — `build_codebook_consolidation_prompt()`
- **Input**: all codes across domains + code frequencies (derived from attribute assignment counts)
- **Output**: `ConsolidatedCode` list — final MECE codebook
- **Runs**: cross-domain; hierarchical if many codes

### Code Assignment — `build_single_dual_assignment_prompt()`
- **Input**: single idea + codebook (optionally pre-filtered by embeddings)
- **Output**: `CodeAttributeAssignment` — code ID + confidence + rationale
- **Runs**: per idea, concurrent via queue-based workers in `CodeAssigner`
- **Note**: despite the legacy "dual" name, this assigns codes only; attributes are already assigned in P4a

## Data Flow: What Step 3 Provides

`ExtractionMetadata` from step 3 cache:
- `primary_dimension` — dimension key (e.g., `"PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS"`)
- `primary_dimension_description` — context-specific description
- `domains` — list of `{key, label, definition}` for discovered domains
- `lang`, `sector`, `entity`, `topic`, `perspective`, `intent` — survey context

Per idea (`IdeasExtractedSubmodel`):
- `domain` — Domain (L2) assignment
- `facet` — Facet (L3): step 3 hint; overwritten by P2 assignment
- `attribute` — Attribute (L4): populated by P4a assignment
- `instance` — abstraction ladder rung 1: verbatim span
- `interpretation` — abstraction ladder rung 2: concrete meaning
- `abstraction` — abstraction ladder rung 3: broader significance
- `idea` — full idea text with template prefix
- `valence` — directional effect (+, -, 0)

## Key Data Structures

### PartitionLabelMapping (from domain_discoverer.py)
```python
@dataclass
class PartitionLabelMapping:
    partition_name: str              # domain key
    partition: DomainDescription     # partition metadata
    labels: List[str]                # unique formatted observations
    label_count: int                 # count of unique labels
    label_domains: List[Optional[str]]
    ideas: List                      # IdeasExtractedSubmodel objects
```

### DomainResult (from qualitative_researcher.py)
```python
@dataclass
class DomainResult:
    partition_name: str
    n_labels: int
    n_batches: int
    facets: List[DiscoveredFacet]
    facet_assignments: Dict[str, str]              # idea_id → facet_name
    attributes: Dict[str, List[DiscoveredAttribute]]  # facet_name → [attrs]
    attribute_assignments: Dict[str, str]           # idea_id → attribute_name
```

### PipelineResult (from qualitative_researcher.py)
```python
@dataclass
class PipelineResult:
    partition_results: Dict[str, DomainResult]  # per-domain results
    codebook_narrative: str
    codes: List[ConsolidatedCode]               # final codebook
```

### CodeAssignedSubmodel (from models_exp.py)
```python
class CodeAssignedSubmodel(IdeasExtractedSubmodel):
    assigned_code: Optional[str]       # code name
    assigned_attribute: Optional[str]  # attribute name (from P4a)
    confidence: Optional[float]
    rationale: Optional[str]
    partition_name: Optional[str]
```

## Label Sources & Valence

Configurable in `CategoriesConfig.label_source`. Labels are formatted by `partition_labels.format_label()`.

**Stored fields** (direct attributes on ideas): `"instance"`, `"interpretation"`, `"abstraction"`, `"facet"`, `"domain"`, `"idea"`

**Computed composites**:
- `"ladder"` — `instance → interpretation → abstraction` (default)
- `"idea_rungs"` — `idea → interpretation → abstraction`

**Valence tags** (when `CategoriesConfig.include_valence=True`): prepends `[+]`, `[-]`, or `[0]` to each label.

## Concurrency & Rate Limiting

Bootstrap pattern shared across P1–P4a:
1. Fetch real rate limits from API response headers
2. Run 3 probe calls to measure avg latency and token usage
3. Compute optimal concurrency via Little's Law
4. Shared `asyncio.Semaphore` + `AsyncLimiter` for all LLM calls in QualitativeResearcher

Code assignment (`CodeAssigner`) has its own rate-limiting stack:
1. ConcurrencyGate: completion-based ramp from 50% → 90% of Little's Law
2. TokenBucket: TPM safety rail with reconciliation
3. AsyncLimiter: PID-adjusted RPM arrival rate
4. Circuit breaker: monitors timeout rate, adjusts concurrency

## Configuration

### CategoriesConfig (config_classNcoder_exp.py)

| Setting | Default | Purpose |
|---------|---------|---------|
| `label_source` | `"ladder"` | Which fields to use as observations |
| `include_valence` | `False` | Prepend valence tags |
| `qr_model_p1` | `gpt-4.1-mini` | P1: Facet Discovery |
| `qr_model_p1_5` | `gpt-4.1` | P1.5: Facet Consolidation |
| `qr_model_p2` | `gpt-4.1-nano` | P2: Facet Assignment |
| `qr_model_p3` | `gpt-4.1-mini` | P3: Attribute Discovery |
| `qr_model_p4` | `gpt-4.1` | P4: Code Generation + P4.5 Consolidation |
| `batch_size_min/max` | 100/150 | P1 chunk sizing |
| `p3_batch_size_min/max` | 100/150 | P3 chunk sizing |
| `facet_assignment_batch_size` | 10 | P2 ideas per LLM call |

### AssignmentConfig (config_classNcoder_exp.py)

| Setting | Default | Purpose |
|---------|---------|---------|
| `assignment_model` | `gpt-4.1-mini` | Code assignment model |
| `use_embedding_prefilter` | `True` | Embedding-based code narrowing |
| `embedding_top_n` | 5 | Codes per idea after pre-filter |
| `embedding_model` | `text-embedding-3-large` | Embedding model |
| `include_other_category` | `True` | Add catch-all "Other" code |

## Files

| File | Purpose |
|------|---------|
| `qualitative_researcher.py` | Main orchestrator: P1–P4.5 pipeline with async concurrency |
| `prompts_exp.py` | All prompt builders + Pydantic response models (DiscoveredFacet, DiscoveredAttribute, CodeFromAttributes, ConsolidatedCode, CodeAttributeAssignment, etc.) |
| `code_assignment.py` | Code assignment orchestrator (Stage 2) with queue-based workers, embedding pre-filter, 4-layer rate limiting |
| `domain_discoverer.py` | Partition ideas by domain, collect unique labels per partition (`DomainDiscoverer`, `PartitionLabelMapping`) |
| `partition_labels.py` | Label extraction/formatting (ladder composites, valence tags, prefixes) |
| `config_classNcoder_exp.py` | Configuration (`CategoriesConfig`, `AssignmentConfig`) |
| `models_exp.py` | Data models (`DomainSet`, `CodingResultsCache`, `CodeAssignedModel`, `CodeAssignedSubmodel`) |
| `embedding_matcher.py` | Embedding-based pre-filter for code assignment |
| `run_experiment.py` | Experiment runner: full pipeline + assignment-only mode, caching, prompt capture |
| `view_assignments.py` | View/analyze assignment results |
| `view_ideas.py` | Full idea-level view grouped by code |
| `debug_assignment_prompt.py` | Debug helper for inspecting assignment prompts |
| `debug_generation_prompts.py` | Debug helper for inspecting pipeline prompts |
| `debug_lookup.py` | Debug helper for ID lookups |

## Source of Truth

- **Dimension definitions**: `step_3_ideaExtractor/dimension_data.py` — `DIMENSIONS` registry
- **Taxonomy logic**: `step_3_ideaExtractor/taxonomy_logic.md` — conceptual mapping per dimension
- **Step 3 output**: cached `ExtractionMetadata` + `IdeasExtractedModel[]`
