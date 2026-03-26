# Step 4 Taxonomy Classifier — Architecture & Data Flow

## Taxonomy

Dimension → Domain → Facet → Attribute.

| Level | Name | Source | Role |
|-------|------|--------|------|
| L1 | Dimension | Step 3 | Information type (selected once per dataset) |
| L2 | Domain | Step 3 | Thematic subject (assigned per idea) |
| L3 | Facet | **Step 4 P1-P3** | Analytical lens — how the domain is examined |
| L4 | Attribute | **Step 4 P4-P7** | Observable property — concrete characteristic |

**Step 4** completes L3 (Facet) and L4 (Attribute) via a 7-phase inductive discovery pipeline operating per-domain in parallel.

## Design Intent

- **Domain-partitioned processing**: each domain (from step 3) is processed independently through all 7 phases, enabling concurrent discovery tailored to each domain's semantics
- **Hierarchical consolidation**: raw discoveries from overlapping chunks are merged into coherent, non-overlapping MECE sets
- **Dimension-shaped taxonomy**: the selected dimension's `prompt_rules` flow into every prompt, adapting facet/attribute semantics to the information type
- **Adaptive batching**: observations are chunked with configurable overlap to balance prompt context with discovery quality
- **4-layer rate limiting**: completion-based ramping with warm-up calibration and circuit breaker protection

## Pipeline Overview

```
Input: Step 3 IdeasExtractedModel[] + ExtractionMetadata (dimension, domains, specifiers)

Stage 1 — PARTITION DISCOVERY (DomainDiscoverer):
  Group ideas by domain → collect unique observations per domain
  → DomainSet + Dict[domain → PartitionLabelMapping]

Stage 2 — TAXONOMY CLASSIFICATION (TaxonomyClassifier, per domain):

  P1: Facet Discovery (chunked, concurrent)
    Chunk observations → discover facets per chunk

  P2: Facet Consolidation (hierarchical merge)
    Merge chunk-level facets → single MECE facet list per domain

  P3: Facet Assignment (batched, concurrent)
    Batch ideas → assign each idea to one discovered facet
    → {idea_id → facet_name}

  P4: Attribute Discovery (per facet, chunked, concurrent)
    For each (domain, facet): chunk assigned ideas' observations → discover attributes

  P5: Attribute Chunk Consolidation (hierarchical merge)
    Merge chunk-level attributes → single MECE attribute list per facet

  P6: Attribute Assignment (per facet, batched, concurrent)
    Batch ideas per facet → assign each to one discovered attribute
    → {idea_id → attribute_name}

  P7: Cross-Facet Attribute Consolidation (per domain)
    Detect and merge duplicate/overlapping attributes across facet boundaries
    → remap attribute assignments

Output:
  - TaxonomyResultsCache (metadata: partition_set, per-domain facets/attributes/assignments)
  - List[TaxonomyClassifiedModel] (growing model: per-idea facet, attribute, partition_name)
```

## Dimension-Specific Prompt Injection

When step 3 selects a primary dimension (e.g., `PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS`), its `DimensionDefinition` from `dimension_data.py` is loaded via the cached dimension key in `ExtractionMetadata`. The definition's `prompt_rules` provide:

- `facet_instruction` — dimension-specific facet semantics for P1/P2
- `attribute_instruction` — dimension-specific attribute semantics for P4/P5
- `domain_diagnostic`, `facet_diagnostic` — diagnostic questions

These are injected into every prompt builder via `build_dimension_context_block()`, so all discovery and assignment prompts are dynamically shaped by the selected dimension.

## Prompt Builders & Response Models

All in `prompts_classifier.py`:

| Phase | Builder | Response Model | Notes |
|-------|---------|---------------|-------|
| P1 | `build_facet_discovery_prompt()` | `FacetDiscoveryResult` (scratchpad + List[DiscoveredFacet]) | Per-chunk, with excluded_domains |
| P2 | `build_facet_consolidation_prompt()` | `FacetConsolidatedResponse` (scratchpad + List[DiscoveredFacet]) | Hierarchical merge |
| P3 | `build_facet_assignment_prompt()` | `FacetAssignmentBatch` (List[FacetAssignment]: idea_id, facet_id, confidence, rationale) | 10 ideas/batch |
| P4 | `build_attribute_discovery_prompt()` | `AttributeDiscoveryResult` (scratchpad + List[DiscoveredAttribute]) | Per-facet, chunked |
| P5 | `build_attribute_chunk_consolidation_prompt()` | `AttributeChunkConsolidatedResponse` (scratchpad + List[DiscoveredAttribute]) | Hierarchical merge |
| P6 | `build_attribute_assignment_prompt()` | `AttributeAssignmentBatch` (List[AttributeAssignment]: idea_id, attribute_id, confidence, rationale) | 10 ideas/batch |
| P7 | `build_attribute_consolidation_prompt()` | `AttributeConsolidatedResponse` (scratchpad + List[ConsolidatedAttribute] with source_attributes) | Cross-facet dedup |

## Data Flow

### From Step 3 (input)

- **`List[IdeasExtractedModel]`**: per-response ideas with domain (L2), partial facet hint, abstraction ladder
- **`ExtractionMetadata`**: primary_dimension, primary_dimension_description, domains, lang, sector, entity, topic, perspective, intent, var_lab

### Step 4 Output (cached)

- **Metadata** (`TaxonomyResultsCache`, step="taxonomy", prefix 005):
  - `partition_set: DomainSet` — domain partition definitions
  - `partition_results: Dict[str, DomainResultModel]` — per-domain: facets, facet_assignments, attributes, attribute_assignments
  - `label_counts`, `label_source`

- **Growing model** (`List[TaxonomyClassifiedModel]`, step="taxonomy_classified", prefix 005):
  - Per-idea: facet (L3), attribute (L4), partition_name populated from taxonomy results
  - Consumed by step 6 (code assignment)

### To Step 5 (output contract)

Step 5 loads `TaxonomyResultsCache` metadata. It needs:
- `partition_set` — domain partition definitions for P8 per-domain code generation
- `partition_results` — per-domain facets, attributes, and assignment counts for codebook derivation

## Key Data Structures

**`PartitionLabelMapping`** (dataclass):
```
partition_name, partition (DomainDescription), labels (unique observations),
label_count, label_domains, ideas (IdeasExtractedSubmodel objects)
```

**`TaxonomyResult`** (dataclass, internal):
```
partition_n_labels, partition_n_batches,
partition_facets: {domain → [DiscoveredFacet]},
partition_assignments: {domain → {idea_id → facet_name}},
partition_attributes: {domain → {facet → [DiscoveredAttribute]}},
attribute_assignments: {idea_id → attribute_name}
```

**`DiscoveredFacet`** / **`DiscoveredAttribute`** (Pydantic):
```
*_name, *_description, example_observations[]
```

**`TaxonomyClassifiedSubmodel`** (extends IdeasExtractedSubmodel):
```
partition_name (new), facet (populated), attribute (populated)
```

## Label Sources & Valence

**Configurable via `CategoriesConfig.label_source`**:
- `"idea"` (default) — idea text (includes template prefix)
- `"instance"`, `"interpretation"`, `"abstraction"` — individual ladder rungs
- `"ladder"` — computed: instance → interpretation → abstraction
- `"idea_rungs"` — computed: idea → interpretation → abstraction

**`label_prefix`**: optional static prefix prepended to all observations.

**`include_valence`**: if True, prepends `[+]`, `[-]`, or `[0]` tag to each observation.

Formatting handled by `partition_labels.py`: `format_label()`, `collect_unique_labels()`.

## Concurrency & Rate Limiting

### 4-Layer Stack (P1, P3, P4, P6 — large phases)

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | ConcurrencyGate + ConcurrencyRamp | Completion-based concurrency control; starts at 50% of Little's Law, ramps to 90% |
| 2 | TokenBucket | TPM enforcement; wait-and-acquire with reconciliation |
| 3 | AsyncLimiter | RPM pacing; arrival rate from Little's Law |
| 4 | LatencyTracker + ConcurrencyCircuitBreaker | Adaptive timeout (P95 × margin); backs off on sustained timeout pressure |

**Light mode** (P2, P5, P7 — small phases): default semaphore + AsyncLimiter only. No TokenBucket, LatencyTracker, or CircuitBreaker.

### Per-Phase State (`PhaseRampState`)

Each large phase gets its own `PhaseRampState` with all 4 layers, warm-up tracking (actual_total_tokens deque, calibration flag), and completion/timeout counters.

### Background Monitor (`_phase_monitor()`)

Runs per-phase as a background coroutine:
- Every 0.5s: circuit breaker check, ramp feed, warm-up calibration check
- Every 20s: PID-based arrival rate adjustment (asymmetric: aggressive speed-up kp=0.4, gentle slow-down kp=0.2)
- Every 2s: progress reporting (concurrency, TPM%, RPM%, latency, timeouts, CB state)

### Warm-Up Calibration (one-shot per large phase)

After 15-30 completions: recalculates Little's Law from measured latency (P10) and token counts, recalibrates ramp target, recomputes arrival rate, and resets PID state.

### Adaptive Batching

**P1 (facet discovery)**: observations chunked into `target_batches` (default 6) with overlap (`chunk_overlap` default 0.2). Batch size clamped to `batch_size_min`–`batch_size_max` (100–150).

**P4 (attribute discovery)**: same logic with `p4_*` config variants (target 5 chunks, same size range).

### Hierarchical Consolidation (`_hierarchical_consolidate()`)

Used by P2, P5, P7. Rules:
1. If ≤ `consolidation_max_chunks_per_call` (6) chunks: merge all in one call
2. If total items ≤ `consolidation_max_items_per_call` (150): merge all in one call
3. Otherwise: group into batches, consolidate each, recurse (max `consolidation_max_rounds` = 5 rounds)

## Configuration

**`CategoriesConfig`** (dataclass in `config_steps/config_classifier.py`):

| Group | Fields | Defaults |
|-------|--------|----------|
| Label extraction | `label_source`, `label_prefix`, `include_valence` | `"idea"`, `""`, `False` |
| Models | `qr_model_p1` ... `qr_model_p7` | Per-phase from `get_step_model()` |
| Temperature | `qr_temperature` | 0.3 |
| Token budgets | `qr_max_tokens_facet_discovery`, `_facet_assignment`, `_attribute_discovery` | 4000 each |
| P1 batching | `batch_size_min/max`, `target_batches`, `chunk_overlap` | 100/150, 6, 0.2 |
| P4 batching | `p4_batch_size_min/max`, `p4_target_batches`, `p4_chunk_overlap` | 100/150, 5, 0.2 |
| Consolidation | `consolidation_max_chunks_per_call`, `_max_items_per_call`, `_max_rounds` | 6, 150, 5 |
| Rate limiting | `ramp_config: ClassifierRampConfig` | See below |

**`ClassifierRampConfig`**:

| Field | Default | Purpose |
|-------|---------|---------|
| `estimated_latency_seconds` | 10.0 | Conservative latency for Little's Law |
| `estimated_avg_tokens` | 3000 | Conservative per-call tokens |
| `start_fraction` / `target_fraction` | 0.50 / 0.90 | Ramp from 50% to 90% of ceiling |
| `min_initial` | 5 | Concurrency floor |
| `warm_up_sample_min/max` | 15 / 30 | Completions before recalibration |
| `circuit_breaker_enabled` | True | Enable circuit breaker |
| `circuit_breaker_min_tasks` | 20 | Skip CB for small phases |
| `timeout_floor_seconds` | 60.0 | Cold-start timeout floor |

## Files

- **`classifier.py`** — TaxonomyClassifier: 7-phase pipeline with 4-layer concurrency (~2400 lines)
- **`domain_discoverer.py`** — DomainDiscoverer: partition ideas by domain, collect observations
- **`partition_labels.py`** — format_label(), collect_unique_labels() utilities
- **`prompts_classifier.py`** — prompt builders + Pydantic response models (P1-P7)
- **`models_classifier.py`** — DomainDescription, DomainSet, TaxonomyClassifiedModel, cache models
- **`run_experiment.py`** — orchestrator: loads step 3 cache, runs classifier, saves results + growing model
- **`debug_taxonomy_prompts.py`** — inspect captured taxonomy prompts
- **`view_assignments_attributes.py`** — view per-idea facet/attribute assignments
- **`view_taxonomy.py`** — view discovered taxonomy structure
