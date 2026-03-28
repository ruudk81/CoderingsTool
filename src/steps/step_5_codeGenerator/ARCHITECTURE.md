# Step 5 Codebook Generator — Architecture & Data Flow

## Design Intent

MECE codebook generation from taxonomy. Step 5 transforms per-domain taxonomy results (facets + attributes from step 4) into a final, parsimonious codebook via two sequential LLM phases:

- **P8**: per-domain code derivation from attributes (parallel)
- **P9**: cross-domain consolidation into final MECE codebook (single call)

Key design choices:
- **Attribute-grounded codes**: every code traces back to discovered attributes, ensuring data groundedness
- **Prevalence weighting**: idea counts per attribute guide code granularity (high-prevalence = core structure, low-prevalence = abstracted upward)
- **Valence sensitivity**: positive/negative/neutral codes kept separate
- **Dimension-specific diagnostics**: each code includes a diagnostic test completing the dimension's diagnostic stem
- **Constraint-based response model**: P8 uses dynamic Pydantic models with `source_attributes` constrained to an enum of valid attribute names

## Pipeline Overview

```
Input: TaxonomyResultsCache (from step 4, prefix 005)
       + ExtractionMetadata (from step 3, prefix 004)

P8 — Code Generation from Attributes (per-domain, PARALLEL):
  For each domain:
    Attributes + idea frequencies + domain context
      → [Prompt P8] derive codes from attributes
      → List[CodeFromAttributes] per domain

P9 — Codebook Consolidation (cross-domain, SINGLE CALL):
  All P8 codes + provenance + frequencies
    → [Prompt P9] 8-step MECE workflow
    → List[ConsolidatedCode] (final codebook)

Output: CodingResultsCache (metadata: partition_set, partition_results, raw_codes)
        Saved as step="mece_codes", prefix 006
```

## Input Contract (from Step 4)

Step 5 loads `TaxonomyResultsCache` metadata via `load_taxonomy_cache()`:

- **`partition_set: DomainSet`** — domain partition definitions
- **`partition_results: Dict[str, DomainResultModel]`** — per-domain:
  - `facets` — discovered facets (P1-P2 output)
  - `attributes` — per-facet attributes (P4-P5 output)
  - `facet_assignments` — {idea_id → facet_name} for frequency counting
  - `attribute_assignments` — {idea_id → attribute_name} for frequency counting

The runner reconstructs `TaxonomyResult` and typed `DiscoveredFacet`/`DiscoveredAttribute` objects from cached dicts.

## Prompt Builders & Response Models

All in `prompts_codeGenerator.py`:

| Phase | Builder | Response Model | Notes |
|-------|---------|---------------|-------|
| P8 | `build_code_from_attributes_prompt()` | `CodeGenerationFromAttributesResult` (scratchpad + List[CodeFromAttributes]) | Constrained source_attributes enum |
| P9 | `build_codebook_consolidation_prompt()` | `CodebookConsolidationResult` (scratchpad + List[ConsolidatedCode]) | 8-step MECE workflow |

### P8: Code Generation from Attributes

**Input per domain**: attribute inventory (per-facet, with idea counts), domain definition, excluded domains list.

**Response model**:
```
CodeFromAttributes:
  code_name (3-5 word noun phrase), definition (1-2 sentences),
  typical_indicators[], source_attributes[] (enum-constrained to valid attribute names)
```

**Dynamic constraint**: `source_attributes` is built as `Literal[tuple(attribute_names)]` via `create_model()`, ensuring codes can only reference attributes that exist in that domain.

**Rules**: phenomenon-level (not evaluative), dimension-specific, prevalence-weighted, MECE within domain.

### P9: Codebook Consolidation

**Input**: all P8 codes with domain provenance tags + frequency counts.

**8-step mandatory workflow**:
1. Valence separation (positive/negative/neutral split)
2. Aggressive merging within clusters
3. Mechanism purity check (values vs. functional vs. perception vs. cause)
4. Neighbor stress test (coder hesitation between codes)
5. One-sentence coverage test
6. Non-redundancy kill step
7. Final diagnostic uniqueness check (dimension-specific diagnostic stem)
8. Prevalence weighting & structural balancing

**Response model**:
```
ConsolidatedCode:
  code_name, definition, diagnostic_test, valence (positive/negative/neutral),
  typical_indicators[], source_attributes[] (all merged origins)
```

## Key Data Structures

**`CodebookResult`** (dataclass, internal):
```
codes: List[ConsolidatedCode]
codebook_narrative: str (concatenated scratchpads from P8 + P9)
```

**`CodingResultsCache`** (Pydantic, cached):
```
partition_set: DomainSet
partition_results: Dict[str, DomainResultModel]
label_counts, label_source
total_categories: int
raw_codes: List[Dict] (ConsolidatedCode dicts, reconstructable via ConsolidatedCode(**d))
```

## Concurrency & Rate Limiting

Step 5 uses the same 4-layer rate limiting stack as step 4:

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | ConcurrencyGate + ConcurrencyRamp | Completion-based concurrency |
| 2 | TokenBucket | TPM enforcement |
| 3 | AsyncLimiter | RPM pacing |
| 4 | LatencyTracker + ConcurrencyCircuitBreaker | Adaptive timeout + pressure detection |

**P8**: per-domain parallelization (one task per domain, all concurrent within gate). Full 4-layer stack if ≥20 tasks.

**P9**: single cross-domain call (no parallelization needed).

Background monitor with warm-up calibration and PID adjustment (same pattern as step 4).

### Error Handling

Both P8 and P9 use 2-attempt retry:
- P8: returns empty `CodeGenerationFromAttributesResult` on final failure
- P9: re-raises on final failure (raw P8 codes pass through)

## Configuration

**`CodebookConfig`** (dataclass in `config_steps/config_codeGenerator.py`):

| Field | Default | Purpose |
|-------|---------|---------|
| `model_p8` | `get_step_model("codegen_p8")` | Model for per-domain code generation |
| `model_p9` | `get_step_model("codegen_p9")` | Model for cross-domain consolidation |
| `temperature` | 0.3 | LLM temperature |
| `max_tokens_code_from_attributes` | 16000 | Token budget for P8 |
| `max_tokens_codebook_consolidation` | 16000 | Token budget for P9 |
| `consolidation_max_chunks_per_call` | 6 | For future hierarchical P9 batching |
| `consolidation_max_items_per_call` | 150 | For future hierarchical P9 batching |
| `ramp_config` | `ClassifierRampConfig()` | 4-layer rate limiting config |

## Data Flow Summary

```
Step 3 cache (prefix 004)
  └─ ExtractionMetadata → survey_question, language, dimension, specifiers

Step 4 cache (prefix 005)
  └─ TaxonomyResultsCache → partition_set, partition_results (facets, attributes, assignments)

        ↓ CodebookGenerator.generate()

Step 5 cache (prefix 006)
  └─ CodingResultsCache → partition_set, partition_results, raw_codes

        ↓ consumed by Step 6 (code assignment)
```

## Files

- **`codebook_generator.py`** — CodebookGenerator: P8-P9 orchestration with 4-layer rate limiting
- **`prompts_codeGenerator.py`** — prompt builders + Pydantic response models (P8-P9), dynamic constraint model
- **`models_codeGenerator.py`** — CodingResultsCache (output model for caching)
- **`run_experiment.py`** — orchestrator: loads step 3/4 cache, runs generator, saves results
- **`debug_codebook_prompts.py`** — inspect captured codebook prompts
- **`debug_lookup.py`** — inspect code-to-attribute mappings
- **`view_codebook.py`** — view final codebook with codes, definitions, diagnostics
