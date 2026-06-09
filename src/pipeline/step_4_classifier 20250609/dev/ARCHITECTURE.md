# Step 4 — Architecture

## Taxonomy

Dimension → Domain → Facet → Attribute.

| Level | Name | Source | Role |
|-------|------|--------|------|
| L1 | Dimension | Step 3 | Information type (selected once per dataset) |
| L2 | Domain | Step 3 | Thematic subject (assigned per idea) |
| L3 | Facet | **Step 4 P1-P3** | Analytical lens — how the domain is examined |
| L4 | Attribute | **Step 4 P4-P7** | Observable property — concrete characteristic |

**Step 4** completes L3 (Facet) and L4 (Attribute) via an inductive discovery pipeline operating per-domain in parallel.

## Design Intent

- **Domain-partitioned processing**: each domain (from step 3) is processed independently, enabling concurrent discovery tailored to each domain's semantics
- **SmoothRequester for all LLM calls**: every phase (P1-P7) dispatches through `SmoothRequester.process_all()` with prepare/parse/fallback callbacks. Rate pacing, retry, and adaptive concurrency are handled internally.
- **Hierarchical consolidation**: raw discoveries from overlapping chunks are merged via iterative multi-round SmoothRequester calls when chunk counts exceed thresholds
- **Dimension-shaped taxonomy**: the selected dimension's `prompt_rules` flow into every prompt, adapting facet/attribute semantics to the information type
- **Adaptive batching**: observations are chunked with configurable overlap to balance prompt context with discovery quality
- **Auto-assignment**: single-facet domains and single-attribute facets skip the LLM assignment call entirely

## Pipeline Overview

```
Input: Step 3 IdeasExtractedModel[] + ExtractionMetadata (dimension, domains, specifiers)

Stage 1 — PARTITION DISCOVERY (DomainDiscoverer):
  Group ideas by domain → collect unique observations per domain
  → DomainSet + Dict[domain → PartitionLabelMapping]

Stage 2 — TAXONOMY CLASSIFICATION (TaxonomyClassifier):

  Phase 1: Facet Discovery + Consolidation
    P1: Chunk observations → discover facets per chunk (SmoothRequester, concurrent)
    P2: Consolidate chunk facets → single MECE facet list per domain (SmoothRequester, concurrent)

  Phase 2: Facet Assignment + Valence
    P3: Batch ideas → assign each to one facet + valence (SmoothRequester, concurrent)
    Auto-assign domains with 1 facet (skip LLM, keep step 3 valence)

  Phase 3: Attribute Discovery + Consolidation
    P4: Per (domain, facet): chunk observations → discover attributes (SmoothRequester, concurrent)
    P5: Consolidate chunk attributes → single MECE list per facet (SmoothRequester, concurrent)

  Phase 4: Attribute Assignment + Valence
    P6: Batch ideas per facet → assign each to one attribute + valence (SmoothRequester, concurrent)
    Auto-assign facets with 1 attribute (skip LLM, keep P3 valence)

  Phase 5: Cross-Facet Attribute Consolidation
    P7: Detect and merge duplicate/overlapping attributes across facet boundaries
    → remap attribute assignments (SmoothRequester, concurrent per domain)

  Phase 5.5: Valence-Neutral Attribute Merge (valence_consolidator.py)
    P7.5: Detect attribute pairs within a facet that differ only in valence;
          merge safe ones into one descriptive attribute (LLM-renamed, fallback)
    → valence stays in the per-idea valence field; overwrites the P7 cache

  Phase 6: Cross-Domain Attribute Consolidation (cross_domain_consolidator.py)
    P8: Embed all ideas → attribute centroids → seriate into 1D order → sliding
        windows of ~10 attributes → LLM merge per window (SmoothRequester, concurrent)
    → remap attribute assignments across domains; verify consistency

Output (P8 overwrites the P7 output in-place — same cache keys):
  - TaxonomyResultsCache (metadata: partition_set, per-domain facets/attributes/assignments/valence)
  - List[TaxonomyClassifiedModel] (growing model: per-idea facet, attribute, partition_name, domain, valence)
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
| P2 | `build_facet_consolidation_prompt()` | `FacetConsolidatedResponse` (scratchpad + List[DiscoveredFacet]) | Per-domain merge |
| P3 | `build_facet_assignment_prompt_single()` | `FacetAssignmentResult` (facet_id, confidence, valence) | 1 idea/task |
| P4 | `build_attribute_discovery_prompt()` | `AttributeDiscoveryResult` (scratchpad + List[DiscoveredAttribute]) | Per-facet, chunked |
| P5 | `build_attribute_chunk_consolidation_prompt()` | `AttributeChunkConsolidatedResponse` (scratchpad + List[DiscoveredAttribute]) | Per-facet merge |
| P6 | `build_attribute_assignment_prompt_single()` | `AttributeAssignmentResult` (attribute_id, confidence, valence) | 1 idea/task |
| P7 | `build_attribute_consolidation_prompt()` | `AttributeConsolidatedResponse` (scratchpad + List[ConsolidatedAttribute] with source_attributes) | Cross-facet dedup |
| P8 | `build_cross_domain_consolidation_prompt()` | `CrossDomainConsolidatedResponse` (scratchpad + List[CrossDomainConsolidatedAttribute] with source_attributes, parent_domain, parent_facet) | Cross-domain dedup (in `cross_domain_consolidator.py`) |
| P7.5 | `build_valence_neutral_rename_prompt()` | `ValenceNeutralRenameResponse` (List[ValenceNeutralAttribute]) | Valence-split merge rename (in `valence_consolidator.py`) |

Note: `build_facet_discovery_prompt()` (P1) also takes the domain's `boundary_test` + `exclusions` and renders them in the `<taxonomy_domain>` block; the P8 prompt lists each domain's exclusions and adds a RESPECT DOMAIN BOUNDARIES rule.

## Label Sources & Valence

**Configurable via `CategoriesConfig.label_source`** (drives both discovery observations and assignment idea text):
- `"idea"` — idea text (includes template prefix); the dataclass default
- `"instance"`, `"interpretation"`, `"abstraction"` — individual ladder rungs
- `"ladder"` — computed: instance → interpretation → abstraction
- `"idea_interpretation"` — computed: idea → interpretation

The active pipeline overrides the default in `run_classifier.py` — currently **`"ladder"`** (full 3-rung context).

**`label_prefix`**: optional static prefix prepended to all observations.

**Valence assignment**: P3 and P6 independently assign valence (+, -, 0) as evaluative direction relative to the facet (P3) or attribute (P6). The growing model carries the most precise: P6 valence > P3 valence > step 3 valence. Auto-assigned ideas keep the previous level's valence. Valence is not emotional sentiment but evaluative direction (e.g., "meeting expectations" vs "failing expectations").

Formatting handled by `partition_labels.py`: `format_label()`, `collect_unique_labels()`.

## Concurrency & Rate Limiting

### SmoothRequester (all phases)

All LLM calls go through `SmoothRequester.process_all()`. Each phase creates its own SR instance with:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `known_limits` | `self._fetched_limits` | Skip probe — limits fetched once at startup via `llm.fetch_rate_limits()` |
| `show_setup` | `False` | Suppress setup block — pipeline prints its own |
| `quiet` | `True` | Suppress all SR output — caller builds verbose report |
| `default_timeout` | 60.0s (discovery/consolidation), 10.0s (assignment) | Cold-start timeout before empirical data available |

### Adaptive Timeout Multiplier

`min(6, round(ln(num_tasks) + 1))` — scales with task count:
- 1 task → 1x P50 (estimate is the actual latency)
- 10 tasks → 3x P50
- 50 tasks → 5x P50
- 100+ tasks → 6x P50 (need headroom for outlier tail)

### Empirical Stats Cache

After each phase, SmoothRequester persists P50 latency, avg tokens, and header availability to `model_perf_stats.json` keyed by `model:phase_key:dataset_key`. On subsequent runs, these seed the timeout and token estimates.

**Empirical capacity** (server concurrency limit) is only saved when the concurrency controller actually found a ceiling — i.e., the state machine exited `RAMP_UP` and transitioned to `STEADY`, `BACKOFF`, or `RECOVER`. For small-task phases (e.g., 8 P1 tasks), the controller typically never detects pressure and stays in `RAMP_UP`, so no capacity is saved and the next run cold-starts at `COLD_START_CAP` (50) again. This prevents a self-reinforcing feedback loop where a bad calibration from too few tasks would cap concurrency on every subsequent run.

### Multi-Round Consolidation (P2, P5)

When chunk count exceeds thresholds:
1. Pre-group chunks into sub-groups respecting `consolidation_max_chunks_per_call` (6) and `consolidation_max_items_per_call` (150)
2. Run round 1 via `process_all()` — one task per group
3. Check if intermediate results fit in one call
4. If yes, run round 2 via `process_all()` for final merge

### Adaptive Batching

**P1 (facet discovery)**: observations chunked into `target_batches` (default 6) with overlap (`chunk_overlap` default 0.2). Batch size clamped to `batch_size_min`–`batch_size_max` (100–150).

**P4 (attribute discovery)**: same logic with `p4_*` config variants (target 5 chunks, same size range).

## Cross-Domain Consolidation (P8)

P7 consolidates attributes *within* each domain. P8 (`cross_domain_consolidator.py`) consolidates attributes *across* domains — too many attributes (35-50+) for a single LLM call, so embeddings find what's semantically close and chunk it into digestible windows. Five stages:

1. **Embed** — embed all ideas (via `SharedEmbedder`), compute a centroid per attribute (text format from `p8_code_source`).
2. **Similarity** — pairwise cosine similarity between centroids (`p8_similarity_threshold` noise floor).
3. **Seriate** — agglomerative clustering → 1D ordering where similar attributes are adjacent.
4. **Sliding window** — windows of `p8_window_size` (10) with `p8_window_overlap` (2); one concurrent LLM merge call per window (SmoothRequester, `phase_key="step4_p8_cross_domain_consolidation"`).
5. **Remap** — apply merge decisions to cache + growing model, then `_verify_consistency()`.

**Remap invariants (post-bugfix):**
- The merge map is keyed by `(domain, attribute_name)`, not bare name — names are not unique across domains. LLM source names are resolved against the attributes present in each window; unknown names are ignored.
- On a remap, `attribute_valence`/`attribute_confidence` (and facet valence/confidence for cross-domain moves) move with the idea; the growing model's `partition_name` and `domain` are both updated.
- "Merge wins, first window takes precedence" (seriation order).
- Zero idea loss; `_verify_consistency()` checks idea count, no dropped valence/confidence, and no orphan assignments, printing `P8 consistency: OK` or warnings.
- Skipped when fewer than 2 attributes exist.

See [CONSOLIDATION_LOGIC.md](CONSOLIDATION_LOGIC.md) for the full algorithm reference.

## Valence-Neutral Merge (P7.5)

Attributes must be descriptive; the `+/-/0` direction lives in the per-idea `valence` field. The P4/P5/P7 prompts forbid splitting a concept by evaluative direction, and `valence_consolidator.py` (P7.5, after P7 / before P8) is the safety net:

1. **Detect** (deterministic, shared with `view_valence_split.py`): attribute pairs within a facet with near-identical labels (token-set / char similarity) AND opposite valence skew (one mostly `+`, the other mostly not-`+`). Both signals are required — label-similarity alone over-flags descriptive near-duplicates; valence-skew alone over-flags homogeneous solo attributes.
2. **Merge** the safe pairs (auto-safe similarity + a clean single-token diff): the neutral merged name/description comes from a small LLM call (`classifier_p7` model, `build_valence_neutral_rename_prompt`) with a deterministic single-token fallback. Idea reassignment is deterministic and preserves valence/confidence.

Overwrites the `taxonomy` / `taxonomy_classified` cache in place; cost-tracked as `p7_5_valence_merge`. See [WORK_VALENCE_NEUTRALITY.md](WORK_VALENCE_NEUTRALITY.md).

## Domain boundary_test & exclusions

Step 3 persists `boundary_test` (str) and `exclusions` (list[str]) per domain in `ExtractionMetadata.domains`. `domain_discoverer.py` reads the persisted values (fallback to derived for old caches) into `DomainDescription` (which now carries `exclusions`). They are injected into **P1 facet discovery** (so facets stay within the domain) and the **P8** cross-domain prompt (so merges respect boundaries).

## Configuration

**`CategoriesConfig`** (dataclass in `config_classifier.py`):

| Group | Fields | Defaults |
|-------|--------|----------|
| Label extraction | `label_source`, `label_prefix` | `"idea"`, `""` |
| Models | `qr_model_p1` … `qr_model_p8` (8 keys) | Per-phase from `get_step_model()`: p1/p4 mini, p2/p5/p7/p8 default, p3/p6 nano |
| Temperature | `qr_temperature` | 0.3 |
| Token budgets | `qr_max_tokens_facet_discovery`, `_facet_assignment`, `_attribute_discovery`, `_cross_domain` | 4000 each, P8 16000 |
| P1 batching | `batch_size_min/max`, `target_batches`, `chunk_overlap` | 100/150, 6, 0.2 |
| P4 batching | `p4_batch_size_min/max`, `p4_target_batches`, `p4_chunk_overlap` | 100/150, 5, 0.2 |
| Consolidation | `consolidation_max_chunks_per_call`, `_max_items_per_call`, `_max_rounds` | 6, 150, 5 |
| P8 (cross-domain) | `p8_code_source`, `p8_embedding_model`, `p8_window_size`, `p8_window_overlap`, `p8_similarity_threshold` | `instance_interpretation`, `text-embedding-3-large`, 10, 2, 0.6 |
