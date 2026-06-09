# Step 4 — Taxonomy Classifier

## Purpose
Completes the taxonomy hierarchy by discovering and assigning L3 (Facet), L4 (Attribute), and valence. Runs a 6-phase pipeline (internally P1-P8) partitioned by domain: discover facets, consolidate, assign to ideas (+ valence), discover attributes, consolidate, assign to ideas (+ valence), cross-facet attribute consolidation (P7), and cross-domain attribute consolidation (P8).

## Key Files
- `run_classifier.py` — step runner / orchestrator
- `classifier.py` — `TaxonomyClassifier` class: P1-P7 async pipeline (~1950 lines)
- `cross_domain_consolidator.py` — `CrossDomainConsolidator` class: P8 (embedding → seriation → sliding-window LLM merge → remap)
- `run_consolidator.py` — standalone P8 runner (re-runs P8 on cached P7 output for fine-tuning, no P1-P7 rerun)
- `valence_consolidator.py` — `ValenceConsolidator` class (P7.5) + shared `detect_valence_splits()`: merge attribute pairs that differ only in valence
- `view_valence_split.py` — read-only detector for valence-split attribute pairs
- `domain_discoverer.py` — `DomainDiscoverer`: partitions ideas by domain, collects observations
- `partition_labels.py` — label formatting utilities
- `prompts_classifier.py` — prompt builders for P1-P8 + Pydantic response models
- `models_classifier.py` — `TaxonomyClassifiedModel`, `TaxonomyResultsCache`, `DomainSet`, `DomainResultModel`
- `config_classifier.py` — `CategoriesConfig`

## Input / Output Contract
- **Input**: `List[IdeasExtractedModel]` + `ExtractionMetadata` from cache key `extracted_ideas/{variable_key}`
- **Output**:
  - `List[TaxonomyClassifiedModel]` — ideas enriched with `partition_name`, `facet`, `attribute`, `valence`
  - `TaxonomyResultsCache` cached at `taxonomy/{variable_key}` — full taxonomy structure (DomainSet + per-domain results including facet/attribute valence)

## LLM Usage
- **Models**: via `get_step_model()` — 8 model config keys in `CategoriesConfig` (`qr_model_p1`–`qr_model_p8`)
  - P1/P4 (discovery): mini tier; P2/P5/P7/P8 (consolidation): default tier
  - P3 (facet assignment), P6 (attribute assignment): nano tier
- **Prompt file**: `prompts_classifier.py` (8 phase-specific builders)
- **Response models**: phase-specific Pydantic models (instructor-based)
- **Temperature**: 0.3 (configurable via `CategoriesConfig.qr_temperature`)
- **Dispatch**: all LLM calls via `SmoothRequester.process_all()` with `prepare_fn`/`parse_fn`/`fallback_fn` callbacks

## Shared Utils
- `utils/llm.py` — `fetch_rate_limits()` for API probe, `token_tracker` for cost tracking
- `utils/smoothRequester.py` — `SmoothRequester` for all LLM dispatch (rate pacing, retry, concurrency)
- `utils/cacheManager.py` — cache load/save
- `utils/promptPrinter.py` — prompt capture for debugging

## Imported from Other Steps
- Dimension data from `step_3_ideaExtractor`: `get_dimension`, `DimensionDefinition`

## Gotchas
- **All phases use SmoothRequester**: each phase creates its own SR instance with `known_limits` (fetched once at startup), `quiet=True` (caller handles reporting), and phase-specific `default_timeout`.
- **Auto-assignment**: domains with 1 facet skip LLM (P3), facets with 1 attribute skip LLM (P6). Saves API calls on trivial assignments.
- **Adaptive timeout multiplier**: `min(6, round(ln(num_tasks) + 1))` — few tasks get tight timeouts (P50 is reliable), many tasks get generous headroom.
- **Multi-round consolidation** (P2, P5): when chunks exceed thresholds (`consolidation_max_chunks_per_call`, `consolidation_max_items_per_call`), tasks are pre-grouped into round-1 SR tasks, then checked for a round-2 pass.
- **Label source**: `CategoriesConfig.label_source` controls what text represents an idea in all P1-P6 prompts (discovery observations *and* assignment idea text). Options: `"idea"`, `"instance"`, `"interpretation"`, `"abstraction"`, `"ladder"`, `"idea_interpretation"`. Significantly affects taxonomy quality. The dataclass default is `"idea"`, but the active pipeline overrides it in `run_classifier.py` — currently `"ladder"` (full `instance → interpretation → abstraction`). Step 4 facet/attribute assignment is the only place a rung choice bottlenecks the LLM input; step 3 domain assignment already sees the full response + all rungs.
- **Growing model**: `TaxonomyClassifiedModel` extends `IdeasExtractedModel`. P3 adds facet assignment + valence, P6 adds attribute assignment + valence, P7 consolidates attributes cross-facet. Raw P6 state (pre-P7) is also persisted as `raw_partition_attributes` and `raw_attribute_assignments` in both `TaxonomyResult` and `DomainResultModel`.
- **Valence cascade**: P3 assigns valence relative to facet, P6 assigns valence relative to attribute. The growing model carries the most precise valence available: P6 > P3 > step 3. Auto-assigned ideas (single-facet/single-attribute) keep the previous level's valence. Valence values: `"+"` (positive), `"-"` (negative), `"0"` (neutral). Not emotional sentiment — evaluative direction.
- **Dimension-aware prompts**: `prompt_rules` from step 3's selected `DimensionDefinition` are injected into all phase prompts to shape facet/attribute semantics.
- **Assignment validation**: P3 and P6 include ID drift detection, content similarity checks (SequenceMatcher threshold 0.7), duplicate detection, and `__UNASSIGNED__` fallback for missing assignments.
- **`force_recalc`**: not explicitly checked — step always runs full pipeline.
- **`debug_stop_after_phase`**: config option (default `None`) to stop after phase 2 or 5. Useful for testing specific phases without running the full pipeline.
- **Empirical capacity not saved for small phases**: SmoothRequester only persists `empirical_capacity` when the concurrency controller actually found a server-side ceiling. For phases with few tasks (e.g., 8 domains in P1/P2), the controller stays in `RAMP_UP` and no capacity is saved — preventing a feedback loop where bad calibration from too few tasks would cap concurrency on every subsequent run.

### P8 (cross-domain consolidation) gotchas
- **P8 overwrites P7 output**: P8 reloads the just-cached `taxonomy` / `taxonomy_classified`, consolidates attributes across domains, and overwrites the same cache keys (no separate keys). Destructive — no pre-P8 snapshot is kept.
- **Attribute identity is `(domain, attribute_name)`**: names are not unique across domains, so the merge map is keyed by `(domain, name)`. The LLM returns bare source names; they are resolved against the attributes actually present in each sliding window (unknown names are ignored).
- **Remap carries valence/confidence and keeps both domain fields in sync**: on a merge, attribute/facet valence + confidence move with the idea, and the growing model's `partition_name` *and* `domain` are both updated. `domain` is canonicalized to `partition_name` when the growing model is built (`_build_taxonomy_enriched_models`), so the two never diverge; P8 keeps both updated together when it moves an idea across domains.
- **Self-check**: `consolidate()` runs `_verify_consistency()` (idea count preserved, no valence/confidence dropped, no orphan assignments) and prints `P8 consistency: OK` or `⚠` warnings. P8 is skipped entirely when fewer than 2 attributes exist (nothing to consolidate; also avoids a seriation crash).

### P7.5 (valence-neutral merge) gotchas
- **Attributes never encode valence**: P4/P5/P7 prompts forbid splitting a concept by evaluative direction. P7.5 (`valence_consolidator.py`, after P7 / before P8) is the safety net: it detects attribute pairs within a facet with near-identical labels AND opposite valence skew, and merges the safe ones (auto-safe + single-token diff) into one descriptive attribute. The merged name/description comes from an LLM call (`classifier_p7` model) with a single-token deterministic fallback. Valence stays in the per-idea `valence` field. Overwrites `taxonomy` / `taxonomy_classified` (like P8). Cost-tracked as `p7_5_valence_merge`. `view_valence_split.py` is the read-only detector.

### Domain boundary_test + exclusions
- **From step 3**: `ExtractionMetadata.domains` dicts carry `boundary_test` (str) + `exclusions` (list[str]). `domain_discoverer.py` uses the persisted values (fallback to derived for old caches); `DomainDescription` carries `exclusions`. Injected into **P1 facet-discovery** (keep facets within-domain) and the **P8** cross-domain prompt (respect boundaries when merging).

## Processing Phases (display: 1-6, internal: P1-P8)
1. **Phase 1 — Facet Discovery + Consolidation** (P1 discovery concurrent, P2 consolidation concurrent)
2. **Phase 2 — Facet Assignment + Valence** (P3, concurrent, auto-assign for single-facet domains)
3. **Phase 3 — Attribute Discovery + Consolidation** (P4 discovery concurrent, P5 consolidation concurrent)
4. **Phase 4 — Attribute Assignment + Valence** (P6, concurrent, auto-assign for single-attribute facets)
5. **Phase 5 — Cross-facet Attribute Consolidation** (P7, concurrent per domain)
   - **P7.5 — Valence-neutral Attribute Merge** (`valence_consolidator.py`, after P7 / before P8): deterministically detect attribute pairs that differ only in valence, merge safe ones into one descriptive attribute (LLM-renamed, deterministic fallback); valence carries +/-.
6. **Phase 6 — Cross-domain Attribute Consolidation** (P8, embedding seriation + sliding-window LLM merge, global)

## Dev Docs
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [CACHE_LOGIC.md](CACHE_LOGIC.md) — caching contracts
- [PROCESSING.md](PROCESSING.md) — processing flow
- [WORK_CROSS_DOMAIN_CONSOLIDATION.md](WORK_CROSS_DOMAIN_CONSOLIDATION.md) — P8 status & remaining work
- [WORK_VALENCE_NEUTRALITY.md](WORK_VALENCE_NEUTRALITY.md) — valence-neutral attributes (P7.5)
