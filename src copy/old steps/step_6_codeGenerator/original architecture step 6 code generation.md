# Step 6 `codeGenerator_exp` — Architecture

## 1. Input Data (loaded from cache)

`run_experiment.py` loads from cache via `CacheManager`:

| Cache Key | Pydantic Model | Source |
|---|---|---|
| `extracted_ideas` metadata | `ExtractionMetadata` | Step 3 — context specifiers (domain, topic, perspective, taxonomy axis, etc.) |
| `initial_clusters` | `List[ClusterModel]` | Step 5 — clustered ideas with embeddings, cluster assignments, probabilities |
| `clustering_metadata` | `ClusteringMetadataModel` | Step 5 — cluster labels used as "starter codes" |
| `mece_categories` metadata | `MECEResultsCache` | Step 5 categories — MECE partition results |
| `category_assignment` | `List[CategoryAssignedModel]` | Step 5 categories — ideas with category assignments |

The `STAGE1_INPUT_SOURCE` config in `config_exp.py:37` selects which input path: `"mece_categories"` (default), `"mece_topics"` (legacy), or `"ideas"` (raw idea sampling).

## 2. Processing — `InductiveCodeGenerator` class

`codeGenerator_exp.py` is ~6000 lines. The `InductiveCodeGenerator` class (line 1331) orchestrates everything.

Key supporting classes:
- **`SharedCodebook`** (line 596) — thread-safe codebook that grows as themes are processed; tracks versions and cached embeddings
- **`SimilarityEngine`** (line 856) — embeds themes, creates dissimilarity batches (groups dissimilar themes together so each batch builds codebook diversity), uses UMAP/cosine similarity
- **`TokenBucket`** / **`LatencyTracker`** — adaptive rate limiting with bootstrap measurement

## 3. The Four-Chain Logic (the core pipeline)

The `design()` method (line 4533) orchestrates these stages:

### Stage 0+1: Data Extraction + Theme Extraction (Prompt 1)

- Extracts cluster/category data → calls `_extract_single_theme()` (line 3539) or `_extract_single_theme_from_category()` (line 3407) per cluster
- Uses unified `CLUSTER_SUMMARY_PROMPT` from `prompts_exp.py` (with route-specific params: `CLUSTER_PROMPT_PARAMS` or `CATEGORY_PROMPT_PARAMS`)
- Response model: `ClusterSummaryOutput` — contains COCs (Central Organizing Concepts) + atomic themes with labels, definitions, assignment examples
- Ideas are sampled by **probability band** (inner/border/fringe members)
- Multi-theme clusters are **expanded** into sub-clusters (`expand_multi_theme_clusters` at line 2595), e.g., cluster 5 with 3 themes → `5-1`, `5-2`, `5-3`

### Stage 1.5: Theme Embedding + Redistribution

- `SimilarityEngine.embed_themes()` embeds all extracted themes
- Multi-theme clusters get ideas **redistributed** to sub-themes via embedding similarity

### Stage 2: Dissimilarity Batching

- Groups themes into batches of **maximally dissimilar** themes so each batch covers diverse topics
- This prevents the codebook from over-concentrating on similar concepts early

### Stage 3: Sequential Batch Processing

For each theme in a batch, runs **Prompts 2 → 3 → 4**:

| Chain | Method | Prompt | Response Model | Purpose |
|---|---|---|---|---|
| **Prompt 2** | `_select_candidate_codes()` (line 5464) | `CODING_DECISION_PROMPT` | `CodingDecisionOutput` | Decides **USE** / **MODIFY_HORIZONTAL** / **MODIFY_VERTICAL** / **CREATE** by comparing theme to existing codebook with cosine similarity |
| **Prompt 3** | `_generate_code()` (line 5639) | `CODE_CREATION_PROMPT` or `CODING_MODIFICATION_PROMPT` | `CodeGenerationOutput` | Generates/modifies the code label + definition + assignment examples. Skipped for USE decisions. |
| **Prompt 4** | `_validate_code()` (line 5881) | `VALIDATION_PROMPT` with scenario-specific instructions | `ValidationResult` | APPROVE/REJECT the proposal; can override to a different decision. Applies fuzzy matching on source codes. |

After validation, the result is merged into `SharedCodebook` and the theme → code assignment is recorded.

### Post-processing

- **Error leak recovery** — retries failed clusters
- **Modification leak recovery** — retries MODIFY decisions that failed due to race conditions (concurrent codebook mutations)

## 4. Verbose Report (export)

`run_experiment.py` wraps execution in a `VerboseCapture` context manager (`saveVerbose.py`) that tees all stdout to both console and a timestamped log file in `exports/verbose_logs/`. This captures the full processing trace including the `_generate_final_report()` output (line 4739) with stage timings, success rates, codebook growth, and decision breakdowns.

## 5. Cached Output (validated by Pydantic)

`run_experiment.py` saves two items to cache (lines 313-314):

| Cache Key | Model | Content |
|---|---|---|
| `codebook_generation_reasoning` | `CodeGeneratorReasoningResults` (line 169) | Complete reasoning chain: step1-4 inputs/outputs, codebook, cluster assignments, stats, redistribution stats |
| `expanded_clusters` | `List[ClusterModel]` | Updated cluster models with `expanded_cluster` assignments reflecting sub-theme redistribution |

The `CodeGeneratorReasoningResults` Pydantic model stores everything for downstream consumption and the verbose report: all four chain results per cluster (`step1_summaries`, `step2_analysis`, `step3_recommendations`, `step4_validations`), the final `codebook` list, and `cluster_assignments` mapping.

### CodeGeneratorReasoningResults structure

```python
class CodeGeneratorReasoningResults(BaseModel):
    # Per-cluster raw results
    cluster_results: List[Dict[str, Any]]

    # Full prompt I/O per stage per cluster
    step1_inputs:  Dict[cluster_id, Dict]   # What Prompt 1 received
    step2_inputs:  Dict[cluster_id, Dict]   # What Prompt 2 received
    step3_inputs:  Dict[cluster_id, Dict]   # What Prompt 3 received
    step4_inputs:  Dict[cluster_id, Dict]   # What Prompt 4 received

    step1_summaries:        Dict[cluster_id, Dict]   # Theme extraction results
    step2_analysis:         Dict[cluster_id, Dict]   # Coding decisions
    step3_recommendations:  Dict[cluster_id, Dict]   # Generated/modified codes
    step4_validations:      Dict[cluster_id, Dict]   # Validation results
    step4_validated_codes:  Dict[cluster_id, Dict]   # Final validated codes

    # Aggregated outputs
    codebook:             List[Dict[str, str]]              # Final code list
    cluster_assignments:  Dict[cluster_id, Dict]            # cluster → code mapping
    cluster_data:         Dict[cluster_id, Dict]            # Raw cluster data
    stats:                Dict[str, Any]                    # Processing statistics
    validation_details:   Optional[Dict[cluster_id, Any]]   # Validation metadata
    redistribution_stats: Optional[Dict[str, Any]]          # Multi-theme redistribution stats

    # Metadata
    generator_version: str
    var_lab: str
    total_clusters: int
    total_ideas: int
    processing_timestamp: str
```

## 6. Key Data Flow Diagram

```
Cache Input
├── ExtractionMetadata (step 3)
├── ClusterModel[] (step 5)
├── ClusteringMetadataModel (step 5) → starter_codes
├── MECEResultsCache (step 5 categories)
└── CategoryAssignedModel[] (step 5 categories)
         │
         ▼
┌─────────────────────────────────────┐
│     InductiveCodeGenerator          │
│                                     │
│  Stage 0+1: Extract + Theme (P1)   │
│       │                             │
│  Stage 1.5: Embed + Redistribute   │
│       │                             │
│  Stage 2: Dissimilarity Batching   │
│       │                             │
│  Stage 3: Per-theme chain          │
│    ├── P2: Coding Decision         │
│    ├── P3: Code Generation         │
│    └── P4: Validation              │
│       │                             │
│  Post: Error/Modification Recovery │
└───────┬─────────────────────────────┘
        │
        ▼
Cache Output
├── codebook_generation_reasoning (CodeGeneratorReasoningResults)
└── expanded_clusters (List[ClusterModel])
```

## 7. All Python Files Used

### Experiment-specific

| File | Role |
|---|---|
| `run_experiment.py` | Orchestrator — loads cache, instantiates generator, runs, caches results, saves verbose log |
| `codeGenerator_exp.py` | Main engine (~6000 lines) — `InductiveCodeGenerator` class with 4-stage pipeline |
| `config_exp.py` | Experimental config — `CodeDesignerConfigExp` + 30+ tuning constants (thresholds, batch sizes, timeouts) |
| `prompts_exp.py` | 11 prompt artifacts (1 unified Stage 1 with route params, 1 Stage 2, 2 Stage 3 + 2 modification instructions, 1 Stage 4 base + 4 validation instruction variants) + all Pydantic response models co-located with their prompts |
| `debug_prompts.py` | Debug script for testing prompt generation for specific clusters |
| `debug_reasoning.py` | Debug script for inspecting reasoning chain outputs |
| `codegenPromptTester_exp.py` | Prompt testing utility — reconstructs prompts from cached params for inspection |
| `../test_data.py` | Shared test dataset config |
| `../models_exp.py` | Experimental Pydantic models (different field names from production) |

### Production utilities imported

| File | What's used |
|---|---|
| `utils/cacheManager.py` | `CacheManager`, `generate_enhanced_variable_key` |
| `utils/verboseReporter.py` | `VerboseReporter` — structured console logging |
| `utils/saveVerbose.py` | `VerboseCapture` — tee stdout to file |
| `utils/promptPrinter.py` | `PromptPrinter` — optional prompt display |
| `utils/llm.py` | `create_client`, `llm_create_sync`, `token_tracker`, `RateLimits` |
| `utils/dataLoader.py` | `DataLoader.get_varlab()` — fetches survey question text |
| `utils/clusterer.py` | `clean_cluster_ideas()` — pre-processing |
| `config.py` | `CacheConfig`, `ModelConfig`, `ProcessingConfig`, re-exported through config_exp |
