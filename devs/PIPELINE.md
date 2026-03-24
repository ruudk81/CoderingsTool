# PIPELINE.md — CoderingsTool Architecture Reference

> Living document for migrating development steps into production.
> Last updated: 2026-03-23

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION                         │
│  pipeline.py (standalone)  ←──→  app.py (Streamlit UI)      │
└──────────────────────────┬──────────────────────────────────┘
                           │ calls step functions
┌──────────────────────────▼──────────────────────────────────┐
│                      PROCESSING LAYER                        │
│  src/utils/*.py  — one or more utility per step              │
│  All LLM calls go through utils/llm.py (OpenAI + Azure)     │
└──────────────────────────┬──────────────────────────────────┘
                           │ reads / writes
┌──────────────────────────▼──────────────────────────────────┐
│                      PERSISTENCE LAYER                       │
│  utils/cacheManager.py → data/cache/ (SQLite + pickle)       │
│  exports/verbose_logs/   exports/prompts/   exports/*.xlsx   │
└─────────────────────────────────────────────────────────────┘
```

**Key files:**

| Role | File(s) |
|------|---------|
| Pipeline orchestration | `src/pipeline.py` |
| Streamlit UI | `src/app.py` |
| Global config | `src/config.py` (ModelConfig, CacheConfig, ProcessingConfig) |
| Pydantic models | `src/models.py` (production), `src/development/models_exp.py` (experimental) |
| LLM abstraction | `src/utils/llm.py` (OpenAI Responses API + Azure Chat Completions) |
| Cache management | `src/utils/cacheManager.py` |
| UI text (bilingual) | `src/ui_text.py` |

---

## 2. Current Production Pipeline (Steps 0–9)

Defined in `pipeline.py` → `STEP_NAMES`:

```python
STEP_NAMES = {
    0: "data",
    1: "preprocessed",
    2: "quality_filter",
    3: "extracted_ideas",
    4: "embeddings",
    5: "code_assignment",
    6: "codebook_generation",
    7: "codebook_refinement",
    8: "code_assignment_direct",
    9: "export"
}
```

### Step-by-step detail

#### Step 0 — Load Data (`data`, prefix `001`)

| | |
|---|---|
| **Function** | `step_0_load_data()` |
| **Utility** | `utils/dataLoader.py` → `DataLoader` |
| **Input** | SPSS file (.sav) |
| **Output** | `List[ResponseModel]` |
| **Config** | — |
| **Prompts** | — |
| **Notes** | Handles single + merged variables, sampling |

#### Step 1 — Preprocess (`preprocessed`, prefix `002`)

| | |
|---|---|
| **Function** | `step_1_preprocess()` |
| **Utilities** | `utils/textNormalizer.py`, `utils/spellChecker.py`, `utils/textFinalizer.py` |
| **Input** | `List[ResponseModel]` |
| **Output** | `List[PreprocessedModel]` + stats dict |
| **Config** | `config_steps/config_preprocess.py` → `SpellCheckConfig` |
| **Prompts** | `prompts_steps/prompts_spellChecker.py` |

#### Step 2 — Quality Filter (`quality_filter`, prefix `003`)

| | |
|---|---|
| **Function** | `step_2_quality_filter()` |
| **Utility** | `utils/qualityFilter.py` → `Grader` |
| **Input** | `List[PreprocessedModel]` |
| **Output** | `List[QualityFilteredModel]` |
| **Config** | `config_steps/config_qualityFilter.py` → `QualityFilterConfig` |
| **Prompts** | `prompts_steps/prompts_qualityFilter.py` |
| **Notes** | Codes: 0=meaningful, 99999997=don't know, 99999998=empty, 99999999=gibberish |

#### Step 3 — Extract Ideas (`extracted_ideas`, prefix `004`)

| | |
|---|---|
| **Function** | `step_3_extract_ideas()` |
| **Utility** | `utils/ideaExtractor.py` → `IdeaExtractor` |
| **Input** | `List[QualityFilteredModel]` |
| **Output** | `List[IdeasExtractedModel]` + `ExtractionMetadata` |
| **Config** | `config_steps/config_ideaExtractor.py` → `SegmentationConfig` |
| **Prompts** | `prompts_steps/prompts_ideaExtractor.py` |
| **Helpers** | `utils/dimension_data.py` (10 MECE dimensions + decision tree) |
| **Notes** | Also caches `ExtractionMetadata` (dimension, domains, template_prefix, context specifiers) |

#### Step 4 — Generate Embeddings (`embeddings`)

| | |
|---|---|
| **Function** | `step_4_generate_embeddings()` |
| **Utility** | `utils/embedder.py` → `Embedder` |
| **Input** | `List[IdeasExtractedModel]` |
| **Output** | `List[EmbeddingsModel]` |
| **Config** | `config_steps/config_embedder.py` → `EmbedderConfig` |
| **Prompts** | — |
| **Notes** | 6 embedding types: idea, interpretation, abstraction, facet, domain, ladder. Loads ExtractionMetadata from step 3 cache. |

#### Step 5 — Category Discovery & Assignment (`code_assignment`)

| | |
|---|---|
| **Function** | `step_4_classNcoder()` (naming mismatch — legacy) |
| **Utilities** | `utils/domain_discoverer.py`, `utils/map_reduce_mece.py`, `utils/code_assignment.py` |
| **Input** | `List[EmbeddingsModel]` |
| **Output** | `List[CodeAssignedModel]` + `CodingResultsCache` (metadata) |
| **Config** | `config_steps/config_categories.py` |
| **Prompts** | `prompts.py` → MAP/REDUCE/MECE prompts |
| **Notes** | Three-stage: partition by domain → MAP/REDUCE/MECE → assign ideas to categories |

#### Step 6 — Generate Codebook (`codebook_generation`)

| | |
|---|---|
| **Function** | `step_6_generate_codebook()` |
| **Utility** | `utils/codeGenerator.py` → `InductiveCodeGenerator` |
| **Input** | `List[CodeAssignedModel]` + MECE results + ExtractionMetadata |
| **Output** | `CodeGeneratorReasoningResults` + `List[ClusterModel]` (expanded clusters) |
| **Config** | `config_steps/config_codeGenerator.py` |
| **Prompts** | `prompts.py` → codebook generation prompts |

#### Step 7 — Refine Codebook (`codebook_refinement`)

| | |
|---|---|
| **Function** | `step_7_refine_codebook()` |
| **Utility** | `utils/codebookRefinement.py` |
| **Input** | `CodeGeneratorReasoningResults` |
| **Output** | `CodeRefinementResults` + `ThemeEnrichedCodebookModel` |
| **Config** | — |
| **Prompts** | `prompts.py` → refinement prompts |

#### Step 8 — Assign Codes (`code_assignment_direct`)

| | |
|---|---|
| **Function** | `step_8_assign_codes()` |
| **Utility** | `utils/codeAssigner.py` → `CodeAssigner` |
| **Input** | `ThemeEnrichedCodebookModel` + expanded clusters from step 6 |
| **Output** | `List[CodeAssignedModel]` |
| **Config** | `config_steps/config_codeAssigner.py` |
| **Prompts** | `prompts.py` → code evaluation prompts |

#### Step 9 — Export (`export`)

| | |
|---|---|
| **Function** | `step_9_export_results()` |
| **Utility** | `utils/resultsExporter.py` → `ResultsExporter` |
| **Input** | `List[CodeAssignedModel]` + `ThemeEnrichedCodebookModel` |
| **Output** | Excel file in `exports/` |
| **Notes** | Optional visualizations via `utils/exportVisualizer.py` |

---

## 3. Development Steps (New Pipeline — Source of Truth)

Located in `src/development/`. These steps will replace production steps 4–8.

Steps 0–3 remain largely the same (data loading, preprocessing, quality filter, idea extraction).

### Development directory structure

```
src/development/
├── models_exp.py                    # Experimental Pydantic models
├── test_data.py                     # Shared test dataset config
├── step_1_preProcessor/             # Largely same as production
│   ├── spellChecker_exp.py
│   ├── textNormalizer_exp.py
│   ├── textFinalizer_exp.py
│   ├── config_exp.py
│   ├── prompts_exp.py
│   └── run_experiment.py
├── step_2_qualityFilter/            # Largely same as production
│   ├── qualityFilter_exp.py
│   ├── config_exp.py
│   ├── prompts_exp.py
│   └── run_experiment.py
├── step_3_ideaExtractor/            # Largely same as production
│   ├── ideaExtractor_exp.py
│   ├── dimension_data.py
│   ├── models_exp.py               # Step-specific experimental models
│   ├── prompts_exp.py
│   └── run_experiment.py
├── step_4_classifier/               # NEW — Taxonomy classification (P1-P7)
│   ├── classifier.py                # TaxonomyClassifier (main)
│   ├── domain_discoverer.py         # Domain partition logic
│   ├── partition_labels.py          # Label formatting helpers
│   ├── config_classifier.py         # CategoriesConfig, ClassifierRampConfig
│   ├── prompts_classifier.py        # P1-P7 prompts
│   ├── models_classifier.py         # DomainSet, DomainDescription, TaxonomyResultsCache
│   └── run_experiment.py
├── step_5_codeGenerator/            # NEW — Codebook generation (P8-P9)
│   ├── codebook_generator.py        # CodebookGenerator (main)
│   ├── config_codeGenerator.py      # CodebookConfig
│   ├── prompts_codeGenerator.py     # P8-P9 prompts
│   ├── models_codeGenerator.py      # CodingResultsCache (with raw_codes)
│   └── run_experiment.py
├── step_6_codeAssigner/             # NEW — Code assignment (P10)
│   ├── code_assignment.py           # CodeAssigner (main)
│   ├── embedding_matcher.py         # Cosine similarity candidate selection
│   ├── config_codeAssigner.py       # AssignmentConfig
│   ├── prompts_codeAssigner.py      # P10 prompt
│   ├── models_codeAssigner.py       # CodeAssignment, CodeAssignedSubmodel, CodeAssignedModel
│   └── run_experiment.py
└── step_7_export/                   # TODO — Needs development
```

### Dev Step 1: PreProcessor (largely unchanged)

| | |
|---|---|
| **Utility** | `step_1_preProcessor/spellChecker_exp.py`, `textNormalizer_exp.py`, `textFinalizer_exp.py` |
| **Config** | `step_1_preProcessor/config_exp.py` |
| **Prompts** | `step_1_preProcessor/prompts_exp.py` |
| **Models** | Uses production `PreprocessedModel` |

### Dev Step 2: Quality Filter (largely unchanged)

| | |
|---|---|
| **Utility** | `step_2_qualityFilter/qualityFilter_exp.py` |
| **Config** | `step_2_qualityFilter/config_exp.py` |
| **Prompts** | `step_2_qualityFilter/prompts_exp.py` |
| **Models** | Uses production `QualityFilteredModel` |

### Dev Step 3: Idea Extractor (largely unchanged)

| | |
|---|---|
| **Utility** | `step_3_ideaExtractor/ideaExtractor_exp.py` |
| **Config** | — |
| **Prompts** | `step_3_ideaExtractor/prompts_exp.py` |
| **Models** | `step_3_ideaExtractor/models_exp.py` → diverges from production (see §5) |
| **Helpers** | `step_3_ideaExtractor/dimension_data.py` |

### Dev Step 4: Taxonomy Classifier — NEW (P1-P7)

> Replaces production step 5 (code_assignment). Classifies ideas into a taxonomy: domain → facet → attribute.

| | |
|---|---|
| **Utility** | `step_4_classifier/classifier.py` → `TaxonomyClassifier` |
| **Config** | `step_4_classifier/config_classifier.py` → `CategoriesConfig`, `ClassifierRampConfig` |
| **Prompts** | `step_4_classifier/prompts_classifier.py` |
| **Models** | `step_4_classifier/models_classifier.py` → `DomainSet`, `DomainDescription`, `DomainResultModel`, `TaxonomyResultsCache` |
| **Helpers** | `step_4_classifier/domain_discoverer.py`, `step_4_classifier/partition_labels.py` |
| **Cache** | `taxonomy` (prefix `005`) + `taxonomy_metadata` (prefix `005`) |

**Sub-stages (per domain, concurrent):**
1. P1: Facet Discovery (chunked)
2. P2: Facet Consolidation (hierarchical merge)
3. P3: Facet Assignment (batched)
4. P4: Attribute Discovery (per facet)
5. P5: Attribute Consolidation (per facet, hierarchical)
6. P6: Attribute Assignment (per facet)
7. P7: Cross-facet Attribute Consolidation

### Dev Step 5: Code Generator — NEW (P8-P9)

> Replaces production steps 6-7 (codebook_generation + codebook_refinement). Generates MECE codes from taxonomy attributes.

| | |
|---|---|
| **Utility** | `step_5_codeGenerator/codebook_generator.py` → `CodebookGenerator` |
| **Config** | `step_5_codeGenerator/config_codeGenerator.py` → `CodebookConfig` |
| **Prompts** | `step_5_codeGenerator/prompts_codeGenerator.py` |
| **Models** | `step_5_codeGenerator/models_codeGenerator.py` → `CodingResultsCache` (extends step 4's models) |
| **Cache** | `mece_codes` (prefix `006`) + `mece_codes_metadata` (prefix `006`) |

**Sub-stages:**
1. P8: Code Generation from Attributes (per domain)
2. P9: Codebook Consolidation (cross-domain merge)

### Dev Step 6: Code Assigner — NEW (P10)

> Replaces production step 8 (code_assignment_direct). Assigns codes + attributes to each idea.

| | |
|---|---|
| **Utility** | `step_6_codeAssigner/code_assignment.py` → `CodeAssigner` |
| **Config** | `step_6_codeAssigner/config_codeAssigner.py` → `AssignmentConfig` |
| **Prompts** | `step_6_codeAssigner/prompts_codeAssigner.py` |
| **Models** | `step_6_codeAssigner/models_codeAssigner.py` → `CodeAssignment`, `CodeAssignedSubmodel`, `CodeAssignedModel` |
| **Helpers** | `step_6_codeAssigner/embedding_matcher.py` (cosine similarity for candidate selection) |
| **Cache** | `taxonomy_codes` (prefix `007`) |

**Features:**
- Dual assignment: code + best-matching attribute per idea
- 4-layer rate limiting: ConcurrencyGate + TokenBucket + AsyncLimiter + Timeout
- Circuit breaker for sustained pressure

### Dev Step 7: Export — TODO

> Still needs development. Will replace production step 9.

---

## 4. Migration Mapping (Production → Development)

| Production Step | Production Name | → | Dev Step | Dev Name |
|---|---|---|---|---|
| 0 | data | = | 0 | data (unchanged) |
| 1 | preprocessed | ≈ | 1 | preProcessor (largely same) |
| 2 | quality_filter | ≈ | 2 | qualityFilter (largely same) |
| 3 | extracted_ideas | ≈ | 3 | ideaExtractor (largely same, models diverge) |
| 4 | embeddings | → | — | TBD (may fold into step 3 or 4) |
| 5 | code_assignment | → | 4 | classifier (P1-P7) |
| 6 | codebook_generation | → | 5 | codeGenerator (P8-P9) |
| 7 | codebook_refinement | → | 5 | (merged into codeGenerator) |
| 8 | code_assignment_direct | → | 6 | codeAssigner (P10) |
| 9 | export | → | 7 | export (needs development) |

---

## 5. Pydantic Model Chain

### Production (`src/models.py`)

```
ResponseModel
  └─ PreprocessedModel (adds: quality_filter, quality_filter_code)
       └─ QualityFilteredModel (no new fields)
            └─ IdeasExtractedModel (adds: response_ideas[IdeasExtractedSubmodel], idea_count, template_prefix)
                 │  Submodel fields: idea_id, idea, instance, interpretation, abstraction,
                 │                   domain, facet, attribute, valence
                 └─ EmbeddingsModel (adds: embeddings per idea)
                      │  Submodel fields: idea_embedding, interpretation_embedding, abstraction_embedding,
                      │                   facet_embedding, domain_embedding, ladder_embedding
                      └─ CodeAssignedModel (adds: category assignments)
                           Submodel fields: assigned_category, category_confidence, category_rationale,
                                           partition_name, initial_cluster, expanded_cluster, cluster_theme
```

### Development (`src/development/models_exp.py`)

Key divergences from production:

| Field | Production | Development |
|-------|-----------|-------------|
| IdeasExtractedSubmodel.interpretation | `interpretation` | `rung_1` |
| IdeasExtractedSubmodel.abstraction | `abstraction` | `rung_2` |
| IdeasExtractedSubmodel.domain | `domain` | `concept_type` |
| EmbeddingsSubmodel | interpretation_embedding, abstraction_embedding, facet_embedding, domain_embedding | rung_1_embedding, rung_2_embedding, concept_type_embedding |
| CodeAssignedSubmodel | assigned_category, category_confidence | assigned_code, assigned_attribute, confidence |

**NOTE:** `models.py` will need to be updated during migration to accommodate new step outputs. The development models in `models_exp.py` and `models_codeAssigner.py` are the source of truth for the target schema.

### Metadata models

- `ExtractionMetadata` — dataset-level (language, sector, topic, template_prefix, primary_dimension, domains)
- `TaxonomyResultsCache` (dev step 4) — domains + facets + attributes per domain
- `CodingResultsCache` (dev step 5) — extends TaxonomyResultsCache with `raw_codes`

---

## 6. File Organization Reference

### Per-step config files (`src/config_steps/`)

| Step | Config File | Key Classes |
|------|------------|-------------|
| 1 | `config_preprocess.py` | `SpellCheckConfig` |
| 2 | `config_qualityFilter.py` | `QualityFilterConfig` |
| 3 | `config_ideaExtractor.py` | `SegmentationConfig`, ramp-up, PID controller |
| 4 | `config_embedder.py` | `EmbedderConfig`, `MULTI_PASS_SPECS` |
| 5 | `config_categories.py` | `CategoriesConfig`, `AssignmentConfig` |
| 6 | `config_codeGenerator.py` | `CodeDesignerConfig` |
| 8 | `config_codeAssigner.py` | `CodeAssignmentConfig` |

### Per-step prompt files (`src/prompts_steps/`)

| Step | Prompts File | Key Prompts |
|------|-------------|-------------|
| 1 | `prompts_spellChecker.py` | `SPELLCHECK_INSTRUCTIONS` |
| 2 | `prompts_qualityFilter.py` | `GRADER_INSTRUCTIONS` + `QualityFilterLLMResponse` |
| 3 | `prompts_ideaExtractor.py` | Context specifiers, dimension decision tree, domain discovery |
| 5-8 | `prompts.py` (main) | MAP/REDUCE/MECE, codebook gen, refinement, code evaluation |

### Dev step-specific config and prompt files

| Dev Step | Config | Prompts |
|----------|--------|---------|
| 4 | `step_4_classifier/config_classifier.py` | `step_4_classifier/prompts_classifier.py` |
| 5 | `step_5_codeGenerator/config_codeGenerator.py` | `step_5_codeGenerator/prompts_codeGenerator.py` |
| 6 | `step_6_codeAssigner/config_codeAssigner.py` | `step_6_codeAssigner/prompts_codeAssigner.py` |

---

## 7. LLM Integration

All LLM calls go through `src/utils/llm.py`:

- **`create_client(model, async_mode)`** — returns instructor-wrapped OpenAI/Azure client
- **`llm_create_async()`** — async call with Pydantic response model
- **`llm_create_sync()`** — sync variant
- **`create_embedding_client()`** — for embedding models
- **`get_model_limits(model)`** — context window + max output tokens
- **`token_tracker`** — global usage/cost monitoring
- **`_is_reasoning_model()`** — disables temperature for GPT-5/o1 models

Models configured in `config.py` → `ModelConfig`:
- Per-stage model selection (spell_check, quality_filter, embedding, codebook_generation, etc.)
- Per-stage temperature, reasoning_effort, text_verbosity
- Model type mapping (chat vs reasoning)

---

## 8. Output Artifacts

| Location | Contents |
|----------|----------|
| `exports/*.xlsx` | Final Excel exports (step 9) |
| `exports/verbose_logs/` | Terminal output captures per run. Naming: `{filename}_{variable}_{sample}_step{N}_{timestamp}.txt` |
| `exports/prompts/` | Saved prompts per step as JSON. Naming: `step{N}_{utility}_{variable}_{sample}_{type}.json` |
| `data/cache/` | Cached pipeline results (see CACHE_LOGIC.md) |
