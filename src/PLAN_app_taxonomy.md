# Plan: app_taxonomy.py — Streamlit UI for Taxonomy Pipeline

## Context

The development folder contains a new pipeline approach (`pipeline_taxonomy.py`) that replaces the 10-step pipeline with a shorter, dimension-aware flow. The key innovation is **classNcoder** (step 4), which replaces the old steps 4-8 (embeddings, clustering, codebook generation, refinement, code assignment) with a single integrated step that uses domain assignments from step 3 directly.

This document plans the second Streamlit app: `app_taxonomy.py`.

---

## Pipeline Comparison

### Old pipeline (app.py)
```
Step 0: Load Data
Step 1: Preprocess (spell check)
Step 2: Quality Filter
Step 3: Extract Ideas (basic segmentation)
Step 4: Generate Embeddings          ← removed
Step 5: Cluster (UMAP + HDBSCAN)    ← removed
Step 6: Generate Codebook            ← removed
Step 7: Refine Codebook              ← removed
Step 8: Assign Codes                 ← removed
Step 9: Export
```

### New pipeline (app_taxonomy.py)
```
Step 0: Load Data
Step 1: Preprocess (spell check)
Step 2: Quality Filter
Step 3: Extract Ideas (dimensional — discovers dimension, domains, extracts with taxonomy)
Step 4: classNcoder (partition → facet discovery → attribute discovery → code generation → assignment)
Step 5: Export
```

**6 steps instead of 10.** Steps 0-2 are identical. Step 3 is enhanced. Step 4 is entirely new. Step 5 is export (adapted for new output).

---

## What Can Be Reused from app.py

### Reuse as-is (copy or import)
- **DatasetConfig dataclass** — identical logic for file/variable/sample management
- **Lazy loaders** — `_get_data_loader()`, `_get_cache_manager()`, `_load_or_recover()`
- **Session state initialization** — same pattern, fewer config objects needed
- **Navigation system** — same prev/next/home/step-navigator pattern, just 6 steps instead of 10
- **Step page structure** — same 5-block pattern (green box, blue box, yellow box, data loading, processing button)
- **Cache invalidation** — `invalidate_from_step()` with updated step mappings
- **Verbose log capture** — `_run_with_verbose_capture()`, `show_verbose_log_expander()`
- **Step 0 upload page** — `show_upload_page()` nearly identical
- **Step 1 preprocessing page** — `show_preprocessing_page()` identical
- **Step 2 filtering page** — `show_filtering_page()` identical

### Reuse with modifications
- **Advanced settings sidebar** — much simpler: only models for spell check, quality filter, segmentation, classNcoder phases (P1-P4)
- **Step result keys mapping** — updated for new step numbers
- **Cache step name mapping** — updated for new pipeline
- **`show_info_panel()`** — routes to new sampling functions for steps 3-4

### New / significantly different
- **Step 3 results display** — must show dimension, domains, taxonomy, abstraction ladder
- **Step 4 results display** — must show partitions → facets → attributes → codes hierarchy
- **Step 5 export page** — adapted for new output models

---

## Step-by-Step UI Design

### Step 0: Load Data (no change)
Identical to current app.py. Load from cache or upload SPSS file.

### Step 1: Preprocess (no change)
Identical. Spell check + normalization.

### Step 2: Quality Filter (no change)
Identical. Remove gibberish, empty, "don't know" responses.

### Step 3: Extract Ideas (significantly different display)

**Processing** is the same pattern (button → call pipeline → store results → mark complete).

**New display elements needed:**

1. **Blue box** (input info): Same as current — question text + response count
2. **Yellow box** (stats): Enhanced with taxonomy info:
   - Dimension selected (e.g., `ATTRIBUTES_ASSOCIATIONS`)
   - Dimension description
   - Number of domains discovered
   - Domain list with idea counts
   - Total ideas extracted / unique ideas
   - Single vs multi-idea response breakdown
3. **Info panel / sampling section**:
   - **Taxonomy overview card**: Show dimension → domains hierarchy
   - **Domain breakdown**: For each domain, show idea count + sample ideas
   - **Random sample viewer**: Show response → ideas with their abstraction ladder:
     ```
     Response: "Duurzaam en groen bankieren"
       → Instance: "duurzaam bankieren"
         Interpretation: "de bank hanteert duurzame principes"
         Abstraction: "milieubewust financieel beleid"
         Domain: milieu- en duurzaamheidswaarden
         Valence: +
     ```
   - **Context specifiers display**: Show detected lang, perspective, intent, sector, topic, entity

### Step 4: classNcoder (entirely new)

**Processing**: Button → call `pipeline_taxonomy.step_4_classNcoder()` → store results → mark complete.

**Display elements needed:**

1. **Blue box**: Question + idea count from step 3
2. **Yellow box** (processing stats):
   - Pipeline timing (total + per phase)
   - Partition count + idea distribution
   - Facets discovered (total across domains)
   - Attributes discovered (total)
   - Codes generated (raw → consolidated)
3. **Main results display — hierarchical codebook browser**:
   The core output is a 4-level taxonomy: **Domain → Facet → Attribute → Code**

   **Option A: Nested expanders** (simplest, familiar from step 7)
   ```
   📂 Domain: milieu- en duurzaamheidswaarden (46 ideas)
     📁 Facet: Duurzaamheid en milieubewustzijn
       🏷️ Attribute: milieuvriendelijk beleggingsbeleid
         → Code: "Duurzaamheid en milieu"
         → Definition: "Waargenomen inzet van Merk X voor duurzaamheid..."
     📁 Facet: Natuur en natuurbescherming
       🏷️ Attribute: ...
   ```

   **Option B: Domain tab navigation** (like current category browser)
   - Top-level: prev/next domain navigation (like `show_category_samples()`)
   - Inside each domain: expandable facets with their attributes and codes
   - Code definitions shown inline

   **Option C: Codebook-first view** (like current step 6 display)
   - List of final consolidated codes with definitions
   - Each code expandable to show: source domains, facets, attributes, sample ideas
   - Pure CSS tabs per code (reuse `CSS_STYLES` pattern from app.py)

   **Recommendation**: Combine B + C:
   - **Tab 1: "Codebook"** — flat list of consolidated codes with definitions (most useful for end user)
   - **Tab 2: "Taxonomy"** — domain → facet → attribute hierarchy with idea counts
   - **Tab 3: "Assignments"** — random sample viewer showing idea → assigned code + rationale

4. **Info panel / sampling section**:
   - Code frequency distribution (bar chart or sorted list)
   - Random assignment sample (like current step 8/9 display)
   - Per-domain code distribution

### Step 5: Export (adapted)

Same pattern as current step 9. Export to Excel with:
- Code assignments per idea
- Taxonomy hierarchy (domain, facet, attribute per idea)
- Codebook sheet
- Optionally: domain distribution stats

**Key difference**: No reasoning_results from old step 6/7. Instead export the taxonomy metadata (dimension, domains, facets, attributes).

---

## Technical Decisions

### File structure
- `src/app_taxonomy.py` — standalone Streamlit app
- Run with: `streamlit run src/app_taxonomy.py`
- Imports from `pipeline_taxonomy.py` (which needs to be finalized first)

### Pipeline dependency
`app_taxonomy.py` depends on `pipeline_taxonomy.py` having clean function signatures for:
- `step_0_load_data(...)` → same as current
- `step_1_preprocess(...)` → same as current
- `step_2_quality_filter(...)` → same as current
- `step_3_extract_ideas(...)` → enhanced, returns ideas + ExtractionMetadata
- `step_4_classNcoder(...)` → new, returns CodeAssignedModel[]
- `step_5_export_results(...)` → adapted for new output

### Models dependency
The new app uses models from:
- `src/models.py` — ResponseModel, PreprocessedModel, QualityFilteredModel (steps 0-2)
- `src/development/step_3_ideaExtractor/models_exp.py` — IdeasExtractedModel, ExtractionMetadata
- `src/development/step_4_classNcoder/models_exp.py` — CodeAssignedModel, CodeAssignedSubmodel

**Note**: Before app_taxonomy.py can work, the development models need to be either:
1. Imported directly from development/ (quick, for testing)
2. Migrated to production src/models.py (proper, for release)

### Cache step name mapping (new)
```python
step_mapping = {
    0: "data",
    1: "preprocessed",
    2: "quality_filter",
    3: "extracted_ideas",      # + "extraction_metadata"
    4: "code_assignment",      # classNcoder output
    5: "export"
}
```

### Session state changes
**Removed** (not needed):
- `code_designer_config`
- `code_assignment_config` (the old one for step 8)
- Step-specific stats: `codebook_stats`, `theme_stats`

**Added**:
- `classncoder_config` — CategoriesConfig from step_4_classNcoder
- `extraction_metadata` — ExtractionMetadata from step 3
- `taxonomy_stats` — dimension, domains, facet/attribute counts
- `classncoder_stats` — phase timings, code counts

### Advanced settings (simplified)
```
Step 1: Spell check model
Step 2: Quality filter model
Step 3: Segmentation model
Step 4: classNcoder phase models (P1, P1.5, P2, P3, P4)
        + label_source selector (ladder, idea_rungs, etc.)
        + batch sizing parameters
```

---

## Sequencing / Prerequisites

### Before app_taxonomy.py can be built:

1. **`pipeline_taxonomy.py` must be finalized** with clean step functions
   - Currently exists but may need function signature cleanup
   - Step functions should accept the same pattern as current pipeline.py:
     `(input_data, filename, var_lab, variable_key, cache_manager, model_config, force_recalc, verbose)`

2. **Model imports must work** from development/ or be migrated
   - Step 3 models (IdeasExtractedModel with domain, facet, attribute fields)
   - Step 4 models (CodeAssignedModel with assigned_category, assigned_attribute)
   - ExtractionMetadata

3. **Cache compatibility** — verify that classNcoder results can be loaded from cache
   the same way the old pipeline results can (pickle-based CacheManager should handle this)

### Build order for app_taxonomy.py:

1. **Phase 1: Skeleton** — Copy app.py, strip steps 4-9, renumber to 0-5
   - Keep all infrastructure (DatasetConfig, lazy loaders, navigation, session state)
   - Steps 0-2 work immediately (identical)
   - Step 3 placeholder (processing works, display is basic)
   - Step 4 placeholder (processing works, display is basic)
   - Step 5 placeholder (basic export)

2. **Phase 2: Step 3 display** — Build the taxonomy/dimension display
   - Taxonomy overview card
   - Domain breakdown with idea counts
   - Enhanced random sample viewer with abstraction ladder

3. **Phase 3: Step 4 display** — Build the classNcoder results display
   - Codebook tab (consolidated codes + definitions)
   - Taxonomy tab (domain → facet → attribute hierarchy)
   - Assignment sample tab

4. **Phase 4: Step 5 export** — Adapt export for taxonomy output
   - Include taxonomy columns in Excel
   - Codebook sheet with hierarchy info

5. **Phase 5: Polish** — Advanced settings, codebook editing, UI text (bilingual)

---

## Open Questions

1. **Codebook editing in step 4?** Current app.py has a codebook editor after step 7. Do we want a similar editor after step 4 (classNcoder) to allow manual tweaks before export?

2. **Visualization in export?** Current export includes dendrogram, word clouds, network graph (all cluster-based). These don't apply to the new pipeline. What visualizations (if any) should the taxonomy export include? Domain distribution chart? Facet treemap?

3. **Shared code between app.py and app_taxonomy.py?** Should we extract common infrastructure (DatasetConfig, lazy loaders, navigation) into a shared module (e.g., `app_common.py`)? Or keep them independent for now?

4. **Cache isolation?** Should app_taxonomy.py use the same cache database as app.py, or a separate one? Steps 0-2 produce identical results, so sharing makes sense. But step 3 outputs are different (dimensional vs basic), so cache keys need to distinguish them.

5. **Model migration timing?** Build app_taxonomy.py with imports from `development/` first (fast iteration), then migrate models to production later? Or migrate models first?