# Job 3: Rewrite app.py for 8-Step Pipeline

## Status: PLANNED (not yet started)

Jobs 1-2 complete. app.py still references old 10-step pipeline.

---

## app.py Architecture (Reverse-Engineered)

### Core Design: 5311 lines, 3 architectural layers

**1. Session State (3-tier)**
- **Navigation**: `step` (int), `completed_steps` (set), `max_step_reached` (int)
- **Config**: DatasetConfig with `_config` suffix pattern (avoids Streamlit widget key conflicts)
- **Pipeline Results**: `pipeline_results` dict — scratchpad for intermediate data between steps

**2. Step Page Template** (every step follows this pattern)
```
Block 1: Green success box     (if step completed)
Block 2: Blue info box         (input stats from prev step)
Block 3: Yellow stats box      (processing stats)
Block 4: Data loading          (cache check → load or trigger processing)
Block 5: Processing button     (🚀 runs pipeline.step_N_func → caches → marks complete → rerun)
```

**3. Two Entry Paths**
- **Upload**: file → variable selection → preview → step 0 → step 1+
- **Cache load**: select dataset → `determine_max_step_from_cache()` → jump to last cached step

### Key Mechanisms
- **Cache-first**: Each step checks cache before processing
- **Cascade invalidation**: `invalidate_from_step(N)` clears steps N..end in both session state AND cache DB
- **Force recalc propagation**: `force_recalculate_from_step` flag checked by each page
- **Lazy-loaded utilities**: CacheManager, DataLoader created once per session
- **Verbose capture**: Wraps pipeline calls, saves logs to `exports/verbose_logs/`
- **Sidebar**: Navigation + step-specific settings + cache management

---

## What Needs to Change (~150 touch points)

### Step Pages: 11 → 9
| Old | New | Action |
|-----|-----|--------|
| Step 0: Upload | Step 0: Upload | keep |
| Step 1: Preprocessing | Step 1: Preprocessing | keep |
| Step 2: Quality Filter | Step 2: Quality Filter | keep |
| Step 3: Idea Extraction | Step 3: Idea Extraction | keep |
| Step 4: Embeddings | — | **DELETE** |
| Step 5: Categories | Step 4: Taxonomy | **REWRITE** (replaces embedding+clustering) |
| Step 6: Coding | Step 5: Codebook | **REWRITE** |
| Step 7: Theme ID | — | **DELETE** |
| Step 8: Code Assignment | Step 6: Code Assignment | **REWRITE** |
| Step 9: Export | Step 7: Export | **REWRITE** (placeholder) |
| Step 10: Results | Step 8: Results | renumber |

### Cache Keys to Update
- `"embeddings"` → remove
- `"code_assignment"` (old step 5) → `"taxonomy"`
- `"codebook_generation"` → remove
- `"codebook_refinement"` → remove
- `"codebook_refinement_enriched"` → remove
- `"code_assignment_direct"` → `"taxonomy_codes"`

### Model Refs to Remove
- `models.EmbeddingsModel`, `models.ThemeEnrichedCodebookModel`, `models.CodeRefinementResults`

### Dead Code to Delete
- `flatten_codebook_to_dataframe()`, `reconstruct_codebook_from_dataframe()`
- `from utils.codebookRefinement import get_refinement_report`

---

## Execution Phases

1. **ui_text.py** — Update STEP_NAMES dicts (NL + EN), remove steps 8-9
2. **Navigation & dispatch** — step_mapping dicts, dispatch branches, step limits
3. **Step page rewrites** — minimal pages for steps 4-7 (run button + results from cache)
4. **Cache & invalidation** — update `invalidate_from_step()`, cache key references
5. **Sidebar settings** — remove embedding/clustering/refinement, add taxonomy/codebook/assignment
6. **Dead code removal** — delete old pages, functions, imports
7. **Verify** — import check + manual Streamlit test

## Available Tooling
- **app-ui-agent**: Purpose-built for app.py (has context files for session state, step pages, navigation, bilingual UI)
- **code-quality-auditor**: Post-migration validation
- **utils-dependency-auditor**: Verify no broken imports

## Files to Modify
- `src/ui_text.py` — step names
- `src/app.py` — major rewrite
