# The Big Migration: Development → Production

> **In plain terms:** We have a working experimental pipeline (steps 1-7 in `src/development/`) that represents the future of CoderingsTool. The production pipeline (`pipeline.py`, `app.py`, `src/utils/`, etc.) still runs the old architecture. Our job is to make the experimental pipeline *become* the production pipeline — cleanly, without dragging along legacy clutter, dead code, or backward-compatibility hacks. This is greenfield-in-disguise: we're not patching old code, we're replacing it with a tested new design.

Last updated: 2026-03-23

---

## Guiding Principles

1. **Development steps 1-7 are the source of truth.** When production and development disagree, development wins.

2. **Clean slate, not backward compatibility.** No bridge mappings, no legacy field aliases, no dead code carried forward "just in case." If something isn't needed by the new pipeline, it doesn't exist.

3. **Inconsistent terminology gets resolved, not bridged.** Where development uses different field names (e.g., `rung_1` vs `interpretation`, `concept_type` vs `domain`), we pick one name and use it everywhere — not both with a mapping layer.

4. **We are in development, not production.** This is our window to get the architecture right. Shortcuts we take now become permanent tech debt.

---

## The Three Major Jobs

### Job 1: Migrate Configs, Prompts, Models, and Utils

Move experimental code from `src/development/step_N_<name>/` into the production file structure.

**What moves where:**

| Experimental file | Production destination |
|---|---|
| `config_exp.py` / `config_<name>.py` | `src/config_steps/config_<stepname>.py` |
| `prompts_exp.py` / `prompts_<name>.py` | `src/prompts_steps/prompts_<stepname>.py` |
| `<name>_exp.py` (main utility) | `src/utils/<name>.py` |
| `models_exp.py` / `models_<name>.py` | `src/models.py` (update) |
| `dimension_data.py`, `partition_labels.py`, `embedding_matcher.py` (helpers) | `src/utils/` |
| `run_experiment.py`, `debug_*.py`, `view_*.py` | NOT migrated — stays in development |

**Per-step inventory:**

#### Steps 1-3 (largely same as production)

These steps have only minor changes (import paths, small behavioral tweaks). Quick wins.

| Step | Key files to migrate | Main change |
|---|---|---|
| 1 (PreProcessor) | `spellChecker_exp.py`, `config_exp.py`, `prompts_exp.py` | Import cleanup, minor behavioral changes |
| 2 (QualityFilter) | `qualityFilter_exp.py`, `config_exp.py`, `prompts_exp.py` | Import cleanup |
| 3 (IdeaExtractor) | `ideaExtractor_exp.py`, `prompts_exp.py`, `models_exp.py`, `dimension_data.py` | Model field changes (see below) |

#### Step 4 (Classifier) — NEW

Entirely new step. No production equivalent to replace — this is additive.

| File | Destination |
|---|---|
| `classifier.py` | `src/utils/classifier.py` (new) |
| `domain_discoverer.py` | `src/utils/domain_discoverer.py` (replaces existing) |
| `partition_labels.py` | `src/utils/partition_labels.py` (new or replaces existing) |
| `config_classifier.py` | `src/config_steps/config_classifier.py` |
| `prompts_classifier.py` | `src/prompts_steps/prompts_classifier.py` |
| `models_classifier.py` | `src/models.py` (integrate `DomainSet`, `DomainDescription`, `DomainResultModel`, `TaxonomyResultsCache`) |

#### Step 5 (CodeGenerator) — NEW

| File | Destination |
|---|---|
| `codebook_generator.py` | `src/utils/codeGenerator.py` (replaces) |
| `config_codeGenerator.py` | `src/config_steps/config_codeGenerator.py` (replaces) |
| `prompts_codeGenerator.py` | `src/prompts_steps/prompts_codeGenerator.py` (new) |
| `models_codeGenerator.py` | `src/models.py` (integrate `CodingResultsCache`) |

#### Step 6 (CodeAssigner) — NEW

| File | Destination |
|---|---|
| `code_assignment.py` | `src/utils/codeAssigner.py` (replaces) |
| `embedding_matcher.py` | `src/utils/embedding_matcher.py` (new) |
| `config_codeAssigner.py` | `src/config_steps/config_codeAssigner.py` (replaces) |
| `prompts_codeAssigner.py` | `src/prompts_steps/prompts_codeAssigner.py` (new) |
| `models_codeAssigner.py` | `src/models.py` (integrate `CodeAssignment`, `CodeAssignedSubmodel`, `CodeAssignedModel`) |

#### Step 7 (Export) — TODO

Not yet developed. Will be built fresh.

**Model consolidation (critical cross-cutting concern):**

`src/models.py` needs a significant update. The development models (`models_exp.py` + per-step `models_*.py`) diverge from production:

| Area | Production (`models.py`) | Development (`models_exp.py` etc.) | Resolution |
|---|---|---|---|
| Abstraction ladder | `interpretation`, `abstraction` | `rung_1`, `rung_2` | Pick one naming convention |
| Taxonomy L2 | `domain` | `concept_type` | Pick one |
| Embedding fields | `interpretation_embedding`, `abstraction_embedding`, `facet_embedding`, `domain_embedding` | `rung_1_embedding`, `rung_2_embedding`, `concept_type_embedding` | Align with ladder/taxonomy naming |
| Code assignment | `assigned_category`, `category_confidence`, `category_rationale` | `assigned_code`, `assigned_attribute`, `confidence`, `rationale` | Use dev version |
| Model chain | `EmbeddingsModel → CodeAssignedModel` (direct) | `EmbeddingsModel → ClusterModel → CodeAssignedModel` (via cluster) | Decide if ClusterModel stays |
| ExtractionMetadata | `primary_dimension`, `domains`, `sector` | `primary_facet`, `concept_types`, `domain` | Use dev version |
| Taxonomy cache | `DomainResultModel` (with MECE fields) | `DomainResultModel` (with facets + attributes) | Use dev version |
| Codebook extensions | — | `CodebookExp`, `ThemeEnrichedCodebookEntryExp`, `ThemeEnrichedCodebookModelExp` | Integrate or replace |

**Decision needed:** Settle on consistent field names BEFORE migrating utilities, since every utility reads/writes these models.

---

### Job 2: Update `pipeline.py`

Rewrite the step functions in `pipeline.py` to call the new utilities.

**What changes:**

| Current step | Current function | New function | Notes |
|---|---|---|---|
| 0 (data) | `step_0_load_data()` | unchanged | |
| 1 (preprocessed) | `step_1_preprocess()` | minor import updates | |
| 2 (quality_filter) | `step_2_quality_filter()` | minor import updates | |
| 3 (extracted_ideas) | `step_3_extract_ideas()` | update for new models | |
| 4 (embeddings) | `step_4_generate_embeddings()` | update embedding fields to match new model | |
| 5 (classifier) | currently `step_4_classNcoder()` | **REPLACE** with new `step_4_classify()` calling `TaxonomyClassifier` | |
| 6 (codebook) | `step_6_generate_codebook()` | **REPLACE** with new `step_5_generate_codebook()` calling `CodebookGenerator` | |
| 7 (refinement) | `step_7_refine_codebook()` | **REMOVE** — folded into step 5 | |
| 8 (code assignment) | `step_8_assign_codes()` | **REPLACE** with new `step_6_assign_codes()` calling dev `CodeAssigner` | |
| 9 (export) | `step_9_export_results()` | **REPLACE** with new `step_7_export()` — needs development | |

**`STEP_NAMES` will change:**

```python
# Current (10 steps, 0-9)
STEP_NAMES = {
    0: "data", 1: "preprocessed", 2: "quality_filter",
    3: "extracted_ideas", 4: "embeddings",
    5: "code_assignment", 6: "codebook_generation",
    7: "codebook_refinement", 8: "code_assignment_direct",
    9: "export"
}

# New (8 steps, 0-7)
STEP_NAMES = {
    0: "data", 1: "preprocessed", 2: "quality_filter",
    3: "extracted_ideas", 4: "taxonomy",
    5: "codebook", 6: "code_assignment",
    7: "export"
}
```

> **Note:** Whether step 4 (embeddings) stays as a separate step or gets folded into step 3/4 is an open question. The dev pipeline currently doesn't have a standalone embedding step — embeddings are generated as part of the classifier's domain discovery and the code assigner's candidate matching.

**Cache key implications:**
- `CacheConfig.step_prefixes` needs updating to match new step names
- Old cache files (with old prefixes/names) become orphaned — not harmful, just stale
- See `CACHE_LOGIC.md` for full prefix mapping

---

### Job 3: Update `app.py`

Rewrite the Streamlit UI to match the new pipeline.

**What changes:**

| Area | Change |
|---|---|
| Step count | 10 → 8 steps (0-7) |
| Step page functions | Remove `render_step_5/6/7/8_page()`, replace with new `render_step_4/5/6/7_page()` |
| STEP_NAMES | Must match pipeline.py |
| Session state | Update `pipeline_results` keys, `completed_steps` range, `max_step_reached` |
| `invalidate_from_step()` | Update cascade logic for new step numbering |
| Cache interactions | Update step_name strings for save/load calls |
| UI text | Update `ui_text.py` step labels (Dutch + English) |
| Config display | Show new step-specific configs (ClassifierConfig, CodebookConfig, AssignmentConfig) |
| Progress tracking | Adjust for 8 steps instead of 10 |

**Approach:** This is a large file (254KB). Best tackled step-page by step-page, after pipeline.py is working.

---

## Execution Strategy

### Recommended order

```
Phase A: Foundation (models + config)
  1. Settle naming conventions (resolve model field divergences)
  2. Update models.py with new model definitions
  3. Migrate config files to config_steps/
  4. Migrate prompt files to prompts_steps/

Phase B: Utilities (one step at a time)
  5. Migrate step 1 utils (quick win, validates the process)
  6. Migrate step 2 utils
  7. Migrate step 3 utils (includes model changes)
  8. Migrate step 4 utils (new — classifier)
  9. Migrate step 5 utils (new — codeGenerator)
  10. Migrate step 6 utils (new — codeAssigner)

Phase C: Pipeline orchestration
  11. Rewrite pipeline.py step functions + STEP_NAMES
  12. Update cache config (prefixes, step names)
  13. Test full pipeline run

Phase D: UI
  14. Update app.py step pages
  15. Update ui_text.py
  16. Test full app walkthrough

Phase E: Cleanup
  17. Remove dead production code (old utils, old prompts)
  18. Remove backward-compatibility bridges from models.py
  19. Audit utils/ for orphaned files (use utils-dependency-auditor agent)
```

### Per-step migration pattern

For each utility step, follow this repeatable pattern:

1. **Diff** experimental vs production to understand scope
2. **Migrate config** → `config_steps/`
3. **Migrate prompts** → `prompts_steps/`
4. **Migrate utility** → `src/utils/`, fix imports
5. **Update models.py** if step introduces new model fields
6. **Verify imports** (`python -c "from utils.<name> import ..."`)
7. **Commit**

---

## Available Tooling

### Existing agents and skills (`.claude/`)

| Tool | Type | Purpose | Useful for |
|---|---|---|---|
| `migrate-exp-step` | Skill | Migrate one experimental step to production (config, prompts, utils, imports, verification) | **Core tool for Phase B** — handles the repeatable per-step migration pattern |
| `migrate-output-schema` | Skill | Migrate JSON output instructions from prompts into Pydantic Field descriptions | Prompt cleanup during Phase A/B |
| `app-ui-agent` | Agent | Modify Streamlit app — UI changes, session state, navigation, step pages | **Core tool for Phase D** |
| `config-parameter-curator` | Agent | Audit and centralize hardcoded params into config.py | Phase A config cleanup |
| `utils-dependency-auditor` | Agent | Find unused/redundant utilities | **Phase E cleanup** |
| `code-quality-auditor` | Agent | Audit single Python file for issues | Post-migration quality checks |
| `azure-client-migrator` | Agent | Migrate files to centralized llm.py | Not needed (already done) |
| `responses-api-migrator` | Agent | Migrate to OpenAI Responses API | Not needed (already done) |

### Agents/skills we might want to develop

| Tool | Purpose | Why |
|---|---|---|
| **model-field-migrator** | Rename model fields across the codebase (models.py + all consumers) | The naming convention resolution (Job 1) touches every file that reads/writes Pydantic models. A dedicated tool could automate the find-and-replace across utils, pipeline, app, prompts. |
| **pipeline-step-rewriter** | Rewrite a pipeline.py step function from an experimental `run_experiment.py` | Each `run_experiment.py` contains the orchestration logic that needs to become a `step_N_*()` function in pipeline.py. A skill could automate the translation. |
| **cache-prefix-updater** | Update CacheConfig step_prefixes and all cache read/write calls | When step names/numbers change, every `save_to_cache(step_name=...)` and `load_from_cache(step_name=...)` call needs updating. A targeted tool could handle this. |

---

## Resolved Questions

> **Core principle applied to all:** The development steps are the source of truth. If something exists in production but not in development, it gets removed. If development does it differently, development wins.

1. **Embedding step:** No standalone embedding step. Development doesn't have one — embeddings are generated on-demand within steps 4 and 6. The production-only `embedder.py` / step 4 (embeddings) gets removed.

2. **Field naming convention:** Development naming wins. `rung_1`/`rung_2` (not `interpretation`/`abstraction`), `concept_type` (not `domain`). Production models.py gets rewritten to match development models.

3. **ClusterModel:** If development doesn't use it, it goes. No backward-compatibility bridging.

4. **Step 7 (export):** Still needs development — open item, not a migration question.

5. **`prompts.py` vs `prompts_steps/`:** ALL step-specific prompts go in `prompts_steps/` (one file per step). ALL step-specific configs go in `config_steps/` (one file per step). Keeps hardcoded parameters easy to find. The monolithic `prompts.py` gets emptied as its contents move to per-step files.

## Backlog

1. **Step 7 (export):** Needs to be designed and built, not migrated. Will be addressed after steps 1-6 are in production.

---

## Reference Documents

- **[PIPELINE.md](../../PIPELINE.md)** — Full architecture: current production pipeline, development steps, model chain, file organization
- **[CACHE_LOGIC.md](../../CACHE_LOGIC.md)** — Cache system: key generation, prefix mapping, invalidation, per-step cache inventory
- **[CLAUDE.md](../../CLAUDE.md)** — Project overview, environment setup, development notes
- **[.claude/skills/migrate-exp-step/SKILL.md](../../.claude/skills/migrate-exp-step/SKILL.md)** — Detailed per-step migration procedure
