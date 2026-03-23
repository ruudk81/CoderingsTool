# Job 1: Migrate Configs, Prompts, Models, and Utils

> Migrate development steps (1-7) into the production file structure. Development is the source of truth — no backward compatibility.

Last updated: 2026-03-23

---

## Key Finding: Naming Convention

The step-specific model files (`step_3_ideaExtractor/models_exp.py`, `step_4_classifier/models_classifier.py`, `step_6_codeAssigner/models_codeAssigner.py`) form a consistent chain using **production-style naming**: `interpretation`, `abstraction`, `domain`, `facet`, `attribute`. The root `development/models_exp.py` (with `rung_1`/`rung_2`/`concept_type`) is an older branch NOT imported by any step. **No rename needed.**

---

## Phase A: Rewrite `src/models.py`

The foundation — every utility reads/writes these models, so they must be settled first.

### What stays (used by dev steps)

**Base pipeline chain** (identical in production and step 3 dev):
- `ResponseModel` → `PreprocessedModel` → `QualityFilteredModel`
- `IdeasExtractedSubmodel` (idea_id, idea, instance, interpretation, abstraction, domain, facet, attribute, valence)
- `IdeasExtractedModel`

**Metadata:**
- `ExtractionMetadata` (from step 3: filename, var_name, template_prefix, lang, sector, topic, perspective, entity, intent, primary_dimension, domains)

**Taxonomy cache models** (from step 4):
- `DomainDescription`, `DomainSet` (partition definitions)
- `DomainResultModel` (step 4 version: facets, facet_assignments, attributes, attribute_assignments)
- `TaxonomyResultsCache` (partition_set, partition_results, label_counts, label_source)

**Codebook cache models** (from step 5):
- `CodingResultsCache` (extends taxonomy with raw_codes)

**Code assignment models** (from step 6):
- `CodeAssignment`, `CodeAssignmentBatch` (internal wrappers)
- `CodeAssignedSubmodel` (extends IdeasExtractedSubmodel — assigned_code, assigned_attribute, confidence, rationale, partition_name)
- `CodeAssignedModel` (extends IdeasExtractedModel)

**Codebook models** (used by step 5 codeGenerator):
- `CodebookEntry`, `CodebookModel` — keep if step 5 uses them, otherwise remove

### What gets removed

- `EmbeddingsSubmodel`, `EmbeddingsModel` — no standalone embedding step; embeddings computed on-the-fly
- `CodeAssignedSubmodel` (production version with assigned_category, category_confidence — replaced by step 6 version)
- `CodeAssignedModel` (production version extending EmbeddingsModel — replaced by step 6 version extending IdeasExtractedModel)
- `ClusterSubmodel`, `ClusterModel`, `AssignedIdeaSubmodel` — not in dev step chain
- `Codebook` (standalone class duplicating CodebookEntry with theme fields)
- `ThemeEnrichedCodebookEntry`, `ThemeEnrichedCodebookModel` — production-only refinement output
- `RefinedSubcode`, `RefinedCodebookCategory`, `RefinedCodebookModel` — old refinement pipeline
- `CodeTransformation`, `BatchTransformationRecord`, `RefinementLineage`, `CodeRefinementResults` — old lineage tracking
- `CodeDefinition` — unused
- Production `DomainResultModel` (MECE-focused version with categories/mece_verifications — replaced by step 4 taxonomy version)
- Production `CodingResultsCache` (replaced by step 5 version)
- Import of `DomainSet`, `MECECode`, `MECEVerification` from `prompts.py` — these move to models.py or get removed

### Model chain after rewrite

```
ResponseModel → PreprocessedModel → QualityFilteredModel
  → IdeasExtractedModel [IdeasExtractedSubmodel]
    → CodeAssignedModel [CodeAssignedSubmodel]  (step 6 output)

ExtractionMetadata (dataset-level, step 3)
DomainSet / DomainDescription (partition definitions, step 4)
TaxonomyResultsCache (taxonomy cache, step 4)
CodingResultsCache (codebook cache, step 5)
```

**Confirmed:** No `EmbeddingsModel`/`EmbeddingsSubmodel`. Embeddings are computed on-the-fly within steps that need them (classifier, codeAssigner). No separate embedding layer in the model chain.

### Files to modify
- `src/models.py` — rewrite

### Files to read before executing
- `src/development/step_6_codeAssigner/embedding_matcher.py` — understand how embeddings are used
- `src/development/step_6_codeAssigner/code_assignment.py` — check if it references EmbeddingsModel

---

## Phase B: Migrate Config Files

### Steps 1-3: Already in `config_steps/`
- `config_preprocess.py` — verify matches dev `step_1_preProcessor/config_exp.py`
- `config_qualityFilter.py` — verify matches dev `step_2_qualityFilter/config_exp.py`
- `config_ideaExtractor.py` — verify matches step 3 dev

### Steps 4-6: New config files to create/replace
- `config_steps/config_classifier.py` ← from `step_4_classifier/config_classifier.py`
- `config_steps/config_codeGenerator.py` ← from `step_5_codeGenerator/config_codeGenerator.py` (replaces existing)
- `config_steps/config_codeAssigner.py` ← from `step_6_codeAssigner/config_codeAssigner.py` (replaces existing)

### Config files to REMOVE from `config_steps/`
- `config_embedder.py` — no standalone embedding step
- `config_categories.py` — replaced by config_classifier.py

### Update `config_steps/__init__.py`
- Add new re-exports, remove dead ones

### Clean up `src/config.py`
- Remove any step-specific config that moved to config_steps/
- Keep only universal config (API, ModelConfig, CacheConfig, ProcessingConfig)
- Update `CacheConfig.step_prefixes` to match new step names

---

## Phase C: Migrate Prompt Files

### Steps 1-3: Already in `prompts_steps/`
- `prompts_spellChecker.py` — verify matches dev
- `prompts_qualityFilter.py` — verify matches dev
- `prompts_ideaExtractor.py` — verify matches dev

### Steps 4-6: New prompt files to create
- `prompts_steps/prompts_classifier.py` ← from `step_4_classifier/prompts_classifier.py`
- `prompts_steps/prompts_codeGenerator.py` ← from `step_5_codeGenerator/prompts_codeGenerator.py`
- `prompts_steps/prompts_codeAssigner.py` ← from `step_6_codeAssigner/prompts_codeAssigner.py`

### Update `prompts_steps/__init__.py`
- Add re-exports for new prompt files

### Retire `src/prompts.py`
- All step-specific prompts now live in `prompts_steps/`
- `prompts.py` can be emptied or deleted
- Models that were defined in prompts.py (DomainSet, MECECode, etc.) move to `models.py` or their respective `prompts_steps/` file

---

## Phase D: Migrate Utility Files

One step at a time. For each: copy dev file → fix imports → verify.

### Steps 1-3 (verify, minor tweaks)
- `src/utils/spellChecker.py` — diff with `step_1_preProcessor/spellChecker_exp.py`
- `src/utils/qualityFilter.py` — diff with `step_2_qualityFilter/qualityFilter_exp.py`
- `src/utils/ideaExtractor.py` — diff with `step_3_ideaExtractor/ideaExtractor_exp.py`
- Also: `textNormalizer.py`, `textFinalizer.py`

### Step 4 (new: classifier)
- `src/utils/classifier.py` ← `step_4_classifier/classifier.py`
- `src/utils/domain_discoverer.py` ← `step_4_classifier/domain_discoverer.py` (replaces existing)
- `src/utils/partition_labels.py` ← `step_4_classifier/partition_labels.py`
- `src/utils/dimension_data.py` ← `step_3_ideaExtractor/dimension_data.py` (verify if already matches)

### Step 5 (new: codeGenerator)
- `src/utils/codeGenerator.py` ← `step_5_codeGenerator/codebook_generator.py` (replaces existing 316KB file)

### Step 6 (new: codeAssigner)
- `src/utils/codeAssigner.py` ← `step_6_codeAssigner/code_assignment.py` (replaces existing 110KB file)
- `src/utils/embedding_matcher.py` ← `step_6_codeAssigner/embedding_matcher.py` (new)

### Utils to REMOVE (no longer used by dev pipeline)
- `src/utils/embedder.py` — no standalone embedding step
- `src/utils/map_reduce_mece.py` — replaced by classifier
- `src/utils/code_assignment.py` — replaced by codeAssigner
- `src/utils/category_assignment.py` — replaced by classifier
- `src/utils/partition_discoverer.py` — replaced by domain_discoverer
- `src/utils/codebookRefinement.py` — folded into step 5
- `src/utils/clusterer.py` — no clustering step
- `src/utils/exportVisualizer.py` — TBD with step 7
- `src/utils/speculativeStarterCodes.py` — production-only
- `src/utils/codegenPromptTester.py` — production-only debug
- `src/utils/assignPromptTester.py` — production-only debug
- `src/utils/codegenResults.py` — production-only debug
- `src/utils/exportCleaner.py` — production-only debug

### Import fixup pattern for ALL migrated utils
```python
# Replace dev-style:
from development.step_N_name.config_name import X
from development.step_N_name.models_name import Y
from development.step_N_name.prompts_name import Z

# With production-style:
from config_steps.config_name import X
from models import Y
from prompts_steps.prompts_name import Z
```

---

## Execution Order

```
1. Rewrite models.py (Phase A)
2. Migrate config files for steps 4-6 (Phase B)
3. Migrate prompt files for steps 4-6 (Phase C)
4. Migrate utils step-by-step (Phase D):
   a. Steps 1-3 (verify/diff, minor fixes)
   b. Step 4 (classifier — new)
   c. Step 5 (codeGenerator — new)
   d. Step 6 (codeAssigner — new)
5. Remove dead utils
6. Verify full import chain
```

Each step gets a git commit via the `migrate-exp-step` skill.

**After Job 1:** All production files reflect development source of truth. pipeline.py and app.py still reference old step functions — that's Job 2 and Job 3.

---

## Verification

After each phase:
```bash
cd src
python -c "import models; print('models OK')"
python -c "from config_steps import *; print('configs OK')"
python -c "from prompts_steps import *; print('prompts OK')"
```

After Phase D (all utils migrated):
```bash
python -c "from utils.spellChecker import SpellChecker; print('step 1 OK')"
python -c "from utils.qualityFilter import Grader; print('step 2 OK')"
python -c "from utils.ideaExtractor import IdeaExtractor; print('step 3 OK')"
python -c "from utils.classifier import TaxonomyClassifier; print('step 4 OK')"
python -c "from utils.codeGenerator import CodebookGenerator; print('step 5 OK')"  # class name TBD
python -c "from utils.codeAssigner import CodeAssigner; print('step 6 OK')"
```
