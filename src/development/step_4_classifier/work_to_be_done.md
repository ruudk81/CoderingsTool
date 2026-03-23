# Work To Be Done — step_4_classifier & step_5_codebookGenerator

## 1. Step 4: Add growing model cache output

Currently step 4 (classifier) only caches `TaxonomyResultsCache` as **metadata** (single object). It does not produce a growing model (`List[PydanticModel]`) — the growing model is only written after step 5 (assignment) as `List[CodeAssignedModel]`.

**To do:** Introduce a growing model output for step 4 so that downstream consumers can load step 4 results without running step 5. This would be a `List[TaxonomyClassifiedModel]` (or similar) extending `IdeasExtractedModel` with facet (L3) and attribute (L4) fields populated by P3/P6.

- Define the new model in `models_classifier.py`
- Add `save_to_cache()` call in `run_experiment.py` after taxonomy completes
- Register a cache key + prefix in `config.py` (e.g. `"taxonomy_classified": "005"`)

## 2. Write ARCHITECTURE.md for step_4_classifier

Create `step_4_classifier/ARCHITECTURE.md` covering:
- Design intent (taxonomy discovery, P1-P7)
- Taxonomy structure (L1-L4)
- Pipeline overview (P1-P7 only)
- Dimension-specific prompt injection
- Prompt builders and response models (P1-P7)
- Data flow (what step 3 provides, what step 4 outputs)
- Key data structures (PartitionLabelMapping, DomainResult, TaxonomyResult)
- Concurrency & rate limiting
- Configuration (CategoriesConfig)
- File listing

Reference: `step_4_classNcoder/ARCHITECTURE.md` (sections covering P1-P7)

## 3. Write ARCHITECTURE.md for step_5_codebookGenerator

Create `step_5_codebookGenerator/ARCHITECTURE.md` covering:
- Design intent (codebook generation + code assignment, P8-P10)
- Pipeline overview (P8-P10)
- Input contract (TaxonomyResultsCache from step 4)
- Prompt builders and response models (P8-P10)
- Code assignment flow (embedding pre-filter, LLM assignment)
- Key data structures (CodebookResult, CodingResultsCache, CodeAssignedModel)
- Concurrency & rate limiting (4-layer stack for P10)
- Configuration (CodebookConfig, AssignmentConfig)
- File listing

Reference: `step_4_classNcoder/ARCHITECTURE.md` (sections covering P8-P10)

## 4. Write CACHE_LOGIC.md for step_4_classifier

Create `step_4_classifier/CACHE_LOGIC.md` covering:
- Input: step 3 cache (2 files: extracted_ideas + metadata)
- Output: TaxonomyResultsCache (prefix 005)
- Output: growing model (once item 1 above is done)
- Cache key scheme

Reference: `step_4_classNcoder/CACHE_LOGIC.md` (taxonomy sections)

## 5. Write CACHE_LOGIC.md for step_5_codebookGenerator

Create `step_5_codebookGenerator/CACHE_LOGIC.md` covering:
- Input: step 4 taxonomy cache + step 3 ideas
- Intermediate: CodingResultsCache / mece_codes (prefix 006)
- Output: List[CodeAssignedModel] / taxonomy_codes (prefix 006)
- RUN_MODE and which caches each mode reads/writes
- How the growing model is assembled in _build_output()

Reference: `step_4_classNcoder/CACHE_LOGIC.md` (codebook + assignment sections)
