# Work To Be Done — step_4_classifier, step_5_codeGenerator & step_6_codeAssigner

## 1. Step 4: Add growing model cache output

Currently step 4 (classifier) only caches `TaxonomyResultsCache` as **metadata** (single object). It does not produce a growing model (`List[PydanticModel]`) — the growing model is only written after step 6 (assignment) as `List[CodeAssignedModel]`.

**To do:** Introduce a growing model output for step 4 so that downstream consumers can load step 4 results without running step 5/6. This would be a `List[TaxonomyClassifiedModel]` (or similar) extending `IdeasExtractedModel` with facet (L3) and attribute (L4) fields populated by P3/P6.

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

Reference: `old steps/step_4_classNcoder/ARCHITECTURE.md` (sections covering P1-P7)

## 3. Write ARCHITECTURE.md for step_5_codeGenerator

Create `step_5_codeGenerator/ARCHITECTURE.md` covering:
- Design intent (codebook generation, P8-P9)
- Pipeline overview (P8-P9 only)
- Input contract (TaxonomyResultsCache from step 4)
- Prompt builders and response models (P8-P9)
- Key data structures (CodebookResult, CodingResultsCache)
- Concurrency & rate limiting
- Configuration (CodebookConfig)
- File listing

Reference: `old steps/step_4_classNcoder/ARCHITECTURE.md` (sections covering P8-P9)

## 4. Write ARCHITECTURE.md for step_6_codeAssigner

Create `step_6_codeAssigner/ARCHITECTURE.md` covering:
- Design intent (code assignment, P10)
- Pipeline overview (P10 only)
- Input contract (CodingResultsCache from step 5 + ideas from step 3)
- Code assignment flow (embedding pre-filter, LLM assignment)
- Key data structures (CodeAssignedModel, CodeAssignedSubmodel)
- Concurrency & rate limiting (4-layer stack)
- Configuration (AssignmentConfig)
- File listing

Reference: `old steps/step_4_classNcoder/ARCHITECTURE.md` (sections covering P10)

## 5. Write CACHE_LOGIC.md for step_4_classifier

Create `step_4_classifier/CACHE_LOGIC.md` covering:
- Input: step 3 cache (2 files: extracted_ideas + metadata)
- Output: TaxonomyResultsCache (prefix 005)
- Output: growing model (once item 1 above is done)
- Cache key scheme

Reference: `old steps/step_4_classNcoder/CACHE_LOGIC.md` (taxonomy sections)

## 6. Write CACHE_LOGIC.md for step_5_codeGenerator

Create `step_5_codeGenerator/CACHE_LOGIC.md` covering:
- Input: step 4 taxonomy cache (prefix 005)
- Output: CodingResultsCache / mece_codes (prefix 006)
- Cache key scheme

Reference: `old steps/step_4_classNcoder/CACHE_LOGIC.md` (codebook sections)

## 7. Write CACHE_LOGIC.md for step_6_codeAssigner

Create `step_6_codeAssigner/CACHE_LOGIC.md` covering:
- Input: step 5 mece_codes cache (prefix 006) + step 3 ideas (prefix 004)
- Output: List[CodeAssignedModel] / taxonomy_codes (prefix 007)
- How the growing model is assembled in _build_output()

Reference: `old steps/step_4_classNcoder/CACHE_LOGIC.md` (assignment sections)
