# Step 4 classNcoder — Cache Logic

## Caching in CoderingsTool: Core Logic

### Two types of cached data

1. **Growing per-respondent model** — A pickle file per pipeline step containing a `List[PydanticModel]`. Each step's model extends the previous via Pydantic inheritance, so each pickle is a cumulative superset of all prior steps:

   ```
   ResponseModel (step 0)
     → PreprocessedModel (step 1)
       → QualityFilteredModel (step 2)
         → IdeasExtractedModel (step 3)
           → CodeAssignedModel (step 4)
   ```

2. **Standalone metadata files** — Single Pydantic model objects (not lists) containing dataset-level or pipeline-level reference data (codebooks, taxonomy inventories, extraction context).

### Storage mechanics

- All cache files are pickles in `<project_root>/data/cache/`
- Filename pattern: `{prefix}_{step}_{datafile}_{variable_key}.pkl`
- Prefixes are defined in `CacheConfig.step_prefixes` in `config.py`
- A SQLite database (`cache.db`) indexes all entries by `(filename, step_name, variable_key)` and stores the full file path
- **On save**: prefix builds the filename → pickle written → path recorded in SQLite
- **On load**: SQLite lookup by `(filename, step_name, variable_key)` → returns stored path → opens pickle. The prefix is not involved on read — it's looked up from the database.

### Two save/load methods

| Method | Input | Cache key suffix | Use case |
|---|---|---|---|
| `save_to_cache()` / `load_from_cache()` | `List[PydanticModel]` | none | Growing per-respondent data |
| `save_metadata_to_cache()` / `load_metadata_from_cache()` | single `PydanticModel` | `"_metadata"` appended | Standalone reference data |

Both use pickle. The `_metadata` suffix prevents cache key collisions so a step can store both a per-respondent list and a summary object.

---

## Step 4 classNcoder: What is loaded and stored

### Input: reads from step 3 cache (2 files)

| Cache key | Model | Method | Contents |
|---|---|---|---|
| `"extracted_ideas"` | `List[IdeasExtractedModel]` | `load_from_cache()` | Per-respondent → per-idea data (ladder, domain, facet hint, valence) |
| `"extracted_ideas_metadata"` | `ExtractionMetadata` | `load_metadata_from_cache()` | Dataset-level context (dimension, domains, lang, survey question) |

### Intermediate checkpoints (2 metadata files)

| Cache key | Model | Written after | Contents |
|---|---|---|---|
| `"taxonomy_metadata"` | `TaxonomyResultsCache` | P1-P7 | Facet/attribute inventories + assignment dicts per domain |
| `"mece_codes_metadata"` | `CodingResultsCache` | P8-P9 | Taxonomy + codebook (`raw_codes`) |

These checkpoints enable modular re-execution via `RUN_MODE` in `run_experiment.py`:

| RUN_MODE | Reads | Runs | Writes |
|---|---|---|---|
| `"taxonomy"` | step 3 cache (2 files) | P1-P7 | `taxonomy_metadata` |
| `"codebook"` | `taxonomy_metadata` | P8-P9 | `mece_codes_metadata` |
| `"assignment"` | step 3 cache + `mece_codes_metadata` | P10 | `taxonomy_codes` |
| `"all"` | step 3 cache (2 files) | P1-P10 | all three |

### Final output (1 growing-model file)

| Cache key | Model | Method | Contents |
|---|---|---|---|
| `"taxonomy_codes"` | `List[CodeAssignedModel]` | `save_to_cache()` | Steps 0-4 cumulative |

Fields added by step 4 on top of step 3's `IdeasExtractedSubmodel`:

| Field | Source phase | Description |
|---|---|---|
| `facet` | P3 | **Overwrites** step 3's facet hint with actual assignment |
| `assigned_attribute` | P6 | Attribute name (L4) |
| `assigned_code` | P10 | Code name from codebook |
| `confidence` | P10 | Assignment confidence (0.0–1.0) |
| `rationale` | P10 | Brief rationale for code assignment |
| `partition_name` | P10 | Domain key (normalized form of `domain`) |

---

## How the growing model is assembled

The growing model is **not built incrementally** as P3, P6, and P10 complete. It is assembled **once**, at the very end, in `CodeAssigner._build_output()` (`code_assignment.py:1187`).

### Why not incremental?

P3 and P6 results are produced inside `QualitativeResearcher` (the taxonomy pipeline), while P10 runs in a separate `CodeAssigner`. These are different classes, potentially run in different sessions (via `RUN_MODE`). Rather than mutating the step 3 models in place across pipeline stages, each stage stores its results as lightweight dicts, and a single assembly step merges everything at the end.

### Where do the in-memory dicts live?

During pipeline execution, the P3/P6 results exist as dicts inside `DomainResult` (per-domain dataclass in `qualitative_researcher.py`):

```python
@dataclass
class DomainResult:
    facet_assignments: Dict[str, str]       # idea_id → facet_name (from P3)
    attribute_assignments: Dict[str, str]   # idea_id → attribute_name (from P6)
    ...
```

These are per-domain. When the taxonomy/codebook pipeline finishes, these dicts are serialized into `DomainResultModel` and stored in the checkpoint caches (`TaxonomyResultsCache` / `CodingResultsCache`).

### How do dicts survive between independent runs?

When `RUN_MODE = "assignment"` runs independently, it **reconstructs the dicts from the checkpoint cache**. In `run_experiment.py:run_assignment_only()` (line 930):

```python
# Load the MECE checkpoint from cache
mece_cache = load_mece_cache()
pydantic_results = mece_cache.partition_results  # Dict[str, DomainResultModel]

# Reconstruct the attribute_assignments dict from all domains
all_attr_assignments = {}
for domain_result in pydantic_results.values():
    all_attr_assignments.update(domain_result.attribute_assignments)

# Pass into CodeAssigner
assigned_results = run_code_assignment(
    ...,
    attribute_assignments=all_attr_assignments,
)
```

Similarly, the facet assignments are reconstructed inside `CodeAssigner._assign_all_async()` (line 252):

```python
# Build facet lookup from DomainResultModel.facet_assignments
self._facet_lookup = {}
for name, mece_res in self._mece_results.items():
    if mece_res.facet_assignments:
        self._facet_lookup.update(mece_res.facet_assignments)
```

### The assembly step

`_build_output()` (`code_assignment.py:1187`) iterates over the original step 3 `IdeasExtractedModel` list and, for each idea:

1. **Copies all step 0-3 fields** via `idea.model_dump()`
2. **Looks up facet** from `self._facet_lookup` (P3 assignments, reconstructed from cache or in-memory)
3. **Looks up attribute** from `self._attribute_assignments` (P6 assignments, passed in from cache or in-memory)
4. **Looks up code, confidence, rationale** from the P10 assignment results (just completed)
5. **Wraps** everything into a `CodeAssignedSubmodel`

The resulting `List[CodeAssignedModel]` is then saved to cache under key `"taxonomy_codes"`.

### Data flow diagram

```
TAXONOMY RUN (P1-P7)                    CODEBOOK RUN (P8-P9)
  QualitativeResearcher                   QualitativeResearcher
    → DomainResult per domain               → PipelineResult.codes
      .facet_assignments (dict)
      .attribute_assignments (dict)
            │                                       │
            ▼                                       ▼
    TaxonomyResultsCache              CodingResultsCache
    (checkpoint pickle)               (checkpoint pickle)
            │                                       │
            └──────────────┬────────────────────────┘
                           │
                    ASSIGNMENT RUN (P10)
                      CodeAssigner.__init__()
                        ← mece_results (from cache: facet + attribute dicts)
                        ← attribute_assignments (from cache: merged across domains)
                        ← ideas_models (from step 3 cache)
                        ← codes (from cache: codebook)
                           │
                      _assign_all_async()
                        → builds _facet_lookup from mece_results
                        → runs P10 LLM calls
                        → assignment_lookup (idea_id → code)
                           │
                      _build_output()
                        → merges step 3 data + facet + attribute + code
                        → List[CodeAssignedModel]
                           │
                           ▼
                    "taxonomy_codes" pickle
                    (growing model: steps 0-4)
```

---

## Cache file prefix scheme

All cache files follow the pattern `{prefix}_{step_name}_{datafile}_{variable_key}.pkl` and live in `data/cache/`. Prefixes are defined in `CacheConfig.step_prefixes` in `config.py`.

| Prefix | Step name | Type | Pipeline step |
|---|---|---|---|
| `001` | `data` | Growing model | Step 0: load data |
| `002` | `preprocessed` | Growing model | Step 1: spell check |
| `003` | `quality_filter` | Growing model | Step 2: quality filter |
| `004` | `extracted_ideas` | Growing model | Step 3: idea extraction |
| `004` | `extracted_ideas_metadata` | Metadata | Step 3: extraction context |
| `005` | `taxonomy_metadata` | Checkpoint | Step 4: P1-P7 taxonomy |
| `005` | `mece_codes_metadata` | Checkpoint | Step 4: P8-P9 codebook |
| `005` | `taxonomy_codes` | Growing model | Step 4: P10 final output |
