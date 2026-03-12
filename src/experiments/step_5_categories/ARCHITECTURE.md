# Step 5 Categories v2 — Architecture & Data Flow

## Design Intent

Overhaul of v1's per-partition pipeline. Key changes:
- **Overlapping chunks** for theme discovery (no blind spots at chunk boundaries)
- **LLM-based theme consolidation** per partition (aggressive dedup of chunk overlap redundancy)
- **Reflexive Thematic Analysis** (Braun & Clarke, 2019) producing interpretive analytical themes with subthemes
- **Hierarchical codebook** as final output (themes → subthemes, recursive `MECECategory` tree)

## Pipeline Overview

```
Data loading:
  Step 3 cache --> IdeasExtractedModel[] + ExtractionMetadata

Partition discovery:
  Ideas --> group by concept_type --> PartitionSet + label mappings

Per partition (concurrent):
  Labels --> [Prompt 1] Chunked Theme Discovery (overlap) --> themes per chunk
         --> programmatic dedup (case-insensitive)
         --> [Prompt 1.5] Theme Consolidation (LLM) --> consolidated partition themes

Cross-partition (single call):
  All consolidated themes --> [Prompt 2] Reflexive Thematic Analysis
                          --> analytical themes + subthemes (hierarchical codebook)

Assignment:
  Ideas + Codebook --> [Prompt 3] Category Assignment --> assigned ideas
```

## Prompts

### Prompt 1: Chunked Theme Discovery (MAP)
- **Builder**: `build_theme_discovery_prompt()`
- **Response model**: `ThemeDiscoveryResult`
- **Input**: ~30-100 labels per chunk, with ~20% overlap between adjacent chunks
- **Output**: list of themes/insights for this chunk
- **Runs**: N chunks per partition, all concurrent
- **Taxonomy context**: dimension name/description (L1) + partition/domain name/definition (L2) from step 3
- **Batch sizing**: adaptive — `batch_size_min` (30) to `batch_size_max` (100), targeting ~15 chunks per partition

### Prompt 1.5: Theme Consolidation (REDUCE)
- **Builder**: `build_theme_consolidation_prompt()`
- **Response model**: `ConsolidatedThemesResult`
- **Input**: all programmatically-deduped themes from one partition (after Prompt 1)
- **Output**: aggressively consolidated theme list (expects 50-80% reduction)
- **Runs**: once per partition, all partitions concurrent
- **Purpose**: merge near-duplicates and paraphrases created by overlapping chunks into a clean, distinct set

### Prompt 2: Reflexive Thematic Analysis
- **Builder**: `build_thematic_analysis_prompt()`
- **Response model**: `ThematicAnalysisResult`
- **Input**: all consolidated themes from ALL partitions, grouped by partition
- **Output**: analytical themes with subthemes (`MECECategory` tree) + thematic map narrative
- **Runs**: once (cross-partition)
- **Purpose**: synthesize descriptive codes into interpretive analytical themes (Braun & Clarke)
- **Multi-phase prompt**: (1) interpretive memos, (2) name themes + subthemes, (3) operationalize for coding, (4) self-check, (5) thematic map

### Prompt 3: Category Assignment
- **Builder**: `build_category_assignment_prompt()` (batch mode) or `build_single_idea_assignment_prompt()` (single mode)
- **Response model**: `CategoryAssignmentBatch` or `SingleCategoryAssignment`
- **Input**: ideas + codebook leaf categories (from Prompt 2 output)
- **Runs**: batched per partition, concurrent
- **Modes**: "batch" (N ideas per call, hierarchical codebook) or "single" (one idea per call, flat codebook)

## Concurrency & Rate Limiting

Both `QualitativeResearcher` and `CategoryAssigner` use the same bootstrap pattern:
1. Fetch real rate limits from API response headers
2. Run 3 probe calls to measure avg latency and token usage
3. Compute optimal concurrency via Little's Law
4. Shared `asyncio.Semaphore` + `AsyncLimiter` for all LLM calls

## Key Differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Chunking | No overlap | ~20% overlap |
| Theme consolidation | Per-partition codebook+consolidate (2 LLM calls) | Programmatic dedup + LLM consolidation (1 LLM call) |
| Cross-partition synthesis | Per-partition MECE reduction | Global Reflexive Thematic Analysis |
| Output approach | Structural MECE categories | Interpretive analytical themes (Braun & Clarke) |
| Codebook depth | Flat codes | Hierarchical themes → subthemes |
| Prompt count | 3 + assignment | 4 + assignment |

## Files

- `prompts_exp.py` — prompt builders + Pydantic response models (all 4 prompts)
- `qualitative_researcher.py` — main orchestrator (prompts 1, 1.5, 2) with async concurrency
- `category_assignment.py` — assignment orchestrator (prompt 3) with queue-based workers + retry
- `config_categories_exp.py` — configuration (`CategoriesConfig`, `AssignmentConfig`)
- `models_exp.py` — data models (`PartitionSet`, `MECEResultsCache`, `CategoryAssignedModel`)
- `partition_discoverer.py` — partition ideas by domain, collect unique labels
- `partition_labels.py` — label extraction/formatting (ladder composites, prefixes)
- `run_experiment.py` — experiment runner (full pipeline + assignment-only mode)
- `debug_assignment_prompt.py` — debug helper for inspecting assignment prompts
- `debug_full_prompts.py` — debug helper for inspecting all pipeline prompts
- `view_assignments.py` — view/analyze assignment results
