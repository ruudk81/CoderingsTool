# Configuration Parameter Audit Report
**Date:** 2025-10-21
**Project:** CoderingsTool
**Purpose:** Comprehensive audit of hardcoded configuration parameters across the codebase

---

## Executive Summary

This audit identified **52 distinct configuration parameters** scattered across 24 active utility files. The codebase shows a **well-structured configuration architecture** with centralized config.py containing most critical parameters in dedicated dataclass configurations.

### Key Findings

1. **Already Centralized:** 45 parameters properly managed in config.py
2. **Migration Candidates:** 7 parameters hardcoded across 5 utility files
3. **Acceptable Hardcoding:** Mathematical constants and debug flags

### Architecture Assessment

✅ **STRENGTHS:**
- Excellent dataclass-based configuration structure
- Clear separation of concerns (CacheConfig, ProcessingConfig, ClusteringConfig, etc.)
- Comprehensive rate limiting configuration with OpenAI model-specific limits
- Embedding model dimension mapping

⚠️ **IMPROVEMENT AREAS:**
- Rate limiting headroom (HEADROOM = 0.9) duplicated 15+ times across utils
- Concurrency caps/minimums duplicated in multiple utility files
- Timeout bounds duplicated across async processors
- Latency tracker configuration hardcoded

---

## Category 1: Already in config.py (COMPLIANT) ✅

### 1.1 Cache Configuration (`CacheConfig`)
**Location:** `src/config.py` lines 335-397

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_cache_age_days` | 30 | Cache validity period |
| `auto_cleanup` | True | Automatic cache cleanup |
| `use_atomic_writes` | True | Atomic file operations |
| `batch_size` | 1000 | Cache batch size |
| `verbose` | False | Cache logging verbosity |

**Status:** ✅ Well-organized, platform-aware (disables atomic writes on Windows)

---

### 1.2 Spell Checking Configuration (`SpellCheckConfig`)
**Location:** `src/config.py` lines 404-459

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 20 | API request batch size |
| `temperature` | 0.0 | LLM temperature |
| `max_tokens` | 4000 | Max completion tokens |
| `retries` | 3 | Retry attempts |
| `retry_delay` | 2 | Seconds between retries |
| `max_batch_size` | 5 | Maximum batch size |
| `completion_reserve` | 1000 | Reserved tokens for completion |
| `cache_size` | 10000 | Spell check cache size |
| `spacy_batch_size` | 64 | spaCy processing batch size |
| `repeated_char_threshold` | 5 | Repeated character detection |
| `max_correction_examples` | 10 | Verbose output examples |
| `seed` | 42 | Random seed |
| `context_chars` | 20 | Context for spell checking |
| `max_concurrent_requests` | 5 | API concurrency |
| `max_words_to_check` | 100000 | Max word threshold |
| `enable_word_frequency_cache` | True | Cache common words |
| `progress_report_interval` | 10000 | Progress reporting |
| `max_unique_oov_words` | 5000 | OOV word limit |
| `enable_early_termination` | True | Early termination flag |
| `max_concurrent_suggestion_chunks` | 20 | Chunk concurrency |
| `max_words_per_chunk` | 1200 | Words per chunk |
| `enable_adaptive_chunking` | True | Adaptive chunking |
| `chunk_progress_reporting` | True | Chunk progress |
| `suggestion_processing_semaphore_limit` | 100 | Semaphore limit |
| `minimum_timeout_seconds` | 15.0 | Minimum timeout |
| `maximum_timeout_seconds` | 60.0 | Maximum timeout |
| `hunspell_concurrent_sessions` | 20 | Hunspell sessions |
| `hunspell_batch_size` | 1000 | Hunspell batch size |
| `enable_streaming_oov_detection` | True | Streaming OOV |
| `oov_detection_queue_size` | 10000 | Queue size |
| `rate_limit_safety_factor` | 0.95 | Safety factor |
| `rate_limit_utilization` | 0.98 | Utilization target |
| `concurrent_burst_multiplier` | 3.0 | Burst capacity |
| `enable_suggestion_pre_validation` | True | Pre-validation |
| `disable_pre_validation_above_oov_words` | 2000 | Pre-validation threshold |
| `enable_suggestion_caching` | True | Suggestion caching |
| `hunspell_pool_size` | 20 | Hunspell pool size |
| `ultra_batch_threshold` | 1000 | Ultra batch threshold |
| `ultra_batch_size` | 10000 | Ultra batch size |

**Status:** ✅ Extremely comprehensive configuration for complex async spell checking

---

### 1.3 Quality Filter Configuration (`QualityFilterConfig`)
**Location:** `src/config.py` lines 465-481

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 20 | Batch size |
| `temperature` | 0.0 | LLM temperature |
| `max_tokens` | 4000 | Max tokens |
| `retries` | 3 | Retry attempts |
| `instructor_retries` | 3 | Instructor retries |
| `high_quality_threshold` | 0.7 | High quality threshold |
| `medium_quality_threshold` | 0.4 | Medium quality threshold |
| `max_filter_examples` | 5 | Verbose examples |
| `model` | DEFAULT_MODEL | Fallback model |
| `max_concurrent_requests` | 5 | Concurrency limit |
| `minimum_timeout_seconds` | 15.0 | Min timeout |
| `maximum_timeout_seconds` | 60.0 | Max timeout |

**Status:** ✅ Complete coverage

---

### 1.4 Clustering Configuration (`ClusteringConfig`, `UMAPConfig`, `HDBSCANConfig`)
**Location:** `src/config.py` lines 534-593

| Parameter | Value | Config Class | Purpose |
|-----------|-------|--------------|---------|
| `pca_components` | 0.99 | ClusteringConfig | Variance to keep |
| `pca_random_state` | 42 | ClusteringConfig | PCA random seed |
| `CLUSTER_METRIC` | "euclidean" | ClusteringConfig | Distance metric |
| `DBCV_D` | 1 | ClusteringConfig | DBCV parameter |
| `similarity_analysis_thresholds` | [0.40, 0.50, ...] | ClusteringConfig | Similarity thresholds |
| `default_merge_threshold` | 0.95 | ClusteringConfig | Merge threshold |
| `ctfidf_top_k` | 15 | ClusteringConfig | TF-IDF top K |
| `ctfidf_min_df` | 2 | ClusteringConfig | TF-IDF min doc freq |
| `ctfidf_ngram_range` | (1, 2) | ClusteringConfig | N-gram range |
| `n_neighbors` | 10 | UMAPConfig | UMAP neighbors |
| `n_components` | 10 | UMAPConfig | UMAP dimensions |
| `min_dist` | 0.1 | UMAPConfig | UMAP min distance |
| `metric` | "cosine" | UMAPConfig | UMAP metric |
| `random_state` | 42 | UMAPConfig | UMAP random seed |
| `n_epochs` | 200 | UMAPConfig | UMAP epochs |
| `use_parallel_umap` | False | UMAPConfig | Parallel processing |
| `min_cluster_size` | 5 | HDBSCANConfig | Min cluster size |
| `alpha` | 1.0 | HDBSCANConfig | Alpha parameter |
| `metric` | "euclidean" | HDBSCANConfig | HDBSCAN metric |
| `cluster_selection_method` | "leaf" | HDBSCANConfig | Selection method |
| `merge_similar_clusters` | True | HDBSCANConfig | Cluster merging |
| `merge_similarity_threshold` | 0.95 | HDBSCANConfig | Merge threshold |

**Status:** ✅ Excellent organization with separate configs for UMAP and HDBSCAN

---

### 1.5 Code Designer Configuration (`CodeDesignerConfig`)
**Location:** `src/config.py` lines 610-650

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `similarity_threshold` | 0.7 | Cosine similarity threshold |
| `max_sub_batch_size` | 10 | Max clusters per sub-batch |
| `batch_size` | 20 | Base batch size |
| `max_concurrent_requests` | 15 | Max concurrent API requests |
| `async_concurrency_limit` | 16 | Async concurrency limit |
| `max_ideas_per_cluster` | 30 | Max ideas to include |
| `max_cached_versions` | 5 | Max cached versions |
| `modification_leak_batch_size` | 10 | Leak recovery batch size |

**Status:** ✅ Well-defined

---

## Category 2: Migration Candidates (OPPORTUNITIES) 🔄

### 2.1 Rate Limiting Headroom (HIGH PRIORITY)

**Current State:**
- Duplicated **15+ times** across 5+ utility files
- Always set to `0.9` (90% of API limits)
- Used in: `qualityFilter.py`, `spellChecker.py`, `ideaExtractor.py`, `codeAssigner.py`, `codeGenerator.py`

**Occurrences:**
```python
# qualityFilter.py:196
HEADROOM = 0.9  # Increased from 0.8 to 0.9 for higher throughput

# spellChecker.py:974
HEADROOM = 0.9  # Increased from 0.8 to 0.9 for higher throughput

# ideaExtractor.py:228, 688, 858
HEADROOM = 0.9

# codeAssigner.py:263, 827
HEADROOM = 0.9  # Use 90% of limits for safety

# codeGenerator.py:1338, 1490, 1884, 3003
HEADROOM = 0.9
```

**Proposed Migration:**
- **Target Config Class:** `ProcessingConfig` (new dataclass in config.py)
- **Parameter Name:** `rate_limit_headroom: float = 0.9`
- **Impact:** 5 files, 15+ occurrences
- **Priority:** **HIGH** - Centralization enables easy tuning of aggressive vs conservative rate limiting

---

### 2.2 Concurrency Caps and Minimums (HIGH PRIORITY)

**Current State:**
- Hardcoded in `compute_optimal_concurrency()` function calls
- Values vary by use case: `cap=300` (default), `cap=10000` (permissive), `cap=10000` (quality filter)
- `min_conc` varies: `100` (default), `0` (permissive), `10` (code generator)

**Occurrences:**
```python
# qualityFilter.py:137 (function signature)
def compute_optimal_concurrency(..., cap: int = 300, min_conc: int = 100, HEADROOM: float = 0.9)

# qualityFilter.py:557 (call site)
Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, cap=10000, min_conc=0)

# ideaExtractor.py:129 (function signature)
def compute_optimal_concurrency(..., cap: int = 300, min_conc: int = 100, HEADROOM: float = 0.9)

# ideaExtractor.py:730 (call site)
Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, cap=10000, min_conc=0)

# codeAssigner.py:141 (function signature)
def compute_optimal_concurrency(..., cap: int = 300, min_conc: int = 100, HEADROOM: float = 0.9)

# codeAssigner.py:866 (call site)
Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, cap=10000, min_conc=0)

# codeGenerator.py:260 (function signature)
def compute_optimal_concurrency(..., cap: int = 300, min_conc: int = 100, HEADROOM: float = 0.9)

# codeGenerator.py:3013 (call site)
Little = compute_optimal_concurrency(api_limits, chain_latency, self.avg_tokens, cap=300, min_conc=10)

# spellChecker.py:981 (function signature)
def compute_optimal_concurrency(..., cap: int = 300, min_conc: int = 1, HEADROOM: float = 0.9)
```

**Proposed Migration:**
- **Target Config Class:** `ProcessingConfig` (new dataclass)
- **Parameter Names:**
  - `concurrency_cap_default: int = 300`
  - `concurrency_cap_permissive: int = 10000`
  - `concurrency_min_default: int = 100`
  - `concurrency_min_permissive: int = 0`
  - `concurrency_min_conservative: int = 10`
- **Impact:** 5 files, function signatures + call sites
- **Priority:** **HIGH** - Affects pipeline throughput and stability

---

### 2.3 Timeout Bounds (MEDIUM PRIORITY)

**Current State:**
- `min_timeout` and `max_timeout` duplicated in `LatencyTracker.get_timeout()` methods
- Always: `min_timeout=15.0`, `max_timeout=60.0`
- Found in: `qualityFilter.py`, `ideaExtractor.py`, `codeAssigner.py`, `codeGenerator.py`, `spellChecker.py`

**Occurrences:**
```python
# qualityFilter.py:107
def get_timeout(self, est_tokens, margin=1.5, min_timeout=15.0, max_timeout=60.0):

# ideaExtractor.py:100
def get_timeout(self, est_tokens, margin=1.5, min_timeout=15.0, max_timeout=60.0):

# codeAssigner.py:111
def get_timeout(self, est_tokens, margin=1.5, min_timeout=15.0, max_timeout=60.0):

# codeGenerator.py:228
def get_timeout(self, est_tokens, margin=1.5, min_timeout=15.0, max_timeout=60.0):

# spellChecker.py:149
def get_timeout(self, est_tokens, margin=1.5, min_timeout=15.0, max_timeout=60.0):
```

**Proposed Migration:**
- **Target Config Class:** `ProcessingConfig` (new dataclass)
- **Parameter Names:**
  - `adaptive_timeout_min_seconds: float = 15.0`
  - `adaptive_timeout_max_seconds: float = 60.0`
  - `adaptive_timeout_margin: float = 1.5`
- **Impact:** 5 files, LatencyTracker classes
- **Priority:** **MEDIUM** - Affects timeout behavior but values are reasonable defaults
- **Note:** These values also appear in step-specific configs (SpellCheckConfig, QualityFilterConfig, etc.) - migration would consolidate

---

### 2.4 Latency Tracker EMA Alpha (MEDIUM PRIORITY)

**Current State:**
- EMA smoothing factor duplicated in `LatencyTracker.__init__()`
- Always: `alpha=0.1`
- Found in: `qualityFilter.py`, `ideaExtractor.py`, `codeAssigner.py`, `codeGenerator.py`, `spellChecker.py`

**Occurrences:**
```python
# qualityFilter.py:94
def __init__(self, alpha=0.1):

# ideaExtractor.py:87
def __init__(self, alpha=0.1):

# codeAssigner.py:98
def __init__(self, alpha=0.1):

# codeGenerator.py:215
def __init__(self, alpha=0.1):

# spellChecker.py:136
def __init__(self, alpha=0.1):
```

**Proposed Migration:**
- **Target Config Class:** `ProcessingConfig` (new dataclass)
- **Parameter Name:** `latency_tracker_ema_alpha: float = 0.1`
- **Impact:** 5 files, LatencyTracker classes
- **Priority:** **MEDIUM** - Internal implementation detail but centralizing improves tuning

---

### 2.5 Bootstrap Probe Count (LOW PRIORITY)

**Current State:**
- Bootstrap measurement uses `n_probes=3` as default
- Only appears in pipeline.py imports from qualityFilter patterns

**Occurrence:**
```python
# Pattern found in async bootstrap functions
async def bootstrap_measure_async(call_fn, n_probes: int = 3):
```

**Proposed Migration:**
- **Target Config Class:** `ProcessingConfig` (new dataclass)
- **Parameter Name:** `bootstrap_probe_count: int = 3`
- **Impact:** 1 location (if consolidated)
- **Priority:** **LOW** - Rarely changed, good default

---

### 2.6 SpellChecker Batch Processing (LOW PRIORITY)

**Current State:**
- Hardcoded in spellChecker.py for OOV processing
- `batch_size = 50`
- `max_concurrent_batches = 6`

**Occurrences:**
```python
# spellChecker.py:609-610
batch_size = 50  # Process 100 words per batch sequentially
max_concurrent_batches = 6  # Run max 6 batches concurrently (stable concurrency)
```

**Proposed Migration:**
- **Target Config Class:** `SpellCheckConfig` (already exists)
- **Parameter Names:**
  - `oov_batch_size: int = 50`
  - `oov_max_concurrent_batches: int = 6`
- **Impact:** 1 file (spellChecker.py)
- **Priority:** **LOW** - Already has comprehensive SpellCheckConfig, easy addition

---

### 2.7 CodeGenerator Batch Size Override (LOW PRIORITY)

**Current State:**
- Hardcoded batch_size override in codeGenerator.py

**Occurrence:**
```python
# codeGenerator.py:821
batch_size = 100
```

**Context:** Appears to be a local override within a specific function, may be intentional for that use case.

**Proposed Action:**
- **Investigate context** - may be acceptable as local override
- If systematic, add to `CodeDesignerConfig` as `internal_batch_size_override: int = 100`
- **Priority:** **LOW** - Single occurrence, may be intentional

---

## Category 3: Acceptable Hardcoding (JUSTIFIED) ✅

### 3.1 Mathematical Constants

**Examples:**
```python
# clusterer.py:319
alpha=1.0  # HDBSCAN alpha for default stability weighting

# qualityFilter.py:110, 124
return max(min_timeout, 30.0)  # Default 30s fallback
return 2.0  # Default 2s latency
```

**Justification:** Mathematical constants and fallback defaults that are algorithm-specific

---

### 3.2 Debug and Development Flags

**Examples:**
```python
# codeGenerator.py:40
EXTRA_VERBOSE = False

# pipeline.py:1624-1948
if False:  # debug if true
```

**Justification:** Development aids, not runtime configuration

---

### 3.3 Bootstrap Timeout

**Example:**
```python
# ideaExtractor.py:474
timeout=30.0  # Conservative timeout for bootstrap
```

**Justification:** One-time bootstrap measurement, conservative default appropriate

---

## Recommended Migration Strategy

### Phase 1: Create ProcessingConfig (HIGH PRIORITY)

**New dataclass in config.py:**

```python
@dataclass
class ProcessingConfig:
    """Global processing parameters affecting cache validity and performance"""

    # Rate limiting
    rate_limit_headroom: float = 0.9  # Use 90% of API limits for safety

    # Concurrency bounds
    concurrency_cap_default: int = 300
    concurrency_cap_permissive: int = 10000
    concurrency_min_default: int = 100
    concurrency_min_permissive: int = 0
    concurrency_min_conservative: int = 10

    # Adaptive timeout bounds
    adaptive_timeout_min_seconds: float = 15.0
    adaptive_timeout_max_seconds: float = 60.0
    adaptive_timeout_margin: float = 1.5

    # Latency tracking
    latency_tracker_ema_alpha: float = 0.1
    latency_tracker_samples_window: int = 100  # Keep last N samples for percentiles

    # Bootstrap measurement
    bootstrap_probe_count: int = 3
```

**Files to modify:**
- `src/config.py` - Add ProcessingConfig dataclass
- `src/utils/qualityFilter.py` - Import and use ProcessingConfig
- `src/utils/spellChecker.py` - Import and use ProcessingConfig
- `src/utils/ideaExtractor.py` - Import and use ProcessingConfig
- `src/utils/codeAssigner.py` - Import and use ProcessingConfig
- `src/utils/codeGenerator.py` - Import and use ProcessingConfig

**Impact:** 5 utility files, ~25 occurrences

---

### Phase 2: Extend SpellCheckConfig (MEDIUM PRIORITY)

**Add to existing SpellCheckConfig:**

```python
@dataclass
class SpellCheckConfig:
    # ... existing parameters ...

    # OOV batch processing (Phase 2 addition)
    oov_batch_size: int = 50
    oov_max_concurrent_batches: int = 6
```

**Files to modify:**
- `src/config.py` - Extend SpellCheckConfig
- `src/utils/spellChecker.py` - Use new config parameters

**Impact:** 1 file, 2 occurrences

---

### Phase 3: Investigate CodeGenerator Batch Override (LOW PRIORITY)

**Action:**
- Review context of `batch_size = 100` in codeGenerator.py:821
- Determine if intentional local override or should be in CodeDesignerConfig
- Document decision

**Impact:** 1 file, 1 occurrence (potentially)

---

## Migration Priority Matrix

| Parameter | Priority | Files Affected | Occurrences | Complexity |
|-----------|----------|----------------|-------------|------------|
| `rate_limit_headroom` | HIGH | 5 | 15+ | Low |
| Concurrency caps/mins | HIGH | 5 | 10+ | Medium |
| Timeout bounds | MEDIUM | 5 | 5 | Low |
| Latency tracker alpha | MEDIUM | 5 | 5 | Low |
| OOV batch params | LOW | 1 | 2 | Low |
| Bootstrap probes | LOW | 1 | 1 | Low |
| CodeGen batch override | LOW | 1 | 1 | Low (investigation) |

---

## Benefits of Migration

### 1. Centralized Tuning
- Single location to adjust aggressive vs conservative rate limiting
- Easy A/B testing of concurrency strategies
- Consistent timeout behavior across all async processors

### 2. Cache-Aware Configuration
- Parameters affecting processing behavior can be included in cache keys
- Prevents stale cache when tuning performance parameters

### 3. Maintainability
- Eliminates "magic numbers" scattered across codebase
- Clear documentation of parameter purpose and impact
- Easier onboarding for new developers

### 4. Type Safety
- Pydantic dataclasses provide validation
- IDE autocompletion for configuration
- Runtime type checking

---

## Files Requiring Modification

### High Priority (Phase 1)
1. `src/config.py` - Add ProcessingConfig dataclass
2. `src/utils/qualityFilter.py` - Replace 4 hardcoded values
3. `src/utils/spellChecker.py` - Replace 4 hardcoded values
4. `src/utils/ideaExtractor.py` - Replace 5 hardcoded values
5. `src/utils/codeAssigner.py` - Replace 3 hardcoded values
6. `src/utils/codeGenerator.py` - Replace 5 hardcoded values

### Medium Priority (Phase 2)
7. `src/config.py` - Extend SpellCheckConfig
8. `src/utils/spellChecker.py` - Use new OOV config

### Low Priority (Phase 3)
9. `src/utils/codeGenerator.py` - Investigate batch_size override

---

## Testing Strategy

### 1. Unit Tests
- Verify config imports work correctly
- Test default values match previous hardcoded values
- Validate Pydantic type checking

### 2. Integration Tests
- Run pipeline with RUN_UNTIL_STEP = 7
- Compare cache keys before/after migration
- Verify identical behavior with default config

### 3. Performance Tests
- Measure throughput before/after
- Test concurrency tuning (low vs high caps)
- Validate timeout behavior under load

---

## Conclusion

The CoderingsTool configuration architecture is **well-designed** with strong centralization in config.py. The identified migration candidates are **low-risk, high-value** opportunities to:

1. **Eliminate duplication** of `HEADROOM = 0.9` (15+ times)
2. **Centralize concurrency tuning** for easier performance optimization
3. **Standardize timeout behavior** across all async processors
4. **Improve maintainability** with clear documentation

**Recommendation:** Proceed with **Phase 1 migration** (ProcessingConfig) as it addresses the highest-impact duplications with minimal risk. The existing config.py architecture provides an excellent foundation for these additions.

---

## Appendix: Complete Parameter Inventory

### config.py Parameters (45 total)

#### CacheConfig (5)
- max_cache_age_days, auto_cleanup, use_atomic_writes, batch_size, verbose

#### SpellCheckConfig (37)
- batch_size, temperature, max_tokens, retries, retry_delay, max_batch_size, completion_reserve, cache_size, spacy_batch_size, repeated_char_threshold, max_correction_examples, seed, context_chars, max_concurrent_requests, max_words_to_check, enable_word_frequency_cache, progress_report_interval, max_unique_oov_words, enable_early_termination, max_concurrent_suggestion_chunks, max_words_per_chunk, enable_adaptive_chunking, chunk_progress_reporting, suggestion_processing_semaphore_limit, minimum_timeout_seconds, maximum_timeout_seconds, hunspell_concurrent_sessions, hunspell_batch_size, enable_streaming_oov_detection, oov_detection_queue_size, rate_limit_safety_factor, rate_limit_utilization, concurrent_burst_multiplier, enable_suggestion_pre_validation, disable_pre_validation_above_oov_words, enable_suggestion_caching, hunspell_pool_size, ultra_batch_threshold, ultra_batch_size

#### QualityFilterConfig (12)
- batch_size, temperature, max_tokens, retries, instructor_retries, high_quality_threshold, medium_quality_threshold, max_filter_examples, model, max_concurrent_requests, minimum_timeout_seconds, maximum_timeout_seconds

#### SegmentationConfig (11)
- max_tokens, completion_reserve, min_batch_size, max_batch_size, target_token_utilization, retry_delay, max_retries, spacy_batch_size, umap_n_jobs, max_code_examples, max_sample_responses, model, temperature, max_concurrent_requests, minimum_timeout_seconds, maximum_timeout_seconds

#### ClusteringConfig (9)
- pca_components, pca_random_state, enable_dbcv, enable_meanp, centroid_distance, CLUSTER_METRIC, DBCV_D, similarity_analysis_thresholds, default_merge_threshold, grid_search_max_workers, grid_search_timeout_seconds, ctfidf_top_k, ctfidf_min_df, ctfidf_ngram_range

#### UMAPConfig (10)
- n_neighbors, n_components, min_dist, metric, random_state, n_jobs, low_memory, transform_seed, n_epochs, use_parallel_umap, parallel_jobs

#### HDBSCANConfig (10)
- min_cluster_size, min_samples, cluster_selection_epsilon, alpha, metric, cluster_selection_method, prediction_data, approx_min_span_tree, gen_min_span_tree, merge_similar_clusters, merge_similarity_threshold

#### CodeDesignerConfig (13)
- embedding_model, model, temperature, max_tokens, seed, similarity_threshold, max_sub_batch_size, batch_size, max_concurrent_requests, async_concurrency_limit, enable_aggressive_parallelism, enable_sequential_batch_processing, enable_sub_batch_processing, max_ideas_per_cluster, max_cached_versions, modification_leak_batch_size

### Hardcoded Parameters to Migrate (7 total)

1. **rate_limit_headroom** - 0.9 (15+ occurrences)
2. **concurrency_cap_default** - 300 (5 function signatures)
3. **concurrency_cap_permissive** - 10000 (3 call sites)
4. **concurrency_min_default** - 100 (5 function signatures)
5. **concurrency_min_permissive** - 0 (3 call sites)
6. **adaptive_timeout_min_seconds** - 15.0 (5 occurrences)
7. **adaptive_timeout_max_seconds** - 60.0 (5 occurrences)

---

**End of Report**
