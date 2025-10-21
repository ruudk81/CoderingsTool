# Configuration Migration Roadmap
**CoderingsTool Configuration Centralization Project**

---

## Quick Reference: Migration Candidates

### 🔴 HIGH PRIORITY - Immediate Action Recommended

#### 1. Rate Limiting Headroom
```python
# Current: Duplicated 15+ times across 5 files
HEADROOM = 0.9  # qualityFilter.py, spellChecker.py, ideaExtractor.py, codeAssigner.py, codeGenerator.py

# Proposed: Single source in config.py
@dataclass
class ProcessingConfig:
    rate_limit_headroom: float = 0.9  # Use 90% of API limits for safety
```

**Impact:** 5 files, 15+ occurrences
**Benefit:** Single knob to tune aggressive vs conservative rate limiting globally

---

#### 2. Concurrency Caps and Minimums
```python
# Current: Function signatures with different defaults
def compute_optimal_concurrency(..., cap: int = 300, min_conc: int = 100, HEADROOM: float = 0.9)
# Call sites override with different values:
Little = compute_optimal_concurrency(..., cap=10000, min_conc=0)  # Permissive
Little = compute_optimal_concurrency(..., cap=300, min_conc=10)   # Conservative

# Proposed: Named configurations in config.py
@dataclass
class ProcessingConfig:
    concurrency_cap_default: int = 300
    concurrency_cap_permissive: int = 10000
    concurrency_min_default: int = 100
    concurrency_min_permissive: int = 0
    concurrency_min_conservative: int = 10
```

**Impact:** 5 files, 10+ occurrences
**Benefit:** Clear semantic meaning, easy to adjust for different pipeline stages

---

### 🟡 MEDIUM PRIORITY - Recommended for Next Phase

#### 3. Adaptive Timeout Bounds
```python
# Current: Duplicated in LatencyTracker classes
def get_timeout(self, est_tokens, margin=1.5, min_timeout=15.0, max_timeout=60.0):

# Proposed: Centralized timeout configuration
@dataclass
class ProcessingConfig:
    adaptive_timeout_min_seconds: float = 15.0
    adaptive_timeout_max_seconds: float = 60.0
    adaptive_timeout_margin: float = 1.5
```

**Impact:** 5 files, LatencyTracker classes
**Benefit:** Consistent timeout behavior, easier to tune for different API conditions

---

#### 4. Latency Tracker EMA Alpha
```python
# Current: Duplicated in LatencyTracker.__init__
def __init__(self, alpha=0.1):

# Proposed: Configuration parameter
@dataclass
class ProcessingConfig:
    latency_tracker_ema_alpha: float = 0.1
    latency_tracker_samples_window: int = 100  # For percentiles
```

**Impact:** 5 files, LatencyTracker classes
**Benefit:** Tunable smoothing for latency tracking

---

### 🟢 LOW PRIORITY - Nice to Have

#### 5. SpellChecker OOV Batch Processing
```python
# Current: Hardcoded in spellChecker.py
batch_size = 50
max_concurrent_batches = 6

# Proposed: Extend existing SpellCheckConfig
@dataclass
class SpellCheckConfig:
    # ... existing 37 parameters ...
    oov_batch_size: int = 50
    oov_max_concurrent_batches: int = 6
```

**Impact:** 1 file, 2 occurrences
**Benefit:** Completes SpellCheckConfig coverage

---

## Migration Execution Plan

### Phase 1: ProcessingConfig Creation (Week 1)

**Objective:** Centralize rate limiting and concurrency parameters

**Tasks:**

1. **Create ProcessingConfig dataclass in config.py**
   - [ ] Add dataclass with all Phase 1 parameters
   - [ ] Create DEFAULT_PROCESSING_CONFIG instance
   - [ ] Add comprehensive docstrings

2. **Update qualityFilter.py**
   - [ ] Import ProcessingConfig
   - [ ] Replace HEADROOM = 0.9 (lines 196, 523)
   - [ ] Update compute_optimal_concurrency signature
   - [ ] Update LatencyTracker class
   - [ ] Run tests

3. **Update spellChecker.py**
   - [ ] Import ProcessingConfig
   - [ ] Replace HEADROOM = 0.9 (line 974)
   - [ ] Update compute_optimal_concurrency signature
   - [ ] Update LatencyTracker class
   - [ ] Run tests

4. **Update ideaExtractor.py**
   - [ ] Import ProcessingConfig
   - [ ] Replace HEADROOM = 0.9 (lines 228, 688, 858)
   - [ ] Update compute_optimal_concurrency signature
   - [ ] Update LatencyTracker class
   - [ ] Run tests

5. **Update codeAssigner.py**
   - [ ] Import ProcessingConfig
   - [ ] Replace HEADROOM = 0.9 (lines 263, 827)
   - [ ] Update compute_optimal_concurrency signature
   - [ ] Update LatencyTracker class
   - [ ] Run tests

6. **Update codeGenerator.py**
   - [ ] Import ProcessingConfig
   - [ ] Replace HEADROOM = 0.9 (lines 1338, 1490, 1884, 3003)
   - [ ] Update compute_optimal_concurrency signature
   - [ ] Update LatencyTracker class
   - [ ] Run tests

7. **Integration Testing**
   - [ ] Run pipeline with RUN_UNTIL_STEP = 7
   - [ ] Verify identical behavior
   - [ ] Check cache key compatibility
   - [ ] Performance baseline comparison

8. **Git Commit**
   - [ ] Pre-migration checkpoint
   - [ ] Commit with detailed message
   - [ ] Push to GitHub

**Estimated Effort:** 4-6 hours

---

### Phase 2: SpellCheckConfig Extension (Week 2)

**Objective:** Complete SpellCheckConfig coverage

**Tasks:**

1. **Extend SpellCheckConfig in config.py**
   - [ ] Add oov_batch_size and oov_max_concurrent_batches
   - [ ] Update DEFAULT_SPELLCHECK_CONFIG

2. **Update spellChecker.py**
   - [ ] Replace hardcoded batch_size = 50
   - [ ] Replace hardcoded max_concurrent_batches = 6
   - [ ] Run tests

3. **Git Commit**
   - [ ] Commit Phase 2 changes
   - [ ] Push to GitHub

**Estimated Effort:** 1 hour

---

### Phase 3: Code Review and Cleanup (Week 3)

**Objective:** Investigate remaining edge cases

**Tasks:**

1. **Investigate codeGenerator.py:821**
   - [ ] Review context of batch_size = 100
   - [ ] Determine if intentional override or should be configurable
   - [ ] Document decision

2. **Documentation Update**
   - [ ] Update README with new configuration
   - [ ] Add migration notes to CHANGELOG
   - [ ] Document configuration best practices

**Estimated Effort:** 2 hours

---

## File Change Summary

### Files to Modify (Phase 1)
| File | Lines to Change | Type of Change |
|------|-----------------|----------------|
| `src/config.py` | +25 | Add ProcessingConfig |
| `src/utils/qualityFilter.py` | ~10 | Replace hardcoded values |
| `src/utils/spellChecker.py` | ~10 | Replace hardcoded values |
| `src/utils/ideaExtractor.py` | ~12 | Replace hardcoded values |
| `src/utils/codeAssigner.py` | ~8 | Replace hardcoded values |
| `src/utils/codeGenerator.py` | ~12 | Replace hardcoded values |

**Total:** 6 files, ~77 lines of code

### Files to Modify (Phase 2)
| File | Lines to Change | Type of Change |
|------|-----------------|----------------|
| `src/config.py` | +2 | Extend SpellCheckConfig |
| `src/utils/spellChecker.py` | ~4 | Replace hardcoded values |

**Total:** 2 files, ~6 lines of code

---

## Before/After Examples

### Example 1: Rate Limiting Headroom

**Before (qualityFilter.py:196):**
```python
class Grader:
    def __init__(...):
        # ...
        limits = get_openai_rate_limits(self.model)
        HEADROOM = 0.9  # Increased from 0.8 to 0.9 for higher throughput

        self.tpm_bucket = TokenBucket(limits.tokens_per_minute * HEADROOM)
```

**After:**
```python
from config import ProcessingConfig, DEFAULT_PROCESSING_CONFIG

class Grader:
    def __init__(..., processing_config: ProcessingConfig = None):
        # ...
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        limits = get_openai_rate_limits(self.model)

        self.tpm_bucket = TokenBucket(
            limits.tokens_per_minute * self.processing_config.rate_limit_headroom
        )
```

---

### Example 2: Concurrency Caps

**Before (ideaExtractor.py:730):**
```python
Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, cap=10000, min_conc=0)
```

**After:**
```python
Little = compute_optimal_concurrency(
    api_limits,
    avg_latency_s,
    avg_tokens,
    cap=self.processing_config.concurrency_cap_permissive,
    min_conc=self.processing_config.concurrency_min_permissive,
    headroom=self.processing_config.rate_limit_headroom
)
```

---

### Example 3: Adaptive Timeout

**Before (codeAssigner.py:111):**
```python
class LatencyTracker:
    def get_timeout(self, est_tokens, margin=1.5, min_timeout=15.0, max_timeout=60.0):
        # ...
        return max(min_timeout, min(max_timeout, timeout * margin))
```

**After:**
```python
class LatencyTracker:
    def __init__(self, processing_config: ProcessingConfig = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        # ...

    def get_timeout(self, est_tokens):
        # ...
        config = self.processing_config
        return max(
            config.adaptive_timeout_min_seconds,
            min(config.adaptive_timeout_max_seconds, timeout * config.adaptive_timeout_margin)
        )
```

---

## Testing Checklist

### Pre-Migration
- [ ] Record current pipeline performance baseline (run pipeline.py with sample_size=50)
- [ ] Save current cache keys for comparison
- [ ] Document current behavior in test scenarios

### Post-Migration (Each Phase)
- [ ] Unit tests: Import and instantiate new config classes
- [ ] Unit tests: Verify default values match previous hardcoded values
- [ ] Integration test: Run pipeline with RUN_UNTIL_STEP = 7
- [ ] Integration test: Verify cache key compatibility
- [ ] Integration test: Compare processing times (should be identical ±5%)
- [ ] Regression test: Run full pipeline (RUN_UNTIL_STEP = None)
- [ ] Code review: Check for missed occurrences of hardcoded values

### Performance Validation
- [ ] Measure throughput with default config
- [ ] Test with aggressive config (HEADROOM = 0.95, high caps)
- [ ] Test with conservative config (HEADROOM = 0.8, low caps)
- [ ] Verify adaptive timeout behavior under simulated API delays

---

## Risk Assessment

### Low Risk ✅
- **Type-safe migrations:** Pydantic dataclasses prevent configuration errors
- **Backward compatible defaults:** All defaults match existing hardcoded values
- **Isolated changes:** Each utility file is self-contained
- **Easy rollback:** Git checkpoints before each phase

### Potential Issues ⚠️
1. **Cache invalidation:** If ProcessingConfig parameters are added to cache keys
   - **Mitigation:** Keep cache key generation unchanged in Phase 1
   - **Future:** Consider adding HEADROOM to cache key for reproducibility

2. **Import cycles:** If circular dependencies arise
   - **Mitigation:** ProcessingConfig is in config.py (low-level module)
   - **Verification:** Test imports in each modified file

3. **Performance regression:** If config lookups add overhead
   - **Mitigation:** Config instances are created once per processor
   - **Verification:** Benchmark before/after

---

## Success Metrics

### Phase 1 Success Criteria
- ✅ All 15+ occurrences of HEADROOM = 0.9 replaced
- ✅ Zero test failures
- ✅ Pipeline completes with identical output
- ✅ Cache keys remain compatible
- ✅ Processing time within ±5% of baseline

### Phase 2 Success Criteria
- ✅ SpellCheckConfig complete
- ✅ Zero test failures
- ✅ Spell checking behavior unchanged

### Overall Success
- ✅ **7 hardcoded parameters centralized**
- ✅ **Single source of truth for processing configuration**
- ✅ **Clear documentation for future tuning**
- ✅ **No behavioral changes from user perspective**

---

## Post-Migration Benefits

### Immediate Benefits
1. **Easy experimentation:** Change HEADROOM once to test 0.8 vs 0.9 vs 0.95
2. **Clear semantics:** `concurrency_cap_permissive` vs magic number 10000
3. **Reduced duplication:** 15+ HEADROOM occurrences → 1 definition

### Long-term Benefits
1. **Cache-aware tuning:** Can add ProcessingConfig to cache keys for reproducibility
2. **Configuration profiles:** Easy to create "aggressive" vs "conservative" presets
3. **Testing:** Mock ProcessingConfig for unit tests of rate limiting logic
4. **Documentation:** Single location to document performance tuning parameters

---

## Questions for User

1. **Should ProcessingConfig parameters affect cache keys?**
   - Pro: Perfect reproducibility - changing HEADROOM invalidates cache
   - Con: More cache misses during tuning experiments
   - **Recommendation:** Start with NO (Phase 1), add later if needed

2. **Should we create configuration presets?**
   ```python
   AGGRESSIVE_PROCESSING = ProcessingConfig(
       rate_limit_headroom=0.95,
       concurrency_cap_default=500,
       # ...
   )

   CONSERVATIVE_PROCESSING = ProcessingConfig(
       rate_limit_headroom=0.8,
       concurrency_cap_default=200,
       # ...
   )
   ```
   - **Recommendation:** Add in Phase 3 after Phase 1 is validated

3. **Should compute_optimal_concurrency be refactored into a class?**
   - Currently duplicated across 5 files with identical logic
   - Could be `RateLimitOptimizer` class in utils/
   - **Recommendation:** Consider for Phase 3 cleanup

---

## Next Steps

1. **User Review:** Review this roadmap and approve Phase 1 scope
2. **Create Git Branch:** `feature/centralize-processing-config`
3. **Execute Phase 1:** Follow task checklist above
4. **Create Pull Request:** Detailed PR with before/after examples
5. **Merge and Deploy:** Test in production environment

---

**Prepared by:** Claude (Config Parameter Curator Agent)
**Date:** 2025-10-21
**Status:** READY FOR USER APPROVAL
