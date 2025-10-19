# Step 4 Architecture Refactoring Plan

**Generated:** 2025-01-19
**Status:** Pending Implementation
**Priority:** High (Critical bugs will cause crashes)

---

## Executive Summary

Step 4 (`show_embedding_page()`) **DOES NOT** follow the standardized 5-block architecture pattern. The implementation uses an outdated "waiting_for_continue" pattern, is missing critical blocks (Blue Box, Yellow Box, Data Loading Block), has incorrect completion step numbering, and lacks proper force_recalc logic.

**Recommendation:** Complete rewrite of the function following the standardized pattern.

---

## Critical Issues Found

### Priority 1: Will Cause Crashes

1. **Missing Data Loading Block (Bug #1)**
   - **Location:** Entire function (lines 1753-1843)
   - **Problem:** No block to load `encoded_text` from step 3 before processing
   - **Impact:** `KeyError: 'encoded_text'` when clicking processing button
   - **Fix:** Add cache-first data loading block

2. **Missing Metadata Population (Bug #2)**
   - **Location:** Missing from data loading block
   - **Problem:** `var_lab` not populated into `pipeline_results`
   - **Impact:** Fragile fallback logic at line 1820
   - **Fix:** Populate metadata in data loading block

3. **Wrong Completion Step Numbers**
   - **Location:** Lines 1833, 1836
   - **Problem:** Marks step 5 complete instead of step 4
   - **Impact:** Navigation system corruption
   - **Fix:** Change to `mark_step_completed(4)`

4. **Incorrect force_recalc Logic**
   - **Location:** Line 1824
   - **Problem:** Only checks global flag, ignores step-specific invalidation
   - **Impact:** Cache invalidation from step 3 won't trigger recalculation
   - **Fix:** Use combined logic: `force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= 4)`

### Priority 2: Architectural Inconsistencies

5. **Wrong Architecture Pattern** - Uses `waiting_for_continue_embedding` instead of `is_step_completed(4)`
6. **Missing Blue Box** - No input data info display
7. **Missing Yellow Box** - No stats/results display
8. **Processing Button Wrong Condition** - No check for step 3 completion
9. **Header Off-By-One Error** - Says "Step 5" instead of "Step 4"

### Priority 3: Minor Issues

10. **Deprecated Session State Key** - Uses `completed_step` instead of only `mark_step_completed()`
11. **No Stats Return** - Pipeline function may not return stats (needs verification)

---

## Pattern Compliance Assessment

| Block | Status | Notes |
|-------|--------|-------|
| Header | ❌ INCORRECT | "Stap 5" instead of "Stap 4" |
| Green Box | ❌ WRONG PATTERN | Uses `waiting_for_continue_embedding` flag |
| Blue Box | ❌ MISSING | Should show input data info |
| Yellow Box | ❌ MISSING | Should display embedding stats |
| Data Loading | ❌ MISSING | Critical - will cause KeyError |
| Processing Button | ⚠️ PARTIAL | Has button but wrong structure |
| Sample Display | ✅ CORRECT | Properly implemented |

---

## Recommended Implementation

**Approach:** Complete rewrite of `show_embedding_page()` (lines 1753-1843)

**New Structure:**
1. Header - "Stap 4: Genereer Embeddings"
2. Green Box - Show when `is_step_completed(4)`
3. Blue Box - Show input info when `is_step_completed(3)`
4. Yellow Box - Show stats when `is_step_completed(4)`
5. Data Loading Block - Cache-first approach for loading `encoded_text`
6. Processing Button - Proper conditions and force_recalc logic

**Estimated Effort:** 30-45 minutes
- Implementation: 15 minutes
- Testing: 15 minutes
- Edge cases: 15 minutes

**Risk:** Low (isolated function, clear template from steps 1-2)

---

## Implementation Details

### Data Loading Block Pattern

```python
if is_step_completed(3) and not is_step_completed(4):
    progress_container = st.empty()
    try:
        if 'encoded_text' not in st.session_state.pipeline_results:
            # Generate variable_key
            selected_variables = st.session_state.get('selected_variables_config', [st.session_state.selected_variable])
            is_merged = st.session_state.get('is_merged_variable', False)
            sample_size = st.session_state.get('sample_size_config')
            merge_config = st.session_state.get('merge_config')
            variable_key = generate_enhanced_variable_key(
                selected_variables,
                is_merged=is_merged,
                sample_size=sample_size,
                merge_config=merge_config
            )

            cache_manager = _get_cache_manager()

            # Try cache first (works for both upload and cache routes)
            if cache_manager.is_cache_valid(st.session_state.filename, "extracted_ideas", variable_key):
                encoded_text = cache_manager.load_from_cache(
                    st.session_state.filename,
                    "extracted_ideas",
                    variable_key,
                    models.IdeasExtractedModel
                )
                st.session_state.pipeline_results['encoded_text'] = encoded_text
            else:
                st.error("Input data not found. Please run idea extraction (step 3) first.")

        # Populate metadata
        if 'var_lab' not in st.session_state.pipeline_results:
            st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')

    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
```

### Force Recalc Pattern

```python
force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= 4)
```

### Completion Tracking

```python
# Store results
st.session_state.pipeline_results['embedded_text'] = embedded_text

# Mark step completed (NOT step 5!)
mark_step_completed(4)
st.rerun()
```

---

## Testing Checklist

### Upload Route
- [ ] Navigate to step 4 after completing step 3
- [ ] Blue box shows idea count from step 3
- [ ] Data loading block populates `encoded_text`
- [ ] Processing button appears
- [ ] Click button → Processing completes
- [ ] Green box appears
- [ ] Yellow box shows stats (if available)
- [ ] Samples appear in right panel

### Cache Route
- [ ] Load from cache → Jump to step 4
- [ ] Blue/Green/Yellow boxes show correctly
- [ ] Samples show immediately
- [ ] Continue button works

### Edge Cases
- [ ] Navigate to step 4 when step 3 not completed
- [ ] Force recalculate from step 3 → Step 4 recalculates
- [ ] Missing input data handled gracefully
- [ ] Bilingual text works

---

## Dependencies

**Must complete first:**
- Step 3 refactoring (idea extraction)

**Affects:**
- Step 5 (clustering) - depends on step 4 output

---

## Notes

1. **Pipeline Function:** Verify if `step_4_generate_embeddings` returns stats tuple or just data
2. **Sample Display:** Already correctly implemented in `show_step_samples()` (line 3194)
3. **Obsolete Keys:** Remove `waiting_for_continue_embedding` and `completed_step` after refactoring

---

## Related Documentation

- `docs/step_architecture_pattern.md` - Standardized 5-block pattern
- `docs/step_architecture_pattern.md` (lines 247-323) - Critical pitfalls section
