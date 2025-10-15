# Implementation Plan: Dynamic Sample Size Tracking

## Overview
Update `st.session_state.sample_size` after each pipeline step to reflect the actual number of cases/ideas being processed, while keeping `st.session_state.sample_size_config` constant for cache key stability.

## Strategy

Use the existing **two-variable system**:
- **`sample_size_config`**: Initial user input (for cache keys) - **NEVER changes**
- **`sample_size`**: Current actual count (for display) - **UPDATES after each step**

This provides:
- Stable cache keys based on initial user selection
- Accurate display showing current processing count
- Clean separation of concerns

## Current State Analysis

### `st.session_state.sample_size` (widget key)
- **Line 1339**: Display only - "**Sample:** {st.session_state.sample_size} cases"
- **Line 1529**: Display only - "**Sample:** {st.session_state.sample_size} cases"
- **Currently unused** except for display

### `st.session_state.sample_size_config` (stored config)
- **Line 46**: Set by `DatasetConfig.to_session_state()`
- **Line 64**: Used by `DatasetConfig.from_session_state()` to rebuild config
- **Line 1266**: Display metric in sidebar
- **Used for**: Configuration and cache key generation

## Changes Required in app.py

### 1. After Step 0 (Data Load) - around line 1461
```python
st.session_state.pipeline_results['raw_text_list'] = raw_text_list
st.session_state.pipeline_results['var_lab'] = var_lab[last_bracket + 1:].strip()

# ADD THIS:
st.session_state.sample_size = len(raw_text_list)
```

### 2. After Step 1 (Preprocessing) - around line 1489
```python
st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text

# ADD THIS:
st.session_state.sample_size = len(preprocessed_text)
```

### 3. After Step 2 (Quality Filter) - around line 1579 (in show_filtering_page)
After the quality filter step completes, add:
```python
quality_filtered_text = pipeline.step_2_quality_filter(...)
st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text

# ADD THIS:
meaningful_responses = [item for item in quality_filtered_text if not item.quality_filter]
st.session_state.sample_size = len(meaningful_responses)
```

### 4. After Step 3 (Idea Extraction) - around line 1706 (in show_segmentation_page)
After idea extraction completes, add:
```python
encoded_text = pipeline.step_3_extract_ideas(...)
st.session_state['pipeline_results']['encoded_text'] = encoded_text

# ADD THIS:
total_ideas = sum(item.idea_count for item in encoded_text)
st.session_state.sample_size = total_ideas  # Now showing idea count instead of response count
```

### 5. After Step 4 (Embeddings) - around line 1789
```python
embedded_text = pipeline.step_4_generate_embeddings(...)
st.session_state.pipeline_results['embedded_text'] = embedded_text

# ADD THIS:
total_ideas = sum(len(resp.response_ideas) for resp in embedded_text if resp.response_ideas)
st.session_state.sample_size = total_ideas
```

### 6. After Step 5 (Clustering) - around line 1871
```python
initial_cluster_results = pipeline.step_5_cluster(...)
st.session_state.pipeline_results['initial_cluster_results'] = initial_cluster_results

# ADD THIS:
total_segments = sum(len(resp.response_ideas) for resp in initial_cluster_results if resp.response_ideas)
st.session_state.sample_size = total_segments
```

### 7. Steps 6-9
Continue updating with actual count (will match step 5's segment count through the remaining steps)

## Additional Fix: Pass sample_size to pipeline.step_0_load_data()

Currently, `sample_size` is NOT being passed to the pipeline function, so truncation doesn't work in the app.

### Line 1427 (Multiple variables mode)
```python
raw_text_list = pipeline.step_0_load_data(
    filename=st.session_state.filename,
    id_column=st.session_state.selected_id_column,
    var_name=selected_vars[0],
    variable_key=variable_key,
    cache_manager=_get_cache_manager(),
    force_recalc=st.session_state.get('force_recalculate_all', False),
    verbose=True,
    # ADD THIS:
    sample_size=st.session_state.get('sample_size_config')
)
```

### Line 1449 (Single variable mode)
```python
raw_text_list = pipeline.step_0_load_data(
    filename=st.session_state.filename,
    id_column=st.session_state.selected_id_column,
    var_name=st.session_state.selected_variable,
    variable_key=variable_key,
    cache_manager=_get_cache_manager(),
    force_recalc=st.session_state.get('force_recalculate_all', False),
    verbose=True,
    # ADD THIS:
    sample_size=st.session_state.get('sample_size_config')
)
```

## Expected Behavior After Implementation

### Example Flow (Starting with 100 cases):

```
Step 0: Load Data
  → Display: "Sample: 100 cases"

Step 1: Preprocess
  → 5 cases removed (system missing/empty)
  → Display: "Sample: 95 cases"

Step 2: Quality Filter
  → 13 cases filtered (meaningless responses)
  → Display: "Sample: 82 cases"

Step 3: Extract Ideas
  → 82 responses split into 127 ideas
  → Display: "Sample: 127 ideas"

Steps 4-9: Process Ideas
  → Display continues showing: "Sample: 127 ideas"
```

## Benefits

1. **Accurate User Feedback**: Users see exactly how many cases/ideas are being processed at each step
2. **Stable Cache Keys**: `sample_size_config` ensures cache remains valid across reruns
3. **Minimal Changes**: Uses existing session state variables, no architectural changes needed
4. **Clear Semantics**: Display reflects reality (responses → ideas transition is visible)

## Testing Checklist

After implementation, verify:
- [ ] Initial display shows user-selected sample size
- [ ] Display updates after each step completion
- [ ] Cache keys remain stable (check cache hit rate)
- [ ] Transition from "cases" to "ideas" is clear in UI
- [ ] Final export matches displayed idea count
