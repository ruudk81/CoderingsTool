# Step Architecture Pattern Documentation

## Purpose and Scope

This document defines the standardized architecture pattern for processing steps (1-9) in the CoderingsTool Streamlit application. It serves as a comprehensive reference for maintaining consistency across all pipeline steps and guides modifications to existing steps or creation of new ones.

**Target Audience:** Specialized agents and developers working on step flow architecture
**Reference Implementation:** Step 1 (Preprocessing) in `src/app.py` lines 1268-1468
**Last Updated:** 2025-01-19

---

## Table of Contents

1. [Overview](#overview)
2. [Complete Architecture Pattern](#complete-architecture-pattern)
3. [Visual Flow Diagram](#visual-flow-diagram)
4. [Code Templates](#code-templates)
5. [Step-by-Step Differences](#step-by-step-differences)
6. [Critical Implementation Details](#critical-implementation-details)
7. [Agent Specification](#agent-specification)

---

## Overview

### Two Routes Through the System

The application supports two distinct user journeys, controlled by session state flags:

| Route | Trigger | Session State Flag | Behavior |
|-------|---------|-------------------|----------|
| **Upload Route** | User uploads SPSS file | `force_recalculate_all = True`<br>`loaded_from_cache = False` | - Processes data freshly<br>- Does NOT show cached samples until step is completed in current session<br>- Session-based filtering active |
| **Cache Route** | User loads from cache | `force_recalculate_all = False`<br>`loaded_from_cache = True` | - Jumps to max completed step<br>- Shows cached samples immediately<br>- No session-based filtering |

**Critical File:** `src/app.py` function `show_step_samples()` lines 3060-3067 controls this routing:

```python
# Session-based filtering: Only show results from current session when in force_recalculate mode
if st.session_state.get('force_recalculate_all', False):
    # Upload from file route - only show if step was completed in current session
    if not is_step_completed(step_number):
        st.write("⏳ Data not yet processed in current session - run processing first")
        return
```

---

## Complete Architecture Pattern

### Standard Step Structure (5 Blocks)

Every processing step follows this consistent 5-block structure:

```
┌─ BLOCK 1: Header
├─ BLOCK 2: Green Box (Completion Status)
├─ BLOCK 3: Blue Box (Input Data Info)
├─ BLOCK 4: Yellow Box (Results/Stats)
├─ BLOCK 5: Data Loading Block (if needed)
├─ BLOCK 6: Processing Button Block
└─ [Right Panel]: Sample Display (via show_step_samples)
```

### Block 1: Header

```python
def show_{step_name}_page():
    lang = st.session_state.language
    st.header("Stap {N}: {Dutch Title}" if lang == "nl" else "Step {N}: {English Title}")
```

**Example:** `show_preprocessing_page()` at line 1268

### Block 2: Green Box - Completion Status

**Purpose:** Indicate when the current step is completed
**Condition:** `if is_step_completed({current_step})`
**Location:** First visual element after header

```python
# 1. green box/completion
if is_step_completed({CURRENT_STEP}):
    st.success("✅ " + ("{Dutch completion message}" if lang == "nl" else "{English completion message}"))
```

**Example:**
```python
# Line 1291 in show_preprocessing_page()
if is_step_completed(1):
    st.success("✅ " + ("Tekstverwerking voltooid! Bekijk de resultaten en klik dan op doorgaan." if lang == "nl" else "Preprocessing completed! Review the results on the right, then click continue."))
```

### Block 3: Blue Box - Input Data Info

**Purpose:** Show context about the data being processed
**Condition:** `if is_step_completed({previous_step})`
**Displays:** Question text, sample size, variable info

```python
# 2. blue box/sample info
if is_step_completed({PREVIOUS_STEP}):
    sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
    sample_info += (f"\n\n**Data:** {st.session_state.sample_size_config} antwoorden" if lang == "nl" else f"\n\n**Data:** {st.session_state.sample_size_config} responses")
    st.info(sample_info)
```

**Example:**
```python
# Lines 1295-1298 in show_preprocessing_page()
if is_step_completed(0):
    sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
    sample_info += (f"\n\n**Data:** {st.session_state.sample_size_config} antwoorden" if lang == "nl" else f"\n\n**Data:** {st.session_state.sample_size_config} responses")
    st.info(sample_info)
```

### Block 4: Yellow Box - Results/Stats

**Purpose:** Display processing statistics and results
**Condition:** `if is_step_completed({current_step})`
**Data Source:** `st.session_state['{step_name}_stats']`

```python
# 3. yellow box/results
if is_step_completed({CURRENT_STEP}):
    if st.session_state.get('{step_name}_stats', {}):
        summary_info = ""
        stats = st.session_state.get('{step_name}_stats', {})

        # Build stats display from stats dict
        # ... (step-specific stats formatting)

        st.markdown(f"""
        <div style="
        border-radius: 10px;
        padding: 12px 16px;
        background-color: #FFF8E6;
        margin-top: 8px;
        color: #5C4102;">
        {summary_info}
        </div>
        """, unsafe_allow_html=True)
```

**Example:**
```python
# Lines 1301-1350 in show_preprocessing_page()
if is_step_completed(1):
    if st.session_state.get('preprocessing_stats', {}):
        summary_info  = ""
        stats = st.session_state.get('preprocessing_stats', {})

        # a) Normalizer stats
        norm_stats = stats.get('normalizer_stats') or {}
        if norm_stats:
            nl = (st.session_state.language == "nl")
            summary_info += (
                "\n\n" + ("**Normalisatie:**" if nl else "**Normalization:**")
                + f"\n- {'Hoofdletterwijzigingen' if nl else 'Case changes'}: {norm_stats.get('case_changes', 0)} "
                  f"{'reacties' if nl else 'responses'}"
                # ... more stats
            )
        # ... more subsections

        st.markdown(f"""<div style="...yellow box styling...">{summary_info}</div>""", unsafe_allow_html=True)
```

### Block 5: Data Loading Block

**Purpose:** Load input data from previous step when not already in pipeline_results
**Condition:** `if is_step_completed({previous_step}) and not is_step_completed({current_step})`
**Critical Logic:** Handles both upload and cache routes

```python
# Getting data from step{N-1} selections
if is_step_completed({PREVIOUS_STEP}) and not is_step_completed({CURRENT_STEP}):
    progress_container = st.empty()
    try:
        if '{input_data_key}' not in st.session_state.pipeline_results:
            # Load data from file (upload route)
            if not st.session_state.get('loaded_from_cache', False):
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

                # Call previous step's pipeline function to load data
                {input_data} = pipeline.step_{N-1}_{operation}(
                    filename=st.session_state.filename,
                    # ... step-specific params
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    force_recalc=st.session_state.get('force_recalculate_all', False),
                    verbose=True
                )

                st.session_state.pipeline_results['{input_data_key}'] = {input_data}
                st.session_state.pipeline_results['{metadata_key}'] = {metadata}
            else:
                # Populate from session state (cache route)
                st.session_state.pipeline_results['{metadata_key}'] = st.session_state.get('{metadata_key}')
    except Exception as e:
        st.error(f"Error: {str(e)}")
```

**Example:**
```python
# Lines 1353-1425 in show_preprocessing_page()
if is_step_completed(0) and not is_step_completed(1):
    progress_container = st.empty()
    try:
        if 'raw_text_list' not in st.session_state.pipeline_results:
            if not st.session_state.get('loaded_from_cache', False):
                # ... variable_key generation ...
                raw_text_list = pipeline.step_0_load_data(
                    filename=st.session_state.filename,
                    id_column=st.session_state.selected_id_column,
                    var_name=st.session_state.selected_variable,
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    sample_size=sample_size,
                    merge_config=merge_config,
                    force_recalc=st.session_state.get('force_recalculate_all', False),
                    verbose=True)
                st.session_state.pipeline_results['raw_text_list'] = raw_text_list
                st.session_state.pipeline_results['var_lab'] = var_lab[last_bracket + 1:].strip()
            else:
                st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab')
    except Exception as e:
        st.error(f"Preprocessing fout: {str(e)}")
```

**Why This Block Exists:**
- **Upload route**: User selected variables/settings on step 0, but hasn't loaded the actual data yet
- **Cache route**: Data was pre-loaded during cache button click, but metadata might need population
- **Purpose**: Ensure `pipeline_results` has required input data before processing button is shown

### Block 6: Processing Button Block

**Purpose:** Execute the current step's processing when user clicks the button
**Condition:** `if is_step_completed({previous_step}) and not is_step_completed({current_step})`
**Critical Actions:** Call pipeline function, store results, mark complete, rerun

```python
# Processing button block
if is_step_completed({PREVIOUS_STEP}) and not is_step_completed({CURRENT_STEP}):
    st.markdown(ui.get_text("{STEP_INFO_KEY}", lang))

    # Show button to start processing
    if st.button("🚀 " + ("{Dutch button text}" if lang == "nl" else "{English button text}"), type="primary"):
        progress_container = st.empty()
        try:
            progress_container.text("🔄 {Processing message}...")

            # Generate variable_key for caching
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

            # Set force_recalc flag (respects both global and step-specific invalidation)
            force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= {CURRENT_STEP})

            # Call pipeline processing function
            {output_data}, {stats} = pipeline.step_{N}_{operation}(
                {input_param}=st.session_state.pipeline_results['{input_data_key}'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                variable_key=variable_key,
                cache_manager=_get_cache_manager(),
                model_config=st.session_state.model_config,
                force_recalc=force_recalc,
                verbose=True,
                prompt_printer_enabled=False
            )

            progress_container.success("✅ {Completion message}")

            # Store results
            st.session_state.pipeline_results['{output_data_key}'] = {output_data}
            st.session_state['{stats_key}'] = {stats}

            # Mark step completed
            mark_step_completed({CURRENT_STEP})
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
```

**Example:**
```python
# Lines 1428-1467 in show_preprocessing_page()
if is_step_completed(0) and not is_step_completed(1):
    st.markdown(ui.get_text("PREPROCESSING_INFO", lang))

    if st.button("🚀 " + ("Start Voorbewerking" if lang == "nl" else "Start Preprocessing"), type="primary"):
        progress_container = st.empty()
        try:
            progress_container.text("🔄 Tekst aan het voorbewerken...")
            # ... variable_key generation ...
            force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= 1)

            preprocessed_text, preprocessing_stats = pipeline.step_1_preprocess(
                raw_text_list=st.session_state.pipeline_results['raw_text_list'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                variable_key=variable_key,
                cache_manager=_get_cache_manager(),
                model_config=st.session_state.model_config,
                force_recalc=force_recalc,
                verbose=True,
                prompt_printer_enabled=False)

            progress_container.success("✅ Voorbewerking voltooid")
            st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text
            st.session_state['preprocessing_stats'] = preprocessing_stats

            mark_step_completed(1)
            st.rerun()

        except Exception as e:
            st.error(f"Preprocessing fout: {str(e)}")
```

---

## Visual Flow Diagram

### User Flow Through a Step

```
┌─────────────────────────────────────────────────────────────────┐
│ USER ENTERS STEP N                                               │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────┐
│ RENDER: Header                                                     │
└───────────────┬───────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────┐
│ CHECK: is_step_completed(N)?                                       │
├─────────YES──────┬──────────────NO─────────────────┐              │
│                  │                                  │              │
│ SHOW: ✅ Green  │                                  │              │
│ Box (Complete)   │                                  │              │
└──────────────────┴──────────────────────────────────┴──────────────┘
                   │                                  │
                   ▼                                  ▼
      ┌────────────────────────┐      ┌──────────────────────────┐
      │ SHOW: Blue Box         │      │ CHECK: is_step_completed │
      │ (Input data info)      │      │ (N-1)?                   │
      └────────────────────────┘      └────YES─────┬────NO───────┘
                   │                               │        │
                   ▼                               ▼        │
      ┌────────────────────────┐      ┌──────────────────┐ │
      │ SHOW: Yellow Box       │      │ SHOW: Blue Box   │ │
      │ (Results/Stats)        │      │ (Input info)     │ │
      └────────────────────────┘      └──────────────────┘ │
                   │                               │        │
                   │                               ▼        │
                   │                  ┌────────────────────────┐
                   │                  │ CHECK: input_data in   │
                   │                  │ pipeline_results?      │
                   │                  └──YES───────┬────NO─────┘
                   │                              │        │
                   │                              │        ▼
                   │                              │   ┌──────────────┐
                   │                              │   │ DATA LOADING │
                   │                              │   │ BLOCK        │
                   │                              │   │ - Load data  │
                   │                              │   │ - Populate   │
                   │                              │   │   results    │
                   │                              │   └──────────────┘
                   │                              │        │
                   │                              ▼        ▼
                   │                  ┌────────────────────────┐
                   │                  │ SHOW: Processing       │
                   │                  │ Button                 │
                   │                  └────────────────────────┘
                   │                               │
                   │                               │ USER CLICKS
                   │                               ▼
                   │                  ┌────────────────────────┐
                   │                  │ EXECUTE:               │
                   │                  │ - Call pipeline fn     │
                   │                  │ - Store results        │
                   │                  │ - Mark complete        │
                   │                  │ - st.rerun()           │
                   │                  └────────────────────────┘
                   │                               │
                   └───────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│ RIGHT PANEL: show_step_samples(N)                                  │
│                                                                    │
│ - Check force_recalculate_all flag                                │
│ - If True AND not is_step_completed(N): Show waiting message      │
│ - Else: Load from cache and display samples                       │
│ - Show "Continue to Next Step" button                             │
└────────────────────────────────────────────────────────────────────┘
```

---

## Code Templates

### Full Step Template

```python
def show_{step_name}_page():
    """
    Step {N}: {Step Description}

    Processes {input_description} and produces {output_description}.

    Pipeline function: step_{N}_{operation}
    Cache name: {cache_step_name}
    Model: models.{ModelClass}
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap {N}: {Dutch Title}" if lang == "nl" else "Step {N}: {English Title}")

    # ==================== BLOCK 1: GREEN BOX ====================
    # Show completion status
    if is_step_completed({CURRENT_STEP}):
        st.success("✅ " + (
            "{Dutch completion message}" if lang == "nl"
            else "{English completion message}"
        ))

    # ==================== BLOCK 2: BLUE BOX ====================
    # Show input data info when previous step is complete
    if is_step_completed({PREVIOUS_STEP}):
        sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        sample_info += (f"\n\n**Data:** {st.session_state.{sample_size_key}} {'antwoorden' if lang == 'nl' else 'responses'}")
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    # Show results/stats when current step is complete
    if is_step_completed({CURRENT_STEP}):
        if st.session_state.get('{stats_key}', {}):
            summary_info = ""
            stats = st.session_state.get('{stats_key}', {})

            # Build step-specific stats display
            # Example structure:
            # stat_section = stats.get('{section_name}') or {}
            # if stat_section:
            #     nl = (st.session_state.language == "nl")
            #     summary_info += (
            #         "\n\n" + ("**{Dutch Section}:**" if nl else "**{English Section}:**")
            #         + f"\n- {'{Dutch Label}' if nl else '{English Label}'}: {stat_section.get('{key}', 0)}"
            #     )

            st.markdown(f"""
            <div style="
            border-radius: 10px;
            padding: 12px 16px;
            background-color: #FFF8E6;
            margin-top: 8px;
            color: #5C4102;">
            {summary_info}
            </div>
            """, unsafe_allow_html=True)

    # ==================== BLOCK 4: DATA LOADING ====================
    # Load input data if not already in pipeline_results
    if is_step_completed({PREVIOUS_STEP}) and not is_step_completed({CURRENT_STEP}):
        progress_container = st.empty()
        try:
            if '{input_data_key}' not in st.session_state.pipeline_results:
                # Load data from file (upload route)
                if not st.session_state.get('loaded_from_cache', False):
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

                    # Optional: Load data via previous step's function if needed
                    # {input_data} = pipeline.step_{N-1}_{operation}(...)

                    # Store in pipeline_results
                    # st.session_state.pipeline_results['{input_data_key}'] = {input_data}
                else:
                    # Cache route: populate metadata from session state
                    st.session_state.pipeline_results['{metadata_key}'] = st.session_state.get('{metadata_key}')
        except Exception as e:
            st.error(f"{'{Step name}'} fout: {str(e)}" if lang == "nl" else f"{'{Step name}'} error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    # Show processing button when ready to process
    if is_step_completed({PREVIOUS_STEP}) and not is_step_completed({CURRENT_STEP}):
        st.markdown(ui.get_text("{INFO_KEY}", lang))

        # Show button to start processing
        if st.button("🚀 " + (
            "{Dutch button text}" if lang == "nl"
            else "{English button text}"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "{Dutch processing message}..." if lang == "nl"
                    else "{English processing message}..."
                ))

                # Generate variable_key for caching
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

                # Set force_recalc flag
                force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= {CURRENT_STEP})

                # Call pipeline processing function
                {output_data}, {stats} = pipeline.step_{N}_{operation}(
                    {input_param}=st.session_state.pipeline_results['{input_data_key}'],
                    filename=st.session_state.filename,
                    var_lab=st.session_state.pipeline_results['var_lab'],
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    model_config=st.session_state.model_config,
                    force_recalc=force_recalc,
                    verbose=True,
                    prompt_printer_enabled=False
                )

                progress_container.success("✅ " + (
                    "{Dutch completion message}" if lang == "nl"
                    else "{English completion message}"
                ))

                # Store results
                st.session_state.pipeline_results['{output_data_key}'] = {output_data}
                st.session_state['{stats_key}'] = {stats}

                # Mark step completed
                mark_step_completed({CURRENT_STEP})
                st.rerun()

            except Exception as e:
                st.error(f"{'{Step name}'} fout: {str(e)}" if lang == "nl" else f"{'{Step name}'} error: {str(e)}")
```

### Sample Display Template (in show_step_samples)

```python
elif step_number == {N}:
    # Step {N}: {Description}
    data = cache_manager.load_from_cache(filename, "{cache_step_name}", variable_key, models.{ModelClass})

    if data:
        # Optional: Update sample size tracking
        # st.session_state.step{N+1}_sample_size = len(data)

        # Call step-specific sample display function
        show_{step_name}_samples(data)

        # Show continue button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                f"{'🔄 Ga naar volgende stap' if st.session_state.language == 'nl' else '🔄 Continue to Next Step'}",
                type="primary",
                use_container_width=True,
                key="{step_name}_continue"
            ):
                st.session_state.step = {N+1}
                st.rerun()
    else:
        st.write("⏳ No {step_name} data in cache - run {step_name} first")
```

---

## Step-by-Step Differences

### Complete Step Reference Table

| Step | Function Name | UI Step # | Cache Name | Model Class | Pipeline Function | Input Param | Output Param | Stats Key | Sample Function |
|------|---------------|-----------|------------|-------------|-------------------|-------------|--------------|-----------|-----------------|
| 0 | `show_upload_page` | 0 | `data` | `ResponseModel` | `step_0_load_data` | - | `raw_text_list` | - | - |
| 1 | `show_preprocessing_page` | 1 | `preprocessed` | `PreprocessedModel` | `step_1_preprocess` | `raw_text_list` | `preprocessed_text` | `preprocessing_stats` | `show_preprocessed_samples` |
| 2 | `show_filtering_page` | 2 | `quality_filter` | `QualityFilteredModel` | `step_2_quality_filter` | `preprocessed_text` | `quality_filtered_text` | `quality_filter_stats` | `show_filtered_samples` |
| 3 | `show_idea_extraction_page` | 3 | `extracted_ideas` | `IdeasExtractedModel` | `step_3_extract_ideas` | `quality_filtered_text` | `extracted_ideas` | `idea_extraction_stats` | `show_idea_samples` |
| 4 | `show_embeddings_page` | 4 | `embeddings` | `EmbeddingsModel` | `step_4_generate_embeddings` | `extracted_ideas` | `embeddings` | `embedding_stats` | - |
| 5 | `show_clustering_page` | 5 | `initial_clusters` | `ClusterModel` | `step_5_cluster` | `embeddings` | `clusters` | `clustering_stats` | `show_cluster_samples` |
| 6 | `show_codebook_page` | 6 | `codebook_generation_reasoning` | `CodeGeneratorReasoningResults` | `step_6_generate_codebook` | `clusters` | `codebook_reasoning` | `codebook_stats` | `show_codebook_samples` |
| 7 | `show_theme_page` | 7 | `theme_identification` | `ThemeEnrichedCodebookModel` | `step_7_refine_codebook` | `codebook_reasoning` | `themed_codebook` | `theme_stats` | `show_theme_samples` |
| 8 | `show_refined_codebook_page` | 8 | `refined_codebook` | `RefinedCodebookModel` | `step_8_assign_codes` | `themed_codebook` | `assignments` | `assignment_stats` | `show_step8_refined_codebook` |
| 9 | `show_export_page` | 9 | `export` | - | `step_9_export_results` | `assignments` | `excel_file` | `export_stats` | `show_step9_assignment_stats` |

### Sample Size Tracking

As the pipeline progresses, the "sample size" (number of items being processed) changes:

| After Step | Sample Size Variable | Represents |
|------------|---------------------|------------|
| 0 (Upload) | `sample_size_config` | Number of responses selected from SPSS |
| 1 (Preprocess) | `sample_size_config` | Still original response count |
| 2 (Quality Filter) | `step3_sample_size` | Responses after filtering (excludes filtered) |
| 3 (Extract Ideas) | `step4_sample_size` | **Total ideas extracted** (can be MORE than responses if multiple ideas per response) |
| 4+ | `step4_sample_size` | Remains constant (idea count) |

**Critical Code:**
```python
# Step 2 sample display (line 3111)
valid_responses = sum(1 for item in data if not getattr(item, 'quality_filter', False))
st.session_state.step3_sample_size = valid_responses

# Step 3 sample display (line 3130)
total_ideas = sum(item.idea_count for item in data)
st.session_state.step4_sample_size = total_ideas  # Lock in for remaining steps
```

---

## Critical Implementation Details

### 1. Session State Architecture

**Two Parallel Storage Systems:**

| Storage Location | Purpose | Example Keys | When Populated |
|-----------------|---------|--------------|----------------|
| `st.session_state.{key}` | Configuration, metadata, UI state | `filename`, `var_lab`, `sample_size_config`, `force_recalculate_all` | Upload/Cache button click |
| `st.session_state.pipeline_results['{key}']` | Processing data flowing through pipeline | `raw_text_list`, `preprocessed_text`, `quality_filtered_text` | Processing button click or data loading block |

**Why Both?**
- Configuration keys persist across reruns for UI display
- `pipeline_results` is ephemeral, cleared on new upload
- This separation allows caching system to work correctly

### 2. Completion Tracking

**Functions:**
```python
def is_step_completed(step_number: int) -> bool:
    """Check if a step has been completed"""
    return step_number in st.session_state.completed_steps

def mark_step_completed(step_number: int):
    """Mark a step as completed"""
    if 'completed_steps' not in st.session_state:
        st.session_state.completed_steps = set()
    st.session_state.completed_steps.add(step_number)

    # Update max step reached
    if step_number > st.session_state.get('max_step_reached', 0):
        st.session_state.max_step_reached = step_number
```

**Location:** `src/app.py` lines ~245-265

**Critical Usage:**
- ALWAYS check `is_step_completed({N-1}) and not is_step_completed({N})` before showing processing button
- ALWAYS call `mark_step_completed({N})` immediately after successful processing
- NEVER mark step complete before data is stored in session state

### 3. Variable Key Generation

**Purpose:** Create unique cache key combining filename + variables + sample size + merge config

```python
def generate_enhanced_variable_key(
    selected_variables: List[str],
    is_merged: bool,
    sample_size: Optional[int],
    merge_config: Optional[dict] = None
) -> str:
    """
    Generate enhanced variable key with sample size and merge config.

    Format: "VAR" or "VAR1+VAR2" or "VAR_SS50" or "VAR1+VAR2_SS50_MCconcat"
    """
```

**Location:** `src/app.py` lines ~900-940

**Critical:** ALWAYS use this function to generate `variable_key` before calling any pipeline function or cache operations.

### 4. Force Recalculation Logic

**Two Levels of Force Recalculation:**

```python
# Global: Set on upload, forces ALL steps to recalculate
st.session_state.force_recalculate_all = True  # Upload route
st.session_state.force_recalculate_all = False  # Cache route

# Step-specific: Set when cache is invalidated from a specific step onwards
st.session_state.force_recalculate_from_step = 3  # Reprocess from step 3 onwards

# Combined logic in processing button:
force_recalc = st.session_state.get('force_recalculate_all', False) or \
               (st.session_state.get('force_recalculate_from_step', 99) <= {CURRENT_STEP})
```

**Where Used:**
- Passed to every `pipeline.step_X_...()` function
- Controls whether to load from cache or recalculate
- Essential for correct cache invalidation behavior

### 5. Sample Display Routing (Upload vs Cache)

**Critical Code in `show_step_samples()` (lines 3060-3067):**

```python
# Session-based filtering: Only show results from current session when in force_recalculate mode
if st.session_state.get('force_recalculate_all', False):
    # Upload from file route - only show if step was completed in current session
    # step_number maps directly to completion tracking (preprocessing=1, quality_filter=2, etc.)
    if not is_step_completed(step_number):
        lang = st.session_state.language
        st.write("⏳ " + ("Data nog niet verwerkt in huidige sessie - voer eerst verwerking uit" if lang == "nl" else "Data not yet processed in current session - run processing first"))
        return
```

**What This Does:**
- **Upload route**: User uploads file → `force_recalculate_all = True` → samples NOT shown until button clicked
- **Cache route**: User loads cache → `force_recalculate_all = False` → samples shown immediately from cache

**Why This Matters:**
- Upload route should feel "fresh" - don't show old cached data
- Cache route should feel "instant" - show everything immediately
- This prevents confusing UX where samples appear before processing

### 6. Error Handling Pattern

**Standard Try-Except Structure:**

```python
try:
    # Processing logic
    output_data, stats = pipeline.step_X_...()
    st.session_state.pipeline_results['output_key'] = output_data
    st.session_state['stats_key'] = stats
    mark_step_completed(X)
    st.rerun()
except Exception as e:
    st.error(f"Processing fout: {str(e)}" if lang == "nl" else f"Processing error: {str(e)}")
    # DO NOT mark step as completed
    # DO NOT rerun
```

**Critical:** Never mark step complete or rerun if exception occurs - this prevents broken state.

---

## Agent Specification

### Proposed Specialized Agent: `step-flow-architect`

**Purpose:** Maintain consistency and assist with modifications to step flow architecture across all pipeline steps.

**Capabilities:**

1. **Step Analysis**
   - Analyze any step function to identify deviations from standard pattern
   - Compare two steps and highlight architectural differences
   - Validate that a step follows the 5-block structure

2. **Step Modification**
   - Update stats display formatting in yellow box
   - Add/modify sample display functions
   - Adjust button text and messaging
   - Update pipeline function integration

3. **Step Creation**
   - Generate new step from template
   - Wire up all required session state keys
   - Create corresponding sample display function
   - Update step navigation mapping

4. **Consistency Enforcement**
   - Ensure all steps use `generate_enhanced_variable_key()` correctly
   - Verify force_recalc logic is consistent
   - Check completion tracking is properly implemented
   - Validate error handling follows standard pattern

**Tools Available:**
- Read: Examine existing step implementations
- Edit: Modify step functions following patterns
- Grep: Find all instances of specific patterns
- Glob: Locate related files and functions

**Usage Examples:**

```
Example 1: Update stats display
User: "The preprocessing stats should also show the total processing time"
Agent:
1. Reads show_preprocessing_page() yellow box section (lines 1301-1350)
2. Identifies stats structure from preprocessing_stats
3. Edits to add new time stat display following existing format
4. Verifies stats key exists in pipeline.step_1_preprocess return value

Example 2: Add new step
User: "I need to add a new step 10 for sentiment analysis between clustering and codebook generation"
Agent:
1. Uses template from this document
2. Creates show_sentiment_page() with all 5 blocks
3. Adds show_sentiment_samples() in show_step_samples()
4. Updates step mapping and navigation
5. Creates stub pipeline.step_10_sentiment_analysis() signature

Example 3: Fix consistency issue
User: "Step 6 doesn't follow the same button pattern as step 1"
Agent:
1. Reads show_codebook_page() processing button block
2. Compares to show_preprocessing_page() template (lines 1428-1467)
3. Identifies differences (missing force_recalc logic, different error handling)
4. Edits to match standard pattern
5. Verifies all required components present
```

**Agent Configuration:**

```python
{
    "name": "step-flow-architect",
    "description": "Specialized agent for maintaining step flow architecture consistency in CoderingsTool",
    "tools": ["Read", "Edit", "Grep", "Glob"],
    "reference_docs": [".claude/step_architecture_pattern.md"],
    "reference_implementation": "src/app.py:show_preprocessing_page (lines 1268-1468)",
    "prohibited_actions": [
        "Modify pipeline function signatures without user confirmation",
        "Change cache key generation logic",
        "Alter completion tracking system",
        "Remove error handling"
    ],
    "required_checks": [
        "Verify 5-block structure maintained",
        "Confirm variable_key generation present",
        "Validate force_recalc logic included",
        "Check completion tracking called",
        "Ensure bilingual text (NL/EN) for all UI strings"
    ]
}
```

---

## Appendix: Common Pitfalls and Solutions

### Pitfall 1: Forgetting Data Loading Block

**Symptom:** `KeyError: 'input_data_key'` when processing button clicked

**Cause:** Block 5 (Data Loading) missing or input data not populated

**Solution:** Always include data loading block with proper checks:
```python
if 'input_data_key' not in st.session_state.pipeline_results:
    # Load data here
```

### Pitfall 2: Wrong Completion Check

**Symptom:** Processing button shows at wrong time

**Cause:** Using `is_step_completed({N})` instead of `is_step_completed({N-1}) and not is_step_completed({N})`

**Solution:** Processing button block should check:
- Previous step IS completed
- Current step is NOT completed

### Pitfall 3: Missing force_recalc Flag

**Symptom:** Cache not being invalidated when it should be

**Cause:** Passing `force_recalc=False` or not passing it at all

**Solution:** Always use combined logic:
```python
force_recalc = st.session_state.get('force_recalculate_all', False) or \
               (st.session_state.get('force_recalculate_from_step', 99) <= {CURRENT_STEP})
```

### Pitfall 4: Marking Complete Before Storing Data

**Symptom:** Step marked complete but samples not showing

**Cause:** Calling `mark_step_completed()` before storing results in session state

**Solution:** Always store results BEFORE marking complete:
```python
st.session_state.pipeline_results['output_key'] = output_data
st.session_state['stats_key'] = stats
mark_step_completed(N)  # After storage
```

### Pitfall 5: Not Using generate_enhanced_variable_key()

**Symptom:** Cache misses, data not found

**Cause:** Manually creating variable_key or using outdated method

**Solution:** Always call `generate_enhanced_variable_key()` with all parameters:
```python
variable_key = generate_enhanced_variable_key(
    selected_variables,
    is_merged=is_merged,
    sample_size=sample_size,
    merge_config=merge_config
)
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-19 | 1.0.0 | Initial documentation created based on show_preprocessing_page() |

---

## Related Documentation

- **Pipeline Functions:** `src/pipeline.py` - All step_X_* processing functions
- **Models:** `src/models.py` - Pydantic models for each processing stage
- **Cache System:** `src/utils/cacheManager.py` - Cache operations and invalidation
- **UI Text:** `src/uiTexts.py` - Bilingual text strings for UI elements
- **Configuration:** `src/config.py` - Processing and cache configuration

---

**End of Documentation**
