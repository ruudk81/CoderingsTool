"""Web interface for CoderingsTool - Enhanced Pipeline App"""

import streamlit as st
import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
import time

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "utils"))

import models
from config import (
    ALLOWED_EXTENSIONS, 
    CacheConfig,
    ModelConfig,
    SpellCheckConfig,
    QualityFilterConfig, 
    SegmentationConfig,
    EmbeddingConfig,
    HDBSCANConfig,
    CodeDesignerConfig,
    CodeAssignmentConfig,
    DEFAULT_SPELLCHECK_CONFIG,
    DEFAULT_QUALITY_FILTER_CONFIG,
    DEFAULT_SEGMENTATION_CONFIG,
    DEFAULT_EMBEDDING_CONFIG,
    DEFAULT_HDBSCAN_CONFIG,
    DEFAULT_CODEDESIGNER_CONFIG,
    DEFAULT_CODE_ASSIGNMENT_CONFIG
)
from utils.dataLoader import DataLoader
from utils.cacheManager import CacheManager
import ui_text as ui

# Debug imports
from utils.streamlit_debug import display_debug_controls, create_debug_capture_from_session, display_all_debug_info

# Lazy loading functions to improve startup performance
def _get_pipeline_runner():
    if st.session_state.pipeline_runner is None:
        from pipeline_runner import get_pipeline_runner
        st.session_state.pipeline_runner = get_pipeline_runner()
    return st.session_state.pipeline_runner

def _get_data_loader():
    if st.session_state.data_loader is None:
        st.session_state.data_loader = DataLoader(verbose=False)
    return st.session_state.data_loader

def _get_cache_manager():
    if st.session_state.cache_manager is None:
        st.session_state.cache_manager = CacheManager(CacheConfig())
    return st.session_state.cache_manager

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="CoderingsTool - Survey Response Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'language' not in st.session_state:
    st.session_state.language = ui.DEFAULT_LANGUAGE
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'available_variables' not in st.session_state:
    st.session_state.available_variables = None
if 'selected_variable' not in st.session_state:
    st.session_state.selected_variable = None
if 'selected_id_column' not in st.session_state:
    st.session_state.selected_id_column = None
if 'variable_preview' not in st.session_state:
    st.session_state.variable_preview = None
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = {}
if 'cache_manager' not in st.session_state:
    st.session_state.cache_manager = None  # Lazy load when needed
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None  # Lazy load when needed
if 'pipeline_runner' not in st.session_state:
    st.session_state.pipeline_runner = None  # Lazy load when needed

# Initialize configuration objects for session-specific settings
if 'model_config' not in st.session_state:
    st.session_state.model_config = ModelConfig()
if 'spellcheck_config' not in st.session_state:
    st.session_state.spellcheck_config = SpellCheckConfig()
if 'quality_filter_config' not in st.session_state:
    st.session_state.quality_filter_config = QualityFilterConfig()
if 'segmentation_config' not in st.session_state:
    st.session_state.segmentation_config = SegmentationConfig()
if 'embedding_config' not in st.session_state:
    st.session_state.embedding_config = EmbeddingConfig()
if 'hdbscan_config' not in st.session_state:
    st.session_state.hdbscan_config = HDBSCANConfig()
if 'code_designer_config' not in st.session_state:
    st.session_state.code_designer_config = CodeDesignerConfig()
if 'code_assignment_config' not in st.session_state:
    st.session_state.code_assignment_config = CodeAssignmentConfig()


def show_advanced_settings():
    """Show advanced settings UI in sidebar"""
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("### Pipeline Configuration")
        st.markdown("*Settings apply only to current session*")
        
        # Model options for dropdowns
        gpt4_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]
        gpt5_models = ["gpt-5-mini", "gpt-5-nano", "gpt-5"]
        all_models = gpt4_models + gpt5_models
        embedding_models = ["text-embedding-3-large", "text-embedding-3-small", "gemini-embedding-001"]
        
        # Reasoning and verbosity options
        reasoning_options = ["minimal", "low", "medium", "high"]
        verbosity_options = ["low", "medium", "high"]
        
        # Step 2: Preprocessing
        st.markdown("#### 📝 Step 2: Preprocessing")
        current_spell_model = st.session_state.model_config.get_model_for_stage('spell_check')
        spell_model = st.selectbox(
            "Spell Check Model",
            options=all_models,
            index=all_models.index(current_spell_model) if current_spell_model in all_models else 0,
            key="spell_check_model"
        )
        if spell_model != current_spell_model:
            st.session_state.model_config.spell_check_model = spell_model
        
        st.markdown("---")
        
        # Step 3: Quality Filter
        st.markdown("#### 🔍 Step 3: Quality Filter")
        current_quality_model = st.session_state.model_config.get_model_for_stage('quality_filter')
        quality_model = st.selectbox(
            "Quality Filter Model",
            options=all_models,
            index=all_models.index(current_quality_model) if current_quality_model in all_models else 0,
            key="quality_filter_model"
        )
        if quality_model != current_quality_model:
            st.session_state.model_config.quality_filter_model = quality_model
        
        st.markdown("---")
        
        # Step 4: Idea Extraction  
        st.markdown("#### 💡 Step 4: Idea Extraction")
        current_seg_model = st.session_state.model_config.get_model_for_stage('segmentation')
        seg_model = st.selectbox(
            "Segmentation Model",
            options=all_models,
            index=all_models.index(current_seg_model) if current_seg_model in all_models else 0,
            key="segmentation_model"
        )
        if seg_model != current_seg_model:
            st.session_state.model_config.segmentation_model = seg_model
        
        st.markdown("---")
        
        # Step 5: Embeddings
        st.markdown("#### 🔗 Step 5: Embeddings")
        current_emb_model = st.session_state.model_config.get_model_for_stage('embedding')
        emb_model = st.selectbox(
            "Embedding Model",
            options=embedding_models,
            index=embedding_models.index(current_emb_model) if current_emb_model in embedding_models else 0,
            key="embedding_model"
        )
        if emb_model != current_emb_model:
            st.session_state.model_config.embedding_model = emb_model
        
        st.markdown("---")
        
        # Step 6: Clustering (CRITICAL)
        st.markdown("#### 📊 Step 6: Clustering ⭐")
        st.markdown("*These parameters significantly impact clustering results*")
        
        epsilon = st.slider(
            "Cluster Selection Epsilon",
            min_value=0.1,
            max_value=1.0,
            value=float(st.session_state.hdbscan_config.cluster_selection_epsilon or 0.5),
            step=0.1,
            help="Controls cluster granularity. Lower = more clusters, Higher = fewer clusters",
            key="cluster_epsilon"
        )
        if epsilon != st.session_state.hdbscan_config.cluster_selection_epsilon:
            st.session_state.hdbscan_config.cluster_selection_epsilon = epsilon
        
        alpha = st.slider(
            "Alpha (Size vs Distance Balance)",
            min_value=0.5,
            max_value=2.0,
            value=float(st.session_state.hdbscan_config.alpha or 1.0),
            step=0.1,
            help="Balances cluster size vs distance. 1.0 = default, >1.0 = prefer larger clusters",
            key="cluster_alpha"
        )
        if alpha != st.session_state.hdbscan_config.alpha:
            st.session_state.hdbscan_config.alpha = alpha
        
        st.markdown("---")
        
        # Step 7: Code Generation (CRITICAL)
        st.markdown("#### 🏗️ Step 7: Code Generation ⭐")
        st.markdown("*Core code generation models and parameters*")
        
        # Theme Summary Model
        current_theme_model = st.session_state.model_config.get_model_for_stage('theme_extraction')
        theme_model = st.selectbox(
                "Theme Summary Model",
                options=gpt5_models + gpt4_models,
                index=(gpt5_models + gpt4_models).index(current_theme_model) if current_theme_model in (gpt5_models + gpt4_models) else 0,
                key="theme_summary_model"
        )
        if theme_model != current_theme_model:
            st.session_state.model_config.thematic_summary_model = theme_model
        
        # Candidate Selection Model
            current_candidate_model = st.session_state.model_config.get_model_for_stage('candidate_selection')
            candidate_model = st.selectbox(
                "Candidate Selection Model",
                options=gpt5_models + gpt4_models,
                index=(gpt5_models + gpt4_models).index(current_candidate_model) if current_candidate_model in (gpt5_models + gpt4_models) else 0,
                key="candidate_selection_model"
            )
            if candidate_model != current_candidate_model:
                st.session_state.model_config.candidate_selection_model = candidate_model
            
            # Code Generation Model
            current_codegen_model = st.session_state.model_config.get_model_for_stage('code_recommendation')
            codegen_model = st.selectbox(
                "Code Generation Model",
                options=gpt5_models + gpt4_models,
                index=(gpt5_models + gpt4_models).index(current_codegen_model) if current_codegen_model in (gpt5_models + gpt4_models) else 0,
                key="code_generation_model"
            )
            if codegen_model != current_codegen_model:
                st.session_state.model_config.code_generation_model = codegen_model
            
            # Validation Model
            current_validation_model = st.session_state.model_config.get_model_for_stage('recommendation_validation')
            validation_model = st.selectbox(
                "Validation Model",
                options=gpt5_models + gpt4_models,
                index=(gpt5_models + gpt4_models).index(current_validation_model) if current_validation_model in (gpt5_models + gpt4_models) else 0,
                key="validation_model"
            )
            if validation_model != current_validation_model:
                st.session_state.model_config.validation_model = validation_model
            
            st.markdown("**GPT-5 Reasoning Parameters**")
            
            # Theme Extraction Parameters
            st.markdown("*Theme Extraction*")
            theme_reasoning = st.selectbox(
                "Reasoning Effort",
                options=reasoning_options,
                index=reasoning_options.index(st.session_state.model_config.theme_extraction_reasoning_effort),
                key="theme_reasoning"
            )
            if theme_reasoning != st.session_state.model_config.theme_extraction_reasoning_effort:
                st.session_state.model_config.theme_extraction_reasoning_effort = theme_reasoning
            
            theme_verbosity = st.selectbox(
                "Text Verbosity",
                options=verbosity_options,
                index=verbosity_options.index(st.session_state.model_config.theme_extraction_text_verbosity),
                key="theme_verbosity"
            )
            if theme_verbosity != st.session_state.model_config.theme_extraction_text_verbosity:
                st.session_state.model_config.theme_extraction_text_verbosity = theme_verbosity
            
            # Candidate Selection Parameters
            st.markdown("*Candidate Selection*")
            candidate_reasoning = st.selectbox(
                "Reasoning Effort",
                options=reasoning_options,
                index=reasoning_options.index(st.session_state.model_config.candidate_selection_reasoning_effort),
                key="candidate_reasoning"
            )
            if candidate_reasoning != st.session_state.model_config.candidate_selection_reasoning_effort:
                st.session_state.model_config.candidate_selection_reasoning_effort = candidate_reasoning
            
            candidate_verbosity = st.selectbox(
                "Text Verbosity",
                options=verbosity_options,
                index=verbosity_options.index(st.session_state.model_config.candidate_selection_text_verbosity),
                key="candidate_verbosity"
            )
            if candidate_verbosity != st.session_state.model_config.candidate_selection_text_verbosity:
                st.session_state.model_config.candidate_selection_text_verbosity = candidate_verbosity
            
            # Code Generation Parameters
            st.markdown("*Code Generation*")
            codegen_reasoning = st.selectbox(
                "Reasoning Effort",
                options=reasoning_options,
                index=reasoning_options.index(st.session_state.model_config.code_generation_reasoning_effort),
                key="codegen_reasoning"
            )
            if codegen_reasoning != st.session_state.model_config.code_generation_reasoning_effort:
                st.session_state.model_config.code_generation_reasoning_effort = codegen_reasoning
            
            codegen_verbosity = st.selectbox(
                "Text Verbosity",
                options=verbosity_options,
                index=verbosity_options.index(st.session_state.model_config.code_generation_text_verbosity),
                key="codegen_verbosity"
            )
            if codegen_verbosity != st.session_state.model_config.code_generation_text_verbosity:
                st.session_state.model_config.code_generation_text_verbosity = codegen_verbosity
            
            # Validation Parameters
            st.markdown("*Validation*")
            validation_reasoning = st.selectbox(
                "Reasoning Effort",
                options=reasoning_options,
                index=reasoning_options.index(st.session_state.model_config.validation_reasoning_effort),
                key="validation_reasoning"
            )
            if validation_reasoning != st.session_state.model_config.validation_reasoning_effort:
                st.session_state.model_config.validation_reasoning_effort = validation_reasoning
            
            validation_verbosity = st.selectbox(
                "Text Verbosity",
                options=verbosity_options,
                index=verbosity_options.index(st.session_state.model_config.validation_text_verbosity),
                key="validation_verbosity"
            )
            if validation_verbosity != st.session_state.model_config.validation_text_verbosity:
                st.session_state.model_config.validation_text_verbosity = validation_verbosity
        
        st.markdown("---")
        
        # Step 9: Code Assignment
        st.markdown("#### 🎯 Step 9: Code Assignment")
        current_assign_model = st.session_state.model_config.get_model_for_stage('code_assignment')
        assign_model = st.selectbox(
            "Code Assignment Model",
            options=all_models,
            index=all_models.index(current_assign_model) if current_assign_model in all_models else 0,
            key="code_assignment_model"
        )
        if assign_model != current_assign_model:
            st.session_state.model_config.code_assignment_model = assign_model
        
        top_k = st.number_input(
            "Top K Similar Codes",
            min_value=1,
            max_value=10,
            value=st.session_state.code_assignment_config.top_k_similar_codes,
            help="Number of most similar codes to present to the model",
            key="top_k_codes"
        )
        if top_k != st.session_state.code_assignment_config.top_k_similar_codes:
            st.session_state.code_assignment_config.top_k_similar_codes = top_k
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.code_assignment_config.min_confidence_threshold,
            step=0.1,
            help="Minimum confidence for valid assignment",
            key="confidence_threshold"
        )
        if confidence != st.session_state.code_assignment_config.min_confidence_threshold:
            st.session_state.code_assignment_config.min_confidence_threshold = confidence
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=5,
            max_value=50,
            value=st.session_state.code_assignment_config.batch_size,
            help="Ideas processed per batch",
            key="assignment_batch_size"
        )
        if batch_size != st.session_state.code_assignment_config.batch_size:
            st.session_state.code_assignment_config.batch_size = batch_size
        
        # Reset to defaults button
        if st.button("🔄 Reset All to Defaults", type="secondary"):
            st.session_state.model_config = ModelConfig()
            st.session_state.spellcheck_config = SpellCheckConfig()
            st.session_state.quality_filter_config = QualityFilterConfig()
            st.session_state.segmentation_config = SegmentationConfig()
            st.session_state.embedding_config = EmbeddingConfig()
            st.session_state.hdbscan_config = HDBSCANConfig()
            st.session_state.code_designer_config = CodeDesignerConfig()
            st.session_state.code_assignment_config = CodeAssignmentConfig()
            # Clear stored variable key to ensure fresh cache naming
            if 'current_variable_key' in st.session_state:
                del st.session_state.current_variable_key
            st.rerun()


def main():
    st.title(ui.get_text("APP_TITLE", st.session_state.language))
    st.markdown(ui.get_text("APP_DESCRIPTION", st.session_state.language))
    
    # Sidebar
    with st.sidebar:
        # Language selector at the top
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**{ui.get_text('LANGUAGE_LABEL', st.session_state.language)}**")
        with col2:
            language_options = {"Nederlands": "nl", "English": "en"}
            current_language_name = next(k for k, v in language_options.items() if v == st.session_state.language)
            selected_language = st.selectbox(
                "Language",
                options=list(language_options.keys()),
                index=list(language_options.keys()).index(current_language_name),
                label_visibility="collapsed"
            )
            if language_options[selected_language] != st.session_state.language:
                st.session_state.language = language_options[selected_language]
                st.rerun()
        
        st.markdown("---")
        
        st.header(ui.get_text("SIDEBAR_HEADER", st.session_state.language))
        st.markdown(ui.get_text("SIDEBAR_DESCRIPTION", st.session_state.language))
        
        # Progress indicator - Updated to 10 steps
        progress = st.progress(st.session_state.step / 10)
        st.markdown(f"**{ui.get_text('CURRENT_STEP', st.session_state.language)}** {st.session_state.step + 1}/10")
        
        st.markdown("---")
        
        # Advanced Settings
        show_advanced_settings()
        
        # Debug Controls
        st.markdown("---")
        display_debug_controls()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.step == 0:
            show_upload_page()
        elif st.session_state.step == 1:
            show_preprocessing_page()
        elif st.session_state.step == 2:
            show_filtering_page()
        elif st.session_state.step == 3:
            show_idea_extraction_page()
        elif st.session_state.step == 4:
            show_embedding_page()
        elif st.session_state.step == 5:
            show_clustering_page()
        elif st.session_state.step == 6:
            show_codebook_generation_page()
        elif st.session_state.step == 7:
            show_theme_identification_page()
        elif st.session_state.step == 8:
            show_code_assignment_page()
        elif st.session_state.step == 9:
            show_export_page()
        elif st.session_state.step == 10:
            show_results_page()
    
    with col2:
        show_info_panel()

def show_upload_page():
    lang = st.session_state.language
    st.header(f"Stap 1: {ui.get_text('BTN_UPLOAD', lang)}" if lang == "nl" else "Step 1: Upload Data")
    
    uploaded_file = st.file_uploader(
        "Kies een SPSS bestand (.sav)" if lang == "nl" else "Choose a SPSS file (.sav)",
        type=['sav'],
        help=ui.get_text("UPLOAD_HELP", lang)
    )
    
    if uploaded_file is not None:
        if st.button(ui.get_text("BTN_UPLOAD", lang), type="primary"):
            with st.spinner("Data wordt geladen..." if lang == "nl" else "Loading data..."):
                try:
                    # Save uploaded file
                    file_path = Path("data") / uploaded_file.name
                    file_path.parent.mkdir(exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.filename = uploaded_file.name
                    st.session_state.uploaded_file_path = str(file_path)
                    
                    # Load variables from SPSS file
                    try:
                        variables = _get_data_loader().list_variables(uploaded_file.name)
                        st.session_state.available_variables = variables
                        st.success(f"Bestand geladen met {len(variables)} variabelen!" if lang == "nl" else f"File loaded with {len(variables)} variables!")
                        # Don't advance to step 1 yet - let user select variables first
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fout bij het laden van variabelen: {str(e)}" if lang == "nl" else f"Error loading variables: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Fout bij het uploaden: {str(e)}" if lang == "nl" else f"Upload error: {str(e)}")
    
    # Show variable selection if file is uploaded
    if st.session_state.available_variables:
        st.subheader("Variabele Selectie" if lang == "nl" else "Variable Selection")
        
        # Show available variables in a nice table
        var_data = []
        for var_name, var_label in st.session_state.available_variables.items():
            var_data.append({"Variable": var_name, "Label": var_label or "(No label)"})
        
        if var_data:
            st.dataframe(pd.DataFrame(var_data), use_container_width=True)
            
            # Variable selection mode
            st.subheader("📝 " + ("Variabele Selectie" if lang == "nl" else "Variable Selection"))
            
            # Single vs Multiple variable toggle
            variable_mode = st.radio(
                "Selectie Mode" if lang == "nl" else "Selection Mode",
                ["single", "multiple"],
                format_func=lambda x: "Enkele variabele" if x == "single" and lang == "nl" 
                                    else "Single variable" if x == "single"
                                    else "Meerdere variabelen" if lang == "nl"
                                    else "Multiple variables",
                key="variable_mode",
                horizontal=True,
                help="Selecteer enkele variabele voor standaard analyse, of meerdere voor tekstsamenvoeging" if lang == "nl"
                     else "Select single variable for standard analysis, or multiple for text merging"
            )
            
            # ID column selection (always needed)
            id_var = st.selectbox(
                "🆔 " + ("Selecteer ID kolom" if lang == "nl" else "Select ID column"),
                options=list(st.session_state.available_variables.keys()),
                format_func=lambda x: f"{x} - {st.session_state.available_variables[x] or '(No label)'}",
                key="id_variable"
            )
            
            # Variable selection based on mode
            if variable_mode == "single":
                # Single variable selection
                text_var = st.selectbox(
                    "📄 " + ("Selecteer tekst variabele" if lang == "nl" else "Select text variable"),
                    options=list(st.session_state.available_variables.keys()),
                    format_func=lambda x: f"{x} - {st.session_state.available_variables[x] or '(No label)'}",
                    key="text_variable"
                )
                selected_variables = [text_var] if text_var else []
            else:
                # Multiple variable selection
                selected_variables = st.multiselect(
                    "📄 " + ("Selecteer tekst variabelen om samen te voegen" if lang == "nl" 
                           else "Select text variables to merge"),
                    options=list(st.session_state.available_variables.keys()),
                    format_func=lambda x: f"{x} - {st.session_state.available_variables[x] or '(No label)'}",
                    key="text_variables_multi",
                    help="Selecteer meerdere variabelen die samengevoegd zullen worden tot één tekst" if lang == "nl"
                         else "Select multiple variables that will be merged into one text"
                )
                
                # Merge configuration for multiple variables
                if selected_variables and len(selected_variables) > 1:
                    with st.expander("🔧 " + ("Samenvoeg Opties" if lang == "nl" else "Merge Options"), expanded=True):
                        merge_col1, merge_col2 = st.columns(2)
                        
                        with merge_col1:
                            merge_strategy = st.selectbox(
                                "Samenvoeg Strategie" if lang == "nl" else "Merge Strategy",
                                ["concatenate", "first_available", "all_combined"],
                                format_func=lambda x: {
                                    "concatenate": "Alles samenvoegen" if lang == "nl" else "Concatenate all",
                                    "first_available": "Eerste beschikbare" if lang == "nl" else "First available",
                                    "all_combined": "Alle met labels" if lang == "nl" else "All with labels"
                                }[x],
                                key="merge_strategy",
                                help="Kies hoe meerdere variabelen samengevoegd worden" if lang == "nl"
                                     else "Choose how multiple variables are merged"
                            )
                        
                        with merge_col2:
                            separator_options = [" ", "\n", " | ", "; ", ", "]
                            separator = st.selectbox(
                                "Scheidingsteken" if lang == "nl" else "Separator",
                                separator_options,
                                format_func=lambda x: {
                                    " ": "Spatie" if lang == "nl" else "Space",
                                    "\n": "Nieuwe regel" if lang == "nl" else "New line", 
                                    " | ": "Pijp symbool" if lang == "nl" else "Pipe symbol",
                                    "; ": "Puntkomma" if lang == "nl" else "Semicolon",
                                    ", ": "Komma" if lang == "nl" else "Comma"
                                }[x],
                                key="merge_separator",
                                help="Scheidingsteken tussen samengevoegde teksten" if lang == "nl"
                                     else "Separator between merged texts"
                            )
                        
                        skip_empty = st.checkbox(
                            "Lege waarden overslaan" if lang == "nl" else "Skip empty values",
                            value=True,
                            key="skip_empty",
                            help="Variabelen zonder inhoud niet opnemen in samengevoegde tekst" if lang == "nl"
                                 else "Don't include variables without content in merged text"
                        )
                
                # Set text_var for backward compatibility
                text_var = selected_variables[0] if selected_variables else None
            
            # Advanced encoding options
            with st.expander("🔧 " + ("Geavanceerde Opties" if lang == "nl" else "Advanced Options"), expanded=False):
                encoding_options = [
                    ("Automatisch detecteren", "auto"),
                    ("UTF-8", "utf-8"),
                    ("Windows-1252 (West-Europees)", "windows-1252"),
                    ("ISO-8859-1 (Latin-1)", "iso-8859-1"),
                    ("CP1252 (Windows West-Europees)", "cp1252"),
                    ("ISO-8859-15 (Latin-9, Euro)", "iso-8859-15"),
                    ("Windows-1250 (Centraal-Europees)", "windows-1250")
                ]
                
                encoding_choice = st.selectbox(
                    "Bestandscodering" if lang == "nl" else "File Encoding",
                    options=[opt[1] for opt in encoding_options],
                    format_func=lambda x: next(opt[0] for opt in encoding_options if opt[1] == x),
                    index=0,  # Default to auto-detect
                    key="file_encoding",
                    help="Kies een specifieke codering als het bestand niet correct wordt geladen" if lang == "nl" 
                         else "Choose a specific encoding if the file is not loading correctly"
                )
                
                # Show encoding success message if available
                if hasattr(st.session_state, 'encoding_success_message'):
                    st.success(st.session_state.encoding_success_message)
                    del st.session_state.encoding_success_message  # Clear after showing
            
            # Preview selected variable(s)
            preview_button_label = "Voorbeeld Bekijken" if lang == "nl" else "Preview Variables"
            if variable_mode == "multiple" and len(selected_variables) > 1:
                preview_button_label = f"Voorbeeld van {len(selected_variables)} variabelen" if lang == "nl" else f"Preview {len(selected_variables)} variables"
            
            if st.button(preview_button_label):
                if selected_variables and id_var:
                    with st.spinner("Data wordt geladen..." if lang == "nl" else "Loading data..."):
                        try:
                            # Use selected encoding, None if auto-detect
                            encoding = st.session_state.get('file_encoding', 'auto')
                            encoding = None if encoding == 'auto' else encoding
                            
                            if variable_mode == "single" or len(selected_variables) == 1:
                                # Single variable preview
                                preview_data = _get_data_loader().get_variable_with_IDs(
                                    st.session_state.filename, id_var, selected_variables[0], encoding=encoding
                                )
                                st.session_state.variable_preview = preview_data
                                st.session_state.selected_variable = selected_variables[0]
                                st.session_state.selected_variables = selected_variables
                                # Don't modify variable_mode here - it's tied to the widget
                                # Instead, store confirmed values separately
                                st.session_state.variable_mode_confirmed = variable_mode
                                st.session_state.selected_variables_confirmed = selected_variables
                                st.session_state.is_merged_variable = False
                            else:
                                # Multiple variables preview - use merge functionality
                                merge_strategy = st.session_state.get('merge_strategy', 'concatenate')
                                separator = st.session_state.get('merge_separator', ' ')
                                skip_empty = st.session_state.get('skip_empty', True)
                                
                                preview_data = _get_data_loader().get_multiple_variables_with_IDs(
                                    filename=st.session_state.filename,
                                    id_column=id_var,
                                    var_names=selected_variables,
                                    merge_strategy=merge_strategy,
                                    separator=separator,
                                    skip_empty=skip_empty,
                                    encoding=encoding
                                )
                                st.session_state.variable_preview = preview_data
                                st.session_state.selected_variable = "merged_text"  # For backward compatibility
                                st.session_state.selected_variables = selected_variables
                                # Don't modify variable_mode here - it's tied to the widget
                                st.session_state.merge_config = {
                                    'strategy': merge_strategy,
                                    'separator': separator,
                                    'skip_empty': skip_empty
                                }
                                # Ensure persistence for merged variables using separate keys
                                st.session_state.variable_mode_confirmed = variable_mode
                                st.session_state.selected_variables_confirmed = selected_variables
                                st.session_state.merge_config_confirmed = st.session_state.merge_config
                                st.session_state.is_merged_variable = True
                            
                            st.session_state.selected_id_column = id_var
                            st.success("Preview geladen!" if lang == "nl" else "Preview loaded!")
                        except Exception as e:
                            st.error(f"Fout bij preview: {str(e)}" if lang == "nl" else f"Preview error: {str(e)}")
                else:
                    st.warning("Selecteer eerst variabelen en ID kolom" if lang == "nl" else "Please select variables and ID column first")
            
            # Show preview if available
            if st.session_state.variable_preview is not None:
                st.subheader("📊 Data Preview")
                preview_df = st.session_state.variable_preview
                
                # Determine the text column name based on mode
                text_column = st.session_state.selected_variable
                if st.session_state.get('variable_mode') == 'multiple' and text_column == 'merged_text':
                    display_text_column = 'merged_text'
                else:
                    display_text_column = text_column
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Totaal" if lang == "nl" else "Total", len(preview_df))
                with col2:
                    non_null = preview_df[display_text_column].notna().sum()
                    st.metric("Niet-leeg" if lang == "nl" else "Non-empty", non_null)
                with col3:
                    unique_vals = preview_df[display_text_column].nunique()
                    st.metric("Uniek" if lang == "nl" else "Unique", unique_vals)
                
                # Show merge information for multiple variables
                if st.session_state.get('variable_mode') == 'multiple' and len(st.session_state.get('selected_variables', [])) > 1:
                    merge_config = st.session_state.get('merge_config', {})
                    st.info(
                        f"🔗 **Samengevoegd:** {len(st.session_state.selected_variables)} variabelen "
                        f"({', '.join(st.session_state.selected_variables)}) | "
                        f"**Strategie:** {merge_config.get('strategy', 'concatenate')} | "
                        f"**Scheidingsteken:** '{merge_config.get('separator', ' ')}'" 
                        if lang == "nl" else
                        f"🔗 **Merged:** {len(st.session_state.selected_variables)} variables "
                        f"({', '.join(st.session_state.selected_variables)}) | "
                        f"**Strategy:** {merge_config.get('strategy', 'concatenate')} | "
                        f"**Separator:** '{merge_config.get('separator', ' ')}'"
                    )
                
                # Show sample data
                st.subheader("📝 " + ("Voorbeeldgegevens" if lang == "nl" else "Sample Data"))
                sample_data = preview_df[preview_df[display_text_column].notna()].head(10)
                if len(sample_data) > 0:
                    st.dataframe(sample_data, use_container_width=True)
                else:
                    st.warning("Geen niet-lege gegevens gevonden" if lang == "nl" else "No non-empty data found")
                
                # Ready to proceed button
                if st.button("Doorgaan naar Preprocessing" if lang == "nl" else "Continue to Preprocessing", type="primary"):
                    # Double-check persistence of merge-related session state before proceeding
                    current_mode = variable_mode  # Use the current widget value instead of session_state
                    if current_mode == 'multiple' and len(selected_variables) > 1:
                        # Ensure merge configuration is properly stored
                        if 'variable_mode_confirmed' not in st.session_state:
                            st.session_state['variable_mode_confirmed'] = current_mode
                        if 'selected_variables_confirmed' not in st.session_state:
                            st.session_state['selected_variables_confirmed'] = selected_variables
                        if 'merge_config_confirmed' not in st.session_state and 'merge_config' in st.session_state:
                            st.session_state['merge_config_confirmed'] = st.session_state['merge_config']
                        st.session_state['is_merged_variable'] = True
                    else:
                        st.session_state['is_merged_variable'] = False
                    
                    st.session_state.step = 1
                    st.rerun()

def show_preprocessing_page():
    lang = st.session_state.language
    st.header("Stap 2: Preprocessing" if lang == "nl" else "Step 2: Preprocessing")
    st.markdown(ui.get_text("PREPROCESSING_INFO", lang))
    
    # Show current selection with better validation for merged variables
    if st.session_state.selected_variable and st.session_state.selected_id_column:
        # Check if this is a merged variable scenario (use confirmed values to avoid widget conflicts)
        is_multiple_mode = (st.session_state.get('variable_mode_confirmed') == 'multiple' or
                          st.session_state.get('is_merged_variable', False))
        
        selected_vars = (st.session_state.get('selected_variables') or 
                        st.session_state.get('selected_variables_confirmed', []))
        
        if is_multiple_mode and len(selected_vars) > 1:
            merge_config = (st.session_state.get('merge_config') or 
                           st.session_state.get('merge_config_confirmed', {}))
            st.info(
                f"**Samengevoegde Variabelen:** {', '.join(selected_vars)}\n\n"
                f"**ID Kolom:** {st.session_state.selected_id_column}\n\n"
                f"**Samenvoeg Strategie:** {merge_config.get('strategy', 'concatenate')}"
                if lang == "nl" else
                f"**Merged Variables:** {', '.join(selected_vars)}\n\n"
                f"**ID Column:** {st.session_state.selected_id_column}\n\n"
                f"**Merge Strategy:** {merge_config.get('strategy', 'concatenate')}"
            )
        else:
            st.info(f"**Variabele:** {st.session_state.selected_variable}\n\n**ID Kolom:** {st.session_state.selected_id_column}")
    else:
        # Enhanced error message with debug information
        missing_items = []
        if not st.session_state.selected_variable:
            missing_items.append("selected_variable")
        if not st.session_state.selected_id_column:
            missing_items.append("selected_id_column")
        
        error_msg = (
            f"Ga terug en selecteer een variabele. Ontbrekend: {', '.join(missing_items)}"
            if lang == "nl" else
            f"Go back and select a variable. Missing: {', '.join(missing_items)}"
        )
        st.warning(error_msg)
        
        # Debug information for development
        with st.expander("🔧 Debug Info" if lang == "en" else "🔧 Debug Informatie"):
            st.write("Session State Variables:")
            st.write(f"- selected_variable: {st.session_state.get('selected_variable')}")
            st.write(f"- selected_id_column: {st.session_state.get('selected_id_column')}")
            st.write(f"- variable_mode_confirmed: {st.session_state.get('variable_mode_confirmed')}")
            st.write(f"- selected_variables: {st.session_state.get('selected_variables')}")
            st.write(f"- selected_variables_confirmed: {st.session_state.get('selected_variables_confirmed')}")
            st.write(f"- merge_config: {st.session_state.get('merge_config')}")
            st.write(f"- merge_config_confirmed: {st.session_state.get('merge_config_confirmed')}")
            st.write(f"- is_merged_variable: {st.session_state.get('is_merged_variable')}")
        return
    
    # Check if we're waiting for debug continue
    if st.session_state.get('waiting_for_debug_continue_preprocessing'):
        # Display the stored debug information
        debug_capture = st.session_state.get('debug_capture_preprocessing')
        if debug_capture:
            display_all_debug_info(debug_capture)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="preprocessing_continue"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_preprocessing'] 
                if 'debug_capture_preprocessing' in st.session_state:
                    del st.session_state['debug_capture_preprocessing']
                st.session_state.step = 2
                st.rerun()
    elif st.button(ui.get_text("BTN_PREPROCESS", lang), type="primary"):
        progress_container = st.empty()
        
        # Create debug capture from session state
        debug_capture = create_debug_capture_from_session()
        
        try:
            # Step 1: Load data if not already loaded
            if 'raw_text_list' not in st.session_state.pipeline_results:
                # Use selected encoding, None if auto-detect
                encoding = st.session_state.get('file_encoding', 'auto')
                encoding = None if encoding == 'auto' else encoding
                
                # Handle variable label for single vs multiple variables (use confirmed values to avoid widget conflicts)
                is_multiple_mode = (st.session_state.get('variable_mode_confirmed') == 'multiple' or
                                  st.session_state.get('is_merged_variable', False))
                
                selected_vars = (st.session_state.get('selected_variables') or 
                                st.session_state.get('selected_variables_confirmed', []))
                
                if is_multiple_mode and len(selected_vars) > 1:
                    # Multiple variables - create combined label
                    merge_config = (st.session_state.get('merge_config') or 
                                   st.session_state.get('merge_config_confirmed', {}))
                    var_labels = []
                    for var in selected_vars:
                        label = _get_data_loader().get_varlab(st.session_state.filename, var, encoding=encoding)
                        var_labels.append(label or var)
                    var_lab = f"Combined ({merge_config.get('strategy', 'concatenate')}): {' + '.join(var_labels)}"
                    
                    # Load multiple variables
                    raw_text_list = _get_pipeline_runner().step_1_load_data(
                        filename=st.session_state.filename,
                        id_column=st.session_state.selected_id_column,
                        var_names=selected_vars,
                        streamlit_container=progress_container,
                        encoding=encoding
                    )
                else:
                    # Single variable (backward compatibility)
                    var_lab = _get_data_loader().get_varlab(st.session_state.filename, st.session_state.selected_variable, encoding=encoding)
                    raw_text_list = _get_pipeline_runner().step_1_load_data(
                        filename=st.session_state.filename,
                        id_column=st.session_state.selected_id_column,
                        var_name=st.session_state.selected_variable,
                        streamlit_container=progress_container,
                        encoding=encoding
                    )
                st.session_state.pipeline_results['raw_text_list'] = raw_text_list
                st.session_state.pipeline_results['var_lab'] = var_lab
            
            # Step 2: Preprocessing
            preprocessed_text = _get_pipeline_runner().step_2_preprocess(
                raw_text_list=st.session_state.pipeline_results['raw_text_list'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                spellcheck_config=st.session_state.spellcheck_config,
                streamlit_container=progress_container,
                debug_capture=debug_capture
            )
            st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text
            
            # Check if debug features are enabled and have captured data
            debug_has_data = (debug_capture and 
                            (debug_capture.verbose_outputs or 
                             debug_capture.first_prompts or 
                             debug_capture.sample_results))
            
            if debug_has_data:
                # Store debug capture and set waiting state
                st.session_state['debug_capture_preprocessing'] = debug_capture
                st.session_state['waiting_for_debug_continue_preprocessing'] = True
                st.rerun()  # Rerun to show the continue button interface
            else:
                # No debug data - advance automatically
                st.session_state.step = 2
                st.rerun()
        except Exception as e:
            progress_container.error(f"Preprocessing fout: {str(e)}" if lang == "nl" else f"Preprocessing error: {str(e)}")

def show_filtering_page():
    lang = st.session_state.language
    st.header("Stap 3: Kwaliteitsfiltering" if lang == "nl" else "Step 3: Quality Filtering")
    st.markdown(ui.get_text("FILTERING_INFO", lang))
    
    if st.button(ui.get_text("BTN_FILTER", lang), type="primary"):
        progress_container = st.empty()
        try:
            quality_filtered_text = _get_pipeline_runner().step_3_quality_filter(
                preprocessed_text=st.session_state.pipeline_results['preprocessed_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                quality_filter_config=st.session_state.quality_filter_config,
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text
            st.session_state.step = 3
            st.rerun()
        except Exception as e:
            progress_container.error(f"Filtering fout: {str(e)}" if lang == "nl" else f"Filtering error: {str(e)}")

def show_idea_extraction_page():
    lang = st.session_state.language
    st.header("Stap 4: Idee Extractie" if lang == "nl" else "Step 4: Idea Extraction")
    
    info_text = """
    Deze stap zal:
    - Responsies segmenteren in discrete ideeën
    - Elke idee een unieke ID geven
    - Voorbereiden voor embedding generatie
    """ if lang == "nl" else """
    This step will:
    - Segment responses into discrete ideas  
    - Assign unique IDs to each idea
    - Prepare for embedding generation
    """
    st.markdown(info_text)
    
    # Check if we're waiting for debug continue
    if st.session_state.get('waiting_for_debug_continue_idea_extraction'):
        # Display the stored debug information
        debug_capture = st.session_state.get('debug_capture_idea_extraction')
        if debug_capture:
            display_all_debug_info(debug_capture)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="idea_extraction_continue"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_idea_extraction']
                if 'debug_capture_idea_extraction' in st.session_state:
                    del st.session_state['debug_capture_idea_extraction']
                st.session_state.step = 4
                st.rerun()
    elif st.button("Start Idee Extractie" if lang == "nl" else "Start Idea Extraction", type="primary"):
        progress_container = st.empty()
        
        # Create debug capture from session state
        debug_capture = create_debug_capture_from_session()
        
        try:
            encoded_text = _get_pipeline_runner().step_4_extract_ideas(
                quality_filtered_text=st.session_state.pipeline_results['quality_filtered_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                segmentation_config=st.session_state.segmentation_config,
                streamlit_container=progress_container,
                debug_capture=debug_capture
            )
            st.session_state.pipeline_results['encoded_text'] = encoded_text
            
            # Check if debug features are enabled and have captured data
            debug_has_data = (debug_capture and 
                            (debug_capture.verbose_outputs or 
                             debug_capture.first_prompts or 
                             debug_capture.sample_results))
            
            if debug_has_data:
                # Store debug capture and set waiting state
                st.session_state['debug_capture_idea_extraction'] = debug_capture
                st.session_state['waiting_for_debug_continue_idea_extraction'] = True
                st.rerun()  # Rerun to show the continue button interface
            else:
                # No debug data - advance automatically
                st.session_state.step = 4
                st.rerun()
        except Exception as e:
            progress_container.error(f"Extractie fout: {str(e)}" if lang == "nl" else f"Extraction error: {str(e)}")

def show_embedding_page():
    lang = st.session_state.language
    st.header("Stap 5: Genereer Embeddings" if lang == "nl" else "Step 5: Generate Embeddings")
    st.markdown(ui.get_text("EMBEDDING_INFO", lang))
    
    # Embedding provider selection
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox(
            "Embedding Provider",
            options=["gemini", "openai"],
            index=0
        )
    with col2:
        model = st.selectbox(
            "Model",
            options=[
                "gemini-embedding-001" if provider == "gemini" else "text-embedding-3-large",
                "text-embedding-3-small" if provider == "openai" else "gemini-embedding-001"
            ] if provider == "openai" else ["gemini-embedding-001"]
        )
    
    if st.button(ui.get_text("BTN_EMBED", lang), type="primary"):
        progress_container = st.empty()
        try:
            # Update embedding config with UI values
            embedding_config = st.session_state.embedding_config
            
            embedded_text = _get_pipeline_runner().step_5_generate_embeddings(
                encoded_text=st.session_state.pipeline_results['encoded_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                embedding_config=embedding_config,
                provider=provider,
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['embedded_text'] = embedded_text
            st.session_state.step = 5
            st.rerun()
        except Exception as e:
            progress_container.error(f"Embedding fout: {str(e)}" if lang == "nl" else f"Embedding error: {str(e)}")

def show_clustering_page():
    lang = st.session_state.language
    st.header("Stap 6: Clustering" if lang == "nl" else "Step 6: Clustering")
    st.markdown(ui.get_text("CLUSTERING_INFO", lang))
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    with col1:
        epsilon = st.slider("Cluster Epsilon", 0.1, 1.0, 0.5, 0.1, 
                           help="Lower values create more clusters")
    with col2:
        alpha = st.slider("Alpha Parameter", 0.5, 2.0, 1.0, 0.1,
                         help="Balance between cluster size and distance")
    
    # Check if we're waiting for debug continue
    if st.session_state.get('waiting_for_debug_continue_clustering'):
        # Display the stored debug information
        debug_capture = st.session_state.get('debug_capture_clustering')
        if debug_capture:
            display_all_debug_info(debug_capture)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="clustering_continue"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_clustering']
                if 'debug_capture_clustering' in st.session_state:
                    del st.session_state['debug_capture_clustering']
                st.session_state.step = 6
                st.rerun()
    elif st.button(ui.get_text("BTN_CLUSTER", lang), type="primary"):
        progress_container = st.empty()
        
        # Create debug capture from session state
        debug_capture = create_debug_capture_from_session()
        
        try:
            # Update hdbscan_config with UI values
            clustering_config = st.session_state.hdbscan_config
            clustering_config.cluster_selection_epsilon = epsilon
            clustering_config.alpha = alpha
            
            initial_cluster_results = _get_pipeline_runner().step_6_cluster(
                embedded_text=st.session_state.pipeline_results['embedded_text'],
                filename=st.session_state.filename,
                hdbscan_config=clustering_config,
                streamlit_container=progress_container,
                debug_capture=debug_capture
            )
            st.session_state.pipeline_results['initial_cluster_results'] = initial_cluster_results
            
            # Check if debug features are enabled and have captured data
            debug_has_data = (debug_capture and 
                            (debug_capture.verbose_outputs or 
                             debug_capture.first_prompts or 
                             debug_capture.sample_results))
            
            if debug_has_data:
                # Store debug capture and set waiting state
                st.session_state['debug_capture_clustering'] = debug_capture
                st.session_state['waiting_for_debug_continue_clustering'] = True
                st.rerun()  # Rerun to show the continue button interface
            else:
                # No debug data - advance automatically
                st.session_state.step = 6
                st.rerun()
        except Exception as e:
            progress_container.error(f"Clustering fout: {str(e)}" if lang == "nl" else f"Clustering error: {str(e)}")

def show_codebook_generation_page():
    lang = st.session_state.language
    st.header("Stap 7: Codebook Generatie" if lang == "nl" else "Step 7: Codebook Generation")
    
    info_text = """
    Deze stap zal:
    - Codes genereren voor elk cluster
    - Een gestructureerd codebook maken
    - Codes optimaliseren en dedupliceren
    """ if lang == "nl" else """
    This step will:
    - Generate codes for each cluster
    - Create a structured codebook
    - Optimize and deduplicate codes
    """
    st.markdown(info_text)
    
    # Codebook generation options
    use_speculative = st.checkbox(
        "Gebruik speculatieve starter codes" if lang == "nl" else "Use speculative starter codes",
        value=False
    )
    
    if st.button("Genereer Codebook" if lang == "nl" else "Generate Codebook", type="primary"):
        progress_container = st.empty()
        try:
            # Determine variable name for codebook generation (use meaningful name for merged variables)
            var_name_for_codebook = st.session_state.selected_variable
            if (st.session_state.get('is_merged_variable', False) and 
                st.session_state.get('selected_variables_confirmed')):
                # Use first variable name or create composite name for merged variables
                selected_vars = st.session_state.get('selected_variables_confirmed', [])
                if len(selected_vars) > 1:
                    var_name_for_codebook = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
            
            codebook_main, reasoning_results = _get_pipeline_runner().step_7_generate_codebook(
                initial_cluster_results=st.session_state.pipeline_results['initial_cluster_results'],
                filename=st.session_state.filename,
                var_name=var_name_for_codebook,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                code_designer_config=st.session_state.code_designer_config,
                use_speculative_starter_codes=use_speculative,
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['codebook_main'] = codebook_main
            st.session_state.pipeline_results['reasoning_results'] = reasoning_results
            st.session_state.step = 7
            st.rerun()
        except Exception as e:
            progress_container.error(f"Codebook fout: {str(e)}" if lang == "nl" else f"Codebook error: {str(e)}")

def show_theme_identification_page():
    lang = st.session_state.language
    st.header("Stap 8: Thema Identificatie" if lang == "nl" else "Step 8: Theme Identification")
    
    info_text = """
    Deze stap zal:
    - Codes groeperen in thema's
    - Hiërarchische thema structuur maken
    - Thema beschrijvingen genereren
    """ if lang == "nl" else """
    This step will:
    - Group codes into themes
    - Create hierarchical theme structure
    - Generate theme descriptions
    """
    st.markdown(info_text)
    
    if st.button("Identificeer Thema's" if lang == "nl" else "Identify Themes", type="primary"):
        progress_container = st.empty()
        try:
            # Determine variable name for theme identification (use meaningful name for merged variables)
            var_name_for_themes = st.session_state.selected_variable
            if (st.session_state.get('is_merged_variable', False) and 
                st.session_state.get('selected_variables_confirmed')):
                # Use first variable name or create composite name for merged variables
                selected_vars = st.session_state.get('selected_variables_confirmed', [])
                if len(selected_vars) > 1:
                    var_name_for_themes = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
            
            theme_enriched_codebook = _get_pipeline_runner().step_8_identify_themes(
                codebook_main=st.session_state.pipeline_results['codebook_main'],
                filename=st.session_state.filename,
                var_name=var_name_for_themes,
                var_lab=st.session_state.pipeline_results['var_lab'],
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['theme_enriched_codebook'] = theme_enriched_codebook
            st.session_state.step = 8
            st.rerun()
        except Exception as e:
            progress_container.error(f"Thema fout: {str(e)}" if lang == "nl" else f"Theme error: {str(e)}")

def show_code_assignment_page():
    lang = st.session_state.language
    st.header("Stap 9: Code Toewijzing" if lang == "nl" else "Step 9: Code Assignment")
    
    info_text = """
    Deze stap zal:
    - Codes toewijzen aan individuele ideeën
    - Thema's koppelen aan toegewezen codes
    - Vertrouwensscores berekenen
    """ if lang == "nl" else """
    This step will:
    - Assign codes to individual ideas
    - Link themes to assigned codes
    - Calculate confidence scores
    """
    st.markdown(info_text)
    
    # Assignment method selection
    method = st.radio(
        "Toewijzing Methode" if lang == "nl" else "Assignment Method",
        options=["direct_llm", "embedding_similarity"],
        format_func=lambda x: "Directe LLM Verwerking" if x == "direct_llm" else "Embedding Similariteit"
        if lang == "nl" else "Direct LLM Processing" if x == "direct_llm" else "Embedding Similarity"
    )
    
    # Check if we're waiting for debug continue
    if st.session_state.get('waiting_for_debug_continue_code_assignment'):
        # Display the stored debug information
        debug_capture = st.session_state.get('debug_capture_code_assignment')
        if debug_capture:
            display_all_debug_info(debug_capture)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="code_assignment_continue"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_code_assignment']
                if 'debug_capture_code_assignment' in st.session_state:
                    del st.session_state['debug_capture_code_assignment']
                st.session_state.step = 9
                st.rerun()
    elif st.button("Wijs Codes Toe" if lang == "nl" else "Assign Codes", type="primary"):
        progress_container = st.empty()
        
        # Create debug capture from session state
        debug_capture = create_debug_capture_from_session()
        
        try:
            code_assigned_results = _get_pipeline_runner().step_9a_assign_codes(
                initial_cluster_results=st.session_state.pipeline_results['initial_cluster_results'],
                theme_enriched_codebook=st.session_state.pipeline_results['theme_enriched_codebook'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                method=method,
                model_config=st.session_state.model_config,
                code_assignment_config=st.session_state.code_assignment_config,
                streamlit_container=progress_container,
                debug_capture=debug_capture
            )
            st.session_state.pipeline_results['code_assigned_results'] = code_assigned_results
            
            # Check if debug features are enabled and have captured data
            debug_has_data = (debug_capture and 
                            (debug_capture.verbose_outputs or 
                             debug_capture.first_prompts or 
                             debug_capture.sample_results))
            
            if debug_has_data:
                # Store debug capture and set waiting state
                st.session_state['debug_capture_code_assignment'] = debug_capture
                st.session_state['waiting_for_debug_continue_code_assignment'] = True
                st.rerun()  # Rerun to show the continue button interface
            else:
                # No debug data - advance automatically
                st.session_state.step = 9
                st.rerun()
        except Exception as e:
            progress_container.error(f"Toewijzing fout: {str(e)}" if lang == "nl" else f"Assignment error: {str(e)}")

def show_export_page():
    lang = st.session_state.language
    st.header("Stap 10: Exporteren" if lang == "nl" else "Step 10: Export Results")
    
    info_text = """
    Exporteer uw resultaten naar Excel met:
    - Alle code toewijzingen
    - Thema informatie
    - Vertrouwensscores
    - Rationales voor toewijzingen
    """ if lang == "nl" else """
    Export your results to Excel with:
    - All code assignments
    - Theme information  
    - Confidence scores
    - Assignment rationales
    """
    st.markdown(info_text)
    
    # Check if we need to load results from cache
    code_assigned_results = None
    theme_enriched_codebook = None
    
    if 'code_assigned_results' in st.session_state.pipeline_results and 'theme_enriched_codebook' in st.session_state.pipeline_results:
        # Use results from current session
        code_assigned_results = st.session_state.pipeline_results['code_assigned_results']
        theme_enriched_codebook = st.session_state.pipeline_results['theme_enriched_codebook']
        st.success("✅ " + ("Resultaten beschikbaar vanuit huidige sessie" if lang == "nl" else "Results available from current session"))
    elif st.session_state.filename and st.session_state.selected_variable:
        # Try to load from cache
        st.info("🔍 " + ("Zoeken naar resultaten in cache..." if lang == "nl" else "Looking for results in cache..."))
        
        try:
            # Load code assignments from cache (step 9a - direct assignment)
            code_assigned_results = st.session_state.cache_manager.load_from_cache(
                st.session_state.filename, "code_assignment_direct", models.CodeAssignedModel
            )
            
            # Load theme enriched codebook from cache  
            theme_enriched_codebooks = st.session_state.cache_manager.load_from_cache(
                st.session_state.filename, "theme_identification", models.ThemeEnrichedCodebookModel
            )
            
            if theme_enriched_codebooks and len(theme_enriched_codebooks) > 0:
                theme_enriched_codebook = theme_enriched_codebooks[0]
            
            if code_assigned_results and theme_enriched_codebook:
                st.success("✅ " + ("Resultaten geladen uit cache!" if lang == "nl" else "Results loaded from cache!"))
                st.info(f"📊 " + (f"Gevonden: {len(code_assigned_results)} responsen met {len(theme_enriched_codebook.codes)} codes" 
                        if lang == "nl" else f"Found: {len(code_assigned_results)} responses with {len(theme_enriched_codebook.codes)} codes"))
            else:
                st.warning("⚠️ " + ("Geen volledige resultaten gevonden in cache" if lang == "nl" else "No complete results found in cache"))
                
        except Exception as e:
            st.error("❌ " + (f"Fout bij laden uit cache: {str(e)}" if lang == "nl" else f"Error loading from cache: {str(e)}"))
    else:
        st.warning("⚠️ " + ("Geen bestand of variabele geselecteerd" if lang == "nl" else "No file or variable selected"))
    
    # Show export options only if we have data
    if code_assigned_results and theme_enriched_codebook:
        # Export options
        export_format = st.selectbox(
            "Export Formaat" if lang == "nl" else "Export Format",
            options=["excel", "csv"],
            format_func=lambda x: "Excel (.xlsx)" if x == "excel" else "CSV (.csv)"
        )
        
        include_rationale = st.checkbox(
            "Rationales opnemen" if lang == "nl" else "Include rationales",
            value=True
        )
        
        # Add option for enhanced export with reasoning data
        include_reasoning = st.checkbox(
            "🧠 Inclusief stap 7 redenering data (beslissingen, rechtvaardigingen, validatie)" 
            if lang == "nl" else "🧠 Include step 7 reasoning data (decisions, justifications, validation)",
            help=("Exporteer extra kolommen met LLM redenering uit stap 7 (code generatie)" 
                  if lang == "nl" else "Export extra columns with LLM reasoning from step 7 (code generation)"),
            value=True  # Default to enhanced export
        )
        
        if st.button("Exporteer Resultaten" if lang == "nl" else "Export Results", type="primary"):
            progress_container = st.empty()
            try:
                if include_reasoning:
                    # Use enhanced export with reasoning data via pipeline runner
                    progress_container.text("🔄 " + ("Resultaten exporteren naar Excel met redenering..." if lang == "nl" else "Exporting results to Excel with reasoning..."))
                    
                    # Determine variable name for export (use meaningful name for merged variables)
                    var_name_for_export = st.session_state.selected_variable
                    if (st.session_state.get('is_merged_variable', False) and 
                        st.session_state.get('selected_variables_confirmed')):
                        # Use first variable name or create composite name for merged variables
                        selected_vars = st.session_state.get('selected_variables_confirmed', [])
                        if len(selected_vars) > 1:
                            var_name_for_export = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
                    
                    excel_path = _get_pipeline_runner().step_10_export_excel_with_reasoning(
                        code_assigned_results=code_assigned_results,
                        theme_enriched_codebook=theme_enriched_codebook,
                        filename=st.session_state.filename,
                        var_name=var_name_for_export,
                        export_dir=None,
                        reasoning_results=st.session_state.pipeline_results.get('reasoning_results'),
                        streamlit_container=progress_container
                    )
                    
                    progress_container.success("✅ " + (f"Code toewijzingen met redenering geëxporteerd naar Excel: {excel_path}" 
                                              if lang == "nl" else f"Code assignments with reasoning exported to Excel: {excel_path}"))
                else:
                    # Use regular export without reasoning data
                    from utils.codeAssignmentExporter import CodeAssignmentExporter
                    
                    progress_container.text("🔄 " + ("Resultaten exporteren naar Excel..." if lang == "nl" else "Exporting results to Excel..."))
                    
                    # Determine variable name for export (use meaningful name for merged variables)
                    var_name_for_export = st.session_state.selected_variable
                    if (st.session_state.get('is_merged_variable', False) and 
                        st.session_state.get('selected_variables_confirmed')):
                        # Use first variable name or create composite name for merged variables
                        selected_vars = st.session_state.get('selected_variables_confirmed', [])
                        if len(selected_vars) > 1:
                            var_name_for_export = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
                    
                    exporter = CodeAssignmentExporter(verbose=True)
                    excel_path = exporter.export_to_excel(
                        code_assigned_results,
                        theme_enriched_codebook,
                        st.session_state.filename,
                        var_name_for_export,
                        export_dir=None  # Will create default export directory
                    )
                    
                    progress_container.success("✅ " + (f"Code toewijzingen geëxporteerd naar Excel: {excel_path}" 
                                              if lang == "nl" else f"Code assignments exported to Excel: {excel_path}"))
                
                # Store in session for download
                st.session_state.pipeline_results['excel_path'] = excel_path
                st.session_state.pipeline_results['code_assigned_results'] = code_assigned_results
                st.session_state.pipeline_results['theme_enriched_codebook'] = theme_enriched_codebook
                st.session_state.step = 10
                st.rerun()
                
            except Exception as e:
                progress_container.error("⚠️ " + (f"Excel export mislukt: {str(e)}" if lang == "nl" else f"Excel export failed: {str(e)}"))
    else:
        st.info("💡 " + ("Voer eerst de volledige pipeline uit of selecteer een bestand met gecachte resultaten" 
                        if lang == "nl" else "Run the complete pipeline first or select a file with cached results"))


def show_results_page():
    lang = st.session_state.language
    st.header("Resultaten" if lang == "nl" else "Results")
    st.markdown(ui.get_text("RESULTS_INFO", lang))
    
    # Show completion celebration
    st.balloons()
    st.success("🎉 Analyse voltooid!" if lang == "nl" else "🎉 Analysis completed!")
    
    # Show results summary if available
    if 'theme_enriched_codebook' in st.session_state.pipeline_results:
        codebook = st.session_state.pipeline_results['theme_enriched_codebook']
        st.subheader("📊 Resultaten Samenvatting" if lang == "nl" else "📊 Results Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Totaal Codes" if lang == "nl" else "Total Codes", len(codebook.codes))
        with col2:
            theme_count = len(set(entry.theme for entry in codebook.codes if entry.theme))
            st.metric("Thema's" if lang == "nl" else "Themes", theme_count)
        with col3:
            if 'code_assigned_results' in st.session_state.pipeline_results:
                total_assignments = sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) 
                                      for resp in st.session_state.pipeline_results['code_assigned_results'] 
                                      if resp.response_ideas)
                st.metric("Toewijzingen" if lang == "nl" else "Assignments", total_assignments)
        
        # Show codebook preview
        with st.expander("📋 Codebook Voorbeeld" if lang == "nl" else "📋 Codebook Preview"):
            codebook_data = []
            for entry in codebook.codes[:10]:  # Show first 10
                codebook_data.append({
                    "Code": entry.code,
                    "Definition": entry.definition[:100] + "..." if len(entry.definition) > 100 else entry.definition,
                    "Theme": entry.theme or "No Theme"
                })
            if codebook_data:
                st.dataframe(pd.DataFrame(codebook_data), use_container_width=True)
    
    # Download options
    st.subheader("Download Opties" if lang == "nl" else "Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'excel_path' in st.session_state.pipeline_results:
            excel_path = st.session_state.pipeline_results['excel_path']
            try:
                with open(excel_path, "rb") as file:
                    excel_data = file.read()
                # Determine variable name for download filename (use meaningful name for merged variables)
                var_name_for_filename = st.session_state.selected_variable
                if (st.session_state.get('is_merged_variable', False) and 
                    st.session_state.get('selected_variables_confirmed')):
                    # Use first variable name or create composite name for merged variables
                    selected_vars = st.session_state.get('selected_variables_confirmed', [])
                    if len(selected_vars) > 1:
                        var_name_for_filename = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
                
                st.download_button(
                    label="📊 Download Excel Resultaten" if lang == "nl" else "📊 Download Excel Results",
                    data=excel_data,
                    file_name=f"{st.session_state.filename}_{var_name_for_filename}_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except FileNotFoundError:
                st.error("Excel bestand niet gevonden" if lang == "nl" else "Excel file not found")
        else:
            st.download_button(
                label="📊 Download Excel Resultaten" if lang == "nl" else "📊 Download Excel Results",
                data="",
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=True
            )
    
    with col2:
        if 'theme_enriched_codebook' in st.session_state.pipeline_results:
            codebook = st.session_state.pipeline_results['theme_enriched_codebook']
            codebook_csv = "Code,Definition,Theme,Theme Description\n"
            for entry in codebook.codes:
                codebook_csv += f'"{entry.code}","{entry.definition}","{entry.theme or ""}","{entry.theme_description or ""}"\n'
            
            # Determine variable name for codebook filename (use meaningful name for merged variables)
            var_name_for_codebook_filename = st.session_state.selected_variable
            if (st.session_state.get('is_merged_variable', False) and 
                st.session_state.get('selected_variables_confirmed')):
                # Use first variable name or create composite name for merged variables
                selected_vars = st.session_state.get('selected_variables_confirmed', [])
                if len(selected_vars) > 1:
                    var_name_for_codebook_filename = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
            
            st.download_button(
                label="📋 Download Codebook" if lang == "nl" else "📋 Download Codebook",
                data=codebook_csv,
                file_name=f"{st.session_state.filename}_{var_name_for_codebook_filename}_codebook.csv",
                mime="text/csv"
            )
        else:
            st.download_button(
                label="📋 Download Codebook" if lang == "nl" else "📋 Download Codebook",
                data="",
                file_name="codebook.csv",
                mime="text/csv",
                disabled=True
            )
    
    if st.button(ui.get_text("BTN_RESTART", lang)):
        # Clear all session state
        for key in list(st.session_state.keys()):
            if key != 'language':  # Keep language setting
                del st.session_state[key]
        st.session_state.step = 0
        st.rerun()

def show_info_panel():
    lang = st.session_state.language
    st.subheader("Informatie" if lang == "nl" else "Information")
    
    # Current file info
    if st.session_state.filename:
        st.markdown(f"**{'Huidig bestand' if lang == 'nl' else 'Current file'}:** {st.session_state.filename}")
    
    # Selected variable info
    if st.session_state.selected_variable:
        st.markdown(f"**{'Geselecteerde variabele' if lang == 'nl' else 'Selected variable'}:** {st.session_state.selected_variable}")
    
    if st.session_state.selected_id_column:
        st.markdown(f"**{'ID kolom' if lang == 'nl' else 'ID column'}:** {st.session_state.selected_id_column}")
    
    st.markdown("---")
    
    # Extended step descriptions for 10 steps
    step_descriptions = [
        "Upload uw SPSS databestand en selecteer variabelen voor analyse." if lang == "nl" else "Upload your SPSS data file and select variables for analysis.",
        "Preprocessing normaliseert en schoont de tekstdata voor analyse." if lang == "nl" else "Preprocessing normalizes and cleans the text data for analysis.",
        "Kwaliteitsfiltering verwijdert lage kwaliteit of betekenisloze antwoorden." if lang == "nl" else "Quality filtering removes low-quality or meaningless responses.",
        "Idee extractie segmenteert responsen in discrete ideeën." if lang == "nl" else "Idea extraction segments responses into discrete ideas.",
        "Embeddings zetten tekst om in numerieke representaties voor clustering." if lang == "nl" else "Embeddings convert text into numerical representations for clustering.",
        "Clustering groepeert vergelijkbare antwoorden in hiërarchische thema's." if lang == "nl" else "Clustering groups similar responses into hierarchical themes.",
        "Codebook generatie maakt gestructureerde codes voor elk cluster." if lang == "nl" else "Codebook generation creates structured codes for each cluster.",
        "Thema identificatie groepeert codes in betekenisvolle thema's." if lang == "nl" else "Theme identification groups codes into meaningful themes.",
        "Code toewijzing koppelt codes aan individuele responsen." if lang == "nl" else "Code assignment links codes to individual responses.",
        "Exporteer resultaten naar Excel met alle toewijzingen en thema's." if lang == "nl" else "Export results to Excel with all assignments and themes.",
        "Bekijk en download uw geanalyseerde resultaten." if lang == "nl" else "Review and download your analyzed results."
    ]
    
    if st.session_state.step < len(step_descriptions):
        st.markdown(step_descriptions[st.session_state.step])

# Helper functions for pipeline integration
def create_pipeline_progress_container():
    """Create a container for showing pipeline progress"""
    return st.empty()

def show_pipeline_progress(container, step_name: str, progress: float, status: str):
    """Update pipeline progress display"""
    with container:
        st.progress(progress)
        st.text(f"{step_name}: {status}")

if __name__ == "__main__":
    main()