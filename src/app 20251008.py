import streamlit as st
import sys
import pandas as pd
from pathlib import Path
import random

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "utils"))

import models
from config import CacheConfig, ModelConfig, SpellCheckConfig,QualityFilterConfig,  SegmentationConfig, EmbeddingConfig, HDBSCANConfig, CodeDesignerConfig, CodeAssignmentConfig

from utils.dataLoader import DataLoader
from utils.cacheManager import CacheManager
import ui_text as ui

# Lazy loaders ################################################################################################################################

def _get_pipeline_runner():
    if st.session_state.pipeline_runner is None:
        from pipelineRunner import get_pipeline_runner
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

# Session state ################################################################################################################################

st.set_page_config(page_title="CoderingsTool - Survey Response Analysis", page_icon="📊", layout="wide",initial_sidebar_state="collapsed")

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
if 'force_recalculate_all' not in st.session_state:
    st.session_state.force_recalculate_all = False  # Default to using cache
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

# Session settings (cofiguration of processing models) ################################################################################################################################

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
        
        # Step 6: Clustering
        st.markdown("#### 📊 Step 6: Clustering")
        st.markdown("*Automatic clustering determines optimal parameters*")
        st.info("Clustering now uses an automatic approach that analyzes the data to find the optimal epsilon value based on k-nearest neighbor distances.")
        
        st.markdown("---")
        
        # Step 7: Code Generation 
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

# App architecture ################################################################################################################################

def main():
    st.title(ui.get_text("APP_TITLE", st.session_state.language))
    st.markdown(ui.get_text("APP_DESCRIPTION", st.session_state.language))
    
    #---------
    # Sidebar
    #---------
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
        #progress = st.progress(st.session_state.step / 10)
        st.markdown(f"**{ui.get_text('CURRENT_STEP', st.session_state.language)}** {st.session_state.step + 1}/10")
        
        st.markdown("---")
        
        # Advanced Settings
        show_advanced_settings()
 
    #---------
    # Main body  
    #---------
    sampling_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    if not st.session_state.step in sampling_steps: 
        show_upload_page()
    else:
        
        #-----
        # Split screen : vertcial
        #-----
        if False: 
            col1, col2 = st.columns([1, 1])
            # LEFT SECTION: Processing step controls
            with col1:
                if st.session_state.step == 1:
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
                # RIGHT SECTION: Processing step controls
                show_info_panel()
            
        #-----
        # Split screen : horizontal
        #-----
        if True: 
            # TOP SECTION: Processing step controls
            if st.session_state.step == 1:
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
                
            # BOTTOM SECTION: Results display
            show_info_panel()


# STEP 0. RETRIEVING / UPLOADING DATA  ################################################################################################################################

def get_available_cached_datasets():
    """Get available cached datasets (001_data_* files) with metadata"""
    cache_manager = _get_cache_manager()
    cache_dir = cache_manager.config.cache_dir
    
    if not cache_dir.exists():
        return []
    
    cache_files = list(cache_dir.glob("001_data_*.pkl")) #001_data = ID + Response/String variables 
    
    datasets = []
    for cache_file in cache_files:
        try:
            filename_parts = cache_file.stem.split('_')
            if len(filename_parts) < 3:
                continue
            prefix_end = 2  # After "001_data"
            sample_suffix = "" # sample size suffix (_250, _full, etc.)
            if filename_parts[-1].isdigit():
                sample_suffix = f"_{filename_parts[-1]}"
                variable_key = "_".join(filename_parts[prefix_end:-1])
            elif filename_parts[-1] == "full":
                sample_suffix = "_full"
                variable_key = "_".join(filename_parts[prefix_end:-1])
            else:
                variable_key = "_".join(filename_parts[prefix_end:])
            
            if variable_key:
                parts = variable_key.split('_')
                var_start_idx = None
                for i, part in enumerate(parts):
                    if (part.startswith('Q') and (len(part) <= 4 or '+' in part)) or '+' in part:
                        var_start_idx = i
                        break
                if var_start_idx is not None:
                    dataset_name = "_".join(parts[:var_start_idx])
                    variables = "_".join(parts[var_start_idx:])
                else:
                    dataset_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
                    variables = parts[-1] if len(parts) > 1 else "unknown"
            else:
                dataset_name = "unknown"
                variables = "unknown"
            
            file_stats = cache_file.stat()
            datasets.append({
                'cache_file': cache_file,
                'dataset_name': dataset_name,
                'variables': variables,
                'sample_suffix': sample_suffix,
                'file_size': file_stats.st_size,
                'created_date': file_stats.st_ctime,
                'display_name': f"{dataset_name} ({variables}){sample_suffix}",
                'cache_key': f"{dataset_name}_{variables}{sample_suffix}"
            })
            
        except Exception:
            # Skip files that can't be parsed
            continue
    
    datasets.sort(key=lambda x: x['created_date'], reverse=True)
    return datasets


def load_cached_dataset(dataset_info):
    """Load a cached dataset and set up session state"""
    try:
        cache_manager = _get_cache_manager()
        
        dataset_name = dataset_info['dataset_name']
        variables = dataset_info['variables']
        sample_suffix = dataset_info['sample_suffix']
        
        # Reconstruct the variable key for cache manager
        if sample_suffix:
            variable_key = f"{variables}{sample_suffix}"
        else:
            variable_key = variables
            
        # Construct filename for cache lookup
        filename = f"{dataset_name}.sav"
        
        # Load from cache
        data = cache_manager.load_from_cache(filename, "data", variable_key, models.ResponseModel)
        
        if data:
            # Set up session state to continue with cached data
            st.session_state.filename = filename
            st.session_state.uploaded_file_path = None  # No physical file
            
            # Parse variables from variable key
            if '+' in variables:
                # Multiple variables
                parsed_vars = variables.split('+')
                st.session_state.selected_variables = parsed_vars
                st.session_state.selected_variable = parsed_vars[0]  # Backward compatibility
                st.session_state.variable_mode = 'multiple'
                st.session_state.is_merged_variable = True
            else:
                # Single variable
                st.session_state.selected_variable = variables
                st.session_state.selected_variables = [variables]
                st.session_state.variable_mode = 'single'
                st.session_state.is_merged_variable = False
            
            # Set sample size if specified
            if sample_suffix and sample_suffix != "_full":
                sample_size = sample_suffix.replace("_", "")
                if sample_size.isdigit():
                    st.session_state.selected_sample_size = int(sample_size)
                    st.session_state.truncate_data = True
                else:
                    st.session_state.selected_sample_size = None
                    st.session_state.truncate_data = False
            else:
                st.session_state.selected_sample_size = None
                st.session_state.truncate_data = False
            
            # Get var_dict
            if '+' in variables:
                var_dict = {var: f"Variable {var}" for var in variables.split('+')}
            else:
                var_dict = {variables: f"Variable {variables}"}
            
            # Add ID column (assume first response has id_column set)
            if data and hasattr(data[0], 'id_column') and data[0].id_column:
                var_dict[data[0].id_column] = f"ID Column ({data[0].id_column})"
                st.session_state.selected_id_column = data[0].id_column
            else:
                var_dict['id'] = 'ID Column (assumed)'
                st.session_state.selected_id_column = 'id'
            
            st.session_state.available_variables = var_dict
            
            # Store the cache key for consistent use throughout the session
            st.session_state.current_cache_key = variable_key
            st.session_state.current_variable_key = variable_key
            
            # Store original cache info for reference
            st.session_state.loaded_from_cache = True
            st.session_state.force_recalculate_all = False  # Use cache for all steps
            st.session_state.cache_dataset_info = {
                'dataset_name': dataset_name,
                'variables': variables,
                'sample_suffix': sample_suffix,
                'variable_key': variable_key,
                'filename': filename
            }
            
            return True, len(data)
        else:
            return False, 0
            
    except Exception as e:
        st.error(f"Error loading cached dataset: {str(e)}")
        return False, 0

def show_upload_page():
    lang = st.session_state.language
    st.header(f"Stap 1: {ui.get_text('BTN_UPLOAD', lang)}" if lang == "nl" else "Step 1: Upload Data")
    
    #----------------------
    # Option 1: from cache
    #----------------------
    st.subheader("📂 " + ("Laad uit Cache" if lang == "nl" else "Load from Cache"))
    cached_datasets = get_available_cached_datasets()
    
    if cached_datasets:
        st.markdown("**" + ("Beschikbare datasets in cache:" if lang == "nl" else "Available datasets in cache:") + "**")
        
        # Create a selectbox with cached datasets
        dataset_options = [""] + [dataset['display_name'] for dataset in cached_datasets]
        selected_dataset_name = st.selectbox(
            "Selecteer dataset" if lang == "nl" else "Select dataset",
            options=dataset_options,
            help="Selecteer een eerder verwerkte dataset om verder te gaan" if lang == "nl" 
                 else "Select a previously processed dataset to continue"
        )
        
        if selected_dataset_name:
            # Find the selected dataset info
            selected_dataset = next((d for d in cached_datasets if d['display_name'] == selected_dataset_name), None)
            
            if selected_dataset:
                # Show dataset information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                	st.write("**Dataset:** " + selected_dataset['dataset_name'])

                with col2:
                	st.write("**Variables:** " + selected_dataset['variables'])

                with col3:
                    file_size_mb = selected_dataset['file_size'] / (1024 * 1024)
                    st.write(f"**Size:** {file_size_mb:.1f} MB")
                
             
                # Load from cache button
                if st.button("📂 " + ("Laad uit Cache" if lang == "nl" else "Load from Cache"), type="primary"):
                    with st.spinner("Data wordt geladen uit cache..." if lang == "nl" else "Loading data from cache..."):
                        success, record_count = load_cached_dataset(selected_dataset)
                        
                        if success:
                            st.success("✅ " + (f"Dataset geladen uit cache! ({record_count} records)" if lang == "nl"
                                                else f"Dataset loaded from cache! ({record_count} records)"))
                            st.session_state.step = 1  
                            st.rerun()
                        else:
                            st.error("❌ " + ("Fout bij laden uit cache" if lang == "nl" else "Error loading from cache"))
        
        st.markdown("---")
    else:
        st.info("ℹ️ " + ("Geen cached datasets beschikbaar" if lang == "nl" else "No cached datasets available"))
        st.markdown("---")
    
    
    #----------------------
    # Option 2: from file
    #----------------------
    st.subheader("📤 " + ("Upload Nieuw Bestand" if lang == "nl" else "Upload New File"))
    
    uploaded_file = st.file_uploader(
        "Kies een SPSS bestand (.sav)" if lang == "nl" else "Choose a SPSS file (.sav)",
        type=['sav'],
        help=ui.get_text("UPLOAD_HELP", lang))
    
    if uploaded_file is not None:
        if st.button(ui.get_text("BTN_UPLOAD", lang), type="primary"):
            with st.spinner("Data wordt geladen..." if lang == "nl" else "Loading data..."):
                try:
                    # 1. Save uploaded file
                    file_path = project_root / "data" / uploaded_file.name
                    file_path.parent.mkdir(exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.filename = uploaded_file.name
                    st.session_state.uploaded_file_path = str(file_path)
                    st.session_state.loaded_from_cache = False
                    st.session_state.force_recalculate_all = True  # Force recalc for new data
                    
                    # 2. Load variables from SPSS file with type information
                    try:
                        variables_with_types = _get_data_loader().list_variables_with_types(uploaded_file.name)
                        # Create simple dict for backward compatibility
                        simple_variables = {var_name: info['label'] for var_name, info in variables_with_types.items()}
                        st.session_state.available_variables = simple_variables
                        st.session_state.available_variables_types = variables_with_types  # Store full type info
                        st.success(f"Bestand geladen met {len(simple_variables)} variabelen!" if lang == "nl" else f"File loaded with {len(simple_variables)} variables!")
                        # Don't advance to step 1 yet - let user select variables first
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fout bij het laden van variabelen: {str(e)}" if lang == "nl" else f"Error loading variables: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Fout bij het uploaden: {str(e)}" if lang == "nl" else f"Upload error: {str(e)}")
    
    # 3. Show variable selection if file is uploaded
    if st.session_state.available_variables:
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
        
        # 3a Variable selection based on mode
        if variable_mode == "single":
            # Single variable selection
            # Filter for string variables only
            if hasattr(st.session_state, 'available_variables_types') and st.session_state.available_variables_types:
                string_vars = [var for var, info in st.session_state.available_variables_types.items() 
                             if info.get('is_string', False)]  # Only include confirmed string variables
                # If no string variables found, show all variables with warning
                if not string_vars:
                    st.warning("Geen tekstvariabelen gevonden. Alle variabelen worden getoond." if lang == "nl" 
                             else "No text variables found. Showing all variables.")
                    string_vars = list(st.session_state.available_variables.keys())
            else:
                string_vars = list(st.session_state.available_variables.keys())
            
            text_var = st.selectbox(
                "📄 " + ("Selecteer tekst variabele" if lang == "nl" else "Select text variable"),
                options=string_vars,
                format_func=lambda x: (
                    f"{x} - {st.session_state.available_variables.get(x, '(No label)')} [{'Tekst' if lang == 'nl' else 'Text'}]" 
                    if hasattr(st.session_state, 'available_variables_types') and 
                       st.session_state.available_variables_types and 
                       st.session_state.available_variables_types.get(x, {}).get('is_string', False)
                    else f"{x} - {st.session_state.available_variables.get(x, '(No label)')} [{st.session_state.available_variables_types.get(x, {}).get('dtype', 'unknown') if hasattr(st.session_state, 'available_variables_types') and st.session_state.available_variables_types else 'unknown'}]"
                ),
                key="text_variable"
            )
            selected_variables = [text_var] if text_var else []
        else:
            # 3b Multiple variable selection
            # Filter for string variables only
            if hasattr(st.session_state, 'available_variables_types') and st.session_state.available_variables_types:
                string_vars = [var for var, info in st.session_state.available_variables_types.items() 
                             if info.get('is_string', False)]  # Only include confirmed string variables
                # If no string variables found, show all variables with warning
                if not string_vars:
                    st.warning("Geen tekstvariabelen gevonden. Alle variabelen worden getoond." if lang == "nl" 
                             else "No text variables found. Showing all variables.")
                    string_vars = list(st.session_state.available_variables.keys())
            else:
                string_vars = list(st.session_state.available_variables.keys())
            
            selected_variables = st.multiselect(
                "📄 " + ("Selecteer tekst variabelen om samen te voegen" if lang == "nl" 
                       else "Select text variables to merge"),
                options=string_vars,
                format_func=lambda x: (
                    f"{x} - {st.session_state.available_variables.get(x, '(No label)')} [{'Tekst' if lang == 'nl' else 'Text'}]" 
                    if hasattr(st.session_state, 'available_variables_types') and 
                       st.session_state.available_variables_types and 
                       st.session_state.available_variables_types.get(x, {}).get('is_string', False)
                    else f"{x} - {st.session_state.available_variables.get(x, '(No label)')} [{st.session_state.available_variables_types.get(x, {}).get('dtype', 'unknown') if hasattr(st.session_state, 'available_variables_types') and st.session_state.available_variables_types else 'unknown'}]"
                ),
                key="text_variables_multi",
                help="Selecteer meerdere variabelen die samengevoegd zullen worden tot één tekst" if lang == "nl"
                     else "Select multiple variables that will be merged into one text"
            )
            
            # 3c Merge configuration for multiple variables
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
            
        text_var = selected_variables[0] if selected_variables else None 
        
        # 4. Data truncation options
        st.subheader("📊 " + ("Steekproef Optie" if lang == "nl" else "Sample Options"))
        
        sample_option = st.radio(
            "Kies steekproef grootte" if lang == "nl" else "Choose sample size",
            ["Gebruik volledige steekproef" if lang == "nl" else "Use full sample",
             "Beperk steekproefgrootte" if lang == "nl" else "Limit sample size"],
            index=0,
            key="sample_option",
            help="Volledige steekproef gebruikt alle gevallen, beperkte steekproef voor snellere verwerking" if lang == "nl"
                 else "Full sample uses all cases, limited sample for faster processing"
        )
        
        sample_size = None
        if sample_option == ("Beperk steekproefgrootte" if lang == "nl" else "Limit sample size"):
            sample_size = st.number_input(
                "Aantal gevallen" if lang == "nl" else "Number of cases",
                min_value=10,
                max_value=10000,
                value=250,
                step=10,
                key="sample_size",
                help="Aantal gevallen om te gebruiken (bijv. 250 voor snelle tests)" if lang == "nl"
                     else "Number of cases to use (e.g., 250 for quick tests)"
            )
            
        # 5. saving specs in sesion state for previewing and preprocessing data
        # NOTE: data is not cached, but selection specs are stored in session state 
        preview_button_label = "Voorbeeld Bekijken" if lang == "nl" else "Preview Variables"
        if variable_mode == "multiple" and len(selected_variables) > 1:
            preview_button_label = f"Voorbeeld van {len(selected_variables)} variabelen" if lang == "nl" else f"Preview {len(selected_variables)} variables"
        if sample_size:
            preview_button_label += f" (eerste {sample_size} gevallen)" if lang == "nl" else f" (first {sample_size} cases)"
        else:
            preview_button_label += " (volledige dataset)" if lang == "nl" else " (full dataset)"
            
        if st.button(preview_button_label):
            if selected_variables and id_var:
                # Check if any selected variables are non-string types
                if hasattr(st.session_state, 'available_variables_types') and st.session_state.available_variables_types:
                    non_string_vars = []
                    for var in selected_variables:
                        var_info = st.session_state.available_variables_types.get(var, {})
                        if not var_info.get('is_string', False):  # Changed default to False
                            dtype = var_info.get('dtype', 'numeric')
                            non_string_vars.append(f"{var} (type: {dtype})")
                    
                    if non_string_vars:
                        st.error(
                            f"⚠️ {'Let op: De volgende variabelen zijn geen tekstvariabelen' if lang == 'nl' else 'Warning: The following variables are not text variables'}:\n\n"
                            f"{chr(10).join('• ' + var for var in non_string_vars)}\n\n"
                            f"{'Deze tool is ontworpen voor het analyseren van open tekstvragen. Numerieke variabelen kunnen niet worden verwerkt.' if lang == 'nl' else 'This tool is designed for analyzing open text responses. Numeric variables cannot be processed.'}\n\n"
                            f"{'Selecteer alleen variabelen met het type [Tekst] of [object].' if lang == 'nl' else 'Please select only variables marked as [Text] or [object].'}"
                        )
                        return
                
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
                            
                            # Apply sample size truncation if specified
                            if sample_size and len(preview_data) > sample_size:
                                preview_data = preview_data.head(sample_size)
                                st.info(f"Getoond: eerste {sample_size} van totaal {len(preview_data)} gevallen" if lang == "nl"
                                        else f"Showing: first {sample_size} of {len(preview_data)} cases")
                            
                            st.session_state.variable_preview = preview_data
                            st.session_state.selected_variable = selected_variables[0]
                            st.session_state.selected_variables = selected_variables
                            st.session_state.selected_sample_size = sample_size
                            # Store confirmed values separately
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
                            
                            # Apply sample size truncation if specified
                            if sample_size and len(preview_data) > sample_size:
                                preview_data = preview_data.head(sample_size)
                                st.info(f"Getoond: eerste {sample_size} van totaal {len(preview_data)} gevallen" if lang == "nl"
                                        else f"Showing: first {sample_size} of {len(preview_data)} cases")
                            
                            st.session_state.variable_preview = preview_data
                            st.session_state.selected_variable = "merged_text"  # For backward compatibility
                            st.session_state.selected_variables = selected_variables
                            st.session_state.selected_sample_size = sample_size
                            # Store merge configuration
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

  
        # 6. Displaying preview data
        if st.session_state.variable_preview is not None:
            st.subheader("📊 Data Preview")
            preview_df = st.session_state.variable_preview

            # Determine the text column name based on mode
            text_column = st.session_state.selected_variable
            if st.session_state.get('variable_mode_confirmed') == 'multiple' and text_column == 'merged_text':
                display_text_column = 'merged_text'
            else:
                display_text_column = text_column

            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Totaal" if lang == "nl" else "Total", len(preview_df))
            with col2:
                non_null = preview_df[display_text_column].notna().sum()
                st.metric("Niet-leeg" if lang == "nl" else "Non-empty", non_null)
            with col3:
                unique_vals = preview_df[display_text_column].nunique()
                st.metric("Uniek" if lang == "nl" else "Unique", unique_vals)
            with col4:
                sample_size_display = st.session_state.get('selected_sample_size')
                if sample_size_display:
                    st.metric("Steekproef" if lang == "nl" else "Sample", sample_size_display)
                else:
                    st.metric("Steekproef" if lang == "nl" else "Sample", "Volledig" if lang == "nl" else "Full")

            # Show merge information for multiple variables
            if st.session_state.get('variable_mode_confirmed') == 'multiple' and len(st.session_state.get('selected_variables', [])) > 1:
                merge_config = st.session_state.get('merge_config', {})
                sample_info = ""
                if st.session_state.get('selected_sample_size'):
                    sample_info = f" | **Steekproef:** {st.session_state.selected_sample_size} gevallen" if lang == "nl" else f" | **Sample:** {st.session_state.selected_sample_size} cases"

                st.info(
                    f"🔗 **Samengevoegd:** {len(st.session_state.selected_variables)} variabelen "
                    f"({', '.join(st.session_state.selected_variables)}) | "
                    f"**Strategie:** {merge_config.get('strategy', 'concatenate')} | "
                    f"**Scheidingsteken:** '{merge_config.get('separator', ' ')}'{sample_info}"
                    if lang == "nl" else
                    f"🔗 **Merged:** {len(st.session_state.selected_variables)} variables "
                    f"({', '.join(st.session_state.selected_variables)}) | "
                    f"**Strategy:** {merge_config.get('strategy', 'concatenate')} | "
                    f"**Separator:** '{merge_config.get('separator', ' ')}'{sample_info}"
                )
            else:
                # Single variable display
                sample_info = ""
                if st.session_state.get('selected_sample_size'):
                    sample_info = f" | **Steekproef:** {st.session_state.selected_sample_size} gevallen" if lang == "nl" else f" | **Sample:** {st.session_state.selected_sample_size} cases"

                st.info(
                    f"📊 **Variabele:** {st.session_state.selected_variable}{sample_info}"
                )

            # Show sample data
            st.subheader("📝 " + ("Voorbeeldgegevens" if lang == "nl" else "Sample Data"))
            sample_data = preview_df[preview_df[display_text_column].notna()].head(10)
            if len(sample_data) > 0:
                # Format data for clean display
                sample_data = sample_data.copy()  # Avoid modifying original

                # Convert all columns to appropriate display format
                for col in sample_data.columns:
                    if col == id_var:
                        # Convert ID column to string, handling floats with .0
                        try:
                            # Check if it's numeric
                            if sample_data[col].dtype in ['int64', 'float64']:
                                # Convert to string and remove .0 for whole numbers
                                sample_data[col] = sample_data[col].apply(
                                    lambda x: str(int(x)) if pd.notna(x) and x == int(x) else str(x)
                                )
                            else:
                                # Already string or other type
                                sample_data[col] = sample_data[col].astype(str)
                        except Exception:
                            # Fallback: just convert to string
                            sample_data[col] = sample_data[col].astype(str)
                    elif sample_data[col].dtype in ['int64', 'float64'] and col != display_text_column:
                        # For other numeric columns, keep as numeric but handle display
                        try:
                            # Don't convert type, let streamlit handle display
                            pass
                        except Exception:
                            pass
                    elif sample_data[col].dtype == 'object' and col != display_text_column:
                        # For string columns, ensure no format string issues
                        try:
                            # Replace format string placeholders to avoid errors
                            sample_data[col] = sample_data[col].apply(
                                lambda x: str(x).replace('%s', '%%s').replace('%d', '%%d').replace('%f', '%%f')
                                if isinstance(x, str) else x
                            )
                        except Exception:
                            pass

                st.dataframe(sample_data, use_container_width=True)
            else:
                st.warning("Geen niet-lege gegevens gevonden" if lang == "nl" else "No non-empty data found")

            # 7. Ready to proceed button
            if st.button("Doorgaan naar Preprocessing" if lang == "nl" else "Continue to Preprocessing", type="primary"):
                # Store session state based on variable mode
                current_mode = st.session_state.get('variable_mode_confirmed', 'single')
                selected_vars = st.session_state.get('selected_variables_confirmed', [])

                if current_mode == 'multiple' and len(selected_vars) > 1:
                    # Ensure merge configuration is properly stored
                    if 'merge_config_confirmed' in st.session_state:
                        st.session_state['merge_config'] = st.session_state['merge_config_confirmed']
                    st.session_state['is_merged_variable'] = True
                else:
                    st.session_state['is_merged_variable'] = False

                st.session_state.step = 1  # Move to preprocessing step
                st.rerun()

# STEP 1. PREPROCESSING RESPONSES ################################################################################################################################

def show_preprocessing_page():
    lang = st.session_state.language
    st.header("Stap 2: Preprocessing" if lang == "nl" else "Step 2: Preprocessing")
    
    #----------------- 
    # state: AFTER preprocessing.  
    #------------------
    if st.session_state.get('waiting_for_continue_preprocessing', False): # Waiting false = AFTER 
    
        # st.success = green box
        st.success("✅ " + ("Preprocessing voltooid! Bekijk de resultaten en klik dan op doorgaan." if lang == "nl" else "Preprocessing completed! Review the results on the right, then click continue."))
        
        # Ensure we have the core selections
        if st.session_state.get('selected_variable') and st.session_state.get('selected_id_column'):
            # Sample / dataset label
            # if st.session_state.get('selected_sample_size'):
            #     sample_info = (
            #         f"\n\n**Steekproef:** {st.session_state.selected_sample_size} gevallen"
            #         if lang == "nl"
            #         else f"\n\n**Sample:** {st.session_state.selected_sample_size} cases"
            #     )
            # else:
            #     sample_info = (
            #         "\n\n**Dataset:** Volledig"
            #         if lang == "nl"
            #         else "\n\n**Dataset:** Full"
            #     )
            
            # st.info (
            #     f"**{'ID Kolom' if lang == 'nl' else 'ID Column'}:** {st.session_state.selected_id_column}\n\n"
            #     f"**{'Variabele(n)' if lang == 'nl' else 'Variable(s)'}:** {st.session_state.selected_variable}\n\n"
            #     f"{sample_info}")
            
            # Stats sections (safe defaults)
            normal_info = ""
            spell_info = ""
            final_info = ""
        
            stats = st.session_state.get('preprocessing_stats', {})
        
            # A) Normalizer stats
            norm_stats = stats.get('normalizer_stats') or {}
            if norm_stats:
                nl = (lang == "nl")
                normal_info = (
                    "\n\n" + ("**Normalisatie:**" if nl else "**Normalization:**")
                    + f"\n- { 'Hoofdletterwijzigingen' if nl else 'Case changes' }: {norm_stats.get('case_changes', 0)} "
                      f"{ 'responsen' if nl else 'responses' }"
                    + f"\n- { 'Witruimte opgeschoond' if nl else 'Whitespace cleanup' }: {norm_stats.get('whitespace_changes', 0)} "
                      f"{ 'responsen' if nl else 'responses' }"
                    + f"\n- { 'Schuine strepen vervangen' if nl else 'Slash replacements' }: {norm_stats.get('slash_changes', 0)} "
                      f"{ 'responsen' if nl else 'responses' }"
                    + f"\n- { 'Lege strings gefilterd' if nl else 'Empty strings filtered' }: {norm_stats.get('invalid_filtered', 0)} "
                      f"{ 'responsen' if nl else 'responses' }"
                )
        
            # B) Spell checker stats
            spell_stats = stats.get('spellchecker_stats') or {}
            if spell_stats:
                nl = (lang == "nl")
                spell_info = (
                    "\n\n" + ("**Spellingcontrole:**" if nl else "**Spell checking:**")
                    + f"\n- { 'Correcties' if nl else 'Corrections' }: {spell_stats.get('corrections_applied', 0)}"
                )
        
            # C) Finalizer stats
            final_stats = stats.get('finalizer_stats') or {}
            if final_stats:
                nl = (lang == "nl")
                final_info = (
                    "\n\n" + ("**Finaliseren:**" if nl else "**Finalization:**")
                    + f"\n- { 'Leestekens toegevoegd' if nl else 'Punctuation additions' }: {final_stats.get('punctuation_additions', 0)} "
                      f"{ 'responsen' if nl else 'responses' }"
                    + f"\n- { 'Opmaak opgeschoond' if nl else 'Format cleanup' }: {final_stats.get('format_cleanup', 0)} "
                      f"{ 'responsen' if nl else 'responses' }"
                    + f"\n- { 'Spatieaanpassingen' if nl else 'Spacing fixes' }: {final_stats.get('spacing_fixes', 0)} "
                      f"{ 'responsen' if nl else 'responses' }"
                )
        
            # Compose the blue info box content (once)
            summary_info = (
                #f"{'Samenvatting' if nl else 'Summary'}"
                f"{normal_info}"
                f"{spell_info}"
                f"{final_info}"
            )
        
            #st.code(summary_info, language= "text")
            st.info(summary_info)
     
    #-----------------
    #state = DEBUGGING
    #-----------------
    elif st.session_state.get('waiting_for_debug_continue_preprocessing'):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="preprocessing_continue_debug"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_preprocessing'] 
                if 'debug_capture_preprocessing' in st.session_state:
                    del st.session_state['debug_capture_preprocessing']
                st.session_state.step = 2
                st.rerun()
        
    #------------------------
    #state = BEFORE processing
    #------------------------
    # A. Information text
    elif st.button(ui.get_text("BTN_PREPROCESS", lang), type="primary"):
        
        if st.session_state.selected_variable and st.session_state.selected_id_column:
            # Check if this is a merged variable scenario
            is_multiple_mode = (st.session_state.get('variable_mode_confirmed') == 'multiple' or st.session_state.get('is_merged_variable', False))
            
            selected_vars = (st.session_state.get('selected_variables') or st.session_state.get('selected_variables_confirmed', []))
            
            sample_info = ""
            if st.session_state.get('selected_sample_size'):
                sample_info = f"\n\n**Steekproef:** {st.session_state.selected_sample_size} gevallen" if lang == "nl" else f"\n\n**Sample:** {st.session_state.selected_sample_size} cases"
            else:
                sample_info = "\n\n**Dataset:** Volledig" if lang == "nl" else "\n\n**Dataset:** Full"
            
            if is_multiple_mode and len(selected_vars) > 1:
                merge_config = (st.session_state.get('merge_config') or 
                               st.session_state.get('merge_config_confirmed', {}))
        
            else:
                #st.info = blue box
                st.info (
                    f"**{'ID Kolom' if lang == 'nl' else 'ID Column'}:** {st.session_state.selected_id_column}\n\n"
                    f"**{'Variabele(n)' if lang == 'nl' else 'Variable(s)'}:** {st.session_state.selected_variables}\n\n"
                    f"**{'Steekproef' if lang == 'nl' else 'Sample'}:** {sample_info}") 
                
            st.markdown(ui.get_text("PREPROCESSING_INFO", lang))

        else:
            # ERROR / DEBUG INFO
            missing_items = []
            if not st.session_state.selected_variable:
                missing_items.append("selected_variable")
            if not st.session_state.selected_id_column:
                missing_items.append("selected_id_column")
            
            error_msg = (f"Ga terug en selecteer een variabele. Ontbrekend: {', '.join(missing_items)}" if lang == "nl" else f"Go back and select a variable. Missing: {', '.join(missing_items)}")
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
        
        progress_container = st.empty()
        
        debug_capture = None  # Disabled debug functionality
        
        try:
            # A. Load data 
            if 'raw_text_list' not in st.session_state.pipeline_results:
                
                # Use selected encoding, None if auto-detect
                encoding = st.session_state.get('file_encoding', 'auto')
                encoding = None if encoding == 'auto' else encoding
                
                # Handle variable label for single vs multiple variables (use confirmed values to avoid widget conflicts)
                is_multiple_mode = (st.session_state.get('variable_mode_confirmed') == 'multiple' or st.session_state.get('is_merged_variable', False))
                selected_vars = (st.session_state.get('selected_variables') or st.session_state.get('selected_variables_confirmed', []))
                
                if is_multiple_mode and len(selected_vars) > 1:
                    # Multiple variables - create combined label
                    merge_config = (st.session_state.get('merge_config') or st.session_state.get('merge_config_confirmed', {}))
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
                        sample_size=st.session_state.get('selected_sample_size'),
                        force_recalc=st.session_state.get('force_recalculate_all', False),
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
                        sample_size=st.session_state.get('selected_sample_size'),
                        force_recalc=st.session_state.get('force_recalculate_all', False),
                        streamlit_container=progress_container,
                        encoding=encoding
                    )
                st.session_state.pipeline_results['raw_text_list'] = raw_text_list
                st.session_state.pipeline_results['var_lab'] = var_lab
            
            # B. Preprocessing
            preprocessed_text = _get_pipeline_runner().step_2_preprocess(
                raw_text_list=st.session_state.pipeline_results['raw_text_list'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                spellcheck_config=st.session_state.spellcheck_config,
                force_recalc=st.session_state.get('force_recalculate_all', False),
                streamlit_container=progress_container,
                debug_capture=debug_capture
            )
            st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text

            # Explicitly store variable_key in session_state for column 2 display after rerun
            pipeline_runner = _get_pipeline_runner()
            if hasattr(pipeline_runner, 'get_variable_key'):
                st.session_state['current_variable_key'] = pipeline_runner.get_variable_key()

            # Collect preprocessing statistics for display
            if hasattr(pipeline_runner, 'preprocessing_stats'):
                st.session_state['preprocessing_stats'] = pipeline_runner.preprocessing_stats

            # App-level cache storage with correct cache key (force_recalculate_all route)
            if st.session_state.get('force_recalculate_all', False):
                cache_manager = _get_cache_manager()
                app_cache_key = st.session_state.get('current_cache_key')
                if app_cache_key:
                    cache_manager.save_to_cache(
                        preprocessed_text, 
                        st.session_state.filename, 
                        "preprocessed", 
                        app_cache_key
                    )
            
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
                # Set waiting state so user can see results before continuing
                st.session_state['completed_step'] = 1  # Mark preprocessing as completed - show preprocessed results
                st.session_state['waiting_for_continue_preprocessing'] = True
                st.rerun()  # Rerun to show the continue button interface
        except Exception as e:
            progress_container.error(f"Preprocessing fout: {str(e)}" if lang == "nl" else f"Preprocessing error: {str(e)}")

def show_filtering_page():
    lang = st.session_state.language
    st.header("Stap 3: Kwaliteitsfiltering" if lang == "nl" else "Step 3: Quality Filtering")
    
    #------------------------
    #state = AFTER processing
    #------------------------
    if st.session_state.get('waiting_for_continue_filtering', False):
        
        st.success("✅ " + ("Kwaliteitsfiltering voltooid! Bekijk de resultaten en klik dan op doorgaan." if lang == "nl" else "Quality filtering completed! Review the results on the right, then click continue."))
        
        # Ensure we have the core selections
        if st.session_state.get('selected_variable') and st.session_state.get('selected_id_column'):
          
            # Sample / dataset label
            if st.session_state.get('selected_sample_size'):
                sample_info = (
                    f"\n\n**Steekproef:** {st.session_state.selected_sample_size} gevallen"
                    if lang == "nl"
                    else f"\n\n**Sample:** {st.session_state.selected_sample_size} cases"
                )
            else:
                sample_info = (
                    "\n\n**Dataset:** Volledig"
                    if lang == "nl"
                    else "\n\n**Dataset:** Full"
                )
            
            st.info (
                f"**{'ID Kolom' if lang == 'nl' else 'ID Column'}:** {st.session_state.selected_id_column} \n\n"
                f"**{'Variabele(n)' if lang == 'nl' else 'Variable(s)'}:** {st.session_state.selected_variables}"
                f"{sample_info}")
        
        if 'quality_filter_stats' in st.session_state:
            stats = st.session_state['quality_filter_stats']
        
            lines = []  # collect all code lines
            code_counts = stats.get('code_counts', {})
            code_meanings = stats.get('code_meanings', {})
        
            # sort safely even if keys are strings/ints
            for code in sorted(code_counts.keys(), key=str):
                count = code_counts.get(code, 0)
                # try both int and str keys for meanings
                meaning = (
                    code_meanings.get(code)
                    or code_meanings.get(str(code))
                    or 'Unknown'
                )
                lines.append(f"- Code {code}: {count} " + ("item(s)" if lang == "en" else "item(s)") + f" - {meaning}")
        
            total = stats.get('total_with_codes', 0) + stats.get('total_without_codes', 0)
            perc_with = (stats.get('total_with_codes', 0) / total * 100) if total else 0
        
            header = "Summary:" if lang == "en" else "Samenvatting:"
            filtered_label = "Filtered" if lang == "en" else "Uitgefilterd"
        
            summary_text = f"{header}\n- {filtered_label}: {stats.get('total_with_codes', 0)} item(s) ({perc_with:.0f}%)\n" + "\n".join(lines)
        
            # no syntax highlighting / colors
            st.code(summary_text, language="text")
                
    #------------------------
    #state = BEFORE processing
    #------------------------
    elif st.button(ui.get_text("BTN_FILTER", lang), type="primary"):
        st.markdown(ui.get_text("FILTERING_INFO", lang))
        progress_container = st.empty()
        try:
            quality_filtered_text = _get_pipeline_runner().step_3_quality_filter(
                preprocessed_text=st.session_state.pipeline_results['preprocessed_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                quality_filter_config=st.session_state.quality_filter_config,
                force_recalc=st.session_state.get('force_recalculate_all', False),
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text

            # Calculate statistics from results for display
            code_counts = {}
            code_meanings = {
                99999997: "User missing: Don't know/only expressing uncertainty",
                99999998: "System missing: NA",
                99999999: "No answer: Empty strings/Single Characters/Only numbers/Nonsensical/gibberish/meaningless content"
            }

            for item in quality_filtered_text:
                if item.quality_filter and item.quality_filter_code:
                    code = item.quality_filter_code
                    code_counts[code] = code_counts.get(code, 0) + 1

            total_with_codes = sum(code_counts.values())
            total_without_codes = len(quality_filtered_text) - total_with_codes

            # Store statistics in session_state for display in waiting state
            st.session_state['quality_filter_stats'] = {
                'code_counts': code_counts,
                'code_meanings': code_meanings,
                'total_with_codes': total_with_codes,
                'total_without_codes': total_without_codes
            }

            # Set waiting state and mark step as completed so left panel shows results
            st.session_state['completed_step'] = 2
            st.session_state['waiting_for_continue_filtering'] = True
            st.rerun()  # Rerun to show the continue button interface
        except Exception as e:
            progress_container.error(f"Filtering fout: {str(e)}" if lang == "nl" else f"Filtering error: {str(e)}")

def show_idea_extraction_page():
    lang = st.session_state.language
    st.header("Stap 4: Idee Extractie" if lang == "nl" else "Step 4: Idea Extraction")
   
    #------------------------
    #state = AFTER processing
    #------------------------
    if st.session_state.get('waiting_for_continue_idea_extraction', False):
        st.success("✅ " + ("Idee extractie voltooid! Bekijk de resultaten en klik dan op doorgaan." if lang == "nl" else "Idea extraction completed! Review the results, then click continue."))
        
        if st.session_state.get('selected_variable') and st.session_state.get('selected_id_column'):
            
            stats = st.session_state['idea_extraction_stats']
            
            if st.session_state.get('selected_sample_size'):
                sample_info = (
                    f"\n\n**Steekproef:** {st.session_state.selected_sample_size} gevallen"
                    if lang == "nl"
                    else f"\n\n**Sample:** {st.session_state.selected_sample_size} cases"
                )
            else:
                sample_info = (
                    "\n\n**Dataset:** Volledig"
                    if lang == "nl"
                    else "\n\n**Dataset:** Full"
                )
            
            st.info (
                f"**{'ID Kolom' if lang == 'nl' else 'ID Column'}:** {st.session_state.selected_id_column} \n\n"
                f"**{'Variabele(n)' if lang == 'nl' else 'Variable(s)'}:** {st.session_state.selected_variables}"
                f"{sample_info}\n\n"
                f"**{'Zonder ruis' if lang == 'nl' else 'Without noise'}**: {stats['total_responses']} {'gevallen' if lang == 'nl' else 'cases'}" 
                )

        # Display idea extraction statistics
        summary_info = ""
        if 'idea_extraction_stats' in st.session_state:
            summary_info =(
            ("Samenvatting:" if lang == 'nl' else "Summary:")
            #+ f"\n- {'Responses verwerkt' if lang == 'nl' else 'Responses processed'} : {stats['total_responses']}" 
            + f"\n- {'Ideeën geëxtraheerd' if lang == 'nl' else 'Ideas extracted'} : {stats['total_ideas']}" 
            + f"\n- {'Unieke ideeën' if lang == 'nl' else 'Total unique ideas'} : {stats['unique_ideas']}" 
            + f"\n- {'Enkelvoudige responsen' if lang == 'nl' else 'Total single responses'} : {stats['single_idea_responses']} ({stats['single_idea_percentage']:.1f}%)" 
            + f"\n- {'Meervoudige responsen' if lang == 'nl' else 'Total multiple responses'} : {stats['multi_idea_responses']} ({stats['multi_idea_percentage']:.1f}%)" 
            )
            
        st.code(summary_info, language="text")

    #------------------------
    #state = BEFORE
    #------------------------
    elif st.session_state.get('waiting_for_debug_continue_idea_extraction'):
        
        st.markdown(ui.get_text("EXTRACTION_INFO", lang))

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="idea_extraction_continue_debug"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_idea_extraction']
                if 'debug_capture_idea_extraction' in st.session_state:
                    del st.session_state['debug_capture_idea_extraction']
                st.session_state.step = 4
                st.rerun()
    
    elif st.button("Start Idee Extractie" if lang == "nl" else "Start Idea Extraction", type="primary"):
        progress_container = st.empty()
        
        # Create debug capture from session state
        # debug_capture = create_debug_capture_from_session()
        debug_capture = None  # Disabled debug functionality
        
        try:
            encoded_text = _get_pipeline_runner().step_4_extract_ideas(
                quality_filtered_text=st.session_state.pipeline_results['quality_filtered_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                model_config=st.session_state.model_config,
                segmentation_config=st.session_state.segmentation_config,
                force_recalc=st.session_state.get('force_recalculate_all', False),
                streamlit_container=progress_container,
                debug_capture=debug_capture
            )
            st.session_state.pipeline_results['encoded_text'] = encoded_text

            # Collect idea extraction statistics for display
            pipeline_runner = _get_pipeline_runner()
            if hasattr(pipeline_runner, 'idea_extraction_stats'):
                st.session_state['idea_extraction_stats'] = pipeline_runner.idea_extraction_stats

            
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
                # Set waiting state so user can see results before continuing
                st.session_state['completed_step'] = 3  # Mark idea extraction as completed
                st.session_state['waiting_for_continue_idea_extraction'] = True
                st.rerun()  # Rerun to show the continue button interface
        except Exception as e:
            progress_container.error(f"Extractie fout: {str(e)}" if lang == "nl" else f"Extraction error: {str(e)}")

def show_embedding_page():
    lang = st.session_state.language
    st.header("Stap 5: Genereer Embeddings" if lang == "nl" else "Step 5: Generate Embeddings")
    st.markdown(ui.get_text("EMBEDDING_INFO", lang))
    
    # Check if we're waiting for user to continue after embedding
    if st.session_state.get('waiting_for_continue_embedding', False):
        st.success("✅ " + ("Embeddings gegenereerd! Bekijk de resultaten links en klik dan op doorgaan." 
                           if lang == "nl" else "Embeddings generated! Review the results on the left, then click continue."))
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="embedding_continue"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_continue_embedding']
                if 'completed_step' in st.session_state:
                    del st.session_state['completed_step']
                st.session_state.step = 5
                st.rerun()
    else:
        # Embedding provider selection
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox(
                "Embedding Provider",
                options=["openai", "gemini"],
                index=0
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
                    force_recalc=st.session_state.get('force_recalculate_all', False),
                    streamlit_container=progress_container
                )
                st.session_state.pipeline_results['embedded_text'] = embedded_text
                
                # Set waiting state and mark step as completed so left panel shows results
                st.session_state['completed_step'] = 4
                st.session_state['waiting_for_continue_embedding'] = True
                st.rerun()  # Rerun to show the continue button interface
            except Exception as e:
                progress_container.error(f"Embedding fout: {str(e)}" if lang == "nl" else f"Embedding error: {str(e)}")

def show_clustering_page():
    lang = st.session_state.language
    st.header("Stap 6: Clustering" if lang == "nl" else "Step 6: Clustering")
    st.markdown(ui.get_text("CLUSTERING_INFO", lang))
    
    # Automatic clustering info
    st.info("🎯 " + ("Automatische clustering bepaalt de optimale parameters op basis van de data" 
             if lang == "nl" else 
             "Automatic clustering determines optimal parameters based on the data"))
    
    # Check if we're waiting for user to continue after clustering
    if st.session_state.get('waiting_for_continue_clustering', False):
        st.success("✅ " + ("Clustering voltooid! Bekijk de resultaten rechts en klik dan op doorgaan." 
                           if lang == "nl" else "Clustering completed! Review the results on the right, then click continue."))
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="clustering_continue_normal"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_continue_clustering']
                if 'completed_step' in st.session_state:
                    del st.session_state['completed_step']
                st.session_state.step = 6
                st.rerun()
    # Check if we're waiting for debug continue
    elif st.session_state.get('waiting_for_debug_continue_clustering'):
        # Display the stored debug information - commented out as requested
        # debug_capture = st.session_state.get('debug_capture_clustering')
        # if debug_capture:
        #     display_all_debug_info(debug_capture)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="clustering_continue_debug"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_clustering']
                if 'debug_capture_clustering' in st.session_state:
                    del st.session_state['debug_capture_clustering']
                st.session_state.step = 6
                st.rerun()
    elif st.button(ui.get_text("BTN_CLUSTER", lang), type="primary"):
        progress_container = st.empty()
        
        # Create debug capture from session state
        # debug_capture = create_debug_capture_from_session()
        debug_capture = None  # Disabled debug functionality
        
        try:
            # Use the default hdbscan_config (automatic clustering)
            clustering_config = st.session_state.hdbscan_config
            
            initial_cluster_results = _get_pipeline_runner().step_6_cluster(
                embedded_text=st.session_state.pipeline_results['embedded_text'],
                filename=st.session_state.filename,
                hdbscan_config=clustering_config,
                force_recalc=st.session_state.get('force_recalculate_all', False),
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
                # Set waiting state so user can see results before continuing
                st.session_state['completed_step'] = 5  # Mark clustering as completed
                st.session_state['waiting_for_continue_clustering'] = True
                st.rerun()  # Rerun to show the continue button interface
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
    
    # Check if we're waiting for user to continue after codebook generation
    if st.session_state.get('waiting_for_continue_codebook_generation', False):
        st.success("✅ " + ("Codebook gegenereerd! Bekijk de resultaten links en klik dan op doorgaan." 
                           if lang == "nl" else "Codebook generated! Review the results on the left, then click continue."))
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="codebook_generation_continue"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_continue_codebook_generation']
                if 'completed_step' in st.session_state:
                    del st.session_state['completed_step']
                st.session_state.step = 7
                st.rerun()
    else:
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
                    force_recalc=st.session_state.get('force_recalculate_all', False),
                    streamlit_container=progress_container
                )
                st.session_state.pipeline_results['codebook_main'] = codebook_main
                st.session_state.pipeline_results['reasoning_results'] = reasoning_results
                
                # Set waiting state and mark step as completed so left panel shows results
                st.session_state['completed_step'] = 6
                st.session_state['waiting_for_continue_codebook_generation'] = True
                st.rerun()  # Rerun to show the continue button interface
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
    
    # Check if we're waiting for user to continue after theme identification
    if st.session_state.get('waiting_for_continue_theme_identification', False):
        st.success("✅ " + ("Thema's geïdentificeerd! Bekijk de resultaten links en klik dan op doorgaan." 
                           if lang == "nl" else "Themes identified! Review the results on the left, then click continue."))
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="theme_identification_continue"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_continue_theme_identification']
                if 'completed_step' in st.session_state:
                    del st.session_state['completed_step']
                st.session_state.step = 8
                st.rerun()
    elif st.button("Identificeer Thema's" if lang == "nl" else "Identify Themes", type="primary"):
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
            
            # Step 8a: Refine codebook (matches pipeline's refine_codebook() call)
            refinement_results = _get_pipeline_runner().step_8_refine_codebook(
                codebook_reasoning=st.session_state.pipeline_results['reasoning_results'],
                filename=st.session_state.filename,
                var_name=var_name_for_themes,
                var_lab=st.session_state.pipeline_results['var_lab'],
                force_recalc=st.session_state.get('force_recalculate_all', False),
                streamlit_container=progress_container
            )
            
            # Step 8b: Create theme enriched codebook (matches pipeline's inline conversion)
            theme_enriched_codebook = _get_pipeline_runner().step_8_identify_themes(
                refinement_results=refinement_results,  # CORRECT: use refinement_results, not codebook_main
                filename=st.session_state.filename,
                var_name=var_name_for_themes,
                var_lab=st.session_state.pipeline_results['var_lab'],
                force_recalc=st.session_state.get('force_recalculate_all', False),
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['theme_enriched_codebook'] = theme_enriched_codebook
            
            # Set waiting state and mark step as completed so left panel shows results
            st.session_state['completed_step'] = 7
            st.session_state['waiting_for_continue_theme_identification'] = True
            st.rerun()  # Rerun to show the continue button interface
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
    
    # Check if we're waiting for user to continue after code assignment
    if st.session_state.get('waiting_for_continue_code_assignment', False):
        st.success("✅ " + ("Code toewijzing voltooid! Bekijk de resultaten rechts en klik dan op doorgaan." 
                           if lang == "nl" else "Code assignment completed! Review the results on the right, then click continue."))
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="code_assignment_continue_normal"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_continue_code_assignment']
                if 'completed_step' in st.session_state:
                    del st.session_state['completed_step']
                st.session_state.step = 9
                st.rerun()
    # Check if we're waiting for debug continue
    elif st.session_state.get('waiting_for_debug_continue_code_assignment'):
        # Display the stored debug information - commented out as requested
        # debug_capture = st.session_state.get('debug_capture_code_assignment')
        # if debug_capture:
        #     display_all_debug_info(debug_capture)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="code_assignment_continue_debug"):
                # Clear the waiting state and advance
                del st.session_state['waiting_for_debug_continue_code_assignment']
                if 'debug_capture_code_assignment' in st.session_state:
                    del st.session_state['debug_capture_code_assignment']
                st.session_state.step = 9
                st.rerun()
    elif st.button("Wijs Codes Toe" if lang == "nl" else "Assign Codes", type="primary"):
        progress_container = st.empty()
        
        # Create debug capture from session state
        # debug_capture = create_debug_capture_from_session()
        debug_capture = None  # Disabled debug functionality
        
        try:
            code_assigned_results = _get_pipeline_runner().step_9a_assign_codes(
                initial_cluster_results=st.session_state.pipeline_results['initial_cluster_results'],
                theme_enriched_codebook=st.session_state.pipeline_results['theme_enriched_codebook'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                method=method,
                model_config=st.session_state.model_config,
                code_assignment_config=st.session_state.code_assignment_config,
                force_recalc=st.session_state.get('force_recalculate_all', False),
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
                # Set waiting state so user can see results before continuing
                st.session_state['completed_step'] = 8  # Mark code assignment as completed
                st.session_state['waiting_for_continue_code_assignment'] = True
                st.rerun()  # Rerun to show the continue button interface
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
                st.info("📊 " + ("Gevonden: {len(code_assigned_results)} responsen met {len(theme_enriched_codebook.codes)} codes" 
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
        # export_format = st.selectbox(
        #     "Export Formaat" if lang == "nl" else "Export Format",
        #     options=["excel", "csv"],
        #     format_func=lambda x: "Excel (.xlsx)" if x == "excel" else "CSV (.csv)"
        # )
        
        # include_rationale = st.checkbox(
        #     "Rationales opnemen" if lang == "nl" else "Include rationales",
        #     value=True
        # )
        
        # Add option for enhanced export with reasoning data
        include_reasoning = st.checkbox(
            "🧠 Inclusief stap 7 redenering data (beslissingen, rechtvaardigingen, validatie)" 
            if lang == "nl" else "🧠 Include step 7 reasoning data (decisions, justifications, validation)",
            help=("Exporteer extra kolommen met LLM redenering uit stap 7 (code generatie)" 
                  if lang == "nl" else "Export extra columns with LLM reasoning from step 7 (code generation)"),
            value=True  # Default to enhanced export
        )
        
        # Check if we're waiting for user to continue after export
        if st.session_state.get('waiting_for_continue_export', False):
            st.success("✅ " + ("Resultaten geëxporteerd! Bekijk de resultaten links en klik dan op doorgaan." 
                               if lang == "nl" else "Results exported! Review the results on the left, then click continue."))
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Continue to Next Step", type="primary", use_container_width=True, key="export_continue"):
                    # Clear the waiting state and advance
                    del st.session_state['waiting_for_continue_export']
                    if 'completed_step' in st.session_state:
                        del st.session_state['completed_step']
                    st.session_state.step = 10
                    st.rerun()
        elif st.button("Exporteer Resultaten" if lang == "nl" else "Export Results", type="primary"):
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
                    # Use regular export without reasoning data (via pipelineRunner for consistency)
                    progress_container.text("🔄 " + ("Resultaten exporteren naar Excel..." if lang == "nl" else "Exporting results to Excel..."))
                    
                    # Determine variable name for export (use meaningful name for merged variables)
                    var_name_for_export = st.session_state.selected_variable
                    if (st.session_state.get('is_merged_variable', False) and 
                        st.session_state.get('selected_variables_confirmed')):
                        # Use first variable name or create composite name for merged variables
                        selected_vars = st.session_state.get('selected_variables_confirmed', [])
                        if len(selected_vars) > 1:
                            var_name_for_export = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
                    
                    excel_path = _get_pipeline_runner().step_10_export_excel(
                        code_assigned_results=code_assigned_results,
                        theme_enriched_codebook=theme_enriched_codebook,
                        filename=st.session_state.filename,
                        var_name=var_name_for_export,
                        export_dir=None,
                        streamlit_container=progress_container
                    )
                    
                    progress_container.success("✅ " + (f"Code toewijzingen geëxporteerd naar Excel: {excel_path}" 
                                              if lang == "nl" else f"Code assignments exported to Excel: {excel_path}"))
                
                # Store in session for download
                st.session_state.pipeline_results['excel_path'] = excel_path
                st.session_state.pipeline_results['code_assigned_results'] = code_assigned_results
                st.session_state.pipeline_results['theme_enriched_codebook'] = theme_enriched_codebook
                
                # Set waiting state and mark step as completed so left panel shows results
                st.session_state['completed_step'] = 9
                st.session_state['waiting_for_continue_export'] = True
                st.rerun()  # Rerun to show the continue button interface
                
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

# Data sampling functions
def load_preview_raw_data(n_samples=5):
    """Load a small preview sample of raw data from uploaded SPSS file"""
    try:
        if not st.session_state.get('filename'):
            return None
        
        # Use pipeline runner to load a small sample
        pipeline_runner = _get_pipeline_runner()
        
        # Check if we're in multiple variable mode
        is_multiple_mode = (st.session_state.get('variable_mode_confirmed') == 'multiple' or
                           st.session_state.get('is_merged_variable', False))
        
        selected_vars = (st.session_state.get('selected_variables') or 
                        st.session_state.get('selected_variables_confirmed', []))
        
        if is_multiple_mode and selected_vars and len(selected_vars) > 1:
            # Multiple variables mode
            preview_data = pipeline_runner.step_1_load_data(
                filename=st.session_state.filename,
                id_column=st.session_state.selected_id_column,
                var_names=selected_vars,
                sample_size=n_samples,
                force_recalc=True,  # Always force recalc for preview
                streamlit_container=None
            )
        elif st.session_state.get('selected_variable'):
            # Single variable mode
            preview_data = pipeline_runner.step_1_load_data(
                filename=st.session_state.filename,
                id_column=st.session_state.selected_id_column,
                var_name=st.session_state.selected_variable,
                sample_size=n_samples,
                force_recalc=True,  # Always force recalc for preview
                streamlit_container=None
            )
        else:
            return None
            
        return preview_data
            
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")
        return None

def show_raw_samples(raw_text_list, n_samples=5):
    """Show random samples from Step 1 - Raw Data"""
    if not raw_text_list:
        st.write("No raw data available")
        return
    
    if st.button("🎲 Draw new random examples", key="raw_samples"):
        st.rerun()
    
    # Original pattern: random.sample(range(len(raw_text_list)), n_samples)
    indices = random.sample(range(len(raw_text_list)), min(n_samples, len(raw_text_list)))
    
    st.write(f"**Random samples from {len(raw_text_list)} filtered responses:**")
    
    if indices:
        sample_text = ""
        for i in indices:
            response_text = raw_text_list[i].response if raw_text_list[i].response is not None else "(empty response)"
            sample_text += f"{response_text}\n"
        
        # Display in gray container
        st.code(sample_text.strip(), language=None)
    

def show_preprocessed_samples(preprocessed_text, n_samples=10):
    """Show random samples from Step 1.5 - Preprocessed Data"""
    
    if not preprocessed_text:
        st.write("{'Geen verwerkte data beschikbaar' if st.session_state.language == 'nl' else 'No preprocessed data available'}")
        return
        
    if st.button(f"{'🎲 Toon nieuwe selectie' if st.session_state.language == 'nl' else '🎲 Draw random examples'}", key="preprocessed_samples"):
        st.rerun()
    
    # Original pattern: random.sample(range(len(preprocessed_text)), n_samples)
    indices = random.sample(range(len(preprocessed_text)), min(n_samples, len(preprocessed_text)))
    
    #st.write(f"**Random samples from {len(preprocessed_text)} preprocessed responses:**")
    
    if indices:
        sample_text = ""
        for i in indices:
            response_text = preprocessed_text[i].response if preprocessed_text[i].response is not None else "(empty response)"
            sample_text += f"{response_text}\n"
        
        # Display in gray container
        st.code(sample_text.strip(), language=None)
    

def show_filtered_samples(quality_filtered_text, n_samples=10):
    """Show random samples from Step 3 - Quality Filtered Data"""
    
    if not quality_filtered_text:
        st.write("{'Geen gefiltered data beschikbaar' if st.session_state.language == 'nl' else 'No filtered data available'}")
        return
    
    if st.button(f"{'🎲 Toon nieuwe selectie' if st.session_state.language == 'nl' else '🎲 Draw random examples'}", key="filtered_samples"):
        st.rerun()
    
    
    #indices = random.sample(range(len(quality_filtered_text)), min(n_samples, len(quality_filtered_text)))
    filtered_text = [item for item in quality_filtered_text if item.quality_filter]
    indices = random.sample(range(len(filtered_text)), min(n_samples, len(filtered_text)))
    
    st.write(f"**Random samples from {len(quality_filtered_text)} filtered responses:**")
    
    sample_text = ""
    for i in indices:
        response_text = filtered_text[i].response if filtered_text[i].response is not None else "(empty response)"
        sample_text += f"{response_text}\n"
    
    # Display in gray container
    st.code(sample_text.strip(), language=None)

def show_idea_samples(encoded_text, n_samples=10):
    """Show random samples from Step 4 - Ideas"""
    
    if not encoded_text:
        st.write("{'geen data beschikbaar' if st.session_state.language == 'nl' else 'No idea data available'}")
        return
    
    if st.button(f"{'🎲 Toon nieuwe selectie' if st.session_state.language == 'nl' else '🎲 Draw new random examples'}", key="idea_samples"):
        st.rerun()
    
    # Original pattern: random.sample(encoded_text, n_samples)
    sampled_items = random.sample(encoded_text, min(n_samples, len(encoded_text)))
    
    #st.write(f"**Random sample from {len(encoded_text)} encoded responses:**")
    
    lines = []
    for item in sampled_items:
        lines.append(f"Response: {item.response.strip()}")
        if hasattr(item, 'response_ideas') and item.response_ideas:
            for segment in item.response_ideas:
                if hasattr(segment, 'idea'):
                    lines.append(f"- {segment.idea}")
        lines.append("")  # Empty line between items
        
    sample_text = "\n".join(lines)
    
    # Display in gray container
    st.code(sample_text.strip())

def show_cluster_samples(initial_cluster_results):
    """Show cluster samples using EXACT pattern from user's original code"""
    if not initial_cluster_results:
        st.write("No cluster data available")
        return
    
    # Original pattern: Get cluster IDs
    cluster_ids = list(set([
        response_idea.initial_cluster 
        for result in initial_cluster_results 
        for response_idea in result.response_ideas  
        if response_idea.initial_cluster is not None
    ]))
    
    if not cluster_ids:
        st.write("No clusters found")
        return
    
    cluster_ids.sort()
    
    # Initialize session state for navigation (simulating the input() pattern)
    if 'cluster_batch' not in st.session_state:
        st.session_state.cluster_batch = 0
    
    # Original pattern: for x in range(1, round(len(cluster_ids) / 1) + 1)
    batch_size = 1  # Show 1 cluster at a time
    total_batches = round(len(cluster_ids) / batch_size)
    current_batch = st.session_state.cluster_batch
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Previous") and current_batch > 0:
            st.session_state.cluster_batch -= 1
            st.rerun()
    
    with col2:
        st.write(f"Batch {current_batch + 1} of {total_batches}")
    
    with col3:
        if st.button("➡️ Next") and current_batch < total_batches - 1:
            st.session_state.cluster_batch += 1
            st.rerun()
    
    # Original pattern logic
    x = current_batch + 1
    y = x * batch_size
    st.write(f"\n=== Showing clusters {y-1} to {min(y, len(cluster_ids)-1)} ===\n")
    
    for z in range(y - 1, y):
        if z < len(cluster_ids):
            cluster_id = cluster_ids[z]  # Use actual cluster ID, not index
            st.write(f"\n**Cluster {cluster_id}**")
            
            cluster_text = ""
            for item in initial_cluster_results:
                for subitem in item.response_ideas:
                    if subitem.initial_cluster == cluster_id:
                        cluster_text += f"{subitem.idea}\n"
            
            # Display in gray container
            if cluster_text.strip():
                st.code(cluster_text.strip(), language=None)

def show_codebook_samples(codebook_reasoning):
    """Show codebook samples using display_cluster_analysis"""
    if not codebook_reasoning:
        st.write("No codebook data available")
        return
    
    from utils.codegenResults import display_cluster_analysis
    import io
    import sys
    
    # Capture the output from display_cluster_analysis
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        display_cluster_analysis(codebook_reasoning)
        output = captured_output.getvalue()
        
        # Display the captured output in gray container
        if output.strip():
            st.code(output, language=None)
    except Exception as e:
        st.error(f"Error displaying codebook analysis: {e}")
    finally:
        sys.stdout = old_stdout

def show_theme_samples(theme_enriched_codebook):
    """Show theme samples using EXACT pattern from user's original code"""
    if not theme_enriched_codebook:
        st.write("No theme data available")
        return
    
    # Original pattern: for theme in themes
    if hasattr(theme_enriched_codebook, 'themes_summary') and theme_enriched_codebook.themes_summary:
        themes = theme_enriched_codebook.themes_summary
        
        theme_text = ""
        for theme in themes:
            # Original pattern: print(f"\n📂 {theme['theme_name'].upper()}")
            if isinstance(theme, dict) and 'theme_name' in theme:
                theme_text += f"\n📂 **{theme['theme_name'].upper()}**\n"
                theme_text += '-' * len(theme['theme_name']) + "\n"
                
                # Original pattern: for code in theme['codes']
                if 'codes' in theme and theme['codes']:
                    for code in theme['codes']:
                        if isinstance(code, dict) and 'code_name' in code:
                            theme_text += f"  • {code['code_name']}\n"
                        elif isinstance(code, str):
                            theme_text += f"  • {code}\n"
                theme_text += "\n"
        
        # Display in gray container
        if theme_text.strip():
            st.code(theme_text.strip(), language=None)
    else:
        st.write("No themes summary available")

def show_assignment_samples(code_assigned_results):
    """Show assignment samples using EXACT pattern from user's original code"""
    if not code_assigned_results:
        st.write("No assignment data available")
        return
    
    # Original pattern: PipelineSummarizer part
    from utils.pipelineSummarizer import PipelineSummarizer
    import io
    import sys
    
    summarizer = PipelineSummarizer(verbose=True)
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        summarizer.generate_summary(
            code_assigned_results=code_assigned_results,
            theme_enriched_codebook=None,
            enriched_codebook=None
        )
        output = captured_output.getvalue()
        
        # Display in gray container
        if output.strip():
            st.code(output, language=None)
    except Exception as e:
        st.error(f"Error displaying assignment summary: {e}")
    finally:
        sys.stdout = old_stdout
    
    st.write("\n---\n")
    
    # Original pattern: random.choice(code_assigned_results)
    sampled_result = random.choice(code_assigned_results)
    
    st.write(f"**Respondent ID:** {sampled_result.respondent_id}")
    st.write(f"**Response:** {sampled_result.response}")
    
    # Original pattern: for idea in sampled_result.response_ideas
    for idea in sampled_result.response_ideas:
        st.write("-" * 40)
        st.write(f"**Idea ID:** {idea.idea_id}")
        st.write(f"**Idea:** {idea.idea}")
        st.write(f"**Assigned Codes:** {', '.join(idea.assigned_codes) if idea.assigned_codes else 'None'}")
        st.write(f"**Rationale:** {idea.assignment_rationale}")
        st.write(f"**Assignment Confidence:** {idea.assignment_confidence}")
        st.write("-" * 40)

def _get_variable_key_for_cache():
    """Get variable key for cache operations - matches existing pattern"""
    return st.session_state.get('current_cache_key')

def show_step8_refined_codebook():
    """Display refined codebook structure - exact pipeline pattern"""
    import random
    
    # Get cache manager and load refinement results
    cache_manager = _get_cache_manager()
    filename = st.session_state.filename
    
    # Get variable key for cache lookup
    variable_key = _get_variable_key_for_cache()
    if not variable_key:
        st.write("❌ Unable to determine variable key for cache lookup")
        return
    
    try:
        # Load refinement_results from step 8a (refine_codebook)
        refinement_results = cache_manager.load_from_cache(filename, "codebook_refinement", variable_key, None)
        
        if refinement_results and len(refinement_results) > 0:
            refinement_result = refinement_results[0]  # Get first result
            
            if hasattr(refinement_result, 'refined_codebook') and refinement_result.refined_codebook:
                final_codebook = refinement_result.refined_codebook
                
                st.write("📋 **Refined Codebook Structure:**")
                st.write("")
                
                # Display structure exactly like pipeline
                for entry in final_codebook.refined_codebook:
                    st.write(f"**{entry.category}**")
                    for x in entry.subcodes:
                        st.write(f"- {x.code}")
                    st.write("")
            else:
                st.write("⚠️ No refined codebook structure available")
        else:
            st.write("❌ No refinement results available in cache")
            
    except Exception as e:
        st.write(f"❌ Error loading refined codebook: {str(e)}")

def show_step9_assignment_stats():
    """Display assignment statistics - fixed summary"""
    from utils.pipelineSummarizer import PipelineSummarizer
    import io
    import sys
    
    # Get cache manager and load results
    cache_manager = _get_cache_manager()
    filename = st.session_state.filename
    variable_key = _get_variable_key_for_cache()
    
    if not variable_key:
        st.write("❌ Unable to determine variable key for cache lookup")
        return
    
    try:
        # Load code assignment results
        code_assigned_results = cache_manager.load_from_cache(filename, "code_assignment_direct", variable_key, None)
        
        # Load theme enriched codebook
        theme_enriched_codebook_results = cache_manager.load_from_cache(filename, "theme_enriched_codebook", variable_key, None)
        theme_enriched_codebook = theme_enriched_codebook_results[0] if theme_enriched_codebook_results else None
        
        if code_assigned_results:
            st.write("📊 **Assignment Statistics:**")
            
            # Use PipelineSummarizer exactly like pipeline
            summarizer = PipelineSummarizer(verbose=True)
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                summarizer.generate_summary(
                    code_assigned_results=code_assigned_results,
                    theme_enriched_codebook=theme_enriched_codebook
                )
                output = captured_output.getvalue()
                
                if output.strip():
                    st.code(output, language=None)
                    
            finally:
                sys.stdout = old_stdout
        else:
            st.write("❌ No code assignment results available")
            
    except Exception as e:
        st.write(f"❌ Error loading assignment statistics: {str(e)}")

def show_step9_random_sample():
    """Display random assignment sample with refresh button"""
    import random
    
    # Get cache manager and load results
    cache_manager = _get_cache_manager()
    filename = st.session_state.filename
    variable_key = _get_variable_key_for_cache()
    
    if not variable_key:
        st.write("❌ Unable to determine variable key for cache lookup")
        return
    
    try:
        # Load code assignment results
        code_assigned_results = cache_manager.load_from_cache(filename, "code_assignment_direct", variable_key, None)
        
        if code_assigned_results and len(code_assigned_results) > 0:
            st.write("🎲 **Random Assignment Sample:**")
            
            # Button to refresh random sample
            if st.button("🔄 Draw New Random Sample", key="random_sample_refresh"):
                # Force a rerun to get new random sample
                st.rerun()
            
            # Draw random sample (pipeline pattern)
            sampled_result = random.choice(code_assigned_results)
            
            st.write(f"**Respondent ID:** {sampled_result.respondent_id}")
            st.write(f"**Response:** {sampled_result.response}")
            st.write("")
            
            # Display ideas and assignments (pipeline pattern)
            for idea in sampled_result.response_ideas:
                st.write("-" * 40)
                st.write(f"**Idea ID:** {idea.idea_id}")
                st.write(f"**Idea:** {idea.idea}")
                st.write(f"**Assigned Codes:** {', '.join(idea.assigned_codes) if idea.assigned_codes else 'None'}")
                st.write(f"**Rationale:** {idea.assignment_rationale}")
                st.write(f"**Assignment Confidence:** {idea.assignment_confidence}")
                st.write("-" * 40)
        else:
            st.write("❌ No code assignment results available")
            
    except Exception as e:
        st.write(f"❌ Error loading random assignment sample: {str(e)}")

def show_step_samples(step_number):
    """Master function to route to appropriate sampling function - loads data from cache"""
        
    # Check if we have the required info to load from cache
    if not st.session_state.get('filename') or not st.session_state.get('selected_variable'):
        st.write("❌ No filename or variable selected - cannot load data")
        return
    
    # Get cache manager
    cache_manager = _get_cache_manager()
    filename = st.session_state.filename
    
    if False: #debug
        #Get variable key for cache lookup (similar to pipeline_runner)
        try:
            # Try to get variable key from session state first (if loaded from cache)
            st.write("🔍 **VARIABLE KEY DEBUG:**")
            variable_key = None
            
            # Check if we have a stored cache key from cache loading
            if st.session_state.get('loaded_from_cache', False):
                stored_key = st.session_state.get('current_cache_key')
                cache_info = st.session_state.get('cache_dataset_info', {})
                
                st.write("✅ **Loaded from cache - using stored key**")
                st.write(f"- Stored cache key: {stored_key}")
                st.write(f"- Dataset: {cache_info.get('dataset_name', 'unknown')}")
                st.write(f"- Variables: {cache_info.get('variables', 'unknown')}")
                st.write(f"- Sample suffix: {cache_info.get('sample_suffix', 'none')}")
                
                if stored_key:
                    variable_key = stored_key
                    st.write(f"- **Using stored key: {variable_key}**")
            
            # If no stored key, try pipeline runner
            if not variable_key:
                st.write("🔄 **Trying pipeline runner for variable key...**")
                try:
                    pipeline_runner = _get_pipeline_runner()
                    st.write(f"- Pipeline runner loaded: {type(pipeline_runner)}")
                    
                    if hasattr(pipeline_runner, 'get_variable_key'):
                        st.write("- get_variable_key method exists, calling it...")
                        variable_key = pipeline_runner.get_variable_key()
                        st.write(f"- Pipeline runner variable key: {variable_key}")
                    else:
                        st.write("- get_variable_key method does not exist")
                        
                except Exception as e:
                    st.write(f"- Error with pipeline runner: {e}")
            
            # Fallback: generate basic variable key
            if not variable_key or variable_key == "unknown":
                st.write("- Using fallback variable key generation...")
                try:
                    selected_variables = [st.session_state.selected_variable]
                    st.write(f"- Selected variables: {selected_variables}")
                    
                    from utils.cacheManager import generate_variable_key
                    variable_key = generate_variable_key(selected_variables, False)
                    st.write(f"- Fallback generated key: {variable_key}")
                    
                except Exception as e:
                    st.write(f"- Error in fallback generation: {e}")
                    
            if not variable_key:
                variable_key = "unknown"
                
            st.write(f"- **Final variable key: {variable_key}**")
            st.write("---")
            
        except Exception as e:
            st.write(f"❌ Error generating variable key: {e}")
            return
        
        st.write("🔍 **CACHE DEBUG INFO:**")
        st.write(f"- Current step: {st.session_state.step}")
        st.write(f"- Requested step_number: {step_number}")
        st.write(f"- Filename: {filename}")
        st.write(f"- Selected variable: {st.session_state.selected_variable}")
        st.write(f"- Variable key: {variable_key}")
        st.write("---")
    
    # Load variabe/cache key
    variable_key = st.session_state.get('current_variable_key')

    if False:  # Additional debug for column 2 display
        st.write("🔍 **Column 2 Debug:**")
        st.write(f"- step_number: {step_number}")
        st.write(f"- variable_key: {variable_key}")
        st.write(f"- filename: {filename}")

    # Load data from cache based on step
    try:
        if step_number == 1:
            # Step 1: Preprocessed data (after preprocessing completion)
            data = cache_manager.load_from_cache(filename, "preprocessed", variable_key, models.PreprocessedModel)
            if False:  # Debug cache lookup result
                st.write(f"- Cache lookup result: {' Found data!' if data else 'No data'}")
                if data:
                    st.write(f"- Number of items: {len(data)}")
            if data:
                show_preprocessed_samples(data)
   
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"{'🔄 Ga naar volgende stap' if st.session_state.language == 'nl' else '🔄 Continue to Next Step'}", type="primary", use_container_width=True, key="preprocessing_continue_normal"):
                        # Clear the waiting state and advance
                        del st.session_state['waiting_for_continue_preprocessing']
                        if 'completed_step' in st.session_state:
                            del st.session_state['completed_step']
                        st.session_state.step = 2
                        st.rerun() 
            else:
                st.write("⏳ No preprocessed data in cache - run preprocessing first")

        elif step_number == 2:
            # Step 2: Quality filtered data
            data = cache_manager.load_from_cache(filename, "quality_filter", variable_key, models.QualityFilteredModel)
            if data:
                #st.write(f"✅ Loaded {len(data)} quality filtered responses from cache")
                show_filtered_samples(data)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"{'🔄 Ga naar volgende stap' if st.session_state.language == 'nl' else '🔄 Continue to Next Step'}", type="primary", use_container_width=True, key="filtering_continue"):    
                        # Clear the waiting state and advance
                        del st.session_state['waiting_for_continue_filtering']
                        if 'completed_step' in st.session_state:
                            del st.session_state['completed_step']
                        st.session_state.step = 3
                        st.rerun()
                
            else:
                st.write("⏳ No quality filtered data in cache - run quality filtering first")
                
        elif step_number == 3:
            # Step 3: Extracted ideas
            data = cache_manager.load_from_cache(filename, "extracted_ideas", variable_key, models.IdeasExtractedModel)
            if data:
                #total_ideas = sum(item.idea_count for item in data)
                #st.write(f"✅ Loaded {len(data)} responses with {total_ideas} ideas from cache")
                show_idea_samples(data)
         
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"{'🔄 Ga naar volgende stap' if st.session_state.language == 'nl' else '🔄 Continue to Next Step'}", type="primary", use_container_width=True, key="idea_extraction_continue_normal"):        
                        # Clear the waiting state and advance
                        del st.session_state['waiting_for_continue_idea_extraction']
                        if 'completed_step' in st.session_state:
                            del st.session_state['completed_step']
                        st.session_state.step = 4
                        st.rerun()
      
            else:
                st.write("⏳ No extracted ideas in cache - run idea extraction first")
                
        elif step_number == 4:
            # Step 4: Embeddings
            data = cache_manager.load_from_cache(filename, "embeddings", variable_key, models.EmbeddingsModel)
            if data:
                total_embeddings = sum(len(resp.response_ideas) for resp in data if resp.response_ideas)
                st.write(f"✅ Embeddings generated for {total_embeddings} items (from cache)")
            else:
                st.write("⏳ No embeddings in cache - run embedding generation first")
                
        elif step_number == 5:
            # Step 5: Clusters
            data = cache_manager.load_from_cache(filename, "initial_clusters", variable_key, models.ClusterModel)
            if data:
                #cluster_ids = set([segment.initial_cluster for result in data for segment in result.response_ideas if segment.initial_cluster is not None])
                #st.write(f"✅ Loaded {len(cluster_ids)} clusters from cache")
                show_cluster_samples(data)
            else:
                st.write("⏳ No clusters in cache - run clustering first")
                
        elif step_number == 6:
            # Step 6: Codebook reasoning
            try:
                from utils.codeGenerator import CodeGeneratorReasoningResults
                data = cache_manager.load_from_cache(filename, "codebook_generation_reasoning", variable_key, CodeGeneratorReasoningResults)
                if data and len(data) > 0:
                    #st.write("✅ Loaded codebook reasoning from cache")
                    show_codebook_samples(data[0])
                else:
                    st.write("⏳ No codebook reasoning in cache - run codebook generation first")
            except Exception as e:
                st.write(f"⚠️ Error loading codebook reasoning: {e}")
                
        elif step_number == 7:
            # Step 7: Themes
            data = cache_manager.load_from_cache(filename, "theme_identification", variable_key, models.ThemeEnrichedCodebookModel)
            if data and len(data) > 0:
                #st.write(f"✅ Loaded {len(data[0].codes)} codes with themes from cache")
                show_theme_samples(data[0])
            else:
                st.write("⏳ No themes in cache - run theme identification first")
                
        elif step_number == 8:
            # Step 8: Refined Codebook Structure
            show_step8_refined_codebook()
                
        elif step_number == 9:
            # Step 9: Code Assignment Results (with both fixed stats and random sample)
            show_step9_assignment_stats()  # Fixed display
            st.markdown("---")
            show_step9_random_sample()     # Interactive display
                
        else:
            st.write(f"❓ No sample display available for step {step_number}")
            
    except Exception as e:
        st.write(f"❌ Error loading data from cache: {e}")
        st.write("This might indicate a cache format issue or missing dependencies.")


def show_info_panel():
    lang = st.session_state.language
    
    # Show samples - use completed_step if we're waiting for user to continue, otherwise use current step
    sampling_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Check if we're in a wait state and should show results from completed step
    in_wait_state = (st.session_state.get('waiting_for_continue_preprocessing', False) or
                     st.session_state.get('waiting_for_continue_filtering', False) or
                     st.session_state.get('waiting_for_continue_idea_extraction', False) or
                     st.session_state.get('waiting_for_continue_embedding', False) or
                     st.session_state.get('waiting_for_continue_clustering', False) or
                     st.session_state.get('waiting_for_continue_codebook_generation', False) or
                     st.session_state.get('waiting_for_continue_theme_identification', False) or
                     st.session_state.get('waiting_for_continue_code_assignment', False) or
                     st.session_state.get('waiting_for_continue_export', False) or
                     
                     st.session_state.get('waiting_for_debug_continue_preprocessing', False) or
                     st.session_state.get('waiting_for_debug_continue_idea_extraction', False) or
                     st.session_state.get('waiting_for_debug_continue_clustering', False) or
                     st.session_state.get('waiting_for_debug_continue_code_assignment', False))
    
    if in_wait_state and st.session_state.get('completed_step'):
        # Show results from the step that just completed
        display_step = st.session_state.completed_step
        #st.header(f"Stap {display_step + 1}: Resultaten" if lang == "nl" else f"Step {display_step + 1}: Results")
        show_step_samples(display_step)
    elif st.session_state.step in sampling_steps:
        # Normal display for current step
        #st.header(f"Stap {st.session_state.step + 1}: Voorbeelden" if lang == "nl" else f"Step {st.session_state.step + 1}: Samples")
        show_step_samples(st.session_state.step)
    
    if st.session_state.step in sampling_steps or (in_wait_state and st.session_state.get('completed_step')):
        st.markdown("---")
    
    if False: # Extended step descriptions for 10 steps
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
        
        # Interactive data sampling section
        st.markdown("---")
    


if __name__ == "__main__":
    main()