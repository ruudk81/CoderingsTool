import streamlit as st
import sys
import pandas as pd
from pathlib import Path
import html, random

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "utils"))

import models
from config import CacheConfig, ModelConfig, SpellCheckConfig,QualityFilterConfig,  SegmentationConfig, EmbeddingConfig, HDBSCANConfig, CodeDesignerConfig, CodeAssignmentConfig

from utils.dataLoader import DataLoader
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
import ui_text as ui

# Import pipeline functions directly
import pipeline
from dataclasses import dataclass
from typing import Optional

# DatasetConfig dataclass ################################################################################################################################

@dataclass
class DatasetConfig:
    """Single source of truth for dataset configuration in Step 0"""
    filename: str
    id_column: str
    selected_variables: list[str]
    variable_mode: str  # 'single' or 'multiple'
    sample_size: Optional[int] = None
    merge_config: Optional[dict] = None  # {strategy, separator, skip_empty}
    encoding: Optional[str] = None
    var_lab: str = ""
    is_merged_variable: bool = False
    loaded_from_cache: bool = False
    force_recalculate_all: bool = False

    def to_session_state(self):
        """Save configuration to storage keys (avoiding widget key conflicts)"""
        # Core configuration - using _config suffix to avoid widget conflicts
        st.session_state.filename = self.filename
        st.session_state.id_column_config = self.id_column
        st.session_state.selected_variables_config = self.selected_variables
        st.session_state.variable_mode_config = self.variable_mode
        st.session_state.sample_size_config = self.sample_size
        st.session_state.merge_config = self.merge_config
        st.session_state.encoding = self.encoding
        st.session_state.var_lab = self.var_lab
        st.session_state.is_merged_variable = self.is_merged_variable
        st.session_state.loaded_from_cache = self.loaded_from_cache

    @classmethod
    def from_session_state(cls) -> Optional['DatasetConfig']:
        """Load configuration from storage keys"""
        if not st.session_state.get('filename'):
            return None

        return cls(
            filename=st.session_state.get('filename', ''),
            id_column=st.session_state.get('id_column_config', ''),
            selected_variables=st.session_state.get('selected_variables_config', []),
            variable_mode=st.session_state.get('variable_mode_config', 'single'),
            sample_size=st.session_state.get('sample_size_config'),
            merge_config=st.session_state.get('merge_config'),
            encoding=st.session_state.get('encoding'),
            var_lab=st.session_state.get('var_lab', ''),
            is_merged_variable=st.session_state.get('is_merged_variable', False),
            loaded_from_cache=st.session_state.get('loaded_from_cache', False)
        )

    def validate_text_variables(self, var_types: dict) -> tuple[bool, list[str]]:
        """ Validate that selected variables are text/string types. """
        non_string_vars = []
        for var in self.selected_variables:
            var_info = var_types.get(var, {})
            if not var_info.get('is_string', False):
                dtype = var_info.get('dtype', 'numeric')
                non_string_vars.append(f"{var} (type: {dtype})")

        return len(non_string_vars) == 0, non_string_vars

    def get_validation_error_message(self, non_string_vars: list[str], lang: str = "en") -> str:
        """  Format error message for non-string variable selection. """
        if lang == "nl":
            msg = "⚠️ Let op: De volgende variabelen zijn geen tekstvariabelen:\n\n"
            msg += "\n".join(f"• {var}" for var in non_string_vars)
            msg += "\n\nDeze tool is ontworpen voor het analyseren van open tekstvragen. "
            msg += "Numerieke variabelen kunnen niet worden verwerkt.\n\n"
            msg += "Selecteer alleen variabelen met het type [Tekst] of [object]."
        else:
            msg = "⚠️ Warning: The following variables are not text variables:\n\n"
            msg += "\n".join(f"• {var}" for var in non_string_vars)
            msg += "\n\nThis tool is designed for analyzing open text responses. "
            msg += "Numeric variables cannot be processed.\n\n"
            msg += "Please select only variables marked as [Text] or [object]."

        return msg

    def get_preview_summary(self, lang: str = "en") -> str:
        """ Generate summary info string for preview display (st.info box)."""
        if self.variable_mode == 'multiple' and len(self.selected_variables) > 1:
            var_list = ', '.join(self.selected_variables)
            strategy = self.merge_config.get('strategy', 'concatenate') if self.merge_config else 'concatenate'
            separator = self.merge_config.get('separator', ' ') if self.merge_config else ' '

            info = (f"🔗 **{'Samengevoegd' if lang == 'nl' else 'Merged'}:** {len(self.selected_variables)} "
                    f"{'variabelen' if lang == 'nl' else 'variables'} ({var_list})")
            info += f" | **{'Strategie' if lang == 'nl' else 'Strategy'}:** {strategy}"
            info += f" | **{'Scheidingsteken' if lang == 'nl' else 'Separator'}:** '{separator}'"
        else:
            var_name = self.selected_variables[0] if self.selected_variables else "Unknown"
            info = f"📊 **{'Variabele' if lang == 'nl' else 'Variable'}:** {var_name}"

        if self.sample_size:
            info += f" | **{'Steekproef' if lang == 'nl' else 'Sample'}:** {self.sample_size} "
            info += "gevallen" if lang == "nl" else "cases"

        return info

    def format_preview_dataframe(self, df, text_col: str):
        """ Clean up dataframe for display: remove .0 from IDs, escape format strings. """
        import pandas as pd

        df = df.copy()  # Avoid modifying original

        if self.id_column in df.columns:
            if df[self.id_column].dtype in ['int64', 'float64']:
                df[self.id_column] = df[self.id_column].apply(
                    lambda x: str(int(x)) if pd.notna(x) and x == int(x) else str(x)
                )
            else:
                df[self.id_column] = df[self.id_column].astype(str)

        for col in df.columns:
            if col != text_col and df[col].dtype == 'object':
                try:
                    df[col] = df[col].apply(
                        lambda x: str(x).replace('%s', '%%s').replace('%d', '%%d').replace('%f', '%%f')
                        if isinstance(x, str) else x
                    )
                except Exception:
                    pass  # Skip if conversion fails

        return df

    @staticmethod
    def filter_string_variables(available_vars: dict, var_types: dict) -> list[str]:
        """ Filter variables to only string/text types. """
        if not var_types:
            return list(available_vars.keys())

        string_vars = [
            var for var, info in var_types.items()
            if info.get('is_string', False)
        ]

        # Fallback: if no string vars found, return all vars
        return string_vars if string_vars else list(available_vars.keys())

    @staticmethod
    def build_variable_format_func(available_vars: dict, var_types: dict, lang: str):
        """ Create format function for variable selectbox that shows type info. """
        def format_var(var_name: str) -> str:
            label = available_vars.get(var_name, '(No label)')

            if var_types and var_name in var_types:
                var_info = var_types[var_name]
                if var_info.get('is_string', False):
                    type_label = "Tekst" if lang == "nl" else "Text"
                else:
                    type_label = var_info.get('dtype', 'unknown')
                return f"{var_name} - {label} [{type_label}]"
            else:
                return f"{var_name} - {label} [unknown]"

        return format_var

# Lazy loaders ################################################################################################################################

def _get_data_loader():
    if st.session_state.data_loader is None:
        st.session_state.data_loader = DataLoader(verbose=False)
    return st.session_state.data_loader

def _get_cache_manager():
    if st.session_state.cache_manager is None:
        st.session_state.cache_manager = CacheManager(CacheConfig())
    return st.session_state.cache_manager

def _load_or_recover(filename: str, step_name: str, variable_key: str, model_cls):
    """
    Safely load from cache with automatic recovery from file handle errors.

    This wrapper handles the "I/O operation on closed file" error that can occur
    when loading cached pickle files on Windows. If this error occurs, it:
    1. Warns the user that the cache file appears corrupted
    2. Invalidates the cache entry
    3. Returns None to trigger reprocessing

    Args:
        filename: SPSS filename
        step_name: Pipeline step name (e.g., "preprocessed", "embeddings")
        variable_key: Cache key for the variable
        model_cls: Pydantic model class for reconstruction

    Returns:
        List of models or None if cache is invalid/corrupted
    """
    cm = _get_cache_manager()
    lang = st.session_state.get('language', 'en')

    try:
        return cm.load_from_cache(filename, step_name, variable_key, model_cls)
    except Exception as e:
        msg = str(e)
        if "I/O operation on closed file" in msg or "closed file" in msg.lower():
            # User-friendly warning in current language
            warning_msg = (
                "⚠️ Cache-bestand lijkt beschadigd. Ik herstel dit en probeer opnieuw."
                if lang == "nl" else
                "⚠️ Cache file appears corrupted. Recovering and retrying."
            )
            st.warning(warning_msg)

            # Invalidate this cache entry to trigger reprocessing
            try:
                cm.db.invalidate_cache(filename, step_name, variable_key)
            except Exception:
                pass  # Silently fail if invalidation fails

            # Return None to trigger reprocessing from previous step
            return None
        else:
            # Re-raise other exceptions
            raise

# Session state ################################################################################################################################

st.set_page_config(page_title="CoderingsTool - Survey Response Analysis", page_icon="📊", layout="wide",initial_sidebar_state="collapsed")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'completed_steps' not in st.session_state:
    st.session_state.completed_steps = set() 
if 'max_step_reached' not in st.session_state:
    st.session_state.max_step_reached = 0  
if 'force_recalculate_all' not in st.session_state:
    st.session_state.force_recalculate_all = False   
  
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
if 'sample_size' not in st.session_state:
    st.session_state.sample_size = None
if 'sample_size_config' not in st.session_state:
    st.session_state.sample_size_config = None

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

# helpers
def get_step_name(step_num: int, lang: str = "en") -> str:
    """Get localized step name"""
    step_names = ui.get_text("STEP_NAMES", lang)
    return step_names.get(step_num, f"Step {step_num}")

def is_step_completed(step_num: int) -> bool:
    """Check if a step has been completed in the current session"""
    return step_num in st.session_state.completed_steps

def mark_step_completed(step_num: int):
    """Mark a step as completed and update max step reached"""
    st.session_state.completed_steps.add(step_num)
    if step_num > st.session_state.max_step_reached:
        st.session_state.max_step_reached = step_num

def can_navigate_to_step(target_step: int) -> bool:
    """Check if user can navigate to a specific step"""
    # Can navigate to any completed step or the next sequential step
    return target_step <= st.session_state.max_step_reached or target_step == st.session_state.step + 1

def reset_navigation_tracking():
    """Reset navigation tracking (e.g., when uploading new file)"""
    st.session_state.completed_steps = set()
    st.session_state.max_step_reached = 0
    st.session_state.pipeline_results = {}

def clear_all_wait_states():
    """Clear all debug waiting states when navigating between steps"""
    # Note: We no longer use waiting_for_continue_* states
    # Only clearing debug states which may still be needed
    wait_states = [
        'waiting_for_debug_continue_preprocessing',
        'waiting_for_debug_continue_idea_extraction',
        'waiting_for_debug_continue_clustering',
        'waiting_for_debug_continue_code_assignment'
    ]
    for state in wait_states:
        if state in st.session_state:
            del st.session_state[state]

def invalidate_from_step(start_step: int):
    """Invalidate cache and completion tracking from start_step onwards"""

    # 1. Clear completion tracking
    steps_to_remove = [s for s in st.session_state.completed_steps if s >= start_step]
    for step in steps_to_remove:
        st.session_state.completed_steps.remove(step)

    # 2. Update max_step_reached
    if start_step <= st.session_state.max_step_reached:
        st.session_state.max_step_reached = start_step - 1

    # 3. Set force_recalculate flag for pipeline functions
    st.session_state.force_recalculate_from_step = start_step

    # 4. Clear pipeline_results for affected steps to force reprocessing
    step_result_keys = {
        0: ['cached_data', 'raw_text_list'],
        1: ['preprocessed_text'],
        2: ['quality_filtered_text'],
        3: ['extracted_ideas', 'encoded_text'],
        4: ['embedded_text'],
        5: ['initial_cluster_results'],
        6: ['codebook_main', 'reasoning_results'],
        7: ['theme_enriched_codebook', 'refinement_results'],
        8: ['code_assigned_results'],
        9: ['excel_path']
    }

    all_cleared_keys = []
    for step_num in range(start_step, 10):
        keys_to_remove = step_result_keys.get(step_num, [])
        for key in keys_to_remove:
            if key in st.session_state.pipeline_results:
                del st.session_state.pipeline_results[key]
                all_cleared_keys.append(f"{key}(step{step_num})")

    # 5. Invalidate cache entries in database
    cache_manager = _get_cache_manager()
    step_mapping = {
        0: "data", 1: "preprocessed", 2: "quality_filter",
        3: "extracted_ideas", 4: "embeddings", 5: "initial_clusters",
        6: "codebook_generation", 7: "theme_identification",
        8: "code_assignment", 9: "export"
    }

    # Get current dataset info
    filename = st.session_state.get('filename')
    selected_vars = st.session_state.get('selected_variables_config', [])
    is_merged = st.session_state.get('is_merged_variable', False)
    sample_size = st.session_state.get('sample_size_config')
    merge_config = st.session_state.get('merge_config')

    if filename and selected_vars:
        variable_key = generate_enhanced_variable_key(
            selected_vars, is_merged, sample_size, merge_config
        )

        # Invalidate cache entries for affected steps
        for step_num in range(start_step, 10):
            step_name = step_mapping.get(step_num)
            if step_name:
                cache_manager.db.invalidate_cache(filename, step_name, variable_key)

# Side bar ################################################################################################################################

def show_advanced_settings(current_step=0):
    """Show advanced settings UI in sidebar - only shows settings relevant to current step """
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

        # Step 1: Preprocessing (
        if current_step == 1:
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

        # Step 2: Quality Filter 
        if current_step == 2:
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

        # Step 3: Idea Extraction 
        if current_step == 3:
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

        # Step 4: Embeddings 
        if current_step == 4:
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

        # Step 5: Clustering 
        if current_step == 5:
            st.markdown("#### 📊 Step 6: Clustering")
            st.markdown("*Automatic clustering determines optimal parameters*")
            st.info("Clustering now uses an automatic approach that analyzes the data to find the optimal epsilon value based on k-nearest neighbor distances.")

            st.markdown("---")

        # Step 6 and 7: Codebook Generation 
        if current_step in [6, 7]:
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

        # Step 8: Code Assignment 
        if current_step == 8:
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

            st.markdown("---")

        # Reset to defaults button (always shown)
        if st.button("🔄 Reset All to Defaults", type="secondary"):
            st.session_state.model_config = ModelConfig()
            st.session_state.spellcheck_config = SpellCheckConfig()
            st.session_state.quality_filter_config = QualityFilterConfig()
            st.session_state.segmentation_config = SegmentationConfig()
            st.session_state.embedding_config = EmbeddingConfig()
            st.session_state.hdbscan_config = HDBSCANConfig()
            st.session_state.code_designer_config = CodeDesignerConfig()
            st.session_state.code_assignment_config = CodeAssignmentConfig()
            st.rerun()

# App architecture ################################################################################################################################

def main():
    st.title(ui.get_text("APP_TITLE", st.session_state.language))
    st.markdown(ui.get_text("APP_DESCRIPTION", st.session_state.language))

    with st.sidebar:
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

        # Progress indicator with step name
        if st.session_state.step == 0:
            st.markdown(f"**{get_step_name(0, st.session_state.language)}**")
        else:
            step_name = get_step_name(st.session_state.step, st.session_state.language)
            st.markdown(f"**{ui.get_text('CURRENT_STEP', st.session_state.language)}** {st.session_state.step}/9 - {step_name}")

        # Navigation buttons
        if st.session_state.step > 0:  # Only show navigation when not on upload page
            if st.button("🏠 " + ("Start" if st.session_state.language == "nl" else "Home"), use_container_width=True, key="nav_home"):

                st.session_state.step = 0
                st.rerun()

            nav_col1, nav_col2 = st.columns(2)

            with nav_col1:
                loaded_from_cache = st.session_state.get('loaded_from_cache', False)
                can_go_back = st.session_state.step > 1 and (loaded_from_cache or(st.session_state.step - 1) in st.session_state.completed_steps)
                if st.button("⬅️ " + ("Vorige" if st.session_state.language == "nl" else "Previous"),disabled=not can_go_back, use_container_width=True, key="nav_previous"):
                    clear_all_wait_states()  # Clear wait states before navigation
                    st.session_state.step -= 1
                    st.rerun()

            with nav_col2:
                can_go_forward = st.session_state.step < 10 and (loaded_from_cache or st.session_state.step in st.session_state.completed_steps)
                if st.button(("Volgende" if st.session_state.language == "nl" else "Next") + " ➡️", disabled=not can_go_forward, use_container_width=True, key="nav_next"):
                    clear_all_wait_states()  # Clear wait states before navigation
                    st.session_state.step += 1
                    st.rerun()

            # Step jump navigation
            st.markdown("---")
            lang = st.session_state.language
            loaded_from_cache = st.session_state.get('loaded_from_cache', False)

            # Build list of available steps
            available_steps = [0]  # Upload always available
            for step in range(1, 10):
                if loaded_from_cache or step in st.session_state.completed_steps or step <= st.session_state.max_step_reached:
                    available_steps.append(step)

            # Only show dropdown if there are steps to jump to (more than just current)
            if len(available_steps) > 1:
                current_index = available_steps.index(st.session_state.step) if st.session_state.step in available_steps else 0

                selected_step = st.selectbox(
                    "⏩ " + ("Spring naar stap:" if lang == "nl" else "Jump to step:"),
                    options=available_steps,
                    index=current_index,
                    format_func=lambda x: f"{get_step_name(x, lang)}" if x == 0 else f"Stap {x}: {get_step_name(x, lang)}" if lang == "nl" else f"Step {x}: {get_step_name(x, lang)}",
                    key="step_jump_selector"
                )

                if selected_step != st.session_state.step:
                    clear_all_wait_states()
                    st.session_state.step = selected_step
                    st.rerun()

            # Visual step progress indicator
            st.markdown("---")
            with st.expander("📋 " + ("Alle stappen" if lang == "nl" else "All steps"), expanded=False):
                for step in range(10):
                    step_name = get_step_name(step, lang)

                    if step == 0:
                        # Upload step
                        if st.session_state.step == 0:
                            st.markdown(f"🔵 **{step_name}**")
                        else:
                            st.markdown(f"✅ {step_name}")
                    else:
                        # Processing steps 1-9
                        if step == st.session_state.step:
                            st.markdown(f"🔵 **Stap {step}: {step_name}**" if lang == "nl" else f"🔵 **Step {step}: {step_name}**")
                        elif step in st.session_state.completed_steps or (loaded_from_cache and step <= st.session_state.max_step_reached):
                            st.markdown(f"✅ Stap {step}: {step_name}" if lang == "nl" else f"✅ Step {step}: {step_name}")
                        else:
                            st.markdown(f"⚪ Stap {step}: {step_name}" if lang == "nl" else f"⚪ Step {step}: {step_name}")

        st.markdown("---")

        # Advanced Settings
        show_advanced_settings(st.session_state.step)

        # Cache Management
        if st.session_state.step > 0 and is_step_completed(st.session_state.step):
            with st.expander("🔧 Cache Management", expanded=False):
                st.markdown("### Reprocess Pipeline")
                st.warning("⚠️ **Warning**: Reprocessing will invalidate all downstream cached steps and require recalculation.")

                current_step = st.session_state.step

                if st.button(
                    "🔄 " + ("Herverwerk vanaf stap" if st.session_state.language == "nl" else "Reprocess from step") + f" {current_step} " + ("en verder" if st.session_state.language == "nl" else "onwards"),
                    type="secondary",
                    use_container_width=True,
                    key="reprocess_from_step"
                ):
                    invalidate_from_step(current_step)
                    st.success("✅ " + ("Cache gewist vanaf stap" if st.session_state.language == "nl" else "Cache cleared from step") + f" {current_step}")
                    st.rerun()

    # Main body  
    sampling_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    if not st.session_state.step in sampling_steps: 
        show_upload_page()
    else:
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

# STEP0 HELPER  ################################################################################################################################

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

def determine_max_step_from_cache(filename: str, variable_key: str, cache_manager) -> int:
    """Determine the maximum completed step from cached data"""
    
    # Step name to step number mapping
    step_mapping = {
        "data": 0,
        "preprocessed": 1,
        "quality_filter": 2,
        "extracted_ideas": 3,
        "embeddings": 4,
        "initial_clusters": 5,
        "codebook_generation": 6,
        "codebook_refinement": 7,
        "code_assignment_direct": 8,
        "export": 9
    }

    # Get all cached steps from database
    cached_steps = cache_manager.db.get_all_cached_steps(filename, variable_key)
  
    # Map step names to numbers
    step_numbers = [step_mapping.get(step, -1) for step in cached_steps if step in step_mapping]
  
    # Return max step number (or 0 if none found)
    max_step = max(step_numbers) if step_numbers else 0
    return max_step

def load_from_cache(dataset_info: dict) -> tuple[DatasetConfig, list, int, int]:
    """ Load cached dataset and build DatasetConfig """
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
     
        if not data:
            return None, None, 0, 0

        # Parse variables from variable key
        if '+' in variables:
            # Multiple variables
            parsed_vars = variables.split('+')
            variable_mode = 'multiple'
            is_merged = True
        else:
            # Single variable
            parsed_vars = [variables]
            variable_mode = 'single'
            is_merged = False

        # Parse sample size from suffix
        sample_size = None
        if sample_suffix and sample_suffix != "_full":
            size_str = sample_suffix.replace("_", "")
            if size_str.isdigit():
                sample_size = int(size_str)
                
        if sample_size is None and data:
            sample_size = len(data)  

        id_column = 'id'
        if data and hasattr(data[0], 'id_column') and data[0].id_column:
            id_column = data[0].id_column

        # Get var_lab
        var_lab = variables
        try:
            data_loader = _get_data_loader()
            first_var = variables.split('+')[0] if '+' in variables else variables
            var_lab = data_loader.get_varlab(filename, first_var)
            
        except Exception:
            pass

        last_bracket = var_lab.rfind("]")
        
        # Build config
        config = DatasetConfig(
            filename=filename,
            id_column=id_column,
            selected_variables=parsed_vars,
            variable_mode=variable_mode,
            sample_size=sample_size,
            merge_config=None,  # Merge config not stored in cache metadata
            encoding=None,
            var_lab=var_lab[last_bracket + 1:].strip(),
            is_merged_variable=is_merged,
            loaded_from_cache=True,
            force_recalculate_all = False
        )

        # Determine max step reached from cache
        max_step = determine_max_step_from_cache(filename, variable_key, cache_manager)
        return config, data, len(data), max_step

    except Exception as e:
        st.error(f"Error loading cached dataset: {str(e)}")
        return None, None, 0, 0


def load_from_file(uploaded_file) -> tuple[str, dict, dict]:
    """ Save uploaded file and load variable metadata """
    try:
        # Save uploaded file
        file_path = project_root / "data" / uploaded_file.name
        file_path.parent.mkdir(exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load variables with type information
        variables_with_types = _get_data_loader().list_variables_with_types(uploaded_file.name)

        # Create simple dict for backward compatibility
        simple_variables = {var_name: info['label'] for var_name, info in variables_with_types.items()}

        return uploaded_file.name, simple_variables, variables_with_types

    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")
        

def build_config_from_ui(filename: str, id_var: str, selected_vars: list[str],
                          encoding: Optional[str] = None) -> DatasetConfig:
    """ Build DatasetConfig from UI selections. Reads widget-controlled settings directly from session_state widget keys. """
    # Read widget-controlled settings from widget session_state keys
    variable_mode = st.session_state.get('variable_mode', 'single')
    sample_size = st.session_state.get('sample_size')  # From number_input widget

    # Determine if merged
    is_merged = variable_mode == 'multiple' and len(selected_vars) > 1

    # Build merge config if needed (read from widget keys)
    merge_config = None
    if is_merged:
        merge_config = {
            'strategy': st.session_state.get('merge_strategy', 'concatenate'),
            'separator': st.session_state.get('merge_separator', ' '),
            'skip_empty': st.session_state.get('skip_empty', True)
        }

    # Get var_lab from SPSS file
    var_lab = selected_vars[0] if selected_vars else ""
    try:
        data_loader = _get_data_loader()
        first_var = selected_vars[0]
        var_lab = data_loader.get_varlab(filename, first_var, encoding)
    except Exception:
        pass

    # Build config
    config = DatasetConfig(
        filename=filename,
        id_column=id_var,
        selected_variables=selected_vars,
        variable_mode=variable_mode,
        sample_size=sample_size,
        merge_config=merge_config,
        encoding=encoding or st.session_state.get('file_encoding'),
        var_lab=var_lab,
        is_merged_variable=is_merged,
        loaded_from_cache=False)

    return config


# def preview_dataset(config: DatasetConfig) -> pd.DataFrame:
#     """ Load and preview dataset based on configuration """
#     data_loader = _get_data_loader()

#     # Determine encoding
#     encoding = config.encoding
#     if encoding == 'auto':
#         encoding = None

#     # Load data based on mode
#     if config.variable_mode == "single" or len(config.selected_variables) == 1:
#         # Single variable
#         preview_data = data_loader.get_variable_with_IDs(
#             config.filename,
#             config.id_column,
#             config.selected_variables[0],
#             encoding=encoding
#         )
#     else:
#         # Multiple variables - merge
#         merge_config = config.merge_config or {}
#         preview_data = data_loader.get_multiple_variables_with_IDs(
#             filename=config.filename,
#             id_column=config.id_column,
#             var_names=config.selected_variables,
#             merge_strategy=merge_config.get('strategy', 'concatenate'),
#             separator=merge_config.get('separator', ' '),
#             skip_empty=merge_config.get('skip_empty', True),
#             encoding=encoding
#         )

#     # Apply sampling if specified
#     if config.sample_size and len(preview_data) > config.sample_size:
#         preview_data = preview_data.head(config.sample_size)

#     return preview_data


def convert_response_models_to_preview_df(response_models: list, id_column: str, text_column: str) -> pd.DataFrame:
    """Convert list of ResponseModel objects to a DataFrame suitable for preview display

    Args:
        response_models: List of ResponseModel objects from step_0_load_data
        id_column: Name for the ID column in the DataFrame
        text_column: Name for the text/response column in the DataFrame

    Returns:
        pd.DataFrame with columns [id_column, text_column]
    """
    import pandas as pd

    # Extract data from ResponseModel objects
    ids = [model.respondent_id for model in response_models]
    responses = [model.response for model in response_models]

    # Create DataFrame with the specified column names
    preview_df = pd.DataFrame({
        id_column: ids,
        text_column: responses
    })

    return preview_df

# STEP0 UPLOAD PAGE  ################################################################################################################################

def show_upload_page():
    lang = st.session_state.language
    st.header(f"{ui.get_text('BTN_UPLOAD', lang)}" if lang == "nl" else "Upload Data")
    
    # ==================== OPTION 1: RETRIEVE FROM CACHE ====================
    st.subheader("📂 " + ("Laad uit Cache" if lang == "nl" else "Load from Cache"))
    cached_datasets = get_available_cached_datasets()
    if cached_datasets:
        st.markdown("**" + ("Beschikbare datasets in cache:" if lang == "nl" else "Available datasets in cache:") + "**")
        dataset_options = [None] + [dataset['display_name'] for dataset in cached_datasets]
        selected_dataset_name = st.selectbox(
            "Selecteer dataset" if lang == "nl" else "Select dataset",
            options=dataset_options,
            format_func=lambda x: ("-- Kies een dataset --" if lang == "nl" else "-- Choose a dataset --") if x is None else x,
            help="Selecteer een eerder verwerkte dataset om verder te gaan" if lang == "nl" else "Select a previously processed dataset to continue"
        )
        if selected_dataset_name:
            selected_dataset = next((d for d in cached_datasets if d['display_name'] == selected_dataset_name), None)
            if selected_dataset:
                col1, col2, col3 = st.columns(3)
                with col1:
                	st.write("**Dataset:** " + selected_dataset['dataset_name'])
                with col2:
                	st.write("**Variables:** " + selected_dataset['variables'])
                with col3:
                    file_size_mb = selected_dataset['file_size'] / (1024 * 1024)
                    st.write(f"**Size:** {file_size_mb:.1f} MB")
                if st.button("📂 " + ("Laad uit Cache" if lang == "nl" else "Load from Cache"), type="primary"):
                    with st.spinner("Data wordt geladen uit cache..." if lang == "nl" else "Loading data from cache..."):
                        config, data, record_count, max_step = load_from_cache(selected_dataset)

                        st.session_state.pipeline_results['cached_data'] = data
                        st.session_state.pipeline_results['raw_text_list'] = data  # Also populate for preprocessing

                        if config and data:
                            config.to_session_state()

                            # Also populate non-config session state variables needed for preprocessing
                            st.session_state.selected_id_column = config.id_column
                            selected_vars = config.selected_variables

                            if config.variable_mode == 'multiple' and len(selected_vars) > 1:
                                st.session_state.selected_variables = selected_vars
                                st.session_state.is_merged_variable = True
                            else:
                                st.session_state.selected_variable = selected_vars[0] if selected_vars else None
                                st.session_state.selected_variables = selected_vars
                                st.session_state.is_merged_variable = False

                            # Mark all completed steps based on cached data
                            for step in range(max_step + 1):
                                mark_step_completed(step)

                            # Set max step reached
                            st.session_state.max_step_reached = max_step

                            # Jump to next step (minimum step 1 to avoid infinite loop on step 0)
                            target_step = max(1, max_step)
                            st.session_state.step = target_step

                            st.success("✅ " + (f"Dataset geladen uit cache! ({record_count} records, voltooid t/m stap {max_step})" if lang == "nl" else f"Dataset loaded from cache! ({record_count} records, completed through step {max_step})"))
                            st.rerun()
                        else:
                            st.error("❌ " + ("Fout bij laden uit cache" if lang == "nl" else "Error loading from cache"))
        st.markdown("---")
    else:
        st.info("ℹ️ " + ("Geen cached datasets beschikbaar" if lang == "nl" else "No cached datasets available"))
        st.markdown("---")
    
    
    # ==================== OPTION 2: UPLOAD FROM FILE ====================
    st.subheader("📤 " + ("Upload Nieuw Bestand" if lang == "nl" else "Upload New File"))
    uploaded_file = st.file_uploader("Kies een SPSS bestand (.sav)" if lang == "nl" else "Choose a SPSS file (.sav)", type=['sav'], help=ui.get_text("UPLOAD_HELP", lang))
    if uploaded_file is not None:
        if st.button(ui.get_text("BTN_UPLOAD", lang), type="primary"):
            with st.spinner("Data wordt geladen..." if lang == "nl" else "Loading data..."):
                try:
                    filename, simple_variables, variables_with_types = load_from_file(uploaded_file)
                    st.session_state.filename = filename
                    st.session_state.uploaded_file_path = str(project_root / "data" / filename)
                    st.session_state.available_variables = simple_variables
                    st.session_state.available_variables_types = variables_with_types
                    st.session_state.loaded_from_cache = False
                    st.session_state.force_recalculate_all = True
                    reset_navigation_tracking()

                    st.success(f"Bestand geladen met {len(simple_variables)} variabelen!" if lang == "nl" else f"File loaded with {len(simple_variables)} variables!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Fout bij het uploaden: {str(e)}" if lang == "nl" else f"Upload error: {str(e)}")
    
    # Variabel selection
    if st.session_state.available_variables:
        st.subheader("📝 " + ("Variabele Selectie" if lang == "nl" else "Variable Selection"))
        variable_mode = st.radio( "Selectie Mode" if lang == "nl" else "Selection Mode", ["single", "multiple"],
            format_func=lambda x: "Enkele variabele" if x == "single" and lang == "nl" else "Single variable" if x == "single" else "Meerdere variabelen" if lang == "nl" else "Multiple variables",
            key="variable_mode",
            horizontal=True,
            help="Selecteer enkele variabele voor standaard analyse, of meerdere voor tekstsamenvoeging" if lang == "nl" else "Select single variable for standard analysis, or multiple for text merging")
        id_var = st.selectbox("🆔 " + ("Selecteer ID kolom" if lang == "nl" else "Select ID column"), options=list(st.session_state.available_variables.keys()), format_func=lambda x: f"{x} - {st.session_state.available_variables[x] or '(No label)'}", key="id_variable")
        
        if variable_mode == "single":
            string_vars = DatasetConfig.filter_string_variables(st.session_state.available_variables, st.session_state.get('available_variables_types', {}))

            if not string_vars:
                st.warning("Geen tekstvariabelen gevonden. Alle variabelen worden getoond." if lang == "nl" else "No text variables found. Showing all variables.")

            format_var = DatasetConfig.build_variable_format_func(
                st.session_state.available_variables,
                st.session_state.get('available_variables_types', {}),
                lang)

            text_var = st.selectbox("📄 " + ("Selecteer tekst variabele" if lang == "nl" else "Select text variable"), options=string_vars, format_func=format_var, key="text_variable")
            selected_variables = [text_var] if text_var else []

        else: #muliple
            string_vars = DatasetConfig.filter_string_variables( st.session_state.available_variables, st.session_state.get('available_variables_types', {}))

            if not string_vars:
                st.warning("Geen tekstvariabelen gevonden. Alle variabelen worden getoond." if lang == "nl"else "No text variables found. Showing all variables.")

            format_var = DatasetConfig.build_variable_format_func(
                st.session_state.available_variables,
                st.session_state.get('available_variables_types', {}),
                lang)

            selected_variables = st.multiselect("📄 " + ("Selecteer tekst variabelen om samen te voegen" if lang == "nl" else "Select text variables to merge"), options=string_vars, format_func=format_var, key="text_variables_multi", help="Selecteer meerdere variabelen die samengevoegd zullen worden tot één tekst" if lang == "nl" else "Select multiple variables that will be merged into one text")
            
            # Merge params 
            if selected_variables and len(selected_variables) > 1:
                with st.expander("🔧 " + ("Samenvoeg Opties" if lang == "nl" else "Merge Options"), expanded=True):
                    merge_col1, merge_col2 = st.columns(2)
                    
                    with merge_col1:
                        st.selectbox("Samenvoeg Strategie" if lang == "nl" else "Merge Strategy", ["concatenate", "first_available", "all_combined"],format_func=lambda x: {"concatenate": "Alles samenvoegen" if lang == "nl" else "Concatenate all", "first_available": "Eerste beschikbare" if lang == "nl" else "First available", "all_combined": "Alle met labels" if lang == "nl" else "All with labels"}[x], key="merge_strategy", help="Kies hoe meerdere variabelen samengevoegd worden" if lang == "nl" else "Choose how multiple variables are merged")
                    
                    with merge_col2:
                        separator_options = ["; ", ", ", " | "]
                        st.selectbox("Scheidingsteken" if lang == "nl" else "Separator", separator_options, format_func=lambda x: {"; ": "Puntkomma" if lang == "nl" else "Semicolon", ", ": "Komma" if lang == "nl" else "Comma", " | ": "Pijp symbool" if lang == "nl" else "Pipe symbol" }[x], key="merge_separator", help="Scheidingsteken tussen samengevoegde teksten" if lang == "nl" else "Separator between merged texts")
                    
                    st.checkbox("Lege waarden overslaan" if lang == "nl" else "Skip empty values", value=True, key="skip_empty", help="Variabelen zonder inhoud niet opnemen in samengevoegde tekst" if lang == "nl" else "Don't include variables without content in merged text")
            
        text_var = selected_variables[0] if selected_variables else None 
        
        # Sample size/truncation
        st.subheader("📊 " + ("Steekproef Optie" if lang == "nl" else "Sample Options"))
        sample_option = st.radio("Kies steekproef grootte" if lang == "nl" else "Choose sample size", ["Gebruik volledige steekproef" if lang == "nl" else "Use full sample", "Beperk steekproefgrootte" if lang == "nl" else "Limit sample size"], index=0, key="sample_option", help="Volledige steekproef gebruikt alle gevallen, beperkte steekproef voor snellere verwerking" if lang == "nl" else "Full sample uses all cases, limited sample for faster processing")
        
        sample_size = None
        if sample_option == ("Beperk steekproefgrootte" if lang == "nl" else "Limit sample size"):
            sample_size = st.number_input("Aantal gevallen" if lang == "nl" else "Number of cases",min_value=10, max_value=10000, value=50, step=10, key="sample_size", help="Aantal gevallen om te gebruiken (bijv. 250 voor snelle tests)" if lang == "nl" else "Number of cases to use (e.g., 250 for quick tests)" )
            
        # Preview config
        preview_button_label = "Voorbeeld Bekijken" if lang == "nl" else "Preview Variables"
        if variable_mode == "multiple" and len(selected_variables) > 1:
            preview_button_label = f"Voorbeeld van {len(selected_variables)} variabelen" if lang == "nl" else f"Preview {len(selected_variables)} variables"
        if sample_size:
            preview_button_label += f" (eerste {sample_size} gevallen)" if lang == "nl" else f" (first {sample_size} cases)"
        else:
            preview_button_label += " (volledige dataset)" if lang == "nl" else " (full dataset)"
            
        if st.button(preview_button_label):
            if selected_variables and id_var:
                if hasattr(st.session_state, 'available_variables_types') and st.session_state.available_variables_types:
                    temp_config = build_config_from_ui(
                        filename=st.session_state.filename,
                        id_var=id_var,
                        selected_vars=selected_variables,
                        encoding=st.session_state.get('file_encoding'))

                    is_valid, non_string_vars = temp_config.validate_text_variables(st.session_state.available_variables_types)

                    if not is_valid:
                        error_msg = temp_config.get_validation_error_message(non_string_vars, lang)
                        st.error(error_msg)
                        return

                with st.spinner("Data wordt geladen..." if lang == "nl" else "Loading data..."):
                    try:
                        encoding = st.session_state.get('file_encoding')
                        config = build_config_from_ui(
                            filename=st.session_state.filename,
                            id_var=id_var,
                            selected_vars=selected_variables,
                            encoding=encoding)

                        # Load data using pipeline step_0 (loads once for both preview and processing)
                        from utils.cacheManager import generate_enhanced_variable_key

                        # Determine parameters for step_0_load_data
                        is_multiple_mode = config.variable_mode == 'multiple' and len(config.selected_variables) > 1
                        sample_size = config.sample_size
                        merge_config = config.merge_config

                        # Generate variable key
                        variable_key = generate_enhanced_variable_key(
                            config.selected_variables,
                            is_merged=is_multiple_mode,
                            sample_size=sample_size,
                            merge_config=merge_config
                        )

                        # Call step_0_load_data to load data as ResponseModel list
                        if is_multiple_mode:
                            raw_text_list = pipeline.step_0_load_data(
                                filename=config.filename,
                                id_column=config.id_column,
                                var_names=config.selected_variables,
                                variable_key=variable_key,
                                cache_manager=_get_cache_manager(),
                                sample_size=sample_size,
                                merge_config=merge_config,
                                encoding=encoding if encoding != 'auto' else None,
                                force_recalc=st.session_state.get('force_recalculate_all', False),
                                verbose=True
                            )
                            text_column = 'merged_text'
                        else:
                            raw_text_list = pipeline.step_0_load_data(
                                filename=config.filename,
                                id_column=config.id_column,
                                var_name=config.selected_variables[0],
                                variable_key=variable_key,
                                cache_manager=_get_cache_manager(),
                                sample_size=sample_size,
                                encoding=encoding if encoding != 'auto' else None,
                                force_recalc=st.session_state.get('force_recalculate_all', False),
                                verbose=True
                            )
                            text_column = config.selected_variables[0]

                        # Convert ResponseModel list to DataFrame for preview display
                        preview_data = convert_response_models_to_preview_df(
                            raw_text_list,
                            id_column=config.id_column,
                            text_column=text_column
                        )

                        # Store config, preview DataFrame, and pipeline data
                        config.to_session_state()
                        st.session_state.variable_preview = preview_data

                        # Store data for pipeline processing (avoids duplicate load in step 1)
                        if 'pipeline_results' not in st.session_state:
                            st.session_state.pipeline_results = {}
                        st.session_state.pipeline_results['raw_text_list'] = raw_text_list

                        # Convert None (full sample) to actual count for display purposes
                        if st.session_state.sample_size_config is None:
                            st.session_state.sample_size_config = len(raw_text_list)

                        # Store variable label for pipeline
                        # Only fetch from SPSS if user hasn't edited var_lab
                        if st.session_state.get('var_lab'):
                            st.session_state.pipeline_results['var_lab'] = st.session_state.var_lab
                        else:
                            var_for_label = config.selected_variables[0] if config.selected_variables else None
                            if var_for_label:
                                var_lab = _get_data_loader().get_varlab(config.filename, var_for_label, encoding=encoding if encoding != 'auto' else None)
                                last_bracket = var_lab.rfind("]")
                                st.session_state.pipeline_results['var_lab'] = var_lab[last_bracket + 1:].strip()
                            else:
                                st.session_state.pipeline_results['var_lab'] = "Unknown Variable"

                    except Exception as e:
                        st.error(f"Fout bij preview: {str(e)}" if lang == "nl" else f"Preview error: {str(e)}")
            else:
                st.warning("Selecteer eerst variabelen en ID kolom" if lang == "nl" else "Please select variables and ID column first")

  
        # Display preview
        if st.session_state.variable_preview is not None:
            st.subheader("📊 Data Preview")
            preview_df = st.session_state.variable_preview

            variable_mode = st.session_state.get('variable_mode_config', 'single')
            if variable_mode == 'multiple':
                display_text_column = 'merged_text'
            else:
                # For single variable mode, get the selected variable from config
                selected_vars = st.session_state.get('selected_variables_config', [])
                display_text_column = selected_vars[0] if selected_vars else 'merged_text'

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
                sample_size_display = st.session_state.get('sample_size_config')
                if sample_size_display:
                    st.metric("Steekproef" if lang == "nl" else "Sample", sample_size_display)
                else:
                    st.metric("Steekproef" if lang == "nl" else "Sample", "Volledig" if lang == "nl" else "Full")

            config = DatasetConfig.from_session_state()
            if config:
                summary_info = config.get_preview_summary(lang)
                st.info(summary_info)

            st.subheader("📝 " + ("Voorbeeldgegevens" if lang == "nl" else "Sample Data"))
            sample_data = preview_df[preview_df[display_text_column].notna()].head(10)
            if len(sample_data) > 0:
                config = DatasetConfig.from_session_state()
                if config:
                    sample_data = config.format_preview_dataframe(sample_data, display_text_column)

                st.dataframe(sample_data, use_container_width=True)
            else:
                st.warning("Geen niet-lege gegevens gevonden" if lang == "nl" else "No non-empty data found")
            
            # Stor in session state
            selected_vars_config = st.session_state.get('selected_variables_config', [])
            var_for_label = selected_vars_config[0] if selected_vars_config else None

            if var_for_label:
                # Only fetch from SPSS if var_lab not already set by user
                if not st.session_state.get('var_lab'):
                    var_lab = _get_data_loader().get_varlab(st.session_state.filename, var_for_label)
                    last_bracket = var_lab.rfind("]")
                    st.session_state.var_lab = var_lab[last_bracket + 1:].strip()
            else:
                if not st.session_state.get('var_lab'):
                    st.session_state.var_lab = "Unknown Variable"

            # Allow user to edit survey question text
            st.markdown("---")
            edited_var_lab = st.text_area(
                "📝 " + ("Vraag tekst (bewerk indien nodig):" if lang == "nl" else "Question text (edit if needed):"),
                value=st.session_state.get('var_lab', ''),
                height=100,
                help="Deze vraag wordt gebruikt als context in de analyse" if lang == "nl" else "This question is used as context throughout the analysis",
                key="var_lab_editor"
            )

            # Update session state if modified
            if edited_var_lab != st.session_state.get('var_lab', ''):
                st.session_state.var_lab = edited_var_lab

            # Proceed
            if st.button("Doorgaan naar Preprocessing" if lang == "nl" else "Continue to Preprocessing", type="primary"):
                st.session_state.selected_id_column = st.session_state.get('id_column_config')
                current_mode = st.session_state.get('variable_mode_config', 'single')
                selected_vars = st.session_state.get('selected_variables_config', [])
                if current_mode == 'multiple' and len(selected_vars) > 1:
                    st.session_state.selected_variables = selected_vars
                    st.session_state['is_merged_variable'] = True
                else:
                    st.session_state.selected_variable = selected_vars[0] if selected_vars else None
                    st.session_state.selected_variables = selected_vars
                    st.session_state['is_merged_variable'] = False

                # Ensure sample_size_config reflects actual count (not user's None choice)
                if st.session_state.sample_size_config is None and 'raw_text_list' in st.session_state.pipeline_results:
                    st.session_state.sample_size_config = len(st.session_state.pipeline_results['raw_text_list'])

                mark_step_completed(0) 
                st.session_state.step = 1
                
                st.rerun()

# STEP 1. PREPROCESSING DATA ################################################################################################################################

def show_preprocessing_page():
    lang = st.session_state.language
    
    if False: #debug    
        debug_info = ""
        debug_info += f"Filename: {st.session_state.get('filename')}\n\n"
        debug_info += f"ID column: {st.session_state.get('id_column')}\n\n"
        debug_info += f"Selected variables: {st.session_state.get('selected_variables')}\n\n"
        debug_info += f"Variable mode: {st.session_state.get('variable_mode')}\n\n"
        debug_info += f"Sample size: {st.session_state.get('sample_size')}\n\n"
        debug_info += f"Merge config: {st.session_state.get('merge_config')}\n\n"
        debug_info += f"Encoding: {st.session_state.get('encoding')}\n\n"
        debug_info += f"Variable labels: {st.session_state.get('var_lab')}\n\n"
        debug_info += f"Is merged variable: {st.session_state.get('is_merged_variable')}\n\n"
        debug_info += f"Loaded from cache: {st.session_state.get('loaded_from_cache')}\n\n"
        debug_info += f"Force recalculate all: {st.session_state.get('force_recalculate_all')}\n\n"
    
        st.info(debug_info)
      
    st.header("Stap 1: Tekstverwerking" if lang == "nl" else "Step 1: Text preprocessing")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(1): 
        st.success("✅ " + ("Tekstverwerking voltooid! Bekijk de resultaten en klik dan op doorgaan." if lang == "nl" else "Preprocessing completed! Review the results on the right, then click continue."))
    
    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(0):
        sample_info =  (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        sample_info += (f"\n\n**Data:** {st.session_state.sample_size_config} antwoorden" if lang == "nl" else f"\n\n**Data:** { st.session_state.sample_size_config} responses")
        st.info(sample_info)    

    # ==================== BLOCK 3: YELLOW BOX ====================
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
                        + f"\n- { 'Hoofdletterwijzigingen' if nl else 'Case changes' }: {norm_stats.get('case_changes', 0)} "
                          f"{ 'reacties' if nl else 'responses' }"
                        + f"\n- { 'Witruimte opgeschoond' if nl else 'Whitespace cleanup' }: {norm_stats.get('whitespace_changes', 0)} "
                          f"{ 'reacties' if nl else 'responses' }"
                        + f"\n- { 'Schuine strepen vervangen' if nl else 'Slash replacements' }: {norm_stats.get('slash_changes', 0)} "
                          f"{ 'reacties' if nl else 'responses' }")
            
           # b) Spell checker stats
           spell_stats = stats.get('spellchecker_stats') or {}
           if spell_stats:
                nl = (st.session_state.language == "nl")
                summary_info += (
                        "\n\n" + ("**Spellingcontrole:**" if nl else "**Spell checking:**")
                        + f"\n- { 'Correcties' if nl else 'Corrections' }: {spell_stats.get('corrections_applied', 0)}")
            
           # c) Finalizer stats
           final_stats = stats.get('finalizer_stats') or {}
           if final_stats:
               nl = (st.session_state.language == "nl")
               summary_info += (
                        "\n\n" + ("**Finaliseren:**" if nl else "**Finalization:**")
                        + f"\n- { 'Leestekens toegevoegd' if nl else 'Punctuation additions' }: {final_stats.get('punctuation_additions', 0)} "
                          f"{ 'reacties' if nl else 'responses' }"
                        + f"\n- { 'Opmaak opgeschoond' if nl else 'Format cleanup' }: {final_stats.get('format_cleanup', 0)} "
                          f"{ 'reacties' if nl else 'responses' }"
                        + f"\n- { 'Spatieaanpassingen' if nl else 'Spacing fixes' }: {final_stats.get('spacing_fixes', 0)} "
                          f"{ 'reacties' if nl else 'responses' }")
    
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
    if is_step_completed(0) and not is_step_completed(1): 
        progress_container = st.empty()
        try: 
            if 'raw_text_list' not in st.session_state.pipeline_results: 
                #load data from file
                if not st.session_state.get('loaded_from_cache', False):
                    encoding = st.session_state.get('file_encoding', 'auto')
                    encoding = None if encoding == 'auto' else encoding
                    is_multiple_mode = (st.session_state.get('variable_mode_config') == 'multiple' or st.session_state.get('is_merged_variable', False))
                    selected_vars = st.session_state.get('selected_variables_config', [])
    
                    #multiple vars
                    if is_multiple_mode and len(selected_vars) > 1:
                        # Check if user edited var_lab exists, otherwise build from SPSS
                        if st.session_state.get('var_lab'):
                            var_lab = st.session_state.var_lab
                        else:
                            merge_config = st.session_state.get('merge_config', {})
                            var_labels = []
                            for var in selected_vars:
                                    label = _get_data_loader().get_varlab(st.session_state.filename, var, encoding=encoding)
                                    var_labels.append(label or var)
                            var_labs = f"Combined ({merge_config.get('strategy', 'concatenate')}): {' + '.join(var_labels)}"
                            var_lab = var_labs
                            st.session_state.var_lab = var_lab

                        progress_container.text("🔄 Data laden...")
                        merge_config = st.session_state.get('merge_config', {})
                        # Generate enhanced variable key with sample size
                        sample_size = st.session_state.get('sample_size_config')
                        variable_key = generate_enhanced_variable_key(
                            selected_vars,
                            is_merged=True,
                            sample_size=sample_size,
                            merge_config=merge_config
                        )
                        raw_text_list = pipeline.step_0_load_data(
                                filename=st.session_state.filename,
                                id_column=st.session_state.selected_id_column,
                                var_name=selected_vars[0],  # Use first variable (merged not supported yet)
                                variable_key=variable_key,
                                cache_manager=_get_cache_manager(),
                                sample_size=sample_size,
                                merge_config=merge_config,
                                force_recalc=st.session_state.get('force_recalculate_all', False),
                                verbose=True)
                        progress_container.success("✅ Data laden voltooid")
                            
                    else: #single vars
                        # Check if user edited var_lab exists, otherwise fetch from SPSS
                        if st.session_state.get('var_lab'):
                            var_lab = st.session_state.var_lab
                        else:
                            var_lab = _get_data_loader().get_varlab(st.session_state.filename, st.session_state.selected_variable, encoding=encoding)
                            last_bracket = var_lab.rfind("]")
                            var_lab = var_lab[last_bracket + 1:].strip()
                            st.session_state.var_lab = var_lab

                        progress_container.text("🔄 Data laden...")
                        # Generate enhanced variable key with sample size
                        sample_size = st.session_state.get('sample_size_config')
                        merge_config = st.session_state.get('merge_config')
                        variable_key = generate_enhanced_variable_key(
                            [st.session_state.selected_variable],
                            is_merged=False,
                            sample_size=sample_size,
                            merge_config=merge_config
                        )
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
                        progress_container.success("✅ Data laden voltooid")

                    st.session_state.pipeline_results['raw_text_list'] = raw_text_list
                    st.session_state.pipeline_results['var_lab'] = var_lab
            else:
                 # Data already loaded from preview - var_lab should already be in pipeline_results
                 if 'var_lab' not in st.session_state.pipeline_results:
                     st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', 'Unknown Variable')
        except Exception as e:
             st.error(f"Preprocessing fout: {str(e)}" if lang == "nl" else f"Preprocessing error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    if is_step_completed(0) and not is_step_completed(1):
        st.markdown(ui.get_text("PREPROCESSING_INFO", lang))

        # Show button to start preprocessing
        if st.button("🚀 " + ("Start Voorbewerking" if lang == "nl" else "Start Preprocessing"), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 Tekst aan het voorbewerken...")
                selected_variables = st.session_state.get('selected_variables_config', [st.session_state.selected_variable])
                is_merged = st.session_state.get('is_merged_variable', False)
                # Generate enhanced variable key with sample size
                sample_size = st.session_state.get('sample_size_config')
                merge_config = st.session_state.get('merge_config')
                variable_key = generate_enhanced_variable_key(
                    selected_variables,
                    is_merged=is_merged,
                    sample_size=sample_size,
                    merge_config=merge_config
                )
                # Check if we need to force recalculation due to cache invalidation
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
                st.error(f"Preprocessing fout: {str(e)}" if lang == "nl" else f"Preprocessing error: {str(e)}")

def show_filtering_page():
    """
    Step 2: Quality filtering (kwaliteitsfilter)

    Processes preprocessed_text from step 1 and exclude noise from further analysis.

    Pipeline function: step_2_extract_ideas
    Cache name: quality_filter
    Model: models.QualityFilteredModel
    """
    lang = st.session_state.language

    st.header("Stap 2: Kwaliteitsfiltering" if lang == "nl" else "Step 2: Quality Filtering")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(2):
        st.success("✅ " + ("Kwaliteitsfiltering voltooid! Bekijk de resultaten en klik dan op doorgaan." if lang == "nl" else "Quality filtering completed! Review the results on the right, then click continue."))

    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(1):
        sample_info =  (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        sample_info += (f"\n\n**Data:** {st.session_state.sample_size_config} antwoorden" if lang == "nl" else f"\n\n**Data:** {st.session_state.sample_size_config} responses")
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    if is_step_completed(2):
        if st.session_state.get('quality_filter_stats', {}):
            stats = st.session_state.get('quality_filter_stats', {})

            lines = []  # collect all code lines
            code_counts = stats.get('code_counts', {})
            code_meanings = stats.get('code_meanings', {})

            # sort safely even if keys are strings/ints
            for code in sorted(code_counts.keys(), key=str):
                count = code_counts.get(code, 0)
                meaning = (code_meanings.get(code) or code_meanings.get(str(code)) or 'Unknown')
                lines.append(f"- **Code {code}**:  {count} " + ("item(s)" if lang == "en" else "item(s)") + f" - {meaning}")

            total = stats.get('total_with_codes', 0) + stats.get('total_without_codes', 0)
            perc_with = (stats.get('total_with_codes', 0) / total * 100) if total else 0

            filtered_label = "**Uitgesloten van verdere analyse**" if lang == "nl" else "**Excluded from further analysis**"

            summary_text = f"\n\n- {filtered_label}:  {stats.get('total_with_codes', 0)} item(s) ({perc_with:.0f}%)\n\n" + "\n\n".join(lines)

            st.markdown(f"""
            <div style="
            border-radius: 10px;
            padding: 12px 16px;
            background-color: #FFF8E6;
            margin-top: 8px;
            color: #5C4102;">
            {summary_text}
            </div>
            """, unsafe_allow_html=True)

    # ==================== BLOCK 4: DATA LOADING ====================
    if is_step_completed(1) and not is_step_completed(2):
        progress_container = st.empty()
        try:
            if 'preprocessed_text' not in st.session_state.pipeline_results:
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

                # Try to load from cache first (works for both upload and cache routes)
                if cache_manager.is_cache_valid(st.session_state.filename, "preprocessed", variable_key):
                    progress_container.text("🔄 Voorbewerkte data laden uit cache..." if lang == "nl" else "🔄 Loading preprocessed data from cache...")
                    preprocessed_text = _load_or_recover(
                        st.session_state.filename,
                        "preprocessed",
                        variable_key,
                        models.PreprocessedModel
                    )
                    st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text
                    # Also populate var_lab if not already in pipeline_results
                    if 'var_lab' not in st.session_state.pipeline_results:
                        st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')
                    progress_container.success("✅ Data geladen uit cache" if lang == "nl" else "✅ Data loaded from cache")
                else:
                    # Upload route: process from raw_text_list
                    progress_container.text("🔄 Voorbewerkte data verwerken..." if lang == "nl" else "🔄 Processing preprocessed data...")
                    preprocessed_text, _ = pipeline.step_1_preprocess(
                        raw_text_list=st.session_state.pipeline_results['raw_text_list'],
                        filename=st.session_state.filename,
                        var_lab=st.session_state.pipeline_results['var_lab'],
                        variable_key=variable_key,
                        cache_manager=cache_manager,
                        model_config=st.session_state.model_config,
                        force_recalc=False,
                        verbose=True,
                        prompt_printer_enabled=False
                    )
                    st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text
                    progress_container.success("✅ Data verwerkt" if lang == "nl" else "✅ Data processed")
        except Exception as e:
            st.error(f"Filtering fout: {str(e)}" if lang == "nl" else f"Filtering error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    if is_step_completed(1) and not is_step_completed(2):
        st.markdown(ui.get_text("FILTERING_INFO", lang))

        # Show button to start quality filtering
        if st.button("🚀 " + ("Start Kwaliteitsfiltering" if lang == "nl" else "Start Quality Filtering"), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + ("Kwaliteit aan het filteren..." if lang == "nl" else "Filtering quality..."))

                # Get variable_key for caching
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
                force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= 2)

                quality_filtered_text = pipeline.step_2_quality_filter(
                    preprocessed_text=st.session_state.pipeline_results['preprocessed_text'],
                    filename=st.session_state.filename,
                    var_lab=st.session_state.pipeline_results['var_lab'],
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    model_config=st.session_state.model_config,
                    force_recalc=force_recalc,
                    verbose=True,
                    prompt_printer_enabled=False
                )

                progress_container.success("✅ " + ("Kwaliteitsfiltering voltooid" if lang == "nl" else "Quality filtering completed"))

                # Calculate statistics from results for display
                code_counts = {}
                code_meanings = {
                    99999997: "User missing: Don't know/only expressing uncertainty",
                    99999998: "System missing: NA",
                    99999999: "No answer: Empty strings/Single Characters/Only numbers/Nonsensical/gibberish/meaningless content"
                }

                for item in quality_filtered_text:
                    if item.quality_filter and item.quality_filter_code is not None:
                        code = item.quality_filter_code
                        code_counts[code] = code_counts.get(code, 0) + 1

                total_with_codes = sum(code_counts.values())
                total_without_codes = len(quality_filtered_text) - total_with_codes

                # Store results
                st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text
                st.session_state['quality_filter_stats'] = {
                    'code_counts': code_counts,
                    'code_meanings': code_meanings,
                    'total_with_codes': total_with_codes,
                    'total_without_codes': total_without_codes
                }

                # Mark step completed
                mark_step_completed(2)
                st.rerun()

            except Exception as e:
                st.error(f"Filtering fout: {str(e)}" if lang == "nl" else f"Filtering error: {str(e)}")

def show_idea_extraction_page():
    """
    Step 3: Idea Extraction (Idee-extractie)

    Processes quality_filtered_text from step 2 and extracts discrete ideas.

    Pipeline function: step_3_extract_ideas
    Cache name: extracted_ideas
    Model: models.IdeasExtractedModel
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 3: Opdelen antwoorden" if lang == "nl" else "Step 3: Splitting responses")

    # ==================== BLOCK 1: GREEN BOX ====================
    # Show completion status
    if is_step_completed(3):
        st.success("✅ " + (
            "Opdeling voltooid! Bekijk de resultaten en klik dan op doorgaan."
            if lang == "nl" else
            "Unitization completed! Review the results, then click continue."
        ))

    # ==================== BLOCK 2: BLUE BOX ====================
    # Show input data info when previous step is complete
    if is_step_completed(2):
        sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        sample_info += (f"\n\n**Data:** {st.session_state.get('step3_sample_size', st.session_state.sample_size_config)} {'antwoorden' if lang == 'nl' else 'responses'}")    
        if is_step_completed(3):
            sample_info += (f" / {st.session_state.get('step4_sample_size', st.session_state.sample_size_config)} {'deelantwoorden' if lang == 'nl' else 'answer parts'}")
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    # Show results/stats when current step is complete
    if is_step_completed(3):
        if st.session_state.get('idea_extraction_stats', {}):
            stats = st.session_state.get('idea_extraction_stats', {})
            summary_info = ""

            # Build stats display
            summary_info += (
                f"\n\n- {'Aantal (deel-) antwoorden' if lang == 'nl' else 'Total ideas'}: {stats.get('total_ideas', 0)}"
                + f"\n\n- {'Unieke (deel-) antwoorden' if lang == 'nl' else 'Unique ideas'}: {stats.get('unique_ideas', 0)}"
                + f"\n\n- {'Enkelvoudige reacties' if lang == 'nl' else 'Single-idea responses'}: {stats.get('single_idea_responses', 0)} ({stats.get('single_idea_percentage', 0):.1f}%)"
                + f"\n\n- {'Meervoudige reacties' if lang == 'nl' else 'Multi-idea responses'}: {stats.get('multi_idea_responses', 0)} ({stats.get('multi_idea_percentage', 0):.1f}%)"
            )

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
    # Load quality_filtered_text if not already in pipeline_results
    if is_step_completed(2) and not is_step_completed(3):
        progress_container = st.empty()
        try:
            if 'quality_filtered_text' not in st.session_state.pipeline_results:
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

                # Try to load from cache first (works for both upload and cache routes)
                if cache_manager.is_cache_valid(st.session_state.filename, "quality_filter", variable_key):
                    progress_container.text("🔄 " + ("Gefilterde data laden uit cache..." if lang == "nl" else "Loading filtered data from cache..."))
                    quality_filtered_text = _load_or_recover(
                        st.session_state.filename,
                        "quality_filter",
                        variable_key,
                        models.QualityFilteredModel
                    )
                    st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text
                    # Also populate var_lab if not already in pipeline_results
                    if 'var_lab' not in st.session_state.pipeline_results:
                        st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')
                    progress_container.success("✅ " + ("Data geladen uit cache" if lang == "nl" else "Data loaded from cache"))
                else:
                    # Upload route: process from preprocessed_text
                    progress_container.text("🔄 " + ("Gefilterde data verwerken..." if lang == "nl" else "Processing filtered data..."))
                    quality_filtered_text = pipeline.step_2_quality_filter(
                        preprocessed_text=st.session_state.pipeline_results['preprocessed_text'],
                        filename=st.session_state.filename,
                        var_lab=st.session_state.pipeline_results['var_lab'],
                        variable_key=variable_key,
                        cache_manager=cache_manager,
                        model_config=st.session_state.model_config,
                        force_recalc=False,
                        verbose=True,
                        prompt_printer_enabled=False
                    )
                    st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text
                    progress_container.success("✅ " + ("Data verwerkt" if lang == "nl" else "Data processed"))
        except Exception as e:
            st.error(f"Idee-extractie fout: {str(e)}" if lang == "nl" else f"Idea extraction error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    # Show processing button when ready to process
    if is_step_completed(2) and not is_step_completed(3):
        st.markdown(ui.get_text("EXTRACTION_INFO", lang))

        # Show button to start idea extraction
        if st.button("🚀 " + (
            "Start Idee-extractie" if lang == "nl"
            else "Start Idea Extraction"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "Ideeën aan het extraheren..." if lang == "nl"
                    else "Extracting ideas..."
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

                # Set force_recalc flag (respects both global and step-specific invalidation)
                force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= 3)

                # Call pipeline processing function
                extracted_ideas = pipeline.step_3_extract_ideas(
                    quality_filtered_text=st.session_state.pipeline_results['quality_filtered_text'],
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
                    "Idee-extractie voltooid" if lang == "nl"
                    else "Idea extraction completed"
                ))

                # Calculate statistics from results for display
                total_responses = len(extracted_ideas)
                total_ideas = sum(item.idea_count for item in extracted_ideas)

                # Count unique ideas
                unique_ideas_set = set()
                for item in extracted_ideas:
                    if hasattr(item, 'response_ideas') and item.response_ideas:
                        for idea_obj in item.response_ideas:
                            if hasattr(idea_obj, 'idea'):
                                unique_ideas_set.add(idea_obj.idea)
                unique_ideas = len(unique_ideas_set)

                # Count single vs multi-idea responses
                single_idea_responses = sum(1 for item in extracted_ideas if item.idea_count == 1)
                multi_idea_responses = sum(1 for item in extracted_ideas if item.idea_count > 1)

                single_idea_percentage = (single_idea_responses / total_responses * 100) if total_responses > 0 else 0
                multi_idea_percentage = (multi_idea_responses / total_responses * 100) if total_responses > 0 else 0

                # Store results (Note: for backward compatibility, also store as 'encoded_text')
                st.session_state.pipeline_results['extracted_ideas'] = extracted_ideas
                st.session_state.pipeline_results['encoded_text'] = extracted_ideas  # Backward compatibility
                st.session_state['idea_extraction_stats'] = {
                    'total_responses': total_responses,
                    'total_ideas': total_ideas,
                    'unique_ideas': unique_ideas,
                    'single_idea_responses': single_idea_responses,
                    'multi_idea_responses': multi_idea_responses,
                    'single_idea_percentage': single_idea_percentage,
                    'multi_idea_percentage': multi_idea_percentage
                }

                # Mark step completed
                mark_step_completed(3)
                st.rerun()

            except Exception as e:
                st.error(f"Idee-extractie fout: {str(e)}" if lang == "nl" else f"Idea extraction error: {str(e)}")

def show_embedding_page():
    """
    Step 4: Embedding Generation (Genereer Embeddings)

    Generates vector embeddings for extracted ideas from step 3.

    Pipeline function: step_4_generate_embeddings
    Cache name: embeddings
    Model: models.EmbeddingsModel
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 4: Genereer Embeddings" if lang == "nl" else "Step 4: Generate Embeddings")

    # ==================== BLOCK 1: GREEN BOX ====================
    # Show completion status
    if is_step_completed(4):
        st.success("✅ " + ("Embeddings gegenereerd! Klik op doorgaan." if lang == "nl" else "Embeddings generated! Click continue."))

    # ==================== BLOCK 2: BLUE BOX ====================
    # Show input data info when previous step is complete
    if is_step_completed(3):
        sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        if is_step_completed(4):
            sample_info += (f"**Data**: {st.session_state.get('step4_sample_size', st.session_state.sample_size_config)} {'embeddings' if lang == 'nl' else 'embeddings'}")
        else:
            sample_info += (f"**Data**: {st.session_state.get('step4_sample_size', st.session_state.sample_size_config)} {'deelantwoorden te embedden' if lang == 'nl' else 'answer parts to embed'}")
        st.info(sample_info)

    # ==================== BLOCK 3: DATA LOADING ====================
    # Load encoded_text if not already in pipeline_results
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

                # Try to load from cache first (works for both upload and cache routes)
                if cache_manager.is_cache_valid(st.session_state.filename, "extracted_ideas", variable_key):
                    progress_container.text("🔄 " + ("Geëxtraheerde ideeën laden uit cache..." if lang == "nl" else "Loading extracted ideas from cache..."))
                    encoded_text = _load_or_recover(
                        st.session_state.filename,
                        "extracted_ideas",
                        variable_key,
                        models.IdeasExtractedModel
                    )
                    st.session_state.pipeline_results['encoded_text'] = encoded_text
                    # Also populate var_lab if not already in pipeline_results
                    if 'var_lab' not in st.session_state.pipeline_results:
                        st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')
                    progress_container.success("✅ " + ("Data geladen uit cache" if lang == "nl" else "Data loaded from cache"))
                else:
                    progress_container.error("❌ " + ("Geen geëxtraheerde ideeën gevonden. Voer eerst stap 3 uit." if lang == "nl" else "No extracted ideas found. Please run step 3 first."))
        except Exception as e:
            st.error(f"Embedding fout: {str(e)}" if lang == "nl" else f"Embedding error: {str(e)}")

    # ==================== BLOCK 4: PROCESSING BUTTON ====================
    # Show processing button when ready to process
    if is_step_completed(3) and not is_step_completed(4):
        st.markdown(ui.get_text("EMBEDDING_INFO", lang))

        # Show button to start embedding generation
        if st.button("🚀 " + (
            "Genereer Embeddings" if lang == "nl"
            else "Generate Embeddings"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "Embeddings aan het genereren..." if lang == "nl"
                    else "Generating embeddings..."
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

                # Set force_recalc flag (respects both global and step-specific invalidation)
                force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= 4)

                # Call pipeline processing function
                embedded_text = pipeline.step_4_generate_embeddings(
                    encoded_text=st.session_state.pipeline_results['encoded_text'],
                    filename=st.session_state.filename,
                    var_lab=st.session_state.pipeline_results['var_lab'],
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    model_config=st.session_state.model_config,
                    force_recalc=force_recalc,
                    verbose=True
                )

                progress_container.success("✅ " + (
                    "Embeddings gegenereerd" if lang == "nl"
                    else "Embeddings generated"
                ))

                # Store results
                st.session_state.pipeline_results['embedded_text'] = embedded_text

                # Mark step completed
                mark_step_completed(4)
                st.rerun()

            except Exception as e:
                st.error(f"Embedding fout: {str(e)}" if lang == "nl" else f"Embedding error: {str(e)}")


def show_clustering_page():
    """
    Step 5: Clustering

    Performs UMAP dimensionality reduction and HDBSCAN clustering on embeddings from step 4.

    Pipeline function: step_5_cluster
    Cache name: initial_clusters
    Model: models.ClusterModel
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 5: Clustering" if lang == "nl" else "Step 5: Clustering")

    # ==================== BLOCK 1: GREEN BOX ====================
    # Show completion status
    if is_step_completed(5):
        st.success("✅ " + (
            "Clustering voltooid! Bekijk de resultaten en klik dan op doorgaan."
            if lang == "nl" else
            "Clustering completed! Review the results, then click continue."))

    # ==================== BLOCK 2: BLUE BOX ====================
    # Show input data info when previous step is complete
    if is_step_completed(4):
        sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        if not is_step_completed(5):
            sample_info += (f"**Data**: {st.session_state.get('step4_sample_size', st.session_state.sample_size_config)} {'embeddings te clusteren' if lang == 'nl' else 'embeddings to cluster'}")
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    # Show results/stats when current step is complete
    if is_step_completed(5):
        if st.session_state.get('clustering_stats', {}):
            stats = st.session_state.get('clustering_stats', {})
            nl = (lang == "nl")

            summary_info = (
                f"\n\n- {'Aantal clusters' if nl else 'Total clusters'}: {stats.get('num_clusters', 0)}"
                +f"\n\n- {'Aantal embeddings geclustered' if nl else 'Embeddings clustered'}: {stats.get('total_segments', 0)}"
                + f"\n\n- {'Ruis' if nl else 'Noise'}: {stats.get('outliers', 0)} "
                + f"({stats.get('outlier_percentage', 0):.1f}%)"
            )

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
    # Load embedded_text if not already in pipeline_results
    if is_step_completed(4) and not is_step_completed(5):
        progress_container = st.empty()
        try:
            if 'embedded_text' not in st.session_state.pipeline_results:
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

                # Try to load from cache first (works for both upload and cache routes)
                if cache_manager.is_cache_valid(st.session_state.filename, "embeddings", variable_key):
                    progress_container.text("🔄 " + ("Embeddings laden uit cache..." if lang == "nl" else "Loading embeddings from cache..."))
                    embedded_text = _load_or_recover(
                        st.session_state.filename,
                        "embeddings",
                        variable_key,
                        models.EmbeddingsModel
                    )
                    st.session_state.pipeline_results['embedded_text'] = embedded_text
                    # Also populate var_lab if not already in pipeline_results
                    if 'var_lab' not in st.session_state.pipeline_results:
                        st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')
                    progress_container.success("✅ " + ("Data geladen uit cache" if lang == "nl" else "Data loaded from cache"))
                else:
                    progress_container.error("❌ " + ("Geen embeddings gevonden. Voer eerst stap 4 uit." if lang == "nl" else "No embeddings found. Please run step 4 first."))
        except Exception as e:
            st.error(f"Clustering fout: {str(e)}" if lang == "nl" else f"Clustering error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    # Show processing button when ready to process
    if is_step_completed(4) and not is_step_completed(5):
        st.markdown(ui.get_text("CLUSTERING_INFO", lang))

        # Show button to start clustering
        if st.button("🚀 " + ("Start Clustering" if lang == "nl"else "Start Clustering"), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + ("Clustering aan het uitvoeren..." if lang == "nl" else "Running clustering..."))

                # Generate variable_key for caching
                selected_variables = st.session_state.get('selected_variables_config', [st.session_state.selected_variable])
                is_merged = st.session_state.get('is_merged_variable', False)
                sample_size = st.session_state.get('sample_size_config')
                merge_config = st.session_state.get('merge_config')
                variable_key = generate_enhanced_variable_key(
                    selected_variables,
                    is_merged=is_merged,
                    sample_size=sample_size,
                    merge_config=merge_config)

                # Set force_recalc flag (respects both global and step-specific invalidation)
                force_recalc = st.session_state.get('force_recalculate_all', False) or (st.session_state.get('force_recalculate_from_step', 99) <= 5)

                # Call pipeline processing function
                initial_cluster_results = pipeline.step_5_cluster(
                    embedded_text=st.session_state.pipeline_results['embedded_text'],
                    filename=st.session_state.filename,
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    force_recalc=force_recalc,
                    verbose=True)

                progress_container.success("✅ " + ("Clustering voltooid" if lang == "nl" else "Clustering completed"))

                # Store results
                st.session_state.pipeline_results['initial_cluster_results'] = initial_cluster_results

                # Calculate clustering statistics
                cluster_ids = set([
                    segment.initial_cluster
                    for result in initial_cluster_results
                    for segment in result.response_ideas
                    if segment.initial_cluster is not None and segment.initial_cluster >= 0])

                outliers = sum(
                    1 for result in initial_cluster_results
                    for segment in result.response_ideas
                    if segment.initial_cluster == -1)

                total_segments = sum(len(result.response_ideas) for result in initial_cluster_results)

                st.session_state['clustering_stats'] = {
                    'num_clusters': len(cluster_ids),
                    'total_segments': total_segments,
                    'outliers': outliers,
                    'outlier_percentage': (outliers / total_segments * 100) if total_segments > 0 else 0}

                # Mark step completed
                mark_step_completed(5)
                st.rerun()

            except Exception as e:
                st.error(f"Clustering fout: {str(e)}" if lang == "nl" else f"Clustering error: {str(e)}")

def show_codebook_generation_page():
    """
    Step 6: Codebook Generation (Codebook Generatie)

    Generates codes from step 5's "reduced" data. 

    Pipeline function: step_6_generate_codebook
    Cache name: codebook_generation
    Model: models.CodebookModel
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 6: Generatie ruwe codes" if lang == "nl" else "Step 6: Unpolised code generation")

    # ==================== BLOCK 1: GREEN BOX ====================
    # Show completion status
    if is_step_completed(6):
        st.success("✅ " + (
            "Codes gegenereerd! Bekijk de resultaten en klik dan op doorgaan."
            if lang == "nl" else
            "Codes generated! Review the results, then click continue."))

    # ==================== BLOCK 2: BLUE BOX ====================
    # Show input data info when previous step is complete
    if is_step_completed(5):
        # Get clustering stats from single source of truth
        # If not in session state yet, calculate from cached cluster data
        if 'clustering_stats' not in st.session_state:
            cache_manager = _get_cache_manager()
            variable_key = _get_variable_key_for_cache()
            if variable_key:
                cluster_data = cache_manager.load_from_cache(
                    st.session_state.filename,
                    "initial_clusters",
                    variable_key,
                    models.ClusterModel
                )
                if cluster_data:
                    cluster_ids = set(
                        segment.initial_cluster
                        for result in cluster_data
                        for segment in result.response_ideas
                        if segment.initial_cluster != -1
                    )
                    st.session_state['clustering_stats'] = {
                        'num_clusters': len(cluster_ids)
                    }

        num_clusters = st.session_state.get('clustering_stats', {}).get('num_clusters', 0)
        sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        sample_info += (f"\n\n**Data:** {num_clusters} {'clusters om te coderen' if lang == 'nl' else 'clusters to code'}")
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    # skipped / not necesarry
     
    # ==================== BLOCK 4: DATA LOADING ====================
    # Load initial_cluster_results if not already in pipeline_results
    if is_step_completed(5) and not is_step_completed(6):
        progress_container = st.empty()
        try:
            if 'initial_cluster_results' not in st.session_state.pipeline_results:
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

                # Try to load from cache first (works for both upload and cache routes)
                if cache_manager.is_cache_valid(st.session_state.filename, "initial_clusters", variable_key):
                    progress_container.text("🔄 " + ("Cluster resultaten laden uit cache..." if lang == "nl" else "Loading cluster results from cache..."))
                    initial_cluster_results = _load_or_recover(
                        st.session_state.filename,
                        "initial_clusters",
                        variable_key,
                        models.ClusterModel
                    )
                    st.session_state.pipeline_results['initial_cluster_results'] = initial_cluster_results

                    # Populate clustering_stats if not already present (for cache route)
                    if 'clustering_stats' not in st.session_state:
                        cluster_ids = set(
                            segment.initial_cluster
                            for result in initial_cluster_results
                            for segment in result.response_ideas
                            if segment.initial_cluster != -1
                        )
                        outliers = sum(
                            1 for result in initial_cluster_results
                            for segment in result.response_ideas
                            if segment.initial_cluster == -1
                        )
                        total_segments = sum(len(result.response_ideas) for result in initial_cluster_results)

                        st.session_state['clustering_stats'] = {
                            'num_clusters': len(cluster_ids),
                            'total_segments': total_segments,
                            'outliers': outliers,
                            'outlier_percentage': (outliers / total_segments * 100) if total_segments > 0 else 0
                        }

                    # Also populate var_lab if not already in pipeline_results
                    if 'var_lab' not in st.session_state.pipeline_results:
                        st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')
                    progress_container.success("✅ " + ("Data geladen uit cache" if lang == "nl" else "Data loaded from cache"))
                else:
                    progress_container.error("❌ " + ("Geen cluster resultaten gevonden. Voer eerst stap 5 uit." if lang == "nl" else "No cluster results found. Please run step 5 first."))
        except Exception as e:
            st.error(f"Codebook fout: {str(e)}" if lang == "nl" else f"Codebook error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    # Show processing button when ready to process
    if is_step_completed(5) and not is_step_completed(6):
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

        # Show button to start codebook generation
        if st.button("🚀 " + (
            "Start Codebook Generatie" if lang == "nl"
            else "Start Codebook Generation"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "Codebook aan het genereren..." if lang == "nl"
                    else "Generating codebook..."
                ))

                # Determine variable name for codebook generation
                var_name_for_codebook = st.session_state.selected_variable
                if (st.session_state.get('is_merged_variable', False) and
                    st.session_state.get('selected_variables_config')):
                    selected_vars = st.session_state.get('selected_variables_config', [])
                    if len(selected_vars) > 1:
                        var_name_for_codebook = f"merged_{'-'.join(selected_vars[:3])}"

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
                force_recalc = st.session_state.get('force_recalculate_all', False) or \
                               (st.session_state.get('force_recalculate_from_step', 99) <= 6)

                # Call pipeline processing function
                codebook_main, reasoning_results = pipeline.step_6_generate_codebook(
                    initial_cluster_results=st.session_state.pipeline_results['initial_cluster_results'],
                    filename=st.session_state.filename,
                    var_name=var_name_for_codebook,
                    var_lab=st.session_state.pipeline_results['var_lab'],
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    model_config=st.session_state.model_config,
                    use_speculative_starter_codes=use_speculative,
                    force_recalc=force_recalc,
                    verbose=True,
                    verbose_detailed=True,
                    prompt_printer_enabled=False,
                    cache_reasoning=True
                )

                progress_container.success("✅ " + (
                    "Codebook generatie voltooid" if lang == "nl"
                    else "Codebook generation completed"
                ))

                # Store results
                st.session_state.pipeline_results['codebook_main'] = codebook_main
                st.session_state.pipeline_results['reasoning_results'] = reasoning_results

                # Calculate codebook statistics
                st.session_state['codebook_stats'] = {
                    'num_codes': len(codebook_main.codes) if codebook_main and hasattr(codebook_main, 'codes') else 0,
                    'unique_clusters': len(set([entry.source_cluster for entry in codebook_main.codes if entry.source_cluster])) if codebook_main and hasattr(codebook_main, 'codes') else 0
                }

                # Mark step completed
                mark_step_completed(6)
                st.rerun()

            except Exception as e:
                st.error(f"Codebook fout: {str(e)}" if lang == "nl" else f"Codebook error: {str(e)}")

def show_theme_identification_page():
    """ Step 7: Thematic codebook (Thematisch codeboek)"""
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 7: Thematisch codeboek" if lang == "nl" else "Step 7: Thematic codebook")

    # ==================== BLOCK 1: GREEN BOX ====================
    # Show completion status
    if is_step_completed(7):
        st.success("✅ " + ("Thema's geïdentificeerd! Bekijk de resultaten en klik op doorgaan."
                           if lang == "nl" else "Themes identified! Review the results and click continue."))

    # ==================== BLOCK 2: BLUE BOX ====================
    # Show input data info when previous step is complete
    if is_step_completed(6):
        # Get num_codes from session state (set by step 6 or data loading block)
        num_codes = st.session_state.get('codebook_stats', {}).get('num_codes', 0)

        sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        codes_text = "codes om te groeperen in thema's" if lang == 'nl' else 'codes to group into themes'
        sample_info += (f"\n\n**Data:** {num_codes} {codes_text}")
        st.info(sample_info)

    # # ==================== BLOCK 3: YELLOW BOX ====================
    # skipped/not necesarry
  
    # ==================== BLOCK 4: DATA LOADING ====================
    # Load reasoning_results and codebook_main if not already in pipeline_results
    if is_step_completed(6) and not is_step_completed(7):
        progress_container = st.empty()
        try:
            if 'reasoning_results' not in st.session_state.pipeline_results:
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
                # Note: Step 6 caches two separate items:
                # 1. "codebook_generation" - contains [codebook_main]
                # 2. "codebook_generation_reasoning" - contains [reasoning_results]
                if cache_manager.is_cache_valid(st.session_state.filename, "codebook_generation", variable_key):
                    progress_container.text("🔄 " + ("Codebook laden uit cache..." if lang == "nl" else "Loading codebook from cache..."))

                    # Load codebook_main from main cache
                    codebook_list = _load_or_recover(
                        st.session_state.filename,
                        "codebook_generation",
                        variable_key,
                        models.CodebookModel
                    )

                    if codebook_list and len(codebook_list) > 0:
                        codebook_main = codebook_list[0]
                        st.session_state.pipeline_results['codebook_main'] = codebook_main

                        # Populate codebook_stats if not already present (for cache route)
                        if 'codebook_stats' not in st.session_state:
                            st.session_state['codebook_stats'] = {
                                'num_codes': len(codebook_main.codes) if codebook_main and hasattr(codebook_main, 'codes') else 0,
                                'unique_clusters': len(set([entry.source_cluster for entry in codebook_main.codes if entry.source_cluster])) if codebook_main and hasattr(codebook_main, 'codes') else 0
                            }

                        # Load reasoning_results from separate reasoning cache
                        try:
                            from utils import codeGenerator
                            reasoning_list = _load_or_recover(
                                st.session_state.filename,
                                "codebook_generation_reasoning",
                                variable_key,
                                codeGenerator.CodeGeneratorReasoningResults
                            )
                            if reasoning_list and len(reasoning_list) > 0:
                                st.session_state.pipeline_results['reasoning_results'] = reasoning_list[0]
                                progress_container.success("✅ " + ("Codebook en reasoning geladen uit cache" if lang == "nl" else "Codebook and reasoning loaded from cache"))
                            else:
                                st.error("⚠️ " + ("Reasoning resultaten niet gevonden in cache. Voer stap 6 opnieuw uit." if lang == "nl"
                                               else "Reasoning results not found in cache. Please re-run step 6."))
                        except Exception as e:
                            st.error("⚠️ " + ("Reasoning resultaten laden mislukt: {str(e)}. Voer stap 6 opnieuw uit." if lang == "nl"
                                              else f"Failed to load reasoning results: {str(e)}. Please re-run step 6."))
                    else:
                        st.error("Ongeldige cache data voor codebook generatie. Voer stap 6 opnieuw uit." if lang == "nl"
                               else "Invalid cache data for codebook generation. Please re-run step 6.")
                else:
                    st.error("Invoer data niet gevonden. Voer eerst codebook generatie (stap 6) uit." if lang == "nl"
                           else "Input data not found. Please run codebook generation (step 6) first.")

            # Populate metadata
            if 'var_lab' not in st.session_state.pipeline_results:
                st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')

        except Exception as e:
            st.error(f"Data laad fout: {str(e)}" if lang == "nl" else f"Data loading error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    # Show processing button when ready to process
    if is_step_completed(6) and not is_step_completed(7):
        st.markdown(ui.get_text("THEME_IDENTIFICATION_INFO", lang))

        # Show button to start theme identification
        if st.button("🚀 " + (
            "Start Thema Identificatie" if lang == "nl"
            else "Start Theme Identification"
        ), type="primary"):
            progress_container = st.empty()
            try:
                # Determine variable name for theme identification
                var_name_for_themes = st.session_state.selected_variable
                if (st.session_state.get('is_merged_variable', False) and
                    st.session_state.get('selected_variables_config')):
                    selected_vars = st.session_state.get('selected_variables_config', [])
                    if len(selected_vars) > 1:
                        var_name_for_themes = f"merged_{'-'.join(selected_vars[:3])}"

                progress_container.text("🔄 Codebook aan het verfijnen..." if lang == "nl"
                                       else "🔄 Refining codebook...")

                # Get variable_key for caching
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

                # Calculate force_recalc with step-specific logic
                force_recalc = st.session_state.get('force_recalculate_all', False) or \
                              (st.session_state.get('force_recalculate_from_step', 99) <= 7)

                refinement_results, theme_enriched_codebook, refinement_report = pipeline.step_7_refine_codebook(
                    codebook_reasoning=st.session_state.pipeline_results['reasoning_results'],
                    filename=st.session_state.filename,
                    var_name=var_name_for_themes,
                    var_lab=st.session_state.pipeline_results['var_lab'],
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    model_config=st.session_state.model_config,
                    default_language=st.session_state.get('language', 'nl'),
                    force_recalc=force_recalc,
                    verbose=True
                )

                progress_container.success("✅ " + (
                    "Thema identificatie voltooid" if lang == "nl"
                    else "Theme identification completed"
                ))

                # Store results
                st.session_state.pipeline_results['theme_enriched_codebook'] = theme_enriched_codebook
                st.session_state.pipeline_results['refinement_results'] = refinement_results
                st.session_state.pipeline_results['refinement_report'] = refinement_report

                # Calculate and store theme statistics
                num_themes = len(theme_enriched_codebook.themes_summary) if theme_enriched_codebook.themes_summary else 0
                num_codes = len(theme_enriched_codebook.codes) if hasattr(theme_enriched_codebook, 'codes') else 0

                st.session_state['theme_stats'] = {
                    'num_themes': num_themes,
                    'num_codes': num_codes
                }

                # Mark step 7 completed
                mark_step_completed(7)
                st.rerun()
            except Exception as e:
                st.error(f"Thema fout: {str(e)}" if lang == "nl" else f"Theme error: {str(e)}")

def show_code_assignment_page():
    """
    Step 8: Code Assignment

    Assigns codes from the theme-enriched codebook to individual ideas.

    Pipeline function: step_8_assign_codes
    Cache name: code_assignment_direct
    Model: models.CodeAssignedModel
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 8: Code Toewijzing" if lang == "nl" else "Step 8: Code Assignment")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(8):
        st.success("✅ " + (
            "Code toewijzing voltooid! Bekijk de resultaten en klik dan op doorgaan."
            if lang == "nl"
            else "Code assignment completed! Review the results, then click continue."
        ))

    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(7):
        sample_info = (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        sample_info += (f"\n\n**Data:** {st.session_state.get('step4_sample_size', 0)} {'ideeën' if lang == 'nl' else 'ideas'}")
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    if is_step_completed(8):
        if st.session_state.get('code_assignment_stats', {}):
            stats = st.session_state.get('code_assignment_stats', {})
            nl = (lang == "nl")

            # Translations for stats labels
            t_responses = "Antwoorden verwerkt" if nl else "Responses processed"
            t_ideas = "Ideeën verwerkt" if nl else "Ideas processed"
            t_unique_codes = "Unieke codes toegewezen" if nl else "Unique codes assigned"
            t_unique_themes = "Unieke themas toegewezen" if nl else "Unique themes assigned"
            t_total_codes = "Totaal code toewijzingen" if nl else "Total code assignments"
            t_total_themes = "Totaal thema toewijzingen" if nl else "Total theme assignments"
            t_avg_codes = "Gemiddeld codes per idee" if nl else "Average codes per idea"
            t_avg_themes = "Gemiddeld themas per idee" if nl else "Average themes per idea"
            t_time = "Verwerkingstijd" if nl else "Processing time"

            summary_info = (
                "\n\n" + ("**Toewijzing:**" if nl else "**Assignment:**")
                + f"\n- {t_responses}: {stats.get('total_responses', 0)}"
                + f"\n- {t_ideas}: {stats.get('total_ideas', 0)}"
                + f"\n- {t_unique_codes}: {stats.get('unique_codes_assigned', 0)}"
                + f"\n- {t_unique_themes}: {stats.get('unique_themes_assigned', 0)}"
                + f"\n- {t_total_codes}: {stats.get('total_code_assignments', 0)}"
                + f"\n- {t_total_themes}: {stats.get('total_theme_assignments', 0)}"
                + f"\n- {t_avg_codes}: {stats.get('avg_codes_per_idea', 0):.2f}"
                + f"\n- {t_avg_themes}: {stats.get('avg_themes_per_idea', 0):.2f}"
                + f"\n- {t_time}: {stats.get('processing_time', 0):.1f}s"
            )

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
    if is_step_completed(7) and not is_step_completed(8):
        progress_container = st.empty()
        try:
            cache_manager = _get_cache_manager()

            # Generate variable_key for cache lookup
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

            # Load initial_cluster_results if not present (from step 5)
            if 'initial_cluster_results' not in st.session_state.pipeline_results:
                if cache_manager.is_cache_valid(st.session_state.filename, "initial_clusters", variable_key):
                    progress_container.text("🔄 " + ("Cluster resultaten laden uit cache..." if lang == "nl" else "Loading cluster results from cache..."))
                    initial_cluster_results = cache_manager.load_from_cache(
                        st.session_state.filename, "initial_clusters", variable_key, models.ClusterModel
                    )
                    st.session_state.pipeline_results['initial_cluster_results'] = initial_cluster_results
                    progress_container.success("✅ " + ("Cluster data geladen" if lang == "nl" else "Cluster data loaded"))

            # Load theme_enriched_codebook if not present (from step 7)
            if 'theme_enriched_codebook' not in st.session_state.pipeline_results:
                if cache_manager.is_cache_valid(st.session_state.filename, "codebook_refinement_enriched", variable_key):
                    progress_container.text("🔄 " + ("Codebook laden uit cache..." if lang == "nl" else "Loading codebook from cache..."))
                    # Load theme_enriched_codebook from its dedicated cache (returns list, extract first element)
                    theme_enriched_data = cache_manager.load_from_cache(
                        st.session_state.filename, "codebook_refinement_enriched", variable_key, models.ThemeEnrichedCodebookModel
                    )
                    if theme_enriched_data and len(theme_enriched_data) > 0:
                        st.session_state.pipeline_results['theme_enriched_codebook'] = theme_enriched_data[0]
                        progress_container.success("✅ " + ("Codebook geladen" if lang == "nl" else "Codebook loaded"))

            # Populate var_lab if not present
            if 'var_lab' not in st.session_state.pipeline_results:
                st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')

            # Clear progress container after successful loading
            progress_container.empty()

        except Exception as e:
            st.error(f"Data laad fout: {str(e)}" if lang == "nl" else f"Data loading error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    if is_step_completed(7) and not is_step_completed(8):
        info_text = """
        Deze stap zal:
        - Codes toewijzen aan individuele ideeën
        - Thema's koppelen aan toegewezen codes
        - Statistieken berekenen
        """ if lang == "nl" else """
        This step will:
        - Assign codes to individual ideas
        - Link themes to assigned codes
        - Calculate statistics
        """
        st.markdown(info_text)

        # Show button to start code assignment
        if st.button("🚀 " + (
            "Wijs Codes Toe" if lang == "nl"
            else "Assign Codes"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "Codes aan het toewijzen..." if lang == "nl"
                    else "Assigning codes..."
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

                # Set force_recalc flag (respects both global and step-specific invalidation)
                force_recalc = st.session_state.get('force_recalculate_all', False) or \
                               (st.session_state.get('force_recalculate_from_step', 99) <= 8)

                # Call pipeline processing function
                code_assigned_results, code_assignment_stats = pipeline.step_8_assign_codes(
                    initial_cluster_results=st.session_state.pipeline_results['initial_cluster_results'],
                    theme_enriched_codebook=st.session_state.pipeline_results['theme_enriched_codebook'],
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
                    "Code toewijzing voltooid" if lang == "nl"
                    else "Code assignment completed"
                ))

                # Store results
                st.session_state.pipeline_results['code_assigned_results'] = code_assigned_results
                st.session_state['code_assignment_stats'] = code_assignment_stats

                # Mark step completed
                mark_step_completed(8)
                st.rerun()

            except Exception as e:
                st.error(f"Toewijzing fout: {str(e)}" if lang == "nl" else f"Assignment error: {str(e)}")

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
                    # Advance to next step
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
                        st.session_state.get('selected_variables_config')):
                        # Use first variable name or create composite name for merged variables
                        selected_vars = st.session_state.get('selected_variables_config', [])
                        if len(selected_vars) > 1:
                            var_name_for_export = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
                    
                    progress_container.text("🔄 " + ("Resultaten exporteren naar Excel..." if lang == "nl" else "Exporting results to Excel..."))

                    excel_path = pipeline.step_9_export_results(
                        code_assigned_results=code_assigned_results,
                        theme_enriched_codebook=theme_enriched_codebook,
                        filename=st.session_state.filename,
                        var_name=var_name_for_export,
                        verbose=True
                    )

                    progress_container.success("✅ " + (f"Code toewijzingen geëxporteerd naar Excel: {excel_path}"
                                              if lang == "nl" else f"Code assignments exported to Excel: {excel_path}"))
                else:
                    # Use regular export without reasoning data (via pipelineRunner for consistency)
                    progress_container.text("🔄 " + ("Resultaten exporteren naar Excel..." if lang == "nl" else "Exporting results to Excel..."))
                    
                    # Determine variable name for export (use meaningful name for merged variables)
                    var_name_for_export = st.session_state.selected_variable
                    if (st.session_state.get('is_merged_variable', False) and 
                        st.session_state.get('selected_variables_config')):
                        # Use first variable name or create composite name for merged variables
                        selected_vars = st.session_state.get('selected_variables_config', [])
                        if len(selected_vars) > 1:
                            var_name_for_export = f"merged_{'-'.join(selected_vars[:3])}"  # Limit to first 3 for readability
                    
                    excel_path = pipeline.step_9_export_results(
                        code_assigned_results=code_assigned_results,
                        theme_enriched_codebook=theme_enriched_codebook,
                        filename=st.session_state.filename,
                        var_name=var_name_for_export,
                        verbose=True
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

                # Mark step 10 (export) as completed in navigation tracker
                mark_step_completed(10)
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
                    st.session_state.get('selected_variables_config')):
                    # Use first variable name or create composite name for merged variables
                    selected_vars = st.session_state.get('selected_variables_config', [])
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
                st.session_state.get('selected_variables_config')):
                # Use first variable name or create composite name for merged variables
                selected_vars = st.session_state.get('selected_variables_config', [])
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
        
        # Check if we're in multiple variable mode
        is_multiple_mode = (st.session_state.get('variable_mode_config') == 'multiple' or
                           st.session_state.get('is_merged_variable', False))

        selected_vars = (st.session_state.get('selected_variables') or
                        st.session_state.get('selected_variables_config', []))

        if is_multiple_mode and selected_vars and len(selected_vars) > 1:
            # Multiple variables mode - use first variable for preview
            merge_config = st.session_state.get('merge_config')
            sample_size = st.session_state.get('sample_size_config')
            # Generate enhanced variable key with sample size for preview
            variable_key = generate_enhanced_variable_key(
                selected_vars,
                is_merged=True,
                sample_size=sample_size,
                merge_config=merge_config
            )

            preview_data = pipeline.step_0_load_data(
                filename=st.session_state.filename,
                id_column=st.session_state.selected_id_column,
                var_name=selected_vars[0],  # Use first variable only
                variable_key=variable_key,
                cache_manager=_get_cache_manager(),
                sample_size=sample_size,
                merge_config=merge_config,
                force_recalc=True,  # Always force recalc for preview
                verbose=False
            )
            # Take first n_samples for preview
            preview_data = preview_data[:n_samples] if len(preview_data) > n_samples else preview_data
        elif st.session_state.get('selected_variable'):
            # Single variable mode
            sample_size = st.session_state.get('sample_size_config')
            merge_config = st.session_state.get('merge_config')
            # Generate enhanced variable key with sample size for preview
            variable_key = generate_enhanced_variable_key(
                [st.session_state.selected_variable],
                is_merged=False,
                sample_size=sample_size,
                merge_config=merge_config
            )

            preview_data = pipeline.step_0_load_data(
                filename=st.session_state.filename,
                id_column=st.session_state.selected_id_column,
                var_name=st.session_state.selected_variable,
                variable_key=variable_key,
                cache_manager=_get_cache_manager(),
                sample_size=sample_size,
                merge_config=merge_config,
                force_recalc=True,  # Always force recalc for preview
                verbose=False
            )
            # Take first n_samples for preview
            preview_data = preview_data[:n_samples] if len(preview_data) > n_samples else preview_data
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

    # Filter out empty/filtered items before sampling
    valid_items = [item for item in raw_text_list if not getattr(item, 'quality_filter', False)]

    if not valid_items:
        st.write("No valid data to display")
        return

    # Original pattern: random.sample(range(len(raw_text_list)), n_samples)
    indices = random.sample(range(len(valid_items)), min(n_samples, len(valid_items)))

    st.write(f"**Random samples from {len(valid_items)} filtered responses:**")

    if indices:
        sample_text = ""
        for i in indices:
            response_text = valid_items[i].response if valid_items[i].response is not None else ""
            sample_text += f"{response_text}\n"
        
        # Display in gray container
        st.code(sample_text.strip(), language=None)
    

def show_preprocessed_samples(preprocessed_text, n_samples=10):
    """Show random samples from Step 1.5 - Preprocessed Data"""

    st.write("\n")

    if not preprocessed_text:
        st.write(f"{'Geen verwerkte data beschikbaar' if st.session_state.language == 'nl' else 'No preprocessed data available'}")
        return

    # Filter out empty/filtered items before sampling
    valid_items = [item for item in preprocessed_text if not getattr(item, 'quality_filter', False)]

    if not valid_items:
        st.write(f"{'Geen geldige data om weer te geven' if st.session_state.language == 'nl' else 'No valid data to display'}")
        return

    indices = random.sample(range(len(valid_items)), min(n_samples, len(valid_items)))

    if indices:
        items = "".join( f"<li>{html.escape(valid_items[i].response or '')}</li>"  for i in indices)
        header = ('Willekeurige selectie' if st.session_state.language == 'nl' else 'Random sample')
        
        st.markdown(f"""
        <div style="
            border-radius: 10px;
            padding: 16px 20px;
            background-color: #F8F9FB;
            margin-top: 8px;
            line-height: 1.6;">
        <b style="display:block; margin-bottom:12px;">{header}:</b>
        <ul style="margin-top:10px;margin-bottom:0;padding-left:0;">
        {items}
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
        st.write("\n")
        
        if st.button(f"{'🎲 Toon nieuwe selectie' if st.session_state.language == 'nl' else '🎲 Draw new sample'}", key="preprocessed_samples"):
            st.rerun()

def show_filtered_samples(quality_filtered_text, n_samples=10):
    """Show random samples from Step 3 - Quality Filtered Data"""

    #st.write("\n")

    if not quality_filtered_text:
        st.write("{'Geen gefilterde data beschikbaar' if st.session_state.language == 'nl' else 'No filtered data available'}")
        return

    # Filter out empty strings (code 99999999) from display while keeping them in statistics
    filtered_text = [item for item in quality_filtered_text if item.quality_filter and item.quality_filter_code != 99999999]

    if not filtered_text:
        st.write(f"{'Geen gefilterde data om weer te geven' if st.session_state.language == 'nl' else 'No filtered data to display'}")
        return

    indices = random.sample(range(len(filtered_text)), min(n_samples, len(filtered_text)))
    
    if indices:
        items = "".join( f"<li>{html.escape(filtered_text[i].response or '')}</li>"  for i in indices)
        
        header = (
            'Willekeurige selectie' 
            if st.session_state.language == 'nl' 
            else 'Random selection'
        )
        
        caption = (
            "Uitgesloten"
            if st.session_state.language == "nl"
            else "Excluded"
        )
    
        st.markdown(f"""
        <div style="
            border-radius: 10px;
            padding: 16px 20px;
            background-color: #F8F9FB;
            margin-top: 8px;
            line-height: 1.6;">
        <b style="display:block; margin-bottom:12px;">{header}</b>
        <span style="display:block; margin-bottom:0px; font-style:italic;">{caption}:</span>
        <ul style="margin-top:10px;margin-bottom:0;padding-left:0;">
        {items}
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
        st.write("\n")
        
        if st.button(f"{'🎲 Toon nieuwe selectie' if st.session_state.language == 'nl' else '🎲 Draw new sample'}", key="preprocessed_samples"):
            st.rerun()
    

def show_idea_samples(encoded_text, n_samples=5):
    """Show random samples from Step 4 - Ideas"""
    
    if not encoded_text:
        st.write("Geen data beschikbaar" if st.session_state.language == "nl" else "No data available")
    else:
        sampled_items = random.sample(encoded_text, min(n_samples, len(encoded_text)))
    
        sections = []
        for item in sampled_items:
            resp = html.escape(item.response.strip()) if getattr(item, "response", None) else ""
            section_html = f"{resp}"
    
            # Add ideas if available
            if getattr(item, "response_ideas", None):
                idea_lines = []
                for seg in item.response_ideas:
                    if getattr(seg, "idea", None):
                        idea_lines.append(f"<li>{html.escape(seg.idea)}</li>")
                if idea_lines:
                    section_html += "<ul style='margin-top:4px; margin-bottom:8px; padding-left:1.2em;'>" + "".join(idea_lines) + "</ul>"
    
            sections.append(section_html)
    
        # Combine all responses with spacing between them
        full_html = "<br>".join(sections)
    
        header = (
            "Willekeurige selectie"
            if st.session_state.language == "nl"
            else "Random sample"
        )
        
        caption = (
            "Antwoord → descriptieve code van deelantwoord"
            if st.session_state.language == "nl"
            else "Response → descriptive code of answer part"
        )
    
        st.markdown(f"""
        <div style="
            border: 1px solid #dce1eb;
            border-radius: 10px;
            padding: 16px 20px;
            background-color: #F8F9FB;
            margin-top: 8px;
            line-height: 1.6;">
          <b style="display:block; margin-bottom:12px;">{header}</b>
          <span style="display:block; margin-bottom:12px; font-style:italic;">{caption}:</span>
          <hr style="border: 0; border-top: 1px solid white; margin: 8px 0;">
          {full_html}
        </div>
        """, unsafe_allow_html=True)
        
        st.write("\n")
        
        if st.button(f"{'🎲 Toon nieuwe selectie' if st.session_state.language == 'nl' else '🎲 Draw new random examples'}", key="idea_samples"):
            st.rerun()


def show_cluster_samples(initial_cluster_results):
    """Show cluster samples with HTML rendering and back/forward navigation."""

    # i18n helpers
    NL = st.session_state.get("language", "en") == "nl"
    t_no_data = "Geen data beschikbaar" if NL else "No data available"
    t_no_clusters = "Geen clusters gevonden" if NL else "No clusters found"
    t_prev = "⬅️ Vorige" if NL else "⬅️ Previous"
    t_next = "➡️ Volgende" if NL else "➡️ Next"
    t_of = "van" if NL else "of"
    t_cluster_label = "Cluster" if NL else "Cluster"
    t_items = "items" if NL else "items"

    if not initial_cluster_results:
        st.markdown(f"""
        <div style="border:1px solid #dce1eb;border-radius:10px;padding:16px 20px;background:#F8F9FB;margin-top:8px;">
          <span style="display:block;">{t_no_data}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # Build {cluster_id: [idea, idea, ...]}
    cluster_dict = {}
    for result in initial_cluster_results:
        for ri in getattr(result, "response_ideas", []) or []:
            cid = getattr(ri, "initial_cluster", None)
            idea = getattr(ri, "idea", None)
            if cid is None or idea is None:
                continue
            if cid == -1:
                continue  # skip noise cluster
            cluster_dict.setdefault(cid, []).append(idea)

    if not cluster_dict:
        st.markdown(f"""
        <div style="border:1px solid #dce1eb;border-radius:10px;padding:16px 20px;background:#F8F9FB;margin-top:8px;">
          <span style="display:block;">{t_no_clusters}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    cluster_ids = sorted(cluster_dict.keys())

    # Session state for current position
    if "cluster_idx" not in st.session_state:
        st.session_state.cluster_idx = 0

    total = len(cluster_ids)
    idx = st.session_state.cluster_idx

    # Navigation header (buttons + indicator + optional jump)
    nav1, nav2, nav3, nav4 = st.columns([1, 2, 2, 2])
    with nav1:
        if st.button(t_prev, use_container_width=True, disabled=(idx <= 0)):
            st.session_state.cluster_idx = max(0, idx - 1)
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='margin-top:6px;text-align:center;'>"
            f"{idx + 1} {t_of} {total}"
            f"</div>",
            unsafe_allow_html=True
        )
    with nav3:
        # Quick jump by index (keeps UI snappy for many clusters)
        new_idx = st.number_input(
            label="X",
            min_value=1, max_value=total, value=idx + 1,
            step=1, key="cluster_jump_number", label_visibility="collapsed"
        )
        if new_idx - 1 != idx:
            st.session_state.cluster_idx = new_idx - 1
            st.rerun()
    with nav4:
        if st.button(t_next, use_container_width=True, disabled=(idx >= total - 1)):
            st.session_state.cluster_idx = min(total - 1, idx + 1)
            st.rerun()

    # Active cluster
    active_cid = cluster_ids[st.session_state.cluster_idx]
    ideas = cluster_dict[active_cid]

    # Build HTML list of ideas
    li_items = []
    for idea in ideas:
        li_items.append(f"<li style='margin:4px 0;'>{html.escape(idea)}</li>")
    ideas_html = "".join(li_items)

    # Render nicely styled card (HTML only)
    st.markdown(f"""
    <div style="
        border: 1px solid #dce1eb;
        border-radius: 10px;
        padding: 16px 20px;
        background-color: #F8F9FB;
        margin-top: 8px;
        line-height: 1.6;">
      
      <div style="
          border: 1px solid #e6eaf2;
          border-radius: 8px;
          padding: 12px 14px;
          background-color: #ffffff;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <div><b>{t_cluster_label} {html.escape(str(active_cid))}</b></div>
          <div style="font-size:12px;opacity:0.8;">{len(ideas)} {t_items}</div>
        </div>
        <ul style="margin:8px 0 0 1.2em; padding:0;">
          {ideas_html}
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    #save number of clusters
    st.session_state.num_clusters = len(cluster_ids)

def show_codebook_samples(codebook_reasoning):
    import re

    NL = st.session_state.get("language", "en") == "nl"
    t_no_data = "Geen codebook-data beschikbaar" if NL else "No codebook data available"
    t_header = "Codebook-analyse" if NL else "Codebook analysis"
    t_error = "Fout bij weergeven van codebook-analyse" if NL else "Error displaying codebook analysis"

    # Translations for sections
    t_ideas = "Cluster Ideeën" if NL else "Cluster Ideas"
    t_code_in_codebook = "Code in Codebook" if NL else "Code in Codebook"
    t_analysis = "Cluster Analyse" if NL else "Cluster Analysis"
    t_theme = "Cluster Thema & Beschrijving" if NL else "Cluster Theme & Description"
    t_candidates = "Kandidaat Codes" if NL else "Candidate Codes"
    t_recommendations = "Aanbeveling" if NL else "Recommendation"
    t_decision = "Beslissing" if NL else "Decision"
    t_motivation = "Motivering" if NL else "Motivation"
    t_verdict = "Verdict" if NL else "Verdict"
    t_validated_decision = "Gevalideerde Beslissing" if NL else "Validated Decision"
    t_code = "Code" if NL else "Code"
    t_definition = "Definitie" if NL else "Definition"
    t_theme_label = "Thema Label" if NL else "Theme Label"
    t_theme_desc = "Thema Beschrijving" if NL else "Theme Description"
    t_decision_type = "Beslissingstype" if NL else "Decision Type"
    t_source_code = "Broncode" if NL else "Source Code"
    t_justification = "Rechtvaardiging" if NL else "Justification"
    t_final_code = "Finale Code" if NL else "Final Code"
    t_found = "gevonden" if NL else "found"
    t_responses = "reacties" if NL else "responses"
    t_existing = "(Bestaand)" if NL else "(Existing)"

    # Navigation translations
    t_prev = "⬅️ Vorige" if NL else "⬅️ Previous"
    t_next = "➡️ Volgende" if NL else "➡️ Next"
    t_of = "van" if NL else "of"
    # t_cluster = "Cluster" if NL else "Cluster"
    # t_ideas_count = "ideeën" if NL else "ideas"
    # t_coded = "✅ Gecodeerd" if NL else "✅ Coded"
    # t_not_coded = "⚠️ Niet gecodeerd" if NL else "⚠️ Not coded"

    if not codebook_reasoning:
        st.markdown(f"""
        <div style="border:1px solid #dce1eb;border-radius:10px;padding:16px 20px;background:#F8F9FB;margin-top:8px;">
          <span style="display:block;">{t_no_data}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # Get available cluster ids
    step3_recs = getattr(codebook_reasoning, "step3_recommendations", {}) or {}
    if not step3_recs:
        st.markdown(f"""
        <div style="border:1px solid #dce1eb;border-radius:10px;padding:16px 20px;background:#F8F9FB;margin-top:8px;">
          <span style="display:block;">{t_no_data}</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    
    # Calculate cluster sizes and sort by size (largest first)
    step1_inputs = getattr(codebook_reasoning, "step1_inputs", {}) or {}
    cluster_sizes = {}
    for cid in step3_recs.keys():
        if cid in step1_inputs:
            cluster_text = step1_inputs[cid].get('cluster_text', '')
            # Count ideas (newline-separated items)
            ideas = [idea.strip() for idea in cluster_text.split('\n') if idea.strip()]
            cluster_sizes[cid] = len(ideas)
        else:
            cluster_sizes[cid] = 0

    # Sort clusters by size (largest first)
    available_ids = sorted(step3_recs.keys(), key=lambda cid: cluster_sizes.get(cid, 0), reverse=True)
    total_clusters = len(available_ids)

    # Initialize session state for cluster navigation
    if "codebook_cluster_idx" not in st.session_state:
        st.session_state.codebook_cluster_idx = 0

    # Boundary check (handles cache invalidation changing cluster count)
    if st.session_state.codebook_cluster_idx >= total_clusters:
        st.session_state.codebook_cluster_idx = 0

    # Select cluster based on current index
    idx = st.session_state.codebook_cluster_idx
    cluster_id = available_ids[idx]

    # Get structured cluster data using new function
    try:
        from utils.codegenResults import get_cluster_analysis
        cluster_data = get_cluster_analysis(codebook_reasoning, cluster_id=cluster_id)

        if 'error' in cluster_data:
            st.markdown(f"""
            <div style="border:1px solid #ffd2d2;border-radius:10px;padding:16px 20px;background:#fff5f5;margin-top:8px;">
              <span style="display:block;">{t_error}: {html.escape(cluster_data['error'])}</span>
            </div>
            """, unsafe_allow_html=True)
            return
    except Exception as e:
        st.markdown(f"""
        <div style="border:1px solid #ffd2d2;border-radius:10px;padding:16px 20px;background:#fff5f5;margin-top:8px;">
          <span style="display:block;">{t_error}: {html.escape(str(e))}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # # Header with cluster ID
    # cluster_size = cluster_sizes.get(cluster_id, 0)
    # has_code = cluster_data['validation']['verdict']
    # code_status = t_coded if has_code else t_not_coded

    # st.markdown(f"""
    # <div style="margin-bottom:16px;">
    #   <b style="display:block; font-size:1.1em; margin-bottom:4px;">{t_header} - {t_cluster} {cluster_id}</b>
    #   <span style="display:block; font-style:italic; color:#666;">{idx + 1} {t_of} {total_clusters}</span>
    # </div>
    # """, unsafe_allow_html=True)

    # Cluster metadata info panel
    #st.info(f"{t_cluster} {cluster_id} • {cluster_size} {t_ideas_count} • {code_status}")

    # Navigation controls (4-column layout)
    nav1, nav2, nav3, nav4 = st.columns([1, 2, 2, 2])

    # Previous button
    with nav1:
        if st.button(t_prev, use_container_width=True, disabled=(idx <= 0), key="codebook_prev"):
            st.session_state.codebook_cluster_idx = max(0, idx - 1)
            st.rerun()

    # Position indicator
    with nav2:
        st.markdown(
            f"<div style='margin-top:6px;text-align:center;'>"
            f"{idx + 1} {t_of} {total_clusters}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Jump to cluster
    with nav3:
        new_idx = st.number_input(
            label="X",
            min_value=1, max_value=total_clusters, value=idx + 1,
            step=1, key="codebook_jump_number", label_visibility="collapsed"
        )
        if new_idx - 1 != idx:
            st.session_state.codebook_cluster_idx = new_idx - 1
            st.rerun()

    # Next button
    with nav4:
        if st.button(t_next, use_container_width=True, disabled=(idx >= total_clusters - 1), key="codebook_next"):
            st.session_state.codebook_cluster_idx = min(total_clusters - 1, idx + 1)
            st.rerun()

    # Visual separator
    st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)

    # Section 1: Code in Codebook (expanded by default, yellow content background)
    validation = cluster_data['validation']
    if validation['verdict']:
        validated_code = validation.get('validated_code', {}) or {}
        code_val = validated_code.get('code', '')
        def_val = validated_code.get('definition', '')

        if code_val or def_val:
            st.markdown(f"""
            <details open style="margin-bottom:12px;">
              <summary style="
                cursor: pointer;
                padding: 12px 16px;
                background: #f0f2f6;
                border-radius: 8px;
                font-weight: 600;
                margin-bottom: 8px;
                user-select: none;">
                📋 {t_code_in_codebook}
              </summary>
              <div style="
                padding: 16px;
                background: #fffbe6;
                border: 1px solid #ffe58f;
                border-radius: 8px;
                margin-top: 8px;
                line-height: 1.6;">
                {f'<div style="margin-bottom:8px;"><b>{t_code}:</b> {html.escape(str(code_val))}</div>' if code_val else ''}
                {f'<div><b>{t_definition}:</b> {html.escape(str(def_val))}</div>' if def_val else ''}
              </div>
            </details>
            """, unsafe_allow_html=True)

    # Section 2: Cluster Ideas (expanded by default)
    ideas = cluster_data['ideas']
    if ideas['count'] > 0:
        ideas_html = ''.join([f'<li style="margin-bottom:8px;">{html.escape(idea)}</li>'
                              for idea in ideas['ideas_list']])
        st.markdown(f"""
        <details open style="margin-bottom:12px;">
          <summary style="
            cursor: pointer;
            padding: 12px 16px;
            background: #f0f2f6;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            user-select: none;">
            💡 {t_ideas} ({ideas['count']} {t_responses})
          </summary>
          <div style="
            padding: 16px;
            background: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            margin-top: 8px;">
            <ol style="line-height: 1.6; margin: 0; padding-left: 20px;">
              {ideas_html}
            </ol>
          </div>
        </details>
        """, unsafe_allow_html=True)

    # Section 3: Cluster Analysis (collapsed by default)
    analysis = cluster_data['analysis']
    if analysis['text']:
        # Format analysis text with line breaks before sentences containing colons
        analysis_text = html.escape(str(analysis['text']))
        analysis_formatted = re.sub(r'\. ([A-Z0-9][^.]*:)', r'.<br>\1', analysis_text)

        st.markdown(f"""
        <details style="margin-bottom:12px;">
          <summary style="
            cursor: pointer;
            padding: 12px 16px;
            background: #f0f2f6;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            user-select: none;">
            🧠 {t_analysis}
          </summary>
          <div style="
            padding: 16px;
            background: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            margin-top: 8px;
            line-height: 1.6;">
            {analysis_formatted}
          </div>
        </details>
        """, unsafe_allow_html=True)

    # Section 3: Cluster Theme & Description (collapsed by default)
    theme = cluster_data['theme']
    if theme['theme_label'] or theme['theme_description']:
        st.markdown(f"""
        <details style="margin-bottom:12px;">
          <summary style="
            cursor: pointer;
            padding: 12px 16px;
            background: #f0f2f6;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            user-select: none;">
            🏷️ {t_theme}
          </summary>
          <div style="
            padding: 16px;
            background: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            margin-top: 8px;
            line-height: 1.6;">
            {f'<div style="margin-bottom:8px;"><b>{t_theme_label}:</b> {html.escape(str(theme["theme_label"]))}</div>' if theme['theme_label'] else ''}
            {f'<div><b>{t_theme_desc}:</b> {html.escape(str(theme["theme_description"]))}</div>' if theme['theme_description'] else ''}
          </div>
        </details>
        """, unsafe_allow_html=True)

    # Section 4: Candidate Codes (collapsed by default)
    candidates = cluster_data['candidate_codes']
    if candidates:
        candidates_html = ''.join([
            f'''<div style="margin-bottom:12px;">
              <b>{html.escape(str(c.get('code', 'N/A')))}</b><br>
              <span style="color:#666;">{html.escape(str(c.get('definition', 'N/A')))}</span>
            </div>'''
            for c in candidates
        ])
        st.markdown(f"""
        <details style="margin-bottom:12px;">
          <summary style="
            cursor: pointer;
            padding: 12px 16px;
            background: #f0f2f6;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            user-select: none;">
            🔍 {t_candidates} {t_existing} ({len(candidates)} {t_found})
          </summary>
          <div style="
            padding: 16px;
            background: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            margin-top: 8px;
            line-height: 1.6;">
            {candidates_html}
          </div>
        </details>
        """, unsafe_allow_html=True)

    # Section 5: Recommendation (collapsed by default)
    recommendations = cluster_data['recommendations']
    if recommendations['decision_type']:
        st.markdown(f"""
        <details style="margin-bottom:12px;">
          <summary style="
            cursor: pointer;
            padding: 12px 16px;
            background: #f0f2f6;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            user-select: none;">
            🎯 {t_recommendations}
          </summary>
          <div style="
            padding: 16px;
            background: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            margin-top: 8px;
            line-height: 1.6;">
            <div style="margin-bottom:12px;">
              <b>{t_decision_type}:</b> {html.escape(str(recommendations['decision_type']))}
            </div>
            <div style="margin-bottom:12px;">
              <b>{t_source_code}:</b> {html.escape(str(recommendations.get('source_code', 'N/A')))}
            </div>
            <div style="margin-bottom:12px;">
              <b>{t_justification}:</b> {html.escape(str(recommendations['justification'] or 'N/A'))}
            </div>
            <div>
              <b>{t_final_code}:</b> {html.escape(str(recommendations.get('final_code_label', 'N/A')))}
            </div>
          </div>
        </details>
        """, unsafe_allow_html=True)

    # Section 7: Decision (collapsed by default, includes motivation)
    validation = cluster_data['validation']
    if validation['verdict']:
        rationale = validation.get('rationale')
        st.markdown(f"""
        <details style="margin-bottom:12px;">
          <summary style="
            cursor: pointer;
            padding: 12px 16px;
            background: #f0f2f6;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            user-select: none;">
            ✅ {t_decision}
          </summary>
          <div style="
            padding: 16px;
            background: white;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            margin-top: 8px;
            line-height: 1.6;">
            <div style="margin-bottom:12px;">
              <b>{t_verdict}:</b> {html.escape(str(validation['verdict']))}
            </div>
            <div style="margin-bottom:12px;">
              <b>{t_validated_decision}:</b> {html.escape(str(validation['validated_decision']))}
            </div>
            {f'<div><b>{t_motivation}:</b> {html.escape(str(rationale))}</div>' if rationale else ''}
          </div>
        </details>
        """, unsafe_allow_html=True)

def show_theme_samples(refinement_report):
    import html
    import streamlit as st

    if not refinement_report:
        st.write("No refinement data available")
        return

    lang = st.session_state.get('language', 'nl')
    t_analysis = "LLM Analyse" if lang == "nl" else "LLM Analysis"
    t_categories = "Codeboek" if lang == "nl" else "Codebook"
    t_subcodes = "subcodes" if lang == "nl" else "subcodes"

    # --- Codebook first (expanded) ---
    categories = refinement_report.get('categories', [])
    if categories:
        st.markdown(f"### {t_categories}")
        for category in categories:
            category_name = category.get('category_name', 'Unknown')
            subcode_count = category.get('subcode_count', 0)
            subcodes = category.get('subcodes', []) or []

            # IMPORTANT: no indentation in the generated HTML
            subcodes_html = "".join([
                (
                    '<div style="margin-bottom:12px;padding-left:8px;">'
                    f'<b style="color:#1f77b4;">{html.escape(str(sc.get("code", "N/A")))}</b><br>'
                    f'<span style="color:#666;font-size:14px;">{html.escape(str(sc.get("description", "N/A")))}</span>'
                    '</div>'
                )
                for sc in subcodes
            ])

            # Also ensure {subcodes_html} is flush-left in the f-string (no leading spaces)
            st.markdown(f"""
<details open style="margin-bottom:16px;">
  <summary style="cursor:pointer;padding:12px 16px;background:#f0f2f6;border-radius:8px;font-weight:600;margin-bottom:8px;user-select:none;">
    📂 {html.escape(str(category_name))} ({subcode_count} {t_subcodes})
  </summary>
  <div style="padding:16px;background:white;border:1px solid #e6e6e6;border-radius:8px;margin-top:8px;line-height:1.6;">
{subcodes_html}
  </div>
</details>
""", unsafe_allow_html=True)
    else:
        st.write("No categories available")

    # --- Analysis second (collapsed) ---
    analysis_text = (refinement_report.get('analysis') or {}).get('text')
    if analysis_text:
        st.markdown(f"""
<details style="margin:16px 0;">
  <summary style="cursor:pointer;padding:12px 16px;background:#f0f2f6;border-radius:8px;font-weight:600;margin-bottom:8px;user-select:none;">
    📝 {t_analysis}
  </summary>
  <div style="padding:16px;background:white;border:1px solid #e6e6e6;border-radius:8px;margin-top:8px;line-height:1.6;">
    {html.escape(str(analysis_text))}
  </div>
</details>
""", unsafe_allow_html=True)



# def show_theme_samples(refinement_report):
#     """Display refinement report with HTML formatting in collapsible sections"""
#     import html

#     if not refinement_report:
#         st.write("No refinement data available")
#         return

#     lang = st.session_state.get('language', 'nl')

#     # Translations
#     t_analysis = "LLM Analyse" if lang == "nl" else "LLM Analysis"
#     t_categories = "Codeboek" if lang == "nl" else "Codebook"
#     t_subcodes = "subcodes" if lang == "nl" else "subcodes"

#     # Section 1: LLM Analysis (collapsed by default)
#     analysis_text = refinement_report.get('analysis', {}).get('text')
#     if analysis_text:
#         st.markdown(f"""
#         <details style="margin-bottom: 16px;">
#           <summary style="
#             cursor: pointer;
#             padding: 12px 16px;
#             background: #f0f2f6;
#             border-radius: 8px;
#             font-weight: 600;
#             margin-bottom: 8px;
#             user-select: none;">
#             📝 {t_analysis}
#           </summary>
#           <div style="
#             padding: 16px;
#             background: white;
#             border: 1px solid #e6e6e6;
#             border-radius: 8px;
#             margin-top: 8px;
#             line-height: 1.6;">
#             {html.escape(str(analysis_text))}
#           </div>
#         </details>
#         """, unsafe_allow_html=True)

#     # Section 2: Refined Categories (each category in its own collapsible section, expanded by default)
#     categories = refinement_report.get('categories', [])
#     if categories:
#         st.markdown(f"### {t_categories}")

#         for category in categories:
#             category_name = category.get('category_name', 'Unknown')
#             subcode_count = category.get('subcode_count', 0)
#             subcodes = category.get('subcodes', [])

#             # Build subcodes HTML list
#             subcodes_html = ''.join([
#                 f'''<div style="margin-bottom: 12px; padding-left: 8px;">
#                   <b style="color: #1f77b4;">{html.escape(str(sc.get('code', 'N/A')))}</b><br>
#                   <span style="color: #666; font-size: 14px;">{html.escape(str(sc.get('description', 'N/A')))}</span>
#                 </div>'''
#                 for sc in subcodes
#             ])

#             st.markdown(f"""
#             <details open style="margin-bottom: 16px;">
#               <summary style="
#                 cursor: pointer;
#                 padding: 12px 16px;
#                 background: #f0f2f6;
#                 border-radius: 8px;
#                 font-weight: 600;
#                 margin-bottom: 8px;
#                 user-select: none;">
#                 📂 {html.escape(str(category_name))} ({subcode_count} {t_subcodes})
#               </summary>
#               <div style="
#                 padding: 16px;
#                 background: white;
#                 border: 1px solid #e6e6e6;
#                 border-radius: 8px;
#                 margin-top: 8px;
#                 line-height: 1.6;">
#                 {subcodes_html}
#               </div>
#             </details>
#             """, unsafe_allow_html=True)
#     else:
#         st.write("No categories available")

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
    """Generate enhanced variable key for cache operations"""
    selected_variables = st.session_state.get('selected_variables_config', [st.session_state.selected_variable])
    is_merged = st.session_state.get('is_merged_variable', False)
    sample_size = st.session_state.get('sample_size_config')
    merge_config = st.session_state.get('merge_config')
    return generate_enhanced_variable_key(
        selected_variables,
        is_merged=is_merged,
        sample_size=sample_size,
        merge_config=merge_config
    )

def show_step8_refined_codebook():
    """Display refined codebook structure - exact pipeline pattern"""
    #import random
    
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
        code_assigned_results = cache_manager.load_from_cache(filename, "code_assignment_direct", variable_key, models.CodeAssignedModel)

        # Load theme enriched codebook
        theme_enriched_codebook_results = cache_manager.load_from_cache(filename, "codebook_refinement_enriched", variable_key, models.ThemeEnrichedCodebookModel)
        theme_enriched_codebook = theme_enriched_codebook_results[0] if theme_enriched_codebook_results else None
        
        if code_assigned_results:
            st.write("📊 **Assignment Statistics:**")

            # Calculate frequencies directly (matching PipelineSummarizer logic)
            code_frequency = {}
            theme_frequency = {}
            total_ideas = 0

            for resp in code_assigned_results:
                if resp.response_ideas:
                    for idea in resp.response_ideas:
                        total_ideas += 1
                        if idea and idea.assigned_codes:
                            for code in idea.assigned_codes:
                                code_frequency[code] = code_frequency.get(code, 0) + 1
                        if idea and idea.assigned_themes:
                            for theme in idea.assigned_themes:
                                theme_frequency[theme] = theme_frequency.get(theme, 0) + 1

            # Display theme counts
            st.write("")
            st.write(f"**📋 THEME COUNTS** (Total: {len(theme_frequency)} themes assigned to {sum(theme_frequency.values())} ideas)")
            sorted_themes = sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)
            for i, (theme, count) in enumerate(sorted_themes, 1):
                pct = (count / total_ideas * 100) if total_ideas > 0 else 0
                st.write(f"{i}. {theme}: **{count}** ideas ({pct:.1f}%)")

            # Display code counts
            st.write("")
            st.write(f"**🏷️ CODE COUNTS** (Total: {len(code_frequency)} codes assigned to {sum(code_frequency.values())} ideas)")
            sorted_codes = sorted(code_frequency.items(), key=lambda x: x[1], reverse=True)
            for i, (code, count) in enumerate(sorted_codes, 1):
                pct = (count / total_ideas * 100) if total_ideas > 0 else 0
                st.write(f"{i}. {code}: **{count}** ideas ({pct:.1f}%)")

            # Export button at the end
            st.markdown("---")
            if st.button("🎉 " + (
                "Exporteer naar Excel" if st.session_state.language == "nl"
                else "Export to Excel"
            ), type="primary", use_container_width=True, key="export_button"):
                try:
                    import pipeline

                    # Run pipeline step 9 export
                    excel_path = pipeline.step_9_export_results(
                        code_assigned_results=code_assigned_results,
                        theme_enriched_codebook=theme_enriched_codebook,
                        filename=st.session_state.filename,
                        var_name=st.session_state.get('var_name', st.session_state.get('selected_variable', 'unknown')),
                        verbose=True
                    )

                    st.balloons()  # Celebration!
                    st.success(f"✅ " + (f"Geëxporteerd naar: {excel_path}" if st.session_state.language == "nl" else f"Exported to: {excel_path}"))
                    mark_step_completed(9)

                except Exception as e:
                    st.error(f"❌ " + (f"Export fout: {str(e)}" if st.session_state.language == "nl" else f"Export error: {str(e)}"))

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

    # Initialize language variable at the start
    lang = st.session_state.language

    # Check if we have the required info to load from cache
    if not st.session_state.get('filename') or not st.session_state.get('selected_variables_config'):
        st.write("❌ No filename or variable selected - cannot load data")
        return

    # Session-based filtering: Check both force_recalculate_all and force_recalculate_from_step
    if st.session_state.get('force_recalculate_all', False):
        # Upload from file route - only show if step was completed in current session
        # step_number maps directly to completion tracking (preprocessing=1, quality_filter=2, etc.)
        if not is_step_completed(step_number):
            lang = st.session_state.language
            st.write("⏳ " + ("Data nog niet verwerkt in huidige sessie - voer eerst verwerking uit" if lang == "nl" else "Data not yet processed in current session - run processing first"))
            return

    # Also check if this step was invalidated by force_recalculate_from_step (cache route)
    force_recalc_from = st.session_state.get('force_recalculate_from_step', 99)
    if force_recalc_from <= step_number:
        # Step has been invalidated - only show if reprocessed in current session
        if not is_step_completed(step_number):
            lang = st.session_state.language
            st.write("⏳ " + ("Stap geïnvalideerd - verwerk opnieuw" if lang == "nl" else "Step invalidated - reprocess required"))
            return

    # CRITICAL: For cache route, also check if step has stale data (beyond max_step)
    # Old cache data may exist in cache files but is invalidated because max_step indicates only steps 0-N are valid.
    # When a user re-processes a particular step, all following steps are invalidated and their cached data
    # becomes stale - it should not be used for further analysis or display.
    loaded_from_cache = st.session_state.get('loaded_from_cache', False)
    if loaded_from_cache and not is_step_completed(step_number):
        lang = st.session_state.language
        st.write("⏳ " + ("Stap nog niet uitgevoerd - verwerk eerst" if lang == "nl" else "Step not yet processed - run processing first"))
        return

    # Get cache manager
    cache_manager = _get_cache_manager()
    filename = st.session_state.filename

    # Generate enhanced variable key for cache lookup
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

    # Load data from cache based on step
    try:
        if step_number == 1:
            # Step 1: Preprocessed data (after preprocessing completion)
            data = cache_manager.load_from_cache(filename, "preprocessed", variable_key, models.PreprocessedModel)

            if data:
                # Count valid responses (not quality filtered)
                valid_responses = sum(1 for item in data if not getattr(item, 'quality_filter', False))
                st.session_state.step2_sample_size = valid_responses
                show_preprocessed_samples(data)
   
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"{'🔄 Ga naar volgende stap' if st.session_state.language == 'nl' else '🔄 Continue to Next Step'}", type="primary", use_container_width=True, key="preprocessing_continue_normal"):
                        # Advance to next step
                        st.session_state.step = 2
                        st.rerun() 
            else:
                st.write("⏳ No preprocessed data in cache - run preprocessing first")

        elif step_number == 2:
            # Step 2: Quality filtered data
            data = cache_manager.load_from_cache(filename, "quality_filter", variable_key, models.QualityFilteredModel)
            if data:
                # Count valid responses (not quality filtered)
                valid_responses = sum(1 for item in data if not getattr(item, 'quality_filter', False))
                st.session_state.step3_sample_size = valid_responses
                show_filtered_samples(data)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"{'🔄 Ga naar volgende stap' if st.session_state.language == 'nl' else '🔄 Continue to Next Step'}", type="primary", use_container_width=True, key="filtering_continue"):
                        # Advance to next step
                        st.session_state.step = 3
                        st.rerun()
                
            else:
                st.write("⏳ No quality filtered data in cache - run quality filtering first")
                
        elif step_number == 3:
            # Step 3: Extracted ideas
            data = cache_manager.load_from_cache(filename, "extracted_ideas", variable_key, models.IdeasExtractedModel)
            if data:
                # Count total ideas across all responses and lock in as sample size for remaining steps
                total_ideas = sum(item.idea_count for item in data)
                st.session_state.step4_sample_size = total_ideas  # Track idea count for step 4 onwards
                show_idea_samples(data)
         
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"{'🔄 Ga naar volgende stap' if st.session_state.language == 'nl' else '🔄 Continue to Next Step'}", type="primary", use_container_width=True, key="idea_extraction_continue_normal"):
                        # Advance to next step
                        st.session_state.step = 4
                        st.rerun()
      
            else:
                st.write("⏳ No extracted ideas in cache - run idea extraction first")
                
        elif step_number == 4:
            # Step 4: Embeddings
            data = cache_manager.load_from_cache(filename, "embeddings", variable_key, models.EmbeddingsModel)
            if data:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 " + ("Ga door naar volgende stap" if lang == "nl" else "Continue to Next Step"), type="primary", use_container_width=True, key="embedding_continue"):
                        st.session_state.step = 5
                        st.rerun()
                
            else:
                st.write("⏳ No embeddings in cache - run embedding generation first")
                
        
        elif step_number == 5:
            # Step 5: Clusters
            data = cache_manager.load_from_cache(filename, "initial_clusters", variable_key, models.ClusterModel)
            if data:
                show_cluster_samples(data)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 " + ("Ga door naar volgende stap" if lang == "nl" else "Continue to Next Step"), type="primary", use_container_width=True, key="cluster_continue"):
                        st.session_state.step = 6
                        st.rerun()
            else:
                st.write("⏳ No clusters in cache - run clustering first")
                
        elif step_number == 6:
            # Step 6: Codebook reasoning
            try:
                from utils.codeGenerator import CodeGeneratorReasoningResults
                data = cache_manager.load_from_cache(filename, "codebook_generation_reasoning", variable_key, CodeGeneratorReasoningResults)
                
                if data and len(data) > 0:
                    show_codebook_samples(data[0])

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔄 " + ("Ga door naar volgende stap" if lang == "nl" else "Continue to Next Step"), type="primary", use_container_width=True, key="codeGen_continue"):
                            st.session_state.step = 7
                            st.rerun()
                else:
                    st.write("⏳ No codebook reasoning in cache - run codebook generation first")
            except Exception as e:
                st.write(f"⚠️ Error loading codebook reasoning: {e}")
                
        elif step_number == 7:
            # Step 7: Codebook Refinement
            from utils.codebookRefinement import get_refinement_report

            data = cache_manager.load_from_cache(filename, "codebook_refinement", variable_key, models.CodeRefinementResults)
            if data and len(data) > 0:
                # Convert CodeRefinementResults to structured dict for display
                refinement_report = get_refinement_report(data[0])
                show_theme_samples(refinement_report)

                # Continue button (pattern from other steps)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 " + ("Ga door naar volgende stap" if st.session_state.language == 'nl' else "Continue to Next Step"),
                                type="primary", use_container_width=True, key="theme_continue"):
                        st.session_state.step = 8
                        st.rerun()
            else:
                st.write("⏳ No refinement results in cache - run codebook refinement first")
                
        elif step_number == 8:
            # Step 8: Code Assignment Results
            show_step9_assignment_stats()  # Reuse step 9's assignment display
                
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

    # Show samples for current step
    sampling_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    if st.session_state.step in sampling_steps:
        # Always display for current step - step-specific data
        show_step_samples(st.session_state.step)

    if st.session_state.step in sampling_steps:
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