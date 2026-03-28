#%%
import sys

# Suppress SystemExit in IPython/Jupyter environments
if 'IPython' in sys.modules:
    import IPython
    _original_showtraceback = IPython.core.interactiveshell.InteractiveShell.showtraceback
    def _custom_showtraceback(self, *args, **kwargs):
        etype, value, tb = sys.exc_info()
        if etype == SystemExit and str(value) == '0':
            return  # Suppress clean exits
        _original_showtraceback(self, *args, **kwargs)
    IPython.core.interactiveshell.InteractiveShell.showtraceback = _custom_showtraceback

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import html, random
import re

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "utils"))

import models
from config import CacheConfig, ModelConfig, get_step_model
from config_steps.config_classifier import CategoriesConfig
from config_steps.config_codeGenerator import CodebookConfig
from config_steps.config_codeAssigner import AssignmentConfig
from config_steps.config_preProcessor import SpellCheckConfig
from config_steps.config_qualityFilter import QualityFilterConfig
from config_steps.config_ideaExtractor import SegmentationConfig

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
        data_dir = str(project_root / "data")
        st.session_state.data_loader = DataLoader(data_dir=data_dir, verbose=False)
    return st.session_state.data_loader

def _get_cache_manager():
    if st.session_state.cache_manager is None:
        st.session_state.cache_manager = CacheManager(CacheConfig())
    return st.session_state.cache_manager

def _load_or_recover(filename: str, step_name: str, variable_key: str, model_cls):
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

def _get_verbose_capture():
    """Get or create a VerboseCapture instance for the current session."""
    from utils.saveVerbose import VerboseCapture

    # Check if we have the required session state
    if not st.session_state.get('filename'):
        return None

    # Get current step from session state
    current_step = st.session_state.get('step', 0)

    # Build variable_key using the same logic as step pages (for cache consistency)
    selected_variables = st.session_state.get('selected_variables_config', [st.session_state.get('selected_variable', 'unknown')])
    is_merged = st.session_state.get('is_merged_variable', False)
    sample_size = st.session_state.get('sample_size_config')
    merge_config = st.session_state.get('merge_config')

    variable_key = generate_enhanced_variable_key(
        selected_variables,
        is_merged=is_merged,
        sample_size=sample_size,
        merge_config=merge_config
    )

    return VerboseCapture(
        filename=st.session_state.filename,
        variable_key=variable_key,
        sample_size=sample_size,
        run_until_step=current_step,
        append_mode=True  # Append for each step in Streamlit
    )

def _run_with_verbose_capture(step_func, *args, **kwargs):
    """Run a pipeline step function with verbose output capture."""
    capture = _get_verbose_capture()
    if capture:
        capture.__enter__()
        try:
            result = step_func(*args, **kwargs)
            return result
        finally:
            capture.__exit__(None, None, None)
    else:
        # No capture available, run directly
        return step_func(*args, **kwargs)

def show_verbose_log_expander(step: int):
    """
    Display verbose log for a completed step in a collapsible expander.

    Only shows the log if:
    - The step is completed (loaded from cache or just ran)
    - A matching log file exists
    - The cache is not invalidated (force_recalculate_from_step > step)

    Args:
        step: The pipeline step number (1-9)
    """
    from utils.saveVerbose import VerboseCapture

    # Check if we should show the log (cache not invalidated for this step)
    force_recalc_from = st.session_state.get('force_recalculate_from_step', 99)
    if force_recalc_from <= step:
        return  # Cache is invalidated, don't show old log

    # Get parameters needed to find the log file
    if not st.session_state.get('filename'):
        return

    selected_variables = st.session_state.get('selected_variables_config', [st.session_state.get('selected_variable', 'unknown')])
    is_merged = st.session_state.get('is_merged_variable', False)
    sample_size = st.session_state.get('sample_size_config')
    merge_config = st.session_state.get('merge_config')

    variable_key = generate_enhanced_variable_key(
        selected_variables,
        is_merged=is_merged,
        sample_size=sample_size,
        merge_config=merge_config
    )

    # Find the most recent matching log file
    log_path = VerboseCapture.find_latest_log(
        filename=st.session_state.filename,
        variable_key=variable_key,
        step=step
    )

    if log_path is None:
        return  # No log file found

    # Load and display the log
    log_content = VerboseCapture.load_log_content(log_path)
    if log_content is None:
        return  # Failed to load log

    lang = st.session_state.get('language', 'en')
    expander_label = f"📋 {'Uitvoeringslog' if lang == 'nl' else 'Execution Log'} (Step {step})"

    with st.expander(expander_label, expanded=False):
        st.code(log_content, language=None)

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
if 'spellcheck_config' not in st.session_state:
    st.session_state.spellcheck_config = SpellCheckConfig()
if 'quality_filter_config' not in st.session_state:
    st.session_state.quality_filter_config = QualityFilterConfig()
if 'segmentation_config' not in st.session_state:
    st.session_state.segmentation_config = SegmentationConfig()
if 'categories_config' not in st.session_state:
    st.session_state.categories_config = CategoriesConfig()
if 'codebook_config' not in st.session_state:
    st.session_state.codebook_config = CodebookConfig()
if 'assignment_config' not in st.session_state:
    st.session_state.assignment_config = AssignmentConfig()

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

def get_display_sample_size(lang: str = "en") -> str:
    """Get sample size for display - preserves 'full' semantic while showing actual count

    Returns display string like '579 (Full)' when sample_size_config is None,
    or just the number when user chose a specific sample size.
    """
    size = st.session_state.get('sample_size_config')
    if size is None:
        # User chose 'full sample' - show actual count with 'Full' label
        if 'raw_text_list' in st.session_state.pipeline_results:
            count = len(st.session_state.pipeline_results['raw_text_list'])
            full_label = "Volledig" if lang == "nl" else "Full"
            return f"{count} ({full_label})"
        return "Volledig" if lang == "nl" else "Full"
    return str(size)

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
        3: ['extracted_ideas'],
        4: ['taxonomy_metadata'],
        5: ['codebook_metadata'],
        6: ['code_assigned_results'],
        7: ['excel_path']
    }

    all_cleared_keys = []
    for step_num in range(start_step, 8):
        keys_to_remove = step_result_keys.get(step_num, [])
        for key in keys_to_remove:
            if key in st.session_state.pipeline_results:
                del st.session_state.pipeline_results[key]
                all_cleared_keys.append(f"{key}(step{step_num})")

    # 5. Invalidate cache entries in database
    cache_manager = _get_cache_manager()
    step_mapping = {
        0: "data", 1: "preprocessed", 2: "quality_filter",
        3: "extracted_ideas", 4: "taxonomy", 5: "codebook",
        6: "code_assignment", 7: "export"
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
        for step_num in range(start_step, 8):
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
        gpt5_models = ["gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-5-chat", "gpt-5-chat-latest"]
        all_models = gpt4_models + gpt5_models
        embedding_models = ["text-embedding-3-large", "text-embedding-3-small", "gemini-embedding-001"]

        # Step 1: Preprocessing
        if current_step == 1:
            st.markdown("#### 📝 Step 2: Preprocessing")
            current_spell_model = st.session_state.spellcheck_config.model
            spell_model = st.selectbox(
                "Spell Check Model",
                options=all_models,
                index=all_models.index(current_spell_model) if current_spell_model in all_models else 0,
                key="spell_check_model"
            )
            if spell_model != current_spell_model:
                st.session_state.spellcheck_config.model = spell_model

            st.markdown("---")

        # Step 2: Quality Filter
        if current_step == 2:
            st.markdown("#### 🔍 Step 3: Quality Filter")
            current_quality_model = st.session_state.quality_filter_config.model
            quality_model = st.selectbox(
                "Quality Filter Model",
                options=all_models,
                index=all_models.index(current_quality_model) if current_quality_model in all_models else 0,
                key="quality_filter_model"
            )
            if quality_model != current_quality_model:
                st.session_state.quality_filter_config.model = quality_model

            st.markdown("---")

        # Step 3: Idea Extraction
        if current_step == 3:
            st.markdown("#### 💡 Step 4: Idea Extraction")
            current_seg_model = st.session_state.segmentation_config.model
            seg_model = st.selectbox(
                "Segmentation Model",
                options=all_models,
                index=all_models.index(current_seg_model) if current_seg_model in all_models else 0,
                key="segmentation_model"
            )
            if seg_model != current_seg_model:
                st.session_state.segmentation_config.model = seg_model

            st.markdown("---")

        # Step 4: Embeddings
        if current_step == 4:
            st.markdown("#### 🔗 Step 5: Embeddings")
            current_emb_model = ModelConfig().embedding_model
            emb_model = st.selectbox(
                "Embedding Model",
                options=embedding_models,
                index=embedding_models.index(current_emb_model) if current_emb_model in embedding_models else 0,
                key="embedding_model"
            )

            st.markdown("---")

        # Step 5: Category Discovery
        if current_step == 5:
            st.markdown("#### 📊 Step 6: Category Discovery")
            st.markdown("*Automatic MECE category discovery from idea partitions*")
            st.info("Category discovery partitions ideas by concept type, discovers MECE categories per partition using MAP/REDUCE, and assigns each idea to exactly one category.")

            st.markdown("---")

        # Step 6: Code Assignment
        if current_step == 6:
            st.markdown("#### 🎯 Step 6: Code Assignment")
            current_assign_model = st.session_state.assignment_config.assignment_model
            assign_model = st.selectbox(
                "Code Assignment Model",
                options=all_models,
                index=all_models.index(current_assign_model) if current_assign_model in all_models else 0,
                key="code_assignment_model"
            )
            if assign_model != current_assign_model:
                st.session_state.assignment_config.assignment_model = assign_model

            top_n = st.number_input(
                "Embedding Top N",
                min_value=1,
                max_value=20,
                value=st.session_state.assignment_config.embedding_top_n,
                help="Number of top codes pre-filtered via embedding similarity",
                key="embedding_top_n"
            )
            if top_n != st.session_state.assignment_config.embedding_top_n:
                st.session_state.assignment_config.embedding_top_n = top_n

            st.markdown("---")

        # Reset to defaults button (always shown)
        if st.button("🔄 Reset All to Defaults", type="secondary"):
            st.session_state.spellcheck_config = SpellCheckConfig()
            st.session_state.quality_filter_config = QualityFilterConfig()
            st.session_state.segmentation_config = SegmentationConfig()
            st.session_state.categories_config = CategoriesConfig()
            st.session_state.codebook_config = CodebookConfig()
            st.session_state.assignment_config = AssignmentConfig()
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

        st.header(ui.get_text("SIDEBAR_HEADER", st.session_state.language))
        st.markdown(ui.get_text("SIDEBAR_DESCRIPTION", st.session_state.language))

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
                can_go_forward = st.session_state.step < 8 and (loaded_from_cache or st.session_state.step in st.session_state.completed_steps)
                if st.button(("Volgende" if st.session_state.language == "nl" else "Next") + " ➡️", disabled=not can_go_forward, use_container_width=True, key="nav_next"):
                    clear_all_wait_states()  # Clear wait states before navigation
                    st.session_state.step += 1
                    st.rerun()

            # Interactive step navigator
            lang = st.session_state.language
            loaded_from_cache = st.session_state.get('loaded_from_cache', False)
            with st.expander("📋 " + ("Alle stappen" if lang == "nl" else "All steps"), expanded=True):

                for step in range(1, 9):  # Start from 1, skip step 0 (Upload Data)
                    step_name = get_step_name(step, lang)

                    # Determine if step is accessible (using OLD correct logic from All steps display)
                    is_current = (step == st.session_state.step)
                    is_accessible = (step in st.session_state.completed_steps or
                                    (loaded_from_cache and step <= st.session_state.max_step_reached))

                    # Format step label with status icon
                    # Only show checkmark for accessible steps, no icon otherwise
                    if is_accessible:
                        icon = "✅"  # Accessible steps (completed or cached) get checkmark
                    else:
                        icon = ""   # All other steps get no icon

                    prefix = "Stap" if lang == "nl" else "Step"
                    label = f"{icon} {prefix} {step}: {step_name}".strip()

                    # Enable button if accessible OR if it's the current step (aesthetic)
                    is_enabled = is_accessible or is_current

                    if st.button(
                        label,
                        key=f"nav_step_{step}",
                        use_container_width=True,
                        disabled=not is_enabled,
                        type="secondary"
                    ):
                        clear_all_wait_states()
                        st.session_state.step = step
                        st.rerun()

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
    sampling_steps = [1, 2, 3, 4, 5, 6, 7, 8]

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
            show_taxonomy_page()
        elif st.session_state.step == 5:
            show_codebook_generation_page()
        elif st.session_state.step == 6:
            show_code_assignment_page()
        elif st.session_state.step == 7:
            show_export_page()
        elif st.session_state.step == 8:
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
        "taxonomy": 4,
        "codebook": 5,
        "code_assignment": 6,
        "export": 7
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
        # Keep sample_size as None when suffix is "_full" - preserves "full sample" semantic

        id_column = 'id'
        if data and hasattr(data[0], 'id_column') and data[0].id_column:
            id_column = data[0].id_column

        # Get var_lab - prioritize cached var_lab (preserves user edits)
        var_lab = variables  # Default
        cache_info = cache_manager.db.get_cache_info(filename, "preprocessed", variable_key)

        if cache_info and cache_info.get('var_lab'):
            # Use cached var_lab (preserves user edits)
            var_lab = cache_info['var_lab']
        else:
            # Fallback: Fetch from SPSS (for old cache entries without var_lab)
            try:
                data_loader = _get_data_loader()
                first_var = variables.split('+')[0] if '+' in variables else variables
                var_lab = data_loader.get_varlab(filename, first_var)
                last_bracket = var_lab.rfind("]")
                var_lab = var_lab[last_bracket + 1:].strip()
            except :
                pass
        
        # Build config
        config = DatasetConfig(
            filename=filename,
            id_column=id_column,
            selected_variables=parsed_vars,
            variable_mode=variable_mode,
            sample_size=sample_size,
            merge_config=None,  # Merge config not stored in cache metadata
            encoding=None,
            var_lab=var_lab,
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
        # Clean up var_lab by removing SPSS metadata before last bracket
        last_bracket = var_lab.rfind("]")
        var_lab = var_lab[last_bracket + 1:].strip()
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


def convert_response_models_to_preview_df(response_models: list, id_column: str, text_column: str) -> pd.DataFrame:

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
                        data_dir = str(project_root / "data")
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
                                verbose=True,
                                data_dir=data_dir
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
                                verbose=True,
                                data_dir=data_dir
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
        sample_info += (f"\n\n**Data:** {get_display_sample_size(lang)} antwoorden" if lang == "nl" else f"\n\n**Data:** {get_display_sample_size(lang)} responses")
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

    # ==================== VERBOSE LOG EXPANDER ====================
    if is_step_completed(1):
        show_verbose_log_expander(1)

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
                                verbose=True,
                                data_dir=str(project_root / "data"))
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
                                verbose=True,
                                data_dir=str(project_root / "data"))
                        progress_container.success("✅ Data laden voltooid")

                    st.session_state.pipeline_results['raw_text_list'] = raw_text_list
                    st.session_state.pipeline_results['var_lab'] = var_lab
            else:
                 # Data already loaded from preview - update var_lab in case user edited it
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

                preprocessed_text, preprocessing_stats = _run_with_verbose_capture(
                        pipeline.step_1_preprocess,
                        raw_text_list=st.session_state.pipeline_results['raw_text_list'],
                        filename=st.session_state.filename,
                        var_lab=st.session_state.pipeline_results['var_lab'],
                        variable_key=variable_key,
                        cache_manager=_get_cache_manager(),

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

    # Helper function to calculate filter statistics
    def calculate_quality_filter_stats(quality_filtered_text):
        """Calculate statistics from quality filtered data"""
        code_meanings = {
            99999997: "Don't know (expresses uncertainty)",
            99999998: "No response (empty/NA)",
            99999999: "Meaningless answer (gibberish/irrelevant text)"
        }

        code_counts = {}
        for item in quality_filtered_text:
            if item.quality_filter and item.quality_filter_code is not None:
                code = item.quality_filter_code
                code_counts[code] = code_counts.get(code, 0) + 1

        total_filtered = sum(code_counts.values())
        total_validated = len(quality_filtered_text) - total_filtered

        return {
            'code_counts': code_counts,
            'code_meanings': code_meanings,
            'total_filtered': total_filtered,
            'total_validated': total_validated
        }

    st.header("Stap 2: Kwaliteitsfiltering" if lang == "nl" else "Step 2: Quality Filtering")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(2):
        st.success("✅ " + ("Kwaliteitsfiltering voltooid! Bekijk de resultaten en klik dan op doorgaan." if lang == "nl" else "Quality filtering completed! Review the results on the right, then click continue."))

    # ==================== BLOCK 1.5: LOAD DATA IF FROM CACHE ====================
    # Only load and calculate stats when step 2 is completed BUT quality_filter_stats doesn't exist
    # This means we're in cache route, not fresh processing route
    if is_step_completed(2) and not st.session_state.get('quality_filter_stats'):
        if 'quality_filtered_text' not in st.session_state.pipeline_results:
            # Load quality filtered data from cache
            cache_manager = _get_cache_manager()
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

            quality_filtered_text = cache_manager.load_from_cache(
                st.session_state.filename,
                "quality_filter",
                variable_key,
                models.QualityFilteredModel
            )

            if quality_filtered_text:
                st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text

        # Calculate stats from cached data (for blue box display)
        if 'quality_filtered_text' in st.session_state.pipeline_results:
            # Create temporary stats object (NOT stored in quality_filter_stats to avoid yellow box)
            quality_filtered_text = st.session_state.pipeline_results['quality_filtered_text']
            st.session_state['_cache_filter_stats'] = calculate_quality_filter_stats(quality_filtered_text)

    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(1):
        sample_info =  (f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n")
        sample_info += (f"\n\n**Data:** {get_display_sample_size(lang)} " +
                       ("antwoorden" if lang == "nl" else "responses"))

        # If step 2 is completed AND we're in cache route (not fresh processing), add filter breakdown
        if is_step_completed(2) and not st.session_state.get('quality_filter_stats'):
            stats = st.session_state.get('_cache_filter_stats', {})
            if stats:
                valid_count = stats.get('total_validated', 0)
                sample_info += (f"\n\n**{'Gevalideerde antwoorden' if lang == 'nl' else 'Validated responses'}:** {valid_count}")

                # Add breakdown of filtered items (same format as yellow box)
                code_counts = stats.get('code_counts', {})
                code_meanings = stats.get('code_meanings', {})
                if code_counts:
                    total_filtered = stats.get('total_filtered', 0)
                    sample_info += (f"\n\n**{'Uitgesloten van verdere analyse' if lang == 'nl' else 'Excluded from further analysis'}:** {total_filtered}")
                    for code in sorted(code_counts.keys()):
                        count = code_counts[code]
                        meaning = code_meanings.get(code, 'Unknown')
                        sample_info += f"\n- Code {code}: {count} item(s) - {meaning}"

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

    # ==================== VERBOSE LOG EXPANDER ====================
    if is_step_completed(2):
        show_verbose_log_expander(2)

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

                    # Check if cache load was successful
                    if preprocessed_text is not None:
                        st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text

                        # Populate var_lab if not already in pipeline_results
                        if 'var_lab' not in st.session_state.pipeline_results:
                            # Try to get from cache metadata first
                            cache_info = cache_manager.db.get_cache_info(st.session_state.filename, "preprocessed", variable_key)
                            if cache_info and cache_info.get('var_lab'):
                                st.session_state.pipeline_results['var_lab'] = cache_info['var_lab']
                            else:
                                # Fallback to session state
                                st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')
                        progress_container.success("✅ Data geladen uit cache" if lang == "nl" else "✅ Data loaded from cache")

                # If cache was invalid or corrupted, fall through to reprocessing
                if 'preprocessed_text' not in st.session_state.pipeline_results:
                    # Upload route: process from raw_text_list
                    progress_container.text("🔄 Voorbewerkte data verwerken..." if lang == "nl" else "🔄 Processing preprocessed data...")
                    preprocessed_text, _ = pipeline.step_1_preprocess(
                        raw_text_list=st.session_state.pipeline_results['raw_text_list'],
                        filename=st.session_state.filename,
                        var_lab=st.session_state.pipeline_results['var_lab'],
                        variable_key=variable_key,
                        cache_manager=cache_manager,

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

                quality_filtered_text = _run_with_verbose_capture(
                    pipeline.step_2_quality_filter,
                    preprocessed_text=st.session_state.pipeline_results['preprocessed_text'],
                    filename=st.session_state.filename,
                    var_lab=st.session_state.pipeline_results['var_lab'],
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),

                    force_recalc=force_recalc,
                    verbose=True,
                    prompt_printer_enabled=False
                )

                progress_container.success("✅ " + ("Kwaliteitsfiltering voltooid" if lang == "nl" else "Quality filtering completed"))

                # Calculate statistics from results for display
                code_counts = {}
                code_meanings = {
                    99999997: "Don't know (expresses uncertainty)",
                    99999998: "No response (empty/NA)",
                    99999999: "Meaningless answer (gibberish/irrelevant text)"
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
        # Use step3_sample_size if available, otherwise use display helper
        step3_size = st.session_state.get('step3_sample_size')
        sample_info += (f"\n\n**Data:** {step3_size if step3_size else get_display_sample_size(lang)} {'antwoorden' if lang == 'nl' else 'responses'}")
        if is_step_completed(3):
            step4_size = st.session_state.get('step4_sample_size')
            sample_info += (f" / {step4_size if step4_size else get_display_sample_size(lang)} {'deelantwoorden' if lang == 'nl' else 'answer parts'}")
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

    # ==================== VERBOSE LOG EXPANDER ====================
    if is_step_completed(3):
        show_verbose_log_expander(3)

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

                    # Check if cache load was successful
                    if quality_filtered_text is not None:
                        st.session_state.pipeline_results['quality_filtered_text'] = quality_filtered_text

                        # Populate var_lab if not already in pipeline_results
                        if 'var_lab' not in st.session_state.pipeline_results:
                            # Try to get from cache metadata first
                            cache_info = cache_manager.db.get_cache_info(st.session_state.filename, "quality_filter", variable_key)
                            if cache_info and cache_info.get('var_lab'):
                                st.session_state.pipeline_results['var_lab'] = cache_info['var_lab']
                            else:
                                # Fallback to session state
                                st.session_state.pipeline_results['var_lab'] = st.session_state.get('var_lab', '')
                        progress_container.success("✅ " + ("Data geladen uit cache" if lang == "nl" else "Data loaded from cache"))

                # If cache was invalid or corrupted, fall through to reprocessing
                if 'quality_filtered_text' not in st.session_state.pipeline_results:
                    # Upload route: process from preprocessed_text
                    progress_container.text("🔄 " + ("Gefilterde data verwerken..." if lang == "nl" else "Processing filtered data..."))
                    quality_filtered_text = pipeline.step_2_quality_filter(
                        preprocessed_text=st.session_state.pipeline_results['preprocessed_text'],
                        filename=st.session_state.filename,
                        var_lab=st.session_state.pipeline_results['var_lab'],
                        variable_key=variable_key,
                        cache_manager=cache_manager,

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
                extracted_ideas = _run_with_verbose_capture(
                    pipeline.step_3_extract_ideas,
                    quality_filtered_text=st.session_state.pipeline_results['quality_filtered_text'],
                    filename=st.session_state.filename,
                    var_lab=st.session_state.pipeline_results['var_lab'],
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),

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

def show_taxonomy_page():
    """
    Step 4: Taxonomy Classification (Taxonomie-classificatie)

    Runs domain/facet/attribute discovery on extracted ideas from step 3.

    Pipeline function: step_4_classify_taxonomy
    Cache name: taxonomy (metadata)
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 4: Taxonomie" if lang == "nl" else "Step 4: Taxonomy")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(4):
        st.success("✅ " + (
            "Taxonomie-classificatie voltooid! Bekijk de resultaten en klik dan op doorgaan."
            if lang == "nl" else
            "Taxonomy classification completed! Review the results, then click continue."
        ))

    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(3):
        sample_info = f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n"
        step3_size = st.session_state.get('step3_sample_size')
        sample_info += f"\n\n**Data:** {step3_size if step3_size else get_display_sample_size(lang)} {'ideeën' if lang == 'nl' else 'ideas'}"
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    if is_step_completed(4):
        stats = st.session_state.get('taxonomy_stats', {})
        if stats:
            summary_info = (
                f"\n\n- {'Domeinen' if lang == 'nl' else 'Domains'}: {stats.get('n_domains', 0)}"
                + f"\n\n- {'Facetten' if lang == 'nl' else 'Facets'}: {stats.get('n_facets', 0)}"
                + f"\n\n- {'Attributen' if lang == 'nl' else 'Attributes'}: {stats.get('n_attributes', 0)}"
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

    # ==================== VERBOSE LOG EXPANDER ====================
    if is_step_completed(4):
        show_verbose_log_expander(4)

    # ==================== BLOCK 4: DATA LOADING ====================
    if is_step_completed(3) and not is_step_completed(4):
        if 'extracted_ideas' not in st.session_state.pipeline_results:
            progress_container = st.empty()
            try:
                variable_key = _get_variable_key_for_cache()
                cache_manager = _get_cache_manager()
                if cache_manager.is_cache_valid(st.session_state.filename, "extracted_ideas", variable_key):
                    progress_container.text("🔄 " + (
                        "Geëxtraheerde ideeën laden uit cache..." if lang == "nl"
                        else "Loading extracted ideas from cache..."
                    ))
                    extracted_ideas = _load_or_recover(
                        st.session_state.filename,
                        "extracted_ideas",
                        variable_key,
                        models.IdeasExtractedModel
                    )
                    if extracted_ideas is not None:
                        st.session_state.pipeline_results['extracted_ideas'] = extracted_ideas
                        progress_container.empty()
            except Exception as e:
                st.error(f"Data laad fout: {str(e)}" if lang == "nl" else f"Data loading error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    if is_step_completed(3) and not is_step_completed(4):
        info_text = (
            "Deze stap classificeert de geëxtraheerde ideeën in domeinen, facetten en attributen."
            if lang == "nl" else
            "This step classifies the extracted ideas into domains, facets, and attributes."
        )
        st.markdown(info_text)

        if st.button("🚀 " + (
            "Start Taxonomie-classificatie" if lang == "nl" else "Start Taxonomy Classification"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "Taxonomie aan het classificeren..." if lang == "nl"
                    else "Classifying taxonomy..."
                ))

                variable_key = _get_variable_key_for_cache()
                force_recalc = (
                    st.session_state.get('force_recalculate_all', False) or
                    (st.session_state.get('force_recalculate_from_step', 99) <= 4)
                )

                extracted_ideas = st.session_state.pipeline_results.get('extracted_ideas', [])

                _run_with_verbose_capture(
                    pipeline.step_4_classify_taxonomy,
                    encoded_text=extracted_ideas,
                    filename=st.session_state.filename,
                    var_lab=st.session_state.pipeline_results.get('var_lab', st.session_state.var_lab),
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    force_recalc=force_recalc,
                    verbose=True,
                    prompt_printer_enabled=False
                )

                progress_container.success("✅ " + (
                    "Taxonomie-classificatie voltooid" if lang == "nl"
                    else "Taxonomy classification completed"
                ))

                # Load stats from cache for display
                from models import TaxonomyResultsCache
                cache_manager = _get_cache_manager()
                taxonomy_cache = cache_manager.load_metadata_from_cache(
                    st.session_state.filename, "taxonomy", variable_key, TaxonomyResultsCache
                )
                if taxonomy_cache:
                    n_domains = len(taxonomy_cache.partition_results)
                    n_facets = sum(len(r.facets) for r in taxonomy_cache.partition_results.values())
                    n_attributes = sum(
                        len(attrs)
                        for r in taxonomy_cache.partition_results.values()
                        for attrs in r.attributes.values()
                    )
                    st.session_state['taxonomy_stats'] = {
                        'n_domains': n_domains,
                        'n_facets': n_facets,
                        'n_attributes': n_attributes,
                    }

                mark_step_completed(4)
                st.rerun()

            except Exception as e:
                st.error(f"Taxonomie fout: {str(e)}" if lang == "nl" else f"Taxonomy error: {str(e)}")


def show_codebook_generation_page():
    """
    Step 5: Codebook Generation (Codebook Generatie)

    Generates a MECE codebook from the taxonomy produced in step 4.

    Pipeline function: step_5_generate_codebook
    Cache name: mece_codes (metadata)
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 5: Codebook generatie" if lang == "nl" else "Step 5: Codebook Generation")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(5):
        st.success("✅ " + (
            "Codebook gegenereerd! Bekijk de resultaten en klik dan op doorgaan."
            if lang == "nl" else
            "Codebook generated! Review the results, then click continue."
        ))

    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(4):
        sample_info = f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n"
        stats = st.session_state.get('taxonomy_stats', {})
        n_domains = stats.get('n_domains', 0)
        sample_info += f"\n\n**{'Taxonomie' if lang == 'nl' else 'Taxonomy'}:** {n_domains} {'domeinen' if lang == 'nl' else 'domains'}"
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    if is_step_completed(5):
        stats = st.session_state.get('codebook_stats', {})
        if stats:
            summary_info = f"\n\n- {'Codes in codebook' if lang == 'nl' else 'Codes in codebook'}: {stats.get('num_codes', 0)}"
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

    # ==================== VERBOSE LOG EXPANDER ====================
    if is_step_completed(5):
        show_verbose_log_expander(5)

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    if is_step_completed(4) and not is_step_completed(5):
        st.markdown(
            "Deze stap genereert een MECE codebook op basis van de taxonomie uit stap 4."
            if lang == "nl" else
            "This step generates a MECE codebook from the taxonomy produced in step 4."
        )

        if st.button("🚀 " + (
            "Start Codebook Generatie" if lang == "nl" else "Start Codebook Generation"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "Codebook aan het genereren..." if lang == "nl"
                    else "Generating codebook..."
                ))

                variable_key = _get_variable_key_for_cache()
                force_recalc = (
                    st.session_state.get('force_recalculate_all', False) or
                    (st.session_state.get('force_recalculate_from_step', 99) <= 5)
                )

                _run_with_verbose_capture(
                    pipeline.step_5_generate_codebook,
                    filename=st.session_state.filename,
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
                    force_recalc=force_recalc,
                    verbose=True,
                    prompt_printer_enabled=False
                )

                progress_container.success("✅ " + (
                    "Codebook generatie voltooid" if lang == "nl"
                    else "Codebook generation completed"
                ))

                # Load stats from cache for display
                from models import CodingResultsCache
                cache_manager = _get_cache_manager()
                mece_cache = cache_manager.load_metadata_from_cache(
                    st.session_state.filename, "mece_codes", variable_key, CodingResultsCache
                )
                if mece_cache:
                    st.session_state['codebook_stats'] = {
                        'num_codes': mece_cache.total_categories,
                    }

                mark_step_completed(5)
                st.rerun()

            except Exception as e:
                st.error(f"Codebook fout: {str(e)}" if lang == "nl" else f"Codebook error: {str(e)}")

def show_code_assignment_page():
    """
    Step 6: Code Assignment

    Assigns MECE codes from the codebook to individual extracted ideas.

    Pipeline function: step_6_assign_codes
    Cache name: taxonomy_codes
    Model: models.CodeAssignedModel
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 6: Code toewijzing" if lang == "nl" else "Step 6: Code Assignment")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(6):
        st.success("✅ " + (
            "Code toewijzing voltooid! Bekijk de resultaten en klik dan op doorgaan."
            if lang == "nl" else
            "Code assignment completed! Review the results, then click continue."
        ))

    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(5):
        sample_info = f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n"
        codebook_stats = st.session_state.get('codebook_stats', {})
        num_codes = codebook_stats.get('num_codes', 0)
        sample_info += f"\n\n**{'Codebook' if lang == 'nl' else 'Codebook'}:** {num_codes} {'codes' if lang == 'nl' else 'codes'}"
        st.info(sample_info)

    # ==================== BLOCK 3: YELLOW BOX ====================
    if is_step_completed(6):
        stats = st.session_state.get('code_assignment_stats', {})
        if stats:
            summary_info = (
                f"\n\n- {'Antwoorden verwerkt' if lang == 'nl' else 'Responses processed'}: {stats.get('total_responses', 0)}"
                + f"\n\n- {'Ideeën verwerkt' if lang == 'nl' else 'Ideas processed'}: {stats.get('total_ideas', 0)}"
                + f"\n\n- {'Ideeën toegewezen' if lang == 'nl' else 'Ideas assigned'}: {stats.get('assigned_count', 0)}"
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

    # ==================== VERBOSE LOG EXPANDER ====================
    if is_step_completed(6):
        show_verbose_log_expander(6)

    # ==================== BLOCK 4: DATA LOADING ====================
    if is_step_completed(5) and not is_step_completed(6):
        if 'extracted_ideas' not in st.session_state.pipeline_results:
            progress_container = st.empty()
            try:
                variable_key = _get_variable_key_for_cache()
                cache_manager = _get_cache_manager()
                if cache_manager.is_cache_valid(st.session_state.filename, "extracted_ideas", variable_key):
                    progress_container.text("🔄 " + (
                        "Geëxtraheerde ideeën laden uit cache..." if lang == "nl"
                        else "Loading extracted ideas from cache..."
                    ))
                    extracted_ideas = _load_or_recover(
                        st.session_state.filename,
                        "extracted_ideas",
                        variable_key,
                        models.IdeasExtractedModel
                    )
                    if extracted_ideas is not None:
                        st.session_state.pipeline_results['extracted_ideas'] = extracted_ideas
                        progress_container.empty()
            except Exception as e:
                st.error(f"Data laad fout: {str(e)}" if lang == "nl" else f"Data loading error: {str(e)}")

    # ==================== BLOCK 5: PROCESSING BUTTON ====================
    if is_step_completed(5) and not is_step_completed(6):
        st.markdown(
            "Deze stap wijst MECE codes toe aan individuele geëxtraheerde ideeën."
            if lang == "nl" else
            "This step assigns MECE codes to each individual extracted idea."
        )

        if st.button("🚀 " + (
            "Start Code Toewijzing" if lang == "nl" else "Start Code Assignment"
        ), type="primary"):
            progress_container = st.empty()
            try:
                progress_container.text("🔄 " + (
                    "Codes aan het toewijzen..." if lang == "nl"
                    else "Assigning codes..."
                ))

                variable_key = _get_variable_key_for_cache()
                force_recalc = (
                    st.session_state.get('force_recalculate_all', False) or
                    (st.session_state.get('force_recalculate_from_step', 99) <= 6)
                )

                extracted_ideas = st.session_state.pipeline_results.get('extracted_ideas', [])

                code_assigned_results = _run_with_verbose_capture(
                    pipeline.step_6_assign_codes,
                    encoded_text=extracted_ideas,
                    filename=st.session_state.filename,
                    variable_key=variable_key,
                    cache_manager=_get_cache_manager(),
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

                # Calculate stats for display
                total_responses = len(code_assigned_results) if code_assigned_results else 0
                total_ideas = sum(len(r.response_ideas or []) for r in (code_assigned_results or []))
                assigned_count = sum(
                    1 for r in (code_assigned_results or [])
                    for idea in (r.response_ideas or [])
                    if idea.assigned_code
                )
                st.session_state['code_assignment_stats'] = {
                    'total_responses': total_responses,
                    'total_ideas': total_ideas,
                    'assigned_count': assigned_count,
                }

                mark_step_completed(6)
                st.rerun()

            except Exception as e:
                st.error(f"Toewijzing fout: {str(e)}" if lang == "nl" else f"Assignment error: {str(e)}")

def show_export_page():
    """
    Step 7: Export (placeholder)

    Export will be implemented in a later phase.
    """
    lang = st.session_state.language

    # ==================== HEADER ====================
    st.header("Stap 7: Exporteren" if lang == "nl" else "Step 7: Export")

    # ==================== BLOCK 1: GREEN BOX ====================
    if is_step_completed(7):
        st.success("✅ " + (
            "Export voltooid!"
            if lang == "nl" else
            "Export completed!"
        ))

    # ==================== BLOCK 2: BLUE BOX ====================
    if is_step_completed(6):
        stats = st.session_state.get('code_assignment_stats', {})
        total_responses = stats.get('total_responses', 0)
        total_ideas = stats.get('total_ideas', 0)
        assigned_count = stats.get('assigned_count', 0)
        sample_info = f"**{'Vraag' if lang == 'nl' else 'Question'}:** {st.session_state.var_lab}\n\n"
        sample_info += (
            f"\n\n**{'Resultaten' if lang == 'nl' else 'Results'}:** "
            f"{total_responses} {'antwoorden' if lang == 'nl' else 'responses'}, "
            f"{assigned_count}/{total_ideas} {'ideeën toegewezen' if lang == 'nl' else 'ideas assigned'}"
        )
        st.info(sample_info)

    # ==================== NOT YET IMPLEMENTED ====================
    st.info(
        "Export wordt binnenkort geïmplementeerd."
        if lang == "nl" else
        "Export will be implemented soon."
    )


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
                verbose=False,
                data_dir=str(project_root / "data")
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
                verbose=False,
                data_dir=str(project_root / "data")
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
    valid_items = [item for item in preprocessed_text if not getattr(item, 'quality_filter', False) and item.response and item.response.strip()]

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

    # Filter out empty strings (code 99999998) from display while keeping them in statistics
    filtered_text = [item for item in quality_filtered_text if item.quality_filter and item.quality_filter_code == 99999999]

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

        if st.button(f"{'🎲 Toon nieuwe selectie' if st.session_state.language == 'nl' else '🎲 Draw new sample'}", key="filtered_samples"):
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


def show_category_samples(category_results):
    """Show category samples with HTML rendering and back/forward navigation."""

    # i18n helpers
    NL = st.session_state.get("language", "en") == "nl"
    t_no_data = "Geen data beschikbaar" if NL else "No data available"
    t_no_categories = "Geen categorieën gevonden" if NL else "No categories found"
    t_prev = "⬅️ Vorige" if NL else "⬅️ Previous"
    t_next = "➡️ Volgende" if NL else "➡️ Next"
    t_of = "van" if NL else "of"

    if not category_results:
        st.markdown(f"""
        <div style="border:1px solid #dce1eb;border-radius:10px;padding:16px 20px;background:#F8F9FB;margin-top:8px;">
          <span style="display:block;">{t_no_data}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # Build {assigned_category: [idea, idea, ...]}
    category_dict = {}
    for result in category_results:
        for ri in getattr(result, "response_ideas", []) or []:
            cat = getattr(ri, "assigned_category", None)
            idea = getattr(ri, "idea", None)
            if cat is None or idea is None:
                continue
            category_dict.setdefault(cat, []).append(idea)

    if not category_dict:
        st.markdown(f"""
        <div style="border:1px solid #dce1eb;border-radius:10px;padding:16px 20px;background:#F8F9FB;margin-top:8px;">
          <span style="display:block;">{t_no_categories}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    category_names = sorted(category_dict.keys())

    # Session state for current position
    if "category_idx" not in st.session_state:
        st.session_state.category_idx = 0

    total = len(category_names)
    idx = st.session_state.category_idx

    # Navigation header (buttons + indicator + optional jump)
    nav1, nav2, nav3, nav4 = st.columns([1, 2, 2, 2])
    with nav1:
        if st.button(t_prev, use_container_width=True, disabled=(idx <= 0)):
            st.session_state.category_idx = max(0, idx - 1)
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='margin-top:6px;text-align:center;'>"
            f"{idx + 1} {t_of} {total}"
            f"</div>",
            unsafe_allow_html=True
        )
    with nav3:
        # Quick jump by index
        new_idx = st.number_input(
            label="X",
            min_value=1, max_value=total, value=idx + 1,
            step=1, key="category_jump_number", label_visibility="collapsed"
        )
        if new_idx - 1 != idx:
            st.session_state.category_idx = new_idx - 1
            st.rerun()
    with nav4:
        if st.button(t_next, use_container_width=True, disabled=(idx >= total - 1)):
            st.session_state.category_idx = min(total - 1, idx + 1)
            st.rerun()

    # Active category
    active_cat = category_names[st.session_state.category_idx]
    ideas = category_dict[active_cat]

    # Show category name as subheader
    st.markdown(f"**{active_cat}** ({len(ideas)} {'ideeën' if NL else 'ideas'})")

    # Build HTML list of ideas (strip metadata)
    li_items = []
    for idea in ideas:
        # Remove all [key=value] metadata brackets
        cleaned_idea = re.sub(r'\[.*?=.*?\]', '', idea).strip()
        li_items.append(f"<li style='margin:4px 0;'>{html.escape(cleaned_idea)}</li>")
    ideas_html = "".join(li_items)

    # Render clean idea list
    st.markdown(f"""
    <div style="
        border: 1px solid #e6eaf2;
        border-radius: 8px;
        padding: 16px 20px;
        background-color: #ffffff;
        margin-top: 8px;">
      <ul style="margin:0; padding:0 0 0 1.2em;">
        {ideas_html}
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # Save number of categories
    st.session_state.num_categories = len(category_names)

def show_step9_assignment_stats():
    """Display assignment statistics - fixed summary"""
    # from utils.pipelineSummarizer import PipelineSummarizer
    # import io
    # import sys
    
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

                    # Generate variable_key for visualization data loading
                    variable_key = generate_enhanced_variable_key(
                        st.session_state.selected_variable,
                        merge_config=st.session_state.get('merge_config'),
                        sample_size=st.session_state.get('sample_size')
                    )

                    # Run pipeline step 9 export with visualizations
                    excel_path = _run_with_verbose_capture(
                        pipeline.step_9_export_results,
                        code_assigned_results=code_assigned_results,
                        theme_enriched_codebook=theme_enriched_codebook,
                        filename=st.session_state.filename,
                        var_name=st.session_state.get('var_name', st.session_state.get('selected_variable', 'unknown')),
                        verbose=True,
                        include_visualizations=True,
                        cache_manager=_get_cache_manager(),
                        variable_key=variable_key
                    )

                    st.balloons()  # Celebration!
                    st.success("✅ " + (f"Geëxporteerd naar: {excel_path}" if st.session_state.language == "nl" else f"Exported to: {excel_path}"))
                    mark_step_completed(9)

                except Exception as e:
                    st.error("❌ " + (f"Export fout: {str(e)}" if st.session_state.language == "nl" else f"Export error: {str(e)}"))

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
        code_assigned_results = cache_manager.load_from_cache(filename, "code_assignment_direct", variable_key, models.CodeAssignedModel)
        
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
            data = cache_manager.load_from_cache(filename, "embeddings", variable_key, None)
            if data:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 " + ("Ga door naar volgende stap" if lang == "nl" else "Continue to Next Step"), type="primary", use_container_width=True, key="embedding_continue"):
                        st.session_state.step = 5
                        st.rerun()
                
            else:
                st.write("⏳ No embeddings in cache - run embedding generation first")
                
        
        elif step_number == 5:
            # Step 5: Categories
            data = cache_manager.load_from_cache(filename, "code_assignment", variable_key, models.CodeAssignedModel)
            if data:
                show_category_samples(data)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 " + ("Ga door naar volgende stap" if lang == "nl" else "Continue to Next Step"), type="primary", use_container_width=True, key="cluster_continue"):
                        st.session_state.step = 6
                        st.rerun()
            else:
                st.write("⏳ No categories in cache - run category discovery first")
                
        elif step_number == 6:
            # Step 6: Codebook reasoning
            try:
                from utils.codeGenerator import CodeGeneratorReasoningResults
                data = cache_manager.load_from_cache(filename, "codebook_generation_reasoning", variable_key, CodeGeneratorReasoningResults)
                
                if data and len(data) > 0:
                    # Set session state for display function
                    if 'pipeline_results' not in st.session_state:
                        st.session_state.pipeline_results = {}
                    st.session_state.pipeline_results['reasoning_results'] = data[0]

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
            # Step 7: Export
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 " + ("Ga door naar volgende stap" if st.session_state.language == 'nl' else "Continue to Next Step"),
                            type="primary", use_container_width=True, key="export_step_continue"):
                    st.session_state.step = 8
                    st.rerun()
                
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