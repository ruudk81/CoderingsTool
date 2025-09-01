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
from config import ALLOWED_EXTENSIONS
from utils.dataLoader import DataLoader
from utils.cacheManager import CacheManager
from config import CacheConfig
from pipeline_runner import get_pipeline_runner
import ui_text as ui

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
    st.session_state.cache_manager = CacheManager(CacheConfig())
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(verbose=False)
if 'pipeline_runner' not in st.session_state:
    st.session_state.pipeline_runner = get_pipeline_runner()


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
                "",
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
                        variables = st.session_state.data_loader.list_variables(uploaded_file.name)
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
            
            # Variable selection
            col1, col2 = st.columns(2)
            
            with col1:
                # Select text variable to analyze
                text_var = st.selectbox(
                    "Selecteer tekst variabele" if lang == "nl" else "Select text variable",
                    options=list(st.session_state.available_variables.keys()),
                    format_func=lambda x: f"{x} - {st.session_state.available_variables[x] or '(No label)'}",
                    key="text_variable"
                )
                
            with col2:
                # Select ID column
                id_var = st.selectbox(
                    "Selecteer ID kolom" if lang == "nl" else "Select ID column",
                    options=list(st.session_state.available_variables.keys()),
                    format_func=lambda x: f"{x} - {st.session_state.available_variables[x] or '(No label)'}",
                    key="id_variable"
                )
            
            # Preview selected variable
            if st.button("Voorbeeld Bekijken" if lang == "nl" else "Preview Variable"):
                if text_var and id_var:
                    with st.spinner("Data wordt geladen..." if lang == "nl" else "Loading data..."):
                        try:
                            preview_data = st.session_state.data_loader.get_variable_with_IDs(
                                st.session_state.filename, id_var, text_var
                            )
                            st.session_state.variable_preview = preview_data
                            st.session_state.selected_variable = text_var
                            st.session_state.selected_id_column = id_var
                            st.success("Preview geladen!" if lang == "nl" else "Preview loaded!")
                        except Exception as e:
                            st.error(f"Fout bij preview: {str(e)}" if lang == "nl" else f"Preview error: {str(e)}")
            
            # Show preview if available
            if st.session_state.variable_preview is not None:
                st.subheader("Data Preview")
                preview_df = st.session_state.variable_preview
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Totaal" if lang == "nl" else "Total", len(preview_df))
                with col2:
                    non_null = preview_df[st.session_state.selected_variable].notna().sum()
                    st.metric("Niet-leeg" if lang == "nl" else "Non-empty", non_null)
                with col3:
                    unique_vals = preview_df[st.session_state.selected_variable].nunique()
                    st.metric("Uniek" if lang == "nl" else "Unique", unique_vals)
                
                # Show sample data
                st.subheader("Sample Responses")
                sample_data = preview_df[preview_df[st.session_state.selected_variable].notna()].head(10)
                st.dataframe(sample_data, use_container_width=True)
                
                # Ready to proceed button
                if st.button("Doorgaan naar Preprocessing" if lang == "nl" else "Continue to Preprocessing", type="primary"):
                    st.session_state.step = 1
                    st.rerun()

def show_preprocessing_page():
    lang = st.session_state.language
    st.header("Stap 2: Preprocessing" if lang == "nl" else "Step 2: Preprocessing")
    st.markdown(ui.get_text("PREPROCESSING_INFO", lang))
    
    # Show current selection
    if st.session_state.selected_variable and st.session_state.selected_id_column:
        st.info(f"**Variabele:** {st.session_state.selected_variable}\n\n**ID Kolom:** {st.session_state.selected_id_column}")
    else:
        st.warning("Ga terug en selecteer een variabele" if lang == "nl" else "Go back and select a variable")
        return
    
    if st.button(ui.get_text("BTN_PREPROCESS", lang), type="primary"):
        progress_container = st.empty()
        try:
            # Step 1: Load data if not already loaded
            if 'raw_text_list' not in st.session_state.pipeline_results:
                var_lab = st.session_state.data_loader.get_varlab(st.session_state.filename, st.session_state.selected_variable)
                raw_text_list = st.session_state.pipeline_runner.step_1_load_data(
                    filename=st.session_state.filename,
                    id_column=st.session_state.selected_id_column,
                    var_name=st.session_state.selected_variable,
                    streamlit_container=progress_container
                )
                st.session_state.pipeline_results['raw_text_list'] = raw_text_list
                st.session_state.pipeline_results['var_lab'] = var_lab
            
            # Step 2: Preprocessing
            preprocessed_text = st.session_state.pipeline_runner.step_2_preprocess(
                raw_text_list=st.session_state.pipeline_results['raw_text_list'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['preprocessed_text'] = preprocessed_text
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
            quality_filtered_text = st.session_state.pipeline_runner.step_3_quality_filter(
                preprocessed_text=st.session_state.pipeline_results['preprocessed_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
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
    
    if st.button("Start Idee Extractie" if lang == "nl" else "Start Idea Extraction", type="primary"):
        progress_container = st.empty()
        try:
            encoded_text = st.session_state.pipeline_runner.step_4_extract_ideas(
                quality_filtered_text=st.session_state.pipeline_results['quality_filtered_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['encoded_text'] = encoded_text
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
            embedded_text = st.session_state.pipeline_runner.step_5_generate_embeddings(
                encoded_text=st.session_state.pipeline_results['encoded_text'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                provider=provider,
                embedding_model=model,
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
    
    if st.button(ui.get_text("BTN_CLUSTER", lang), type="primary"):
        progress_container = st.empty()
        try:
            initial_cluster_results = st.session_state.pipeline_runner.step_6_cluster(
                embedded_text=st.session_state.pipeline_results['embedded_text'],
                filename=st.session_state.filename,
                epsilon=epsilon,
                alpha=alpha,
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['initial_cluster_results'] = initial_cluster_results
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
            codebook_main = st.session_state.pipeline_runner.step_7_generate_codebook(
                initial_cluster_results=st.session_state.pipeline_results['initial_cluster_results'],
                filename=st.session_state.filename,
                var_name=st.session_state.selected_variable,
                var_lab=st.session_state.pipeline_results['var_lab'],
                use_speculative_starter_codes=use_speculative,
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['codebook_main'] = codebook_main
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
            theme_enriched_codebook = st.session_state.pipeline_runner.step_8_identify_themes(
                codebook_main=st.session_state.pipeline_results['codebook_main'],
                filename=st.session_state.filename,
                var_name=st.session_state.selected_variable,
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
    
    if st.button("Wijs Codes Toe" if lang == "nl" else "Assign Codes", type="primary"):
        progress_container = st.empty()
        try:
            code_assigned_results = st.session_state.pipeline_runner.step_9a_assign_codes(
                initial_cluster_results=st.session_state.pipeline_results['initial_cluster_results'],
                theme_enriched_codebook=st.session_state.pipeline_results['theme_enriched_codebook'],
                filename=st.session_state.filename,
                var_lab=st.session_state.pipeline_results['var_lab'],
                method=method,
                streamlit_container=progress_container
            )
            st.session_state.pipeline_results['code_assigned_results'] = code_assigned_results
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
        
        if st.button("Exporteer Resultaten" if lang == "nl" else "Export Results", type="primary"):
            progress_container = st.empty()
            try:
                # Use the same export logic as pipeline.py lines 997-1010
                from utils.codeAssignmentExporter import CodeAssignmentExporter
                
                progress_container.text("🔄 " + ("Resultaten exporteren naar Excel..." if lang == "nl" else "Exporting results to Excel..."))
                
                exporter = CodeAssignmentExporter(verbose=True)
                excel_path = exporter.export_to_excel(
                    code_assigned_results,
                    theme_enriched_codebook,
                    st.session_state.filename,
                    st.session_state.selected_variable,
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
                st.download_button(
                    label="📊 Download Excel Resultaten" if lang == "nl" else "📊 Download Excel Results",
                    data=excel_data,
                    file_name=f"{st.session_state.filename}_{st.session_state.selected_variable}_results.xlsx",
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
            
            st.download_button(
                label="📋 Download Codebook" if lang == "nl" else "📋 Download Codebook",
                data=codebook_csv,
                file_name=f"{st.session_state.filename}_{st.session_state.selected_variable}_codebook.csv",
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