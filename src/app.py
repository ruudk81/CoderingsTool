"""Web interface for CoderingsTool"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "modules"))
sys.path.append(str(project_root / "src" / "modules" / "utils"))

import models
import pipeline
from config import ALLOWED_EXTENSIONS
import ui_text as ui

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'language' not in st.session_state:
    st.session_state.language = ui.DEFAULT_LANGUAGE

# Page config
st.set_page_config(
    page_title=ui.get_text("APP_TITLE", st.session_state.language),
    page_icon="📊",
    layout="wide"
)

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
        
        # Progress indicator
        progress = st.progress(st.session_state.step / 6)
        st.markdown(f"**{ui.get_text('CURRENT_STEP', st.session_state.language)}** {st.session_state.step + 1}/6")
    
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
            show_embedding_page()
        elif st.session_state.step == 4:
            show_clustering_page()
        elif st.session_state.step == 5:
            show_labeling_page()
        elif st.session_state.step == 6:
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
                # Save uploaded file
                file_path = Path("data") / uploaded_file.name
                file_path.parent.mkdir(exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.filename = uploaded_file.name
                st.session_state.step = 1
                st.rerun()

def show_preprocessing_page():
    lang = st.session_state.language
    st.header("Stap 2: Preprocessing" if lang == "nl" else "Step 2: Preprocessing")
    st.markdown(ui.get_text("PREPROCESSING_INFO", lang))
    
    if st.button(ui.get_text("BTN_PREPROCESS", lang), type="primary"):
        with st.spinner("Tekst wordt verwerkt..." if lang == "nl" else "Preprocessing text..."):
            # Run preprocessing logic here
            st.success(ui.get_text("SUCCESS_PREPROCESSING", lang))
            st.session_state.step = 2
            st.rerun()

def show_filtering_page():
    lang = st.session_state.language
    st.header("Stap 3: Kwaliteitsfiltering" if lang == "nl" else "Step 3: Quality Filtering")
    st.markdown(ui.get_text("FILTERING_INFO", lang))
    
    if st.button(ui.get_text("BTN_FILTER", lang), type="primary"):
        with st.spinner("Antwoorden worden gefilterd..." if lang == "nl" else "Filtering responses..."):
            # Run filtering logic here
            st.success(ui.get_text("SUCCESS_FILTERING", lang))
            st.session_state.step = 3
            st.rerun()

def show_embedding_page():
    lang = st.session_state.language
    st.header("Stap 4: Genereer Embeddings" if lang == "nl" else "Step 4: Generate Embeddings")
    st.markdown(ui.get_text("EMBEDDING_INFO", lang))
    
    if st.button(ui.get_text("BTN_EMBED", lang), type="primary"):
        with st.spinner("Embeddings worden aangemaakt..." if lang == "nl" else "Creating embeddings..."):
            # Run embedding logic here
            st.success(ui.get_text("SUCCESS_EMBEDDING", lang))
            st.session_state.step = 4
            st.rerun()

def show_clustering_page():
    lang = st.session_state.language
    st.header("Stap 5: Clustering" if lang == "nl" else "Step 5: Clustering")
    st.markdown(ui.get_text("CLUSTERING_INFO", lang))
    
    if st.button(ui.get_text("BTN_CLUSTER", lang), type="primary"):
        with st.spinner("Antwoorden worden geclusterd..." if lang == "nl" else "Clustering responses..."):
            # Run clustering logic here
            st.success(ui.get_text("SUCCESS_CLUSTERING", lang))
            st.session_state.step = 5
            st.rerun()

def show_labeling_page():
    lang = st.session_state.language
    st.header("Stap 6: Thematisch Labelen" if lang == "nl" else "Step 6: Thematic Labeling")
    st.markdown(ui.get_text("LABELING_INFO", lang))
    
    if st.button(ui.get_text("BTN_LABEL", lang), type="primary"):
        with st.spinner("Labels worden gegenereerd..." if lang == "nl" else "Generating labels..."):
            # Run labeling logic here
            st.success(ui.get_text("SUCCESS_LABELING", lang))
            st.session_state.step = 6
            st.rerun()

def show_results_page():
    lang = st.session_state.language
    st.header("Resultaten" if lang == "nl" else "Results")
    st.markdown(ui.get_text("RESULTS_INFO", lang))
    
    # Show results and download options
    st.download_button(
        label=ui.get_text("BTN_DOWNLOAD", lang),
        data="", # Replace with actual data
        file_name="results.csv",
        mime="text/csv"
    )
    
    if st.button(ui.get_text("BTN_RESTART", lang)):
        st.session_state.step = 0
        st.session_state.data = None
        st.session_state.filename = None
        st.rerun()

def show_info_panel():
    lang = st.session_state.language
    st.subheader("Informatie" if lang == "nl" else "Information")
    
    if st.session_state.filename:
        st.markdown(f"**{'Huidig bestand' if lang == 'nl' else 'Current file'}:** {st.session_state.filename}")
    
    st.markdown("---")
    descriptions = ui.get_text("STEP_DESCRIPTIONS", lang)
    st.markdown(descriptions[st.session_state.step])

if __name__ == "__main__":
    main()