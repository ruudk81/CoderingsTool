"""
Centralized cached resources for Streamlit performance optimization.
Provides session-wide cached instances of expensive resources.
"""

import streamlit as st
import instructor
import tiktoken
import logging
from openai import AsyncOpenAI
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY

logger = logging.getLogger(__name__)

@st.cache_resource
def get_openai_client(api_key: str = None):
    """Create cached OpenAI instructor client for session-wide reuse"""
    api_key = api_key or OPENAI_API_KEY
    with st.spinner("Initializing OpenAI client..."):
        return instructor.patch(AsyncOpenAI(api_key=api_key))

@st.cache_resource
def get_tiktoken_encoding(model_name: str):
    """Create cached tiktoken encoding for session-wide reuse"""
    try:
        with st.spinner(f"Loading tokenizer for {model_name}..."):
            return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Using cl100k_base encoding as fallback for {model_name}")
        return tiktoken.get_encoding("cl100k_base")

@st.cache_resource
def get_spacy_nlp():
    """Load SpaCy language model with Streamlit caching for session-wide reuse"""
    import spacy
    from config import DEFAULT_LANGUAGE
    
    try:
        vocab = "nl_core_news_lg" if DEFAULT_LANGUAGE == "Dutch" else "en_core_web_lg"
        with st.spinner(f"Loading {vocab} language model..."):
            nlp = spacy.load(vocab)
        return nlp
    except OSError:
        vocab = "nl_core_news_lg" if DEFAULT_LANGUAGE == "Dutch" else "en_core_web_lg"
        st.error(f"SpaCy model not found. Please install it with: python -m spacy download {vocab}")
        raise RuntimeError(f"SpaCy model not found. Please install it with: python -m spacy download {vocab}")