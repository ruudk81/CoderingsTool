"""
Centralized cached resources for Streamlit and bare mode compatibility.
Provides session-wide cached instances of expensive resources.
"""

import instructor
import tiktoken
import logging
from openai import AsyncOpenAI
from config import DEFAULT_LANGUAGE, OPENAI_API_KEY
from .bare_mode_utils import conditional_cache_resource, conditional_spinner

logger = logging.getLogger(__name__)

@conditional_cache_resource
def get_openai_client(api_key: str = None):
    """Create cached OpenAI instructor client for session-wide reuse"""
    api_key = api_key or OPENAI_API_KEY
    with conditional_spinner("Initializing OpenAI client..."):
        return instructor.patch(AsyncOpenAI(api_key=api_key))

@conditional_cache_resource
def get_tiktoken_encoding(model_name: str):
    """Create cached tiktoken encoding for session-wide reuse"""
    try:
        with conditional_spinner(f"Loading tokenizer for {model_name}..."):
            return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Using cl100k_base encoding as fallback for {model_name}")
        return tiktoken.get_encoding("cl100k_base")

@conditional_cache_resource
def get_spacy_nlp_conditional(spell_check_enabled: bool = True):
    """Load SpaCy language model conditionally based on configuration"""
    if not spell_check_enabled:
        return None  # Skip loading entirely to save memory and time
        
    import spacy
    from config import DEFAULT_LANGUAGE
    from .bare_mode_utils import conditional_error
    
    try:
        vocab = "nl_core_news_lg" if DEFAULT_LANGUAGE == "Dutch" else "en_core_web_lg"
        with conditional_spinner(f"Loading {vocab} language model..."):
            nlp = spacy.load(vocab)
        return nlp
    except OSError:
        vocab = "nl_core_news_lg" if DEFAULT_LANGUAGE == "Dutch" else "en_core_web_lg"
        error_msg = f"SpaCy model not found. Please install it with: python -m spacy download {vocab}"
        conditional_error(error_msg)
        raise RuntimeError(error_msg)

@conditional_cache_resource
def get_spacy_nlp():
    """Load SpaCy language model with caching for session-wide reuse (always loads)"""
    return get_spacy_nlp_conditional(True)

@conditional_cache_resource
def get_embedder_for_provider(provider: str, config=None, model_config=None):
    """Load embedding provider conditionally - only load what's needed"""
    with conditional_spinner(f"Initializing {provider} embedder..."):
        if provider.lower() == "openai":
            from utils.embedder import Embedder
            return Embedder(config=config, model_config=model_config, provider="openai", verbose=False)
        elif provider.lower() == "gemini":
            from utils.embedder import Embedder  
            return Embedder(config=config, model_config=model_config, provider="gemini", verbose=False)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

@conditional_cache_resource
def get_clusterer_conditional(config_hash: str = "default"):
    """Load clusterer resources conditionally"""
    with conditional_spinner("Loading clustering algorithms..."):
        from utils.clusterer import Clusterer
        return Clusterer