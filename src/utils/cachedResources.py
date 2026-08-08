"""Process-wide caches for resources that are expensive to construct.

Both are loaded once per process. Streamlit reruns the script on every
interaction but keeps imported modules in sys.modules, so an lru_cache at
module level survives a rerun just like st.cache_resource would — without
needing to know whether Streamlit is running at all.
"""

import logging
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_tiktoken_encoding(model_name: str):
    """Tokenizer for one model, falling back to cl100k_base for unknown names."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.debug(f"Using cl100k_base encoding as fallback for {model_name}")
        return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=None)
def get_spacy_nlp_conditional(spell_check_enabled: bool = True):
    """SpaCy language model, or None when spell checking is off.

    Skipping the load entirely is the point: the large vocabularies cost
    seconds and hundreds of MB.
    """
    if not spell_check_enabled:
        return None

    import spacy

    from config import DEFAULT_LANGUAGE

    vocab = "nl_core_news_lg" if DEFAULT_LANGUAGE == "Dutch" else "en_core_web_lg"
    try:
        logger.info(f"Loading {vocab} language model...")
        return spacy.load(vocab)
    except OSError:
        raise RuntimeError(
            f"SpaCy model not found. Please install it with: python -m spacy download {vocab}"
        )
