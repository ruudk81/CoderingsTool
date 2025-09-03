"""
Bare Mode Utilities for CoderingsTool
Provides compatibility between Streamlit and bare mode execution.
"""

import functools
import time
import logging
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global state storage for bare mode
_bare_mode_cache = {}
_bare_mode_session_state = {}

def is_streamlit_context() -> bool:
    """
    Detect if we're running in Streamlit context.
    Returns True if Streamlit context is available, False otherwise.
    """
    try:
        import streamlit as st
        # Check for script run context, not just session_state existence
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx is not None
    except (ImportError, RuntimeError, AttributeError):
        return False

def conditional_cache_resource(func: Callable) -> Callable:
    """
    Decorator that uses st.cache_resource in Streamlit context,
    or functools.lru_cache in bare mode.
    """
    if is_streamlit_context():
        import streamlit as st
        return st.cache_resource(func)
    else:
        # Use LRU cache with maxsize=1 for resource caching
        return functools.lru_cache(maxsize=1)(func)

def conditional_cache_data(show_spinner: str = None, **kwargs) -> Callable:
    """
    Decorator that uses st.cache_data in Streamlit context,
    or functools.lru_cache in bare mode.
    """
    def decorator(func: Callable) -> Callable:
        if is_streamlit_context():
            import streamlit as st
            return st.cache_data(show_spinner=show_spinner, **kwargs)(func)
        else:
            # Use LRU cache for data caching in bare mode
            return functools.lru_cache(maxsize=128)(func)
    return decorator

class BareSessionState:
    """
    Simple session state implementation for bare mode.
    Mimics Streamlit's session_state behavior.
    """
    def __init__(self):
        self._state = _bare_mode_session_state
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __delitem__(self, key):
        del self._state[key]
    
    def __contains__(self, key):
        return key in self._state
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def setdefault(self, key, default):
        return self._state.setdefault(key, default)
    
    def __getattr__(self, key):
        if key.startswith('_'):
            return super().__getattribute__(key)
        return self._state.get(key)
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._state[key] = value
    
    def __hasattr__(self, key):
        return key in self._state

def get_session_state():
    """
    Get session state object - Streamlit's in Streamlit context,
    our bare mode implementation otherwise.
    """
    if is_streamlit_context():
        import streamlit as st
        return st.session_state
    else:
        return BareSessionState()

@contextmanager
def conditional_spinner(message: str = "Processing..."):
    """
    Context manager that shows Streamlit spinner in Streamlit context,
    or simple logging message in bare mode.
    """
    if is_streamlit_context():
        import streamlit as st
        with st.spinner(message):
            yield
    else:
        logger.info(f"⏳ {message}")
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.info(f"✅ Completed in {elapsed:.2f}s")

def conditional_error(message: str):
    """
    Show error message in Streamlit context or log as error in bare mode.
    """
    if is_streamlit_context():
        import streamlit as st
        st.error(message)
    else:
        logger.error(f"ERROR: {message}")

def conditional_info(message: str):
    """
    Show info message in Streamlit context or log as info in bare mode.
    """
    if is_streamlit_context():
        import streamlit as st
        st.info(message)
    else:
        logger.info(f"INFO: {message}")

def conditional_success(message: str):
    """
    Show success message in Streamlit context or log as info in bare mode.
    """
    if is_streamlit_context():
        import streamlit as st
        st.success(message)
    else:
        logger.info(f"SUCCESS: {message}")

def conditional_warning(message: str):
    """
    Show warning message in Streamlit context or log as warning in bare mode.
    """
    if is_streamlit_context():
        import streamlit as st
        st.warning(message)
    else:
        logger.warning(f"WARNING: {message}")

def clear_bare_mode_cache():
    """
    Clear the bare mode cache. Useful for testing or when memory is tight.
    """
    global _bare_mode_cache, _bare_mode_session_state
    _bare_mode_cache.clear()
    _bare_mode_session_state.clear()
    logger.info("Bare mode cache cleared")

# For debugging: log the context detection result
if __name__ == "__main__":
    context = "Streamlit" if is_streamlit_context() else "Bare mode"
    print(f"Running in: {context}")