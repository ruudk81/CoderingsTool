"""
Experimental Configuration for Step 1: Preprocess

Re-exports production config items that don't need modification.
Defines experimental constants for spell checking.

Pattern follows step_6_codeGenerator/config_exp.py
"""

from dataclasses import dataclass

# =============================================================================
# RE-EXPORTS FROM PRODUCTION CONFIG (read-only items)
# =============================================================================
from config import (
    OPENAI_API_KEY,
    DEFAULT_LANGUAGE,
    ModelConfig,
    ProcessingConfig,
    DEFAULT_PROCESSING_CONFIG,
    API_PROVIDER,
    FALLBACK_TPM,
    FALLBACK_RPM,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)
from config_steps.config_preprocess import (
    HUNSPELL_PATH,
    DUTCH_DICT_PATH,
    ENGLISH_DICT_PATH,
    SpellCheckConfig,
    DEFAULT_SPELLCHECK_CONFIG,
)

# =============================================================================
# EXPERIMENTAL CONSTANTS (moved from spellChecker_exp.py)
# =============================================================================

# Hunspell processing limits
MAX_HUNSPELL_PROCESSES = 20          # Max parallel Hunspell processes to prevent resource exhaustion
MAX_SAFE_BATCH_SIZE = 1000           # Maximum batch size for Hunspell word checking

# Suggestion generation
SUGGESTION_BATCH_SIZE = 50           # Words per batch for suggestion generation
MAX_CONCURRENT_SUGGESTION_BATCHES = 6  # Concurrent batches for suggestion processing

# Token estimation
OUTPUT_TOKEN_RATIO = 0.15            # Estimated output/input token ratio for spell correction

# SpaCy validation
SPACY_VECTOR_NORM_THRESHOLD = 5      # Minimum vector norm for valid SpaCy tokens


# =============================================================================
# EXPERIMENTAL CONFIG CLASS (for future use)
# =============================================================================
@dataclass
class SpellCheckConfigExp(SpellCheckConfig):
    """Experimental spell check config - override fields as needed."""
    pass


DEFAULT_SPELLCHECK_CONFIG_EXP = SpellCheckConfigExp()
