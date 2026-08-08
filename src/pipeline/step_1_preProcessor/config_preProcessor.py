"""
Preprocessing-specific configuration — separate from main config.py

This module contains all configuration for the Step 1 preprocessing pipeline:
- Hunspell paths (cross-platform)
- SpellCheckConfig dataclass
- Processing constants for spell checking
"""
import os
import platform
import shutil
from pathlib import Path
from dataclasses import dataclass
from config import get_step_model


# =============================================================================
# HUNSPELL CONFIGURATION (Cross-platform)
# =============================================================================

def _get_hunspell_paths():
    """
    Get Hunspell executable and dictionary directory paths.
    Supports Windows (bundled .exe) and macOS/Linux (system install via brew/apt).

    Returns:
        Tuple of (hunspell_executable_path, hunspell_dict_directory)
    """
    # Determine project root: step_1_preProcessor/ -> pipeline/ -> src/ -> project root
    config_file_dir = Path(__file__).parent  # step_1_preProcessor/
    project_root = config_file_dir.parent.parent.parent  # project root
    hunspell_dir = str(project_root / "hunspell")

    system = platform.system()

    if system == "Windows":
        # Use bundled Windows executable
        hunspell_exe = os.path.join(hunspell_dir, "hunspell.exe")
    else:
        # macOS or Linux: try system-installed hunspell
        # Check common locations in order of preference
        system_hunspell = shutil.which("hunspell")

        if system_hunspell:
            hunspell_exe = system_hunspell
        elif system == "Darwin":  # macOS
            # Homebrew paths (Apple Silicon and Intel)
            brew_paths = [
                "/opt/homebrew/bin/hunspell",  # Apple Silicon
                "/usr/local/bin/hunspell",      # Intel Mac
            ]
            hunspell_exe = next((p for p in brew_paths if os.path.exists(p)), None)
            if not hunspell_exe:
                # Fallback to bundled (won't work but provides clear error)
                hunspell_exe = os.path.join(hunspell_dir, "hunspell")
        else:  # Linux
            linux_paths = [
                "/usr/bin/hunspell",
                "/usr/local/bin/hunspell",
            ]
            hunspell_exe = next((p for p in linux_paths if os.path.exists(p)), None)
            if not hunspell_exe:
                hunspell_exe = os.path.join(hunspell_dir, "hunspell")

    return hunspell_exe, hunspell_dir

# Initialize cross-platform paths
HUNSPELL_PATH, _hunspell_dir = _get_hunspell_paths()
DUTCH_DICT_PATH = os.path.join(_hunspell_dir, "dict", "nl_NL")
ENGLISH_DICT_PATH = os.path.join(_hunspell_dir, "dict", "en_GB")


# =============================================================================
# SPELL CHECKER PROCESSING CONSTANTS
# =============================================================================

MAX_HUNSPELL_PROCESSES = 20              # Max parallel Hunspell processes to prevent resource exhaustion
MAX_SAFE_BATCH_SIZE = 1000               # Maximum batch size for Hunspell word checking
SUGGESTION_BATCH_SIZE = 50               # Words per batch for suggestion generation
MAX_CONCURRENT_SUGGESTION_BATCHES = 6    # Concurrent batches for suggestion processing
OUTPUT_TOKEN_RATIO = 0.15                # Estimated output/input token ratio for spell correction
SPACY_VECTOR_NORM_THRESHOLD = 5          # Minimum vector norm for valid SpaCy tokens

# Unrepairable input: a token that is not language at all. Hunspell flags it like
# any other unknown word, but there is nothing to correct it to, so the LLM has to
# invent something — and it invents plausible words, which is worse than leaving
# the noise visible. Doubles as acronym protection: BLG and ZZP have no vowel
# either, and must reach the output as written.
WORD_VOWELS = "aeiouyàáâäèéêëìíîïòóôöùúûü"
MAX_REPEATED_CHARS = 3      # "allles" is a typo worth fixing, "xxxx" is a hammered key
MAX_CONSONANT_RUN = 4       # "maatschappelihjk" survives this, "Jsisjdkdjd" does not


# =============================================================================
# SPELL CHECK CONFIGURATION
# =============================================================================

@dataclass
class SpellCheckConfig:
    """Configuration for spell checking step.

    Only the LLM call's own parameters and the local Hunspell/SpaCy machinery live
    here. Workers, pacing, concurrency, timeouts and retries belong to
    SmoothRequester (phase key `step1_spell_check`) — a copy here would not
    override it, it would simply do nothing.
    """
    model: str = get_step_model("spell_check")
    temperature: float = 0.0

    # Batching for the local NLP work
    spacy_batch_size: int = 64

    # Hunspell
    hunspell_pool_size: int = 20        # always used, so the pool never auto-tunes
    hunspell_batch_size: int = 1000     # words per check batch

    # Suggestion caching
    enable_suggestion_caching: bool = True
    enable_word_frequency_cache: bool = True  # skip Hunspell for words already seen
    max_unique_oov_words: int = 5000          # cap on unique OOV words processed

    # Dataset vocabulary: an unknown word that recurs across many responses is
    # this dataset's own vocabulary (a brand, an abbreviation, a term), not a
    # typo. Threshold is the higher of the two, so it scales with sample size
    # without collapsing on a small one. Calibrated on one dataset — remeasure
    # before trusting it on a corpus of a very different size.
    dataset_vocab_min_responses: int = 4
    dataset_vocab_response_ratio: float = 0.0025

    # Output formatting
    repeated_char_threshold: int = 5    # a "word" of 5+ identical chars is not spell-checkable
    max_correction_examples: int = 10   # verbose output only


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

DEFAULT_SPELLCHECK_CONFIG = SpellCheckConfig()
