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
from config import get_model


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
    # Determine project root: config_steps/ -> src/ -> project root
    config_file_dir = Path(__file__).parent  # config_steps/
    project_root = config_file_dir.parent.parent  # project root
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


# =============================================================================
# SPELL CHECK CONFIGURATION
# =============================================================================

@dataclass
class SpellCheckConfig:
    """Configuration for spell checking step"""
    model: str = get_model("nano")
    batch_size: int = 20
    temperature: float = 0.0
    max_tokens: int = 4000
    retries: int = 3
    retry_delay: int = 2
    max_batch_size: int = 5
    completion_reserve: int = 1000
    cache_size: int = 10000
    spacy_batch_size: int = 64  # Increased for better performance
    repeated_char_threshold: int = 5  # Characters repeated 5+ times
    max_correction_examples: int = 10  # For verbose output
    seed: int = 42
    context_chars: int = 20  # Characters of context for spell checking
    max_concurrent_requests: int = 5  # For API rate limiting

    # Performance optimization settings
    max_words_to_check: int = 100000  # Skip spell checking if more words than this
    enable_word_frequency_cache: bool = True  # Cache common words
    progress_report_interval: int = 10000  # Report progress every N words
    max_unique_oov_words: int = 5000  # Limit unique OOV words to process
    enable_early_termination: bool = True  # Allow early termination for large datasets

    # Aggressive parallel processing settings for suggestion generation
    max_concurrent_suggestion_chunks: int = 20
    max_words_per_chunk: int = 1200
    enable_adaptive_chunking: bool = True
    chunk_progress_reporting: bool = True
    suggestion_processing_semaphore_limit: int = 100

    # Timeout configuration for API calls
    minimum_timeout_seconds: float = 15.0
    maximum_timeout_seconds: float = 60.0

    # Hunspell optimization
    hunspell_concurrent_sessions: int = 20
    hunspell_batch_size: int = 1000
    enable_streaming_oov_detection: bool = True
    oov_detection_queue_size: int = 10000

    # Rate limiting optimization parameters
    rate_limit_safety_factor: float = 0.95
    rate_limit_utilization: float = 0.98
    concurrent_burst_multiplier: float = 3.0

    # Suggestion validation parameters
    enable_suggestion_pre_validation: bool = True
    disable_pre_validation_above_oov_words: int = 2000
    enable_suggestion_caching: bool = True

    # Performance optimization parameters
    hunspell_pool_size: int = 20
    ultra_batch_threshold: int = 1000
    ultra_batch_size: int = 10000


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

DEFAULT_SPELLCHECK_CONFIG = SpellCheckConfig()
