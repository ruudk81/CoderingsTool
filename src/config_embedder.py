"""
Embedder-specific configuration - separate from main config.py

This module contains all configuration constants for the Embedder,
organized into logical dataclasses for easier management and tuning.

These settings control:
- Embedding text format selection
- Batch processing settings
- Provider-specific configurations
- Question-aware embedding transformation
- Embedding quality analysis
"""
from dataclasses import dataclass
from typing import Literal, Optional


# =============================================================================
# HARDCODED SETTINGS FOR "BOTH" MODE
# =============================================================================
# Which idea format to use when embedding_text_format="both"
# Options: "idea" or "idea_without_template_prefix"
BOTH_MODE_IDEA_FORMAT: Literal["idea", "idea_without_template_prefix"] = "idea_without_template_prefix"


# =============================================================================
# TEXT FORMAT CONFIGURATION
# =============================================================================

@dataclass
class TextFormatConfig:
    """Configuration for embedding text format selection.

    Controls what text is extracted from ideas for embedding:
    - "idea": Embed the clean idea text (idea.idea field)
    - "taxonomy_phrase": Embed the taxonomy phrase (idea.taxonomy_phrase field)
    - "idea_without_template_prefix": Embed idea text with template_prefix stripped
    - "both": Embed BOTH idea text AND taxonomy_phrase (dual embeddings)
    """
    embedding_text_format: Literal["idea", "taxonomy_phrase", "idea_without_template_prefix", "both"] = "idea"


# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing settings.

    Provider-specific batch sizes and concurrency limits for optimal
    throughput while respecting rate limits.
    """
    # OpenAI settings
    openai_batch_size: int = 100            # Texts per batch for OpenAI
    openai_max_concurrent: int = 5          # Max concurrent OpenAI requests

    # Gemini settings
    gemini_batch_size: int = 20             # Texts per batch for Gemini
    gemini_max_concurrent: int = 10         # Max concurrent Gemini requests


# =============================================================================
# QUESTION-AWARE EMBEDDING CONFIGURATION
# =============================================================================

@dataclass
class QuestionAwareConfig:
    """Configuration for question-aware embedding transformation.

    Combines response embeddings with question embeddings and domain anchors
    to create context-aware representations.
    """
    use_question_aware: bool = False        # Enable question-aware transformation

    # Weights for combining embeddings (should sum to 1.0)
    response_weight: float = 0.6            # Weight for response embeddings
    question_weight: float = 0.3            # Weight for question embeddings
    domain_anchor_weight: float = 0.1       # Weight for domain-relative positioning


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for embedding quality analysis.

    Controls whether to compute quality metrics and statistics
    during embedding generation.
    """
    analyze_embeddings: bool = False        # Enable embedding quality analysis
    compute_similarity_stats: bool = False  # Compute pairwise similarity statistics
    max_embeddings_for_similarity: int = 1000  # Max embeddings for pairwise computation


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for API retry behavior."""
    retry_backoff_base: float = 0.8         # Base multiplier for exponential backoff
    default_retries: int = 3                # Default retry attempts for API calls


# =============================================================================
# COMBINED EMBEDDER CONFIGURATION
# =============================================================================

@dataclass
class EmbedderConfig:
    """Combined configuration for the Embedder.

    Aggregates all embedder settings into a single config object.
    This is the main config class used by the Embedder.
    """
    # Text format settings
    embedding_text_format: Literal["idea", "taxonomy_phrase", "idea_without_template_prefix", "both"] = "idea"

    # Provider selection
    provider: str = "openai"                # "openai" or "gemini"

    # OpenAI batch settings
    openai_batch_size: int = 100
    openai_max_concurrent: int = 5

    # Gemini batch settings
    gemini_batch_size: int = 20
    gemini_max_concurrent: int = 10

    # Question-aware settings
    use_question_aware: bool = False
    response_weight: float = 0.6
    question_weight: float = 0.3
    domain_anchor_weight: float = 0.1

    # Analysis settings
    analyze_embeddings: bool = False
    compute_similarity_stats: bool = False
    max_embeddings_for_similarity: int = 1000

    # Retry settings
    retry_backoff_base: float = 0.8
    default_retries: int = 3

    # Verbose output
    verbose: bool = True


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

DEFAULT_TEXT_FORMAT_CONFIG = TextFormatConfig()
DEFAULT_BATCH_PROCESSING_CONFIG = BatchProcessingConfig()
DEFAULT_QUESTION_AWARE_CONFIG = QuestionAwareConfig()
DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()
DEFAULT_RETRY_CONFIG = RetryConfig()
DEFAULT_EMBEDDER_CONFIG = EmbedderConfig()


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Configuration for taxonomy phrase embedding
TAXONOMY_PHRASE_CONFIG = EmbedderConfig(
    embedding_text_format="taxonomy_phrase",
    analyze_embeddings=True,
    compute_similarity_stats=True,
)

# Configuration for idea embedding without template prefix
IDEA_WITHOUT_PREFIX_CONFIG = EmbedderConfig(
    embedding_text_format="idea_without_template_prefix",
    analyze_embeddings=True,
    compute_similarity_stats=True,
)

# Configuration for dual embedding (both idea and taxonomy_phrase)
BOTH_EMBEDDINGS_CONFIG = EmbedderConfig(
    embedding_text_format="both",
    analyze_embeddings=True,
    compute_similarity_stats=True,
)

# Configuration for question-aware embedding
QUESTION_AWARE_EMBEDDER_CONFIG = EmbedderConfig(
    use_question_aware=True,
    response_weight=0.6,
    question_weight=0.3,
    domain_anchor_weight=0.1,
    analyze_embeddings=True,
)

# Configuration for Gemini provider
GEMINI_EMBEDDER_CONFIG = EmbedderConfig(
    provider="gemini",
    gemini_batch_size=20,
    gemini_max_concurrent=10,
)
