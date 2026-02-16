"""
Experimental Configuration for Step 4: Embedder

Copied from production config_embedder.py — tweak settings here
without affecting production runs.

Usage:
    Set USE_EXPERIMENTAL = True in run_experiment.py to use this config.
"""
from dataclasses import dataclass
from typing import Literal

EmbeddingTextFormat = Literal[
    "idea", "node", "category", "taxonomy", "all"
]


# =============================================================================
# MULTI-PASS EMBEDDING SPECIFICATIONS
# =============================================================================

@dataclass
class EmbeddingPass:
    """Specification for a single embedding pass in multi-pass modes."""
    text_format: str      # Format key for _get_text_for_embedding
    target_field: str     # Field on EmbeddingsSubmodel to store result
    label: str            # Human-readable label for logging


MULTI_PASS_SPECS = {
    "all": [
        EmbeddingPass("idea",     "idea_embedding",     "idea text (template_prefix + idea)"),
        EmbeddingPass("node",     "node_embedding",     "node (canonical concept)"),
        EmbeddingPass("category", "category_embedding", "semantic_category"),
        EmbeddingPass("taxonomy", "taxonomy_embedding", "taxonomy chain (node → category_label → semantic_category → root)"),
    ],
}


@dataclass
class EmbedderConfigExp:
    """Experimental embedder config — modify freely.

    Production defaults copied from config_embedder.py EmbedderConfig.
    """
    # Text format: "idea", "node", "category", "taxonomy", "all"
    embedding_text_format: EmbeddingTextFormat = "all"
    # Provider: "openai" or "gemini"
    provider: str = "openai"

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
    analyze_embeddings: bool = True
    compute_similarity_stats: bool = True
    max_embeddings_for_similarity: int = 1000

    # Retry settings
    retry_backoff_base: float = 0.8
    default_retries: int = 3

    # Verbose output
    verbose: bool = True