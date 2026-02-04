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
    "idea", "taxonomy_phrase", "idea_without_template_prefix",
    "both_taxonomy_phrase", "ontology", "both_ontology", "all"
]

BOTH_MODE_IDEA_FORMAT: Literal["idea", "idea_without_template_prefix"] = "idea_without_template_prefix"


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
    "both_taxonomy_phrase": [
        EmbeddingPass(BOTH_MODE_IDEA_FORMAT, "idea_embedding", "idea text"),
        EmbeddingPass("taxonomy_phrase", "taxonomy_embedding", "taxonomy_phrase"),
    ],
    "both_ontology": [
        EmbeddingPass(BOTH_MODE_IDEA_FORMAT, "idea_embedding", "idea text"),
        EmbeddingPass("ontology", "ontology_embedding", "ontology string"),
    ],
    "all": [
        EmbeddingPass(BOTH_MODE_IDEA_FORMAT, "idea_embedding", "idea text"),
        EmbeddingPass("taxonomy_phrase", "taxonomy_embedding", "taxonomy_phrase"),
        EmbeddingPass("ontology", "ontology_embedding", "ontology string"),
    ],
}


@dataclass
class EmbedderConfigExp:
    """Experimental embedder config — modify freely.

    Production defaults copied from config_embedder.py EmbedderConfig.
    """
    # Text format: "idea", "taxonomy_phrase", "idea_without_template_prefix",
    #              "both_taxonomy_phrase", "ontology", "both_ontology", "all"
    embedding_text_format: EmbeddingTextFormat = "both_ontology"

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