"""
Experimental Configuration for Step 4: Embedder

v5-aligned embedding formats and multi-pass specifications.

Available single-pass formats (stored in idea_embedding):
    "idea"            — idea text as-is (natural sentence incl. template_prefix)
    "idea_bare"       — idea with template_prefix stripped
    "concept"         — canonical concept noun phrase
    "concept_type"    — discovered concept type
    "concept_defined"      — concept → concept_type_definition
    "concept_typed"        — concept (concept_type)
    "idea_concept_defined" — idea → concept → concept_type_definition
    "ladder"               — instance → concept → concept_type → concept_type_definition

Available multi-pass formats (each pass stored in its own field):
    "default"         — 4 passes: idea, ladder, concept_defined, idea_concept_defined
    "all"             — 4 passes: idea, concept, concept_type, ladder

Usage:
    Set USE_EXPERIMENTAL = True in run_experiment.py to use this config.
"""
from dataclasses import dataclass
from typing import Literal

EmbeddingTextFormat = Literal[
    # Single-pass (stored in idea_embedding)
    "idea", "idea_bare", "concept", "concept_type",
    "concept_defined", "concept_typed", "idea_concept_defined", "ladder",
    # Multi-pass (each pass stored in its own field)
    "default", "all",
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
    "default": [
        EmbeddingPass("idea",                 "idea_embedding",                  "idea (natural sentence)"),
        EmbeddingPass("ladder",               "ladder_embedding",                "abstraction ladder (instance → concept → concept_type → concept_type_definition)"),
        EmbeddingPass("concept_defined",      "concept_embedding",               "concept → concept_type_definition"),
        EmbeddingPass("idea_concept_defined", "idea_concept_defined_embedding",  "idea → concept → concept_type_definition"),
    ],
    "all": [
        EmbeddingPass("idea",         "idea_embedding",         "idea (natural sentence)"),
        EmbeddingPass("concept",      "concept_embedding",      "concept (canonical noun phrase)"),
        EmbeddingPass("concept_type", "concept_type_embedding", "concept_type"),
        EmbeddingPass("ladder",       "ladder_embedding",       "abstraction ladder"),
    ],
}


# =============================================================================
# EXPERIMENTAL EMBEDDER CONFIGURATION
# =============================================================================

@dataclass
class EmbedderConfigExp:
    """Experimental embedder config — modify freely."""

    # Text format (see module docstring for options)
    embedding_text_format: EmbeddingTextFormat = "default"

    # OpenAI batch settings
    openai_batch_size: int = 100
    openai_max_concurrent: int = 5

    # Analysis settings
    analyze_embeddings: bool = True
    compute_similarity_stats: bool = True
    max_embeddings_for_similarity: int = 1000

    # Retry settings
    retry_backoff_base: float = 0.8
    default_retries: int = 3

    # Verbose output
    verbose: bool = True
