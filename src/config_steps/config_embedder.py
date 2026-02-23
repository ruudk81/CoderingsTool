"""
Embedder-specific configuration — v5-aligned text formats and multi-pass specs.

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
"""
from dataclasses import dataclass
from typing import Literal, Optional

# Shared type for all embedding text format options
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
# EMBEDDER CONFIGURATION
# =============================================================================

@dataclass
class EmbedderConfig:
    """Combined configuration for the Embedder.

    Default settings: 4-pass "default" mode with analysis enabled.
    """
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


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

DEFAULT_EMBEDDER_CONFIG = EmbedderConfig()


# =============================================================================
# STANDALONE TEXT FORMATTING FOR EMBEDDING
# =============================================================================

def _format_ladder(idea, separator: str = " → ") -> str:
    """Format abstraction ladder: instance → concept → concept_type → concept_type_definition."""
    parts = []
    for field in ('instance', 'concept', 'concept_type', 'concept_type_definition'):
        val = (getattr(idea, field, '') or '').strip()
        if val:
            parts.append(val)
    return separator.join(parts) if parts else idea.idea


def _format_single_field(idea, field: str, separator: str = " → ") -> str:
    """Format a single named field from an idea object."""
    if field == "ladder":
        return _format_ladder(idea, separator)
    if field == "idea":
        return idea.idea
    return (getattr(idea, field, '') or '').strip()


def format_idea_text(
    idea,
    fmt: str,
    separator: str = " → ",
    template_prefix: Optional[str] = None,
) -> str:
    """Format idea text for embedding — standalone, no Embedder instance required.

    Extracted from embedder for use by any step.

    Args:
        idea: Object with idea/instance/concept/concept_type/concept_type_definition fields.
        fmt: One of:
            - Named: "idea", "idea_bare", "concept", "concept_type",
              "concept_typed", "concept_defined", "idea_concept_defined", "ladder"
            - Composite: "concept+concept_type_definition", "idea+concept", etc.
              Fields joined with `separator`.
        separator: Join string for multi-field and composite formats (default " → ").
        template_prefix: For "idea_bare" — prefix to strip from idea.idea.

    Returns:
        Formatted text string (never empty — falls back to idea.idea).
    """
    # --- Composite "field1+field2+..." syntax ---
    if "+" in fmt:
        parts = [_format_single_field(idea, f.strip(), separator) for f in fmt.split("+")]
        result = separator.join(p for p in parts if p)
        return result if result else idea.idea

    # --- Named formats ---
    if fmt == "idea_bare":
        if template_prefix and idea.idea.startswith(template_prefix):
            stripped = idea.idea[len(template_prefix):].strip()
            return stripped if stripped else idea.idea
        return idea.idea

    if fmt == "concept":
        val = (getattr(idea, 'concept', '') or '').strip()
        return val if val else idea.idea

    if fmt == "concept_type":
        val = (getattr(idea, 'concept_type', '') or '').strip()
        return val if val else idea.idea

    if fmt == "concept_typed":
        concept = (getattr(idea, 'concept', '') or '').strip()
        concept_type = (getattr(idea, 'concept_type', '') or '').strip()
        if concept and concept_type:
            return f"{concept} ({concept_type})"
        return concept or idea.idea

    if fmt == "concept_defined":
        concept = (getattr(idea, 'concept', '') or '').strip()
        definition = (getattr(idea, 'concept_type_definition', '') or '').strip()
        if concept and definition:
            return f"{concept}{separator}{definition}"
        return concept or idea.idea

    if fmt == "idea_concept_defined":
        concept = (getattr(idea, 'concept', '') or '').strip()
        definition = (getattr(idea, 'concept_type_definition', '') or '').strip()
        parts = [idea.idea]
        if concept:
            parts.append(concept)
        if definition:
            parts.append(definition)
        return separator.join(parts)

    if fmt == "ladder":
        return _format_ladder(idea, separator)

    # Default: "idea" — full idea text (natural sentence incl. template_prefix)
    return idea.idea
