"""
Embedder-specific configuration — v5-aligned text formats and multi-pass specs.

Available single-pass formats (stored in idea_embedding):
    "idea"            — idea text as-is (natural sentence incl. template_prefix)
    "idea_bare"       — idea with template_prefix stripped
    "interpretation"  — interpretation (ladder rung 2)
    "abstraction"     — abstraction (ladder rung 3)
    "ladder"          — instance → interpretation → abstraction

Available multi-pass formats (each pass stored in its own field):
    "default"         — 3 passes: idea, ladder, interpretation
    "all"             — 4 passes: idea, interpretation, abstraction, ladder
"""
from dataclasses import dataclass
from typing import Literal, Optional

# Shared type for all embedding text format options
EmbeddingTextFormat = Literal[
    # Single-pass (stored in idea_embedding)
    "idea", "idea_bare", "interpretation", "abstraction", "ladder",
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
        EmbeddingPass("idea",           "idea_embedding",           "idea (natural sentence)"),
        EmbeddingPass("ladder",         "ladder_embedding",         "abstraction ladder (instance → interpretation → abstraction)"),
        EmbeddingPass("interpretation", "interpretation_embedding", "interpretation (ladder rung 2)"),
    ],
    "all": [
        EmbeddingPass("idea",           "idea_embedding",           "idea (natural sentence)"),
        EmbeddingPass("interpretation", "interpretation_embedding", "interpretation (ladder rung 2)"),
        EmbeddingPass("abstraction",    "abstraction_embedding",    "abstraction (ladder rung 3)"),
        EmbeddingPass("ladder",         "ladder_embedding",         "abstraction ladder"),
    ],
}


# =============================================================================
# EMBEDDER CONFIGURATION
# =============================================================================

@dataclass
class EmbedderConfig:
    """Combined configuration for the Embedder.

    Default settings: 3-pass "default" mode with analysis enabled.
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
    """Format abstraction ladder: instance → interpretation → abstraction."""
    parts = []
    for field in ('instance', 'interpretation', 'abstraction'):
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
        idea: Object with idea/instance/interpretation/abstraction fields.
        fmt: One of:
            - Named: "idea", "idea_bare", "interpretation", "abstraction", "ladder"
            - Composite: "interpretation+abstraction", "idea+interpretation", etc.
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

    if fmt == "interpretation":
        val = (getattr(idea, 'interpretation', '') or '').strip()
        return val if val else idea.idea

    if fmt == "abstraction":
        val = (getattr(idea, 'abstraction', '') or '').strip()
        return val if val else idea.idea

    if fmt == "ladder":
        return _format_ladder(idea, separator)

    # Default: "idea" — full idea text (natural sentence incl. template_prefix)
    return idea.idea
