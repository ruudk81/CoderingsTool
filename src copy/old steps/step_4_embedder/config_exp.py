"""
Experimental Configuration for Step 4: Embedder

v5-aligned embedding formats with rung_1/rung_2 terminology (from step 3).

Available single-pass formats (stored in idea_embedding):
    "idea"              — idea text as-is (natural sentence incl. template_prefix)
    "idea_bare"         — idea with template_prefix stripped
    "rung_1"            — concrete interpretation (what it means)
    "rung_2"            — broader significance (why it matters)
    "concept_type"      — discovered concept type
    "rung_1_defined"    — rung_1 → rung_2
    "rung_1_typed"      — rung_1 (concept_type)
    "idea_rung_1_defined" — idea → rung_1 → rung_2
    "ladder"            — instance → rung_1 → rung_2

Available multi-pass formats (each pass stored in its own field):
    "default"         — 4 passes: idea, ladder, rung_1_defined, idea_rung_1_defined
    "all"             — 5 passes: idea, rung_1, rung_2, concept_type, ladder

Usage:
    Set USE_EXPERIMENTAL = True in run_experiment.py to use this config.
"""
from dataclasses import dataclass
from typing import Literal, Optional

EmbeddingTextFormat = Literal[
    # Single-pass (stored in idea_embedding)
    "idea", "idea_bare", "rung_1", "rung_2", "concept_type",
    "rung_1_defined", "rung_1_typed", "idea_rung_1_defined", "ladder",
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
        EmbeddingPass("idea",               "idea_embedding",          "idea (natural sentence)"),
        EmbeddingPass("ladder",             "ladder_embedding",        "abstraction ladder (instance → rung_1 → rung_2)"),
        EmbeddingPass("rung_1_defined",     "rung_1_embedding",        "rung_1 → rung_2"),
        EmbeddingPass("idea_rung_1_defined", "rung_2_embedding",       "idea → rung_1 → rung_2"),
    ],
    "all": [
        EmbeddingPass("idea",         "idea_embedding",         "idea (natural sentence)"),
        EmbeddingPass("rung_1",       "rung_1_embedding",       "rung_1 (concrete interpretation)"),
        EmbeddingPass("rung_2",       "rung_2_embedding",       "rung_2 (broader significance)"),
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


# =============================================================================
# STANDALONE TEXT FORMATTING FOR EMBEDDING
# =============================================================================

def _format_ladder(idea, separator: str = " → ") -> str:
    """Format abstraction ladder: instance → rung_1 → rung_2."""
    parts = []
    for field in ('instance', 'rung_1', 'rung_2'):
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

    Extracted from embedder_exp.py _get_text_for_embedding() for use by any step.

    Args:
        idea: Object with idea/instance/rung_1/rung_2/concept_type fields.
        fmt: One of:
            - Named: "idea", "idea_bare", "rung_1", "rung_2", "concept_type",
              "rung_1_typed", "rung_1_defined", "idea_rung_1_defined", "ladder"
            - Composite: "rung_1+rung_2", "idea+rung_1", etc.
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

    # --- Named formats (mirrors embedder_exp.py _get_text_for_embedding) ---
    if fmt == "idea_bare":
        if template_prefix and idea.idea.startswith(template_prefix):
            stripped = idea.idea[len(template_prefix):].strip()
            return stripped if stripped else idea.idea
        return idea.idea

    if fmt == "rung_1":
        val = (getattr(idea, 'rung_1', '') or '').strip()
        return val if val else idea.idea

    if fmt == "rung_2":
        val = (getattr(idea, 'rung_2', '') or '').strip()
        return val if val else idea.idea

    if fmt == "concept_type":
        val = (getattr(idea, 'concept_type', '') or '').strip()
        return val if val else idea.idea

    if fmt == "rung_1_typed":
        rung_1 = (getattr(idea, 'rung_1', '') or '').strip()
        concept_type = (getattr(idea, 'concept_type', '') or '').strip()
        if rung_1 and concept_type:
            return f"{rung_1} ({concept_type})"
        return rung_1 or idea.idea

    if fmt == "rung_1_defined":
        rung_1 = (getattr(idea, 'rung_1', '') or '').strip()
        rung_2 = (getattr(idea, 'rung_2', '') or '').strip()
        if rung_1 and rung_2:
            return f"{rung_1}{separator}{rung_2}"
        return rung_1 or idea.idea

    if fmt == "idea_rung_1_defined":
        rung_1 = (getattr(idea, 'rung_1', '') or '').strip()
        rung_2 = (getattr(idea, 'rung_2', '') or '').strip()
        parts = [idea.idea]
        if rung_1:
            parts.append(rung_1)
        if rung_2:
            parts.append(rung_2)
        return separator.join(parts)

    if fmt == "ladder":
        return _format_ladder(idea, separator)

    # Default: "idea" — full idea text (natural sentence incl. template_prefix)
    return idea.idea
