"""
Configuration for Category Discovery v2.

Pipeline: per-partition chunked theme discovery → theme consolidation →
concept discovery → codebook construction → category assignment.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CategoriesConfig:
    """Configuration for Category Discovery."""

    # ==========================================================================
    # PARTITION SOURCE
    # ==========================================================================

    PARTITION_SOURCE = "domain"

    # ==========================================================================
    # LABEL SOURCE
    # ==========================================================================

    # Which text to collect as "labels" for theme discovery input.
    #
    # Stored fields (direct attributes on IdeasExtractedSubmodel from step 3):
    #   "interpretation" — concrete interpretation (what it means)
    #   "abstraction"    — broader significance (why it matters)
    #   "domain"         — discovered domain (L2), e.g., "recommendation"
    #   "idea"           — full idea text incl. template prefix
    #   "instance"       — verbatim span from response
    #
    # Computed composites (assembled from stored fields by format_label()):
    #   "ladder"     — instance → interpretation → abstraction
    #   "idea_rungs" — idea → interpretation → abstraction
    label_source: str = "ladder"

    # Optional prefix prepended to each label string before processing.
    # "" = no prefix (default)
    # Any literal string = static prefix for all labels
    label_prefix: str = ""

    # ==========================================================================
    # QUALITATIVE RESEARCHER PIPELINE
    # ==========================================================================

    # LLM settings
    qr_model: str = "gpt-4.1"
    qr_temperature: float = 0.3
    qr_max_tokens_theme_discovery: int = 4000
    qr_max_tokens_consolidation: int = 4000          # Prompt 1.5: per-partition theme consolidation
    qr_max_tokens_concept_discovery: int = 4000      # Prompt 2a: concept discovery
    qr_max_tokens_coc_consolidation: int = 4000      # Prompt 3: cross-partition COC consolidation
    qr_max_tokens_hierarchical_codebook: int = 8000  # Prompt 4: hierarchical codebook construction
    qr_max_tokens_codebook_construction: int = 4000  # Prompt 2b: per-partition codebook (legacy)
    qr_max_tokens_thematic_analysis: int = 8000      # Legacy prompt 2 (deprecated)

    # Adaptive batching: batch size scales with partition size to keep
    # theme discovery chunk count in a productive range (~5-20 chunks).
    #   n ≤ 30  → 1 chunk
    #   n ~ 500 → ~15 chunks of ~33
    #   n ~ 2000 → ~20 chunks of 100 (hits ceiling)
    batch_size_min: int = 30       # floor: enough labels for theme discovery
    batch_size_max: int = 100      # ceiling: keeps prompt quality high
    target_batches: int = 15       # ideal number of chunks
    chunk_overlap: float = 0.2     # overlap fraction between adjacent chunks

    # Theme consolidation: hierarchical chunking
    # When unique themes exceed this count, split into chunks and consolidate
    # in successive rounds until the list fits in a single call.
    consolidation_chunk_size: int = 45   # max themes per consolidation call
    consolidation_max_rounds: int = 5    # safety cap on recursive rounds

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True


# =============================================================================
# PRESETS
# =============================================================================

DEFAULT_CONFIG = CategoriesConfig()


# =============================================================================
# CATEGORY ASSIGNMENT CONFIG
# =============================================================================

@dataclass
class AssignmentConfig:
    """Configuration for MECE category assignment to individual ideas."""

    # LLM settings
    assignment_model: str = "gpt-4.1-mini"
    assignment_temperature: float = 0.1    # Low for consistent assignment
    assignment_max_tokens: int = 4000

    # Assignment mode: "single" (one idea per call, flat codebook) or
    # "batch" (N ideas per call, hierarchical codebook)
    assignment_mode: str = "single"

    # Batching: ideas per LLM call (only used in "batch" mode)
    assignment_batch_size: int = 10

    # Fallback category for ideas that don't clearly fit any MECE category.
    # Resolved by language from extraction_metadata.lang.
    include_other_category: bool = True

    # Output
    verbose: bool = True


# Language → "Other/Miscellaneous" label mapping
OTHER_CATEGORY_LABELS: Dict[str, str] = {
    "Dutch": "overig/anders",
    "nl-NL": "overig/anders",
    "English": "other/miscellaneous",
    "en-GB": "other/miscellaneous",
    "en-US": "other/miscellaneous",
    "German": "sonstiges",
    "de-DE": "sonstiges",
    "French": "autre/divers",
    "fr-FR": "autre/divers",
    "Spanish": "otro/varios",
    "es-ES": "otro/varios",
}
OTHER_CATEGORY_DEFAULT = "other/miscellaneous"


def get_other_category_label(language: str) -> str:
    """Resolve the Other category label for a given language."""
    return OTHER_CATEGORY_LABELS.get(language, OTHER_CATEGORY_DEFAULT)
