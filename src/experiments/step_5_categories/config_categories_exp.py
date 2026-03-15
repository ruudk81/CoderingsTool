"""
Configuration for Category Discovery v3.

Pipeline: facet discovery → facet assignment → attribute discovery →
code generation from attributes → code assignment.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CategoriesConfig:
    """Configuration for Category Discovery v3."""

    # ==========================================================================
    # PARTITION SOURCE
    # ==========================================================================

    PARTITION_SOURCE = "domain"

    # ==========================================================================
    # LABEL SOURCE
    # ==========================================================================

    # Which text to collect as "observations" for facet/attribute discovery input.
    #
    # Stored fields (direct attributes on IdeasExtractedSubmodel from step 3):
    #   "instance"        — Attribute (L4): verbatim span from response
    #   "interpretation"  — Ladder rung 2: concrete meaning (survey language)
    #   "abstraction"     — Ladder rung 3: broader significance (survey language)
    #   "facet"           — Facet (L3): dimension-specific aspect
    #   "domain"          — Domain (L2): thematic domain
    #   "idea"            — full idea text incl. template prefix
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
    # QUALITATIVE RESEARCHER PIPELINE (v3: 4 phases)
    # ==========================================================================

    # LLM settings
    qr_model: str = "gpt-4.1"
    qr_temperature: float = 0.3

    # P1: Facet Discovery (per-domain, chunked)
    qr_max_tokens_facet_discovery: int = 4000

    # P2: Facet Assignment (per-domain, batched)
    qr_max_tokens_facet_assignment: int = 4000
    facet_assignment_batch_size: int = 15  # ideas per assignment call

    # P3: Attribute Discovery (per facet within domain)
    qr_max_tokens_attribute_discovery: int = 4000

    # P4: Code Generation from Attributes (cross-domain)
    qr_max_tokens_code_from_attributes: int = 16000

    # Adaptive batching for P1 (facet discovery chunks)
    batch_size_min: int = 30       # floor: enough observations for discovery
    batch_size_max: int = 100      # ceiling: keeps prompt quality high
    target_batches: int = 15       # ideal number of chunks
    chunk_overlap: float = 0.2     # overlap fraction between adjacent chunks

    # Facet consolidation: when unique facets exceed this count after
    # programmatic dedup, an LLM consolidation pass merges near-duplicates.
    consolidation_chunk_size: int = 45   # threshold to trigger LLM consolidation
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
