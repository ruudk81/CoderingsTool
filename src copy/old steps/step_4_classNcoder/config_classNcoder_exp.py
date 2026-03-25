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
    label_source: str = "idea"

    # Optional prefix prepended to each label string before processing.
    # "" = no prefix (default)
    # Any literal string = static prefix for all labels
    label_prefix: str = ""

    # Prepend a valence tag ([+], [-], [0]) to each label.
    # Useful so the LLM can distinguish positive/negative observations during
    # facet discovery (P1), facet assignment (P3), and attribute discovery (P4).
    include_valence: bool = False

    # ==========================================================================
    # QUALITATIVE RESEARCHER PIPELINE (v3: 10 stages)
    # ==========================================================================

    # LLM settings — per-stage model selection
    qr_model_p1: str = "gpt-4.1-mini"       # P1: Facet Discovery
    qr_model_p2: str = "gpt-4.1"            # P2: Facet Consolidation
    qr_model_p3: str = "gpt-4.1-nano"       # P3: Facet Assignment (classification)
    qr_model_p4: str = "gpt-4.1-mini"       # P4: Attribute Discovery
    qr_model_p5: str = "gpt-4.1"            # P5: Attribute Chunk Consolidation
    qr_model_p6: str = "gpt-4.1-nano"       # P6: Attribute Assignment (classification)
    qr_model_p7: str = "gpt-4.1-mini"       # P7: Cross-facet Attribute Consolidation
    qr_model_p8: str = "gpt-4.1"            # P8: Code Generation
    qr_model_p9: str = "gpt-4.1"            # P9: Codebook Consolidation
    qr_temperature: float = 0.3

    # P1: Facet Discovery (per-domain, chunked)
    qr_max_tokens_facet_discovery: int = 4000

    # P3: Facet Assignment (per-domain, batched)
    qr_max_tokens_facet_assignment: int = 4000
    facet_assignment_batch_size: int = 10  # ideas per assignment call (nano-friendly)

    # P4: Attribute Discovery (per facet within domain)
    qr_max_tokens_attribute_discovery: int = 4000

    # P8: Code Generation from Attributes (per-domain, valence-split)
    qr_max_tokens_code_from_attributes: int = 16000

    # P9: Codebook Consolidation (cross-domain review)
    qr_max_tokens_codebook_consolidation: int = 16000

    # Adaptive batching for P1 (facet discovery chunks)
    batch_size_min: int = 100      # no splitting below this (single batch)
    batch_size_max: int = 150      # ceiling: keeps prompt quality high
    target_batches: int = 6        # ideal number of chunks
    chunk_overlap: float = 0.2     # overlap fraction between adjacent chunks

    # Adaptive batching for P4 (attribute discovery chunks within a facet)
    p4_batch_size_min: int = 100   # no splitting below this (single batch)
    p4_batch_size_max: int = 150   # ceiling per chunk
    p4_target_batches: int = 5     # ideal number of chunks
    p4_chunk_overlap: float = 0.2  # overlap fraction between adjacent chunks

    # Hierarchical consolidation (shared by P2 and P5)
    # When chunk count or total item count exceeds these limits,
    # consolidation becomes hierarchical: group → consolidate → recurse.
    consolidation_max_chunks_per_call: int = 6   # Rule 2: max chunks per consolidation call
    consolidation_max_items_per_call: int = 150  # Rule 3: max total items per consolidation call
    consolidation_max_rounds: int = 5            # safety cap on recursive rounds

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

    # Fallback category for ideas that don't clearly fit any MECE category.
    # Resolved by language from extraction_metadata.lang.
    include_other_category: bool = True

    # Embedding pre-filtering (scopes codebook to top-N codes per idea)
    use_embedding_prefilter: bool = True
    embedding_top_n: int = 5
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100
    embedding_max_concurrent: int = 5

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
