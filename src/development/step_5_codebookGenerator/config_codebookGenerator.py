"""
Configuration for Codebook Generator (P8-P10).

Pipeline: code generation from attributes → codebook consolidation → code assignment.
"""

from dataclasses import dataclass
from typing import Dict


# =============================================================================
# CODEBOOK GENERATION CONFIG (P8-P9)
# =============================================================================

@dataclass
class CodebookConfig:
    """Configuration for Codebook Generation (P8-P9)."""

    # LLM settings
    model_p8: str = "gpt-4.1"            # P8: Code Generation from Attributes
    model_p9: str = "gpt-4.1"            # P9: Codebook Consolidation
    temperature: float = 0.3

    # P8: Code Generation from Attributes (per-domain)
    max_tokens_code_from_attributes: int = 16000

    # P9: Codebook Consolidation (cross-domain review)
    max_tokens_codebook_consolidation: int = 16000

    # Hierarchical consolidation for P9
    consolidation_max_chunks_per_call: int = 6
    consolidation_max_items_per_call: int = 150
    consolidation_max_rounds: int = 5

    # Output
    verbose: bool = True


# =============================================================================
# CODE ASSIGNMENT CONFIG (P10)
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


# =============================================================================
# OTHER CATEGORY LABELS
# =============================================================================

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
