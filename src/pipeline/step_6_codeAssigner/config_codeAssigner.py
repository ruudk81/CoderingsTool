"""
Configuration for Code Assigner (P10).

Assigns each idea to exactly one MECE code from the codebook.
"""

from dataclasses import dataclass
from typing import Dict
from config import get_step_model


@dataclass
class AssignmentConfig:
    """Configuration for MECE category assignment to individual ideas."""

    # LLM settings (derived from MODEL_FAMILY toggle)
    assignment_model: str = get_step_model("code_assignment")
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

    # Consistency binding (same word → same single code).
    # Group identical/near-identical instances before assigning, code one
    # representative per cluster, broadcast that code to all members. Removes
    # sampling-noise divergence at the source and saves LLM calls.
    bind_enabled: bool = True
    bind_use_embeddings: bool = True        # False = exact-normalized grouping only
    bind_cosine_threshold: float = 0.85     # near-duplicate instance merge (measured: variants ~0.85-0.91, distinct ≤0.56)
    bind_min_cluster_size: int = 2          # only bind groups of ≥ this size

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
