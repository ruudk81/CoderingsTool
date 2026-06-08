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

    # Offer the LLM an explicit "no code fits" option so it is never forced to
    # mis-assign. Choosing it resolves to the __UNASSIGNED__ sentinel — step 6
    # never invents a catch-all label that is absent from step 5's codebook.
    allow_no_fit: bool = True

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
# NO-FIT OPTION LABELS
# =============================================================================

# Language → display phrase for the "no code fits" prompt option. This is shown
# to the LLM only; when chosen it resolves to __UNASSIGNED__, never to a label
# written into assigned_code.
NO_FIT_LABELS: Dict[str, str] = {
    "Dutch": "geen passende code",
    "nl-NL": "geen passende code",
    "English": "no matching code",
    "en-GB": "no matching code",
    "en-US": "no matching code",
    "German": "keine passende Kategorie",
    "de-DE": "keine passende Kategorie",
    "French": "aucun code correspondant",
    "fr-FR": "aucun code correspondant",
    "Spanish": "ningún código aplicable",
    "es-ES": "ningún código aplicable",
}
NO_FIT_DEFAULT = "no matching code"


def get_no_fit_label(language: str) -> str:
    """Resolve the no-fit display phrase for a given language."""
    return NO_FIT_LABELS.get(language, NO_FIT_DEFAULT)
