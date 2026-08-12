"""
Configuration for Code Generator (P8-P9).

Pipeline: code generation from attributes → codebook consolidation.
"""

from dataclasses import dataclass
from config import get_step_model


@dataclass
class CodebookConfig:
    """Configuration for Codebook Generation (P8-P9)."""

    # LLM settings (derived from MODEL_FAMILY toggle)
    model_p8: str = get_step_model("codegen_p8")  # P8: Code Generation from Attributes
    model_p9: str = get_step_model("codegen_p9")  # P9: Codebook Consolidation
    temperature_p8: float = 0.3
    temperature_p9: float = 0.0

    # P8: Code Generation from Attributes (per-domain)
    max_tokens_code_from_attributes: int = 16000

    # P9: Codebook Consolidation (cross-domain review)
    max_tokens_codebook_consolidation: int = 16000

    # Embedding-based representative samples
    code_source: str = "instance_interpretation"  # Text format for embedding: idea, instance, instance_interpretation, full_abstraction_ladder
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100
    embedding_max_concurrent: int = 5
    max_representative_samples: int = 3  # Max samples per attribute per valence group

    # Output
    verbose: bool = True

    # Prevalentiedrempel: een concept krijgt een eigen code als het door minstens
    # dit aandeel van de respondenten wordt genoemd (1% → later eventueel 5% als
    # het codeboek te veel fragmenteert). Let op: deze drempel en het Overig-
    # plafond van 10% bewegen tegen elkaar in — hoger hier betekent meer Overig.
    t_keep_share: float = 0.01
    t_keep_min_respondents: int = 3
