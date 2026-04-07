"""
Configuration for Code Generator (P8-P9).

Pipeline: code generation from attributes → codebook consolidation.
"""

from dataclasses import dataclass, field
from config import get_step_model
from pipeline.step_4_classifier.config_classifier import ClassifierRampConfig


@dataclass
class CodebookConfig:
    """Configuration for Codebook Generation (P8-P9)."""

    # LLM settings (derived from MODEL_FAMILY toggle)
    model_p8: str = get_step_model("codegen_p8")  # P8: Code Generation from Attributes
    model_p9: str = get_step_model("codegen_p9")  # P9: Codebook Consolidation
    temperature: float = 0.3

    # P8: Code Generation from Attributes (per-domain)
    max_tokens_code_from_attributes: int = 16000

    # P9: Codebook Consolidation (cross-domain review)
    max_tokens_codebook_consolidation: int = 16000

    # Hierarchical consolidation for P9
    consolidation_max_chunks_per_call: int = 6
    consolidation_max_items_per_call: int = 150
    consolidation_max_rounds: int = 5

    # Embedding-based representative samples
    code_source: str = "instance_interpretation"  # Text format for embedding: idea, instance, instance_interpretation, full_abstraction_ladder
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100
    embedding_max_concurrent: int = 5
    max_representative_samples: int = 3  # Max samples per attribute per valence group

    # 4-layer rate limiting (reuses ClassifierRampConfig)
    ramp_config: ClassifierRampConfig = field(default_factory=ClassifierRampConfig)

    # Output
    verbose: bool = True
