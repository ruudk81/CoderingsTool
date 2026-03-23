"""
Configuration for Code Generator (P8-P9).

Pipeline: code generation from attributes → codebook consolidation.
"""

from dataclasses import dataclass


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
