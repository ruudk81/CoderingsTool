"""
Configuration for codeGenerator_v2 experiment.

This module provides experiment configuration settings for running
the codeGenerator in isolation for testing and prompt modification.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ExperimentConfig:
    """Configuration for the codeGenerator experiment."""

    # Dataset settings (matching pipeline.py selection)
    filename: str = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column: str = "DLNMID"
    var_name: str = "Q20"
    sample_size: int = 500

    # Generator settings (passed to InductiveCodeGenerator)
    stages_to_run: str = 'all'  # 'all' or 'theme_extraction_only'
    verbose: bool = True
    verbose_detailed: bool = False
    prompt_printer_enabled: bool = False

    # Language setting
    language: str = "nl"

    # Output settings
    save_results_to_file: bool = True
    sample_codebook_count: int = 0  # Number of codes to display (0 = show all)


# Default configuration - modify this for your experiments
EXPERIMENT_CONFIG = ExperimentConfig(
    # Dataset settings
    filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
    id_column="DLNMID",
    var_name="Q20",
    sample_size=500,

    # Alternative datasets (uncomment to use):
    # filename="M250480 Associatiemonitor ASN Bank net databestand.sav",
    # var_name="Qd1_combined",
    # sample_size=2000,

    # Generator settings
    stages_to_run='all',
    verbose=True,
    verbose_detailed=False,
    prompt_printer_enabled=False,

    # Output settings
    save_results_to_file=True,
    sample_codebook_count=0,  # 0 = show all codes
)
