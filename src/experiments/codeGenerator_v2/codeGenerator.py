"""
CodeGenerator V2 Experiment - Thin Wrapper

This module re-exports the production InductiveCodeGenerator and related classes.
For experiments, modify prompts.py and inject via run_experiment.py.

Usage:
    from experiments.codeGenerator_v2.codeGenerator import InductiveCodeGenerator

For experiment-specific extensions, add them to this file as needed.
"""

# Re-export production classes for experiment use
from utils.codeGenerator import (
    # Main generator class
    InductiveCodeGenerator,

    # Codebook management
    SharedCodebook,

    # Pydantic models for structured outputs
    ClusterSummaryOutput,
    ClusterSummaryItem,
    ClusterThemeItem,
    AssignmentExamples,
    NearNeighbor,

    # Result models
    CodeGeneratorReasoningResults,

    # Coding decision models
    CodingDecisionOutput,
    GeneratedCode,
    CodeValidation,
)


# =============================================================================
# EXPERIMENT-SPECIFIC EXTENSIONS (add as needed)
# =============================================================================

# Example: Custom generator with overridden behavior
# class ExperimentalCodeGenerator(InductiveCodeGenerator):
#     """
#     Extend InductiveCodeGenerator for experiments.
#
#     Override specific methods to test different approaches
#     while keeping the rest of the pipeline unchanged.
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # Add custom initialization here
#
#     # Override specific methods as needed
#     # async def extract_themes(self, clusters):
#     #     # Custom theme extraction logic
#     #     pass
