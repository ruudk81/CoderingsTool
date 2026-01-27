"""
codeGenerator V2 Module

NOTE: After migration, v2 is now the PRODUCTION codeGenerator.
Import from utils.codeGenerator instead.

This module provides backward-compatible aliases.
"""

from utils.codeGenerator import InductiveCodeGenerator, CodeGeneratorReasoningResults

__all__ = ["InductiveCodeGenerator", "CodeGeneratorReasoningResults"]
