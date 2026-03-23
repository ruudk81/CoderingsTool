"""
Data models for Code Assigner (P10).

Code assignment output models — final pipeline output.
Imports taxonomy types from step 4 (upstream dependency).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from development.step_3_ideaExtractor.models_exp import (
    IdeasExtractedSubmodel,
    IdeasExtractedModel,
)


# --- Internal wrapper models for code assignment (data-flow, not LLM response) ---

class CodeAssignment(BaseModel):
    """Single idea-to-code assignment (internal wrapper)."""
    idea_id: str = Field(..., description="The idea_id from the input")
    assigned_code_id: str = Field(
        ..., description="The code ID from [C#] prefix (e.g. 'C1', 'C7'). ONLY the ID."
    )
    confidence: float = Field(..., description="Confidence (0.0 to 1.0)")
    rationale: str = Field(..., description="Brief rationale")


class CodeAssignmentBatch(BaseModel):
    """Batch wrapper for uniform downstream handling."""
    assignments: List[CodeAssignment] = Field(
        ..., description="One assignment per idea"
    )


class CodeAssignedSubmodel(IdeasExtractedSubmodel):
    """Per-idea data with code + attribute assignment.

    Extends step 3's IdeasExtractedSubmodel.
    Populated by: facet (L3) from step 4, attribute (L4) from step 4,
    plus code assignment fields from step 6.
    """
    assigned_code: Optional[str] = None
    assigned_attribute: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    partition_name: Optional[str] = None


class CodeAssignedModel(IdeasExtractedModel):
    """Response-level model with code-assigned ideas."""
    response_ideas: Optional[List[CodeAssignedSubmodel]] = None
    assignment_metadata: Optional[Dict[str, Any]] = None
