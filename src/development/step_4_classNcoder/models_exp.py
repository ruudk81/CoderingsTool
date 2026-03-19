"""
Data models for Step 5 category discovery.

Partition models, MECE cache models, and category assignment output models.
All idea-level models extend step 3's IdeasExtractedSubmodel directly
(no embeddings dependency).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from development.step_3_ideaExtractor.models_exp import (
    IdeasExtractedSubmodel,
    IdeasExtractedModel,
)


# =============================================================================
# PARTITION MODELS (data-driven domain groups)
# =============================================================================

class DomainDescription(BaseModel):
    """Description of a domain partition."""
    partition_name: str = Field(
        ...,
        description="Concept type name (data-driven, e.g., 'recommendation', 'product_feature')"
    )
    inclusion_definition: str = Field(
        ...,
        description=(
            "What kinds of statements belong to this partition. "
            "Uses observable criteria."
        )
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a coder asks to determine if a statement "
            "belongs to this partition."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description="3-5 concrete words or phrases that indicate this partition"
    )


class DomainSet(BaseModel):
    """Complete set of domain partitions."""
    partitions: List[DomainDescription] = Field(
        ...,
        description="List of populated domain partitions"
    )


# =============================================================================
# MECE CACHE MODELS
# =============================================================================

class DomainResultModel(BaseModel):
    """Pydantic-serializable partition result for caching (v3)."""
    partition_name: str
    n_labels: int
    n_batches: int
    facets: List[Dict[str, Any]] = Field(default_factory=list)
    facet_assignments: Dict[str, str] = Field(default_factory=dict)
    attributes: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class CodingResultsCache(BaseModel):
    """Top-level cache wrapper for all category results."""
    partition_set: DomainSet
    partition_results: Dict[str, DomainResultModel]
    label_counts: Dict[str, int] = Field(default_factory=dict)
    label_source: str = ""
    total_categories: int = 0
    raw_codes: List[Dict] = Field(default_factory=list)  # ConsolidatedCode dicts


# =============================================================================
# CATEGORY ASSIGNMENT OUTPUT MODELS
# =============================================================================

class CodeAssignedSubmodel(IdeasExtractedSubmodel):
    """Per-idea data with code + attribute assignment.

    Extends step 3's IdeasExtractedSubmodel.
    Step 5 populates: facet (L3), attribute (L4) on the base model,
    plus code assignment fields below.
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
