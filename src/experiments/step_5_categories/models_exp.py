"""
Data models for Step 5 category discovery.

Partition models, MECE cache models, and category assignment output models.
All idea-level models extend step 3's IdeasExtractedSubmodel directly
(no embeddings dependency).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from experiments.step_3_ideaExtractor.models_exp import (
    IdeasExtractedSubmodel,
    IdeasExtractedModel,
)
from .prompts_exp import MECECategory, MECEVerification


# =============================================================================
# PARTITION MODELS (data-driven domain groups)
# =============================================================================

class PartitionDescription(BaseModel):
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


class PartitionSet(BaseModel):
    """Complete set of domain partitions."""
    partitions: List[PartitionDescription] = Field(
        ...,
        description="List of populated domain partitions"
    )


# =============================================================================
# MECE CACHE MODELS
# =============================================================================

class PartitionMECEResultModel(BaseModel):
    """Pydantic-serializable partition result for caching."""
    partition_name: str
    n_labels: int
    n_batches: int
    categories: List[MECECategory] = Field(default_factory=list)


class MECEResultsCache(BaseModel):
    """Top-level cache wrapper for all category results."""
    partition_set: PartitionSet
    partition_results: Dict[str, PartitionMECEResultModel]
    label_counts: Dict[str, int] = Field(default_factory=dict)
    label_source: str = ""
    total_categories: int = 0


# =============================================================================
# CATEGORY ASSIGNMENT OUTPUT MODELS
# =============================================================================

class CategoryAssignedSubmodel(IdeasExtractedSubmodel):
    """Per-idea data with MECE category assignment.

    Extends step 3's IdeasExtractedSubmodel (no embeddings).
    """
    assigned_category: Optional[str] = None
    category_confidence: Optional[float] = None
    category_rationale: Optional[str] = None
    partition_name: Optional[str] = None
    parent_category: Optional[str] = None


class CategoryAssignedModel(IdeasExtractedModel):
    """Response-level model with category-assigned ideas."""
    response_ideas: Optional[List[CategoryAssignedSubmodel]] = None
    assignment_metadata: Optional[Dict[str, Any]] = None
