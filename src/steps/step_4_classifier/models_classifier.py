"""
Data models for Taxonomy Classifier (P1-P7).

Partition models, taxonomy cache models, and growing model output.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from steps.step_3_ideaExtractor.models import (
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
# CACHE MODELS
# =============================================================================

class DomainResultModel(BaseModel):
    """Pydantic-serializable partition result for caching (v3)."""
    partition_name: str
    n_labels: int
    n_batches: int
    facets: List[Dict[str, Any]] = Field(default_factory=list)
    facet_assignments: Dict[str, str] = Field(default_factory=dict)
    attributes: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    attribute_assignments: Dict[str, str] = Field(default_factory=dict)


class TaxonomyResultsCache(BaseModel):
    """Cache for taxonomy results (P1-P7): domains, facets, attributes."""
    partition_set: DomainSet
    partition_results: Dict[str, DomainResultModel]
    label_counts: Dict[str, int] = Field(default_factory=dict)
    label_source: str = ""


# =============================================================================
# GROWING MODEL (per-respondent output with taxonomy classification)
# =============================================================================

class TaxonomyClassifiedSubmodel(IdeasExtractedSubmodel):
    """Per-idea data with taxonomy classification.

    Extends step 3's IdeasExtractedSubmodel.
    facet (L3) and attribute (L4) are inherited and populated by step 4 P3/P6.
    """
    partition_name: Optional[str] = None  # Domain partition this idea belongs to


class TaxonomyClassifiedModel(IdeasExtractedModel):
    """Response-level model with taxonomy-classified ideas."""
    response_ideas: Optional[List[TaxonomyClassifiedSubmodel]] = None
    classification_metadata: Optional[Dict[str, Any]] = None
