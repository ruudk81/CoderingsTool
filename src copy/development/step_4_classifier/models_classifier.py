"""
Data models for Taxonomy Classifier (P1-P7).

Partition models and taxonomy cache models.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


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
