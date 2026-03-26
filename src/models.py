"""
Pipeline Data Models — Pydantic models for the CoderingsTool pipeline.

Model chain (per-response, progressive enrichment):
    ResponseModel → PreprocessedModel → QualityFilteredModel
    → IdeasExtractedModel [IdeasExtractedSubmodel]
    → CodeAssignedModel [CodeAssignedSubmodel]

Metadata & cache models (dataset-level):
    ExtractionMetadata (step 3)
    DomainDescription / DomainSet (step 4 partition definitions)
    DomainResultModel / TaxonomyResultsCache (step 4 taxonomy cache)
    CodingResultsCache (step 5 codebook cache)

Source of truth: development step model files (step_3, step_4, step_5, step_6).
"""

from typing import List, Any, Optional, Union, Dict
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# STEP 3: EXTRACTION METADATA (dataset-level)
# =============================================================================

class ExtractionMetadata(BaseModel):
    """Extraction-level metadata from step 3 (applies to entire dataset, not per-idea).

    Taxonomy: Dimension (L1) > Domain (L2) > Facet (L3) > Attribute (L4).
    """

    # File/variable info
    filename: str = ""
    var_name: str = ""
    var_lab: str = ""                     # Survey question

    # Template
    template_prefix: str = ""             # e.g., "Merk X heeft de associatie"

    # Context specifiers (6 fields from GenericSpecifierGroup1/2Response)
    lang: str = ""                        # e.g., "nl-NL"
    sector: str = ""                      # e.g., "finance" (industry/sector)
    topic: str = ""                       # e.g., "brand_association"
    perspective: str = ""                 # e.g., "consumer"
    entity: str = ""                      # e.g., "merk_x"
    intent: str = ""                      # e.g., "evaluate"

    # Primary dimension (L1 in taxonomy: Dimension > Domain > Facet > Attribute)
    primary_dimension: str = ""               # e.g., "EVALUATION_PRIORITIZATION"
    primary_dimension_description: str = ""   # Context-specific description of the dimension
    decision_tree_stop_position: int = 0      # 1-10, which decision tree step triggered selection
    # Domains (L2, data-driven)
    domains: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Data-driven domains [{key, label, definition}, ...]"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# =============================================================================
# PIPELINE MODEL CHAIN (per-response, progressive enrichment)
# =============================================================================

class ResponseModel(BaseModel):
    respondent_id: Any
    response: Union[str, float, int, None]
    response_type: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_model(self, model_class: type) -> 'BaseModel':
        data = self.model_dump()
        return model_class(**data)


class PreprocessedModel(ResponseModel):
    quality_filter: Optional[bool] = None
    quality_filter_code: Optional[int] = None  # 0=meaningful, 99999997=don't know, 99999998=no response/empty, 99999999=gibberish


class QualityFilteredModel(PreprocessedModel):
    pass


class IdeasExtractedSubmodel(BaseModel):
    """Per-idea data from step 3 extraction.

    Taxonomy fields: domain (L2), facet (L3), attribute (L4).
    - Domain assigned by step 3; facet partially populated by step 3.
    - Facet and attribute completed by step 4 (classifier).

    Abstraction ladder (extraction metadata): instance → interpretation → abstraction.
    """
    idea_id: str                          # Format: {respondent_id}_{sequence_number}
    idea: str                             # Clean text (starts with template prefix)
    # --- Abstraction ladder (extraction metadata, NOT taxonomy levels) ---
    instance: str = ""                    # Rung 1: verbatim span from response
    interpretation: str = ""              # Rung 2: concrete meaning (survey language)
    abstraction: str = ""                 # Rung 3: broader significance (survey language)
    # --- Taxonomy levels ---
    domain: str = ""                      # Domain (L2): thematic domain
    facet: str = ""                       # Facet (L3): analytical lens (step 3 hint; completed by step 4)
    attribute: str = ""                   # Attribute (L4): named observable property (assigned by step 4)
    # --- Classification metadata ---
    valence: str = ""                     # Directional effect: +, -, or 0
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0
    template_prefix: Optional[str] = None  # Canonical phrasing prefix for embedding text


# =============================================================================
# STEP 4: TAXONOMY CLASSIFIED MODELS (growing model output)
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


# =============================================================================
# STEP 6: CODE ASSIGNMENT MODELS
# =============================================================================

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


class CodeAssignedSubmodel(TaxonomyClassifiedSubmodel):
    """Per-idea data with code + attribute assignment.

    Extends step 4's TaxonomyClassifiedSubmodel (which provides facet, attribute,
    partition_name). Adds code assignment fields from step 6.
    """
    assigned_code: Optional[str] = None
    assigned_attribute: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None


class CodeAssignedModel(TaxonomyClassifiedModel):
    """Response-level model with code-assigned ideas."""
    response_ideas: Optional[List[CodeAssignedSubmodel]] = None
    assignment_metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# STEP 4: TAXONOMY CACHE MODELS (partition definitions + P1-P7 results)
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


class DomainResultModel(BaseModel):
    """Pydantic-serializable partition result for caching (taxonomy version)."""
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
# STEP 5: CODEBOOK CACHE MODELS (extends taxonomy with generated codes)
# =============================================================================

class CodingResultsCache(BaseModel):
    """Cache for codebook results (taxonomy + codes).

    Extends TaxonomyResultsCache fields with raw_codes from P8-P9.
    """
    partition_set: DomainSet
    partition_results: Dict[str, DomainResultModel]
    label_counts: Dict[str, int] = Field(default_factory=dict)
    label_source: str = ""
    total_categories: int = 0
    raw_codes: List[Dict] = Field(default_factory=list)  # ConsolidatedCode dicts
