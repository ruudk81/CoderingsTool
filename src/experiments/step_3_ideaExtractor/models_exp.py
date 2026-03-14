"""
Local Pipeline Models for step_3_ideaExtractor v5.

Taxonomy: Dimension (L1) > Domain (L2) > Facet (L3) > Attribute (L4).
v5 overhaul: 10 MECE dimensions with decision-tree ordering.

Per-idea fields map to taxonomy levels:
- instance  → Attribute (L4): verbatim span
- facet     → Facet (L3): dimension-specific aspect
- domain    → Domain (L2): thematic domain

Keeps shared models_exp.py untouched so v2 remains runnable.
"""

from typing import List, Any, Optional, Union, Dict
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt

# === RE-EXPORTS from production models (unchanged) ============================================

from models import (
    ResponseModel,
    PreprocessedModel,
    QualityFilteredModel,
    CodebookEntry,
    CodebookModel,
    RefinedSubcode,
    RefinedCodebookCategory,
    RefinedCodebookModel,
    CodeTransformation,
    BatchTransformationRecord,
    RefinementLineage,
    CodeRefinementResults,
    CodeDefinition,
    Codebook,
    ThemeEnrichedCodebookEntry,
    ThemeEnrichedCodebookModel,
)

from prompts import QualityFilterLLMResponse


# === v3 METADATA MODEL ========================================================================

class ExtractionMetadata(BaseModel):
    """Extraction-level metadata from step 3 v5 (applies to entire dataset, not per-idea)."""

    # File/variable info
    filename: str = ""
    var_name: str = ""
    var_lab: str = ""                     # Survey question

    # Template
    template_prefix: str = ""             # e.g., "ASN Bank heeft de associatie"

    # Context specifiers (6 fields from GenericSpecifierGroup1/2Response)
    lang: str = ""                        # e.g., "nl-NL"
    sector: str = ""                      # e.g., "finance" (industry/sector)
    topic: str = ""                       # e.g., "brand_association"
    perspective: str = ""                 # e.g., "consumer"
    entity: str = ""                      # e.g., "asn_bank"
    intent: str = ""                      # e.g., "evaluate"

    # Primary dimension (L1 in taxonomy: Dimension > Domain > Facet > Attribute)
    primary_dimension: str = ""               # e.g., "EVALUATION_PRIORITIZATION"
    primary_dimension_description: str = ""   # Context-specific description of the dimension
    decision_tree_stop_position: int = 0      # 1-10, which decision tree step triggered selection
    # Domains (L2, data-driven, replaces topical_categories)
    domains: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Data-driven domains [{key, label, definition}, ...]"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === v3 PIPELINE MODELS =======================================================================

class IdeasExtractedSubmodel(BaseModel):
    """Per-idea data from step 3 extraction.

    Taxonomy levels: instance (L4 Attribute) + facet (L3) + domain (L2)
    Abstraction ladder: instance → interpretation → abstraction (survey language)
    """
    idea_id: str                          # Format: {respondent_id}_{sequence_number}
    idea: str                             # Clean text (starts with template prefix)
    instance: str = ""                    # Attribute (L4): verbatim span from response
    interpretation: str = ""              # Ladder rung 2: concrete meaning (survey language)
    abstraction: str = ""                 # Ladder rung 3: broader significance (survey language)
    facet: str = ""                       # Facet (L3): dimension-specific aspect
    domain: str = ""                      # Domain (L2): thematic domain
    valence: str = ""                     # +, -, or 0
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0
    template_prefix: Optional[str] = None


class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None               # idea (natural sentence incl. template_prefix)
    interpretation_embedding: Optional[npt.NDArray[np.float32]] = None     # Ladder rung 2: interpretation
    abstraction_embedding: Optional[npt.NDArray[np.float32]] = None        # Ladder rung 3: abstraction
    facet_embedding: Optional[npt.NDArray[np.float32]] = None              # Facet (L3)
    domain_embedding: Optional[npt.NDArray[np.float32]] = None             # Domain (L2)
    ladder_embedding: Optional[npt.NDArray[np.float32]] = None             # instance → interpretation → abstraction


class EmbeddingsModel(IdeasExtractedModel):
    response_ideas: Optional[List[EmbeddingsSubmodel]] = None
    embedding_text_format: str = "idea"


class ClusterSubmodel(EmbeddingsSubmodel):
    initial_cluster: Optional[Union[int, str]] = None
    cluster_probability: Optional[float] = None
    expanded_cluster: Optional[str] = None
    cluster_theme: Optional[str] = None


class ClusterModel(EmbeddingsModel):
    response_ideas: Optional[List[ClusterSubmodel]] = None


class AssignedIdeaSubmodel(ClusterSubmodel):
    assigned_codes: Optional[List[str]] = None
    assigned_themes: Optional[List[str]] = None
    assignment_confidence: Optional[float] = None
    assignment_rationale: Optional[str] = None


class CodeAssignedModel(ClusterModel):
    response_ideas: Optional[List[AssignedIdeaSubmodel]] = None
    assignment_metadata: Optional[Dict[str, Any]] = None
