"""
Local Pipeline Models for step_3_ideaExtractor v5.

Taxonomy: Dimension > Domain > Facet > Attribute (progressive narrowing).
v5 overhaul: 10 MECE dimensions with decision-tree ordering.

Differences from shared models_exp.py:
- ExtractionMetadata: taxonomy_axis → primary_dimension + decision_tree_stop_position, topical_categories → domains
- IdeasExtractedSubmodel: dropped root/category_label, semantic_category → domain,
  added valence; ladder: instance → interpretation → abstraction
- EmbeddingsSubmodel: category_embedding → domain_embedding, taxonomy_embedding → ladder_embedding

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
    # Domains (L2, data-driven, replaces topical_categories)
    domains: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Data-driven domains [{key, label, definition}, ...]"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === v3 PIPELINE MODELS =======================================================================

class IdeasExtractedSubmodel(BaseModel):
    """Per-idea data from step 3 extraction.

    Ladder: instance → interpretation → abstraction (bottom-up) + domain (L2 classification)
    """
    idea_id: str                          # Format: {respondent_id}_{sequence_number}
    idea: str                             # Clean text (starts with template prefix)
    instance: str = ""                    # Verbatim span from response
    interpretation: str = ""              # Concrete interpretation (what it means)
    abstraction: str = ""                 # Broader significance (why it matters)
    domain: str = ""                      # Discovered domain (L2, e.g., "recommendation")
    valence: str = ""                     # +, -, or 0
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0
    template_prefix: Optional[str] = None


class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None               # idea (natural sentence incl. template_prefix)
    interpretation_embedding: Optional[npt.NDArray[np.float32]] = None     # interpretation (concrete)
    abstraction_embedding: Optional[npt.NDArray[np.float32]] = None        # abstraction (broader significance)
    domain_embedding: Optional[npt.NDArray[np.float32]] = None             # domain (L2)
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
