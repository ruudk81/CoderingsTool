"""
Local Pipeline Models for step_3_ideaExtractor v4.

Differences from shared models_exp.py:
- ExtractionMetadata: taxonomy_axis → primary_facet, topical_categories → concept_types
- IdeasExtractedSubmodel: dropped root/category_label, semantic_category → concept_type,
  added valence/agency_focus/prescriptiveness
- EmbeddingsSubmodel: category_embedding → concept_type_embedding

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
    QualityFilterLLMResponse,
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
    ClusterLabelModel,
    ClusterRepresentationModel,
    ClusterRepresentationsModel,
    ClusterRepresentationCacheModel,
    ClusteringMetricsModel,
    LLMContextModel,
    ClusteringMetadataModel,
)


# === v3 METADATA MODEL ========================================================================

class ExtractionMetadata(BaseModel):
    """Extraction-level metadata from step 3 v3 (applies to entire dataset, not per-idea)."""

    # File/variable info
    filename: str = ""
    var_name: str = ""
    var_lab: str = ""                     # Survey question

    # Template
    template_prefix: str = ""             # e.g., "Merk X heeft de associatie"

    # Context specifiers (6 fields from GenericSpecifierGroup1/2Response)
    lang: str = ""                        # e.g., "nl-NL"
    domain: str = ""                      # e.g., "finance"
    topic: str = ""                       # e.g., "brand_association"
    perspective: str = ""                 # e.g., "consumer"
    entity: str = ""                      # e.g., "merk_x"
    intent: str = ""                      # e.g., "evaluate"

    # Primary facet (replaces taxonomy_axis)
    primary_facet: str = ""               # e.g., "EVALUATION_JUDGMENT"
    primary_facet_description: str = ""   # Context-specific description of the facet
    taxonomy_actionable_type: str = ""    # e.g., "attributes", "features", "concepts"

    # Concept types (data-driven, replaces topical_categories)
    concept_types: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Data-driven concept types [{key, label, definition}, ...]"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === v3 PIPELINE MODELS =======================================================================

class IdeasExtractedSubmodel(BaseModel):
    """Per-idea data from step 3 v3 extraction.

    Hierarchy: instance → node → concept_type → primary_facet (dataset-level)
    Secondary facets: valence, agency_focus, prescriptiveness
    """
    idea_id: str                          # Format: {respondent_id}_{sequence_number}
    idea: str                             # Clean text (starts with template prefix)
    instance: str = ""                    # Verbatim span from response
    node: str = ""                        # Canonical, reusable concept (noun phrase)
    concept_type: str = ""                # Discovered concept type (e.g., "recommendation")
    valence: str = ""                     # positive / negative / neutral_mixed
    agency_focus: str = ""                # system_entity / stakeholder_actor / respondent
    prescriptiveness: str = ""            # descriptive / prescriptive
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0
    template_prefix: Optional[str] = None


class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None        # template_prefix + idea
    node_embedding: Optional[npt.NDArray[np.float32]] = None        # node (canonical concept)
    concept_type_embedding: Optional[npt.NDArray[np.float32]] = None  # concept_type
    taxonomy_embedding: Optional[npt.NDArray[np.float32]] = None    # node → concept_type → primary_facet


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
