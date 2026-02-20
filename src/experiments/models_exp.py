"""
Experimental Models — Clean Pydantic models aligned with step 3 v5 prompt outputs.

Differences from production models.py:
- ExtractionMetadata: taxonomy_primary_axis → primary_facet + primary_facet_description
  + decision_tree_stop_position; taxonomy_sample_phrases → concept_types
- IdeasExtractedSubmodel: node → concept, semantic_category → concept_type,
  category_label → concept_type_definition; dropped root; added valence
- EmbeddingsSubmodel: 4 embedding fields (idea, concept, concept_type, ladder)
- OntologySubmodel: removed entirely (was unused)

Usage in experimental steps:
    from experiments import models_exp as models
"""

from typing import List, Any, Optional, Type, Union, Dict, Tuple
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt

# === UNCHANGED MODELS (re-exported from production) ================================================

from models import (
    # Base pipeline models (unchanged)
    ResponseModel,
    PreprocessedModel,
    QualityFilteredModel,
    QualityFilterLLMResponse,

    # Codebook models (unchanged)
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

    # Cluster representation models (unchanged)
    ClusterLabelModel,
    ClusterRepresentationModel,
    ClusterRepresentationsModel,
    ClusterRepresentationCacheModel,
    ClusteringMetricsModel,
    LLMContextModel,
    ClusteringMetadataModel,
)


# === CLEANED METADATA MODEL ========================================================================

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
    domain: str = ""                      # e.g., "finance"
    topic: str = ""                       # e.g., "brand_association"
    perspective: str = ""                 # e.g., "consumer"
    entity: str = ""                      # e.g., "merk_x"
    intent: str = ""                      # e.g., "evaluate"

    # Primary facet (from MECE decision tree)
    primary_facet: str = ""               # e.g., "EVALUATION_PRIORITIZATION"
    primary_facet_description: str = ""   # Context-specific description of the facet
    decision_tree_stop_position: int = 0  # 1-10, which decision tree step triggered facet selection
    taxonomy_actionable_type: str = ""    # e.g., "attributes", "features", "concepts"

    # Concept types (data-driven)
    concept_types: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Data-driven concept types [{key, label, definition}, ...]"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === CLEANED PIPELINE MODELS =======================================================================

class IdeasExtractedSubmodel(BaseModel):
    """Per-idea data from step 3 v5 extraction.

    Hierarchy: instance → concept → concept_type → primary_facet (dataset-level)
    Secondary facets: valence
    """
    idea_id: str                          # Format: {respondent_id}_{sequence_number}
    idea: str                             # Clean text (starts with template prefix)
    instance: str = ""                    # Verbatim span from response
    concept: str = ""                     # Canonical, reusable concept (noun phrase)
    concept_type: str = ""                # Discovered concept type (e.g., "recommendation")
    concept_type_definition: str = ""     # High-level framing of concept_type in survey context
    valence: str = ""                     # positive / negative / neutral_mixed
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0
    template_prefix: Optional[str] = None


class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None        # idea (natural sentence incl. template_prefix)
    concept_embedding: Optional[npt.NDArray[np.float32]] = None     # concept → concept_type_definition
    concept_type_embedding: Optional[npt.NDArray[np.float32]] = None  # concept_type
    ladder_embedding: Optional[npt.NDArray[np.float32]] = None      # instance → concept → concept_type → concept_type_definition
    idea_concept_defined_embedding: Optional[npt.NDArray[np.float32]] = None  # idea → concept → concept_type_definition


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
