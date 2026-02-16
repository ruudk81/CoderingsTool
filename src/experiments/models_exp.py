"""
Experimental Models — Clean Pydantic models aligned with step 3 prompt outputs.

Differences from production models.py:
- ExtractionMetadata: removed dead fields (taxonomy_secondary_axis, taxonomy_sample_phrases,
  taxonomy_rationale, extraction_timestamp); renamed taxonomy_primary_axis → taxonomy_axis
- IdeasExtractedSubmodel: removed legacy fields (taxonomy_phrase, sentiment, sense)
- EmbeddingsSubmodel: 4 embedding fields (idea, node, category, taxonomy)
- OntologySubmodel: removed entirely (was unused)

Usage in experimental steps:
    from experiments import models_exp as models
"""

from typing import List, Any, Optional, Type, Union, Dict, Tuple
from pydantic import BaseModel, ConfigDict
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
    """Extraction-level metadata from step 3 (applies to entire dataset, not per-idea)."""

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

    # Taxonomy (from CodingDimensionConsolidatedResponse + SubjectExtractionResponse)
    taxonomy_axis: str = ""               # e.g., "WHAT" (was: taxonomy_primary_axis)
    taxonomy_axis_description: str = ""   # Context-specific description of the axis
    taxonomy_actionable_type: str = ""    # e.g., "attributes", "features", "concepts"

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === CLEANED PIPELINE MODELS =======================================================================

class IdeasExtractedSubmodel(BaseModel):
    """Per-idea data from step 3 extraction.

    Fields match SemanticTaxonomyResponse from prompts_exp.py:
    instance → node → semantic_category (category_label) → root
    """
    idea_id: str                    # Format: {respondent_id}_{sequence_number}
    idea: str                       # Clean text
    instance: str = ""              # Verbatim span from response
    node: str = ""                  # Canonical, reusable concept (noun phrase)
    semantic_category: str = ""     # One of: identity, attribute, function, state, evaluation, relation
    category_label: str = ""        # Concise descriptive label within the category
    root: str = ""                  # Top-level domain framing
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0
    template_prefix: Optional[str] = None


class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None        # template_prefix + idea
    node_embedding: Optional[npt.NDArray[np.float32]] = None        # node (canonical concept)
    category_embedding: Optional[npt.NDArray[np.float32]] = None    # semantic_category
    taxonomy_embedding: Optional[npt.NDArray[np.float32]] = None    # node → category_label → semantic_category → root


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
