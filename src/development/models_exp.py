"""
Experimental Models — Clean Pydantic models aligned with step 3 v5 prompt outputs.

Differences from production models.py:
- ExtractionMetadata: taxonomy_primary_axis → primary_facet + primary_facet_description
  + decision_tree_stop_position; taxonomy_sample_phrases → concept_types
- IdeasExtractedSubmodel: concept → rung_1 (concrete interpretation),
  concept_type_definition → rung_2 (broader significance); dropped root; added valence
- EmbeddingsSubmodel: 5 embedding fields (idea, rung_1, rung_2, concept_type, ladder)
- OntologySubmodel: removed entirely (was unused)

Usage in experimental steps:
    from development import models_exp as models
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
)


class CodebookExp(Codebook):
    """Extended Codebook with concept_type for partition-based code assignment."""
    concept_type: Optional[str] = None
    boundary_test: Optional[str] = None
    diagnostic_signals: Optional[List[str]] = None


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

    Ladder: instance → rung_1 → rung_2 (bottom-up abstraction) + concept_type (domain classification)
    Secondary facets: valence
    """
    idea_id: str                          # Format: {respondent_id}_{sequence_number}
    idea: str                             # Clean text (starts with template prefix)
    instance: str = ""                    # Verbatim span from response
    rung_1: str = ""                      # Concrete interpretation (what it means)
    rung_2: str = ""                      # Broader significance (why it matters)
    concept_type: str = ""                # Discovered concept type (e.g., "recommendation")
    valence: str = ""                     # +, -, or 0
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0
    template_prefix: Optional[str] = None


class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None        # idea (natural sentence incl. template_prefix)
    rung_1_embedding: Optional[npt.NDArray[np.float32]] = None      # rung_1 (concrete interpretation)
    rung_2_embedding: Optional[npt.NDArray[np.float32]] = None      # rung_2 (broader significance)
    concept_type_embedding: Optional[npt.NDArray[np.float32]] = None  # concept_type
    ladder_embedding: Optional[npt.NDArray[np.float32]] = None      # instance → rung_1 → rung_2


class EmbeddingsModel(IdeasExtractedModel):
    response_ideas: Optional[List[EmbeddingsSubmodel]] = None
    embedding_text_format: str = "idea"


class ClusterSubmodel(EmbeddingsSubmodel):
    initial_cluster: Optional[Union[int, str]] = None
    cluster_probability: Optional[float] = None
    iteration_assigned: Optional[int] = None
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


# === MECE CACHE MODELS (step 5 categories) ========================================================

from development.step_4_classNcoder.models_exp import DomainSet
from development.step_4_classNcoder.prompts_exp import (
    MECECode, MECEVerification,
)


class DomainResultModel(BaseModel):
    """Pydantic-serializable version of PartitionMECEResult for caching."""
    partition_name: str
    n_labels: int
    n_batches: int
    reduce_skipped: bool
    categories: List[MECECode] = Field(default_factory=list)
    mece_verifications: List[MECEVerification] = Field(default_factory=list)
    hierarchy_depth: Optional[dict] = None


class CodingResultsCache(BaseModel):
    """Top-level cache wrapper for all MECE results.

    Stores the complete output of the MAP/REDUCE/MECE pipeline:
    partition definitions, MECE category sets, and label counts.
    Designed for save_metadata_to_cache() (single model).
    """
    partition_set: DomainSet
    partition_results: Dict[str, DomainResultModel]
    label_counts: Dict[str, int] = Field(default_factory=dict)
    processing_mode: str = ""
    label_source: str = ""
    total_categories: int = 0


# === CATEGORY ASSIGNMENT MODELS (parallel branch from EmbeddingsSubmodel) ==========================

class CodeAssignedSubmodel(EmbeddingsSubmodel):
    """Per-idea data with MECE category assignment.

    Extends EmbeddingsSubmodel (not ClusterSubmodel) since category
    assignment operates on ideas partitioned by concept_type, independent
    of clustering.
    """
    assigned_category: Optional[str] = None        # MECECode.category_label
    category_confidence: Optional[float] = None    # 0.0 - 1.0
    category_rationale: Optional[str] = None       # LLM reasoning
    partition_name: Optional[str] = None           # concept_type partition
    parent_category: Optional[str] = None          # parent MECECode label (None for flat)
    # Bridge fields for step 6 codeGenerator compatibility (set at runtime)
    initial_cluster: Optional[Union[int, str]] = None
    expanded_cluster: Optional[str] = None
    cluster_theme: Optional[str] = None


class CodeAssignedModel(EmbeddingsModel):
    """Response-level model with category-assigned ideas."""
    response_ideas: Optional[List[CodeAssignedSubmodel]] = None
    assignment_metadata: Optional[Dict[str, Any]] = None


# === STEP 7: MECE-ENRICHED CODEBOOK MODELS ========================================================

class ThemeEnrichedCodebookEntryExp(ThemeEnrichedCodebookEntry):
    """Extended codebook entry with MECE-verified assignment instructions.

    Replaces step 6's stale inclusion/exclusion examples with fresh MECE-verified
    versions and adds boundary_test + diagnostic_signals for downstream code assignment.
    """
    boundary_test: Optional[str] = None               # Yes/no question for independent assignment
    diagnostic_signals: Optional[List[str]] = None     # 3-5 trigger words/phrases
    concept_type: Optional[str] = None                 # Source concept_type from step 3
    mece_verified: bool = False                        # Whether MECE enforcement was applied


class ThemeEnrichedCodebookModelExp(CodebookModel):
    """Extended codebook model with MECE-enriched entries."""
    codes: List[ThemeEnrichedCodebookEntryExp]
    themes_summary: Optional[List[Dict[str, Any]]] = None
    code_to_theme_mapping: Optional[Dict[str, str]] = None
    theme_methodology: Optional[str] = None
    source_variable: Optional[str] = None
    mece_partition_results: Optional[Dict[str, Any]] = None   # Raw MECE results per partition
    concept_type_mapping: Optional[Dict[str, str]] = None     # source_cluster -> concept_type
    cross_partition_results: Optional[Dict[str, Any]] = None  # Cross-partition judge results
    partition_remap: Optional[Dict[str, str]] = None          # old_partition -> new_partition (for split partitions)
    dominance_axes: Optional[Dict[str, str]] = None           # partition_name -> dominance_axis routing question
    other_idea_assignments: Optional[Dict[str, str]] = None  # idea_id -> pre-assigned code label (DIRECT_OTHER from step 6)
