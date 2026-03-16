from typing import List, Any, Optional, Type, Union, Dict, Tuple
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt

# ===  METADATA MODEL ========================================================================================================

class ExtractionMetadata(BaseModel):
    # File/variable info
    filename: str = ""
    var_name: str = ""
    var_lab: str = ""                     # Survey question

    # Template
    template_prefix: str = ""             # e.g., "Merk X has the association"

    # Context specifiers (6 fields)
    lang: str = ""                        # e.g., "nl-NL"
    domain: str = ""                      # e.g., "finance"
    topic: str = ""                       # e.g., "brand_association"
    perspective: str = ""                 # e.g., "consumer"
    entity: str = ""                      # e.g., "merk_x"
    intent: str = ""                      # e.g., "evaluate"

    # Primary facet (v5: 10 MECE facets with decision-tree ordering)
    primary_facet: str = ""               # e.g., "EVALUATION_PRIORITIZATION"
    primary_facet_description: str = ""   # Context-specific description of the facet
    decision_tree_stop_position: int = 0  # 1-10, which decision tree step triggered facet selection
    # Concept types (data-driven)
    concept_types: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Data-driven concept types [{key, label, definition}, ...]"
    )

    # Timestamp
    extraction_timestamp: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === GROWING RESULT MODELS ========================================================================================================

class ResponseModel(BaseModel):
    respondent_id: Any
    response: Union[str, float, int, None]   
    response_type: Optional[str] = None   
    model_config = ConfigDict(arbitrary_types_allowed=True) # for arrays with embeddings
 
    def to_model(self, model_class: Type['BaseModel']) -> 'BaseModel':
        data = self.model_dump()
        return model_class(**data)

class PreprocessedModel(ResponseModel):
    quality_filter: Optional[bool] = None
    quality_filter_code: Optional[int] = None  # 0=meaningful, 99999997=don't know, 99999998=no response/empty, 99999999=gibberish

class QualityFilteredModel(PreprocessedModel):
    pass

# QualityFilterLLMResponse moved to prompts.py (co-located with GRADER_INSTRUCTIONS)

class IdeasExtractedSubmodel(BaseModel):
    """Per-idea data from step 3 extraction.

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
    template_prefix: Optional[str] = None  # Canonical phrasing prefix for embedding text extraction

class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None        # idea (natural sentence incl. template_prefix)
    concept_embedding: Optional[npt.NDArray[np.float32]] = None      # concept → concept_type_definition
    concept_type_embedding: Optional[npt.NDArray[np.float32]] = None  # concept_type
    ladder_embedding: Optional[npt.NDArray[np.float32]] = None      # instance → concept → concept_type → concept_type_definition
    idea_concept_defined_embedding: Optional[npt.NDArray[np.float32]] = None  # idea → concept → concept_type_definition

class EmbeddingsModel(IdeasExtractedModel):
    response_ideas: Optional[List[EmbeddingsSubmodel]] = None
    embedding_text_format: str = "idea"  # "idea", "concept", "ladder", "default", "all", etc.

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
    # Bridge fields for step 6 codeGenerator compatibility (set at runtime)
    initial_cluster: Optional[Union[int, str]] = None
    expanded_cluster: Optional[str] = None
    cluster_theme: Optional[str] = None


class CodeAssignedModel(EmbeddingsModel):
    """Response-level model with category-assigned ideas."""
    response_ideas: Optional[List[CodeAssignedSubmodel]] = None
    assignment_metadata: Optional[Dict[str, Any]] = None


# === CODEBOOK  MODELS ========================================================================================================
class CodebookEntry(BaseModel):
    code: str
    definition: str
    source_cluster: Optional[str] = None    # Support sub-clusters like "12-1", "12-2"
    inclusion_examples: Optional[List[str]] = None
    exclusion_examples: Optional[List[str]] = None
    near_neighbor_label: Optional[str] = None
    tell_apart_rule: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodebookModel(BaseModel):
    codes: List[CodebookEntry]
    generation_metadata: Optional[Dict[str, Any]] = None
    source_variable: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RefinedSubcode(BaseModel):
    id: str  # Original code ID(s) - multiple if merged (e.g., "1,2,3")
    code: str
    description: str
    category: str = ""  # Empty string when code is directly under theme; category name for 3-level hierarchy
    source_cluster: str = ""  # Cluster ID(s) from Step 6 - may be comma-separated for merged codes (e.g., "8,11,23")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RefinedCodebookCategory(BaseModel):
    category: str
    subcodes: List[RefinedSubcode]
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RefinedCodebookModel(BaseModel):
    analysis: str
    refined_codebook: List[RefinedCodebookCategory]
    generation_metadata: Optional[Dict[str, Any]] = None
    source_variable: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeTransformation(BaseModel):
    phase: str  # "MAP" or "REDUCE"
    batch_id: Optional[int] = None
    transformation_type: str  # "PRESERVED", "MERGED", "DROPPED"
    input_ids: List[str]  # Original sequential IDs
    output_id: Optional[str] = None  # Result ID (None if DROPPED)
    source_cluster_ids: List[str]  # Cluster IDs for traceability
    final_code_label: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class BatchTransformationRecord(BaseModel):
    batch_id: int
    input_ids: List[str]
    input_cluster_map: Dict[str, str]  # id -> cluster_id
    output_ids: List[str]
    transformations: List[CodeTransformation]
    dropped_ids: List[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RefinementLineage(BaseModel):
    original_codes: List[Dict[str, Any]]
    master_id_to_cluster_map: Dict[str, str]
    map_batches: List[BatchTransformationRecord]
    reduce_record: Optional[BatchTransformationRecord] = None
    orphaned_clusters: List[str] = []
    reconciled_mappings: Dict[str, str] = {}  # orphaned_cluster -> target_code_source_cluster
    timestamp: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeRefinementResults(BaseModel):
    original_codebook: List[Dict[str, Any]]
    refined_codebook: RefinedCodebookModel
    processing_stats: Dict[str, Any]
    timestamp: str
    lineage: Optional[RefinementLineage] = None  # Transformation tracking
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeDefinition(BaseModel):
    code: str
    definition: str

class Codebook(BaseModel):
    code: str
    definition: str
    source_cluster:  Optional[str] = None    # Support sub-clusters like "12-1", "12-2"
    theme: Optional[str] = None
    theme_description: Optional[str] = None
    inclusion_examples: Optional[List[str]] = None
    exclusion_examples: Optional[List[str]] = None
    near_neighbor_label: Optional[str] = None
    tell_apart_rule: Optional[str] = None   

class ThemeEnrichedCodebookEntry(CodebookEntry):
    code: Optional[str] = None
    definition: Optional[str] = None
    theme: Optional[str] = None
    theme_description: Optional[str] = None
    category: str = ""  # Empty string for 2-level hierarchy; category name for 3-level hierarchy
    category_description: str = ""  # Category description (empty for 2-level)
    source_cluster: Optional[Union[int, str]] = None  
    
class ThemeEnrichedCodebookModel(CodebookModel):
    codes: List[ThemeEnrichedCodebookEntry]  # Override with enriched version
    themes_summary: Optional[List[Dict[str, Any]]] = None
    code_to_theme_mapping: Optional[Dict[str, str]] = None
    theme_methodology: Optional[str] = None


# === MECE CACHE MODELS (step 5 categories) ========================================================================================================

from prompts import (
    DomainSet, MECECode, MECEVerification,
)


class DomainResultModel(BaseModel):
    """Pydantic-serializable version of PartitionMECEResult for caching."""
    partition_name: str
    n_labels: int
    n_batches: int
    reduce_skipped: bool
    categories: List[MECECode] = Field(default_factory=list)
    mece_verifications: List[MECEVerification] = Field(default_factory=list)


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



