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
    template_prefix: str = ""             # e.g., "ASN Bank has the association"

    # Context specifiers (6 fields)
    lang: str = ""                        # e.g., "nl-NL"
    domain: str = ""                      # e.g., "finance"
    topic: str = ""                       # e.g., "brand_association"
    perspective: str = ""                 # e.g., "consumer"
    entity: str = ""                      # e.g., "asn_bank"
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

class ClusterSubmodel(EmbeddingsSubmodel):
    initial_cluster: Optional[Union[int, str]] = None
    cluster_probability: Optional[float] = None  # HDBSCAN membership probability (0-1)
    expanded_cluster: Optional[str] = None
    cluster_theme: Optional[str] = None  # Theme name from Step 6 Chain 1
    
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


# === CLUSTER REPRESENTATION MODELS ========================================================================================================

class ClusterLabelModel(BaseModel):
    cluster_id: int
    theme: str                    # Short atomic label (≤10 words)
    description: str              # 1-2 sentence description
    key_concepts: List[str]       # 3-5 key concepts
    n_ideas: int                  # Number of ideas in cluster
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClusterRepresentationModel(BaseModel):
    cluster_id: int
    keywords: List[Tuple[str, float]]           # c-TF-IDF keywords [(word, score), ...]
    llm_label: Optional[ClusterLabelModel] = None  # LLM-generated label (if enabled)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClusterRepresentationsModel(BaseModel):
    representations: List[ClusterRepresentationModel]
    generation_metadata: Optional[Dict[str, Any]] = None  # Algorithm, config, timestamps
    model_config = ConfigDict(arbitrary_types_allowed=True)


# === CLUSTERING METADATA CACHE MODELS ========================================================================================================

class ClusterRepresentationCacheModel(BaseModel):
    cluster_id: int
    size: int

    # What was given to LLM (audit trail)
    representative_samples: List[Tuple[str, float]]  # (text, probability/score)
    keywords_ctfidf: List[Tuple[str, float]]
    keywords_mmr: List[Tuple[str, float]]
    keywords_tfidf: List[Tuple[str, float]]

    # Cluster distributions (also given to LLM)
    sentiment_distribution: Optional[Dict[str, float]] = None  # e.g., {"positive": 0.3, "neutral": 0.6}
    sense_distribution: Optional[Dict[str, float]] = None      # e.g., {"factual": 0.7, "evaluative": 0.3}

    # LLM output
    label_theme: Optional[str] = None
    label_description: Optional[str] = None
    label_key_concepts: Optional[List[str]] = None

    # Cluster-level metrics
    mean_probability: Optional[float] = None
    coherence: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusteringMetricsModel(BaseModel):
    n_clusters: int
    noise_rate: float
    noise_count: int
    mean_coherence: float
    coherence_breakdown: str
    silhouette: Optional[float] = None
    dbcv: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMContextModel(BaseModel):
    survey_question: str
    language: str

    # Dataset context
    domain: Optional[str] = None
    entity: Optional[str] = None
    topic: Optional[str] = None
    perspective: Optional[str] = None
    intent: Optional[str] = None

    # Taxonomy context
    taxonomy_axis: Optional[str] = None
    taxonomy_description: Optional[str] = None
    taxonomy_actionable_type: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusteringMetadataModel(BaseModel):
    # Per-cluster data
    clusters: Dict[int, ClusterRepresentationCacheModel]

    # Global LLM context (shared across all clusters)
    llm_context: Optional[LLMContextModel] = None

    # Global metrics
    metrics: ClusteringMetricsModel

    # Provenance
    algorithm_used: str
    algorithm_params: Dict[str, Any]
    timestamp: str

    model_config = ConfigDict(arbitrary_types_allowed=True)



