from typing import List, Any, Optional, Type, Union, Dict
from pydantic import BaseModel, ConfigDict, Field, RootModel
import numpy as np
import numpy.typing as npt

# === GROWING RESULT MODELS ========================================================================================================

class ResponseModel(BaseModel):
    respondent_id: Any
    response: Union[str, float, int]   
    response_type: Optional[str] = None   
    model_config = ConfigDict(arbitrary_types_allowed=True) # for arrays with embeddings
 
    def to_model(self, model_class: Type['BaseModel']) -> 'BaseModel':
        data = self.model_dump()
        return model_class(**data)

class PreprocessedModel(ResponseModel):
    quality_filter: Optional[bool] = None
    quality_filter_code: Optional[int] = None  # 0=meaningful, 99999997=user_missing, 99999998=system_missing, 99999999=no_answer

class QualityFilteredModel(PreprocessedModel):
    pass

class IdeasExtractedSubmodel(BaseModel):
    idea_id: str  # Format: {respondent_id}_{sequence_number}
    idea: str
    model_config = ConfigDict(arbitrary_types_allowed=True)   
    
class IdeasExtractedModel(QualityFilteredModel):
    response_ideas: Optional[List[IdeasExtractedSubmodel]] = None
    idea_count: int = 0

class EmbeddingsSubmodel(IdeasExtractedSubmodel):
    idea_embedding: Optional[npt.NDArray[np.float32]] = None
    
class EmbeddingsModel(IdeasExtractedModel):
    response_ideas: Optional[List[EmbeddingsSubmodel]] = None

class ClusterSubmodel(EmbeddingsSubmodel):
    initial_cluster: Optional[int] = None   
    
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

class CodeDefinition(BaseModel):
    code: str
    definition: str

class Codebook(BaseModel):
    code: str
    definition: str    
    theme: Optional[str] = None
    theme_description: Optional[str] = None   

class CodebookEntry(BaseModel):
    code: str
    definition: str
    source_clusters: Optional[List[int]] = None  # Which clusters influenced this code
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodebookModel(BaseModel):
    codes: List[CodebookEntry]
    generation_metadata: Optional[Dict[str, Any]] = None
    source_variable: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ThemeEnrichedCodebookEntry(CodebookEntry):
    theme: Optional[str] = None
    theme_description: Optional[str] = None
    theme_cluster_id: Optional[int] = None
    is_miscellaneous: Optional[bool] = False

class ThemeEnrichedCodebookModel(CodebookModel):
    codes: List[ThemeEnrichedCodebookEntry]  # Override with enriched version
    themes_summary: Optional[List[Dict[str, Any]]] = None
    code_to_theme_mapping: Optional[Dict[str, str]] = None
    theme_methodology: Optional[str] = None

    
# === CODE GENERATOR MODELS ========================================================================================================

class ClusterThemeItem(BaseModel):
    """Individual theme item from CLUSTER_SUMMARY_PROMPT output"""
    theme_id: int = Field(description="Theme identifier (1, 2, etc.)")
    theme_statement: str = Field(description="≤25 words, atomic, grounded, operational theme statement")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClusterSummaryOutput(BaseModel):
    """Array output from CLUSTER_SUMMARY_PROMPT - instructor compatible"""
    themes: List[ClusterThemeItem] = Field(description="Array of themes extracted from cluster")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # @property
    # def root(self):
    #     """Backward compatibility property to mimic RootModel behavior"""
    #     return self.themes

# class ClusterThemeAnalysis(BaseModel):
#     """Step 1 JSON response: Cluster summary with themes array"""
#     cluster_summary: str
#     themes: List[str]  # Array of theme statements with rationales
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

class CandidateCode(BaseModel):
    """Individual candidate code from Step 2"""  
    code: str
    definition: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

# class ActionDetails(BaseModel):
#     """Action details for coding decisions in Step 3"""
#     codes_to_use: Optional[List[str]] = None
#     codes_to_modify: Optional[str] = None
#     modified_code_name: Optional[str] = None
#     modified_code_definition: Optional[str] = None
#     new_code_name: Optional[str] = None
#     new_code_definition: Optional[str] = None
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

class CodingDecision(BaseModel):
    """Individual coding decision for a theme from Step 3 - flattened structure"""
    theme_number: int
    decision: str  # use | modify | create
    final_code_label: str
    final_code_definition: str
    source_code: Optional[str] = None  # name of reused/modified existing code, or null if new
    justification: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeRecommendation(BaseModel):
    """Simplified Step 3 JSON response: Only coding decisions (flattened format)"""
    coding_decisions: List[CodingDecision]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# class ClusterAnalysis(BaseModel):
#     """Cluster analysis metadata from Step 3"""
#     number_of_themes: int
#     theme_descriptions: List[str]
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class CodeRecommendation(BaseModel):
#     """Step 3 JSON response: Multi-theme code recommendations"""
#     cluster_analysis: ClusterAnalysis
#     coding_decisions: List[CodingDecision]
#     overall_justification: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)


# class CodeEvaluation(BaseModel):
#     """Evaluation criteria for Step 4 validation"""
#     semantic_fit: str
#     atomicity: str
#     parsimony: str
#     redundancy: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

class ValidatedCode(BaseModel):
    """Final validated code from Step 4"""
    code: str
    definition: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OriginalRecommendation(BaseModel):
    """Original recommendation structure for Step 4"""
    code: str
    definition: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeValidation(BaseModel):
    """Individual code validation from Step 4 - simplified structure"""
    theme_number: int
    original_recommendation: OriginalRecommendation
    decision: str  # APPROVE | REJECT
    decision_rationale: str
    validated_code: ValidatedCode
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationResult(BaseModel):
    """Step 4 JSON response: Multi-theme validation results"""
    # theme_assessment: ThemeAssessment
    code_validations: List[CodeValidation]
    # overall_validation: OverallValidation
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    
# class ThemeAssessment(BaseModel):
#     """Theme assessment from Step 4"""
#     number_of_themes_identified: int
#     theme_separation_valid: bool
#     theme_separation_reasoning: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class OverallValidation(BaseModel):
#     """Overall validation summary from Step 4"""
#     all_themes_coded: bool
#     final_code_count: int
#     summary: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)


class CodeGeneratorReasoningResults(BaseModel):
    cluster_results: List[Dict[str, Any]]  # Raw results from each cluster
    
    # ACTUAL prompt inputs for complete transparency (now supports string cluster IDs for sub-clusters)
    step1_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 1 received
    step2_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 2 received
    step3_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 3 received
    step4_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 4 received
    step3_validation_warnings: Dict[Union[int, str], List[Dict[str, Any]]] = {}  # Validation warnings
    
    step1_summaries: Dict[Union[int, str], Dict[str, Any]]  # ClusterThemeAnalysis: {cluster_summary, themes[]}
    step2_analysis: Dict[Union[int, str], List[Dict[str, str]]]  # List[CandidateCode]: Array of candidate codes
    step3_recommendations: Dict[Union[int, str], Dict[str, Any]]  # CodeRecommendation: {coding_decisions[]}  
    step4_validations: Dict[Union[int, str], Dict[str, Any]]  # ValidationResult: {code_validations[]}
    step4_validated_codes: Dict[Union[int, str], Dict[str, Any]] = {}  # Final validated codes from Step 4
    
    stats: Dict[str, Any]
    generator_version: str
    var_lab: str
    total_clusters: int
    total_ideas: int
    processing_timestamp: str
    
    cluster_assignments: Dict[Union[int, str], Dict[str, Any]]
    
    codebook: List[Dict[str, str]]  # Final deduplicated codebook from SharedCodebook
    cluster_data: Dict[Union[int, str], Dict[str, Any]]  # Raw cluster data for stats calculations (supports sub-clusters)
    validation_details: Optional[Dict[Union[int, str], Any]] = None  # Detailed validation results (maps to step4_validations)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get(self, key: str, default=None):
        """Dictionary-style access for promptTester compatibility"""
        return getattr(self, key, default)


# === DEDUPLICATION MODELS ====================================================================================================

# class MergeDecision(BaseModel):
#     """Individual merge decision for deduplication"""
#     codes_to_merge: List[str]
#     final_code_name: str
#     final_definition: str
#     justification: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class DeduplicationResult(BaseModel):
#     """Deduplication JSON response"""
#     merge_decisions: List[MergeDecision]
#     codes_to_keep_unchanged: List[str]
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# === CODE GENERATOR OUTPUT MODELS ====================================================================================================

# class CodeGenerationOutput(BaseModel):
#     """Step 3 JSON response: Multi-theme code recommendations"""
#     cluster_analysis: ClusterAnalysis
#     coding_decisions: List[CodingDecision]
#     overall_justification: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class ValidationOutput(BaseModel):
#     """Step 4 JSON response: Multi-theme validation results"""
#     theme_assessment: ThemeAssessment
#     code_validations: List[CodeValidation]
#     overall_validation: OverallValidation
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# === SMART PHASE PROCESSING MODELS ========================================================================================================

# class ExtractedTheme(BaseModel):
#     """Individual theme extracted from cluster in Phase 1"""
#     cluster_id: int
#     theme_id: str  # Format: cluster_{cluster_id}_theme_{index}
#     theme_text: str
#     theme_embedding: Optional[npt.NDArray[np.float32]] = None
#     extraction_confidence: Optional[float] = None
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class ClusterThemeExtraction(BaseModel):
#     """All themes extracted from a single cluster"""
#     cluster_id: int
#     cluster_data: Dict[str, Any]  # Original cluster data
#     extracted_themes: List[ExtractedTheme]
#     extraction_status: str  # "success", "partial", "failed"
#     extraction_error: Optional[str] = None
#     processing_time: float
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class ThemeSimilarity(BaseModel):
#     """Similarity between two themes"""
#     theme_1_id: str
#     theme_2_id: str
#     similarity_score: float
#     distance_metric: str = "cosine"
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class SimilarityBatch(BaseModel):
#     """Batch of clusters formed by theme similarity"""
#     batch_id: int
#     cluster_ids: List[int]
#     representative_themes: List[str]  # Most representative themes for this batch
#     avg_inter_theme_similarity: float  # Average similarity within batch
#     batch_size: int
#     formation_rationale: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class SmartPhaseStats(BaseModel):
#     """Performance and timing statistics for smart phase processing"""
#     total_clusters: int
#     phase_1_duration: float  # Theme extraction time
#     phase_2_duration: float  # Batch formation time  
#     phase_3_duration: float  # Sequential processing time
#     total_duration: float
    
#     themes_extracted: int
#     batches_formed: int
#     avg_themes_per_cluster: float
#     avg_similarity_per_batch: float
    
#     fallback_triggered: bool = False
#     fallback_reason: Optional[str] = None
    
#     # Performance comparison with current system
#     current_system_duration: Optional[float] = None
#     performance_improvement: Optional[float] = None  # Percentage improvement
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class SmartPhaseResult(BaseModel):
#     """Complete result from smart phase processing"""
#     codebook: List[Dict[str, str]]
#     cluster_assignments: Dict[int, Any]
#     stats: SmartPhaseStats
#     theme_extractions: List[ClusterThemeExtraction]
#     similarity_batches: List[SimilarityBatch]
    
#     # Validation results (if enabled)
#     output_validation: Optional[Dict[str, Any]] = None
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# # === CODEDESIGNER MODELS (INSTRUCTOR-COMPATIBLE) ========================================================================================================


# class CandidateCodeSelectionOutput(RootModel):
#     """Array output from CANDIDATE_CODE_SELECTION_PROMPT - matches prompt exactly"""
#     root: List[CandidateCode]
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class CodeDesignerGenerationOutput(BaseModel):
#     """Stage 4b: Code generation decision and details"""
#     decision: str = Field(description="Action: 'create', 'modify', or 'use'")
#     code: str = Field(description="Final code to use")
#     definition: str = Field(description="Code definition")
#     original_code: Optional[str] = Field(None, description="Original code if modifying")
#     reasoning: str = Field(description="Justification for the decision")
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class CodeDesignerValidationOutput(BaseModel):
#     """Stage 4c: Validation of code assignment"""
#     is_valid: bool = Field(description="Whether the code assignment is valid")
#     final_code: str = Field(description="Validated final code")
#     final_definition: str = Field(description="Validated definition")
#     validation_notes: str = Field(description="Validation reasoning")
#     confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in assignment")
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)



