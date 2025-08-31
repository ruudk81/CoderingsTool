from typing import List, Any, Optional, Type, Union, Dict
from pydantic import BaseModel, ConfigDict #, Field, RootModel
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

# === CODE GENERATOR MODELS ========================================================================================================

# class ClusterThemeItem(BaseModel):
#     """Individual theme item from CLUSTER_SUMMARY_PROMPT output"""
#     theme_id: int = Field(description="Theme identifier (1, 2, etc.)")
#     theme_name: str = Field(description="Short thematic name (≤10 words)")
#     theme_statement: str = Field(description="≤25 words, atomic, grounded, operational theme statement")
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class ClusterSummaryOutput(RootModel[List[ClusterThemeItem]]):
#     """Array output from CLUSTER_SUMMARY_PROMPT - matches prompt exactly"""
#     root: List[ClusterThemeItem]
    

# class CandidateCode(BaseModel):
#     """Individual candidate code from Step 2"""  
#     code: str
#     definition: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class CandidateCodeSelectionOutput(RootModel[List[CandidateCode]]):
#     """Array output from CANDIDATE_CODE_SELECTION_PROMPT - matches prompt exactly"""
#     root: List[CandidateCode]

# class CodingDecision(BaseModel):
#     """Individual coding decision for a theme from Step 3 - flattened structure"""
#     theme_number: int
#     theme_name: str 
#     decision: str  # use | modify | create
#     final_code_label: str
#     final_code_definition: str
#     source_code: Optional[str] = None  # name of reused/modified existing code, or null if new
#     justification: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class CodeRecommendation(BaseModel):
#     """Simplified Step 3 JSON response: Only coding decisions (flattened format)"""
#     coding_decisions: List[CodingDecision]
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class ValidatedCode(BaseModel):
#     """Final validated code from Step 4"""
#     code: str
#     definition: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class OriginalRecommendation(BaseModel):
#     """Original recommendation structure for Step 4"""
#     code: str
#     definition: str
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class CodeValidation(BaseModel):
#     """Individual code validation from Step 4 - simplified structure"""
#     theme_number: int
#     theme_name: str 
#     original_recommendation: OriginalRecommendation
#     decision: str  # APPROVE | REJECT
#     decision_rationale: str
#     validated_code: ValidatedCode
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class ValidationResult(BaseModel):
#     """Step 4 JSON response: Multi-theme validation results"""
#     code_validations: List[CodeValidation]
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)
 
# class CodeGeneratorReasoningResults(BaseModel):
#     cluster_results: List[Dict[str, Any]]  # Raw results from each cluster
    
#     # ACTUAL prompt inputs for complete transparency (now supports string cluster IDs for sub-clusters)
#     step1_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 1 received
#     step2_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 2 received
#     step3_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 3 received
#     step4_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 4 received
#     step3_validation_warnings: Dict[Union[int, str], List[Dict[str, Any]]] = {}  # Validation warnings
    
#     step1_summaries: Dict[Union[int, str], Dict[str, Any]]  # ClusterThemeAnalysis: {cluster_summary, themes[]}
#     step2_analysis: Dict[Union[int, str], List[Dict[str, str]]]  # List[CandidateCode]: Array of candidate codes
#     step3_recommendations: Dict[Union[int, str], Dict[str, Any]]  # CodeRecommendation: {coding_decisions[]}  
#     step4_validations: Dict[Union[int, str], Dict[str, Any]]  # ValidationResult: {code_validations[]}
#     step4_validated_codes: Dict[Union[int, str], Dict[str, Any]] = {}  # Final validated codes from Step 4
    
#     stats: Dict[str, Any]
#     generator_version: str
#     var_lab: str
#     total_clusters: int
#     total_ideas: int
#     processing_timestamp: str
    
#     cluster_assignments: Dict[Union[int, str], Dict[str, Any]]
    
#     codebook: List[Dict[str, str]]  # Final deduplicated codebook from SharedCodebook
#     cluster_data: Dict[Union[int, str], Dict[str, Any]]  # Raw cluster data for stats calculations (supports sub-clusters)
#     validation_details: Optional[Dict[Union[int, str], Any]] = None  # Detailed validation results (maps to step4_validations)
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)
    
#     def get(self, key: str, default=None):
#         """Dictionary-style access for promptTester compatibility"""
#         return getattr(self, key, default)

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




