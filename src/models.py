from typing import List, Any, Optional, Type, Union, Dict
from pydantic import BaseModel, ConfigDict
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
    topic: Optional[str] = None   
    theme: Optional[str] = None
    theme_description: Optional[str] = None   

class CodebookEntry(BaseModel):
    """Individual code from generated codebook"""
    code: str
    definition: str
    source_clusters: Optional[List[int]] = None  # Which clusters influenced this code
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodebookModel(BaseModel):
    """Generated codebook from cluster analysis (Step 7)"""
    codes: List[CodebookEntry]
    generation_metadata: Optional[Dict[str, Any]] = None
    source_variable: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ThemeEnrichedCodebookEntry(CodebookEntry):
    """Code enriched with theme information"""
    theme: Optional[str] = None
    theme_description: Optional[str] = None
    theme_cluster_id: Optional[int] = None
    is_miscellaneous: Optional[bool] = False

class ThemeEnrichedCodebookModel(CodebookModel):
    """Codebook enriched with themes (Step 8)"""
    codes: List[ThemeEnrichedCodebookEntry]  # Override with enriched version
    themes_summary: Optional[List[Dict[str, Any]]] = None
    code_to_theme_mapping: Optional[Dict[str, str]] = None
    theme_methodology: Optional[str] = None

    
# === CODE GENERATOR REASONING RESULTS MODEL ========================================================================================================

class CodeGeneratorReasoningResults(BaseModel):
    """Detailed LLM reasoning and decisions from code generation process - Multi-theme support"""
    
    # All cluster processing details
    cluster_results: List[Dict[str, Any]]  # Raw results from each cluster
    
    # Separated by step for easy analysis - Multi-theme JSON structures
    step1_summaries: Dict[int, Dict[str, Any]]  # ClusterThemeAnalysis: {cluster_summary, themes[]}
    step2_analysis: Dict[int, List[Dict[str, str]]]  # List[CandidateCode]: Array of candidate codes
    step3_recommendations: Dict[int, Dict[str, Any]]  # CodeRecommendation: {coding_decisions[]}  
    step4_validations: Dict[int, Dict[str, Any]]  # ValidationResult: {code_validations[]}
    step4_validated_codes: Dict[int, Dict[str, Any]] = {}  # Final validated codes from Step 4
    
    # Processing metadata
    stats: Dict[str, Any]
    generator_version: str
    var_lab: str
    total_clusters: int
    total_ideas: int
    processing_timestamp: str
    
    # Cluster assignments for cross-reference
    cluster_assignments: Dict[int, str]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# === MULTI-THEME JSON RESPONSE MODELS ========================================================================================================

class ClusterThemeAnalysis(BaseModel):
    """Step 1 JSON response: Cluster summary with themes array"""
    cluster_summary: str
    themes: List[str]  # Array of theme statements with rationales
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CandidateCode(BaseModel):
    """Individual candidate code from Step 2"""  
    code: str
    definition: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ActionDetails(BaseModel):
    """Action details for coding decisions in Step 3"""
    codes_to_use: Optional[List[str]] = None
    codes_to_modify: Optional[str] = None
    modified_code_name: Optional[str] = None
    modified_code_definition: Optional[str] = None
    new_code_name: Optional[str] = None
    new_code_definition: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodingDecision(BaseModel):
    """Individual coding decision for a theme from Step 3"""
    theme_number: int
    theme_description: str
    decision: str  # use_existing | modify_existing | create_new
    action_details: ActionDetails
    justification: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClusterAnalysis(BaseModel):
    """Cluster analysis metadata from Step 3"""
    number_of_themes: int
    theme_descriptions: List[str]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeRecommendation(BaseModel):
    """Step 3 JSON response: Multi-theme code recommendations"""
    cluster_analysis: ClusterAnalysis
    coding_decisions: List[CodingDecision]
    overall_justification: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeEvaluation(BaseModel):
    """Evaluation criteria for Step 4 validation"""
    semantic_fit: str
    atomicity: str
    parsimony: str
    redundancy: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ValidatedCode(BaseModel):
    """Final validated code from Step 4"""
    code: str
    definition: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeValidation(BaseModel):
    """Individual code validation from Step 4"""
    theme_number: int
    theme_description: str
    original_recommendation: str
    evaluation: CodeEvaluation
    decision: str  # APPROVE | REVISE | REJECT | MERGE | SPLIT
    decision_rationale: str
    validated_code: ValidatedCode
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ThemeAssessment(BaseModel):
    """Theme assessment from Step 4"""
    number_of_themes_identified: int
    theme_separation_valid: bool
    theme_separation_reasoning: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OverallValidation(BaseModel):
    """Overall validation summary from Step 4"""
    all_themes_coded: bool
    final_code_count: int
    summary: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ValidationResult(BaseModel):
    """Step 4 JSON response: Multi-theme validation results"""
    theme_assessment: ThemeAssessment
    code_validations: List[CodeValidation]
    overall_validation: OverallValidation
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# class SuggestedTheme(BaseModel):
#     theme_name: str
#     concept: str
#     codes: List[str]
#     relationship: str

# class Reflection(BaseModel):
#     broad_or_narrow_themes: Optional[str] = None
#     contradictions_or_unexpected_patterns: Optional[str] = None
#     potential_subthemes: Optional[str] = None
#     unclassified_codes: Optional[List[str]] = None

# class ThemeAnalysis(BaseModel):
#     initial_observations: List[str]
#     suggested_themes: List[SuggestedTheme]
#     reflection: Reflection



