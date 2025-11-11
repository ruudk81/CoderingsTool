from typing import List, Any, Optional, Type, Union, Dict
from pydantic import BaseModel, ConfigDict #, Field, RootModel
import numpy as np
import numpy.typing as npt

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
    initial_cluster: Optional[Union[int, str]] = None
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

"""refined_codebook"""
class RefinedSubcode(BaseModel):
    id: str  # Original code ID(s) - multiple if merged (e.g., "1,2,3")
    code: str
    description: str
    category: str = ""  # Empty string when code is directly under theme; category name for 3-level hierarchy
    source_cluster: str = ""  # Cluster ID(s) from Step 6 - may be comma-separated for merged codes (e.g., "8,11,23")
    signals: List[str] = []  # 2 observable cues for code assignment (from Stage 2)
    boundary_rule: str = ""  # Comparative rule vs nearest confusable code (from Stage 2)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RefinedCodebookCategory(BaseModel):
    category: str
    subcodes: List[RefinedSubcode]
    central_pattern: str = ""  # One-sentence unifying idea for the theme (from Stage 2)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class RefinedCodebookModel(BaseModel):
    analysis: str
    refined_codebook: List[RefinedCodebookCategory]
    generation_metadata: Optional[Dict[str, Any]] = None
    source_variable: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeRefinementResults(BaseModel):
    original_codebook: List[Dict[str, Any]]
    refined_codebook: RefinedCodebookModel
    processing_stats: Dict[str, Any]
    timestamp: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

"""speculative codes"""
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

"""theme enriched_codebook"""
class ThemeEnrichedCodebookEntry(CodebookEntry):
    code: Optional[str] = None
    definition: Optional[str] = None
    theme: Optional[str] = None
    theme_description: Optional[str] = None
    category: str = ""  # Empty string for 2-level hierarchy; category name for 3-level hierarchy
    category_description: str = ""  # Category description (empty for 2-level)
    source_cluster: Optional[Union[int, str]] = None
    signals: Optional[List[str]] = None  # 2 observable cues from Step 7 refinement
    boundary_rule: Optional[str] = None  # Comparative boundary rule from Step 7 refinement
    central_pattern: Optional[str] = None  # Unifying theme idea from Step 7 refinement (stored at theme level but accessible here)  
    
class ThemeEnrichedCodebookModel(CodebookModel):
    codes: List[ThemeEnrichedCodebookEntry]  # Override with enriched version
    themes_summary: Optional[List[Dict[str, Any]]] = None
    code_to_theme_mapping: Optional[Dict[str, str]] = None
    theme_methodology: Optional[str] = None





