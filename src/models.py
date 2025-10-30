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

"""codebook_main"""

class CodebookEntry(BaseModel):
    code: str
    definition: str
    source_cluster: Optional[str] = None    # Support sub-clusters like "12-1", "12-2"
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

class CodeRefinementResults(BaseModel):
    original_codebook: List[Dict[str, Any]]
    refined_codebook: RefinedCodebookModel
    processing_stats: Dict[str, Any]
    timestamp: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

"""enriched_codebook"""
class CodeDefinition(BaseModel):
    code: str
    definition: str

class Codebook(BaseModel):
    code: str
    definition: str
    source_cluster:  Optional[str] = None    # Support sub-clusters like "12-1", "12-2"    
    theme: Optional[str] = None
    theme_description: Optional[str] = None   

"""theme enriched_codebook"""
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





