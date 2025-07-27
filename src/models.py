from typing import List, Any, Optional, Type, Union
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt

# === GROWING RESPONSE/RESULT MODELS ========================================================================================================

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
    idea_embeddings: Optional[List[EmbeddingsSubmodel]] = None

class ClusterSubmodel(EmbeddingsSubmodel):
    initial_cluster: Optional[int] = None   
    
class ClusterModel(IdeasExtractedModel):
    response_ideas: Optional[List[ClusterSubmodel]] = None  

# === CODEBOOK MODELS ========================================================================================================

class CodeDefinition(BaseModel):
    code: str
    definition: str

class Codebook(BaseModel):
    code: str
    definition: str    
    topic: Optional[str] = None   
    theme: Optional[str] = None   
    
class SuggestedTheme(BaseModel):
    theme_name: str
    concept: str
    codes: List[str]
    relationship: str

class Reflection(BaseModel):
    broad_or_narrow_themes: Optional[str] = None
    contradictions_or_unexpected_patterns: Optional[str] = None
    potential_subthemes: Optional[str] = None
    unclassified_codes: Optional[List[str]] = None

class ThemeAnalysis(BaseModel):
    initial_observations: List[str]
    suggested_themes: List[SuggestedTheme]
    reflection: Reflection



