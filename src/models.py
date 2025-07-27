from typing import List, Any, Optional, Type, Union, Dict
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt


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

class CodeDefinition(BaseModel):
    """Single code with definition for GATOS codebook"""
    code: str
    definition: str
    
# class CodeGenerationDecision(BaseModel):
#     """Decision from inductive codebook generation"""
#     step_by_step_analysis: str
#     create_new_code: bool
#     new_code: Optional[str] = None
#     new_definition: Optional[str] = None
#     reasoning: str

# class Theme(BaseModel):
#     """Theme from theme identification"""
#     theme_name: str
#     concept: str
#     codes: List[str]
#     relationship: str

# class ThemeAnalysis(BaseModel):
#     """Complete theme analysis output"""
#     initial_observations: List[str]
#     suggested_themes: List[Theme]
#     reflection: Dict[str, str]


# class GATOSFinalResults(BaseModel):
#     """Final GATOS results with themes"""
#     codebook: GATOSCodebook
#     themes: List[Theme]
#     theme_analysis: ThemeAnalysis

# class LabelSubmodel(ClusterSubmodel):
#     """Submodel for ideas with hierarchical labeling information"""
#     Theme: Optional[Dict[int, str]] = None 
#     Topic: Optional[Dict[float, str]] = None  # Float keys for topic IDs like 1.1, 1.2, etc.
#     Code: Optional[Dict[float, str]] = None  # Float keys for code IDs like 1.1.1
    
#     model_config = ConfigDict(arbitrary_types_allowed=True)  # Force model rebuild

# class HierarchicalTopic(BaseModel):
#     """Represents a single topic in the hierarchical structure"""
#     topic_id: str  # e.g., "1.1"
#     numeric_id: float  # e.g., 1.1
#     label: str
#     description: str
#     parent_id: str  # e.g., "1"
#     level: int  # 2 for topics
    
# class HierarchicalTheme(BaseModel):
#     """Represents a single theme in the hierarchical structure"""
#     theme_id: str  # e.g., "1"
#     numeric_id: float  # e.g., 1.0
#     label: str
#     description: str
#     level: int  # 1 for themes
#     topics: List[HierarchicalTopic] = []
    
# class ClusterMapping(BaseModel):
#     """Maps clusters to the hierarchical structure"""
#     cluster_id: int
#     cluster_label: str
#     theme_id: str
#     topic_id: str
#     code_id: str
#     confidence: float = 1.0

# class LabelModel(ClusterModel):
#     """Model for Step 7: Hierarchical labeling with themes, topics, and codes"""
#     summary: Optional[str] = None
#     idea_embeddings: Optional[List[LabelSubmodel]] = None
#     # Hierarchical structure data
#     themes: Optional[List[HierarchicalTheme]] = None
#     cluster_mappings: Optional[List[ClusterMapping]] = None

# # conversion    
# def to_model(self, model_class: Type['BaseModel']) -> 'BaseModel':
#     data = self.model_dump()
    
#     if hasattr(self, 'idea_embeddings') and self.idea_embeddings:
    
#         from typing import get_type_hints, get_args
#         type_hints = get_type_hints(model_class)
        
#         if 'idea_embeddings' in type_hints:
#             submodel_type = get_args(get_args(type_hints['idea_embeddings'])[0])[0]
#             converted_embeddings = []
#             for embedding in self.idea_embeddings:
#                 embedding_data = embedding.model_dump()
#                 converted_embeddings.append(submodel_type(**embedding_data))
#             data['idea_embeddings'] = converted_embeddings
    
#     return model_class(**data)

# # Backward compatibility aliases
# PreprocessModel = PreprocessedModel

