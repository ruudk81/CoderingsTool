from typing import List, Any, Optional, Type, Dict, Union
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt


class ResponseModel(BaseModel):
    respondent_id: Any
    response: Union[str, float, int]  # Allow string, float, or int types
    response_type: Optional[str] = None  # Track original type: 'string', 'numeric', 'nan'
    model_config = ConfigDict(arbitrary_types_allowed=True) # for arrays with embeddings
 
    def to_model(self, model_class: Type['BaseModel']) -> 'BaseModel':
        data = self.model_dump()
        return model_class(**data)

class PreprocessedModel(ResponseModel):
    """Model for Step 2: Preprocessed text data with initial quality assessment"""
    quality_filter: Optional[bool] = None
    quality_filter_code: Optional[int] = None  # 0=meaningful, 99999997=user_missing, 99999998=system_missing, 99999999=no_answer

class QualityFilteredModel(PreprocessedModel):
    """Model for Step 3: Quality filtered data (same fields, but after LLM quality grading)"""
    pass

class IdeaSubmodel(BaseModel):
    """Submodel for individual extracted ideas with unique tracking"""
    idea_id: str  # Format: {respondent_id}_{sequence_number}
    idea_summary: str   
    

class IdeaModel(QualityFilteredModel):
    """Model for GATOS Step 1: Extracted ideas from responses"""
    response_ideas: Optional[List[IdeaSubmodel]] = None
    idea_count: int = 0


class EmbeddingSubmodel(IdeaSubmodel):
    """Submodel for ideas with embeddings"""
    idea_embedding: Optional[npt.NDArray[np.float32]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
class EmbeddingModel(IdeaModel):
    """Model for Step 5: Ideas with embeddings"""
    idea_embeddings: Optional[List[EmbeddingSubmodel]] = None

class ClusterSubmodel(EmbeddingSubmodel):
    """Submodel for ideas with embeddings and cluster assignment"""
    initial_cluster: Optional[int] = None  # Single cluster ID from initial clustering
    
class ClusterModel(EmbeddingModel):
    """Model for Step 6: Ideas with embeddings and clusters"""
    idea_embeddings: Optional[List[ClusterSubmodel]] = None

class LabelSubmodel(ClusterSubmodel):
    """Submodel for ideas with hierarchical labeling information"""
    Theme: Optional[Dict[int, str]] = None 
    Topic: Optional[Dict[float, str]] = None  # Float keys for topic IDs like 1.1, 1.2, etc.
    Code: Optional[Dict[float, str]] = None  # Float keys for code IDs like 1.1.1
    
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Force model rebuild

class HierarchicalTopic(BaseModel):
    """Represents a single topic in the hierarchical structure"""
    topic_id: str  # e.g., "1.1"
    numeric_id: float  # e.g., 1.1
    label: str
    description: str
    parent_id: str  # e.g., "1"
    level: int  # 2 for topics
    
class HierarchicalTheme(BaseModel):
    """Represents a single theme in the hierarchical structure"""
    theme_id: str  # e.g., "1"
    numeric_id: float  # e.g., 1.0
    label: str
    description: str
    level: int  # 1 for themes
    topics: List[HierarchicalTopic] = []
    
class ClusterMapping(BaseModel):
    """Maps clusters to the hierarchical structure"""
    cluster_id: int
    cluster_label: str
    theme_id: str
    topic_id: str
    code_id: str
    confidence: float = 1.0

class LabelModel(ClusterModel):
    """Model for Step 7: Hierarchical labeling with themes, topics, and codes"""
    summary: Optional[str] = None
    idea_embeddings: Optional[List[LabelSubmodel]] = None
    # Hierarchical structure data
    themes: Optional[List[HierarchicalTheme]] = None
    cluster_mappings: Optional[List[ClusterMapping]] = None

# conversion    
def to_model(self, model_class: Type['BaseModel']) -> 'BaseModel':
    data = self.model_dump()
    
    if hasattr(self, 'idea_embeddings') and self.idea_embeddings:
    
        from typing import get_type_hints, get_args
        type_hints = get_type_hints(model_class)
        
        if 'idea_embeddings' in type_hints:
            submodel_type = get_args(get_args(type_hints['idea_embeddings'])[0])[0]
            converted_embeddings = []
            for embedding in self.idea_embeddings:
                embedding_data = embedding.model_dump()
                converted_embeddings.append(submodel_type(**embedding_data))
            data['idea_embeddings'] = converted_embeddings
    
    return model_class(**data)

# Backward compatibility aliases
PreprocessModel = PreprocessedModel
EmbeddingsModel = EmbeddingModel  # For pipeline compatibility
EmbeddingsSubmodel = EmbeddingSubmodel  # For pipeline compatibility