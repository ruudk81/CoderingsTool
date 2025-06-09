from typing import List, Any, Optional, Type, Dict, Union
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt

class ResponseSegmentModel(BaseModel):
    segment_id: str
    segment_response: str
    model_config = ConfigDict(arbitrary_types_allowed=True) # for arrays with embeddings

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

class SegmentSubmodel(ResponseSegmentModel):
    """Submodel for response segments with labels and descriptions"""
    segment_label: Optional[str] = None
    segment_description: Optional[str] = None

class SegmentedModel(QualityFilteredModel):
    """Model for Step 4: Segmented responses with codes and descriptions"""
    response_segment: Optional[List[SegmentSubmodel]] = None

class ClusterSubmodel(SegmentSubmodel):
    """Submodel for segments with embeddings and initial cluster assignment"""
    code_embedding: Optional[npt.NDArray[np.float32]] = None
    description_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_cluster: Optional[int] = None  # Single cluster ID from initial clustering
    
class ClusterModel(SegmentedModel):
    """Model for Step 5: Initial clusters with embeddings"""
    response_segment: Optional[List[ClusterSubmodel]] = None

class LabelSubmodel(ClusterSubmodel):
    """Submodel for segments with hierarchical labeling information"""
    Theme: Optional[Dict[int, str]] = None 
    Topic: Optional[Dict[float, str]] = None  # Float keys for topic IDs like 1.1, 1.2, etc.
    Code: Optional[Dict[float, str]] = None  # Float keys for code IDs like 1.1.1
    
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Force model rebuild

class HierarchicalCode(BaseModel):
    """Represents a single code in the hierarchical structure"""
    code_id: str  # e.g., "1.1.1"
    numeric_id: float  # e.g., 1.11
    label: str
    description: str
    parent_id: str  # e.g., "1.1"
    level: int  # 3 for codes
    
class HierarchicalTopic(BaseModel):
    """Represents a single topic in the hierarchical structure"""
    topic_id: str  # e.g., "1.1"
    numeric_id: float  # e.g., 1.1
    label: str
    description: str
    parent_id: str  # e.g., "1"
    level: int  # 2 for topics
    codes: List[HierarchicalCode] = []
    
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
    """Model for Step 6: Hierarchical labeling with themes, topics, and codes"""
    summary: Optional[str] = None
    response_segment: Optional[List[LabelSubmodel]] = None
    # Hierarchical structure data
    themes: Optional[List[HierarchicalTheme]] = None
    cluster_mappings: Optional[List[ClusterMapping]] = None

# conversion    
def to_model(self, model_class: Type['BaseModel']) -> 'BaseModel':
    data = self.model_dump()
    
    if hasattr(self, 'response_segment') and self.response_segment:
    
        from typing import get_type_hints, get_args
        type_hints = get_type_hints(model_class)
        
        if 'response_segment' in type_hints:
            submodel_type = get_args(get_args(type_hints['response_segment'])[0])[0]
            converted_segments = []
            for segment in self.response_segment:
                segment_data = segment.model_dump()
                converted_segments.append(submodel_type(**segment_data))
            data['response_segment'] = converted_segments
    
    return model_class(**data)

# Backward compatibility aliases
DescriptiveModel = SegmentedModel
DescriptiveSubmodel = SegmentSubmodel
PreprocessModel = PreprocessedModel
EmbeddingsModel = ClusterModel  # For pipeline compatibility
EmbeddingsSubmodel = ClusterSubmodel  # For pipeline compatibility