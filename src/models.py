from typing import List, Any, Optional, Type, Dict
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt

class ResponseSegmentModel(BaseModel):
    segment_id: str
    segment_response: str
    model_config = ConfigDict(arbitrary_types_allowed=True) # for arrays with embeddings

class ResponseModel(BaseModel):
    respondent_id: Any
    response: str
    model_config = ConfigDict(arbitrary_types_allowed=True) # for arrays with embeddings
 
    def to_model(self, model_class: Type['BaseModel']) -> 'BaseModel':
        data = self.model_dump()
        return model_class(**data)

class PreprocessModel(ResponseModel):
    pass

class DescriptiveSubmodel(ResponseSegmentModel):
    descriptive_code: Optional[str] = None
    code_description: Optional[str] = None

class DescriptiveModel(ResponseModel):
    quality_filter: Optional[bool] = None
    response_segment: Optional[List[DescriptiveSubmodel]] = None

class EmbeddingsSubmodel(DescriptiveSubmodel):
    code_embedding: Optional[npt.NDArray[np.float32]] = None
    description_embedding: Optional[npt.NDArray[np.float32]] = None
    
class EmbeddingsModel(DescriptiveModel):
    response_segment: Optional[List[EmbeddingsSubmodel]] = None

class ClusterSubmodel(EmbeddingsSubmodel):
    #clusters with keywords
    meta_cluster: Optional[Dict[int, str]] = None 
    macro_cluster: Optional[Dict[int, str]] = None
    micro_cluster: Optional[Dict[int, str]] = None
    
class ClusterModel(EmbeddingsModel):
    response_segment: Optional[List[ClusterSubmodel]] = None

class LabelSubmodel(ClusterSubmodel):
    #clusters with labels
    Theme: Optional[Dict[int, str]] = None 
    Topic: Optional[Dict[int, str]] = None
    Keyword: Optional[Dict[int, str]] = None

class LabelModel(ClusterModel):
    summary: Optional[str] = None
    response_segment: Optional[List[LabelSubmodel]] = None

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