from typing import List, Any, Optional, Type, Union, Dict
from pydantic import BaseModel, ConfigDict, Field, validator
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

# === HIERARCHICAL THEME IDENTIFICATION MODELS ========================================================================================================

class CodeAssignment(BaseModel):
    """Assignment of a code to a domain with rationale"""
    code_number: int = Field(description="The original code number (1-64)")
    code_name: str = Field(description="The original code name")
    fit_rationale: str = Field(description="Brief explanation why this code fits this domain")

class DomainDefinition(BaseModel):
    """Domain level grouping of related codes"""
    domain_name: str = Field(description="Clear, descriptive name for the domain")
    domain_description: str = Field(description="Brief description of what unites these codes")
    codes: List[CodeAssignment] = Field(description="Codes belonging to this domain")
    
    @validator('codes')
    def validate_codes_count(cls, v):
        if len(v) < 2:
            raise ValueError("A domain should contain at least 2 codes")
        return v

class ThemeDefinition(BaseModel):
    """High-level theme containing multiple domains"""
    theme_name: str = Field(description="Conceptual name for the theme")
    theme_concept: str = Field(description="Explanation of the overarching concept")
    domains: List[DomainDefinition] = Field(description="Domains belonging to this theme")
    
    @validator('domains')
    def validate_domains_count(cls, v):
        if len(v) < 1:
            raise ValueError("A theme should contain at least 1 domain")
        return v

class DomainClusteringResult(BaseModel):
    """Output for Stage 1: Domain clustering per batch"""
    batch_id: int = Field(description="Identifier for this batch")
    identified_domains: List[DomainDefinition] = Field(description="Domains identified in this batch")
    processing_notes: str = Field(description="Any observations about domain identification")

class CoverageStatistics(BaseModel):
    """Statistics about the hierarchical structure coverage"""
    total_codes: int = Field(description="Total number of original codes")
    classified_codes: int = Field(description="Number of codes assigned to domains")
    coverage_percentage: float = Field(ge=0, le=100, description="Percentage of codes classified")
    themes_count: int = Field(description="Number of themes identified")
    domains_count: int = Field(description="Number of domains identified")
    avg_codes_per_domain: float = Field(description="Average codes per domain")
    
    @validator('coverage_percentage')
    def validate_coverage(cls, v):
        if v < 95:
            raise ValueError("Coverage should be at least 95%")
        return v

class HierarchicalStructure(BaseModel):
    """Complete three-level hierarchical structure"""
    themes: List[ThemeDefinition] = Field(description="All themes with their domains and codes")
    coverage_statistics: CoverageStatistics = Field(description="Coverage metrics")
    quality_notes: str = Field(description="Reflections on the structure quality")
    
    def get_code_lookup(self) -> Dict[int, Dict[str, str]]:
        """Build lookup table: code_id -> {domain, theme}"""
        lookup = {}
        for theme in self.themes:
            for domain in theme.domains:
                for code in domain.codes:
                    lookup[code.code_number] = {
                        'domain': domain.domain_name,
                        'theme': theme.theme_name
                    }
        return lookup



