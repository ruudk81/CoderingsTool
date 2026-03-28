"""
Data models for Code Generator (P8-P9).

Coding results cache for the codebook checkpoint.
Imports taxonomy types from step 4 (upstream dependency).
"""

from typing import List, Dict
from pydantic import BaseModel, Field

from development.step_4_classifier.models_classifier import (
    DomainSet,
    DomainResultModel,
)


class CodingResultsCache(BaseModel):
    """Cache for codebook results (taxonomy + codes)."""
    partition_set: DomainSet
    partition_results: Dict[str, DomainResultModel]
    label_counts: Dict[str, int] = Field(default_factory=dict)
    label_source: str = ""
    total_categories: int = 0
    raw_codes: List[Dict] = Field(default_factory=list)  # ConsolidatedCode dicts
