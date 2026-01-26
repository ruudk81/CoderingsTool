"""
BERTopic-inspired representation models for cluster keyword extraction

This module contains experimental implementations of various keyword extraction
methods adapted from BERTopic for use with CoderingsTool's clustering results.

Available representations:
- CTfidfRepresentation: Class-based TF-IDF (BERTopic's core algorithm)
- TfidfRepresentation: Basic per-cluster TF-IDF
- MMRRepresentation: Maximal Marginal Relevance (diversity-aware)
- KeyBERTRepresentation: Embedding-based keyword selection
- LLMRepresentation: LLM-enhanced keyword refinement
"""

from .base import BaseRepresentation
from .ctfidf_representation import ClassTfidfTransformer, CTfidfRepresentation
from .tfidf_representation import TfidfRepresentation
from .mmr_representation import MMRRepresentation
from .keybert_representation import KeyBERTRepresentation
from .llm_representation import LLMRepresentation

__all__ = [
    "BaseRepresentation",
    "ClassTfidfTransformer",
    "CTfidfRepresentation",
    "TfidfRepresentation",
    "MMRRepresentation",
    "KeyBERTRepresentation",
    "LLMRepresentation",
]
