"""
Base interface for cluster representation models

All representation models must inherit from BaseRepresentation and implement
the extract_topics() method.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class BaseRepresentation(ABC):
    """Base class for cluster representation models"""

    @abstractmethod
    def extract_topics(
        self,
        cluster_id: int,
        ctfidf_scores: np.ndarray,
        vocabulary: List[str],
        cluster_texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Extract representative keywords for a cluster

        Args:
            cluster_id: Cluster identifier
            ctfidf_scores: c-TF-IDF scores for this cluster (1D array)
            vocabulary: Feature names from vectorizer
            cluster_texts: Original idea texts in this cluster
            embeddings: Optional embeddings for ideas in cluster
            **kwargs: Additional model-specific parameters

        Returns:
            List of (keyword, score) tuples, ordered by relevance
        """
        pass
