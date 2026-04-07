"""
Embedding-based code pre-filtering for P10 assignment.

Embeds ideas and codes, computes cosine similarity, and returns
top-N candidate code indices per idea. Used to scope the P10
assignment prompt to a small subset of relevant codes.

Delegates embedding to SharedEmbedder (src/utils/embedder.py).
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.embedder import SharedEmbedder


class EmbeddingMatcher:
    """Embed texts and compute top-N matches via cosine similarity."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_concurrent: int = 5,
    ):
        self._embedder = SharedEmbedder(
            model=model,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts in batches. Returns [N × dim] numpy array."""
        return await self._embedder.embed_texts(texts)

    def compute_top_n(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        n: int = 5,
    ) -> List[List[int]]:
        """For each query, return indices of top-N corpus items by cosine similarity.

        Returns list of lists, each containing N indices sorted by
        similarity (highest first).
        """
        n = min(n, corpus_embeddings.shape[0])
        sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)

        top_indices = []
        for row in sim_matrix:
            sorted_idx = np.argsort(row)[::-1][:n]
            top_indices.append(sorted_idx.tolist())

        return top_indices

    # =========================================================================
    # TEXT BUILDERS
    # =========================================================================

    @staticmethod
    def build_idea_text(idea, facet_lookup: Dict[str, str]) -> str:
        """Build embedding text for an idea."""
        domain = getattr(idea, 'domain', '') or ''
        facet = facet_lookup.get(idea.idea_id, '') or getattr(idea, 'facet', '') or ''
        interpretation = getattr(idea, 'interpretation', '') or ''
        abstraction = getattr(idea, 'abstraction', '') or ''
        return f"{domain} | {facet} | {interpretation} | {abstraction}"

    @staticmethod
    def build_code_text(code) -> str:
        """Build embedding text for a code."""
        indicators = ', '.join(code.typical_indicators[:5]) if code.typical_indicators else ''
        return f"{code.code_name} | {code.definition} | {indicators}"
