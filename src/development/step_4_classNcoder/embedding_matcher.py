"""
Embedding-based code pre-filtering for P5 assignment.

Embeds ideas and codes, computes cosine similarity, and returns
top-N candidate code indices per idea. Used to scope the P5
assignment prompt to a small subset of relevant codes.
"""

import asyncio
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.llm import create_embedding_client


class EmbeddingMatcher:
    """Embed texts and compute top-N matches via cosine similarity."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_concurrent: int = 5,
    ):
        self._model = model
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = create_embedding_client(async_mode=True)

    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a single batch via OpenAI API."""
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    async def _with_retries(self, fn, retries: int = 3, base: float = 0.8):
        """Retry with exponential backoff."""
        for i in range(retries):
            try:
                return await fn()
            except asyncio.CancelledError:
                raise
            except Exception:
                if i == retries - 1:
                    raise
                await asyncio.sleep(base * (2 ** i))

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts in batches. Returns [N × dim] numpy array."""
        await self._ensure_client()

        if not texts:
            return np.array([], dtype=np.float32)

        # Create batches
        batches = []
        for i in range(0, len(texts), self._batch_size):
            batches.append(texts[i:i + self._batch_size])

        # Process concurrently with semaphore
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def process_batch(batch: List[str]) -> List[np.ndarray]:
            async with semaphore:
                return await self._with_retries(lambda: self._embed_batch(batch))

        batch_results = await asyncio.gather(*[process_batch(b) for b in batches])

        # Flatten
        all_embeddings = []
        for result in batch_results:
            all_embeddings.extend(result)

        return np.array(all_embeddings, dtype=np.float32)

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
