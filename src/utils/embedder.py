"""
Shared embedding utility for the CoderingsTool pipeline.

Provides batched async text embedding, text formatting for different
code_source modes, medoid computation, and representative sample selection.

Used by step 5 (code generation) and step 6 (code assignment).
"""

import asyncio
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from utils.llm import create_embedding_client


# =============================================================================
# SHARED EMBEDDER
# =============================================================================

class SharedEmbedder:
    """Batched async text embedding via OpenAI API."""

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
        """Embed texts in batches. Returns [N x dim] numpy array."""
        await self._ensure_client()

        if not texts:
            return np.array([], dtype=np.float32)

        batches = [
            texts[i:i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def process_batch(batch: List[str]) -> List[np.ndarray]:
            async with semaphore:
                return await self._with_retries(lambda: self._embed_batch(batch))

        batch_results = await asyncio.gather(*[process_batch(b) for b in batches])

        all_embeddings = []
        for result in batch_results:
            all_embeddings.extend(result)

        return np.array(all_embeddings, dtype=np.float32)


# =============================================================================
# TEXT FORMATTING
# =============================================================================

def format_idea_text(idea, code_source: str) -> str:
    """Format idea text for embedding based on code_source mode.

    Modes:
        idea                   — idea.idea (natural sentence, includes template_prefix)
        instance               — idea.instance (shortest verbatim span)
        instance_interpretation — "{instance} | {interpretation}"
        full_abstraction_ladder — "{instance} | {interpretation} | {abstraction}"
    """
    if code_source == "idea":
        return idea.idea

    if code_source == "instance":
        return getattr(idea, "instance", "") or idea.idea

    instance = (getattr(idea, "instance", "") or "").strip()
    interpretation = (getattr(idea, "interpretation", "") or "").strip()
    abstraction = (getattr(idea, "abstraction", "") or "").strip()

    if code_source == "instance_interpretation":
        parts = [p for p in (instance, interpretation) if p]
        return " | ".join(parts) if parts else idea.idea

    if code_source == "full_abstraction_ladder":
        parts = [p for p in (instance, interpretation, abstraction) if p]
        return " | ".join(parts) if parts else idea.idea

    # Fallback to idea text for unknown code_source
    return idea.idea


# =============================================================================
# MEDOID & REPRESENTATIVE SAMPLES
# =============================================================================

def compute_medoid(embeddings: np.ndarray) -> int:
    """Return index of the medoid — the point minimizing total distance to all others.

    Uses cosine distance. For a single point, returns 0.
    """
    if len(embeddings) <= 1:
        return 0

    dist_matrix = cosine_distances(embeddings)
    total_distances = dist_matrix.sum(axis=1)
    return int(np.argmin(total_distances))


def find_representative_samples(embeddings: np.ndarray, n: int = 3) -> List[int]:
    """Return indices of up to n representative samples: the medoid + closest neighbors.

    For groups with <= n points, returns all indices.
    """
    if len(embeddings) <= n:
        return list(range(len(embeddings)))

    medoid_idx = compute_medoid(embeddings)

    # Distances from medoid to all other points
    dist_matrix = cosine_distances(embeddings)
    distances_from_medoid = dist_matrix[medoid_idx]

    # Sort by distance to medoid, take n closest (medoid itself has distance 0)
    sorted_indices = np.argsort(distances_from_medoid)
    return sorted_indices[:n].tolist()
