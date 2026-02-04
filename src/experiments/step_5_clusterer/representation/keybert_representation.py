"""
KeyBERT-inspired embedding-based keyword representation

Uses semantic embeddings to select keywords most representative of cluster content.
Adapted from KeyBERT methodology for cluster analysis.

The algorithm:
1. Calculate cluster centroid from idea embeddings
2. Generate embeddings for candidate keywords
3. Rank keywords by cosine similarity to cluster centroid
4. Optionally combine with c-TF-IDF scores

This approach captures semantic meaning beyond statistical frequency,
selecting keywords that best represent the cluster's conceptual center.

Usage:
    from experiments.representation.keybert_representation import KeyBERTRepresentation

    keybert = KeyBERTRepresentation(top_k=10, weight=0.5)
    keywords = keybert.extract_topics(
        cluster_id, ctfidf_scores, vocabulary, cluster_texts, embeddings
    )
"""
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.llm import create_embedding_client
from .base import BaseRepresentation


class KeyBERTRepresentation(BaseRepresentation):
    """
    Embedding-based keyword selection (KeyBERT-inspired)

    Args:
        top_k: Number of keywords to extract
        embedding_model: Model name for keyword embeddings
        weight: Balance between embedding similarity and c-TF-IDF (0.0-1.0)
                0.0 = pure c-TF-IDF
                1.0 = pure embedding similarity
                0.5 = equal weight (default)
        candidate_multiplier: Extract top_k * candidate_multiplier candidates by c-TF-IDF
    """

    def __init__(
        self,
        top_k: int = 10,
        embedding_model: str = "text-embedding-3-large",
        weight: float = 0.5,
        candidate_multiplier: int = 3
    ):
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be between 0.0 and 1.0, got {weight}")

        self.top_k = top_k
        self.embedding_model = embedding_model
        self.weight = weight
        self.candidate_multiplier = candidate_multiplier
        self.client = None

    def _get_client(self):
        """Lazy initialization of embedding client"""
        if self.client is None:
            self.client = create_embedding_client(async_mode=False)
        return self.client

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        client = self._get_client()

        response = client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )

        embeddings = np.array([data.embedding for data in response.data])
        return embeddings

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
        Extract keywords using embedding similarity to cluster centroid

        Args:
            cluster_id: Cluster identifier
            ctfidf_scores: c-TF-IDF scores for this cluster (1D array)
            vocabulary: Feature names from vectorizer
            cluster_texts: Original idea texts
            embeddings: Pre-computed embeddings for cluster texts (optional)

        Returns:
            List of (keyword, combined_score) tuples
        """
        # Step 1: Get candidate keywords by c-TF-IDF
        n_candidates = min(
            self.top_k * self.candidate_multiplier,
            len([s for s in ctfidf_scores if s > 0])
        )

        if n_candidates == 0:
            return []

        candidate_indices = np.argsort(ctfidf_scores)[-n_candidates:][::-1]
        candidate_keywords = [vocabulary[i] for i in candidate_indices]
        candidate_ctfidf = ctfidf_scores[candidate_indices]

        # Normalize c-TF-IDF scores to [0, 1]
        if candidate_ctfidf.max() > 0:
            normalized_ctfidf = candidate_ctfidf / candidate_ctfidf.max()
        else:
            normalized_ctfidf = candidate_ctfidf

        # Step 2: Calculate cluster centroid from embeddings
        if embeddings is not None and len(embeddings) > 0:
            cluster_centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        else:
            # Generate embeddings for cluster texts
            text_embeddings = self._get_embeddings(cluster_texts)
            cluster_centroid = np.mean(text_embeddings, axis=0).reshape(1, -1)

        # Step 3: Generate embeddings for candidate keywords
        keyword_embeddings = self._get_embeddings(candidate_keywords)

        # Step 4: Calculate cosine similarity to cluster centroid
        similarities = cosine_similarity(keyword_embeddings, cluster_centroid).flatten()

        # Normalize similarities to [0, 1]
        if similarities.max() > 0:
            normalized_similarities = (similarities - similarities.min()) / (
                similarities.max() - similarities.min()
            )
        else:
            normalized_similarities = similarities

        # Step 5: Combine c-TF-IDF and embedding similarity
        combined_scores = (
            (1 - self.weight) * normalized_ctfidf +
            self.weight * normalized_similarities
        )

        # Step 6: Select top-k keywords by combined score
        top_indices = np.argsort(combined_scores)[-self.top_k:][::-1]

        keywords = [
            (candidate_keywords[i], float(combined_scores[i]))
            for i in top_indices
        ]

        return keywords

    def get_keyword_analysis(
        self,
        keywords: List[Tuple[str, float]],
        cluster_texts: List[str],
        ctfidf_scores: np.ndarray,
        vocabulary: List[str]
    ) -> dict:
        """
        Analyze keyword selection showing c-TF-IDF vs embedding contributions

        Args:
            keywords: Selected (keyword, combined_score) tuples
            cluster_texts: Original cluster texts
            ctfidf_scores: Full c-TF-IDF score array
            vocabulary: Full vocabulary

        Returns:
            Dict with detailed keyword analysis
        """
        if not keywords:
            return {}

        # Get embeddings
        keyword_list = [kw for kw, _ in keywords]
        keyword_embeddings = self._get_embeddings(keyword_list)

        # Calculate cluster centroid
        text_embeddings = self._get_embeddings(cluster_texts)
        cluster_centroid = np.mean(text_embeddings, axis=0).reshape(1, -1)

        # Get similarities and c-TF-IDF scores
        similarities = cosine_similarity(keyword_embeddings, cluster_centroid).flatten()

        analysis = []
        for idx, (keyword, combined_score) in enumerate(keywords):
            # Find keyword in vocabulary
            vocab_idx = vocabulary.tolist().index(keyword) if keyword in vocabulary else None

            ctfidf_score = ctfidf_scores[vocab_idx] if vocab_idx is not None else 0.0
            embedding_sim = similarities[idx]

            analysis.append({
                "keyword": keyword,
                "combined_score": float(combined_score),
                "ctfidf_score": float(ctfidf_score),
                "embedding_similarity": float(embedding_sim),
                "weight": self.weight
            })

        return {
            "keywords": analysis,
            "weight": self.weight,
            "embedding_model": self.embedding_model
        }
