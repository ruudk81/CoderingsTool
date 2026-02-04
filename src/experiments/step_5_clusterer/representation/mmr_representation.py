"""
MMR (Maximal Marginal Relevance) representation model

Implements diversity-based keyword selection that balances relevance with diversity.
Adapted from BERTopic's MMR implementation.

The MMR algorithm iteratively selects keywords that are:
1. Highly relevant to the cluster (high c-TF-IDF score)
2. Diverse from already-selected keywords (low similarity)

Formula: MMR = argmax[λ * relevance(w) - (1-λ) * max_similarity(w, selected)]

where:
- λ (diversity parameter): 0.0 = maximum diversity, 1.0 = maximum relevance
- relevance(w): normalized c-TF-IDF score for keyword w
- max_similarity(w, selected): maximum cosine similarity to already-selected keywords

Usage:
    from experiments.representation.mmr_representation import MMRRepresentation

    mmr = MMRRepresentation(diversity=0.3, top_k=10)
    keywords = mmr.extract_topics(cluster_id, ctfidf_scores, vocabulary, cluster_texts)
"""
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from .base import BaseRepresentation


class MMRRepresentation(BaseRepresentation):
    """
    MMR keyword selection balancing relevance and diversity

    Args:
        diversity: Lambda parameter in MMR formula (0.0-1.0)
                  0.0 = maximum diversity (ignore relevance)
                  1.0 = maximum relevance (ignore diversity)
                  0.3 = good balance (BERTopic default)
        top_k: Number of keywords to extract
        candidate_multiplier: Extract top_k * candidate_multiplier candidates before MMR
                            (allows more diverse selection from larger pool)
    """

    def __init__(
        self,
        diversity: float = 0.3,
        top_k: int = 10,
        candidate_multiplier: int = 3
    ):
        if not 0.0 <= diversity <= 1.0:
            raise ValueError(f"diversity must be between 0.0 and 1.0, got {diversity}")

        self.diversity = diversity
        self.top_k = top_k
        self.candidate_multiplier = candidate_multiplier

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
        Extract keywords using MMR for diversity

        Args:
            cluster_id: Cluster identifier
            ctfidf_scores: c-TF-IDF scores for this cluster (1D array)
            vocabulary: Feature names from vectorizer
            cluster_texts: Original idea texts (used for word co-occurrence)
            embeddings: Not used in this implementation

        Returns:
            List of (keyword, score) tuples, ordered by MMR selection
        """
        # Step 1: Get top candidate keywords by c-TF-IDF
        n_candidates = min(
            self.top_k * self.candidate_multiplier,
            len([s for s in ctfidf_scores if s > 0])
        )

        if n_candidates == 0:
            return []

        # Get candidate indices and scores
        candidate_indices = np.argsort(ctfidf_scores)[-n_candidates:][::-1]
        candidate_keywords = [vocabulary[i] for i in candidate_indices]
        candidate_scores = ctfidf_scores[candidate_indices]

        # Normalize scores to [0, 1] for MMR
        if candidate_scores.max() > 0:
            normalized_scores = candidate_scores / candidate_scores.max()
        else:
            normalized_scores = candidate_scores

        # Step 2: Calculate word co-occurrence matrix for similarity
        word_similarity = self._calculate_word_similarity(
            candidate_keywords,
            cluster_texts
        )

        # Step 3: Apply MMR iterative selection
        selected_keywords = []
        selected_indices = []

        for _ in range(min(self.top_k, len(candidate_keywords))):
            if len(selected_indices) == 0:
                # First selection: highest relevance
                best_idx = 0
            else:
                # Subsequent selections: balance relevance and diversity
                mmr_scores = []
                for idx in range(len(candidate_keywords)):
                    if idx in selected_indices:
                        mmr_scores.append(-np.inf)
                        continue

                    # Relevance component
                    relevance = normalized_scores[idx]

                    # Diversity component: max similarity to selected keywords
                    similarities = [
                        word_similarity[idx, sel_idx]
                        for sel_idx in selected_indices
                    ]
                    max_similarity = max(similarities) if similarities else 0.0

                    # MMR formula
                    mmr = self.diversity * relevance - (1 - self.diversity) * max_similarity
                    mmr_scores.append(mmr)

                best_idx = np.argmax(mmr_scores)

            selected_indices.append(best_idx)
            keyword = candidate_keywords[best_idx]
            score = float(candidate_scores[best_idx])  # Use original c-TF-IDF score
            selected_keywords.append((keyword, score))

        return selected_keywords

    def _calculate_word_similarity(
        self,
        keywords: List[str],
        cluster_texts: List[str]
    ) -> np.ndarray:
        """
        Calculate word similarity matrix based on co-occurrence in texts

        Uses a simple heuristic: words that appear together in texts are similar.
        This is cheaper than embedding-based similarity and works well for MMR.

        Args:
            keywords: List of candidate keywords
            cluster_texts: Texts to analyze for co-occurrence

        Returns:
            Similarity matrix (n_keywords x n_keywords) with values in [0, 1]
        """
        n_keywords = len(keywords)

        # Build binary keyword occurrence matrix
        occurrence = np.zeros((len(cluster_texts), n_keywords), dtype=int)

        for text_idx, text in enumerate(cluster_texts):
            text_lower = text.lower()
            for kw_idx, keyword in enumerate(keywords):
                if keyword.lower() in text_lower:
                    occurrence[text_idx, kw_idx] = 1

        # Calculate cosine similarity between keyword occurrence vectors
        # If a keyword doesn't appear in any text, similarity defaults to 0
        similarity = np.zeros((n_keywords, n_keywords))

        for i in range(n_keywords):
            for j in range(i, n_keywords):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    # Cosine similarity of occurrence vectors
                    vec_i = occurrence[:, i]
                    vec_j = occurrence[:, j]

                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)

                    if norm_i > 0 and norm_j > 0:
                        sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    else:
                        sim = 0.0

                    similarity[i, j] = sim
                    similarity[j, i] = sim  # Symmetric

        return similarity

    def get_diversity_stats(
        self,
        keywords: List[Tuple[str, float]],
        cluster_texts: List[str]
    ) -> dict:
        """
        Calculate diversity statistics for selected keywords

        Args:
            keywords: List of (keyword, score) tuples
            cluster_texts: Texts used for analysis

        Returns:
            Dict with diversity metrics
        """
        if not keywords:
            return {"avg_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}

        keyword_list = [kw for kw, _ in keywords]
        similarity_matrix = self._calculate_word_similarity(keyword_list, cluster_texts)

        # Get upper triangle (excluding diagonal) for pairwise similarities
        n = len(keyword_list)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sims.append(similarity_matrix[i, j])

        if not pairwise_sims:
            return {"avg_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}

        return {
            "avg_similarity": float(np.mean(pairwise_sims)),
            "min_similarity": float(np.min(pairwise_sims)),
            "max_similarity": float(np.max(pairwise_sims)),
            "n_keywords": len(keywords)
        }
