"""
Basic TF-IDF per-cluster keyword extraction

Unlike c-TF-IDF which compares terms across clusters, this computes TF-IDF
independently for each cluster's texts. This shows what terms are important
within each cluster without cross-cluster comparison.

Usage:
    from experiments.representation.tfidf_representation import TfidfRepresentation

    tfidf = TfidfRepresentation(top_k=10)
    keywords = tfidf.extract_keywords(clusters)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseRepresentation


class TfidfRepresentation(BaseRepresentation):
    """
    Basic TF-IDF keyword extraction per cluster

    Computes TF-IDF independently for each cluster's texts, treating each
    text as a document. Keywords are selected by averaging TF-IDF scores
    across all texts in the cluster.

    Args:
        top_k: Number of top keywords to extract per cluster
        ngram_range: N-gram range (e.g., (1, 2) for unigrams + bigrams)
        min_df: Minimum document frequency within cluster
        max_df: Maximum document frequency proportion
    """

    def __init__(
        self,
        top_k: int = 15,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95
    ):
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

    def extract_keywords(
        self,
        clusters: Dict[int, List[str]],
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top keywords for each cluster using per-cluster TF-IDF

        Args:
            clusters: Dict mapping cluster_id to list of idea texts
            verbose: Print progress information

        Returns:
            Dict mapping cluster_id to list of (keyword, score) tuples
        """
        if not clusters:
            if verbose:
                print("[TF-IDF] Warning: No clusters provided")
            return {}

        if verbose:
            print(f"\n[TF-IDF] Processing {len(clusters)} clusters (per-cluster)")

        cluster_keywords = {}

        for cluster_id, texts in sorted(clusters.items()):
            if len(texts) < 2:
                # Need at least 2 texts for TF-IDF to be meaningful
                if verbose:
                    print(f"[TF-IDF] Cluster {cluster_id}: skipped (only {len(texts)} text)")
                cluster_keywords[cluster_id] = []
                continue

            keywords = self._extract_cluster_keywords(cluster_id, texts, verbose)
            cluster_keywords[cluster_id] = keywords

        if verbose:
            print(f"[TF-IDF] Extracted keywords for {len(cluster_keywords)} clusters\n")

        return cluster_keywords

    def _extract_cluster_keywords(
        self,
        cluster_id: int,
        texts: List[str],
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """Extract keywords from a single cluster's texts using TF-IDF."""
        try:
            # Adjust min_df based on cluster size
            effective_min_df = min(self.min_df, max(1, len(texts) // 3))

            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                min_df=effective_min_df,
                max_df=self.max_df,
                lowercase=True,
                token_pattern=r"(?u)\b\w\w+\b"
            )

            tfidf_matrix = vectorizer.fit_transform(texts)
            vocabulary = vectorizer.get_feature_names_out()

            if len(vocabulary) == 0:
                if verbose:
                    print(f"[TF-IDF] Cluster {cluster_id}: no vocabulary extracted")
                return []

            # Average TF-IDF scores across all texts in cluster
            avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()

            # Get top-k by average score
            top_indices = np.argsort(avg_scores)[-self.top_k:][::-1]

            keywords = [
                (vocabulary[i], float(avg_scores[i]))
                for i in top_indices
                if avg_scores[i] > 0
            ]

            return keywords

        except ValueError as e:
            if verbose:
                print(f"[TF-IDF] Cluster {cluster_id}: error - {e}")
            return []

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
        Extract keywords using BaseRepresentation interface

        Note: This implementation ignores ctfidf_scores and vocabulary,
        computing TF-IDF directly on cluster_texts instead.

        Args:
            cluster_id: Cluster identifier
            ctfidf_scores: Not used (ignored)
            vocabulary: Not used (ignored)
            cluster_texts: Texts in this cluster
            embeddings: Not used

        Returns:
            List of (keyword, score) tuples
        """
        return self._extract_cluster_keywords(cluster_id, cluster_texts, verbose=False)
