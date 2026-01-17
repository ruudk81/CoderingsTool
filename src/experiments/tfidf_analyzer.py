"""
TF-IDF Analyzer for Cluster Keyword Extraction

This module provides TF-IDF-based keyword extraction for clusters,
designed to enhance cluster descriptions in the CoderingsTool pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


@dataclass
class TfidfConfig:
    """Configuration for TF-IDF keyword extraction"""
    max_features: int = 1000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.8
    top_k_keywords: int = 10
    language: str = "nl"

    # Custom stopwords (can be extended)
    custom_stopwords: List[str] = field(default_factory=list)

    # Preprocessing options
    lowercase: bool = True
    strip_accents: str = "unicode"
    token_pattern: str = r"(?u)\b\w\w+\b"  # Words with 2+ characters


class TfidfAnalyzer:
    """Extract cluster keywords using TF-IDF analysis"""

    def __init__(self, config: TfidfConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.vectorizer = None
        self.stopwords = self._get_stopwords()

    def _get_stopwords(self) -> List[str]:
        """Get stopwords for the configured language"""
        # Base stopwords for Dutch and English
        dutch_stopwords = [
            "de", "het", "een", "en", "van", "in", "op", "is", "te", "voor",
            "dat", "met", "die", "aan", "uit", "ook", "door", "er", "zijn",
            "bij", "om", "naar", "als", "over", "tot", "maar", "want", "worden",
            "of", "heeft", "had", "kan", "meer", "moet", "deze", "niet", "al",
            "zijn", "was", "zo", "zoals", "nog", "na", "wordt", "werd", "worden",
            "mijn", "je", "jij", "we", "hij", "zij", "ze", "hun", "haar", "hem",
            "mij", "ons", "jullie", "hun", "der", "den", "dit", "wat", "wie",
            "waar", "wanneer", "hoe", "waarom", "welke", "hier", "daar", "nu",
            "toen", "dan", "heel", "zeer", "veel", "weinig", "alle", "geen",
            "iets", "niets", "alles", "iedereen", "niemand", "andere", "andere"
        ]

        english_stopwords = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they", "my",
            "your", "his", "her", "its", "our", "their", "what", "which", "who",
            "when", "where", "why", "how", "all", "some", "any", "no", "not",
            "very", "much", "more", "most", "other", "another", "such", "so"
        ]

        # Select based on language
        if self.config.language.lower() in ["nl", "dutch", "nederlands"]:
            base_stopwords = dutch_stopwords
        elif self.config.language.lower() in ["en", "english", "engels"]:
            base_stopwords = english_stopwords
        else:
            # Use both for multilingual support
            base_stopwords = dutch_stopwords + english_stopwords

        # Add custom stopwords
        return list(set(base_stopwords + self.config.custom_stopwords))

    def extract_keywords(
        self,
        clusters: Dict[int, List[str]]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top keywords for each cluster using TF-IDF

        Args:
            clusters: Dict mapping cluster_id to list of idea texts

        Returns:
            Dict mapping cluster_id to list of (keyword, score) tuples
        """
        if not clusters:
            if self.verbose:
                print("Warning: No clusters provided for TF-IDF analysis")
            return {}

        # Prepare documents (one document per cluster)
        cluster_ids = []
        documents = []

        for cluster_id, ideas in clusters.items():
            if not ideas:
                continue
            cluster_ids.append(cluster_id)
            # Join all ideas in cluster into single document
            documents.append(" ".join(ideas))

        if not documents:
            if self.verbose:
                print("Warning: No valid cluster documents for TF-IDF analysis")
            return {}

        if self.verbose:
            print(f"\n[TF-IDF] Processing {len(documents)} clusters")
            print(f"[TF-IDF] Config: ngrams={self.config.ngram_range}, "
                  f"min_df={self.config.min_df}, max_df={self.config.max_df}")

        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            stop_words=self.stopwords,
            lowercase=self.config.lowercase,
            strip_accents=self.config.strip_accents,
            token_pattern=self.config.token_pattern
        )

        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            feature_names = self.vectorizer.get_feature_names_out()

            if self.verbose:
                print(f"[TF-IDF] Vocabulary size: {len(feature_names)}")
                print(f"[TF-IDF] Matrix shape: {tfidf_matrix.shape}")

        except ValueError as e:
            if self.verbose:
                print(f"Error: TF-IDF vectorization failed: {e}")
            return {}

        # Extract top keywords per cluster
        cluster_keywords = {}

        for idx, cluster_id in enumerate(cluster_ids):
            # Get TF-IDF scores for this cluster
            tfidf_scores = tfidf_matrix[idx].toarray()[0]

            # Get top K indices
            top_indices = np.argsort(tfidf_scores)[-self.config.top_k_keywords:][::-1]

            # Extract keywords with scores
            keywords = [
                (feature_names[i], tfidf_scores[i])
                for i in top_indices
                if tfidf_scores[i] > 0  # Only include non-zero scores
            ]

            cluster_keywords[cluster_id] = keywords

        if self.verbose:
            print(f"[TF-IDF] Extracted keywords for {len(cluster_keywords)} clusters")

        return cluster_keywords

    def get_cluster_summary(
        self,
        cluster_id: int,
        keywords: List[Tuple[str, float]],
        max_keywords: Optional[int] = None
    ) -> str:
        """
        Generate a text summary of cluster keywords

        Args:
            cluster_id: Cluster identifier
            keywords: List of (keyword, score) tuples
            max_keywords: Maximum keywords to include (None = all)

        Returns:
            Formatted string summary
        """
        if max_keywords:
            keywords = keywords[:max_keywords]

        summary_parts = [f"Cluster {cluster_id} keywords:"]
        for i, (keyword, score) in enumerate(keywords, 1):
            summary_parts.append(f"  {i}. {keyword} ({score:.3f})")

        return "\n".join(summary_parts)

    def get_keyword_context(
        self,
        cluster_id: int,
        keywords: List[Tuple[str, float]],
        ideas: List[str],
        max_examples: int = 3
    ) -> Dict[str, List[str]]:
        """
        Find example ideas containing each keyword

        Args:
            cluster_id: Cluster identifier
            keywords: List of (keyword, score) tuples
            ideas: List of idea texts from the cluster
            max_examples: Maximum examples per keyword

        Returns:
            Dict mapping keyword to list of example ideas
        """
        keyword_examples = {}

        for keyword, _ in keywords:
            examples = []
            keyword_lower = keyword.lower()

            for idea in ideas:
                if keyword_lower in idea.lower():
                    examples.append(idea)
                    if len(examples) >= max_examples:
                        break

            if examples:
                keyword_examples[keyword] = examples

        return keyword_examples

    def compare_clusters(
        self,
        cluster_keywords: Dict[int, List[Tuple[str, float]]],
        cluster1_id: int,
        cluster2_id: int,
        top_n: int = 5
    ) -> Dict[str, List[str]]:
        """
        Compare keywords between two clusters

        Args:
            cluster_keywords: Full keyword mapping
            cluster1_id: First cluster ID
            cluster2_id: Second cluster ID
            top_n: Number of top keywords to compare

        Returns:
            Dict with 'unique_to_1', 'unique_to_2', 'shared' keyword lists
        """
        if cluster1_id not in cluster_keywords or cluster2_id not in cluster_keywords:
            return {'unique_to_1': [], 'unique_to_2': [], 'shared': []}

        keywords1 = set(kw for kw, _ in cluster_keywords[cluster1_id][:top_n])
        keywords2 = set(kw for kw, _ in cluster_keywords[cluster2_id][:top_n])

        return {
            'unique_to_1': list(keywords1 - keywords2),
            'unique_to_2': list(keywords2 - keywords1),
            'shared': list(keywords1 & keywords2)
        }
