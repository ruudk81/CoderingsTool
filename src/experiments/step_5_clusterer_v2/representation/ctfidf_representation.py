"""
Class-based TF-IDF implementation adapted from BERTopic
This is a standalone implementation without BERTopic dependencies

Usage:
    from experiments.representation.ctfidf_representation import CTfidfRepresentation

    # Initialize
    ctfidf = CTfidfRepresentation(top_k=15, bm25_weighting=True)

    # Extract keywords from clusters
    clusters = {1: ['idea1', 'idea2'], 2: ['idea3', 'idea4']}
    keywords = ctfidf.extract_keywords(clusters)
"""
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array

from .base import BaseRepresentation


class ClassTfidfTransformer(TfidfTransformer):
    """
    A Class-based TF-IDF procedure using scikit-learn's TfidfTransformer as a base.
    
    c-TF-IDF is a TF-IDF formula adapted for multiple classes by joining all documents per class.
    Each class is converted to a single document instead of a set of documents.
    
    The formula:
    1. Term Frequency: Frequency of each word x for each class c, L1 normalized
    2. Inverse Document Frequency: log(1 + (avg_words_per_class / freq_of_word_across_classes))
    3. With BM25 weighting: log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
    """
    
    def __init__(self):
        """Initialize with BERTopic's recommended settings"""
        self.bm25_weighting = True  # Use BM25 for better short text performance
        self.reduce_frequent_words = True  # Square root to reduce impact of very frequent words
        super(ClassTfidfTransformer, self).__init__()
    
    def fit(self, X: sp.csr_matrix):
        """
        Learn the idf vector (global term weights).
        
        Arguments:
            X: A matrix of term/token counts where each row represents a class
        """
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64
        
        if self.use_idf:
            _, n_features = X.shape
            
            # Calculate the frequency of words across all classes
            df = np.squeeze(np.asarray(X.sum(axis=0)))
            
            # Calculate the average number of samples as regularization
            avg_nr_samples = int(X.sum(axis=1).mean())
            
            # BM25-inspired weighting procedure
            if self.bm25_weighting:
                idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
            else:
                # Standard c-TF-IDF
                idf = np.log((avg_nr_samples / df) + 1)
            
            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )
        
        return self
    
    def transform(self, X: sp.csr_matrix):
        """
        Transform a count-based matrix to c-TF-IDF.
        
        Arguments:
            X: A matrix of term/token counts
            
        Returns:
            X: A c-TF-IDF matrix
        """
        if self.use_idf:
            # L1 normalize (sum of row = 1)
            X = normalize(X, axis=1, norm="l1", copy=False)
            
            # Reduce impact of very frequent words
            if self.reduce_frequent_words:
                X.data = np.sqrt(X.data)
            
            # Apply IDF weighting
            X = X * self._idf_diag

        return X


class CTfidfRepresentation(BaseRepresentation):
    """
    c-TF-IDF keyword extraction for clusters

    Wrapper around ClassTfidfTransformer that provides a complete keyword
    extraction pipeline compatible with CoderingsTool's cluster structure.

    Args:
        top_k: Number of top keywords to extract per cluster
        bm25_weighting: Use BM25-inspired weighting (recommended for short texts)
        reduce_frequent_words: Apply square root to reduce impact of very frequent words
        ngram_range: N-gram range for keyword extraction (e.g., (1, 2) for unigrams + bigrams)
        min_df: Minimum document frequency (clusters) for a keyword
        max_df: Maximum document frequency (proportion) for a keyword
        language: Language for stopwords ("nl", "en", or None)
    """

    def __init__(
        self,
        top_k: int = 15,
        bm25_weighting: bool = True,
        reduce_frequent_words: bool = True,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        language: str = "nl"
    ):
        self.top_k = top_k
        self.bm25_weighting = bm25_weighting
        self.reduce_frequent_words = reduce_frequent_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.language = language

        # Initialize transformer
        self.transformer = ClassTfidfTransformer()
        self.transformer.bm25_weighting = bm25_weighting
        self.transformer.reduce_frequent_words = reduce_frequent_words

        # Vectorizer for building count matrix
        self.vectorizer = None
        self.vocabulary = None
        self.ctfidf_matrix = None

    def extract_keywords(
        self,
        clusters: Dict[int, List[str]],
        verbose: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top keywords for each cluster using c-TF-IDF

        Args:
            clusters: Dict mapping cluster_id to list of idea texts
            verbose: Print progress information

        Returns:
            Dict mapping cluster_id to list of (keyword, score) tuples
        """
        if not clusters:
            if verbose:
                print("[c-TF-IDF] Warning: No clusters provided")
            return {}

        # Prepare documents (one document per cluster)
        cluster_ids = sorted(clusters.keys())
        cluster_docs = [" ".join(clusters[cid]) for cid in cluster_ids]

        if verbose:
            print(f"\n[c-TF-IDF] Processing {len(cluster_docs)} clusters")
            print(f"[c-TF-IDF] Config: ngrams={self.ngram_range}, min_df={self.min_df}, "
                  f"max_df={self.max_df}, bm25={self.bm25_weighting}")

        # Build count matrix
        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b"  # Words with 2+ characters
        )

        try:
            count_matrix = self.vectorizer.fit_transform(cluster_docs)
            self.vocabulary = self.vectorizer.get_feature_names_out()

            if verbose:
                print(f"[c-TF-IDF] Vocabulary size: {len(self.vocabulary)}")
                print(f"[c-TF-IDF] Matrix shape: {count_matrix.shape}")

        except ValueError as e:
            if verbose:
                print(f"[c-TF-IDF] Error: Vectorization failed: {e}")
            return {}

        # Apply c-TF-IDF transformation
        self.ctfidf_matrix = self.transformer.fit_transform(count_matrix)

        # Extract keywords per cluster
        cluster_keywords = {}
        for idx, cluster_id in enumerate(cluster_ids):
            ctfidf_scores = self.ctfidf_matrix[idx].toarray()[0]
            keywords = self.extract_topics(
                cluster_id=cluster_id,
                ctfidf_scores=ctfidf_scores,
                vocabulary=self.vocabulary,
                cluster_texts=clusters[cluster_id]
            )
            cluster_keywords[cluster_id] = keywords

        if verbose:
            print(f"[c-TF-IDF] Extracted keywords for {len(cluster_keywords)} clusters\n")

        return cluster_keywords

    def extract_topics(
        self,
        cluster_id: int,
        ctfidf_scores: np.ndarray,
        vocabulary: List[str],
        cluster_texts: List[str],
        embeddings: np.ndarray = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Extract top keywords for a single cluster

        Implementation of BaseRepresentation abstract method.

        Args:
            cluster_id: Cluster identifier
            ctfidf_scores: c-TF-IDF scores for this cluster (1D array)
            vocabulary: Feature names from vectorizer
            cluster_texts: Original idea texts (not used in basic c-TF-IDF)
            embeddings: Optional embeddings (not used in basic c-TF-IDF)

        Returns:
            List of (keyword, score) tuples, ordered by c-TF-IDF score
        """
        # Get top K indices by score
        top_indices = np.argsort(ctfidf_scores)[-self.top_k:][::-1]

        # Extract keywords with non-zero scores
        keywords = [
            (vocabulary[i], float(ctfidf_scores[i]))
            for i in top_indices
            if ctfidf_scores[i] > 0
        ]

        return keywords

    def get_cluster_summary(
        self,
        cluster_id: int,
        keywords: List[Tuple[str, float]],
        max_keywords: int = None
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

        summary_parts = [f"Cluster {cluster_id} c-TF-IDF keywords:"]
        for i, (keyword, score) in enumerate(keywords, 1):
            summary_parts.append(f"  {i}. {keyword:<30} ({score:.4f})")

        return "\n".join(summary_parts)