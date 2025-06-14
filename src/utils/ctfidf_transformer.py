import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
import scipy.sparse as sp
from typing import Union, List, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

from .verboseReporter import VerboseReporter


@dataclass
class CtfidfConfig:
    """Configuration for c-TF-IDF transformer"""
    bm25_weighting: bool = False  # Use BM25-inspired weighting
    reduce_frequent_words: bool = False  # Apply square root to frequent words
    verbose: bool = True


class CtfidfTransformer:
    """
    Class-based TF-IDF transformer adapted from BERTopic implementation.
    
    This transformer calculates TF-IDF scores for topics by treating each topic
    as a single document (created by aggregating all documents in that topic).
    """
    
    def __init__(self, config: CtfidfConfig = None, verbose: bool = False):
        self.config = config or CtfidfConfig()
        self.verbose_reporter = VerboseReporter(verbose or self.config.verbose)
        
        # Will be set during fit
        self.idf_diag = None
        self.vocabulary_size = None
        self.fitted = False
        
    def fit(self, X: Union[sp.csr_matrix, np.ndarray], n_samples_per_topic: List[int] = None):
        """
        Fit the c-TF-IDF transformer.
        
        Parameters:
        -----------
        X : sparse matrix or array, shape (n_topics, n_features)
            Topic-term matrix where each row represents aggregated term counts for a topic
        n_samples_per_topic : list of int, optional
            Number of documents per topic for proper IDF calculation
        """
        if sp.issparse(X):
            X = X.toarray()
            
        self.vocabulary_size = X.shape[1]
        
        # Calculate document frequency (how many topics contain each term)
        df = np.sum(X > 0, axis=0)
        
        # Calculate average number of samples per topic for IDF
        if n_samples_per_topic is not None:
            avg_nr_samples = np.mean(n_samples_per_topic)
        else:
            # Fallback: use number of topics as proxy
            avg_nr_samples = X.shape[0]
        
        self.verbose_reporter.stat_line(f"Calculating IDF for {len(df)} terms across {X.shape[0]} topics")
        self.verbose_reporter.stat_line(f"Average samples per topic: {avg_nr_samples:.1f}")
        
        # Calculate IDF using BERTopic's formula
        if self.config.bm25_weighting:
            # BM25-inspired weighting: log(1 + (avg - df + 0.5) / (df + 0.5))
            idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
            self.verbose_reporter.stat_line("Using BM25-inspired IDF weighting")
        else:
            # Standard c-TF-IDF: log((avg + 1) / (df + 1))
            idf = np.log((avg_nr_samples / df) + 1)
            self.verbose_reporter.stat_line("Using standard c-TF-IDF weighting")
        
        # Create diagonal matrix for efficient multiplication
        self.idf_diag = sp.diags(idf, shape=(self.vocabulary_size, self.vocabulary_size))
        
        # Report IDF statistics
        self.verbose_reporter.stat_line(f"IDF range: {np.min(idf):.3f} to {np.max(idf):.3f}")
        self.verbose_reporter.stat_line(f"Mean IDF: {np.mean(idf):.3f}")
        
        self.fitted = True
        return self
    
    def transform(self, X: Union[sp.csr_matrix, np.ndarray]) -> sp.csr_matrix:
        """
        Transform topic-term matrix to c-TF-IDF representation.
        
        Parameters:
        -----------
        X : sparse matrix or array, shape (n_topics, n_features)
            Topic-term matrix
            
        Returns:
        --------
        c_tf_idf : sparse matrix, shape (n_topics, n_features)
            c-TF-IDF transformed matrix
        """
        if not self.fitted:
            raise ValueError("CtfidfTransformer must be fitted before transform")
        
        # Convert to sparse matrix if needed
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        
        self.verbose_reporter.stat_line(f"Transforming {X.shape[0]} topics with {X.shape[1]} features")
        
        # Step 1: L1 normalization (TF calculation)
        # This ensures term frequencies sum to 1 per topic
        X_normalized = normalize(X, axis=1, norm='l1', copy=True)
        
        # Step 2: Optional frequency reduction (square root)
        if self.config.reduce_frequent_words:
            X_normalized.data = np.sqrt(X_normalized.data)
            self.verbose_reporter.stat_line("Applied square root frequency reduction")
        
        # Step 3: Apply IDF weighting
        c_tf_idf = X_normalized * self.idf_diag
        
        self.verbose_reporter.stat_line("c-TF-IDF transformation completed")
        
        return c_tf_idf
    
    def fit_transform(self, X: Union[sp.csr_matrix, np.ndarray], 
                     n_samples_per_topic: List[int] = None) -> sp.csr_matrix:
        """Fit the transformer and transform the data in one step."""
        return self.fit(X, n_samples_per_topic).transform(X)
    
    def get_feature_importance(self, topic_idx: int, X_transformed: sp.csr_matrix, 
                              feature_names: List[str], top_k: int = 10) -> List[tuple]:
        """
        Get top features for a specific topic based on c-TF-IDF scores.
        
        Parameters:
        -----------
        topic_idx : int
            Index of the topic
        X_transformed : sparse matrix
            c-TF-IDF transformed matrix
        feature_names : list of str
            Feature names (vocabulary)
        top_k : int
            Number of top features to return
            
        Returns:
        --------
        top_features : list of tuples (feature_name, score)
            Top features with their c-TF-IDF scores
        """
        if topic_idx >= X_transformed.shape[0]:
            raise ValueError(f"Topic index {topic_idx} out of range")
        
        # Get c-TF-IDF scores for the topic
        scores = X_transformed[topic_idx].toarray().flatten()
        
        # Get top indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Create list of (feature, score) tuples
        top_features = [(feature_names[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        return top_features
    
    def calculate_similarity(self, X_outliers: Union[sp.csr_matrix, np.ndarray],
                           X_topics: sp.csr_matrix) -> np.ndarray:
        """
        Calculate cosine similarity between outlier documents and topic representations.
        
        Parameters:
        -----------
        X_outliers : sparse matrix or array
            c-TF-IDF representations of outlier documents
        X_topics : sparse matrix
            c-TF-IDF representations of topics
            
        Returns:
        --------
        similarity : array, shape (n_outliers, n_topics)
            Cosine similarity matrix
        """
        similarity = cosine_similarity(X_outliers, X_topics)
        self.verbose_reporter.stat_line(f"Calculated similarity for {X_outliers.shape[0]} outliers vs {X_topics.shape[0]} topics")
        
        return similarity