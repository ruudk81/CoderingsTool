"""
Class-based TF-IDF implementation adapted from BERTopic
This is a standalone implementation without BERTopic dependencies
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array


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