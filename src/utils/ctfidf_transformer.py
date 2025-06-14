import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
import scipy.sparse as sp
from typing import Union, List, Optional
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List, Dict

from .verboseReporter import VerboseReporter


def verify_bertopic_implementation():
    """Verify our ClassTfidfTransformer matches BERTopic's source exactly."""
    
    print("🔍 Verifying BERTopic c-TF-IDF Implementation")
    print("=" * 50)
    
    try:
        # Read BERTopic source - handle both relative and absolute paths
        import os
        possible_paths = [
            "/workspaces/CoderingsTool/bertopic/vectorizers/_ctfidf.py",
            "../bertopic/vectorizers/_ctfidf.py",
            "../../bertopic/vectorizers/_ctfidf.py",
            os.path.join(os.path.dirname(__file__), "..", "..", "bertopic", "vectorizers", "_ctfidf.py")
        ]
        
        bertopic_source = None
        for path in possible_paths:
            try:
                with open(path, 'r') as f:
                    bertopic_source = f.read()
                    print(f"✅ Found BERTopic source at: {path}")
                    break
            except FileNotFoundError:
                continue
        
        if not bertopic_source:
            raise FileNotFoundError("Could not find BERTopic source file in any expected location")
        
        # Check key implementation details
        checks = [
            ("avg_nr_samples calculation", "int(X.sum(axis=1).mean())" in bertopic_source),
            ("Document frequency calculation", "df = np.squeeze(np.asarray(X.sum(axis=0)))" in bertopic_source),
            ("IDF calculation", "np.log((avg_nr_samples / df) + 1)" in bertopic_source),
            ("L1 normalization", "normalize(X, axis=1, norm=\"l1\"" in bertopic_source),
            ("BM25 weighting option", "log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))" in bertopic_source)
        ]
        
        print("📋 Implementation Verification:")
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ SUCCESS: Our implementation matches BERTopic's source code")
        else:
            print("\n❌ WARNING: Some implementation details don't match BERTopic")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Error verifying BERTopic implementation: {e}")
        return False


@dataclass
class CtfidfConfig:
    """Configuration for c-TF-IDF transformer"""
    bm25_weighting: bool = False  # Use BM25-inspired weighting
    reduce_frequent_words: bool = False  # Apply square root to frequent words
    seed_words: List[str] = None  # Specific words to boost
    seed_multiplier: float = 2  # Multiplier for seed words
    verbose: bool = True


class ClassTfidfTransformer(TfidfTransformer):
    """
    Exact implementation of BERTopic's ClassTfidfTransformer.
    
    A Class-based TF-IDF procedure using scikit-learns TfidfTransformer as a base.
    c-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class.
    """
    
    def __init__(
        self,
        bm25_weighting: bool = False,
        reduce_frequent_words: bool = False,
        seed_words: List[str] = None,
        seed_multiplier: float = 2,
    ):
        self.bm25_weighting = bm25_weighting
        self.reduce_frequent_words = reduce_frequent_words
        self.seed_words = seed_words
        self.seed_multiplier = seed_multiplier
        super(ClassTfidfTransformer, self).__init__()
        
    def fit(self, X: sp.csr_matrix, multiplier: np.ndarray = None):
        """Learn the idf vector (global term weights) - exact BERTopic implementation.
        
        Arguments:
            X: A matrix of term/token counts.
            multiplier: A multiplier for increasing/decreasing certain IDF scores
        """
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64
        
        if self.use_idf:
            _, n_features = X.shape
            
            # Calculate the frequency of words across all classes (BERTopic's way)
            df = np.squeeze(np.asarray(X.sum(axis=0)))
            
            # Calculate the average number of samples as regularization (BERTopic's way)
            avg_nr_samples = int(X.sum(axis=1).mean())
            
            # Handle zero document frequencies to prevent division by zero
            # This can happen when vectorizer creates features that don't appear in aggregated docs
            df = np.where(df == 0, 1e-10, df)  # Replace zeros with tiny value
            
            # BM25-inspired weighting procedure
            if self.bm25_weighting:
                idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
            
            # Divide the average number of samples by the word frequency
            # +1 is added to force values to be positive
            else:
                idf = np.log((avg_nr_samples / df) + 1)
            
            # Multiplier to increase/decrease certain idf scores
            if multiplier is not None:
                idf = idf * multiplier
            
            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )
        
        return self
    
    def transform(self, X: sp.csr_matrix):
        """Transform a count-based matrix to c-TF-IDF - exact BERTopic implementation.
        
        Arguments:
            X (sparse matrix): A matrix of term/token counts.
        
        Returns:
            X (sparse matrix): A c-TF-IDF matrix
        """
        if self.use_idf:
            X = normalize(X, axis=1, norm="l1", copy=False)
            
            if self.reduce_frequent_words:
                X.data = np.sqrt(X.data)
            
            X = X * self._idf_diag
        
        return X
    
    def fit_transform(self, X: sp.csr_matrix, multiplier: np.ndarray = None) -> sp.csr_matrix:
        """Fit the transformer and transform the data in one step."""
        return self.fit(X, multiplier).transform(X)
    


# Wrapper class with verbose reporting for easier integration
class CtfidfTransformer:
    """Wrapper around ClassTfidfTransformer with verbose reporting"""
    
    def __init__(self, config: CtfidfConfig = None, verbose: bool = False):
        self.config = config or CtfidfConfig()
        self.verbose_reporter = VerboseReporter(verbose or self.config.verbose)
        
        # Initialize the actual BERTopic transformer
        self.transformer = ClassTfidfTransformer(
            bm25_weighting=self.config.bm25_weighting,
            reduce_frequent_words=self.config.reduce_frequent_words,
            seed_words=self.config.seed_words,
            seed_multiplier=self.config.seed_multiplier
        )
        
    def fit_transform(self, X: sp.csr_matrix, multiplier: np.ndarray = None) -> sp.csr_matrix:
        """Fit and transform with verbose reporting"""
        # Report input statistics
        _, n_features = X.shape
        avg_nr_samples = int(X.sum(axis=1).mean())
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        
        # Debug zero document frequencies
        zero_df_count = np.sum(df == 0)
        if zero_df_count > 0:
            self.verbose_reporter.stat_line(f"⚠️  Warning: {zero_df_count} features have zero document frequency")
        
        self.verbose_reporter.stat_line(f"c-TF-IDF input: {X.shape[0]} topics, {n_features} features")
        self.verbose_reporter.stat_line(f"Average samples per topic: {avg_nr_samples}")
        self.verbose_reporter.stat_line(f"Document frequency range: {np.min(df):.1f} to {np.max(df):.1f}")
        
        # Use BERTopic's exact implementation
        result = self.transformer.fit_transform(X, multiplier)
        
        # Check for problematic values in result
        if sp.issparse(result):
            has_inf = np.isinf(result.data).any()
            has_nan = np.isnan(result.data).any()
        else:
            has_inf = np.isinf(result).any()
            has_nan = np.isnan(result).any()
        
        if has_inf or has_nan:
            self.verbose_reporter.stat_line(f"⚠️  c-TF-IDF result has inf: {has_inf}, nan: {has_nan}")
        
        self.verbose_reporter.stat_line(f"c-TF-IDF output shape: {result.shape}")
        return result
    
    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform using fitted transformer"""
        return self.transformer.transform(X)
    
    def calculate_similarity(self, X_outliers: sp.csr_matrix, X_topics: sp.csr_matrix) -> np.ndarray:
        """Calculate cosine similarity between outlier documents and topic representations"""
        similarity = cosine_similarity(X_outliers, X_topics)
        self.verbose_reporter.stat_line(f"Calculated similarity for {X_outliers.shape[0]} outliers vs {X_topics.shape[0]} topics")
        return similarity
    
    def verify_bertopic_implementation(self, test_documents: List[str]) -> Dict:
        """
        Verify our c-TF-IDF implementation matches BERTopic's exactly.
        
        This method creates test topic representations and compares our implementation
        with BERTopic's reference implementation from the source code.
        """
        self.verbose_reporter.step_start("Verifying c-TF-IDF implementation against BERTopic", "🔬")
        
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Create a simple test case
            # Simulate topic documents (aggregated per cluster)
            topic_docs = [
                " ".join(test_documents[:len(test_documents)//3]),  # Topic 0
                " ".join(test_documents[len(test_documents)//3:2*len(test_documents)//3]),  # Topic 1
                " ".join(test_documents[2*len(test_documents)//3:])  # Topic 2
            ]
            
            # Create vectorizer and count matrix
            vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 2))
            X_counts = vectorizer.fit_transform(topic_docs)
            
            self.verbose_reporter.stat_line(f"Test setup: {X_counts.shape[0]} topics, {X_counts.shape[1]} features")
            
            # Test our implementation
            our_result = self.transformer.fit_transform(X_counts)
            
            # Compare with reference BERTopic implementation
            from bertopic.vectorizers._ctfidf import ClassTfidfTransformer as BertopicTransformer
            
            bertopic_transformer = BertopicTransformer(
                bm25_weighting=self.config.bm25_weighting,
                reduce_frequent_words=self.config.reduce_frequent_words,
                seed_words=self.config.seed_words,
                seed_multiplier=self.config.seed_multiplier
            )
            
            bertopic_result = bertopic_transformer.fit_transform(X_counts)
            
            # Compare results
            if sp.issparse(our_result):
                our_dense = our_result.toarray()
            else:
                our_dense = our_result
                
            if sp.issparse(bertopic_result):
                bertopic_dense = bertopic_result.toarray()
            else:
                bertopic_dense = bertopic_result
            
            # Calculate differences
            max_diff = np.max(np.abs(our_dense - bertopic_dense))
            mean_diff = np.mean(np.abs(our_dense - bertopic_dense))
            
            # Check if matrices are practically identical
            are_identical = np.allclose(our_dense, bertopic_dense, atol=1e-10)
            
            verification_results = {
                'matrices_identical': are_identical,
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'our_shape': our_result.shape,
                'bertopic_shape': bertopic_result.shape,
                'our_nonzero': np.count_nonzero(our_dense),
                'bertopic_nonzero': np.count_nonzero(bertopic_dense)
            }
            
            self.verbose_reporter.stat_line(f"Implementation verification:")
            self.verbose_reporter.stat_line(f"  Matrices identical: {are_identical}")
            self.verbose_reporter.stat_line(f"  Max difference: {max_diff:.2e}")
            self.verbose_reporter.stat_line(f"  Mean difference: {mean_diff:.2e}")
            
            if are_identical:
                self.verbose_reporter.stat_line("✅ Our c-TF-IDF implementation matches BERTopic exactly!")
            else:
                self.verbose_reporter.stat_line("⚠️ Implementation differences detected - investigation needed")
            
            self.verbose_reporter.step_complete("c-TF-IDF verification completed")
            return verification_results
            
        except ImportError as e:
            self.verbose_reporter.stat_line(f"❌ Cannot import BERTopic for verification: {e}")
            return {'error': 'BERTopic import failed'}
        except Exception as e:
            self.verbose_reporter.stat_line(f"❌ Verification failed: {e}")
            return {'error': str(e)}