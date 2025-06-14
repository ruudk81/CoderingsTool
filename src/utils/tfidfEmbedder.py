import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
import spacy
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path

from config import TfidfConfig, DEFAULT_LANGUAGE
from .verboseReporter import VerboseReporter

class TfidfEmbedder:
    """Handles TF-IDF embedding generation for ensemble approach"""
    
    def __init__(
        self, 
        config: TfidfConfig = None,
        verbose: bool = False,
        cache_dir: Path = None
    ):
        self.config = config or TfidfConfig()
        self.verbose_reporter = VerboseReporter(verbose)
        self.cache_dir = cache_dir or Path("cache/tfidf_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize spaCy for POS filtering if needed
        self.nlp = None
        if self.config.allowed_pos_tags:
            try:
                model_name = "nl_core_news_lg" if DEFAULT_LANGUAGE == "Dutch" else "en_core_web_lg"
                self.nlp = spacy.load(model_name)
                self.verbose_reporter.stat_line(f"Loaded spaCy model: {model_name}")
            except:
                self.verbose_reporter.stat_line("Warning: spaCy model not found, skipping POS filtering")
                self.config.allowed_pos_tags = None
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            use_idf=self.config.use_idf,
            sublinear_tf=self.config.sublinear_tf,
            norm=self.config.norm,
            tokenizer=self._custom_tokenizer if self.config.allowed_pos_tags else None
        )
        
        # For dimensionality reduction if needed
        self.svd = None
        self.is_fitted = False
    
    def _custom_tokenizer(self, text: str) -> List[str]:
        """Custom tokenizer that filters by POS tags"""
        if not self.nlp:
            return text.split()
        
        doc = self.nlp(text.lower())
        tokens = []
        
        for token in doc:
            # Keep tokens with allowed POS tags
            if (token.pos_ in self.config.allowed_pos_tags and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 1):
                tokens.append(token.lemma_)
        
        return tokens
    
    def fit(self, texts: List[str]) -> 'TfidfEmbedder':
        """Fit the TF-IDF vectorizer on texts"""
        self.verbose_reporter.step_start("Fitting TF-IDF model", emoji="📊")
        
        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        self.verbose_reporter.stat_line(f"Fitting on {len(valid_texts)} texts")
        
        # Fit vectorizer
        self.vectorizer.fit(valid_texts)
        self.is_fitted = True
        
        # Report vocabulary statistics
        vocab_size = len(self.vectorizer.vocabulary_)
        self.verbose_reporter.stat_line(f"Vocabulary size: {vocab_size} terms")
        
        # Show sample important terms
        if hasattr(self.vectorizer, 'idf_'):
            # Get all terms sorted by IDF score (most important first)
            terms = self.vectorizer.get_feature_names_out()
            idf_scores = self.vectorizer.idf_
            
            # Sort by IDF score (highest first = most domain-specific)
            sorted_indices = np.argsort(idf_scores)[::-1]
            sorted_terms = [terms[i] for i in sorted_indices]
            sorted_scores = [idf_scores[i] for i in sorted_indices]
            
            # Show top terms with their IDF scores
            self.verbose_reporter.sample_list("Top domain-specific terms (high IDF)", 
                                            [f"{term} ({score:.2f})" for term, score in zip(sorted_terms[:20], sorted_scores[:20])])
            
            # Optionally show all terms if verbose mode and reasonable size
            if vocab_size <= 50:
                self.verbose_reporter.sample_list("All TF-IDF vocabulary", sorted_terms)
            elif self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Full vocabulary available - showing top 20 of {vocab_size} terms")
        
        self.verbose_reporter.step_complete("TF-IDF model fitted")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF embeddings"""
        if not self.is_fitted:
            raise ValueError("TfidfEmbedder must be fitted before transform")
        
        # Transform to sparse matrix
        tfidf_sparse = self.vectorizer.transform(texts)
        
        # Convert to dense array
        tfidf_dense = tfidf_sparse.toarray().astype(np.float32)
        
        return tfidf_dense
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce TF-IDF dimensions using SVD"""
        if self.svd is None or self.svd.n_components != n_components:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(embeddings)
        
        return self.svd.transform(embeddings)
    
    def save(self, filename: str = "tfidf_model.pkl"):
        """Save fitted model to disk"""
        filepath = self.cache_dir / filename
        model_data = {
            'vectorizer': self.vectorizer,
            'svd': self.svd,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.verbose_reporter.stat_line(f"Saved TF-IDF model to {filepath}")
    
    def load(self, filename: str = "tfidf_model.pkl") -> bool:
        """Load fitted model from disk"""
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.svd = model_data.get('svd')
            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']
            
            self.verbose_reporter.stat_line(f"Loaded TF-IDF model from {filepath}")
            return True
        except Exception as e:
            self.verbose_reporter.stat_line(f"Failed to load TF-IDF model: {e}")
            return False
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from fitted vectorizer"""
        if not self.is_fitted:
            raise ValueError("TfidfEmbedder must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def print_full_vocabulary(self):
        """Print all vocabulary terms sorted by importance (IDF score)"""
        if not self.is_fitted:
            raise ValueError("TfidfEmbedder must be fitted first")
        
        terms = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        
        # Sort by IDF score (highest first)
        sorted_indices = np.argsort(idf_scores)[::-1]
        sorted_terms = [terms[i] for i in sorted_indices]
        sorted_scores = [idf_scores[i] for i in sorted_indices]
        
        self.verbose_reporter.stat_line(f"Complete TF-IDF vocabulary ({len(terms)} terms):")
        for i, (term, score) in enumerate(zip(sorted_terms, sorted_scores), 1):
            print(f"{i:3d}. {term:<30} (IDF: {score:.3f})")
        
        return list(zip(sorted_terms, sorted_scores))