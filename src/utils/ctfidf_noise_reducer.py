import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from dataclasses import dataclass

from .ctfidf_transformer import CtfidfTransformer, CtfidfConfig
from .verboseReporter import VerboseReporter


@dataclass
class CtfidfNoiseRescueConfig:
    """Configuration for c-TF-IDF noise rescue"""
    enabled: bool = True
    similarity_threshold: float = 0.1  # Minimum cosine similarity for assignment
    min_topic_size: int = 2  # Minimum documents per topic for c-TF-IDF calculation
    max_rescue_attempts: int = 1000  # Maximum outliers to attempt rescue
    
    # c-TF-IDF specific settings
    ctfidf_config: CtfidfConfig = None
    
    # Text preprocessing
    min_doc_length: int = 5  # Minimum characters for document inclusion
    
    # Reporting
    verbose: bool = True
    show_examples: bool = True
    max_examples: int = 3


class CtfidfNoiseReducer:
    """
    BERTopic-style noise reduction using c-TF-IDF similarity.
    
    This class implements the strategy from BERTopic's reduce_outliers method
    with the "c-tf-idf" strategy, adapted for the CoderingsTool pipeline.
    """
    
    def __init__(self, 
                 vectorizer: CountVectorizer,
                 config: CtfidfNoiseRescueConfig = None,
                 verbose: bool = False):
        self.vectorizer = vectorizer
        self.config = config or CtfidfNoiseRescueConfig()
        self.verbose_reporter = VerboseReporter(verbose or self.config.verbose)
        
        # Initialize c-TF-IDF transformer
        ctfidf_config = self.config.ctfidf_config or CtfidfConfig()
        self.ctfidf_transformer = CtfidfTransformer(ctfidf_config, verbose)
        
        # Will be set during rescue process
        self.topic_representations = None
        self.topic_documents = None
        self.feature_names = None
        
    def rescue_noise_points(self, 
                          documents: List[str],
                          cluster_labels: List[int],
                          segment_ids: List[str] = None) -> Dict:
        """
        Rescue noise points using c-TF-IDF similarity to topic representations.
        
        Parameters:
        -----------
        documents : list of str
            All documents (including noise points)
        cluster_labels : list of int
            Cluster assignments (-1 for noise)
        segment_ids : list of str, optional
            Segment IDs for tracking
            
        Returns:
        --------
        results : dict
            Rescue statistics and updated labels
        """
        self.verbose_reporter.step_start("c-TF-IDF noise rescue", "🔍")
        
        # Prepare data
        df = pd.DataFrame({
            'document': documents,
            'cluster': cluster_labels,
            'segment_id': segment_ids if segment_ids else range(len(documents))
        })
        
        # Filter out very short documents
        df = df[df['document'].str.len() >= self.config.min_doc_length]
        
        # Identify topics and outliers
        topics_df = df[df['cluster'] != -1]
        outliers_df = df[df['cluster'] == -1]
        
        self.verbose_reporter.stat_line(f"Total documents: {len(df)}")
        self.verbose_reporter.stat_line(f"Valid topics: {len(topics_df)}")
        self.verbose_reporter.stat_line(f"Outliers to rescue: {len(outliers_df)}")
        
        # Debug: Show document length distribution
        doc_lengths = df['document'].str.len()
        self.verbose_reporter.stat_line(f"Document lengths - Min: {doc_lengths.min()}, Max: {doc_lengths.max()}, Mean: {doc_lengths.mean():.1f}")
        
        # Debug: Show sample documents
        if len(df) > 0:
            sample_docs = df['document'].head(3).tolist()
            self.verbose_reporter.sample_list("Sample documents", [f"'{doc[:80]}..." for doc in sample_docs])
        
        if len(outliers_df) == 0:
            self.verbose_reporter.stat_line("No outliers to rescue")
            self.verbose_reporter.step_complete("c-TF-IDF rescue completed")
            return self._create_results(0, 0, {})
        
        # Limit rescue attempts
        if len(outliers_df) > self.config.max_rescue_attempts:
            self.verbose_reporter.stat_line(f"Limiting rescue to {self.config.max_rescue_attempts} attempts")
            outliers_df = outliers_df.head(self.config.max_rescue_attempts)
        
        # Build topic representations
        topic_info = self._build_topic_representations(topics_df)
        if not topic_info:
            self.verbose_reporter.stat_line("No valid topics for c-TF-IDF calculation")
            self.verbose_reporter.step_complete("c-TF-IDF rescue completed")
            return self._create_results(0, len(outliers_df), {})
        
        # Rescue outliers
        rescue_results = self._rescue_outliers(outliers_df, topic_info)
        
        self.verbose_reporter.step_complete("c-TF-IDF rescue completed")
        return rescue_results
    
    def _build_topic_representations(self, topics_df: pd.DataFrame) -> Optional[Dict]:
        """Build c-TF-IDF representations for existing topics."""
        self.verbose_reporter.stat_line("Building topic representations...")
        
        # Aggregate documents per topic
        topic_docs = topics_df.groupby('cluster')['document'].apply(lambda x: ' '.join(x)).reset_index()
        topic_docs.columns = ['cluster', 'aggregated_document']
        
        # Filter topics with minimum size
        topic_sizes = topics_df['cluster'].value_counts()
        valid_topics = topic_sizes[topic_sizes >= self.config.min_topic_size].index
        topic_docs = topic_docs[topic_docs['cluster'].isin(valid_topics)]
        
        if len(topic_docs) == 0:
            self.verbose_reporter.stat_line("No topics meet minimum size requirement")
            return None
        
        self.verbose_reporter.stat_line(f"Creating representations for {len(topic_docs)} topics")
        
        # Debug: Show topic sizes and sample aggregated docs
        self.verbose_reporter.stat_line(f"Topic sizes: {dict(topic_sizes[topic_sizes >= self.config.min_topic_size])}")
        if len(topic_docs) > 0:
            sample_topic_doc = topic_docs['aggregated_document'].iloc[0]
            self.verbose_reporter.stat_line(f"Sample aggregated doc length: {len(sample_topic_doc)} chars")
            self.verbose_reporter.stat_line(f"Sample: '{sample_topic_doc[:100]}...'")
        
        # Vectorize aggregated documents
        try:
            # Check if vectorizer is fitted
            try:
                self.feature_names = self.vectorizer.get_feature_names_out()
                self.verbose_reporter.stat_line(f"Using fitted vectorizer with {len(self.feature_names)} features")
            except Exception as e:
                raise ValueError(f"Vectorizer not fitted. Please fit vectorizer before calling c-TF-IDF rescue. Error: {e}")
            
            aggregated_docs = topic_docs['aggregated_document'].tolist()
            self.verbose_reporter.stat_line(f"Vectorizing {len(aggregated_docs)} aggregated documents...")
            
            X_topics = self.vectorizer.transform(aggregated_docs)
            
            self.verbose_reporter.stat_line(f"Topic matrix shape: {X_topics.shape}")
            self.verbose_reporter.stat_line(f"Vocabulary size: {len(self.feature_names)}")
            
            # Debug: Check if matrix is empty
            non_zero_topics = np.sum(X_topics.sum(axis=1) > 0)
            self.verbose_reporter.stat_line(f"Topics with non-zero terms: {non_zero_topics}/{X_topics.shape[0]}")
            
            # Debug: Show sample vocabulary
            if len(self.feature_names) > 0:
                sample_vocab = list(self.feature_names[:10])
                self.verbose_reporter.sample_list("Sample vocabulary", sample_vocab)
            
            # Transform to c-TF-IDF using BERTopic's exact method
            self.topic_representations = self.ctfidf_transformer.fit_transform(X_topics)
            
            return {
                'cluster_ids': topic_docs['cluster'].tolist(),
                'topic_representations': self.topic_representations,
                'aggregated_docs': topic_docs['aggregated_document'].tolist()
            }
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"❌ Error building topic representations: {e}")
            self.verbose_reporter.stat_line(f"Error type: {type(e).__name__}")
            import traceback
            self.verbose_reporter.stat_line(f"Traceback: {traceback.format_exc()[-200:]}")
            return None
    
    def _rescue_outliers(self, outliers_df: pd.DataFrame, topic_info: Dict) -> Dict:
        """Rescue outlier documents using c-TF-IDF similarity."""
        self.verbose_reporter.stat_line("Calculating outlier similarities...")
        
        # Vectorize outlier documents
        outlier_docs = outliers_df['document'].tolist()
        self.verbose_reporter.stat_line(f"Vectorizing {len(outlier_docs)} outlier documents...")
        
        try:
            X_outliers = self.vectorizer.transform(outlier_docs)
            self.verbose_reporter.stat_line(f"Outlier matrix shape: {X_outliers.shape}")
        except Exception as e:
            self.verbose_reporter.stat_line(f"❌ Error vectorizing outliers: {e}")
            return self._create_results(0, len(outliers_df), {})
        
        # Transform to c-TF-IDF using the fitted transformer
        X_outliers_ctfidf = self.ctfidf_transformer.transform(X_outliers)
        
        # Calculate similarities
        similarities = self.ctfidf_transformer.calculate_similarity(
            X_outliers_ctfidf, topic_info['topic_representations']
        )
        
        # Debug: Check similarity matrix
        if len(similarities) > 0:
            max_sims = np.max(similarities, axis=1)
            self.verbose_reporter.stat_line(f"Similarity matrix shape: {similarities.shape}")
            self.verbose_reporter.stat_line(f"Max similarities - Min: {np.min(max_sims):.4f}, Max: {np.max(max_sims):.4f}, Mean: {np.mean(max_sims):.4f}")
            
            # Show distribution around threshold
            threshold = self.config.similarity_threshold
            above_threshold = np.sum(max_sims >= threshold)
            above_half_threshold = np.sum(max_sims >= threshold/2)
            self.verbose_reporter.stat_line(f"Above threshold ({threshold}): {above_threshold}/{len(max_sims)}")
            self.verbose_reporter.stat_line(f"Above half threshold ({threshold/2}): {above_half_threshold}/{len(max_sims)}")
        
        # Apply threshold and assign topics
        rescued_count = 0
        new_assignments = {}
        rescue_examples = []
        
        threshold = self.config.similarity_threshold
        cluster_ids = topic_info['cluster_ids']
        
        for i, (idx, row) in enumerate(outliers_df.iterrows()):
            sim_scores = similarities[i]
            best_topic_idx = np.argmax(sim_scores)
            best_similarity = sim_scores[best_topic_idx]
            
            if best_similarity >= threshold:
                new_cluster = cluster_ids[best_topic_idx]
                new_assignments[row['segment_id']] = new_cluster
                rescued_count += 1
                
                # Collect examples for reporting
                if self.config.show_examples and len(rescue_examples) < self.config.max_examples:
                    rescue_examples.append({
                        'document': row['document'][:60] + "..." if len(row['document']) > 60 else row['document'],
                        'cluster': new_cluster,
                        'similarity': best_similarity
                    })
        
        # Report results
        total_outliers = len(outliers_df)
        success_rate = rescued_count / total_outliers if total_outliers > 0 else 0.0
        
        self.verbose_reporter.stat_line(f"c-TF-IDF rescued {rescued_count}/{total_outliers} remaining outliers ({success_rate:.1%} success rate for Phase 2)")
        self.verbose_reporter.stat_line(f"Used similarity threshold: {threshold}")
        
        # Show confidence distribution
        if len(similarities) > 0:
            max_sims = np.max(similarities, axis=1)
            above_threshold = np.sum(max_sims >= threshold)
            
            self.verbose_reporter.stat_line(f"Similarity scores - Min: {np.min(max_sims):.3f}, Max: {np.max(max_sims):.3f}, Mean: {np.mean(max_sims):.3f}")
            self.verbose_reporter.stat_line(f"Points above threshold ({threshold}): {above_threshold}/{len(max_sims)}")
        
        # Show rescue examples
        if rescue_examples:
            example_texts = [
                f"→ Cluster {ex['cluster']} (sim: {ex['similarity']:.3f}): {ex['document']}"
                for ex in rescue_examples
            ]
            self.verbose_reporter.sample_list("c-TF-IDF rescue examples", example_texts)
        
        return self._create_results(rescued_count, total_outliers, new_assignments)
    
    def _create_results(self, rescued_count: int, total_outliers: int, new_assignments: Dict) -> Dict:
        """Create standardized results dictionary."""
        return {
            'rescued_count': rescued_count,
            'total_outliers': total_outliers,
            'success_rate': rescued_count / total_outliers if total_outliers > 0 else 0.0,
            'new_assignments': new_assignments,
            'method': 'c-tf-idf'
        }
    
    def get_topic_terms(self, topic_cluster_id: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top terms for a specific topic based on c-TF-IDF scores."""
        if self.topic_representations is None:
            raise ValueError("Topic representations not built yet")
        
        # Find topic index
        topic_info = getattr(self, '_last_topic_info', None)
        if not topic_info:
            raise ValueError("Topic information not available")
        
        try:
            topic_idx = topic_info['cluster_ids'].index(topic_cluster_id)
            return self.ctfidf_transformer.get_feature_importance(
                topic_idx, self.topic_representations, self.feature_names, top_k
            )
        except ValueError:
            return []