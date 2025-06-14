import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        
        # Debug: Track assignment decisions
        above_threshold_count = 0
        assignment_decisions = []
        
        for i, (idx, row) in enumerate(outliers_df.iterrows()):
            sim_scores = similarities[i]
            best_topic_idx = np.argmax(sim_scores)
            best_similarity = sim_scores[best_topic_idx]
            
            # Track all decisions for debugging
            decision = {
                'index': i,
                'df_idx': idx,
                'segment_id': row['segment_id'],
                'best_similarity': best_similarity,
                'threshold': threshold,
                'above_threshold': best_similarity >= threshold
            }
            assignment_decisions.append(decision)
            
            if best_similarity >= threshold:
                above_threshold_count += 1
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
        statistical_above_threshold = 0
        if len(similarities) > 0:
            max_sims = np.max(similarities, axis=1)
            statistical_above_threshold = np.sum(max_sims >= threshold)
            
            self.verbose_reporter.stat_line(f"Similarity scores - Min: {np.min(max_sims):.3f}, Max: {np.max(max_sims):.3f}, Mean: {np.mean(max_sims):.3f}")
            self.verbose_reporter.stat_line(f"Points above threshold ({threshold}): {statistical_above_threshold}/{len(max_sims)}")
        
        # Show rescue examples
        if rescue_examples:
            example_texts = [
                f"→ Cluster {ex['cluster']} (sim: {ex['similarity']:.3f}): {ex['document']}"
                for ex in rescue_examples
            ]
            self.verbose_reporter.sample_list("c-TF-IDF rescue examples", example_texts)
        
        # Debug: Verify assignment count matches rescued count
        actual_assignments = len(new_assignments)
        
        # Report detailed assignment analysis
        self.verbose_reporter.stat_line(f"Assignment loop analysis:")
        self.verbose_reporter.stat_line(f"  - Total outliers processed: {len(assignment_decisions)}")
        self.verbose_reporter.stat_line(f"  - Above threshold in loop: {above_threshold_count}")
        self.verbose_reporter.stat_line(f"  - rescued_count: {rescued_count}")
        self.verbose_reporter.stat_line(f"  - new_assignments length: {actual_assignments}")
        
        if actual_assignments != rescued_count:
            self.verbose_reporter.stat_line(f"⚠️  CRITICAL BUG: rescued_count={rescued_count} but new_assignments has {actual_assignments} entries!")
            self.verbose_reporter.stat_line(f"new_assignments keys: {list(new_assignments.keys())[:10]}")
        
        if above_threshold_count != rescued_count:
            self.verbose_reporter.stat_line(f"⚠️  CRITICAL BUG: above_threshold_count={above_threshold_count} but rescued_count={rescued_count}!")
        
        if statistical_above_threshold != above_threshold_count:
            self.verbose_reporter.stat_line(f"⚠️  CRITICAL BUG: statistical_above_threshold={statistical_above_threshold} but loop above_threshold_count={above_threshold_count}!")
        
        # Show a few assignment decisions for debugging
        if len(assignment_decisions) > 0:
            sample_decisions = assignment_decisions[:5]
            decision_texts = [
                f"idx={d['index']}, seg_id={d['segment_id']}, sim={d['best_similarity']:.4f}, above_thresh={d['above_threshold']}"
                for d in sample_decisions
            ]
            self.verbose_reporter.sample_list("Assignment decisions (first 5)", decision_texts)
        
        self.verbose_reporter.stat_line(f"✅ c-TF-IDF rescue verification: {rescued_count} rescued, {actual_assignments} assignments")
        
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
    
    def rescue_noise_points_with_embedding_comparison(self, 
                                                    documents: List[str],
                                                    embeddings: np.ndarray,
                                                    cluster_labels: List[int],
                                                    segment_ids: List[str] = None) -> Dict:
        """
        Enhanced rescue that compares c-TF-IDF and embedding-based similarities.
        
        This method implements the debugging approach to compare:
        1. c-TF-IDF similarity (text-based, used by BERTopic)
        2. Embedding-based cosine similarity (using ensemble embeddings)
        
        The goal is to identify discrepancies and understand the semantic gap.
        """
        self.verbose_reporter.step_start("Enhanced c-TF-IDF + embedding comparison rescue", "🔍")
        
        # Prepare data
        df = pd.DataFrame({
            'document': documents,
            'embedding': list(embeddings),
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
        
        if len(outliers_df) == 0:
            self.verbose_reporter.stat_line("No outliers to rescue")
            self.verbose_reporter.step_complete("Enhanced rescue completed")
            return self._create_results(0, 0, {})
        
        # Limit rescue attempts
        if len(outliers_df) > self.config.max_rescue_attempts:
            self.verbose_reporter.stat_line(f"Limiting rescue to {self.config.max_rescue_attempts} attempts")
            outliers_df = outliers_df.head(self.config.max_rescue_attempts)
        
        # Build both c-TF-IDF and embedding representations
        topic_info = self._build_topic_representations(topics_df)
        if not topic_info:
            self.verbose_reporter.stat_line("No valid topics for comparison")
            self.verbose_reporter.step_complete("Enhanced rescue completed")
            return self._create_results(0, len(outliers_df), {})
        
        # Calculate embedding-based centroids
        embedding_centroids = self._calculate_embedding_centroids(topics_df, topic_info['cluster_ids'])
        
        # Perform dual similarity rescue
        rescue_results = self._rescue_outliers_with_comparison(outliers_df, topic_info, embedding_centroids)
        
        self.verbose_reporter.step_complete("Enhanced c-TF-IDF + embedding comparison rescue completed")
        return rescue_results
    
    def _calculate_embedding_centroids(self, topics_df: pd.DataFrame, cluster_ids: List[int]) -> Dict[int, np.ndarray]:
        """Calculate cluster centroids from ensemble embeddings"""
        self.verbose_reporter.stat_line("Calculating embedding centroids...")
        
        centroids = {}
        for cluster_id in cluster_ids:
            cluster_embeddings = topics_df[topics_df['cluster'] == cluster_id]['embedding'].tolist()
            if cluster_embeddings:
                # Convert to numpy array and calculate mean
                embeddings_array = np.array(cluster_embeddings)
                centroids[cluster_id] = np.mean(embeddings_array, axis=0)
        
        self.verbose_reporter.stat_line(f"Calculated {len(centroids)} embedding centroids")
        return centroids
    
    def _rescue_outliers_with_comparison(self, outliers_df: pd.DataFrame, 
                                       topic_info: Dict, 
                                       embedding_centroids: Dict[int, np.ndarray]) -> Dict:
        """Rescue outliers while comparing c-TF-IDF and embedding similarities"""
        self.verbose_reporter.stat_line("Calculating both c-TF-IDF and embedding similarities...")
        
        # Get outlier data
        outlier_docs = outliers_df['document'].tolist()
        outlier_embeddings = np.array(outliers_df['embedding'].tolist())
        
        # Calculate c-TF-IDF similarities (existing approach)
        try:
            X_outliers = self.vectorizer.transform(outlier_docs)
            X_outliers_ctfidf = self.ctfidf_transformer.transform(X_outliers)
            ctfidf_similarities = self.ctfidf_transformer.calculate_similarity(
                X_outliers_ctfidf, topic_info['topic_representations']
            )
        except Exception as e:
            self.verbose_reporter.stat_line(f"❌ Error calculating c-TF-IDF similarities: {e}")
            return self._create_results(0, len(outliers_df), {})
        
        # Calculate embedding-based similarities
        cluster_ids = topic_info['cluster_ids']
        centroid_embeddings = np.array([embedding_centroids[cid] for cid in cluster_ids])
        embedding_similarities = cosine_similarity(outlier_embeddings, centroid_embeddings)
        
        # Compare and analyze similarities
        self._analyze_similarity_comparison(ctfidf_similarities, embedding_similarities, 
                                          outlier_docs, cluster_ids)
        
        # Apply rescue using both approaches
        rescue_results = self._apply_dual_rescue(outliers_df, ctfidf_similarities, 
                                               embedding_similarities, cluster_ids)
        
        return rescue_results
    
    def _analyze_similarity_comparison(self, ctfidf_sims: np.ndarray, 
                                     embedding_sims: np.ndarray,
                                     outlier_docs: List[str],
                                     cluster_ids: List[int]) -> None:
        """Detailed analysis comparing c-TF-IDF vs embedding similarities"""
        self.verbose_reporter.stat_line("")
        self.verbose_reporter.stat_line("=== SIMILARITY COMPARISON ANALYSIS ===")
        
        # Calculate best similarities for each approach
        ctfidf_best = np.max(ctfidf_sims, axis=1)
        embedding_best = np.max(embedding_sims, axis=1)
        
        # Basic statistics
        self.verbose_reporter.stat_line(f"c-TF-IDF similarities:")
        self.verbose_reporter.stat_line(f"  Min: {np.min(ctfidf_best):.4f}, Max: {np.max(ctfidf_best):.4f}")
        self.verbose_reporter.stat_line(f"  Mean: {np.mean(ctfidf_best):.4f}, Std: {np.std(ctfidf_best):.4f}")
        
        self.verbose_reporter.stat_line(f"Embedding similarities:")
        self.verbose_reporter.stat_line(f"  Min: {np.min(embedding_best):.4f}, Max: {np.max(embedding_best):.4f}")
        self.verbose_reporter.stat_line(f"  Mean: {np.mean(embedding_best):.4f}, Std: {np.std(embedding_best):.4f}")
        
        # Threshold analysis
        ctfidf_threshold = self.config.similarity_threshold
        
        ctfidf_above_thresh = np.sum(ctfidf_best >= ctfidf_threshold)
        embedding_above_thresh = np.sum(embedding_best >= ctfidf_threshold)
        
        self.verbose_reporter.stat_line(f"")
        self.verbose_reporter.stat_line(f"Above threshold ({ctfidf_threshold}):")
        self.verbose_reporter.stat_line(f"  c-TF-IDF: {ctfidf_above_thresh}/{len(ctfidf_best)} ({ctfidf_above_thresh/len(ctfidf_best):.1%})")
        self.verbose_reporter.stat_line(f"  Embedding: {embedding_above_thresh}/{len(embedding_best)} ({embedding_above_thresh/len(embedding_best):.1%})")
        
        # Correlation analysis
        correlation = np.corrcoef(ctfidf_best, embedding_best)[0, 1]
        self.verbose_reporter.stat_line(f"Correlation between approaches: {correlation:.4f}")
        
        # Show examples of high discrepancy cases
        differences = np.abs(ctfidf_best - embedding_best)
        high_discrepancy_indices = np.argsort(differences)[-3:]  # Top 3 discrepancies
        
        self.verbose_reporter.stat_line(f"")
        self.verbose_reporter.stat_line(f"HIGH DISCREPANCY EXAMPLES:")
        for i, idx in enumerate(high_discrepancy_indices):
            doc_preview = outlier_docs[idx][:60] + "..." if len(outlier_docs[idx]) > 60 else outlier_docs[idx]
            
            # Find best clusters for each approach
            ctfidf_best_cluster_idx = np.argmax(ctfidf_sims[idx])
            embedding_best_cluster_idx = np.argmax(embedding_sims[idx])
            
            ctfidf_best_cluster = cluster_ids[ctfidf_best_cluster_idx]
            embedding_best_cluster = cluster_ids[embedding_best_cluster_idx]
            
            self.verbose_reporter.stat_line(f"  Example {i+1}: '{doc_preview}'")
            self.verbose_reporter.stat_line(f"    c-TF-IDF: {ctfidf_best[idx]:.4f} → cluster {ctfidf_best_cluster}")
            self.verbose_reporter.stat_line(f"    Embedding: {embedding_best[idx]:.4f} → cluster {embedding_best_cluster}")
            self.verbose_reporter.stat_line(f"    Difference: {differences[idx]:.4f}")
        
        # Agreement analysis
        ctfidf_best_clusters = [cluster_ids[np.argmax(ctfidf_sims[i])] for i in range(len(ctfidf_sims))]
        embedding_best_clusters = [cluster_ids[np.argmax(embedding_sims[i])] for i in range(len(embedding_sims))]
        
        agreement = np.sum(np.array(ctfidf_best_clusters) == np.array(embedding_best_clusters))
        self.verbose_reporter.stat_line(f"")
        self.verbose_reporter.stat_line(f"Cluster assignment agreement: {agreement}/{len(ctfidf_best_clusters)} ({agreement/len(ctfidf_best_clusters):.1%})")
        
        self.verbose_reporter.stat_line("=== END SIMILARITY COMPARISON ===")
        self.verbose_reporter.stat_line("")
    
    def _apply_dual_rescue(self, outliers_df: pd.DataFrame,
                          ctfidf_sims: np.ndarray,
                          embedding_sims: np.ndarray,
                          cluster_ids: List[int]) -> Dict:
        """Apply rescue using primary c-TF-IDF with embedding as fallback"""
        rescued_count = 0
        new_assignments = {}
        rescue_examples = []
        
        threshold = self.config.similarity_threshold
        
        # Track rescue methods
        ctfidf_rescues = 0
        embedding_rescues = 0
        both_rescues = 0
        
        for i, (idx, row) in enumerate(outliers_df.iterrows()):
            ctfidf_scores = ctfidf_sims[i]
            embedding_scores = embedding_sims[i]
            
            ctfidf_best_idx = np.argmax(ctfidf_scores)
            embedding_best_idx = np.argmax(embedding_scores)
            
            ctfidf_best_sim = ctfidf_scores[ctfidf_best_idx]
            embedding_best_sim = embedding_scores[embedding_best_idx]
            
            # Primary: Use c-TF-IDF if above threshold
            if ctfidf_best_sim >= threshold:
                new_cluster = cluster_ids[ctfidf_best_idx]
                rescue_method = "c-TF-IDF"
                best_sim = ctfidf_best_sim
                ctfidf_rescues += 1
                
                # Check if embedding would also rescue
                if embedding_best_sim >= threshold:
                    both_rescues += 1
                    if cluster_ids[embedding_best_idx] == new_cluster:
                        rescue_method = "both (agree)"
                    else:
                        rescue_method = "both (disagree)"
            
            # Fallback: Use embedding if c-TF-IDF failed but embedding succeeds
            elif embedding_best_sim >= threshold:
                new_cluster = cluster_ids[embedding_best_idx]
                rescue_method = "embedding"
                best_sim = embedding_best_sim
                embedding_rescues += 1
            
            else:
                # Neither approach rescues this point
                continue
            
            # Record the rescue
            new_assignments[row['segment_id']] = new_cluster
            rescued_count += 1
            
            # Collect examples for reporting
            if self.config.show_examples and len(rescue_examples) < self.config.max_examples:
                rescue_examples.append({
                    'document': row['document'][:60] + "..." if len(row['document']) > 60 else row['document'],
                    'cluster': new_cluster,
                    'similarity': best_sim,
                    'method': rescue_method,
                    'ctfidf_sim': ctfidf_best_sim,
                    'embedding_sim': embedding_best_sim
                })
        
        # Report results
        total_outliers = len(outliers_df)
        success_rate = rescued_count / total_outliers if total_outliers > 0 else 0.0
        
        self.verbose_reporter.stat_line(f"")
        self.verbose_reporter.stat_line(f"=== DUAL RESCUE RESULTS ===")
        self.verbose_reporter.stat_line(f"Total rescued: {rescued_count}/{total_outliers} ({success_rate:.1%})")
        self.verbose_reporter.stat_line(f"c-TF-IDF rescues: {ctfidf_rescues}")
        self.verbose_reporter.stat_line(f"Embedding rescues: {embedding_rescues}")
        self.verbose_reporter.stat_line(f"Both methods agreed: {ctfidf_rescues - (both_rescues - embedding_rescues)}")
        self.verbose_reporter.stat_line(f"Both methods disagreed: {both_rescues - (ctfidf_rescues - (both_rescues - embedding_rescues))}")
        
        # Show rescue examples
        if rescue_examples:
            example_texts = []
            for ex in rescue_examples:
                example_texts.append(
                    f"→ Cluster {ex['cluster']} via {ex['method']} (c-TF-IDF: {ex['ctfidf_sim']:.3f}, emb: {ex['embedding_sim']:.3f}): {ex['document']}"
                )
            self.verbose_reporter.sample_list("Dual rescue examples", example_texts)
        
        return self._create_results(rescued_count, total_outliers, new_assignments)
    
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