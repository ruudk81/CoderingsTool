import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List, Any, Optional, Dict
import numpy.typing as npt
from collections import defaultdict, Counter
from umap import UMAP
import hdbscan
import spacy 
from sklearn.feature_extraction.text import CountVectorizer

# config
import models
from config import DEFAULT_LANGUAGE, ClusteringConfig, DEFAULT_CLUSTERING_CONFIG
from utils.clusterQualifier import ClusterQualityAnalyzer
from utils.verboseReporter import VerboseReporter
from utils.ctfidf_noise_reducer import CtfidfNoiseReducer, CtfidfNoiseRescueConfig
import warnings  # hard coded warning in umap about hidden stat
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")

# Structured formats 
class ResultMapper(BaseModel):
    # input
    respondent_id: Any
    segment_id: str
    segment_label: str
    segment_description: str
    code_embedding: npt.NDArray[np.float32]
    description_embedding: npt.NDArray[np.float32]
    # add
    reduced_code_embedding: Optional[npt.NDArray[np.float32]] = None
    reduced_description_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_code_cluster: Optional[int] = None
    initial_description_cluster: Optional[int] = None
    # config
    model_config = ConfigDict(arbitrary_types_allowed=True)  # for arrays with embeddings


# Main utils 
class ClusterGenerator:
    def __init__(
        self,
        input_list: List[models.EmbeddingsModel] = None,
        var_lab=None,
        dim_reduction_model=None,
        cluster_model=None,
        vectorizer_model=None,
        config: ClusteringConfig = None,
        embedding_type: str = None,  # Can be "code" OR "description" 
        verbose: bool = None):
        
        # Initialize configuration
        self.config = config or DEFAULT_CLUSTERING_CONFIG
        
        self.var_lab = var_lab if var_lab else ""
        self.embedding_type = embedding_type or self.config.embedding_type
        self.output_list: List[ResultMapper] = []
        self.verbose = verbose if verbose is not None else self.config.verbose
        # Store the original input list to preserve response data
        self.original_input_list = input_list if input_list else []
        
        # Initialize verbose reporter
        self.verbose_reporter = VerboseReporter(self.verbose)
       
        if input_list:
            self.populate_from_input_list(input_list)
        
        # Initialize dimensionality reduction model
        if dim_reduction_model is None:
            umap_config = self.config.umap
            self.dim_reduction_model = UMAP(
                n_neighbors=umap_config.n_neighbors,
                n_components=umap_config.n_components,
                min_dist=umap_config.min_dist,
                metric=umap_config.metric,
                random_state=umap_config.random_state,
                n_jobs=umap_config.n_jobs,
                low_memory=umap_config.low_memory,
                transform_seed=umap_config.transform_seed)
            self.verbose_reporter.stat_line("Using configured UMAP dimensionality reduction")
        else:
            self.dim_reduction_model = dim_reduction_model
            
        # Initialize clustering model
        if cluster_model is None:
            hdbscan_config = self.config.hdbscan
            hdbscan_params = {
                'metric': hdbscan_config.metric,
                'cluster_selection_method': hdbscan_config.cluster_selection_method,
                'prediction_data': hdbscan_config.prediction_data,
                'approx_min_span_tree': hdbscan_config.approx_min_span_tree,
                'gen_min_span_tree': hdbscan_config.gen_min_span_tree
            }
            # Add optional parameters if configured
            if hdbscan_config.min_cluster_size is not None:
                hdbscan_params['min_cluster_size'] = hdbscan_config.min_cluster_size
            if hdbscan_config.min_samples is not None:
                hdbscan_params['min_samples'] = hdbscan_config.min_samples
                
            self.cluster_model = hdbscan.HDBSCAN(**hdbscan_params)
            self.verbose_reporter.stat_line("Using configured HDBSCAN clustering")
        else:
            self.cluster_model = cluster_model
        
        # Initialize vectorizer model
        if vectorizer_model is None:
            vectorizer_config = self.config.vectorizer
            stop_words = self._get_stop_words() if vectorizer_config.use_language_stop_words else None
            
            vectorizer_params = {
                'ngram_range': vectorizer_config.ngram_range,
                'min_df': vectorizer_config.min_df,
                'max_df': vectorizer_config.max_df
            }
            if stop_words is not None:
                vectorizer_params['stop_words'] = stop_words
            if vectorizer_config.max_features is not None:
                vectorizer_params['max_features'] = vectorizer_config.max_features
                
            self.vectorizer_model = CountVectorizer(**vectorizer_params)
            self.verbose_reporter.stat_line("Using configured CountVectorizer")
        else:
            self.vectorizer_model = vectorizer_model

    def _get_stop_words(self):
        if DEFAULT_LANGUAGE == "Dutch":
            try:
                return list(spacy.load("nl_core_news_lg").Defaults.stop_words)
            except:
                self.verbose_reporter.stat_line("Warning: Dutch language model not found. Using English stop words.")
                return 'english'
        else:
            return 'english'
    
    def _get_top_clusters_by_size(self, clusters, n=5):
        """Get top N clusters by size with representative descriptions"""
        cluster_counts = Counter(clusters)
        # Remove noise cluster (-1) for top clusters
        valid_clusters = {k: v for k, v in cluster_counts.items() if k != -1}
        top_clusters = sorted(valid_clusters.items(), key=lambda x: x[1], reverse=True)[:n]
        
        cluster_examples = []
        for cluster_id, count in top_clusters:
            # Get representative description for this cluster
            cluster_items = [item for item in self.output_list 
                           if (self.embedding_type == "code" and item.initial_code_cluster == cluster_id) or
                              (self.embedding_type == "description" and item.initial_description_cluster == cluster_id)]
            
            if cluster_items:
                rep_desc = self._get_representative_description(cluster_items)
                cluster_examples.append(f"Cluster {cluster_id}: {count} items - {rep_desc}")
        
        return cluster_examples
    
    def _get_representative_description(self, cluster_items, max_length=50):
        """Get most representative description using centroid similarity"""
        if not cluster_items:
            return "No items"
            
        # Use the first description as fallback
        if len(cluster_items) == 1:
            desc = cluster_items[0].segment_label if self.embedding_type == "code" else cluster_items[0].segment_description
            return desc[:max_length] + "..." if len(desc) > max_length else desc
        
        # For multiple items, find most representative
        descriptions = [item.segment_label if self.embedding_type == "code" else item.segment_description for item in cluster_items]
        embeddings = [item.code_embedding if self.embedding_type == "code" else item.description_embedding for item in cluster_items 
                     if (item.code_embedding if self.embedding_type == "code" else item.description_embedding) is not None]
        
        if embeddings:
            # Calculate centroid and find closest description
            embeddings_array = np.array(embeddings)
            centroid = np.mean(embeddings_array, axis=0)
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(embeddings_array, centroid.reshape(1, -1)).flatten()
            best_idx = np.argmax(similarities)
            desc = descriptions[best_idx]
        else:
            # Fallback to first description
            desc = descriptions[0]
        
        return desc[:max_length] + "..." if len(desc) > max_length else desc

    def populate_from_input_list(self, input_list: List[models.EmbeddingsModel]) -> None:
        self.verbose_reporter.stat_line("Populating output list from input models")
        
        self.output_list = []
        
        for response_item in input_list:
            if response_item.response_segment:
                for segment_item in response_item.response_segment:
                    # Check if segment_item has the required attributes
                    if (hasattr(segment_item, 'segment_label') and 
                        hasattr(segment_item, 'segment_description') and
                        hasattr(segment_item, 'code_embedding') and
                        hasattr(segment_item, 'description_embedding')):
                        
                        self.output_list.append(ResultMapper(
                            respondent_id=response_item.respondent_id,
                            segment_id=segment_item.segment_id,
                            segment_label=segment_item.segment_label or "NA",
                            segment_description=segment_item.segment_description or "NA",
                            code_embedding=segment_item.code_embedding,
                            description_embedding=segment_item.description_embedding
                        ))

    def add_reduced_embeddings(self) -> None:
        # Reduces dimensionality of embeddings and adds them to output_list.
        self.verbose_reporter.step_start("Reducing dimensionality of embeddings", "📊")
        
        # Process code embeddings if needed
        if self.embedding_type == "code":
            code_embeddings_array = np.array([item.code_embedding for item in self.output_list])
            reduced_code_embeddings = self.dim_reduction_model.fit_transform(code_embeddings_array)
            
            # Add reduced embeddings to output list
            for i, item in enumerate(self.output_list):
                item.reduced_code_embedding = reduced_code_embeddings[i]
        
        # Process description embeddings if needed
        if self.embedding_type == "description":
            description_embeddings_array = np.array([item.description_embedding for item in self.output_list])
            reduced_description_embeddings = self.dim_reduction_model.fit_transform(description_embeddings_array)
            
            # Add reduced embeddings to output list
            for i, item in enumerate(self.output_list):
                item.reduced_description_embedding = reduced_description_embeddings[i]
                
        self.verbose_reporter.step_complete("Dimensionality reduction completed")

    def add_initial_clusters(self) -> None:
        # Performs initial clustering on reduced embeddings and adds cluster labels to output_list.
        self.verbose_reporter.step_start("Clustering reduced embeddings", "🔍")

        # Cluster code embeddings if needed
        if self.embedding_type == "code":
            reduced_code_embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
            initial_code_clusters = self.cluster_model.fit_predict(reduced_code_embeddings)
            
            # Add cluster labels to output list
            for i, item in enumerate(self.output_list):
                item.initial_code_cluster = initial_code_clusters[i]
            
            cluster_counts = Counter(initial_code_clusters)
            self.verbose_reporter.stat_line(f"Code clusters: {len(set(initial_code_clusters))} clusters found")
            noise_count = cluster_counts.get(-1, 0)
            if noise_count > 0:
                self.verbose_reporter.stat_line(f"Noise cluster (-1): {noise_count} items")
            
            # Show top clusters with smart examples
            top_clusters = self._get_top_clusters_by_size(initial_code_clusters, 5)
            if top_clusters:
                self.verbose_reporter.sample_list("Largest clusters", top_clusters)
                
        # Cluster description embeddings if needed
        if self.embedding_type == "description":
            reduced_description_embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
            initial_description_clusters = self.cluster_model.fit_predict(reduced_description_embeddings)
            
            # Add cluster labels to output list
            for i, item in enumerate(self.output_list):
                item.initial_description_cluster = initial_description_clusters[i]
            
            cluster_counts = Counter(initial_description_clusters)
            self.verbose_reporter.stat_line(f"Description clusters: {len(set(initial_description_clusters))} clusters found")
            noise_count = cluster_counts.get(-1, 0)
            if noise_count > 0:
                self.verbose_reporter.stat_line(f"Noise cluster (-1): {noise_count} items")
            
            # Show top clusters with smart examples
            top_clusters = self._get_top_clusters_by_size(initial_description_clusters, 5)
            if top_clusters:
                self.verbose_reporter.sample_list("Largest clusters", top_clusters)
        
        self.verbose_reporter.step_complete("Initial clustering completed")

    def _calculate_cluster_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculate cluster centroids using simple mean of embeddings"""
        centroids = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id != -1:  # Skip noise points
                cluster_mask = labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        return centroids

    def _cosine_similarity_rescue(self, noise_indices: np.ndarray, embeddings: np.ndarray, labels: np.ndarray) -> int:
        """Rescue noise points using cosine similarity to cluster centroids"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate cluster centroids
        centroids = self._calculate_cluster_centroids(embeddings, labels)
        
        if not centroids:
            self.verbose_reporter.stat_line("No clusters found for centroid calculation")
            return 0
        
        cluster_ids = list(centroids.keys())
        centroid_vectors = np.array(list(centroids.values()))
        
        self.verbose_reporter.stat_line(f"Calculated centroids for {len(centroids)} clusters")
        
        rescued_count = 0
        threshold = self.config.noise_rescue.cosine_similarity_threshold
        
        # Process each noise point
        for noise_idx in noise_indices:
            noise_embedding = embeddings[noise_idx].reshape(1, -1)
            
            # Calculate cosine similarities to all centroids
            similarities = cosine_similarity(noise_embedding, centroid_vectors)[0]
            
            # Find best cluster
            best_cluster_idx = np.argmax(similarities)
            best_similarity = similarities[best_cluster_idx]
            
            # Assign if above threshold
            if best_similarity >= threshold:
                best_cluster_id = cluster_ids[best_cluster_idx]
                
                # Update the appropriate cluster field
                if self.embedding_type == "code":
                    self.output_list[noise_idx].initial_code_cluster = best_cluster_id
                else:
                    self.output_list[noise_idx].initial_description_cluster = best_cluster_id
                
                rescued_count += 1
        
        return rescued_count
    
    def _embedding_based_similarity_comparison(self, noise_indices: np.ndarray, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate embedding-based similarities for comparison with c-TF-IDF"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate cluster centroids (same as in cosine rescue)
        centroids = self._calculate_cluster_centroids(embeddings, labels)
        
        if not centroids:
            return {}
        
        cluster_ids = list(centroids.keys())
        centroid_vectors = np.array(list(centroids.values()))
        
        similarities = {}
        for noise_idx in noise_indices:
            noise_embedding = embeddings[noise_idx].reshape(1, -1)
            sim_scores = cosine_similarity(noise_embedding, centroid_vectors)[0]
            best_cluster_idx = np.argmax(sim_scores)
            best_similarity = sim_scores[best_cluster_idx]
            best_cluster_id = cluster_ids[best_cluster_idx]
            
            similarities[noise_idx] = {
                'best_cluster': best_cluster_id,
                'similarity': best_similarity,
                'all_similarities': dict(zip(cluster_ids, sim_scores))
            }
        
        return similarities

    def _original_response_embedding_similarity(self, noise_indices: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate similarities using ORIGINAL response embeddings for fairer c-TF-IDF comparison"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get ORIGINAL response embeddings (before ensemble weighting)
        try:
            original_embeddings = np.array([item.description_embedding for item in self.output_list])
            self.verbose_reporter.stat_line("🔬 Calculating similarities with original response embeddings")
        except:
            self.verbose_reporter.stat_line("⚠️  Original embeddings not available, using ensemble embeddings")
            if self.embedding_type == "code":
                original_embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
            else:
                original_embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
        
        # Calculate centroids from ORIGINAL embeddings (not ensemble)
        centroids = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id != -1:  # Skip noise points
                cluster_mask = labels == cluster_id
                cluster_embeddings = original_embeddings[cluster_mask]
                centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        if not centroids:
            return {}
        
        cluster_ids = list(centroids.keys())
        centroid_vectors = np.array(list(centroids.values()))
        
        similarities = {}
        for noise_idx in noise_indices:
            noise_embedding = original_embeddings[noise_idx].reshape(1, -1)
            sim_scores = cosine_similarity(noise_embedding, centroid_vectors)[0]
            best_cluster_idx = np.argmax(sim_scores)
            best_similarity = sim_scores[best_cluster_idx]
            best_cluster_id = cluster_ids[best_cluster_idx]
            
            similarities[noise_idx] = {
                'best_cluster': best_cluster_id,
                'similarity': best_similarity,
                'all_similarities': dict(zip(cluster_ids, sim_scores))
            }
        
        return similarities

    def _manual_embedding_comparison(self, embedding_similarities: Dict, ctfidf_results: Dict) -> None:
        """Manual comparison of embedding vs c-TF-IDF similarities when enhanced method fails."""
        
        if not embedding_similarities:
            return
            
        self.verbose_reporter.step_start("Embedding vs c-TF-IDF Similarity Analysis", "🔬")
        
        # Extract embedding similarity statistics
        embedding_sims = [data['similarity'] for data in embedding_similarities.values()]
        
        if embedding_sims:
            avg_emb_sim = np.mean(embedding_sims)
            max_emb_sim = np.max(embedding_sims) 
            min_emb_sim = np.min(embedding_sims)
            
            self.verbose_reporter.stat_line(f"📊 Embedding similarities - Min: {min_emb_sim:.3f}, Max: {max_emb_sim:.3f}, Mean: {avg_emb_sim:.3f}")
            
            # Count embeddings above cosine threshold
            emb_above_threshold = sum(1 for sim in embedding_sims if sim >= 0.7)
            total_noise = len(embedding_sims)
            
            self.verbose_reporter.stat_line(f"🔍 Embedding similarities above cosine threshold (0.7): {emb_above_threshold}/{total_noise}")
            self.verbose_reporter.stat_line(f"🔍 c-TF-IDF rescued: {ctfidf_results.get('rescued_count', 0)}/{total_noise}")
            
            # Analysis
            if emb_above_threshold > ctfidf_results.get('rescued_count', 0):
                ratio = emb_above_threshold / max(ctfidf_results.get('rescued_count', 0), 1)
                self.verbose_reporter.stat_line(f"⚠️  DISCREPANCY: Embedding-based rescue could rescue {ratio:.1f}x more points")
                self.verbose_reporter.stat_line("💡 Consider using embedding-based rescue as primary method")
                
                # Show example high-embedding, low-ctfidf points
                high_emb_examples = [(idx, data) for idx, data in embedding_similarities.items() 
                                   if data['similarity'] >= 0.7][:3]
                if high_emb_examples:
                    example_texts = [f"Point {idx}: {data['similarity']:.3f} similarity to cluster {data['best_cluster']}" 
                                   for idx, data in high_emb_examples]
                    self.verbose_reporter.sample_list("High embedding similarity examples", example_texts)
            else:
                self.verbose_reporter.stat_line("✅ c-TF-IDF and embedding similarities are reasonably aligned")
        
        self.verbose_reporter.step_complete("Similarity analysis completed")

    def _enhanced_manual_comparison(self, ensemble_similarities: Dict, original_similarities: Dict, ctfidf_results: Dict) -> None:
        """Enhanced comparison showing both ensemble and original embedding similarities vs c-TF-IDF."""
        
        self.verbose_reporter.step_start("Enhanced Embedding vs c-TF-IDF Analysis", "🔬")
        
        # Compare ensemble embeddings (current clustering basis)
        if ensemble_similarities:
            ensemble_sims = [data['similarity'] for data in ensemble_similarities.values()]
            ensemble_above_threshold = sum(1 for sim in ensemble_sims if sim >= 0.7)
            
            self.verbose_reporter.stat_line(f"📊 Ensemble embeddings - Min: {np.min(ensemble_sims):.3f}, Max: {np.max(ensemble_sims):.3f}, Mean: {np.mean(ensemble_sims):.3f}")
            self.verbose_reporter.stat_line(f"🔍 Ensemble above threshold (0.7): {ensemble_above_threshold}/{len(ensemble_sims)}")
        
        # Compare original response embeddings (fairer comparison to c-TF-IDF)
        if original_similarities:
            original_sims = [data['similarity'] for data in original_similarities.values()]
            original_above_threshold = sum(1 for sim in original_sims if sim >= 0.7)
            
            self.verbose_reporter.stat_line(f"📊 Original response embeddings - Min: {np.min(original_sims):.3f}, Max: {np.max(original_sims):.3f}, Mean: {np.mean(original_sims):.3f}")
            self.verbose_reporter.stat_line(f"🔍 Original embeddings above threshold (0.7): {original_above_threshold}/{len(original_sims)}")
            
            # Compare c-TF-IDF performance
            ctfidf_rescued = ctfidf_results.get('rescued_count', 0)
            total_noise = len(original_sims)
            
            self.verbose_reporter.stat_line(f"🔍 c-TF-IDF rescued: {ctfidf_rescued}/{total_noise}")
            
            # Analysis
            if original_above_threshold > ctfidf_rescued:
                ratio = original_above_threshold / max(ctfidf_rescued, 1)
                self.verbose_reporter.stat_line(f"⚠️  DISCREPANCY (vs original embeddings): {ratio:.1f}x more points could be rescued")
                self.verbose_reporter.stat_line("💡 Even with original embeddings, text-based c-TF-IDF underperforms")
            else:
                self.verbose_reporter.stat_line("✅ c-TF-IDF performance aligns better with original embeddings")
            
            # Show the difference between ensemble and original similarities
            if ensemble_similarities and len(ensemble_sims) == len(original_sims):
                mean_diff = np.mean(ensemble_sims) - np.mean(original_sims)
                self.verbose_reporter.stat_line(f"📈 Ensemble boost: {mean_diff:+.3f} average similarity increase from question context")
                
                # Show examples of both
                for i, (ensemble_idx, original_idx) in enumerate(zip(list(ensemble_similarities.keys())[:3], list(original_similarities.keys())[:3])):
                    ensemble_sim = ensemble_similarities[ensemble_idx]['similarity']
                    original_sim = original_similarities[original_idx]['similarity']
                    boost = ensemble_sim - original_sim
                    self.verbose_reporter.stat_line(f"  Example {i+1}: Original={original_sim:.3f} → Ensemble={ensemble_sim:.3f} (boost: {boost:+.3f})")
        
        self.verbose_reporter.step_complete("Enhanced similarity analysis completed")

    def _ctfidf_rescue(self, current_labels: np.ndarray) -> int:
        """Rescue remaining noise points using c-TF-IDF similarity with embedding comparison"""
        
        # Prepare documents, embeddings, and labels for hybrid rescue
        documents = []
        embeddings = []
        cluster_labels = []        
        segment_ids = []
        
        for item in self.output_list:
            # Use description for text analysis (more content than labels)
            if self.embedding_type == "code":
                text = item.segment_label or ""
                embedding = item.code_embedding
                cluster = item.initial_code_cluster
            else:
                text = item.segment_description or ""
                embedding = item.description_embedding
                cluster = item.initial_description_cluster
            
            if text and len(text.strip()) > 0 and embedding is not None:  # Require both text and embedding
                documents.append(text)
                embeddings.append(embedding)
                cluster_labels.append(cluster)
                segment_ids.append(item.segment_id)
        
        # Debug: Check for duplicate segment IDs
        unique_segment_ids = set(segment_ids)
        duplicate_count = len(segment_ids) - len(unique_segment_ids)
        
        if duplicate_count > 0:
            self.verbose_reporter.stat_line(f"⚠️  WARNING: Found {duplicate_count} duplicate segment IDs in c-TF-IDF input")
            
            # Show format analysis
            compound_format = sum(1 for sid in segment_ids if '_' in str(sid))
            simple_format = sum(1 for sid in segment_ids if str(sid).isdigit())
            
            self.verbose_reporter.stat_line(f"📊 Format analysis: {compound_format} compound, {simple_format} simple format")
            
            # Show sample duplicates
            from collections import Counter
            segment_counts = Counter(segment_ids)
            duplicates = [(sid, count) for sid, count in segment_counts.items() if count > 1]
            if duplicates[:3]:  # Show first 3 duplicates
                dup_texts = [f"'{sid}': {count}x" for sid, count in duplicates[:3]]
                self.verbose_reporter.sample_list("Sample duplicates", dup_texts)
        else:
            self.verbose_reporter.stat_line(f"✅ SEGMENT ID VALIDATION: All {len(segment_ids)} segment IDs are unique!")
            
            # Show format verification
            compound_format = sum(1 for sid in segment_ids if '_' in str(sid))
            if compound_format == len(segment_ids):
                self.verbose_reporter.stat_line(f"✅ All segment IDs use proper compound format (respondent_id_segment)")
            else:
                simple_format = sum(1 for sid in segment_ids if str(sid).isdigit())
                self.verbose_reporter.stat_line(f"📊 Format mix: {compound_format} compound, {simple_format} simple format")
        
        # Debug output
        valid_clusters = [c for c in cluster_labels if c != -1]
        unique_clusters = len(set(valid_clusters)) if valid_clusters else 0
        noise_count = cluster_labels.count(-1)
        
        self.verbose_reporter.stat_line(f"c-TF-IDF input: {len(documents)} documents")
        self.verbose_reporter.stat_line(f"Valid clusters: {unique_clusters}, Noise points: {noise_count}")
        self.verbose_reporter.stat_line(f"Embedding type: {self.embedding_type}")
        
        # Sample document check
        if documents:
            sample_doc = documents[0]
            self.verbose_reporter.stat_line(f"Sample document: '{sample_doc[:60]}...' (cluster: {cluster_labels[0]})")
        
        if not documents:
            self.verbose_reporter.stat_line("❌ No valid documents for c-TF-IDF rescue")
            return 0
        
        if unique_clusters == 0:
            self.verbose_reporter.stat_line("❌ No valid clusters for c-TF-IDF comparison")
            return 0
        
        # Fit vectorizer on all documents if not already fitted
        try:
            # Test if vectorizer is fitted
            self.vectorizer_model.get_feature_names_out()
            self.verbose_reporter.stat_line("✅ Vectorizer already fitted")
        except:
            # Fit vectorizer on all available documents
            self.verbose_reporter.stat_line("Fitting vectorizer on all documents...")
            self.vectorizer_model.fit(documents)
            vocab_size = len(self.vectorizer_model.get_feature_names_out())
            self.verbose_reporter.stat_line(f"Vectorizer fitted: {vocab_size} features")
        
        # Calculate embedding-based similarities for comparison
        noise_mask = np.array(cluster_labels) == -1
        noise_indices = np.where(noise_mask)[0]
        
        # Get embeddings for noise points comparison
        # Use ORIGINAL response embeddings instead of ensemble embeddings for fairer comparison
        if self.embedding_type == "code":
            rescue_embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
        else:
            # Try to get original response embeddings before ensemble weighting
            try:
                # Access original response embeddings if available
                rescue_embeddings = np.array([item.description_embedding for item in self.output_list])
                self.verbose_reporter.stat_line("🔬 Using original response embeddings for comparison (before ensemble weighting)")
            except:
                # Fallback to reduced ensemble embeddings
                rescue_embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
                self.verbose_reporter.stat_line("⚠️  Using ensemble embeddings for comparison (original not available)")
        
        # Calculate both ensemble and original response embedding similarities for comparison
        ensemble_similarities = self._embedding_based_similarity_comparison(
            noise_indices, rescue_embeddings, current_labels
        )
        
        original_similarities = self._original_response_embedding_similarity(
            noise_indices, current_labels
        )
        
        # Configure c-TF-IDF rescue with embedding comparison
        ctfidf_config = CtfidfNoiseRescueConfig(
            enabled=True,
            similarity_threshold=self.config.noise_rescue.ctfidf_similarity_threshold,
            min_topic_size=self.config.noise_rescue.ctfidf_min_topic_size,
            verbose=self.verbose
        )
        
        # Initialize enhanced c-TF-IDF noise reducer with fitted vectorizer
        ctfidf_reducer = CtfidfNoiseReducer(
            vectorizer=self.vectorizer_model,
            config=ctfidf_config,
            verbose=self.verbose
        )
        
        # Perform hybrid c-TF-IDF + embedding similarity rescue
        try:
            rescue_results = ctfidf_reducer.rescue_noise_points_with_embedding_comparison(
                documents=documents,
                cluster_labels=cluster_labels,
                segment_ids=segment_ids,
                embedding_similarities=original_similarities  # Use original embeddings for fairer comparison
            )
        except TypeError as e:
            self.verbose_reporter.stat_line(f"⚠️  Enhanced rescue failed: {e}")
            self.verbose_reporter.stat_line("Falling back to standard c-TF-IDF rescue")
            rescue_results = ctfidf_reducer.rescue_noise_points(
                documents=documents,
                cluster_labels=cluster_labels,
                segment_ids=segment_ids
            )
            
            # Add manual embedding comparison (with both ensemble and original)
            self._enhanced_manual_comparison(ensemble_similarities, original_similarities, rescue_results)
            
            # Verify BERTopic implementation
            from .ctfidf_transformer import verify_bertopic_implementation
            verify_bertopic_implementation()
        
        # Apply rescue results to output_list
        rescued_count = 0
        new_assignments = rescue_results.get('new_assignments', {})
        
        # Debug: Report what we received from c-TF-IDF rescue
        reported_rescued = rescue_results.get('rescued_count', 0)
        self.verbose_reporter.stat_line(f"c-TF-IDF reported: {reported_rescued} rescued, {len(new_assignments)} assignments to apply")
        
        # Debug: Show what assignments we received
        if new_assignments:
            assignment_items = list(new_assignments.items())[:3]  # First 3
            assignment_texts = [f"seg_id={seg_id} → cluster {cluster}" for seg_id, cluster in assignment_items]
            self.verbose_reporter.sample_list("Assignments to apply", assignment_texts)
        
        # Debug: Track assignment application
        items_checked = 0
        items_matched = 0
        items_was_noise = 0
        items_was_not_noise = 0
        
        for item in self.output_list:
            items_checked += 1
            if item.segment_id in new_assignments:
                items_matched += 1
                new_cluster = new_assignments[item.segment_id]
                
                # Update the appropriate cluster field
                if self.embedding_type == "code":
                    if item.initial_code_cluster == -1:  # Only update if it was noise
                        item.initial_code_cluster = new_cluster
                        rescued_count += 1
                        items_was_noise += 1
                    else:
                        items_was_not_noise += 1
                else:
                    if item.initial_description_cluster == -1:  # Only update if it was noise
                        item.initial_description_cluster = new_cluster
                        rescued_count += 1
                        items_was_noise += 1
                    else:
                        items_was_not_noise += 1
        
        # Debug: Report assignment application results
        self.verbose_reporter.stat_line(f"🔍 DEBUG: Checked {items_checked} items in output_list")
        self.verbose_reporter.stat_line(f"🔍 DEBUG: Found {items_matched} items with matching segment_ids")
        self.verbose_reporter.stat_line(f"🔍 DEBUG: Applied to {items_was_noise} noise points")
        self.verbose_reporter.stat_line(f"🔍 DEBUG: Skipped {items_was_not_noise} non-noise points")
        
        # Debug: Verify final counts
        self.verbose_reporter.stat_line(f"c-TF-IDF assignment result: applied {rescued_count} assignments from {len(new_assignments)} available")
        if rescued_count != len(new_assignments):
            self.verbose_reporter.stat_line(f"⚠️  Assignment mismatch: some segment_ids in new_assignments were not found in output_list or were not noise points")
        
        return rescued_count
    
    def run_similarity_diagnostics(self) -> Dict:
        """
        Run comprehensive diagnostics to compare c-TF-IDF vs embedding similarities.
        This method helps identify why c-TF-IDF similarities are low.
        """
        self.verbose_reporter.section_header("SIMILARITY DIAGNOSTICS", "🔬")
        
        # Collect documents and embeddings for analysis
        documents = []
        embeddings = []
        cluster_labels = []
        
        for item in self.output_list:
            if self.embedding_type == "code":
                text = item.segment_label or ""
                embedding = item.code_embedding 
                cluster = item.initial_code_cluster
            else:
                text = item.segment_description or ""
                embedding = item.description_embedding
                cluster = item.initial_description_cluster
            
            if text and len(text.strip()) > 0 and embedding is not None:
                documents.append(text)
                embeddings.append(embedding)
                cluster_labels.append(cluster)
        
        if not documents:
            self.verbose_reporter.stat_line("❌ No valid documents for diagnostics")
            return {'error': 'No valid documents'}
        
        # 1. Verify c-TF-IDF implementation against BERTopic
        self.verbose_reporter.step_start("Step 1: Verifying c-TF-IDF implementation", "🔍")
        
        try:
            # Fit vectorizer if not already fitted
            try:
                self.vectorizer_model.get_feature_names_out()
            except:
                self.vectorizer_model.fit(documents)
            
            # Initialize c-TF-IDF transformer
            from .ctfidf_transformer import CtfidfTransformer, CtfidfConfig
            ctfidf_config = CtfidfConfig(verbose=True)
            ctfidf_transformer = CtfidfTransformer(ctfidf_config, verbose=True)
            
            # Run BERTopic verification
            verification_result = ctfidf_transformer.verify_bertopic_implementation(documents[:20])  # Use sample
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"❌ c-TF-IDF verification failed: {e}")
            verification_result = {'error': str(e)}
        
        # 2. Analyze embedding properties
        self.verbose_reporter.step_start("Step 2: Analyzing ensemble embeddings", "📊")
        
        embeddings_array = np.array(embeddings)
        self.verbose_reporter.stat_line(f"Embedding shape: {embeddings_array.shape}")
        self.verbose_reporter.stat_line(f"Embedding stats - Mean: {np.mean(embeddings_array):.4f}, Std: {np.std(embeddings_array):.4f}")
        
        # Check for ensemble embedding characteristics
        embedding_norms = np.linalg.norm(embeddings_array, axis=1)
        self.verbose_reporter.stat_line(f"Embedding norms - Min: {np.min(embedding_norms):.4f}, Max: {np.max(embedding_norms):.4f}")
        
        # 3. Compare approaches on actual noise points
        self.verbose_reporter.step_start("Step 3: Comparing approaches on noise points", "🔍")
        
        noise_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        if noise_indices:
            noise_sample = noise_indices[:10]  # Sample first 10 noise points
            
            self.verbose_reporter.stat_line(f"Analyzing {len(noise_sample)} noise points out of {len(noise_indices)} total")
            
            # This will trigger the detailed comparison in the enhanced rescue method
            try:
                # Prepare sample data
                sample_docs = [documents[i] for i in noise_sample]
                sample_embeddings = embeddings_array[noise_sample]
                sample_labels = [cluster_labels[i] for i in noise_sample]
                sample_ids = [f"diagnostic_{i}" for i in range(len(sample_docs))]
                
                # Use the enhanced rescue method for detailed analysis
                from .ctfidf_noise_reducer import CtfidfNoiseReducer, CtfidfNoiseRescueConfig
                
                config = CtfidfNoiseRescueConfig(
                    similarity_threshold=0.05,  # Lower threshold for diagnostic
                    verbose=True,
                    show_examples=True,
                    max_examples=5
                )
                
                reducer = CtfidfNoiseReducer(self.vectorizer_model, config, verbose=True)
                
                # Run enhanced rescue for diagnostic purposes
                diagnostic_result = reducer.rescue_noise_points_with_embedding_comparison(
                    documents=documents,  # Full document set for proper c-TF-IDF
                    embeddings=embeddings_array,
                    cluster_labels=cluster_labels,
                    segment_ids=[f"diag_{i}" for i in range(len(documents))]
                )
                
            except Exception as e:
                self.verbose_reporter.stat_line(f"❌ Comparative analysis failed: {e}")
                diagnostic_result = {'error': str(e)}
        else:
            self.verbose_reporter.stat_line("No noise points found for comparison")
            diagnostic_result = {'no_noise': True}
        
        # 4. Analyze text vs embedding semantic gap
        self.verbose_reporter.step_start("Step 4: Analyzing semantic gap", "🕰️")
        
        # Sample document analysis
        if len(documents) >= 5:
            sample_indices = np.random.choice(len(documents), 5, replace=False)
            
            self.verbose_reporter.stat_line("Sample document analysis:")
            for i, idx in enumerate(sample_indices):
                doc = documents[idx]
                doc_preview = doc[:80] + "..." if len(doc) > 80 else doc
                
                # Analyze text characteristics
                word_count = len(doc.split())
                char_count = len(doc)
                
                self.verbose_reporter.stat_line(f"  {i+1}. '{doc_preview}'")
                self.verbose_reporter.stat_line(f"      Words: {word_count}, Chars: {char_count}, Cluster: {cluster_labels[idx]}")
        
        # Summary and recommendations
        self.verbose_reporter.step_start("Step 5: Summary and recommendations", "📜")
        
        recommendations = []
        
        if verification_result.get('matrices_identical', False):
            recommendations.append("✅ c-TF-IDF implementation is correct")
        else:
            recommendations.append("⚠️ c-TF-IDF implementation may have issues")
        
        # Check embedding properties
        avg_norm = np.mean(embedding_norms)
        if avg_norm < 0.5:
            recommendations.append("⚠️ Embeddings have low norms - may indicate scaling issues")
        elif avg_norm > 2.0:
            recommendations.append("⚠️ Embeddings have high norms - may need normalization")
        else:
            recommendations.append("✅ Embedding norms appear reasonable")
        
        # Check noise ratio
        noise_ratio = len(noise_indices) / len(cluster_labels) if cluster_labels else 0
        if noise_ratio > 0.5:
            recommendations.append(f"⚠️ High noise ratio ({noise_ratio:.1%}) suggests clustering issues")
        elif noise_ratio > 0.3:
            recommendations.append(f"⚠️ Moderate noise ratio ({noise_ratio:.1%}) - rescue methods helpful")
        else:
            recommendations.append(f"✅ Low noise ratio ({noise_ratio:.1%}) indicates good clustering")
        
        # Ensemble embedding considerations
        recommendations.append("💡 Ensemble embeddings (0.6 response + 0.3 question + 0.1 domain) may differ semantically from raw text")
        recommendations.append("💡 c-TF-IDF uses raw text while similarities use ensemble embeddings - this creates semantic gap")
        recommendations.append("💡 Consider adjusting similarity thresholds or using embedding-based rescue as primary method")
        
        self.verbose_reporter.sample_list("Diagnostic recommendations", recommendations)
        
        self.verbose_reporter.section_header("DIAGNOSTICS COMPLETE", "✅")
        
        return {
            'verification_result': verification_result,
            'diagnostic_result': diagnostic_result,
            'embedding_stats': {
                'shape': embeddings_array.shape,
                'mean': np.mean(embeddings_array),
                'std': np.std(embeddings_array),
                'avg_norm': np.mean(embedding_norms)
            },
            'noise_ratio': noise_ratio,
            'recommendations': recommendations
        }

    def rescue_noise_points(self) -> Dict[str, int]:
        """Rescue noise points using hybrid strategy: cosine similarity → c-TF-IDF"""
        if not self.config.noise_rescue.enabled:
            return {"rescued_count": 0, "total_noise": 0, "success_rate": 0.0}
            
        self.verbose_reporter.step_start("Hybrid noise rescue", "🚀")
        self.verbose_reporter.stat_line("Two-phase rescue strategy: Phase 1 (cosine similarity) → Phase 2 (c-TF-IDF)")
        
        # Get current cluster labels and embeddings based on embedding type
        if self.embedding_type == "code":
            labels = np.array([item.initial_code_cluster for item in self.output_list])
            embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
        else:
            labels = np.array([item.initial_description_cluster for item in self.output_list])
            embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
        
        # Find noise points
        initial_noise_mask = labels == -1
        total_initial_noise = initial_noise_mask.sum()
        
        if total_initial_noise == 0:
            self.verbose_reporter.stat_line("No noise points to rescue")
            self.verbose_reporter.step_complete("Noise rescue completed")
            return {"rescued_count": 0, "total_noise": 0, "success_rate": 1.0}
        
        self.verbose_reporter.stat_line(f"Initial noise points: {total_initial_noise}")
        
        # Limit rescue attempts for safety
        noise_indices = np.where(initial_noise_mask)[0]
        if len(noise_indices) > self.config.noise_rescue.max_rescue_attempts:
            self.verbose_reporter.stat_line(f"Limiting rescue to {self.config.noise_rescue.max_rescue_attempts} attempts")
            noise_indices = noise_indices[:self.config.noise_rescue.max_rescue_attempts]
        
        total_rescued = 0
        
        try:
            # Phase 1: Cosine similarity rescue (if enabled)
            cosine_rescued = 0
            if self.config.noise_rescue.use_cosine_rescue:
                self.verbose_reporter.stat_line("Phase 1: Cosine similarity rescue")
                cosine_rescued = self._cosine_similarity_rescue(noise_indices, embeddings, labels)
                total_rescued += cosine_rescued
                
                # Update labels after cosine rescue
                if self.embedding_type == "code":
                    labels = np.array([item.initial_code_cluster for item in self.output_list])
                else:
                    labels = np.array([item.initial_description_cluster for item in self.output_list])
            
            # Phase 2: c-TF-IDF rescue (if enabled and there are remaining noise points)
            ctfidf_rescued = 0
            if self.config.noise_rescue.use_ctfidf_rescue:
                # Find remaining noise points after cosine rescue
                remaining_noise_mask = labels == -1
                remaining_noise_count = remaining_noise_mask.sum()
                
                if remaining_noise_count > 0:
                    self.verbose_reporter.stat_line(f"Phase 2: c-TF-IDF rescue ({remaining_noise_count} remaining noise points after cosine rescue)")
                    ctfidf_rescued = self._ctfidf_rescue(labels)
                    total_rescued += ctfidf_rescued
                    self.verbose_reporter.stat_line(f"Phase 2 result: {ctfidf_rescued} additional points rescued via c-TF-IDF")
                else:
                    self.verbose_reporter.stat_line("Phase 2: No remaining noise points for c-TF-IDF rescue")
                
            # Report combined results
            success_rate = total_rescued / total_initial_noise if total_initial_noise > 0 else 0.0
            remaining_noise = total_initial_noise - total_rescued
            
            self.verbose_reporter.stat_line(f"")
            self.verbose_reporter.stat_line(f"=== HYBRID RESCUE SUMMARY ===")
            if self.config.noise_rescue.use_cosine_rescue:
                self.verbose_reporter.stat_line(f"Phase 1 (Cosine): {cosine_rescued} rescued")
            if self.config.noise_rescue.use_ctfidf_rescue:
                self.verbose_reporter.stat_line(f"Phase 2 (c-TF-IDF): {ctfidf_rescued} rescued")
            self.verbose_reporter.stat_line(f"Total rescued: {total_rescued}/{total_initial_noise} ({success_rate:.1%} success rate)")
            self.verbose_reporter.stat_line(f"Final noise: {remaining_noise} points")
            
            self.verbose_reporter.step_complete("Hybrid noise rescue completed")
            
            return {
                "rescued_count": total_rescued,
                "total_noise": total_initial_noise,
                "success_rate": success_rate,
                "cosine_rescued": cosine_rescued,
                "ctfidf_rescued": ctfidf_rescued
            }
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Hybrid rescue failed: {str(e)}")
            self.verbose_reporter.step_complete("Hybrid noise rescue failed")
            return {"rescued_count": 0, "total_noise": total_initial_noise, "success_rate": 0.0}

    def calculate_and_display_quality_metrics(self) -> Dict:
        """Calculate quality metrics for the clustering results - informational only"""
        self.verbose_reporter.step_start("Quality assessment", "📈")
        
        metrics = {}
        
        # Get embeddings and labels based on embedding type
        if self.embedding_type == "code":
            embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
            labels = np.array([item.initial_code_cluster for item in self.output_list])
        else:
            embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
            labels = np.array([item.initial_description_cluster for item in self.output_list])
        
        # Calculate quality metrics
        quality_analyzer = ClusterQualityAnalyzer(embeddings, labels)
        metrics = quality_analyzer.get_full_report()
        
        # Calculate overall quality score
        metrics['overall_quality'] = quality_analyzer.calculate_quality_score(metrics)
        
        # Display metrics using VerboseReporter
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted_metrics[key] = f"{value:.3f}"
            else:
                formatted_metrics[key] = str(value)
        
        self.verbose_reporter.summary("Clustering Quality Metrics", formatted_metrics, "📊")
        
        return metrics

    def filter_and_remap_clusters(self) -> None:
        """Filter out NA items and remap cluster IDs to be sequential"""
        self.verbose_reporter.step_start("Filtering NA items and remapping clusters", "🧹")
        
        # Get current clusters and items based on embedding type
        if self.embedding_type == "code":
            clusters = [item.initial_code_cluster for item in self.output_list]
            items = [item.segment_label for item in self.output_list]
        else:
            clusters = [item.initial_description_cluster for item in self.output_list]
            items = [item.segment_description for item in self.output_list]
        
        # Get IDs for filtering
        respondent_ids = [item.respondent_id for item in self.output_list]
        segment_ids = [item.segment_id for item in self.output_list]
        
        # Filter out outliers (cluster -1) and NA items
        filtered_respondent_ids = []
        filtered_segment_ids = []
        filtered_clusters = []
        filtered_items = []
        
        for r_id, s_id, cluster, item in zip(respondent_ids, segment_ids, clusters, items):
            if cluster != -1 and item.lower() != "na":
                filtered_respondent_ids.append(r_id)
                filtered_segment_ids.append(s_id)
                filtered_clusters.append(cluster)
                filtered_items.append(item)
        
        self.verbose_reporter.stat_line(f"Filtered out {len(clusters) - len(filtered_clusters)} items (noise or NA)")
        
        # Remap cluster IDs to be sequential
        unique_clusters = sorted(set(filtered_clusters))
        mapping = {original: new for new, original in enumerate(unique_clusters)}
        remapped_clusters = [mapping[cluster] for cluster in filtered_clusters]
        
        # Create mapping for updating output_list
        updated_clusters = {}
        for r_id, s_id, cluster in zip(filtered_respondent_ids, filtered_segment_ids, remapped_clusters):
            updated_clusters[(r_id, s_id)] = cluster
        
        # Update output_list with remapped clusters
        # Items not in filtered list will get None
        for item in self.output_list:
            key = (item.respondent_id, item.segment_id)
            if self.embedding_type == "code":
                # Store remapped cluster, or None if filtered out
                item.initial_code_cluster = updated_clusters.get(key, None)
            else:
                item.initial_description_cluster = updated_clusters.get(key, None)
        
        self.verbose_reporter.stat_line(f"Remapped {len(unique_clusters)} clusters to sequential IDs (0-{len(unique_clusters)-1})")
        self.verbose_reporter.step_complete("Filter and remap completed")

    def run_pipeline(self, embedding_type: str = None) -> None:
        
        if embedding_type:
            self.embedding_type = embedding_type
        
        # Start the main clustering pipeline
        self.verbose_reporter.section_header("CLUSTERING PHASE", "🔬")
        self.verbose_reporter.step_start(f"Clustering with embedding type: {self.embedding_type}", "🎯")
            
        if not self.output_list:
            raise ValueError("Output list is empty. Please populate it first.")
        
        # Step 1: Reduce dimensionality of embeddings
        self.add_reduced_embeddings()
        
        # Step 2: Perform initial clustering
        self.add_initial_clusters()
        
        # Step 2.5: Run similarity diagnostics (if in debug mode)
        if self.config.enable_similarity_diagnostics:
            self.diagnostic_results = self.run_similarity_diagnostics()
        
        # Step 2.6: Rescue noise points (if enabled)
        self.rescue_stats = self.rescue_noise_points()
        
        # Step 3: Calculate and display quality metrics (if enabled)
        if self.config.enable_quality_metrics:
            metrics = self.calculate_and_display_quality_metrics()
            
            # Show noise rescue impact if rescue was performed
            if self.rescue_stats["total_noise"] > 0:
                self.verbose_reporter.step_start("Noise rescue impact", "📈")
                original_noise_ratio = self.rescue_stats["total_noise"] / len(self.output_list)
                final_noise_ratio = metrics.get("noise_ratio", 0.0)
                noise_reduction = original_noise_ratio - final_noise_ratio
                
                self.verbose_reporter.stat_line(f"Original noise ratio: {original_noise_ratio:.1%}")
                self.verbose_reporter.stat_line(f"Final noise ratio: {final_noise_ratio:.1%}")
                self.verbose_reporter.stat_line(f"Noise reduction: {noise_reduction:.1%}")
                self.verbose_reporter.stat_line(f"Coverage improvement: {self.rescue_stats['rescued_count']} points rescued")
                self.verbose_reporter.step_complete("Noise rescue impact calculated")
        
        # Step 4: Filter NA items and remap clusters (if enabled)
        if self.config.filter_na_items or self.config.remap_cluster_ids:
            if self.config.filter_na_items and self.config.remap_cluster_ids:
                self.filter_and_remap_clusters()
            elif self.config.filter_na_items:
                # Just filter, don't remap
                pass  # TODO: Implement filter-only method if needed
            elif self.config.remap_cluster_ids:
                # Just remap, don't filter
                pass  # TODO: Implement remap-only method if needed
        
        # Final summary
        if self.embedding_type == "code":
            clusters = [item.initial_code_cluster for item in self.output_list if item.initial_code_cluster is not None]
        else:
            clusters = [item.initial_description_cluster for item in self.output_list if item.initial_description_cluster is not None]
        
        unique_clusters = set(clusters)
        
        # Create final summary
        summary_stats = {
            "Total items processed": len(self.output_list),
            "Items with valid clusters": len(clusters),
            "Clusters found": len(unique_clusters),
            "Items filtered out": len(self.output_list) - len(clusters)
        }
        
        # Add noise rescue information if rescue was performed
        if hasattr(self, 'rescue_stats') and self.rescue_stats["total_noise"] > 0:
            summary_stats["Original noise points"] = self.rescue_stats["total_noise"]
            summary_stats["Points rescued"] = self.rescue_stats["rescued_count"]
            summary_stats["Rescue success rate"] = f"{self.rescue_stats['success_rate']:.1%}"
        
        self.verbose_reporter.summary("Final Clustering Summary", summary_stats, "📊")
        self.verbose_reporter.step_complete("Clustering pipeline completed successfully")
            
    def to_cluster_model(self) -> List[models.ClusterModel]:
       
        if not self.output_list:
            raise ValueError("Output list is empty. Nothing to convert.")
            
        # Group output items by respondent_id
        items_by_respondent = defaultdict(list)
        for item in self.output_list:
            items_by_respondent[item.respondent_id].append(item)
        
        # Create mapping of respondent_id to original response data
        response_mapping = {}
        segment_mapping = {}
        for original_item in self.original_input_list:
            response_mapping[original_item.respondent_id] = original_item.response
            if original_item.response_segment:
                for segment in original_item.response_segment:
                    segment_key = (original_item.respondent_id, segment.segment_id)
                    segment_mapping[segment_key] = segment.segment_response
        
        # Create ClusterModel instances
        result_models = []
        
        for respondent_id, items in items_by_respondent.items():
            # Get the original response from mapping
            response = response_mapping.get(respondent_id, "")
            
            # Create submodels for each segment
            submodels = []
            for item in items:
                # Get original segment response
                segment_key = (respondent_id, item.segment_id)
                segment_response = segment_mapping.get(segment_key, "")
                
                # Store initial cluster ID
                initial_cluster = None
                if self.embedding_type == "code" and item.initial_code_cluster is not None:
                    initial_cluster = item.initial_code_cluster
                elif self.embedding_type == "description" and item.initial_description_cluster is not None:
                    initial_cluster = item.initial_description_cluster
                
                # Create submodel with embeddings and initial cluster
                submodel = models.ClusterSubmodel(
                    segment_id=item.segment_id,
                    segment_response=segment_response,
                    segment_label=item.segment_label,
                    segment_description=item.segment_description,
                    code_embedding=item.code_embedding,
                    description_embedding=item.description_embedding,
                    initial_cluster=initial_cluster  # Single cluster ID
                )
                
                submodels.append(submodel)
            
            # Create ClusterModel
            model = models.ClusterModel(
                respondent_id=respondent_id,
                response=response,
                response_segment=submodels
            )
            
            result_models.append(model)
        
        return result_models    
