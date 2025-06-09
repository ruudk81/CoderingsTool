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
            desc = cluster_items[0].segment_description
            return desc[:max_length] + "..." if len(desc) > max_length else desc
        
        # For multiple items, find most representative
        descriptions = [item.segment_description for item in cluster_items]
        embeddings = [item.description_embedding for item in cluster_items 
                     if item.description_embedding is not None]
        
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

    def rescue_noise_points(self) -> Dict[str, int]:
        """Rescue noise points using cosine similarity or HDBSCAN methods"""
        if not self.config.noise_rescue.enabled:
            return {"rescued_count": 0, "total_noise": 0, "success_rate": 0.0}
            
        self.verbose_reporter.step_start("Noise rescue", "🚀")
        
        # Get current cluster labels and embeddings based on embedding type
        if self.embedding_type == "code":
            labels = np.array([item.initial_code_cluster for item in self.output_list])
            embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
        else:
            labels = np.array([item.initial_description_cluster for item in self.output_list])
            embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
        
        # Find noise points
        noise_mask = labels == -1
        total_noise = noise_mask.sum()
        
        if total_noise == 0:
            self.verbose_reporter.stat_line("No noise points to rescue")
            self.verbose_reporter.step_complete("Noise rescue completed")
            return {"rescued_count": 0, "total_noise": 0, "success_rate": 1.0}
        
        self.verbose_reporter.stat_line(f"Noise before: n={total_noise} noise points")
        
        # Limit rescue attempts for safety
        noise_indices = np.where(noise_mask)[0]
        if len(noise_indices) > self.config.noise_rescue.max_rescue_attempts:
            self.verbose_reporter.stat_line(f"Limiting rescue to {self.config.noise_rescue.max_rescue_attempts} attempts")
            noise_indices = noise_indices[:self.config.noise_rescue.max_rescue_attempts]
            noise_mask = np.zeros_like(labels, dtype=bool)
            noise_mask[noise_indices] = True
        
        try:
            # Choose rescue method based on configuration
            if self.config.noise_rescue.use_cosine_rescue:
                # Use cosine similarity rescue
                self.verbose_reporter.stat_line("Using cosine similarity rescue method")
                rescued_count = self._cosine_similarity_rescue(noise_indices, embeddings, labels)
                
            else:
                # Use existing HDBSCAN methods
                self.verbose_reporter.stat_line("Using HDBSCAN rescue methods")
                
                # Debug: Check if clusterer has prediction data
                if not hasattr(self.cluster_model, 'prediction_data_') or self.cluster_model.prediction_data_ is None:
                    self.verbose_reporter.stat_line("⚠️  Warning: HDBSCAN clusterer has no prediction data")
                    self.verbose_reporter.stat_line("This might be because prediction_data=True wasn't set during fit")
                else:
                    self.verbose_reporter.stat_line("✅ HDBSCAN has prediction data available")
                    
                # Try using membership_vector approach as alternative
                try:
                    # Get membership vectors for noise points
                    test_membership_vectors = hdbscan.membership_vector(self.cluster_model, embeddings[noise_mask])
                    
                    # Find best cluster for each point based on membership strength
                    best_clusters = np.argmax(test_membership_vectors, axis=1)
                    membership_strengths = np.max(test_membership_vectors, axis=1)
                    
                    self.verbose_reporter.stat_line(f"membership_vector approach - Max strength: {np.max(membership_strengths):.3f}")
                    self.verbose_reporter.stat_line(f"membership_vector approach - Mean strength: {np.mean(membership_strengths):.3f}")
                    
                    # Use membership vector results instead of approximate_predict
                    rescued_clusters = best_clusters
                    strengths = membership_strengths
                    
                except Exception as e:
                    self.verbose_reporter.stat_line(f"membership_vector failed: {e}")
                    # Fall back to approximate_predict
                    rescued_clusters, strengths = hdbscan.approximate_predict(
                        self.cluster_model, 
                        embeddings[noise_mask]
                    )
                
                # Debug: Check what we got
                self.verbose_reporter.stat_line(f"Final method returned {len(rescued_clusters)} cluster predictions")
                unique_predictions = np.unique(rescued_clusters)
                self.verbose_reporter.stat_line(f"Unique predicted clusters: {unique_predictions[:10]}...")  # Show first 10
                
                # Apply threshold and update cluster assignments
                rescued_count = 0
                threshold = self.config.noise_rescue.rescue_threshold
                
                # Show confidence distribution for debugging
                if len(strengths) > 0:
                    max_conf = np.max(strengths)
                    min_conf = np.min(strengths)
                    mean_conf = np.mean(strengths)
                    above_threshold = np.sum(strengths > threshold)
                    
                    self.verbose_reporter.stat_line(f"Confidence scores - Min: {min_conf:.3f}, Max: {max_conf:.3f}, Mean: {mean_conf:.3f}")
                    self.verbose_reporter.stat_line(f"Points above threshold ({threshold}): {above_threshold}/{len(strengths)}")
                
                for i, noise_idx in enumerate(noise_indices):
                    if strengths[i] > threshold:
                        # Update the appropriate cluster field
                        if self.embedding_type == "code":
                            self.output_list[noise_idx].initial_code_cluster = rescued_clusters[i]
                        else:
                            self.output_list[noise_idx].initial_description_cluster = rescued_clusters[i]
                        rescued_count += 1
            
            success_rate = rescued_count / total_noise if total_noise > 0 else 0.0
            remaining_noise = total_noise - rescued_count
            
            self.verbose_reporter.stat_line(f"Rescued {rescued_count}/{total_noise} noise points ({success_rate:.1%} success rate)")
            self.verbose_reporter.stat_line(f"Noise after: n={remaining_noise} noise points")
            
            # Only show threshold info for HDBSCAN methods
            if not self.config.noise_rescue.use_cosine_rescue:
                self.verbose_reporter.stat_line(f"Used confidence threshold: {threshold}")
            
            # Suggest threshold adjustment if no rescues but there are predictions (HDBSCAN methods only)
            if not self.config.noise_rescue.use_cosine_rescue and rescued_count == 0 and len(strengths) > 0 and max_conf > 0:
                suggested_threshold = max(0.1, max_conf * 0.8)  # 80% of max confidence, minimum 0.1
                self.verbose_reporter.stat_line(f"💡 Suggestion: Try lowering threshold to {suggested_threshold:.2f} to rescue {np.sum(strengths > suggested_threshold)} points")
            
            # Show examples only for HDBSCAN methods (which have confidence scores)
            if not self.config.noise_rescue.use_cosine_rescue:
                if rescued_count > 0:
                    examples = []
                    example_count = 0
                    for i, noise_idx in enumerate(noise_indices):
                        if strengths[i] > threshold and example_count < 3:
                            if self.embedding_type == "code":
                                item_text = self.output_list[noise_idx].segment_label or "N/A"
                                cluster_id = self.output_list[noise_idx].initial_code_cluster
                            else:
                                item_text = self.output_list[noise_idx].segment_description or "N/A"
                                cluster_id = self.output_list[noise_idx].initial_description_cluster
                            examples.append(f"→ Cluster {cluster_id} (conf: {strengths[i]:.2f}): {item_text[:60]}...")
                            example_count += 1
                    
                    if examples:
                        self.verbose_reporter.sample_list("Rescued examples", examples)
                else:
                    # Show highest confidence candidates that weren't rescued
                    if len(strengths) > 0:
                        top_indices = np.argsort(strengths)[-3:][::-1]  # Top 3 confidence scores
                        examples = []
                        for idx in top_indices:
                            noise_idx = noise_indices[idx]
                            if self.embedding_type == "code":
                                item_text = self.output_list[noise_idx].segment_label or "N/A"
                            else:
                                item_text = self.output_list[noise_idx].segment_description or "N/A"
                            examples.append(f"→ Cluster {rescued_clusters[idx]} (conf: {strengths[idx]:.2f}): {item_text[:60]}...")
                        
                        if examples:
                            self.verbose_reporter.sample_list("Highest confidence candidates (not rescued)", examples)
            
            self.verbose_reporter.step_complete("Noise rescue completed")
            
            return {
                "rescued_count": rescued_count,
                "total_noise": total_noise,
                "success_rate": success_rate
            }
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Rescue failed: {str(e)}")
            self.verbose_reporter.step_complete("Noise rescue failed")
            return {"rescued_count": 0, "total_noise": total_noise, "success_rate": 0.0}

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
        
        # Step 2.5: Rescue noise points (if enabled)
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


# Example usage and testing
if __name__ == "__main__":
    """Test the simple clusterer with actual embeddings"""
    from utils.cacheManager import CacheManager
    from config import CacheConfig
    import models
    import dataLoader
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load embeddings from cache
    input_list = cache_manager.load_from_cache(filename, "embeddings", models.EmbeddingsModel)
    
    if input_list:
        print(f"Loaded {len(input_list)} embeddings from cache")
        
        # Get variable label
        data_loader = dataLoader.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        
        print("\n=== Running simple clustering ===")
        clusterer = ClusterGenerator(
            input_list=input_list,
            var_lab=var_lab,
            embedding_type="description",  # or "description"
            verbose=True
            )
        
        clusterer.run_pipeline()
        cluster_results = clusterer.to_cluster_model()
        
        for result in cluster_results:
            print(result)
            break
        
        # Save to cache
        cache_manager.save_to_cache(cluster_results, filename, 'clusters')
        print(f"Saved {len(cluster_results)} cluster results to cache")
        
        # Print cluster summary
        cluster_counts = defaultdict(int)
        for response_items in cluster_results:
            for segment_items in response_items.response_segment:
                if segment_items.micro_cluster is not None:
                    cluster_id = list(segment_items.micro_cluster.keys())[0]
                    cluster_counts[cluster_id] += 1
    
        print(f"\nFound {len(cluster_counts)} unique clusters")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"  Cluster {cluster_id}: {count} items")
        
    else:
        print("No cached embeddings found. Please run the pipeline first.")