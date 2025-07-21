import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List, Any, Optional, Tuple
import numpy.typing as npt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from dataclasses import dataclass

# config
import models
from config import DEFAULT_LANGUAGE, UMAPConfig, HDBSCANConfig, VectorizerConfig
from utils.verboseReporter import VerboseReporter
import warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")


@dataclass
class DimensionalityReductionStats:
    """Statistics from dimensionality reduction process"""
    original_dims: int
    pca_dims: int
    pca_variance_retained: float
    final_dims: int
    embeddings_shape: Tuple[int, int]
    

class ResultMapper(BaseModel):
    """Enhanced result mapper for tracking data through the pipeline"""
    respondent_id: Any
    idea_id: str
    idea: str
    idea_embedding: npt.NDArray[np.float32]
    pca_embedding: Optional[npt.NDArray[np.float32]] = None
    umap_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_idea_cluster: Optional[int] = None
    processing_order: Optional[int] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusterGenerator:
    def __init__(
        self,
        input_list: List[models.EmbeddingsModel] = None,
        var_lab=None,
        variance_threshold: float = 0.90,  # 90% variance retention for PCA
        dim_reduction_model=None,
        cluster_model=None,
        vectorizer_model=None,
        verbose: bool = None):
        
        self.UMAPConfig = UMAPConfig
        self.HDBSCANConfig = HDBSCANConfig
        self.VectorizerConfig = VectorizerConfig
        self.verbose = verbose 
        self.verbose_reporter = VerboseReporter(self.verbose)
        
        # Initialize principal components
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized dynamically
        self.optimal_pca_dims = None
        self.dr_stats = None
        self.variance_threshold = variance_threshold
     
        # Data structures
        self.output_list: List[ResultMapper] = []
        self._original_input_list = None
        
        if input_list:
            self.populate_from_input_list(input_list)
        
        self.var_lab = var_lab if var_lab else ""
        
        # Initialize models
        self._initialize_models(dim_reduction_model, cluster_model, vectorizer_model)
    
    def _initialize_models(self, dim_reduction_model, cluster_model, vectorizer_model):
        """Initialize clustering and vectorizer models"""

        if dim_reduction_model is None:
            umap_config = self.UMAPConfig
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
        

        if cluster_model is None:
            hdbscan_config = self.HDBSCANConfig 
            
            # Handle None values for unconstrained clustering
            hdbscan_params = {
                'metric': hdbscan_config.metric,
                'cluster_selection_method': hdbscan_config.cluster_selection_method,  
                'prediction_data': hdbscan_config.prediction_data,
                'approx_min_span_tree': hdbscan_config.approx_min_span_tree,   
                'gen_min_span_tree': hdbscan_config.gen_min_span_tree
            }
            
            # Only add parameters if they are not None (let HDBSCAN use defaults)
            if hdbscan_config.min_cluster_size is not None:
                hdbscan_params['min_cluster_size'] = hdbscan_config.min_cluster_size
            if hdbscan_config.min_samples is not None:
                hdbscan_params['min_samples'] = hdbscan_config.min_samples
                
            self.cluster_model = hdbscan.HDBSCAN(**hdbscan_params)
            min_cluster_display = hdbscan_config.min_cluster_size if hdbscan_config.min_cluster_size is not None else "default"
            min_samples_display = hdbscan_config.min_samples if hdbscan_config.min_samples is not None else "default"
            self.verbose_reporter.stat_line(
                f"Using HDBSCAN with min_cluster_size={min_cluster_display}, "
                f"min_samples={min_samples_display} (unconstrained clustering)"
            )
        else:
            self.cluster_model = cluster_model
        
     
        if vectorizer_model is None:
            vectorizer_config = self.VectorizerConfig
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
    
    def populate_from_input_list(self, input_list: List[models.EmbeddingsModel]) -> None:
        """Populate output list from input models"""
        self.verbose_reporter.stat_line("Populating output list from input models")
        
        self._original_input_list = input_list
        self.output_list = []
        processing_order = 0
        
        for respondent_item in input_list:
            if respondent_item.response_ideas:
                for response_item in respondent_item.response_ideas:
                    if response_item.idea_embedding is None:
                        self.verbose_reporter.stat_line(
                            f"Warning: Missing embedding for respondent {respondent_item.respondent_id}, "
                            f"idea {response_item.idea_id}. Skipping."
                        )
                        continue
                    
                    result = ResultMapper(
                        respondent_id=respondent_item.respondent_id,
                        idea_id=response_item.idea_id,
                        idea=response_item.idea or "NA",
                        idea_embedding=response_item.idea_embedding,
                        processing_order=processing_order
                    )
                    
                    self.output_list.append(result)
                    processing_order += 1
        
        # Validation: Check processing order assignment
        processing_orders = [item.processing_order for item in self.output_list]
        expected_orders = list(range(len(self.output_list)))
        if processing_orders != expected_orders:
            self.verbose_reporter.stat_line("⚠️  WARNING: Processing orders are not sequential!")
            self.verbose_reporter.stat_line(f"Expected: {expected_orders[:10]}... Got: {processing_orders[:10]}...")
        
        self.verbose_reporter.stat_line(f"Populated {len(self.output_list)} items successfully")
    
    def _find_optimal_pca_dimensions(self, embeddings_array: np.ndarray) -> Tuple[int, float]:
        """
        Find the optimal number of PCA dimensions that retain the specified variance threshold.
        Now uses pure 90% variance retention without constraints.
        Returns: (optimal_dimensions, actual_variance_retained)
        """
        self.verbose_reporter.step_start(
            f"Finding optimal PCA dimensions for {self.variance_threshold*100}% variance retention", "🔍"
        )
        
        # Standardize the embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings_array)
        
        # Fit full PCA to analyze variance
        n_samples, n_features = embeddings_array.shape
        max_components = min(n_samples - 1, n_features)
        
        pca_full = PCA(n_components=max_components)
        pca_full.fit(embeddings_scaled)
        
        # Find number of components for desired variance
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Find where we hit the variance threshold (90%)
        optimal_dims = np.argmax(cumsum_variance >= self.variance_threshold) + 1
        
        # Ensure we don't exceed available components
        optimal_dims = min(optimal_dims, max_components)
        
        actual_variance = cumsum_variance[optimal_dims - 1] if optimal_dims <= len(cumsum_variance) else cumsum_variance[-1]
        
        self.verbose_reporter.stat_line(
            f"Optimal PCA dimensions: {optimal_dims} (retaining {actual_variance*100:.2f}% variance)"
        )
        
        return optimal_dims, actual_variance
  
    def add_pca_embeddings(self) -> None:
        """Apply PCA with dynamically determined dimensions"""
        self.verbose_reporter.step_start("Applying PCA dimensionality reduction", "📊")
        
        # Sort by processing order
        sorted_items = sorted(self.output_list, key=lambda x: x.processing_order)
        
        # Create embedding array
        idea_embeddings_array = np.array([item.idea_embedding for item in sorted_items])
        original_shape = idea_embeddings_array.shape
        
        # Find optimal PCA dimensions
        self.optimal_pca_dims, variance_retained = self._find_optimal_pca_dimensions(idea_embeddings_array)
        
        # Apply PCA with optimal dimensions
        embeddings_scaled = self.scaler.transform(idea_embeddings_array)
        self.pca = PCA(n_components=self.optimal_pca_dims, random_state=42)
        pca_embeddings = self.pca.fit_transform(embeddings_scaled)
        
        # Assign PCA embeddings
        for i, item in enumerate(sorted_items):
            # Find the item in the original list and update it
            for orig_item in self.output_list:
                if orig_item.processing_order == item.processing_order:
                    orig_item.pca_embedding = pca_embeddings[i]
                    break
        
        # Store statistics
        self.dr_stats = DimensionalityReductionStats(
            original_dims=original_shape[1],
            pca_dims=self.optimal_pca_dims,
            pca_variance_retained=variance_retained,
            final_dims=self.UMAPConfig.n_components,
            embeddings_shape=pca_embeddings.shape
        )
        
        self.verbose_reporter.stat_line(
            f"PCA complete: {original_shape[1]} → {self.optimal_pca_dims} dimensions"
        )
        self.verbose_reporter.step_complete("PCA dimensionality reduction completed")
    
    def add_umap_embeddings(self) -> None:
        """Apply UMAP to PCA-reduced embeddings to preserve global structure"""
        self.verbose_reporter.step_start(
            f"Applying UMAP: {self.optimal_pca_dims} → {self.UMAPConfig.n_components} dimensions", "🗺️"
        )
        
        # Sort by processing order
        sorted_items = sorted(self.output_list, key=lambda x: x.processing_order)
        
        # Create PCA embedding array
        pca_embeddings_array = np.array([item.pca_embedding for item in sorted_items])
        
        # Apply UMAP
        umap_embeddings = self.dim_reduction_model.fit_transform(pca_embeddings_array)
        
        # Assign UMAP embeddings
        for i, item in enumerate(sorted_items):
            # Find the item in the original list and update it
            for orig_item in self.output_list:
                if orig_item.processing_order == item.processing_order:
                    orig_item.umap_embedding = umap_embeddings[i]
                    break
        
        self.verbose_reporter.stat_line(
            f"UMAP complete: {self.optimal_pca_dims} → {self.UMAPConfig.n_components} dimensions"
        )
        self.verbose_reporter.step_complete("UMAP dimensionality reduction completed")
   
    def add_initial_clusters(self) -> None:
        """Cluster the UMAP embeddings using HDBSCAN"""
        self.verbose_reporter.step_start("Clustering with HDBSCAN (most permissive settings)", "🔍")
        
        # Sort by processing order
        sorted_items = sorted(self.output_list, key=lambda x: x.processing_order)
        
        # Create UMAP embedding array
        umap_embeddings_array = np.array([item.umap_embedding for item in sorted_items])
        
        # Perform clustering
        initial_idea_clusters = self.cluster_model.fit_predict(umap_embeddings_array)
        
        # Assign clusters
        assignment_count = 0
        for i, item in enumerate(sorted_items):
            # Find the item in the original list and update it
            found = False
            for orig_item in self.output_list:
                if orig_item.processing_order == item.processing_order:
                    orig_item.initial_idea_cluster = initial_idea_clusters[i]
                    assignment_count += 1
                    found = True
                    break
            if not found:
                self.verbose_reporter.stat_line(
                    f"⚠️  WARNING: Could not find original item for processing_order {item.processing_order}"
                )
        
        # Validation: Check for processing order consistency
        processing_orders = [item.processing_order for item in self.output_list]
        unique_orders = set(processing_orders)
        if len(unique_orders) != len(processing_orders):
            self.verbose_reporter.stat_line("⚠️  WARNING: Duplicate processing orders detected!")
            from collections import Counter
            duplicates = {k: v for k, v in Counter(processing_orders).items() if v > 1}
            self.verbose_reporter.stat_line(f"Processing order duplicates: {duplicates}")
        
        self.verbose_reporter.stat_line(f"Cluster assignments completed: {assignment_count}/{len(sorted_items)}")
        
        # Report statistics
        cluster_counts = Counter(initial_idea_clusters)
        n_clusters = len([c for c in cluster_counts if c != -1])  # Exclude noise
        
        self.verbose_reporter.stat_line(f"Clusters found: {n_clusters} (excluding noise)")
        
        # Report cluster sizes
        noise_count = cluster_counts.get(-1, 0)
        if noise_count > 0:
            self.verbose_reporter.stat_line(f"Noise points (-1): {noise_count} items ({noise_count/len(initial_idea_clusters)*100:.1f}%)")
        
        # Report size distribution for valid clusters
        valid_clusters = [c for c in cluster_counts if c != -1]
        if valid_clusters:
            sizes = [cluster_counts[c] for c in valid_clusters]
            self.verbose_reporter.stat_line(
                f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
                f"median={np.median(sizes):.1f}, mean={np.mean(sizes):.1f}"
            )
            
            # Report cluster persistence scores if available
            if hasattr(self.cluster_model, 'cluster_persistence_'):
                persistence = self.cluster_model.cluster_persistence_
                self.verbose_reporter.stat_line(
                    f"Cluster persistence: min={min(persistence):.4f}, "
                    f"max={max(persistence):.4f}, mean={np.mean(persistence):.4f}"
                )
              
        self.verbose_reporter.step_complete("HDBSCAN clustering completed")
    
    def run_pipeline(self) -> None:
        """Run the complete PCA + UMAP + HDBSCAN clustering pipeline"""
        self.verbose_reporter.section_header("CLUSTERING PHASE (PCA + UMAP + HDBSCAN)", "🔬")
        
        try:
            # Step 1: PCA for initial dimensionality reduction
            self.add_pca_embeddings()
            
            # Step 2: UMAP for final dimensionality reduction
            self.add_umap_embeddings()
            
            # Step 3: HDBSCAN clustering
            self.add_initial_clusters()
            
            # Final summary
            self._create_final_summary()
            
            self.verbose_reporter.step_complete("Clustering pipeline completed successfully")
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Pipeline failed: {str(e)}")
            raise
    
    def _create_final_summary(self) -> None:
        """Create and display final summary statistics"""
        clusters = [item.initial_idea_cluster for item in self.output_list]
        unique_clusters = [c for c in set(clusters) if c != -1]  # Exclude noise
        noise_count = clusters.count(-1)
        
        summary_stats = {
            "Total items processed": len(self.output_list),
            "Clusters found": len(unique_clusters),
            "Noise points": f"{noise_count} ({noise_count/len(clusters)*100:.1f}%)",
            "Clustering method": f"HDBSCAN (min_cluster_size={self.HDBSCANConfig.min_cluster_size or 'default'}, min_samples={self.HDBSCANConfig.min_samples or 'default'})",
            "Dimensionality reduction": f"{self.dr_stats.original_dims} → {self.dr_stats.pca_dims} (PCA) → {self.dr_stats.final_dims} (UMAP)",
            "PCA variance retained": f"{self.dr_stats.pca_variance_retained*100:.2f}%"
        }
        
        self.verbose_reporter.summary("Final Clustering Summary", summary_stats, "📊")
    
    def to_cluster_model(self) -> List[models.ClusterModel]:
        """Convert results to ClusterModel format"""
        self.verbose_reporter.step_start("Converting results to ClusterModel format", "🔄")
        
        # Group results by respondent_id
        respondent_groups = {}
        for item in self.output_list:
            if item.respondent_id not in respondent_groups:
                respondent_groups[item.respondent_id] = []
            respondent_groups[item.respondent_id].append(item)
        
        # Create ClusterModel objects
        cluster_models = []
        
        for respondent_id, items in respondent_groups.items():
            # Sort items by processing order
            items.sort(key=lambda x: x.processing_order)
            
            # Find original model
            original_model = None
            if hasattr(self, '_original_input_list'):
                for model in self._original_input_list:
                    if model.respondent_id == respondent_id:
                        original_model = model
                        break
            
            # Create ClusterSubmodel objects
            cluster_submodels = []
            for item in items:
                cluster_submodel = models.ClusterSubmodel(
                    idea_id=item.idea_id,
                    idea=item.idea,
                    idea_embedding=item.idea_embedding,
                    initial_cluster=item.initial_idea_cluster
                )
                cluster_submodels.append(cluster_submodel)
            
            # Create ClusterModel
            if original_model:
                cluster_model = models.ClusterModel(
                    **original_model.model_dump(exclude={'response_ideas', 'idea_embeddings'}),
                    response_ideas=cluster_submodels,
                    idea_embeddings=cluster_submodels
                )
            else:
                cluster_model = models.ClusterModel(
                    respondent_id=respondent_id,
                    response_ideas=cluster_submodels,
                    idea_embeddings=cluster_submodels,
                    idea_count=len(cluster_submodels)
                )
            
            cluster_models.append(cluster_model)
        
        self.verbose_reporter.step_complete(
            f"Converted {len(cluster_models)} respondents to ClusterModel format"
        )
        
        return cluster_models
  