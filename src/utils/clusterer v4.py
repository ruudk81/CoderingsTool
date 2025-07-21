import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Any, Optional, Tuple, Dict
import numpy.typing as npt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from umap import UMAP
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.cluster.hierarchy as sch

# config
import models
from config import DEFAULT_LANGUAGE, ClusteringConfig, DEFAULT_CLUSTERING_CONFIG
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
    
    # Dimensionality reduction stages
    pca_embedding: Optional[npt.NDArray[np.float32]] = None
    umap_embedding: Optional[npt.NDArray[np.float32]] = None
    
    # Clustering result
    initial_idea_cluster: Optional[int] = None
    
    # Metadata
    processing_order: Optional[int] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusterGenerator:
    def __init__(
        self,
        input_list: List[models.EmbeddingsModel] = None,
        var_lab=None,
        variance_threshold: float = 0.90,  # 90% variance retention for PCA
        umap_n_components: int = 5,  # Final UMAP dimensions
        n_clusters: Optional[int] = None,  # Number of clusters for agglomerative clustering
        max_clusters_to_try: int = 50,  # Maximum clusters to try if n_clusters not specified
        linkage: str = 'ward',  # Linkage criterion for agglomerative clustering
        dim_reduction_model=None,
        cluster_model=None,
        vectorizer_model=None,
        config: ClusteringConfig = None,
        verbose: bool = None):
        
        # Initialize config and verbose settings
        self.config = config or DEFAULT_CLUSTERING_CONFIG
        self.verbose = verbose if verbose is not None else self.config.verbose
        self.verbose_reporter = VerboseReporter(self.verbose)
        
        # Dimensionality reduction parameters
        self.variance_threshold = variance_threshold
        self.umap_n_components = umap_n_components
        
        # Clustering parameters
        self.n_clusters = n_clusters
        self.max_clusters_to_try = max_clusters_to_try
        self.linkage = linkage
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized dynamically
        self.optimal_pca_dims = None
        self.dr_stats = None
        self._linkage_matrix = None  # Store linkage matrix for reuse
        self._inconsistency_threshold = None  # Store threshold for reuse
        
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
        # UMAP model (will use after PCA)
        if dim_reduction_model is None:
            # Use UMAP settings that preserve global structure
            self.dim_reduction_model = UMAP(
                n_neighbors=15,  # Default for preserving global structure
                n_components=self.umap_n_components,
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
                n_jobs=-1,
                low_memory=False,
                transform_seed=42
            )
            self.verbose_reporter.stat_line(f"Using UMAP for final reduction to {self.umap_n_components} dimensions")
        else:
            self.dim_reduction_model = dim_reduction_model
        
        # Agglomerative clustering (replacing HDBSCAN)
        if cluster_model is None:
            if self.n_clusters is not None:
                self.cluster_model = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    linkage=self.linkage,
                    metric='euclidean'
                )
                self.verbose_reporter.stat_line(f"Using Agglomerative clustering with {self.n_clusters} clusters")
            else:
                # Will determine optimal clusters dynamically
                self.cluster_model = None
                self.verbose_reporter.stat_line("Will determine optimal number of clusters dynamically")
        else:
            self.cluster_model = cluster_model
        
        # Vectorizer
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
        
        # Optional: Plot variance explained
        if self.verbose and len(cumsum_variance) > 10:
            self._plot_variance_explained(cumsum_variance, optimal_dims)
        
        return optimal_dims, actual_variance
    
    def _plot_variance_explained(self, cumsum_variance: np.ndarray, optimal_dims: int):
        """Plot cumulative variance explained by PCA components"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'b-', linewidth=2)
            plt.axhline(y=self.variance_threshold, color='r', linestyle='--', 
                       label=f'{self.variance_threshold*100}% threshold')
            plt.axvline(x=optimal_dims, color='g', linestyle='--', 
                       label=f'Optimal dims: {optimal_dims}')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.title('PCA Variance Explained')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('pca_variance_explained.png', dpi=150)
            plt.close()
            self.verbose_reporter.stat_line("Saved PCA variance plot to pca_variance_explained.png")
        except Exception as e:
            self.verbose_reporter.stat_line(f"Failed to create variance plot: {str(e)}")
    
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
            final_dims=self.umap_n_components,
            embeddings_shape=pca_embeddings.shape
        )
        
        self.verbose_reporter.stat_line(
            f"PCA complete: {original_shape[1]} → {self.optimal_pca_dims} dimensions"
        )
        self.verbose_reporter.step_complete("PCA dimensionality reduction completed")
    
    def add_umap_embeddings(self) -> None:
        """Apply UMAP to PCA-reduced embeddings to preserve global structure"""
        self.verbose_reporter.step_start(
            f"Applying UMAP: {self.optimal_pca_dims} → {self.umap_n_components} dimensions", "🗺️"
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
            f"UMAP complete: {self.optimal_pca_dims} → {self.umap_n_components} dimensions"
        )
        self.verbose_reporter.step_complete("UMAP dimensionality reduction completed")
    
    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using inconsistency method"""
        self.verbose_reporter.step_start("Finding optimal number of clusters using inconsistency method", "🔍")
        
        # Perform hierarchical clustering
        linkage_matrix = sch.linkage(embeddings, method=self.linkage, metric='euclidean')
        
        # Calculate inconsistency for different depths
        depths = range(2, min(10, len(embeddings) // 10))  # Test different depths
        inconsistency_stats = []
        
        for depth in depths:
            incons = sch.inconsistent(linkage_matrix, d=depth)
            # Get the mean of the inconsistency index (4th column)
            mean_incons = np.mean(incons[:, 3])
            inconsistency_stats.append(mean_incons)
            
            if self.verbose:
                self.verbose_reporter.stat_line(f"Depth {depth}: mean inconsistency = {mean_incons:.4f}")
        
        # Find the elbow in inconsistency values
        # Look for the maximum rate of change
        if len(inconsistency_stats) > 2:
            # Calculate second derivative to find elbow
            first_deriv = np.diff(inconsistency_stats)
            second_deriv = np.diff(first_deriv)
            
            # Find the depth with maximum change in rate (elbow)
            elbow_idx = np.argmax(np.abs(second_deriv)) + 2  # +2 because of double diff
            optimal_depth = depths[min(elbow_idx, len(depths) - 1)]
        else:
            optimal_depth = depths[0]
        
        # Use inconsistency threshold based on optimal depth
        incons = sch.inconsistent(linkage_matrix, d=optimal_depth)
        
        # Set threshold as mean + 1.5 * std of inconsistency values
        threshold = np.mean(incons[:, 3]) + 1.5 * np.std(incons[:, 3])
        
        # Find clusters using fcluster with inconsistency criterion
        clusters = sch.fcluster(linkage_matrix, threshold, criterion='inconsistent')
        optimal_k = len(np.unique(clusters))
        
        self.verbose_reporter.stat_line(
            f"Optimal depth: {optimal_depth}, threshold: {threshold:.4f}"
        )
        self.verbose_reporter.stat_line(
            f"Optimal clusters: {optimal_k} (using inconsistency method)"
        )
        
        # Store linkage matrix for later use
        self._linkage_matrix = linkage_matrix
        self._inconsistency_threshold = threshold
        
        # Optional: Plot inconsistency analysis
        if self.verbose:
            self._plot_inconsistency_analysis(depths, inconsistency_stats, optimal_depth)
        
        return optimal_k
    
    def _plot_inconsistency_analysis(self, depths: range, inconsistency_stats: List[float], optimal_depth: int):
        """Plot inconsistency analysis for different depths"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot 1: Inconsistency values
            ax1.plot(list(depths), inconsistency_stats, 'b-', linewidth=2, marker='o')
            ax1.axvline(x=optimal_depth, color='r', linestyle='--', 
                       label=f'Optimal depth: {optimal_depth}')
            ax1.set_xlabel('Depth')
            ax1.set_ylabel('Mean Inconsistency')
            ax1.set_title('Mean Inconsistency vs Depth')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Rate of change (to show elbow)
            if len(inconsistency_stats) > 1:
                rates = np.diff(inconsistency_stats)
                ax2.plot(list(depths)[1:], rates, 'g-', linewidth=2, marker='s')
                ax2.set_xlabel('Depth')
                ax2.set_ylabel('Rate of Change')
                ax2.set_title('Rate of Change in Inconsistency')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('inconsistency_analysis.png', dpi=150)
            plt.close()
            self.verbose_reporter.stat_line("Saved inconsistency analysis plot to inconsistency_analysis.png")
        except Exception as e:
            self.verbose_reporter.stat_line(f"Failed to create inconsistency plot: {str(e)}")
    
    def add_initial_clusters(self) -> None:
        """Cluster the UMAP embeddings using agglomerative clustering"""
        self.verbose_reporter.step_start("Applying agglomerative clustering", "🔍")
        
        # Sort by processing order
        sorted_items = sorted(self.output_list, key=lambda x: x.processing_order)
        
        # Create UMAP embedding array
        umap_embeddings_array = np.array([item.umap_embedding for item in sorted_items])
        
        # Determine number of clusters if not specified
        if self.cluster_model is None:
            optimal_clusters = self._find_optimal_clusters(umap_embeddings_array)
            self.cluster_model = AgglomerativeClustering(
                n_clusters=optimal_clusters,
                linkage=self.linkage,
                metric='euclidean'
            )
        
        # Perform clustering
        initial_idea_clusters = self.cluster_model.fit_predict(umap_embeddings_array)
        
        # Assign clusters
        for i, item in enumerate(sorted_items):
            # Find the item in the original list and update it
            for orig_item in self.output_list:
                if orig_item.processing_order == item.processing_order:
                    orig_item.initial_idea_cluster = initial_idea_clusters[i]
                    break
        
        # Report statistics
        cluster_counts = Counter(initial_idea_clusters)
        self.verbose_reporter.stat_line(f"Clusters: {len(set(initial_idea_clusters))} clusters found")
        
        # Report size distribution
        sizes = list(cluster_counts.values())
        self.verbose_reporter.stat_line(
            f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
            f"median={np.median(sizes):.1f}, mean={np.mean(sizes):.1f}"
        )
        
        # Optional: Create dendrogram for hierarchical clustering
        if self.verbose and len(umap_embeddings_array) < 500:  # Only for smaller datasets
            self._create_dendrogram(umap_embeddings_array)
        
        self.verbose_reporter.step_complete("Agglomerative clustering completed")
    
    def _create_dendrogram(self, embeddings: np.ndarray):
        """Create a dendrogram visualization of the hierarchical clustering"""
        try:
            plt.figure(figsize=(15, 8))
            dendrogram = sch.dendrogram(
                sch.linkage(embeddings, method=self.linkage),
                no_labels=True,
                color_threshold=0
            )
            plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage} linkage)')
            plt.xlabel('Sample Index')
            plt.ylabel('Distance')
            plt.tight_layout()
            plt.savefig('clustering_dendrogram.png', dpi=150)
            plt.close()
            self.verbose_reporter.stat_line("Saved dendrogram to clustering_dendrogram.png")
        except Exception as e:
            self.verbose_reporter.stat_line(f"Failed to create dendrogram: {str(e)}")
    
    def run_pipeline(self) -> None:
        """Run the complete PCA + UMAP + agglomerative clustering pipeline"""
        self.verbose_reporter.section_header("CLUSTERING PHASE (PCA + UMAP + Agglomerative)", "🔬")
        
        try:
            # Step 1: PCA for initial dimensionality reduction
            self.add_pca_embeddings()
            
            # Step 2: UMAP for final dimensionality reduction
            self.add_umap_embeddings()
            
            # Step 3: Agglomerative clustering
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
        unique_clusters = set(clusters)
        
        summary_stats = {
            "Total items processed": len(self.output_list),
            "Clusters found": len(unique_clusters),
            "Clustering method": f"Agglomerative ({self.linkage} linkage)",
            "Dimensionality reduction": f"{self.dr_stats.original_dims} → {self.dr_stats.pca_dims} (PCA) → {self.dr_stats.final_dims} (UMAP)",
            "PCA variance retained": f"{self.dr_stats.pca_variance_retained*100:.2f}%",
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
    
    def get_cluster_samples(self, cluster_id: int, n_samples: int = 10) -> List[Dict[str, Any]]:
        """Get sample items from a specific cluster for inspection"""
        cluster_items = []
        
        for item in self.output_list:
            if item.initial_idea_cluster == cluster_id:
                cluster_items.append({
                    'respondent_id': item.respondent_id,
                    'idea_id': item.idea_id,
                    'idea': item.idea
                })
        
        # Random sample if more than requested
        if len(cluster_items) > n_samples:
            import random
            cluster_items = random.sample(cluster_items, n_samples)
        
        return cluster_items