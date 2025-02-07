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
from utils.cluster_quality import ClusterQualityAnalyzer
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
            if self.verbose:
                print("Using configured UMAP dimensionality reduction")
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
            if self.verbose:
                print("Using configured HDBSCAN clustering")
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
            if self.verbose:
                print("Using configured CountVectorizer")
        else:
            self.vectorizer_model = vectorizer_model

    def _get_stop_words(self):
        if DEFAULT_LANGUAGE == "Dutch":
            try:
                return list(spacy.load("nl_core_news_lg").Defaults.stop_words)
            except:
                print("Warning: Dutch language model not found. Using English stop words.")
                return 'english'
        else:
            return 'english'

    def populate_from_input_list(self, input_list: List[models.EmbeddingsModel]) -> None:
        if self.verbose:
            print("Populating output list from input models...")
        
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
        if self.verbose:
            print("Reducing dimensionality of embeddings...")
        
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
                
        if self.verbose:
            print("Completed dimensionality reduction")

    def add_initial_clusters(self) -> None:
        # Performs initial clustering on reduced embeddings and adds cluster labels to output_list.
        if self.verbose:
            print("Clustering reduced embeddings...")

        # Cluster code embeddings if needed
        if self.embedding_type == "code":
            reduced_code_embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
            initial_code_clusters = self.cluster_model.fit_predict(reduced_code_embeddings)
            
            # Add cluster labels to output list
            for i, item in enumerate(self.output_list):
                item.initial_code_cluster = initial_code_clusters[i]
            
            if self.verbose:
                cluster_counts = Counter(initial_code_clusters)
                print(f"Code clusters: {len(set(initial_code_clusters))} clusters found")
                for cluster_id, count in sorted(cluster_counts.items()):
                    print(f"Code Cluster {cluster_id}: {count} items")
                
                # Show a sample of codes per cluster
                self._print_sample_items_per_cluster(initial_code_clusters, "segment_label", "Code")
                
        # Cluster description embeddings if needed
        if self.embedding_type == "description":
            reduced_description_embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
            initial_description_clusters = self.cluster_model.fit_predict(reduced_description_embeddings)
            
            # Add cluster labels to output list
            for i, item in enumerate(self.output_list):
                item.initial_description_cluster = initial_description_clusters[i]
            
            if self.verbose:
                cluster_counts = Counter(initial_description_clusters)
                print(f"Description clusters: {len(set(initial_description_clusters))} clusters found")
                for cluster_id, count in sorted(cluster_counts.items()):
                    print(f"Description Cluster {cluster_id}: {count} items")
                
                # Show a sample of descriptions per cluster
                self._print_sample_items_per_cluster(initial_description_clusters, "segment_description", "Description")

    def _print_sample_items_per_cluster(self, clusters, attribute, label_prefix, sample_size=5):
        # Helper method to print samples from each cluster
        cluster_to_items = defaultdict(list)
        for cluster, item in zip(clusters, self.output_list):
            value = getattr(item, attribute)
            if value and value.lower() != "na":
                cluster_to_items[cluster].append(value)
                
        for cluster in sorted(cluster_to_items.keys()):
            print(f"{label_prefix} Cluster {cluster}:")
            for item_text in cluster_to_items[cluster][:sample_size]:
                print(f" - {item_text}")

    def calculate_and_display_quality_metrics(self) -> Dict:
        """Calculate quality metrics for the clustering results - informational only"""
        if self.verbose:
            print("\n📊 Clustering Quality Metrics (informational only):")
        
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
        
        # Display metrics
        if self.verbose:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        return metrics

    def filter_and_remap_clusters(self) -> None:
        """Filter out NA items and remap cluster IDs to be sequential"""
        if self.verbose:
            print("\n🧹 Filtering NA items and remapping clusters...")
        
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
        
        if self.verbose:
            print(f"Filtered out {len(clusters) - len(filtered_clusters)} items (noise or NA)")
        
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
        
        if self.verbose:
            print(f"Remapped {len(unique_clusters)} clusters to sequential IDs (0-{len(unique_clusters)-1})")

    def run_pipeline(self, embedding_type: str = None) -> None:
        
        if embedding_type:
            self.embedding_type = embedding_type
            
        if self.verbose:
            print(f"\n🚀 Running simple clustering pipeline with embedding type: {self.embedding_type}")
            
        if not self.output_list:
            raise ValueError("Output list is empty. Please populate it first.")
        
        # Step 1: Reduce dimensionality of embeddings
        if self.verbose:
            print("\n📊 STEP 1: Dimensionality Reduction")
        self.add_reduced_embeddings()
        
        # Step 2: Perform initial clustering
        if self.verbose:
            print("\n🔍 STEP 2: Initial Clustering")
        self.add_initial_clusters()
        
        # Step 3: Calculate and display quality metrics (if enabled)
        if self.config.enable_quality_metrics:
            if self.verbose:
                print("\n📈 STEP 3: Quality Assessment")
            self.calculate_and_display_quality_metrics()
        
        # Step 4: Filter NA items and remap clusters (if enabled)
        if self.config.filter_na_items or self.config.remap_cluster_ids:
            if self.verbose:
                print("\n🧹 STEP 4: Filter and Remap")
            if self.config.filter_na_items and self.config.remap_cluster_ids:
                self.filter_and_remap_clusters()
            elif self.config.filter_na_items:
                # Just filter, don't remap
                pass  # TODO: Implement filter-only method if needed
            elif self.config.remap_cluster_ids:
                # Just remap, don't filter
                pass  # TODO: Implement remap-only method if needed
        
        if self.verbose:
            print("\n✅ Pipeline completed successfully")
            
            # Print some summary statistics after filtering
            if self.embedding_type == "code":
                clusters = [item.initial_code_cluster for item in self.output_list if item.initial_code_cluster is not None]
            else:
                clusters = [item.initial_description_cluster for item in self.output_list if item.initial_description_cluster is not None]
            
            unique_clusters = set(clusters)
            print("\nSUMMARY (after filtering):")
            print(f"- Total items processed: {len(self.output_list)}")
            print(f"- Items with valid clusters: {len(clusters)}")
            print(f"- Clusters found: {len(unique_clusters)}")
            print(f"- Items filtered out: {len(self.output_list) - len(clusters)}")
            
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
                
                # Only store micro clusters (initial clusters)
                micro_cluster = {}
                if self.embedding_type == "code" and item.initial_code_cluster is not None:
                    micro_cluster[item.initial_code_cluster] = ""
                elif self.embedding_type == "description" and item.initial_description_cluster is not None:
                    micro_cluster[item.initial_description_cluster] = ""
                
                # Use None if no valid clusters were added
                if not micro_cluster:
                    micro_cluster = None
                
                # Create submodel with no meta clusters
                submodel = models.ClusterSubmodel(
                    segment_id=item.segment_id,
                    segment_response=segment_response,
                    segment_label=item.segment_label,
                    segment_description=item.segment_description,
                    code_embedding=item.code_embedding,
                    description_embedding=item.description_embedding,
                    meta_cluster=None,  # No meta clusters in simple version
                    macro_cluster=None,  # No macro clusters 
                    micro_cluster=micro_cluster  # Only initial clusters
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
    from utils.cache_manager import CacheManager
    from config import CacheConfig
    import models
    import data_io
    
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
        data_loader = data_io.DataLoader()
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