import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List, Any, Optional, Dict
import numpy.typing as npt
from collections import  Counter
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
    respondent_id: Any
    idea_id: str
    idea: str
    idea_embedding: npt.NDArray[np.float32]
    reduced_idea_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_idea_cluster: Optional[int] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)  

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
        verbose: bool = None):

        # Initialize config and verbose settings first
        self.config = config or DEFAULT_CLUSTERING_CONFIG
        self.verbose = verbose if verbose is not None else self.config.verbose
        self.verbose_reporter = VerboseReporter(self.verbose)
        
        # Initialize output_list before populate_from_input_list
        self.output_list: List[ResultMapper] = []
        
        if input_list:
            self.populate_from_input_list(input_list)
        
        self.var_lab = var_lab if var_lab else ""

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
        self.verbose_reporter.stat_line("Populating output list from input models")
        
        # Store original input list for later conversion
        self._original_input_list = input_list
        
        self.output_list = []
        
        for respondent_item in input_list:
            if respondent_item.response_ideas:
                for response_item in respondent_item.response_ideas:
                    self.output_list.append(ResultMapper(
                            respondent_id=respondent_item.respondent_id,
                            idea_id=response_item.idea_id,
                            idea = response_item.idea or "NA",
                            idea_embedding=response_item.idea_embedding
                    ))

    def add_reduced_embeddings(self) -> None:
        self.verbose_reporter.step_start("Reducing dimensionality of embeddings", "📊")
        
        idea_embeddings_array = np.array([item.idea_embedding for item in self.output_list])
        reduced_idea_embeddings = self.dim_reduction_model.fit_transform(idea_embeddings_array)
        
        for i, item in enumerate(self.output_list):
            item.reduced_idea_embedding = reduced_idea_embeddings[i]
                
        self.verbose_reporter.step_complete("Dimensionality reduction completed")

    def add_initial_clusters(self) -> None:
        self.verbose_reporter.step_start("Clustering reduced embeddings", "🔍")

        reduced_idea_embeddings = np.array([item.reduced_idea_embedding for item in self.output_list])
        initial_idea_clusters = self.cluster_model.fit_predict(reduced_idea_embeddings)
        
        for i, item in enumerate(self.output_list):
            item.initial_idea_cluster = initial_idea_clusters[i]
        
        cluster_counts = Counter(initial_idea_clusters)
        self.verbose_reporter.stat_line(f"Clusters: {len(set(initial_idea_clusters))} clusters found")
        noise_count = cluster_counts.get(-1, 0)
        if noise_count > 0:
            self.verbose_reporter.stat_line(f"Noise cluster (-1): {noise_count} items")
        
        self.verbose_reporter.step_complete("Initial clustering completed")


    def calculate_and_display_quality_metrics(self) -> Dict:
        """Calculate quality metrics for the clustering results - informational only"""
        self.verbose_reporter.step_start("Quality assessment", "📈")
        
        metrics = {}
        
        # Get embeddings and labels based on embedding type
        embeddings = np.array([item.reduced_idea_embedding for item in self.output_list])
        labels = np.array([item.initial_idea_cluster for item in self.output_list])
        
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

    def run_pipeline(self) -> None:
        
        self.verbose_reporter.section_header("CLUSTERING PHASE", "🔬")
        
        self.add_reduced_embeddings()
        self.add_initial_clusters()
          
        clusters = [item.initial_idea_cluster for item in self.output_list if item.initial_idea_cluster is not None]
        
        unique_clusters = set(clusters)
        
        # Create final summary
        summary_stats = {
            "Total items processed": len(self.output_list),
            "Items with valid clusters": len(clusters),
            "Clusters found": len(unique_clusters),
            "Items filtered out": len(self.output_list) - len(clusters)
        }
        
        
        self.verbose_reporter.summary("Final Clustering Summary", summary_stats, "📊")
        self.verbose_reporter.step_complete("Clustering pipeline completed successfully")
    
    def to_cluster_model(self) -> List[models.ClusterModel]:
        """Convert ClusterGenerator results to list of ClusterModel objects"""
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
            # Find the original EmbeddingsModel for this respondent
            original_model = None
            if hasattr(self, '_original_input_list'):
                for model in self._original_input_list:
                    if model.respondent_id == respondent_id:
                        original_model = model
                        break
            
            # Create ClusterSubmodel objects for each idea
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
                # Use original model as base and update with cluster data
                cluster_model = models.ClusterModel(
                    **original_model.model_dump(exclude={'response_ideas', 'idea_embeddings'}),
                    response_ideas=cluster_submodels,
                    idea_embeddings=cluster_submodels
                )
            else:
                # Create minimal ClusterModel
                cluster_model = models.ClusterModel(
                    respondent_id=respondent_id,
                    response_ideas=cluster_submodels,
                    idea_embeddings=cluster_submodels,
                    idea_count=len(cluster_submodels)
                )
            
            cluster_models.append(cluster_model)
        
        self.verbose_reporter.step_complete(f"Converted {len(cluster_models)} respondents to ClusterModel format")
        return cluster_models
            
   
