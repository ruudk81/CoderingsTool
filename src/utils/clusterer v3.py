import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, validator
from typing import List, Any, Optional, Dict
import numpy.typing as npt
from collections import Counter
from umap import UMAP
import hdbscan
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import hashlib
from dataclasses import dataclass
import pandas as pd

# config
import models
from config import DEFAULT_LANGUAGE, ClusteringConfig, DEFAULT_CLUSTERING_CONFIG
from utils.verboseReporter import VerboseReporter
import warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")


# Data integrity tracking
@dataclass
class DataIntegrityCheck:
    """Track data integrity throughout processing"""
    item_count: int
    respondent_ids: set
    idea_ids: set
    embedding_shapes: set
    
    def validate_against(self, other: 'DataIntegrityCheck', operation: str) -> None:
        """Validate data integrity between operations"""
        if self.item_count != other.item_count:
            raise ValueError(f"Item count mismatch after {operation}: {self.item_count} vs {other.item_count}")
        if self.idea_ids != other.idea_ids:
            raise ValueError(f"Idea IDs mismatch after {operation}")


# Enhanced ResultMapper with validation
class ResultMapper(BaseModel):
    """Enhanced result mapper with built-in validation and unique ID"""
    # Unique identifier for tracking
    unique_id: str = Field(default_factory=lambda: "")
    
    # Original fields
    respondent_id: Any
    idea_id: str
    idea: str
    idea_embedding: npt.NDArray[np.float32]
    
    # Processing results
    reduced_idea_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_idea_cluster: Optional[int] = None
    
    # Processing metadata
    processing_order: Optional[int] = None
    embedding_hash: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Generate unique ID if not provided
        if not self.unique_id:
            self.unique_id = f"{self.respondent_id}_{self.idea_id}_{id(self)}"
        # Calculate embedding hash for integrity checking
        self.embedding_hash = hashlib.md5(self.idea_embedding.tobytes()).hexdigest()
    
    @validator('idea_embedding')
    def validate_embedding(cls, v):
        if v is None:
            raise ValueError("idea_embedding cannot be None")
        if not isinstance(v, np.ndarray):
            raise ValueError("idea_embedding must be a numpy array")
        if len(v.shape) != 1:
            raise ValueError(f"idea_embedding must be 1-dimensional, got shape {v.shape}")
        return v
    
    @validator('reduced_idea_embedding')
    def validate_reduced_embedding(cls, v):
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("reduced_idea_embedding must be a numpy array")
            if len(v.shape) != 1:
                raise ValueError(f"reduced_idea_embedding must be 1-dimensional, got shape {v.shape}")
        return v


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
        
        # Initialize data structures
        self.output_list: List[ResultMapper] = []
        self._processing_df: Optional[pd.DataFrame] = None
        self._integrity_checks: Dict[str, DataIntegrityCheck] = {}
        
        if input_list:
            self.populate_from_input_list(input_list)
        
        self.var_lab = var_lab if var_lab else ""
        
        # Initialize models
        self._initialize_models(dim_reduction_model, cluster_model, vectorizer_model)
    
    def _initialize_models(self, dim_reduction_model, cluster_model, vectorizer_model):
        """Initialize all models with proper configuration"""
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
    
    def _create_integrity_check(self, stage: str) -> DataIntegrityCheck:
        """Create an integrity check snapshot at a given stage"""
        check = DataIntegrityCheck(
            item_count=len(self.output_list),
            respondent_ids={item.respondent_id for item in self.output_list},
            idea_ids={item.idea_id for item in self.output_list},
            embedding_shapes={item.idea_embedding.shape for item in self.output_list}
        )
        self._integrity_checks[stage] = check
        return check
    
    def _validate_integrity(self, stage: str, previous_stage: str) -> None:
        """Validate data integrity between stages"""
        if previous_stage in self._integrity_checks:
            current_check = self._create_integrity_check(stage)
            previous_check = self._integrity_checks[previous_stage]
            current_check.validate_against(previous_check, stage)
    
    def populate_from_input_list(self, input_list: List[models.EmbeddingsModel]) -> None:
        """Populate output list with enhanced validation"""
        self.verbose_reporter.stat_line("Populating output list from input models")
        
        # Store original input list for later conversion
        self._original_input_list = input_list
        
        self.output_list = []
        processing_order = 0
        
        # Track all unique combinations to detect duplicates
        seen_combinations = set()
        
        for respondent_item in input_list:
            if respondent_item.response_ideas:
                for response_item in respondent_item.response_ideas:
                    # Create unique key
                    key = (respondent_item.respondent_id, response_item.idea_id)
                    
                    # Check for duplicates
                    if key in seen_combinations:
                        self.verbose_reporter.stat_line(
                            f"Warning: Duplicate found for respondent {respondent_item.respondent_id}, "
                            f"idea {response_item.idea_id}. Skipping."
                        )
                        continue
                    
                    seen_combinations.add(key)
                    
                    # Validate embedding
                    if response_item.idea_embedding is None:
                        self.verbose_reporter.stat_line(
                            f"Warning: Missing embedding for respondent {respondent_item.respondent_id}, "
                            f"idea {response_item.idea_id}. Skipping."
                        )
                        continue
                    
                    # Create ResultMapper with processing order
                    result = ResultMapper(
                        respondent_id=respondent_item.respondent_id,
                        idea_id=response_item.idea_id,
                        idea=response_item.idea or "NA",
                        idea_embedding=response_item.idea_embedding,
                        processing_order=processing_order
                    )
                    
                    self.output_list.append(result)
                    processing_order += 1
        
        # Create initial integrity check
        self._create_integrity_check("initial_population")
        
        # Create DataFrame for additional tracking
        self._create_tracking_dataframe()
        
        self.verbose_reporter.stat_line(f"Populated {len(self.output_list)} items successfully")
    
    def _create_tracking_dataframe(self) -> None:
        """Create a pandas DataFrame for additional alignment tracking"""
        data = []
        for item in self.output_list:
            data.append({
                'unique_id': item.unique_id,
                'processing_order': item.processing_order,
                'respondent_id': item.respondent_id,
                'idea_id': item.idea_id,
                'embedding_hash': item.embedding_hash,
                'embedding_shape': item.idea_embedding.shape
            })
        
        self._processing_df = pd.DataFrame(data)
        self._processing_df.set_index('unique_id', inplace=True)
    
    def add_reduced_embeddings(self) -> None:
        """Add reduced embeddings with comprehensive validation"""
        self.verbose_reporter.step_start("Reducing dimensionality of embeddings", "📊")
        
        # Validate before processing
        self._validate_integrity("before_reduction", "initial_population")
        
        # Sort by processing order to ensure consistency
        sorted_items = sorted(self.output_list, key=lambda x: x.processing_order)
        
        # Create embedding array with validation
        idea_embeddings_list = []
        unique_ids_order = []
        
        for item in sorted_items:
            if item.idea_embedding is None:
                raise ValueError(f"Missing embedding for item {item.unique_id}")
            idea_embeddings_list.append(item.idea_embedding)
            unique_ids_order.append(item.unique_id)
        
        idea_embeddings_array = np.array(idea_embeddings_list)
        
        # Validate array shape
        expected_shape = (len(sorted_items), sorted_items[0].idea_embedding.shape[0])
        if idea_embeddings_array.shape != expected_shape:
            raise ValueError(f"Embedding array shape mismatch. Expected {expected_shape}, got {idea_embeddings_array.shape}")
        
        # Perform dimensionality reduction
        reduced_idea_embeddings = self.dim_reduction_model.fit_transform(idea_embeddings_array)
        
        # Validate output
        if len(reduced_idea_embeddings) != len(sorted_items):
            raise ValueError(f"Dimensionality reduction output mismatch. Expected {len(sorted_items)}, got {len(reduced_idea_embeddings)}")
        
        # Create mapping for safe assignment
        unique_id_to_reduced = dict(zip(unique_ids_order, reduced_idea_embeddings))
        
        # Assign reduced embeddings using unique IDs
        for item in self.output_list:
            if item.unique_id not in unique_id_to_reduced:
                raise ValueError(f"Missing reduced embedding for item {item.unique_id}")
            item.reduced_idea_embedding = unique_id_to_reduced[item.unique_id]
        
        # Update tracking DataFrame
        self._processing_df['has_reduced_embedding'] = [
            item.reduced_idea_embedding is not None for item in self.output_list
        ]
        
        # Validate after processing
        self._validate_integrity("after_reduction", "before_reduction")
        
        # Additional validation
        for item in self.output_list:
            if item.reduced_idea_embedding is None:
                raise ValueError(f"Failed to assign reduced embedding for item {item.unique_id}")
        
        self.verbose_reporter.step_complete("Dimensionality reduction completed with validation")
    
    def add_initial_clusters(self) -> None:
        """Add initial clusters with comprehensive validation"""
        self.verbose_reporter.step_start("Clustering reduced embeddings", "🔍")
        
        # Validate before processing
        for item in self.output_list:
            if item.reduced_idea_embedding is None:
                raise ValueError(f"Missing reduced embedding for item {item.unique_id}")
        
        # Sort by processing order to ensure consistency
        sorted_items = sorted(self.output_list, key=lambda x: x.processing_order)
        
        # Create reduced embedding array with tracking
        reduced_embeddings_list = []
        unique_ids_order = []
        
        for item in sorted_items:
            reduced_embeddings_list.append(item.reduced_idea_embedding)
            unique_ids_order.append(item.unique_id)
        
        reduced_embeddings_array = np.array(reduced_embeddings_list)
        
        # Validate array
        if len(reduced_embeddings_array) != len(sorted_items):
            raise ValueError(f"Reduced embedding array mismatch. Expected {len(sorted_items)}, got {len(reduced_embeddings_array)}")
        
        # Perform clustering
        initial_idea_clusters = self.cluster_model.fit_predict(reduced_embeddings_array)
        
        # Validate output
        if len(initial_idea_clusters) != len(sorted_items):
            raise ValueError(f"Clustering output mismatch. Expected {len(sorted_items)}, got {len(initial_idea_clusters)}")
        
        # Create mapping for safe assignment
        unique_id_to_cluster = dict(zip(unique_ids_order, initial_idea_clusters))
        
        # Assign clusters using unique IDs
        for item in self.output_list:
            if item.unique_id not in unique_id_to_cluster:
                raise ValueError(f"Missing cluster assignment for item {item.unique_id}")
            item.initial_idea_cluster = unique_id_to_cluster[item.unique_id]
        
        # Update tracking DataFrame
        self._processing_df['cluster'] = [
            item.initial_idea_cluster for item in self.output_list
        ]
        
        # Validate all items have clusters
        for item in self.output_list:
            if item.initial_idea_cluster is None:
                raise ValueError(f"Failed to assign cluster for item {item.unique_id}")
        
        # Report statistics
        cluster_counts = Counter(initial_idea_clusters)
        self.verbose_reporter.stat_line(f"Clusters: {len(set(initial_idea_clusters))} clusters found")
        noise_count = cluster_counts.get(-1, 0)
        if noise_count > 0:
            self.verbose_reporter.stat_line(f"Noise cluster (-1): {noise_count} items")
        
        self.verbose_reporter.step_complete("Initial clustering completed with validation")
    
    def run_pipeline(self) -> None:
        """Run the complete pipeline with integrity checks"""
        self.verbose_reporter.section_header("CLUSTERING PHASE", "🔬")
        
        try:
            # Run pipeline steps
            self.add_reduced_embeddings()
            self.add_initial_clusters()
            
            # Final validation
            self._perform_final_validation()
            
            # Create summary
            clusters = [item.initial_idea_cluster for item in self.output_list]
            unique_clusters = set(clusters)
            
            summary_stats = {
                "Total items processed": len(self.output_list),
                "Items with valid clusters": len(clusters),
                "Clusters found": len(unique_clusters),
                "Unique respondents": len(set(item.respondent_id for item in self.output_list)),
                "Integrity checks passed": len(self._integrity_checks)
            }
            
            self.verbose_reporter.summary("Final Clustering Summary", summary_stats, "📊")
            self.verbose_reporter.step_complete("Clustering pipeline completed successfully")
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Pipeline failed: {str(e)}")
            self._export_debug_info()
            raise
    
    def _perform_final_validation(self) -> None:
        """Perform comprehensive final validation"""
        self.verbose_reporter.step_start("Performing final validation", "✓")
        
        # Check all items have all required fields
        for item in self.output_list:
            if item.idea_embedding is None:
                raise ValueError(f"Missing idea_embedding for {item.unique_id}")
            if item.reduced_idea_embedding is None:
                raise ValueError(f"Missing reduced_idea_embedding for {item.unique_id}")
            if item.initial_idea_cluster is None:
                raise ValueError(f"Missing initial_idea_cluster for {item.unique_id}")
        
        # Verify data consistency with DataFrame
        df_check = pd.DataFrame([
            {
                'unique_id': item.unique_id,
                'respondent_id': item.respondent_id,
                'idea_id': item.idea_id,
                'has_embedding': item.idea_embedding is not None,
                'has_reduced': item.reduced_idea_embedding is not None,
                'has_cluster': item.initial_idea_cluster is not None
            }
            for item in self.output_list
        ])
        
        if not df_check['has_embedding'].all():
            raise ValueError("Some items missing embeddings")
        if not df_check['has_reduced'].all():
            raise ValueError("Some items missing reduced embeddings")
        if not df_check['has_cluster'].all():
            raise ValueError("Some items missing clusters")
        
        self.verbose_reporter.step_complete("Final validation passed")
    
    def _export_debug_info(self) -> None:
        """Export debug information in case of failure"""
        try:
            debug_data = []
            for item in self.output_list:
                debug_data.append({
                    'unique_id': item.unique_id,
                    'respondent_id': item.respondent_id,
                    'idea_id': item.idea_id,
                    'has_embedding': item.idea_embedding is not None,
                    'embedding_shape': item.idea_embedding.shape if item.idea_embedding is not None else None,
                    'has_reduced': item.reduced_idea_embedding is not None,
                    'reduced_shape': item.reduced_idea_embedding.shape if item.reduced_idea_embedding is not None else None,
                    'cluster': item.initial_idea_cluster
                })
            
            debug_df = pd.DataFrame(debug_data)
            debug_df.to_csv('clustering_debug.csv', index=False)
            self.verbose_reporter.stat_line("Debug information exported to clustering_debug.csv")
        except Exception as e:
            self.verbose_reporter.stat_line(f"Failed to export debug info: {str(e)}")
    
    def to_cluster_model(self) -> List[models.ClusterModel]:
        """Convert results to ClusterModel with validation"""
        self.verbose_reporter.step_start("Converting results to ClusterModel format", "🔄")
        
        # Group results by respondent_id while maintaining order
        respondent_groups = {}
        for item in self.output_list:
            if item.respondent_id not in respondent_groups:
                respondent_groups[item.respondent_id] = []
            respondent_groups[item.respondent_id].append(item)
        
        # Create ClusterModel objects
        cluster_models = []
        
        for respondent_id, items in respondent_groups.items():
            # Sort items by processing order to maintain consistency
            items.sort(key=lambda x: x.processing_order)
            
            # Find the original EmbeddingsModel
            original_model = None
            if hasattr(self, '_original_input_list'):
                for model in self._original_input_list:
                    if model.respondent_id == respondent_id:
                        original_model = model
                        break
            
            # Create ClusterSubmodel objects
            cluster_submodels = []
            for item in items:
                # Validate item has all required fields
                if item.initial_idea_cluster is None:
                    raise ValueError(f"Missing cluster for item {item.unique_id}")
                
                cluster_submodel = models.ClusterSubmodel(
                    idea_id=item.idea_id,
                    idea=item.idea,
                    initial_cluster=item.initial_idea_cluster
                )
                cluster_submodels.append(cluster_submodel)
            
            # Create ClusterModel
            if original_model:
                # Validate consistency
                original_idea_ids = {idea.idea_id for idea in original_model.response_ideas}
                processed_idea_ids = {item.idea_id for item in items}
                
                if original_idea_ids != processed_idea_ids:
                    self.verbose_reporter.stat_line(
                        f"Warning: Idea ID mismatch for respondent {respondent_id}. "
                        f"Original: {original_idea_ids}, Processed: {processed_idea_ids}"
                    )
                
                # Use original model as base
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
                    idea_count=len(cluster_submodels)
                )
            
            cluster_models.append(cluster_model)
        
        # Final validation
        total_ideas_original = sum(len(items) for items in respondent_groups.values())
        total_ideas_converted = sum(len(model.response_ideas) for model in cluster_models)
        
        if total_ideas_original != total_ideas_converted:
            raise ValueError(
                f"Idea count mismatch in conversion: {total_ideas_original} vs {total_ideas_converted}"
            )
        
        self.verbose_reporter.step_complete(
            f"Converted {len(cluster_models)} respondents to ClusterModel format"
        )
        
        return cluster_models