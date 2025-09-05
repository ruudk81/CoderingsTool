import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
from typing import List, Optional, Any, Dict, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from umap import UMAP
import hdbscan
import random

if TYPE_CHECKING:
    from config import HDBSCANConfig

# === MODELS ========================================================================================================
from pydantic import BaseModel
from models import EmbeddingsModel, ClusterModel, ClusterSubmodel

# === UTILS ========================================================================================================
from utils import verboseReporter

class ResultMapper(BaseModel):
    respondent_id: Any
    idea_id: str
    idea: str
    idea_embedding: npt.NDArray[np.float32]
    pca_embedding: Optional[npt.NDArray[np.float32]] = None
    umap_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_idea_cluster: Optional[int] = None
    processing_order: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class Clusterer:
    def __init__(self, input_list: List[EmbeddingsModel],
                 variance_threshold: float = 0.9,
                 umap_n_components: int = 10,
                 umap_n_neighbors: int = 5,
                 hdbscan_config: Optional['HDBSCANConfig'] = None,
                 verbose: bool = False,
                 verbose_reporter: Optional['VerboseReporter'] = None):

        self.verbose_reporter = verbose_reporter or verboseReporter.VerboseReporter(verbose, capture_logging=True)
        self.variance_threshold = variance_threshold
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        
        # Handle HDBSCAN configuration
        if hdbscan_config is not None:
            self.hdbscan_config = hdbscan_config
        else:
            # Import here to avoid circular import
            from config import DEFAULT_HDBSCAN_CONFIG
            self.hdbscan_config = DEFAULT_HDBSCAN_CONFIG

        self._original_input_list = input_list
        self.output_list: List[ResultMapper] = []
        self._populate_from_input_list(input_list)

    def _populate_from_input_list(self, input_list: List[EmbeddingsModel]) -> None:
        self.output_list = []
        processing_order = 0
        for respondent_item in input_list:
            if respondent_item.response_ideas: 
                for embedding_item in respondent_item.response_ideas:  
                    if embedding_item.idea_embedding is None:
                        continue
                    result = ResultMapper(
                        respondent_id=respondent_item.respondent_id,
                        idea_id=embedding_item.idea_id,
                        idea=embedding_item.idea or "NA",
                        idea_embedding=embedding_item.idea_embedding,
                        processing_order=processing_order
                    )
                    self.output_list.append(result)
                    processing_order += 1

    def run(self):
        # Display input statistics and configuration
        self.verbose_reporter.step_start("Clustering embedded ideas", emoji="🔄")
        
        embeddings = np.array([item.idea_embedding for item in self.output_list])
        
        self.verbose_reporter.stat_line(f"Input: {len(self.output_list)} idea embeddings ({embeddings.shape[1]} dimensions)")
        self.verbose_reporter.stat_line(f"UMAP configuration: {self.umap_n_neighbors} neighbors, {self.umap_n_components} components, cosine metric")
        self.verbose_reporter.stat_line(f"HDBSCAN configuration: min_cluster_size={self.hdbscan_config.min_cluster_size}, epsilon={self.hdbscan_config.cluster_selection_epsilon}, alpha={self.hdbscan_config.alpha}, {self.hdbscan_config.metric} metric")
        
        #from sklearn.decomposition import PCA
        #from sklearn.preprocessing import StandardScaler

        # # === Step 1: PCA ===
        # scaler = StandardScaler()
        # scaled = scaler.fit_transform(embeddings)

        # pca = PCA()
        # pca_embeddings = pca.fit_transform(scaled)

        # total_variance = 0.0
        # optimal_dims = 0
        # for i, variance in enumerate(pca.explained_variance_ratio_):
        #     total_variance += variance
        #     if total_variance >= self.variance_threshold:
        #         optimal_dims = i + 1
        #         break

        # pca_embeddings = pca_embeddings[:, :optimal_dims]

        # for item, pca_embed in zip(self.output_list, pca_embeddings):
        #     item.pca_embedding = pca_embed

        # print(f"[PCA] Reduced {embeddings.shape[1]} → {optimal_dims} dims ({total_variance * 100:.2f}% variance retained)")

        # === Step 2: UMAP ===
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line(f"Reducing dimensions with UMAP... ({embeddings.shape[1]} → {self.umap_n_components} dims)")
        
        umap = UMAP(
            n_neighbors = self.umap_n_neighbors,  # Higher for better semantic relationships
            n_components = self.umap_n_components,  # More dimensions to preserve semantic nuances
            min_dist = 0.1,  # Slight separation for better cluster distinction
            metric = "cosine",   
            random_state  = 42,
            n_jobs  = 1,
            low_memory = True,
            transform_seed = 42
        )
        #umap_embeddings = umap.fit_transform(pca_embeddings)
        umap_embeddings = umap.fit_transform(embeddings)

        for item, umap_embed in zip(self.output_list, umap_embeddings):
            item.umap_embedding = umap_embed

        #print(f"[UMAP] Reduced {optimal_dims} → {self.umap_n_components} dims")

        # === Step 3: HDBSCAN ===
        self.verbose_reporter.stat_line("Clustering with HDBSCAN...")
        
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_config.min_cluster_size,
            min_samples=self.hdbscan_config.min_samples,
            cluster_selection_epsilon=self.hdbscan_config.cluster_selection_epsilon,
            alpha=self.hdbscan_config.alpha,
            metric=self.hdbscan_config.metric,
            cluster_selection_method=self.hdbscan_config.cluster_selection_method,
            prediction_data=self.hdbscan_config.prediction_data,
            approx_min_span_tree=self.hdbscan_config.approx_min_span_tree,
            gen_min_span_tree=self.hdbscan_config.gen_min_span_tree,
        )
        labels = hdb.fit_predict(umap_embeddings)

        for item, label in zip(self.output_list, labels):
            item.initial_idea_cluster = int(label)

        # Calculate cluster statistics
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = list(labels).count(-1)
        
        # Calculate cluster size statistics
        cluster_sizes = {}
        for label in labels:
            if label != -1:  # Exclude noise
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        if cluster_sizes:
            sizes = list(cluster_sizes.values())
            min_size = min(sizes)
            max_size = max(sizes)
            median_size = sorted(sizes)[len(sizes)//2]
        else:
            min_size = max_size = median_size = 0

        # Report statistics
        self.verbose_reporter.stat_line(f"Found {num_clusters} clusters")
        self.verbose_reporter.stat_line(f"Noise points: {noise_points} / {len(self.output_list)} ({noise_points / len(self.output_list) * 100:.1f}%)")
        if num_clusters > 0:
            self.verbose_reporter.stat_line(f"Cluster sizes: min={min_size}, max={max_size}, median={median_size}")
        
        # Display sample clusters
        self._display_sample_clusters(labels, cluster_sizes)
        
        # Complete the step
        self.verbose_reporter.step_complete("Initial clustering completed", emoji="✅")

    def _display_sample_clusters(self, labels: np.ndarray, cluster_sizes: Dict[int, int]) -> None:
        """Display sample clusters with example ideas"""
        if not cluster_sizes or not self.verbose_reporter.enabled:
            return
        
        # Select up to 3 sample clusters
        num_samples = min(3, len(cluster_sizes))
        if num_samples == 0:
            return
        
        # Sort clusters by size and select some interesting ones
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Get a mix of large and medium clusters
        sample_clusters = []
        if len(sorted_clusters) >= 3:
            # Get largest, a middle one, and a smaller one
            sample_clusters = [
                sorted_clusters[0],  # Largest
                sorted_clusters[len(sorted_clusters)//3],  # Middle
                sorted_clusters[2*len(sorted_clusters)//3]  # Smaller
            ]
        else:
            sample_clusters = sorted_clusters[:num_samples]
        
        self.verbose_reporter.empty_line()
        print("📋 Sample clusters:")
        
        for cluster_id, size in sample_clusters:
            # Get ideas from this cluster
            cluster_ideas = []
            for i, (item, label) in enumerate(zip(self.output_list, labels)):
                if label == cluster_id:
                    cluster_ideas.append(item.idea)
            
            # Display cluster info
            print(f"  Cluster {cluster_id} ({size} ideas):")
            
            # Show up to 3 random examples from this cluster
            examples = random.sample(cluster_ideas, min(3, len(cluster_ideas)))
            for example in examples:
                # Truncate long ideas to 80 characters
                if len(example) > 80:
                    example = example[:77] + "..."
                print(f"    → \"{example}\"")
            
            # Add spacing between clusters
            if (cluster_id, size) != sample_clusters[-1]:
                print()

    def to_cluster_model(self) -> List[ClusterModel]:
        respondent_groups = {}
        for item in self.output_list:
            respondent_groups.setdefault(item.respondent_id, []).append(item)

        cluster_models = []
        for respondent_id, items in respondent_groups.items():
            items.sort(key=lambda x: x.processing_order)

            original_model = next((m for m in self._original_input_list if m.respondent_id == respondent_id), None)

            cluster_submodels = [
                ClusterSubmodel(
                    idea_id=item.idea_id,
                    idea=item.idea,
                    idea_embedding=item.idea_embedding,
                    initial_cluster=item.initial_idea_cluster
                ) for item in items
            ]

            if original_model:
                cluster_model = ClusterModel(
                    **original_model.model_dump(exclude={'response_ideas'}),
                    response_ideas=cluster_submodels)
            else:
                cluster_model = ClusterModel(
                    respondent_id=respondent_id,
                    response_ideas=cluster_submodels,
                    #idea_embeddings=cluster_submodels,
                    idea_count=len(cluster_submodels)
                )

            cluster_models.append(cluster_model)

        return cluster_models
