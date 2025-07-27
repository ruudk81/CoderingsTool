from typing import List, Optional, Any
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
from pydantic import BaseModel
from models import EmbeddingsModel, ClusterModel, ClusterSubmodel


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
                 umap_n_components: int = 5,
                 umap_n_neighbors: int = 15,
                 hdbscan_min_cluster_size: int = 10):

        self.variance_threshold = variance_threshold
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size

        self._original_input_list = input_list
        self.output_list: List[ResultMapper] = []
        self._populate_from_input_list(input_list)

    def _populate_from_input_list(self, input_list: List[EmbeddingsModel]) -> None:
        self.output_list = []
        processing_order = 0
        for respondent_item in input_list:
            if respondent_item.idea_embeddings: 
                for embedding_item in respondent_item.idea_embeddings:  
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
        print(f"[INFO] Running clustering on {len(self.output_list)} items")

        embeddings = np.array([item.idea_embedding for item in self.output_list])

        # === Step 1: PCA ===
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)

        pca = PCA()
        pca_embeddings = pca.fit_transform(scaled)

        total_variance = 0.0
        optimal_dims = 0
        for i, variance in enumerate(pca.explained_variance_ratio_):
            total_variance += variance
            if total_variance >= self.variance_threshold:
                optimal_dims = i + 1
                break

        pca_embeddings = pca_embeddings[:, :optimal_dims]

        for item, pca_embed in zip(self.output_list, pca_embeddings):
            item.pca_embedding = pca_embed

        print(f"[PCA] Reduced {embeddings.shape[1]} → {optimal_dims} dims ({total_variance * 100:.2f}% variance retained)")

        # === Step 2: UMAP ===
        umap = UMAP(
            n_neighbors = 5,  # Higher for better semantic relationships
            n_components = 10,  # More dimensions to preserve semantic nuances
            min_dist = 0.1,  # Slight separation for better cluster distinction
            metric = "cosine",   
            random_state  = 42,
            n_jobs  = 1,
            low_memory = True,
            transform_seed = 42
        )
        umap_embeddings = umap.fit_transform(pca_embeddings)

        for item, umap_embed in zip(self.output_list, umap_embeddings):
            item.umap_embedding = umap_embed

        print(f"[UMAP] Reduced {optimal_dims} → {self.umap_n_components} dims")

        # === Step 3: HDBSCAN ===
        hdb = hdbscan.HDBSCAN(
            min_cluster_size= 2,  # Smaller clusters for better semantic coherence
            min_samples= None, # Lower threshold for more selective clustering
            metric= "euclidean",   
            cluster_selection_method = "eom",
            prediction_data= True,
            approx_min_span_tree= False,
            gen_min_span_tree= True,
        )
        labels = hdb.fit_predict(umap_embeddings)

        for item, label in zip(self.output_list, labels):
            item.initial_idea_cluster = int(label)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = list(labels).count(-1)

        print(f"[HDBSCAN] Found {num_clusters} clusters")
        print(f"[HDBSCAN] Noise points: {noise_points} / {len(self.output_list)} ({noise_points / len(self.output_list) * 100:.1f}%)")

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
