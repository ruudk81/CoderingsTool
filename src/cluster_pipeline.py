from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
from collections import Counter


@dataclass
class ResultMapper:
    respondent_id: str
    idea_id: str
    idea: str
    idea_embedding: npt.NDArray[np.float32]
    processing_order: int
    pca_embedding: Optional[npt.NDArray[np.float32]] = None
    umap_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_idea_cluster: Optional[int] = None


@dataclass
class DimensionalityReductionStats:
    original_dims: int
    pca_dims: int
    pca_variance_retained: float
    umap_dims: int
    total_items: int


class ClusterPipeline:
    def __init__(
        self,
        result_items: List[ResultMapper],
        variance_threshold: float = 0.90,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: Optional[int] = None,
        verbose: bool = True,
    ):
        self.items = [item for item in result_items if item.idea_embedding is not None]
        self.items.sort(key=lambda x: x.processing_order)
        self.variance_threshold = variance_threshold
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.pca = None
        self.umap = None
        self.hdbscan = None
        self.dr_stats: Optional[DimensionalityReductionStats] = None

    def run(self):
        if not self.items:
            raise ValueError("No valid items with embeddings to process.")

        embeddings = np.array([item.idea_embedding for item in self.items])
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        # === Step 1: PCA ===
        scaled = self.scaler.fit_transform(embeddings)
        n_samples, n_features = scaled.shape
        max_components = min(n_samples - 1, n_features)

        pca_full = PCA(n_components=max_components, random_state=42)
        pca_full.fit(scaled)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        optimal_dims = np.argmax(cumsum >= self.variance_threshold) + 1
        optimal_dims = min(optimal_dims, max_components)
        actual_variance = cumsum[optimal_dims - 1]

        self.pca = PCA(n_components=optimal_dims, random_state=42)
        pca_embeddings = self.pca.fit_transform(scaled)
        for item, pca_embed in zip(self.items, pca_embeddings):
            item.pca_embedding = pca_embed

        if self.verbose:
            print(f"[PCA] Reduced {n_features} → {optimal_dims} dims "
                  f"({actual_variance*100:.2f}% variance retained)")

        # === Step 2: UMAP ===
        self.umap = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            n_jobs=1,
            low_memory=True,
            transform_seed=42
        )
        umap_embeddings = self.umap.fit_transform(pca_embeddings)
        for item, umap_embed in zip(self.items, umap_embeddings):
            item.umap_embedding = umap_embed

        if self.verbose:
            print(f"[UMAP] Reduced {optimal_dims} → {self.umap_n_components} dims")

        # === Step 3: HDBSCAN ===
        hdbscan_params = {
            "min_cluster_size": self.hdbscan_min_cluster_size,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
            "approx_min_span_tree": True,
            "gen_min_span_tree": False,
            "random_state": 42
        }
        if self.hdbscan_min_samples is not None:
            hdbscan_params["min_samples"] = self.hdbscan_min_samples

        self.hdbscan = hdbscan.HDBSCAN(**hdbscan_params)
        labels = self.hdbscan.fit_predict(umap_embeddings)
        for item, label in zip(self.items, labels):
            item.initial_idea_cluster = int(label)

        cluster_counts = Counter(labels)
        n_clusters = len([c for c in cluster_counts if c != -1])
        noise_points = cluster_counts.get(-1, 0)

        if self.verbose:
            print(f"[HDBSCAN] Found {n_clusters} clusters")
            print(f"[HDBSCAN] Noise points: {noise_points} / {len(labels)} "
                  f"({noise_points / len(labels) * 100:.1f}%)")

        # === Store stats ===
        self.dr_stats = DimensionalityReductionStats(
            original_dims=n_features,
            pca_dims=optimal_dims,
            pca_variance_retained=actual_variance,
            umap_dims=self.umap_n_components,
            total_items=len(self.items)
        )

    def get_results(self) -> List[ResultMapper]:
        return self.items

    def get_stats(self) -> Optional[DimensionalityReductionStats]:
        return self.dr_stats
