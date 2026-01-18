import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
import numpy.typing as npt
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize #StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from collections import defaultdict
import random
import warnings
import concurrent.futures
import multiprocessing
import time
import re

from config import UMAPConfig, ClusteringConfig, HDBSCANConfig

# === CONSTANTS ========================================================================================================
SILHOUETTE_LOW_THRESHOLD = 0.30       # Threshold for "low quality" silhouette score
TOP_K_CLUSTERS_DEFAULT = 5            # Default number of top clusters for centroid analysis
PCA_SIZE_THRESHOLD = 10_000           # Apply PCA when embeddings exceed this count
POLISH_ROUNDS = 3                     # Number of polish iterations in auto-HDBSCAN

# === MODELS ========================================================================================================
from pydantic import BaseModel
from models import EmbeddingsModel, ClusterModel, ClusterSubmodel

# === UTILS ========================================================================================================
from utils import verboseReporter

# === DATACLASSES ========================================================================================================
from dataclasses import dataclass

@dataclass
class ClusterSummary:
    """Summary statistics for a clustering result"""
    n_points: int
    min_cluster_size: int
    min_samples : int
    n_clusters: int
    noise_rate: float
    median_cluster_size: int

@dataclass
class HParams:
    min_samples: int
    min_cluster_size: int
    factor: float
    notes: str

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
    def __init__(
        self, input_list: List[EmbeddingsModel],
        umap_config: Optional['UMAPConfig'] = None,
        clustering_config: Optional['ClusteringConfig'] = None,
        hdbscan_config: Optional['HDBSCANConfig'] = None,
        verbose: bool = False,
        verbose_reporter = None):

        self.verbose_reporter = verbose_reporter or verboseReporter.VerboseReporter(verbose, capture_logging=True)
        
        if umap_config is not None:
            self.umap_config = umap_config
        else:
            from config import DEFAULT_UMAP_CONFIG
            self.umap_config = DEFAULT_UMAP_CONFIG
            
        if clustering_config is not None:
            self.clustering_config = clustering_config
        else:
            from config import DEFAULT_CLUSTERING_CONFIG
            self.clustering_config = DEFAULT_CLUSTERING_CONFIG
        
        if hdbscan_config is not None:
            self.hdbscan_config = hdbscan_config
        else:
            from config import DEFAULT_HDBSCAN_CONFIG
            self.hdbscan_config = DEFAULT_HDBSCAN_CONFIG

        self._original_input_list = input_list
        self.output_list: List[ResultMapper] = []
        self._populate_from_input_list(input_list)
        self.rs = np.random.default_rng(42)
        self.DBCV_D = ClusteringConfig.DBCV_D  # safe for DBCV (avoid overflow)
        self.enable_dbcv = self.clustering_config.enable_dbcv
        self.enable_meanp  = self.clustering_config.enable_meanp
        self.centroid_distance = self.clustering_config.centroid_distance
        self.CLUSTER_METRIC = ClusteringConfig.CLUSTER_METRIC
        
      
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

    def _pca_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        
        pca = PCA(
            n_components=self.clustering_config.pca_components,
            svd_solver="full",
            random_state=self.clustering_config.pca_random_state
        )
        embeddings = pca.fit_transform(embeddings)
            
        return embeddings
    
    def _umap_embed(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():  # Suppress UMAP n_jobs warning when using random_state
            warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state", category=UserWarning, module="umap",)

            # FIX: Check use_parallel_umap flag instead of parallel_jobs value
            if self.umap_config.use_parallel_umap:
                random_state = None
                transform_seed = None
                n_jobs = self.umap_config.parallel_jobs  # Use configured value
            else:
                random_state = self.umap_config.random_state
                transform_seed = self.umap_config.transform_seed
                n_jobs = 1
    
            n = X.shape[0]
            if n <= 100:
                n_components = 8
                n_neighbors  = 8
                min_dist     = 0.00
            elif n < 2000:
                n_components = 10
                n_neighbors  = 15
                min_dist     = 0.05
            else:
                n_components = 12
                n_neighbors  = 30
                min_dist     = 0.10
    
            n_neighbors = int(min(max(2, n_neighbors), max(2, n - 1)))
    
            X = np.asarray(X, dtype=np.float32)
         
            umap_params = {
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_dist": float(min_dist),
                "metric": self.umap_config.metric, 
                "n_epochs": self.umap_config.n_epochs,
                "n_jobs": n_jobs,              
                "low_memory": self.umap_config.low_memory,
                "verbose": False,
                "random_state": random_state,   
                "transform_seed": transform_seed,
                "init": "random",
            }
    
            umap = UMAP(**umap_params)
            U = umap.fit_transform(X).astype(np.float32, copy=False)
            return U
    
    def _dbcv_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate DBCV score for clustering quality"""
    
        mask = labels >= 0
        X_clean, y_clean = X[mask], labels[mask]
        if X_clean.shape[0] == 0:
            return np.nan

        clusters, counts = np.unique(y_clean, return_counts=True)
        keep = np.isin(y_clean, clusters[counts >= 2])
        X_clean, y_clean = X_clean[keep], y_clean[keep]
    
        if np.unique(y_clean).size < 2:
            return np.nan
    
        X_clean = X_clean.astype(np.float64, copy=False)
    
        try:
            return float(validity_index(
                X_clean,
                y_clean,
                metric=self.CLUSTER_METRIC , 
                d=self.DBCV_D
            ))
        except Exception as e:
            print(f"DBCV error: {type(e).__name__}: {str(e)}")
            return np.nan
    

    def _mean_probability(self, hdbscan_model: HDBSCAN, labels: np.ndarray) -> float:
        """Calculate mean probability for clustered points"""
        mask = labels >= 0
        if not np.any(mask): 
            return np.nan
        return float(np.mean(hdbscan_model.probabilities_[mask]))
    
    def _cluster_stability(self, model: HDBSCAN, labels: np.ndarray) -> float:
        """ Size-weighted cluster persistence (Stability*), aligned with HDBSCAN's objective. Returns 0.0 if not available. """

        # labels: -1 = noise
        mask = labels >= 0
        if not np.any(mask):
            return 0.0
        
        # Try both attribute names (depends on version/wrapper)
        persistence = getattr(model, "cluster_persistence_", None)
        if persistence is None: 
            persistence = getattr(model, "cluster_stability_", None)
        if persistence is None:
            return 0.0

        labels_non_noise = labels[mask]
        n = labels_non_noise.size
    
        # cluster sizes
        max_lab = int(labels_non_noise.max())
        counts = np.bincount(labels_non_noise, minlength=max_lab + 1).astype(float)
    
        # Safety: match lengths
        k = min(len(persistence), len(counts))
        if k == 0:
            return 0.0
    
        # Stability* = (1/N_non_noise) sum_c persistence[c] * |c|
        stab_star = float(np.dot(persistence[:k], counts[:k]) / max(n, 1.0))
        return stab_star

    
    def _silhouette_score(self, X: np.ndarray, labels: np.ndarray, sample_size: Optional[int] = None) -> float:
        """Calculate silhouette score for clustering quality"""
        mask = labels >= 0
        X_clean, y_clean = X[mask], labels[mask]
        if X_clean.shape[0] == 0: 
            return np.nan
        
        _, counts = np.unique(y_clean, return_counts=True)
        if (counts >= 2).sum() < 2: 
            return np.nan
        
        return float(silhouette_score(X_clean, y_clean, metric=self.CLUSTER_METRIC, sample_size=sample_size, random_state=self.clustering_config.pca_random_state))
    
    
    def _geom_indices(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calinski–Harabasz (CH ↑) and Davies–Bouldin (DB ↓) on non-noise points."""
        mask = labels >= 0
        Xn, yn = X[mask], labels[mask]
        if Xn.shape[0] < 3: 
            return {"CH": np.nan, "DB": np.nan}
        uniq = np.unique(yn)
        if uniq.size < 2:
            return {"CH": np.nan, "DB": np.nan}
        CH = np.nan
        DB = np.nan
        try:
            CH = float(calinski_harabasz_score(Xn, yn))
        except Exception:
            pass
        try:
            DB = float(davies_bouldin_score(Xn, yn))
        except Exception:
            pass
        return {"CH": CH, "DB": DB}
    
    def _centroid_distance_topk(self, U, labels, topk_clusters):
        uniq, counts = np.unique(labels[labels != -1], return_counts=True)
        if uniq.size < 2:
            return dict(mean_all=np.nan, mean_topk=np.nan, topk_ids=[])
    
        # 1) sort clusters by size (descending) and pick top-K
        order = np.argsort(counts)[::-1]
        top_ids = uniq[order[:topk_clusters]]
     
        # 2) centroids for *all* clusters and for top-K clusters
        def get_centroids(ids):
            cents = []
            for c in ids:
                idx = np.where(labels == c)[0]
                cents.append(U[idx].mean(axis=0))
            return np.vstack(cents)
    
        cents_all = get_centroids(uniq)
        cents_top = get_centroids(top_ids)
    
        # 3) angular distances between centroids
        def ang_stats(cents):
            k = cents.shape[0]
            if k < 2: return np.nan
            C = cosine_similarity(cents)
            S = np.clip(C, -1.0, 1.0)
            D = np.arccos(S) / np.pi
            np.fill_diagonal(D, 0.0)
            iu = np.triu_indices(k, k=1)
            return float(D[iu].mean())
    
        mean_all  = ang_stats(cents_all)
        mean_topk = ang_stats(cents_top)
    
        return mean_all, mean_topk
    
    def _compute_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """Compute cluster centroids and sizes"""
        by_cluster = defaultdict(list)
        for vec, label in zip(embeddings, labels):
            if label is not None and label >= 0:
                by_cluster[int(label)].append(vec)

        centroids = {}
        sizes = {}
        for cluster_id, vectors in by_cluster.items():
            vectors_array = np.vstack(vectors)
            centroids[cluster_id] = vectors_array.mean(axis=0)
            sizes[cluster_id] = vectors_array.shape[0]

        return centroids, sizes

    def _compute_cluster_centroids(self, labels: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """
        Compute L2-normalized centroids from pca_embedding for each cluster (excluding noise).

        Returns:
            centroids: Dict mapping cluster_id -> centroid vector
            sizes: Dict mapping cluster_id -> cluster size
        """
        by_cluster = defaultdict(list)

        # Group embeddings by cluster
        for item, label in zip(self.output_list, labels):
            if label >= 0:  # Exclude noise (-1)
                by_cluster[int(label)].append(item.pca_embedding)

        centroids = {}
        sizes = {}

        # Compute centroids (mean of L2-normalized embeddings)
        for cluster_id, embeddings in by_cluster.items():
            embeddings_array = np.vstack(embeddings)
            centroid = embeddings_array.mean(axis=0)
            # Re-normalize the centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            centroids[cluster_id] = centroid
            sizes[cluster_id] = len(embeddings)

        return centroids, sizes

    def _pairwise_cluster_similarity(self, cluster_a_ids: np.ndarray, cluster_b_ids: np.ndarray) -> Dict[str, float]:
        """
        Calculate all pairwise cosine similarities between two clusters.

        Args:
            cluster_a_ids: Indices of items in cluster A
            cluster_b_ids: Indices of items in cluster B

        Returns:
            Dictionary with quantile statistics: q25, q50 (median), q75, mean
        """
        # Get embeddings for both clusters
        embeddings_a = np.vstack([self.output_list[i].pca_embedding for i in cluster_a_ids])
        embeddings_b = np.vstack([self.output_list[i].pca_embedding for i in cluster_b_ids])

        # Compute all pairwise cosine similarities (dot product since already L2-normalized)
        similarities = embeddings_a @ embeddings_b.T

        # Flatten to 1D array
        similarities_flat = similarities.flatten()

        # Calculate quantiles
        q25 = float(np.quantile(similarities_flat, 0.25))
        q50 = float(np.quantile(similarities_flat, 0.50))
        q75 = float(np.quantile(similarities_flat, 0.75))
        mean = float(np.mean(similarities_flat))

        return {
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'mean': mean
        }

    def _renumber_clusters(self, labels: np.ndarray) -> np.ndarray:
        """
        Renumber clusters sequentially: noise=-1, valid clusters=0 to n-1.

        Args:
            labels: Array with potentially non-sequential cluster IDs

        Returns:
            Array with sequential cluster IDs
        """
        # Get unique cluster IDs (excluding noise)
        unique_clusters = sorted([label for label in np.unique(labels) if label >= 0])

        # Create mapping: old_id -> new_id
        mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        mapping[-1] = -1  # Noise stays as -1

        # Apply mapping
        renumbered = np.array([mapping[label] for label in labels])

        return renumbered

    def _assess_noise_quality(self, U: np.ndarray, labels: np.ndarray, cluster_members: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Assess whether noise points are 'soft noise' (assignable) or 'hard noise' (true outliers).
        Uses member-based quantile logic (consistent with merge logic).

        Args:
            U: UMAP embeddings array
            labels: Cluster labels (with -1 for noise)
            cluster_members: Dict mapping cluster_id -> array of member embeddings

        Returns:
            Dictionary with:
                - total_noise_rate: fraction of all points labeled as noise
                - soft_noise_rate: fraction assignable (quantile similarity >= threshold)
                - hard_noise_rate: fraction truly problematic (quantile similarity < threshold)
                - mean_noise_similarity: average best quantile score across noise points
        """
        noise_mask = labels == -1
        n_total = len(labels)

        # No noise case
        if not np.any(noise_mask):
            return {
                'total_noise_rate': 0.0,
                'soft_noise_rate': 0.0,
                'hard_noise_rate': 0.0,
                'mean_noise_similarity': 1.0
            }

        # No clusters case (all noise)
        if len(cluster_members) == 0:
            return {
                'total_noise_rate': float(noise_mask.sum() / n_total),
                'soft_noise_rate': 0.0,
                'hard_noise_rate': float(noise_mask.sum() / n_total),
                'mean_noise_similarity': 0.0
            }

        # Get noise points
        noise_points = U[noise_mask]
        n_noise = noise_points.shape[0]

        # For each noise point, find best cluster match using quantile-based scoring
        # Note: U (pca_embeddings) are already L2-normalized, so dot product = cosine similarity
        best_matches = []
        threshold = self.clustering_config.noise_assignability_threshold

        for noise_point in noise_points:
            cluster_scores = []
            for cluster_id, members in cluster_members.items():
                # Compute all pairwise similarities (dot product = cosine for L2-normalized)
                similarities = noise_point @ members.T  # (n_members,)

                # Calculate quantile-based score (same as merge logic)
                q25 = float(np.quantile(similarities, 0.25))
                q50 = float(np.quantile(similarities, 0.50))
                q75 = float(np.quantile(similarities, 0.75))
                score = np.mean([q25, q50, q75])

                cluster_scores.append(score)

            # Best match = highest quantile score across all clusters
            best_match = max(cluster_scores) if cluster_scores else 0.0
            best_matches.append(best_match)

        best_matches = np.array(best_matches)

        # Classify by threshold
        assignable = best_matches >= threshold
        n_soft = assignable.sum()
        n_hard = n_noise - n_soft

        # Calculate hard noise score distribution
        hard_noise_stats = {}
        if n_hard > 0:
            hard_scores = best_matches[~assignable]  # Hard noise = NOT assignable
            hard_noise_stats = {
                'hard_noise_q25': float(np.quantile(hard_scores, 0.25)),
                'hard_noise_median': float(np.quantile(hard_scores, 0.50)),
                'hard_noise_q75': float(np.quantile(hard_scores, 0.75)),
                'hard_noise_mean': float(hard_scores.mean())
            }
        else:
            hard_noise_stats = {
                'hard_noise_q25': None,
                'hard_noise_median': None,
                'hard_noise_q75': None,
                'hard_noise_mean': None
            }

        return {
            'total_noise_rate': float(n_noise / n_total),
            'soft_noise_rate': float(n_soft / n_total),
            'hard_noise_rate': float(n_hard / n_total),
            'mean_noise_similarity': float(best_matches.mean()),
            **hard_noise_stats
        }

    class UnionFind:
        """Union-find data structure with path compression for cluster merging"""
        def __init__(self, elements):
            self.parent = {e: e for e in elements}
            self.rank = {e: 0 for e in elements}

        def find(self, x):
            """Find root with path compression"""
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            """Union by rank"""
            root_x, root_y = self.find(x), self.find(y)
            if root_x == root_y:
                return
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

        def get_components(self):
            """Return mapping of each element to its component representative"""
            return {e: self.find(e) for e in self.parent}

    def _merge_similar_clusters(self, labels: np.ndarray) -> np.ndarray:
        """
        Merge clusters using graph-based transitive closure with union-find.

        Args:
            labels: Initial cluster assignments

        Returns:
            Updated cluster assignments with merged clusters
        """
        if not self.clustering_config.merge_similar_clusters:
            return labels

        self.verbose_reporter.empty_line()
        self.verbose_reporter.section_header("CLUSTER MERGING")

        # Step 1: Compute centroids and build cluster membership
        centroids, sizes = self._compute_cluster_centroids(labels)
        n_initial_clusters = len(centroids)

        if n_initial_clusters < 2:
            self.verbose_reporter.stat_line("Less than 2 clusters - skipping merge")
            return labels

        self.verbose_reporter.stat_line(f"Initial clusters: {n_initial_clusters}")
        self.verbose_reporter.stat_line(f"Centroid threshold: {self.clustering_config.merge_centroid_threshold}")
        self.verbose_reporter.stat_line(f"Pairwise threshold: {self.clustering_config.merge_pairwise_threshold}")

        # Build cluster_to_indices mapping
        cluster_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:
                cluster_to_indices[int(label)].append(i)

        # Step 2: Find ALL candidate pairs based on centroid similarity
        cluster_ids = sorted(centroids.keys())
        centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])

        # Compute centroid similarities (cosine = dot product for L2-normalized)
        centroid_similarities = centroid_matrix @ centroid_matrix.T

        candidates = []
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                sim = centroid_similarities[i, j]
                if sim >= self.clustering_config.merge_centroid_threshold:
                    candidates.append((cluster_ids[i], cluster_ids[j], sim))

        self.verbose_reporter.stat_line(f"Candidate pairs (centroid > {self.clustering_config.merge_centroid_threshold}): {len(candidates)}")

        if not candidates:
            self.verbose_reporter.stat_line("No similar clusters found - no merging needed")
            return labels

        # Step 3: Evaluate pairwise similarity for ALL candidates
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Evaluating candidate pairs:")

        merge_edges = []  # Store all merge-worthy pairs
        for cluster_a, cluster_b, centroid_sim in candidates:
            # Get item indices for both clusters
            indices_a = np.array(cluster_to_indices[cluster_a])
            indices_b = np.array(cluster_to_indices[cluster_b])

            # Calculate pairwise similarities
            stats = self._pairwise_cluster_similarity(indices_a, indices_b)

            # Merge decision based on quantile mean
            quantile_mean = np.mean([stats['q25'], stats['q50'], stats['q75']])

            if quantile_mean >= self.clustering_config.merge_pairwise_threshold:
                merge_edges.append((cluster_a, cluster_b, centroid_sim, quantile_mean, sizes[cluster_a], sizes[cluster_b]))
                self.verbose_reporter.stat_line(
                    f"  ✓ Edge {cluster_a}↔{cluster_b} | "
                    f"sizes: {sizes[cluster_a]}, {sizes[cluster_b]} | "
                    f"centroid: {centroid_sim:.3f} | "
                    f"quantile_mean: {quantile_mean:.3f}"
                )

        if not merge_edges:
            self.verbose_reporter.stat_line("No merge-worthy pairs found")
            return labels

        # Step 4: Use union-find to group clusters into connected components
        uf = self.UnionFind(cluster_ids)
        for cluster_a, cluster_b, _, _, _, _ in merge_edges:
            uf.union(cluster_a, cluster_b)

        # Get component mapping (each cluster -> its component representative)
        component_map = uf.get_components()

        # Step 5: Relabel all points to their component representative
        labels_merged = labels.copy()
        for old_id, component_id in component_map.items():
            labels_merged[labels == old_id] = component_id

        # Step 6: Renumber to sequential IDs
        labels_final = self._renumber_clusters(labels_merged)

        # Step 7: Report results
        unique_components = set(component_map.values())
        n_final_clusters = len(unique_components)
        n_merged = n_initial_clusters - n_final_clusters

        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Merging complete:")
        self.verbose_reporter.stat_line(f"  Initial clusters: {n_initial_clusters}")
        self.verbose_reporter.stat_line(f"  Merge edges: {len(merge_edges)}")
        self.verbose_reporter.stat_line(f"  Final clusters: {n_final_clusters}")
        self.verbose_reporter.stat_line(f"  Reduction: {n_merged} clusters removed")

        return labels_final
    
    def _evaluate_hdbscan(self, U: np.ndarray, ms: int, mcs: int) -> Dict[str, Any]:
        """Evaluate HDBSCAN configuration with kappa-based scoring and all metrics"""
        start_time = time.time()

        # Fit HDBSCAN (deterministic by design)
        db = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=self.CLUSTER_METRIC,
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.0,
            alpha=1.0
        ).fit(U)

        labels = db.labels_
        noise_rate = float(np.mean(labels == -1))
        n_clusters = int(np.unique(labels[labels >= 0]).size)

        # Group PCA embeddings by cluster for noise assessment
        # (Use pca_embedding from output_list - already L2-normalized, same as merge logic)
        pca_embeddings_array = np.vstack([item.pca_embedding for item in self.output_list])

        cluster_members = {}
        if n_clusters > 0:
            by_cluster = defaultdict(list)
            for vec, label in zip(pca_embeddings_array, labels):
                if label >= 0:
                    by_cluster[int(label)].append(vec)
            # Convert lists to arrays
            cluster_members = {cid: np.vstack(members) for cid, members in by_cluster.items()}

        # Assess noise quality (soft vs hard) using member-based quantile logic
        noise_breakdown = self._assess_noise_quality(pca_embeddings_array, labels, cluster_members)

        # Calculate ALL metrics including expensive ones
        M = self._calculate_metrics(U, labels, db)

        # Add all noise breakdown metrics (including distribution stats)
        M.update(noise_breakdown)

        # Calculate median cluster size
        _, counts = np.unique(labels[labels >= 0], return_counts=True)
        med_size = int(np.median(counts)) if counts.size else 0

        summary = ClusterSummary(
            n_points=U.shape[0],
            min_cluster_size=int(mcs),
            min_samples = int(ms),
            n_clusters=int(n_clusters),
            noise_rate=float(round(noise_rate, 3)),
            median_cluster_size=int(med_size),
        )

        elapsed_time = time.time() - start_time

        return {
            "mcs": mcs,
            "ms": ms,
            "hdbscan_model": db,
            "labels": labels,
            "metrics": M,
            "summary": summary,
            "elapsed_time": elapsed_time
        }
    
    def _calculate_metrics(self, U: np.ndarray, labels: np.ndarray, hdbscan_model: HDBSCAN) -> Dict[str, float]:
        
        # Mask for labeled points
        mask = labels >= 0
        X, y = U[mask], labels[mask]
        n_clusters = int(np.unique(y).size) if X.shape[0] > 0 else 0
        noise_rate = float(np.mean(labels == -1))
        
        # density based validation index 
        dbcv = (self._dbcv_score(U, labels) if (self.enable_dbcv and n_clusters >= 2) else None)
   
        # mean probability of clusster membership
        has_probs = hasattr(hdbscan_model, "probabilities_") and hdbscan_model.probabilities_ is not None
        has_members = np.any(labels != -1)
        meanp = (self._mean_probability(hdbscan_model, labels) if (self.enable_meanp and has_probs and has_members) else None)
   
        # mean distance to cluster centroid embedding
        if self.centroid_distance and n_clusters >= 1:
            topk = min(TOP_K_CLUSTERS_DEFAULT, n_clusters)
            cdist, cdist5 = self._centroid_distance_topk(U, labels, topk_clusters=topk)
        else:
            cdist, cdist5 = None, None
    
        # silhouette score for distinctiveness/seperation/exlusiveness
        sil = self._silhouette_score(U, labels)
          
        # Size-weighted cluster persistence 
        stab = self._cluster_stability(hdbscan_model, labels)

        # Per-sample silhouettes 
        frac_low_points = 0.0
        frac_low_clusters = 0.0 
        if X.shape[0] and n_clusters >= 2:
            sil_samples = silhouette_samples(X, y, metric=self.CLUSTER_METRIC)
            y_sample = y
            frac_low_points = float((sil_samples < SILHOUETTE_LOW_THRESHOLD).mean())
            clusters, counts = np.unique(y_sample, return_counts=True)
            cluster_means = []
            for c in clusters:
                cluster_mask = (y_sample == c)
                cluster_sil = sil_samples[cluster_mask]
                cluster_means.append(cluster_sil.mean())
            cluster_means = np.array(cluster_means)
            frac_low_clusters = float((cluster_means < SILHOUETTE_LOW_THRESHOLD).mean())
        
        # CH / DB on U (non-noise)
        geom = self._geom_indices(U, labels)
           
        return {
            "n_clusters": n_clusters,           # number of clusters (labels >= 0)
            "dbcv": dbcv,                       # density-based clustering validation (higher=better; may be None) 
            "noise_rate": noise_rate,           # fraction of points labeled -1
            "meanp": meanp,                     # mean membership probability over clustered points
            "sil": sil,                         # silhouetee score - for distinctiveness/separation
            "CH": geom["CH"],                   # Calinski–Harabasz (higher=better)  - overall variance-explained metric.
            "DB": geom["DB"],                   # Davies–Bouldin (lower=better) - for cluster overlap metric
            "cdist": cdist, "cdist5": cdist5,   # mean distance of ALL/TOP5 centroid pairs (off-diagonal)
            "stab": stab,                       # Size-weighted cluster persistence (0=balanced, 1=skewed)
            "frac_low_points": frac_low_points, # share of points with sil < threshold (e.g., 0.30)
            "frac_low_clusters": frac_low_clusters, # share of clusters with mean sil < threshold
         }
                
        
    def _grid_search(self, U: np.ndarray, ms_grid: List[int], mcs: int) -> List[Dict[str, Any]]:
        """Execute grid search in parallel using ThreadPoolExecutor"""
        
        max_workers = self.clustering_config.grid_search_max_workers
       
        if max_workers is None: 
            max_workers = max(1, multiprocessing.cpu_count() - 1) # Auto-detect: use CPU count - 1, but at least 1
        elif max_workers == -1: 
            max_workers = multiprocessing.cpu_count() # Use all available cores
        
        timeout = self.clustering_config.grid_search_timeout_seconds
        
        self.verbose_reporter.stat_line(f"Running evaluation in parallel with {max_workers} workers")
        
        try: # Use ThreadPoolExecutor for better memory sharing with numpy arrays
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                start_time = time.time()
                future_to_ms = {executor.submit(self._evaluate_hdbscan, U, ms, mcs): ms for ms in ms_grid}
                results = []  
                completed = 0
                total = len(ms_grid)
                
                for future in concurrent.futures.as_completed(future_to_ms, timeout=timeout):
                    ms = future_to_ms[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        if self.verbose_reporter.enabled:
                            self.verbose_reporter.stat_line(
                                f"  Completed {completed}/{total} configs "
                                f"(ms={ms}) "
                            )
                    except Exception as e:
                        self.verbose_reporter.stat_line(f"  Failed to evaluate mcs={mcs}: {str(e)}")
                        # Continue with other configurations
                
                elapsed_total = time.time() - start_time
                self.verbose_reporter.stat_line(f"Parallel evaluation completed in {elapsed_total:.1f}s")
                
                return results
                
        except concurrent.futures.TimeoutError:
            self.verbose_reporter.stat_line(f"Parallel grid search timed out after {timeout}s")
            raise
        except Exception as e:
            self.verbose_reporter.stat_line(f"Parallel grid search failed: {str(e)}")
            raise
    
    def _scale_metric01(self,
        x, 
        winsor=3.0, # cap z-scores at ±winsor
        temperature=1.5        # ↓ makes tails pop more (e.g., 1.0 = stronger, 2.0 = milder)
        ):

        x = np.asarray(x, float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        # 1.4826 makes MAD ~ std for Gaussian
        scale = (1.4826 * mad) if mad > 0 else (np.nanstd(x, ddof=1) or 1.0)
        z = (x - med) / scale
        z = np.clip(z, -winsor, winsor)
        y = 1.0 / (1.0 + np.exp(-z / max(temperature, 1e-9)))
    
        return np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0)

    def _structure_factor_from_space(self, U: np.ndarray, subsample: int = 3000, knn_k: int = 15) -> tuple[float, str]:
        """
        Data-driven scaling factor from the actual embedding space U.
        factor < 1 -> go smaller/finer; factor > 1 -> go larger/coarser.
        Uses q90 of pairwise cosine similarity and CV of kNN distances.
        """
        rs = self.rs 
        X = U
        n = X.shape[0]
    
        # L2-normalize so dot product ≈ cosine on U
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Xn = Xn.astype(np.float32, copy=False)
    
        # Subsample for diagnostics speed
        if n > subsample:
            idx = rs.choice(n, subsample, replace=False)
            Xsub = Xn[idx]
        else:
            Xsub = Xn
    
        m = min(Xsub.shape[0], 2000)
        idy = rs.choice(Xsub.shape[0], m, replace=False)
        Y = Xsub[idy]
    
        # q90 cosine similarity
        S = Y @ Y.T
        tri = S[np.triu_indices_from(S, k=1)]
        q90 = float(np.quantile(tri, 0.90))
    
        # kNN distances (1 - cosine)
        D = 1.0 - (Xsub @ Xsub.T)
        np.fill_diagonal(D, np.inf)
        knn_k = int(max(5, min(knn_k, max(5, Xsub.shape[0] - 1))))
        knn_d = np.partition(D, knn_k, axis=1)[:, :knn_k]
        kd_mean = float(np.mean(knn_d))
        kd_std  = float(np.std(knn_d))
        kd_cv   = float(kd_std / (kd_mean + 1e-12))
    
        # Map diagnostics -> factor
        f = 1.0
        notes = [f"Cosine similarity: q90={q90:.2f}", f"Coefficient of variation ={kd_cv:.2f}"]
    
        # Dense space → smaller params; Sparse → larger
        if q90 >= 0.65:   f *= 0.80; notes.append("dense ×0.80")
        elif q90 < 0.50:  f *= 1.20; notes.append("sparse ×1.20")
        else:             notes.append("moderate ×1.00")
    
        # Heterogeneous densities → preserve tight islands
        if kd_cv >= 0.60: f *= 0.85; notes.append("var.density ×0.85")
        elif kd_cv <= 0.30: f *= 1.10; notes.append("uniform ×1.10")
    
        f = float(np.clip(f, 0.50, 1.50))
        notes.append(f"factor={f:.2f}")
        return f, "; ".join(notes)


    def _detect_data_regime(self, U: np.ndarray, n_points: int) -> Dict[str, Any]:
        """Detect data regime to guide clustering strategy.

        Returns regime classification and characteristics for adaptive strategy selection.
        Uses simplified regime detection based on size × structure matrix.
        """
        # === SIZE CLASSIFICATION ===
        if n_points < 100:
            size_class = "small"
        elif n_points <= 300:
            size_class = "medium"
        else:
            size_class = "large"

        # === STRUCTURE CLASSIFICATION ===
        # Reuse existing space diagnostics (from _structure_factor_from_space logic)
        rs = self.rs
        Xn = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        Xn = Xn.astype(np.float32, copy=False)

        # Subsample for speed
        subsample = min(2000, n_points)
        if n_points > subsample:
            idx = rs.choice(n_points, subsample, replace=False)
            Xsub = Xn[idx]
        else:
            Xsub = Xn

        m = min(Xsub.shape[0], 1000)
        idy = rs.choice(Xsub.shape[0], m, replace=False)
        Y = Xsub[idy]

        # Cosine similarity diagnostics
        S = Y @ Y.T
        tri = S[np.triu_indices_from(S, k=1)]
        q90 = float(np.quantile(tri, 0.90))
        q50 = float(np.quantile(tri, 0.50))

        # kNN distance variance (measure of density heterogeneity)
        D = 1.0 - (Xsub @ Xsub.T)
        np.fill_diagonal(D, np.inf)
        knn_k = min(15, max(5, Xsub.shape[0] - 1))
        knn_d = np.partition(D, knn_k, axis=1)[:, :knn_k]
        kd_mean = float(np.mean(knn_d))
        kd_std = float(np.std(knn_d))
        kd_cv = float(kd_std / (kd_mean + 1e-12))

        # Structure classification based on q90
        if q90 >= 0.80:
            structure_class = "coherent"  # Dense, similar ideas
        elif q90 < 0.65:
            structure_class = "diffuse"   # Sparse, distinct ideas
        else:
            structure_class = "mixed"     # Moderate structure

        # === REGIME ASSIGNMENT ===
        regime_map = {
            ("small", "diffuse"): "R1",
            ("small", "mixed"): "R2",
            ("small", "coherent"): "R3",
            ("medium", "diffuse"): "R4",
            ("medium", "mixed"): "R5",
            ("medium", "coherent"): "R6",
            ("large", "diffuse"): "R7",
            ("large", "mixed"): "R8",
            ("large", "coherent"): "R9",
        }

        regime_id = regime_map[(size_class, structure_class)]

        # === REGIME DESCRIPTIONS & GOALS ===
        regime_descriptions = {
            "R1": "Small+Diffuse: Seek broad holistic themes",
            "R2": "Small+Mixed: Balance consolidation with distinctions",
            "R3": "Small+Coherent: Consolidate for quality (broad, stable themes)",
            "R4": "Medium+Diffuse: Maximize coverage with balanced clusters",
            "R5": "Medium+Mixed: Standard case (sweet spot)",
            "R6": "Medium+Coherent: Preserve fine distinctions, high separation",
            "R7": "Large+Diffuse: Many specific clusters with clear boundaries",
            "R8": "Large+Mixed: Hierarchical with good separation",
            "R9": "Large+Coherent: Aggressive consolidation to avoid over-splitting",
        }

        regime_goals = {
            "R1": "broad_themes",
            "R2": "balanced",
            "R3": "consolidation",
            "R4": "coverage",
            "R5": "standard",
            "R6": "fine_distinctions",
            "R7": "fine_distinctions",
            "R8": "balanced",
            "R9": "consolidation",
        }

        return {
            'regime_id': regime_id,
            'size_class': size_class,
            'structure_class': structure_class,
            'description': regime_descriptions[regime_id],
            'regime_goal': regime_goals[regime_id],
            'n_points': n_points,
            'q90': q90,
            'q50': q50,
            'kd_cv': kd_cv,
            'diagnostics': f"n={n_points}, q90={q90:.2f}, q50={q50:.2f}, cv={kd_cv:.2f}"
        }


    def _baseline_by_n(self, n: int) -> tuple[int, int, str]:
        """
        Calculate baseline parameters using sqrt-based formula:
        baseline = max(5, 0.25 * sqrt(n))

        This provides:
        - Stable floor at 5 for small datasets (n < 400)
        - Scales appropriately for larger datasets
        - Same baseline used as starting point for both mcs and ms
        """
        baseline = max(5, int(np.ceil(0.25 * np.sqrt(n))))

        # Both start from same baseline (will be differentiated by regime factors and ms_ratio)
        mcs = baseline
        ms = baseline

        return ms, mcs, f"baseline(sqrt formula)={baseline} → ms={ms}, mcs={mcs}"
    
    
    def _suggest_params(self, U: np.ndarray, min_ms: int = 2, min_mcs: int = None, max_mcs: int = 250) -> tuple[int, int, str, str, Dict[str, Any]]:
        """
        Regime-aware parameter suggestion using 3-step approach with dual metric control:

        Step 1: Calculate baseline using sqrt formula: max(5, 0.25*sqrt(n))
        Step 2a: Detect regime (q90 for coherence, kd_cv for density heterogeneity)
        Step 2b: Apply q90-based factor to min_cluster_size
        Step 2c: Apply kd_cv-based ratio to min_samples
        Step 3: Grid search explores factor-based variations

        Dual metric rationale:
        - q90 (cosine similarity) → controls min_cluster_size scale (semantic coherence)
        - kd_cv (kNN distance variation) → controls min_samples ratio (density heterogeneity)

        Returns: (ms, mcs, notes_regime, notes_baseline, regime_info)
        """
        n = U.shape[0]

        # Adaptive floor based on dataset size
        if min_mcs is None:
            if n <= 100:
                min_mcs = 2
            elif n <= 300:
                min_mcs = 3
            elif n <= 1000:
                min_mcs = 5
            elif n <= 5000:
                min_mcs = 8
            else:
                min_mcs = 10

        # === STEP 1: Baseline calculation (sqrt formula) ===
        ms0, mcs0, notes_baseline = self._baseline_by_n(n)

        # === STEP 2a: Detect regime ===
        regime_info = self._detect_data_regime(U, n)
        q90 = regime_info['q90']
        kd_cv = regime_info['kd_cv']

        # === STEP 2b: Apply q90-based factor to min_cluster_size ===
        if q90 >= 0.80:
            # Coherent data: LARGER mcs for consolidation
            mcs_factor = 1.5
            mcs_rationale = "coherent (q90≥0.80) → ×1.5 for consolidation"
        elif q90 < 0.65:
            # Diffuse data: SMALLER mcs to capture distinct clusters
            mcs_factor = 0.7
            mcs_rationale = "diffuse (q90<0.65) → ×0.7 for distinction"
        else:
            # Mixed: baseline unchanged
            mcs_factor = 1.0
            mcs_rationale = "mixed (0.65≤q90<0.80) → ×1.0 baseline"

        mcs_adjusted = mcs0 * mcs_factor
        mcs = int(np.clip(int(round(mcs_adjusted)), min_mcs, max_mcs))

        # === STEP 2c: Apply kd_cv-based ratio to min_samples ===
        # High kd_cv → tight islands in sparse space → allow lower min_samples
        # Low kd_cv → uniform density → higher min_samples prevents over-merging
        if kd_cv > 0.8:
            # Strong "islands" - heterogeneous density
            ms_ratio = 0.4
            ms_rationale = "heterogeneous (kd_cv>0.8) → ms=0.4×mcs"
        else:
            # Uniform density
            ms_ratio = 1.0
            ms_rationale = "uniform (kd_cv≤0.8) → ms=1.0×mcs"

        ms_adjusted = mcs * ms_ratio
        ms = max(min_ms, int(np.clip(int(round(ms_adjusted)), min_ms, mcs)))

        # Construct notes
        notes_regime = (
            f"Regime: {regime_info['regime_id']} ({regime_info['description']}); "
            f"{mcs_rationale}; {ms_rationale}; "
            f"baseline={mcs0} → mcs={mcs}, ms={ms}"
        )

        return ms, mcs, notes_regime, notes_baseline, regime_info


    def _generate_grid_pairs(self, mcs_baseline: int, ms_ratio: float, regime_goal: str,
                            min_mcs: int, min_ms: int) -> List[Tuple[int, int]]:
        """
        Generate factor-based grid for (min_samples, min_cluster_size) exploration.

        Applies multiplicative factors to baseline, then splits into ms and mcs
        using the provided ratio. This maintains consistent scaling across both parameters.

        Uses multiplicative factors that scale with dataset size:
        - Consolidation regimes: [0.5, 1.0, 1.5, 2.0] (wide exploration)
        - Balanced regimes: [0.75, 1.0, 1.5] (moderate exploration)
        - Fine-tune regimes: [0.75, 1.0, 1.25] (narrow exploration)

        Example for consolidation with baseline=8, ratio=0.4:
        factors = [0.5, 1.0, 1.5, 2.0]
        Factor 0.5:  8×0.5=4   → mcs=4,  ms=4×0.4=2   → (2, 4)
        Factor 1.0:  8×1.0=8   → mcs=8,  ms=8×0.4=3   → (3, 8)
        Factor 1.5:  8×1.5=12  → mcs=12, ms=12×0.4=5  → (5, 12)
        Factor 2.0:  8×2.0=16  → mcs=16, ms=16×0.4=6  → (6, 16)

        Args:
            mcs_baseline: Regime-suggested min_cluster_size baseline
            ms_ratio: Ratio of min_samples to min_cluster_size (e.g., 0.4 for coherent data)
            regime_goal: Strategy goal ('consolidation', 'balanced', 'fine_distinctions', etc.)
            min_mcs: Minimum allowed min_cluster_size (adaptive floor)
            min_ms: Minimum allowed min_samples

        Returns:
            Sorted list of unique (ms, mcs) tuples for grid search
        """
        # Select factors based on regime goal
        if regime_goal in ['consolidation', 'broad_themes']:
            # Wide exploration for consolidation strategies
            factors = [0.5, 1.0, 1.5, 2.0]
        elif regime_goal in ['balanced', 'standard', 'coverage']:
            # Moderate exploration
            factors = [0.75, 1.0, 1.5]
        else:
            # Narrow fine-tuning for distinction-preserving strategies
            factors = [0.75, 1.0, 1.25]

        # Generate (ms, mcs) pairs by applying factors
        grid_pairs = []
        for factor in factors:
            # Apply factor to baseline
            scaled_mcs = max(min_mcs, int(round(factor * mcs_baseline)))

            # Calculate ms using the ratio
            scaled_ms = max(min_ms, int(round(scaled_mcs * ms_ratio)))

            # Ensure ms <= mcs (HDBSCAN requirement)
            scaled_ms = min(scaled_ms, scaled_mcs)

            grid_pairs.append((scaled_ms, scaled_mcs))

        # Remove duplicates and sort by mcs (second element)
        unique_pairs = sorted(set(grid_pairs), key=lambda x: x[1])

        return unique_pairs

    def _generate_expansion_grid(
        self,
        winner: dict,
        dbcv_best: float,
        all_results: list,
        ms_ratio: float,
        min_mcs: int,
        min_ms: int
    ) -> List[Tuple[int, int]]:
        """
        Generate expansion candidates based on winner's weaknesses.

        Step 5 expansion logic:
        - If DBCV < 0.60: try smaller mcs (seek coherent structure)
        - If hard_noise > 0.10: try smaller ms (reduce orphaning)

        Args:
            winner: Best config from initial grid search
            dbcv_best: Maximum DBCV across all initial results
            all_results: All results from initial grid search
            ms_ratio: Ratio of min_samples to min_cluster_size
            min_mcs: Minimum allowed min_cluster_size
            min_ms: Minimum allowed min_samples

        Returns:
            List of (ms, mcs) pairs to evaluate, excluding already-tried configs
        """
        expansion_pairs = []

        # Trigger 1: Low DBCV → try smaller mcs (seek coherent structure)
        if dbcv_best < 0.60:
            # Find config with best DBCV to use as anchor
            cfg_dbcv_best = max(all_results, key=lambda r: r['metrics'].get('dbcv', -1))
            base_mcs = cfg_dbcv_best['mcs']
            for factor in [0.8, 0.67, 0.5]:
                new_mcs = max(min_mcs, int(round(factor * base_mcs)))
                new_ms = max(min_ms, int(round(new_mcs * ms_ratio)))
                new_ms = min(new_ms, new_mcs)  # Ensure ms <= mcs
                expansion_pairs.append((new_ms, new_mcs))

        # Trigger 2: High noise → try smaller ms for ALL mcs sizes in original grid
        winner_noise = winner.get('hard_noise', winner.get('metrics', {}).get('hard_noise_rate', 0))
        if winner_noise > 0.10:
            for r in all_results:
                ms_orig, mcs_orig = r['ms'], r['mcs']
                new_ms = max(min_ms, int(round(0.5 * ms_orig)))
                if new_ms < ms_orig:  # Only add if actually smaller
                    expansion_pairs.append((new_ms, mcs_orig))

        # Remove duplicates and configs already in original grid
        existing = {(r['ms'], r['mcs']) for r in all_results}
        unique_new = sorted(set(expansion_pairs) - existing, key=lambda x: (x[1], x[0]))

        return unique_new

    def _select_final_config(self, all_results: list, original_winner: dict) -> Tuple[dict, str]:
        """
        Select final config with gated acceptance for smaller params.

        Policy:
        - Prefer the original broad/holistic clustering by default
        - Allow smaller min_cluster_size only if DBCV >= 0.65 (must prove validity)
        - Allow smaller min_samples only if hard_noise <= 0.10 (must fix noise)
        - If a config changes both, it must satisfy both conditions
        - Among eligible configs, select highest composite score
        - If none qualify, fall back to original winner

        Args:
            all_results: All results (initial + expansion)
            original_winner: Best config from initial grid (before expansion)

        Returns:
            Tuple of (selected config, selection reason)
        """
        eligible = []
        original_mcs = original_winner['mcs']
        original_ms = original_winner['ms']

        for r in all_results:
            ok = True

            # Smaller mcs must prove validity
            if r['mcs'] < original_mcs:
                r_dbcv = r.get('dbcv', r.get('metrics', {}).get('dbcv', 0))
                ok &= (r_dbcv >= 0.65)

            # Smaller ms must fix noise
            if r['ms'] < original_ms:
                r_noise = r.get('hard_noise', r.get('metrics', {}).get('hard_noise_rate', 1.0))
                ok &= (r_noise <= 0.10)

            if ok:
                eligible.append(r)

        # Fallback: if everything filtered out, keep original winner
        if not eligible:
            return original_winner, "fallback to original (no config passed gates)"

        final = max(eligible, key=lambda r: r['score'])

        # Determine selection reason
        if final['mcs'] < original_mcs and final['ms'] < original_ms:
            reason = f"smaller mcs+ms passed both gates (DBCV={final.get('dbcv', 0):.2f}>=0.65, noise={final.get('hard_noise', 0):.1%}<=10%)"
        elif final['mcs'] < original_mcs:
            reason = f"smaller mcs passed DBCV gate ({final.get('dbcv', 0):.2f}>=0.65)"
        elif final['ms'] < original_ms:
            reason = f"smaller ms passed noise gate ({final.get('hard_noise', 0):.1%}<=10%)"
        elif final == original_winner:
            reason = "original winner (best eligible score)"
        else:
            reason = f"best eligible score ({final['score']:.3f})"

        return final, reason


    @staticmethod
    def _apply_threshold_rule(ms: int, mcs: int, dbcv: Optional[float], hard_noise: float,
                              n_points: int = None,
                              min_ms: int = 2, min_mcs: int = None,
                              dbcv_cut: float = 0.50, noise_cut: float = 0.15) -> tuple[int, int, str, bool]:
        """
        Polish loop noise handling (Step 4 of regime-aware approach):
        If hard_noise > threshold: ONLY decrease min_samples (by 0.7×), keep min_cluster_size unchanged.

        This follows HDBSCAN best practice: high noise → lower sample requirement to recover noisy points.
        We do NOT change min_cluster_size because that's set by the regime strategy.

        Adaptive floor: adjusts min_mcs based on dataset size if n_points provided.
        hard_noise = fraction of noise points with low similarity to all clusters (true outliers).
        Returns (new_ms, new_mcs, note, changed_flag).
        """
        # Calculate adaptive floor if not provided
        if min_mcs is None and n_points is not None:
            if n_points <= 100:
                min_mcs = 2
            elif n_points <= 300:
                min_mcs = 3
            elif n_points <= 1000:
                min_mcs = 5
            elif n_points <= 5000:
                min_mcs = 8
            else:
                min_mcs = 10
        elif min_mcs is None:
            min_mcs = 5  # Fallback to original default

        trigger = False
        note_parts = []
        if hard_noise is not None and hard_noise > noise_cut:
            trigger = True
            note_parts.append(f"hard_noise {hard_noise:.2f}>{noise_cut:.2f}")
        if (dbcv is not None) and (dbcv < dbcv_cut):
            trigger = True
            note_parts.append(f"dbcv {dbcv:.2f}<{dbcv_cut:.2f}")

        if trigger:
            # ONLY decrease min_samples (by 0.7×), keep min_cluster_size unchanged
            ms_new = max(min_ms, int(np.ceil(0.7 * ms)))
            mcs_new = mcs  # Keep mcs unchanged! Regime strategy sets this.
            note_parts.append(f"reduce ms only: {ms}→{ms_new}, mcs unchanged={mcs}")
            return ms_new, mcs_new, "; ".join(note_parts), True

        return ms, mcs, "no change", False
    

    def _auto_hdbscan_grid(self, U: np.ndarray, original_embeddings: Optional[np.ndarray] = None) -> Tuple[HDBSCAN, np.ndarray, ClusterSummary]:
        """Auto-tune HDBSCAN parameters using grid search with polish refinement.

        Args:
            U: UMAP-reduced embeddings array (used for HDBSCAN clustering)
            original_embeddings: Original embeddings (used for regime detection to avoid UMAP artifacts)

        Returns:
            Tuple of (fitted HDBSCAN model, cluster labels, ClusterSummary)
        """
        # Get regime-aware parameter suggestion
        # Use original embeddings for regime detection to avoid UMAP q90=1.00 artifact
        embeddings_for_regime = original_embeddings if original_embeddings is not None else U
        n = U.shape[0]
        ms, mcs, notes_regime, notes_baseline, regime_info = self._suggest_params(embeddings_for_regime)

        # Determine adaptive floor for logging
        if n <= 100:
            floor_note = f"n={n} → adaptive floor: min_mcs=2 (small dataset)"
        elif n <= 300:
            floor_note = f"n={n} → adaptive floor: min_mcs=3 (small-medium dataset)"
        elif n <= 1000:
            floor_note = f"n={n} → adaptive floor: min_mcs=5 (medium dataset)"
        elif n <= 5000:
            floor_note = f"n={n} → adaptive floor: min_mcs=8 (large dataset)"
        else:
            floor_note = f"n={n} → adaptive floor: min_mcs=10 (very large dataset)"

        # Report regime detection and parameter suggestion
        self.verbose_reporter.empty_line()
        self.verbose_reporter.section_header("REGIME-AWARE PARAMETER SELECTION")
        self.verbose_reporter.stat_line(f"Detected regime: {regime_info['regime_id']} - {regime_info['description']}")
        self.verbose_reporter.stat_line(f"Diagnostics: {regime_info['diagnostics']}")
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("3-Step Parameter Selection:")
        self.verbose_reporter.stat_line(f"  [Adaptive] {floor_note}")
        self.verbose_reporter.stat_line(f"  [Step 1 - Baseline] {notes_baseline}")
        self.verbose_reporter.stat_line(f"  [Step 2 - Regime] {notes_regime}")

        # Adaptive floor for mcs grid
        if n <= 100:
            min_mcs_floor = 2
        elif n <= 300:
            min_mcs_floor = 3
        elif n <= 1000:
            min_mcs_floor = 5
        elif n <= 5000:
            min_mcs_floor = 8
        else:
            min_mcs_floor = 10

        # === STEP 3: Factor-based grid search ===
        # Generate grid using regime-specific factors applied to BOTH ms and mcs baseline
        regime_goal = regime_info.get('regime_goal', 'balanced')  # From regime strategy

        # Recalculate ms_ratio (same logic as in _suggest_params)
        kd_cv = regime_info.get('kd_cv', 0.0)
        ms_ratio = 0.4 if kd_cv > 0.8 else 1.0

        grid_pairs = self._generate_grid_pairs(mcs, ms_ratio, regime_goal, min_mcs_floor, min_ms=2)

        self.verbose_reporter.stat_line(f"  [Step 3 - Grid] Factor-based grid for goal '{regime_goal}': {grid_pairs}")
        self.verbose_reporter.empty_line()

        # Round 0: evaluate starters (manually iterate over mcs_grid)
        max_workers = self.clustering_config.grid_search_max_workers or max(1, multiprocessing.cpu_count() - 1)
        timeout = self.clustering_config.grid_search_timeout_seconds

        self.verbose_reporter.stat_line(f"Running evaluation in parallel with {max_workers} workers")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                start_time = time.time()
                future_to_config = {
                    executor.submit(self._evaluate_hdbscan, U, ms_val, mcs_val): (ms_val, mcs_val)
                    for ms_val, mcs_val in grid_pairs
                }
                results = []
                completed = 0
                total = len(grid_pairs)

                for future in concurrent.futures.as_completed(future_to_config, timeout=timeout):
                    ms_val, mcs_val = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1

                        if self.verbose_reporter.enabled:
                            self.verbose_reporter.stat_line(
                                f"  Completed {completed}/{total} configs "
                                f"(ms={ms_val}, mcs={mcs_val})"
                            )
                    except Exception as e:
                        self.verbose_reporter.stat_line(f"  Failed to evaluate (ms={ms_val}, mcs={mcs_val}): {str(e)}")
                        # Continue with other configurations

                elapsed_total = time.time() - start_time
                self.verbose_reporter.stat_line(f"Parallel evaluation completed in {elapsed_total:.1f}s")

        except concurrent.futures.TimeoutError:
            self.verbose_reporter.stat_line(f"Parallel grid search timed out after {timeout}s")
            raise
        except Exception as e:
            self.verbose_reporter.stat_line(f"Parallel grid search failed: {str(e)}")
            raise
        if not results:
            raise RuntimeError("All clustering configurations failed. No valid results to evaluate.")
    
        # === STEP 4: DBCV-primary composite scoring ===
        # DBCV is primary metric (HDBSCAN is density-based)
        # Silhouette weight depends on kd_cv (density uniformity)
        # Penalties are lightweight guardrails, not drivers

        kd_cv = regime_info.get('kd_cv', 0.0)

        # Extract raw metrics
        dbcv = np.array([r["metrics"].get("dbcv", 0.0) for r in results], dtype=float)
        sil = np.clip([r["metrics"].get("sil", 0.0) for r in results], 0, 1)
        hard_noise = np.array([r["metrics"].get("hard_noise_rate",
                              r["metrics"].get("noise_rate", 0.0)) for r in results], dtype=float)
        k = np.array([r["metrics"].get("n_clusters", 1) for r in results], dtype=float)

        # Base score: kd_cv-adaptive weighting of DBCV vs silhouette
        # High kd_cv → heterogeneous density → trust DBCV more
        # Low kd_cv → uniform density → silhouette is meaningful
        if kd_cv > 0.6:
            # Heterogeneous density ("islands") - DBCV dominant
            base_score = 0.90 * dbcv + 0.10 * sil
            weight_rationale = f"kd_cv={kd_cv:.2f}>0.6 → DBCV-dominant (0.90/0.10)"
        else:
            # Uniform density - balanced weighting
            base_score = 0.50 * dbcv + 0.50 * sil
            weight_rationale = f"kd_cv={kd_cv:.2f}≤0.6 → balanced (0.50/0.50)"

        # Noise penalty: only penalize excess above 10% tolerance
        noise_tolerance = 0.10
        excess_noise = np.maximum(0.0, hard_noise - noise_tolerance)
        noise_penalty = 0.50 * excess_noise

        # Symmetric k penalty: penalize deviation from k_target (both under and over)
        k_target = max(3, n // mcs)
        k_dev = np.abs(k - k_target) / k_target
        k_penalty = 0.10 * k_dev

        # Final score
        final_score = base_score - noise_penalty - k_penalty

        self.verbose_reporter.stat_line(f"  [Step 4 - Score] {weight_rationale}")
        self.verbose_reporter.stat_line(f"  [Step 4 - Score] k_target={k_target} (from regime mcs={mcs})")

        for i, r in enumerate(results):
            r["score"] = float(final_score[i])
            r["score_base"] = float(base_score[i])
            r["dbcv"] = float(dbcv[i])
            r["sil"] = float(sil[i])
            r["noise_penalty"] = float(noise_penalty[i])
            r["k_penalty"] = float(k_penalty[i])
            r["hard_noise"] = float(hard_noise[i])
            r["k"] = int(k[i])
            r["k_target"] = k_target

        results.sort(key=lambda r: r["score"], reverse=True)

        # Report grid search results
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Grid search results (DBCV-primary scoring):")
        for i, r in enumerate(results[:5]):  # Show top 5
            marker = "→" if i == 0 else " "
            self.verbose_reporter.stat_line(
                f"  {marker} ms={r['ms']:>2}, mcs={r['mcs']:>2} | "
                f"k={r['k']:>2} | score={r['score']:.3f} | "
                f"base={r['score_base']:.3f} (dbcv={r['dbcv']:.2f}, sil={r['sil']:.2f}) | "
                f"pen: noise={r['noise_penalty']:.3f}, k={r['k_penalty']:.3f}"
            )

        # === STEP 5: CONDITIONAL GRID EXPANSION ===
        # Triggers:
        # - DBCV < 0.60: try smaller mcs (seek coherent structure)
        # - hard_noise > 0.10: try smaller ms (reduce orphaning)

        # Get initial winner (before expansion)
        initial_winner = results[0]
        dbcv_best = max(r['metrics'].get('dbcv', -1) for r in results)
        winner_noise = initial_winner.get('hard_noise', initial_winner['metrics'].get('hard_noise_rate', 0))

        # Check if expansion is needed
        needs_mcs_expansion = dbcv_best < 0.60
        needs_ms_expansion = winner_noise > 0.10

        if needs_mcs_expansion or needs_ms_expansion:
            self.verbose_reporter.empty_line()
            self.verbose_reporter.stat_line("[Step 5 - Expansion] Checking expansion triggers:")
            if needs_mcs_expansion:
                self.verbose_reporter.stat_line(f"  → DBCV={dbcv_best:.3f} < 0.60: will try smaller mcs")
            if needs_ms_expansion:
                self.verbose_reporter.stat_line(f"  → hard_noise={winner_noise:.1%} > 10%: will try smaller ms")

            # Generate expansion candidates
            expansion_grid = self._generate_expansion_grid(
                winner=initial_winner,
                dbcv_best=dbcv_best,
                all_results=results,
                ms_ratio=ms_ratio,
                min_mcs=min_mcs_floor,
                min_ms=2
            )

            if expansion_grid:
                self.verbose_reporter.stat_line(f"  Expansion candidates: {expansion_grid}")

                # Evaluate expansion grid in parallel
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_config = {
                            executor.submit(self._evaluate_hdbscan, U, ms_val, mcs_val): (ms_val, mcs_val)
                            for ms_val, mcs_val in expansion_grid
                        }
                        expansion_results = []

                        for future in concurrent.futures.as_completed(future_to_config, timeout=timeout):
                            ms_val, mcs_val = future_to_config[future]
                            try:
                                result = future.result()
                                expansion_results.append(result)
                                self.verbose_reporter.stat_line(f"  Evaluated expansion (ms={ms_val}, mcs={mcs_val})")
                            except Exception as e:
                                self.verbose_reporter.stat_line(f"  Failed (ms={ms_val}, mcs={mcs_val}): {str(e)}")

                    # Merge expansion with original and re-score ALL together
                    if expansion_results:
                        # Score expansion results using same raw scoring
                        exp_dbcv = np.array([r["metrics"].get("dbcv", 0.0) for r in expansion_results], dtype=float)
                        exp_sil = np.clip([r["metrics"].get("sil", 0.0) for r in expansion_results], 0, 1)
                        exp_hard_noise = np.array([r["metrics"].get("hard_noise_rate",
                                          r["metrics"].get("noise_rate", 0.0)) for r in expansion_results], dtype=float)
                        exp_k = np.array([r["metrics"].get("n_clusters", 1) for r in expansion_results], dtype=float)

                        # Same kd_cv-adaptive weighting as main scoring
                        if kd_cv > 0.6:
                            exp_base_score = 0.90 * exp_dbcv + 0.10 * exp_sil
                        else:
                            exp_base_score = 0.50 * exp_dbcv + 0.50 * exp_sil

                        # Same noise penalty
                        exp_excess_noise = np.maximum(0.0, exp_hard_noise - noise_tolerance)
                        exp_noise_penalty = 0.50 * exp_excess_noise

                        # Same symmetric k penalty
                        exp_k_dev = np.abs(exp_k - k_target) / k_target
                        exp_k_penalty = 0.10 * exp_k_dev

                        exp_final_score = exp_base_score - exp_noise_penalty - exp_k_penalty

                        for i, r in enumerate(expansion_results):
                            r["score"] = float(exp_final_score[i])
                            r["score_base"] = float(exp_base_score[i])
                            r["dbcv"] = float(exp_dbcv[i])
                            r["sil"] = float(exp_sil[i])
                            r["noise_penalty"] = float(exp_noise_penalty[i])
                            r["k_penalty"] = float(exp_k_penalty[i])
                            r["hard_noise"] = float(exp_hard_noise[i])
                            r["k"] = int(exp_k[i])
                            r["k_target"] = k_target

                        # Report expansion results
                        expansion_results.sort(key=lambda r: r["score"], reverse=True)
                        self.verbose_reporter.empty_line()
                        self.verbose_reporter.stat_line("Expansion grid results:")
                        for i, r in enumerate(expansion_results):
                            marker = "→" if i == 0 else " "
                            self.verbose_reporter.stat_line(
                                f"  {marker} ms={r['ms']:>2}, mcs={r['mcs']:>2} | "
                                f"k={r['k']:>2} | score={r['score']:.3f} | "
                                f"base={r['score_base']:.3f} (dbcv={r['dbcv']:.2f}, sil={r['sil']:.2f}) | "
                                f"pen: noise={r['noise_penalty']:.3f}, k={r['k_penalty']:.3f}"
                            )

                        # Merge with original results
                        results.extend(expansion_results)
                        results.sort(key=lambda r: r["score"], reverse=True)

                        self.verbose_reporter.empty_line()
                        self.verbose_reporter.stat_line(f"✓ Expansion complete: {len(expansion_results)} new configs evaluated")

                except Exception as e:
                    self.verbose_reporter.stat_line(f"  Grid expansion failed: {str(e)}")
            else:
                self.verbose_reporter.stat_line("  No new expansion candidates (all already tried)")
        else:
            self.verbose_reporter.empty_line()
            self.verbose_reporter.stat_line("[Step 5 - Expansion] No expansion needed:")
            self.verbose_reporter.stat_line(f"  → DBCV={dbcv_best:.3f} >= 0.60 ✓")
            self.verbose_reporter.stat_line(f"  → hard_noise={winner_noise:.1%} <= 10% ✓")

        # === STEP 6: FINAL SELECTION WITH GATES ===
        best, selection_reason = self._select_final_config(results, initial_winner)
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line(f"[Final Selection] {selection_reason}")

        # Verbose reporting
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Complete evaluation results (sorted by score):")
        for r in results:
            total_pen = r.get('noise_penalty', 0) + r.get('k_penalty', 0)
            self.verbose_reporter.stat_line(
                f"mcs={r['mcs']:>3} | ms={r['ms']:>3} | k={r['k']:>3} | "
                f"score={r['score']:.3f} | base={r['score_base']:.3f} | pen={total_pen:.3f}"
            )
        self.verbose_reporter.empty_line()
        for r in results:
            self.verbose_reporter.stat_line(
                f"mcs={r['mcs']:>3} | ms={r['ms']:>3} | dbcv={r['dbcv']:.3f} | sil={r['sil']:.3f}"
            )
        self.verbose_reporter.empty_line()
        for r in results:
            m = r["metrics"]
            total_noise = m.get('noise_rate', 0.0)
            soft_noise = m.get('soft_noise_rate', 0.0)
            hard_noise_rate = m.get('hard_noise_rate', 0.0)
            k_dev = abs(r['k'] - r['k_target']) / r['k_target']
            self.verbose_reporter.stat_line(
                f"mcs={r['mcs']:>3} | ms={r['ms']:>3} | k_pen={r['k_penalty']:.3f} (k_dev={k_dev:.2f}) | "
                f"noise: {total_noise:.1%} (soft: {soft_noise:.1%}, hard: {hard_noise_rate:.1%})"
            )

        # Display hard noise similarity distribution in separate section
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Hard noise similarity distribution:")
        for r in results:
            m = r["metrics"]
            if m.get('hard_noise_q25') is not None:
                self.verbose_reporter.stat_line(
                    f"  mcs={r['mcs']:>3} | ms={r['ms']:>3}: "
                    f"q25={m['hard_noise_q25']:.2f}, "
                    f"median={m['hard_noise_median']:.2f}, "
                    f"q75={m['hard_noise_q75']:.2f}, "
                    f"mean={m['hard_noise_mean']:.2f}"
                )
        self.verbose_reporter.empty_line()
        for r in results:
            m = r["metrics"]
            dbcv_str = '' if m['dbcv'] is None else f"DBCV={m['dbcv']:>6.3f} | "
            meanp_str = '' if m['meanp'] is None else f"meanp={m.get('meanp', float('nan')):.3f} | "
            cdist = '' if m['cdist'] is None else  f"cdist={m.get('cdist', float('nan')):.3f} | "
            cdist5 = '' if m['cdist5'] is None else f"cdist5={m.get('cdist5', float('nan')):.3f} | "
            self.verbose_reporter.stat_line(
                f"mcs={r['mcs']:>3}: | ms={r['ms']:>3} | {meanp_str}{dbcv_str}"
                f"Sil={m.get('sil', float('nan')):.3f} | DB={m.get('DB', float('nan')):.3f} | {cdist}{cdist5}"
            )
    
        # --- Polish loop: halve if hard_noise/DBCV trigger; re-eval up to 3 rounds
        ms_best, mcs_best = best["ms"], best["mcs"]
        dbcv0 = best["metrics"].get("dbcv", None)
        hard_noise_best = best["metrics"].get("hard_noise_rate", np.nan)

        rounds = POLISH_ROUNDS
        changed = True
        note_all = []
        while rounds > 0 and changed:
            rounds -= 1
            ms_new, mcs_new, note, changed = self._apply_threshold_rule(
                ms_best, mcs_best,
                dbcv=dbcv0, hard_noise=hard_noise_best,
                n_points=n,
                min_ms=2, min_mcs=None, dbcv_cut=0.50, noise_cut=0.20
            )
            note_all.append(note)
            if not changed:
                break

            # re-eval a micro-grid around the new ms
            ms_grid = sorted({int(np.clip(f * ms_new, 1, mcs_new)) for f in [0.8, 1.0, 1.2]})
            results2 = self._grid_search(U, ms_grid, mcs_new)
            if not results2:
                break

            # re-score same as above
            sil  = np.clip([r["metrics"].get("sil", np.nan) for r in results2], 0, 1)
            db   = [r["metrics"].get("DB",  np.nan) for r in results2]; db = 1.0 - np.clip(db, 0, 1)
            stab = np.asarray([r["metrics"].get("stab", np.nan) for r in results2], float)
            geometry   = 0.5 * sil + 0.5 * db
            stability  = stab
            base_score = (geometry + stability) / 2
            noise = np.array([r["metrics"].get("noise_rate", np.nan) for r in results2], dtype=float)
            k     = np.array([r["metrics"].get("n_clusters", np.nan) for r in results2], dtype=float)
            k_n   = self._scale_metric01(np.sqrt(k))
            penalties = (noise + 0.5 * k_n) / 2
            final_score = 1 + base_score - penalties
            for i, r in enumerate(results2):
                r["score"] = float(final_score[i])
            results2.sort(key=lambda r: r["score"], reverse=True)
            candidate = results2[0]

            # Check if halving improved hard_noise
            if candidate["metrics"].get("hard_noise_rate", np.nan) < hard_noise_best:
                # Halving helped - accept new config
                best = candidate
                ms_best, mcs_best = candidate["ms"], candidate["mcs"]
                dbcv0 = candidate["metrics"].get("dbcv", None)
                hard_noise_best = candidate["metrics"].get("hard_noise_rate", np.nan)
            else:
                # Halving didn't help - keep original best and stop
                break

        if note_all:
            self.verbose_reporter.empty_line()
            self.verbose_reporter.stat_line("Polish loop decisions: " + " | ".join(note_all))

        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line(f"🏆 Best configuration: min_samples={best['ms']}, min_cluster_size={best['mcs']}")
        return best["hdbscan_model"], best["labels"], best["summary"]

    def _calculate_noise_parameters(self, n_noise: int, original_mcs: int, original_ms: int) -> Tuple[int, int]:
        """
        Calculate HDBSCAN parameters for noise reclustering based on strategy.

        Args:
            n_noise: Number of noise points to recluster
            original_mcs: Original min_cluster_size from main clustering
            original_ms: Original min_samples from main clustering

        Returns:
            Tuple of (min_samples, min_cluster_size) for noise reclustering
        """
        strategy = self.clustering_config.noise_parameter_strategy

        if strategy == "adaptive":
            # Adaptive based on noise point count
            if n_noise < 50:
                mcs = 3
                ms = 2
            elif n_noise < 200:
                mcs = 5
                ms = 3
            else:
                mcs = max(3, int(0.02 * n_noise))  # 2% of noise points
                ms = max(2, mcs // 2)

        elif strategy == "aggressive":
            # Fraction of original parameters
            mcs = max(3, original_mcs // self.clustering_config.noise_mcs_divisor)
            ms = max(2, original_ms // self.clustering_config.noise_ms_divisor)

        elif strategy == "fixed":
            # Fixed values from config
            mcs = self.clustering_config.noise_fixed_mcs
            ms = self.clustering_config.noise_fixed_ms

        else:
            raise ValueError(f"Unknown noise_parameter_strategy: {strategy}")

        # Ensure ms <= mcs
        ms = min(ms, mcs)

        return ms, mcs

    def _assess_noise_cluster_quality(self, pca_embeddings_noise: np.ndarray,
                                     noise_labels: np.ndarray) -> List[int]:
        """
        Filter noise-derived clusters by quality thresholds.

        Args:
            pca_embeddings_noise: PCA embeddings for noise points only (L2-normalized)
            noise_labels: Cluster labels from noise reclustering

        Returns:
            List of valid cluster IDs that meet quality criteria
        """
        valid_clusters = []

        for cluster_id in np.unique(noise_labels[noise_labels >= 0]):
            cluster_mask = noise_labels == cluster_id
            cluster_points = pca_embeddings_noise[cluster_mask]
            n_points = cluster_points.shape[0]

            # Check 1: Minimum size
            if n_points < self.clustering_config.noise_min_cluster_size:
                continue

            # Check 2: Internal cohesion (mean pairwise similarity)
            if n_points > 1:
                pairwise_sim = cluster_points @ cluster_points.T
                # Get upper triangle (excluding diagonal)
                triu_indices = np.triu_indices_from(pairwise_sim, k=1)
                if len(triu_indices[0]) > 0:
                    mean_cohesion = float(np.mean(pairwise_sim[triu_indices]))
                else:
                    mean_cohesion = 1.0  # Single point, perfect cohesion
            else:
                mean_cohesion = 1.0  # Single point

            if mean_cohesion < self.clustering_config.noise_cluster_cohesion_threshold:
                continue

            valid_clusters.append(int(cluster_id))

        return valid_clusters

    def _recluster_noise_points(self, labels: np.ndarray, U: np.ndarray,
                               pca_embeddings: np.ndarray,
                               original_mcs: int, original_ms: int) -> np.ndarray:
        """
        Two-pass clustering: Attempt to find viable clusters among ALL noise points.

        Args:
            labels: Current cluster labels (with -1 for noise)
            U: UMAP embeddings
            pca_embeddings: PCA embeddings (L2-normalized)
            original_mcs: Original min_cluster_size from main clustering
            original_ms: Original min_samples from main clustering

        Returns:
            Updated labels with noise-derived clusters numbered sequentially
        """
        if not self.clustering_config.enable_noise_reclustering:
            return labels

        # Section 1: Identify ALL noise points
        noise_mask = labels == -1
        n_total_noise = noise_mask.sum()

        if n_total_noise < self.clustering_config.noise_min_total_points:
            self.verbose_reporter.stat_line(
                f"Skipping noise reclustering: only {n_total_noise} noise points "
                f"(minimum: {self.clustering_config.noise_min_total_points})"
            )
            return labels

        # Section 2: Calculate noise reclustering parameters
        noise_ms, noise_mcs = self._calculate_noise_parameters(
            n_total_noise, original_mcs, original_ms
        )

        # Section 3: Run HDBSCAN on ALL noise points
        U_noise = U[noise_mask]

        self.verbose_reporter.empty_line()
        self.verbose_reporter.section_header("NOISE RECLUSTERING PASS")
        self.verbose_reporter.stat_line(f"Total noise points: {n_total_noise}")
        self.verbose_reporter.stat_line(f"Parameters: min_samples={noise_ms}, min_cluster_size={noise_mcs}")

        noise_hdbscan = HDBSCAN(
            min_cluster_size=noise_mcs,
            min_samples=noise_ms,
            metric=self.CLUSTER_METRIC,
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.0,
            alpha=1.0
        ).fit(U_noise)

        noise_labels = noise_hdbscan.labels_

        # Section 4: Quality filtering
        pca_embeddings_noise = pca_embeddings[noise_mask]
        valid_noise_clusters = self._assess_noise_cluster_quality(
            pca_embeddings_noise, noise_labels
        )

        if len(valid_noise_clusters) < self.clustering_config.noise_min_clusters:
            self.verbose_reporter.stat_line(
                f"Noise reclustering found {len(valid_noise_clusters)} viable clusters "
                f"(below minimum of {self.clustering_config.noise_min_clusters})"
            )
            return labels

        # Section 5: Renumber and integrate
        labels_updated = labels.copy()
        max_main_cluster = labels[labels >= 0].max() if np.any(labels >= 0) else -1
        next_cluster_id = max_main_cluster + 1

        noise_indices = np.where(noise_mask)[0]
        n_recovered = 0

        for old_noise_cluster_id in valid_noise_clusters:
            cluster_mask_in_noise = noise_labels == old_noise_cluster_id
            global_indices = noise_indices[cluster_mask_in_noise]
            labels_updated[global_indices] = next_cluster_id
            n_recovered += len(global_indices)
            next_cluster_id += 1

        # Section 6: Reporting
        n_noise_clusters = len(valid_noise_clusters)
        recovery_rate = n_recovered / n_total_noise if n_total_noise > 0 else 0.0
        final_noise = (labels_updated == -1).sum()

        self.verbose_reporter.stat_line(f"Viable clusters discovered: {n_noise_clusters}")
        self.verbose_reporter.stat_line(f"Points recovered: {n_recovered} ({recovery_rate:.1%})")
        self.verbose_reporter.stat_line(f"Residual noise: {final_noise} ({final_noise/len(labels):.1%})")

        return labels_updated

    def run(self):
        """Enhanced clustering pipeline with automatic optimization"""
        # Display input statistics

        embeddings = np.array([item.idea_embedding for item in self.output_list])
        self.verbose_reporter.stat_line(f"Input: {len(self.output_list)} idea embeddings ({embeddings.shape[1]} dimensions)")

        # PCA Reduction, if applicable
        self.verbose_reporter.empty_line()
       
        if embeddings.shape[0] > PCA_SIZE_THRESHOLD:
            self.verbose_reporter.stat_line("Step 1: PCA dimensionality reduction...")
            start_time = time.time()
            pca_embeddings = self._pca_reduce(embeddings)
            elapsed_time = time.time() - start_time
            self.verbose_reporter.stat_line(f"Reduced {embeddings.shape[1]} → {pca_embeddings.shape[1]} dimensions")
            self.verbose_reporter.stat_line(f"PCA completed in {elapsed_time:.1f}s")
        else:
            self.verbose_reporter.stat_line("Step 1 Skipped: NO PCA dimensionality reduction")
            pca_embeddings = embeddings
            
        # L2-normalization 
        L2_embeddings = normalize(pca_embeddings, norm="l2", copy=False)

        # Store normalized embeddings
        for item, pca_embed in zip(self.output_list, L2_embeddings):
            item.pca_embedding = pca_embed
        
        # UMAP reduction 
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Step 2: UMAP embedding...")
        self.verbose_reporter.stat_line(f"Configuration: {self.umap_config.n_neighbors} neighbors, {self.umap_config.n_components} components")
        start_time = time.time()
        umap_embeddings = self._umap_embed(L2_embeddings)
        elapsed_time = time.time() - start_time
        self.verbose_reporter.stat_line(f"Reduced: {L2_embeddings.shape[1]} → {umap_embeddings.shape[1]} dimensions")
        self.verbose_reporter.stat_line(f"UMAP completed in {elapsed_time:.1f}s")
      
        # Store UMAP embeddings 
        for item, umap_embed in zip(self.output_list, umap_embeddings):
            item.umap_embedding = umap_embed

        U = umap_embeddings
        
        # HDBSCAN
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Step 3: Finding optimal clustering parameters...")

        # Pass L2-normalized embeddings for regime detection (avoid UMAP q90=1.00 artifact)
        best_model, labels, summary = self._auto_hdbscan_grid(U, original_embeddings=L2_embeddings)
        
        # Report best configuration
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Optimal clustering configuration found:")
        self.verbose_reporter.stat_line(f"  Min cluster size: {summary.min_cluster_size}")
        self.verbose_reporter.stat_line(f"  Clusters: {summary.n_clusters}")
        self.verbose_reporter.stat_line(f"  Noise rate: {summary.noise_rate:.1%}")
        self.verbose_reporter.stat_line(f"  Median cluster size: {summary.median_cluster_size}")

        # Assign labels to items
        for item, label in zip(self.output_list, labels):
            item.initial_idea_cluster = int(label)

        # Merge similar clusters if enabled
        if self.clustering_config.merge_similar_clusters:
            labels = self._merge_similar_clusters(labels)
            # Update items with merged labels
            for item, label in zip(self.output_list, labels):
                item.initial_idea_cluster = int(label)

        # Noise reclustering (two-pass clustering)
        if self.clustering_config.enable_noise_reclustering:
            # Get PCA embeddings array (already stored in output_list)
            pca_embeddings_array = np.vstack([item.pca_embedding for item in self.output_list])

            # Recluster noise points
            labels = self._recluster_noise_points(
                labels=labels,
                U=umap_embeddings,
                pca_embeddings=pca_embeddings_array,
                original_mcs=summary.min_cluster_size,
                original_ms=summary.min_samples
            )

            # Update items with final labels (including noise-derived clusters)
            for item, label in zip(self.output_list, labels):
                item.initial_idea_cluster = int(label)

        # Processing stats
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        #noise_points = list(labels).count(-1)

        # Calculate final hard noise metrics (after merging)
        pca_embeddings_array = np.vstack([item.pca_embedding for item in self.output_list])
        cluster_members = {}
        if num_clusters > 0:
            by_cluster = defaultdict(list)
            for vec, label in zip(pca_embeddings_array, labels):
                if label >= 0:
                    by_cluster[int(label)].append(vec)
            cluster_members = {cid: np.vstack(members) for cid, members in by_cluster.items()}

        final_noise_breakdown = self._assess_noise_quality(pca_embeddings_array, labels, cluster_members)
        hard_noise_count = int(final_noise_breakdown['hard_noise_rate'] * len(self.output_list))
        
        cluster_sizes = {}
        for label in labels:
            if label != -1:
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        if cluster_sizes:
            sizes = list(cluster_sizes.values())
            # Calculate quartiles
            sizes_sorted = sorted(sizes)
            q1 = sizes_sorted[len(sizes_sorted)//4]
            median = sizes_sorted[len(sizes_sorted)//2]
            q3 = sizes_sorted[3*len(sizes_sorted)//4]
            
            # Report statistics
            self.verbose_reporter.empty_line()
            self.verbose_reporter.section_header("FINAL CLUSTERING RESULTS")
            self.verbose_reporter.stat_line(f"Total clusters: {num_clusters}")
            self.verbose_reporter.stat_line(f"Hard noise points: {hard_noise_count} ({final_noise_breakdown['hard_noise_rate']:.1%})")
            self.verbose_reporter.stat_line(f"Cluster sizes - Min: {min(sizes)}, Q1: {q1}, Median: {median}, Q3: {q3}, Max: {max(sizes)}")
            
            # Show top 5 largest clusters
            self.verbose_reporter.empty_line()
            self.verbose_reporter.stat_line("Top 5 largest clusters:")
            top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            for cluster_id, size in top_clusters:
                self.verbose_reporter.stat_line(f"  Cluster {cluster_id}: {size} ideas")
        
        # Display sample clusters 
        self._display_sample_clusters(labels, cluster_sizes)
        
        # Complete 
        self.verbose_reporter.step_complete("Enhanced clustering completed", emoji="✅")

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
        self.verbose_reporter.stat_line("📋 Sample clusters:")
        
        for cluster_id, size in sample_clusters:
            # Get ideas from this cluster
            cluster_ideas = []
            for i, (item, label) in enumerate(zip(self.output_list, labels)):
                if label == cluster_id:
                    cluster_ideas.append(item.idea)
            
            # Display cluster info
            self.verbose_reporter.stat_line(f"  Cluster {cluster_id} ({size} ideas):")
            
            # Show up to 3 random examples from this cluster
            examples = random.sample(cluster_ideas, min(3, len(cluster_ideas)))
            for example in examples:
                cleaned_example = re.sub(r"\[.*?\]", "", example)
                cleaned_example = re.sub(r"\s+", " ", cleaned_example).strip()

                # Truncate long ideas to 80 characters
                if len(cleaned_example) > 80:
                    cleaned_example = cleaned_example[:77] + "..."
                self.verbose_reporter.stat_line(f"    → \"{cleaned_example}\"")
            
            # Add spacing between clusters
            if (cluster_id, size) != sample_clusters[-1]:
                self.verbose_reporter.empty_line()

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
                    idea_count=len(cluster_submodels)
                )

            cluster_models.append(cluster_model)

        return cluster_models


def clean_cluster_ideas(cluster_results: List[ClusterModel]) -> List[ClusterModel]:
    """Clean cluster idea texts by removing bracketed annotations and normalizing whitespace.

    Args:
        cluster_results: List of ClusterModel objects with idea texts to clean

    Returns:
        List of ClusterModel objects with cleaned idea texts
    """
    cleaned_results = []

    for result in cluster_results:
        cleaned_response_ideas = []

        if result.response_ideas:
            for idea_submodel in result.response_ideas:
                # Extract and clean idea text
                cleaned_idea = idea_submodel.idea
                cleaned_idea = re.sub(r"\[.*?\]", "", cleaned_idea)
                cleaned_idea = re.sub(r"\s+", " ", cleaned_idea).strip()

                # Create new ClusterSubmodel with cleaned text
                cleaned_submodel = ClusterSubmodel(
                    idea_id=idea_submodel.idea_id,
                    idea=cleaned_idea,
                    idea_embedding=idea_submodel.idea_embedding,
                    initial_cluster=idea_submodel.initial_cluster,
                    expanded_cluster=idea_submodel.expanded_cluster,
                    cluster_theme=idea_submodel.cluster_theme
                )
                cleaned_response_ideas.append(cleaned_submodel)

        # Create new ClusterModel with cleaned ideas
        cleaned_result = ClusterModel(
            respondent_id=result.respondent_id,
            response=result.response,
            response_type=result.response_type,
            quality_filter=result.quality_filter,
            quality_filter_code=result.quality_filter_code,
            response_ideas=cleaned_response_ideas,
            idea_count=len(cleaned_response_ideas)
        )
        cleaned_results.append(cleaned_result)

    return cleaned_results
