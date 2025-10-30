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

from config import UMAPConfig, ClusteringConfig, HDBSCANConfig

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
        self.random_state = 42
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
    
            if self.umap_config.parallel_jobs:
                random_state = None
                transform_seed = None
                n_jobs = -1  # use all cores; UMAP may cap/override
            else:
                random_state = 42
                transform_seed = 42
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
    
    def _evaluate_hdbscan(self, U: np.ndarray, ms: int, mcs: int) -> Dict[str, Any]:
        """Evaluate HDBSCAN configuration with kappa-based scoring and all metrics"""
        start_time = time.time()

        # Fit HDBSCAN
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
      
        # Calculate ALL metrics including expensive ones
        M = self._calculate_metrics(U, labels, db)
        
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
            topk = min(5, n_clusters)   
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
            frac_low_points = float((sil_samples < 0.30).mean())
            clusters, counts = np.unique(y_sample, return_counts=True)
            cluster_means = []
            for c in clusters:
                cluster_mask = (y_sample == c)
                cluster_sil = sil_samples[cluster_mask]
                cluster_means.append(cluster_sil.mean())
            cluster_means = np.array(cluster_means)
            frac_low_clusters = float((cluster_means < 0.30).mean())
        
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
    
    
    def _baseline_by_n(self, n: int) -> tuple[int, int, str]:
        # min_samples (piecewise)
        if n <= 100:
            ms = int(np.ceil(0.05 * n))
        elif n < 500:
            ms = int(np.ceil(max(np.log(n), 0.02 * n, 0.40 * np.sqrt(n))))
        else:
            ms = int(np.ceil(max(np.log(n), 0.01 * n, 0.50 * np.sqrt(n))))
    
        # min_cluster_size ladder (your preference)
        if n <= 500:
            mcs = int(np.ceil(0.01 * n))       # 1%
            ladder = "1%"
        elif n < 2000:
            mcs = int(np.ceil(0.015 * n))      # 1.5%
            ladder = "1.5%"
        else:
            mcs = int(np.ceil(0.02 * n))       # 2%
            ladder = "2%"

        ms = min(ms, mcs)          
    
        return ms, mcs, f"baseline(ms={ms}, mcs={mcs}, ladder={ladder})"
    
    
    def _suggest_params(self, U: np.ndarray, min_ms: int = 2, min_mcs: int = 5, max_mcs: int = 250) -> tuple[int, int, str]:
        """
        A → B: structure sets direction (factor), size sets scale (baseline).
        """
        f, notes_a = self._structure_factor_from_space(U)
        ms0, mcs0, notes_b = self._baseline_by_n(U.shape[0])
    
        ms  = max(min_ms, int(np.clip(int(round(ms0 * f)), min_ms, U.shape[0])))
        mcs = int(np.clip(int(round(mcs0 * f)), min_mcs, max_mcs))
    
        notes = f"[A] {notes_a}; [B] {notes_b}; → scaled(ms={ms}, mcs={mcs})"
        return ms, mcs, notes
    
    
    @staticmethod
    def _apply_threshold_rule(ms: int, mcs: int, dbcv: Optional[float], noise: float,
                              min_ms: int = 2, min_mcs: int = 5,
                              dbcv_cut: float = 0.50, noise_cut: float = 0.20) -> tuple[int, int, str, bool]:
        """
        If noise > 20% OR (DBCV available and < 0.50): halve ms and mcs. Floors applied.
        Returns (new_ms, new_mcs, note, changed_flag).
        """
        trigger = False
        note_parts = []
        if noise is not None and noise > noise_cut:
            trigger = True
            note_parts.append(f"noise {noise:.2f}>{noise_cut:.2f}")
        if (dbcv is not None) and (dbcv < dbcv_cut):
            trigger = True
            note_parts.append(f"dbcv {dbcv:.2f}<{dbcv_cut:.2f}")
    
        if trigger:
            ms_new  = max(min_ms, int(np.ceil(0.5 * ms)))
            mcs_new = max(min_mcs, int(np.ceil(0.5 * mcs)))
            note_parts.append(f"halve→ ms={ms_new}, mcs={mcs_new}")
            return ms_new, mcs_new, "; ".join(note_parts), True
    
        return ms, mcs, "no change", False
    

    def _auto_hdbscan_grid(self, U: np.ndarray):

        # Get structure-scaled baseline
        ms, mcs, notes = self._suggest_params(U)
        self.verbose_reporter.stat_line(f"Param suggestion: {notes}")
    
        # Small micro-grid around ms (keep mcs fixed for now)
        ms_grid = sorted({int(np.clip(f * ms, 1, mcs)) for f in [0.8, 1.0, 1.2]})
    
        # Round 0: evaluate starters
        results = self._grid_search(U, ms_grid, mcs)
        if not results:
            raise RuntimeError("All clustering configurations failed. No valid results to evaluate.")
    
        # Score & rank (reuse your logic)
        sil  = np.clip([r["metrics"].get("sil", np.nan) for r in results], 0, 1)
        db   = [r["metrics"].get("DB",  np.nan) for r in results]; db = 1.0 - np.clip(db, 0, 1)
        stab = np.asarray([r["metrics"].get("stab", np.nan) for r in results], float)
    
        geometry   = 0.5 * sil + 0.5 * db
        stability  = stab
        base_score = (geometry + stability) / 2
    
        noise = np.array([r["metrics"].get("noise_rate", np.nan) for r in results], dtype=float)
        k     = np.array([r["metrics"].get("n_clusters", np.nan) for r in results], dtype=float)
        k_n   = self._scale_metric01(np.sqrt(k))
        penalties = (noise + 0.5 * k_n) / 2
    
        final_score = 1 + base_score - penalties
        for i, r in enumerate(results):
            r["score"] = float(final_score[i])
            r["score_base"] = float(base_score[i])
            r["geometry"] = float(geometry[i])
            r["stability"] = float(stability[i])
            r["penalties"] = float(penalties[i])
            r["noise"] = float(noise[i])
            r["k"] = float(k_n[i])
    
        results.sort(key=lambda r: r["score"], reverse=True)
        best = results[0]
    
        # Verbose reporting (kept from your original)
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Complete evaluation results (sorted by score):")
        for r in results:
            self.verbose_reporter.stat_line(
                f"mcs={r['mcs']:>3} | ms={r['ms']:>3} | clusters={r['metrics'].get('n_clusters', -1):>3} | "
                f"score={r['score']:.3f} | score_base={r['score_base']:.3f} | penalties={r['penalties']:.3f}"
            )
        self.verbose_reporter.empty_line()
        for r in results:
            self.verbose_reporter.stat_line(
                f"mcs={r['mcs']:>3} | ms={r['ms']:>3} | geom={r['geometry']:.3f} | stab={r['stability']:.3f}"
            )
        self.verbose_reporter.empty_line()
        for r in results:
            self.verbose_reporter.stat_line(
                f"mcs={r['mcs']:>3} | ms={r['ms']:>3} | k={r['k']:.3f} | noise={r['noise']:.3f}"
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
    
        # --- Polish loop: halve if noise/DBCV trigger; re-eval up to 3 rounds
        ms_best, mcs_best = best["ms"], best["mcs"]
        dbcv0 = best["metrics"].get("dbcv", None)
        noise0 = best["metrics"].get("noise_rate", np.nan)
    
        rounds = 3
        changed = True
        note_all = []
        while rounds > 0 and changed:
            rounds -= 1
            ms_new, mcs_new, note, changed = self._apply_threshold_rule(
                ms_best, mcs_best,
                dbcv=dbcv0, noise=noise0,
                min_ms=2, min_mcs=5, dbcv_cut=0.50, noise_cut=0.20
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
            best2 = results2[0]
    
            # update best
            best = best2
            ms_best, mcs_best = best2["ms"], best2["mcs"]
            dbcv0 = best2["metrics"].get("dbcv", None)
            noise0 = best2["metrics"].get("noise_rate", np.nan)
    
        if note_all:
            self.verbose_reporter.stat_line("Polish loop decisions: " + " | ".join(note_all))
    
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line(f"🏆 Best configuration: min_samples={best['ms']}, min_cluster_size={best['mcs']}")
        return best["hdbscan_model"], best["labels"], best["summary"]

    def run(self):
        """Enhanced clustering pipeline with automatic optimization"""
        # Display input statistics
        
        embeddings = np.array([item.idea_embedding for item in self.output_list])
        self.verbose_reporter.stat_line(f"Input: {len(self.output_list)} idea embeddings ({embeddings.shape[1]} dimensions)")
        
        # PCA Reduction, if applicable
        self.verbose_reporter.empty_line()
       
        if embeddings.shape[0] > 10_000:
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
        
        best_model, labels, summary = self._auto_hdbscan_grid(U)
        
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
        
        # Processing stats
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = list(labels).count(-1)
        
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
            self.verbose_reporter.stat_line(f"Noise points: {noise_points} ({noise_points / len(self.output_list) * 100:.1f}%)")
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
                # Truncate long ideas to 80 characters
                if len(example) > 80:
                    example = example[:77] + "..."
                self.verbose_reporter.stat_line(f"    → \"{example}\"")
            
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
                    #idea_embeddings=cluster_submodels,
                    idea_count=len(cluster_submodels)
                )

            cluster_models.append(cluster_model)

        return cluster_models
