import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
from typing import List, Optional, Any, Dict, Tuple, Iterable
import numpy as np
import numpy.typing as npt
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize #StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from collections import defaultdict
import math, re
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

# === CONSTANTS ========================================================================================================


# === DATACLASSES ========================================================================================================
from dataclasses import dataclass

@dataclass
class ClusterSummary:
    """Summary statistics for a clustering result"""
    n_points: int
    min_cluster_size: int
    n_clusters: int
    noise_rate: float
    median_cluster_size: int

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
        self.DBCV_D = ClusteringConfig.DBCV_D  # safe for DBCV (avoid overflow)
        self.enable_dbcv = clustering_config.enable_dbcv
        self.enable_meanp  = clustering_config.enable_meanp
        self.centroid_distance = clustering_config.centroid_distance
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
            
        # L2-normalize rows so cosine UMAP works properly
        embeddings = normalize(embeddings, norm="l2", copy=False)
            
        return embeddings
    
    def _umap_embed(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings(): # Suppress UMAP n_jobs warning when using random_state
            warnings.filterwarnings("ignore", 
                                  message="n_jobs value .* overridden to 1 by setting random_state",
                                  category=UserWarning,
                                  module="umap")
            
            if self.umap_config.parallel_jobs:
                random_state = None
                transform_seed = None
            else:
                random_state = 42
                transform_seed = 42
            
            umap_params = {
                    'n_neighbors': self.umap_config.n_neighbors,
                    'n_components': self.umap_config.n_components,
                    'min_dist': self.umap_config.min_dist,
                    'metric': self.umap_config.metric,
                    'n_epochs': self.umap_config.n_epochs,
                    'n_jobs': self.umap_config.parallel_jobs,  # Use multiple cores if true, but default is false, because of random state
                    'low_memory': self.umap_config.low_memory,
                    'verbose': False,
                    'random_state': random_state, #only if parallel is false
                    'transform_seed': transform_seed
                    }
            umap = UMAP(**umap_params)
            U = umap.fit_transform(X).astype(np.float64, copy=False)
            return U
    
    def _dbcv_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate DBCV score for clustering quality"""
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return np.nan
       
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
        except Exception:
            return np.nan
       
    
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
    
    def _mean_probability(self, hdbscan_model: HDBSCAN, labels: np.ndarray) -> float:
        """Calculate mean probability for clustered points"""
        mask = labels >= 0
        if not np.any(mask): 
            return np.nan
        return float(np.mean(hdbscan_model.probabilities_[mask]))
    
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
    
    def _topk_intra_centroid_angular(self, X, labels, topk, trim_q, robust):
        
        mask = labels >= 0
        if not np.any(mask): return {'wm_q25_ang': np.nan}
    
        y = labels[mask]
        Xn = X[mask]
        uniq, counts = np.unique(y, return_counts=True)
    
        order = np.argsort(counts)[::-1]
        top_ids = uniq[order[:min(topk, len(uniq))]]
    
        sizes = []
        mean_list = []
    
        for cid in top_ids:
            Xi = Xn[y == cid]
            if Xi.shape[0] < 2:
                continue
            centroid = Xi.mean(axis=0, keepdims=True)
            sims = cosine_similarity(Xi, centroid).ravel()         # [-1,1], usually [0,1]
            sims = np.clip(sims, -1.0, 1.0)
            ang = np.arccos(sims) / np.pi                          # angular distance in [0,1]
            if trim_q > 0.0:
                lo, hi = np.quantile(ang, [trim_q, 1 - trim_q])
                ang = ang[(ang >= lo) & (ang <= hi)]
            if ang.size:
                sizes.append(Xi.shape[0])
                mean_list.append(float(np.mean(ang)))
    
        if not sizes:
            return np.nan
        
        w = np.array(sizes, float); w /= w.sum()
        wm_ang = float(np.nansum(w * np.array(mean_list, float)))

        return  wm_ang*180 #degrees

    def _ctfidf_metrics(self, labels: np.ndarray, docs: Iterable[str], top_k: int = 15, stopwords=None, min_df: int = 2, ngram_range=(1,2)) -> Dict[str, float]:
        """Compute c-TF-IDF coherence/exclusivity/diversity across clusters."""
        labels = np.asarray(labels)
        mask = labels >= 0
        if not np.any(mask):
            return {"cTF_coh": np.nan, "cTF_exc": np.nan, "cTF_div": np.nan}
        uniq = [l for l in np.unique(labels[mask]) if l != -1]
        if len(uniq) < 2:
            return {"cTF_coh": np.nan, "cTF_exc": np.nan, "cTF_div": np.nan}
    
        docs = np.asarray(list(docs), dtype=object)
        cluster_docs = []
        for l in uniq:
            cl = docs[labels == l]
            if cl.size == 0: 
                continue
            joined = " ".join(cl)
            joined = re.sub(r"\s+", " ", joined).strip()
            cluster_docs.append(joined)
        if len(cluster_docs) < 2:
            return {"cTF_coh": np.nan, "cTF_exc": np.nan, "cTF_div": np.nan}
    
        try:
            vect = CountVectorizer(stop_words=stopwords, min_df=min_df, ngram_range=ngram_range)
            counts = vect.fit_transform(cluster_docs)        # (C, V)
            tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
            ctfidf = tfidf.fit_transform(counts).astype(float)
            ctfidf = normalize(ctfidf, norm="l2", axis=1)
            C = ctfidf.toarray()
    
            # coherence: mean of top-k weights
            coh_vals, div_vals, exc_vals = [], [], []
            for i in range(C.shape[0]):
                row = C[i]
                if not np.any(row > 0): 
                    continue
                k = min(top_k, int((row > 0).sum()))
                idx = np.argsort(row)[::-1][:k]
                w = row[idx]
                coh_vals.append(float(w.mean()))
                p = w / (w.sum() if w.sum() > 0 else 1.0)
                # normalized entropy over top-k
                p = p[p > 0]
                H = -(p * np.log(p)).sum() / math.log(len(p)) if len(p) > 1 else 0.0
                div_vals.append(float(H))
    
            # exclusivity: 1 − cos(this, mean(others))
            for i in range(C.shape[0]):
                this_vec = C[i].reshape(1, -1)
                others = np.delete(C, i, axis=0)
                if others.size == 0: 
                    continue
                other_mean = others.mean(axis=0, keepdims=True)
                sim = float(cosine_similarity(this_vec, other_mean)[0, 0])
                exc_vals.append(1.0 - sim)
    
            return {
                "cTF_coh": float(np.mean(coh_vals)) if coh_vals else np.nan,
                "cTF_exc": float(np.mean(exc_vals)) if exc_vals else np.nan,
                "cTF_div": float(np.mean(div_vals)) if div_vals else np.nan,
            }
        except Exception:
            return {"cTF_coh": np.nan, "cTF_exc": np.nan, "cTF_div": np.nan}
    
    @staticmethod
    def gini_from_sizes(sizes):
        """Compute Gini coefficient of cluster size distribution."""
        s = np.array(sizes, dtype=float)
        s = s[s > 0]             # ignore empties
        if s.size == 0:
            return np.nan
        s.sort()
        n = s.size
        cum = np.cumsum(s)
        gini = (n + 1 - 2*np.sum(cum) / cum[-1]) / n
        return float(gini)
    
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
    
    def _evaluate_hdbscan(self, U: np.ndarray, mcs: int, embeddings: np.ndarray) -> Dict[str, Any]:
        """Evaluate HDBSCAN configuration with kappa-based scoring and all metrics"""
        start_time = time.time()

        # Fit HDBSCAN
        db = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=None,
            metric=self.CLUSTER_METRIC,
            cluster_selection_method="eom",
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
            n_clusters=int(n_clusters),
            noise_rate=float(round(noise_rate, 3)),
            median_cluster_size=int(med_size),
        )
        
        
        elapsed_time = time.time() - start_time
        
        return {
            "mcs": mcs,
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
        
        # inbalance : mean angular degree distance topK clusters - for coherence topk clusters
        wm_ang5 = self._topk_intra_centroid_angular(U, labels, topk=min(5, n_clusters) , trim_q=0.0, robust=False)
      
        # Per-sample silhouettes (with sampling)
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
        
        # gini for inbalance
        _, counts = np.unique(labels[mask], return_counts=True)
        gini = self.gini_from_sizes(counts)
        
        # c-TF-IDF suite on raw ideas text
        docs = [item.idea for item in self.output_list]
        ct = self._ctfidf_metrics(
            labels=labels, docs=docs,
            top_k=getattr(self.clustering_config, "ctfidf_top_k", 15),
            stopwords=getattr(self.clustering_config, "ctfidf_stopwords", None),
            min_df=getattr(self.clustering_config, "ctfidf_min_df", 2),
            ngram_range=getattr(self.clustering_config, "ctfidf_ngram_range", (1,2))
        )
        
        return {
            "n_clusters": n_clusters,           # number of clusters (labels >= 0)
            "dbcv": dbcv,                       # density-based clustering validation (higher=better; may be None) 
            "noise_rate": noise_rate,           # fraction of points labeled -1
            "meanp": meanp,                     # mean membership probability over clustered points
            "sil": sil,                         # silhouetee score - for distinctiveness/separation
            "CH": geom["CH"],                   # Calinski–Harabasz (higher=better)  - overall variance-explained metric.
            "DB": geom["DB"],                   # Davies–Bouldin (lower=better) - for cluster overlap metric
            "cTF_coh": ct["cTF_coh"],           # c-TF-IDF coherence (0..1; higher=tighter top terms)
            "cTF_exc": ct["cTF_exc"],           # c-TF-IDF exclusivity (0..1; higher=more distinctive vs others)
            "cTF_div": ct["cTF_div"],           # diversity/entropy of top terms within clusters
            "wm_ang5": wm_ang5,                 # angled distance to cluster mean/centroid for top5 clusters (in degrees)- for cluster coherence 
            "cdist": cdist, "cdist5": cdist5,   # mean angular distance of ALL/TOP5 centroid pairs (off-diagonal)
            "gini": gini,                       # inequality of cluster sizes (0=balanced, 1=skewed
            "frac_low_points": frac_low_points, # share of points with sil < threshold (e.g., 0.30)
            "frac_low_clusters": frac_low_clusters, # share of clusters with mean sil < threshold
         }
                
        
    def _grid_search(self, U: np.ndarray, mcs_grid: List[int], embeddings: np.ndarray) -> List[Dict[str, Any]]:
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
                future_to_mcs = {executor.submit(self._evaluate_hdbscan, U, mcs, embeddings): mcs for mcs in mcs_grid}
                results = []  
                completed = 0
                total = len(mcs_grid)
                
                for future in concurrent.futures.as_completed(future_to_mcs, timeout=timeout):
                    mcs = future_to_mcs[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        if self.verbose_reporter.enabled:
                            self.verbose_reporter.stat_line(
                                f"  Completed {completed}/{total} configs "
                                f"(mcs={mcs}) "
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

    
    def _auto_hdbscan_grid(self,  U: np.ndarray,  embeddings: np.ndarray, mcs_grid: Optional[Iterable[int]] = None) -> Tuple[HDBSCAN, np.ndarray, ClusterSummary]:
        
        n = U.shape[0]
        if mcs_grid is None:
            mcs_grid = sorted(set([
                max(5,  int(0.25 * np.sqrt(n))),
                max(10, int(0.50 * np.sqrt(n))),
                max(15, int(0.75 * np.sqrt(n))),
                max(20, int(1.00 * np.sqrt(n))),
            ]))
        else:
            mcs_grid = sorted(list(mcs_grid))
                
        
        results = self._grid_search(U, mcs_grid, embeddings)
               
        # 1) scores
        sil  = [r["metrics"].get("sil", np.nan) for r in results]
        db   = [r["metrics"].get("DB",  np.nan) for r in results]
        coh  = [r["metrics"].get("cTF_coh", np.nan) for r in results]
        exc  = [r["metrics"].get("cTF_exc", np.nan) for r in results]
        gini = [r["metrics"].get("gini", np.nan) for r in results]    
        
        sil_t = np.clip(sil, 0, 1)**2
        dbb_t = 1.0 / (1.0 + np.square(db))      # higher=better
        coh_t = np.clip(coh, 0, 1)**2
        exc_t = np.clip(exc, 0, 1)**2
        g_t   = 1.0 - np.clip(gini, 0, 1)        # balance benefit
        
        sil_n = self._scale_metric01(sil_t)
        db_n  = self._scale_metric01(dbb_t)
        coh_n = self._scale_metric01(coh_t)
        exc_n = self._scale_metric01(exc_t)
        g_n   = self._scale_metric01(g_t)     

        geometry  = 0.5*sil_n + 0.5 *db_n
        semantics = 0.5 *coh_n + 0.5* exc_n
        balance   = g_n
        
        base_raw  = (geometry + semantics + balance) / 3.0           
        base_n    = self._scale_metric01(base_raw) 
        
        # 2) penalties
        ang5       = np.array([r["metrics"].get("wm_ang5", np.nan) for r in results], float)
        noise      = np.array([r["metrics"].get("noise_rate", np.nan) for r in results], dtype=float)
        badC       = np.array([r["metrics"].get("frac_low_clusters", np.nan) for r in results], dtype=float)
        k          = np.array([r["metrics"].get("n_clusters", np.nan) for r in results], dtype=float)

        ang5_n     = self._scale_metric01(ang5)
        noise_n    = self._scale_metric01(noise)
        badC_n     = self._scale_metric01(badC)
        k_n        = self._scale_metric01(np.sqrt(k))         
        
        penalties  = (ang5_n + noise_n + badC_n + k_n) / 4
        pen_n      = self._scale_metric01(penalties) 
        
        final_score = 1 + base_n - pen_n
        
        # noise   = np.array([r["metrics"].get("noise_rate", np.nan) for r in results], dtype=float)
        # badC    = np.array([r["metrics"].get("frac_low_clusters", np.nan) for r in results], dtype=float)
        
        # k          = np.array([r["metrics"].get("n_clusters", np.nan) for r in results], dtype=float)
        # pen_sizeC = self._scale_metric01( k)         
        
        # ang5       = np.clip(wm_ang5 / np.pi, 0.0, 1.0)
        # pen_ang5 = self._scale_metric01(ang5)         
        
        # # Blend relative penalties (per-term standardized)  
        # w_ang5, w_siceC= 1.0, 1.0
        # pen_rel = (w_siceC*pen_sizeC + w_ang5* pen_ang5) / (w_ang5+w_siceC) 
        # pen_n = self._scale_metric01(pen_rel)
        
        # # # 4) composite score
        # # base_score  = base_raw  
        # penalties   = pen_n + noise + badC
        # # final_score = 1 + base_score - penalties
        
        # # 3) Center both sides and equalize sensitivity by spread (MAD) 
        # def _mad(v): 
        #     med = np.nanmedian(v) 
        #     return np.nanmedian(np.abs(v - med)) 
    
        # base_med, pen_med = np.nanmedian(base_n), np.nanmedian(pen_n) 
        # base_mad, pen_mad = _mad(base_n), _mad(pen_n) 
        # lam = base_mad / (pen_mad + 1e-9) # calibration so equal deltas have equal impact 
        
        # delta_base = base_n - base_med 
        # delta_pen = pen_n - pen_med 
        
        # final_score = 1 + delta_base - (lam * delta_pen) - noise - badC 
        # base_score = base_n
    

        for i, r in enumerate(results):
            r["score"]           = float(final_score[i])
            r["score_base"]      = float(base_n[i])
            r["geometry"]        = float(geometry[i])
            r["semantics"]       = float(semantics[i])
            r["balance"]         = float(balance[i])
            r["penalties"]       = float(pen_n[i]) 
            r["pen_ang5"]        = float(ang5_n[i])
            r["pen_sizeC"]       = float(k_n[i])
            r["pen_noise"]       = float(noise_n[i])
            r["pen_badC"]        = float(badC_n[i])
            
        if not results:
            raise RuntimeError("All clustering configurations failed. No valid results to evaluate.")
        
        results.sort(key=lambda r: r["score"], reverse=True)
        best_result = results[0]
        
        # Report all results
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Complete evaluation results (sorted by score):")
        
        for i, result in enumerate(results):
            metrics = result["metrics"]
            self.verbose_reporter.stat_line(
                f"mcs={result['mcs']:>4} | " 
                f"clusters={metrics.get('n_clusters', -1):>3} | "
                f"score={result['score']:.3f} | "
                f"score_base={result['score_base']:.3f} | "
                f"pentalties={result['penalties']:.3f}")
            
        self.verbose_reporter.empty_line()    
        for i, result in enumerate(results):
            metrics = result["metrics"]
            self.verbose_reporter.stat_line(
              f"mcs={result['mcs']:>4} | "   
              f"geom={result['geometry']:.3f} | "
              f"sem={result['semantics']:.3f} | "
              f"bal={result['balance']:.3f}")
            
        self.verbose_reporter.empty_line()    
        for i, result in enumerate(results):
            metrics = result["metrics"]
            self.verbose_reporter.stat_line(
              f"mcs={result['mcs']:>4} | "   
              f"ang5={result['pen_ang5']:.3f} | "
              f"sizeC={result['pen_sizeC']:.3f} | "
              f"noise={result.get('pen_noise', float('nan')):.3f} | "
              f"badC={result['pen_badC']:.3f}")
           
        self.verbose_reporter.empty_line()
        for i, result in enumerate(results):
            metrics = result["metrics"]
            dbcv_str = '' if metrics['dbcv'] is None else f"DBCV={metrics['dbcv']:>6.3f} | "
            meanp_str = '' if metrics['meanp'] is None else f"meanp={metrics.get('meanp', float('nan')):.3f} | "
            cdist = '' if metrics['cdist'] is None else  f"cdist={metrics.get('cdist', float('nan')):.3f} | "
            cdist5 = '' if metrics['cdist5'] is None else   f"cdist5={metrics.get('cdist5', float('nan')):.3f} | "

            self.verbose_reporter.stat_line(
                f"mcs={result['mcs']:>4}: | " 
                f"{meanp_str}"
                f"{dbcv_str}"
                f"Sil={metrics.get('sil', float('nan')):.3f} | "
                f"DB={metrics.get('DB', float('nan')):.3f} | "
                f"{cdist}"
                f"{cdist5}"
                f"coh={metrics.get('cTF_coh', float('nan')):.3f} | "
                f"exc={metrics.get('cTF_exc', float('nan')):.3f} | "
                f"div={metrics.get('cTF_div', float('nan')):.3f} | "
                f"gini={metrics.get('gini', float('nan')):.3f} | "
                )
        
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line(f"🏆 Best configuration: min_cluster_size={best_result['summary'].min_cluster_size}")
        
        return best_result["hdbscan_model"], best_result["labels"], best_result["summary"]
    
    def _merge_clusters_by_similarity(self, embeddings: np.ndarray, labels: np.ndarray, sim_threshold: Optional[float] = None, linkage: str = "complete") -> np.ndarray:
        """Merge clusters based on centroid cosine similarity"""
        assert embeddings.shape[0] == labels.shape[0], "embeddings/labels length mismatch"
        
        # Use provided threshold or default from config
        if sim_threshold is None:
            sim_threshold = self.clustering_config.default_merge_threshold
        
        # Build centroids in original embedding space
        centroids, sizes = self._compute_centroids(embeddings, labels)
        
        if not centroids:
            return labels.copy()
        
        orig_ids = sorted(centroids.keys())
        C = np.vstack([centroids[cid] for cid in orig_ids])
        
        # L2-normalize for cosine similarity
        Cn = normalize(C)
        
        # Agglomerative clustering on centroids
        dist_threshold = 1.0 - sim_threshold
        
        try:
            ag = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage=linkage,
                distance_threshold=dist_threshold,
                compute_full_tree=True,
            )
        except TypeError:
            # Older scikit-learn versions
            ag = AgglomerativeClustering(
                n_clusters=None,
                affinity="cosine",
                linkage=linkage,
                distance_threshold=dist_threshold,
                compute_full_tree=True,
            )
        
        merged_ids = ag.fit_predict(Cn)
        
        # Build mapping old_id -> merged_group_id
        uniq_groups = {g: i for i, g in enumerate(sorted(np.unique(merged_ids)))}
        old_to_new_group = {old: uniq_groups[g] for old, g in zip(orig_ids, merged_ids)}
        
        # Remap point labels
        new_labels = labels.copy()
        for i, y in enumerate(labels):
            if y is not None and y >= 0:
                new_labels[i] = old_to_new_group[int(y)]
            else:
                new_labels[i] = -1
        
        # Report merges
        groups = defaultdict(list)
        for old, g in old_to_new_group.items():
            groups[g].append(old)
        groups_list = [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: kv[0])]
        
        self.verbose_reporter.section_header(f"CLUSTER MERGING (cosine ≥ {sim_threshold:.2f})")
        self.verbose_reporter.stat_line(f"Original clusters: {len(orig_ids)} → Merged clusters: {len(groups_list)}")
        
        # Show merged groups
        merged_groups = [g for g in groups_list if len(g) > 1]
        if merged_groups:
            self.verbose_reporter.stat_line("Merged groups:")
            for g in merged_groups[:10]:
                total_n = sum(sizes.get(cid, 0) for cid in g)
                parts = ", ".join(f"{cid}(n={sizes.get(cid,0)})" for cid in g)
                self.verbose_reporter.stat_line(f"  {{{parts}}} → total n={total_n}")
            if len(merged_groups) > 10:
                self.verbose_reporter.stat_line(f"  ... and {len(merged_groups)-10} more groups")
        else:
            self.verbose_reporter.stat_line("No merges needed at this threshold")
        
        return new_labels
    
    def _analyze_cluster_similarity(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Analyze and report similarity between clusters"""
        self.verbose_reporter.section_header("CLUSTER SIMILARITY ANALYSIS")
        
        # Extract cluster embeddings and calculate centroids
        cluster_embeddings = defaultdict(list)
        cluster_sizes = defaultdict(int)
        
        # Group embeddings by cluster ID
        for embedding, label in zip(embeddings, labels):
            if label is not None and label != -1:  # Exclude noise points
                cluster_id = label
                cluster_embeddings[cluster_id].append(embedding)
                cluster_sizes[cluster_id] += 1
        
        # Calculate cluster centroids
        cluster_centroids = {}
        for id_cluster, embeddings_cluster in cluster_embeddings.items():
            if embeddings_cluster:
                centroid = np.mean(embeddings_cluster, axis=0)
                cluster_centroids[id_cluster] = centroid
        
        # Sort cluster IDs for consistent output
        sorted_cluster_ids = sorted(cluster_centroids.keys())
        num_clusters = len(sorted_cluster_ids)
        
        if num_clusters > 1:
            # Create centroid matrix
            centroid_matrix = np.array([cluster_centroids[cid] for cid in sorted_cluster_ids])
            
            # Calculate pairwise cosine similarities
            similarity_matrix = cosine_similarity(centroid_matrix)
            
            # Extract upper triangle (excluding diagonal)
            similarities = similarity_matrix[np.triu_indices(num_clusters, k=1)]
            total_pairs = len(similarities)
            
            # Report similarity distribution
            self.verbose_reporter.stat_line(f"Analyzing {num_clusters} clusters ({total_pairs} unique pairs)")
            
            # Thresholds to analyze
            thresholds = self.clustering_config.similarity_analysis_thresholds
            
            for threshold in thresholds:
                count = np.sum(similarities >= threshold)
                percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
                self.verbose_reporter.stat_line(f"Similarity >= {threshold:.2f}: {count:4d} pairs ({percentage:5.1f}%)")
            
            # Find and display most similar cluster pairs
            self.verbose_reporter.empty_line()
            self.verbose_reporter.stat_line("Top 10 most similar cluster pairs:")
            
            # Get indices of top similarities
            top_k = min(10, total_pairs)
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
            
            # Convert flat indices back to cluster pairs
            triu_indices = np.triu_indices(num_clusters, k=1)
            
            for rank, idx in enumerate(top_indices, 1):
                i = triu_indices[0][idx]
                j = triu_indices[1][idx]
                cluster_i = sorted_cluster_ids[i]
                cluster_j = sorted_cluster_ids[j]
                similarity = similarities[idx]
                size_i = cluster_sizes.get(cluster_i, 0)
                size_j = cluster_sizes.get(cluster_j, 0)
                
                self.verbose_reporter.stat_line(
                    f"  {rank:2d}. Cluster {cluster_i} ({size_i} ideas) <-> "
                    f"Cluster {cluster_j} ({size_j} ideas): {similarity:.3f}"
                )
        else:
            self.verbose_reporter.stat_line("Not enough clusters for similarity analysis (need at least 2)")

    def run(self):
        """Enhanced clustering pipeline with automatic optimization"""
        # Display input statistics
        
        embeddings = np.array([item.idea_embedding for item in self.output_list])
        self.verbose_reporter.stat_line(f"Input: {len(self.output_list)} idea embeddings ({embeddings.shape[1]} dimensions)")
        
        # === Step 1: PCA preprocessing ===
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
      
        # Store PCA embeddings
        for item, pca_embed in zip(self.output_list, pca_embeddings):
            item.pca_embedding = pca_embed
        
        # === Step 2: UMAP embedding ===
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Step 2: UMAP embedding...")
        self.verbose_reporter.stat_line(f"Configuration: {self.umap_config.n_neighbors} neighbors, {self.umap_config.n_components} components")
        start_time = time.time()
        umap_embeddings = self._umap_embed(pca_embeddings)
        elapsed_time = time.time() - start_time
        self.verbose_reporter.stat_line(f"Reduced: {pca_embeddings.shape[1]} → {umap_embeddings.shape[1]} dimensions")
        self.verbose_reporter.stat_line(f"UMAP completed in {elapsed_time:.1f}s")
      
        
        # Store UMAP embeddings 
        for item, umap_embed in zip(self.output_list, umap_embeddings):
            item.umap_embedding = umap_embed
        
        # Use UMAP embeddings directly (no normalization needed for euclidean clustering)
        U = umap_embeddings
        
        # === Step 3: Automatic HDBSCAN parameter optimization ===
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Step 3: Finding optimal clustering parameters...")
        
        best_model, labels, summary = self._auto_hdbscan_grid(U, embeddings)
        
        # Report best configuration
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("Optimal clustering configuration found:")
        self.verbose_reporter.stat_line(f"  Min cluster size: {summary.min_cluster_size}")
        self.verbose_reporter.stat_line(f"  Clusters: {summary.n_clusters}")
        self.verbose_reporter.stat_line(f"  Noise rate: {summary.noise_rate:.1%}")
        self.verbose_reporter.stat_line(f"  Median cluster size: {summary.median_cluster_size}")
       
        # === Step 4: Cluster similarity analysis ===
        self.verbose_reporter.empty_line()
        self._analyze_cluster_similarity(embeddings, labels)
        
        # === Step 5: Optional cluster merging ===
        if self.hdbscan_config.merge_similar_clusters:
            self.verbose_reporter.empty_line()
            self.verbose_reporter.stat_line("Step 5: Merging similar clusters...")
            labels = self._merge_clusters_by_similarity(
                embeddings,
                labels,
                sim_threshold=self.hdbscan_config.merge_similarity_threshold
            )
        
        # === Step 6: Assign final labels to items ===
        for item, label in zip(self.output_list, labels):
            item.initial_idea_cluster = int(label)
        
        # === Step 7: Calculate final statistics ===
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = list(labels).count(-1)
        
        # Calculate cluster size statistics
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
            
            # Report final statistics
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
        
        # === Step 8: Display sample clusters ===
        self._display_sample_clusters(labels, cluster_sizes)
        
        # Complete the step
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
