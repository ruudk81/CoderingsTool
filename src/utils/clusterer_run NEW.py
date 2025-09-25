from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from umap import UMAP

from hdbscan.validity import validity_index
from sklearn.metrics import silhouette_score, silhouette_samples

CLUSTER_METRIC = "euclidean"     
DBCOV_D = 1                      # safe for DBCV (avoid overflow)

@dataclass
class ClusterSummary:
    n_points: int
    min_cluster_size: int
    n_clusters: int
    noise_rate: float
    dbcv: float
    silhouette: float
    mean_probability: float
    median_cluster_size: int

def pca_reduce(
    embeddings: np.ndarray, 
    n_components: int = 100,
    random_state: int = 42, 
    dtype: str = "float64") -> np.ndarray:
    
    pca = PCA(n_components=n_components, random_state=random_state)
    Xp = pca.fit_transform(embeddings).astype(dtype, copy=False)
    return Xp

def umap_embed(
    X: np.ndarray, 
    n_neighbors: int = 15, 
    n_components: int = 10,
    metric: str = "cosine", 
    n_epochs: int = 200,
    random_state: int = 42, 
    low_memory: bool = True,
    normalize_output: bool = False
    ) -> np.ndarray:
   
    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.1,
        metric=metric, # cosine in INPUT space
        n_epochs=n_epochs,
        random_state=random_state,
        transform_seed=random_state,
        low_memory=low_memory,
        verbose=False
    )
    U = umap.fit_transform(X).astype(np.float64, copy=False)

    return U

def dbcv_score(
    X: np.ndarray, 
    labels: np.ndarray,
    metric: str = CLUSTER_METRIC, 
    d_override: int = DBCOV_D
    ) -> float:
    
    mask = labels >= 0
    X, y = X[mask], labels[mask]
    if X.shape[0] == 0: return np.nan
    # drop singleton clusters
    clusters, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, clusters[counts >= 2])
    X, y = X[keep], y[keep]
    if np.unique(y).size < 2: return np.nan
    X = X.astype(np.float64, copy=False)
    return float(validity_index(X, y, metric=metric, d=d_override))

def silhouette_full(
    X: np.ndarray, 
    labels: np.ndarray,
    metric: str = CLUSTER_METRIC,
    sample_size: Optional[int] = None
    ) -> float:
    
    mask = labels >= 0
    Xn, yn = X[mask], labels[mask]
    if Xn.shape[0] == 0: return np.nan
    _, counts = np.unique(yn, return_counts=True)
    if (counts >= 2).sum() < 2: return np.nan
    return float(silhouette_score(Xn, yn, metric=metric, sample_size=sample_size, random_state=42))

def mean_probability(
    db: HDBSCAN, 
    labels: np.ndarray
    ) -> float:
    
    mask = labels >= 0
    if not np.any(mask): return np.nan
    return float(np.mean(db.probabilities_[mask]))

def scoring_metrics(
    U, 
    labels, 
    db, 
    metric="euclidean",
    sil_cluster_thresh=0.40, 
    sil_point_thresh=0.20,
    sample_size_for_sil=None):
    
    # mask labeled
    mask = labels >= 0
    X, y = U[mask], labels[mask]
    n_clusters = int(np.unique(y).size)
    noise_rate = float(np.mean(labels == -1))

    # global metrics
    dbcv_val = dbcv_score(U, labels, metric=metric, d_override=DBCOV_D)
    sil_val  = silhouette_full(U, labels, metric=metric, sample_size=sample_size_for_sil)
    meanp    = mean_probability(db, labels)

    # per-sample silhouettes (on labeled points)
    if X.shape[0] and n_clusters >= 2:
        sil_samp = silhouette_samples(X, y, metric=metric)
        # fraction of low-sil points
        frac_low_points = float((sil_samp < sil_point_thresh).mean())
        # mean sil per cluster
        clus, counts = np.unique(y, return_counts=True)
        means = []
        # faster grouping using boolean masks
        frac_low_clusters = 0.0
        for c in clus:
            w = (y == c)
            s = sil_samp[w]
            means.append(s.mean())
        means = np.array(means)
        frac_low_clusters = float((means < sil_cluster_thresh).mean())
    else:
        frac_low_points = 0.0
        frac_low_clusters = 0.0

    return {
        "dbcv": dbcv_val,                        # Density-Based Clustering Validation Index [-1,1], higher = better density separation. >0.5 is good.
        "sil": sil_val,                          # Global silhouette [-1,1], higher = clearer separation. >0.5 good, >0.7 excellent.
        "meanp": meanp,                          # Mean HDBSCAN cluster probability [0,1], higher = more stable/dense clusters. >0.7 good, >0.9 excellent.
        "n_clusters": n_clusters,                # Number of clusters (ignoring noise). Needs to land in interpretable range (e.g. 30–80).
        "noise_rate": noise_rate,                # Fraction of points labeled as noise [-1]. 0.1–0.25 often healthy, >0.35 = too much discarded.
        "frac_low_points": frac_low_points,      # Share of points with silhouette < 0.0. High (>0.2) means many points are badly placed/ambiguous.
        "frac_low_clusters": frac_low_clusters,  # Share of clusters with mean silhouette < 0.2. High (>0.2) = lots of weak/overlapping clusters.
    }

def auto_hdbscan_grid(
    U: np.ndarray,
    mcs_grid: Optional[Iterable[int]] = None,
    sample_size_for_sil: Optional[int] = None
    ) -> Tuple[HDBSCAN, np.ndarray, ClusterSummary]:

    n = U.shape[0]
    if mcs_grid is None:
        mcs_grid = sorted(set([
            5, 10, 15,
            max(5, int(0.25 * np.sqrt(n))),
            max(5, int(0.50 * np.sqrt(n))),
            max(5, int(1.00 * np.sqrt(n))),
            max(5, int(1.50 * np.sqrt(n))),
            max(5, int(0.005 * n)),
            max(5, int(0.01 * n)),
        ]))

    best_tuple = None  # (score, db, labels, summary)

    for mcs in mcs_grid:
        db = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=None,   # ties to mcs by default
            metric=CLUSTER_METRIC,
            cluster_selection_method="eom",
            cluster_selection_epsilon=0.0,
            alpha=1.0
        ).fit(U)

        labels = db.labels_
        noise_rate = float(np.mean(labels == -1))
        n_clusters = int(np.unique(labels[labels >= 0]).size)

        # per-run metrics
        dbcv = dbcv_score(U, labels, metric=CLUSTER_METRIC, d_override=DBCOV_D)
        sil  = silhouette_full(U, labels, metric=CLUSTER_METRIC, sample_size=sample_size_for_sil)
        meanp = mean_probability(db, labels)

        # median cluster size (ignoring noise)
        _, counts = np.unique(labels[labels >= 0], return_counts=True)
        med_size = int(np.median(counts)) if counts.size else 0

        M = scoring_metrics(
            U, 
            labels, 
            db, 
            metric=CLUSTER_METRIC,
            sil_cluster_thresh=0.20, 
            sil_point_thresh=0.00,
            sample_size_for_sil=None)
        
        w_dbcv, w_sil, w_prob = 1.0, 1.0, 0.5
        w_k, w_noise, w_badC, w_badP = 0.0005, 0.5, 0.5, 0.3
        
        score = (
          np.nan_to_num(M["dbcv"]) * w_dbcv
          + np.nan_to_num(M["sil"])  * w_sil
          + np.nan_to_num(M["meanp"]) * w_prob
          - w_k * M["n_clusters"]
          - w_noise * M["noise_rate"]
          - w_badC * M["frac_low_clusters"]
          - w_badP * M["frac_low_points"]
          )

        #score = (np.nan_to_num(dbcv) + np.nan_to_num(sil) + np.nan_to_num(meanp)) - 0.0005 * n_clusters

        summary = ClusterSummary(
            n_points=n,
            min_cluster_size=int(mcs),
            n_clusters=int(n_clusters),
            noise_rate=float(round(noise_rate, 3)),
            dbcv=float(dbcv) if not np.isnan(dbcv) else np.nan,
            silhouette=float(sil) if not np.isnan(sil) else np.nan,
            mean_probability=float(meanp) if not np.isnan(meanp) else np.nan,
            median_cluster_size=int(med_size),
        )

        if (best_tuple is None) or (score > best_tuple[0]):
            best_tuple = (score, db, labels, summary)

        print(f"score={score:.3f} | mcs={mcs:>4} | K={M['n_clusters']:>4} | noise={M['noise_rate']:.3f} | "
              f"DBCV={M['dbcv']:.3f} | Sil={M['sil']:.3f} | P={M['meanp']:.3f} | "
              f"lowC={M['frac_low_clusters']:.2f} | lowP={M['frac_low_points']:.2f}")
        
    _, best_db, best_labels, best_summary = best_tuple
    return best_db, best_labels, best_summary


def cluster_embeddings(
    embeddings: np.ndarray,
    pca_components: int = 100,
    umap_neighbors: int = 15,
    umap_components: int = 10,
    umap_metric: str = "cosine",
    umap_epochs: int = 200,
    umap_low_memory: bool = False,
    mcs_grid: Optional[Iterable[int]] = None):
    
    Xp = pca_reduce(embeddings, n_components=pca_components)
    U = umap_embed(
        Xp,
        n_neighbors=umap_neighbors,
        n_components=umap_components,
        metric=umap_metric,        # cosine in INPUT space
        n_epochs=umap_epochs,
        low_memory=umap_low_memory,
        normalize_output=False     
    )
    model, labels, summary = auto_hdbscan_grid(U, mcs_grid=mcs_grid, sample_size_for_sil=None)
    return U, labels, Xp, summary, model


U, labels, X50, summary, model = cluster_embeddings(
    embeddings, 
    pca_components=100, 
    umap_neighbors=15, 
    umap_components=10, 
    umap_metric="cosine", 
    umap_epochs=200, 
    umap_low_memory=False, 
    mcs_grid=None, 
    ) 


print(summary) 
print("#clusters:", summary.n_clusters, "noise_rate:", summary.noise_rate)

        
unique, counts = np.unique(labels[labels >= 0], return_counts=True)
cluster_sizes = dict(zip(unique, counts))
cluster_report = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

n_clusters = len(unique)
mean_size = counts.mean()
median_size = np.median(counts)

# quartiles
n1, q1, q2, q3, q4 = np.percentile(counts, [0, 25, 50, 75, 100])

print(f"Number of clusters: {n_clusters}\n")
print(f"Average size: {mean_size:.1f}")
print(f"Median size: {round(median_size)}\n")
print(f"min: {round(n1)}")
print(f"Q1 (25th pct): {round(q1)}")
print(f"Q2 (50th pct): {round(q2)}")
print(f"Q3 (75th pct): {round(q3)}")
print(f"max: {round(q4)}")
print("\nTop5:")
for cid, n in cluster_report[:5]:   # top 10 largest clusters
    print(f"-Cluster {cid}: n= {n}")

if False: #debug
    sampled_cluster = random.randint(0, n_clusters - 1)
    #sampled_cluster = 21
    cluster_ideas = [idea for idea, label in zip(ideas, labels) if label == sampled_cluster]
    sample_size = min(50, len(cluster_ideas))
    sampled_ideas = random.sample(cluster_ideas, sample_size)
    for idea in sampled_ideas:
        print(idea)


# === CLUSTER SIMILARITY ANALYSIS ========================================================================================================
"""Analyze similarity between clusters"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

if True:
    verbose_reporter.section_header("CLUSTER SIMILARITY ANALYSIS")
    
    # Extract cluster embeddings and calculate centroids
    cluster_embeddings = defaultdict(list)
    cluster_sizes = defaultdict(int)
    
    # Group embeddings by cluster ID
    for idea, embedding, label in zip(ideas, embeddings, labels): 
        if label is not None and label != -1:  # Exclude noise points
            cluster_id = label
            cluster_embeddings[cluster_id].append(embedding)
            cluster_sizes[cluster_id] += 1
    
    # Calculate cluster centroids (mean of all embeddings in cluster)
    cluster_centroids = {}
    for id_cluster, embeddings_cluster in cluster_embeddings.items():
        if embeddings_cluster:  # Only process clusters with embeddings
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
        print("\nCLUSTER SIMILARITY DISTRIBUTION")
        print(f"Analyzing {num_clusters} clusters ({total_pairs} unique pairs)")
        print("-" * 60)
        
        # Thresholds to analyze
        thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        
        for threshold in thresholds:
            count = np.sum(similarities >= threshold)
            percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
            print(f"Similarity >= {threshold:.2f}: {count:4d} pairs ({percentage:5.1f}%)")
        
        # Find and display most similar cluster pairs
        print("\nTOP 10 MOST SIMILAR CLUSTER PAIRS:")
        print("-" * 60)
        
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
            
            print(f"{rank:2d}. Cluster {cluster_i} ({size_i} ideas) <-> Cluster {cluster_j} ({size_j} ideas): {similarity:.3f}")
    else:
        print("\nCLUSTER SIMILARITY ANALYSIS")
        print("Not enough clusters for similarity analysis (need at least 2)")
    
    print("\n" + "=" * 80)

# === CLUSTER MERGING ========================================================================================================
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from collections import defaultdict

def compute_centroids(embeddings: np.ndarray, labels: np.ndarray):
    """Return {cluster_id: centroid} dict and {cluster_id: size} dict for labels>=0."""
    by_cluster = defaultdict(list)
    for v, y in zip(embeddings, labels):
        if y is not None and y >= 0:
            by_cluster[int(y)].append(v)
    centroids = {}
    sizes = {}
    for cid, vecs in by_cluster.items():
        V = np.vstack(vecs)
        centroids[cid] = V.mean(axis=0)
        sizes[cid] = V.shape[0]
    return centroids, sizes

def merge_clusters_by_centroid_cosine(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sim_threshold: float = 0.90,
    linkage: str = "complete",
):
    """
    Merge HDBSCAN clusters whose centroids have cosine similarity >= sim_threshold,
    using complete-linkage agglomerative clustering on centroid vectors.
    Returns: new_labels (same shape as labels), merge_map (old->new), groups (list of sets of old ids)
    """
    assert embeddings.shape[0] == labels.shape[0], "embeddings/labels length mismatch"

    # 1) Build centroids in ORIGINAL embedding space
    centroids, sizes = compute_centroids(embeddings, labels)
    if not centroids:
        # no clusters to merge
        return labels.copy(), {}, []

    orig_ids = sorted(centroids.keys())
    C = np.vstack([centroids[cid] for cid in orig_ids])

    # 2) L2-normalize (cosine distance is scale-invariant but this is a safe habit)
    Cn = normalize(C)  # shape: (k, d)

    # 3) Agglomerative on centroids with cosine distance
    # cosine similarity >= 0.90  -> cosine distance <= 0.10
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
        # Older scikit-learn: use 'affinity' instead of 'metric'
        ag = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage=linkage,
            distance_threshold=dist_threshold,
            compute_full_tree=True,
        )

    merged_ids = ag.fit_predict(Cn)  # length k (number of original clusters)

    # 4) Build mapping old_id -> merged_group_id (renumber groups to 0..G-1 for neatness)
    # Keep groups stable & compact
    uniq_groups = {g: i for i, g in enumerate(sorted(np.unique(merged_ids)))}
    old_to_new_group = {old: uniq_groups[g] for old, g in zip(orig_ids, merged_ids)}

    # 5) Remap point labels: noise stays -1; cluster y>=0 becomes its group's id
    new_labels = labels.copy()
    for i, y in enumerate(labels):
        if y is not None and y >= 0:
            new_labels[i] = old_to_new_group[int(y)]
        else:
            new_labels[i] = -1

    # 6) Human-friendly report of merges (groups of original cluster ids)
    groups = defaultdict(list)
    for old, g in old_to_new_group.items():
        groups[g].append(old)
    groups = [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: kv[0])]

    # Optional: print a compact summary
    print("\n=== MERGE SUMMARY (cosine, complete) ===")
    print(f"Original clusters: {len(orig_ids)}  →  Merged clusters: {len(groups)}")
    print(f"Threshold: cosine ≥ {sim_threshold:.2f} (distance ≤ {dist_threshold:.2f})")
    # Show only groups that actually merged (size > 1)
    merged_groups = [g for g in groups if len(g) > 1]
    if merged_groups:
        print("\nMerged groups (original cluster IDs):")
        for g in merged_groups[:20]:
            total_n = sum(sizes.get(cid, 0) for cid in g)
            parts = ", ".join(f"{cid}(n={sizes.get(cid,0)})" for cid in g)
            print(f"- {{ {parts} }}  → total n={total_n}")
        if len(merged_groups) > 20:
            print(f"... and {len(merged_groups)-20} more groups.")
    else:
        print("No merges needed at this threshold.")

    return new_labels, old_to_new_group, groups

new_labels, old_to_new_map, groups = merge_clusters_by_centroid_cosine(
    embeddings=embeddings,
    labels=labels,
    sim_threshold=0.95,
    linkage="complete",
)

# Quick size report after merging (ignores noise)
uniq, counts = np.unique(new_labels[new_labels >= 0], return_counts=True)
order = np.argsort(counts)[::-1]
print("\nTop5 after merging:")
for cid, n in zip(uniq[order][:5], counts[order][:5]):
    print(f"- Cluster {cid}: n={n}")

if False: #debug
    sampled_cluster = random.randint(0, n_clusters - 1)
    sampled_cluster =5
    cluster_ideas = [idea for idea, label in zip(ideas, new_labels) if label == sampled_cluster]
    sample_size = min(50, len(cluster_ideas))
    sampled_ideas = random.sample(cluster_ideas, sample_size)
    for idea in sampled_ideas:
        print(idea)