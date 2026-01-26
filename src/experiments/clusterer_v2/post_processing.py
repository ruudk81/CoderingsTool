"""
Clusterer Post-Processing Module

Implements:
- Cluster merging (graph-based transitive closure with union-find)
- Noise reduction strategies:
  - BERTopic-style: Assign noise to nearest cluster by embedding similarity
  - Legacy HDBSCAN: Re-run HDBSCAN on noise points
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

from .config import ClustererV2Config


class UnionFind:
    """Union-Find data structure for transitive closure in cluster merging."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def get_components(self) -> Dict[int, int]:
        """Return mapping from element to component representative."""
        return {e: self.find(e) for e in self.parent}


def compute_cluster_centroids(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """
    Compute centroid for each cluster.

    Args:
        labels: Cluster labels
        embeddings: L2-normalized embeddings

    Returns:
        (centroids dict, sizes dict)
    """
    centroids = {}
    sizes = {}

    unique_labels = [l for l in set(labels) if l >= 0]
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        sizes[label] = len(cluster_embeddings)

        # Compute centroid and L2-normalize
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid

    return centroids, sizes


def pairwise_cluster_similarity(
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    embeddings: np.ndarray
) -> Dict[str, float]:
    """
    Calculate pairwise similarity statistics between two clusters.

    Args:
        indices_a: Indices of cluster A points
        indices_b: Indices of cluster B points
        embeddings: All embeddings

    Returns:
        Dict with q25, q50, q75 percentile similarities
    """
    emb_a = embeddings[indices_a]
    emb_b = embeddings[indices_b]

    # Pairwise cosine similarities (dot product for L2-normalized)
    sim_matrix = emb_a @ emb_b.T
    all_sims = sim_matrix.flatten()

    return {
        'q25': float(np.percentile(all_sims, 25)),
        'q50': float(np.percentile(all_sims, 50)),
        'q75': float(np.percentile(all_sims, 75)),
        'mean': float(np.mean(all_sims))
    }


def renumber_clusters(labels: np.ndarray) -> np.ndarray:
    """
    Renumber cluster labels to be sequential starting from 0.

    Args:
        labels: Cluster labels (may have gaps, -1 for noise)

    Returns:
        Renumbered labels with sequential IDs
    """
    new_labels = labels.copy()
    unique_labels = sorted(set(labels) - {-1})

    mapping = {old: new for new, old in enumerate(unique_labels)}
    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels


def reduce_noise_by_embedding_similarity(
    labels: np.ndarray,
    embeddings: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    BERTopic-style noise reduction: assign noise points to nearest cluster by embedding similarity.

    Based on BERTopic Strategy #4 (embeddings). Each noise point is assigned to the
    cluster whose centroid has the highest cosine similarity, if above threshold.

    Args:
        labels: Cluster labels with -1 for noise points
        embeddings: L2-normalized embeddings
        threshold: Minimum cosine similarity to assign noise to cluster (default 0.5)
        verbose: Print progress

    Returns:
        Tuple of (updated_labels, stats_dict)
    """
    noise_mask = labels == -1
    n_total_noise = noise_mask.sum()

    if n_total_noise == 0:
        if verbose:
            print("  No noise points to reduce")
        return labels, {'n_noise_initial': 0, 'n_assigned': 0, 'n_noise_final': 0}

    if verbose:
        print(f"\n[Noise Reduction - Embedding Similarity]")
        print(f"  Initial noise points: {n_total_noise}")
        print(f"  Similarity threshold: {threshold}")

    # Compute cluster centroids
    centroids, sizes = compute_cluster_centroids(labels, embeddings)

    if len(centroids) == 0:
        if verbose:
            print("  No clusters found - cannot reduce noise")
        return labels, {'n_noise_initial': n_total_noise, 'n_assigned': 0, 'n_noise_final': n_total_noise}

    # Build centroid matrix
    cluster_ids = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])

    # Get noise embeddings
    noise_indices = np.where(noise_mask)[0]
    noise_embeddings = embeddings[noise_indices]

    # Compute cosine similarities between noise points and cluster centroids
    # embeddings are L2-normalized, so dot product = cosine similarity
    similarities = noise_embeddings @ centroid_matrix.T

    # Find best cluster for each noise point
    best_cluster_indices = np.argmax(similarities, axis=1)
    best_scores = np.max(similarities, axis=1)

    # Assign noise points above threshold
    new_labels = labels.copy()
    n_assigned = 0

    for i, (noise_idx, cluster_idx, score) in enumerate(zip(noise_indices, best_cluster_indices, best_scores)):
        if score >= threshold:
            new_labels[noise_idx] = cluster_ids[cluster_idx]
            n_assigned += 1

    n_noise_final = (new_labels == -1).sum()
    assignment_rate = n_assigned / n_total_noise if n_total_noise > 0 else 0.0

    if verbose:
        print(f"  Assigned: {n_assigned} ({assignment_rate:.1%})")
        print(f"  Remaining noise: {n_noise_final} ({n_noise_final/len(labels):.1%})")

    stats = {
        'n_noise_initial': n_total_noise,
        'n_assigned': n_assigned,
        'n_noise_final': n_noise_final,
        'assignment_rate': assignment_rate
    }

    return new_labels, stats


def merge_similar_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    config: ClustererV2Config,
    verbose: bool = True
) -> np.ndarray:
    """
    Merge clusters using graph-based transitive closure with union-find.

    Process:
    1. Compute centroids for all clusters
    2. Find candidate pairs with centroid similarity >= threshold
    3. Validate candidates with pairwise quantile analysis
    4. Use union-find for transitive closure
    5. Renumber to sequential IDs

    Args:
        labels: Initial cluster assignments
        embeddings: L2-normalized embeddings
        config: ClustererV2Config
        verbose: Print progress

    Returns:
        Updated cluster labels with merged clusters
    """
    if not config.enable_merging:
        return labels

    # Compute centroids
    centroids, sizes = compute_cluster_centroids(labels, embeddings)
    n_initial_clusters = len(centroids)

    if n_initial_clusters < 2:
        if verbose:
            print("  Less than 2 clusters - skipping merge")
        return labels

    if verbose:
        print(f"\n[Cluster Merging]")
        print(f"  Initial clusters: {n_initial_clusters}")
        print(f"  Centroid threshold: {config.merge_centroid_threshold}")
        print(f"  Pairwise threshold: {config.merge_pairwise_threshold}")

    # Build cluster_to_indices mapping
    cluster_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_to_indices[int(label)].append(i)

    # Find candidate pairs based on centroid similarity
    cluster_ids = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
    centroid_similarities = centroid_matrix @ centroid_matrix.T

    candidates = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            sim = centroid_similarities[i, j]
            if sim >= config.merge_centroid_threshold:
                candidates.append((cluster_ids[i], cluster_ids[j], sim))

    if verbose:
        print(f"  Candidate pairs (centroid >= {config.merge_centroid_threshold}): {len(candidates)}")

    if not candidates:
        if verbose:
            print("  No similar clusters found - no merging needed")
        return labels

    # Evaluate pairwise similarity for all candidates
    merge_edges = []
    for cluster_a, cluster_b, centroid_sim in candidates:
        indices_a = np.array(cluster_to_indices[cluster_a])
        indices_b = np.array(cluster_to_indices[cluster_b])

        stats = pairwise_cluster_similarity(indices_a, indices_b, embeddings)
        quantile_mean = np.mean([stats['q25'], stats['q50'], stats['q75']])

        if quantile_mean >= config.merge_pairwise_threshold:
            merge_edges.append((cluster_a, cluster_b, centroid_sim, quantile_mean))
            if verbose:
                print(f"    ✓ Merge {cluster_a}↔{cluster_b} | "
                      f"sizes: {sizes[cluster_a]}, {sizes[cluster_b]} | "
                      f"centroid: {centroid_sim:.3f} | quantile_mean: {quantile_mean:.3f}")

    if not merge_edges:
        if verbose:
            print("  No merge-worthy pairs found")
        return labels

    # Union-find for transitive closure
    uf = UnionFind(cluster_ids)
    for cluster_a, cluster_b, _, _ in merge_edges:
        uf.union(cluster_a, cluster_b)

    # Get component mapping
    component_map = uf.get_components()

    # Relabel all points to their component representative
    labels_merged = labels.copy()
    for old_id, component_id in component_map.items():
        labels_merged[labels == old_id] = component_id

    # Renumber to sequential IDs
    labels_final = renumber_clusters(labels_merged)

    # Report results
    unique_components = set(component_map.values())
    n_final_clusters = len(unique_components)
    n_merged = n_initial_clusters - n_final_clusters

    if verbose:
        print(f"  Merging complete:")
        print(f"    Initial: {n_initial_clusters} → Final: {n_final_clusters}")
        print(f"    Reduction: {n_merged} clusters removed")

    return labels_final


def assess_noise_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cohesion_threshold: float
) -> List[int]:
    """
    Assess quality of noise-derived clusters and return valid ones.

    A cluster is valid if:
    - Size >= min_cluster_size (implicit from HDBSCAN)
    - Internal cohesion >= threshold

    Args:
        embeddings: L2-normalized embeddings (noise subset)
        labels: Cluster labels from noise reclustering
        cohesion_threshold: Minimum mean pairwise similarity

    Returns:
        List of valid cluster IDs
    """
    valid_clusters = []
    unique_labels = [l for l in set(labels) if l >= 0]

    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) < 2:
            valid_clusters.append(label)  # Single-point clusters pass
            continue

        # Calculate cohesion (mean pairwise similarity)
        similarities = cluster_embeddings @ cluster_embeddings.T
        n = len(cluster_embeddings)
        upper_tri = np.triu_indices(n, k=1)
        mean_sim = float(np.mean(similarities[upper_tri]))

        if mean_sim >= cohesion_threshold:
            valid_clusters.append(label)

    return valid_clusters


def recluster_noise(
    labels: np.ndarray,
    umap_embeddings: np.ndarray,
    original_embeddings: np.ndarray,
    config: ClustererV2Config,
    verbose: bool = True
) -> np.ndarray:
    """
    Two-pass clustering: Attempt to find viable clusters among noise points.

    Args:
        labels: Current cluster labels (with -1 for noise)
        umap_embeddings: UMAP embeddings
        original_embeddings: L2-normalized original embeddings
        config: ClustererV2Config
        verbose: Print progress

    Returns:
        Updated labels with noise-derived clusters numbered sequentially
    """
    if not config.enable_noise_reclustering:
        return labels

    noise_mask = labels == -1
    n_total_noise = noise_mask.sum()

    # Need minimum noise points
    min_total = 10  # Default minimum
    if n_total_noise < min_total:
        if verbose:
            print(f"  Skipping noise reclustering: only {n_total_noise} noise points (minimum: {min_total})")
        return labels

    if verbose:
        print(f"\n[Noise Reclustering]")
        print(f"  Total noise points: {n_total_noise}")

    # Calculate noise reclustering parameters
    # Use more aggressive (smaller) parameters for noise
    noise_mcs = max(3, config.noise_min_cluster_size)
    noise_ms = max(1, noise_mcs // 2)

    if verbose:
        print(f"  Parameters: min_cluster_size={noise_mcs}, min_samples={noise_ms}")

    # Run HDBSCAN on noise points
    U_noise = umap_embeddings[noise_mask]

    noise_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=noise_mcs,
        min_samples=noise_ms,
        metric='euclidean',
        cluster_selection_method='leaf',
        gen_min_span_tree=True
    )
    noise_labels = noise_hdbscan.fit_predict(U_noise)

    # Quality filtering
    original_noise = original_embeddings[noise_mask]
    valid_noise_clusters = assess_noise_cluster_quality(
        original_noise, noise_labels, config.noise_cohesion_threshold
    )

    if len(valid_noise_clusters) == 0:
        if verbose:
            print(f"  No viable clusters found in noise")
        return labels

    # Renumber and integrate
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

    # Reporting
    n_noise_clusters = len(valid_noise_clusters)
    recovery_rate = n_recovered / n_total_noise if n_total_noise > 0 else 0.0
    final_noise = (labels_updated == -1).sum()

    if verbose:
        print(f"  Viable clusters discovered: {n_noise_clusters}")
        print(f"  Points recovered: {n_recovered} ({recovery_rate:.1%})")
        print(f"  Residual noise: {final_noise} ({final_noise/len(labels):.1%})")

    return labels_updated
