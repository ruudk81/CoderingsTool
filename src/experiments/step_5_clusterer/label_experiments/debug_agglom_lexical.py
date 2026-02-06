"""
Debug script to trace HDBSCAN behavior on c-TF-IDF matrix.

Run from project root: python src/experiments/step_5_clusterer/label_experiments/debug_agglom_lexical.py
"""
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from sklearn.preprocessing import normalize

# Run the main analysis first
from experiments.step_5_clusterer.label_experiments.analyze_agglomerative_lexical_groups import (
    main, CONFIG, build_agglom_ctfidf_matrix, cluster_agglom_by_ctfidf
)

print("=" * 70)
print("DEBUG: Tracing HDBSCAN on c-TF-IDF")
print("=" * 70)

# Run main to get results
results = main()

agglom_results = results["agglom_results"]
global_agglom_labels = results["global_agglom_labels"]
id_mapping = results["id_mapping"]
lexical_results = results["lexical_results"]
point_mapping = results["point_mapping"]

# Pick an EOM with many sub-clusters for debugging
eom_sizes = {}
for eom_id, agglom_result in agglom_results.items():
    if not agglom_result.get("skipped"):
        eom_sizes[eom_id] = agglom_result["optimal_k"]

# Find EOM with most sub-clusters
max_eom_id = max(eom_sizes.keys(), key=lambda x: eom_sizes[x])
print(f"\n\nDEBUG: Analyzing EOM {max_eom_id} with {eom_sizes[max_eom_id]} sub-clusters")
print("=" * 70)

# Build c-TF-IDF for this EOM
local_to_global = {}
for global_id, (eid, lid) in id_mapping.items():
    if eid == max_eom_id:
        local_to_global[lid] = global_id

global_ids_this_eom = list(local_to_global.values())
print(f"Global IDs for this EOM: {global_ids_this_eom}")

# Build c-TF-IDF
ctfidf_matrix, vocab, ordered_global_ids = build_agglom_ctfidf_matrix(
    global_ids_this_eom, global_agglom_labels, point_mapping, CONFIG
)

print(f"\nc-TF-IDF matrix shape: {ctfidf_matrix.shape}")
print(f"Vocabulary size: {len(vocab)}")

# Analyze matrix properties
print("\nMatrix statistics:")
print(f"  Min: {ctfidf_matrix.min():.4f}")
print(f"  Max: {ctfidf_matrix.max():.4f}")
print(f"  Mean: {ctfidf_matrix.mean():.4f}")
print(f"  Sparsity: {(ctfidf_matrix == 0).sum() / ctfidf_matrix.size * 100:.1f}%")

# Per-row statistics
print("\nPer-row (sub-cluster) statistics:")
row_sums = ctfidf_matrix.sum(axis=1)
row_nonzeros = (ctfidf_matrix > 0).sum(axis=1)
print(f"  Row sums - min: {row_sums.min():.4f}, max: {row_sums.max():.4f}, mean: {row_sums.mean():.4f}")
print(f"  Non-zero terms per row - min: {row_nonzeros.min()}, max: {row_nonzeros.max()}, mean: {row_nonzeros.mean():.1f}")

# L2 normalize
normalized_matrix = normalize(ctfidf_matrix, norm='l2')
print("\nAfter L2 normalization:")
print(f"  Row L2 norms: all should be ~1.0")
row_norms = np.linalg.norm(normalized_matrix, axis=1)
print(f"    min: {row_norms.min():.4f}, max: {row_norms.max():.4f}")

# Pairwise distances
from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(normalized_matrix)
upper_triangle = dists[np.triu_indices(len(dists), k=1)]
print(f"\nPairwise Euclidean distances (on normalized matrix):")
print(f"  Min: {upper_triangle.min():.4f}")
print(f"  Max: {upper_triangle.max():.4f}")
print(f"  Mean: {upper_triangle.mean():.4f}")
print(f"  Std: {upper_triangle.std():.4f}")

# Run HDBSCAN manually and trace
import hdbscan
print("\n" + "=" * 70)
print("HDBSCAN Analysis")
print("=" * 70)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=CONFIG.lexical_min_cluster_size,  # 2
    min_samples=CONFIG.lexical_min_samples,            # 1
    metric="euclidean",
    cluster_selection_method="eom",
    allow_single_cluster=True
)
labels = clusterer.fit_predict(normalized_matrix)

print(f"\nHDBSCAN config: min_cluster_size={CONFIG.lexical_min_cluster_size}, min_samples={CONFIG.lexical_min_samples}")
print(f"HDBSCAN labels: {labels}")
print(f"Unique labels: {set(labels)}")
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = sum(labels == -1)
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} out of {len(labels)}")

# Check if the problem is that all points are equidistant
if upper_triangle.std() < 0.1:
    print("\n** WARNING: Points have very similar pairwise distances!")
    print("   This makes it hard for HDBSCAN to find dense regions.")

# Try with different min_cluster_size
print("\n" + "-" * 40)
print("Testing alternative HDBSCAN configs:")
print("-" * 40)

for mcs in [2, 3, 4]:
    for ms in [1, 2]:
        c = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=True
        )
        l = c.fit_predict(normalized_matrix)
        nc = len(set(l)) - (1 if -1 in l else 0)
        nn = sum(l == -1)
        print(f"  mcs={mcs}, ms={ms}: {nc} clusters, {nn} noise")

# Compare to overlay.py's approach: what if we have fewer but larger sub-clusters?
print("\n" + "=" * 70)
print("COMPARISON: What if we had fewer sub-clusters?")
print("=" * 70)

# Check overlay.py's LEAF count for this EOM
# We can't directly, but we can infer from the lexical results
lex_result = lexical_results[max_eom_id]
print(f"With sqrt(N) agglom: {len(ordered_global_ids)} sub-clusters")
print(f"Lexical groups found: {lex_result.get('n_lexical_groups', 'N/A')}")
print(f"Sub-cluster groups: {len(lex_result.get('subcluster_groups', {}))}")
