"""
Test if clustering is deterministic when using identical inputs and order.
"""

import numpy as np
from utils.clean_clusterer import CleanClusterer
from config import DEFAULT_CLUSTERING_CONFIG

def test_clustering_determinism(embedded_text, num_tests=3):
    """Test if clustering produces identical results with same inputs"""
    
    print("="*80)
    print("CLUSTERING DETERMINISM TEST")
    print("="*80)
    
    # Extract embeddings in exact order from embedded_text
    embeddings = []
    segment_ids = []
    descriptions = []
    
    for resp in embedded_text:
        if resp.response_segment:
            for seg in resp.response_segment:
                if seg.description_embedding is not None:
                    embeddings.append(seg.description_embedding)
                    segment_ids.append(seg.segment_id)
                    descriptions.append(seg.segment_description)
    
    embeddings_array = np.array(embeddings)
    print(f"Testing with {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings_array.shape}")
    
    # Run clustering multiple times with same config
    cluster_results = []
    
    for test_num in range(num_tests):
        print(f"\nTest {test_num + 1}:")
        
        # Create fresh clusterer instance
        clusterer = CleanClusterer(DEFAULT_CLUSTERING_CONFIG)
        
        # Apply UMAP
        reduced_embeddings = clusterer.umap_model.fit_transform(embeddings_array)
        print(f"  UMAP reduced to: {reduced_embeddings.shape}")
        
        # Apply HDBSCAN
        cluster_labels = clusterer.hdbscan_model.fit_predict(reduced_embeddings)
        print(f"  Clusters found: {len(set(cluster_labels) - {-1})}")
        print(f"  Noise points: {np.sum(cluster_labels == -1)}")
        
        cluster_results.append(cluster_labels.copy())
    
    # Compare results
    print(f"\n{'='*60}")
    print("DETERMINISM ANALYSIS")
    print(f"{'='*60}")
    
    if num_tests > 1:
        all_identical = True
        for i in range(1, num_tests):
            if not np.array_equal(cluster_results[0], cluster_results[i]):
                all_identical = False
                differences = np.sum(cluster_results[0] != cluster_results[i])
                print(f"❌ Test 1 vs Test {i+1}: {differences} different assignments")
                
                # Show example differences
                diff_indices = np.where(cluster_results[0] != cluster_results[i])[0][:5]
                for idx in diff_indices:
                    print(f"  Segment {segment_ids[idx]}: {cluster_results[0][idx]} → {cluster_results[i][idx]}")
        
        if all_identical:
            print("✅ All clustering runs produced identical results!")
        else:
            print("❌ Clustering is non-deterministic - results vary between runs")
    
    return cluster_results[0]

# Usage:
# cluster_labels = test_clustering_determinism(embedded_text)