import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

def debug_cluster_alignment(cluster_generator, sample_size=5):
    """
    Debug function to verify that embeddings and descriptions are properly aligned
    in the clustering process.
    """
    print("\n" + "="*80)
    print("DEBUGGING CLUSTER-EMBEDDING-DESCRIPTION ALIGNMENT")
    print("="*80)
    
    # Get the output list from cluster generator
    output_list = cluster_generator.output_list
    embedding_type = cluster_generator.embedding_type
    
    print(f"\nTotal items in output_list: {len(output_list)}")
    print(f"Embedding type being used: {embedding_type}")
    
    # Sample random items for detailed inspection
    if len(output_list) > sample_size:
        sample_indices = random.sample(range(len(output_list)), sample_size)
    else:
        sample_indices = range(len(output_list))
    
    print(f"\nSampling {len(sample_indices)} items for detailed inspection:")
    print("-" * 60)
    
    for idx in sample_indices:
        item = output_list[idx]
        print(f"\nIndex: {idx}")
        print(f"Respondent ID: {item.respondent_id}")
        print(f"Segment ID: {item.segment_id}")
        print(f"Segment Label: {item.segment_label}")
        print(f"Segment Description: {item.segment_description[:100]}...")
        
        if embedding_type == "description":
            print(f"Description Cluster: {item.initial_description_cluster}")
            if item.description_embedding is not None:
                print(f"Description Embedding shape: {item.description_embedding.shape}")
                print(f"Description Embedding first 5 values: {item.description_embedding[:5]}")
            if item.reduced_description_embedding is not None:
                print(f"Reduced Description Embedding shape: {item.reduced_description_embedding.shape}")
        else:
            print(f"Code Cluster: {item.initial_code_cluster}")
            if item.code_embedding is not None:
                print(f"Code Embedding shape: {item.code_embedding.shape}")
                print(f"Code Embedding first 5 values: {item.code_embedding[:5]}")
            if item.reduced_code_embedding is not None:
                print(f"Reduced Code Embedding shape: {item.reduced_code_embedding.shape}")
    
    # Check cluster coherence by calculating similarity within clusters
    print("\n" + "="*60)
    print("CLUSTER COHERENCE CHECK")
    print("="*60)
    
    # Get unique clusters
    if embedding_type == "description":
        clusters = [item.initial_description_cluster for item in output_list if item.initial_description_cluster is not None]
        embeddings = [item.description_embedding for item in output_list if item.initial_description_cluster is not None]
    else:
        clusters = [item.initial_code_cluster for item in output_list if item.initial_code_cluster is not None]
        embeddings = [item.code_embedding for item in output_list if item.initial_code_cluster is not None]
    
    unique_clusters = sorted(set(clusters))
    print(f"\nUnique clusters (excluding None): {unique_clusters[:10]}..." if len(unique_clusters) > 10 else f"\nUnique clusters: {unique_clusters}")
    
    # Sample a few clusters to check coherence
    sample_clusters = random.sample(unique_clusters, min(3, len(unique_clusters)))
    
    for cluster_id in sample_clusters:
        print(f"\n--- Cluster {cluster_id} ---")
        
        # Get items in this cluster
        cluster_items = []
        cluster_embeddings = []
        
        for i, item in enumerate(output_list):
            if embedding_type == "description" and item.initial_description_cluster == cluster_id:
                cluster_items.append(item)
                if item.description_embedding is not None:
                    cluster_embeddings.append(item.description_embedding)
            elif embedding_type == "code" and item.initial_code_cluster == cluster_id:
                cluster_items.append(item)
                if item.code_embedding is not None:
                    cluster_embeddings.append(item.code_embedding)
        
        print(f"Items in cluster: {len(cluster_items)}")
        
        # Show sample descriptions
        print("\nSample descriptions from this cluster:")
        for i in range(min(5, len(cluster_items))):
            desc = cluster_items[i].segment_description if embedding_type == "description" else cluster_items[i].segment_label
            print(f"  - {desc[:80]}...")
        
        # Calculate average pairwise similarity within cluster
        if len(cluster_embeddings) > 1:
            embeddings_array = np.array(cluster_embeddings)
            similarities = cosine_similarity(embeddings_array)
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarities, k=1)
            non_zero_similarities = upper_triangle[upper_triangle > 0]
            
            if len(non_zero_similarities) > 0:
                avg_similarity = np.mean(non_zero_similarities)
                min_similarity = np.min(non_zero_similarities)
                max_similarity = np.max(non_zero_similarities)
                
                print(f"\nIntra-cluster similarity stats:")
                print(f"  Average: {avg_similarity:.3f}")
                print(f"  Min: {min_similarity:.3f}")
                print(f"  Max: {max_similarity:.3f}")
                
                # Flag if similarity is suspiciously low
                if avg_similarity < 0.3:
                    print("  ⚠️  WARNING: Low average similarity suggests possible misalignment!")
    
    # Check for duplicate descriptions in different clusters
    print("\n" + "="*60)
    print("DUPLICATE DESCRIPTION CHECK")
    print("="*60)
    
    description_to_clusters = {}
    for item in output_list:
        desc = item.segment_description if embedding_type == "description" else item.segment_label
        cluster = item.initial_description_cluster if embedding_type == "description" else item.initial_code_cluster
        
        if cluster is not None and desc:
            if desc not in description_to_clusters:
                description_to_clusters[desc] = []
            description_to_clusters[desc].append(cluster)
    
    # Find descriptions that appear in multiple clusters
    duplicates_found = False
    for desc, clusters in description_to_clusters.items():
        unique_clusters_for_desc = list(set(clusters))
        if len(unique_clusters_for_desc) > 1:
            if not duplicates_found:
                print("\n⚠️  Descriptions appearing in multiple clusters:")
                duplicates_found = True
            print(f"  '{desc[:60]}...' appears in clusters: {unique_clusters_for_desc}")
    
    if not duplicates_found:
        print("\n✅ No duplicate descriptions found across different clusters")
    
    print("\n" + "="*80)
    print("END OF ALIGNMENT DEBUG")
    print("="*80)


# If running as a script
if __name__ == "__main__":
    print("This is a debug utility. Import and use debug_cluster_alignment() function with a ClusterGenerator instance.")