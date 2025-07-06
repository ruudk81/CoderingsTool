"""
End-to-end alignment test to definitively prove where misalignment occurs.
This will trace specific segments through the entire pipeline.
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import models
import numpy as np
from typing import List, Dict
from utils.enhanced_embedder import EnhancedEmbedder
from utils.clusterer import ClusterGenerator
from config import EmbeddingConfig, ClusteringConfig


def create_distinctive_test_data() -> List[models.ClusterModel]:
    """Create test data with very distinct, easily identifiable segments"""
    
    test_data = [
        models.ClusterModel(
            respondent_id=1000,
            response="Response about different topics",
            response_segment=[
                models.ClusterSubmodel(
                    segment_id="1000_1",
                    segment_response="Salt is unhealthy",
                    segment_label="SALT_CONCERN",
                    segment_description="SALT: This segment is specifically about salt being unhealthy and should cluster with other salt concerns"
                ),
                models.ClusterSubmodel(
                    segment_id="1000_2", 
                    segment_response="Sugar is bad",
                    segment_label="SUGAR_CONCERN",
                    segment_description="SUGAR: This segment is specifically about sugar being bad and should cluster with other sugar concerns"
                )
            ]
        ),
        models.ClusterModel(
            respondent_id=2000,
            response="Another response about topics", 
            response_segment=[
                models.ClusterSubmodel(
                    segment_id="2000_1",
                    segment_response="Too much salt content",
                    segment_label="SALT_CONCERN", 
                    segment_description="SALT: This segment is about excessive salt content and should cluster with salt segment 1000_1"
                ),
                models.ClusterSubmodel(
                    segment_id="2000_2",
                    segment_response="Price is expensive",
                    segment_label="PRICE_CONCERN",
                    segment_description="PRICE: This segment is about pricing being expensive and should form its own cluster"
                )
            ]
        ),
        models.ClusterModel(
            respondent_id=3000,
            response="Third response",
            response_segment=[
                models.ClusterSubmodel(
                    segment_id="3000_1",
                    segment_response="Sugar content too high",
                    segment_label="SUGAR_CONCERN",
                    segment_description="SUGAR: This segment is about high sugar content and should cluster with sugar segment 1000_2"
                )
            ]
        )
    ]
    
    return test_data


def trace_segment_through_pipeline(test_data: List[models.ClusterModel]) -> Dict:
    """Trace specific segments through the entire pipeline"""
    
    print("="*80)
    print("END-TO-END ALIGNMENT TEST")
    print("="*80)
    
    # Expected clusters:
    # - Salt cluster: 1000_1, 2000_1  
    # - Sugar cluster: 1000_2, 3000_1
    # - Price cluster: 2000_2
    
    print("\nINPUT DATA:")
    for resp in test_data:
        print(f"Respondent {resp.respondent_id}:")
        for seg in resp.response_segment:
            print(f"  {seg.segment_id}: {seg.segment_description}")
    
    # Step 1: Generate embeddings with Enhanced Embedder
    print("\n" + "="*60)
    print("STEP 1: ENHANCED EMBEDDER")
    print("="*60)
    
    embedding_config = EmbeddingConfig()
    embedding_config.use_question_aware = False  # Disable for cleaner test
    
    embedder = EnhancedEmbedder(config=embedding_config, verbose=True)
    embedded_data = embedder.get_combined_embeddings_with_tracking(test_data, "Test survey question")
    
    print("\nAFTER EMBEDDING:")
    for resp in embedded_data:
        print(f"Respondent {resp.respondent_id}:")
        for seg in resp.response_segment:
            has_code = seg.code_embedding is not None
            has_desc = seg.description_embedding is not None
            print(f"  {seg.segment_id}: {seg.segment_description[:50]}...")
            print(f"    Code embedding: {'✓' if has_code else '✗'}")
            print(f"    Desc embedding: {'✓' if has_desc else '✗'}")
    
    # Step 2: Run clustering
    print("\n" + "="*60)
    print("STEP 2: CLUSTERING")
    print("="*60)
    
    clustering_config = ClusteringConfig()
    clustering_config.hdbscan.min_cluster_size = 2  # Small for test data
    clustering_config.hdbscan.min_samples = 1
    
    clusterer = ClusterGenerator(
        input_list=embedded_data,
        var_lab="Test survey question",
        embedding_type="description",
        config=clustering_config,
        verbose=True
    )
    
    clusterer.run_pipeline()
    clustered_data = clusterer.to_cluster_model()
    
    print("\nAFTER CLUSTERING:")
    for resp in clustered_data:
        print(f"Respondent {resp.respondent_id}:")
        for seg in resp.response_segment:
            print(f"  {seg.segment_id} → Cluster {seg.initial_cluster}: {seg.segment_description[:50]}...")
    
    # Step 3: Analyze clustering results
    print("\n" + "="*60)
    print("STEP 3: CLUSTERING ANALYSIS")
    print("="*60)
    
    # Group by cluster
    clusters = {}
    for resp in clustered_data:
        for seg in resp.response_segment:
            cluster_id = seg.initial_cluster
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append({
                'segment_id': seg.segment_id,
                'description': seg.segment_description,
                'respondent_id': resp.respondent_id
            })
    
    print("\nCLUSTER COMPOSITION:")
    for cluster_id, segments in clusters.items():
        print(f"\nCluster {cluster_id} ({len(segments)} segments):")
        for seg in segments:
            print(f"  {seg['segment_id']} (Resp {seg['respondent_id']}): {seg['description'][:60]}...")
    
    # Step 4: Validate expected clustering
    print("\n" + "="*60)
    print("STEP 4: VALIDATION")
    print("="*60)
    
    # Find segments by description prefix
    salt_segments = []
    sugar_segments = []
    price_segments = []
    
    for resp in clustered_data:
        for seg in resp.response_segment:
            if seg.segment_description.startswith("SALT:"):
                salt_segments.append((seg.segment_id, seg.initial_cluster))
            elif seg.segment_description.startswith("SUGAR:"):
                sugar_segments.append((seg.segment_id, seg.initial_cluster))
            elif seg.segment_description.startswith("PRICE:"):
                price_segments.append((seg.segment_id, seg.initial_cluster))
    
    print(f"Salt segments: {salt_segments}")
    print(f"Sugar segments: {sugar_segments}")
    print(f"Price segments: {price_segments}")
    
    # Check if salt segments are in same cluster
    salt_clusters = [cluster for _, cluster in salt_segments if cluster is not None]
    sugar_clusters = [cluster for _, cluster in sugar_segments if cluster is not None]
    
    salt_aligned = len(set(salt_clusters)) <= 1 if salt_clusters else True
    sugar_aligned = len(set(sugar_clusters)) <= 1 if sugar_clusters else True
    
    print(f"\nVALIDATION RESULTS:")
    print(f"Salt segments in same cluster: {'✓' if salt_aligned else '❌'}")
    print(f"Sugar segments in same cluster: {'✓' if sugar_aligned else '❌'}")
    
    if salt_aligned and sugar_aligned:
        print("🎯 CLUSTERING APPEARS CORRECT")
    else:
        print("❌ CLUSTERING IS INCORRECT - ALIGNMENT ISSUE DETECTED")
    
    return {
        'salt_aligned': salt_aligned,
        'sugar_aligned': sugar_aligned,
        'clusters': clusters
    }


if __name__ == "__main__":
    test_data = create_distinctive_test_data()
    results = trace_segment_through_pipeline(test_data)