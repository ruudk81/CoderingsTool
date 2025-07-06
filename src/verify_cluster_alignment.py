"""
Comprehensive verification that cluster assignments match their actual segment descriptions.
This verifies there's no misalignment within the pipeline itself.
"""

def verify_cluster_content_alignment(initial_cluster_results, sample_cluster_id=None):
    """
    Verify that cluster assignments actually point to the correct segment descriptions.
    This checks for internal misalignment within the pipeline results.
    """
    
    print("="*80)
    print("CLUSTER CONTENT ALIGNMENT VERIFICATION")
    print("="*80)
    
    # Extract all segments with their cluster assignments
    all_segments = []
    cluster_contents = {}
    
    for resp in initial_cluster_results:
        if resp.response_segment:
            for seg in resp.response_segment:
                segment_info = {
                    'respondent_id': resp.respondent_id,
                    'segment_id': seg.segment_id,
                    'description': seg.segment_description,
                    'cluster': seg.initial_cluster,
                    'has_embedding': seg.description_embedding is not None
                }
                all_segments.append(segment_info)
                
                # Group by cluster
                if seg.initial_cluster is not None:
                    if seg.initial_cluster not in cluster_contents:
                        cluster_contents[seg.initial_cluster] = []
                    cluster_contents[seg.initial_cluster].append(segment_info)
    
    print(f"Total segments found: {len(all_segments)}")
    print(f"Segments with clusters: {len([s for s in all_segments if s['cluster'] is not None])}")
    print(f"Segments with embeddings: {len([s for s in all_segments if s['has_embedding']])}")
    print(f"Total clusters: {len(cluster_contents)}")
    
    # Check a specific cluster for semantic coherence
    if sample_cluster_id is None:
        # Use the cluster from user's example
        sample_cluster_id = 18
    
    if sample_cluster_id in cluster_contents:
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS OF CLUSTER {sample_cluster_id}")
        print(f"{'='*60}")
        
        cluster_segments = cluster_contents[sample_cluster_id]
        print(f"Cluster {sample_cluster_id} contains {len(cluster_segments)} segments:")
        
        # Show all segments in this cluster
        for i, seg in enumerate(cluster_segments):
            print(f"  {i+1:2d}. {seg['segment_id']}: {seg['description']}")
        
        # Analyze semantic coherence
        print(f"\n{'='*40}")
        print("SEMANTIC COHERENCE ANALYSIS")
        print(f"{'='*40}")
        
        # Look for common themes/patterns
        descriptions = [seg['description'].lower() for seg in cluster_segments]
        
        # Check for common keywords that might explain the clustering
        common_words = {}
        for desc in descriptions:
            words = desc.split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    common_words[word] = common_words.get(word, 0) + 1
        
        # Show most common words
        sorted_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
        print("Most frequent words in this cluster:")
        for word, count in sorted_words[:10]:
            if count > 1:  # Only show words that appear multiple times
                print(f"  '{word}': {count} times")
        
        # Identify potential semantic groupings
        themes = {
            'belangrijk': ['belangrijk', 'important'],
            'school': ['school', 'onderwijs', 'leerlingen'],
            'beleid': ['beleid', 'policy', 'regels'],
            'programma': ['programma', 'aanbod', 'activiteiten'],
            'monitoring': ['monitor', 'gemonitord', 'meten'],
            'cultuur': ['cultuur', 'maatschappij', 'omgaan']
        }
        
        found_themes = {}
        for theme, keywords in themes.items():
            count = 0
            for desc in descriptions:
                if any(keyword in desc for keyword in keywords):
                    count += 1
            if count > 0:
                found_themes[theme] = count
        
        print(f"\nThematic analysis:")
        for theme, count in found_themes.items():
            percentage = (count / len(cluster_segments)) * 100
            print(f"  {theme}: {count}/{len(cluster_segments)} segments ({percentage:.1f}%)")
        
        # Determine if cluster makes semantic sense
        if len(found_themes) <= 2 and max(found_themes.values()) >= len(cluster_segments) * 0.7:
            print(f"\n✅ CLUSTER APPEARS SEMANTICALLY COHERENT")
        else:
            print(f"\n❌ CLUSTER APPEARS SEMANTICALLY INCOHERENT")
            print("This suggests either:")
            print("  1. Clustering algorithm is not working properly")
            print("  2. There is misalignment between descriptions and cluster assignments")
    
    else:
        print(f"\n❌ Cluster {sample_cluster_id} not found in results")
    
    return cluster_contents

def cross_reference_cluster_assignments(initial_cluster_results):
    """
    Cross-reference that the same segment_id always has the same cluster assignment
    """
    
    print(f"\n{'='*60}")
    print("CROSS-REFERENCE VERIFICATION")
    print(f"{'='*60}")
    
    segment_cluster_map = {}
    duplicates = []
    
    for resp in initial_cluster_results:
        if resp.response_segment:
            for seg in resp.response_segment:
                seg_id = seg.segment_id
                cluster_id = seg.initial_cluster
                
                if seg_id in segment_cluster_map:
                    if segment_cluster_map[seg_id] != cluster_id:
                        duplicates.append((seg_id, segment_cluster_map[seg_id], cluster_id))
                else:
                    segment_cluster_map[seg_id] = cluster_id
    
    if duplicates:
        print(f"❌ FOUND {len(duplicates)} SEGMENT IDs WITH CONFLICTING CLUSTER ASSIGNMENTS:")
        for seg_id, cluster1, cluster2 in duplicates[:5]:  # Show first 5
            print(f"  {seg_id}: assigned to both cluster {cluster1} and {cluster2}")
    else:
        print("✅ NO CONFLICTING ASSIGNMENTS: Each segment_id has consistent cluster assignment")
    
    return len(duplicates) == 0

# Usage:
# cluster_contents = verify_cluster_content_alignment(initial_cluster_results, sample_cluster_id=18)
# consistency_check = cross_reference_cluster_assignments(initial_cluster_results)