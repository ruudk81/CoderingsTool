#!/usr/bin/env python3
"""
Alignment verification code to add to your pipeline after step 6
"""

import random

def verify_alignment(encoded_text, embedded_text, initial_cluster_results):
    """
    Verify alignment between extracted ideas, embeddings, and clusters
    
    Args:
        encoded_text: List of IdeasExtractedModel from step 4
        embedded_text: List of EmbeddingsModel from step 5
        initial_cluster_results: List of ClusterModel from step 6
    """
    print("\n" + "=" * 80)
    print("ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    # 1. Basic statistics
    print("\n1. Basic Statistics:")
    print(f"   - Responses with extracted ideas: {len(encoded_text)}")
    print(f"   - Responses with embeddings: {len(embedded_text)}")
    print(f"   - Responses with clusters: {len(initial_cluster_results)}")
    
    total_ideas_extracted = sum(len(r.response_ideas) for r in encoded_text if r.response_ideas)
    total_ideas_embedded = sum(len(r.response_ideas) for r in embedded_text if r.response_ideas)
    total_ideas_clustered = sum(len(r.response_ideas) for r in initial_cluster_results if r.response_ideas)
    
    print(f"   - Total ideas extracted: {total_ideas_extracted}")
    print(f"   - Total ideas with embeddings: {total_ideas_embedded}")
    print(f"   - Total ideas with clusters: {total_ideas_clustered}")
    
    # 2. Check respondent ID consistency
    print("\n2. Respondent ID Consistency:")
    encoded_ids = set(r.respondent_id for r in encoded_text)
    embedded_ids = set(r.respondent_id for r in embedded_text)
    cluster_ids = set(r.respondent_id for r in initial_cluster_results)
    
    print(f"   - IDs in extracted only: {len(encoded_ids - embedded_ids)}")
    print(f"   - IDs in embedded only: {len(embedded_ids - encoded_ids)}")
    print(f"   - IDs in clusters only: {len(cluster_ids - encoded_ids)}")
    print(f"   - IDs in all three: {len(encoded_ids & embedded_ids & cluster_ids)}")
    
    # 3. Sample and trace specific ideas
    print("\n3. Tracing Sample Ideas:")
    
    # Get all clusters
    all_clusters = list(set([
        idea.initial_cluster 
        for result in initial_cluster_results 
        for idea in result.response_ideas   
        if idea.initial_cluster is not None
    ]))
    
    if all_clusters:
        sampled_cluster = random.choice(all_clusters)
        print(f"\n   Analyzing cluster {sampled_cluster}:")
        
        # Collect ideas from this cluster
        cluster_ideas = []
        for result in initial_cluster_results:
            for idea in result.response_ideas:
                if idea.initial_cluster == sampled_cluster:
                    cluster_ideas.append({
                        'respondent_id': result.respondent_id,
                        'idea_id': idea.idea_id,
                        'idea_text': idea.idea,
                        'cluster': idea.initial_cluster
                    })
        
        print(f"   - Ideas in cluster: {len(cluster_ideas)}")
        
        # Sample one idea for detailed trace
        if cluster_ideas:
            test_idea = random.choice(cluster_ideas)
            print(f"\n   Detailed trace for idea '{test_idea['idea_id']}':")
            print(f"   - Respondent: {test_idea['respondent_id']}")
            print(f"   - Text: {test_idea['idea_text'][:100]}...")
            
            # Check in encoded_text
            found_encoded = False
            for resp in encoded_text:
                if resp.respondent_id == test_idea['respondent_id']:
                    for idea in resp.response_ideas:
                        if idea.idea_id == test_idea['idea_id']:
                            found_encoded = True
                            text_match = idea.idea == test_idea['idea_text']
                            print(f"   - In extracted ideas: ✓ (text match: {'✓' if text_match else '✗'})")
                            break
                    break
            
            if not found_encoded:
                print(f"   - In extracted ideas: ✗")
            
            # Check in embedded_text
            found_embedded = False
            has_embedding = False
            for resp in embedded_text:
                if resp.respondent_id == test_idea['respondent_id']:
                    for idea in resp.response_ideas:
                        if idea.idea_id == test_idea['idea_id']:
                            found_embedded = True
                            text_match = idea.idea == test_idea['idea_text']
                            has_embedding = hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None
                            print(f"   - In embeddings: ✓ (text match: {'✓' if text_match else '✗'}, has embedding: {'✓' if has_embedding else '✗'})")
                            break
                    break
            
            if not found_embedded:
                print(f"   - In embeddings: ✗")
            
            print(f"   - In clusters: ✓ (cluster {test_idea['cluster']})")
    
    # 4. Check idea ID format
    print("\n4. Idea ID Format Check:")
    sample_size = min(5, len(initial_cluster_results))
    for i in range(sample_size):
        resp = initial_cluster_results[i]
        if resp.response_ideas:
            idea = resp.response_ideas[0]
            expected_prefix = f"{resp.respondent_id}_"
            matches_format = idea.idea_id.startswith(str(resp.respondent_id))
            print(f"   - Respondent {resp.respondent_id}: idea_id = {idea.idea_id} (format OK: {'✓' if matches_format else '✗'})")
    
    print("\n" + "=" * 80)
    return True

# Usage in pipeline:
# After step 6, add:
# verify_alignment(encoded_text, embedded_text, initial_cluster_results)