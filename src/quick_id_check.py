"""
Quick manual verification of segment ID consistency.
Run this in Spyder after the pipeline completes to verify IDs match.
"""

def quick_id_verification(encoded_text, clean_cluster_map, initial_cluster_results):
    """Quick verification that segment IDs are consistent across all sources"""
    
    print("="*80)
    print("QUICK SEGMENT ID VERIFICATION")
    print("="*80)
    
    # Extract IDs from each source
    encoded_ids = set()
    for record in encoded_text:
        if record.response_segment:
            for seg in record.response_segment:
                encoded_ids.add(seg.segment_id)
    
    clean_ids = set(clean_cluster_map.keys())
    
    cluster_ids = set()
    for record in initial_cluster_results:
        if record.response_segment:
            for seg in record.response_segment:
                cluster_ids.add(seg.segment_id)
    
    # Report counts
    print(f"Encoded text segments: {len(encoded_ids)}")
    print(f"Clean cluster map keys: {len(clean_ids)}")
    print(f"Initial cluster results: {len(cluster_ids)}")
    
    # Check for perfect matches
    if encoded_ids == clean_ids == cluster_ids:
        print("✅ PERFECT MATCH: All segment IDs are identical across sources!")
        return True
    else:
        print("❌ MISMATCH DETECTED:")
        
        # Show what's missing where
        only_in_encoded = encoded_ids - clean_ids - cluster_ids
        only_in_clean = clean_ids - encoded_ids - cluster_ids  
        only_in_cluster = cluster_ids - encoded_ids - clean_ids
        
        if only_in_encoded:
            print(f"  Only in encoded_text: {list(only_in_encoded)[:5]}...")
        if only_in_clean:
            print(f"  Only in clean_cluster_map: {list(only_in_clean)[:5]}...")
        if only_in_cluster:
            print(f"  Only in initial_cluster_results: {list(only_in_cluster)[:5]}...")
            
        return False

def quick_description_check(encoded_text, segment_lookup, sample_size=5):
    """Quick check that descriptions match between sources"""
    
    print("\n" + "="*60)
    print("QUICK DESCRIPTION VERIFICATION")
    print("="*60)
    
    # Sample a few segments to verify descriptions match
    import random
    sampled_records = random.sample(encoded_text, min(sample_size, len(encoded_text)))
    
    all_match = True
    for record in sampled_records:
        if record.response_segment:
            for seg in record.response_segment:
                seg_id = seg.segment_id
                encoded_desc = seg.segment_description
                lookup_desc = segment_lookup.get(seg_id)
                
                if encoded_desc != lookup_desc:
                    print(f"❌ DESCRIPTION MISMATCH for {seg_id}:")
                    print(f"  Encoded: {encoded_desc}")
                    print(f"  Lookup:  {lookup_desc}")
                    all_match = False
    
    if all_match:
        print(f"✅ Sampled {sample_size} segments - all descriptions match!")
    
    return all_match

# Usage example:
# After pipeline completes, run:
# id_match = quick_id_verification(encoded_text, clean_cluster_map, initial_cluster_results)
# desc_match = quick_description_check(encoded_text, segment_lookup)