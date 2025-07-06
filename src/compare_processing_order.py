"""
Compare the exact order of segment processing between pipeline and clean clusterer.
"""

def compare_processing_orders(encoded_text, embedded_text):
    """Compare segment processing order between segmentation and embedding steps"""
    
    print("="*80)
    print("PROCESSING ORDER COMPARISON")
    print("="*80)
    
    # Extract order from encoded_text (step 4 - segmentation)
    encoded_order = []
    for resp in encoded_text:
        if resp.response_segment:
            for seg in resp.response_segment:
                encoded_order.append((seg.segment_id, seg.segment_description[:50] + "..."))
    
    # Extract order from embedded_text (step 5 - after Enhanced Embedder)
    embedded_order = []
    for resp in embedded_text:
        if resp.response_segment:
            for seg in resp.response_segment:
                embedded_order.append((seg.segment_id, seg.segment_description[:50] + "..."))
    
    print(f"Encoded text order: {len(encoded_order)} segments")
    print(f"Embedded text order: {len(embedded_order)} segments")
    
    # Compare orders
    if len(encoded_order) != len(embedded_order):
        print("❌ DIFFERENT LENGTHS!")
        return False
    
    order_matches = True
    first_mismatch = None
    
    for i, (encoded_item, embedded_item) in enumerate(zip(encoded_order, embedded_order)):
        if encoded_item[0] != embedded_item[0]:  # Compare segment IDs
            if first_mismatch is None:
                first_mismatch = i
            order_matches = False
    
    if order_matches:
        print("✅ PERFECT ORDER MATCH: Encoded and embedded orders are identical!")
        print("First 10 segments in both orders:")
        for i in range(min(10, len(encoded_order))):
            print(f"  {i}: {encoded_order[i][0]} → {encoded_order[i][1]}")
    else:
        print(f"❌ ORDER MISMATCH: First difference at position {first_mismatch}")
        print("Encoded order around mismatch:")
        start = max(0, first_mismatch - 2)
        end = min(len(encoded_order), first_mismatch + 3)
        for i in range(start, end):
            marker = " →→→ " if i == first_mismatch else "     "
            print(f"  {i}:{marker}{encoded_order[i][0]} → {encoded_order[i][1]}")
        
        print("\nEmbedded order around mismatch:")
        for i in range(start, end):
            marker = " →→→ " if i == first_mismatch else "     "
            print(f"  {i}:{marker}{embedded_order[i][0]} → {embedded_order[i][1]}")
    
    return order_matches

def compare_with_clusterer_extraction(encoded_text):
    """Compare with how clean clusterer extracts segments"""
    
    print("\n" + "="*60)
    print("CLEAN CLUSTERER EXTRACTION ORDER")
    print("="*60)
    
    # How clean clusterer extracts (from clean_clusterer.py)
    clean_extraction_order = []
    for resp in encoded_text:  # Uses encoded_text as input
        if resp.response_segment:
            for seg in resp.response_segment:
                clean_extraction_order.append((seg.segment_id, seg.segment_description[:50] + "..."))
    
    print(f"Clean clusterer would extract: {len(clean_extraction_order)} segments")
    print("First 10 segments that clean clusterer sees:")
    for i in range(min(10, len(clean_extraction_order))):
        print(f"  {i}: {clean_extraction_order[i][0]} → {clean_extraction_order[i][1]}")
    
    return clean_extraction_order

# Usage:
# order_match = compare_processing_orders(encoded_text, embedded_text)
# clean_order = compare_with_clusterer_extraction(encoded_text)