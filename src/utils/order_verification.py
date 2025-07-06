"""
Direct order verification to identify exactly where segment ordering gets scrambled.
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List
import models


def trace_segment_order(step_name: str, data: List[models.ClusterModel], target_segments: List[str] = None) -> List[str]:
    """Extract segment order at a specific step"""
    
    if target_segments is None:
        target_segments = []
    
    segment_order = []
    
    for resp in data:
        if resp.response_segment:
            for seg in resp.response_segment:
                segment_order.append(seg.segment_id)
    
    print(f"\n{step_name}: Found {len(segment_order)} segments")
    
    if target_segments:
        print(f"Target segment positions:")
        for target in target_segments:
            try:
                pos = segment_order.index(target)
                print(f"  {target}: position {pos}")
            except ValueError:
                print(f"  {target}: NOT FOUND")
    
    return segment_order


def compare_segment_descriptions(cluster_results: List[models.ClusterModel], 
                               cluster_id: int = 23) -> None:
    """Compare what the pipeline thinks is in a cluster vs clean clusterer extraction"""
    
    print(f"\n{'='*60}")
    print(f"SEGMENT ORDER VERIFICATION FOR CLUSTER {cluster_id}")
    print(f"{'='*60}")
    
    # Method 1: How clean clusterer extracts segments (clean order)
    clean_segments = []
    for resp in cluster_results:
        if resp.response_segment:
            for seg in resp.response_segment:
                clean_segments.append((seg.segment_id, seg.segment_description))
    
    print(f"Clean extraction found {len(clean_segments)} total segments")
    print("First 10 segments in clean extraction order:")
    for i, (seg_id, desc) in enumerate(clean_segments[:10]):
        print(f"  {i}: {seg_id} → {desc[:60]}...")
    
    # Method 2: How pipeline assigns clusters (what we see in debug)
    pipeline_cluster_segments = []
    for resp in cluster_results:
        if resp.response_segment:
            for seg in resp.response_segment:
                if seg.initial_cluster == cluster_id:
                    pipeline_cluster_segments.append((seg.segment_id, seg.segment_description))
    
    print(f"\nPipeline cluster {cluster_id} has {len(pipeline_cluster_segments)} segments:")
    for i, (seg_id, desc) in enumerate(pipeline_cluster_segments[:10]):
        print(f"  {i}: {seg_id} → {desc[:60]}...")
    
    # Method 3: Check if any segments have the same descriptions as clean clusterer would expect
    if len(clean_segments) >= cluster_id + 1:
        expected_seg_id, expected_desc = clean_segments[cluster_id]
        print(f"\nIf order is correct, cluster {cluster_id} should contain segment:")
        print(f"  {expected_seg_id} → {expected_desc[:60]}...")
        
        # Check if this expected segment is actually in the cluster
        found_expected = False
        for seg_id, desc in pipeline_cluster_segments:
            if seg_id == expected_seg_id:
                found_expected = True
                print(f"  ✅ Expected segment {expected_seg_id} IS in cluster {cluster_id}")
                break
        
        if not found_expected:
            print(f"  ❌ Expected segment {expected_seg_id} is NOT in cluster {cluster_id}")
            
            # Find where this segment actually ended up
            for resp in cluster_results:
                if resp.response_segment:
                    for seg in resp.response_segment:
                        if seg.segment_id == expected_seg_id:
                            actual_cluster = seg.initial_cluster
                            print(f"  → {expected_seg_id} is actually in cluster {actual_cluster}")
                            break


def full_order_diagnostic(cluster_results: List[models.ClusterModel]) -> None:
    """Complete diagnostic of segment ordering"""
    
    print(f"\n{'='*80}")
    print("COMPLETE SEGMENT ORDER DIAGNOSTIC")
    print(f"{'='*80}")
    
    # Extract all segments in the order they appear in the data structure
    all_segments = []
    for resp in cluster_results:
        if resp.response_segment:
            for seg in resp.response_segment:
                all_segments.append({
                    'position': len(all_segments),
                    'respondent_id': resp.respondent_id,
                    'segment_id': seg.segment_id,
                    'description': seg.segment_description,
                    'cluster': seg.initial_cluster
                })
    
    print(f"Total segments: {len(all_segments)}")
    
    # Show first 20 segments with their cluster assignments
    print(f"\nFirst 20 segments in data structure order:")
    for seg in all_segments[:20]:
        desc_short = seg['description'][:50] + "..." if len(seg['description']) > 50 else seg['description']
        print(f"  {seg['position']:3d}: {seg['segment_id']} → Cluster {seg['cluster']} → {desc_short}")
    
    # Check if cluster IDs are sequential based on position
    print(f"\nChecking if cluster assignments follow a pattern:")
    cluster_positions = {}
    for seg in all_segments:
        cluster_id = seg['cluster']
        if cluster_id is not None and cluster_id != -1:
            if cluster_id not in cluster_positions:
                cluster_positions[cluster_id] = []
            cluster_positions[cluster_id].append(seg['position'])
    
    # Show position ranges for first 10 clusters
    print("Position ranges for first 10 clusters:")
    for cluster_id in sorted(cluster_positions.keys())[:10]:
        positions = cluster_positions[cluster_id]
        min_pos = min(positions)
        max_pos = max(positions)
        print(f"  Cluster {cluster_id}: positions {min_pos}-{max_pos} ({len(positions)} segments)")


def run_order_verification(cluster_results: List[models.ClusterModel]) -> None:
    """Run complete order verification diagnostic"""
    
    compare_segment_descriptions(cluster_results, cluster_id=23)
    compare_segment_descriptions(cluster_results, cluster_id=24)
    full_order_diagnostic(cluster_results)


if __name__ == "__main__":
    print("Run this from pipeline.py after clustering is complete")
    print("Add: from utils.order_verification import run_order_verification")
    print("Add: run_order_verification(initial_cluster_results)")