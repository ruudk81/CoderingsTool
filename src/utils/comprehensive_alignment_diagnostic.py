"""
Comprehensive alignment diagnostic tool to track individual segments through the pipeline.

This tool creates a detailed paper trail for each segment from creation to clustering,
helping identify exactly where alignment issues occur.
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List, Dict, Tuple, Set, Any
import models
import pandas as pd
import numpy as np
from collections import defaultdict
import hashlib


class SegmentTracker:
    def __init__(self):
        self.segment_registry = {}  # segment_id -> detailed info
        self.issues = []
        self.warnings = []
        
    def register_segment(self, stage: str, respondent_id: Any, segment_id: str, 
                        segment_description: str, additional_info: Dict = None):
        """Register a segment at a specific pipeline stage"""
        key = (respondent_id, segment_id)
        
        if key not in self.segment_registry:
            self.segment_registry[key] = {
                'respondent_id': respondent_id,
                'segment_id': segment_id,
                'stages': {},
                'description_hash': None,
                'first_seen': stage
            }
        
        # Create a hash of the description for consistency checking
        desc_hash = hashlib.md5(segment_description.encode('utf-8')).hexdigest()[:8]
        
        stage_info = {
            'description': segment_description[:100] + "..." if len(segment_description) > 100 else segment_description,
            'description_hash': desc_hash,
            'description_length': len(segment_description)
        }
        
        if additional_info:
            stage_info.update(additional_info)
            
        self.segment_registry[key]['stages'][stage] = stage_info
        
        # Check for description consistency
        if self.segment_registry[key]['description_hash'] is None:
            self.segment_registry[key]['description_hash'] = desc_hash
        elif self.segment_registry[key]['description_hash'] != desc_hash:
            self.issues.append(f"Description mismatch for {key}: hash changed from {self.segment_registry[key]['description_hash']} to {desc_hash}")
    
    def check_stage_alignment(self, stage1: str, stage2: str):
        """Check if segments are properly aligned between two stages"""
        stage1_segments = set()
        stage2_segments = set()
        
        for key, info in self.segment_registry.items():
            if stage1 in info['stages']:
                stage1_segments.add(key)
            if stage2 in info['stages']:
                stage2_segments.add(key)
        
        missing_in_stage2 = stage1_segments - stage2_segments
        extra_in_stage2 = stage2_segments - stage1_segments
        
        print(f"\n=== ALIGNMENT CHECK: {stage1} → {stage2} ===")
        print(f"Segments in {stage1}: {len(stage1_segments)}")
        print(f"Segments in {stage2}: {len(stage2_segments)}")
        
        if missing_in_stage2:
            self.issues.append(f"{len(missing_in_stage2)} segments lost between {stage1} and {stage2}")
            print(f"❌ Lost {len(missing_in_stage2)} segments")
            for key in list(missing_in_stage2)[:3]:
                print(f"   Lost: {key}")
        else:
            print(f"✅ No segments lost")
            
        if extra_in_stage2:
            self.warnings.append(f"{len(extra_in_stage2)} new segments appeared in {stage2}")
            print(f"⚠️  {len(extra_in_stage2)} new segments appeared")
    
    def verify_embedding_alignment(self):
        """Verify that embeddings are correctly aligned with their descriptions"""
        print(f"\n=== EMBEDDING ALIGNMENT VERIFICATION ===")
        
        embedding_mismatches = 0
        for key, info in self.segment_registry.items():
            if 'embedded' in info['stages'] and 'clustered' in info['stages']:
                embedded_desc = info['stages']['embedded']['description']
                clustered_desc = info['stages']['clustered']['description']
                
                if embedded_desc != clustered_desc:
                    embedding_mismatches += 1
                    if embedding_mismatches <= 3:  # Show first 3 examples
                        print(f"❌ Description mismatch for {key}:")
                        print(f"   Embedded: {embedded_desc}")
                        print(f"   Clustered: {clustered_desc}")
        
        if embedding_mismatches > 0:
            self.issues.append(f"{embedding_mismatches} segments have mismatched descriptions between embedding and clustering")
        else:
            print("✅ All descriptions match between embedding and clustering stages")
    
    def verify_cluster_assignments(self):
        """Check for suspicious cluster assignments"""
        print(f"\n=== CLUSTER ASSIGNMENT VERIFICATION ===")
        
        cluster_contents = defaultdict(list)
        for key, info in self.segment_registry.items():
            if 'clustered' in info['stages']:
                cluster_id = info['stages']['clustered'].get('cluster_id')
                if cluster_id is not None:
                    cluster_contents[cluster_id].append({
                        'key': key,
                        'description': info['stages']['clustered']['description'],
                        'hash': info['stages']['clustered']['description_hash']
                    })
        
        # Check for clusters with identical descriptions but different IDs
        identical_descriptions = defaultdict(list)
        for cluster_id, segments in cluster_contents.items():
            for segment in segments:
                identical_descriptions[segment['hash']].append((cluster_id, segment['key'], segment['description']))
        
        cross_cluster_duplicates = 0
        for desc_hash, occurrences in identical_descriptions.items():
            if len(occurrences) > 1:
                cluster_ids = set(occ[0] for occ in occurrences)
                if len(cluster_ids) > 1:
                    cross_cluster_duplicates += 1
                    if cross_cluster_duplicates <= 3:  # Show first 3 examples
                        print(f"⚠️  Identical description in multiple clusters:")
                        print(f"   Description: {occurrences[0][2]}")
                        print(f"   Clusters: {sorted(cluster_ids)}")
                        print(f"   Segments: {[occ[1] for occ in occurrences]}")
        
        if cross_cluster_duplicates > 0:
            self.warnings.append(f"{cross_cluster_duplicates} identical descriptions found in multiple clusters")
        
        print(f"Found {len(cluster_contents)} clusters")
        print(f"Cross-cluster identical descriptions: {cross_cluster_duplicates}")
    
    def trace_specific_segments(self, sample_size: int = 5):
        """Trace a sample of segments through all stages"""
        print(f"\n=== DETAILED SEGMENT TRACING (Sample: {sample_size}) ===")
        
        # Get segments that appear in all stages
        complete_segments = []
        for key, info in self.segment_registry.items():
            if all(stage in info['stages'] for stage in ['segmented', 'embedded', 'clustered']):
                complete_segments.append((key, info))
        
        if not complete_segments:
            print("❌ No segments found that appear in all stages!")
            return
        
        # Sample segments for detailed tracing
        import random
        sampled_segments = random.sample(complete_segments, min(sample_size, len(complete_segments)))
        
        for i, (key, info) in enumerate(sampled_segments):
            print(f"\n--- SEGMENT {i+1}: {key} ---")
            
            for stage in ['segmented', 'embedded', 'clustered']:
                if stage in info['stages']:
                    stage_info = info['stages'][stage]
                    print(f"{stage.upper()}:")
                    print(f"  Description: {stage_info['description']}")
                    print(f"  Hash: {stage_info['description_hash']}")
                    
                    if 'has_code_embedding' in stage_info:
                        print(f"  Code embedding: {stage_info['has_code_embedding']}")
                    if 'has_desc_embedding' in stage_info:
                        print(f"  Desc embedding: {stage_info['has_desc_embedding']}")
                    if 'cluster_id' in stage_info:
                        print(f"  Cluster: {stage_info['cluster_id']}")
    
    def generate_summary(self):
        """Generate comprehensive summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ALIGNMENT DIAGNOSTIC SUMMARY")
        print("="*80)
        
        total_segments = len(self.segment_registry)
        stages_coverage = defaultdict(int)
        
        for info in self.segment_registry.values():
            for stage in info['stages'].keys():
                stages_coverage[stage] += 1
        
        print(f"\nSEGMENT COVERAGE BY STAGE:")
        for stage, count in sorted(stages_coverage.items()):
            print(f"  {stage}: {count} segments")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.issues and not self.warnings:
            print(f"\n✅ NO ALIGNMENT ISSUES DETECTED!")
            print(f"All {total_segments} segments properly tracked through pipeline.")


def run_comprehensive_diagnostic(segmented: List[models.SegmentedModel],
                                embedded: List[models.ClusterModel],
                                clustered: List[models.ClusterModel]) -> None:
    """Run comprehensive alignment diagnostic"""
    
    tracker = SegmentTracker()
    
    print("="*80)
    print("COMPREHENSIVE ALIGNMENT DIAGNOSTIC")
    print("="*80)
    
    # Register segments from each stage
    print("\nRegistering segments from segmented stage...")
    for result in segmented:
        if result.response_segment:
            for segment in result.response_segment:
                tracker.register_segment(
                    stage='segmented',
                    respondent_id=result.respondent_id,
                    segment_id=segment.segment_id,
                    segment_description=segment.segment_description,
                    additional_info={
                        'label': segment.segment_label
                    }
                )
    
    print("Registering segments from embedded stage...")
    for result in embedded:
        if result.response_segment:
            for segment in result.response_segment:
                tracker.register_segment(
                    stage='embedded',
                    respondent_id=result.respondent_id,
                    segment_id=segment.segment_id,
                    segment_description=segment.segment_description,
                    additional_info={
                        'label': segment.segment_label,
                        'has_code_embedding': segment.code_embedding is not None,
                        'has_desc_embedding': segment.description_embedding is not None
                    }
                )
    
    print("Registering segments from clustered stage...")
    for result in clustered:
        if result.response_segment:
            for segment in result.response_segment:
                tracker.register_segment(
                    stage='clustered',
                    respondent_id=result.respondent_id,
                    segment_id=segment.segment_id,
                    segment_description=segment.segment_description,
                    additional_info={
                        'label': segment.segment_label,
                        'cluster_id': segment.initial_cluster,
                        'has_code_embedding': segment.code_embedding is not None,
                        'has_desc_embedding': segment.description_embedding is not None
                    }
                )
    
    # Run all alignment checks
    tracker.check_stage_alignment('segmented', 'embedded')
    tracker.check_stage_alignment('embedded', 'clustered')
    tracker.verify_embedding_alignment()
    tracker.verify_cluster_assignments()
    tracker.trace_specific_segments(sample_size=3)
    tracker.generate_summary()
    
    return tracker