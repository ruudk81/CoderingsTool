"""
Diagnostic tool to validate ID alignment throughout the pipeline.

This tool checks that respondent_id and segment_id are properly tracked
from ResponseModel through ClusterModel.
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List, Dict, Tuple, Set, Any
import models
from collections import defaultdict


class IDAlignmentDiagnostic:
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def check_segment_id_generation(self, segmented_results: List[models.SegmentedModel]) -> None:
        """Check if segment IDs are properly generated and unique within responses."""
        print("\n=== SEGMENT ID GENERATION CHECK ===")
        
        # Track segment IDs per response
        duplicate_segments = []
        missing_segments = []
        
        for result in segmented_results:
            if not result.response_segment:
                continue
                
            segment_ids = []
            for segment in result.response_segment:
                if not segment.segment_id:
                    missing_segments.append((result.respondent_id, "Missing segment_id"))
                else:
                    segment_ids.append(segment.segment_id)
            
            # Check for duplicates within response
            if len(segment_ids) != len(set(segment_ids)):
                duplicate_segments.append((result.respondent_id, segment_ids))
        
        if missing_segments:
            self.issues.append(f"Found {len(missing_segments)} segments with missing IDs")
            print(f"❌ Missing segment IDs in {len(missing_segments)} cases")
        else:
            print("✅ All segments have IDs")
            
        if duplicate_segments:
            self.issues.append(f"Found {len(duplicate_segments)} responses with duplicate segment IDs")
            print(f"❌ Duplicate segment IDs within responses: {len(duplicate_segments)} cases")
            for resp_id, seg_ids in duplicate_segments[:3]:  # Show first 3
                print(f"   Response {resp_id}: segments {seg_ids}")
        else:
            print("✅ No duplicate segment IDs within responses")
    
    def check_respondent_id_consistency(self, 
                                      raw_data: List[models.ResponseModel],
                                      preprocessed: List[models.PreprocessedModel],
                                      segmented: List[models.SegmentedModel],
                                      clustered: List[models.ClusterModel]) -> None:
        """Check if respondent IDs are consistent across pipeline stages."""
        print("\n=== RESPONDENT ID CONSISTENCY CHECK ===")
        
        # Get respondent IDs at each stage
        raw_ids = {r.respondent_id for r in raw_data}
        preproc_ids = {r.respondent_id for r in preprocessed}
        segment_ids = {r.respondent_id for r in segmented}
        cluster_ids = {r.respondent_id for r in clustered}
        
        # Check for type consistency
        id_types = defaultdict(set)
        for stage_name, id_set in [("raw", raw_ids), ("preprocessed", preproc_ids), 
                                   ("segmented", segment_ids), ("clustered", cluster_ids)]:
            for id_val in id_set:
                id_types[stage_name].add(type(id_val).__name__)
        
        # Report type inconsistencies
        print("\nID type analysis:")
        for stage, types in id_types.items():
            print(f"  {stage}: {', '.join(types)}")
            if len(types) > 1:
                self.warnings.append(f"Mixed ID types in {stage}: {types}")
        
        # Check for missing IDs between stages
        print("\nID preservation check:")
        
        # Raw to preprocessed
        missing_in_preproc = raw_ids - preproc_ids
        if missing_in_preproc:
            self.issues.append(f"{len(missing_in_preproc)} IDs lost between raw and preprocessed")
            print(f"❌ Lost {len(missing_in_preproc)} IDs: raw → preprocessed")
        else:
            print("✅ All IDs preserved: raw → preprocessed")
        
        # Preprocessed to segmented (note: quality filtered items may be removed)
        missing_in_segment = preproc_ids - segment_ids
        print(f"ℹ️  {len(missing_in_segment)} IDs filtered out (quality filter)")
        
        # Segmented to clustered
        missing_in_cluster = segment_ids - cluster_ids
        if missing_in_cluster:
            self.issues.append(f"{len(missing_in_cluster)} IDs lost between segmented and clustered")
            print(f"❌ Lost {len(missing_in_cluster)} IDs: segmented → clustered")
        else:
            print("✅ All IDs preserved: segmented → clustered")
    
    def check_segment_alignment(self,
                               segmented: List[models.SegmentedModel],
                               embedded: List[models.ClusterModel],
                               clustered: List[models.ClusterModel]) -> None:
        """Check if segments are properly aligned through embedding and clustering."""
        print("\n=== SEGMENT ALIGNMENT CHECK ===")
        
        # Create segment inventory from segmented data
        segment_inventory = {}
        for result in segmented:
            if result.response_segment:
                for segment in result.response_segment:
                    key = (result.respondent_id, segment.segment_id)
                    segment_inventory[key] = {
                        'description': segment.segment_description,
                        'label': segment.segment_label
                    }
        
        # Check embedded data
        embedded_segments = {}
        for result in embedded:
            if result.response_segment:
                for segment in result.response_segment:
                    key = (result.respondent_id, segment.segment_id)
                    embedded_segments[key] = {
                        'description': segment.segment_description,
                        'label': segment.segment_label,
                        'has_code_embedding': segment.code_embedding is not None,
                        'has_desc_embedding': segment.description_embedding is not None
                    }
        
        # Check clustered data
        clustered_segments = {}
        for result in clustered:
            if result.response_segment:
                for segment in result.response_segment:
                    key = (result.respondent_id, segment.segment_id)
                    clustered_segments[key] = {
                        'description': segment.segment_description,
                        'label': segment.segment_label,
                        'cluster': segment.initial_cluster
                    }
        
        # Compare inventories
        print(f"\nSegment counts:")
        print(f"  Segmented: {len(segment_inventory)}")
        print(f"  Embedded: {len(embedded_segments)}")
        print(f"  Clustered: {len(clustered_segments)}")
        
        # Check for missing segments
        missing_in_embedded = set(segment_inventory.keys()) - set(embedded_segments.keys())
        missing_in_clustered = set(embedded_segments.keys()) - set(clustered_segments.keys())
        
        if missing_in_embedded:
            self.issues.append(f"{len(missing_in_embedded)} segments lost during embedding")
            print(f"❌ Lost {len(missing_in_embedded)} segments during embedding")
            for key in list(missing_in_embedded)[:3]:
                print(f"   Lost: respondent {key[0]}, segment {key[1]}")
        else:
            print("✅ All segments preserved during embedding")
            
        if missing_in_clustered:
            self.issues.append(f"{len(missing_in_clustered)} segments lost during clustering")
            print(f"❌ Lost {len(missing_in_clustered)} segments during clustering")
        else:
            print("✅ All segments preserved during clustering")
        
        # Check embedding completeness
        segments_without_embeddings = sum(1 for seg in embedded_segments.values() 
                                        if not seg['has_code_embedding'] or not seg['has_desc_embedding'])
        if segments_without_embeddings:
            self.warnings.append(f"{segments_without_embeddings} segments missing embeddings")
            print(f"⚠️  {segments_without_embeddings} segments missing embeddings")
        
        # Check cluster assignment
        unclustered = sum(1 for seg in clustered_segments.values() if seg['cluster'] is None)
        noise_clustered = sum(1 for seg in clustered_segments.values() if seg['cluster'] == -1)
        
        print(f"\nCluster assignment:")
        print(f"  Assigned to clusters: {len(clustered_segments) - unclustered - noise_clustered}")
        print(f"  Noise cluster (-1): {noise_clustered}")
        print(f"  No cluster (None): {unclustered}")
    
    def check_cluster_to_segment_mapping(self, cluster_results: List[models.ClusterModel]) -> None:
        """Verify that cluster IDs map correctly to segments."""
        print("\n=== CLUSTER TO SEGMENT MAPPING CHECK ===")
        
        # Build reverse mapping: cluster_id -> list of (respondent_id, segment_id, description)
        cluster_mapping = defaultdict(list)
        
        for result in cluster_results:
            if result.response_segment:
                for segment in result.response_segment:
                    if segment.initial_cluster is not None:
                        cluster_mapping[segment.initial_cluster].append({
                            'respondent_id': result.respondent_id,
                            'segment_id': segment.segment_id,
                            'description': segment.segment_description[:50] + "..." if len(segment.segment_description) > 50 else segment.segment_description
                        })
        
        # Report cluster statistics
        cluster_sizes = {cid: len(items) for cid, items in cluster_mapping.items()}
        print(f"\nFound {len(cluster_mapping)} clusters")
        print(f"Cluster size distribution:")
        print(f"  Min size: {min(cluster_sizes.values()) if cluster_sizes else 0}")
        print(f"  Max size: {max(cluster_sizes.values()) if cluster_sizes else 0}")
        print(f"  Average size: {sum(cluster_sizes.values()) / len(cluster_sizes) if cluster_sizes else 0:.1f}")
        
        # Show sample mappings
        print("\nSample cluster contents (first 3 clusters):")
        for cluster_id in sorted(cluster_mapping.keys())[:3]:
            items = cluster_mapping[cluster_id]
            print(f"\nCluster {cluster_id} ({len(items)} segments):")
            for item in items[:3]:  # Show first 3 items
                print(f"  - Respondent {item['respondent_id']}, Segment {item['segment_id']}: {item['description']}")
    
    def generate_summary(self) -> None:
        """Generate a summary of all findings."""
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        
        if not self.issues and not self.warnings:
            print("✅ No issues found! ID alignment appears correct.")
        else:
            if self.issues:
                print(f"\n❌ ISSUES ({len(self.issues)}):")
                for issue in self.issues:
                    print(f"  - {issue}")
                    
            if self.warnings:
                print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"  - {warning}")
        
        print("\nRECOMMENDATIONS:")
        if any("Mixed ID types" in w for w in self.warnings):
            print("  - Consider standardizing respondent_id types (e.g., always use strings)")
        if any("segments lost" in i for i in self.issues):
            print("  - Check segment filtering logic in embedding/clustering steps")
        if any("duplicate segment IDs" in i for i in self.issues):
            print("  - Ensure segment IDs are properly enumerated in segmentDescriber")


def run_diagnostic(raw_data: List[models.ResponseModel],
                  preprocessed: List[models.PreprocessedModel], 
                  segmented: List[models.SegmentedModel],
                  embedded: List[models.ClusterModel],
                  clustered: List[models.ClusterModel]) -> None:
    """Run complete diagnostic on pipeline data."""
    
    diagnostic = IDAlignmentDiagnostic()
    
    print("="*60)
    print("ID ALIGNMENT DIAGNOSTIC TOOL")
    print("="*60)
    
    # Run all checks
    diagnostic.check_respondent_id_consistency(raw_data, preprocessed, segmented, clustered)
    diagnostic.check_segment_id_generation(segmented)
    diagnostic.check_segment_alignment(segmented, embedded, clustered)
    diagnostic.check_cluster_to_segment_mapping(clustered)
    
    # Generate summary
    diagnostic.generate_summary()