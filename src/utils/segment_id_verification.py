"""
Segment ID Verification Tool

This diagnostic tool verifies that segment IDs are exactly the same between:
1. The encoded_text from step 4 (segmentation)
2. The clean_cluster_map from the clean clusterer  
3. The initial_cluster_results from step 5

The goal is to ensure we're not comparing different segments when we think we're comparing the same ones.

INTEGRATION WITH PIPELINE:
- Automatically runs when DEBUG_CLUSTER_TRACKING=True in pipeline.py
- Integrates between clean clusterer test and order verification
- Provides comprehensive verification of data consistency

WHAT IT CHECKS:
- Segment IDs are identical across all three sources
- Segment descriptions match for the same IDs
- Respondent IDs match for the same segment IDs
- Identifies missing segments in any source
- Provides detailed examples of mismatches

USAGE:
    from utils.segment_id_verification import run_segment_id_verification
    success = run_segment_id_verification(encoded_text, clean_cluster_map, initial_cluster_results)
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List, Dict, Set, Tuple, Optional
import models
from collections import defaultdict


class SegmentIDVerifier:
    """Comprehensive segment ID verification across pipeline stages"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.issues_found = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with level"""
        if self.verbose:
            prefix = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "SUCCESS": "✅"}
            print(f"{prefix.get(level, '•')} {message}")
    
    def extract_segment_ids_from_encoded_text(self, encoded_text: List[models.SegmentedModel]) -> Dict[str, Tuple[str, str]]:
        """
        Extract segment IDs from encoded_text (step 4)
        Returns: {segment_id: (respondent_id, segment_description)}
        """
        segment_map = {}
        
        for resp in encoded_text:
            if resp.response_segment:
                for segment in resp.response_segment:
                    segment_map[segment.segment_id] = (
                        str(resp.respondent_id), 
                        segment.segment_description
                    )
        
        self.log(f"Extracted {len(segment_map)} segment IDs from encoded_text")
        return segment_map
    
    def extract_segment_ids_from_clean_cluster_map(self, clean_cluster_map: Dict[str, int]) -> Set[str]:
        """
        Extract segment IDs from clean_cluster_map
        Returns: set of segment IDs
        """
        segment_ids = set(clean_cluster_map.keys())
        self.log(f"Extracted {len(segment_ids)} segment IDs from clean_cluster_map")
        return segment_ids
    
    def extract_segment_ids_from_initial_cluster_results(self, initial_cluster_results: List[models.ClusterModel]) -> Dict[str, Tuple[str, str, Optional[int]]]:
        """
        Extract segment IDs from initial_cluster_results (step 5)
        Returns: {segment_id: (respondent_id, segment_description, initial_cluster)}
        """
        segment_map = {}
        
        for resp in initial_cluster_results:
            if resp.response_segment:
                for segment in resp.response_segment:
                    segment_map[segment.segment_id] = (
                        str(resp.respondent_id),
                        segment.segment_description,
                        segment.initial_cluster
                    )
        
        self.log(f"Extracted {len(segment_map)} segment IDs from initial_cluster_results")
        return segment_map
    
    def find_set_differences(self, set1: Set[str], set2: Set[str], name1: str, name2: str) -> Tuple[Set[str], Set[str]]:
        """Find differences between two sets of segment IDs"""
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        
        if only_in_1:
            self.log(f"{len(only_in_1)} segment IDs only in {name1}", "WARNING")
            self.issues_found.append(f"Segment IDs only in {name1}: {len(only_in_1)}")
        
        if only_in_2:
            self.log(f"{len(only_in_2)} segment IDs only in {name2}", "WARNING")
            self.issues_found.append(f"Segment IDs only in {name2}: {len(only_in_2)}")
            
        return only_in_1, only_in_2
    
    def verify_segment_descriptions(self, encoded_segments: Dict[str, Tuple[str, str]], 
                                   cluster_segments: Dict[str, Tuple[str, str, Optional[int]]]) -> List[str]:
        """
        Verify that segment descriptions match for the same IDs
        Returns list of mismatched segment IDs
        """
        mismatches = []
        
        common_ids = set(encoded_segments.keys()) & set(cluster_segments.keys())
        
        for seg_id in common_ids:
            encoded_desc = encoded_segments[seg_id][1]
            cluster_desc = cluster_segments[seg_id][1]
            
            if encoded_desc != cluster_desc:
                mismatches.append(seg_id)
                self.log(f"Description mismatch for {seg_id}:", "ERROR")
                self.log(f"  Encoded: '{encoded_desc}'", "ERROR")
                self.log(f"  Cluster: '{cluster_desc}'", "ERROR")
        
        if mismatches:
            self.issues_found.append(f"Description mismatches: {len(mismatches)}")
        
        return mismatches
    
    def verify_respondent_ids(self, encoded_segments: Dict[str, Tuple[str, str]], 
                             cluster_segments: Dict[str, Tuple[str, str, Optional[int]]]) -> List[str]:
        """
        Verify that respondent IDs match for the same segment IDs
        Returns list of mismatched segment IDs
        """
        mismatches = []
        
        common_ids = set(encoded_segments.keys()) & set(cluster_segments.keys())
        
        for seg_id in common_ids:
            encoded_resp_id = encoded_segments[seg_id][0]
            cluster_resp_id = cluster_segments[seg_id][0]
            
            if encoded_resp_id != cluster_resp_id:
                mismatches.append(seg_id)
                self.log(f"Respondent ID mismatch for {seg_id}:", "ERROR")
                self.log(f"  Encoded: '{encoded_resp_id}'", "ERROR")
                self.log(f"  Cluster: '{cluster_resp_id}'", "ERROR")
        
        if mismatches:
            self.issues_found.append(f"Respondent ID mismatches: {len(mismatches)}")
        
        return mismatches
    
    def show_examples(self, segment_ids: Set[str], segments_dict: Dict[str, Tuple], 
                     source_name: str, max_examples: int = 5):
        """Show examples of segments from a specific source"""
        if not segment_ids:
            return
            
        self.log(f"\nExample segments only in {source_name}:", "INFO")
        
        for i, seg_id in enumerate(sorted(segment_ids)):
            if i >= max_examples:
                self.log(f"  ... and {len(segment_ids) - max_examples} more", "INFO")
                break
                
            if seg_id in segments_dict:
                info = segments_dict[seg_id]
                if len(info) == 2:  # encoded_text format
                    resp_id, desc = info
                    self.log(f"  {seg_id} (resp: {resp_id}): {desc[:100]}...", "INFO")
                elif len(info) == 3:  # cluster_results format
                    resp_id, desc, cluster = info
                    self.log(f"  {seg_id} (resp: {resp_id}, cluster: {cluster}): {desc[:100]}...", "INFO")
            else:
                self.log(f"  {seg_id}: [No details available]", "INFO")
    
    def analyze_segment_id_patterns(self, segment_ids: Set[str]) -> Dict[str, int]:
        """Analyze patterns in segment IDs"""
        patterns = defaultdict(int)
        
        for seg_id in segment_ids:
            # Extract respondent part (everything before the last underscore)
            if '_' in seg_id:
                resp_part = '_'.join(seg_id.split('_')[:-1])
                patterns[resp_part] += 1
        
        return dict(patterns)
    
    def run_verification(self, encoded_text: List[models.SegmentedModel],
                        clean_cluster_map: Dict[str, int],
                        initial_cluster_results: List[models.ClusterModel]) -> bool:
        """
        Run comprehensive segment ID verification
        Returns True if all verifications pass, False otherwise
        """
        self.log("="*80)
        self.log("SEGMENT ID VERIFICATION DIAGNOSTIC")
        self.log("="*80)
        
        # Extract segment IDs from all sources
        encoded_segments = self.extract_segment_ids_from_encoded_text(encoded_text)
        clean_segment_ids = self.extract_segment_ids_from_clean_cluster_map(clean_cluster_map)
        cluster_segments = self.extract_segment_ids_from_initial_cluster_results(initial_cluster_results)
        
        # Convert to sets for comparison
        encoded_ids = set(encoded_segments.keys())
        cluster_ids = set(cluster_segments.keys())
        
        self.log(f"\nSUMMARY:")
        self.log(f"  Encoded text segments: {len(encoded_ids)}")
        self.log(f"  Clean cluster map segments: {len(clean_segment_ids)}")
        self.log(f"  Initial cluster results segments: {len(cluster_ids)}")
        
        # Check for perfect matches
        all_match = (encoded_ids == clean_segment_ids == cluster_ids)
        
        if all_match:
            self.log("✅ ALL SEGMENT IDS MATCH PERFECTLY!", "SUCCESS")
            
            # Verify descriptions match
            desc_mismatches = self.verify_segment_descriptions(encoded_segments, cluster_segments)
            resp_mismatches = self.verify_respondent_ids(encoded_segments, cluster_segments)
            
            if not desc_mismatches and not resp_mismatches:
                self.log("✅ ALL DESCRIPTIONS AND RESPONDENT IDS MATCH!", "SUCCESS")
                return True
            else:
                self.log(f"❌ Found {len(desc_mismatches)} description mismatches and {len(resp_mismatches)} respondent ID mismatches", "ERROR")
                return False
        
        # Find differences
        self.log("\nFINDING DIFFERENCES...", "INFO")
        
        # Compare encoded_text vs clean_cluster_map
        encoded_only, clean_only = self.find_set_differences(
            encoded_ids, clean_segment_ids, 
            "encoded_text", "clean_cluster_map"
        )
        
        # Compare encoded_text vs initial_cluster_results
        encoded_only2, cluster_only = self.find_set_differences(
            encoded_ids, cluster_ids,
            "encoded_text", "initial_cluster_results"
        )
        
        # Compare clean_cluster_map vs initial_cluster_results
        clean_only2, cluster_only2 = self.find_set_differences(
            clean_segment_ids, cluster_ids,
            "clean_cluster_map", "initial_cluster_results"
        )
        
        # Show examples of mismatches
        self.show_examples(encoded_only, encoded_segments, "encoded_text")
        self.show_examples(clean_only, {}, "clean_cluster_map")
        self.show_examples(cluster_only, cluster_segments, "initial_cluster_results")
        
        # Analyze patterns in segment IDs
        self.log("\nANALYZING SEGMENT ID PATTERNS...", "INFO")
        
        encoded_patterns = self.analyze_segment_id_patterns(encoded_ids)
        cluster_patterns = self.analyze_segment_id_patterns(cluster_ids)
        
        self.log(f"Encoded text: {len(encoded_patterns)} unique respondent patterns")
        self.log(f"Cluster results: {len(cluster_patterns)} unique respondent patterns")
        
        # Check for common issues
        if len(encoded_ids) != len(cluster_ids):
            self.log("❌ DIFFERENT NUMBER OF SEGMENTS DETECTED!", "ERROR")
            self.issues_found.append("Different number of segments between sources")
        
        # Verify descriptions for common segments
        if encoded_ids & cluster_ids:  # If there are common segments
            self.log("\nVERIFYING DESCRIPTIONS FOR COMMON SEGMENTS...", "INFO")
            desc_mismatches = self.verify_segment_descriptions(encoded_segments, cluster_segments)
            resp_mismatches = self.verify_respondent_ids(encoded_segments, cluster_segments)
            
            if not desc_mismatches and not resp_mismatches:
                self.log("✅ Descriptions and respondent IDs match for common segments", "SUCCESS")
            
        # Final verdict
        if self.issues_found:
            self.log(f"\n❌ VERIFICATION FAILED - {len(self.issues_found)} issues found:", "ERROR")
            for issue in self.issues_found:
                self.log(f"  - {issue}", "ERROR")
            return False
        else:
            self.log("\n✅ VERIFICATION PASSED - No issues found", "SUCCESS")
            return True


def run_segment_id_verification(encoded_text: List[models.SegmentedModel],
                               clean_cluster_map: Dict[str, int],
                               initial_cluster_results: List[models.ClusterModel],
                               verbose: bool = True) -> bool:
    """
    Run segment ID verification diagnostic
    
    Args:
        encoded_text: Results from step 4 (segmentation)
        clean_cluster_map: Results from clean clusterer
        initial_cluster_results: Results from step 5 (initial clustering)
        verbose: Whether to print detailed output
    
    Returns:
        True if verification passes, False otherwise
    """
    verifier = SegmentIDVerifier(verbose=verbose)
    return verifier.run_verification(encoded_text, clean_cluster_map, initial_cluster_results)


def analyze_segment_id_structure(segment_ids: Set[str], source_name: str) -> None:
    """Analyze the structure of segment IDs to understand the format"""
    print(f"\n{source_name} SEGMENT ID STRUCTURE ANALYSIS:")
    print("-" * 50)
    
    if not segment_ids:
        print("No segment IDs to analyze")
        return
    
    # Sample analysis
    sample_ids = sorted(segment_ids)[:10]
    print(f"Sample segment IDs ({len(sample_ids)}):")
    for seg_id in sample_ids:
        print(f"  {seg_id}")
    
    # Pattern analysis
    patterns = set()
    for seg_id in segment_ids:
        if '_' in seg_id:
            parts = seg_id.split('_')
            pattern = '_'.join(['X'] * len(parts))
            patterns.add(pattern)
    
    print(f"\nID patterns found:")
    for pattern in sorted(patterns):
        count = sum(1 for sid in segment_ids if len(sid.split('_')) == len(pattern.split('_')))
        print(f"  {pattern}: {count} segments")
    
    # Length analysis
    lengths = [len(seg_id) for seg_id in segment_ids]
    print(f"\nID lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    # Example usage for testing
    print("Segment ID Verification Tool")
    print("This tool should be imported and used within the pipeline")
    print("Example usage:")
    print("  from utils.segment_id_verification import run_segment_id_verification")
    print("  success = run_segment_id_verification(encoded_text, clean_cluster_map, initial_cluster_results)")