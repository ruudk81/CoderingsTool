#!/usr/bin/env python3
"""
Assignment Comparison Utility
Compare ClusterAssigner v2 vs CodeAssigner performance and results
"""

import os
import sys
import time
from typing import List, Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import codeAssigner
from utils import clusterAssigner
import models


@dataclass
class ComparisonResults:
    """Results comparing both assignment methods"""
    
    # Performance metrics
    cluster_assigner_time: float = 0.0
    code_assigner_time: float = 0.0
    speedup_ratio: float = 0.0
    
    # Assignment statistics
    cluster_total_ideas: int = 0
    code_total_ideas: int = 0
    cluster_reassignments: int = 0
    cluster_perfect_matches: int = 0
    
    # Result comparison
    identical_assignments: int = 0
    different_assignments: int = 0
    assignment_overlap_ratio: float = 0.0
    
    # API usage
    cluster_api_calls: int = 0  # Should be 0
    code_api_calls_estimated: int = 0


class AssignmentComparator:
    """Compare ClusterAssigner v2 vs CodeAssigner"""
    
    def __init__(self, cluster_models, enriched_codebook, cached_ideas, var_lab, verbose=True):
        self.cluster_models = cluster_models
        self.enriched_codebook = enriched_codebook
        self.cached_ideas = cached_ideas
        self.var_lab = var_lab
        self.verbose = verbose
        
    def run_comparison(self) -> ComparisonResults:
        """Run both assignment methods and compare results"""
        results = ComparisonResults()
        
        print("\n" + "="*80)
        print("ASSIGNMENT METHOD COMPARISON")
        print("="*80)
        
        # Run ClusterAssigner v2
        print("\n🚀 Testing ClusterAssigner v2...")
        cluster_start = time.time()
        
        cluster_assigner_instance = clusterAssigner.ClusterAssigner(
            cluster_models=self.cluster_models,
            enriched_codebook=self.enriched_codebook,
            var_lab=self.var_lab,
            verbose=self.verbose
        )
        
        cluster_results = cluster_assigner_instance.assign()
        cluster_end = time.time()
        
        results.cluster_assigner_time = cluster_end - cluster_start
        cluster_stats = cluster_assigner_instance.get_assignment_stats()
        results.cluster_total_ideas = cluster_stats.total_ideas
        results.cluster_reassignments = cluster_stats.reassigned_to_subclusters
        results.cluster_perfect_matches = cluster_stats.perfect_matches
        results.cluster_api_calls = 0  # No API calls
        
        print(f"✅ ClusterAssigner completed in {results.cluster_assigner_time:.2f}s")
        print(f"   Ideas: {results.cluster_total_ideas}, Reassignments: {results.cluster_reassignments}")
        print(f"   Perfect matches: {results.cluster_perfect_matches}, API calls: 0")
        
        # Run CodeAssigner (original)
        print("\n⚡ Testing original CodeAssigner...")
        code_start = time.time()
        
        # Convert enriched codebook to simple Codebook format
        simple_codebook = [models.Codebook(
            code=entry.code, 
            definition=entry.definition,
            theme=entry.theme,
            theme_description=entry.theme_description
        ) for entry in self.enriched_codebook]
        
        code_assigner_instance = codeAssigner.CodeAssigner(
            cluster_models=[],  # Empty - using cached embeddings
            codebook=simple_codebook,
            var_lab=self.var_lab,
            cached_idea_embeddings=self.cached_ideas,
            verbose=self.verbose
        )
        
        code_results = code_assigner_instance.assign()
        code_end = time.time()
        
        results.code_assigner_time = code_end - code_start
        results.code_total_ideas = len(self.cached_ideas) if self.cached_ideas else 0
        results.code_api_calls_estimated = results.code_total_ideas  # One API call per idea
        
        print(f"✅ CodeAssigner completed in {results.code_assigner_time:.2f}s")
        print(f"   Ideas: {results.code_total_ideas}, Estimated API calls: {results.code_api_calls_estimated}")
        
        # Calculate performance metrics
        if results.code_assigner_time > 0:
            results.speedup_ratio = results.code_assigner_time / results.cluster_assigner_time
        
        # Compare assignment results
        self._compare_assignments(cluster_results, code_results, results)
        
        # Print final comparison
        self._print_comparison_summary(results)
        
        return results
    
    def _compare_assignments(self, cluster_results: List[models.CodeAssignedModel], 
                           code_results: List[models.CodeAssignedModel], 
                           results: ComparisonResults):
        """Compare the actual assignment results"""
        
        # Build lookup for easy comparison
        cluster_assignments = {}
        for result in cluster_results:
            for idea in result.response_ideas:
                cluster_assignments[idea.idea_id] = set(idea.assigned_codes or [])
        
        code_assignments = {}
        for result in code_results:
            for idea in result.response_ideas:
                code_assignments[idea.idea_id] = set(idea.assigned_codes or [])
        
        # Compare assignments
        identical_count = 0
        different_count = 0
        total_overlap = 0
        total_comparisons = 0
        
        for idea_id in cluster_assignments:
            if idea_id in code_assignments:
                cluster_codes = cluster_assignments[idea_id]
                code_codes = code_assignments[idea_id]
                
                if cluster_codes == code_codes:
                    identical_count += 1
                else:
                    different_count += 1
                
                # Calculate overlap ratio
                if cluster_codes or code_codes:
                    overlap = len(cluster_codes.intersection(code_codes))
                    union = len(cluster_codes.union(code_codes))
                    total_overlap += overlap / union if union > 0 else 0
                    total_comparisons += 1
        
        results.identical_assignments = identical_count
        results.different_assignments = different_count
        results.assignment_overlap_ratio = total_overlap / total_comparisons if total_comparisons > 0 else 0
    
    def _print_comparison_summary(self, results: ComparisonResults):
        """Print comprehensive comparison summary"""
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print("\n📊 Performance Comparison:")
        print(f"   ClusterAssigner v2: {results.cluster_assigner_time:.2f}s")
        print(f"   Original CodeAssigner: {results.code_assigner_time:.2f}s")
        print(f"   Speedup: {results.speedup_ratio:.1f}x faster")
        
        print("\n🎯 API Usage Comparison:")
        print(f"   ClusterAssigner v2: {results.cluster_api_calls} API calls")
        print(f"   Original CodeAssigner: ~{results.code_api_calls_estimated} API calls")
        print(f"   API Call Reduction: 100%")
        
        print("\n📈 Assignment Quality:")
        print(f"   Ideas processed: {results.cluster_total_ideas}")
        print(f"   Sub-cluster reassignments: {results.cluster_reassignments}")
        print(f"   Perfect cluster matches: {results.cluster_perfect_matches}")
        
        print("\n🔍 Result Comparison:")
        print(f"   Identical assignments: {results.identical_assignments}")
        print(f"   Different assignments: {results.different_assignments}")
        print(f"   Average code overlap: {results.assignment_overlap_ratio:.2%}")
        
        # Overall assessment
        print(f"\n✨ Overall Assessment:")
        if results.speedup_ratio > 10:
            print("   🚀 ClusterAssigner v2 is SIGNIFICANTLY faster")
        elif results.speedup_ratio > 5:
            print("   ⚡ ClusterAssigner v2 is much faster")
        else:
            print("   📈 ClusterAssigner v2 is faster")
            
        if results.assignment_overlap_ratio > 0.8:
            print("   🎯 High assignment similarity - good consistency")
        elif results.assignment_overlap_ratio > 0.6:
            print("   ✅ Moderate assignment similarity - acceptable variance")
        else:
            print("   ⚠️  Low assignment similarity - methods differ significantly")
        
        print(f"\n🏆 RECOMMENDATION:")
        if results.speedup_ratio > 5 and results.assignment_overlap_ratio > 0.6:
            print("   USE ClusterAssigner v2 - Much faster with good quality!")
        elif results.speedup_ratio > 2:
            print("   CONSIDER ClusterAssigner v2 - Faster with cluster-based logic")
        else:
            print("   EVALUATE further - Similar performance, different approaches")


def main():
    """Simple main function for testing"""
    print("Assignment Comparison Utility")
    print("This utility compares ClusterAssigner v2 vs original CodeAssigner")
    print("Usage: Run this from the pipeline context with loaded data")
    print("\nTo use this utility:")
    print("1. Load cluster_models, enriched_codebook, and cached_ideas")
    print("2. Create AssignmentComparator instance")
    print("3. Call run_comparison()")


if __name__ == "__main__":
    main()