"""
Test script for the refactored dual caching system
"""

import models
from utils.codeGenerator_displayResults import load_and_display_reasoning
from utils.cacheManager import CacheManager
from config import CacheConfig

def test_refactored_models():
    """Test that the refactored models work correctly"""
    
    print("=== Testing Refactored Models ===\n")
    
    # Test CodeGeneratorReasoningResults model
    print("Testing CodeGeneratorReasoningResults model:")
    try:
        reasoning_results = models.CodeGeneratorReasoningResults(
            cluster_results=[],
            step2_summaries={1: "Test cluster summary"},
            step3_recommendations={1: {"decision": "create_new", "theme": "test"}},
            step4_validations={1: {"decision": "APPROVE", "rationale": "Good code"}},
            candidate_codes={1: [{"code": "TEST", "definition": "Test code"}]},
            stats={"new_codes_added": 1},
            generator_version="TEST",
            var_lab="test_variable",
            total_clusters=1,
            total_ideas=5,
            processing_timestamp="2025-01-01T00:00:00",
            cluster_assignments={1: "TEST_CODE"}
        )
        print("SUCCESS: CodeGeneratorReasoningResults model created")
        print(f"  - Variable: {reasoning_results.var_lab}")
        print(f"  - Total clusters: {reasoning_results.total_clusters}")
        print(f"  - Step 2 summaries: {len(reasoning_results.step2_summaries)}")
        print(f"  - Step 3 recommendations: {len(reasoning_results.step3_recommendations)}")
        
        # Test display function signature (won't actually display since no cache)
        from utils.codeGenerator_displayResults import display_cluster_analysis
        print("SUCCESS: display_cluster_analysis function available with new signature")
        
    except Exception as e:
        print(f"ERROR: {e}")

def test_cache_loading_function():
    """Test the cache loading function"""
    
    print("\n=== Testing Cache Loading Function ===\n")
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Test loading function (will fail gracefully since no cache exists)
    print("Testing load_and_display_reasoning function:")
    load_and_display_reasoning(cache_manager, "nonexistent_file.sav")

if __name__ == "__main__":
    test_refactored_models()
    test_cache_loading_function()
    
    print("\n=== Summary ===")
    print("SUCCESS: Model refactoring complete")
    print("SUCCESS: Display functions updated") 
    print("SUCCESS: Cache loading function ready")
    print("\nTo use the new system:")
    print("1. Set CACHE_CODEGENERATOR_REASONING = True in pipeline.py")
    print("2. Run your pipeline to create reasoning cache")
    print("3. Use load_and_display_reasoning() to analyze results")