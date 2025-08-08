"""
Test script for the dual caching functionality
"""

from config import DEFAULT_PROCESSING_CONFIG, ProcessingConfig
import models

def test_detailed_cache_config():
    """Test that the detailed cache configuration is properly set up"""
    
    # Test default config
    print("Testing default configuration:")
    print(f"cache_detailed_step7: {DEFAULT_PROCESSING_CONFIG.cache_detailed_step7}")
    
    # Test custom config with detailed caching enabled
    custom_config = ProcessingConfig(cache_detailed_step7=True)
    print(f"\nTesting custom configuration:")
    print(f"cache_detailed_step7: {custom_config.cache_detailed_step7}")
    
    # Test Step7DetailedResults model
    print(f"\nTesting Step7DetailedResults model creation:")
    try:
        detailed_results = models.Step7DetailedResults(
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
        print("SUCCESS: Step7DetailedResults model created successfully")
        print(f"  - Contains {len(detailed_results.step2_summaries)} cluster summaries")
        print(f"  - Contains {len(detailed_results.step3_recommendations)} recommendations")
        print(f"  - Processing timestamp: {detailed_results.processing_timestamp}")
        
    except Exception as e:
        print(f"ERROR: Error creating Step7DetailedResults: {e}")

if __name__ == "__main__":
    test_detailed_cache_config()