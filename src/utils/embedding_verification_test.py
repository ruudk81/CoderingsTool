"""
Test to verify that embeddings are correctly aligned with their segments.
This can be run to compare the current embedder vs enhanced embedder.
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import models
import numpy as np
from typing import List
from utils.embedder import Embedder
from utils.enhanced_embedder import EnhancedEmbedder
from config import EmbeddingConfig


def create_test_data() -> List[models.ClusterModel]:
    """Create simple test data with known segment IDs"""
    
    test_data = [
        models.ClusterModel(
            respondent_id=1001,
            response="Test response 1",
            response_segment=[
                models.ClusterSubmodel(
                    segment_id="1001_1",
                    segment_response="Salt is bad",
                    segment_label="SALT_CONCERN",
                    segment_description="Concern about salt content"
                ),
                models.ClusterSubmodel(
                    segment_id="1001_2", 
                    segment_response="Sugar is unhealthy",
                    segment_label="SUGAR_CONCERN",
                    segment_description="Concern about sugar content"
                )
            ]
        ),
        models.ClusterModel(
            respondent_id=1002,
            response="Test response 2", 
            response_segment=[
                models.ClusterSubmodel(
                    segment_id="1002_1",
                    segment_response="Price is too high",
                    segment_label="PRICE_CONCERN", 
                    segment_description="Concern about pricing"
                )
            ]
        ),
        models.ClusterModel(
            respondent_id=1003,
            response="Test response 3",
            response_segment=[
                models.ClusterSubmodel(
                    segment_id="1003_1",
                    segment_response="Taste could be better",
                    segment_label="TASTE_IMPROVEMENT",
                    segment_description="Suggestion for taste improvement"
                ),
                models.ClusterSubmodel(
                    segment_id="1003_2",
                    segment_response="Packaging is wasteful", 
                    segment_label="PACKAGING_WASTE",
                    segment_description="Concern about packaging waste"
                )
            ]
        )
    ]
    
    return test_data


def verify_embedding_alignment(data: List[models.ClusterModel], embedder_name: str) -> bool:
    """Verify that embeddings are correctly aligned with segments"""
    
    print(f"\n=== VERIFYING {embedder_name.upper()} ALIGNMENT ===")
    
    alignment_issues = 0
    
    for resp in data:
        print(f"\nRespondent {resp.respondent_id}:")
        
        if resp.response_segment:
            for segment in resp.response_segment:
                # Check if embeddings exist
                has_code = segment.code_embedding is not None
                has_desc = segment.description_embedding is not None
                
                print(f"  Segment {segment.segment_id}:")
                print(f"    Description: {segment.segment_description}")
                print(f"    Code embedding: {'✓' if has_code else '✗'}")
                print(f"    Desc embedding: {'✓' if has_desc else '✗'}")
                
                # Verify embedding dimensions if they exist
                if has_code:
                    if len(segment.code_embedding.shape) != 1:
                        print(f"    ❌ Code embedding has wrong shape: {segment.code_embedding.shape}")
                        alignment_issues += 1
                    else:
                        print(f"    ✓ Code embedding shape: {segment.code_embedding.shape}")
                
                if has_desc:
                    if len(segment.description_embedding.shape) != 1:
                        print(f"    ❌ Description embedding has wrong shape: {segment.description_embedding.shape}")
                        alignment_issues += 1
                    else:
                        print(f"    ✓ Description embedding shape: {segment.description_embedding.shape}")
    
    print(f"\n{embedder_name} alignment issues: {alignment_issues}")
    return alignment_issues == 0


def run_embedding_verification() -> None:
    """Run verification test comparing embedders"""
    
    print("="*60)
    print("EMBEDDING ALIGNMENT VERIFICATION TEST")
    print("="*60)
    
    # Create test data
    test_data = create_test_data()
    print(f"Created test data with {len(test_data)} responses")
    
    # Test current embedder
    print("\n" + "="*40)
    print("TESTING CURRENT EMBEDDER")
    print("="*40)
    
    current_data = [item.model_copy(deep=True) for item in test_data]  # Deep copy
    
    config = EmbeddingConfig()
    current_embedder = Embedder(config=config, verbose=True)
    
    try:
        # Generate embeddings
        current_data = current_embedder.get_code_embeddings(current_data)
        current_data = current_embedder.get_description_embeddings(current_data, "Test survey question")
        current_data = current_embedder.combine_embeddings(current_data, current_data)
        
        # Verify alignment
        current_aligned = verify_embedding_alignment(current_data, "Current Embedder")
        
    except Exception as e:
        print(f"❌ Current embedder failed: {e}")
        current_aligned = False
    
    # Test enhanced embedder  
    print("\n" + "="*40)
    print("TESTING ENHANCED EMBEDDER")
    print("="*40)
    
    enhanced_data = [item.model_copy(deep=True) for item in test_data]  # Deep copy
    
    try:
        enhanced_embedder = EnhancedEmbedder(config=config, verbose=True)
        
        # Generate embeddings with ID tracking
        enhanced_data = enhanced_embedder.get_code_embeddings_with_tracking(enhanced_data)
        enhanced_data = enhanced_embedder.get_description_embeddings_with_tracking(enhanced_data)
        
        # Verify alignment
        enhanced_aligned = verify_embedding_alignment(enhanced_data, "Enhanced Embedder")
        
    except Exception as e:
        print(f"❌ Enhanced embedder failed: {e}")
        enhanced_aligned = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Current Embedder Aligned: {'✓' if current_aligned else '❌'}")
    print(f"Enhanced Embedder Aligned: {'✓' if enhanced_aligned else '❌'}")
    
    if enhanced_aligned and not current_aligned:
        print("\n🎯 RECOMMENDATION: Switch to Enhanced Embedder")
    elif current_aligned:
        print("\n✅ Current embedder appears to be working correctly")
    else:
        print("\n⚠️  Both embedders have alignment issues - investigate further")


if __name__ == "__main__":
    run_embedding_verification()