#!/usr/bin/env python3
"""
Test script to verify cache fix for single models wrapped in lists
"""
import sys
import os
sys.path.append('/workspaces/CoderingsTool/src')
sys.path.append('/workspaces/CoderingsTool/src/utils')

from config import DEFAULT_CACHE_CONFIG
from utils.cacheManager import CacheManager
import models

# Initialize cache manager
cache_manager = CacheManager(DEFAULT_CACHE_CONFIG)

# Test data - create a single CodebookModel
test_codebook = models.CodebookModel(
    codes=[
        models.CodebookEntry(code="TEST_CODE", definition="Test definition", source_clusters=[1])
    ],
    generation_metadata={"test": True},
    source_variable="test_variable"
)

# Test save with wrapped model (should work)
print("Testing cache save with wrapped model...")
success = cache_manager.save_to_cache([test_codebook], "test_file.sav", "test_step", 1.0)
print(f"Save result: {success}")

if success:
    print("✅ Cache save succeeded!")
    
    # Test load back
    print("Testing cache load...")
    loaded_models = cache_manager.load_from_cache("test_file.sav", "test_step", models.CodebookModel)
    
    if loaded_models and len(loaded_models) > 0:
        loaded_model = loaded_models[0]
        print(f"✅ Cache load succeeded! Loaded {len(loaded_model.codes)} codes")
        print(f"   Source variable: {loaded_model.source_variable}")
        print(f"   First code: {loaded_model.codes[0].code}")
    else:
        print("❌ Cache load failed!")
else:
    print("❌ Cache save failed!")

print("\nCache test completed.")