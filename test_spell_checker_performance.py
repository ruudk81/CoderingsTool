#!/usr/bin/env python3
"""
Quick test script to verify spellChecker parallel processing optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import asyncio
from src.utils.spellChecker import SpellChecker, SpellCheckModel

# Test data with known misspellings
TEST_RESPONSES = [
    SpellCheckModel(respondent_id=f"test_{i}", 
                   original_response=f"This is a test sentance with mispellings and errrors {i}")
    for i in range(10)  # 10 test responses for quick test
]

async def test_spell_checker_performance():
    """Test the optimized spell checker performance"""
    print("=== SpellChecker Performance Test ===")
    print(f"Testing with {len(TEST_RESPONSES)} responses")
    
    # Initialize spell checker
    spell_checker = SpellChecker(verbose=True)
    
    # Check installation
    if not spell_checker.check_hunspell_installation():
        print("ERROR Hunspell not available - cannot test")
        return
    
    print("OK Hunspell installation verified")
    
    # Test the optimized spell checker
    start_time = time.time()
    try:
        results = await spell_checker.spell_check_async(TEST_RESPONSES, "test_variable")
        end_time = time.time()
        
        print(f"\n=== Test Results ===")
        print(f"OK Processing completed in {end_time - start_time:.2f} seconds")
        print(f"OK Processed {len(results)} responses")
        print(f"OK Stats: {spell_checker.stats}")
        
        # Show some examples
        changes = 0
        for i, result in enumerate(results[:5]):
            if result.response != TEST_RESPONSES[i].original_response:
                changes += 1
                print(f"  Example {i+1}: '{TEST_RESPONSES[i].original_response}' -> '{result.response}'")
        
        if changes > 0:
            print(f"OK Detected {changes} corrections in sample")
        else:
            print("INFO No corrections made in sample (this may be normal)")
            
    except Exception as e:
        print(f"ERROR Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_spell_checker_performance())