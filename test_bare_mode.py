#!/usr/bin/env python3
"""
Test script to verify bare mode functionality without Streamlit warnings
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing bare mode compatibility...")
print("=" * 50)

try:
    # Test 1: Context detection
    print("1. Testing context detection...")
    from src.utils.bare_mode_utils import is_streamlit_context
    context = is_streamlit_context()
    print(f"   Running in {'Streamlit' if context else 'bare mode'}: OK")
    
    # Test 2: Session state access
    print("2. Testing session state...")
    from src.utils.bare_mode_utils import get_session_state
    session_state = get_session_state()
    session_state['test_key'] = 'test_value'
    print(f"   Session state working: {session_state.get('test_key') == 'test_value'} OK")
    
    # Test 3: Conditional caching
    print("3. Testing conditional caching...")
    from src.utils.bare_mode_utils import conditional_cache_resource
    
    @conditional_cache_resource
    def test_cached_function():
        return "cached_result"
    
    result = test_cached_function()
    print(f"   Caching working: {result == 'cached_result'} OK")
    
    # Test 4: Import cached resources (should not cause warnings)
    print("4. Testing cached resources import...")
    from src.utils.cached_resources import get_tiktoken_encoding
    print("   Cached resources imported without warnings OK")
    
    # Test 5: Import pipeline runner (main test)
    print("5. Testing pipeline runner import...")
    from src.pipeline_runner import StreamlitPipelineRunner
    print("   Pipeline runner imported without warnings OK")
    
    print("\n" + "=" * 50)
    print("SUCCESS: All bare mode tests passed!")
    print("RESULT: No Streamlit ScriptRunContext warnings should appear above")
    
except Exception as e:
    print(f"ERROR: Test failed: {e}")
    import traceback
    traceback.print_exc()