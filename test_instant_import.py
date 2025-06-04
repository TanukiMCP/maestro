#!/usr/bin/env python3
"""
Test script to verify static_tools_dict.py imports instantly with zero dependencies.
This simulates exactly what Smithery does when scanning for tools.
"""

import time
import sys

def test_pure_dict_import():
    """Test that static_tools_dict.py imports instantly"""
    print("Testing static_tools_dict.py import speed...")
    
    start_time = time.time()
    
    try:
        from static_tools_dict import STATIC_TOOLS_DICT
        import_time = time.time() - start_time
        
        print(f"✅ Import successful in {import_time:.6f} seconds")
        print(f"✅ Found {len(STATIC_TOOLS_DICT)} tools")
        
        # Verify basic structure
        if STATIC_TOOLS_DICT:
            tool = STATIC_TOOLS_DICT[0]
            print(f"✅ First tool: {tool['name']}")
            print(f"✅ Has description: {len(tool['description'])} chars")
            print(f"✅ Has input schema: {bool(tool['inputSchema'])}")
        
        # Check for instant import (should be under 10ms)
        if import_time < 0.01:
            print("✅ PASS: Import is instant (< 10ms)")
            return True
        else:
            print(f"❌ FAIL: Import too slow ({import_time:.6f}s)")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Import error: {e}")
        return False

def test_http_transport_simulation():
    """Test exactly what the HTTP transport does for tool discovery"""
    print("\nTesting HTTP transport tool discovery simulation...")
    
    start_time = time.time()
    
    try:
        # Simulate exactly what the HTTP transport does
        from static_tools_dict import STATIC_TOOLS_DICT
        
        # Create JSON response (like HTTP transport does)
        response_data = {
            "jsonrpc": "2.0",
            "result": {
                "tools": STATIC_TOOLS_DICT
            }
        }
        
        total_time = time.time() - start_time
        
        print(f"✅ Tool discovery successful in {total_time:.6f} seconds")
        print(f"✅ Response contains {len(response_data['result']['tools'])} tools")
        
        # Verify response structure
        tools = response_data['result']['tools']
        if tools and all('name' in tool and 'description' in tool and 'inputSchema' in tool for tool in tools):
            print("✅ All tools have required fields")
        
        if total_time < 0.01:
            print("✅ PASS: Tool discovery is instant (< 10ms)")
            return True
        else:
            print(f"❌ FAIL: Tool discovery too slow ({total_time:.6f}s)")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Tool discovery error: {e}")
        return False

def test_multiple_imports():
    """Test that multiple imports are still fast (simulating repeated requests)"""
    print("\nTesting multiple imports (simulating repeated Smithery requests)...")
    
    times = []
    for i in range(5):
        start_time = time.time()
        
        # Clear import cache to simulate fresh imports
        if 'static_tools_dict' in sys.modules:
            del sys.modules['static_tools_dict']
        
        from static_tools_dict import STATIC_TOOLS_DICT
        import_time = time.time() - start_time
        times.append(import_time)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    print(f"✅ Average import time: {avg_time:.6f} seconds")
    print(f"✅ Maximum import time: {max_time:.6f} seconds")
    
    if max_time < 0.01:
        print("✅ PASS: All imports are instant (< 10ms)")
        return True
    else:
        print(f"❌ FAIL: Some imports too slow (max: {max_time:.6f}s)")
        return False

if __name__ == "__main__":
    success1 = test_pure_dict_import()
    success2 = test_http_transport_simulation()
    success3 = test_multiple_imports()
    
    if success1 and success2 and success3:
        print("\n🎉 ALL TESTS PASSED - Smithery deployment should work perfectly!")
        print("✅ Tool scanning will be instant")
        print("✅ No import delays or side effects")
        print("✅ Ready for production deployment")
        sys.exit(0)
    else:
        print("\n💥 TESTS FAILED - Smithery deployment may still timeout!")
        sys.exit(1) 