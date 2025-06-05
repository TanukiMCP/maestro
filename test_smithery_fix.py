#!/usr/bin/env python3
"""
Test script to validate Smithery.ai compatibility
Tests instant tool discovery and proper MCP protocol handling
"""

import asyncio
import time
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_instant_tool_discovery():
    """Test that tool discovery happens instantly"""
    print("🔍 Testing instant tool discovery...")
    
    start_time = time.time()
    
    try:
        # Import the server module
        import server
        
        # Test the instant tool discovery directly using raw tools (no conversion)
        if hasattr(server, 'INSTANT_TOOLS_RAW'):
            tools_raw = server.INSTANT_TOOLS_RAW
            discovery_time = (time.time() - start_time) * 1000  # Convert to ms
            
            print(f"✅ Raw instant tool access completed in {discovery_time:.2f}ms")
            print(f"📊 Found {len(tools_raw)} raw tools")
            
            # Validate raw tool structure
            for tool in tools_raw:
                if not isinstance(tool, dict) or 'name' not in tool or 'description' not in tool or 'inputSchema' not in tool:
                    print(f"❌ Tool {tool.get('name', 'unknown')} missing required attributes")
                    return False
            
            # Check if discovery time meets Smithery requirement (<100ms)
            if discovery_time < 100:
                print(f"✅ PASS: Raw tool access is under 100ms - Smithery compatible!")
                
                # Now test the converted tools method
                start_time2 = time.time()
                converted_tools = server.get_instant_tools()
                conversion_time = (time.time() - start_time2) * 1000
                
                print(f"📊 Tool conversion took {conversion_time:.2f}ms")
                print(f"📊 Converted {len(converted_tools)} tools")
                
                # Test the proxy method
                start_time3 = time.time()
                proxy_tools = await server.mcp.get_tools()
                proxy_time = (time.time() - start_time3) * 1000
                
                print(f"📊 Proxy method took {proxy_time:.2f}ms")
                
                # Validate that all methods return the same number of tools
                if len(tools_raw) == len(converted_tools) == len(proxy_tools):
                    print(f"✅ All methods return consistent tool count: {len(tools_raw)}")
                else:
                    print(f"❌ Inconsistent tool counts: raw={len(tools_raw)}, converted={len(converted_tools)}, proxy={len(proxy_tools)}")
                    return False
                
                # The key requirement is that raw access is instant
                return True
            else:
                print(f"❌ FAIL: Raw tool access took {discovery_time:.2f}ms (should be <100ms)")
                return False
        else:
            print("❌ INSTANT_TOOLS_RAW not available")
            return False
            
    except Exception as e:
        discovery_time = (time.time() - start_time) * 1000
        print(f"❌ Tool discovery failed after {discovery_time:.2f}ms: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_static_tools_loading():
    """Test that static tools load without dynamic imports"""
    print("\n📦 Testing static tools loading...")
    
    start_time = time.time()
    
    try:
        from static_tools_dict import STATIC_TOOLS_DICT
        load_time = (time.time() - start_time) * 1000
        
        print(f"✅ Static tools loaded in {load_time:.2f}ms")
        print(f"📊 Found {len(STATIC_TOOLS_DICT)} static tool definitions")
        
        # Validate tool structure
        for tool in STATIC_TOOLS_DICT:
            if not isinstance(tool, dict) or 'name' not in tool or 'description' not in tool or 'inputSchema' not in tool:
                print(f"❌ Invalid tool structure: {tool}")
                return False
        
        print("✅ All static tools have valid structure")
        return True
        
    except Exception as e:
        load_time = (time.time() - start_time) * 1000
        print(f"❌ Static tools loading failed after {load_time:.2f}ms: {e}")
        return False

async def test_mcp_protocol_compatibility():
    """Test MCP protocol compatibility"""
    print("\n🔌 Testing MCP protocol compatibility...")
    
    try:
        import server
        
        # Test that the server has the required MCP methods
        required_methods = ['get_tools', 'run']
        for method in required_methods:
            if not hasattr(server.mcp, method):
                print(f"❌ Missing required MCP method: {method}")
                return False
        
        print("✅ All required MCP methods present")
        
        # Test tool execution setup (without actually executing)
        tools = await server.mcp.get_tools()
        tool_names = [tool.name for tool in tools]
        
        expected_tools = [
            'maestro_orchestrate',
            'maestro_collaboration_response', 
            'maestro_iae_discovery',
            'maestro_tool_selection',
            'maestro_iae',
            'get_available_engines',
            'maestro_search',
            'maestro_scrape',
            'maestro_execute',
            'maestro_temporal_context',
            'maestro_error_handler'
        ]
        
        for expected_tool in expected_tools:
            if expected_tool not in tool_names:
                print(f"❌ Missing expected tool: {expected_tool}")
                return False
        
        print(f"✅ All {len(expected_tools)} expected tools present")
        return True
        
    except Exception as e:
        print(f"❌ MCP protocol compatibility test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 TanukiMCP Maestro - Smithery Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("Static Tools Loading", test_static_tools_loading),
        ("Instant Tool Discovery", test_instant_tool_discovery),
        ("MCP Protocol Compatibility", test_mcp_protocol_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Server is Smithery.ai compatible!")
        print("🚀 Ready for deployment!")
    else:
        print("⚠️  SOME TESTS FAILED - Server needs fixes for Smithery compatibility")
        print("🔧 Please address the issues above before deployment")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 