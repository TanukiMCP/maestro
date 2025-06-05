#!/usr/bin/env python3
"""
Test script to verify Smithery compatibility of TanukiMCP Maestro.
Tests that the server can start in HTTP mode and tools are discoverable instantly.
"""

import asyncio
import os
import subprocess
import time
import sys
import requests
from typing import Dict, Any
import json

async def test_tool_discovery_speed():
    """Test that tool discovery happens instantly without dynamic imports."""
    print("Testing tool discovery speed...")
    
    # Import the MCP server
    from server import mcp
    
    start_time = time.time()
    
    # Get tool list - this should be instant
    tools = await mcp.get_tools()
    
    discovery_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"   Tool discovery took {discovery_time:.2f}ms")
    print(f"   Found {len(tools)} tools:")
    
    for tool in tools:
        # Handle both string names and tool objects
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        print(f"      - {tool_name}")
    
    # Verify all expected tools are present
    expected_tools = [
        "maestro_orchestrate",
        "maestro_collaboration_response", 
        "maestro_iae_discovery",
        "maestro_tool_selection",
        "maestro_iae",
        "get_available_engines",
        "maestro_search",
        "maestro_scrape",
        "maestro_execute", 
        "maestro_temporal_context",
        "maestro_error_handler"
    ]
    
    tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
    missing_tools = [tool for tool in expected_tools if tool not in tool_names]
    if missing_tools:
        print(f"   Missing tools: {missing_tools}")
        return False
    
    if discovery_time > 100:
        print(f"   WARNING: Tool discovery took {discovery_time:.2f}ms (should be <100ms for Smithery)")
        return False
        
    print("   Tool discovery speed test passed!")
    return True

def test_http_server_startup():
    """Test that the server can start in HTTP mode."""
    print("Testing HTTP server startup...")
    
    # Set PORT environment variable to trigger HTTP mode (use different port to avoid conflicts)
    os.environ["PORT"] = "18001"
    
    try:
        # Start server in a subprocess
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it time to start
        time.sleep(3)
        
        # Check if it's running
        if process.poll() is None:
            print("   HTTP server started successfully")
            
            # Test MCP endpoint
            try:
                response = requests.get("http://localhost:18001/mcp", timeout=5)
                # MCP endpoint might return different status codes, we just want it to exist
                print(f"   MCP endpoint accessible (status: {response.status_code})")
                if response.status_code in [200, 405, 406]:  # 405/406 Method Not Allowed is OK for GET on MCP
                    print("   MCP endpoint responding correctly")
                else:
                    print(f"   Unexpected MCP endpoint status: {response.status_code}")
                    return False
            except requests.RequestException as e:
                print(f"   MCP endpoint test failed: {e}")
                return False
                
            # Test that server is working
            print("   Server is responding to requests")
            
            # Terminate the server
            process.terminate()
            process.wait(timeout=5)
            print("   Server shutdown cleanly")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"   Server failed to start")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return False
            
    except Exception as e:
        print(f"   HTTP server test failed: {e}")
        return False
    finally:
        # Clean up environment
        if "PORT" in os.environ:
            del os.environ["PORT"]

def test_static_imports():
    """Test that no heavy imports happen at module level."""
    print("[TEST] Testing static imports...")
    
    start_time = time.time()
    
    # Import the server module
    import server
    
    import_time = (time.time() - start_time) * 1000
    
    print(f"   [PASS] Module import took {import_time:.2f}ms")
    
    # Check that lazy loading globals are None initially
    if server._maestro_tools is None and server._computational_tools is None:
        print("   [PASS] Lazy loading variables are properly initialized as None")
    else:
        print("   [FAIL] Lazy loading variables were initialized during import")
        return False
    
    if import_time > 1000:
        print(f"   [WARN]  Module import took {import_time:.2f}ms (should be <1000ms for Smithery)")
        return False
    elif import_time > 500:
        print(f"   [WARN]  Module import took {import_time:.2f}ms (ideally <500ms for optimal Smithery performance)")
        # Don't fail the test, just warn
        
    print("   [PASS] Static imports test passed!")
    return True

async def main():
    """Run all compatibility tests."""
    print("[START] TanukiMCP Maestro - Smithery Compatibility Test Suite")
    print("=" * 60)
    
    tests = [
        ("Static imports", test_static_imports),
        ("Tool discovery speed", test_tool_discovery_speed),
        ("HTTP server startup", test_http_server_startup),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n[TARGET] Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"   Test failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("[DATA] Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"   {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Server should work with Smithery.ai deployment.")
        print("[INFO] The server will automatically detect Smithery environment (PORT env var)")
        print("   and use HTTP/SSE transport instead of STDIO.")
        return 0
    else:
        print("[WARN]  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 