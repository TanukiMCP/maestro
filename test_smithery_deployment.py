#!/usr/bin/env python3
"""
Test script to verify Smithery.ai deployment compatibility.
Tests HTTP transport, health check, and tool discovery speed.
"""

import asyncio
import time
import httpx
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_smithery_deployment():
    """Test Smithery.ai deployment requirements."""
    print("🧪 Testing Smithery.ai Deployment Compatibility")
    print("=" * 50)
    
    # Test 1: Import speed (should be fast)
    print("1. Testing import speed...")
    start_time = time.time()
    
    try:
        from server import mcp
        import_time = (time.time() - start_time) * 1000
        print(f"   ✅ Import time: {import_time:.2f}ms")
        
        if import_time > 100:
            print(f"   ⚠️  Warning: Import time ({import_time:.2f}ms) > 100ms")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Tool discovery speed
    print("2. Testing tool discovery speed...")
    start_time = time.time()
    
    try:
        # This should be instant with STATIC_TOOLS_DICT
        tools = mcp._tools if hasattr(mcp, '_tools') else {}
        discovery_time = (time.time() - start_time) * 1000
        print(f"   ✅ Tool discovery time: {discovery_time:.2f}ms")
        print(f"   ✅ Found {len(tools)} tools")
        
        if discovery_time > 100:
            print(f"   ⚠️  Warning: Discovery time ({discovery_time:.2f}ms) > 100ms")
            
    except Exception as e:
        print(f"   ❌ Tool discovery failed: {e}")
        return False
    
    # Test 3: HTTP server startup (simulate Smithery environment)
    print("3. Testing HTTP server startup...")
    
    try:
        # Set PORT environment variable like Smithery does
        os.environ["PORT"] = "8080"
        
        # Start server in background
        server_task = asyncio.create_task(test_http_server())
        
        # Give server time to start
        await asyncio.sleep(2)
        
        # Test health check endpoint
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8080/health", timeout=5.0)
                if response.status_code == 200 and response.text == "ok":
                    print("   ✅ Health check endpoint working")
                else:
                    print(f"   ❌ Health check failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Health check request failed: {e}")
        
        # Test MCP endpoint
        async with httpx.AsyncClient() as client:
            try:
                # Try to connect to MCP endpoint (should at least respond)
                response = await client.get("http://localhost:8080/mcp", timeout=5.0)
                print(f"   ✅ MCP endpoint responding (status: {response.status_code})")
                
            except Exception as e:
                print(f"   ⚠️  MCP endpoint test: {e}")
        
        # Cancel server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
            
    except Exception as e:
        print(f"   ❌ HTTP server test failed: {e}")
        return False
    
    print("\n🎉 Smithery.ai compatibility tests completed!")
    return True

async def test_http_server():
    """Start the server for testing."""
    try:
        from server import mcp
        await mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=8080,
            path="/mcp"
        )
    except Exception as e:
        print(f"Server startup error: {e}")

if __name__ == "__main__":
    asyncio.run(test_smithery_deployment()) 