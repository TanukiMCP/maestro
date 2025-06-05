#!/usr/bin/env python3
"""
Test script to verify TanukiMCP Maestro HTTP endpoints work correctly.
This helps debug deployment issues with Smithery.ai.
"""

import asyncio
import aiohttp
import json
import sys
import time

async def test_endpoints():
    """Test all HTTP endpoints to ensure they work."""
    
    base_url = "http://localhost:8000"
    endpoints = ["/health", "/tools", "/debug", "/mcp"]
    
    print("🧪 Testing TanukiMCP Maestro HTTP endpoints...")
    print(f"🌐 Base URL: {base_url}")
    print("-" * 50)
    
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            print(f"📡 Testing {endpoint}...")
            
            try:
                start_time = time.time()
                async with session.get(url) as response:
                    duration = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        content_type = response.headers.get('content-type', 'unknown')
                        
                        if 'json' in content_type:
                            data = await response.json()
                            print(f"  ✅ Status: {response.status} ({duration:.1f}ms)")
                            print(f"  📄 Response: {json.dumps(data, indent=2)}")
                        else:
                            text = await response.text()
                            print(f"  ✅ Status: {response.status} ({duration:.1f}ms)")
                            print(f"  📄 Response: {text}")
                    else:
                        text = await response.text()
                        print(f"  ❌ Status: {response.status} ({duration:.1f}ms)")
                        print(f"  📄 Error: {text}")
                        
            except asyncio.TimeoutError:
                print(f"  ⏰ Timeout: No response within 10 seconds")
            except aiohttp.ClientConnectorError:
                print(f"  🔌 Connection Error: Server not running or unreachable")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            print()
    
    print("🏁 Testing complete!")

async def test_mcp_tools():
    """Test MCP tools endpoint specifically."""
    print("\n🔧 Testing MCP tools discovery...")
    
    base_url = "http://localhost:8000"
    mcp_url = f"{base_url}/mcp"
    
    # Test MCP protocol endpoint
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # Try to get tools list (this is what Smithery.ai does)
            tools_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            start_time = time.time()
            async with session.post(
                mcp_url,
                json=tools_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                duration = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ MCP tools/list: {response.status} ({duration:.1f}ms)")
                    print(f"  📄 Response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"  ❌ MCP tools/list: {response.status} ({duration:.1f}ms)")
                    print(f"  📄 Error: {text}")
                    
        except Exception as e:
            print(f"  ❌ MCP Error: {e}")

if __name__ == "__main__":
    print("🚀 TanukiMCP Maestro HTTP Endpoint Test")
    print("=" * 50)
    
    try:
        asyncio.run(test_endpoints())
        asyncio.run(test_mcp_tools())
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1) 