#!/usr/bin/env python3
"""
Test script for Maestro MCP Server HTTP/SSE endpoint
"""

import requests
import json
import time
import sys

def test_health_endpoint(port=8001):
    """Test the health check endpoint."""
    try:
        url = f"http://localhost:{port}/"
        print(f"🔍 Testing health endpoint: {url}")
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"📊 Server info:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server on port {port}")
        return False
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        return False

def test_sse_endpoint(port=8001):
    """Test the SSE endpoint availability."""
    try:
        url = f"http://localhost:{port}/sse/"
        print(f"🔍 Testing SSE endpoint: {url}")
        
        response = requests.get(url, timeout=5, stream=True)
        
        if response.status_code == 200:
            print("✅ SSE endpoint is accessible")
            # Check headers
            content_type = response.headers.get('content-type', '')
            if 'text/event-stream' in content_type:
                print("✅ SSE headers are correct")
                return True
            else:
                print(f"⚠️ SSE headers may be incorrect: {content_type}")
                return True
        else:
            print(f"❌ SSE endpoint failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to SSE endpoint on port {port}")
        return False
    except Exception as e:
        print(f"❌ Error testing SSE endpoint: {e}")
        return False

def main():
    port = 8001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number")
            sys.exit(1)
    
    print(f"🎭 Testing Maestro MCP Server on port {port}")
    print("=" * 50)
    
    # Test health endpoint
    health_ok = test_health_endpoint(port)
    print()
    
    # Test SSE endpoint
    sse_ok = test_sse_endpoint(port)
    print()
    
    # Summary
    print("=" * 50)
    if health_ok and sse_ok:
        print("🎉 All tests passed! Server is ready for MCP clients.")
        print(f"🔗 SSE endpoint for MCP clients: http://localhost:{port}/sse/")
        print(f"📡 Messages endpoint: http://localhost:{port}/messages/")
    else:
        print("❌ Some tests failed. Check server status.")
        sys.exit(1)

if __name__ == "__main__":
    main() 