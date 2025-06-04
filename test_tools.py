#!/usr/bin/env python3
"""
Test script for TanukiMCP Maestro tools via HTTP transport
"""

import json
import requests
import asyncio

def test_tool_discovery():
    """Test tool discovery endpoint"""
    print("Testing tool discovery...")
    try:
        response = requests.get("http://localhost:8000/mcp")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            tools = data.get("result", {}).get("tools", [])
            print(f"Found {len(tools)} tools:")
            for tool in tools[:3]:  # Show first 3 tools
                print(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:100]}...")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_tool_execution():
    """Test tool execution"""
    print("\nTesting tool execution...")
    
    # Test get_available_engines (simplest tool)
    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "tools/call",
        "params": {
            "name": "get_available_engines",
            "arguments": {
                "detailed": False
            }
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/mcp",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Response:", json.dumps(data, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_maestro_orchestrate():
    """Test maestro_orchestrate tool"""
    print("\nTesting maestro_orchestrate...")
    
    payload = {
        "jsonrpc": "2.0",
        "id": "test-2",
        "method": "tools/call",
        "params": {
            "name": "maestro_orchestrate",
            "arguments": {
                "task_description": "Explain the difference between Python lists and tuples in simple terms",
                "complexity_level": "basic"
            }
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/mcp",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            result = data.get("result", {})
            content = result.get("content", [])
            if content:
                print("Tool response:")
                for item in content:
                    if item.get("type") == "text":
                        print(item.get("text", "")[:500] + "..." if len(item.get("text", "")) > 500 else item.get("text", ""))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_tool_discovery()
    test_tool_execution()
    test_maestro_orchestrate() 