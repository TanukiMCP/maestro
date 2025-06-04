#!/usr/bin/env python3
"""
Comprehensive test script for all TanukiMCP Maestro tools
"""

import json
import requests
import time

def test_tool(tool_name, arguments, description):
    """Test a single tool"""
    print(f"\n{'='*60}")
    print(f"🔧 Testing: {tool_name}")
    print(f"📝 Description: {description}")
    print(f"{'='*60}")
    
    payload = {
        "jsonrpc": "2.0",
        "id": f"test-{tool_name}",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/mcp",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        print(f"⏱️ Response time: {end_time - start_time:.2f}s")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("result", {})
            content = result.get("content", [])
            if content:
                print("✅ SUCCESS - Tool response:")
                for item in content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        # Show first 300 chars for readability
                        if len(text) > 300:
                            print(text[:300] + "...")
                        else:
                            print(text)
            else:
                print("⚠️ Empty response")
        else:
            print(f"❌ ERROR: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT - Tool took too long to respond")
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

def main():
    """Test all maestro tools"""
    print("🚀 TanukiMCP Maestro - Comprehensive Tool Testing")
    print("=" * 60)
    
    # Test 1: get_available_engines (simplest)
    test_tool(
        "get_available_engines",
        {},
        "List available computational engines"
    )
    
    # Test 2: maestro_iae_discovery
    test_tool(
        "maestro_iae_discovery",
        {
            "task_type": "mathematical_analysis",
            "requirements": ["numerical_computation", "statistical_analysis"],
            "complexity": "medium"
        },
        "Discover optimal Intelligence Amplification Engine"
    )
    
    # Test 3: maestro_tool_selection
    test_tool(
        "maestro_tool_selection",
        {
            "task_description": "Analyze website performance and suggest improvements",
            "available_tools": ["maestro_search", "maestro_scrape", "maestro_iae"],
            "constraints": {"time_limit": "5_minutes"}
        },
        "Intelligent tool selection and recommendation"
    )
    
    # Test 4: maestro_orchestrate (basic task)
    test_tool(
        "maestro_orchestrate",
        {
            "task_description": "Calculate the area of a circle with radius 5",
            "complexity_level": "basic",
            "reasoning_focus": "logical"
        },
        "Enhanced meta-reasoning orchestration"
    )
    
    # Test 5: maestro_iae (computational engine)
    test_tool(
        "maestro_iae",
        {
            "analysis_request": "Calculate basic statistics for the dataset [1, 2, 3, 4, 5]",
            "engine_type": "auto",
            "data": [1, 2, 3, 4, 5]
        },
        "Intelligence Amplification Engine"
    )
    
    # Test 6: maestro_temporal_context
    test_tool(
        "maestro_temporal_context",
        {
            "context_request": "What are the current trends in AI development as of 2024?",
            "time_frame": "current"
        },
        "Time-aware reasoning and context analysis"
    )
    
    # Test 7: maestro_error_handler
    test_tool(
        "maestro_error_handler",
        {
            "error_context": "Python script fails with 'ModuleNotFoundError: No module named numpy'",
            "error_details": {"language": "python", "environment": "local"}
        },
        "Intelligent error analysis and recovery"
    )
    
    # Test 8: maestro_execute
    test_tool(
        "maestro_execute",
        {
            "execution_type": "code",
            "content": "print('Hello from TanukiMCP Maestro!')",
            "language": "python"
        },
        "Secure code and workflow execution"
    )
    
    print(f"\n{'='*60}")
    print("🏁 Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 