#!/usr/bin/env python3
"""
Comprehensive test for all 11 TanukiMCP Maestro tools
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
                        # Show first 200 chars for readability
                        if len(text) > 200:
                            print(text[:200] + "...")
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
    """Test all 11 maestro tools"""
    print("🚀 TanukiMCP Maestro - Complete Tool Testing Suite")
    print("Testing all 11 tools with realistic examples")
    print("=" * 60)
    
    # Test 1: get_available_engines
    test_tool(
        "get_available_engines",
        {},
        "List available computational engines"
    )
    
    # Test 2: maestro_orchestrate
    test_tool(
        "maestro_orchestrate",
        {
            "task_description": "Calculate the factorial of 5 and explain the process",
            "complexity_level": "basic",
            "reasoning_focus": "logical"
        },
        "Enhanced meta-reasoning orchestration"
    )
    
    # Test 3: maestro_iae_discovery
    test_tool(
        "maestro_iae_discovery",
        {
            "task_type": "mathematical_analysis",
            "requirements": ["numerical_computation", "statistical_analysis"],
            "complexity": "medium"
        },
        "Discover optimal Intelligence Amplification Engine"
    )
    
    # Test 4: maestro_tool_selection
    test_tool(
        "maestro_tool_selection",
        {
            "task_description": "Analyze website performance and suggest improvements",
            "available_tools": ["maestro_search", "maestro_scrape", "maestro_iae"],
            "constraints": {"time_limit": "5_minutes"}
        },
        "Intelligent tool selection and recommendation"
    )
    
    # Test 5: maestro_iae
    test_tool(
        "maestro_iae",
        {
            "analysis_request": "Calculate basic statistics for the dataset [1, 2, 3, 4, 5]",
            "engine_type": "auto",
            "data": [1, 2, 3, 4, 5]
        },
        "Intelligence Amplification Engine"
    )
    
    # Test 6: maestro_collaboration_response
    test_tool(
        "maestro_collaboration_response",
        {
            "collaboration_id": "test_collab_001",
            "responses": {"user_choice": "proceed", "feedback": "looks good"},
            "approval_status": "approved"
        },
        "Handle user responses during collaborative workflows"
    )
    
    # Test 7: maestro_search
    test_tool(
        "maestro_search",
        {
            "query": "latest developments in artificial intelligence 2024",
            "max_results": 5,
            "temporal_filter": "recent",
            "result_format": "summary"
        },
        "Enhanced web search with LLM-powered analysis"
    )
    
    # Test 8: maestro_scrape
    test_tool(
        "maestro_scrape",
        {
            "url": "https://example.com/article",
            "extraction_type": "text",
            "selectors": {"title": "h1", "content": ".article-body"}
        },
        "Intelligent web scraping with content extraction"
    )
    
    # Test 9: maestro_execute
    test_tool(
        "maestro_execute",
        {
            "execution_type": "code",
            "content": "print('Hello from TanukiMCP Maestro!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')",
            "language": "python",
            "validation_level": "basic"
        },
        "Secure code and workflow execution"
    )
    
    # Test 10: maestro_temporal_context
    test_tool(
        "maestro_temporal_context",
        {
            "context_request": "What are the current trends in AI development as of 2024?",
            "time_frame": "current",
            "temporal_factors": ["technology_evolution", "market_trends", "research_developments"]
        },
        "Time-aware reasoning and context analysis"
    )
    
    # Test 11: maestro_error_handler
    test_tool(
        "maestro_error_handler",
        {
            "error_context": "Python script fails with 'ModuleNotFoundError: No module named numpy'",
            "error_details": {"language": "python", "environment": "local", "os": "windows"},
            "recovery_preferences": ["quick_fix", "automated_solution"]
        },
        "Intelligent error analysis and recovery"
    )
    
    print(f"\n{'='*60}")
    print("🏁 Complete Tool Testing Finished!")
    print("All 11 TanukiMCP Maestro tools have been tested")
    print("=" * 60)

if __name__ == "__main__":
    main() 