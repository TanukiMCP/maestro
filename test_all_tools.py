#!/usr/bin/env python3
"""
Comprehensive test script to verify all 11 Maestro tools work correctly
and produce real responses (not placeholders, mock logic, or hardcoded responses).
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

async def call_tool(session: aiohttp.ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call a tool via the MCP HTTP interface."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        async with session.post("http://localhost:8000/mcp", json=payload, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                text = await response.text()
                return {"error": f"HTTP {response.status}: {text}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

async def test_all_tools():
    """Test all 11 Maestro tools with real requests."""
    print("🧪 Testing All 11 Maestro Tools")
    print("=" * 60)
    
    # Define test cases for each tool
    test_cases = [
        {
            "name": "maestro_orchestrate",
            "description": "Complex task orchestration",
            "arguments": {
                "task_description": "Analyze the benefits and drawbacks of renewable energy sources",
                "complexity_level": "moderate"
            }
        },
        {
            "name": "maestro_iae_discovery", 
            "description": "Engine discovery",
            "arguments": {
                "task_type": "mathematical_analysis",
                "domain_context": "statistical computation"
            }
        },
        {
            "name": "maestro_tool_selection",
            "description": "Tool selection",
            "arguments": {
                "request_description": "I need to analyze financial data and create visualizations"
            }
        },
        {
            "name": "maestro_iae",
            "description": "Computational analysis",
            "arguments": {
                "analysis_request": "Calculate the mean and standard deviation of the dataset [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "engine_type": "mathematical"
            }
        },
        {
            "name": "maestro_search",
            "description": "Enhanced search",
            "arguments": {
                "query": "latest developments in artificial intelligence 2024",
                "max_results": 3
            }
        },
        {
            "name": "maestro_scrape",
            "description": "Web scraping",
            "arguments": {
                "url": "https://httpbin.org/json",
                "output_format": "json"
            }
        },
        {
            "name": "maestro_execute",
            "description": "Code execution",
            "arguments": {
                "command": "print('Hello from Maestro!'); result = 2 + 2; print(f'2 + 2 = {result}')"
            }
        },
        {
            "name": "maestro_error_handler",
            "description": "Error analysis",
            "arguments": {
                "error_message": "ValueError: invalid literal for int() with base 10: 'abc'"
            }
        },
        {
            "name": "maestro_temporal_context",
            "description": "Temporal reasoning",
            "arguments": {
                "temporal_query": "What major technological events happened in 2023?"
            }
        },
        {
            "name": "get_available_engines",
            "description": "List engines",
            "arguments": {
                "engine_type": "all"
            }
        },
        {
            "name": "maestro_collaboration_response",
            "description": "Collaboration response",
            "arguments": {
                "collaboration_id": "test-123",
                "responses": {"question1": "Yes, proceed with the analysis"}
            }
        }
    ]
    
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            tool_name = test_case["name"]
            description = test_case["description"]
            arguments = test_case["arguments"]
            
            print(f"\n{i:2d}. Testing {tool_name}")
            print(f"    Description: {description}")
            print(f"    Arguments: {json.dumps(arguments, indent=6)}")
            
            start_time = time.time()
            result = await call_tool(session, tool_name, arguments)
            duration = (time.time() - start_time) * 1000
            
            # Analyze the result
            if "error" in result:
                print(f"    ❌ FAILED ({duration:.1f}ms): {result['error']}")
                results.append({"tool": tool_name, "status": "FAILED", "error": result["error"]})
            elif "result" in result:
                response_content = result["result"]
                
                # Check if it's a real response or placeholder
                is_real_response = analyze_response_quality(response_content, tool_name)
                
                if is_real_response:
                    print(f"    ✅ SUCCESS ({duration:.1f}ms): Real response received")
                    print(f"    📄 Response preview: {str(response_content)[:100]}...")
                    results.append({"tool": tool_name, "status": "SUCCESS", "response_length": len(str(response_content))})
                else:
                    print(f"    ⚠️  PLACEHOLDER ({duration:.1f}ms): Mock/placeholder response detected")
                    print(f"    📄 Response: {str(response_content)[:100]}...")
                    results.append({"tool": tool_name, "status": "PLACEHOLDER", "response": str(response_content)[:200]})
            else:
                print(f"    ❓ UNKNOWN ({duration:.1f}ms): Unexpected response format")
                print(f"    📄 Raw response: {json.dumps(result, indent=6)}")
                results.append({"tool": tool_name, "status": "UNKNOWN", "response": str(result)})
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    placeholder_count = sum(1 for r in results if r["status"] == "PLACEHOLDER")
    unknown_count = sum(1 for r in results if r["status"] == "UNKNOWN")
    
    print(f"✅ Real working tools: {success_count}/11")
    print(f"⚠️  Placeholder/mock tools: {placeholder_count}/11")
    print(f"❌ Failed tools: {failed_count}/11")
    print(f"❓ Unknown status: {unknown_count}/11")
    
    if success_count == 11:
        print("\n🎉 ALL TOOLS ARE WORKING WITH REAL RESPONSES!")
    elif success_count > 0:
        print(f"\n⚠️  Only {success_count} out of 11 tools are producing real responses.")
    else:
        print("\n❌ NO TOOLS ARE PRODUCING REAL RESPONSES!")
    
    return results

def analyze_response_quality(response: Any, tool_name: str) -> bool:
    """Analyze if a response is real or a placeholder/mock."""
    response_str = str(response).lower()
    
    # Common placeholder/mock indicators (but exclude legitimate terms)
    placeholder_indicators = [
        "placeholder", "mock", "dummy", "fake", "test data",
        "not implemented", "coming soon", "todo", "fixme",
        "lorem ipsum",
        "this is a simulation", "this would", "would return",
        "hardcoded", "static response"
    ]
    
    # Check for placeholder indicators (but be more specific)
    for indicator in placeholder_indicators:
        if indicator in response_str:
            return False
    
    # Tool-specific checks for real responses
    if tool_name == "maestro_search":
        # Real search should have URLs, titles, and search metadata
        if ("url:" in response_str or "http" in response_str) and \
           ("title" in response_str or "snippet" in response_str) and \
           len(response_str) > 200:
            return True
        return False
    
    if tool_name == "maestro_scrape":
        # Real scrape should have content and metadata
        if ("content:" in response_str or "scraped at:" in response_str) and \
           ("url:" in response_str or "http" in response_str) and \
           len(response_str) > 200:
            return True
        return False
    
    if tool_name == "maestro_iae" and "mean" in response_str and "standard deviation" in response_str:
        # Check if it contains actual calculations
        if any(char.isdigit() for char in response_str):
            return True
    
    if tool_name == "maestro_execute" and "hello from maestro" in response_str:
        # Code execution should show actual output
        return True
    
    # General checks for real content
    if len(response_str) > 50 and not any(indicator in response_str for indicator in placeholder_indicators):
        return True
    
    return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive tool testing...")
    try:
        results = asyncio.run(test_all_tools())
    except KeyboardInterrupt:
        print("\n⛔ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}") 