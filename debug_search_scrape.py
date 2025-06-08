#!/usr/bin/env python3
"""
Debug script to test search and scrape functionality directly
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maestro.llm_web_tools import LLMWebTools

async def test_search_scrape():
    """Test search and scrape directly"""
    print("🔍 Testing LLMWebTools directly...")
    
    llm_tools = LLMWebTools()
    
    # Test search
    print("\n1. Testing search...")
    try:
        search_result = await llm_tools.llm_driven_search(
            query="test search",
            max_results=3,
            engines=['duckduckgo'],
            context=None
        )
        print(f"Search success: {search_result.get('success', False)}")
        print(f"Search results: {search_result.get('total_results', 0)}")
        if search_result.get('error'):
            print(f"Search error: {search_result['error']}")
    except Exception as e:
        print(f"Search exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Test scrape
    print("\n2. Testing scrape...")
    try:
        scrape_result = await llm_tools.llm_driven_scrape(
            url="https://httpbin.org/json",
            output_format="json",
            context=None
        )
        print(f"Scrape success: {scrape_result.get('success', False)}")
        if scrape_result.get('error'):
            print(f"Scrape error: {scrape_result['error']}")
        else:
            print(f"Scrape content length: {len(str(scrape_result.get('result', {})))}")
    except Exception as e:
        print(f"Scrape exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search_scrape()) 