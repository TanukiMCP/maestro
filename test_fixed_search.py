#!/usr/bin/env python3
# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Test script for the fixed search functionality
"""

import sys
sys.path.append('src')

import asyncio
from maestro.llm_web_tools import LLMWebTools

async def test_search():
    """Test the fixed search functionality"""
    print("Testing fixed search functionality...")
    
    llm_tools = LLMWebTools()
    
    try:
        result = await llm_tools.llm_driven_search(
            query="artificial intelligence",
            max_results=3,
            engines=['duckduckgo'],
            temporal_filter=None,
            result_format='structured',
            llm_analysis=False,  # Set to False since we don't have LLM context
            context=None
        )
        
        print(f"Search success: {result.get('success', False)}")
        print(f"Total results: {result.get('total_results', 0)}")
        
        if result.get('success'):
            print("\nResults:")
            for i, search_result in enumerate(result.get('results', []), 1):
                print(f"\n{i}. {search_result['title']}")
                print(f"   URL: {search_result['url']}")
                print(f"   Domain: {search_result['domain']}")
                print(f"   Snippet: {search_result['snippet'][:100]}...")
                print(f"   Relevance: {search_result['relevance_score']:.2f}")
        else:
            print(f"Search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search()) 
