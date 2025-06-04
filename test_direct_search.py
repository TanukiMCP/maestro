#!/usr/bin/env python3
# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Test script for direct enhanced tools search
"""

import sys
sys.path.append('src')

import asyncio
from maestro.enhanced_tools import EnhancedToolHandlers

async def test_direct_search():
    """Test the enhanced tools search handler directly"""
    print("Testing enhanced tools search handler...")
    
    enhanced_tools = EnhancedToolHandlers()
    
    try:
        arguments = {
            'query': 'python programming tutorials',
            'max_results': 3,
            'search_engine': 'duckduckgo',
            'temporal_filter': 'any',
            'result_format': 'structured'
        }
        
        result = await enhanced_tools.handle_maestro_search(arguments)
        
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        
        if result:
            print(f"First result text (first 500 chars):")
            print(result[0].text[:500])
            print("...")
        else:
            print("No results returned")
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_search()) 
