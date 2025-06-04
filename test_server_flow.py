#!/usr/bin/env python3
"""
Test script that mimics the server flow exactly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import logging

# Set up logging like the server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

async def test_server_flow():
    """Test the exact server flow"""
    print("Testing server flow...")
    
    # Import like the server does
    from maestro.enhanced_tools import EnhancedToolHandlers
    
    # Create instance like the server does
    enhanced_tools = EnhancedToolHandlers()
    
    # Create arguments like the server does
    arguments = {
        "query": "javascript tutorials",
        "max_results": 2,
        "search_engine": "duckduckgo",
        "temporal_filter": "any",
        "result_format": "structured"
    }
    
    try:
        print(f"Calling handle_maestro_search with arguments: {arguments}")
        
        # Call the handler like the server does
        result = await enhanced_tools.handle_maestro_search(arguments)
        
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        
        if result:
            text = result[0].text
            print(f"Result text length: {len(text)}")
            
            # Check if it contains "Results: 0 found" or actual results
            if "Results: 0 found" in text:
                print("❌ Search returned 0 results")
            elif "Results:" in text and "found" in text:
                print("✅ Search returned results")
            
            # Print first 1000 characters
            print(f"First 1000 characters:")
            print(text[:1000])
        else:
            print("No results returned")
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_server_flow()) 
