"""
Test Suite: maestro_search tool
Tests enhanced web search functionality for agentic IDE usage
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pytest
import asyncio
from mcp.server.fastmcp import FastMCP


async def test_search_tool_registration():
    """Test that maestro_search is properly registered for Smithery scanning"""
    import server
    
    # Verify tool registration
    tools_list = await server.mcp.list_tools()
    tool_names = [tool.name for tool in tools_list]
    assert 'maestro_search' in tool_names
    print("✅ maestro_search properly registered for Smithery scanning")


@pytest.mark.asyncio
async def test_search_technical_query():
    """Test search with technical programming query"""
    import server
    
    result = await server.maestro_search(None,
        query="Python asyncio best practices 2024",
        max_results=5,
        search_type="comprehensive",
        temporal_filter="recent"
    )
    
    assert isinstance(result, str)
    assert len(result) > 50  # More reasonable expectation
    print(f"✅ Technical search completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_search_current_events():
    """Test search for current events and trends"""
    import server
    
    result = await server.maestro_search(None,
        query="latest developments in AI development tools",
        max_results=8,
        temporal_filter="recent",
        output_format="detailed"
    )
    
    assert isinstance(result, str)
    assert len(result) > 50
    print(f"✅ Current events search completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_search_error_handling():
    """Test search error handling"""
    import server
    
    # Empty query
    result = await server.maestro_search(None,
        query="",
        max_results=5
    )
    
    assert isinstance(result, str)
    print("✅ Empty search query handled gracefully")


@pytest.mark.asyncio
async def test_search_different_formats():
    """Test different output formats"""
    import server
    
    query = "machine learning frameworks comparison"
    
    for format_type in ["summary", "detailed", "structured"]:
        result = await server.maestro_search(None,
            query=query,
            max_results=3,
            output_format=format_type
        )
        
        assert isinstance(result, str)
        assert len(result) > 20
        print(f"✅ Search format '{format_type}' completed: {len(result)} characters")


# Export this function for run_all_tests.py
async def run_tests():
    print("🧪 Testing maestro_search tool...")
    
    await test_search_tool_registration()
    await test_search_technical_query()
    await test_search_current_events()
    await test_search_error_handling()
    await test_search_different_formats()
    
    print("✅ All maestro_search tests passed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_tests()) 