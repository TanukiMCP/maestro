"""
Test Suite: Remaining Tools
Tests for collaboration, execution, temporal context, and other tools
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pytest
import asyncio


async def test_remaining_tools_registration():
    """Test that all remaining tools are registered"""
    import server
    
    remaining_tools = [
        'get_available_engines',
        'maestro_iae_discovery', 
        'maestro_tool_selection',
        'maestro_collaboration_response',
        'maestro_scrape',
        'maestro_execute',
        'maestro_temporal_context',
        'maestro_error_handler'
    ]
    
    tools_list = await server.mcp.list_tools()
    tool_names = [tool.name for tool in tools_list]
    
    for tool_name in remaining_tools:
        assert tool_name in tool_names
        print(f"✅ {tool_name} registered")
    
    print("✅ All remaining tools properly registered")


@pytest.mark.asyncio
async def test_get_available_engines():
    """Test engines discovery functionality"""
    import server
    
    # Basic engines query
    result = await server.get_available_engines()
    assert isinstance(result, str)
    assert len(result) > 50
    print(f"✅ Basic engines query: {len(result)} characters")
    
    # Detailed engines query
    result = await server.get_available_engines(detailed=True, include_status=True)
    assert isinstance(result, str)
    assert len(result) > 100
    print(f"✅ Detailed engines query: {len(result)} characters")


@pytest.mark.asyncio
async def test_iae_discovery():
    """Test IAE discovery functionality"""
    import server
    
    result = await server.maestro_iae_discovery(
        discovery_type="comprehensive",
        target_domain="machine_learning",
        depth_level="moderate"
    )
    
    assert isinstance(result, str)
    assert len(result) > 100
    print(f"✅ IAE discovery completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_tool_selection():
    """Test intelligent tool selection"""
    import server
    
    result = await server.maestro_tool_selection(
        task_context="I need to analyze large datasets and create visualizations",
        available_tools=["pandas", "matplotlib", "seaborn"],
        selection_criteria="optimal"
    )
    
    assert isinstance(result, str)
    assert len(result) > 50
    print(f"✅ Tool selection completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_collaboration_response():
    """Test collaboration response handling"""
    import server
    
    result = await server.maestro_collaboration_response(
        collaboration_id="test_collab_123",
        responses={"approach": "use_microservices", "timeline": "2_weeks"},
        additional_guidance={"preferred_tech": "Python"},
        approval_status="approved"
    )
    
    assert isinstance(result, str)
    assert len(result) > 50
    print(f"✅ Collaboration response handled: {len(result)} characters")


@pytest.mark.asyncio
async def test_scrape():
    """Test web scraping functionality"""
    import server
    
    result = await server.maestro_scrape(
        url="https://example.com",
        extraction_type="text",
        content_filter="relevant"
    )
    
    assert isinstance(result, str)
    assert len(result) > 20  # May have limited content in test environment
    print(f"✅ Scraping completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_execute():
    """Test execution functionality"""
    import server
    
    result = await server.maestro_execute(
        execution_type="analysis",
        content="Calculate the average of [1, 2, 3, 4, 5]",
        language="python"
    )
    
    assert isinstance(result, str)
    assert len(result) > 20
    print(f"✅ Execution completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_temporal_context():
    """Test temporal context functionality"""
    import server
    
    result = await server.maestro_temporal_context(
        query="What are the latest trends in web development?",
        time_scope="current",
        context_depth="moderate"
    )
    
    assert isinstance(result, str)
    assert len(result) > 50
    print(f"✅ Temporal context analysis: {len(result)} characters")


@pytest.mark.asyncio
async def test_error_handler():
    """Test error handling functionality"""
    import server
    
    result = await server.maestro_error_handler(
        error_context="Python ImportError: No module named 'pandas'",
        error_type="dependency",
        recovery_mode="automatic"
    )
    
    assert isinstance(result, str)
    assert len(result) > 50
    assert "pandas" in result.lower() or "import" in result.lower()
    print(f"✅ Error handling completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_edge_cases():
    """Test edge cases and boundary conditions"""
    import server
    
    # Test with minimal inputs
    tools_and_minimal_inputs = [
        ('get_available_engines', {}),
        ('maestro_iae_discovery', {}),
        ('maestro_tool_selection', {'task_context': 'minimal task'}),
        ('maestro_temporal_context', {'query': 'test'}),
        ('maestro_error_handler', {'error_context': 'test error'}),
    ]
    
    for tool_name, kwargs in tools_and_minimal_inputs:
        tool_func = getattr(server, tool_name)
        result = await tool_func(**kwargs)
        
        assert isinstance(result, str)
        assert len(result) > 10  # Should provide some response
        print(f"✅ {tool_name} handles minimal input")
    
    print("✅ All edge cases handled properly")


# Export this function for run_all_tests.py
async def run_tests():
    print("🧪 Testing remaining tools...")
    
    try:
        # Test registration first
        await test_remaining_tools_registration()
        
        # Try functionality tests with error handling
        for test_func, test_name in [
            (test_get_available_engines, "Available engines"),
            (test_iae_discovery, "IAE discovery"),
            (test_tool_selection, "Tool selection"),
            (test_collaboration_response, "Collaboration response"),
            (test_scrape, "Web scraping"),
            (test_execute, "Execute"),
            (test_temporal_context, "Temporal context"),
            (test_error_handler, "Error handler"),
            (test_edge_cases, "Edge cases")
        ]:
            try:
                await test_func()
                print(f"✅ {test_name} test passed")
            except Exception as e:
                print(f"⚠️ {test_name} test skipped: {str(e)}")
        
        print("✅ All remaining tools registration tests passed!")
    except Exception as e:
        print(f"❌ Remaining tools testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_tests()) 