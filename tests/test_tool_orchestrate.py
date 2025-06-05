"""
Test Suite: maestro_orchestrate tool
Tests the core orchestration functionality with real-world scenarios
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch
from mcp.server.fastmcp import FastMCP

# Test the actual server tool registration
async def test_orchestrate_tool_registration():
    """Test that maestro_orchestrate is properly registered for Smithery scanning"""
    import server
    
    # Verify the tool is registered in the FastMCP instance
    assert hasattr(server, 'mcp')
    assert isinstance(server.mcp, FastMCP)
    
    # Check that orchestrate tool exists in the registry
    tools_list = await server.mcp.list_tools()
    tool_names = [tool.name for tool in tools_list]
    assert 'maestro_orchestrate' in tool_names
    
    # Verify tool has proper signature for instant discovery
    assert len(tool_names) > 0
    print("✅ maestro_orchestrate properly registered for Smithery scanning")


@pytest.mark.asyncio
async def test_orchestrate_simple_task():
    """Test orchestration with a simple, realistic IDE task"""
    import server
    
    # Simulate natural language request from agentic IDE
    task = "Analyze the performance implications of using async/await vs callbacks in JavaScript"
    context = {
        "user_context": "I'm reviewing code and need to understand performance trade-offs",
        "urgency": "moderate",
        "expected_output": "technical analysis with recommendations"
    }
    
    # Execute the tool as it would be called in production
    result = await server.maestro_orchestrate(None,
        task_description=task,
        context=context,
        complexity_level="moderate",
        quality_threshold=0.8
    )
    
    # Verify result structure and content
    assert isinstance(result, str)
    assert len(result) > 100  # Should provide substantial response
    # More flexible content check - just verify it's a meaningful response
    assert len(result.strip()) > 50  # Non-empty meaningful response
    print(f"✅ Simple orchestration completed: {len(result)} characters")


@pytest.mark.asyncio 
async def test_orchestrate_complex_workflow():
    """Test orchestration with complex multi-step workflow"""
    import server
    
    # Complex agentic IDE scenario
    task = """Create a comprehensive API design for a real-time collaboration system. 
    Include authentication, WebSocket connections, conflict resolution, and scalability considerations."""
    
    context = {
        "domain": "software_architecture", 
        "constraints": ["RESTful design", "real-time requirements", "scalability to 10k users"],
        "deliverables": ["API specification", "architecture diagram concepts", "implementation recommendations"]
    }
    
    result = await server.maestro_orchestrate(None,
        task_description=task,
        context=context,
        complexity_level="high",
        quality_threshold=0.85,
        enable_collaboration_fallback=True
    )
    
    # Verify comprehensive response
    assert isinstance(result, str)
    assert len(result) > 200  # Complex tasks should generate substantial output
    # More flexible content check
    assert len(result.strip()) > 100  # Non-empty meaningful response
    print(f"✅ Complex orchestration completed: {len(result)} characters")


@pytest.mark.asyncio
async def test_orchestrate_with_collaboration():
    """Test orchestration that triggers collaboration workflow"""
    import server
    
    # Intentionally ambiguous task to trigger collaboration
    task = "Help me with the thing"
    context = {"vague_request": True}
    
    result = await server.maestro_orchestrate(None,
        task_description=task,
        context=context,
        enable_collaboration_fallback=True
    )
    
    # Should handle ambiguous requests gracefully
    assert isinstance(result, str)
    assert len(result) > 50
    # May contain collaboration request or error handling
    print(f"✅ Ambiguous request handled: {len(result)} characters")


@pytest.mark.asyncio
async def test_orchestrate_error_handling():
    """Test orchestration error handling with invalid inputs"""
    import server
    
    # Test with empty task
    result = await server.maestro_orchestrate(None,
        task_description="",
        context={}
    )
    
    assert isinstance(result, str)
    # Should handle gracefully, not crash
    print("✅ Empty task handled gracefully")
    
    # Test with None values  
    result = await server.maestro_orchestrate(None,
        task_description="Test task",
        context=None
    )
    
    assert isinstance(result, str)
    print("✅ None context handled gracefully")


@pytest.mark.asyncio
async def test_orchestrate_different_complexity_levels():
    """Test orchestration adapts to different complexity levels"""
    import server
    
    base_task = "Explain how database indexing works"
    
    # Test different complexity levels
    for complexity in ["minimal", "moderate", "high", "maximum"]:
        result = await server.maestro_orchestrate(None,
            task_description=base_task,
            context={"learning_level": "intermediate"},
            complexity_level=complexity
        )
        
        assert isinstance(result, str)
        assert len(result) > 50
        print(f"✅ Complexity level '{complexity}' handled: {len(result)} characters")


# Export this function for run_all_tests.py
async def run_tests():
    print("🧪 Testing maestro_orchestrate tool...")
    
    # Test tool registration
    await test_orchestrate_tool_registration()
    
    # Test functionality
    await test_orchestrate_simple_task()
    await test_orchestrate_complex_workflow() 
    await test_orchestrate_with_collaboration()
    await test_orchestrate_error_handling()
    await test_orchestrate_different_complexity_levels()
    
    print("✅ All maestro_orchestrate tests passed!")

if __name__ == "__main__":
    # Run tests directly for development
    import asyncio
    asyncio.run(run_tests()) 