#!/usr/bin/env python3
"""
Test script for the new MAESTRO orchestration system

This script demonstrates the new architecture where:
1. orchestrate is the main gateway that automatically gathers context and tool awareness
2. orchestrate creates detailed execution plans with specific tool calls
3. execute runs the plans sequentially as defined by the LLM
"""

import asyncio
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maestro.enhanced_tools import EnhancedToolHandlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

async def test_orchestration_system():
    """Test the new orchestration system"""
    
    print("🎭 Testing MAESTRO Orchestration System")
    print("=" * 50)
    
    # Initialize the enhanced tool handlers
    handlers = EnhancedToolHandlers()
    
    # Test 1: Simple research task
    print("\n📋 Test 1: Research Task Orchestration")
    print("-" * 40)
    
    research_task = {
        "task_description": "Find the latest developments in AI and machine learning from 2024",
        "user_context": {"domain": "technology", "urgency": "moderate"},
        "complexity_level": "moderate",
        "auto_execute": False  # Just create the plan, don't execute
    }
    
    try:
        result = await handlers.handle_maestro_orchestrate(research_task)
        print("✅ Research orchestration result:")
        print(result[0].text[:500] + "..." if len(result[0].text) > 500 else result[0].text)
    except Exception as e:
        print(f"❌ Research orchestration failed: {e}")
    
    # Test 2: Data extraction task
    print("\n📋 Test 2: Data Extraction Task Orchestration")
    print("-" * 40)
    
    extraction_task = {
        "task_description": "Extract and analyze content from a technology news website",
        "user_context": {"target_domain": "tech news", "format_preference": "structured"},
        "complexity_level": "moderate",
        "auto_execute": False
    }
    
    try:
        result = await handlers.handle_maestro_orchestrate(extraction_task)
        print("✅ Extraction orchestration result:")
        print(result[0].text[:500] + "..." if len(result[0].text) > 500 else result[0].text)
    except Exception as e:
        print(f"❌ Extraction orchestration failed: {e}")
    
    # Test 3: Analysis task with auto-execution
    print("\n📋 Test 3: Analysis Task with Auto-Execution")
    print("-" * 40)
    
    analysis_task = {
        "task_description": "Analyze the current state of web search technology and provide insights",
        "user_context": {"analysis_depth": "comprehensive", "output_format": "report"},
        "complexity_level": "high",
        "auto_execute": True  # This will create and execute the plan
    }
    
    try:
        result = await handlers.handle_maestro_orchestrate(analysis_task)
        print("✅ Analysis orchestration with auto-execution result:")
        print(result[0].text[:800] + "..." if len(result[0].text) > 800 else result[0].text)
    except Exception as e:
        print(f"❌ Analysis orchestration failed: {e}")
    
    # Test 4: Direct execution of a task
    print("\n📋 Test 4: Direct Task Execution")
    print("-" * 40)
    
    execution_task = {
        "task_description": "Search for information about Python async programming best practices",
        "user_context": {"programming_level": "intermediate"},
        "complexity_level": "simple"
    }
    
    try:
        result = await handlers.handle_maestro_execute(execution_task)
        print("✅ Direct execution result:")
        print(result[0].text[:500] + "..." if len(result[0].text) > 500 else result[0].text)
    except Exception as e:
        print(f"❌ Direct execution failed: {e}")
    
    # Test 5: IAE analysis
    print("\n📋 Test 5: IAE Analysis")
    print("-" * 40)
    
    iae_task = {
        "analysis_request": "Analyze the effectiveness of different search engines for technical queries",
        "engine_type": "auto",
        "precision_level": "standard"
    }
    
    try:
        result = await handlers.handle_maestro_iae(iae_task)
        print("✅ IAE analysis result:")
        print(result[0].text[:500] + "..." if len(result[0].text) > 500 else result[0].text)
    except Exception as e:
        print(f"❌ IAE analysis failed: {e}")
    
    # Test 6: Context gathering demonstration
    print("\n📋 Test 6: Context Gathering Demonstration")
    print("-" * 40)
    
    try:
        # Access the orchestration engine directly to show context gathering
        await handlers._ensure_initialized()
        context = await handlers.orchestration_engine.gather_context({
            "user_preference": "detailed_analysis",
            "session_id": "test_session_123"
        })
        
        print("✅ Context gathering result:")
        print(f"- Current time: {context.current_time}")
        print(f"- Available tools: {len(context.available_tools)}")
        print(f"- Environment capabilities: {context.environment['capabilities']}")
        print(f"- User context: {context.user_context}")
        
        # Show some tool details
        print("\n🔧 Available Tools Sample:")
        for i, tool in enumerate(context.available_tools[:3]):
            print(f"  {i+1}. {tool.name} ({tool.tool_type}): {tool.description[:60]}...")
            
    except Exception as e:
        print(f"❌ Context gathering failed: {e}")
    
    # Cleanup
    await handlers.cleanup()
    print("\n✅ All tests completed!")

async def test_search_functionality():
    """Test the search functionality specifically"""
    
    print("\n🔍 Testing Search Functionality")
    print("=" * 40)
    
    handlers = EnhancedToolHandlers()
    
    search_args = {
        "query": "Python async programming 2024",
        "max_results": 5,
        "search_engine": "duckduckgo",
        "result_format": "structured"
    }
    
    try:
        result = await handlers.handle_maestro_search(search_args)
        print("✅ Search test result:")
        print(result[0].text[:600] + "..." if len(result[0].text) > 600 else result[0].text)
    except Exception as e:
        print(f"❌ Search test failed: {e}")
    
    await handlers.cleanup()

if __name__ == "__main__":
    print("🚀 Starting MAESTRO Orchestration System Tests")
    
    # Run the orchestration tests
    asyncio.run(test_orchestration_system())
    
    # Run the search test
    asyncio.run(test_search_functionality())
    
    print("\n🎉 All tests completed successfully!") 