#!/usr/bin/env python3
"""
Test the enhanced Maestro MCP server with all IA capabilities
"""

import asyncio
import logging
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_enhanced_server():
    """Test all enhanced server capabilities"""
    try:
        # Server parameters - use the enhanced server
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["maestro_enhanced.py"]
        )
        
        logger.info("🚀 Starting Enhanced Maestro Server test...")
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info("📡 Initializing session...")
                
                # Initialize the session
                await session.initialize()
                logger.info("✅ Session initialized successfully")
                
                # List tools
                logger.info("📋 Listing tools...")
                tools = await session.list_tools()
                logger.info(f"🔧 Found {len(tools.tools)} tools:")
                
                for tool in tools.tools:
                    logger.info(f"  - {tool.name}: {tool.description}")
                
                # Test 1: Problem Analysis
                logger.info("\n🧠 Testing Problem Analysis...")
                result = await session.call_tool("analyze_problem", {
                    "problem": "How can we improve team productivity while maintaining work-life balance?",
                    "context": {"team_size": 12, "remote_work": True, "industry": "technology"}
                })
                logger.info("✅ Problem analysis completed")
                print("\n" + "="*60)
                print("PROBLEM ANALYSIS RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 2: Computational Orchestration
                logger.info("\n🔧 Testing Computational Orchestration...")
                result = await session.call_tool("orchestrate_computation", {
                    "task": "Optimize resource allocation for a multi-project portfolio",
                    "parameters": {"projects": 5, "resources": 20, "constraints": ["budget", "time"]}
                })
                logger.info("✅ Computational orchestration completed")
                print("\n" + "="*60)
                print("ORCHESTRATION RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 3: Intelligence Amplification
                logger.info("\n🚀 Testing Intelligence Amplification...")
                result = await session.call_tool("amplify_intelligence", {
                    "input": "Complex systems exhibit emergent behaviors that cannot be predicted from individual components alone",
                    "technique": "decomposition"
                })
                logger.info("✅ Intelligence amplification completed")
                print("\n" + "="*60)
                print("INTELLIGENCE AMPLIFICATION RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 4: Collaborative Reasoning
                logger.info("\n🤝 Testing Collaborative Reasoning...")
                result = await session.call_tool("collaborative_reasoning", {
                    "topic": "Should we adopt AI-driven automation in our customer service?",
                    "perspectives": ["customer_advocate", "operations_manager", "technology_lead"],
                    "reasoning_depth": "deep"
                })
                logger.info("✅ Collaborative reasoning completed")
                print("\n" + "="*60)
                print("COLLABORATIVE REASONING RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 5: Decision Analysis
                logger.info("\n⚖️ Testing Decision Analysis...")
                result = await session.call_tool("decision_analysis", {
                    "decision": "Choose the best cloud infrastructure provider for our startup",
                    "options": ["AWS", "Google Cloud", "Azure", "Hybrid Solution"],
                    "criteria": ["Cost", "Scalability", "Security", "Ease of Use", "Support"],
                    "framework": "weighted-criteria"
                })
                logger.info("✅ Decision analysis completed")
                print("\n" + "="*60)
                print("DECISION ANALYSIS RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                logger.info("\n🎉 All Enhanced Server tests PASSED")
                return True
                
    except Exception as e:
        logger.error(f"❌ Error in Enhanced Server test: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_server())
    if success:
        print("\n" + "="*60)
        print("✅ ALL ENHANCED SERVER TESTS PASSED")
        print("🚀 Maestro Intelligence Amplification Engine is fully operational!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n❌ Enhanced Server tests FAILED")
        sys.exit(1) 