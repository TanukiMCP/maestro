#!/usr/bin/env python3
"""
Test using ClientSession to verify our MCP server works
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

async def test_client_session():
    """Test using ClientSession"""
    try:
        # Server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_stdio_server.py"]
        )
        
        logger.info("🚀 Starting ClientSession test...")
        
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
                
                # Test calling a tool
                if tools.tools:
                    tool_name = "echo"
                    logger.info(f"🧪 Testing tool: {tool_name}")
                    
                    result = await session.call_tool(tool_name, {"message": "Hello from ClientSession!"})
                    logger.info(f"📤 Tool result: {result}")
                
                logger.info("🎉 ClientSession test PASSED")
                return True
                
    except Exception as e:
        logger.error(f"❌ Error in ClientSession test: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_client_session())
    if success:
        print("✅ ClientSession test PASSED")
        sys.exit(0)
    else:
        print("❌ ClientSession test FAILED")
        sys.exit(1) 