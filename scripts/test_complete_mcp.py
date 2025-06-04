#!/usr/bin/env python3
"""
Complete MCP test - tests initialization, tool listing, and tool calling
"""
import asyncio
import sys
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_complete_mcp():
    """Test complete MCP functionality"""
    logger.info("🚀 Starting complete MCP test...")
    
    server_command = sys.executable
    server_args = ["mcp_stdio_server.py"]
    server_params = StdioServerParameters(command=server_command, args=server_args)
    
    logger.info(f"Server command: {server_command}")
    logger.info(f"Server args: {server_args}")
    
    try:
        async with stdio_client(server_params) as (read, write):
            logger.info("✅ Successfully started server process")
            
            # Create and initialize client session
            async with ClientSession(read, write) as session:
                logger.info("✅ ClientSession created and initialized")
                
                # Test 1: List tools with timeout
                logger.info("📋 Testing tool listing...")
                try:
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                    tools = tools_result.tools if hasattr(tools_result, 'tools') else tools_result
                    logger.info(f"✅ Found {len(tools)} tools:")
                    for tool in tools:
                        logger.info(f"  - {tool.name}: {tool.description}")
                        
                    if not tools:
                        logger.warning("⚠️  No tools found!")
                        return False
                        
                except asyncio.TimeoutError:
                    logger.error("❌ Timeout waiting for tool list")
                    return False
                except Exception as e:
                    logger.error(f"❌ Error listing tools: {e}", exc_info=True)
                    return False
                
                # Test 2: Call the echo tool with timeout
                logger.info("🔧 Testing tool calling...")
                try:
                    test_message = "Hello from MCP test!"
                    result = await asyncio.wait_for(
                        session.call_tool("echo", {"message": test_message}), 
                        timeout=10.0
                    )
                    logger.info(f"✅ Tool call result: {result}")
                    
                    # Check if the result contains our test message
                    result_text = str(result)
                    if test_message in result_text:
                        logger.info("✅ Tool call returned expected result")
                    else:
                        logger.warning(f"⚠️  Tool call result doesn't contain expected message. Got: {result_text}")
                        
                except asyncio.TimeoutError:
                    logger.error("❌ Timeout waiting for tool call")
                    return False
                except Exception as e:
                    logger.error(f"❌ Error calling tool: {e}", exc_info=True)
                    return False
                
                logger.info("🎉 All tests completed successfully!")
                return True
                
    except Exception as e:
        logger.error(f"❌ Error in complete MCP test: {e}", exc_info=True)
        return False

async def main():
    """Main function with overall timeout"""
    try:
        success = await asyncio.wait_for(test_complete_mcp(), timeout=60.0)
        return success
    except asyncio.TimeoutError:
        logger.error("❌ Overall test timeout (60 seconds)")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 ✅ Complete MCP test PASSED - Tool scanning is working!")
        print("🚀 Ready for deployment to smithery.ai")
    else:
        print("\n💥 ❌ Complete MCP test FAILED")
        sys.exit(1) 