#!/usr/bin/env python3
"""
Raw tools/list test - tests the tools/list method directly
"""
import asyncio
import sys
import logging
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from mcp.shared.message import SessionMessage
import mcp.types as types

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_raw_tools_list():
    """Test raw tools/list communication with the MCP server"""
    logger.info("🚀 Starting raw tools/list test...")
    
    server_command = sys.executable
    server_args = ["mcp_stdio_server.py"]  # Back to original server
    server_params = StdioServerParameters(command=server_command, args=server_args)
    
    try:
        async with stdio_client(server_params) as (read, write):
            logger.info("✅ Successfully started server process")
            
            # Step 1: Initialize
            logger.info("📡 Sending initialize request...")
            init_request = types.JSONRPCMessage(
                jsonrpc="2.0",
                id=1,
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            )
            
            session_message = SessionMessage(init_request)
            await write.send(session_message)
            
            # Read initialize response
            init_response = await asyncio.wait_for(read.receive(), timeout=5.0)
            if isinstance(init_response, Exception):
                logger.error(f"❌ Initialize failed: {init_response}")
                return False
            logger.info("✅ Initialize successful")
            
            # Step 2: Send tools/list request
            logger.info("📋 Sending tools/list request...")
            tools_request = types.JSONRPCMessage(
                jsonrpc="2.0",
                id=2,
                method="tools/list",
                params={}
            )
            
            tools_session_message = SessionMessage(tools_request)
            await write.send(tools_session_message)
            logger.info("📤 tools/list request sent, waiting for response...")
            
            # Read tools/list response with timeout
            try:
                tools_response = await asyncio.wait_for(read.receive(), timeout=10.0)
                logger.info(f"📥 Received tools response: {tools_response}")
                
                if isinstance(tools_response, Exception):
                    logger.error(f"❌ Tools list failed: {tools_response}")
                    return False
                elif isinstance(tools_response, SessionMessage):
                    response_msg = tools_response.message
                    logger.info(f"✅ Tools response message: {response_msg}")
                    
                    # Check if it's a successful response
                    if hasattr(response_msg, 'root') and hasattr(response_msg.root, 'result'):
                        result = response_msg.root.result
                        logger.info(f"🔧 Tools result: {result}")
                        
                        if 'tools' in result:
                            tools = result['tools']
                            logger.info(f"✅ Found {len(tools)} tools:")
                            for tool in tools:
                                logger.info(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
                            return True
                        else:
                            logger.warning("⚠️  No 'tools' key in result")
                            return False
                    else:
                        logger.error(f"❌ Unexpected response format: {response_msg}")
                        return False
                else:
                    logger.error(f"❌ Unexpected response type: {type(tools_response)}")
                    return False
                    
            except asyncio.TimeoutError:
                logger.error("❌ Timeout waiting for tools/list response")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error in raw tools/list test: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_raw_tools_list())
    if success:
        print("\n🎉 ✅ Raw tools/list test PASSED!")
        print("🔧 Server correctly responds to tools/list requests")
    else:
        print("\n💥 ❌ Raw tools/list test FAILED")
        sys.exit(1) 