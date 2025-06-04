#!/usr/bin/env python3
"""
Raw stdio test for MCP server communication
"""
import asyncio
import json
import sys
import logging
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from mcp.shared.message import SessionMessage
import mcp.types as types

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_raw_communication():
    """Test raw stdio communication with the MCP server"""
    logger.debug("Starting raw stdio communication test...")
    
    server_command = sys.executable
    server_args = ["mcp_stdio_server.py"]
    server_params = StdioServerParameters(command=server_command, args=server_args)
    
    logger.debug(f"Server command: {server_command}")
    logger.debug(f"Server args: {server_args}")
    
    try:
        async with stdio_client(server_params) as (read, write):
            logger.debug("Successfully started server process. Testing raw communication...")
            
            # Send a simple initialize request manually
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
            
            logger.debug(f"Sending initialize request: {init_request}")
            
            # Create SessionMessage and send it
            session_message = SessionMessage(init_request)
            await write.send(session_message)
            logger.debug("Initialize request sent, waiting for response...")
            
            # Try to read response with timeout
            try:
                response = await asyncio.wait_for(read.receive(), timeout=5.0)
                logger.debug(f"Received response: {response}")
                
                # Check if it's a SessionMessage or Exception
                if isinstance(response, Exception):
                    logger.error(f"Received exception: {response}")
                    return False
                elif isinstance(response, SessionMessage):
                    logger.debug(f"Parsed response message: {response.message}")
                    return True
                else:
                    logger.error(f"Unexpected response type: {type(response)}")
                    return False
                    
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for initialize response")
                return False
                
    except Exception as e:
        logger.error(f"Error in raw communication test: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_raw_communication())
    if success:
        print("✅ Raw stdio communication test PASSED")
    else:
        print("❌ Raw stdio communication test FAILED") 