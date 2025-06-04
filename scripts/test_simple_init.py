#!/usr/bin/env python3
"""
Simple test to verify MCP server initialization works
"""

import asyncio
import logging
import sys
import subprocess
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from mcp.types import InitializeRequest, ClientCapabilities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_simple_init():
    """Test basic server initialization"""
    try:
        # Server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_stdio_server.py"]
        )
        
        logger.info("🚀 Starting simple initialization test...")
        
        async with stdio_client(server_params) as (read, write):
            logger.info("📡 Client connected, sending initialize request...")
            
            # Send initialize request
            init_request = InitializeRequest(
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": ClientCapabilities(),
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            )
            
            await write.send(init_request)
            logger.info("📤 Initialize request sent")
            
            # Wait for response
            logger.info("⏳ Waiting for initialize response...")
            init_response = await asyncio.wait_for(read.receive(), timeout=10.0)
            logger.info(f"📥 Received response: {type(init_response).__name__}")
            
            if hasattr(init_response, 'result'):
                logger.info(f"✅ Server capabilities: {init_response.result.capabilities}")
                logger.info("🎉 Simple initialization test PASSED")
                return True
            else:
                logger.error(f"❌ Unexpected response type: {init_response}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error in simple init test: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_init())
    if success:
        print("✅ Simple initialization test PASSED")
        sys.exit(0)
    else:
        print("❌ Simple initialization test FAILED")
        sys.exit(1) 