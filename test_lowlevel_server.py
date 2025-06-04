#!/usr/bin/env python3
"""
Low-level MCP server test
"""
import asyncio
import logging
import sys
from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    ServerCapabilities,
    Tool,
    TextContent,
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
    """Handle tools/list requests"""
    logger.debug("Handling list_tools request")
    return ListToolsResult(
        tools=[
            Tool(
                name="echo",
                description="Echoes back the input message",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo"
                        }
                    },
                    "required": ["message"]
                }
            )
        ]
    )

async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tools/call requests"""
    logger.debug(f"Handling call_tool request: {request}")
    
    if request.params.name == "echo":
        message = request.params.arguments.get("message", "")
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Echo: {message}"
                )
            ]
        )
    else:
        raise ValueError(f"Unknown tool: {request.params.name}")

async def main():
    """Main server function"""
    logger.debug("Starting low-level MCP server")
    
    async with stdio_server() as (read_stream, write_stream):
        async with ServerSession(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="LowLevelTestServer",
                server_version="1.0.0",
                capabilities=ServerCapabilities(
                    tools={}
                ),
            ),
        ) as session:
            # Register handlers
            session.set_request_handler(ListToolsRequest, handle_list_tools)
            session.set_request_handler(CallToolRequest, handle_call_tool)
            
            logger.debug("Server initialized, waiting for messages...")
            
            # Process messages
            async for message in session.incoming_messages:
                if isinstance(message, Exception):
                    logger.error(f"Error: {message}")
                    continue
                logger.debug(f"Received message: {message}")

if __name__ == "__main__":
    asyncio.run(main()) 