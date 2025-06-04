#!/usr/bin/env python3
"""
Official MCP Server Implementation for TanukiMCP Maestro
Using the standard MCP Python SDK with proper lazy loading for Smithery compatibility.

Based on successful Smithery deployment patterns from:
- MCP-Atlassian: https://smithery.ai/server/mcp-atlassian  
- Memory Cache Server: https://smithery.ai/server/@ibproduct/ib-mcp-cache-server
"""

import os
import sys
import asyncio
import json
from typing import Any, Dict, List, Optional, Union
import logging

# Only import MCP - keep all heavy imports lazy
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import sse_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    ListToolsResult
)

# Global lazy-loaded instances
_maestro_tools = None
_computational_tools = None
_enhanced_tool_handlers = None

# Logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def get_maestro_tools():
    """Lazy load MaestroTools only when first needed"""
    global _maestro_tools
    if _maestro_tools is None:
        try:
            from src.maestro_tools import MaestroTools
            _maestro_tools = MaestroTools()
            logger.info("MaestroTools initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MaestroTools: {e}")
            _maestro_tools = None
    return _maestro_tools

def get_computational_tools():
    """Lazy load ComputationalTools only when first needed"""
    global _computational_tools
    if _computational_tools is None:
        try:
            from src.computational_tools import ComputationalTools
            _computational_tools = ComputationalTools()
            logger.info("ComputationalTools initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ComputationalTools: {e}")
            _computational_tools = None
    return _computational_tools

def get_enhanced_tool_handlers():
    """Lazy load EnhancedToolHandlers only when first needed"""
    global _enhanced_tool_handlers
    if _enhanced_tool_handlers is None:
        try:
            from src.enhanced_tools import EnhancedToolHandlers
            _enhanced_tool_handlers = EnhancedToolHandlers()
            logger.info("EnhancedToolHandlers initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedToolHandlers: {e}")
            _enhanced_tool_handlers = None
    return _enhanced_tool_handlers

# Create the MCP server
app = Server("tanukimcp-maestro")

# Import static tool definitions (shared with HTTP transport)
from static_tools import STATIC_TOOLS

@app.list_tools()
async def list_tools() -> ListToolsResult:
    """List all available tools - returns instantly without heavy imports"""
    return ListToolsResult(tools=STATIC_TOOLS)

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls with lazy loading"""
    try:
        if name.startswith("maestro_") and name != "maestro_iae":
            # Handle MaestroTools
            tools = get_maestro_tools()
            if tools is None:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text="Error: MaestroTools not available. Please check server configuration."
                    )]
                )
            
            # Route to appropriate handler
            if hasattr(tools, 'handle_tool_call'):
                result = await tools.handle_tool_call(name, arguments)
                if isinstance(result, list):
                    return CallToolResult(content=result)
                else:
                    return CallToolResult(content=[TextContent(type="text", text=str(result))])
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text=f"Error: Tool handler method not found for {name}"
                    )]
                )
        
        elif name == "maestro_iae":
            # Handle ComputationalTools  
            tools = get_computational_tools()
            if tools is None:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text="Error: ComputationalTools not available. Please check server configuration."
                    )]
                )
            
            result = await tools.handle_iae_request(
                arguments.get("analysis_request", ""),
                arguments.get("engine_type", "auto"),
                arguments.get("data"),
                arguments.get("parameters", {})
            )
            
            if isinstance(result, list):
                return CallToolResult(content=result)
            else:
                return CallToolResult(content=[TextContent(type="text", text=str(result))])
        
        elif name == "get_available_engines":
            # Handle engine discovery
            tools = get_computational_tools()
            if tools is None:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text="Error: ComputationalTools not available. Please check server configuration."
                    )]
                )
            
            detailed = arguments.get("detailed", False)
            engines = tools.get_available_engines(detailed)
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(engines, indent=2))]
            )
        
        else:
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Error: Unknown tool '{name}'"
                )]
            )
    
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text", 
                text=f"Error executing {name}: {str(e)}"
            )]
        )

async def main():
    """Main server entry point"""
    # Determine transport method
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    
    if transport == "sse":
        # SSE transport for HTTP/web deployment (Smithery)
        port = int(os.getenv("PORT", 8000))
        logger.info(f"Starting SSE server on port {port}")
        
        from mcp.server.sse import SseServerTransport
        transport_instance = SseServerTransport("/mcp")
        
        async with sse_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    else:
        # Default to stdio transport
        logger.info("Starting stdio server")
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, 
                write_stream, 
                app.create_initialization_options()
            )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1) 