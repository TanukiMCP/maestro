#!/usr/bin/env python3
"""
Maestro MCP Server - stdio transport implementation
Uses Model Context Protocol (MCP) FastMCP to expose tools via stdio.
"""

import os
import sys

# Ensure src directory is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Import existing handler functions and utilities
from main import (
    handle_maestro_orchestrate,
    handle_maestro_iae_discovery,
    handle_maestro_tool_selection,
    handle_maestro_iae,
    handle_maestro_search,
    handle_maestro_scrape,
    handle_maestro_execute,
    handle_maestro_error_handler,
    handle_maestro_temporal_context,
    get_computational_tools_instance
)

# Create FastMCP server
mcp = FastMCP("Maestro")

# Register tools
@mcp.tool(name="maestro_orchestrate", description="🎭 Intelligent workflow orchestration for complex tasks")
async def maestro_orchestrate(task_description: str, context: dict = None, success_criteria: dict = None, complexity_level: str = "moderate"):
    """Wrapper for handle_maestro_orchestrate"""
    results = await handle_maestro_orchestrate(
        task_description=task_description,
        context=context,
        success_criteria=success_criteria,
        complexity_level=complexity_level
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_iae_discovery", description="🔍 Integrated Analysis Engine discovery for optimal computation selection")
async def maestro_iae_discovery(task_type: str = "general", domain_context: str = "", complexity_requirements: dict = None):
    """Wrapper for handle_maestro_iae_discovery"""
    results = await handle_maestro_iae_discovery(
        task_type=task_type,
        domain_context=domain_context,
        complexity_requirements=complexity_requirements
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_tool_selection", description="🧰 Intelligent tool selection based on task requirements")
async def maestro_tool_selection(request_description: str, available_context: dict = None, precision_requirements: dict = None):
    """Wrapper for handle_maestro_tool_selection"""
    results = await handle_maestro_tool_selection(
        request_description=request_description,
        available_context=available_context,
        precision_requirements=precision_requirements
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_iae", description="🧠 Intelligence Amplification Engine for specialized computational tasks")
async def maestro_iae(engine_domain: str, computation_type: str, parameters: dict, precision_requirements: str = "machine_precision", validation_level: str = "standard"):
    """Wrapper for handle_maestro_iae"""
    results = await handle_maestro_iae(
        engine_domain=engine_domain,
        computation_type=computation_type,
        parameters=parameters,
        precision_requirements=precision_requirements,
        validation_level=validation_level
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_search", description="🔎 Enhanced search capabilities across multiple sources")
async def maestro_search(arguments: dict):
    """Wrapper for handle_maestro_search"""
    results = await handle_maestro_search(arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_scrape", description="📑 Web scraping functionality with content extraction")
async def maestro_scrape(arguments: dict):
    """Wrapper for handle_maestro_scrape"""
    results = await handle_maestro_scrape(arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_execute", description="⚙️ Command execution with security controls")
async def maestro_execute(arguments: dict):
    """Wrapper for handle_maestro_execute"""
    results = await handle_maestro_execute(arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_error_handler", description="🚨 Advanced error handling and recovery")
async def maestro_error_handler(arguments: dict):
    """Wrapper for handle_maestro_error_handler"""
    results = await handle_maestro_error_handler(arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_temporal_context", description="🕐 Temporal context analysis for time-sensitive information")
async def maestro_temporal_context(arguments: dict):
    """Wrapper for handle_maestro_temporal_context"""
    results = await handle_maestro_temporal_context(arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="get_available_engines", description="📊 Get available computational engines and capabilities")
def get_available_engines():
    """Expose computational tools from ComputationalTools"""
    comp_tools = get_computational_tools_instance()
    return comp_tools.get_available_engines()

if __name__ == "__main__":
    mcp.run() 