#!/usr/bin/env python3
"""
Maestro MCP Server - stdio transport implementation
Uses Model Context Protocol (MCP) FastMCP to expose tools via stdio.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_debug(msg, *args, **kwargs):
    """Helper to log debug messages with a prefix for easy filtering"""
    logger.debug(f"SMITHERY-DEBUG: {msg}", *args, **kwargs)

# Ensure src directory is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Create FastMCP server
mcp = FastMCP("Maestro")

# Lazy loading of handlers
_handlers = None

def get_handlers():
    global _handlers
    if _handlers is None:
        log_debug("Loading handlers")
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
        _handlers = {
            'orchestrate': handle_maestro_orchestrate,
            'iae_discovery': handle_maestro_iae_discovery,
            'tool_selection': handle_maestro_tool_selection,
            'iae': handle_maestro_iae,
            'search': handle_maestro_search,
            'scrape': handle_maestro_scrape,
            'execute': handle_maestro_execute,
            'error_handler': handle_maestro_error_handler,
            'temporal_context': handle_maestro_temporal_context,
            'get_computational_tools': get_computational_tools_instance
        }
    return _handlers

# Register tools with lazy loading
@mcp.tool(name="maestro_orchestrate", description="🎭 Intelligent workflow orchestration for complex tasks")
async def maestro_orchestrate(task_description: str, context: dict = None, success_criteria: dict = None, complexity_level: str = "moderate"):
    """Wrapper for handle_maestro_orchestrate"""
    handlers = get_handlers()
    results = await handlers['orchestrate'](
        task_description=task_description,
        context=context,
        success_criteria=success_criteria,
        complexity_level=complexity_level
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_iae_discovery", description="🔍 Integrated Analysis Engine discovery for optimal computation selection")
async def maestro_iae_discovery(task_type: str = "general", domain_context: str = "", complexity_requirements: dict = None):
    """Wrapper for handle_maestro_iae_discovery"""
    handlers = get_handlers()
    results = await handlers['iae_discovery'](
        task_type=task_type,
        domain_context=domain_context,
        complexity_requirements=complexity_requirements
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_tool_selection", description="🧰 Intelligent tool selection based on task requirements")
async def maestro_tool_selection(request_description: str, available_context: dict = None, precision_requirements: dict = None):
    """Wrapper for handle_maestro_tool_selection"""
    handlers = get_handlers()
    results = await handlers['tool_selection'](
        request_description=request_description,
        available_context=available_context,
        precision_requirements=precision_requirements
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_iae", description="🧠 Intelligence Amplification Engine for specialized computational tasks")
async def maestro_iae(engine_domain: str, computation_type: str, parameters: dict, precision_requirements: str = "machine_precision", validation_level: str = "standard"):
    """Wrapper for handle_maestro_iae"""
    handlers = get_handlers()
    results = await handlers['iae'](
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
    handlers = get_handlers()
    results = await handlers['search'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_scrape", description="📑 Web scraping functionality with content extraction")
async def maestro_scrape(arguments: dict):
    """Wrapper for handle_maestro_scrape"""
    handlers = get_handlers()
    results = await handlers['scrape'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_execute", description="⚙️ Command execution with security controls")
async def maestro_execute(arguments: dict):
    """Wrapper for handle_maestro_execute"""
    handlers = get_handlers()
    results = await handlers['execute'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_error_handler", description="🚨 Advanced error handling and recovery")
async def maestro_error_handler(arguments: dict):
    """Wrapper for handle_maestro_error_handler"""
    handlers = get_handlers()
    results = await handlers['error_handler'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="maestro_temporal_context", description="🕐 Temporal context analysis for time-sensitive information")
async def maestro_temporal_context(arguments: dict):
    """Wrapper for handle_maestro_temporal_context"""
    handlers = get_handlers()
    results = await handlers['temporal_context'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(name="get_available_engines", description="📊 Get available computational engines and capabilities")
def get_available_engines():
    """Expose computational tools from ComputationalTools"""
    handlers = get_handlers()
    comp_tools = handlers['get_computational_tools']()
    return comp_tools.get_available_engines()

if __name__ == "__main__":
    log_debug("Starting Maestro MCP Server")
    mcp.run() 