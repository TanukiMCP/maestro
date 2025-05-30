"""
Maestro MCP Server - Enhanced Workflow Orchestration

Provides intelligent workflow orchestration tools for LLM enhancement.
HTTP transport implementation for Smithery compatibility.
"""

import asyncio
import logging
from typing import Dict, Any, List

from mcp import types # Import types directly
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server with HTTP transport for Smithery
mcp = FastMCP("maestro")

# --- Lazy Loaded Instances ---
_maestro_tools_instance = None
_computational_tools_instance = None

def get_maestro_tools_instance():
    """Lazily get an instance of MaestroTools."""
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        # This import should be safe if maestro_tools.py only defines the class
        # and its __init__ is lightweight.
        from .maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
        logger.info("MaestroTools instance created lazily.")
    return _maestro_tools_instance

def get_computational_tools_instance():
    """Lazily get an instance of ComputationalTools."""
    global _computational_tools_instance
    if _computational_tools_instance is None:
        # This import should be safe if computational_tools.py only defines the class
        # and its __init__ is lightweight.
        from .computational_tools import ComputationalTools
        _computational_tools_instance = ComputationalTools()
        logger.info("ComputationalTools instance created lazily.")
    return _computational_tools_instance

# --- Tool Handlers ---
# These handlers will be registered with MCP.
# They call the actual logic in MaestroTools or ComputationalTools.

async def handle_maestro_orchestrate(task_description: str, context: Dict[str, Any] = None, success_criteria: Dict[str, Any] = None, complexity_level: str = "moderate") -> List[types.TextContent]:
    instance = get_maestro_tools_instance()
    # Assuming _handle_orchestrate in MaestroTools takes arguments as a dictionary
    args = {
        "task_description": task_description,
        "context": context or {},
        "success_criteria": success_criteria or {},
        "complexity_level": complexity_level
    }
    return await instance._handle_orchestrate(arguments=args)

async def handle_maestro_iae_discovery(task_type: str = "general", domain_context: str = "", complexity_requirements: Dict[str, Any] = None) -> List[types.TextContent]:
    instance = get_maestro_tools_instance()
    args = {
        "task_type": task_type,
        "domain_context": domain_context,
        "complexity_requirements": complexity_requirements or {}
    }
    return await instance._handle_iae_discovery(arguments=args)

async def handle_maestro_tool_selection(request_description: str, available_context: Dict[str, Any] = None, precision_requirements: Dict[str, Any] = None) -> List[types.TextContent]:
    instance = get_maestro_tools_instance()
    args = {
        "request_description": request_description,
        "available_context": available_context or {},
        "precision_requirements": precision_requirements or {}
    }
    return await instance._handle_tool_selection(arguments=args)

async def handle_maestro_iae(engine_domain: str, computation_type: str, parameters: Dict[str, Any], precision_requirements: str = "machine_precision", validation_level: str = "standard") -> List[types.TextContent]:
    instance = get_computational_tools_instance()
    # ComputationalTools.handle_tool_call expects 'name' and 'arguments'
    # Here, 'name' is maestro_iae, and arguments are the kwargs.
    args = {
        "engine_domain": engine_domain,
        "computation_type": computation_type,
        "parameters": parameters,
        "precision_requirements": precision_requirements,
        "validation_level": validation_level
    }
    return await instance.handle_tool_call(name="maestro_iae", arguments=args)

# --- Tool Registration ---

def _register_tools():
    """Register MCP tools by defining their schemas and handlers."""
    logger.info("Registering tools for Maestro MCP Server...")

    # Maestro Orchestrate (from MaestroTools)
    mcp.tool(
        name="maestro_orchestrate",
        description="🎭 Intelligent workflow orchestration with context analysis and success criteria validation.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_description": {"type": "string"},
                "context": {"type": "object"},
                "success_criteria": {"type": "object"},
                "complexity_level": {"type": "string", "default": "moderate"}
            },
            "required": ["task_description"]
        }
    )(handle_maestro_orchestrate)
    logger.info("Registered: maestro_orchestrate")

    # Maestro IAE Discovery (from MaestroTools)
    mcp.tool(
        name="maestro_iae_discovery",
        description="💡 Discover Intelligence Amplification Engines and their capabilities.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "default": "general"},
                "domain_context": {"type": "string", "default": ""},
                "complexity_requirements": {"type": "object", "default": {}}
            },
            "required": []
        }
    )(handle_maestro_iae_discovery)
    logger.info("Registered: maestro_iae_discovery")

    # Maestro Tool Selection (from MaestroTools)
    mcp.tool(
        name="maestro_tool_selection",
        description="🎯 Intelligent tool selection based on task requirements and computational needs.",
        inputSchema={
            "type": "object",
            "properties": {
                "request_description": {"type": "string"},
                "available_context": {"type": "object", "default": {}},
                "precision_requirements": {"type": "object", "default": {}}
            },
            "required": ["request_description"]
        }
    )(handle_maestro_tool_selection)
    logger.info("Registered: maestro_tool_selection")
    
    # Maestro IAE (from ComputationalTools)
    # Get schema from ComputationalTools.get_mcp_tools()
    try:
        from .computational_tools import ComputationalTools
        temp_comp_tools = ComputationalTools() # Assumed lightweight __init__
        iae_tool_schema = next((t for t in temp_comp_tools.get_mcp_tools() if t.name == "maestro_iae"), None)
        if iae_tool_schema:
            mcp.tool(
                name=iae_tool_schema.name,
                description=iae_tool_schema.description,
                inputSchema=iae_tool_schema.inputSchema
            )(handle_maestro_iae)
            logger.info(f"Registered: {iae_tool_schema.name}")
        else:
            logger.error("Could not find 'maestro_iae' schema in ComputationalTools.")
    except ImportError:
        logger.error("Could not import ComputationalTools to get 'maestro_iae' schema.")
    except Exception as e:
        logger.error(f"Error getting 'maestro_iae' schema from ComputationalTools: {e}")

    # Placeholder for other original tools from main.py if they need to be kept or adapted
    # Example: maestro_search (if it were to be made real and lazy)
    # @mcp.tool(name="maestro_search", description="...", inputSchema=...)
    # async def handle_maestro_search_placeholder(**kwargs):
    #     # Original main.py search was a placeholder returning a string.
    #     # If it needs to be real, implement lazy loading for its service here.
    #     query = kwargs.get("query", "default query")
    #     # ... (rest of placeholder logic or call to a real service)
    #     return [types.TextContent(text=f"Search results for {query}...")] 
    # logger.info("Registered placeholder: maestro_search")

    logger.info("Tool registration process completed.")

# Register tools when this module is imported
_register_tools()

# The 'app' for Uvicorn is the FastMCP instance itself
app = mcp

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stdio":
        logger.info("🎭 Starting Maestro MCP Server (STDIO Mode)")
        try:
            asyncio.run(mcp.run_stdio_session()) 
        except KeyboardInterrupt:
            logger.info("STDIO session ended by user.")
    else:
        # This branch is typically for information when 'python src/main.py' is run directly.
        # HTTP server startup is handled by 'deploy.py' using Uvicorn.
        logger.info("🎭 Maestro MCP Server (src.main)")
        print("To run with HTTP server, use: python deploy.py [dev|prod|smithery]")
        print("To run with STDIO, use: python src/main.py stdio")