"""
Maestro MCP Server - Enhanced Workflow Orchestration

Provides intelligent workflow orchestration tools for LLM enhancement.
HTTP transport implementation for Smithery compatibility.
"""

import asyncio
import logging
from typing import Dict, Any, List

# Import FastAPI to wrap our MCP server
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from mcp import types # Import types directly
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
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
    # Hardcoded schema instead of loading from ComputationalTools during initialization
    mcp.tool(
        name="maestro_iae",
        description="🧮 Intelligent Amplification Engine for specialized computational tasks across multiple domains.",
        inputSchema={
            "type": "object",
            "properties": {
                "engine_domain": {"type": "string"},
                "computation_type": {"type": "string"},
                "parameters": {"type": "object"},
                "precision_requirements": {"type": "string", "default": "machine_precision"},
                "validation_level": {"type": "string", "default": "standard"}
            },
            "required": ["engine_domain", "computation_type", "parameters"]
        }
    )(handle_maestro_iae)
    logger.info("Registered: maestro_iae")

    logger.info("Tool registration process completed.")

# Register tools when this module is imported
_register_tools()

# Create a FastAPI app
fastapi_app = FastAPI(title="Maestro MCP Server", description="Enhanced Workflow Orchestration")

# MCP request/response handler
async def handle_mcp_request(request: Request):
    """Handle MCP requests and proxy them to the FastMCP instance."""
    logger.info(f"Handling MCP request: {request.method} {request.url.path}")
    
    # Get the request body
    body = await request.body()
    
    # Set up headers dictionary from request headers
    headers = dict(request.headers.items())
    
    # Create response queue for ASGI messages
    response_queue = asyncio.Queue()
    
    # Define send and receive functions for ASGI
    async def send(message):
        await response_queue.put(message)
    
    async def receive():
        return {
            "type": "http.request",
            "body": body,
            "more_body": False
        }
    
    # Call the FastMCP ASGI app with appropriate scope
    # Path is empty because FastMCP doesn't expect a /mcp prefix
    scope = {
        "type": "http",
        "path": "/",
        "method": request.method,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "query_string": request.url.query.encode(),
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 8000),
        "scheme": request.url.scheme,
        "http_version": "1.1",
        "raw_path": request.url.path.encode(),
    }
    
    # Process the request through FastMCP
    await mcp(scope, receive, send)
    
    # Get the response from the queue
    response_start = await response_queue.get()
    response_body = await response_queue.get()
    
    # Extract status code and headers
    status_code = response_start.get("status", 200)
    headers = dict([(k.decode(), v.decode()) for k, v in response_start.get("headers", [])])
    
    # Return the response
    return StreamingResponse(
        content=[response_body.get("body", b"")],
        status_code=status_code,
        headers=headers
    )

# Mount the MCP server at /mcp with route for all methods
@fastapi_app.api_route("/mcp", methods=["GET", "POST", "DELETE", "OPTIONS"])
async def handle_all_mcp_methods(request: Request):
    """Handle all HTTP methods to /mcp for Smithery compatibility."""
    logger.info(f"MCP endpoint called with method: {request.method}")
    return await handle_mcp_request(request)

# The 'app' for Uvicorn is now the FastAPI app that mounts FastMCP at /mcp
app = fastapi_app

# Add a default route for the root path
@fastapi_app.get("/")
async def root():
    """Return information about the server and its endpoints."""
    tools_count = len(mcp.tools)
    return {
        "name": "Maestro MCP Server",
        "description": "Enhanced Workflow Orchestration with Intelligence Amplification",
        "version": "1.0.0",
        "status": "online",
        "tools_count": tools_count,
        "endpoints": {
            "mcp": "/mcp - MCP server endpoint (GET: tool list, POST: tool call, DELETE: cancel)",
            "docs": "/docs - FastAPI auto-generated documentation"
        },
        "smithery_compatible": True,
        "lazy_loading": True
    }

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