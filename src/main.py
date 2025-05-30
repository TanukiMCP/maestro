"""
Maestro MCP Server - Enhanced Workflow Orchestration

Provides intelligent workflow orchestration tools for LLM enhancement.
HTTP transport implementation for Smithery compatibility.
"""

import asyncio
import logging
import time
import os
import sys
import traceback
from typing import Dict, Any, List

# Import FastAPI to wrap our MCP server
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from mcp import types # Import types directly
from mcp.server.fastmcp import FastMCP

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
    """Helper to log debug messages with a SMITHERY-DEBUG prefix for easy filtering"""
    logger.debug(f"SMITHERY-DEBUG: {msg}", *args, **kwargs)

log_debug("Starting Maestro MCP Server")
log_debug(f"Python version: {sys.version}")
log_debug(f"Environment variables: SMITHERY_MODE={os.environ.get('SMITHERY_MODE')}, ENABLE_LAZY_LOADING={os.environ.get('ENABLE_LAZY_LOADING')}")

# Initialize FastMCP server
try:
    log_debug("Initializing FastMCP server")
    start_time = time.time()
    mcp = FastMCP("maestro")
    init_time = time.time() - start_time
    log_debug(f"FastMCP server initialized in {init_time:.2f} seconds")
except Exception as e:
    log_debug(f"Error initializing FastMCP server: {e}")
    log_debug(traceback.format_exc())
    raise

# --- Lazy Loaded Instances ---
_maestro_tools_instance = None
_computational_tools_instance = None

def get_maestro_tools_instance():
    """Lazily get an instance of MaestroTools."""
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        log_debug("Creating MaestroTools instance")
        start_time = time.time()
        # This import should be safe if maestro_tools.py only defines the class
        # and its __init__ is lightweight.
        from .maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
        load_time = time.time() - start_time
        log_debug(f"MaestroTools instance created lazily in {load_time:.2f} seconds")
    return _maestro_tools_instance

def get_computational_tools_instance():
    """Lazily get an instance of ComputationalTools."""
    global _computational_tools_instance
    if _computational_tools_instance is None:
        log_debug("Creating ComputationalTools instance")
        start_time = time.time()
        # This import should be safe if computational_tools.py only defines the class
        # and its __init__ is lightweight.
        from .computational_tools import ComputationalTools
        _computational_tools_instance = ComputationalTools()
        load_time = time.time() - start_time
        log_debug(f"ComputationalTools instance created lazily in {load_time:.2f} seconds")
    return _computational_tools_instance

# --- Enhanced Tool Handlers ---
_enhanced_tool_handlers_instance = None

def get_enhanced_tool_handlers_instance():
    """Lazily get an instance of EnhancedToolHandlers."""
    global _enhanced_tool_handlers_instance
    if _enhanced_tool_handlers_instance is None:
        log_debug("Creating EnhancedToolHandlers instance")
        from .maestro.enhanced_tools import EnhancedToolHandlers
        _enhanced_tool_handlers_instance = EnhancedToolHandlers()
    return _enhanced_tool_handlers_instance

# --- Tool Handlers ---
# These handlers will be registered with MCP.
# They call the actual logic in MaestroTools or ComputationalTools.

async def handle_maestro_orchestrate(task_description: str, context: Dict[str, Any] = None, success_criteria: Dict[str, Any] = None, complexity_level: str = "moderate") -> List[types.TextContent]:
    log_debug("handle_maestro_orchestrate called")
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
    log_debug("handle_maestro_iae_discovery called")
    instance = get_maestro_tools_instance()
    args = {
        "task_type": task_type,
        "domain_context": domain_context,
        "complexity_requirements": complexity_requirements or {}
    }
    return await instance._handle_iae_discovery(arguments=args)

async def handle_maestro_tool_selection(request_description: str, available_context: Dict[str, Any] = None, precision_requirements: Dict[str, Any] = None) -> List[types.TextContent]:
    log_debug("handle_maestro_tool_selection called")
    instance = get_maestro_tools_instance()
    args = {
        "request_description": request_description,
        "available_context": available_context or {},
        "precision_requirements": precision_requirements or {}
    }
    return await instance._handle_tool_selection(arguments=args)

async def handle_maestro_iae(engine_domain: str, computation_type: str, parameters: Dict[str, Any], precision_requirements: str = "machine_precision", validation_level: str = "standard") -> List[types.TextContent]:
    log_debug("handle_maestro_iae called")
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

# Enhanced tool handler wrappers
async def handle_maestro_search(arguments: dict) -> list:
    instance = get_enhanced_tool_handlers_instance()
    return await instance.handle_maestro_search(arguments)

async def handle_maestro_scrape(arguments: dict) -> list:
    instance = get_enhanced_tool_handlers_instance()
    return await instance.handle_maestro_scrape(arguments)

async def handle_maestro_execute(arguments: dict) -> list:
    instance = get_enhanced_tool_handlers_instance()
    return await instance.handle_maestro_execute(arguments)

async def handle_maestro_error_handler(arguments: dict) -> list:
    instance = get_enhanced_tool_handlers_instance()
    return await instance.handle_maestro_error_handler(arguments)

async def handle_maestro_temporal_context(arguments: dict) -> list:
    instance = get_enhanced_tool_handlers_instance()
    return await instance.handle_maestro_temporal_context(arguments)

# --- Tool Registration ---

def _register_tools():
    """Register MCP tools by defining their schemas and handlers."""
    logger.info("Registering tools for Maestro MCP Server...")

    try:
        log_debug("Registering maestro_orchestrate tool")
        mcp.tool(
            name="maestro_orchestrate",
            description="🎭 Intelligent workflow orchestration with context analysis and success criteria validation."
        )(handle_maestro_orchestrate)
        logger.info("Registered: maestro_orchestrate")
    except Exception as e:
        log_debug(f"Error registering maestro_orchestrate: {e}")
        log_debug(traceback.format_exc())

    try:
        log_debug("Registering maestro_iae_discovery tool")
        mcp.tool(
            name="maestro_iae_discovery",
            description="💡 Discover Intelligence Amplification Engines and their capabilities."
        )(handle_maestro_iae_discovery)
        logger.info("Registered: maestro_iae_discovery")
    except Exception as e:
        log_debug(f"Error registering maestro_iae_discovery: {e}")
        log_debug(traceback.format_exc())

    try:
        log_debug("Registering maestro_tool_selection tool")
        mcp.tool(
            name="maestro_tool_selection",
            description="🎯 Intelligent tool selection based on task requirements and computational needs."
        )(handle_maestro_tool_selection)
        logger.info("Registered: maestro_tool_selection")
    except Exception as e:
        log_debug(f"Error registering maestro_tool_selection: {e}")
        log_debug(traceback.format_exc())

    try:
        log_debug("Registering maestro_iae tool")
        mcp.tool(
            name="maestro_iae",
            description="🧮 Intelligent Amplification Engine for specialized computational tasks across multiple domains."
        )(handle_maestro_iae)
        logger.info("Registered: maestro_iae")
    except Exception as e:
        log_debug(f"Error registering maestro_iae: {e}")
        log_debug(traceback.format_exc())

    # Register enhanced tools
    try:
        log_debug("Registering maestro_search tool")
        mcp.tool(
            name="maestro_search",
            description="🌐 LLM-driven web search with temporal filtering and structured results."
        )(handle_maestro_search)
        logger.info("Registered: maestro_search")
    except Exception as e:
        log_debug(f"Error registering maestro_search: {e}")
        log_debug(traceback.format_exc())

    try:
        log_debug("Registering maestro_scrape tool")
        mcp.tool(
            name="maestro_scrape",
            description="🕷️ LLM-driven web scraping and content extraction with selectors and format options."
        )(handle_maestro_scrape)
        logger.info("Registered: maestro_scrape")
    except Exception as e:
        log_debug(f"Error registering maestro_scrape: {e}")
        log_debug(traceback.format_exc())

    try:
        log_debug("Registering maestro_execute tool")
        mcp.tool(
            name="maestro_execute",
            description="⚡ LLM-driven code execution with output capture and validation."
        )(handle_maestro_execute)
        logger.info("Registered: maestro_execute")
    except Exception as e:
        log_debug(f"Error registering maestro_execute: {e}")
        log_debug(traceback.format_exc())

    try:
        log_debug("Registering maestro_error_handler tool")
        mcp.tool(
            name="maestro_error_handler",
            description="🔧 Adaptive error handling and recovery with LLM-driven analysis."
        )(handle_maestro_error_handler)
        logger.info("Registered: maestro_error_handler")
    except Exception as e:
        log_debug(f"Error registering maestro_error_handler: {e}")
        log_debug(traceback.format_exc())

    try:
        log_debug("Registering maestro_temporal_context tool")
        mcp.tool(
            name="maestro_temporal_context",
            description="🕐 Temporal context analysis for information freshness and deadline awareness."
        )(handle_maestro_temporal_context)
        logger.info("Registered: maestro_temporal_context")
    except Exception as e:
        log_debug(f"Error registering maestro_temporal_context: {e}")
        log_debug(traceback.format_exc())

    # Use _tool_map for tool listing
    try:
        tool_map = getattr(mcp, '_tool_map', {})
        log_debug(f"Tool registration process completed. Registered {len(tool_map)} tools: {list(tool_map.keys())}")
    except Exception as e:
        log_debug(f"Error accessing tool map: {e}")
        log_debug(traceback.format_exc())

# Register tools when this module is imported
try:
    _register_tools()
except Exception as e:
    log_debug(f"Error during tool registration: {e}")
    log_debug(traceback.format_exc())
    # Allow server to start even if registration fails, to aid debugging
    # raise # Commented out to allow server to start

# Create a FastAPI app
log_debug("Creating FastAPI app")
fastapi_app = FastAPI(title="Maestro MCP Server", description="Enhanced Workflow Orchestration")

# MCP request/response handler
async def handle_mcp_request(request: Request):
    """Handle MCP requests and proxy them to the FastMCP instance."""
    req_start_time = time.time()
    logger.info(f"Handling MCP request: {request.method} {request.url.path}")
    log_debug(f"Request headers: {request.headers}")
    
    # Get the request body
    body = await request.body()
    log_debug(f"Request body size: {len(body)} bytes")
    
    # Set up headers dictionary from request headers
    headers = dict(request.headers.items())
    
    # Create response queue for ASGI messages
    response_queue = asyncio.Queue()
    
    # Define send and receive functions for ASGI
    async def send(message):
        log_debug(f"ASGI send message type: {message.get('type')}")
        await response_queue.put(message)
    
    async def receive():
        log_debug("ASGI receive called")
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
    
    log_debug(f"Calling FastMCP ASGI app with scope: {scope}")
    
    # Process the request through FastMCP
    try:
        await mcp(scope, receive, send)
        log_debug("FastMCP ASGI app call completed")
    except Exception as e:
        log_debug(f"Error calling FastMCP ASGI app: {e}")
        log_debug(traceback.format_exc())
        raise
    
    # Get the response from the queue
    log_debug("Getting response from queue")
    response_start = await response_queue.get()
    response_body = await response_queue.get()
    
    # Extract status code and headers
    status_code = response_start.get("status", 200)
    headers = dict([(k.decode(), v.decode()) for k, v in response_start.get("headers", [])])
    
    req_time = time.time() - req_start_time
    log_debug(f"MCP request completed in {req_time:.2f} seconds with status {status_code}")
    
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
    req_start_time = time.time()
    log_debug(f"MCP endpoint request: {request.method} {request.url.path}")
    log_debug(f"Request client: {request.client}")
    log_debug(f"Request headers: {dict(request.headers.items())}")
    
    # Fast path for GET requests - specifically for Smithery tool scanning
    if request.method == "GET":
        log_debug("Fast path for tool scanning activated")
        try:
            start_time = time.time()
            tool_list = []
            tool_map = getattr(mcp, '_tool_map', {})
            log_debug(f"Preparing tool list from {len(tool_map)} tools")
            for tool_name, tool_info in tool_map.items():
                log_debug(f"Adding tool {tool_name} to response")
                tool_list.append({
                    "name": tool_name,
                    "description": getattr(tool_info, 'description', None)
                })
            log_debug(f"Tool list prepared with {len(tool_list)} tools")
            response_time = time.time() - start_time
            log_debug(f"Fast path response prepared in {response_time:.2f} seconds")
            log_debug(f"Total GET request time: {time.time() - req_start_time:.2f} seconds")
            return JSONResponse(content=tool_list)
        except Exception as e:
            log_debug(f"Error in fast path for tool scanning: {e}")
            log_debug(traceback.format_exc())
            return JSONResponse(content={"error": "Failed to list tools", "details": str(e)}, status_code=500)
    
    # Normal path for other requests
    log_debug("Using normal MCP request path")
    return await handle_mcp_request(request)

# Add a default route for the root path
@fastapi_app.get("/")
async def root():
    """Return information about the server and its endpoints."""
    log_debug("Root endpoint called")
    tool_map = getattr(mcp, '_tool_map', {})
    tools_count = len(tool_map)
    log_debug(f"Current tool count: {tools_count}")
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

# Debug endpoint to get diagnostic information
@fastapi_app.get("/debug")
async def debug_info():
    """Return debugging information about the server."""
    log_debug("Debug endpoint called")
    import psutil
    import platform
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    tool_map = getattr(mcp, '_tool_map', {})
    return {
        "server_info": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "uptime_seconds": time.time() - process.create_time(),
        },
        "memory_usage": {
            "rss_bytes": memory_info.rss,
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_bytes": memory_info.vms,
            "vms_mb": memory_info.vms / (1024 * 1024),
        },
        "tools": {
            "count": len(tool_map),
            "names": list(tool_map.keys())
        },
        "environment": {
            "smithery_mode": os.environ.get("SMITHERY_MODE", "false"),
            "enable_lazy_loading": os.environ.get("ENABLE_LAZY_LOADING", "false"),
            "optimize_tool_scanning": os.environ.get("OPTIMIZE_TOOL_SCANNING", "false")
        }
    }

# Add a new lightweight /tools endpoint
@fastapi_app.get("/tools")
async def lightweight_tools():
    """Extremely lightweight tool listing endpoint that bypasses FastMCP's internal mechanisms."""
    log_debug("Lightweight tools endpoint called - bypassing FastMCP for fast scanning")
    return [
        {"name": "maestro_orchestrate", "description": "🎭 Intelligent workflow orchestration with context analysis and success criteria validation."},
        {"name": "maestro_iae_discovery", "description": "💡 Discover Intelligence Amplification Engines and their capabilities."},
        {"name": "maestro_tool_selection", "description": "🎯 Intelligent tool selection based on task requirements and computational needs."},
        {"name": "maestro_iae", "description": "🧮 Intelligent Amplification Engine for specialized computational tasks across multiple domains."},
        {"name": "maestro_search", "description": "🌐 LLM-driven web search with temporal filtering and structured results."},
        {"name": "maestro_scrape", "description": "🕷️ LLM-driven web scraping and content extraction with selectors and format options."},
        {"name": "maestro_execute", "description": "⚡ LLM-driven code execution with output capture and validation."},
        {"name": "maestro_error_handler", "description": "🔧 Adaptive error handling and recovery with LLM-driven analysis."},
        {"name": "maestro_temporal_context", "description": "🕐 Temporal context analysis for information freshness and deadline awareness."}
    ]

# Add a dedicated healthcheck endpoint for Smithery
@fastapi_app.get("/health")
async def healthcheck():
    """Dedicated lightweight healthcheck endpoint for Smithery."""
    log_debug("Healthcheck endpoint called")
    return {"status": "healthy", "timestamp": time.time()}

# The 'app' for Uvicorn is now the FastAPI app that mounts FastMCP at /mcp
app = fastapi_app
log_debug("Module initialization complete - app ready to serve requests")

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