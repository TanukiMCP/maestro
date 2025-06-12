#!/usr/bin/env python3
"""
Application Factory for the Maestro MCP Server

This module provides a factory function to create and configure the FastAPI
application instance. This approach allows for deferred initialization,
which is crucial for things like tool discovery in environments like Smithery
where the application shouldn't be fully started just for inspection.
"""

import logging
from fastmcp.server import FastMCP
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware
import os
import sys
import time
import inspect
from functools import wraps

# Import the static tool dictionary for fast tool scanning
# This is critical for Smithery deployments which have strict timeouts
try:
    # Try importing from the src directory first (when run as a module)
    from src.static_tools_dict import MAESTRO_TOOLS_DICT, get_tool_names
except ImportError:
    try:
        # Then try importing from the root directory (when run directly)
        from static_tools_dict import MAESTRO_TOOLS_DICT, get_tool_names
    except ImportError:
        # Fallback to empty dictionary if not found
        print("WARNING: static_tools_dict.py not found, using empty dictionary")
        MAESTRO_TOOLS_DICT = {}
        def get_tool_names(): return []

# Import real tool implementations (lazy-loaded only when needed)
from maestro.dependencies import get_config
from maestro.tools import (
    maestro_orchestrate,
    maestro_iae,
    maestro_web,
    maestro_execute,
    maestro_error_handler
)

logger = logging.getLogger(__name__)

# Check if we're in tool scanning mode (for Smithery deployment)
# This is set when Smithery is scanning for available tools
TOOL_SCANNING_MODE = os.environ.get("SMITHERY_SCANNING", "0") == "1" or "--scan-tools" in sys.argv

# --- Tool Registration ---
# This list is defined statically so it can be inspected without
# initializing the full application.
maestro_tools = [
    maestro_orchestrate,
    maestro_iae,
    maestro_web,
    maestro_execute,
    maestro_error_handler
]

def create_proxy_function(original_func, tool_meta):
    """Dynamically creates a proxy function with the same signature as the original."""
    
    @wraps(original_func)
    async def proxy(**kwargs):
        # Call the original function with all provided kwargs
        # The validation that this will work is done by fastmcp already
        start_time = time.time()
        result = await original_func(**kwargs)
        end_time = time.time()
        logger.debug(f"Tool {original_func.__name__} executed in {(end_time - start_time)*1000:.2f}ms")
        return result
    
    # Add the metadata for fastmcp to find
    proxy.__doc__ = tool_meta.get("description", "")
    proxy.description = tool_meta.get("description", "")
    proxy.parameters = tool_meta.get("parameters", {})
    proxy.category = tool_meta.get("category", "")
    
    return proxy

# Create lazy-loading proxies for each tool
lazy_tools = []
for tool in maestro_tools:
    tool_name = tool.__name__
    if TOOL_SCANNING_MODE and tool_name in MAESTRO_TOOLS_DICT:
        # During scanning, create a proxy with the correct signature and metadata
        lazy_tools.append(create_proxy_function(
            tool,
            MAESTRO_TOOLS_DICT[tool_name]
        ))
    else:
        # In normal mode, just use the tool directly
        lazy_tools.append(tool)

async def health_check(request):
    """Health check endpoint for container orchestration and monitoring."""
    import datetime
    return JSONResponse({
        "status": "healthy",
        "service": "maestro-mcp-server",
        "version": "1.0",
        "tools_count": len(lazy_tools),
        "protocol": "mcp-2024-11-05",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

def create_app():
    """
    Creates and configures the Maestro MCP application.
    
    This factory function handles the configuration loading and application
    setup, ensuring that expensive operations are not performed at module
    import time.
    
    For Smithery deployments, it also ensures the server binds to the
    correct PORT environment variable.
    """
    # --- Minimal Startup Configuration ---
    # Only load the engine mode from env vars for initial server setup.
    # The full config will be loaded per-request by the dependency injector.
    mode_str = os.getenv("MAESTRO_MODE", "production").lower()
    is_dev_mode = (mode_str == "development")
    
    log_level_str = os.getenv("MAESTRO_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level_str,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"✅ Starting server in '{mode_str}' mode.")
    
    # Handle PORT environment variable for Smithery deployment
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"✅ Server will bind to port {port}")
    
    # --- Application Instance ---
    # For Smithery scanning mode, we reduce timeout and use lightweight tools
    # without any dependencies to ensure scans complete within 100ms
    start_time = time.time()
    
    # Determine if this is a tool scanning operation
    is_scanning = TOOL_SCANNING_MODE
    
    if is_scanning:
        logger.info("🔍 Smithery tool scanning mode detected, using lightweight tool definitions.")
        
        # Use extremely fast options for tool scanning
        mcp = FastMCP(
            # Use lazy-loaded tools with pre-defined metadata for fast scanning
            tools=lazy_tools,
            name="Maestro",
            instructions="An MCP server for advanced, backend-only orchestration and intelligence amplification. Provides session management for complex multi-step tasks.",
            on_duplicate_tools="warn",
            mask_error_details=not is_dev_mode,
            # Fast scanning configuration for Smithery
            timeout=10,  # Reduced timeout for tool scanning
            # Skip dependencies completely during scanning to ensure rapid startup
            dependencies={}
        )
    else:
        logger.info("🚀 Normal operation mode detected, using full tool implementations.")
        # Normal operation mode with full features
        mcp = FastMCP(
            # Use the original tools here, not the proxies
            tools=maestro_tools,
            name="Maestro",
            instructions="An MCP server for advanced, backend-only orchestration and intelligence amplification. Provides session management for complex multi-step tasks.",
            on_duplicate_tools="warn",
            mask_error_details=not is_dev_mode,
            # Configure timeouts for Smithery compatibility
            timeout=30,
            # Add dependencies for full operation, which are loaded lazily on request
            dependencies={'config': get_config}
        )

    app = mcp.streamable_http_app()
    
    # Add CORS middleware to allow cross-origin requests from Smithery
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for simplicity, can be restricted
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    end_time = time.time()
    initialization_time = (end_time - start_time) * 1000  # Convert to ms
    logger.info(f"✅ Maestro MCP Server created with {len(lazy_tools)} tools in {initialization_time:.2f}ms.")

    # Add the health route to the app
    if hasattr(app, 'router'):
        app.router.routes.append(Route("/health", health_check, methods=["GET"]))
        logger.info("✅ Health check endpoint added.")

    return app 