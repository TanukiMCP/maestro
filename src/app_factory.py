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
import os
import sys
import time

# Import the static tool dictionary for fast tool scanning
# This is critical for Smithery deployments which have strict timeouts
from static_tools_dict import MAESTRO_TOOLS_DICT, get_tool_names

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

# Create a lightweight proxy for each tool that defers initialization
# This approach ensures that the tool scanning process is extremely fast
class LazyToolProxy:
    """A proxy that defers tool initialization until the tool is actually called."""
    
    def __init__(self, tool_name, tool_impl, tool_meta):
        self.tool_name = tool_name
        self.tool_impl = tool_impl
        self.__name__ = tool_name
        
        # Copy metadata from static dictionary for fast access
        self.__doc__ = tool_meta.get("description", "")
        self.description = tool_meta.get("description", "")
        self.parameters = tool_meta.get("parameters", {})
        self.category = tool_meta.get("category", "")
    
    async def __call__(self, *args, **kwargs):
        # Only now do we actually call the real implementation
        start_time = time.time()
        result = await self.tool_impl(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Tool {self.tool_name} executed in {(end_time - start_time)*1000:.2f}ms")
        return result

# Create lazy-loading proxies for each tool
lazy_tools = []
for i, tool in enumerate(maestro_tools):
    tool_name = tool.__name__
    if tool_name in MAESTRO_TOOLS_DICT:
        lazy_tools.append(LazyToolProxy(
            tool_name=tool_name,
            tool_impl=tool,
            tool_meta=MAESTRO_TOOLS_DICT[tool_name]
        ))
    else:
        # Fallback if tool not in static dictionary
        lazy_tools.append(tool)
        logger.warning(f"Tool {tool_name} not found in static dictionary")

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
        logger.info("🔍 Smithery tool scanning mode detected")
        
        # Use extremely fast options for tool scanning
        mcp = FastMCP(
            # Use tools with very minimal metadata
            tools=lazy_tools,
            name="Maestro",
            instructions="An MCP server for advanced, backend-only orchestration and intelligence amplification. Provides session management for complex multi-step tasks.",
            on_duplicate_tools="warn",
            mask_error_details=not is_dev_mode,
            # Fast scanning configuration for Smithery
            timeout=10,  # Reduced timeout for tool scanning
            json_response=True,  # Enable JSON responses for Smithery
            stateless_http=True,  # Enable stateless mode for faster scanning
            # Skip dependencies completely during scanning
            dependencies={}
        )
    else:
        # Normal operation mode with full features
        mcp = FastMCP(
            tools=lazy_tools,
            name="Maestro",
            instructions="An MCP server for advanced, backend-only orchestration and intelligence amplification. Provides session management for complex multi-step tasks.",
            on_duplicate_tools="warn",
            mask_error_details=not is_dev_mode,
            # Configure timeouts for Smithery compatibility
            timeout=30,  # Reduced timeout for tool scanning
            json_response=True,  # Enable JSON responses for Smithery
            stateless_http=True,  # Enable stateless mode for faster scanning
            # Add dependencies for full operation
            dependencies={'config': get_config}
        )

    app = mcp.streamable_http_app()
    end_time = time.time()
    initialization_time = (end_time - start_time) * 1000  # Convert to ms
    logger.info(f"✅ Maestro MCP Server created with {len(lazy_tools)} tools in {initialization_time:.2f}ms.")

    # Add the health route to the app
    if hasattr(app, 'router'):
        app.router.routes.append(Route("/health", health_check, methods=["GET"]))
        logger.info("✅ Health check endpoint added.")

    return app 