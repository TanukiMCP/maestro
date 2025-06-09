#!/usr/bin/env python3
"""
Maestro MCP Server - Backend-only, headless Intelligence Amplification server
This server implements the Model Context Protocol to provide agentic tools.

Copyright (c) 2025 TanukiMCP Orchestra
Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
Contact tanukimcp@gmail.com for commercial licensing inquiries
"""

import sys
import os
# Add the src directory to Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from fastmcp.server import FastMCP

from maestro.config import MAESTROConfig
from maestro.tools import (
    maestro_orchestrate,
    maestro_iae,
    maestro_web,
    maestro_execute,
    maestro_error_handler,
    maestro_collaboration_response
)

# --- Configuration & Logging ---
config = MAESTROConfig.from_env()
config.validate()

logging.basicConfig(
    level=config.logging.level.value,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Tool Registration ---
maestro_tools = [
    maestro_orchestrate,
    maestro_iae,
    maestro_web,
    maestro_execute,
    maestro_error_handler,
    maestro_collaboration_response
]

# --- Application Instance ---
# Create the FastMCP instance with proper HTTP session management
mcp = FastMCP(
    tools=maestro_tools,
    name="Maestro",
    instructions="An MCP server for advanced, backend-only orchestration and intelligence amplification.",
    on_duplicate_tools="warn",
    mask_error_details=config.engine.mode.value != "development",
)

# Support both stdio and HTTP transports
if __name__ == "__main__":
    import sys
    import anyio
    
    # If run directly, use stdio transport (MCP standard for command-based clients)
    if len(sys.argv) == 1 or "--stdio" in sys.argv:
        logger.info(f"🔌 Starting Maestro MCP Server in stdio mode with {len(maestro_tools)} tools")
        anyio.run(mcp.run_stdio_async)
    else:
        # For HTTP transport, create the streamable HTTP app
        app = mcp.streamable_http_app()
        logger.info(f"🌐 Maestro MCP Server Streamable HTTP app created with {len(maestro_tools)} tools")
else:
    # When imported (e.g., by uvicorn), create the streamable HTTP app
    app = mcp.streamable_http_app()
    logger.info(f"✅ Maestro MCP Server created with {len(maestro_tools)} tools.")

# Add health endpoint for Docker health checks and Smithery monitoring
from starlette.responses import JSONResponse
from starlette.routing import Route

async def health_check(request):
    """Health check endpoint for container orchestration and monitoring."""
    import datetime
    return JSONResponse({
        "status": "healthy",
        "service": "maestro-mcp-server",
        "version": "1.0",
        "tools_count": len(maestro_tools),
        "protocol": "mcp-2024-11-5",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# Add the health route to the app
if hasattr(app, 'router'):
    app.router.routes.append(Route("/health", health_check, methods=["GET"]))
