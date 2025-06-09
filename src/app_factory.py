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

from maestro.config import MAESTROConfig
from maestro.tools import (
    maestro_orchestrate,
    maestro_iae,
    maestro_web,
    maestro_execute,
    maestro_error_handler,
    maestro_collaboration_response
)

logger = logging.getLogger(__name__)

# --- Tool Registration ---
# This list is defined statically so it can be inspected without
# initializing the full application.
maestro_tools = [
    maestro_orchestrate,
    maestro_iae,
    maestro_web,
    maestro_execute,
    maestro_error_handler,
    maestro_collaboration_response
]

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

def create_app():
    """
    Creates and configures the Maestro MCP application.
    
    This factory function handles the configuration loading and application
    setup, ensuring that expensive operations are not performed at module
    import time.
    """
    # --- Configuration & Logging ---
    # Configuration is loaded here, inside the factory, not at the global scope.
    config = MAESTROConfig.from_env()
    config.validate()

    logging.basicConfig(
        level=config.logging.level.value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("✅ Maestro MCP Server configuration loaded.")
    
    # --- Application Instance ---
    mcp = FastMCP(
        tools=maestro_tools,
        name="Maestro",
        instructions="An MCP server for advanced, backend-only orchestration and intelligence amplification.",
        on_duplicate_tools="warn",
        mask_error_details=config.engine.mode.value != "development",
    )

    app = mcp.streamable_http_app()
    logger.info(f"✅ Maestro MCP Server created with {len(maestro_tools)} tools.")

    # Add the health route to the app
    if hasattr(app, 'router'):
        app.router.routes.append(Route("/health", health_check, methods=["GET"]))
        logger.info("✅ Health check endpoint added.")

    return app 