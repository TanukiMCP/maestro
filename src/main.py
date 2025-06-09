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
import logging
import anyio
from fastmcp.server import FastMCP

# Add the src directory to Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app factory and the static tool list
from app_factory import create_app, maestro_tools

logger = logging.getLogger(__name__)

# --- Application Instance ---
# When imported by an ASGI server like Uvicorn, the `app` object is created here.
# The factory ensures that configuration is loaded and the app is initialized
# only when this module is executed or imported for serving, not for inspection.
app = create_app()

# --- Main Execution Block ---
# This block handles direct execution of the server, for example, for stdio mode.
if __name__ == "__main__":
    if len(sys.argv) == 1 or "--stdio" in sys.argv:
        # For stdio mode, we still need to load config to run the server.
        # We create a temporary app instance to get the underlying MCP object.
        # This is a bit of a workaround because stdio mode is tied to the MCP instance.
        
        # We need a simple, temporary app to get the configured MCP instance
        # since the main `app` is a Starlette app.
        from maestro.config import MAESTROConfig
        config = MAESTROConfig.from_env()
        
        logging.basicConfig(
            level=config.logging.level.value,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        mcp_instance = FastMCP(
            tools=maestro_tools,
            name="Maestro",
            instructions="An MCP server for advanced, backend-only orchestration and intelligence amplification.",
            on_duplicate_tools="warn",
            mask_error_details=config.engine.mode.value != "development",
        )
        
        logger.info(f"🔌 Starting Maestro MCP Server in stdio mode with {len(maestro_tools)} tools")
        anyio.run(mcp_instance.run_stdio_async)
    else:
        # This branch is for running via `python src/main.py http` or similar,
        # which is less common than using `uvicorn`. Uvicorn will just import `app`.
        logger.info("To run the HTTP server, please use: uvicorn run:main --reload")
