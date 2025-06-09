#!/usr/bin/env python3
"""
Maestro MCP Server - Backend-only, headless Intelligence Amplification server

This is a production-quality, IDE-agnostic server designed to be called by external
agentic IDEs like Cursor, Claude Desktop, Windsurf, and Cline. It provides no UI
or LLM client functionality, focusing purely on computational intelligence amplification.

Copyright (c) 2025 TanukiMCP Orchestra
Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
Contact tanukimcp@gmail.com for commercial licensing inquiries
"""

import logging
import os
import sys
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from maestro.config import MAESTROConfig, EngineMode
from maestro.api import MAESTROAPIRouter
from maestro.schemas import HealthResponse

# Load and validate configuration
config = MAESTROConfig.from_env()
config.validate()

# Configure logging based on config
logging.basicConfig(
    level=config.logging.level.value,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if config.logging.file_enabled and config.logging.file_path:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        config.logging.file_path,
        maxBytes=config.logging.rotation_size * 1024 * 1024,
        backupCount=config.logging.retention_days
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MAESTRO MCP Server",
    description="Backend-only, headless Intelligence Amplification server",
    version="1.0.0",
    docs_url="/docs" if config.engine.mode != EngineMode.PRODUCTION else None,
    redoc_url="/redoc" if config.engine.mode != EngineMode.PRODUCTION else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.security.allowed_origins
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Create API router
api_router = MAESTROAPIRouter(config=config)

# Include API routes
app.include_router(api_router.router, prefix="/api/v1")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def validate_llm_client(request: Request, call_next):
    """Ensure LLM client is provided for endpoints that require it."""
    if request.url.path.startswith("/api/v1/orchestrate"):
        # Extract LLM client from request headers/body
        llm_client = await extract_llm_client(request)
        request.state.llm_client = llm_client
    return await call_next(request)

async def extract_llm_client(request: Request) -> Any:
    """
    Extract LLM client from request.
    The external IDE must provide this in a standardized format.
    """
    try:
        body = await request.json()
        llm_client = body.get("llm_client")
        if not llm_client:
            raise ValueError("LLM client not provided")
        return llm_client
    except Exception as e:
        logger.error(f"Failed to extract LLM client: {e}")
        raise ValueError("Invalid or missing LLM client")

def main():
    """Run the MAESTRO MCP server."""
    try:
        # Log startup information
        logger.info("🎭 Starting MAESTRO MCP server...")
        logger.info(f"Mode: {config.engine.mode}")
        logger.info(f"Host: {config.server.host}")
        logger.info(f"Port: {config.server.port}")
        
        # Start server
        uvicorn.run(
            "main:app",
            host=config.server.host,
            port=config.server.port,
            workers=config.server.workers,
            timeout_keep_alive=config.server.timeout,
            reload=config.engine.mode == EngineMode.DEVELOPMENT
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
