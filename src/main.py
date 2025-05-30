"""
Maestro MCP Server - Enhanced Workflow Orchestration

Provides intelligent workflow orchestration tools for LLM enhancement.
HTTP transport implementation for Smithery compatibility.
"""

import time # Add time for logging
# VERY EARLY LOGGING - Check if module is even being imported
print(f"SRC.MAIN.PY MODULE IMPORT STARTED AT: {time.time()}", flush=True)

import asyncio
import logging
import os
import sys
import traceback
from typing import Dict, Any, List

# Import FastAPI to wrap our MCP server
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# from mcp import types # MCP NOT USED IN THIS DIAGNOSTIC VERSION
# from mcp.server.fastmcp import FastMCP # MCP NOT USED IN THIS DIAGNOSTIC VERSION

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
    logger.debug(f"DIAGNOSTIC MODE - SMITHERY-DEBUG: {msg}", *args, **kwargs)

log_debug("DIAGNOSTIC MODE - Starting Maestro MCP Server (MCP Functionality Disabled)")
log_debug(f"DIAGNOSTIC MODE - Python version: {sys.version}")

# Check if we should defer MCP initialization (variable still checked but MCP won't be used)
should_defer_mcp_init = os.environ.get("MCP_DEFERRED_INIT", "").lower() == "true"
log_debug(f"DIAGNOSTIC MODE - MCP initialization deferral (not in use): {should_defer_mcp_init}")

# Initialize FastMCP server - COMPLETELY COMMENTED OUT FOR DIAGNOSTIC
_mcp_initialized = False
mcp = None
log_debug("DIAGNOSTIC MODE - FastMCP object creation and initialization SKIPPED")

# --- Lazy Loaded Instances --- (Keep structure, but they won't be called by MCP tools)
_maestro_tools_instance = None
_computational_tools_instance = None

def get_maestro_tools_instance():
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        log_debug("DIAGNOSTIC MODE - Creating MaestroTools instance (if called by non-MCP path)")
        from .maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
    return _maestro_tools_instance

def get_computational_tools_instance():
    global _computational_tools_instance
    if _computational_tools_instance is None:
        log_debug("DIAGNOSTIC MODE - Creating ComputationalTools instance (if called by non-MCP path)")
        from .computational_tools import ComputationalTools
        _computational_tools_instance = ComputationalTools()
    return _computational_tools_instance

_enhanced_tool_handlers_instance = None
def get_enhanced_tool_handlers_instance():
    global _enhanced_tool_handlers_instance
    if _enhanced_tool_handlers_instance is None:
        log_debug("DIAGNOSTIC MODE - Creating EnhancedToolHandlers instance (if called by non-MCP path)")
        from .maestro.enhanced_tools import EnhancedToolHandlers
        _enhanced_tool_handlers_instance = EnhancedToolHandlers()
    return _enhanced_tool_handlers_instance

# --- Tool Handlers --- (Keep definitions, but they won't be registered with MCP)
# ... (Keep all async handle_... functions as they are, they just won't be used by MCP)
async def handle_maestro_orchestrate(task_description: str, context: Dict[str, Any] = None, success_criteria: Dict[str, Any] = None, complexity_level: str = "moderate"): # -> List[types.TextContent]: # MCP Type removed
    log_debug("DIAGNOSTIC MODE - handle_maestro_orchestrate called (MCP path disabled)")
    # ...
    return [{"text": "DIAGNOSTIC - Orchestrate"}] # Placeholder

async def handle_maestro_iae_discovery(task_type: str = "general", domain_context: str = "", complexity_requirements: Dict[str, Any] = None): # -> List[types.TextContent]:
    log_debug("DIAGNOSTIC MODE - handle_maestro_iae_discovery called (MCP path disabled)")
    # ...
    return [{"text": "DIAGNOSTIC - IAE Discovery"}] # Placeholder

async def handle_maestro_tool_selection(request_description: str, available_context: Dict[str, Any] = None, precision_requirements: Dict[str, Any] = None): # -> List[types.TextContent]:
    log_debug("DIAGNOSTIC MODE - handle_maestro_tool_selection called (MCP path disabled)")
    # ...
    return [{"text": "DIAGNOSTIC - Tool Selection"}] # Placeholder

async def handle_maestro_iae(engine_domain: str, computation_type: str, parameters: Dict[str, Any], precision_requirements: str = "machine_precision", validation_level: str = "standard"): # -> List[types.TextContent]:
    log_debug("DIAGNOSTIC MODE - handle_maestro_iae called (MCP path disabled)")
    # ...
    return [{"text": "DIAGNOSTIC - IAE"}] # Placeholder

async def handle_maestro_search(arguments: dict) -> list:
    log_debug("DIAGNOSTIC MODE - handle_maestro_search called (MCP path disabled)")
    return [{"text": "DIAGNOSTIC - Search"}] # Placeholder

async def handle_maestro_scrape(arguments: dict) -> list:
    log_debug("DIAGNOSTIC MODE - handle_maestro_scrape called (MCP path disabled)")
    return [{"text": "DIAGNOSTIC - Scrape"}] # Placeholder

async def handle_maestro_execute(arguments: dict) -> list:
    log_debug("DIAGNOSTIC MODE - handle_maestro_execute called (MCP path disabled)")
    return [{"text": "DIAGNOSTIC - Execute"}] # Placeholder

async def handle_maestro_error_handler(arguments: dict) -> list:
    log_debug("DIAGNOSTIC MODE - handle_maestro_error_handler called (MCP path disabled)")
    return [{"text": "DIAGNOSTIC - Error Handler"}] # Placeholder

async def handle_maestro_temporal_context(arguments: dict) -> list:
    log_debug("DIAGNOSTIC MODE - handle_maestro_temporal_context called (MCP path disabled)")
    return [{"text": "DIAGNOSTIC - Temporal Context"}] # Placeholder


# --- Tool Registration ---
def _register_tools():
    log_debug("DIAGNOSTIC MODE - _register_tools called - MCP registration SKIPPED")
    # All mcp.tool(...) calls are effectively disabled as mcp is None
    return

# Register tools when this module is imported (will do nothing in diagnostic)
try:
    # if not should_defer_mcp_init: # This check is not relevant if mcp is always None here
    _register_tools() 
    # else:
    #    log_debug("DIAGNOSTIC MODE - Tool registration deferred (MCP registration SKIPPED)")
except Exception as e:
    log_debug(f"DIAGNOSTIC MODE - Error during tool registration (should be skipped): {e}")

# Create a FastAPI app
log_debug("DIAGNOSTIC MODE - Creating FastAPI app")
fastapi_app = FastAPI(title="Maestro MCP Server (Diagnostic Mode)", description="Enhanced Workflow Orchestration - MCP DISABLED")

# Add a new lightweight /tools endpoint as the FIRST endpoint
@fastapi_app.get("/tools")
async def lightweight_tools():
    log_debug("DIAGNOSTIC MODE - 💡 Lightweight tools endpoint called")
    response_data = [
        {"name": "maestro_orchestrate", "description": "🎭 Intelligent workflow orchestration... (DIAGNOSTIC)"},
        {"name": "maestro_iae_discovery", "description": "🔍 Integrated Analysis Engine discovery... (DIAGNOSTIC)"},
        {"name": "maestro_tool_selection", "description": "🧰 Intelligent tool selection... (DIAGNOSTIC)"},
        {"name": "maestro_iae", "description": "🧠 Integrated Analysis Engine... (DIAGNOSTIC)"},
        {"name": "maestro_search", "description": "🔎 Enhanced search capabilities... (DIAGNOSTIC)"},
        {"name": "maestro_scrape", "description": "📑 Web scraping functionality... (DIAGNOSTIC)"},
        {"name": "maestro_execute", "description": "⚙️ Command execution... (DIAGNOSTIC)"},
        {"name": "maestro_error_handler", "description": "🚨 Error handling... (DIAGNOSTIC)"},
        {"name": "maestro_temporal_context", "description": "🕐 Temporal context analysis... (DIAGNOSTIC)"}
    ]
    log_debug(f"DIAGNOSTIC MODE - 💡 Returning {len(response_data)} tools from lightweight endpoint")
    return response_data

# Super lightweight ping endpoint
@fastapi_app.get("/ping")
async def ping():
    log_debug("DIAGNOSTIC MODE - 🏓 Ping endpoint called")
    return {"ping": "pong (diagnostic mode)", "timestamp": time.time()}
    
# MCP request/response handler - Modified for diagnostic
async def handle_mcp_request(request: Request):
    log_debug("DIAGNOSTIC MODE - handle_mcp_request called (MCP functionality disabled)")
    return JSONResponse(
        content={"error": "MCP functionality disabled in diagnostic mode"},
        status_code=503
    )

# Mount the MCP server at /mcp - Modified for diagnostic
@fastapi_app.api_route("/mcp", methods=["GET", "POST", "DELETE", "OPTIONS"])
async def handle_all_mcp_methods(request: Request):
    log_debug(f"DIAGNOSTIC MODE - /mcp endpoint called with method: {request.method}")
    if request.method == "GET":
        log_debug("DIAGNOSTIC MODE - /mcp GET request, returning lightweight tools list")
        return await lightweight_tools()
    else:
        log_debug("DIAGNOSTIC MODE - /mcp non-GET request, MCP disabled")
        return JSONResponse(
            content={"error": "MCP functionality disabled in diagnostic mode for non-GET /mcp requests"},
            status_code=503
        )

# Add a default route for the root path
@fastapi_app.get("/")
async def root():
    log_debug("DIAGNOSTIC MODE - Root endpoint called")
    return {
        "name": "Maestro MCP Server (Diagnostic Mode)",
        "status": "online - MCP DISABLED",
        "endpoints": {
            "tools": "/tools - Lightweight tool list (DIAGNOSTIC)",
            "ping": "/ping - Basic server ping (DIAGNOSTIC)",
            "mcp": "/mcp - MCP endpoint (DISABLED IN DIAGNOSTIC)",
            "health": "/health - Health check (DIAGNOSTIC)"
        }
    }

# Debug endpoint (remains useful)
@fastapi_app.get("/debug")
async def debug_info():
    # ... (keep debug_info as is, it's helpful)
    log_debug("DIAGNOSTIC MODE - Debug endpoint called")
    import psutil
    import platform
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "diagnostic_mode_active": True,
        "server_info": { "python_version": sys.version, "platform": platform.platform()},
        "memory_usage": { "rss_mb": memory_info.rss / (1024 * 1024)},
        "environment": { "smithery_mode": os.environ.get("SMITHERY_MODE", "false")}
    }

# Add a dedicated healthcheck endpoint for Smithery
@fastapi_app.get("/health")
async def healthcheck():
    log_debug("DIAGNOSTIC MODE - Healthcheck endpoint called")
    return {"status": "healthy (diagnostic mode)", "timestamp": time.time()}

# The 'app' for Uvicorn is now the FastAPI app
app = fastapi_app
log_debug("DIAGNOSTIC MODE - Module initialization complete - app ready to serve requests (MCP DISABLED)")

# Handle deferred FastMCP initialization - COMMENTED OUT for diagnostic
# async def initialize_mcp_after_startup():
#    log_debug("DIAGNOSTIC MODE - initialize_mcp_after_startup SKIPPED")

# @fastapi_app.on_event("startup")
# async def startup_event():
#    log_debug("DIAGNOSTIC MODE - FastAPI startup event triggered (MCP init SKIPPED)")
#    # if should_defer_mcp_init:
#    #     import asyncio
#    #     asyncio.create_task(initialize_mcp_after_startup())
#    log_debug("DIAGNOSTIC MODE - FastAPI startup complete (MCP init SKIPPED)")

if __name__ == "__main__":
    # ... (keep __main__ block as is, but it won't run MCP stdio session)
    logger.info("DIAGNOSTIC MODE - Running src/main.py directly (MCP STDIO disabled)")
    print("To run with HTTP server (DIAGNOSTIC MODE), use: python deploy.py [dev|prod|smithery]")