"""
Maestro MCP Server - Enhanced Workflow Orchestration

Provides intelligent workflow orchestration tools for LLM enhancement.
HTTP transport implementation for Smithery compatibility.
"""

import time
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
    """Helper to log debug messages with a prefix for easy filtering"""
    logger.debug(f"SMITHERY-DEBUG: {msg}", *args, **kwargs)

log_debug("Starting Maestro MCP Server")
log_debug(f"Python version: {sys.version}")

# Check if we should defer initialization
should_defer_init = os.environ.get("MCP_DEFERRED_INIT", "").lower() == "true"
log_debug(f"Initialization deferral: {should_defer_init}")

# --- Lazy Loaded Instances ---
_maestro_tools_instance = None
_computational_tools_instance = None

def get_maestro_tools_instance():
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        log_debug("Creating MaestroTools instance")
        from .maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
    return _maestro_tools_instance

def get_computational_tools_instance():
    global _computational_tools_instance
    if _computational_tools_instance is None:
        log_debug("Creating ComputationalTools instance")
        from .computational_tools import ComputationalTools
        _computational_tools_instance = ComputationalTools()
    return _computational_tools_instance

_enhanced_tool_handlers_instance = None
def get_enhanced_tool_handlers_instance():
    global _enhanced_tool_handlers_instance
    if _enhanced_tool_handlers_instance is None:
        log_debug("Creating EnhancedToolHandlers instance")
        from .maestro.enhanced_tools import EnhancedToolHandlers
        _enhanced_tool_handlers_instance = EnhancedToolHandlers()
    return _enhanced_tool_handlers_instance

# --- Tool Handlers ---
async def handle_maestro_orchestrate(ctx: mcp.server.fastmcp.Context, task_description: str, context: Dict[str, Any] = None, success_criteria: Dict[str, Any] = None, complexity_level: str = "moderate"):
    log_debug(f"handle_maestro_orchestrate called with task: {task_description}")
    tools_instance = get_maestro_tools_instance()
    try:
        result = await tools_instance.orchestrate_task(
            ctx=ctx,
            task_description=task_description,
            context=context or {},
            success_criteria=success_criteria or {},
            complexity_level=complexity_level
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_orchestrate: {str(e)}")
        return [{"text": f"Error orchestrating task: {str(e)}"}]

async def handle_maestro_iae_discovery(task_type: str = "general", domain_context: str = "", complexity_requirements: Dict[str, Any] = None):
    log_debug(f"handle_maestro_iae_discovery called with task_type: {task_type}")
    tools_instance = get_maestro_tools_instance()
    try:
        result = await tools_instance.discover_integrated_analysis_engines(
            task_type=task_type,
            domain_context=domain_context,
            complexity_requirements=complexity_requirements or {}
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_iae_discovery: {str(e)}")
        return [{"text": f"Error discovering IAE: {str(e)}"}]

async def handle_maestro_tool_selection(request_description: str, available_context: Dict[str, Any] = None, precision_requirements: Dict[str, Any] = None):
    log_debug(f"handle_maestro_tool_selection called with request: {request_description}")
    tools_instance = get_maestro_tools_instance()
    try:
        result = await tools_instance.select_tools(
            request_description=request_description,
            available_context=available_context or {},
            precision_requirements=precision_requirements or {}
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_tool_selection: {str(e)}")
        return [{"text": f"Error selecting tools: {str(e)}"}]

async def handle_maestro_iae(engine_domain: str, computation_type: str, parameters: Dict[str, Any], precision_requirements: str = "machine_precision", validation_level: str = "standard"):
    log_debug(f"handle_maestro_iae called with domain: {engine_domain}, computation: {computation_type}")
    comp_tools = get_computational_tools_instance()
    try:
        result = await comp_tools.run_integrated_analysis(
            engine_domain=engine_domain,
            computation_type=computation_type,
            parameters=parameters,
            precision_requirements=precision_requirements,
            validation_level=validation_level
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_iae: {str(e)}")
        return [{"text": f"Error running IAE: {str(e)}"}]

async def handle_maestro_search(arguments: dict) -> list:
    log_debug(f"handle_maestro_search called with arguments: {arguments}")
    enhanced_tools = get_enhanced_tool_handlers_instance()
    try:
        query = arguments.get("query", "")
        search_type = arguments.get("search_type", "web")
        max_results = arguments.get("max_results", 10)
        
        result = await enhanced_tools.search(
            query=query,
            search_type=search_type,
            max_results=max_results
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_search: {str(e)}")
        return [{"text": f"Error performing search: {str(e)}"}]

async def handle_maestro_scrape(arguments: dict) -> list:
    log_debug(f"handle_maestro_scrape called with arguments: {arguments}")
    enhanced_tools = get_enhanced_tool_handlers_instance()
    try:
        url = arguments.get("url", "")
        elements = arguments.get("elements", [])
        extract_type = arguments.get("extract_type", "text")
        
        result = await enhanced_tools.scrape(
            url=url,
            elements=elements,
            extract_type=extract_type
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_scrape: {str(e)}")
        return [{"text": f"Error scraping content: {str(e)}"}]

async def handle_maestro_execute(arguments: dict) -> list:
    log_debug(f"handle_maestro_execute called with arguments: {arguments}")
    enhanced_tools = get_enhanced_tool_handlers_instance()
    try:
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", 30)
        environment = arguments.get("environment", {})
        
        result = await enhanced_tools.execute(
            command=command,
            timeout=timeout,
            environment=environment
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_execute: {str(e)}")
        return [{"text": f"Error executing command: {str(e)}"}]

async def handle_maestro_error_handler(arguments: dict) -> list:
    log_debug(f"handle_maestro_error_handler called with arguments: {arguments}")
    enhanced_tools = get_enhanced_tool_handlers_instance()
    try:
        error_message = arguments.get("error_message", "")
        error_type = arguments.get("error_type", "general")
        context = arguments.get("context", {})
        
        result = await enhanced_tools.handle_error(
            error_message=error_message,
            error_type=error_type,
            context=context
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_error_handler: {str(e)}")
        return [{"text": f"Error handling error: {str(e)}"}]

async def handle_maestro_temporal_context(arguments: dict) -> list:
    log_debug(f"handle_maestro_temporal_context called with arguments: {arguments}")
    enhanced_tools = get_enhanced_tool_handlers_instance()
    try:
        query = arguments.get("query", "")
        time_range = arguments.get("time_range", "recent")
        domain = arguments.get("domain", "general")
        
        result = await enhanced_tools.analyze_temporal_context(
            query=query,
            time_range=time_range,
            domain=domain
        )
        return [{"text": result}]
    except Exception as e:
        log_debug(f"Error in handle_maestro_temporal_context: {str(e)}")
        return [{"text": f"Error analyzing temporal context: {str(e)}"}]


# --- Tool Registration ---
def _register_tools():
    log_debug("Registering tools")
    return

# Register tools when this module is imported
try:
    _register_tools()
except Exception as e:
    log_debug(f"Error during tool registration: {e}")

# Create a FastAPI app
log_debug("Creating FastAPI app")
fastapi_app = FastAPI(title="Maestro MCP Server", description="Enhanced Workflow Orchestration")

# Add a new lightweight /tools endpoint as the FIRST endpoint
@fastapi_app.get("/tools")
async def lightweight_tools():
    log_debug("💡 Lightweight tools endpoint called")
    response_data = [
        {"name": "maestro_orchestrate", "description": "🎭 Intelligent workflow orchestration for complex tasks"},
        {"name": "maestro_iae_discovery", "description": "🔍 Integrated Analysis Engine discovery for optimal computation selection"},
        {"name": "maestro_tool_selection", "description": "🧰 Intelligent tool selection based on task requirements"},
        {"name": "maestro_iae", "description": "🧠 Integrated Analysis Engine for specialized computational tasks"},
        {"name": "maestro_search", "description": "🔎 Enhanced search capabilities across multiple sources"},
        {"name": "maestro_scrape", "description": "📑 Web scraping functionality with content extraction"},
        {"name": "maestro_execute", "description": "⚙️ Command execution with security controls"},
        {"name": "maestro_error_handler", "description": "🚨 Advanced error handling and recovery"},
        {"name": "maestro_temporal_context", "description": "🕐 Temporal context analysis for time-sensitive information"}
    ]
    log_debug(f"💡 Returning {len(response_data)} tools from lightweight endpoint")
    return response_data

# Super lightweight ping endpoint
@fastapi_app.get("/ping")
async def ping():
    log_debug("🏓 Ping endpoint called")
    return {"ping": "pong", "timestamp": time.time()}

# MCP request/response handler
async def handle_mcp_request(request: Request):
    log_debug("MCP request handler called")
    if request.method == "GET":
        return await lightweight_tools()
    else:
        try:
            body = await request.json()
            tool_name = body.get("tool")
            arguments = body.get("arguments", {})
    
            # Route to the appropriate handler
            if tool_name == "maestro_orchestrate":
                return await handle_maestro_orchestrate(**arguments)
            elif tool_name == "maestro_iae_discovery":
                return await handle_maestro_iae_discovery(**arguments)
            elif tool_name == "maestro_tool_selection":
                return await handle_maestro_tool_selection(**arguments)
            elif tool_name == "maestro_iae":
                return await handle_maestro_iae(**arguments)
            elif tool_name == "maestro_search":
                return await handle_maestro_search(arguments)
            elif tool_name == "maestro_scrape":
                return await handle_maestro_scrape(arguments)
            elif tool_name == "maestro_execute":
                return await handle_maestro_execute(arguments)
            elif tool_name == "maestro_error_handler":
                return await handle_maestro_error_handler(arguments)
            elif tool_name == "maestro_temporal_context":
                return await handle_maestro_temporal_context(arguments)
            else:
                return JSONResponse(
                    content={"error": f"Unknown tool: {tool_name}"},
                    status_code=400
                )
        except Exception as e:
            log_debug(f"Error handling MCP request: {str(e)}")
            return JSONResponse(
                content={"error": f"Error processing request: {str(e)}"},
                status_code=500
            )

# Mount the MCP server at /mcp
@fastapi_app.api_route("/mcp", methods=["GET", "POST", "DELETE", "OPTIONS"])
async def handle_all_mcp_methods(request: Request):
    log_debug(f"/mcp endpoint called with method: {request.method}")
    return await handle_mcp_request(request)

# Add a default route for the root path
@fastapi_app.get("/")
async def root():
    log_debug("Root endpoint called")
    return {
        "name": "Maestro MCP Server",
        "status": "online",
        "endpoints": {
            "tools": "/tools - Tool list",
            "ping": "/ping - Basic server ping",
            "mcp": "/mcp - MCP endpoint",
            "health": "/health - Health check"
        }
    }

# Debug endpoint (remains useful)
@fastapi_app.get("/debug")
async def debug_info():
    log_debug("Debug endpoint called")
    import psutil
    import platform
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "server_info": { "python_version": sys.version, "platform": platform.platform()},
        "memory_usage": { "rss_mb": memory_info.rss / (1024 * 1024)},
        "environment": { "smithery_mode": os.environ.get("SMITHERY_MODE", "false")}
    }

# Add a dedicated healthcheck endpoint for Smithery
@fastapi_app.get("/health")
async def healthcheck():
    log_debug("Healthcheck endpoint called")
    return {"status": "healthy", "timestamp": time.time()}

# The 'app' for Uvicorn is now the FastAPI app
app = fastapi_app
log_debug("Module initialization complete - app ready to serve requests")

if __name__ == "__main__":
    logger.info("Running src/main.py directly")
    print("To run with HTTP server, use: python deploy.py [dev|prod|smithery]")