#!/usr/bin/env python3
"""
HTTP Transport Wrapper for TanukiMCP Maestro - Smithery Compatible
Implements the /mcp endpoint with proper SSE transport as required by Smithery.

Based on successful Smithery deployments:
- Endpoint: /mcp (required)
- Methods: GET, POST, DELETE
- Port: PORT environment variable
- Configuration: Query parameters with dot-notation
"""

import os
import asyncio
import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, JSONResponse, StreamingResponse
from starlette.routing import Route, Mount
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# Lazy load the official MCP server only when needed
_mcp_app = None

def get_mcp_app():
    """Lazy load the MCP server only when tool execution is needed"""
    global _mcp_app
    if _mcp_app is None:
        from mcp_official_server import app as mcp_app
        _mcp_app = mcp_app
    return _mcp_app

# Logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class SmitheryMCPTransport:
    """HTTP transport that wraps the official MCP server for Smithery compatibility"""
    
    def __init__(self):
        # Don't load the MCP server at init - defer until needed
        self._mcp_server = None
    
    def get_mcp_server(self):
        """Lazy load the MCP server only when needed for tool execution"""
        if self._mcp_server is None:
            self._mcp_server = get_mcp_app()
        return self._mcp_server
        
    async def handle_mcp_request(self, request: Request) -> Response:
        """Handle requests to /mcp endpoint"""
        try:
            # Parse configuration from query parameters (Smithery format)
            config = self._parse_config_from_query(str(request.url))
            
            if request.method == "GET":
                # Tool discovery request - treat as tools/list with default id
                return await self._handle_tool_discovery(config, "tools-list-get")
            
            elif request.method == "POST":
                # Tool execution request
                body = await request.body()
                if body:
                    try:
                        data = json.loads(body.decode())
                        # Validate basic JSON-RPC structure
                        if not isinstance(data, dict):
                            return JSONResponse({
                                "jsonrpc": "2.0",
                                "id": None,
                                "error": {
                                    "code": -32600,
                                    "message": "Invalid Request - must be a JSON object"
                                }
                            }, status_code=400)
                        
                        return await self._handle_tool_execution(data, config)
                    except json.JSONDecodeError as e:
                        return JSONResponse({
                            "jsonrpc": "2.0", 
                            "id": None,
                            "error": {
                                "code": -32700,
                                "message": f"Parse error: {str(e)}"
                            }
                        }, status_code=400)
                else:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": None, 
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request - no request body"
                        }
                    }, status_code=400)
            
            elif request.method == "DELETE":
                # Cleanup request
                return JSONResponse({"status": "ok"})
            
            else:
                return JSONResponse({"error": "Method not allowed"}, status_code=405)
                
        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    def _parse_config_from_query(self, url: str) -> Dict[str, Any]:
        """Parse Smithery configuration from query parameters"""
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            # Convert flat params to nested config using dot notation
            config = {}
            for key, values in params.items():
                if values:
                    self._set_nested_value(config, key, values[0])
            
            return config
        except Exception as e:
            logger.warning(f"Failed to parse config from query: {e}")
            return {}
    
    def _set_nested_value(self, obj: Dict, key: str, value: str):
        """Set nested dictionary value using dot notation"""
        parts = key.split('.')
        current = obj
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for planned tools"""
        descriptions = {
            "maestro_search": "Enhanced web search with LLM-powered analysis and filtering",
            "maestro_scrape": "Intelligent web scraping with content extraction and structured data processing", 
            "maestro_execute": "Secure code and workflow execution with validation",
            "maestro_temporal_context": "Time-aware reasoning and context analysis",
            "maestro_error_handler": "Intelligent error analysis and recovery suggestions"
        }
        return descriptions.get(tool_name, "Advanced AI capability")
    
    async def _handle_tool_discovery(self, config: Dict[str, Any], request_id: Optional[str] = None) -> JSONResponse:
        """Handle tool discovery (list_tools) - must be fast for Smithery"""
        try:
            # Use pure dictionary definitions (zero imports, instant loading)
            from static_tools_dict import STATIC_TOOLS_DICT
            
            # Tools are already in dict format - no conversion needed
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "tools": STATIC_TOOLS_DICT
                }
            }
            
            # Add id if provided (required for JSON-RPC)
            if request_id is not None:
                response["id"] = request_id
            
            return JSONResponse(response)
            
        except Exception as e:
            logger.error(f"Error in tool discovery: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            
            # Add id if provided (required for JSON-RPC)
            if request_id is not None:
                error_response["id"] = request_id
                
            return JSONResponse(error_response, status_code=500)
    
    async def _handle_initialize(self, params: Dict[str, Any], request_id: Optional[str]) -> JSONResponse:
        """Handle MCP initialize method"""
        try:
            # Extract client info
            client_info = params.get("clientInfo", {})
            protocol_version = params.get("protocolVersion", "2024-11-05")
            
            # Return server capabilities
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "tanukimcp-maestro",
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {},
                        "logging": {},
                        "experimental": {}
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Error in initialize: {e}")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }, status_code=500)
    
    async def _execute_tool_directly(self, tool_name: str, arguments: dict):
        """Execute tools directly using the same logic as the MCP server"""
        try:
            from mcp.types import TextContent, CallToolResult
            
            if tool_name.startswith("maestro_") and tool_name != "maestro_iae":
                # Handle MaestroTools
                from src.maestro_tools import MaestroTools
                from mcp.server.fastmcp import Context
                tools = MaestroTools()
                
                # Create a minimal context for the tools
                ctx = Context()
                
                if tool_name == "maestro_orchestrate":
                    result_text = await tools.orchestrate_task(
                        ctx=ctx,
                        task_description=arguments.get("task_description", ""),
                        context=arguments.get("context", {}),
                        complexity_level=arguments.get("complexity_level", "moderate"),
                        quality_threshold=arguments.get("quality_threshold", 0.8),
                        resource_level=arguments.get("resource_level", "moderate"),
                        reasoning_focus=arguments.get("reasoning_focus", "auto"),
                        validation_rigor=arguments.get("validation_rigor", "standard"),
                        max_iterations=arguments.get("max_iterations", 3),
                        domain_specialization=arguments.get("domain_specialization", ""),
                        enable_collaboration_fallback=arguments.get("enable_collaboration_fallback", True)
                    )
                    return CallToolResult(content=[TextContent(type="text", text=result_text)])
                
                elif tool_name == "maestro_collaboration_response":
                    result_text = await tools.handle_collaboration_response(
                        collaboration_id=arguments.get("collaboration_id", ""),
                        responses=arguments.get("responses", {}),
                        additional_context=arguments.get("additional_guidance", {}),
                        user_preferences={},
                        approval_status=arguments.get("approval_status", "approved"),
                        confidence_level=1.0
                    )
                    return CallToolResult(content=[TextContent(type="text", text=result_text)])
                
                elif tool_name == "maestro_iae_discovery":
                    result = await tools._handle_iae_discovery(arguments)
                    return CallToolResult(content=result)
                
                elif tool_name == "maestro_tool_selection":
                    result = await tools._handle_tool_selection(arguments)
                    return CallToolResult(content=result)
                
                elif tool_name == "maestro_search":
                    result = await tools._handle_maestro_search(arguments)
                    return CallToolResult(content=result)
                
                elif tool_name == "maestro_scrape":
                    result = await tools._handle_maestro_scrape(arguments)
                    return CallToolResult(content=result)
                
                elif tool_name == "maestro_execute":
                    result = await tools._handle_maestro_execute(arguments)
                    return CallToolResult(content=result)
                
                elif tool_name == "maestro_temporal_context":
                    result = await tools._handle_maestro_temporal_context(arguments)
                    return CallToolResult(content=result)
                
                elif tool_name == "maestro_error_handler":
                    result = await tools._handle_maestro_error_handler(arguments)
                    return CallToolResult(content=result)
                
                else:
                    return CallToolResult(
                        content=[TextContent(
                            type="text", 
                            text=f"Error: Unknown maestro tool '{tool_name}'"
                        )]
                    )
            
            elif tool_name == "maestro_iae":
                # Handle ComputationalTools  
                from src.computational_tools import ComputationalTools
                tools = ComputationalTools()
                
                result = await tools.handle_tool_call(tool_name, arguments)
                
                if isinstance(result, list):
                    return CallToolResult(content=result)
                else:
                    return CallToolResult(content=[TextContent(type="text", text=str(result))])
            
            elif tool_name == "get_available_engines":
                # Handle engine discovery
                from src.computational_tools import ComputationalTools
                tools = ComputationalTools()
                
                # get_available_engines doesn't take parameters
                engines = tools.get_available_engines()
                
                import json
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(engines, indent=2))]
                )
            
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text=f"Error: Unknown tool '{tool_name}'"
                    )]
                )
        
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            from mcp.types import TextContent, CallToolResult
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Error executing {tool_name}: {str(e)}"
                )]
            )
    
    async def _handle_tool_execution(self, data: Dict[str, Any], config: Dict[str, Any]) -> JSONResponse:
        """Handle tool execution requests"""
        try:
            method = data.get("method")
            params = data.get("params", {})
            request_id = data.get("id")
            
            # Validate required JSON-RPC fields
            if method is None:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request - missing 'method' field"
                    }
                }, status_code=400)
            
            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                # Execute the tool directly using our tool handlers
                result = await self._execute_tool_directly(tool_name, arguments)
                
                # Convert result to JSON format
                if hasattr(result, 'content'):
                    # Handle CallToolResult objects
                    content_data = []
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            content_data.append({
                                "type": "text",
                                "text": content_item.text
                            })
                        elif hasattr(content_item, 'data'):
                            content_data.append({
                                "type": "image",
                                "data": content_item.data,
                                "mimeType": getattr(content_item, 'mimeType', 'image/png')
                            })
                    
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": content_data,
                            "isError": False
                        }
                    })
                else:
                    # Handle direct string results
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": str(result)
                            }],
                            "isError": False
                        }
                    })
            
            elif method == "tools/list":
                # Same as GET /mcp for tool discovery
                return await self._handle_tool_discovery(config, request_id)
            
            elif method == "initialize":
                # Handle MCP initialization
                return await self._handle_initialize(params, request_id)
            
            elif method == "notifications/initialized":
                # Handle initialization notification (notifications don't have responses)
                # Return 204 No Content for notifications
                return Response(status_code=204)
            
            elif method == "ping":
                # Handle ping requests
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {}
                })
            
            elif method == "resources/list":
                # Handle resource listing (empty for now)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": []
                    }
                })
            
            elif method == "prompts/list":
                # Handle prompt listing (empty for now)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "prompts": []
                    }
                })
            
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }, status_code=400)
                
        except Exception as e:
            logger.error(f"Error in tool execution: {e}")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }, status_code=500)

# Create transport instance
transport = SmitheryMCPTransport()

# Health check endpoint
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint"""
    return JSONResponse({"status": "healthy", "service": "tanukimcp-maestro"})

# Root endpoint
async def root(request: Request) -> JSONResponse:
    """Root endpoint with service information"""
    return JSONResponse({
        "service": "TanukiMCP Maestro",
        "version": "1.0.0",
        "description": "Meta-Agent Ensemble for Systematic Task Reasoning and Orchestration",
        "endpoints": {
            "/mcp": "MCP protocol endpoint",
            "/health": "Health check",
            "/": "Service information"
        },
        "tools": 11  # Static count - no need to load server for this
    })

# Create Starlette application
routes = [
    Route("/", root, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/mcp", transport.handle_mcp_request, methods=["GET", "POST", "DELETE"]),
]

app = Starlette(routes=routes)

# Add CORS middleware for web compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    # Get port from environment (required by Smithery)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting TanukiMCP Maestro HTTP server on {host}:{port}")
    logger.info("Endpoints available:")
    logger.info(f"  GET/POST/DELETE http://{host}:{port}/mcp - MCP protocol")
    logger.info(f"  GET http://{host}:{port}/health - Health check")
    logger.info(f"  GET http://{host}:{port}/ - Service info")
    
    uvicorn.run(
        "mcp_http_transport:app",
        host=host,
        port=port,
        log_level="warning",
        access_log=False
    ) 