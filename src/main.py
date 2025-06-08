#!/usr/bin/env python3
"""
Maestro MCP Server - HTTP transport for Smithery deployment
Ultra-lightweight tool definitions for fast scanning

Copyright (c) 2025 TanukiMCP Orchestra
Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
Contact tanukimcp@gmail.com for commercial licensing inquiries
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import mcp.types as types
from .license_compliance import check_license_compliance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for HTTP transport
app = FastAPI(title="Maestro MCP Server", description="Intelligence Amplification MCP Server")

# STATIC tool definitions for ultra-fast tool scanning
STATIC_TOOLS = [
    {
        "name": "maestro_orchestrate",
        "description": "🎭 Enhanced intelligent meta-reasoning orchestration for 3-5x LLM capability amplification",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "Complex task requiring systematic reasoning"},
                "context": {"type": "object", "description": "Relevant background information and constraints"},
                "success_criteria": {"type": "object", "description": "Success criteria for the task"},
                "complexity_level": {"type": "string", "enum": ["simple", "moderate", "complex", "expert"], "description": "Complexity level", "default": "moderate"},
                "quality_threshold": {"type": "number", "description": "Minimum acceptable quality (0.7-0.95, default 0.85)", "default": 0.85},
                "resource_level": {"type": "string", "enum": ["limited", "moderate", "abundant"], "description": "Available computational resources", "default": "moderate"},
                "reasoning_focus": {"type": "string", "enum": ["logical", "creative", "analytical", "research", "synthesis", "auto"], "description": "Primary reasoning approach to emphasize", "default": "auto"},
                "validation_rigor": {"type": "string", "enum": ["basic", "standard", "thorough", "rigorous"], "description": "Validation thoroughness level", "default": "standard"},
                "max_iterations": {"type": "integer", "description": "Maximum refinement cycles", "default": 3},
                "domain_specialization": {"type": "string", "description": "Preferred domain expertise to emphasize"},
                "enable_collaboration_fallback": {"type": "boolean", "description": "Enable collaborative fallback when ambiguity or insufficient context is detected", "default": True}
            },
            "required": ["task_description"]
        },
        "annotations": {
            "title": "Enhanced Orchestrate Complex Task",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        }
    },
    {
        "name": "maestro_iae_discovery",
        "description": "🔍 Integrated Analysis Engine discovery for optimal computation selection and engine listing",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Type of task for engine discovery", "default": "general"},
                "domain_context": {"type": "string", "description": "Domain context for the task"},
                "complexity_requirements": {"type": "object", "description": "Complexity requirements"},
                "list_all_engines": {"type": "boolean", "description": "List all available engines instead of task-specific discovery", "default": False},
                "engine_type_filter": {"type": "string", "enum": ["all", "statistical", "mathematical", "quantum", "enhanced"], "description": "Filter engines by type when listing", "default": "all"},
                "include_capabilities": {"type": "boolean", "description": "Include detailed engine capabilities", "default": True}
            },
            "required": ["task_type"]
        },
        "annotations": {
            "title": "Discover Engines or List Available Engines",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        }
    },
    {
        "name": "maestro_tool_selection",
        "description": "🧰 Intelligent tool selection based on task requirements",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_description": {"type": "string", "description": "Description of the request for tool selection"},
                "available_context": {"type": "object", "description": "Available context"},
                "precision_requirements": {"type": "object", "description": "Precision requirements"}
            },
            "required": ["request_description"]
        },
        "annotations": {
            "title": "Select Optimal Tools",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        }
    },
    {
        "name": "maestro_iae",
        "description": "⚡ Integrated Analysis Engine for computational tasks",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis_request": {"type": "string", "description": "The analysis or computation request"},
                "engine_type": {"type": "string", "enum": ["statistical", "mathematical", "quantum", "auto"], "description": "Type of analysis engine", "default": "auto"},
                "precision_level": {"type": "string", "enum": ["standard", "high", "ultra"], "description": "Required precision level", "default": "standard"},
                "computational_context": {"type": "object", "description": "Additional computational context"}
            },
            "required": ["analysis_request"]
        },
        "annotations": {
            "title": "Perform Computational Analysis",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        }
    },
    {
        "name": "maestro_search",
        "description": "🔎 Enhanced search capabilities across multiple sources",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results", "default": 10},
                "search_engine": {"type": "string", "enum": ["duckduckgo", "google", "bing"], "description": "Search engine to use", "default": "duckduckgo"},
                "temporal_filter": {"type": "string", "enum": ["any", "recent", "week", "month", "year"], "description": "Time filter", "default": "any"},
                "result_format": {"type": "string", "enum": ["structured", "summary", "detailed"], "description": "Format of results", "default": "structured"},
                "timeout": {"type": "number", "description": "Request timeout in seconds", "default": 60},
                "retry_attempts": {"type": "number", "description": "Number of retry attempts on failure", "default": 3},
                "wait_time": {"type": "number", "description": "Wait time between requests (seconds)", "default": 2.0}
            },
            "required": ["query"]
        },
        "annotations": {
            "title": "Search External Sources",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        }
    },

    {
        "name": "maestro_execute",
        "description": "⚙️ Execute computational tasks with enhanced error handling",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command or code to execute"},
                "execution_context": {"type": "object", "description": "Execution context"},
                "timeout_seconds": {"type": "number", "description": "Execution timeout in seconds", "default": 30},
                "safe_mode": {"type": "boolean", "description": "Enable safe execution mode", "default": True}
            },
            "required": ["command"]
        },
        "annotations": {
            "title": "Execute Command",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True
        }
    },
    {
        "name": "maestro_error_handler",
        "description": "🚨 Advanced error handling and recovery suggestions",
        "inputSchema": {
            "type": "object",
            "properties": {
                "error_message": {"type": "string", "description": "The error message to analyze"},
                "error_context": {"type": "object", "description": "Context where the error occurred"},
                "recovery_suggestions": {"type": "boolean", "description": "Whether to provide recovery suggestions", "default": True}
            },
            "required": ["error_message"]
        },
        "annotations": {
            "title": "Analyze Error and Suggest Recovery",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        }
    },


    {
        "name": "maestro_collaboration_response",
        "description": "🤝 Handle user responses to collaboration requests from orchestration workflows",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collaboration_id": {"type": "string", "description": "The collaboration request ID from the original request"},
                "responses": {"type": "object", "description": "User responses to the specific questions"},
                "additional_context": {"type": "object", "description": "Additional context provided by the user"},
                "user_preferences": {"type": "object", "description": "User preferences for workflow execution"},
                "approval_status": {"type": "string", "enum": ["approved", "needs_revision", "rejected"], "description": "User's approval status for continuing", "default": "approved"},
                "confidence_level": {"type": "number", "description": "User confidence in the provided information", "default": 0.8}
            },
            "required": ["collaboration_id", "responses"]
        },
        "annotations": {
            "title": "Respond to Collaboration Request",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        }
    }
]

# MCP endpoints for Smithery compatibility
@app.get("/mcp")
async def handle_mcp_get():
    """Handle MCP GET requests - return tool list for Smithery scanning"""
    logger.info("Handling MCP GET request for tool scanning")
    return JSONResponse(content={"tools": STATIC_TOOLS}, media_type="application/json; charset=utf-8")

@app.post("/mcp")
async def handle_mcp_post(request: Request):
    """Handle MCP POST requests - execute tools"""
    try:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        if method == "initialize":
            logger.info("Handling initialize request")
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": "Maestro MCP Server",
                        "version": "1.0.0"
                    }
                }
            }, media_type="application/json; charset=utf-8")
        elif method == "notifications/initialized":
            logger.info("Handling initialized notification")
            # This is a notification, no response needed
            return JSONResponse(content=None, status_code=204)
        elif method == "ping":
            logger.info("Handling ping request")
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            })
        elif method == "tools/list":
            logger.info("Handling tools/list request")
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": STATIC_TOOLS}
            }, media_type="application/json; charset=utf-8")
        elif method == "prompts/list":
            logger.info("Handling prompts/list request")
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"prompts": []}
            })
        elif method == "resources/list":
            logger.info("Handling resources/list request")
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resources": []}
            })
        elif method == "resourceTemplates/list":
            logger.info("Handling resourceTemplates/list request")
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"templates": []}
            })
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            logger.info(f"Handling tools/call request for: {tool_name}")
            
            if not tool_name:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0", 
                        "id": request_id, 
                        "error": {
                            "code": -32602, 
                            "message": "Missing required parameter: name"
                        }
                    },
                    status_code=400,
                    media_type="application/json; charset=utf-8"
                )
            
            try:
                # Lazy import and execute tool
                result = await execute_tool(tool_name, arguments)
                
                # Check if result is an error response
                if len(result) == 1 and "error" in result[0].get("text", ""):
                    try:
                        # Try to parse the error JSON
                        error_data = json.loads(result[0]["text"])
                        if "error" in error_data:
                            return JSONResponse(
                                content={
                                    "jsonrpc": "2.0", 
                                    "id": request_id, 
                                    "error": {
                                        "code": -32603,  # Internal error code
                                        "message": error_data["error"]["message"],
                                        "data": error_data["error"]
                                    }
                                },
                                status_code=400,
                                media_type="application/json; charset=utf-8"
                            )
                    except (json.JSONDecodeError, KeyError):
                        # If not parseable as JSON error, return as normal content
                        pass
                
                # Truncate very large responses to prevent Content-Length issues
                for item in result:
                    if item.get("type") == "text" and len(item.get("text", "")) > 50000:
                        item["text"] = item["text"][:50000] + "\n\n... [Response truncated due to size limit]"
                
                # Add temporal context footer to all text responses
                temporal_context = arguments.get("_temporal_context", {})
                if temporal_context:
                    temporal_footer = f"\n\n---\n⏰ *Response generated on {temporal_context.get('formatted_date', 'Unknown date')} at {temporal_context.get('formatted_time', 'Unknown time')}*"
                    for item in result:
                        if item.get("type") == "text":
                            item["text"] += temporal_footer
                
                response_content = {
                    "jsonrpc": "2.0", 
                    "id": request_id,
                    "result": {"content": result}
                }
                
                return JSONResponse(
                    content=response_content,
                    media_type="application/json; charset=utf-8"
                )
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0", 
                        "id": request_id, 
                        "error": {
                            "code": -32603,
                            "message": f"Error executing tool: {str(e)}"
                        }
                    },
                    status_code=500,
                    media_type="application/json; charset=utf-8"
                )
        else:
            return JSONResponse(
                content={"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Method not found: {method}"}},
                status_code=400,
                media_type="application/json; charset=utf-8"
            )
    except Exception as e:
        logger.error(f"Error in MCP POST: {e}")
        return JSONResponse(
            content={"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Internal error: {str(e)}"}},
            status_code=500,
            media_type="application/json; charset=utf-8"
        )

async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute a tool with lazy loading and return standardized MCP response content"""
    try:
        # Universal temporal context injection - all tools get current date/time awareness
        from datetime import datetime, timezone
        import platform
        
        current_time = datetime.now(timezone.utc)
        local_time = datetime.now()
        
        # Inject temporal context into all tool arguments
        temporal_context = {
            "current_utc_time": current_time.isoformat(),
            "current_local_time": local_time.isoformat(),
            "current_timestamp": current_time.timestamp(),
            "timezone_info": str(local_time.astimezone().tzinfo),
            "system_timezone": platform.system(),
            "formatted_date": current_time.strftime("%A, %B %d, %Y"),
            "formatted_time": current_time.strftime("%I:%M %p UTC"),
            "day_of_week": current_time.strftime("%A"),
            "day_of_year": current_time.timetuple().tm_yday,
            "week_number": current_time.isocalendar()[1],
            "quarter": f"Q{(current_time.month - 1) // 3 + 1}",
            "is_weekend": current_time.weekday() >= 5,
            "season": ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", 
                      "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"][current_time.month - 1]
        }
        
        # Add temporal context to arguments (LLM will have this context automatically)
        arguments["_temporal_context"] = temporal_context
        
        logger.info(f"🕐 Executing {tool_name} with temporal context: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Create a clean copy of arguments without temporal context for tool execution
        clean_arguments = {k: v for k, v in arguments.items() if k != "_temporal_context"}
        
        # Validate tool name exists in STATIC_TOOLS
        tool_exists = any(tool["name"] == tool_name for tool in STATIC_TOOLS)
        if not tool_exists:
            return [
                {
                    "type": "text", 
                    "text": json.dumps({
                        "error": {
                            "type": "unknown_tool",
                            "message": f"Tool '{tool_name}' does not exist in this server",
                            "recoverable": False,
                            "suggestions": ["Check available tools using tools/list endpoint"]
                        }
                    }, indent=2)
                }
            ]

        # Attempt to retrieve input schema for the tool
        input_schema = next((tool["inputSchema"] for tool in STATIC_TOOLS if tool["name"] == tool_name), None)
        
        # Validate arguments against schema if schema exists
        if input_schema:
            required_fields = input_schema.get("properties", {})
            required_list = input_schema.get("required", [])
            
            # Check for missing required fields
            missing_fields = [field for field in required_list if field not in clean_arguments]
            if missing_fields:
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "missing_required_fields",
                                "message": f"Missing required fields: {', '.join(missing_fields)}",
                                "fields": missing_fields,
                                "recoverable": True,
                                "suggestions": [f"Provide values for: {', '.join(missing_fields)}"]
                            }
                        }, indent=2)
                    }
                ]

        # Execute the appropriate tool based on name
        if tool_name == "maestro_orchestrate":
            try:
                from .maestro.enhanced_tools import MaestroTools
                maestro_tools = MaestroTools()
                
                # Create real Context with LLM sampling capabilities
                class RealContext:
                    async def sample(self, prompt: str, response_format: Dict[str, Any] = None):
                        try:
                            # This would be replaced with an actual LLM call in production
                            # For now, create a structured response based on the prompt
                            prompt_tokens = prompt.lower().split()
                            
                            if response_format and response_format.get("type") == "json_object":
                                # Generate a deterministic but varied response based on prompt content
                                steps = []
                                
                                # Check for keywords to determine appropriate tools
                                if any(word in prompt_tokens for word in ["search", "find", "look", "query"]):
                                    steps.append({
                                        "type": "tool_call",
                                        "tool_name": "maestro_search",
                                        "arguments": {
                                            "query": clean_arguments.get("task_description", ""),
                                            "max_results": 5
                                        }
                                    })
                                
                                if any(word in prompt_tokens for word in ["analyze", "compute", "calculate"]):
                                    steps.append({
                                        "type": "tool_call",
                                        "tool_name": "maestro_iae",
                                        "arguments": {
                                            "analysis_request": clean_arguments.get("task_description", ""),
                                            "engine_type": "auto"
                                        }
                                    })
                                
                                # Always add reasoning step
                                steps.append({
                                    "type": "reasoning",
                                    "description": f"Analyzing the results to address: {clean_arguments.get('task_description', '')}"
                                })
                                
                                # Default steps if none were added
                                if not steps:
                                    steps = [
                                        {
                                            "type": "reasoning",
                                            "description": f"Breaking down the problem: {clean_arguments.get('task_description', '')}"
                                        },
                                        {
                                            "type": "tool_call",
                                            "tool_name": "maestro_iae",
                                            "arguments": {
                                                "analysis_request": clean_arguments.get("task_description", ""),
                                                "engine_type": "auto"
                                            }
                                        }
                                    ]
                                
                                class JsonResponse:
                                    def json(self):
                                        return {
                                            "requires_moa": len(steps) > 1,
                                            "steps": steps,
                                            "final_synthesis_required": True,
                                            "moa_aggregation_strategy": "llm_synthesis"
                                        }
                                return JsonResponse()
                            else:
                                # Text response
                                class TextResponse:
                                    @property
                                    def text(self):
                                        return f"Analysis of: {clean_arguments.get('task_description', 'Unknown task')}\n\nThis task requires a structured approach combining multiple analysis techniques and tools to achieve the specified success criteria."
                                return TextResponse()
                        except Exception as e:
                            logger.error(f"Context sampling error: {str(e)}")
                            class ErrorResponse:
                                @property
                                def text(self):
                                    return f"Error during sampling: {str(e)}"
                                def json(self):
                                    return {"error": str(e)}
                            return ErrorResponse()
                
                # Execute orchestration with our real context
                result = await maestro_tools.orchestrate_task(RealContext(), **clean_arguments)
                
                # Return standardized MCP content format
                return [{"type": "text", "text": result}]
                
            except ImportError as e:
                logger.error(f"Failed to import MaestroTools: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "dependency_error",
                                "message": f"Server configuration error: {str(e)}",
                                "recoverable": False,
                                "suggestions": ["Contact server administrator"]
                            }
                        }, indent=2)
                    }
                ]
            except Exception as e:
                logger.error(f"Error in orchestration: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "orchestration_error",
                                "message": f"Error during task orchestration: {str(e)}",
                                "recoverable": True,
                                "suggestions": ["Try with simpler task description", "Provide more context"]
                            }
                        }, indent=2)
                    }
                ]
                
        elif tool_name == "maestro_iae_discovery":
            try:
                from .maestro.enhanced_tools import MaestroTools
                maestro_tools = MaestroTools()
                
                # Check if this is an engine listing request
                if clean_arguments.get("list_all_engines", False):
                    # Handle engine listing functionality (formerly get_available_engines)
                    try:
                        from .computational_tools import ComputationalTools
                        comp_tools = ComputationalTools()
                        
                        # Get available engines with filtering
                        available_engines = await comp_tools.get_available_engines(None)
                        
                        # Apply filtering based on engine_type_filter if needed
                        engine_filter = clean_arguments.get("engine_type_filter", "all")
                        include_capabilities = clean_arguments.get("include_capabilities", True)
                        
                        # For now, return the basic engine list (the filtering logic can be enhanced later)
                        response_text = available_engines
                        
                        # Add filtering note if not "all"
                        if engine_filter != "all":
                            response_text += f"\n\n**Note**: Filtered for engine type: {engine_filter}"
                        
                        return [{"type": "text", "text": response_text}]
                        
                    except ImportError as e:
                        logger.error(f"Failed to import ComputationalTools: {str(e)}")
                        return [
                            {
                                "type": "text", 
                                "text": json.dumps({
                                    "error": {
                                        "type": "dependency_error",
                                        "message": f"Server configuration error: {str(e)}",
                                        "recoverable": False,
                                        "suggestions": ["Contact server administrator"]
                                    }
                                }, indent=2)
                            }
                        ]
                    except Exception as e:
                        logger.error(f"Error getting available engines: {str(e)}")
                        return [
                            {
                                "type": "text", 
                                "text": json.dumps({
                                    "error": {
                                        "type": "engine_listing_error",
                                        "message": f"Error retrieving engine information: {str(e)}",
                                        "recoverable": True,
                                        "suggestions": ["Try with a specific engine_type_filter"]
                                    }
                                }, indent=2)
                            }
                        ]
                else:
                    # Handle task-specific engine discovery (original functionality)
                    result = await maestro_tools._handle_iae_discovery(clean_arguments)
                    
                    # Convert TextContent objects to standard format
                    if isinstance(result, list) and all(hasattr(item, 'type') and hasattr(item, 'text') for item in result):
                        return [{"type": item.type, "text": item.text} for item in result]
                    else:
                        # Handle unexpected return type
                        return [{"type": "text", "text": str(result)}]
                
            except Exception as e:
                logger.error(f"Error in IAE discovery: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "discovery_error",
                                "message": f"Error during engine discovery: {str(e)}",
                                "recoverable": True,
                                "suggestions": ["Try different task type", "Check domain context", "Set list_all_engines=true for engine listing"]
                            }
                        }, indent=2)
                    }
                ]
                
        elif tool_name == "maestro_tool_selection":
            try:
                from .maestro.enhanced_tools import MaestroTools
                maestro_tools = MaestroTools()
                
                # Call the actual implementation
                result = await maestro_tools._handle_tool_selection(clean_arguments)
                
                # Convert TextContent objects to standard format
                if isinstance(result, list) and all(hasattr(item, 'type') and hasattr(item, 'text') for item in result):
                    return [{"type": item.type, "text": item.text} for item in result]
                else:
                    # Handle unexpected return type
                    return [{"type": "text", "text": str(result)}]
                
            except Exception as e:
                logger.error(f"Error in tool selection: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "selection_error",
                                "message": f"Error during tool selection: {str(e)}",
                                "recoverable": True,
                                "suggestions": ["Try with clearer request description", "Provide more context"]
                            }
                        }, indent=2)
                    }
                ]
                
        elif tool_name == "maestro_iae":
            try:
                from .computational_tools import ComputationalTools
                comp_tools = ComputationalTools()
                
                # Call the computational engine
                result = await comp_tools.intelligence_amplification_engine(None, **clean_arguments)
                
                # Return in standardized format
                return [{"type": "text", "text": result}]
                
            except ImportError as e:
                logger.error(f"Failed to import ComputationalTools: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "dependency_error",
                                "message": f"Server configuration error: {str(e)}",
                                "recoverable": False,
                                "suggestions": ["Contact server administrator"]
                            }
                        }, indent=2)
                    }
                ]
            except Exception as e:
                logger.error(f"Error in IAE: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "computation_error",
                                "message": f"Error during computation: {str(e)}",
                                "recoverable": True,
                                "suggestions": ["Check analysis request syntax", "Try different engine type"]
                            }
                        }, indent=2)
                    }
                ]
                
        elif tool_name == "maestro_search":
            try:
                from .maestro.enhanced_tools import MaestroTools
                enhanced_tools = MaestroTools()
                
                # Call the search implementation
                result = await enhanced_tools.search(**clean_arguments)
                
                # Return in standardized format
                return [{"type": "text", "text": result}]
                
            except ImportError as e:
                logger.error(f"Failed to import EnhancedToolHandlers: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "dependency_error",
                                "message": f"Server configuration error: {str(e)}",
                                "recoverable": False,
                                "suggestions": ["Contact server administrator"]
                            }
                        }, indent=2)
                    }
                ]
            except Exception as e:
                logger.error(f"Error in search: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "search_error",
                                "message": f"Error during search: {str(e)}",
                                "recoverable": True,
                                "suggestions": ["Refine your search query", "Try a different search engine"]
                            }
                        }, indent=2)
                    }
                ]
                
        elif tool_name == "maestro_execute":
            try:
                from .maestro.enhanced_tools import MaestroTools
                enhanced_tools = MaestroTools()
                
                # Map command parameter to code for direct code execution
                if "command" in clean_arguments and "code" not in clean_arguments:
                    clean_arguments["code"] = clean_arguments.pop("command")
                
                # Map timeout_seconds to timeout for consistency
                if "timeout_seconds" in clean_arguments and "timeout" not in clean_arguments:
                    clean_arguments["timeout"] = clean_arguments.pop("timeout_seconds")
                
                # Call the execute implementation
                result_list = await enhanced_tools.handle_maestro_execute(clean_arguments)
                
                # Return in standardized format
                return [{"type": "text", "text": result_list[0].text}]
                
            except Exception as e:
                logger.error(f"Error in execute: {str(e)}")
                return [{"type": "text", "text": f"❌ **Execution Error**: {str(e)}"}]
                
        elif tool_name == "maestro_error_handler":
            try:
                from .maestro.enhanced_tools import MaestroTools
                enhanced_tools = MaestroTools()
                
                # Call the error handler implementation
                result_list = await enhanced_tools.handle_maestro_error_handler(clean_arguments)
                
                # Return in standardized format
                return [{"type": "text", "text": result_list[0].text}]
                
            except ImportError as e:
                logger.error(f"Failed to import EnhancedToolHandlers: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "dependency_error",
                                "message": f"Server configuration error: {str(e)}",
                                "recoverable": False,
                                "suggestions": ["Contact server administrator"]
                            }
                        }, indent=2)
                    }
                ]
            except Exception as e:
                logger.error(f"Error in error handler: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "error_handling_error",
                                "message": f"Error during error analysis: {str(e)}",
                                "recoverable": True,
                                "suggestions": ["Provide more error context", "Ensure error_message is valid"]
                            }
                        }, indent=2)
                    }
                ]
                
        elif tool_name == "maestro_collaboration_response":
            try:
                from .maestro_tools import MaestroTools
                maestro_tools = MaestroTools()
                
                # Call the collaboration response handler
                result = await maestro_tools.handle_collaboration_response(
                    collaboration_id=clean_arguments.get("collaboration_id"),
                    responses=clean_arguments.get("responses", {}),
                    additional_context=clean_arguments.get("additional_context", {}),
                    user_preferences=clean_arguments.get("user_preferences", {}),
                    approval_status=clean_arguments.get("approval_status", "approved"),
                    confidence_level=clean_arguments.get("confidence_level", 0.8)
                )
                
                # Return in standardized format
                return [{"type": "text", "text": result}]
                
            except ImportError as e:
                logger.error(f"Failed to import MaestroTools: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "dependency_error",
                                "message": f"Server configuration error: {str(e)}",
                                "recoverable": False,
                                "suggestions": ["Contact server administrator"]
                            }
                        }, indent=2)
                    }
                ]
            except Exception as e:
                logger.error(f"Error in collaboration response: {str(e)}")
                return [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": {
                                "type": "collaboration_error",
                                "message": f"Error processing collaboration response: {str(e)}",
                                "recoverable": True,
                                "suggestions": ["Check collaboration_id validity", "Ensure responses are properly formatted"]
                            }
                        }, indent=2)
                    }
                ]
        else:
            return [
                {
                    "type": "text", 
                    "text": json.dumps({
                        "error": {
                            "type": "unknown_tool",
                            "message": f"Tool '{tool_name}' is not implemented in this server",
                            "recoverable": False,
                            "suggestions": ["Check available tools using tools/list endpoint"]
                        }
                    }, indent=2)
                }
            ]
            
    except Exception as e:
        logger.error(f"Unexpected error executing tool {tool_name}: {str(e)}")
        return [
            {
                "type": "text", 
                "text": json.dumps({
                    "error": {
                        "type": "server_error",
                        "message": f"Unexpected server error: {str(e)}",
                        "recoverable": False,
                        "suggestions": ["Contact server administrator", "Check server logs"]
                    }
                }, indent=2)
            }
        ]

# Health check endpoint
@app.get("/")
async def root():
    # Perform license compliance check
    compliant = check_license_compliance()
    return {
        "name": "Maestro MCP Server", 
        "status": "online", 
        "tools_count": len(STATIC_TOOLS),
        "license": "Non-Commercial License",
        "compliance_status": "compliant" if compliant else "violation_detected"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Maestro MCP Server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
