#!/usr/bin/env python3
# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Maestro MCP Server - stdio transport implementation
Ultra-lightweight with static tool definitions for fast Smithery scanning
"""

import asyncio
import logging
import sys
import os
from typing import Any, Dict, List

# Core MCP imports - minimal for fastest startup
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Configure minimal logging for production - Smithery requires instant tool scanning
logging.basicConfig(
    level=logging.WARNING,  # Reduce verbosity for faster scanning
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Create the server instance
server = Server("maestro-mcp", version="2.0.0")

# NO IMPORTS FROM SRC/ AT MODULE LEVEL - Everything deferred to call time
# This ensures ultra-fast startup for Smithery tool scanning

# STATIC TOOL DEFINITIONS - Pre-computed for instant scanning
STATIC_TOOLS = [
    types.Tool(
        name="maestro_orchestrate",
        description="🎭 Enhanced intelligent meta-reasoning orchestration for 3-5x LLM capability amplification",
        inputSchema={
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Complex task requiring systematic reasoning"
                },
                "context": {
                    "type": "object",
                    "description": "Relevant background information and constraints",
                    "additionalProperties": True
                },
                "success_criteria": {
                    "type": "object",
                    "description": "Success criteria for the task",
                    "additionalProperties": True
                },
                "complexity_level": {
                    "type": "string",
                    "enum": ["simple", "moderate", "complex", "expert"],
                    "description": "Complexity level of the task",
                    "default": "moderate"
                },
                "quality_threshold": {
                    "type": "number",
                    "minimum": 0.7,
                    "maximum": 0.95,
                    "description": "Minimum acceptable quality (0.7-0.95, default 0.85)",
                    "default": 0.85
                },
                "resource_level": {
                    "type": "string",
                    "enum": ["limited", "moderate", "abundant"],
                    "description": "Available computational resources",
                    "default": "moderate"
                },
                "reasoning_focus": {
                    "type": "string",
                    "enum": ["logical", "creative", "analytical", "research", "synthesis", "auto"],
                    "description": "Primary reasoning approach to emphasize",
                    "default": "auto"
                },
                "validation_rigor": {
                    "type": "string",
                    "enum": ["basic", "standard", "thorough", "rigorous"],
                    "description": "Validation thoroughness level",
                    "default": "standard"
                },
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maximum refinement cycles",
                    "default": 3
                },
                "domain_specialization": {
                    "type": "string",
                    "description": "Preferred domain expertise to emphasize"
                },
                "enable_collaboration_fallback": {
                    "type": "boolean",
                    "description": "Enable collaborative fallback when ambiguity or insufficient context is detected",
                    "default": True
                }
            },
            "required": ["task_description"]
        }
    ),
    types.Tool(
        name="maestro_iae_discovery",
        description="🔍 Integrated Analysis Engine discovery for optimal computation selection",
        inputSchema={
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "Type of task for engine discovery",
                    "default": "general"
                },
                "domain_context": {
                    "type": "string",
                    "description": "Domain context for the task"
                },
                "complexity_requirements": {
                    "type": "object",
                    "description": "Complexity requirements",
                    "additionalProperties": True
                }
            },
            "required": ["task_type"]
        }
    ),
    types.Tool(
        name="maestro_tool_selection",
        description="🧰 Intelligent tool selection based on task requirements",
        inputSchema={
            "type": "object",
            "properties": {
                "request_description": {
                    "type": "string",
                    "description": "Description of the request for tool selection"
                },
                "available_context": {
                    "type": "object",
                    "description": "Available context",
                    "additionalProperties": True
                },
                "precision_requirements": {
                    "type": "object",
                    "description": "Precision requirements",
                    "additionalProperties": True
                }
            },
            "required": ["request_description"]
        }
    ),
    types.Tool(
        name="maestro_iae",
        description="⚡ Integrated Analysis Engine for computational tasks",
        inputSchema={
            "type": "object",
            "properties": {
                "analysis_request": {
                    "type": "string",
                    "description": "The analysis or computation request"
                },
                "engine_type": {
                    "type": "string",
                    "enum": ["statistical", "mathematical", "quantum", "auto"],
                    "description": "Type of analysis engine to use",
                    "default": "auto"
                },
                "precision_level": {
                    "type": "string",
                    "enum": ["standard", "high", "ultra"],
                    "description": "Required precision level",
                    "default": "standard"
                },
                "computational_context": {
                    "type": "object",
                    "description": "Additional computational context",
                    "additionalProperties": True
                }
            },
            "required": ["analysis_request"]
        }
    ),
    types.Tool(
        name="maestro_search",
        description="🔎 Enhanced search capabilities across multiple sources",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of results",
                    "default": 10
                },
                "search_engine": {
                    "type": "string",
                    "enum": ["duckduckgo", "google", "bing", "auto"],
                    "description": "Search engine to use",
                    "default": "duckduckgo"
                },
                "temporal_filter": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year", "any"],
                    "description": "Time filter for results",
                    "default": "any"
                },
                "result_format": {
                    "type": "string",
                    "enum": ["structured", "markdown", "json"],
                    "description": "Output format preference",
                    "default": "structured"
                }
            },
            "required": ["query"]
        }
    ),
    types.Tool(
        name="maestro_scrape",
        description="🌐 Web scraping and content extraction",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to scrape"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["markdown", "html", "text", "json"],
                    "description": "Output format",
                    "default": "markdown"
                },
                "selectors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CSS selectors for specific content"
                },
                "wait_time": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 30,
                    "description": "Wait time in seconds",
                    "default": 3
                },
                "extract_links": {
                    "type": "boolean",
                    "description": "Extract links from the page",
                    "default": False
                }
            },
            "required": ["url"]
        }
    ),
    types.Tool(
        name="maestro_execute",
        description="⚙️ Secure code execution in isolated environment",
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to execute"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "bash", "shell"],
                    "description": "Programming language",
                    "default": "python"
                },
                "execution_context": {
                    "type": "object",
                    "description": "Execution environment variables",
                    "additionalProperties": True
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Execution timeout",
                    "default": 30
                },
                "safe_mode": {
                    "type": "boolean",
                    "description": "Enable safety restrictions",
                    "default": True
                }
            },
            "required": ["code"]
        }
    ),
    types.Tool(
        name="maestro_error_handler",
        description="🔧 Intelligent error analysis and recovery suggestions",
        inputSchema={
            "type": "object",
            "properties": {
                "error_message": {
                    "type": "string",
                    "description": "Error message to analyze"
                },
                "error_context": {
                    "type": "object",
                    "description": "Additional error context",
                    "additionalProperties": True
                },
                "recovery_suggestions": {
                    "type": "boolean",
                    "description": "Include recovery suggestions",
                    "default": True
                }
            },
            "required": ["error_message"]
        }
    ),
    types.Tool(
        name="maestro_temporal_context",
        description="⏰ Time-aware context and temporal reasoning",
        inputSchema={
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Task requiring temporal context"
                },
                "temporal_query": {
                    "type": "string",
                    "description": "Temporal query or time reference"
                },
                "time_range": {
                    "type": "object",
                    "description": "Time range specifications",
                    "additionalProperties": True
                },
                "temporal_precision": {
                    "type": "string",
                    "enum": ["minute", "hour", "day", "week", "month", "year"],
                    "description": "Required temporal precision",
                    "default": "day"
                }
            },
            "required": ["task_description", "temporal_query"]
        }
    ),
    types.Tool(
        name="get_available_engines",
        description="🔧 List available computational engines and capabilities",
        inputSchema={
            "type": "object",
            "properties": {
                "engine_type": {
                    "type": "string",
                    "enum": ["all", "mathematical", "statistical", "quantum", "optimization"],
                    "description": "Filter by engine type",
                    "default": "all"
                },
                "include_capabilities": {
                    "type": "boolean",
                    "description": "Include detailed capabilities",
                    "default": True
                }
            }
        }
    ),
    types.Tool(
        name="maestro_collaboration_response",
        description="🤝 Handle user responses to collaboration requests",
        inputSchema={
            "type": "object",
            "properties": {
                "collaboration_id": {
                    "type": "string",
                    "description": "ID of the collaboration request"
                },
                "responses": {
                    "type": "object",
                    "description": "User responses to collaboration questions",
                    "additionalProperties": True
                },
                "additional_context": {
                    "type": "object",
                    "description": "Additional context provided by user",
                    "additionalProperties": True
                },
                "user_preferences": {
                    "type": "object",
                    "description": "User preferences for task execution",
                    "additionalProperties": True
                },
                "approval_status": {
                    "type": "string",
                    "enum": ["approved", "rejected", "modified"],
                    "description": "User approval status",
                    "default": "approved"
                },
                "confidence_level": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "User confidence in provided responses",
                    "default": 0.8
                }
            },
            "required": ["collaboration_id"]
        }
    )
]

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Handle tools/list requests - return ALL tools with STATIC definitions only"""
    # NO LOGGING during tool scanning - Smithery requires instantaneous response
    # Return pre-computed static tools for maximum speed
    return STATIC_TOOLS

# Lazy-loaded instances - ONLY created when tools are actually called
_maestro_tools_instance = None
_computational_tools_instance = None
_enhanced_tool_handlers_instance = None

def get_maestro_tools_instance():
    """Lazy load MaestroTools - only on first tool call"""
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        logger.info("Loading MaestroTools instance")
        # Add src to path only when actually needed
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
    return _maestro_tools_instance

def get_computational_tools_instance():
    """Lazy load ComputationalTools - only on first tool call"""
    global _computational_tools_instance
    if _computational_tools_instance is None:
        logger.info("Loading ComputationalTools instance")
        # Add src to path only when actually needed
        if os.path.join(os.path.dirname(__file__), 'src') not in sys.path:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from computational_tools import ComputationalTools
        _computational_tools_instance = ComputationalTools()
    return _computational_tools_instance

def get_enhanced_tool_handlers_instance():
    """Lazy load EnhancedToolHandlers - only on first tool call"""
    global _enhanced_tool_handlers_instance
    if _enhanced_tool_handlers_instance is None:
        logger.info("Loading EnhancedToolHandlers instance")
        # Add src to path only when actually needed
        if os.path.join(os.path.dirname(__file__), 'src') not in sys.path:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # Force reload of modules to get latest changes
        import importlib
        if 'maestro.enhanced_tools' in sys.modules:
            importlib.reload(sys.modules['maestro.enhanced_tools'])
        if 'maestro.llm_web_tools' in sys.modules:
            importlib.reload(sys.modules['maestro.llm_web_tools'])
            
        from maestro.enhanced_tools import EnhancedToolHandlers
        _enhanced_tool_handlers_instance = EnhancedToolHandlers()
    return _enhanced_tool_handlers_instance

def reset_enhanced_tool_handlers_instance():
    """Reset the enhanced tool handlers instance to force reload"""
    global _enhanced_tool_handlers_instance
    _enhanced_tool_handlers_instance = None

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> List[types.TextContent]:
    """Handle tools/call requests - route to appropriate tool handler"""
    logger.info(f"Handling call_tool request: {name}")
    
    if arguments is None:
        arguments = {}
    
    try:
        if name == "maestro_orchestrate":
            return await _handle_maestro_orchestrate(arguments)
        elif name == "maestro_iae_discovery":
            return await _handle_maestro_iae_discovery(arguments)
        elif name == "maestro_tool_selection":
            return await _handle_maestro_tool_selection(arguments)
        elif name == "maestro_iae":
            return await _handle_maestro_iae(arguments)
        elif name == "maestro_search":
            return await _handle_maestro_search(arguments)
        elif name == "maestro_scrape":
            return await _handle_maestro_scrape(arguments)
        elif name == "maestro_execute":
            return await _handle_maestro_execute(arguments)
        elif name == "maestro_error_handler":
            return await _handle_maestro_error_handler(arguments)
        elif name == "maestro_temporal_context":
            return await _handle_maestro_temporal_context(arguments)
        elif name == "get_available_engines":
            return await _handle_get_available_engines(arguments)
        elif name == "maestro_collaboration_response":
            return await _handle_maestro_collaboration_response(arguments)
        else:
            logger.error(f"Unknown tool: {name}")
            return [types.TextContent(type="text", text=f"Error: Unknown tool '{name}'")]
    
    except Exception as e:
        logger.error(f"Error handling tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error executing {name}: {str(e)}")]

# Tool handler functions - these do the actual work and load dependencies as needed

async def _handle_maestro_iae(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_iae tool calls - Integrated Analysis Engine"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Call the new IAE handler directly
        result = await enhanced_tools.handle_maestro_iae(arguments)
        
        return result
    except Exception as e:
        logger.error(f"Error in maestro_iae: {e}")
        return [types.TextContent(type="text", text=f"Error in IAE analysis: {str(e)}")]

async def _handle_maestro_orchestrate(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_orchestrate tool calls"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Call the new orchestrate handler directly
        result = await enhanced_tools.handle_maestro_orchestrate(arguments)
        
        return result
    except Exception as e:
        logger.error(f"Error in maestro_orchestrate: {e}")
        return [types.TextContent(type="text", text=f"Error in orchestration: {str(e)}")]

async def _handle_maestro_iae_discovery(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_iae_discovery tool calls"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        task_type = arguments.get("task_type", "general")
        domain_context = arguments.get("domain_context", "")
        complexity_requirements = arguments.get("complexity_requirements", {})
        
        result = await maestro_tools.iae_discovery(task_type, domain_context, complexity_requirements)
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_iae_discovery: {e}")
        return [types.TextContent(type="text", text=f"Error in IAE discovery: {str(e)}")]

async def _handle_maestro_tool_selection(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_tool_selection tool calls"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        request_description = arguments.get("request_description", "")
        available_context = arguments.get("available_context", {})
        precision_requirements = arguments.get("precision_requirements", {})
        
        result = await maestro_tools.tool_selection(request_description, available_context, precision_requirements)
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_tool_selection: {e}")
        return [types.TextContent(type="text", text=f"Error in tool selection: {str(e)}")]

async def _handle_maestro_search(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_search tool calls"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Call the handler directly instead of the bridge method
        result = await enhanced_tools.handle_maestro_search(arguments)
        
        return result
    except Exception as e:
        logger.error(f"Error in maestro_search: {e}")
        return [types.TextContent(type="text", text=f"Error in search: {str(e)}")]

async def _handle_maestro_scrape(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_scrape tool calls"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Call the handler directly instead of the bridge method
        result = await enhanced_tools.handle_maestro_scrape(arguments)
        
        return result
    except Exception as e:
        logger.error(f"Error in maestro_scrape: {e}")
        return [types.TextContent(type="text", text=f"Error in scraping: {str(e)}")]

async def _handle_maestro_execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_execute tool calls"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        command = arguments.get("command", "")
        execution_context = arguments.get("execution_context", {})
        timeout_seconds = arguments.get("timeout_seconds", 30)
        safe_mode = arguments.get("safe_mode", True)
        
        result = await enhanced_tools.execute(command, execution_context, timeout_seconds, safe_mode)
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_execute: {e}")
        return [types.TextContent(type="text", text=f"Error in execution: {str(e)}")]

async def _handle_maestro_error_handler(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_error_handler tool calls"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        error_message = arguments.get("error_message", "")
        error_context = arguments.get("error_context", {})
        recovery_suggestions = arguments.get("recovery_suggestions", True)
        
        result = await enhanced_tools.error_handler(error_message, error_context, recovery_suggestions)
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_error_handler: {e}")
        return [types.TextContent(type="text", text=f"Error in error handling: {str(e)}")]

async def _handle_maestro_temporal_context(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_temporal_context tool calls"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        temporal_query = arguments.get("temporal_query", "")
        time_range = arguments.get("time_range", {})
        temporal_precision = arguments.get("temporal_precision", "day")
        
        result = await enhanced_tools.temporal_context(temporal_query, time_range, temporal_precision)
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_temporal_context: {e}")
        return [types.TextContent(type="text", text=f"Error in temporal context: {str(e)}")]

async def _handle_get_available_engines(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle get_available_engines tool calls"""
    try:
        comp_tools = get_computational_tools_instance()
        
        engine_type = arguments.get("engine_type", "all")
        include_capabilities = arguments.get("include_capabilities", True)
        
        # Get available engines from computational tools
        available_engines = comp_tools.get_available_engines()
        
        if include_capabilities:
            engines_info = comp_tools.get_engine_capabilities()
            result = f"Available Engines:\n{available_engines}\n\nCapabilities:\n{engines_info}"
        else:
            result = f"Available Engines:\n{available_engines}"
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in get_available_engines: {e}")
        return [types.TextContent(type="text", text=f"Error getting engines: {str(e)}")]

async def _handle_maestro_collaboration_response(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle collaboration response tool calls"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        collaboration_id = arguments.get("collaboration_id")
        responses = arguments.get("responses", {})
        additional_context = arguments.get("additional_context", {})
        user_preferences = arguments.get("user_preferences", {})
        approval_status = arguments.get("approval_status", "approved")
        confidence_level = arguments.get("confidence_level", 0.8)
        
        if not collaboration_id:
            return [types.TextContent(type="text", text="❌ Error: collaboration_id is required")]
        
        # Process the collaboration response
        result = await maestro_tools.handle_collaboration_response(
            collaboration_id=collaboration_id,
            responses=responses,
            additional_context=additional_context,
            user_preferences=user_preferences,
            approval_status=approval_status,
            confidence_level=confidence_level
        )
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_collaboration_response: {e}")
        return [types.TextContent(type="text", text=f"❌ Error handling collaboration response: {str(e)}")]

async def main():
    """Run the MCP server with stdio transport"""
    # Minimal logging for Smithery deployment - only critical errors
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 
