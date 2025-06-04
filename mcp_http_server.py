#!/usr/bin/env python3
# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Maestro MCP HTTP Server - Smithery deployment implementation
Optimized for instant tool scanning with HTTP transport
"""

import asyncio
import logging
import sys
import os
import json
from typing import Any, Dict, List
from contextlib import asynccontextmanager

# Core MCP imports for HTTP transport
from mcp.server.fastmcp import FastMCP
import mcp.types as types

# Configure minimal logging for production deployment
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Create FastMCP server instance for HTTP transport
app = FastMCP("maestro-mcp")

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

# Register static tools with FastMCP for instant discovery
for tool in STATIC_TOOLS:
    app.add_tool(tool)

# Lazy-loaded instances - ONLY created when tools are actually called
_maestro_tools_instance = None
_computational_tools_instance = None
_enhanced_tool_handlers_instance = None

def get_maestro_tools_instance():
    """Lazy load MaestroTools - only on first tool call"""
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        # Add src to path only when actually needed
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
    return _maestro_tools_instance

def get_computational_tools_instance():
    """Lazy load ComputationalTools - only on first tool call"""
    global _computational_tools_instance
    if _computational_tools_instance is None:
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

# Tool handler functions - these do the actual work and load dependencies as needed

@app.tool()
async def maestro_orchestrate(
    task_description: str,
    context: Dict[str, Any] = None,
    success_criteria: Dict[str, Any] = None,
    complexity_level: str = "moderate",
    quality_threshold: float = 0.85,
    resource_level: str = "moderate",
    reasoning_focus: str = "auto",
    validation_rigor: str = "standard",
    max_iterations: int = 3,
    domain_specialization: str = None,
    enable_collaboration_fallback: bool = True
) -> str:
    """Enhanced intelligent meta-reasoning orchestration for 3-5x LLM capability amplification"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Build arguments dict
        arguments = {
            "task_description": task_description,
            "context": context or {},
            "success_criteria": success_criteria or {},
            "complexity_level": complexity_level,
            "quality_threshold": quality_threshold,
            "resource_level": resource_level,
            "reasoning_focus": reasoning_focus,
            "validation_rigor": validation_rigor,
            "max_iterations": max_iterations,
            "enable_collaboration_fallback": enable_collaboration_fallback
        }
        if domain_specialization:
            arguments["domain_specialization"] = domain_specialization
        
        # Call the handler directly
        result = await enhanced_tools.handle_maestro_orchestrate(arguments)
        return result[0].text if result else "No result returned"
        
    except Exception as e:
        logger.error(f"Error in maestro_orchestrate: {e}")
        return f"Error in orchestration: {str(e)}"

@app.tool()
async def maestro_iae_discovery(
    task_type: str = "general",
    domain_context: str = "",
    complexity_requirements: Dict[str, Any] = None
) -> str:
    """Integrated Analysis Engine discovery for optimal computation selection"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        result = await maestro_tools.iae_discovery(
            task_type, 
            domain_context, 
            complexity_requirements or {}
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in maestro_iae_discovery: {e}")
        return f"Error in IAE discovery: {str(e)}"

@app.tool()
async def maestro_tool_selection(
    request_description: str,
    available_context: Dict[str, Any] = None,
    precision_requirements: Dict[str, Any] = None
) -> str:
    """Intelligent tool selection based on task requirements"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        result = await maestro_tools.tool_selection(
            request_description, 
            available_context or {}, 
            precision_requirements or {}
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in maestro_tool_selection: {e}")
        return f"Error in tool selection: {str(e)}"

@app.tool()
async def maestro_iae(
    analysis_request: str,
    engine_type: str = "auto",
    precision_level: str = "standard",
    computational_context: Dict[str, Any] = None
) -> str:
    """Integrated Analysis Engine for computational tasks"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Build arguments dict
        arguments = {
            "analysis_request": analysis_request,
            "engine_type": engine_type,
            "precision_level": precision_level,
            "computational_context": computational_context or {}
        }
        
        # Call the new IAE handler directly
        result = await enhanced_tools.handle_maestro_iae(arguments)
        return result[0].text if result else "No result returned"
        
    except Exception as e:
        logger.error(f"Error in maestro_iae: {e}")
        return f"Error in IAE analysis: {str(e)}"

@app.tool()
async def maestro_search(
    query: str,
    max_results: int = 10,
    search_engine: str = "duckduckgo",
    temporal_filter: str = "any",
    result_format: str = "structured"
) -> str:
    """Enhanced search capabilities across multiple sources"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Build arguments dict
        arguments = {
            "query": query,
            "max_results": max_results,
            "search_engine": search_engine,
            "temporal_filter": temporal_filter,
            "result_format": result_format
        }
        
        # Call the handler directly
        result = await enhanced_tools.handle_maestro_search(arguments)
        return result[0].text if result else "No result returned"
        
    except Exception as e:
        logger.error(f"Error in maestro_search: {e}")
        return f"Error in search: {str(e)}"

@app.tool()
async def maestro_scrape(
    url: str,
    output_format: str = "markdown",
    selectors: List[str] = None,
    wait_time: int = 3,
    extract_links: bool = False
) -> str:
    """Web scraping and content extraction"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Build arguments dict
        arguments = {
            "url": url,
            "output_format": output_format,
            "wait_time": wait_time,
            "extract_links": extract_links
        }
        if selectors:
            arguments["selectors"] = selectors
        
        # Call the handler directly
        result = await enhanced_tools.handle_maestro_scrape(arguments)
        return result[0].text if result else "No result returned"
        
    except Exception as e:
        logger.error(f"Error in maestro_scrape: {e}")
        return f"Error in scraping: {str(e)}"

@app.tool()
async def maestro_execute(
    code: str,
    language: str = "python",
    execution_context: Dict[str, Any] = None,
    timeout_seconds: int = 30,
    safe_mode: bool = True
) -> str:
    """Secure code execution in isolated environment"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Build arguments dict
        arguments = {
            "code": code,
            "language": language,
            "execution_context": execution_context or {},
            "timeout_seconds": timeout_seconds,
            "safe_mode": safe_mode
        }
        
        # Call the handler directly
        result = await enhanced_tools.handle_maestro_execute(arguments)
        return result[0].text if result else "No result returned"
        
    except Exception as e:
        logger.error(f"Error in maestro_execute: {e}")
        return f"Error in execution: {str(e)}"

@app.tool()
async def maestro_error_handler(
    error_message: str,
    error_context: Dict[str, Any] = None,
    recovery_suggestions: bool = True
) -> str:
    """Intelligent error analysis and recovery suggestions"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Build arguments dict
        arguments = {
            "error_message": error_message,
            "error_context": error_context or {},
            "recovery_suggestions": recovery_suggestions
        }
        
        # Call the handler directly
        result = await enhanced_tools.handle_maestro_error_handler(arguments)
        return result[0].text if result else "No result returned"
        
    except Exception as e:
        logger.error(f"Error in maestro_error_handler: {e}")
        return f"Error in error handling: {str(e)}"

@app.tool()
async def maestro_temporal_context(
    task_description: str,
    temporal_query: str,
    time_range: Dict[str, Any] = None,
    temporal_precision: str = "day"
) -> str:
    """Time-aware context and temporal reasoning"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        # Build arguments dict
        arguments = {
            "task_description": task_description,
            "temporal_query": temporal_query,
            "time_range": time_range or {},
            "temporal_precision": temporal_precision
        }
        
        # Call the handler directly
        result = await enhanced_tools.handle_maestro_temporal_context(arguments)
        return result[0].text if result else "No result returned"
        
    except Exception as e:
        logger.error(f"Error in maestro_temporal_context: {e}")
        return f"Error in temporal context: {str(e)}"

@app.tool()
async def get_available_engines(
    engine_type: str = "all",
    include_capabilities: bool = True
) -> str:
    """List available computational engines and capabilities"""
    try:
        comp_tools = get_computational_tools_instance()
        
        # Get available engines from computational tools
        available_engines = comp_tools.get_available_engines()
        
        if include_capabilities:
            engines_info = comp_tools.get_engine_capabilities()
            result = f"Available Engines:\n{available_engines}\n\nCapabilities:\n{engines_info}"
        else:
            result = f"Available Engines:\n{available_engines}"
        
        return result
    except Exception as e:
        logger.error(f"Error in get_available_engines: {e}")
        return f"Error getting engines: {str(e)}"

@app.tool()
async def maestro_collaboration_response(
    collaboration_id: str,
    responses: Dict[str, Any] = None,
    additional_context: Dict[str, Any] = None,
    user_preferences: Dict[str, Any] = None,
    approval_status: str = "approved",
    confidence_level: float = 0.8
) -> str:
    """Handle user responses to collaboration requests"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        # Process the collaboration response
        result = await maestro_tools.handle_collaboration_response(
            collaboration_id=collaboration_id,
            responses=responses or {},
            additional_context=additional_context or {},
            user_preferences=user_preferences or {},
            approval_status=approval_status,
            confidence_level=confidence_level
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in maestro_collaboration_response: {e}")
        return f"❌ Error handling collaboration response: {str(e)}"

# Health check endpoint for Smithery
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "maestro-mcp", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (required by Smithery)
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastMCP server
    uvicorn.run(app, host="0.0.0.0", port=port) 