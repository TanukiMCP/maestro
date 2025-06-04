#!/usr/bin/env python3
# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Maestro MCP FastMCP Server - Smithery-compatible implementation
Using official FastMCP for proper tool scanning and HTTP transport
"""

import os
import sys
from typing import Any, Dict, List
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Use official FastMCP from the MCP SDK
from mcp.server.fastmcp import FastMCP

# Lazy-loaded instances - ONLY created when tools are actually called
_maestro_tools_instance = None
_computational_tools_instance = None  
_enhanced_tool_handlers_instance = None

def get_maestro_tools_instance():
    """Lazy load MaestroTools - only on first tool call"""
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        from maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
    return _maestro_tools_instance

def get_computational_tools_instance():
    """Lazy load ComputationalTools - only on first tool call"""
    global _computational_tools_instance
    if _computational_tools_instance is None:
        from computational_tools import ComputationalTools
        _computational_tools_instance = ComputationalTools()
    return _computational_tools_instance

def get_enhanced_tool_handlers_instance():
    """Lazy load EnhancedToolHandlers - only on first tool call"""
    global _enhanced_tool_handlers_instance
    if _enhanced_tool_handlers_instance is None:
        # Force reload of modules to get latest changes
        import importlib
        if 'maestro.enhanced_tools' in sys.modules:
            importlib.reload(sys.modules['maestro.enhanced_tools'])
        if 'maestro.llm_web_tools' in sys.modules:
            importlib.reload(sys.modules['maestro.llm_web_tools'])
            
        from maestro.enhanced_tools import EnhancedToolHandlers
        _enhanced_tool_handlers_instance = EnhancedToolHandlers()
    return _enhanced_tool_handlers_instance

# Context manager for app lifecycle
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Manage application lifecycle"""
    # Initialize minimal resources on startup for instant tool scanning
    yield {}

# Create FastMCP server with dependencies for deployment
mcp = FastMCP(
    name="maestro-mcp",
    dependencies=[
        "langchain",
        "langchain-community", 
        "langchain-experimental",
        "langchain-openai",
        "sympy",
        "numpy", 
        "scipy",
        "pandas",
        "matplotlib",
        "plotly",
        "scikit-learn",
        "spacy",
        "nltk", 
        "playwright",
        "selenium",
        "beautifulsoup4",
        "pillow",
        "opencv-python",
        "requests",
        "redis",
        "pydantic",
        "aiohttp",
        "httpx",
        "psutil",
        "asyncio",
        "sqlalchemy",
        "aiomysql"
    ],
    lifespan=app_lifespan
)

# MAESTRO ORCHESTRATION TOOLS

@mcp.tool()
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
    """🎭 Enhanced intelligent meta-reasoning orchestration for 3-5x LLM capability amplification"""
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
        return f"Error in orchestration: {str(e)}"

@mcp.tool()
async def maestro_collaboration_response(
    collaboration_id: str,
    responses: Dict[str, Any] = None,
    additional_context: Dict[str, Any] = None,
    user_preferences: Dict[str, Any] = None,
    approval_status: str = "approved",
    confidence_level: float = 0.8
) -> str:
    """🤝 Handle user responses to collaboration requests"""
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
        return f"❌ Error handling collaboration response: {str(e)}"

# ANALYSIS AND DISCOVERY TOOLS

@mcp.tool()
async def maestro_iae_discovery(
    task_type: str = "general",
    domain_context: str = "",
    complexity_requirements: Dict[str, Any] = None
) -> str:
    """🔍 Integrated Analysis Engine discovery for optimal computation selection"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        result = await maestro_tools.iae_discovery(
            task_type, 
            domain_context, 
            complexity_requirements or {}
        )
        
        return result
    except Exception as e:
        return f"Error in IAE discovery: {str(e)}"

@mcp.tool()
async def maestro_tool_selection(
    request_description: str,
    available_context: Dict[str, Any] = None,
    precision_requirements: Dict[str, Any] = None
) -> str:
    """🧰 Intelligent tool selection based on task requirements"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        result = await maestro_tools.tool_selection(
            request_description, 
            available_context or {}, 
            precision_requirements or {}
        )
        
        return result
    except Exception as e:
        return f"Error in tool selection: {str(e)}"

@mcp.tool()
async def maestro_iae(
    analysis_request: str,
    engine_type: str = "auto",
    precision_level: str = "standard",
    computational_context: Dict[str, Any] = None
) -> str:
    """⚡ Integrated Analysis Engine for computational tasks"""
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
        return f"Error in IAE analysis: {str(e)}"

@mcp.tool()
async def get_available_engines(
    engine_type: str = "all",
    include_capabilities: bool = True
) -> str:
    """🔧 List available computational engines and capabilities"""
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
        return f"Error getting engines: {str(e)}"

# WEB AND DATA TOOLS

@mcp.tool()
async def maestro_search(
    query: str,
    max_results: int = 10,
    search_engine: str = "duckduckgo",
    temporal_filter: str = "any",
    result_format: str = "structured"
) -> str:
    """🔎 Enhanced search capabilities across multiple sources"""
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
        return f"Error in search: {str(e)}"

@mcp.tool()
async def maestro_scrape(
    url: str,
    output_format: str = "markdown",
    selectors: List[str] = None,
    wait_time: int = 3,
    extract_links: bool = False
) -> str:
    """🌐 Web scraping and content extraction"""
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
        return f"Error in scraping: {str(e)}"

@mcp.tool()
async def maestro_execute(
    code: str,
    language: str = "python",
    execution_context: Dict[str, Any] = None,
    timeout_seconds: int = 30,
    safe_mode: bool = True
) -> str:
    """⚙️ Secure code execution in isolated environment"""
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
        return f"Error in execution: {str(e)}"

@mcp.tool()
async def maestro_temporal_context(
    task_description: str,
    temporal_query: str,
    time_range: Dict[str, Any] = None,
    temporal_precision: str = "day"
) -> str:
    """⏰ Time-aware context and temporal reasoning"""
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
        return f"Error in temporal context: {str(e)}"

# ERROR HANDLING TOOL

@mcp.tool()
async def maestro_error_handler(
    error_message: str,
    error_context: Dict[str, Any] = None,
    recovery_suggestions: bool = True
) -> str:
    """🔧 Intelligent error analysis and recovery suggestions"""
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
        return f"Error in error handling: {str(e)}"

if __name__ == "__main__":
    # Run with FastMCP's built-in server - supports all transport types
    import uvicorn
    
    # Get port from environment variable (required by Smithery)
    port = int(os.getenv("PORT", 8000))
    
    # Run the server with HTTP transport for Smithery deployment
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port) 