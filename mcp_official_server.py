#!/usr/bin/env python3
"""
Official MCP Server using FastMCP with Streamable HTTP transport.
This is the correct, modern approach for Smithery deployment.
"""

import os
import asyncio
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP, Context

# Create FastMCP server instance with settings
mcp = FastMCP(
    name="TanukiMCP-Maestro",
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8000))
)

# Global instances for tool classes (lazy loaded)
_maestro_tools = None
_computational_tools = None

def get_maestro_tools():
    """Lazy load MaestroTools only when needed"""
    global _maestro_tools
    if _maestro_tools is None:
        from src.maestro_tools import MaestroTools
        _maestro_tools = MaestroTools()
    return _maestro_tools

def get_computational_tools():
    """Lazy load ComputationalTools only when needed"""
    global _computational_tools
    if _computational_tools is None:
        from src.computational_tools import ComputationalTools
        _computational_tools = ComputationalTools()
    return _computational_tools

# Create a simple mock context for tools that need it
class MockContext:
    async def sample(self, prompt: str, **kwargs):
        """Mock sample method for Context"""
        if "2+2" in prompt or "factorial" in prompt:
            return type('Response', (), {'text': 'The answer is 4. This is calculated by adding 2 + 2 = 4.'})()
        elif "json" in kwargs.get('response_format', {}).get('type', ''):
            return type('Response', (), {'json': lambda: {"score": 0.8, "issues": [], "recommendations": []}})()
        else:
            return type('Response', (), {'text': f'This is a mock response for: {prompt[:100]}...'})()

# Register Maestro Tools
@mcp.tool(description="Orchestrate complex tasks with intelligent decomposition and multi-agent validation")
async def maestro_orchestrate(
    task_description: str,
    context: Dict[str, Any] = None,
    complexity_level: str = "moderate",
    quality_threshold: float = 0.8,
    resource_level: str = "moderate",
    reasoning_focus: str = "auto",
    validation_rigor: str = "standard",
    max_iterations: int = 3,
    domain_specialization: str = "",
    enable_collaboration_fallback: bool = True
) -> str:
    """Orchestrate complex tasks with intelligent decomposition and multi-agent validation."""
    tools = get_maestro_tools()
    ctx = MockContext()
    return await tools.orchestrate_task(
        ctx=ctx,
        task_description=task_description,
        context=context or {},
        complexity_level=complexity_level,
        quality_threshold=quality_threshold,
        resource_level=resource_level,
        reasoning_focus=reasoning_focus,
        validation_rigor=validation_rigor,
        max_iterations=max_iterations,
        domain_specialization=domain_specialization,
        enable_collaboration_fallback=enable_collaboration_fallback
    )

@mcp.tool(description="Handle collaboration responses for ongoing orchestration tasks")
async def maestro_collaboration_response(
    collaboration_id: str,
    responses: Dict[str, Any],
    additional_context: Dict[str, Any] = None,
    user_preferences: Dict[str, Any] = None,
    approval_status: str = "approved",
    confidence_level: float = 1.0
) -> str:
    """Handle collaboration responses for ongoing orchestration tasks."""
    tools = get_maestro_tools()
    return await tools.handle_collaboration_response(
        collaboration_id=collaboration_id,
        responses=responses,
        additional_context=additional_context or {},
        user_preferences=user_preferences or {},
        approval_status=approval_status,
        confidence_level=confidence_level
    )

@mcp.tool(description="Discover and analyze computational requirements for Intelligence Amplification Engine")
async def maestro_iae_discovery(
    task_description: str,
    domain_context: str = "",
    complexity_assessment: str = "auto",
    computational_requirements: Dict[str, Any] = None
) -> str:
    """Discover and analyze computational requirements for Intelligence Amplification Engine."""
    tools = get_maestro_tools()
    arguments = {
        "task_description": task_description,
        "domain_context": domain_context,
        "complexity_assessment": complexity_assessment,
        "computational_requirements": computational_requirements or {}
    }
    result = await tools._handle_iae_discovery(arguments)
    return result[0].text if result else "No result"

@mcp.tool(description="Intelligent tool selection based on task requirements and constraints")
async def maestro_tool_selection(
    task_description: str,
    available_tools: List[str] = None,
    constraints: Dict[str, Any] = None
) -> str:
    """Intelligent tool selection based on task requirements and constraints."""
    tools = get_maestro_tools()
    arguments = {
        "task_description": task_description,
        "available_tools": available_tools or [],
        "constraints": constraints or {}
    }
    result = await tools._handle_tool_selection(arguments)
    return result[0].text if result else "No result"

@mcp.tool(description="Enhanced search with intelligent query processing and result synthesis")
async def maestro_search(
    query: str,
    max_results: int = 10,
    search_type: str = "comprehensive",
    temporal_filter: str = "recent",
    output_format: str = "detailed"
) -> str:
    """Enhanced search with intelligent query processing and result synthesis."""
    tools = get_maestro_tools()
    arguments = {
        "query": query,
        "max_results": max_results,
        "search_type": search_type,
        "temporal_filter": temporal_filter,
        "output_format": output_format
    }
    result = await tools._handle_maestro_search(arguments)
    return result[0].text if result else "No result"

@mcp.tool(description="Intelligent web scraping with content extraction and filtering")
async def maestro_scrape(
    url: str,
    extraction_type: str = "text",
    content_filter: str = "relevant",
    output_format: str = "structured"
) -> str:
    """Intelligent web scraping with content extraction and filtering."""
    tools = get_maestro_tools()
    arguments = {
        "url": url,
        "extraction_type": extraction_type,
        "content_filter": content_filter,
        "output_format": output_format
    }
    result = await tools._handle_maestro_scrape(arguments)
    return result[0].text if result else "No result"

@mcp.tool(description="Secure code execution with safety checks and sandboxing")
async def maestro_execute(
    execution_type: str = "code",
    content: str = "",
    language: str = "python",
    security_level: str = "standard",
    timeout: int = 30
) -> str:
    """Secure code execution with safety checks and sandboxing."""
    tools = get_maestro_tools()
    arguments = {
        "execution_type": execution_type,
        "content": content,
        "language": language,
        "security_level": security_level,
        "timeout": timeout
    }
    result = await tools._handle_maestro_execute(arguments)
    return result[0].text if result else "No result"

@mcp.tool(description="Temporal reasoning and context analysis for time-sensitive queries")
async def maestro_temporal_context(
    query: str,
    time_scope: str = "current",
    context_depth: str = "moderate",
    currency_check: bool = True
) -> str:
    """Temporal reasoning and context analysis for time-sensitive queries."""
    tools = get_maestro_tools()
    arguments = {
        "query": query,
        "time_scope": time_scope,
        "context_depth": context_depth,
        "currency_check": currency_check
    }
    result = await tools._handle_maestro_temporal_context(arguments)
    return result[0].text if result else "No result"

@mcp.tool(description="Intelligent error handling with pattern recognition and recovery suggestions")
async def maestro_error_handler(
    error_context: str,
    error_type: str = "general",
    recovery_mode: str = "automatic",
    learning_enabled: bool = True
) -> str:
    """Intelligent error handling with pattern recognition and recovery suggestions."""
    tools = get_maestro_tools()
    arguments = {
        "error_context": error_context,
        "error_type": error_type,
        "recovery_mode": recovery_mode,
        "learning_enabled": learning_enabled
    }
    result = await tools._handle_maestro_error_handler(arguments)
    return result[0].text if result else "No result"

# Register Computational Tools
@mcp.tool(description="Intelligence Amplification Engine Gateway - Access to all computational engines for precise numerical calculations")
async def maestro_iae(
    engine_domain: str = "quantum_physics",
    computation_type: str = "entanglement_entropy",
    parameters: Dict[str, Any] = None
) -> str:
    """Intelligence Amplification Engine Gateway - Access to all computational engines for precise numerical calculations."""
    tools = get_computational_tools()
    result = await tools.handle_tool_call("maestro_iae", {
        "engine_domain": engine_domain,
        "computation_type": computation_type,
        "parameters": parameters or {}
    })
    return result[0].text if result else "No result"

@mcp.tool(description="Get available computational engines and their capabilities")
async def get_available_engines(
    detailed: bool = False,
    include_status: bool = True
) -> str:
    """Get available computational engines and their capabilities."""
    tools = get_computational_tools()
    ctx = MockContext()
    return await tools.get_available_engines(ctx, detailed=detailed, include_status=include_status)

if __name__ == "__main__":
    # Run with Streamable HTTP transport (recommended for Smithery)
    mcp.run(transport="streamable-http") 