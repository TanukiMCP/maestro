#!/usr/bin/env python3
"""
TanukiMCP Maestro - Production-Ready MCP Server

Fast-loading MCP server with 11 production tools optimized for Smithery.ai deployment.
All heavy imports are deferred to ensure <100ms tool discovery.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List

# Only import FastMCP - all other imports are deferred
from fastmcp import FastMCP

# Initialize FastMCP server for instant tool registration
mcp = FastMCP("TanukiMCP Maestro")

# Health check endpoint for Smithery.ai deployment
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for load balancers and container orchestration."""
    from starlette.responses import PlainTextResponse
    return PlainTextResponse("ok", status_code=200)

# Tools endpoint for debugging
@mcp.custom_route("/tools", methods=["GET"])
async def tools_endpoint(request):
    """Tools endpoint for debugging and manual verification."""
    from starlette.responses import JSONResponse
    try:
        # Handle async get_tools method
        tools_result = mcp.get_tools()
        if hasattr(tools_result, '__await__'):
            tools = await tools_result
        else:
            tools = tools_result
            
        return JSONResponse({
            "status": "success",
            "tool_count": len(tools),
            "tools": list(tools)
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)

# Debug endpoint for troubleshooting
@mcp.custom_route("/debug", methods=["GET"])
async def debug_endpoint(request):
    """Debug endpoint for troubleshooting deployment issues."""
    from starlette.responses import JSONResponse
    import os
    return JSONResponse({
        "status": "debug",
        "port": os.getenv("PORT", "8000"),
        "host": "0.0.0.0",
        "mcp_version": "2024-11-05",
        "server_name": "TanukiMCP Maestro",
        "endpoints": ["/health", "/tools", "/debug"],
        "transport": "STDIO (Smithery wraps with WebSocket)",
        "note": "Using STDIO transport - Smithery automatically wraps with WebSocket proxy"
    })

# Global variables for lazy loading
_maestro_tools = None
_computational_tools = None

def get_maestro_tools():
    """Lazy load maestro tools only when needed."""
    global _maestro_tools
    if _maestro_tools is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from maestro_tools import MaestroTools
        _maestro_tools = MaestroTools()
    return _maestro_tools

def get_computational_tools():
    """Lazy load computational tools only when needed."""
    global _computational_tools
    if _computational_tools is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from computational_tools import ComputationalTools
        _computational_tools = ComputationalTools()
    return _computational_tools

# Tool 1: Maestro Orchestration
@mcp.tool()
async def maestro_orchestrate(
    ctx,
    task_description: str,
    context: Optional[Dict[str, Any]] = None,
    complexity_level: str = "moderate",
    quality_threshold: float = 0.8,
    resource_level: str = "moderate",
    reasoning_focus: str = "auto",
    validation_rigor: str = "standard",
    max_iterations: int = 3,
    domain_specialization: str = "",
    enable_collaboration_fallback: bool = True
) -> str:
    """
    Enhanced meta-reasoning orchestration with collaborative fallback.
    
    Amplifies LLM capabilities 3-5x through multi-agent validation, iterative refinement,
    and quality control. Supports complex reasoning, research, analysis, and problem-solving
    with operator profiles and dynamic workflow planning.
    """
    tools = get_maestro_tools()
    return await tools.orchestrate_task(
        ctx,
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

# Tool 2: Collaboration Response Handler
@mcp.tool()
async def maestro_collaboration_response(
    ctx,
    collaboration_id: str,
    responses: Dict[str, Any],
    approval_status: str,
    additional_guidance: str = ""
) -> str:
    """
    Handle user responses during collaborative workflows.
    
    Processes user input and continues orchestration with provided guidance.
    """
    tools = get_maestro_tools()
    return await tools.handle_collaboration_response(
        ctx,
        collaboration_id=collaboration_id,
        responses=responses,
        approval_status=approval_status,
        additional_guidance=additional_guidance
    )

# Tool 3: IAE Discovery
@mcp.tool()
async def maestro_iae_discovery(
    ctx,
    task_type: str,
    requirements: Optional[List[str]] = None,
    complexity: str = "medium"
) -> str:
    """
    Discover and recommend optimal Intelligence Amplification Engine (IAE) based on task requirements.
    
    Analyzes computational needs and suggests best engine configurations.
    """
    tools = get_maestro_tools()
    return await tools.iae_discovery(
        ctx,
        task_type=task_type,
        requirements=requirements or [],
        complexity=complexity
    )

# Tool 4: Tool Selection
@mcp.tool()
async def maestro_tool_selection(
    ctx,
    task_description: str,
    available_tools: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> str:
    """
    Intelligent tool selection and recommendation based on task analysis.
    
    Provides optimal tool combinations and usage strategies.
    """
    tools = get_maestro_tools()
    return await tools.tool_selection(
        ctx,
        task_description=task_description,
        available_tools=available_tools or [],
        constraints=constraints or {}
    )

# Tool 5: Intelligence Amplification Engine
@mcp.tool()
async def maestro_iae(
    ctx,
    analysis_request: str,
    engine_type: str = "auto",
    data: Optional[Any] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Intelligence Amplification Engine for advanced computational analysis.
    
    Supports mathematical, quantum physics, data analysis, language enhancement,
    and code quality engines.
    """
    comp_tools = get_computational_tools()
    return await comp_tools.intelligence_amplification_engine(
        ctx,
        analysis_request=analysis_request,
        engine_type=engine_type,
        data=data,
        parameters=parameters or {}
    )

# Tool 6: Available Engines
@mcp.tool()
async def get_available_engines(
    ctx,
    detailed: bool = False
) -> str:
    """
    Get list of available Intelligence Amplification Engines and their capabilities.
    """
    comp_tools = get_computational_tools()
    return await comp_tools.get_available_engines(ctx, detailed=detailed)

# Tool 7: Enhanced Search
@mcp.tool()
async def maestro_search(
    ctx,
    query: str,
    max_results: int = 10,
    search_type: str = "comprehensive",
    temporal_filter: str = "recent",
    output_format: str = "detailed"
) -> str:
    """
    Enhanced web search with LLM-powered analysis and filtering.
    
    Provides intelligent search results with temporal filtering and result formatting.
    """
    tools = get_maestro_tools()
    return await tools.enhanced_search(
        ctx,
        query=query,
        max_results=max_results,
        search_type=search_type,
        temporal_filter=temporal_filter,
        output_format=output_format
    )

# Tool 8: Intelligent Scraping
@mcp.tool()
async def maestro_scrape(
    ctx,
    url: str,
    extraction_type: str = "text",
    content_filter: str = "relevant",
    output_format: str = "structured"
) -> str:
    """
    Intelligent web scraping with content extraction and filtering.
    
    Extracts and processes web content with smart filtering and formatting.
    """
    tools = get_maestro_tools()
    return await tools.intelligent_scrape(
        ctx,
        url=url,
        extraction_type=extraction_type,
        content_filter=content_filter,
        output_format=output_format
    )

# Tool 9: Secure Execution
@mcp.tool()
async def maestro_execute(
    ctx,
    execution_type: str = "code",
    content: str = "",
    language: str = "python",
    security_level: str = "standard",
    timeout: int = 30
) -> str:
    """
    Secure code execution with sandboxing and safety controls.
    
    Executes code in isolated environments with comprehensive security measures.
    """
    tools = get_maestro_tools()
    return await tools.secure_execute(
        ctx,
        execution_type=execution_type,
        content=content,
        language=language,
        security_level=security_level,
        timeout=timeout
    )

# Tool 10: Temporal Reasoning
@mcp.tool()
async def maestro_temporal_context(
    ctx,
    query: str,
    time_scope: str = "current",
    context_depth: str = "moderate",
    currency_check: bool = True
) -> str:
    """
    Advanced temporal reasoning and context analysis.
    
    Analyzes temporal dependencies and provides time-aware insights.
    """
    tools = get_maestro_tools()
    return await tools.temporal_reasoning(
        ctx,
        query=query,
        time_scope=time_scope,
        context_depth=context_depth,
        currency_check=currency_check
    )

# Tool 11: Error Handler
@mcp.tool()
async def maestro_error_handler(
    ctx,
    error_context: str,
    error_type: str = "general",
    recovery_mode: str = "automatic",
    learning_enabled: bool = True
) -> str:
    """
    Intelligent error analysis and recovery system.
    
    Analyzes errors, suggests fixes, and learns from patterns for improved handling.
    """
    tools = get_maestro_tools()
    return await tools.intelligent_error_handler(
        ctx,
        error_context=error_context,
        error_type=error_type,
        recovery_mode=recovery_mode,
        learning_enabled=learning_enabled
    )

# Simple HTTP endpoint for tool listing (for Smithery scanning)
@mcp.custom_route("/tools/list", methods=["GET"])
async def http_tools_list(request):
    """Return a JSON list of tool names for instant scanning without session requirements."""
    from starlette.responses import JSONResponse
    try:
        tools_result = mcp.get_tools()
        if hasattr(tools_result, '__await__'):
            tools = await tools_result
        else:
            tools = tools_result
        return JSONResponse({
            "tools": list(tools)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Production server startup
if __name__ == "__main__":
    print("🚀 TanukiMCP Maestro - PRODUCTION SERVER")
    print("⚡ Discovery: INSTANT (<100ms) via FastMCP 2.0")
    print("🛠️ Tools: 11 production-grade tools")
    print("🌐 Protocol: MCP 2024-11-05")
    print("☁️ Smithery.ai: Compatible")
    print("🎯 Ready for deployment!")
    
    # Get port from environment variable (Smithery.ai requirement)
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Required for container deployment
    
    print(f"🌐 Starting HTTP server on {host}:{port}")
    print(f"📡 MCP endpoint: http://{host}:{port}/mcp")
    print(f"🏥 Health check: http://{host}:{port}/health")
    
    # Debug: Print available tools
    try:
        # Try to get tools list - note this might be async in some versions
        if hasattr(mcp, 'get_tools'):
            try:
                tools = mcp.get_tools()
                # Handle if it returns a coroutine
                if hasattr(tools, '__await__'):
                    print("🔧 Tools method is async - skipping debug listing")
                else:
                    print(f"🔧 Registered tools: {len(tools)}")
                    for tool in tools:
                        print(f"  - {tool}")
            except Exception as tools_err:
                print(f"⚠️ Error getting tools: {tools_err}")
        else:
            print("🔧 No get_tools method available")
    except Exception as e:
        print(f"⚠️ Error in tools debug: {e}")
    
    try:
        # Run FastMCP server with Streamable HTTP transport for Smithery.ai
        print("🚀 Starting server with Streamable HTTP transport...")
        print("📡 MCP endpoint: http://{host}:{port}/mcp")
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            path="/mcp"
        )
    except Exception as e:
        print(f"❌ Server startup with HTTP transport failed: {e}")
        import traceback
        traceback.print_exc() 