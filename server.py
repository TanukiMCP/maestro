#!/usr/bin/env python3
"""
TanukiMCP Maestro - Production-Ready MCP Server

Fast-loading MCP server with 11 production tools optimized for Smithery.ai deployment.
All heavy imports are deferred to ensure <100ms tool discovery.
"""

import os
import asyncio
import sys
from typing import Dict, Any, Optional, List

# CRITICAL: Pre-define tools as pure dictionaries for INSTANT access (Smithery requirement)
# NO imports, NO processing, NO conversion - just raw data
INSTANT_TOOLS_RAW = [
    {
        "name": "maestro_orchestrate",
        "description": "Enhanced meta-reasoning orchestration with collaborative fallback. Amplifies LLM capabilities 3-5x through multi-agent validation, iterative refinement, and quality control. Supports complex reasoning, research, analysis, and problem-solving with operator profiles and dynamic workflow planning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "Detailed description of the task to orchestrate"},
                "context": {"type": "object", "description": "Additional context or data for the task", "default": {}},
                "complexity_level": {"type": "string", "enum": ["basic", "moderate", "advanced", "expert"], "description": "Complexity level of the task", "default": "moderate"},
                "quality_threshold": {"type": "number", "minimum": 0.7, "maximum": 0.95, "description": "Minimum acceptable solution quality (0.7-0.95)", "default": 0.8},
                "resource_level": {"type": "string", "enum": ["limited", "moderate", "abundant"], "description": "Available computational resources", "default": "moderate"},
                "reasoning_focus": {"type": "string", "enum": ["logical", "creative", "analytical", "research", "synthesis", "auto"], "description": "Primary reasoning approach", "default": "auto"},
                "validation_rigor": {"type": "string", "enum": ["basic", "standard", "thorough", "rigorous"], "description": "Multi-agent validation thoroughness", "default": "standard"},
                "max_iterations": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Maximum refinement cycles", "default": 3},
                "domain_specialization": {"type": "string", "description": "Preferred domain expertise focus", "default": ""},
                "enable_collaboration_fallback": {"type": "boolean", "description": "Enable intelligent collaboration when ambiguity detected", "default": True}
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "maestro_collaboration_response",
        "description": "Handle user responses during collaborative workflows. Processes user input and continues orchestration with provided guidance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collaboration_id": {"type": "string", "description": "Unique collaboration session identifier"},
                "responses": {"type": "object", "description": "User responses to collaboration questions"},
                "approval_status": {"type": "string", "enum": ["approved", "rejected", "modified"], "description": "User approval status for proposed approach"},
                "additional_guidance": {"type": "string", "description": "Additional user guidance or modifications", "default": ""}
            },
            "required": ["collaboration_id", "responses", "approval_status"]
        }
    },
    {
        "name": "maestro_iae_discovery",
        "description": "Discover and recommend optimal Intelligence Amplification Engine (IAE) based on task requirements. Analyzes computational needs and suggests best engine configurations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Type of computational task"},
                "requirements": {"type": "array", "items": {"type": "string"}, "description": "Specific computational requirements"},
                "complexity": {"type": "string", "enum": ["low", "medium", "high", "expert"], "default": "medium"}
            },
            "required": ["task_type"]
        }
    },
    {
        "name": "maestro_tool_selection",
        "description": "Intelligent tool selection and recommendation based on task analysis. Provides optimal tool combinations and usage strategies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "Description of the task requiring tools"},
                "available_tools": {"type": "array", "items": {"type": "string"}, "description": "List of available tools to choose from", "default": []},
                "constraints": {"type": "object", "description": "Tool selection constraints", "default": {}}
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "maestro_iae",
        "description": "Intelligence Amplification Engine for advanced computational analysis. Supports mathematical, quantum physics, data analysis, language enhancement, and code quality engines.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis_request": {"type": "string", "description": "Description of the analysis to perform"},
                "engine_type": {"type": "string", "enum": ["mathematical", "quantum_physics", "data_analysis", "language_enhancement", "code_quality", "auto"], "description": "Specific engine to use or 'auto' for automatic selection", "default": "auto"},
                "data": {"type": ["string", "object", "array"], "description": "Input data for analysis"},
                "parameters": {"type": "object", "description": "Engine-specific parameters", "default": {}}
            },
            "required": ["analysis_request"]
        }
    },
    {
        "name": "get_available_engines",
        "description": "Get list of available Intelligence Amplification Engines and their capabilities.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "detailed": {"type": "boolean", "description": "Return detailed engine information", "default": False}
            }
        }
    },
    {
        "name": "maestro_search",
        "description": "Enhanced web search with LLM-powered analysis and filtering. Provides intelligent search results with temporal filtering and result formatting.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to execute"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Maximum number of results to return", "default": 10},
                "search_type": {"type": "string", "enum": ["comprehensive", "focused", "academic", "news"], "description": "Type of search to perform", "default": "comprehensive"},
                "temporal_filter": {"type": "string", "enum": ["any", "recent", "week", "month", "year"], "description": "Time-based filtering", "default": "recent"},
                "output_format": {"type": "string", "enum": ["detailed", "summary", "structured"], "description": "Format of search results", "default": "detailed"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "maestro_scrape",
        "description": "Intelligent web scraping with content extraction and structured data processing. Handles dynamic content and provides clean, formatted output.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to scrape content from"},
                "extraction_type": {"type": "string", "enum": ["text", "structured", "media", "links"], "description": "Type of content to extract", "default": "text"},
                "content_filter": {"type": "string", "enum": ["all", "relevant", "main"], "description": "Content filtering level", "default": "relevant"},
                "output_format": {"type": "string", "enum": ["markdown", "json", "text", "html"], "description": "Output format", "default": "structured"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "maestro_execute",
        "description": "Secure code execution sandbox for Python, JavaScript, and shell commands. Enforces security policies and resource limits.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "execution_type": {"type": "string", "enum": ["code", "command", "script"], "description": "Type of execution", "default": "code"},
                "content": {"type": "string", "description": "Code or command to execute"},
                "language": {"type": "string", "enum": ["python", "javascript", "shell", "bash"], "description": "Programming language", "default": "python"},
                "security_level": {"type": "string", "enum": ["standard", "strict", "minimal"], "description": "Security enforcement level", "default": "standard"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300, "description": "Execution timeout in seconds", "default": 30}
            },
            "required": ["content"]
        }
    },
    {
        "name": "maestro_temporal_context",
        "description": "Provides temporal reasoning and context awareness. Analyzes time-sensitive queries and ensures information currency.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query requiring temporal context analysis"},
                "time_scope": {"type": "string", "enum": ["current", "historical", "predictive", "comparative"], "description": "Temporal scope of analysis", "default": "current"},
                "context_depth": {"type": "string", "enum": ["surface", "moderate", "deep"], "description": "Depth of contextual analysis", "default": "moderate"},
                "currency_check": {"type": "boolean", "description": "Verify information currency", "default": True}
            },
            "required": ["query"]
        }
    },
    {
        "name": "maestro_error_handler",
        "description": "Intelligent error analysis and recovery. Diagnoses issues, suggests solutions, and can attempt automated recovery for common problems.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "error_context": {"type": "string", "description": "Context or description of the error"},
                "error_type": {"type": "string", "enum": ["general", "technical", "logical", "data"], "description": "Type of error encountered", "default": "general"},
                "recovery_mode": {"type": "string", "enum": ["automatic", "guided", "manual"], "description": "Recovery approach", "default": "automatic"},
                "learning_enabled": {"type": "boolean", "description": "Enable learning from error patterns", "default": True}
            },
            "required": ["error_context"]
        }
    }
]

print(f"✅ Instant tools pre-defined: {len(INSTANT_TOOLS_RAW)} tools available")

# Lazy loading variables
_mcp = None
_instant_tools_converted = None

def get_instant_tools():
    """Convert raw tools to MCP Tool objects only when needed"""
    global _instant_tools_converted
    if _instant_tools_converted is None:
        from mcp.types import Tool
        _instant_tools_converted = []
        for tool_dict in INSTANT_TOOLS_RAW:
            tool = Tool(
                name=tool_dict["name"],
                description=tool_dict["description"],
                inputSchema=tool_dict["inputSchema"]
            )
            _instant_tools_converted.append(tool)
        print(f"🚀 Converted {len(_instant_tools_converted)} tools to MCP format")
    return _instant_tools_converted

def get_mcp():
    """Lazy load FastMCP only when needed for actual server operations"""
    global _mcp
    if _mcp is None:
        from fastmcp import FastMCP
        _mcp = FastMCP("TanukiMCP Maestro")
        
        # Override get_tools for instant response
        original_get_tools = _mcp.get_tools
        
        async def instant_get_tools():
            """Provide instant tool listing using pre-converted static tools"""
            return get_instant_tools()
        
        # Replace FastMCP's get_tools with our instant version
        _mcp.get_tools = instant_get_tools
        
        # Now register all the tool implementations
        register_tools(_mcp)
        
    return _mcp

# Create a property to access mcp that triggers lazy loading
class MCPProxy:
    def __getattr__(self, name):
        return getattr(get_mcp(), name)
    
    async def get_tools(self):
        """Direct instant tool access without FastMCP overhead"""
        return get_instant_tools()

# Use proxy for instant access
mcp = MCPProxy()

# Global variables for lazy loading (only used during tool execution)
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

def register_tools(mcp_instance):
    """Register all tool implementations with FastMCP"""
    
    # Tool 1: Maestro Orchestration
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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
    @mcp_instance.tool()
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

# Production server startup
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting HTTP MCP server on port {port} (path: /mcp)")
    get_mcp().run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp") 