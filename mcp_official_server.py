#!/usr/bin/env python3
"""
Official MCP Server Implementation for TanukiMCP Maestro
Using the standard MCP Python SDK with proper lazy loading for Smithery compatibility.

Based on successful Smithery deployment patterns from:
- MCP-Atlassian: https://smithery.ai/server/mcp-atlassian  
- Memory Cache Server: https://smithery.ai/server/@ibproduct/ib-mcp-cache-server
"""

import os
import sys
import asyncio
import json
from typing import Any, Dict, List, Optional, Union
import logging

# Only import MCP - keep all heavy imports lazy
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import sse_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    ListToolsResult
)

# Global lazy-loaded instances
_maestro_tools = None
_computational_tools = None
_enhanced_tool_handlers = None

# Logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def get_maestro_tools():
    """Lazy load MaestroTools only when first needed"""
    global _maestro_tools
    if _maestro_tools is None:
        try:
            from src.maestro_tools import MaestroTools
            _maestro_tools = MaestroTools()
            logger.info("MaestroTools initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MaestroTools: {e}")
            _maestro_tools = None
    return _maestro_tools

def get_computational_tools():
    """Lazy load ComputationalTools only when first needed"""
    global _computational_tools
    if _computational_tools is None:
        try:
            from src.computational_tools import ComputationalTools
            _computational_tools = ComputationalTools()
            logger.info("ComputationalTools initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ComputationalTools: {e}")
            _computational_tools = None
    return _computational_tools

def get_enhanced_tool_handlers():
    """Lazy load EnhancedToolHandlers only when first needed"""
    global _enhanced_tool_handlers
    if _enhanced_tool_handlers is None:
        try:
            from src.enhanced_tools import EnhancedToolHandlers
            _enhanced_tool_handlers = EnhancedToolHandlers()
            logger.info("EnhancedToolHandlers initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedToolHandlers: {e}")
            _enhanced_tool_handlers = None
    return _enhanced_tool_handlers

# Create the MCP server
app = Server("tanukimcp-maestro")

# Static tool definitions for instant discovery (no heavy imports needed)
STATIC_TOOLS = [
    Tool(
        name="maestro_orchestrate",
        description="Enhanced meta-reasoning orchestration with collaborative fallback. Amplifies LLM capabilities 3-5x through multi-agent validation, iterative refinement, and quality control. Supports complex reasoning, research, analysis, and problem-solving with operator profiles and dynamic workflow planning.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Detailed description of the task to orchestrate"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context or data for the task",
                    "default": {}
                },
                "complexity_level": {
                    "type": "string",
                    "enum": ["basic", "moderate", "advanced", "expert"],
                    "description": "Complexity level of the task",
                    "default": "moderate"
                },
                "quality_threshold": {
                    "type": "number",
                    "minimum": 0.7,
                    "maximum": 0.95,
                    "description": "Minimum acceptable solution quality (0.7-0.95)",
                    "default": 0.8
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
                    "description": "Primary reasoning approach",
                    "default": "auto"
                },
                "validation_rigor": {
                    "type": "string",
                    "enum": ["basic", "standard", "thorough", "rigorous"],
                    "description": "Multi-agent validation thoroughness",
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
                    "description": "Preferred domain expertise focus",
                    "default": ""
                },
                "enable_collaboration_fallback": {
                    "type": "boolean",
                    "description": "Enable intelligent collaboration when ambiguity detected",
                    "default": True
                }
            },
            "required": ["task_description"]
        }
    ),
    Tool(
        name="maestro_collaboration_response",
        description="Handle user responses during collaborative workflows. Processes user input and continues orchestration with provided guidance.",
        inputSchema={
            "type": "object",
            "properties": {
                "collaboration_id": {
                    "type": "string",
                    "description": "Unique collaboration session identifier"
                },
                "responses": {
                    "type": "object",
                    "description": "User responses to collaboration questions"
                },
                "approval_status": {
                    "type": "string",
                    "enum": ["approved", "rejected", "modified"],
                    "description": "User approval status for proposed approach"
                },
                "additional_guidance": {
                    "type": "string",
                    "description": "Additional user guidance or modifications",
                    "default": ""
                }
            },
            "required": ["collaboration_id", "responses", "approval_status"]
        }
    ),
    Tool(
        name="maestro_iae_discovery",
        description="Discover and recommend optimal Intelligence Amplification Engine (IAE) based on task requirements. Analyzes computational needs and suggests best engine configurations.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "Type of computational task"
                },
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific computational requirements"
                },
                "complexity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "expert"],
                    "default": "medium"
                }
            },
            "required": ["task_type"]
        }
    ),
    Tool(
        name="maestro_tool_selection",
        description="Intelligent tool selection and recommendation based on task analysis. Provides optimal tool combinations and usage strategies.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task requiring tools"
                },
                "available_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of available tools to choose from",
                    "default": []
                },
                "constraints": {
                    "type": "object",
                    "description": "Tool selection constraints",
                    "default": {}
                }
            },
            "required": ["task_description"]
        }
    ),
    Tool(
        name="maestro_iae",
        description="Intelligence Amplification Engine for advanced computational analysis. Supports mathematical, quantum physics, data analysis, language enhancement, and code quality engines.",
        inputSchema={
            "type": "object",
            "properties": {
                "analysis_request": {
                    "type": "string",
                    "description": "Description of the analysis to perform"
                },
                "engine_type": {
                    "type": "string",
                    "enum": ["mathematical", "quantum_physics", "data_analysis", "language_enhancement", "code_quality", "auto"],
                    "description": "Specific engine to use or 'auto' for automatic selection",
                    "default": "auto"
                },
                "data": {
                    "type": ["string", "object", "array"],
                    "description": "Input data for analysis"
                },
                "parameters": {
                    "type": "object",
                    "description": "Engine-specific parameters",
                    "default": {}
                }
            },
            "required": ["analysis_request"]
        }
    ),
    Tool(
        name="get_available_engines",
        description="Get list of available Intelligence Amplification Engines and their capabilities.",
        inputSchema={
            "type": "object",
            "properties": {
                "detailed": {
                    "type": "boolean",
                    "description": "Return detailed engine information",
                    "default": False
                }
            }
        }
    ),
    Tool(
        name="maestro_search",
        description="Enhanced web search with LLM-powered analysis and filtering. Provides intelligent search results with temporal filtering and result formatting.",
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
                    "maximum": 20,
                    "description": "Maximum number of results",
                    "default": 10
                },
                "temporal_filter": {
                    "type": "string",
                    "enum": ["recent", "week", "month", "year", "all"],
                    "description": "Time-based result filtering",
                    "default": "all"
                },
                "result_format": {
                    "type": "string",
                    "enum": ["summary", "detailed", "urls_only"],
                    "description": "Format of search results",
                    "default": "summary"
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific domains to search within",
                    "default": []
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="maestro_scrape",
        description="Intelligent web scraping with content extraction and structured data processing. Handles dynamic content and provides clean, formatted output.",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to scrape"
                },
                "extraction_type": {
                    "type": "string",
                    "enum": ["text", "structured", "images", "links", "all"],
                    "description": "Type of content to extract",
                    "default": "text"
                },
                "selectors": {
                    "type": "object",
                    "description": "CSS selectors for specific content",
                    "default": {}
                },
                "wait_for": {
                    "type": "string",
                    "description": "CSS selector to wait for before scraping",
                    "default": ""
                }
            },
            "required": ["url"]
        }
    ),
    Tool(
        name="maestro_execute",
        description="Secure code and workflow execution with validation. Supports multiple languages and execution modes with comprehensive safety checks.",
        inputSchema={
            "type": "object",
            "properties": {
                "execution_type": {
                    "type": "string",
                    "enum": ["code", "workflow", "plan"],
                    "description": "Type of execution to perform"
                },
                "content": {
                    "type": "string",
                    "description": "Code, workflow definition, or plan to execute"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "bash", "sql", "auto"],
                    "description": "Programming language for code execution",
                    "default": "auto"
                },
                "environment": {
                    "type": "object",
                    "description": "Execution environment variables",
                    "default": {}
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 300,
                    "description": "Execution timeout in seconds",
                    "default": 30
                },
                "validation_level": {
                    "type": "string",
                    "enum": ["none", "basic", "strict"],
                    "description": "Code validation rigor",
                    "default": "basic"
                }
            },
            "required": ["execution_type", "content"]
        }
    ),
    Tool(
        name="maestro_temporal_context",
        description="Time-aware reasoning and context analysis. Provides temporal insights, information currency assessment, and time-based recommendations.",
        inputSchema={
            "type": "object",
            "properties": {
                "context_request": {
                    "type": "string",
                    "description": "Description of temporal context needed"
                },
                "time_frame": {
                    "type": "string",
                    "description": "Relevant time frame for analysis",
                    "default": "current"
                },
                "temporal_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific temporal factors to consider",
                    "default": []
                }
            },
            "required": ["context_request"]
        }
    ),
    Tool(
        name="maestro_error_handler",
        description="Intelligent error analysis and recovery suggestions. Provides adaptive error handling with context-aware recovery strategies.",
        inputSchema={
            "type": "object",
            "properties": {
                "error_context": {
                    "type": "string",
                    "description": "Description of the error or failure"
                },
                "error_details": {
                    "type": "object",
                    "description": "Specific error information",
                    "default": {}
                },
                "recovery_preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred recovery approaches",
                    "default": []
                }
            },
            "required": ["error_context"]
        }
    )
]

@app.list_tools()
async def list_tools() -> ListToolsResult:
    """List all available tools - returns instantly without heavy imports"""
    return ListToolsResult(tools=STATIC_TOOLS)

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls with lazy loading"""
    try:
        if name.startswith("maestro_") and name != "maestro_iae":
            # Handle MaestroTools
            tools = get_maestro_tools()
            if tools is None:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text="Error: MaestroTools not available. Please check server configuration."
                    )]
                )
            
            # Route to appropriate handler
            if hasattr(tools, 'handle_tool_call'):
                result = await tools.handle_tool_call(name, arguments)
                if isinstance(result, list):
                    return CallToolResult(content=result)
                else:
                    return CallToolResult(content=[TextContent(type="text", text=str(result))])
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text=f"Error: Tool handler method not found for {name}"
                    )]
                )
        
        elif name == "maestro_iae":
            # Handle ComputationalTools  
            tools = get_computational_tools()
            if tools is None:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text="Error: ComputationalTools not available. Please check server configuration."
                    )]
                )
            
            result = await tools.handle_iae_request(
                arguments.get("analysis_request", ""),
                arguments.get("engine_type", "auto"),
                arguments.get("data"),
                arguments.get("parameters", {})
            )
            
            if isinstance(result, list):
                return CallToolResult(content=result)
            else:
                return CallToolResult(content=[TextContent(type="text", text=str(result))])
        
        elif name == "get_available_engines":
            # Handle engine discovery
            tools = get_computational_tools()
            if tools is None:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text="Error: ComputationalTools not available. Please check server configuration."
                    )]
                )
            
            detailed = arguments.get("detailed", False)
            engines = tools.get_available_engines(detailed)
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(engines, indent=2))]
            )
        
        else:
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Error: Unknown tool '{name}'"
                )]
            )
    
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text", 
                text=f"Error executing {name}: {str(e)}"
            )]
        )

async def main():
    """Main server entry point"""
    # Determine transport method
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    
    if transport == "sse":
        # SSE transport for HTTP/web deployment (Smithery)
        port = int(os.getenv("PORT", 8000))
        logger.info(f"Starting SSE server on port {port}")
        
        from mcp.server.sse import SseServerTransport
        transport_instance = SseServerTransport("/mcp")
        
        async with sse_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    else:
        # Default to stdio transport
        logger.info("Starting stdio server")
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, 
                write_stream, 
                app.create_initialization_options()
            )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1) 