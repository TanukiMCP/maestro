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
from mcp.types import Tool

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

# NEW: Pre-convert tools at module level
PRE_CONVERTED_MCP_TOOLS = []
for tool_dict_item in INSTANT_TOOLS_RAW:
    tool_obj = Tool(
        name=tool_dict_item["name"],
        description=tool_dict_item["description"],
        inputSchema=tool_dict_item["inputSchema"]
    )
    PRE_CONVERTED_MCP_TOOLS.append(tool_obj)

def get_instant_tools():
    """Return pre-converted MCP Tool objects."""
    # No more lazy loading or conversion here - directly return the pre-converted list
    return PRE_CONVERTED_MCP_TOOLS

# Import actual tool implementations when needed
def get_maestro_tools():
    """Lazy load MaestroTools only when needed"""
    from src.maestro_tools import MaestroTools
    return MaestroTools()

def get_computational_tools():
    """Lazy load ComputationalTools only when needed"""
    from src.computational_tools import ComputationalTools
    return ComputationalTools()

# Production server startup
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting HTTP MCP server on port {port} (path: /mcp)")
    print("Please use mcp_http_transport.py for Smithery-compatible deployment")
    print("This file is now only used for compatibility with existing tools")
    
    # Import the official MCP server implementation
    try:
        from mcp_official_server import app as mcp_app
        from mcp_http_transport import routes, app as http_app
        
        # Run the HTTP server using the official MCP SDK
        import uvicorn
        uvicorn.run(http_app, host="0.0.0.0", port=port)
    except ImportError as e:
        print(f"Error importing MCP server: {e}")
        print("Please make sure mcp_official_server.py and mcp_http_transport.py exist")
        sys.exit(1) 