#!/usr/bin/env python3
"""
Maestro MCP Server - stdio transport implementation
Ultra-lightweight with static tool definitions for fast Smithery scanning
"""

import asyncio
import logging
import sys
import os
from typing import Any, Dict, List

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Configure logging to stderr to avoid interfering with MCP protocol on stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Create the server instance
server = Server("maestro-mcp", version="2.0.0")

# NO IMPORTS FROM SRC/ AT MODULE LEVEL - Everything deferred to call time
# This ensures ultra-fast startup for Smithery tool scanning

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Handle tools/list requests - return ALL tools with STATIC definitions only"""
    logger.info("Handling list_tools request")
    
    # COMPLETELY STATIC tool definitions - no imports, no instantiation, no computations
    return [
        types.Tool(
            name="maestro_orchestrate",
            description="🎭 Intelligent workflow orchestration for complex tasks using Mixture-of-Agents (MoA)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "The complex task to orchestrate"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for the task",
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
                        "description": "Maximum number of results",
                        "default": 10
                    },
                    "search_engine": {
                        "type": "string",
                        "enum": ["duckduckgo", "google", "bing"],
                        "description": "Search engine to use",
                        "default": "duckduckgo"
                    },
                    "temporal_filter": {
                        "type": "string",
                        "enum": ["any", "recent", "week", "month", "year"],
                        "description": "Time filter for results",
                        "default": "any"
                    },
                    "result_format": {
                        "type": "string",
                        "enum": ["structured", "summary", "detailed"],
                        "description": "Format of results",
                        "default": "structured"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="maestro_scrape",
            description="📑 Web scraping functionality with content extraction",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "text", "html", "json"],
                        "description": "Output format",
                        "default": "markdown"
                    },
                    "selectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CSS selectors for specific elements"
                    },
                    "wait_time": {
                        "type": "number",
                        "description": "Time to wait for page load (seconds)",
                        "default": 3
                    },
                    "extract_links": {
                        "type": "boolean",
                        "description": "Whether to extract links",
                        "default": False
                    }
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="maestro_execute",
            description="⚙️ Execute computational tasks with enhanced error handling",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command or code to execute"
                    },
                    "execution_context": {
                        "type": "object",
                        "description": "Execution context",
                        "additionalProperties": True
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    },
                    "safe_mode": {
                        "type": "boolean",
                        "description": "Enable safe execution mode",
                        "default": True
                    }
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="maestro_error_handler",
            description="🚨 Advanced error handling and recovery suggestions",
            inputSchema={
                "type": "object",
                "properties": {
                    "error_message": {
                        "type": "string",
                        "description": "The error message to analyze"
                    },
                    "error_context": {
                        "type": "object",
                        "description": "Context where the error occurred",
                        "additionalProperties": True
                    },
                    "recovery_suggestions": {
                        "type": "boolean",
                        "description": "Whether to provide recovery suggestions",
                        "default": True
                    }
                },
                "required": ["error_message"]
            }
        ),
        types.Tool(
            name="maestro_temporal_context",
            description="📅 Temporal context awareness and time-based reasoning",
            inputSchema={
                "type": "object",
                "properties": {
                    "temporal_query": {
                        "type": "string",
                        "description": "Query requiring temporal reasoning"
                    },
                    "time_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        },
                        "description": "Time range for the query"
                    },
                    "temporal_precision": {
                        "type": "string",
                        "enum": ["year", "month", "day", "hour", "minute"],
                        "description": "Required temporal precision",
                        "default": "day"
                    }
                },
                "required": ["temporal_query"]
            }
        ),
        types.Tool(
            name="get_available_engines",
            description="🔧 Get list of available computational engines and their capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "engine_type": {
                        "type": "string",
                        "enum": ["all", "statistical", "mathematical", "quantum", "enhanced"],
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
        )
    ]

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
        from maestro.enhanced_tools import EnhancedToolHandlers
        _enhanced_tool_handlers_instance = EnhancedToolHandlers()
    return _enhanced_tool_handlers_instance

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
        comp_tools = get_computational_tools_instance()
        
        analysis_request = arguments.get("analysis_request", "")
        engine_type = arguments.get("engine_type", "auto")
        precision_level = arguments.get("precision_level", "standard")
        computational_context = arguments.get("computational_context", {})
        
        # Execute IAE analysis using computational tools
        result = await comp_tools.integrated_analysis_engine(
            analysis_request, engine_type, precision_level, computational_context
        )
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_iae: {e}")
        return [types.TextContent(type="text", text=f"Error in IAE analysis: {str(e)}")]

async def _handle_maestro_orchestrate(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_orchestrate tool calls"""
    try:
        maestro_tools = get_maestro_tools_instance()
        
        task_description = arguments.get("task_description", "")
        context = arguments.get("context", {})
        success_criteria = arguments.get("success_criteria", {})
        complexity_level = arguments.get("complexity_level", "moderate")
        
        # For stdio mode, create a mock context since we don't have FastMCP Context
        class MockContext:
            async def sample(self, prompt: str, response_format: Dict[str, Any] = None):
                # For stdio mode, we'll return a simplified orchestration plan
                # In the full HTTP version, this would use actual LLM sampling
                class MockResponse:
                    def json(self):
                        return {
                            "orchestration_plan": {
                                "steps": [
                                    {"step": 1, "action": "Analyze task requirements", "tools": ["maestro_tool_selection"]},
                                    {"step": 2, "action": "Execute primary computation", "tools": ["maestro_iae"]},
                                    {"step": 3, "action": "Validate results", "tools": ["maestro_error_handler"]},
                                    {"step": 4, "action": "Provide comprehensive output", "tools": ["maestro_temporal_context"]}
                                ],
                                "complexity": complexity_level,
                                "estimated_duration": "2-5 minutes"
                            }
                        }
                    
                    @property
                    def text(self):
                        return f"""# Orchestration Plan for: {task_description}

## Task Analysis
- **Complexity Level**: {complexity_level}
- **Context**: {context}
- **Success Criteria**: {success_criteria}

## Execution Plan
1. **Task Analysis** - Use maestro_tool_selection to identify optimal tools
2. **Core Processing** - Execute using maestro_iae with appropriate engine
3. **Quality Assurance** - Validate results with maestro_error_handler
4. **Final Integration** - Apply temporal context if needed

## Next Steps
The orchestration system would now execute these steps in sequence, 
coordinating between the different Maestro tools to achieve the desired outcome.
"""
                
                return MockResponse()
        
        mock_context = MockContext()
        result = await maestro_tools.orchestrate(mock_context, task_description, context, success_criteria, complexity_level)
        
        return [types.TextContent(type="text", text=result)]
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
        
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        search_engine = arguments.get("search_engine", "duckduckgo")
        temporal_filter = arguments.get("temporal_filter", "any")
        result_format = arguments.get("result_format", "structured")
        
        result = await enhanced_tools.search(query, max_results, search_engine, temporal_filter, result_format)
        
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Error in maestro_search: {e}")
        return [types.TextContent(type="text", text=f"Error in search: {str(e)}")]

async def _handle_maestro_scrape(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_scrape tool calls"""
    try:
        enhanced_tools = get_enhanced_tool_handlers_instance()
        
        url = arguments.get("url", "")
        output_format = arguments.get("output_format", "markdown") 
        selectors = arguments.get("selectors", [])
        wait_time = arguments.get("wait_time", 3)
        extract_links = arguments.get("extract_links", False)
        
        result = await enhanced_tools.scrape(url, output_format, selectors, wait_time, extract_links)
        
        return [types.TextContent(type="text", text=result)]
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

async def main():
    """Run the MCP server with stdio transport"""
    logger.info("Starting Maestro MCP Server with stdio transport")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 