#!/usr/bin/env python3
"""
Maestro MCP Server - stdio transport implementation
Uses existing sophisticated tools from src/ directory
"""

import asyncio
import logging
import sys
import os
from typing import Any, Dict, List

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Add src to path to import existing tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging to stderr to avoid interfering with MCP protocol on stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Create the server instance
server = Server("maestro-mcp", version="2.0.0")

# Lazy-loaded instances of your existing tools
_maestro_tools_instance = None
_computational_tools_instance = None

def get_maestro_tools_instance():
    global _maestro_tools_instance
    if _maestro_tools_instance is None:
        logger.info("Loading MaestroTools instance")
        from maestro_tools import MaestroTools 
        _maestro_tools_instance = MaestroTools()
    return _maestro_tools_instance

def get_computational_tools_instance():
    global _computational_tools_instance
    if _computational_tools_instance is None:
        logger.info("Loading ComputationalTools instance")
        from computational_tools import ComputationalTools
        _computational_tools_instance = ComputationalTools()
    return _computational_tools_instance

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Handle tools/list requests - return your existing sophisticated tools"""
    logger.info("Handling list_tools request")
    
    # Get tools from your existing computational tools
    comp_tools = get_computational_tools_instance()
    mcp_tools = comp_tools.get_mcp_tools()
    
    # Add your orchestration and other tools
    orchestration_tools = [
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
            name="get_available_engines",
            description="📋 Get list of available computational engines and their capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain_filter": {
                        "type": "string",
                        "description": "Filter engines by domain (optional)"
                    }
                }
            }
        )
    ]
    
    # Combine all tools
    all_tools = mcp_tools + orchestration_tools
    logger.info(f"Returning {len(all_tools)} tools")
    return all_tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> List[types.TextContent]:
    """Handle tools/call requests using your existing tool implementations"""
    logger.info(f"Handling call_tool request for: {name}")
    
    if arguments is None:
        arguments = {}
    
    try:
        # Route to your existing tool implementations
        if name == "maestro_iae":
            return await _handle_maestro_iae(arguments)
        elif name == "maestro_orchestrate":
            return await _handle_maestro_orchestrate(arguments)
        elif name == "maestro_iae_discovery":
            return await _handle_maestro_iae_discovery(arguments)
        elif name == "maestro_tool_selection":
            return await _handle_maestro_tool_selection(arguments)
        elif name == "get_available_engines":
            return await _handle_get_available_engines(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [
            types.TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}"
            )
        ]

async def _handle_maestro_iae(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_iae using your existing ComputationalTools"""
    comp_tools = get_computational_tools_instance()
    
    # Extract parameters
    engine_domain = arguments.get("engine_domain", "quantum_physics")
    computation_type = arguments.get("computation_type", "entanglement_entropy")
    parameters = arguments.get("parameters", {})
    precision_requirements = arguments.get("precision_requirements", "machine_precision")
    validation_level = arguments.get("validation_level", "standard")
    
    # Use your existing tool implementation
    result = await comp_tools.handle_tool_call("maestro_iae", {
        "engine_domain": engine_domain,
        "computation_type": computation_type,
        "parameters": parameters,
        "precision_requirements": precision_requirements,
        "validation_level": validation_level
    })
    
    return result

async def _handle_maestro_orchestrate(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_orchestrate using your existing MaestroTools"""
    maestro_tools = get_maestro_tools_instance()
    
    # Extract parameters
    task_description = arguments.get("task_description", "")
    context = arguments.get("context", {})
    success_criteria = arguments.get("success_criteria", {})
    complexity_level = arguments.get("complexity_level", "moderate")
    
    # Create a mock context for the orchestration
    class MockContext:
        async def sample(self, prompt: str, response_format: Dict[str, Any] = None):
            # For stdio mode, we'll return a simplified orchestration plan
            # In the full HTTP version, this would use actual LLM sampling
            class MockResponse:
                def json(self):
                    return {
                        "requires_moa": True,
                        "steps": [
                            {
                                "type": "reasoning",
                                "description": f"Analyzing task: {task_description}"
                            },
                            {
                                "type": "tool_call",
                                "tool_name": "maestro_iae",
                                "arguments": {
                                    "engine_domain": "quantum_physics",
                                    "computation_type": "entanglement_entropy",
                                    "parameters": {}
                                }
                            }
                        ],
                        "final_synthesis_required": True,
                        "moa_aggregation_strategy": "llm_synthesis"
                    }
            return MockResponse()
    
    # Use your existing orchestration implementation
    result = await maestro_tools.orchestrate_task(
        ctx=MockContext(),
        task_description=task_description,
        context=context,
        success_criteria=success_criteria,
        complexity_level=complexity_level
    )
    
    return [
        types.TextContent(
            type="text",
            text=result
        )
    ]

async def _handle_maestro_iae_discovery(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle IAE discovery using your existing tools"""
    maestro_tools = get_maestro_tools_instance()
    
    task_type = arguments.get("task_type", "general")
    domain_context = arguments.get("domain_context", "")
    complexity_requirements = arguments.get("complexity_requirements", {})
    
    result = await maestro_tools.discover_integrated_analysis_engines(
        task_type=task_type,
        domain_context=domain_context,
        complexity_requirements=complexity_requirements
    )
    
    return [
        types.TextContent(
            type="text",
            text=result
        )
    ]

async def _handle_maestro_tool_selection(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool selection using your existing tools"""
    maestro_tools = get_maestro_tools_instance()
    
    request_description = arguments.get("request_description", "")
    available_context = arguments.get("available_context", {})
    precision_requirements = arguments.get("precision_requirements", {})
    
    result = await maestro_tools.select_tools(
        request_description=request_description,
        available_context=available_context,
        precision_requirements=precision_requirements
    )
    
    return [
        types.TextContent(
            type="text",
            text=result
        )
    ]

async def _handle_get_available_engines(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle getting available engines"""
    comp_tools = get_computational_tools_instance()
    
    domain_filter = arguments.get("domain_filter")
    engines = comp_tools.get_available_engines()
    
    if domain_filter:
        engines = {k: v for k, v in engines.items() if domain_filter.lower() in k.lower()}
    
    result = f"""
🔧 **Available Computational Engines**

{len(engines)} engines available:

"""
    
    for engine_name, engine_info in engines.items():
        result += f"**{engine_name}**\n"
        result += f"  - Description: {engine_info.get('description', 'No description')}\n"
        result += f"  - Capabilities: {', '.join(engine_info.get('capabilities', []))}\n"
        result += f"  - Status: {engine_info.get('status', 'Unknown')}\n\n"
    
    return [
        types.TextContent(
            type="text",
            text=result
        )
    ]

async def main():
    """Main server function"""
    logger.info("🚀 Starting Maestro MCP Server (stdio) with existing sophisticated tools")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main()) 