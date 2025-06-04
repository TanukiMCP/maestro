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
_enhanced_tool_handlers_instance = None

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

def get_enhanced_tool_handlers_instance():
    global _enhanced_tool_handlers_instance
    if _enhanced_tool_handlers_instance is None:
        logger.info("Loading EnhancedToolHandlers instance")
        from maestro.enhanced_tools import EnhancedToolHandlers
        _enhanced_tool_handlers_instance = EnhancedToolHandlers()
    return _enhanced_tool_handlers_instance

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Handle tools/list requests - return ALL your existing sophisticated tools"""
    logger.info("Handling list_tools request")
    
    all_defined_tools = [
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
                    "wait_for": {
                        "type": "string",
                        "description": "Element to wait for before scraping"
                    },
                    "extract_links": {
                        "type": "boolean",
                        "description": "Extract links from the page",
                        "default": False
                    },
                    "extract_images": {
                        "type": "boolean",
                        "description": "Extract images from the page",
                        "default": False
                    }
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="maestro_execute",
            description="⚙️ Command execution with security controls",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30
                    },
                    "environment": {
                        "type": "object",
                        "description": "Environment variables",
                        "additionalProperties": True
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for execution"
                    },
                    "capture_output": {
                        "type": "boolean",
                        "description": "Capture stdout and stderr",
                        "default": True
                    },
                    "sandboxed": {
                        "type": "boolean",
                        "description": "Execute in a sandboxed environment",
                        "default": True
                    }
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="maestro_error_handler",
            description="🛡️ Advanced error handling and reporting",
            inputSchema={
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "description": "Type of error encountered"
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Detailed error message"
                    },
                    "context_data": {
                        "type": "object",
                        "description": "Contextual data at the time of error",
                        "additionalProperties": True
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Severity of the error",
                        "default": "medium"
                    }
                },
                "required": ["error_type", "error_message"]
            }
        ),
        types.Tool(
            name="maestro_temporal_context",
            description="⏳ Manages temporal context and event sequencing",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_name": {
                        "type": "string",
                        "description": "Name of the event"
                    },
                    "event_data": {
                        "type": "object",
                        "description": "Data associated with the event",
                        "additionalProperties": True
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Timestamp of the event (ISO 8601)"
                    },
                    "sequence_id": {
                        "type": "string",
                        "description": "Sequence identifier for related events"
                    }
                },
                "required": ["event_name"]
            }
        ),
        types.Tool(
            name="maestro_iae",
            description=(
                "🔬 Intelligence Amplification Engine Gateway - Provides access to all "
                "computational engines for precise numerical calculations. Use this tool "
                "when you need actual computations (not token predictions) for mathematical, "
                "scientific, or engineering problems. Supports quantum physics, statistical "
                "analysis, molecular modeling, and more through the MIA protocol."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "engine_domain": {
                        "type": "string",
                        "description": "Computational domain",
                        "enum": ["quantum_physics", "molecular_modeling", "statistical_analysis", 
                               "classical_mechanics", "relativity", "chemistry", "biology"],
                        "default": "quantum_physics"
                    },
                    "computation_type": {
                        "type": "string", 
                        "description": "Type of calculation to perform",
                        "enum": ["entanglement_entropy", "bell_violation", "quantum_fidelity", 
                               "pauli_decomposition", "molecular_properties", "statistical_test",
                               "regression_analysis", "sequence_alignment"],
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Computation-specific parameters",
                        "properties": {
                            "density_matrix": {
                                "type": "array",
                                "description": "Quantum state density matrix (for quantum calculations)",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object", 
                                        "properties": {
                                            "real": {"type": "number"},
                                            "imag": {"type": "number", "default": 0}
                                        },
                                        "required": ["real"]
                                    }
                                }
                            },
                            "quantum_state": {
                                "type": "array",
                                "description": "Quantum state vector (for Bell violation)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "real": {"type": "number"},
                                        "imag": {"type": "number", "default": 0}
                                    },
                                    "required": ["real"]
                                }
                            },
                            "measurement_angles": {
                                "type": "object",
                                "description": "Measurement angles for Bell test",
                                "properties": {
                                    "alice": {"type": "array", "items": {"type": "number"}},
                                    "bob": {"type": "array", "items": {"type": "number"}}
                                },
                                "required": ["alice", "bob"]
                            },
                            "operator": {
                                "type": "array",
                                "description": "Quantum operator matrix (for Pauli decomposition)",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "real": {"type": "number"},
                                            "imag": {"type": "number", "default": 0}
                                        },
                                        "required": ["real"]
                                    }
                                }
                            },
                            "state1": {
                                "type": "array",
                                "description": "First quantum state (for fidelity)",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "real": {"type": "number"},
                                            "imag": {"type": "number", "default": 0}
                                        },
                                        "required": ["real"]
                                    }
                                }
                            },
                            "state2": {
                                "type": "array", 
                                "description": "Second quantum state (for fidelity)",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "real": {"type": "number"},
                                            "imag": {"type": "number", "default": 0}
                                        },
                                        "required": ["real"]
                                    }
                                }
                            }
                        },
                        "additionalProperties": True
                    },
                    "precision_requirements": {
                        "type": "string",
                        "description": "Required precision level",
                        "enum": ["machine_precision", "extended_precision", "exact_symbolic"],
                        "default": "machine_precision"
                    },
                    "validation_level": {
                        "type": "string",
                        "description": "Input validation strictness",
                        "enum": ["basic", "standard", "strict"],
                        "default": "standard"
                    }
                },
                "required": ["engine_domain", "computation_type", "parameters"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_available_engines",
            description="🚀 Lists all available computational engines within the Maestro IAE framework",
            inputSchema={"type": "object", "properties": {}}, # No arguments needed
        )
    ]
    
    logger.info(f"Found {len(all_defined_tools)} tools.")
    return all_defined_tools

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
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [
            types.TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}"
            )
        ]

# Tool implementation functions (using your existing sophisticated implementations)

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
                
                @property
                def text(self):
                    return f"Synthesized analysis for: {task_description}. The task has been analyzed and appropriate computational engines have been identified for execution."
            
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
    
    # Use your existing _handle_iae_discovery method
    result = await maestro_tools._handle_iae_discovery(arguments)
    
    # Convert the result to TextContent format
    if isinstance(result, list) and len(result) > 0:
        return result  # Already in TextContent format
    else:
        return [
            types.TextContent(
                type="text",
                text=str(result)
            )
        ]

async def _handle_maestro_tool_selection(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool selection using your existing tools"""
    maestro_tools = get_maestro_tools_instance()
    
    # Use your existing _handle_tool_selection method
    result = await maestro_tools._handle_tool_selection(arguments)
    
    # Convert the result to TextContent format
    if isinstance(result, list) and len(result) > 0:
        return result  # Already in TextContent format
    else:
        return [
            types.TextContent(
                type="text",
                text=str(result)
            )
        ]

async def _handle_maestro_search(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_search using your existing EnhancedToolHandlers"""
    enhanced_tools = get_enhanced_tool_handlers_instance()
    
    # Use your existing search implementation
    result = await enhanced_tools.handle_maestro_search(arguments)
    
    return result

async def _handle_maestro_scrape(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_scrape using your existing EnhancedToolHandlers"""
    enhanced_tools = get_enhanced_tool_handlers_instance()
    
    # Use your existing scrape implementation
    result = await enhanced_tools.handle_maestro_scrape(arguments)
    
    return result

async def _handle_maestro_execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_execute using your existing EnhancedToolHandlers"""
    enhanced_tools = get_enhanced_tool_handlers_instance()
    
    # Use your existing execute implementation
    result = await enhanced_tools.handle_maestro_execute(arguments)
    
    return result

async def _handle_maestro_error_handler(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_error_handler using your existing EnhancedToolHandlers"""
    enhanced_tools = get_enhanced_tool_handlers_instance()
    
    # Use your existing error handler implementation
    result = await enhanced_tools.handle_maestro_error_handler(arguments)
    
    return result

async def _handle_maestro_temporal_context(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle maestro_temporal_context using your existing EnhancedToolHandlers"""
    enhanced_tools = get_enhanced_tool_handlers_instance()
    
    # Use your existing temporal context implementation
    result = await enhanced_tools.handle_maestro_temporal_context(arguments)
    
    return result

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
    logger.info("🚀 Starting Maestro MCP Server (stdio) with ALL existing sophisticated tools")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main()) 