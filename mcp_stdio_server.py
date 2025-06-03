#!/usr/bin/env python3
"""
Maestro MCP Server - stdio transport implementation
Uses Model Context Protocol (MCP) FastMCP to expose tools via stdio.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_debug(msg, *args, **kwargs):
    """Helper to log debug messages with a prefix for easy filtering"""
    logger.debug(f"SMITHERY-DEBUG: {msg}", *args, **kwargs)

# Ensure src directory is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Create FastMCP server
mcp = FastMCP("Maestro")

# Lazy loading of handlers
_handlers = None

def get_handlers():
    global _handlers
    if _handlers is None:
        log_debug("Loading handlers")
        from main import (
            handle_maestro_orchestrate,
            handle_maestro_iae_discovery,
            handle_maestro_tool_selection,
            handle_maestro_iae,
            handle_maestro_search,
            handle_maestro_scrape,
            handle_maestro_execute,
            handle_maestro_error_handler,
            handle_maestro_temporal_context,
            get_computational_tools_instance
        )
        _handlers = {
            'orchestrate': handle_maestro_orchestrate,
            'iae_discovery': handle_maestro_iae_discovery,
            'tool_selection': handle_maestro_tool_selection,
            'iae': handle_maestro_iae,
            'search': handle_maestro_search,
            'scrape': handle_maestro_scrape,
            'execute': handle_maestro_execute,
            'error_handler': handle_maestro_error_handler,
            'temporal_context': handle_maestro_temporal_context,
            'get_computational_tools': get_computational_tools_instance
        }
    return _handlers

# Register tools with lazy loading
@mcp.tool(
    name="maestro_orchestrate",
    description="🎭 Intelligent workflow orchestration for complex tasks",
    annotations={
        "title": "Orchestrate Workflow",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def maestro_orchestrate(ctx: mcp.server.fastmcp.Context, task_description: str, context: dict = None, success_criteria: dict = None, complexity_level: str = "moderate"):
    """Wrapper for handle_maestro_orchestrate"""
    handlers = get_handlers()
    results = await handlers['orchestrate'](
        ctx=ctx,
        task_description=task_description,
        context=context,
        success_criteria=success_criteria,
        complexity_level=complexity_level
    )
    return [TextContent(type="text", text=results)]

@mcp.tool(
    name="maestro_iae_discovery",
    description="🔍 Integrated Analysis Engine discovery for optimal computation selection",
    annotations={
        "title": "IAE Discovery",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def maestro_iae_discovery(task_type: str = "general", domain_context: str = "", complexity_requirements: dict = None):
    """Wrapper for handle_maestro_iae_discovery"""
    handlers = get_handlers()
    results = await handlers['iae_discovery'](
        task_type=task_type,
        domain_context=domain_context,
        complexity_requirements=complexity_requirements
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="maestro_tool_selection",
    description="🧰 Intelligent tool selection based on task requirements",
    annotations={
        "title": "Tool Selection",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def maestro_tool_selection(request_description: str, available_context: dict = None, precision_requirements: dict = None):
    """Wrapper for handle_maestro_tool_selection"""
    handlers = get_handlers()
    results = await handlers['tool_selection'](
        request_description=request_description,
        available_context=available_context,
        precision_requirements=precision_requirements
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="maestro_iae",
    description="🧠 Intelligence Amplification Engine for specialized computational tasks",
    annotations={
        "title": "Intelligence Amplification Engine",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def maestro_iae(engine_domain: str, computation_type: str, parameters: dict, precision_requirements: str = "machine_precision", validation_level: str = "standard"):
    """Wrapper for handle_maestro_iae"""
    handlers = get_handlers()
    results = await handlers['iae'](
        engine_domain=engine_domain,
        computation_type=computation_type,
        parameters=parameters,
        precision_requirements=precision_requirements,
        validation_level=validation_level
    )
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="maestro_search",
    description="🔎 Enhanced search capabilities across multiple sources",
    annotations={
        "title": "Enhanced Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def maestro_search(arguments: dict):
    """Wrapper for handle_maestro_search"""
    handlers = get_handlers()
    results = await handlers['search'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="maestro_scrape",
    description="📑 Web scraping functionality with content extraction",
    annotations={
        "title": "Web Scraping",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def maestro_scrape(arguments: dict):
    """Wrapper for handle_maestro_scrape"""
    handlers = get_handlers()
    results = await handlers['scrape'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="maestro_execute",
    description="⚙️ Command execution with security controls",
    annotations={
        "title": "Command Execution",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def maestro_execute(arguments: dict):
    """Wrapper for handle_maestro_execute"""
    handlers = get_handlers()
    results = await handlers['execute'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="maestro_error_handler",
    description="🚨 Advanced error handling and recovery",
    annotations={
        "title": "Error Handler",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def maestro_error_handler(arguments: dict):
    """Wrapper for handle_maestro_error_handler"""
    handlers = get_handlers()
    results = await handlers['error_handler'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="maestro_temporal_context",
    description="🕐 Temporal context analysis for time-sensitive information",
    annotations={
        "title": "Temporal Context Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def maestro_temporal_context(arguments: dict):
    """Wrapper for handle_maestro_temporal_context"""
    handlers = get_handlers()
    results = await handlers['temporal_context'](arguments)
    return [TextContent(type="text", text=item["text"]) for item in results]

@mcp.tool(
    name="get_available_engines",
    description="📊 Get available computational engines and capabilities",
    annotations={
        "title": "Get Available Engines",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
def get_available_engines():
    """Expose computational tools from ComputationalTools"""
    handlers = get_handlers()
    comp_tools = handlers['get_computational_tools']()
    return comp_tools.get_available_engines()

# === RESOURCES ===
@mcp.resource("file://{path}", description="Expose file contents or directory listings.")
def resource_file(path: str) -> str:
    # Security: restrict to allowed roots, prevent traversal
    import os
    allowed_root = os.path.abspath(os.getcwd())
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(allowed_root):
        raise ValueError("Access denied.")
    if os.path.isdir(abs_path):
        return "\n".join(os.listdir(abs_path))
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

@mcp.resource("log://{name}", description="Access logs or execution traces by name.")
def resource_log(name: str) -> str:
    import os
    log_path = os.path.join("logs", name)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log {name} not found.")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

@mcp.resource("engine://{engine_name}", description="Get details about a computational engine.")
def resource_engine(engine_name: str) -> str:
    handlers = get_handlers()
    comp_tools = handlers['get_computational_tools']()
    return comp_tools.get_engine_schema(engine_name)

@mcp.resource("engines://list", description="List all available computational engines.")
def resource_engines_list() -> str:
    handlers = get_handlers()
    comp_tools = handlers['get_computational_tools']()
    return str(comp_tools.get_available_engines())

@mcp.resource("config://{section}", description="Expose server or environment configuration sections.")
def resource_config(section: str) -> str:
    import os
    if section == "env":
        return str(dict(os.environ))
    # Add more config sections as needed
    raise ValueError("Unknown config section.")

# === PROMPTS ===
@mcp.prompt(name="decompose_task", description="Break down a task into actionable steps.")
def prompt_decompose_task(task_description: str) -> str:
    return f"Break down the following task into actionable steps: {task_description}"

@mcp.prompt(name="suggest_tools", description="Suggest tools and order for a given task and context.")
def prompt_suggest_tools(task: str, context: dict) -> str:
    return f"Given the task '{task}', and context {context}, which tools should be used and in what order?"

@mcp.prompt(name="error_recovery", description="Suggest recovery actions for an error and context.")
def prompt_error_recovery(error_message: str, context: dict) -> str:
    return f"Given the error '{error_message}' and context {context}, suggest recovery actions or next steps."

@mcp.prompt(name="code_review", description="Review code according to criteria.")
def prompt_code_review(code: str, criteria: dict) -> str:
    return f"Review the following code according to {criteria}: {code}"

@mcp.prompt(name="summarize_search", description="Summarize search results.")
def prompt_summarize_search(search_results: list) -> str:
    return f"Summarize the following search results: {search_results}"

@mcp.prompt(name="temporal_reasoning", description="Analyze events in a time context.")
def prompt_temporal_reasoning(events: list, time_context: str) -> str:
    return f"Analyze these events {events} in the context of {time_context}."

@mcp.prompt(name="clarify_user_intent", description="Ask clarifying questions for ambiguous input.")
def prompt_clarify_user_intent(ambiguous_input: str) -> str:
    return f"The following input is ambiguous: '{ambiguous_input}'. Ask clarifying questions to the user."

if __name__ == "__main__":
    log_debug("Starting Maestro MCP Server")
    mcp.run() 