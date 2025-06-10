#!/usr/bin/env python3
"""
Static Tools Dictionary for Maestro MCP Server
Provides a static definition of available tools for fast discovery and documentation.

This file is used by Smithery.ai for tool discovery and by the server for
instant tool listing without full initialization.
"""

MAESTRO_TOOLS_DICT = {
    "maestro_orchestrate": {
        "name": "maestro_orchestrate",
        "description": "Unified workflow orchestration and collaboration handler for the MAESTRO Protocol. Handles both task orchestration and user collaboration in a single robust tool with operation_mode parameter.",
        "category": "orchestration", 
        "parameters": {
            "task_description": {
                "type": "string",
                "description": "Description of the task to orchestrate (required for new workflows)",
                "required": False
            },
            "available_tools": {
                "type": "array",
                "description": "List of available tools for workflow execution",
                "required": False
            },
            "context_info": {
                "type": "object",
                "description": "Additional context and configuration for workflow execution",
                "required": False
            },
            "workflow_session_id": {
                "type": "string",
                "description": "Existing workflow session ID to continue or None for new workflow",
                "required": False
            },
            "user_response": {
                "type": "any",
                "description": "User response data for collaboration (when operation_mode is 'collaborate')",
                "required": False
            },
            "operation_mode": {
                "type": "string",
                "description": "Either 'orchestrate' (default) or 'collaborate'",
                "required": False,
                "default": "orchestrate"
            }
        }
    },
    "maestro_iae": {
        "name": "maestro_iae",
        "description": "Invokes a specific capability from an Intelligence Amplification Engine (IAE) using the MCP-native registry and meta-reasoning logic.",
        "category": "intelligence_amplification",
        "parameters": {
            "engine_name": {
                "type": "string",
                "description": "Name of the IAE engine to invoke",
                "required": True
            },
            "method_name": {
                "type": "string",
                "description": "Method to call on the engine",
                "required": True
            },
            "parameters": {
                "type": "object",
                "description": "Parameters to pass to the method",
                "required": True
            }
        }
    },
    "maestro_web": {
        "name": "maestro_web",
        "description": "Unified web tool for LLM-driven research. Supports only web search (no scraping).",
        "category": "research",
        "parameters": {
            "operation": {
                "type": "string",
                "description": "Operation to perform (only 'search' supported)",
                "required": True
            },
            "query_or_url": {
                "type": "string",
                "description": "Search query string",
                "required": True
            },
            "search_engine": {
                "type": "string",
                "description": "Search engine to use",
                "required": False,
                "default": "duckduckgo"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "required": False,
                "default": 5
            }
        }
    },
    "maestro_execute": {
        "name": "maestro_execute",
        "description": "Executes a block of code in a specified language within a secure sandbox.",
        "category": "execution",
        "parameters": {
            "code": {
                "type": "string",
                "description": "The source code to execute",
                "required": True
            },
            "language": {
                "type": "string",
                "description": "Programming language (e.g., 'python', 'javascript', 'bash')",
                "required": True
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "required": False,
                "default": 60
            }
        }
    },
    "maestro_error_handler": {
        "name": "maestro_error_handler",
        "description": "Analyzes an error and provides a structured response for recovery.",
        "category": "error_handling",
        "parameters": {
            "error_message": {
                "type": "string",
                "description": "The error message that occurred",
                "required": True
            },
            "context": {
                "type": "object",
                "description": "Context in which the error occurred",
                "required": True
            }
        }
    },

}

# Tool count for quick reference
TOOL_COUNT = len(MAESTRO_TOOLS_DICT)

# Categories for organization
TOOL_CATEGORIES = {
    "orchestration": ["maestro_orchestrate"],
    "intelligence_amplification": ["maestro_iae"],
    "research": ["maestro_web"],
    "execution": ["maestro_execute"],
    "error_handling": ["maestro_error_handler"],
    "collaboration": ["maestro_orchestrate"]
}

# Version and metadata
MAESTRO_VERSION = "1.0"
PROTOCOL_VERSION = "mcp-2024-11-05"
DISCOVERY_TIME_MS = 50  # Target discovery time in milliseconds

def get_tools_dict():
    """Return the static tools dictionary."""
    return MAESTRO_TOOLS_DICT

def get_tool_names():
    """Return list of tool names."""
    return list(MAESTRO_TOOLS_DICT.keys())

def get_tool_categories():
    """Return tool categories mapping."""
    return TOOL_CATEGORIES

if __name__ == "__main__":
    print(f"Maestro MCP Server - {TOOL_COUNT} tools available")
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Discovery time: <{DISCOVERY_TIME_MS}ms")
    print("\nAvailable tools:")
    for tool_name, tool_info in MAESTRO_TOOLS_DICT.items():
        print(f"  - {tool_name}: {tool_info['description']}") 