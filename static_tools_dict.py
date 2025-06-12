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
        "description": "Intelligent session management for complex multi-step tasks following MCP principles. Enables LLMs to self-direct orchestration through conceptual frameworks and knowledge management.",
        "category": "session_management", 
        "parameters": {
            "action": {
                "type": "string",
                "description": "The action to take. Original: 'create_session', 'declare_capabilities', 'add_task', 'execute_next', 'validate_task', 'mark_complete', 'get_status'. Self-directed: 'create_framework', 'update_workflow_state', 'add_knowledge', 'decompose_task', 'get_relevant_knowledge', 'get_task_hierarchy'. Capability Management: 'validate_capabilities', 'suggest_capabilities'",
                "required": True
            },
            "task_description": {
                "type": "string",
                "description": "Description of task to add (for 'add_task')",
                "required": False
            },
            "session_name": {
                "type": "string",
                "description": "Name for new session (for 'create_session')",
                "required": False
            },
            "validation_criteria": {
                "type": "array",
                "description": "List of criteria for task validation (for 'add_task')",
                "required": False
            },
            "evidence": {
                "type": "string",
                "description": "Evidence of task completion (for 'validate_task')",
                "required": False
            },
            "execution_evidence": {
                "type": "string",
                "description": "Evidence that execution was performed (for tracking completion)",
                "required": False
            },
            "builtin_tools": {
                "type": "array",
                "description": "List of built-in tools available (for 'declare_capabilities')",
                "required": False
            },
            "mcp_tools": {
                "type": "array",
                "description": "List of MCP server tools available (for 'declare_capabilities')",
                "required": False
            },
            "user_resources": {
                "type": "array",
                "description": "List of user-added resources available (for 'declare_capabilities')",
                "required": False
            },
            "next_action_needed": {
                "type": "boolean",
                "description": "Whether more actions are needed",
                "required": False,
                "default": True
            },
            "framework_type": {
                "type": "string",
                "description": "Type of conceptual framework: 'task_decomposition', 'dependency_graph', 'workflow_optimization', 'knowledge_synthesis', 'multi_agent_coordination', 'iterative_refinement', 'custom' (for 'create_framework')",
                "required": False
            },
            "framework_name": {
                "type": "string",
                "description": "Name for the conceptual framework (for 'create_framework')",
                "required": False
            },
            "framework_structure": {
                "type": "object",
                "description": "Dictionary defining the framework structure (for 'create_framework')",
                "required": False
            },
            "task_nodes": {
                "type": "array",
                "description": "List of task nodes for decomposition frameworks (for 'create_framework')",
                "required": False
            },
            "workflow_phase": {
                "type": "string",
                "description": "Current workflow phase: 'planning', 'decomposition', 'execution', 'validation', 'synthesis', 'reflection' (for 'update_workflow_state')",
                "required": False
            },
            "workflow_state_update": {
                "type": "object",
                "description": "Dictionary with workflow state updates (for 'update_workflow_state' and context)",
                "required": False
            },
            "knowledge_type": {
                "type": "string",
                "description": "Type of knowledge: 'tool_effectiveness', 'approach_outcome', 'error_pattern', 'optimization_insight', 'domain_specific', 'workflow_pattern' (for 'add_knowledge')",
                "required": False
            },
            "knowledge_subject": {
                "type": "string",
                "description": "What the knowledge is about (for 'add_knowledge' and 'get_relevant_knowledge')",
                "required": False
            },
            "knowledge_insights": {
                "type": "array",
                "description": "List of insights learned (for 'add_knowledge')",
                "required": False
            },
            "knowledge_confidence": {
                "type": "number",
                "description": "Confidence level 0.0-1.0 (for 'add_knowledge')",
                "required": False
            },
            "parent_task_id": {
                "type": "string",
                "description": "ID of parent task for decomposition (for 'decompose_task' and 'get_task_hierarchy')",
                "required": False
            },
            "subtasks": {
                "type": "array",
                "description": "List of subtask definitions (for 'decompose_task')",
                "required": False
            },
            "decomposition_strategy": {
                "type": "string",
                "description": "Strategy used for task decomposition (for 'decompose_task')",
                "required": False
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