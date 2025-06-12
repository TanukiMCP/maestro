"""
Static Tool Definitions for Fast Scanning

This module contains pre-defined metadata for all Maestro tools.
This allows Smithery to scan available tools without initializing
the full application or loading dependencies.
"""

# Static tool metadata dictionary
# This is used during tool scanning to provide fast access to tool information
MAESTRO_TOOLS_DICT = {
    "maestro_orchestrate": {
        "name": "maestro_orchestrate",
        "description": "Orchestrates complex multi-step tasks through intelligent task decomposition and execution tracking.",
        "category": "orchestration",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task to orchestrate"
                },
                "session_name": {
                    "type": "string",
                    "description": "Optional name for the orchestration session"
                }
            },
            "required": ["task_description"]
        }
    },
    "maestro_iae": {
        "name": "maestro_iae",
        "description": "Intelligence Amplification Engine for enhanced reasoning and problem-solving capabilities.",
        "category": "reasoning",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Current thought or reasoning step"
                },
                "thought_number": {
                    "type": "integer",
                    "description": "Sequential number of this thought"
                },
                "total_thoughts": {
                    "type": "integer",
                    "description": "Total expected thoughts in sequence"
                }
            },
            "required": ["thought", "thought_number", "total_thoughts"]
        }
    },
    "maestro_web": {
        "name": "maestro_web",
        "description": "Web search and information retrieval capabilities for enhanced context and knowledge.",
        "category": "knowledge",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to execute"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["query"]
        }
    },
    "maestro_execute": {
        "name": "maestro_execute",
        "description": "Executes actions and commands based on orchestrated task steps.",
        "category": "execution",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to execute"
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for the action"
                }
            },
            "required": ["action"]
        }
    },
    "maestro_error_handler": {
        "name": "maestro_error_handler",
        "description": "Intelligent error handling and recovery for robust task execution.",
        "category": "error_handling",
        "parameters": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "object",
                    "description": "Error details to handle"
                },
                "context": {
                    "type": "object",
                    "description": "Context of the error"
                }
            },
            "required": ["error"]
        }
    }
}

def get_tool_names():
    """Returns a list of all available tool names."""
    return list(MAESTRO_TOOLS_DICT.keys()) 