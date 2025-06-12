# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries
"""
This module defines the high-level tools exposed by the Maestro MCP server.
These tools are designed to be called by external agentic IDEs and orchestrators.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import anyio
import subprocess
import sys
import platform
import asyncio
import functools
import os
import datetime
import hashlib
from .config import MAESTROConfig
from fastmcp import Context
from dataclasses import asdict, is_dataclass

# Import for type annotation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import MAESTROConfig

logger = logging.getLogger(__name__)

# Add after imports
_orchestration_engine = None

def _create_temporal_awareness_context() -> Dict[str, Any]:
    """Create temporal awareness context with current date/time and research suggestions."""
    current_time = datetime.datetime.now()
    current_utc = datetime.datetime.utcnow()
    
    # Estimate when information might be outdated (basic heuristic)
    # For tech/current events: 1-3 months old suggests web research
    # For general knowledge: 6+ months suggests verification
    knowledge_cutoff_estimate = datetime.datetime(2024, 4, 1)  # Rough estimate for most LLMs
    months_since_cutoff = (current_time.year - knowledge_cutoff_estimate.year) * 12 + \
                         (current_time.month - knowledge_cutoff_estimate.month)
    
    suggests_web_research = months_since_cutoff > 3  # Suggest research if >3 months old
    
    return {
        "temporal_awareness": {
            "current_date": current_time.strftime("%Y-%m-%d"),
            "current_time": current_time.strftime("%H:%M:%S"),
            "current_datetime_iso": current_time.isoformat(),
            "current_utc_datetime_iso": current_utc.isoformat(),
            "current_timezone": str(current_time.astimezone().tzinfo),
            "current_day_of_week": current_time.strftime("%A"),
            "current_month": current_time.strftime("%B"),
            "current_year": current_time.year,
            "knowledge_cutoff_estimate": knowledge_cutoff_estimate.isoformat(),
            "months_since_knowledge_cutoff": months_since_cutoff,
            "suggests_web_research_for_current_info": suggests_web_research,
            "temporal_context_reasoning": (
                "Always consider whether information might be outdated. "
                f"Current date is {current_time.strftime('%Y-%m-%d')}. "
                f"If the task involves current events, recent developments, or time-sensitive information, "
                f"consider using web search tools to get up-to-date information."
            )
        }
    }

async def maestro_orchestrate(
    action: str,
    task_description: str = None,
    session_name: str = None,
    validation_criteria: list = None,
    evidence: str = None,
    execution_evidence: str = None,
    builtin_tools: list = None,
    mcp_tools: list = None,
    user_resources: list = None,
    next_action_needed: bool = True,
    # New parameters for LLM self-directed orchestration
    framework_type: str = None,
    framework_name: str = None,
    framework_structure: dict = None,
    task_nodes: list = None,
    workflow_phase: str = None,
    workflow_state_update: dict = None,
    knowledge_type: str = None,
    knowledge_subject: str = None,
    knowledge_insights: list = None,
    knowledge_confidence: float = None,
    parent_task_id: str = None,
    subtasks: list = None,
    decomposition_strategy: str = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Intelligent session management for complex multi-step tasks following MCP principles.
    Enables LLMs to self-direct orchestration through conceptual frameworks and knowledge management.
    
    Args:
        action: The action to take (see below for full list)
        task_description: Description of task to add (for 'add_task')
        session_name: Name for new session (for 'create_session')
        validation_criteria: List of criteria for task validation (for 'add_task')
        evidence: Evidence of task completion (for 'validate_task')
        execution_evidence: Evidence that execution was performed (for tracking completion)
        builtin_tools: List of built-in tools available (for 'declare_capabilities')
        mcp_tools: List of MCP server tools available (for 'declare_capabilities')
        user_resources: List of user-added resources available (for 'declare_capabilities')
        next_action_needed: Whether more actions are needed
        
        # Conceptual Framework Management
        framework_type: Type of framework ('task_decomposition', 'dependency_graph', etc.)
        framework_name: Name for the conceptual framework
        framework_structure: Dictionary defining the framework structure
        task_nodes: List of task nodes for decomposition frameworks
        
        # Workflow State Management
        workflow_phase: Current workflow phase ('planning', 'execution', etc.)
        workflow_state_update: Dictionary with workflow state updates
        
        # Knowledge Management
        knowledge_type: Type of knowledge ('tool_effectiveness', 'approach_outcome', etc.)
        knowledge_subject: What the knowledge is about
        knowledge_insights: List of insights learned
        knowledge_confidence: Confidence level (0.0-1.0)
        
        # Task Decomposition
        parent_task_id: ID of parent task for decomposition
        subtasks: List of subtask definitions
        decomposition_strategy: Strategy used for decomposition
        
        ctx: MCP context for logging
        
    Actions:
        Original actions:
        - create_session: Create new orchestration session
        - declare_capabilities: Declare available tools and resources
        - add_task: Add task to session
        - execute_next: Execute next pending task
        - validate_task: Validate task completion
        - mark_complete: Mark task as completed
        - get_status: Get session status
        
        New self-directed orchestration actions:
        - create_framework: Create conceptual framework for orchestration
        - update_workflow_state: Update current workflow state
        - add_knowledge: Add learned knowledge to session
        - decompose_task: Decompose task into subtasks
        - get_relevant_knowledge: Retrieve relevant session knowledge
        - get_task_hierarchy: Get hierarchical task structure
        
    Returns:
        Dictionary with current state, suggested actions, and orchestration context
    """
    # Import models after function definition to avoid circular imports
    from .session_models import (
        Session, Task, BuiltInTool, MCPTool, UserResource,
        ConceptualFramework, ConceptualFrameworkType, WorkflowState, 
        WorkflowPhase, SessionKnowledge, KnowledgeType, TaskNode
    )
    import os
    import json
    import uuid
    
    # Global session management
    session_state_file = "maestro/state/current_session.json"
    
    def _load_current_session():
        """Load the current session from disk"""
        os.makedirs(os.path.dirname(session_state_file), exist_ok=True)
        
        if os.path.exists(session_state_file):
            try:
                with open(session_state_file, 'r') as f:
                    session_data = json.load(f)
                return Session(**session_data)
            except Exception as e:
                if ctx:
                    ctx.warning(f"Could not load current session: {e}")
                return None
        return None

    def _save_current_session(session):
        """Save the current session to disk"""
        if session:
            os.makedirs(os.path.dirname(session_state_file), exist_ok=True)
            with open(session_state_file, 'w') as f:
                # Convert datetime objects to ISO format for JSON serialization
                session_dict = session.model_dump()
                json.dump(session_dict, f, indent=2, default=str)

    def _create_new_session(session_name: str = None):
        """Create a new session"""
        session = Session()
        if session_name:
            session.session_name = session_name
        
        session.environment_context = {
            "created_at": str(uuid.uuid4()),
            "capabilities_declared": False,
            "llm_environment": "agentic_coding_assistant"
        }
        
        _save_current_session(session)
        return session

    def _get_relevant_capabilities_for_task(session, task_description: str):
        """Get relevant tools and resources for the current task"""
        if not session or not session.capabilities:
            return {"builtin_tools": [], "mcp_tools": [], "resources": []}
        
        task_lower = task_description.lower()
        result = {"builtin_tools": [], "mcp_tools": [], "resources": []}
        
        # Check built-in tools
        for tool in session.capabilities.built_in_tools:
            for relevance in tool.relevant_for:
                if relevance.lower() in task_lower:
                    result["builtin_tools"].append({
                        "name": tool.name,
                        "description": tool.description,
                        "capabilities": tool.capabilities
                    })
                    break
        
        # Check MCP tools
        for tool in session.capabilities.mcp_tools:
            for relevance in tool.relevant_for:
                if relevance.lower() in task_lower:
                    result["mcp_tools"].append({
                        "name": tool.name,
                        "description": tool.description,
                        "server": tool.server_name,
                        "capabilities": tool.capabilities
                    })
                    break
        
        # Check user resources
        for resource in session.capabilities.user_resources:
            for relevance in resource.relevant_for:
                if relevance.lower() in task_lower:
                    result["resources"].append({
                        "name": resource.name,
                        "type": resource.type,
                        "description": resource.description,
                        "source": resource.source_url
                    })
                    break
        
        return result

    def _get_next_incomplete_task(session):
        """Get the next incomplete task"""
        if not session:
            return None
        
        for task in session.tasks:
            if task.status == "[ ]":
                return task
        return None

    def _suggest_next_actions(session):
        """Suggest what the LLM should do next"""
        if not session:
            return ["create_session"]
        
        # If no capabilities declared yet, suggest declaring them
        if not session.environment_context.get("capabilities_declared", False):
            return ["declare_capabilities"]
        
        # Check if all three categories are declared (mandatory)
        caps = session.capabilities
        if not caps.built_in_tools or not caps.mcp_tools or not caps.user_resources:
            return ["declare_capabilities"]
        
        # Check for active frameworks or knowledge
        if session.conceptual_frameworks:
            actions = ["execute_next", "update_workflow_state", "add_knowledge"]
        else:
            actions = ["create_framework", "add_task"]
        
        incomplete_task = _get_next_incomplete_task(session)
        if incomplete_task:
            if not incomplete_task.execution_started:
                actions.append("execute_next")
            elif incomplete_task.validation_required and not incomplete_task.evidence:
                actions.append("validate_task")
            elif incomplete_task.execution_evidence:
                actions.append("mark_complete")
            else:
                actions.extend(["provide_execution_evidence", "mark_complete"])
            
            # If task can be decomposed
            if not incomplete_task.subtask_ids:
                actions.append("decompose_task")
        else:
            actions.extend(["add_task", "get_status"])
        
        return list(set(actions))  # Remove duplicates

    def _suggest_capabilities_for_framework(framework_type: str, framework_structure: dict) -> dict:
        """Suggest required capabilities based on framework type and structure"""
        suggestions = {
            "builtin_tools": [],
            "mcp_tools": [],
            "resources": []
        }
        
        # Base suggestions by framework type
        if framework_type == "task_decomposition":
            suggestions["mcp_tools"].append("maestro_iae")
            suggestions["builtin_tools"].extend(["edit_file", "read_file"])
        elif framework_type == "workflow_optimization":
            suggestions["mcp_tools"].extend(["maestro_iae", "maestro_execute"])
            suggestions["builtin_tools"].append("run_terminal_cmd")
        elif framework_type == "knowledge_synthesis":
            suggestions["mcp_tools"].extend(["maestro_web", "maestro_iae"])
            suggestions["resources"].append("knowledge_base")
        elif framework_type == "multi_agent_coordination":
            suggestions["mcp_tools"].extend(["maestro_iae", "maestro_orchestrate"])
        
        # Analyze framework structure for additional requirements
        structure_str = str(framework_structure).lower()
        
        # Check for computational needs
        if any(keyword in structure_str for keyword in ["calculate", "compute", "analyze", "mathematical", "statistical"]):
            if "maestro_iae" not in suggestions["mcp_tools"]:
                suggestions["mcp_tools"].append("maestro_iae")
        
        # Check for web research needs
        if any(keyword in structure_str for keyword in ["research", "search", "web", "online", "current"]):
            if "maestro_web" not in suggestions["mcp_tools"]:
                suggestions["mcp_tools"].append("maestro_web")
        
        # Check for code execution needs
        if any(keyword in structure_str for keyword in ["execute", "run", "script", "code", "program"]):
            if "maestro_execute" not in suggestions["mcp_tools"]:
                suggestions["mcp_tools"].append("maestro_execute")
        
        # Check for file operations
        if any(keyword in structure_str for keyword in ["file", "create", "edit", "modify", "write"]):
            if "edit_file" not in suggestions["builtin_tools"]:
                suggestions["builtin_tools"].append("edit_file")
        
        return suggestions

    def _suggest_capabilities_for_task(task_description: str, capabilities) -> dict:
        """Suggest required capabilities based on task description"""
        suggestions = {
            "builtin_tools": [],
            "mcp_tools": [],
            "resources": []
        }
        
        task_lower = task_description.lower()
        
        # Mathematical/computational tasks
        if any(keyword in task_lower for keyword in ["calculate", "compute", "analyze", "mathematical", "statistical", 
                                                      "equation", "formula", "algorithm", "model", "simulation"]):
            suggestions["mcp_tools"].append("maestro_iae")
        
        # Web research tasks
        if any(keyword in task_lower for keyword in ["research", "search", "find", "web", "online", "current", 
                                                      "latest", "news", "information"]):
            suggestions["mcp_tools"].append("maestro_web")
        
        # Code execution tasks
        if any(keyword in task_lower for keyword in ["execute", "run", "script", "code", "program", "test", 
                                                      "python", "javascript", "bash"]):
            suggestions["mcp_tools"].append("maestro_execute")
        
        # Error handling tasks
        if any(keyword in task_lower for keyword in ["error", "debug", "fix", "troubleshoot", "issue", "problem"]):
            suggestions["mcp_tools"].append("maestro_error_handler")
        
        # File operations
        if any(keyword in task_lower for keyword in ["file", "create", "edit", "modify", "write", "read", "update"]):
            suggestions["builtin_tools"].extend(["edit_file", "read_file"])
        
        # Terminal operations
        if any(keyword in task_lower for keyword in ["terminal", "command", "shell", "system", "install", "setup"]):
            suggestions["builtin_tools"].append("run_terminal_cmd")
        
        # Search operations
        if any(keyword in task_lower for keyword in ["search", "find", "locate", "grep", "pattern"]):
            suggestions["builtin_tools"].extend(["grep_search", "codebase_search"])
        
        # Check against available capabilities and filter
        available_builtin = [t.name for t in capabilities.built_in_tools]
        available_mcp = [t.name for t in capabilities.mcp_tools]
        
        suggestions["builtin_tools"] = [t for t in suggestions["builtin_tools"] if t in available_builtin]
        suggestions["mcp_tools"] = [t for t in suggestions["mcp_tools"] if t in available_mcp]
        
        # Remove duplicates
        suggestions["builtin_tools"] = list(set(suggestions["builtin_tools"]))
        suggestions["mcp_tools"] = list(set(suggestions["mcp_tools"]))
        suggestions["resources"] = list(set(suggestions["resources"]))
        
        return suggestions

    # Load current session
    current_session = _load_current_session()
    
    if ctx:
        ctx.info(f"[MaestroOrchestrate] Action: {action}, Session: {current_session.id if current_session else 'none'}")
    
    result = {
        "action": action,
        "session_id": current_session.id if current_session else None,
        "current_task": None,
        "relevant_capabilities": {"builtin_tools": [], "mcp_tools": [], "resources": []},
        "all_capabilities": {"builtin_tools": [], "mcp_tools": [], "resources": []},
        "suggested_next_actions": [],
        "next_action_needed": next_action_needed,
        "completion_guidance": "",
        "status": "success"
    }
    
    try:
        if action == "create_session":
            current_session = _create_new_session(session_name)
            result["session_id"] = current_session.id
            result["session_name"] = getattr(current_session, 'session_name', None)
            result["suggested_next_actions"] = ["declare_capabilities"]
            result["completion_guidance"] = """
🚀 Session created! MANDATORY NEXT STEP: Use 'declare_capabilities' action with ALL THREE categories:

1. builtin_tools: Your core environment tools (edit_file, run_terminal_cmd, web_search, etc.)
2. mcp_tools: Available MCP server tools (maestro_orchestrate, codebase_search, etc.)
3. user_resources: Available docs, codebases, APIs, knowledge bases

This is REQUIRED for intelligent task execution and tool guidance.
"""
            
        elif action == "declare_capabilities":
            if not current_session:
                current_session = _create_new_session()
            
            # ALL THREE CATEGORIES ARE MANDATORY
            if not builtin_tools or not mcp_tools or not user_resources:
                result["error"] = "ALL THREE categories are required: builtin_tools, mcp_tools, AND user_resources"
                result["completion_guidance"] = """
MANDATORY: You must declare ALL THREE categories for effective task execution:

1. builtin_tools: Core tools available in your environment (edit_file, run_terminal_cmd, web_search, etc.)
2. mcp_tools: MCP server tools available (maestro_orchestrate, codebase_search, etc.)  
3. user_resources: Available documentation, codebases, APIs, knowledge bases

Example:
builtin_tools=["edit_file: Create and edit files", "run_terminal_cmd: Execute commands", "web_search: Search web"]
mcp_tools=["maestro_orchestrate: Session management", "codebase_search: Search codebase"]
user_resources=["documentation:React Docs: React documentation", "codebase:Current Project: Project files"]
"""
                return result
            
            # Process all three categories
            current_session.capabilities.built_in_tools = []
            current_session.capabilities.mcp_tools = []
            current_session.capabilities.user_resources = []
            
            # Handle built-in tools (REQUIRED)
            for tool_data in builtin_tools:
                if isinstance(tool_data, dict):
                    tool = BuiltInTool(**tool_data)
                else:
                    # Handle string format: "tool_name: description"
                    parts = str(tool_data).split(":", 1)
                    tool = BuiltInTool(
                        name=parts[0].strip(),
                        description=parts[1].strip() if len(parts) > 1 else "Built-in tool",
                        relevant_for=["file", "code", "terminal"] if any(keyword in parts[0].lower() for keyword in ["file", "edit", "terminal", "run"]) else ["general"]
                    )
                current_session.capabilities.built_in_tools.append(tool)
            
            # Handle MCP tools (REQUIRED)
            for tool_data in mcp_tools:
                if isinstance(tool_data, dict):
                    tool = MCPTool(**tool_data)
                else:
                    # Handle string format: "server_name.tool_name: description"
                    parts = str(tool_data).split(":", 1)
                    name_parts = parts[0].strip().split(".")
                    tool = MCPTool(
                        name=name_parts[-1] if len(name_parts) > 1 else parts[0].strip(),
                        server_name=name_parts[0] if len(name_parts) > 1 else "unknown",
                        description=parts[1].strip() if len(parts) > 1 else "MCP tool",
                        relevant_for=["integration", "external"] if "mcp" in parts[0].lower() else ["general"]
                    )
                current_session.capabilities.mcp_tools.append(tool)
            
            # Handle user resources (REQUIRED)
            for resource_data in user_resources:
                if isinstance(resource_data, dict):
                    resource = UserResource(**resource_data)
                else:
                    # Handle string format: "type:name: description"
                    parts = str(resource_data).split(":", 2)
                    resource = UserResource(
                        type=parts[0].strip() if len(parts) > 2 else "documentation",
                        name=parts[1].strip() if len(parts) > 2 else parts[0].strip(),
                        description=parts[2].strip() if len(parts) > 2 else (parts[1].strip() if len(parts) > 1 else "User resource"),
                        relevant_for=["documentation", "reference"] if "doc" in parts[0].lower() else ["knowledge"]
                    )
                current_session.capabilities.user_resources.append(resource)
            
            # Mark capabilities as fully declared
            current_session.environment_context["capabilities_declared"] = True
            _save_current_session(current_session)
            
            result["capabilities_declared"] = {
                "builtin_tools": len(current_session.capabilities.built_in_tools),
                "mcp_tools": len(current_session.capabilities.mcp_tools),
                "user_resources": len(current_session.capabilities.user_resources)
            }
            result["all_capabilities"] = {
                "builtin_tools": [{"name": t.name, "description": t.description} for t in current_session.capabilities.built_in_tools],
                "mcp_tools": [{"name": t.name, "server": t.server_name, "description": t.description} for t in current_session.capabilities.mcp_tools],
                "resources": [{"name": r.name, "type": r.type, "description": r.description} for r in current_session.capabilities.user_resources]
            }
            result["suggested_next_actions"] = ["add_task"]
            result["completion_guidance"] = f"✅ ALL capabilities registered! ({len(current_session.capabilities.built_in_tools)} built-in, {len(current_session.capabilities.mcp_tools)} MCP, {len(current_session.capabilities.user_resources)} resources) Now add tasks using 'add_task' action."
            
        elif action == "add_task":
            if not current_session:
                current_session = _create_new_session()
            
            # Check if all capabilities are declared first
            if not current_session.environment_context.get("capabilities_declared", False):
                result["error"] = "Must declare capabilities first using 'declare_capabilities' action"
                result["suggested_next_actions"] = ["declare_capabilities"]
                result["completion_guidance"] = "You must declare your available built-in tools, MCP tools, and resources before adding tasks."
                return result
            
            # Verify all three categories are declared
            caps = current_session.capabilities
            if not caps.built_in_tools or not caps.mcp_tools or not caps.user_resources:
                result["error"] = "Incomplete capability declaration. All three categories required."
                result["suggested_next_actions"] = ["declare_capabilities"]
                result["completion_guidance"] = "You must declare ALL THREE: builtin_tools, mcp_tools, AND user_resources."
                return result
            
            if not task_description:
                result["error"] = "task_description is required for add_task"
                return result
            
            task = Task(description=task_description)
            if validation_criteria:
                task.validation_required = True
                task.validation_criteria = validation_criteria
            
            # Get relevant capabilities for this task
            relevant_caps = _get_relevant_capabilities_for_task(current_session, task_description)
            task.suggested_builtin_tools = [t["name"] for t in relevant_caps["builtin_tools"]]
            task.suggested_mcp_tools = [t["name"] for t in relevant_caps["mcp_tools"]]
            task.suggested_resources = [r["name"] for r in relevant_caps["resources"]]
            
            current_session.tasks.append(task)
            _save_current_session(current_session)
            
            result["task_id"] = task.id
            result["task_added"] = task_description
            result["suggested_capabilities"] = {
                "builtin_tools": task.suggested_builtin_tools,
                "mcp_tools": task.suggested_mcp_tools,
                "resources": task.suggested_resources
            }
            result["relevant_capabilities"] = relevant_caps
            result["suggested_next_actions"] = ["execute_next", "add_task"]
            result["completion_guidance"] = f"Task added! Use 'execute_next' to get execution guidance for: {task_description}"
            
        elif action == "execute_next":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            # Ensure capabilities are fully declared before execution
            caps = current_session.capabilities
            if not caps.built_in_tools or not caps.mcp_tools or not caps.user_resources:
                result["error"] = "Cannot execute tasks without complete capability declaration"
                result["suggested_next_actions"] = ["declare_capabilities"]
                result["completion_guidance"] = "You must declare ALL capabilities (built-in tools, MCP tools, user resources) before executing tasks."
                return result
            
            next_task = _get_next_incomplete_task(current_session)
            if not next_task:
                result["message"] = "No incomplete tasks found"
                result["suggested_next_actions"] = ["add_task", "get_status"]
                result["completion_guidance"] = "All tasks complete! Add more tasks or check status."
            else:
                next_task.execution_started = True
                _save_current_session(current_session)
                
                relevant_caps = _get_relevant_capabilities_for_task(current_session, next_task.description)
                
                result["current_task"] = {
                    "id": next_task.id,
                    "description": next_task.description,
                    "validation_required": next_task.validation_required,
                    "validation_criteria": next_task.validation_criteria
                }
                result["relevant_capabilities"] = relevant_caps
                result["execution_guidance"] = f"EXECUTE NOW: {next_task.description}"
                result["available_tools_summary"] = {
                    "builtin": [t["name"] for t in relevant_caps["builtin_tools"]],
                    "mcp": [t["name"] for t in relevant_caps["mcp_tools"]],
                    "resources": [r["name"] for r in relevant_caps["resources"]]
                }
                result["suggested_next_actions"] = ["mark_complete"] if not next_task.validation_required else ["validate_task", "mark_complete"]
                result["completion_guidance"] = f"Execute the task using available tools and resources, then call 'mark_complete' with execution_evidence parameter."
            
        elif action == "validate_task":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            next_task = _get_next_incomplete_task(current_session)
            if not next_task:
                result["error"] = "No task to validate"
                return result
            
            if evidence:
                next_task.evidence.append({"evidence": evidence, "timestamp": str(uuid.uuid4())})
                _save_current_session(current_session)
                result["validation_added"] = evidence
            
            result["current_task"] = {
                "id": next_task.id,
                "description": next_task.description,
                "validation_criteria": next_task.validation_criteria,
                "evidence": next_task.evidence
            }
            result["suggested_next_actions"] = ["mark_complete"]
            result["completion_guidance"] = "Validation evidence added. Use 'mark_complete' to finish this task."
            
        elif action == "mark_complete":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            next_task = _get_next_incomplete_task(current_session)
            if not next_task:
                result["error"] = "No task to complete"
                return result
            
            # Add execution evidence if provided
            if execution_evidence:
                next_task.execution_evidence.append(execution_evidence)
            
            # Auto-detect completion or force complete
            should_complete = True
            if execution_evidence or next_task.execution_evidence:
                next_task.status = "[X]"
                _save_current_session(current_session)
                
                result["completed_task"] = next_task.description
                result["completion_evidence"] = next_task.execution_evidence
                result["suggested_next_actions"] = _suggest_next_actions(current_session)
                result["completion_guidance"] = "Task marked complete! " + ("Ready for next task." if _get_next_incomplete_task(current_session) else "All tasks finished!")
            else:
                result["error"] = "Task cannot be auto-completed. Provide execution_evidence parameter."
                result["completion_guidance"] = "To mark complete, provide execution_evidence parameter showing what you did."
            
        elif action == "get_status":
            if not current_session:
                result["message"] = "No active session"
                result["suggested_next_actions"] = ["create_session"]
                result["completion_guidance"] = "No session active. Use 'create_session' to start."
            else:
                result["session_id"] = current_session.id
                result["session_name"] = current_session.session_name
                result["total_tasks"] = len(current_session.tasks)
                result["completed_tasks"] = len([t for t in current_session.tasks if t.status == "[X]"])
                result["capabilities_declared"] = current_session.environment_context.get("capabilities_declared", False)
                
                current_task = _get_next_incomplete_task(current_session)
                if current_task:
                    result["current_task"] = {
                        "id": current_task.id,
                        "description": current_task.description,
                        "execution_started": current_task.execution_started,
                        "status": current_task.status
                    }
                
                result["all_tasks"] = [
                    {
                        "id": t.id,
                        "description": t.description,
                        "status": t.status,
                        "execution_started": t.execution_started,
                        "validation_required": t.validation_required
                    }
                    for t in current_session.tasks
                ]
                result["all_capabilities"] = {
                    "builtin_tools": [{"name": t.name, "description": t.description} for t in current_session.capabilities.built_in_tools],
                    "mcp_tools": [{"name": t.name, "server": t.server_name, "description": t.description} for t in current_session.capabilities.mcp_tools],
                    "resources": [{"name": r.name, "type": r.type, "description": r.description} for r in current_session.capabilities.user_resources]
                }
                result["suggested_next_actions"] = _suggest_next_actions(current_session)
                result["completion_guidance"] = "Session active. " + ("Continue with current task." if current_task else "All tasks complete!")
        
        # New LLM self-directed orchestration actions
        elif action == "create_framework":
            if not current_session:
                current_session = _create_new_session()
            
            # ENFORCE CAPABILITY DECLARATION
            if not current_session.environment_context.get("capabilities_declared", False):
                result["error"] = "Capabilities must be declared before creating frameworks"
                result["suggested_next_actions"] = ["declare_capabilities"]
                result["completion_guidance"] = """MANDATORY: You must declare ALL capabilities before self-directed orchestration:
1. builtin_tools: Core tools (edit_file, run_terminal_cmd, web_search, etc.)
2. mcp_tools: MCP server tools INCLUDING maestro_iae for computational tasks
3. user_resources: Available documentation, codebases, APIs

Example for computational workflows:
mcp_tools=["maestro_iae: Intelligence Amplification Engines", "maestro_execute: Code execution", "maestro_web: Web research"]"""
                return result
            
            # Verify all three categories are declared
            caps = current_session.capabilities
            if not caps.built_in_tools or not caps.mcp_tools or not caps.user_resources:
                result["error"] = "Incomplete capability declaration. All three categories required."
                result["suggested_next_actions"] = ["declare_capabilities"]
                result["completion_guidance"] = "You must declare ALL THREE: builtin_tools, mcp_tools (including maestro_iae), AND user_resources."
                return result
            
            # Check if IAE is declared for computational frameworks
            has_iae = any(tool.name == "maestro_iae" for tool in caps.mcp_tools)
            if framework_type in ["task_decomposition", "workflow_optimization"] and not has_iae:
                result["warning"] = "maestro_iae not declared but may be needed for computational tasks"
                result["suggestion"] = "Consider declaring maestro_iae for access to mathematical, scientific, and analytical engines"
            
            if not framework_type or not framework_name or not framework_structure:
                result["error"] = "framework_type, framework_name, and framework_structure are required"
                result["completion_guidance"] = "To create a conceptual framework, provide: framework_type (e.g., 'task_decomposition'), framework_name, and framework_structure (dict)"
                return result
            
            # Create the conceptual framework
            try:
                # Extract capability requirements from framework structure
                required_capabilities = framework_structure.get("required_capabilities", {})
                if not required_capabilities:
                    # Auto-suggest capabilities based on framework type
                    required_capabilities = _suggest_capabilities_for_framework(framework_type, framework_structure)
                
                framework = ConceptualFramework(
                    type=ConceptualFrameworkType(framework_type),
                    name=framework_name,
                    description=framework_structure.get("description", f"Framework for {framework_name}"),
                    structure=framework_structure,
                    task_nodes=[TaskNode(**node) for node in (task_nodes or [])],
                    relationships=framework_structure.get("relationships", []),
                    optimization_rules=framework_structure.get("optimization_rules", []),
                    metadata={
                        "required_capabilities": required_capabilities,
                        "declared_capabilities": {
                            "builtin_tools": [t.name for t in caps.built_in_tools],
                            "mcp_tools": [t.name for t in caps.mcp_tools],
                            "user_resources": [r.name for r in caps.user_resources]
                        }
                    }
                )
                
                current_session.add_framework(framework)
                _save_current_session(current_session)
                
                result["framework_id"] = framework.id
                result["framework_created"] = {
                    "type": framework.type.value,
                    "name": framework.name,
                    "task_nodes": len(framework.task_nodes),
                    "relationships": len(framework.relationships),
                    "required_capabilities": required_capabilities
                }
                result["capability_mapping"] = {
                    "suggested": required_capabilities,
                    "available": {
                        "builtin_tools": [t.name for t in caps.built_in_tools],
                        "mcp_tools": [t.name for t in caps.mcp_tools],
                        "iae_available": has_iae
                    }
                }
                result["suggested_next_actions"] = ["update_workflow_state", "add_task", "decompose_task"]
                result["completion_guidance"] = f"Framework '{framework_name}' created with capability mapping! Now update workflow state or add tasks to implement the framework."
                
            except Exception as e:
                result["error"] = f"Failed to create framework: {str(e)}"
                result["completion_guidance"] = "Check framework_type is valid and structure is properly formatted"
                
        elif action == "update_workflow_state":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            # ENFORCE CAPABILITY DECLARATION
            if not current_session.environment_context.get("capabilities_declared", False):
                result["error"] = "Capabilities must be declared before workflow state management"
                result["suggested_next_actions"] = ["declare_capabilities"]
                return result
            
            if not workflow_phase:
                result["error"] = "workflow_phase is required"
                result["completion_guidance"] = "Provide workflow_phase: 'planning', 'decomposition', 'execution', 'validation', 'synthesis', or 'reflection'"
                return result
            
            try:
                # Create new workflow state
                new_state = WorkflowState(
                    current_phase=WorkflowPhase(workflow_phase),
                    current_step=workflow_state_update.get("current_step") if workflow_state_update else None,
                    completed_steps=current_session.current_workflow_state.completed_steps if current_session.current_workflow_state else [],
                    active_frameworks=[f.id for f in current_session.conceptual_frameworks[-3:]] if current_session.conceptual_frameworks else [],
                    execution_context=workflow_state_update.get("execution_context", {}) if workflow_state_update else {},
                    decision_history=current_session.current_workflow_state.decision_history if current_session.current_workflow_state else [],
                    performance_metrics=workflow_state_update.get("metrics", {}) if workflow_state_update else {}
                )
                
                # Track capability usage in workflow state
                if workflow_state_update and workflow_state_update.get("tools_used"):
                    new_state.execution_context["tools_used"] = workflow_state_update["tools_used"]
                    # Validate tools are available
                    available_tools = [t.name for t in current_session.capabilities.built_in_tools] + \
                                    [t.name for t in current_session.capabilities.mcp_tools]
                    for tool in workflow_state_update["tools_used"]:
                        if tool not in available_tools:
                            result["warning"] = f"Tool '{tool}' used but not declared in capabilities"
                
                # Add completed step if provided
                if workflow_state_update and workflow_state_update.get("completed_step"):
                    new_state.completed_steps.append(workflow_state_update["completed_step"])
                
                # Add decision if provided
                if workflow_state_update and workflow_state_update.get("decision"):
                    new_state.decision_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "decision": workflow_state_update["decision"]
                    })
                
                current_session.update_workflow_state(new_state)
                _save_current_session(current_session)
                
                result["workflow_state"] = {
                    "phase": new_state.current_phase.value,
                    "current_step": new_state.current_step,
                    "completed_steps": len(new_state.completed_steps),
                    "active_frameworks": len(new_state.active_frameworks)
                }
                result["suggested_next_actions"] = ["add_knowledge", "execute_next", "add_task"]
                result["completion_guidance"] = f"Workflow state updated to '{workflow_phase}' phase. Continue with task execution or knowledge management."
                
            except Exception as e:
                result["error"] = f"Failed to update workflow state: {str(e)}"
                
        elif action == "add_knowledge":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            # ENFORCE CAPABILITY DECLARATION
            if not current_session.environment_context.get("capabilities_declared", False):
                result["error"] = "Capabilities must be declared before knowledge management"
                result["suggested_next_actions"] = ["declare_capabilities"]
                return result
            
            if not knowledge_type or not knowledge_subject or not knowledge_insights:
                result["error"] = "knowledge_type, knowledge_subject, and knowledge_insights are required"
                result["completion_guidance"] = "Provide: knowledge_type ('tool_effectiveness', 'approach_outcome', etc.), knowledge_subject, and knowledge_insights (list)"
                return result
            
            try:
                # Track which tools/IAEs the knowledge relates to
                related_tools = []
                if knowledge_type == "tool_effectiveness":
                    # Extract tool names from subject
                    for tool in current_session.capabilities.mcp_tools:
                        if tool.name in knowledge_subject.lower():
                            related_tools.append(tool.name)
                
                knowledge = SessionKnowledge(
                    type=KnowledgeType(knowledge_type),
                    subject=knowledge_subject,
                    context=workflow_state_update or {},
                    insights=knowledge_insights,
                    confidence=knowledge_confidence or 0.8,
                    applicable_scenarios=workflow_state_update.get("applicable_scenarios", []) if workflow_state_update else []
                )
                
                # Add tool relationships to metadata
                if related_tools:
                    knowledge.context["related_tools"] = related_tools
                
                current_session.add_knowledge(knowledge)
                _save_current_session(current_session)
                
                result["knowledge_added"] = {
                    "type": knowledge.type.value,
                    "subject": knowledge.subject,
                    "insights_count": len(knowledge.insights),
                    "confidence": knowledge.confidence,
                    "related_tools": related_tools
                }
                result["suggested_next_actions"] = ["get_relevant_knowledge", "execute_next", "update_workflow_state"]
                result["completion_guidance"] = f"Knowledge about '{knowledge_subject}' added to session. Use it to inform future decisions."
                
            except Exception as e:
                result["error"] = f"Failed to add knowledge: {str(e)}"
                
        elif action == "decompose_task":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            # ENFORCE CAPABILITY DECLARATION
            if not current_session.environment_context.get("capabilities_declared", False):
                result["error"] = "Capabilities must be declared before task decomposition"
                result["suggested_next_actions"] = ["declare_capabilities"]
                return result
            
            if not parent_task_id or not subtasks:
                result["error"] = "parent_task_id and subtasks are required"
                result["completion_guidance"] = "Provide parent_task_id and subtasks list with task definitions"
                return result
            
            # Find parent task
            parent_task = next((t for t in current_session.tasks if t.id == parent_task_id), None)
            if not parent_task:
                result["error"] = f"Parent task {parent_task_id} not found"
                return result
            
            # Create subtasks with capability mapping
            created_subtasks = []
            capability_warnings = []
            
            for subtask_def in subtasks:
                # Suggest required capabilities for subtask
                required_caps = subtask_def.get("required_capabilities", [])
                if not required_caps:
                    # Auto-suggest based on task description
                    required_caps = _suggest_capabilities_for_task(
                        subtask_def.get("description", ""),
                        current_session.capabilities
                    )
                
                subtask = Task(
                    description=subtask_def.get("description", ""),
                    parent_task_id=parent_task_id,
                    complexity_score=subtask_def.get("complexity", 0.5),
                    approach_strategy=subtask_def.get("strategy"),
                    workflow_phase=WorkflowPhase(subtask_def["phase"]) if "phase" in subtask_def else None
                )
                
                # Store required capabilities in task
                subtask.suggested_builtin_tools = required_caps.get("builtin_tools", [])
                subtask.suggested_mcp_tools = required_caps.get("mcp_tools", [])
                subtask.suggested_resources = required_caps.get("resources", [])
                
                # Check if required capabilities are available
                available_tools = [t.name for t in current_session.capabilities.built_in_tools] + \
                                [t.name for t in current_session.capabilities.mcp_tools]
                for tool in subtask.suggested_mcp_tools:
                    if tool not in available_tools:
                        capability_warnings.append(f"Subtask '{subtask.description}' requires '{tool}' which is not declared")
                
                if subtask_def.get("validation_criteria"):
                    subtask.validation_required = True
                    subtask.validation_criteria = subtask_def["validation_criteria"]
                
                current_session.tasks.append(subtask)
                parent_task.subtask_ids.append(subtask.id)
                created_subtasks.append(subtask)
            
            # Update parent task strategy if provided
            if decomposition_strategy:
                parent_task.approach_strategy = {"decomposition": decomposition_strategy}
            
            _save_current_session(current_session)
            
            result["decomposition_complete"] = {
                "parent_task": parent_task.description,
                "subtasks_created": len(created_subtasks),
                "subtask_ids": [t.id for t in created_subtasks],
                "capability_mapping": {
                    subtask.id: {
                        "description": subtask.description,
                        "required_tools": subtask.suggested_mcp_tools,
                        "required_builtin": subtask.suggested_builtin_tools
                    }
                    for subtask in created_subtasks
                }
            }
            
            if capability_warnings:
                result["capability_warnings"] = capability_warnings
                result["suggestion"] = "Consider declaring missing tools or adjusting subtask requirements"
            
            result["suggested_next_actions"] = ["execute_next", "get_task_hierarchy"]
            result["completion_guidance"] = f"Task decomposed into {len(created_subtasks)} subtasks with capability mapping. Execute them to complete the parent task."
            
        elif action == "get_relevant_knowledge":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            context = knowledge_subject or task_description or ""
            if not context:
                result["error"] = "Provide knowledge_subject or task_description for context"
                return result
            
            relevant_knowledge = current_session.get_relevant_knowledge(
                context=context,
                knowledge_type=KnowledgeType(knowledge_type) if knowledge_type else None
            )
            
            result["relevant_knowledge"] = [
                {
                    "type": k.type.value,
                    "subject": k.subject,
                    "insights": k.insights,
                    "confidence": k.confidence,
                    "usage_count": k.usage_count
                }
                for k in relevant_knowledge
            ]
            result["knowledge_count"] = len(relevant_knowledge)
            result["suggested_next_actions"] = ["execute_next", "add_knowledge"]
            result["completion_guidance"] = f"Found {len(relevant_knowledge)} relevant knowledge items. Use them to inform your approach."
            
        elif action == "get_task_hierarchy":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            task_id = parent_task_id or (current_session.tasks[0].id if current_session.tasks else None)
            if not task_id:
                result["error"] = "No tasks available or parent_task_id not provided"
                return result
            
            hierarchy = current_session.get_task_hierarchy(task_id)
            
            def format_hierarchy(h, level=0):
                """Format hierarchy for display"""
                if not h:
                    return []
                
                lines = []
                task = h["task"]
                indent = "  " * level
                lines.append(f"{indent}{task.status} {task.description} (ID: {task.id})")
                
                for subtask_h in h.get("subtasks", []):
                    lines.extend(format_hierarchy(subtask_h, level + 1))
                
                return lines
            
            result["task_hierarchy"] = hierarchy
            result["hierarchy_display"] = "\n".join(format_hierarchy(hierarchy))
            result["suggested_next_actions"] = ["execute_next", "decompose_task"]
            result["completion_guidance"] = "Task hierarchy retrieved. Use it to understand task structure and dependencies."
        
        elif action == "validate_capabilities":
            if not current_session:
                result["error"] = "No active session"
                return result
            
            # ENFORCE CAPABILITY DECLARATION
            if not current_session.environment_context.get("capabilities_declared", False):
                result["error"] = "Capabilities must be declared before validation"
                result["suggested_next_actions"] = ["declare_capabilities"]
                return result
            
            # Perform comprehensive validation
            validation_result = current_session.validate_all_capabilities()
            
            result["validation_summary"] = validation_result
            result["session_valid"] = validation_result["session_valid"]
            
            if not validation_result["session_valid"]:
                result["missing_capabilities"] = validation_result["missing_capabilities"]
                result["warnings"] = validation_result["warnings"]
                
                # Provide specific guidance
                guidance = []
                if validation_result["missing_capabilities"]["mcp_tools"]:
                    missing_mcp = validation_result["missing_capabilities"]["mcp_tools"]
                    if "maestro_iae" in missing_mcp:
                        guidance.append(
                            "CRITICAL: Declare maestro_iae to access Intelligence Amplification Engines for:\n"
                            "  - Mathematical computations (solve equations, calculus, statistics)\n"
                            "  - Data analysis (statistical analysis, visualization)\n"
                            "  - Scientific computing (physics, chemistry, biology)\n"
                            "  - Code quality analysis and optimization"
                        )
                    if "maestro_web" in missing_mcp:
                        guidance.append("Declare maestro_web for web research and information gathering")
                    if "maestro_execute" in missing_mcp:
                        guidance.append("Declare maestro_execute for code execution in Python/JavaScript/Bash")
                
                result["remediation_guidance"] = guidance
                result["suggested_next_actions"] = ["declare_capabilities"]
                result["completion_guidance"] = "Validation failed. Declare missing capabilities before proceeding."
            else:
                result["suggested_next_actions"] = ["execute_next", "create_framework"]
                result["completion_guidance"] = "All capability requirements validated! Ready to proceed with workflow execution."
            
            # Generate usage report
            usage_report = current_session.get_capability_usage_report()
            result["capability_usage"] = usage_report
            
        elif action == "suggest_capabilities":
            if not current_session:
                current_session = _create_new_session()
            
            # Analyze what needs capability suggestions
            context = task_description or framework_name or workflow_phase or ""
            if not context:
                result["error"] = "Provide task_description, framework_name, or workflow_phase for suggestions"
                return result
            
            suggestions = {
                "recommended_tools": {
                    "builtin_tools": [],
                    "mcp_tools": [],
                    "resources": []
                },
                "rationale": {},
                "examples": {}
            }
            
            context_lower = context.lower()
            
            # Analyze context and provide intelligent suggestions
            
            # Mathematical/Computational tasks
            if any(keyword in context_lower for keyword in ["math", "calculate", "compute", "equation", "statistical", 
                                                            "analyze", "model", "simulation", "algorithm", "optimize"]):
                suggestions["recommended_tools"]["mcp_tools"].append("maestro_iae")
                suggestions["rationale"]["maestro_iae"] = (
                    "Provides access to 25+ Intelligence Amplification Engines including:\n"
                    "  - Mathematics engine for equations, calculus, linear algebra\n"
                    "  - Data analysis engine for statistics and visualization\n"
                    "  - Scientific computing for modeling and simulations\n"
                    "  - Optimization algorithms for performance tuning"
                )
                suggestions["examples"]["maestro_iae"] = [
                    'await maestro_iae(engine_name="mathematics", method_name="solve_equation", parameters={"equation": "x^2 + 5x + 6 = 0"})',
                    'await maestro_iae(engine_name="data_analysis", method_name="statistical_summary", parameters={"data": [1,2,3,4,5]})'
                ]
            
            # Web Research tasks
            if any(keyword in context_lower for keyword in ["research", "search", "web", "online", "current", 
                                                            "latest", "find information", "gather data"]):
                suggestions["recommended_tools"]["mcp_tools"].append("maestro_web")
                suggestions["rationale"]["maestro_web"] = (
                    "Enables real-time web research and information gathering:\n"
                    "  - Search current information from the web\n"
                    "  - Gather up-to-date data and news\n"
                    "  - Research topics requiring recent information"
                )
                suggestions["examples"]["maestro_web"] = [
                    'await maestro_web(operation="search", query_or_url="latest AI developments 2024", num_results=10)'
                ]
            
            # Code Execution tasks
            if any(keyword in context_lower for keyword in ["execute", "run", "code", "script", "program", 
                                                            "python", "javascript", "bash", "test"]):
                suggestions["recommended_tools"]["mcp_tools"].append("maestro_execute")
                suggestions["rationale"]["maestro_execute"] = (
                    "Secure code execution in multiple languages:\n"
                    "  - Python for data analysis and scripting\n"
                    "  - JavaScript for web-related tasks\n"
                    "  - Bash for system operations"
                )
                suggestions["examples"]["maestro_execute"] = [
                    'await maestro_execute(code="print(\'Hello World\')", language="python", timeout=30)'
                ]
            
            # Error Handling tasks
            if any(keyword in context_lower for keyword in ["error", "debug", "fix", "troubleshoot", "issue"]):
                suggestions["recommended_tools"]["mcp_tools"].append("maestro_error_handler")
                suggestions["rationale"]["maestro_error_handler"] = (
                    "Intelligent error analysis and recovery:\n"
                    "  - Analyze error patterns and root causes\n"
                    "  - Suggest recovery strategies\n"
                    "  - Provide debugging guidance"
                )
            
            # File Operations
            if any(keyword in context_lower for keyword in ["file", "create", "edit", "modify", "write", "read"]):
                suggestions["recommended_tools"]["builtin_tools"].extend(["edit_file", "read_file"])
                suggestions["rationale"]["file_tools"] = "File creation, editing, and reading operations"
            
            # Code Search
            if any(keyword in context_lower for keyword in ["search code", "find", "grep", "codebase"]):
                suggestions["recommended_tools"]["builtin_tools"].extend(["grep_search", "codebase_search"])
                suggestions["rationale"]["search_tools"] = "Search through codebases and find patterns"
            
            # Terminal Operations
            if any(keyword in context_lower for keyword in ["terminal", "command", "shell", "install", "system"]):
                suggestions["recommended_tools"]["builtin_tools"].append("run_terminal_cmd")
                suggestions["rationale"]["terminal"] = "Execute system commands and shell scripts"
            
            # Framework-specific suggestions
            if framework_type:
                framework_suggestions = _suggest_capabilities_for_framework(framework_type, framework_structure or {})
                for tool_type in ["builtin_tools", "mcp_tools", "resources"]:
                    for tool in framework_suggestions.get(tool_type, []):
                        if tool not in suggestions["recommended_tools"][tool_type]:
                            suggestions["recommended_tools"][tool_type].append(tool)
            
            # Validation criteria mapping
            if validation_criteria:
                for criterion in validation_criteria:
                    criterion_lower = criterion.lower()
                    if any(keyword in criterion_lower for keyword in ["compute", "calculate", "analyze"]):
                        if "maestro_iae" not in suggestions["recommended_tools"]["mcp_tools"]:
                            suggestions["recommended_tools"]["mcp_tools"].append("maestro_iae")
                            suggestions["rationale"]["validation_iae"] = "IAE needed for computational validation"
                    if any(keyword in criterion_lower for keyword in ["test", "verify", "check"]):
                        if "maestro_execute" not in suggestions["recommended_tools"]["mcp_tools"]:
                            suggestions["recommended_tools"]["mcp_tools"].append("maestro_execute")
                            suggestions["rationale"]["validation_execute"] = "Code execution needed for testing/verification"
            
            result["capability_suggestions"] = suggestions
            result["declaration_template"] = {
                "builtin_tools": suggestions["recommended_tools"]["builtin_tools"],
                "mcp_tools": [f"{tool}: {suggestions['rationale'].get(tool, 'MCP tool')}" 
                             for tool in suggestions["recommended_tools"]["mcp_tools"]],
                "user_resources": suggestions["recommended_tools"]["resources"]
            }
            result["suggested_next_actions"] = ["declare_capabilities"]
            result["completion_guidance"] = (
                "Capability suggestions generated based on your workflow needs. "
                "Use the declaration_template to declare these capabilities."
            )
        
        else:
            result["error"] = f"Unknown action: {action}"
            result["completion_guidance"] = """Valid actions: 
Original: create_session, declare_capabilities, add_task, execute_next, validate_task, mark_complete, get_status
Self-directed: create_framework, update_workflow_state, add_knowledge, decompose_task, get_relevant_knowledge, get_task_hierarchy
Capability Management: validate_capabilities, suggest_capabilities"""
        
        return result
        
    except Exception as e:
        logger.error(f"Error in maestro_orchestrate: {e}", exc_info=True)
        return {
            "status": "error",
            "action": action,
            "error": str(e),
            "completion_guidance": "An error occurred. Please try again or contact support."
        }


async def _handle_collaboration_response_deprecated(
    engine, user_response: Any, workflow_session_id: Optional[str], 
    context_info: Optional[Dict[str, Any]], ctx: Context
) -> Dict[str, Any]:
    """Handle user collaboration responses with robust session management."""
    
    if not user_response:
        return {
            "status": "error",
            "operation_mode": "collaborate",
            "error": "user_response is required for collaboration mode",
            "message": "No user response provided for collaboration",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    # Handle workflow session collaboration
    if workflow_session_id:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Processing collaboration for session: {workflow_session_id}")
        
        try:
            session = engine.session_manager.get_session(workflow_session_id)
            
            if not session:
                if ctx:
                    ctx.error(f"[MaestroOrchestrate] Session {workflow_session_id} not found or expired")
                return {
                    "status": "session_expired",
                    "operation_mode": "collaborate",
                    "session_id": workflow_session_id,
                    "error": f"Session {workflow_session_id} not found or expired",
                    "message": "The workflow session has expired or is invalid. Please start a new workflow.",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Update session context with user response
            _merge_user_response_to_session(session, user_response, context_info)
            
            # Update session in manager
            engine.session_manager.update_session(session)
            
            if ctx:
                ctx.info(f"[MaestroOrchestrate] Session context updated with user response")
            
            # Continue workflow execution with updated context
            try:
                step_result = await engine.execute_workflow_step(workflow_session_id)
                
                if ctx:
                    ctx.info(f"[MaestroOrchestrate] Workflow step executed successfully after collaboration")
                
                return {
                    "status": "workflow_continued",
                    "operation_mode": "collaborate",
                    "session_id": workflow_session_id,
                    "step_result": step_result,
                    "user_response_processed": True,
                    "message": "User response processed and workflow continued successfully",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
            except Exception as step_error:
                if ctx:
                    ctx.error(f"[MaestroOrchestrate] Error executing workflow step: {step_error}")
                
                return {
                    "status": "execution_error",
                    "operation_mode": "collaborate",
                    "session_id": workflow_session_id,
                    "error": str(step_error),
                    "user_response_processed": True,
                    "message": "User response was processed but workflow execution failed",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            if ctx:
                ctx.error(f"[MaestroOrchestrate] Error updating session: {e}")
            
            return {
                "status": "collaboration_error",
                "operation_mode": "collaborate",
                "session_id": workflow_session_id,
                "error": str(e),
                "message": "Failed to process collaboration response",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    # Handle paused workflow creation collaboration
    creation_id = _extract_creation_id_from_context(context_info)
    if creation_id:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Resuming workflow creation: {creation_id}")
        
        try:
            # Prepare context response
            context_response = _prepare_context_response(user_response, context_info)
            
            # Resume workflow creation with user context
            result = await engine.resume_workflow_creation(creation_id, context_response)
            
            if ctx:
                ctx.info(f"[MaestroOrchestrate] Workflow creation resumed successfully")
            
            # Handle different result types
            if isinstance(result, ContextSurvey):
                return {
                    "status": "context_still_required",
                    "operation_mode": "collaborate", 
                    "creation_id": creation_id,
                    "survey": result,
                    "message": "Additional context is still required to continue workflow creation",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            elif isinstance(result, StepExecutionResult):
                return {
                    "status": "workflow_created_and_started",
                    "operation_mode": "collaborate",
                    "session_id": result.workflow_session_id,
                    "step_result": result,
                    "message": "Workflow created successfully and first step executed",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                return {
                    "status": "workflow_created",
                    "operation_mode": "collaborate",
                    "result": result,
                    "message": "Workflow creation completed successfully",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
        except Exception as resume_error:
            if ctx:
                ctx.error(f"[MaestroOrchestrate] Error resuming workflow creation: {resume_error}")
            
            return {
                "status": "creation_error",
                "operation_mode": "collaborate",
                "creation_id": creation_id,
                "error": str(resume_error),
                "message": "Failed to resume workflow creation with provided context",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    # Fallback for non-workflow collaborations
    return {
        "status": "processed",
        "operation_mode": "collaborate",
        "user_response": user_response,
        "message": "User response has been successfully processed and can be used in the next workflow step.",
        "timestamp": datetime.datetime.now().isoformat()
    }


async def _handle_orchestration(
    engine, task_description: str, available_tools: List[Dict[str, Any]], 
    context_info: Optional[Dict[str, Any]], workflow_session_id: Optional[str], ctx: Context
) -> Dict[str, Any]:
    """Handle workflow orchestration with robust session and context management."""
    
    # Map provided context to expected keys for backward compatibility
    if context_info:
        context_info = _map_context_info(context_info)
    
    # Handle progressive workflow execution (continuing existing workflow)
    if workflow_session_id:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Attempting to execute next step in session {workflow_session_id}")
        
        try:
            orchestration_result = await engine.execute_workflow_step(workflow_session_id)
            
            # Convert result to consistent dictionary format
            return _convert_result_to_dict(orchestration_result)
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "session not found" in error_msg or "expired" in error_msg:
                # Session expired or not found, start new workflow with context
                if ctx:
                    ctx.warning(f"[MaestroOrchestrate] Session {workflow_session_id} expired or not found, starting new workflow")
                
                if not task_description or not available_tools:
                    return {
                        "status": "session_expired_insufficient_context",
                        "operation_mode": "orchestrate",
                        "session_id": workflow_session_id,
                        "error": "task_description and available_tools required for new workflow after session expiry",
                        "message": "Session expired and insufficient context to restart workflow",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                
                # Start new workflow with available context
                return await _start_new_workflow(engine, task_description, available_tools, context_info, ctx)
            else:
                raise
    
    # Start new progressive workflow
    if not task_description:
        return {
            "status": "error",
            "operation_mode": "orchestrate",
            "error": "task_description is required for new workflow orchestration",
            "message": "Cannot start new workflow without task description",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    return await _start_new_workflow(engine, task_description, available_tools, context_info, ctx)


async def _start_new_workflow(
    engine, task_description: str, available_tools: List[Dict[str, Any]], 
    context_info: Optional[Dict[str, Any]], ctx: Context
) -> Dict[str, Any]:
    """Start a new workflow with proper context and tool handling."""
    
    # Check if this is a continuation of a paused workflow creation
    creation_id = f"creation_{hashlib.sha256(task_description.encode()).hexdigest()[:10]}"
    paused_creation = engine.session_manager.get_paused_creation(creation_id)

    if paused_creation and context_info:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Resuming paused workflow creation: {creation_id}")
        
        orchestration_context = context_info.copy()
        if available_tools:
            orchestration_context["available_tools"] = available_tools
        
        result = await engine.resume_workflow_creation(
            creation_id=creation_id,
            context_response=orchestration_context
        )
        
        # Convert result to consistent dictionary format
        return _convert_result_to_dict(result)
    else:
        # Validate required parameters for new workflow
        if not available_tools:
            return {
                "status": "error",
                "operation_mode": "orchestrate",
                "error": "available_tools are required for new workflow orchestration",
                "message": "Cannot start new workflow without available tools",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Merge available_tools into context_info for tool mapping
        orchestration_context = context_info.copy() if context_info else {}
        orchestration_context["available_tools"] = available_tools
        
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Creating new progressive workflow with {len(available_tools)} available tools")
        
        # Run progressive orchestration (creates workflow and executes step 1)
        result = await engine.orchestrate_progressive_workflow(
            task_description=task_description,
            provided_context=orchestration_context
        )
        
        # Convert result to consistent dictionary format
        return _convert_result_to_dict(result)


def _merge_user_response_to_session(session, user_response: Any, context_info: Optional[Dict[str, Any]]):
    """Merge user response into session context data."""
    if isinstance(user_response, dict):
        # If user_response is a dictionary, merge it into context_data
        session.context_data.update(user_response)
    else:
        # If user_response is a string or other type, store it as 'user_context'
        session.context_data['user_context'] = user_response
    
    # Merge additional context if provided
    if context_info:
        session.context_data.update(context_info)
    
    # Store collaboration metadata for reference
    session.context_data['last_collaboration'] = {
        'user_response': user_response,
        'additional_context': context_info,
        'timestamp': datetime.datetime.now().isoformat()
    }


def _extract_creation_id_from_context(context_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract creation ID from context info or survey data."""
    if not context_info:
        return None
        
    # Check direct creation_id
    creation_id = context_info.get('creation_id')
    if creation_id:
        return creation_id
    
    # Check survey data
    survey = context_info.get('survey')
    if survey:
        if isinstance(survey, dict):
            return survey.get('survey_id')
        elif hasattr(survey, 'survey_id'):
            return survey.survey_id
    
    return None


def _prepare_context_response(user_response: Any, context_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare context response for workflow creation resume."""
    context_response = {}
    
    if isinstance(user_response, dict):
        context_response = user_response.copy()
    else:
        context_response['user_context'] = user_response
    
    # Merge additional context if provided
    if context_info:
        context_response.update(context_info)
    
    return context_response


def _map_context_info(context_info: Dict[str, Any]) -> Dict[str, Any]:
    """Map provided context to expected keys and inject temporal awareness."""
    mapped_context = {
        "target_audience": context_info.get("target_audience"),
        "design_preferences": {
            "color_scheme": context_info.get("color_scheme"),
            "design_style": context_info.get("design_style"),
            "inspiration_sites": context_info.get("inspiration_sites")
        },
        "functionality_requirements": {
            "key_features": context_info.get("key_features"),
            "need_forms": context_info.get("need_forms"),
            "need_galleries": context_info.get("need_galleries"),
            "need_ecommerce": context_info.get("need_ecommerce"),
            "external_integrations": context_info.get("external_integrations")
        },
        "content_assets": {
            "existing_content": context_info.get("existing_content"),
            "need_content_creation": context_info.get("need_content_creation"),
            "has_brand_guidelines": context_info.get("has_brand_guidelines")
        },
        "technical_constraints": {
            "hosting": context_info.get("hosting"),
            "technical_constraints": context_info.get("technical_constraints"),
            "cms_needed": context_info.get("cms_needed")
        }
    }
    
    # ALWAYS inject temporal awareness context for date/time awareness
    temporal_context = _create_temporal_awareness_context()
    
    # Merge original context with mapped context and temporal awareness
    return {**context_info, **mapped_context, **temporal_context}


async def maestro_search(
    query: str,
    search_engine: str = "duckduckgo",
    num_results: int = 5,
    ctx: Context = None,
) -> List[Dict[str, str]]:
    """
    Performs a web search using a specified search engine and returns the results.

    Args:
        query: The search query.
        search_engine: The search engine to use (e.g., 'duckduckgo', 'google', 'bing').
        num_results: The desired number of search results.
        ctx: The MCP context.

    Returns:
        A list of search result dictionaries, each containing 'title', 'link', and 'snippet'.
    """
    if ctx:
        ctx.info(f"Performing search for '{query}' using {search_engine}")
    try:
        # Lazy import to prevent delays during tool scanning
        from .web import SearchEngine
        engine = SearchEngine(engine=search_engine)
        results = await engine.search(query, num_results=num_results)
        if ctx:
            ctx.info(f"Found {len(results)} results.")
        return results
    except Exception as e:
        if ctx:
            ctx.error(f"Search failed: {e}")
        raise

async def maestro_scrape(
    url: str,
    ctx: Context = None
) -> Dict[str, str]:
    """
    Scrapes the content of a given URL.

    Args:
        url: The URL to scrape.
        ctx: The MCP context.

    Returns:
        A dictionary containing the URL, title, and text content of the page.
    """
    if ctx:
        ctx.info(f"Scraping URL: {url}")
    try:
        # Lazy import to prevent delays during tool scanning
        from .web import Browser
        async with Browser() as browser:
            content = await browser.scrape(url)
            if ctx:
                ctx.info(f"Successfully scraped {len(content.get('text', ''))} characters from {url}.")
            return content
    except Exception as e:
        if ctx:
            ctx.error(f"Scraping failed: {e}")
        raise

async def maestro_web(
    operation: str,
    query_or_url: str,
    search_engine: str = "duckduckgo",
    num_results: int = 5,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Unified web tool for LLM-driven research. Supports only web search (no scraping).

    Args:
        operation: Only 'search' is supported (performs web search).
        query_or_url: Search query string.
        search_engine: The search engine to use (default: duckduckgo).
        num_results: Number of search results to return.
        ctx: The MCP context.

    Returns:
        {"operation": "search", "query": str, "results": List[Dict]}
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if operation != "search":
        raise ValueError("maestro_web only supports 'search' operation. Scraping is not supported.")
    if ctx:
        ctx.info(f"[MaestroWeb] Performing web search for '{query_or_url}' using {search_engine}")
        if config:
            ctx.info(f"[MaestroWeb] Rate limiting enabled: {config.security.rate_limit_enabled}")
    try:
        # Lazy import to prevent delays during tool scanning
        from .web import SearchEngine
        engine = SearchEngine(engine=search_engine)
        results = await engine.search(query_or_url, num_results=num_results)
        if ctx:
            ctx.info(f"[MaestroWeb] Found {len(results)} search results.")
        return {
            "operation": "search",
            "query": query_or_url,
            "search_engine": search_engine,
            "results": results
        }
    except Exception as e:
        if ctx:
            ctx.error(f"[MaestroWeb] Search failed: {e}")
        raise

async def maestro_execute(
    code: str,
    language: str,
    timeout: int = 60,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Executes a block of code in a specified language within a secure sandbox.

    Args:
        code: The source code to execute.
        language: The programming language (e.g., 'python', 'javascript', 'bash').
        timeout: Execution timeout in seconds.
        ctx: The MCP context.

    Returns:
        A dictionary containing the execution status, stdout, stderr, and exit code.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        ctx.info(f"[MaestroExecute] Executing {language} code")
        if config:
            ctx.info(f"[MaestroExecute] Timeout: {config.engine.task_timeout}s")
    from .number_formatter import clean_output
    cmd = []
    if language == 'python':
        cmd = [sys.executable, '-c', code]
    elif language == 'javascript':
        cmd = ['node', '-e', code]
    elif language == 'bash':
        cmd = ['bash', '-c', code]
    else:
        raise ValueError(f"Unsupported language: {language}")
    def run_sync_subprocess(cmd_list: list, timeout_sec: int) -> dict:
        try:
            process_result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            stdout_cleaned = clean_output(process_result.stdout.strip()) if process_result.stdout else ""
            stderr_cleaned = clean_output(process_result.stderr.strip()) if process_result.stderr else ""
            return {
                "exit_code": process_result.returncode,
                "stdout": stdout_cleaned,
                "stderr": stderr_cleaned,
                "status": "success" if process_result.returncode == 0 else "error",
            }
        except FileNotFoundError:
            return {"status": "error", "error": f"Interpreter for '{language}' not found."}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Execution timed out"}
    if ctx:
        await ctx.info(f"Running command in thread: {' '.join(cmd)}")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, functools.partial(run_sync_subprocess, cmd_list=cmd, timeout_sec=timeout)
        )
        if ctx:
            await ctx.info(f"Execution finished with status: {result.get('status')}")
        return result
    except Exception as e:
        if ctx:
            await ctx.error(f"Execution failed in executor: {e}")
        raise

async def maestro_error_handler(
    error_message: str,
    context: Dict[str, Any],
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Analyzes an error and provides a structured response for recovery.

    Args:
        error_message: The error message that occurred.
        context: The context in which the error occurred (e.g., tool name, parameters).
        ctx: The MCP context.

    Returns:
        A dictionary with error analysis and suggested recovery steps.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        ctx.error(f"[MaestroErrorHandler] Analyzing error: {error_message}")
        if config:
            ctx.info(f"[MaestroErrorHandler] Debug mode: {config.engine.mode.value == 'development'}")
    analysis = {
        "error_type": "GenericError",
        "severity": "High",
        "possible_root_cause": "An unknown error occurred.",
        "recovery_suggestion": "Review the error message and context, and try an alternative approach."
    }
    if "timeout" in error_message.lower():
        analysis["error_type"] = "TimeoutError"
        analysis["possible_root_cause"] = "The operation took too long to complete."
        analysis["recovery_suggestion"] = "Try increasing the timeout or simplifying the operation."
    elif "not found" in error_message.lower():
        analysis["error_type"] = "NotFoundError"
        analysis["possible_root_cause"] = "A required resource or file was not found."
        analysis["recovery_suggestion"] = "Verify that all paths are correct and required resources exist."
    return {
        "original_error": error_message,
        "original_context": context,
        "analysis": analysis,
    }

async def maestro_iae(
    engine_name: str,
    method_name: str,
    parameters: Dict[str, Any],
    ctx: Context = None,
) -> Any:
    """
    Invokes a specific capability from an Intelligence Amplification Engine (IAE) using the MCP-native registry and meta-reasoning logic.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        ctx.info(f"[MaestroIAE] Invoking {engine_name}.{method_name}")
        if config:
            ctx.info(f"[MaestroIAE] Engine mode: {config.engine.mode.value}")
    try:
        import sys
        from pathlib import Path
        engines_dir = Path(__file__).parent.parent
        if str(engines_dir) not in sys.path:
            sys.path.insert(0, str(engines_dir))
        from .iae_discovery import IAERegistry
        from .maestro_iae import IAEIntegrationManager
        registry = IAERegistry()
        manager = IAEIntegrationManager(registry)
        await manager.initialize_engines()
        result = await manager.execute_task(
            engine_id=f"iae_{engine_name.lower()}",
            task_name=method_name,
            parameters=parameters,
            context=ctx
        )
        if ctx:
            await ctx.info(f"[MaestroIAE] IAE method {method_name} executed successfully.")
        return result
    except Exception as e:
        if ctx:
            await ctx.error(f"[MaestroIAE] IAE operation failed: {e}")
        raise 

def _convert_result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert StepExecutionResult or other dataclass objects to dictionaries."""
    if is_dataclass(result) and not isinstance(result, type):
        # Convert dataclass to dictionary
        result_dict = asdict(result)
        # Ensure we have consistent fields
        if not result_dict.get("operation_mode"):
            result_dict["operation_mode"] = "orchestrate"
        if not result_dict.get("timestamp"):
            result_dict["timestamp"] = datetime.datetime.now().isoformat()
        return result_dict
    elif isinstance(result, dict):
        # Already a dictionary, just ensure consistent fields
        if not result.get("operation_mode"):
            result["operation_mode"] = "orchestrate"
        if not result.get("timestamp"):
            result["timestamp"] = datetime.datetime.now().isoformat()
        return result
    else:
        # Convert other types to dictionary format
        return {
            "status": "unknown",
            "operation_mode": "orchestrate",
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        } 