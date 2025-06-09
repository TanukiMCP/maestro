# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Protocol Unified Orchestration Engine

Provides dynamic workflow generation and tool orchestration through LLM-driven planning.
This is a backend-only, headless engine designed to be called by external agentic IDEs.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
import inspect
from dataclasses import dataclass
from pathlib import Path

# Correctly import the client from its dedicated module.
from .llm_client import LLMClient, LLMResponse
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import Tool as MCPTool, TextContent

from .schemas import Workflow

logger = logging.getLogger(__name__)

@dataclass
class ToolInfo:
    """Information about an available tool."""
    name: str
    description: str
    tool_type: str
    server: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None

class OrchestrationEngine:
    """
    Main orchestration engine that serves as the gateway for all MAESTRO operations.
    It uses a powerful LLM to generate a complete workflow based on provided context.
    """
    
    def __init__(self, llm_client: "LLMClient"):
        self.llm_client = llm_client
        self._tool_cache: Optional[List[ToolInfo]] = None
        self._tool_cache_time: Optional[datetime] = None
        self._mcp_servers: Dict[str, FastMCP] = {}

    async def orchestrate(
        self,
        ctx: Context,
        task_description: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """
        Main orchestration method. It gathers context and uses an LLM to create a
        complete, structured workflow plan.
        """
        logger.info(f"🎭 Starting LLM-driven orchestration for task: '{task_description}'")
        
        # Step 1: Gather comprehensive context (tools, time, etc.)
        tools = await self.discover_available_tools()
        current_time_utc = datetime.now(timezone.utc).isoformat()

        # Step 2: Generate the system prompt for the orchestrator LLM
        prompt = self._create_orchestration_prompt(
            task_description=task_description,
            user_context=user_context,
            available_tools=tools,
            current_time_utc=current_time_utc
        )
        
        # Step 3: Call the LLM to generate the workflow
        logger.info("📞 Calling LLM to generate orchestration plan...")
        try:
            llm_response = await self.llm_client.generate(
                ctx=ctx,
                prompt=prompt,
                response_format={"type": "json_object", "schema": Workflow.model_json_schema()}
            )
            
            # Step 4: Parse and validate the response into a Workflow object
            try:
                workflow_json = json.loads(llm_response.text)
                workflow = Workflow.model_validate(workflow_json)
                # Assign a unique ID to the workflow
                workflow.workflow_id = f"wf_{uuid.uuid4()}"
                logger.info(f"✅ Successfully generated and validated workflow: {workflow.workflow_id}")
                return workflow
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"❌ Failed to parse or validate LLM response: {e}")
                logger.error(f"Raw LLM Output:\n{llm_response.text}")
                raise ValueError(f"Failed to generate a valid workflow from the LLM: {str(e)}") from e
        except Exception as e:
            logger.error(f"❌ Failed to generate workflow with LLM: {e}")
            raise ValueError(f"LLM call failed during orchestration: {str(e)}") from e

    def _create_orchestration_prompt(
        self,
        task_description: str,
        user_context: Optional[Dict[str, Any]],
        available_tools: List[ToolInfo],
        current_time_utc: str
    ) -> str:
        """Constructs the detailed system prompt for the LLM."""
        
        # Serialize context and tools for inclusion in the prompt
        tools_json_str = json.dumps([tool.to_dict() for tool in available_tools], indent=2)
        user_context_str = json.dumps(user_context, indent=2) if user_context else "None"
        workflow_schema_str = json.dumps(Workflow.model_json_schema(), indent=2)

        return f"""
You are MAESTRO, a world-class AI Orchestration system. Your purpose is to act as a "Master Orchestrator Agent."
Your primary function is to analyze a user's request and design a comprehensive, step-by-step workflow to accomplish their goal. This workflow will be executed by a team of specialized AI agents, and you must define both the tasks and the expert agent profiles required for each task.

**Current Time (UTC):** {current_time_utc}

**Your Task:**
Based on the user's request and the provided context, you must generate a complete workflow as a JSON object. This workflow must adhere strictly to the provided JSON schema.

**1. User's Goal:**
"{task_description}"

**2. Additional User-Provided Context:**
```json
{user_context_str}
```

**3. Available Tools:**
Here is the list of tools available to the agents who will execute this workflow. You must only specify tools from this list in your plan.
```json
{tools_json_str}
```

**4. Your Instructions:**
- **Decompose the Goal:** Break down the user's goal into a sequence of logical, discrete tasks.
- **Assign Expert Agents:** For each task, define an `AgentProfile`. This includes a `name` (e.g., 'Code-Validator') and a detailed `system_prompt` that will guide the agent's behavior and give it the necessary expertise. This creates a "Mixture-of-Agents" (MoA) architecture.
- **Map Tools to Tasks:** For each task, select the appropriate tools from the list of available tools.
- **Define Validation Criteria:** For each task, specify clear, objective criteria to verify its successful completion.
- **Consider Collaboration:** If a task requires user feedback, set `user_collaboration_required` to `true`.
- **Plan for Failure:** For each task, provide a concise `error_handling_plan`.
- **Define Final Success:** Create a final `e2e_validation_criteria` to confirm the entire workflow has met the user's overall goal.
- **Respond in JSON:** Your entire output must be a single JSON object that strictly validates against the following schema. Do not include any other text or explanation outside of the JSON object.

**JSON Schema for Your Response:**
```json
{workflow_schema_str}
```

Now, generate the complete workflow JSON object.
"""

    async def discover_available_tools(self) -> List[ToolInfo]:
        """Dynamically discover all available tools in the environment."""
        now = datetime.now(timezone.utc)
        if (self._tool_cache and self._tool_cache_time and
            (now - self._tool_cache_time).total_seconds() < 300):
            return self._tool_cache

        logger.info("🔄 Discovering available tools...")
        tools = []

        # 1. Discover MAESTRO built-in tools
        tools.extend(await self._discover_maestro_tools())

        # 2. Discover tools from all mounted MCP servers
        try:
            mcp_tools = await self._discover_mcp_tools()
            tools.extend(mcp_tools)
        except Exception as e:
            logger.warning(f"Failed to discover MCP tools: {e}")

        # 3. Discover IDE's built-in tools (e.g., Cursor's tools)
        try:
            ide_tools = await self._discover_ide_tools()
            tools.extend(ide_tools)
        except Exception as e:
            logger.warning(f"Failed to discover IDE tools: {e}")

        # 4. Discover Smithery-installed tools
        try:
            smithery_tools = await self._discover_smithery_tools()
            tools.extend(smithery_tools)
        except Exception as e:
            logger.warning(f"Failed to discover Smithery tools: {e}")

        self._tool_cache = tools
        self._tool_cache_time = now
        logger.info(f"🔍 Discovered {len(tools)} available tools across all sources")
        return tools

    async def _discover_maestro_tools(self) -> List[ToolInfo]:
        """Fetch definitions for built-in MAESTRO tools."""
        return [
            ToolInfo(
                name="maestro_search",
                description="LLM-enhanced web search across multiple engines with intelligent result synthesis.",
                tool_type="maestro",
                server="maestro"
            ),
            ToolInfo(
                name="maestro_iae",
                description="Integrated Analysis Engine for complex computational analysis and data synthesis.",
                tool_type="maestro",
                server="maestro"
            ),
            ToolInfo(
                name="maestro_execute",
                description="Executes a block of code (e.g., Python) in a secure, sandboxed environment.",
                tool_type="maestro",
                server="maestro"
            ),
            ToolInfo(
                name="maestro_error_handler",
                description="Analyzes errors and suggests adaptive recovery strategies.",
                tool_type="maestro",
                server="maestro"
            )
        ]

    async def _discover_mcp_tools(self) -> List[ToolInfo]:
        """Discover tools from all mounted MCP servers."""
        tools = []
        
        # Get list of MCP servers from environment
        try:
            from mcp.server.fastmcp import get_mcp_servers
            servers = await get_mcp_servers()
            self._mcp_servers = servers
            
            for server_name, server in servers.items():
                try:
                    # Get tools from this server
                    server_tools = await server.list_tools()
                    
                    # Convert to ToolInfo objects
                    for tool in server_tools:
                        tools.append(ToolInfo(
                            name=tool.name,
                            description=tool.description,
                            tool_type="mcp",
                            server=server_name,
                            parameters=tool.inputSchema,
                            annotations=tool.annotations
                        ))
                        
                except Exception as e:
                    logger.warning(f"Failed to get tools from server {server_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to discover MCP servers: {e}")
            
        return tools

    async def _discover_ide_tools(self) -> List[ToolInfo]:
        """Discover built-in tools from the IDE."""
        tools = []
        
        # Example: Cursor's built-in tools
        cursor_tools = [
            ToolInfo(
                name="edit_file",
                description="Edit or create a file in the workspace",
                tool_type="ide",
                server="cursor"
            ),
            ToolInfo(
                name="read_file",
                description="Read the contents of a file",
                tool_type="ide",
                server="cursor"
            ),
            ToolInfo(
                name="list_dir",
                description="List contents of a directory",
                tool_type="ide",
                server="cursor"
            ),
            ToolInfo(
                name="grep_search",
                description="Search for text patterns in files",
                tool_type="ide",
                server="cursor"
            ),
            ToolInfo(
                name="file_search",
                description="Search for files by name",
                tool_type="ide",
                server="cursor"
            ),
            ToolInfo(
                name="codebase_search",
                description="Semantic search over codebase",
                tool_type="ide",
                server="cursor"
            )
        ]
        tools.extend(cursor_tools)
        
        return tools

    async def _discover_smithery_tools(self) -> List[ToolInfo]:
        """Discover tools from Smithery-installed MCP servers."""
        tools = []
        
        # Common Smithery servers
        smithery_servers = {
            "puppeteer": {
                "tools": [
                    ToolInfo(
                        name="puppeteer_navigate",
                        description="Navigate to a URL using Puppeteer",
                        tool_type="smithery",
                        server="puppeteer"
                    ),
                    ToolInfo(
                        name="puppeteer_screenshot",
                        description="Take a screenshot using Puppeteer",
                        tool_type="smithery",
                        server="puppeteer"
                    ),
                    ToolInfo(
                        name="puppeteer_click",
                        description="Click an element using Puppeteer",
                        tool_type="smithery",
                        server="puppeteer"
                    )
                ]
            },
            "clear-thought": {
                "tools": [
                    ToolInfo(
                        name="sequential_thinking",
                        description="Apply sequential thinking to solve problems",
                        tool_type="smithery",
                        server="clear-thought"
                    ),
                    ToolInfo(
                        name="mental_model",
                        description="Apply mental models to problem-solving",
                        tool_type="smithery",
                        server="clear-thought"
                    ),
                    ToolInfo(
                        name="design_pattern",
                        description="Apply design patterns to software architecture",
                        tool_type="smithery",
                        server="clear-thought"
                    )
                ]
            }
        }
        
        # Add tools from each server if available
        for server_name, server_info in smithery_servers.items():
            try:
                # Check if server is actually available
                if server_name in self._mcp_servers:
                    tools.extend(server_info["tools"])
            except Exception as e:
                logger.warning(f"Failed to get tools from Smithery server {server_name}: {e}")
                
        return tools

    def _map_tools_to_workflow(self, workflow: Dict[str, Any], available_tools: List[ToolInfo]) -> Dict[str, Any]:
        """Map discovered tools to workflow phases based on capabilities."""
        tool_map = {}
        
        # Create lookup by tool type
        tools_by_type = {}
        for tool in available_tools:
            if tool.tool_type not in tools_by_type:
                tools_by_type[tool.tool_type] = []
            tools_by_type[tool.tool_type].append(tool)
            
        # Map common phase patterns to tool types
        phase_tool_patterns = {
            "analysis": ["ide", "clear-thought", "maestro"],
            "research": ["puppeteer", "maestro", "ide"],
            "planning": ["clear-thought", "ide", "maestro"],
            "implementation": ["ide", "maestro"],
            "testing": ["ide", "maestro"],
            "documentation": ["ide", "clear-thought"]
        }
        
        # Map tools to each phase
        for phase in workflow.get("phases", []):
            phase_name = phase["name"].lower()
            phase_tools = []
            
            # Get relevant tool types for this phase
            relevant_types = phase_tool_patterns.get(
                phase_name, 
                ["maestro", "ide"]  # Default to maestro and IDE tools
            )
            
            # Add tools of relevant types
            for tool_type in relevant_types:
                if tool_type in tools_by_type:
                    phase_tools.extend(tools_by_type[tool_type])
                    
            tool_map[phase["name"]] = phase_tools
            
        return tool_map