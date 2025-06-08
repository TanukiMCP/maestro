# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Orchestration Engine

Main gateway that automatically gathers context and tool awareness, then
delegates to an LLM to orchestrate detailed execution plans for agent coordination.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

# Correctly import the client from its dedicated module.
from .llm_client import LLMClient, LLMResponse

from .schemas import Workflow

logger = logging.getLogger(__name__)

# The ToolInfo and ContextInfo dataclasses are simplified as their detailed
# contents will now be formatted into the prompt for the LLM.

class ToolInfo:
    """Minimal information about an available tool for logging and context."""
    def __init__(self, name: str, description: str, tool_type: str):
        self.name = name
        self.description = description
        self.tool_type = tool_type

    def to_dict(self):
        return {"name": self.name, "description": self.description, "type": self.tool_type}

class OrchestrationEngine:
    """
    Main orchestration engine that serves as the gateway for all MAESTRO operations.
    It uses a powerful LLM to generate a complete workflow based on provided context.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self._tool_cache: Optional[List[ToolInfo]] = None
        self._tool_cache_time: Optional[datetime] = None

    async def orchestrate(
        self,
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
        llm_response = await self.llm_client.generate(
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
            raise ValueError("Failed to generate a valid workflow from the LLM.") from e

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
        tools.extend(await self._discover_maestro_tools())
        # In a real implementation, you would also discover MCP and other built-in tools.
        # tools.extend(await self._discover_mcp_tools())
        # tools.extend(await self._discover_builtin_tools())

        self._tool_cache = tools
        self._tool_cache_time = now
        logger.info(f"🔍 Discovered {len(tools)} available tools.")
        return tools

    async def _discover_maestro_tools(self) -> List[ToolInfo]:
        """Fetch definitions for built-in MAESTRO tools."""
        return [
            ToolInfo(
                name="maestro_search",
                description="LLM-enhanced web search across multiple engines with intelligent result synthesis.",
                tool_type="maestro"
            ),
            ToolInfo(
                name="maestro_iae",
                description="Integrated Analysis Engine for complex computational analysis and data synthesis.",
                tool_type="maestro"
            ),
            ToolInfo(
                name="maestro_execute",
                description="Executes a block of code (e.g., Python) in a secure, sandboxed environment.",
                tool_type="maestro"
            ),
            ToolInfo(
                name="maestro_error_handler",
                description="Analyzes errors and suggests adaptive recovery strategies.",
                tool_type="maestro"
            )
        ]