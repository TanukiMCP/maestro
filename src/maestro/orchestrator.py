# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Protocol Unified Orchestrator

Provides a single gateway for complex task orchestration by leveraging the
LLM-driven OrchestrationEngine to generate dynamic, multi-agent workflows.
"""

import logging
from typing import Dict, Any, Optional

# Import the new engine and schemas
from .orchestration_engine import OrchestrationEngine, LLMClient
from .schemas import Workflow, Task

logger = logging.getLogger(__name__)

class MAESTROOrchestrator:
    """
    Unified MAESTRO Protocol Orchestrator.
    
    This class serves as the primary entry point for the `maestro_orchestrate` tool.
    It initializes the core `OrchestrationEngine` and delegates the complex task
    of workflow generation to it.
    """
    
    def __init__(self):
        """
        Initializes the MAESTROOrchestrator and its underlying engine.
        """
        # In a real application, the LLMClient would be configured and
        # passed here, likely using dependency injection.
        llm_client = LLMClient()  # Using the placeholder client for now
        
        self.engine = OrchestrationEngine(llm_client=llm_client)
        logger.info("🎭 MAESTRO Unified Orchestrator initialized with LLM-driven engine.")
    
    async def orchestrate_task(
        self, 
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        PRIMARY MCP TOOL: Initiates LLM-driven workflow orchestration.
        
        This tool takes a high-level task description, uses the OrchestrationEngine
        to generate a complete multi-step, multi-agent workflow, and returns
        the structured plan.
        
        Args:
            task_description: Natural language description of the overall goal.
            context: Optional dictionary of additional context for the task.
            
        Returns:
            A dictionary containing the status and the generated workflow plan.
        """
        logger.info(f"🎭 Orchestrating task: {task_description[:100]}...")
        
        try:
            # Step 1: Delegate workflow generation to the new engine
            workflow_plan = await self.engine.orchestrate(
                task_description=task_description,
                user_context=context
            )
            
            # Step 2: Format the successful response
            logger.info(f"✅ Successfully orchestrated workflow {workflow_plan.workflow_id}")
            return {
                "status": "orchestration_complete",
                "workflow": self._format_workflow_for_response(workflow_plan)
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {str(e)}", exc_info=True)
            return {
                "status": "orchestration_failed",
                "error": str(e),
                "message": "The orchestrator failed to generate a valid workflow. Please check the logs for details."
            }

    def _format_workflow_for_response(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Converts the Workflow Pydantic model into the dictionary format
        expected by the tool's user.
        """
        return {
            "workflow_id": workflow.workflow_id,
            "overall_goal": workflow.overall_goal,
            "e2e_validation_criteria": workflow.e2e_validation_criteria,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "agent_profile": {
                        "name": task.agent_profile.name,
                        "system_prompt": task.agent_profile.system_prompt
                    },
                    "tools": task.tools,
                    "validation_criteria": task.validation_criteria,
                    "user_collaboration_required": task.user_collaboration_required,
                    "error_handling_plan": task.error_handling_plan
                }
                for task in workflow.tasks
            ]
        }
