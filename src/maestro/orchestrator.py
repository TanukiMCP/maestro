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

# Import the new engine, schemas, and IAE discovery
from .orchestration_engine import OrchestrationEngine, LLMClient
from .schemas import Workflow, Task
from .iae_discovery import IAEDiscovery

logger = logging.getLogger(__name__)

class MAESTROOrchestrator:
    """
    Unified MAESTRO Protocol Orchestrator.
    
    This class serves as the primary entry point for the `maestro_orchestrate` tool.
    It initializes the core `OrchestrationEngine` and delegates the complex task
    of workflow generation to it.

    **IMPORTANT:** This orchestrator is backend-only, headless, and must be called by an external agentic IDE or LLM client. It must NOT instantiate or assume any built-in LLM client or UI. The LLM client (or orchestration context) must be provided by the external caller at runtime.
    """
    
    def __init__(self, llm_client: "LLMClient"):
        """
        Initializes the MAESTROOrchestrator and its underlying engine.
        Args:
            llm_client: An LLM client or orchestration context provided by the external agentic IDE or LLM client. This must NOT be instantiated internally.
        """
        self.engine = OrchestrationEngine(llm_client=llm_client)
        self.iae_discovery = IAEDiscovery()
        logger.info("🎭 MAESTRO Unified Orchestrator initialized with externally provided LLM-driven engine.")
    
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
            # Step 1: Discover relevant engines for the task
            task_type = await self.engine.infer_task_type(task_description)
            domain_context = await self.engine.infer_domain_context(task_description)
            
            engine_discovery = await self.iae_discovery.discover_engines_for_task(
                task_type=task_type,
                domain_context=domain_context
            )
            
            if engine_discovery["status"] != "success":
                raise Exception(f"Engine discovery failed: {engine_discovery.get('message')}")
                
            # Step 2: Delegate workflow generation to the orchestration engine
            workflow_plan = await self.engine.orchestrate(
                task_description=task_description,
                user_context=context,
                available_engines=engine_discovery["discovered_engines"]
            )
            
            # Step 3: Format the successful response
            logger.info(f"✅ Successfully orchestrated workflow {workflow_plan.workflow_id}")
            return {
                "status": "orchestration_complete",
                "workflow": self._format_workflow_for_response(workflow_plan),
                "available_engines": engine_discovery["discovered_engines"]
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
