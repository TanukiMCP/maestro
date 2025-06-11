# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Protocol Unified Orchestrator

Provides a single gateway for complex task orchestration by leveraging the
LLM-driven OrchestrationEngine to generate dynamic, multi-agent workflows.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

# Import the new engine, schemas, and IAE discovery
from .orchestration_engine import OrchestrationEngine, LLMClient
from .schemas import Workflow, Task
from .iae_discovery import IAEDiscovery
from .workflow_metadata import WorkflowMetadata, LLMExecutionContext, LLMRole, ExecutionPattern

logger = logging.getLogger(__name__)

class StepStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATION_FAILED = "validation_failed"

@dataclass
class StepContext:
    """Context maintained for each workflow step"""
    step_id: str
    status: StepStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    artifacts: Dict[str, Any]
    validation_results: Dict[str, Any]
    llm_decisions: List[str]
    tool_executions: List[Dict]

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

class EnhancedOrchestrator:
    """
    Enhanced orchestration engine that explicitly guides LLM execution.
    
    Key Features:
    1. Explicit LLM role and responsibility definition
    2. Step-by-step execution with validation
    3. Context maintenance between steps
    4. Clear documentation of LLM decisions
    """
    
    def __init__(self, workflow_metadata: WorkflowMetadata):
        self.metadata = workflow_metadata
        self.current_step = 0
        self.step_contexts: Dict[str, StepContext] = {}
        self.start_time = datetime.now()
        
    async def execute_workflow(self) -> Dict:
        """
        Main workflow execution method.
        
        The LLM is expected to:
        1. Read and understand its role from workflow_metadata
        2. Execute each step using available tools
        3. Validate completion before proceeding
        4. Maintain context between steps
        5. Document decisions and actions
        """
        logger.info(f"Starting workflow execution with ID: {self.metadata.workflow_id}")
        logger.info(f"LLM Role: {self.metadata.llm_execution_context.role}")
        
        # Get explicit instructions for LLM
        llm_instructions = self.metadata.get_llm_instructions()
        logger.info(f"LLM Instructions: {llm_instructions}")
        
        results = {
            "workflow_id": self.metadata.workflow_id,
            "start_time": self.start_time,
            "steps_completed": 0,
            "current_status": StepStatus.NOT_STARTED,
            "artifacts": {},
            "llm_execution_summary": {
                "role": self.metadata.llm_execution_context.role.value,
                "decisions_made": [],
                "tools_used": [],
                "validation_results": []
            }
        }
        
        try:
            # Execute each phase sequentially
            for phase in self.metadata.phases:
                phase_results = await self._execute_phase(phase)
                results["artifacts"][phase["phase_id"]] = phase_results
                
            results["end_time"] = datetime.now()
            results["current_status"] = StepStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            results["current_status"] = StepStatus.FAILED
            results["error"] = str(e)
            
        return results
    
    async def _execute_phase(self, phase: Dict) -> Dict:
        """
        Execute a single workflow phase.
        
        The LLM should:
        1. Understand the phase requirements
        2. Execute necessary tools
        3. Validate results
        4. Maintain context
        """
        logger.info(f"Executing phase: {phase['phase_id']}")
        
        # Create step context
        step_ctx = StepContext(
            step_id=phase["phase_id"],
            status=StepStatus.NOT_STARTED,
            start_time=datetime.now(),
            end_time=None,
            artifacts={},
            validation_results={},
            llm_decisions=[],
            tool_executions=[]
        )
        
        self.step_contexts[phase["phase_id"]] = step_ctx
        
        # Update status
        step_ctx.status = StepStatus.IN_PROGRESS
        
        # Execute phase (LLM implements this using available tools)
        try:
            # Phase execution happens here - LLM uses available tools
            step_ctx.status = StepStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Phase execution failed: {str(e)}")
            step_ctx.status = StepStatus.FAILED
            raise
        
        # Validate phase completion
        if not await self._validate_phase(phase, step_ctx):
            step_ctx.status = StepStatus.VALIDATION_FAILED
            raise ValueError(f"Phase validation failed: {phase['phase_id']}")
        
        step_ctx.end_time = datetime.now()
        return {
            "status": step_ctx.status,
            "duration": (step_ctx.end_time - step_ctx.start_time).total_seconds(),
            "artifacts": step_ctx.artifacts,
            "validation_results": step_ctx.validation_results,
            "llm_decisions": step_ctx.llm_decisions,
            "tool_executions": step_ctx.tool_executions
        }
    
    async def _validate_phase(self, phase: Dict, step_ctx: StepContext) -> bool:
        """
        Validate phase completion.
        
        The LLM should:
        1. Check all success criteria
        2. Validate artifacts
        3. Ensure context is maintained
        """
        logger.info(f"Validating phase: {phase['phase_id']}")
        
        # LLM implements validation logic here
        validation_passed = True  # LLM should set this based on actual validation
        
        step_ctx.validation_results = {
            "passed": validation_passed,
            "timestamp": datetime.now(),
            "criteria_checked": []  # LLM should populate this
        }
        
        return validation_passed

# Example usage (this would be done by the LLM in practice):
async def example_llm_execution():
    # Create workflow metadata
    from .workflow_metadata import example_workflow
    
    # Initialize orchestrator
    orchestrator = EnhancedOrchestrator(example_workflow)
    
    # Execute workflow
    results = await orchestrator.execute_workflow()
    
    return results
