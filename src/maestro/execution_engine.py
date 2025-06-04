# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Execution Engine

Sequential execution engine that runs orchestrated plans step by step,
managing dependencies, error handling, and result compilation.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .orchestration_engine import OrchestrationPlan, ExecutionStep

logger = logging.getLogger(__name__)

class StepStatus(Enum):
    """Status of an execution step"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StepResult:
    """Result of an execution step"""
    step_id: int
    status: StepStatus
    output: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class ExecutionState:
    """Current state of plan execution"""
    plan: OrchestrationPlan
    step_results: Dict[int, StepResult]
    current_step: Optional[int]
    execution_start: datetime
    execution_end: Optional[datetime] = None
    overall_status: StepStatus = StepStatus.PENDING

class ExecutionEngine:
    """
    Sequential execution engine for orchestrated plans
    """
    
    def __init__(self, enhanced_tool_handlers):
        self.enhanced_tool_handlers = enhanced_tool_handlers
        self.active_executions = {}
        
    async def execute_plan(
        self,
        plan: OrchestrationPlan,
        execution_id: Optional[str] = None
    ) -> ExecutionState:
        """
        Execute an orchestration plan sequentially
        
        Args:
            plan: The orchestration plan to execute
            execution_id: Optional execution ID for tracking
            
        Returns:
            ExecutionState with results of all steps
        """
        
        if execution_id is None:
            execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🚀 Starting execution of plan: {execution_id}")
        logger.info(f"📋 Plan: '{plan.task_description}' ({len(plan.execution_steps)} steps)")
        
        # Initialize execution state
        state = ExecutionState(
            plan=plan,
            step_results={},
            current_step=None,
            execution_start=datetime.now(timezone.utc),
            overall_status=StepStatus.RUNNING
        )
        
        # Store active execution
        self.active_executions[execution_id] = state
        
        try:
            # Execute steps sequentially respecting dependencies
            await self._execute_steps_sequentially(state)
            
            # Determine overall status
            state.overall_status = self._determine_overall_status(state)
            state.execution_end = datetime.now(timezone.utc)
            
            logger.info(f"✅ Execution completed: {execution_id} - Status: {state.overall_status.value}")
            return state
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {execution_id} - Error: {str(e)}")
            state.overall_status = StepStatus.FAILED
            state.execution_end = datetime.now(timezone.utc)
            raise
        finally:
            # Clean up active execution
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_steps_sequentially(self, state: ExecutionState):
        """Execute steps in dependency order"""
        
        remaining_steps = list(state.plan.execution_steps)
        completed_step_ids = set()
        
        while remaining_steps:
            # Find steps that can be executed (all dependencies met)
            ready_steps = []
            for step in remaining_steps:
                if all(dep_id in completed_step_ids for dep_id in step.dependencies):
                    ready_steps.append(step)
            
            if not ready_steps:
                # No steps are ready - check for circular dependencies
                logger.error("❌ No executable steps found - possible circular dependencies")
                break
            
            # Execute ready steps (could be parallel, but we'll do sequential for now)
            for step in ready_steps:
                logger.info(f"🔄 Executing step {step.step_id}: {step.description}")
                state.current_step = step.step_id
                
                # Execute the step
                result = await self._execute_single_step(step, state)
                state.step_results[step.step_id] = result
                
                # If step completed successfully, mark as completed
                if result.status == StepStatus.COMPLETED:
                    completed_step_ids.add(step.step_id)
                elif result.status == StepStatus.FAILED:
                    # Handle step failure
                    logger.warning(f"⚠️ Step {step.step_id} failed, attempting recovery...")
                    recovery_result = await self._attempt_step_recovery(step, result, state)
                    if recovery_result.status == StepStatus.COMPLETED:
                        completed_step_ids.add(step.step_id)
                        state.step_results[step.step_id] = recovery_result
                    else:
                        logger.error(f"❌ Step {step.step_id} failed and could not be recovered")
                        # For now, we'll continue execution, but mark this in the results
                
                # Remove from remaining steps
                remaining_steps.remove(step)
        
        state.current_step = None
    
    async def _execute_single_step(self, step: ExecutionStep, state: ExecutionState) -> StepResult:
        """Execute a single step"""
        
        start_time = datetime.now(timezone.utc)
        
        result = StepResult(
            step_id=step.step_id,
            status=StepStatus.RUNNING,
            output=None,
            started_at=start_time
        )
        
        try:
            # Replace placeholders in tool arguments
            tool_arguments = await self._resolve_step_arguments(step, state)
            
            # Execute the tool
            if step.tool_name == "maestro_search":
                tool_result = await self.enhanced_tool_handlers.handle_maestro_search(tool_arguments)
            elif step.tool_name == "maestro_scrape":
                tool_result = await self.enhanced_tool_handlers.handle_maestro_scrape(tool_arguments)
            elif step.tool_name == "maestro_iae":
                tool_result = await self.enhanced_tool_handlers.handle_maestro_iae(tool_arguments)
            elif step.tool_name == "maestro_error_handler":
                tool_result = await self.enhanced_tool_handlers.handle_maestro_error_handler(tool_arguments)
            elif step.tool_name == "maestro_temporal_context":
                tool_result = await self.enhanced_tool_handlers.handle_maestro_temporal_context(tool_arguments)
            elif step.tool_name == "python_execute":
                tool_result = await self._execute_python_code(tool_arguments)
            else:
                raise ValueError(f"Unknown tool: {step.tool_name}")
            
            # Process tool result
            result.output = tool_result
            result.status = StepStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)
            result.execution_time = (result.completed_at - start_time).total_seconds()
            
            logger.info(f"✅ Step {step.step_id} completed in {result.execution_time:.2f}s")
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            result.execution_time = (result.completed_at - start_time).total_seconds()
            
            logger.error(f"❌ Step {step.step_id} failed: {str(e)}")
        
        return result
    
    async def _resolve_step_arguments(self, step: ExecutionStep, state: ExecutionState) -> Dict[str, Any]:
        """Resolve placeholders in step arguments"""
        
        resolved_args = {}
        
        for key, value in step.tool_arguments.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # This is a placeholder
                placeholder = value[2:-2]  # Remove {{ }}
                
                if placeholder == "task_description":
                    resolved_args[key] = state.plan.task_description
                elif placeholder == "current_time":
                    resolved_args[key] = state.plan.context.current_time.isoformat()
                elif placeholder.startswith("step_"):
                    # Reference to another step's output
                    try:
                        step_id = int(placeholder.split("_")[1])
                        if step_id in state.step_results:
                            resolved_args[key] = state.step_results[step_id].output
                        else:
                            resolved_args[key] = value  # Keep placeholder if step not completed
                    except (IndexError, ValueError):
                        resolved_args[key] = value
                else:
                    # Unknown placeholder, keep as is
                    resolved_args[key] = value
            else:
                resolved_args[key] = value
        
        return resolved_args
    
    async def _attempt_step_recovery(self, step: ExecutionStep, failed_result: StepResult, state: ExecutionState) -> StepResult:
        """Attempt to recover from a failed step"""
        
        logger.info(f"🔄 Attempting recovery for step {step.step_id}")
        
        # Use the error handler to get recovery suggestions
        try:
            recovery_args = {
                "error_message": failed_result.error or "Step execution failed",
                "error_context": {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "arguments": step.tool_arguments
                },
                "recovery_suggestions": True
            }
            
            error_handler_result = await self.enhanced_tool_handlers.handle_maestro_error_handler(recovery_args)
            
            # For now, we'll mark recovery as attempted but not actually retry
            # In a full implementation, we could parse the error handler result and retry with modified parameters
            
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,  # Still failed, but recovery was attempted
                output=f"Recovery attempted. Error handler result: {error_handler_result}",
                error=f"Original error: {failed_result.error}",
                execution_time=failed_result.execution_time,
                started_at=failed_result.started_at,
                completed_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"❌ Recovery attempt failed: {str(e)}")
            return failed_result
    
    async def _execute_python_code(self, arguments: Dict[str, Any]) -> Any:
        """Execute Python code with security controls"""
        import subprocess
        import tempfile
        import os
        import sys
        
        code = arguments.get("code", "")
        timeout = arguments.get("timeout", 30)
        safe_mode = arguments.get("safe_mode", True)
        
        if not code:
            return {"status": "error", "message": "No code provided"}
        
        # Security validation
        unsafe_patterns = [
            "import os", "import subprocess", "import sys", "exec(", "eval(",
            "__import__", "open(", "file(", "input(", "raw_input(",
            "compile(", "globals(", "locals(", "vars(", "dir(",
            "getattr(", "setattr(", "delattr(", "hasattr("
        ]
        
        if any(pattern in code for pattern in unsafe_patterns):
            return {
                "status": "error", 
                "message": "Code contains unsafe operations",
                "unsafe_patterns": [p for p in unsafe_patterns if p in code]
            }
        
        try:
            # Create secure execution environment
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # Add safety wrapper
                wrapped_code = f"""
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Capture output
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
{chr(10).join('        ' + line for line in code.split(chr(10)))}
    
    result = {{
        "status": "success",
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "execution_completed": True
    }}
    print("EXECUTION_RESULT:", result)
    
except Exception as e:
    error_result = {{
        "status": "error",
        "error_type": type(e).__name__,
        "error_message": str(e),
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue()
    }}
    print("EXECUTION_RESULT:", error_result)
"""
                temp_file.write(wrapped_code)
                temp_file.flush()
                
                # Execute with restrictions
                process = subprocess.run([
                    sys.executable, temp_file.name
                ], 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=tempfile.gettempdir()  # Restrict working directory
                )
                
                # Parse result
                output_lines = process.stdout.split('\n')
                result_line = None
                for line in output_lines:
                    if line.startswith("EXECUTION_RESULT:"):
                        result_line = line[17:].strip()
                        break
                
                if result_line:
                    try:
                        import ast
                        result = ast.literal_eval(result_line)
                        return result
                    except:
                        pass
                
                # Fallback result format
                return {
                    "status": "completed",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "return_code": process.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "message": f"Code execution exceeded {timeout} seconds timeout"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Execution failed: {str(e)}"
            }
        finally:
            # Clean up temp file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file.name)
            except:
                pass
    
    def _determine_overall_status(self, state: ExecutionState) -> StepStatus:
        """Determine the overall execution status"""
        
        if not state.step_results:
            return StepStatus.PENDING
        
        statuses = [result.status for result in state.step_results.values()]
        
        if all(status == StepStatus.COMPLETED for status in statuses):
            return StepStatus.COMPLETED
        elif any(status == StepStatus.FAILED for status in statuses):
            return StepStatus.FAILED
        elif any(status == StepStatus.RUNNING for status in statuses):
            return StepStatus.RUNNING
        else:
            return StepStatus.PENDING
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionState]:
        """Get the current status of an execution"""
        return self.active_executions.get(execution_id)
    
    def list_active_executions(self) -> List[str]:
        """List all active execution IDs"""
        return list(self.active_executions.keys())
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            state = self.active_executions[execution_id]
            state.overall_status = StepStatus.FAILED
            state.execution_end = datetime.now(timezone.utc)
            del self.active_executions[execution_id]
            logger.info(f"🚫 Execution cancelled: {execution_id}")
            return True
        return False 