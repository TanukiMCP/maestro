"""
Enhanced MCP workflow metadata structure that explicitly defines LLM execution context and responsibilities.
This ensures any LLM using the protocol understands its role in the execution process.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class LLMRole(Enum):
    PRIMARY_EXECUTOR = "primary_executor"  # LLM executes steps and validates completion
    COORDINATOR = "coordinator"           # LLM coordinates but delegates execution
    VALIDATOR = "validator"              # LLM only validates execution
    
class ExecutionPattern(Enum):
    SEQUENTIAL = "sequential"                    # Execute steps in strict sequence
    SEQUENTIAL_WITH_VALIDATION = "sequential_with_validation"  # Validate before proceeding
    PARALLEL = "parallel"                        # Execute steps in parallel where possible
    ADAPTIVE = "adaptive"                        # Dynamically determine execution pattern

@dataclass
class LLMExecutionContext:
    """Explicit definition of LLM's role and responsibilities in workflow execution"""
    role: LLMRole
    responsibilities: List[str]
    execution_pattern: ExecutionPattern
    required_capabilities: List[str]
    validation_criteria: Dict[str, Union[str, float]]
    context_maintenance: Dict[str, str]  # How LLM should maintain workflow context
    
    def validate_step_completion(self, step_results: Dict) -> bool:
        """LLM should implement this to validate step completion"""
        pass
    
    def maintain_context(self, current_state: Dict) -> Dict:
        """LLM should implement this to maintain workflow context"""
        pass

@dataclass
class WorkflowMetadata:
    """Enhanced workflow metadata that explicitly includes LLM execution context"""
    workflow_id: str
    task_description: str
    llm_execution_context: LLMExecutionContext  # Explicit LLM context
    phases: List[Dict]
    tools: Dict[str, Dict]
    success_criteria: Dict[str, Union[str, float]]
    
    # Additional metadata fields
    complexity: str
    estimated_duration: str
    dependencies: List[str]
    required_tools: List[str]
    
    def get_llm_instructions(self) -> Dict[str, str]:
        """Returns explicit instructions for LLM execution"""
        return {
            "role": self.llm_execution_context.role.value,
            "primary_responsibility": """
            You (the LLM) are responsible for:
            1. Executing each step using available IDE tools
            2. Validating step completion before proceeding
            3. Maintaining workflow context between steps
            4. Using your capabilities to enhance the execution process
            5. Documenting actions and decisions made
            """,
            "execution_pattern": self.llm_execution_context.execution_pattern.value,
            "validation_requirements": str(self.llm_execution_context.validation_criteria),
            "context_maintenance": str(self.llm_execution_context.context_maintenance)
        }

# Example usage:
example_context = LLMExecutionContext(
    role=LLMRole.PRIMARY_EXECUTOR,
    responsibilities=[
        "Execute each step using available IDE tools",
        "Validate step completion before proceeding",
        "Maintain workflow context and state"
    ],
    execution_pattern=ExecutionPattern.SEQUENTIAL_WITH_VALIDATION,
    required_capabilities=[
        "code_generation",
        "code_analysis",
        "natural_language_understanding"
    ],
    validation_criteria={
        "completion_threshold": 0.85,
        "validation_strategy": "comprehensive"
    },
    context_maintenance={
        "state_tracking": "explicit",
        "context_updates": "continuous"
    }
)

example_workflow = WorkflowMetadata(
    workflow_id="example_workflow_001",
    task_description="Example task",
    llm_execution_context=example_context,
    phases=[],  # Define phases here
    tools={},   # Define tools here
    success_criteria={
        "completion_threshold": 0.85,
        "validation_strategy": "comprehensive"
    },
    complexity="moderate",
    estimated_duration="2 hours",
    dependencies=[],
    required_tools=["edit_file", "read_file"]
) 