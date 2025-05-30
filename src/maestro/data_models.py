"""
Core Data Models for MAESTRO Protocol

Defines all data structures used throughout the orchestration system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid
from datetime import datetime


class TaskType(Enum):
    """Types of tasks that can be orchestrated."""
    MATHEMATICS = "mathematics"
    WEB_DEVELOPMENT = "web_development"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    CODE_DEVELOPMENT = "code_development"
    LANGUAGE_PROCESSING = "language_processing"
    GENERAL = "general"


class ComplexityLevel(Enum):
    """Complexity levels for task assessment."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class VerificationMethod(Enum):
    """Available verification methods."""
    AUTOMATED_TESTING = "automated_testing"
    MATHEMATICAL_VERIFICATION = "mathematical_verification"
    CODE_QUALITY_VERIFICATION = "code_quality_verification"
    VISUAL_VERIFICATION = "visual_verification"
    LANGUAGE_QUALITY_VERIFICATION = "language_quality_verification"
    ACCESSIBILITY_VERIFICATION = "accessibility_verification"


@dataclass
class QualityMetrics:
    """Quality metrics for evaluation."""
    overall_score: float = 0.0
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    quality_score: float = 0.0
    confidence_score: float = 0.0
    verification_scores: Dict[str, float] = field(default_factory=dict)
    
    def meets_threshold(self, threshold: float) -> bool:
        """Check if overall score meets quality threshold."""
        return self.overall_score >= threshold


@dataclass
class TaskAnalysis:
    """Analysis results for a task request."""
    task_type: TaskType
    complexity: ComplexityLevel
    capabilities: List[str]
    estimated_duration: str  # Changed from int to str for better readability
    required_tools: List[str]
    success_criteria: List[str]
    quality_requirements: Dict[str, float]
    assigned_operator: Optional[str] = None  # Operator profile ID assigned
    metadata: Optional[Dict[str, Any]] = None  # For knowledge graph integration
    
    
@dataclass
class VerificationResult:
    """Result of quality verification."""
    success: bool
    confidence_score: float
    quality_metrics: QualityMetrics
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    verification_time: float = 0.0


@dataclass
class WorkflowNode:
    """Individual workflow step with quality control."""
    node_id: str
    task_description: str
    operator_profile_id: str
    required_capabilities: List[str]
    success_criteria: List[str]
    verification_methods: List[VerificationMethod]
    minimum_confidence_threshold: float = 0.85
    max_retry_attempts: int = 3
    input_context: Dict[str, Any] = field(default_factory=dict)
    output_context: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[Any] = None
    verification_result: Optional[VerificationResult] = None
    
    @classmethod
    def create(cls, task_description: str, capabilities: List[str], 
               success_criteria: List[str]) -> "WorkflowNode":
        """Create a new workflow node with generated ID."""
        return cls(
            node_id=f"node_{uuid.uuid4().hex[:8]}",
            task_description=task_description,
            operator_profile_id="",
            required_capabilities=capabilities,
            success_criteria=success_criteria,
            verification_methods=[VerificationMethod.AUTOMATED_TESTING]
        )


@dataclass
class Workflow:
    """Complete workflow definition."""
    workflow_id: str
    task_description: str
    nodes: List[WorkflowNode]
    capabilities_used: List[str]
    quality_threshold: float
    verification_mode: str
    created_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def create(cls, task_description: str, quality_threshold: float = 0.9,
               verification_mode: str = "comprehensive") -> "Workflow":
        """Create a new workflow with generated ID."""
        return cls(
            workflow_id=f"workflow_{uuid.uuid4().hex[:8]}",
            task_description=task_description,
            nodes=[],
            capabilities_used=[],
            quality_threshold=quality_threshold,
            verification_mode=verification_mode
        )


@dataclass
class ExecutionMetrics:
    """Metrics from workflow execution."""
    total_time: float
    nodes_executed: int
    nodes_successful: int
    retries_performed: int
    quality_checks_run: int
    early_stopping_triggered: bool = False


@dataclass
class MAESTROResult:
    """Complete result from MAESTRO orchestration."""
    success: bool
    task_description: str
    detailed_output: str
    summary: str
    workflow_used: Workflow
    operator_profile_id: str
    verification: VerificationResult
    execution_metrics: ExecutionMetrics
    files_affected: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def format_success_response(self) -> str:
        """Format the result for user presentation."""
        if self.success:
            return f"""
## MAESTRO Protocol Execution Complete ✅

**Task Accomplished:** {self.task_description}

**Solution:**
{self.detailed_output}

**Quality Verification Results:**
- **Overall Quality Score:** {self.verification.quality_metrics.overall_score:.2%}
- **Verification Methods Used:** {len(self.verification.detailed_results)}
- **All Success Criteria Met:** ✅ Yes

**Workflow Details:**
- **Operator Profile:** {self.operator_profile_id}
- **Nodes Executed:** {self.execution_metrics.nodes_executed}
- **Capabilities Used:** {', '.join(self.workflow_used.capabilities_used)}
- **Execution Time:** {self.execution_metrics.total_time:.1f}s

**Verification Summary:**
{self.verification.confidence_score:.2%} confidence with {len(self.verification.issues_found)} issues found.

**Files Created/Modified:**
{chr(10).join(f"- {file}" for file in self.files_affected)}

This solution has been automatically verified and meets all quality standards.
"""
        else:
            return f"""
## MAESTRO Protocol Quality Assessment

**Original Task:** {self.task_description}

**Current Status:** Quality verification did not meet success criteria

**What was accomplished:**
{self.summary}

**Quality Analysis:**
Overall Score: {self.verification.quality_metrics.overall_score:.2%}

**Specific Issues Identified:**
{chr(10).join(f"- {issue}" for issue in self.verification.issues_found)}

**Recommended Next Steps:**
{chr(10).join(f"- {rec}" for rec in self.verification.recommendations)}

Would you like me to retry with adjusted parameters or try an alternative approach?
"""


@dataclass
class OperatorProfile:
    """Specialized operator configuration for workflow execution."""
    profile_id: str
    profile_type: TaskType
    system_prompt: str
    capabilities: List[str]
    quality_standards: Dict[str, float]
    verification_requirements: List[VerificationMethod]
    model_preferences: List[str] = field(default_factory=list)
    tool_permissions: List[str] = field(default_factory=list)
    
    @classmethod
    def create(cls, profile_type: TaskType, capabilities: List[str]) -> "OperatorProfile":
        """Create a new operator profile with generated ID."""
        return cls(
            profile_id=f"operator_{profile_type.value}_{uuid.uuid4().hex[:8]}",
            profile_type=profile_type,
            system_prompt="",
            capabilities=capabilities,
            quality_standards={},
            verification_requirements=[]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "profile_id": self.profile_id,
            "profile_type": self.profile_type.value,
            "system_prompt": self.system_prompt,
            "capabilities": self.capabilities,
            "quality_standards": self.quality_standards,
            "verification_requirements": [v.value for v in self.verification_requirements],
            "model_preferences": self.model_preferences,
            "tool_permissions": self.tool_permissions
        } 