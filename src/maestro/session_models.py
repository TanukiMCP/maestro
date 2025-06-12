from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

# Enums for structured types
class ConceptualFrameworkType(str, Enum):
    TASK_DECOMPOSITION = "task_decomposition"
    DEPENDENCY_GRAPH = "dependency_graph"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    CUSTOM = "custom"

class WorkflowPhase(str, Enum):
    PLANNING = "planning"
    DECOMPOSITION = "decomposition"
    EXECUTION = "execution"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"

class KnowledgeType(str, Enum):
    TOOL_EFFECTIVENESS = "tool_effectiveness"
    APPROACH_OUTCOME = "approach_outcome"
    ERROR_PATTERN = "error_pattern"
    OPTIMIZATION_INSIGHT = "optimization_insight"
    DOMAIN_SPECIFIC = "domain_specific"
    WORKFLOW_PATTERN = "workflow_pattern"

# Original models with enhancements
class BuiltInTool(BaseModel):
    name: str
    description: str
    always_available: bool = True
    capabilities: List[str] = []
    relevant_for: List[str] = []
    # New fields for self-directed orchestration
    effectiveness_scores: Dict[str, float] = {}  # task_type -> score
    usage_patterns: List[Dict[str, Any]] = []

class MCPTool(BaseModel):
    name: str
    description: str
    server_name: str
    capabilities: List[str] = []
    relevant_for: List[str] = []
    # New fields for self-directed orchestration
    effectiveness_scores: Dict[str, float] = {}  # task_type -> score
    usage_patterns: List[Dict[str, Any]] = []

class UserResource(BaseModel):
    name: str
    type: str  # "documentation", "codebase", "api", "knowledge_base", etc.
    description: str
    indexed_content: Optional[str] = None
    source_url: Optional[str] = None
    relevant_for: List[str] = []
    # New fields for self-directed orchestration
    access_patterns: List[Dict[str, Any]] = []
    relevance_scores: Dict[str, float] = {}  # context -> score

class EnvironmentCapabilities(BaseModel):
    built_in_tools: List[BuiltInTool] = []
    mcp_tools: List[MCPTool] = []
    user_resources: List[UserResource] = []

# New models for conceptual frameworks
class TaskNode(BaseModel):
    """Represents a node in task decomposition"""
    id: str = Field(default_factory=lambda: f"node_{uuid.uuid4()}")
    name: str
    description: str
    type: str  # "atomic", "composite", "milestone", "decision"
    dependencies: List[str] = []  # IDs of other TaskNodes
    success_criteria: List[str] = []
    estimated_complexity: Optional[float] = None
    assigned_tools: List[str] = []
    required_capabilities: Dict[str, List[str]] = Field(default_factory=lambda: {
        "builtin_tools": [],
        "mcp_tools": [],
        "resources": []
    })
    metadata: Dict[str, Any] = {}
    
    def validate_capabilities(self, available_capabilities: EnvironmentCapabilities) -> Dict[str, Any]:
        """Validate that required capabilities are available"""
        validation_result = {
            "valid": True,
            "missing": {"builtin_tools": [], "mcp_tools": [], "resources": []},
            "warnings": []
        }
        
        # Check builtin tools
        available_builtin = [t.name for t in available_capabilities.built_in_tools]
        for tool in self.required_capabilities.get("builtin_tools", []):
            if tool not in available_builtin:
                validation_result["valid"] = False
                validation_result["missing"]["builtin_tools"].append(tool)
        
        # Check MCP tools
        available_mcp = [t.name for t in available_capabilities.mcp_tools]
        for tool in self.required_capabilities.get("mcp_tools", []):
            if tool not in available_mcp:
                validation_result["valid"] = False
                validation_result["missing"]["mcp_tools"].append(tool)
                
                # Special warning for IAE
                if tool == "maestro_iae":
                    validation_result["warnings"].append(
                        "maestro_iae is required for computational tasks but not declared. "
                        "Declare it to access mathematical, scientific, and analytical engines."
                    )
        
        # Check resources
        available_resources = [r.name for r in available_capabilities.user_resources]
        for resource in self.required_capabilities.get("resources", []):
            if resource not in available_resources:
                validation_result["valid"] = False
                validation_result["missing"]["resources"].append(resource)
        
        return validation_result

class ConceptualFramework(BaseModel):
    """Stores LLM-created conceptual frameworks for orchestration"""
    id: str = Field(default_factory=lambda: f"framework_{uuid.uuid4()}")
    type: ConceptualFrameworkType
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    structure: Dict[str, Any]  # Flexible structure for different framework types
    task_nodes: List[TaskNode] = []  # For task decomposition
    relationships: List[Dict[str, Any]] = []  # For dependency graphs
    optimization_rules: List[Dict[str, Any]] = []  # For workflow optimization
    metadata: Dict[str, Any] = {}
    
    def get_all_required_capabilities(self) -> Dict[str, List[str]]:
        """Aggregate all required capabilities from task nodes"""
        all_capabilities = {
            "builtin_tools": set(),
            "mcp_tools": set(),
            "resources": set()
        }
        
        for node in self.task_nodes:
            for tool in node.required_capabilities.get("builtin_tools", []):
                all_capabilities["builtin_tools"].add(tool)
            for tool in node.required_capabilities.get("mcp_tools", []):
                all_capabilities["mcp_tools"].add(tool)
            for resource in node.required_capabilities.get("resources", []):
                all_capabilities["resources"].add(resource)
        
        # Check if computational tasks need IAE
        if self.type in [ConceptualFrameworkType.TASK_DECOMPOSITION, 
                        ConceptualFrameworkType.WORKFLOW_OPTIMIZATION]:
            structure_str = str(self.structure).lower()
            if any(keyword in structure_str for keyword in ["compute", "calculate", "analyze", "mathematical"]):
                all_capabilities["mcp_tools"].add("maestro_iae")
        
        return {
            "builtin_tools": list(all_capabilities["builtin_tools"]),
            "mcp_tools": list(all_capabilities["mcp_tools"]),
            "resources": list(all_capabilities["resources"])
        }

class WorkflowState(BaseModel):
    """Tracks the current state of workflow execution"""
    id: str = Field(default_factory=lambda: f"state_{uuid.uuid4()}")
    current_phase: WorkflowPhase
    current_step: Optional[str] = None
    completed_steps: List[str] = []
    active_frameworks: List[str] = []  # IDs of ConceptualFrameworks in use
    execution_context: Dict[str, Any] = {}
    decision_history: List[Dict[str, Any]] = []
    performance_metrics: Dict[str, Any] = {}
    capability_usage: Dict[str, List[str]] = Field(default_factory=lambda: {
        "builtin_tools_used": [],
        "mcp_tools_used": [],
        "resources_accessed": []
    })
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def track_capability_usage(self, capability_type: str, capability_name: str):
        """Track usage of a capability"""
        if capability_type == "builtin":
            if capability_name not in self.capability_usage["builtin_tools_used"]:
                self.capability_usage["builtin_tools_used"].append(capability_name)
        elif capability_type == "mcp":
            if capability_name not in self.capability_usage["mcp_tools_used"]:
                self.capability_usage["mcp_tools_used"].append(capability_name)
        elif capability_type == "resource":
            if capability_name not in self.capability_usage["resources_accessed"]:
                self.capability_usage["resources_accessed"].append(capability_name)

class SessionKnowledge(BaseModel):
    """Represents knowledge learned during session execution"""
    id: str = Field(default_factory=lambda: f"knowledge_{uuid.uuid4()}")
    type: KnowledgeType
    subject: str  # What this knowledge is about
    context: Dict[str, Any]  # Context in which this was learned
    insights: List[str]  # Key insights or learnings
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence in this knowledge
    applicable_scenarios: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = 0

class Task(BaseModel):
    """Enhanced task with decomposition and tracking capabilities"""
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4()}")
    description: str
    status: str = "[ ]"  # Enforce "[ ]" or "[X]"
    # Original fields
    validation_required: bool = False
    validation_criteria: List[str] = []
    evidence: List[dict] = []
    execution_started: bool = False
    execution_evidence: List[str] = []
    suggested_builtin_tools: List[str] = []
    suggested_mcp_tools: List[str] = []
    suggested_resources: List[str] = []
    # New fields for self-directed orchestration
    parent_task_id: Optional[str] = None  # For hierarchical decomposition
    subtask_ids: List[str] = []  # Child tasks
    task_node_id: Optional[str] = None  # Link to TaskNode in framework
    complexity_score: Optional[float] = None
    approach_strategy: Optional[Dict[str, Any]] = None
    learning_outcomes: List[str] = []
    workflow_phase: Optional[WorkflowPhase] = None
    capability_validation: Optional[Dict[str, Any]] = None  # Validation results

    @validator('status')
    def validate_status(cls, v):
        if v not in ["[ ]", "[X]"]:
            raise ValueError("Status must be '[ ]' or '[X]'")
        return v
    
    def validate_required_capabilities(self, available_capabilities: EnvironmentCapabilities) -> Dict[str, Any]:
        """Validate that required capabilities for this task are available"""
        validation_result = {
            "valid": True,
            "missing": {"builtin_tools": [], "mcp_tools": [], "resources": []},
            "warnings": []
        }
        
        # Check suggested tools against available
        available_builtin = [t.name for t in available_capabilities.built_in_tools]
        available_mcp = [t.name for t in available_capabilities.mcp_tools]
        available_resources = [r.name for r in available_capabilities.user_resources]
        
        for tool in self.suggested_builtin_tools:
            if tool not in available_builtin:
                validation_result["valid"] = False
                validation_result["missing"]["builtin_tools"].append(tool)
        
        for tool in self.suggested_mcp_tools:
            if tool not in available_mcp:
                validation_result["valid"] = False
                validation_result["missing"]["mcp_tools"].append(tool)
                
                if tool == "maestro_iae" and any(keyword in self.description.lower() 
                                               for keyword in ["compute", "calculate", "analyze"]):
                    validation_result["warnings"].append(
                        f"Task '{self.description}' appears to need computational capabilities "
                        "but maestro_iae is not declared"
                    )
        
        for resource in self.suggested_resources:
            if resource not in available_resources:
                validation_result["valid"] = False
                validation_result["missing"]["resources"].append(resource)
        
        self.capability_validation = validation_result
        return validation_result

class Session(BaseModel):
    """Enhanced session with full self-directed orchestration support"""
    id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")
    session_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Original fields
    tasks: List[Task] = []
    capabilities: EnvironmentCapabilities = Field(default_factory=EnvironmentCapabilities)
    environment_context: Optional[Dict] = None
    
    # New fields for self-directed orchestration
    conceptual_frameworks: List[ConceptualFramework] = []
    workflow_states: List[WorkflowState] = []  # History of workflow states
    current_workflow_state: Optional[WorkflowState] = None
    session_knowledge: List[SessionKnowledge] = []
    
    # Orchestration metadata
    orchestration_strategy: Optional[Dict[str, Any]] = None
    optimization_goals: List[str] = []
    constraints: List[Dict[str, Any]] = []
    performance_history: List[Dict[str, Any]] = []
    
    # Learning and adaptation
    adaptation_rules: List[Dict[str, Any]] = []
    success_patterns: List[Dict[str, Any]] = []
    failure_patterns: List[Dict[str, Any]] = []
    
    def add_framework(self, framework: ConceptualFramework) -> None:
        """Add a conceptual framework to the session"""
        self.conceptual_frameworks.append(framework)
        self.updated_at = datetime.utcnow()
    
    def add_knowledge(self, knowledge: SessionKnowledge) -> None:
        """Add learned knowledge to the session"""
        self.session_knowledge.append(knowledge)
        self.updated_at = datetime.utcnow()
    
    def update_workflow_state(self, new_state: WorkflowState) -> None:
        """Update the current workflow state and maintain history"""
        if self.current_workflow_state:
            self.workflow_states.append(self.current_workflow_state)
        self.current_workflow_state = new_state
        self.updated_at = datetime.utcnow()
    
    def get_relevant_knowledge(self, context: str, knowledge_type: Optional[KnowledgeType] = None) -> List[SessionKnowledge]:
        """Retrieve relevant knowledge based on context"""
        relevant = []
        for knowledge in self.session_knowledge:
            if knowledge_type and knowledge.type != knowledge_type:
                continue
            if context.lower() in knowledge.subject.lower() or \
               any(context.lower() in scenario.lower() for scenario in knowledge.applicable_scenarios):
                relevant.append(knowledge)
        return sorted(relevant, key=lambda k: k.confidence, reverse=True)
    
    def get_active_framework(self, framework_type: Optional[ConceptualFrameworkType] = None) -> Optional[ConceptualFramework]:
        """Get the most recently active framework of a given type"""
        if not self.current_workflow_state:
            return None
        
        for framework_id in reversed(self.current_workflow_state.active_frameworks):
            for framework in self.conceptual_frameworks:
                if framework.id == framework_id:
                    if framework_type is None or framework.type == framework_type:
                        return framework
        return None
    
    def get_task_hierarchy(self, task_id: str) -> Dict[str, Any]:
        """Get the hierarchical structure of a task and its subtasks"""
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            return {}
        
        hierarchy = {
            "task": task,
            "subtasks": []
        }
        
        for subtask_id in task.subtask_ids:
            subtask_hierarchy = self.get_task_hierarchy(subtask_id)
            if subtask_hierarchy:
                hierarchy["subtasks"].append(subtask_hierarchy)
        
        return hierarchy
    
    def validate_all_capabilities(self) -> Dict[str, Any]:
        """Validate all capability requirements across frameworks and tasks"""
        validation_summary = {
            "session_valid": True,
            "framework_validations": {},
            "task_validations": {},
            "missing_capabilities": {"builtin_tools": set(), "mcp_tools": set(), "resources": set()},
            "warnings": []
        }
        
        # Validate frameworks
        for framework in self.conceptual_frameworks:
            required = framework.get_all_required_capabilities()
            for node in framework.task_nodes:
                node_validation = node.validate_capabilities(self.capabilities)
                if not node_validation["valid"]:
                    validation_summary["session_valid"] = False
                    validation_summary["framework_validations"][framework.id] = node_validation
                    
                    # Aggregate missing capabilities
                    for tool in node_validation["missing"]["builtin_tools"]:
                        validation_summary["missing_capabilities"]["builtin_tools"].add(tool)
                    for tool in node_validation["missing"]["mcp_tools"]:
                        validation_summary["missing_capabilities"]["mcp_tools"].add(tool)
                    for resource in node_validation["missing"]["resources"]:
                        validation_summary["missing_capabilities"]["resources"].add(resource)
        
        # Validate tasks
        for task in self.tasks:
            task_validation = task.validate_required_capabilities(self.capabilities)
            if not task_validation["valid"]:
                validation_summary["session_valid"] = False
                validation_summary["task_validations"][task.id] = task_validation
                
                # Aggregate missing capabilities
                for tool in task_validation["missing"]["builtin_tools"]:
                    validation_summary["missing_capabilities"]["builtin_tools"].add(tool)
                for tool in task_validation["missing"]["mcp_tools"]:
                    validation_summary["missing_capabilities"]["mcp_tools"].add(tool)
                for resource in task_validation["missing"]["resources"]:
                    validation_summary["missing_capabilities"]["resources"].add(resource)
            
            validation_summary["warnings"].extend(task_validation.get("warnings", []))
        
        # Convert sets to lists for JSON serialization
        validation_summary["missing_capabilities"] = {
            "builtin_tools": list(validation_summary["missing_capabilities"]["builtin_tools"]),
            "mcp_tools": list(validation_summary["missing_capabilities"]["mcp_tools"]),
            "resources": list(validation_summary["missing_capabilities"]["resources"])
        }
        
        # Add specific warning if maestro_iae is missing but needed
        if "maestro_iae" in validation_summary["missing_capabilities"]["mcp_tools"]:
            validation_summary["warnings"].append(
                "CRITICAL: maestro_iae is required for computational tasks but not declared. "
                "This tool provides access to 25+ Intelligence Amplification Engines for "
                "mathematics, data analysis, scientific computing, and more."
            )
        
        return validation_summary
    
    def get_capability_usage_report(self) -> Dict[str, Any]:
        """Generate a report of capability usage across the session"""
        usage_report = {
            "declared_capabilities": {
                "builtin_tools": [t.name for t in self.capabilities.built_in_tools],
                "mcp_tools": [t.name for t in self.capabilities.mcp_tools],
                "resources": [r.name for r in self.capabilities.user_resources]
            },
            "used_capabilities": {
                "builtin_tools": set(),
                "mcp_tools": set(),
                "resources": set()
            },
            "unused_capabilities": {
                "builtin_tools": [],
                "mcp_tools": [],
                "resources": []
            },
            "usage_by_phase": {}
        }
        
        # Aggregate usage from workflow states
        for state in self.workflow_states:
            phase = state.current_phase.value
            if phase not in usage_report["usage_by_phase"]:
                usage_report["usage_by_phase"][phase] = {
                    "builtin_tools": [],
                    "mcp_tools": [],
                    "resources": []
                }
            
            for tool in state.capability_usage.get("builtin_tools_used", []):
                usage_report["used_capabilities"]["builtin_tools"].add(tool)
                usage_report["usage_by_phase"][phase]["builtin_tools"].append(tool)
            
            for tool in state.capability_usage.get("mcp_tools_used", []):
                usage_report["used_capabilities"]["mcp_tools"].add(tool)
                usage_report["usage_by_phase"][phase]["mcp_tools"].append(tool)
            
            for resource in state.capability_usage.get("resources_accessed", []):
                usage_report["used_capabilities"]["resources"].add(resource)
                usage_report["usage_by_phase"][phase]["resources"].append(resource)
        
        # Calculate unused capabilities
        for tool in usage_report["declared_capabilities"]["builtin_tools"]:
            if tool not in usage_report["used_capabilities"]["builtin_tools"]:
                usage_report["unused_capabilities"]["builtin_tools"].append(tool)
        
        for tool in usage_report["declared_capabilities"]["mcp_tools"]:
            if tool not in usage_report["used_capabilities"]["mcp_tools"]:
                usage_report["unused_capabilities"]["mcp_tools"].append(tool)
        
        for resource in usage_report["declared_capabilities"]["resources"]:
            if resource not in usage_report["used_capabilities"]["resources"]:
                usage_report["unused_capabilities"]["resources"].append(resource)
        
        # Convert sets to lists
        usage_report["used_capabilities"] = {
            "builtin_tools": list(usage_report["used_capabilities"]["builtin_tools"]),
            "mcp_tools": list(usage_report["used_capabilities"]["mcp_tools"]),
            "resources": list(usage_report["used_capabilities"]["resources"])
        }
        
        return usage_report

# Validation models for new actions
class FrameworkCreationRequest(BaseModel):
    """Request model for creating conceptual frameworks"""
    framework_type: ConceptualFrameworkType
    name: str
    description: str
    structure: Dict[str, Any]
    task_nodes: Optional[List[TaskNode]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    optimization_rules: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkflowStateUpdate(BaseModel):
    """Request model for updating workflow state"""
    phase: WorkflowPhase
    current_step: Optional[str] = None
    completed_step: Optional[str] = None
    active_framework_ids: Optional[List[str]] = None
    execution_context: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

class KnowledgeUpdate(BaseModel):
    """Request model for adding session knowledge"""
    knowledge_type: KnowledgeType
    subject: str
    context: Dict[str, Any]
    insights: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    applicable_scenarios: Optional[List[str]] = None

class TaskDecomposition(BaseModel):
    """Request model for task decomposition"""
    parent_task_id: str
    subtasks: List[Dict[str, Any]]  # Each dict contains task details
    decomposition_strategy: Optional[str] = None
    dependencies: Optional[List[Dict[str, str]]] = None  # parent_id -> child_id mappings 