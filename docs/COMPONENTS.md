# TanukiMCP Maestro: Components

This document provides detailed information about the major components of the TanukiMCP Maestro server and how they interact.

## Core Components

### FastMCP Application

**Location**: `src/app_factory.py`

The FastMCP application is the container for all MCP tools and provides the HTTP interface for LLM clients. Key aspects include:

- Static tool registration for fast discovery
- Health check endpoint for container orchestration
- Environment-aware configuration loading
- Production vs. development mode handling

```python
# Tool registration in app_factory.py
maestro_tools = [
    maestro_orchestrate,
    maestro_iae,
    maestro_web,
    maestro_execute,
    maestro_error_handler
]
```

### Session Manager

**Location**: `src/maestro/tools.py` (maestro_orchestrate)

The Session Manager provides state management capabilities for complex multi-step workflows. It enables:

- Creation and tracking of sessions
- Capability declaration and management
- Task addition and execution
- Task validation and completion tracking
- Framework creation and knowledge management

Key internal functions:
- `_load_current_session()`: Loads session from disk
- `_save_current_session()`: Persists session to disk
- `_create_new_session()`: Initializes a new session
- `_get_relevant_capabilities_for_task()`: Maps tasks to relevant tools
- `_suggest_next_actions()`: Provides LLM guidance on next steps

### Intelligence Amplification Engine (IAE)

**Location**: `src/maestro/tools.py` (maestro_iae), `src/maestro/maestro_iae.py`, `src/maestro/iae_discovery.py`

The IAE component provides specialized reasoning engines to extend LLM capabilities. Key features include:

- Registry of specialized engines
- Dynamic engine discovery and initialization
- Method execution with parameter validation
- Result processing and formatting

The IAE system follows a registry pattern:
```python
# From iae_discovery.py
class IAERegistry:
    """Registry for Intelligence Amplification Engines"""
    
    def __init__(self):
        self.engines = {}
        
    def register(self, engine_id, engine_class):
        """Register an engine with the registry"""
        self.engines[engine_id] = engine_class
```

### Web Research System

**Location**: `src/maestro/tools.py` (maestro_web), `src/maestro/web.py`

The Web Research system provides LLMs with access to current information from the web. It includes:

- Search engine integration (primarily DuckDuckGo)
- Query processing and result formatting
- Rate limiting and security controls
- Temporal awareness integration

```python
# From tools.py
async def maestro_web(
    operation: str,
    query_or_url: str,
    search_engine: str = "duckduckgo",
    num_results: int = 5,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Unified web tool for LLM-driven research. Supports only web search (no scraping).
    """
```

### Code Execution Engine

**Location**: `src/maestro/tools.py` (maestro_execute)

The Code Execution Engine provides a sandboxed environment for running code in various languages. Features include:

- Support for Python, JavaScript, and Bash
- Execution timeouts and resource limits
- Stdout/stderr capture and processing
- Security controls and isolation

```python
# Command execution in maestro_execute
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
        # Process results...
```

### Error Handler

**Location**: `src/maestro/tools.py` (maestro_error_handler)

The Error Handler provides structured analysis of errors and recovery suggestions. It includes:

- Error type classification
- Severity assessment
- Root cause analysis
- Recovery suggestions

## Data Models

**Location**: `src/maestro/session_models.py`

### Core Models

#### Session

The primary container for orchestration state:

```python
class Session(BaseModel):
    id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")
    session_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Original fields
    tasks: List[Task] = []
    capabilities: EnvironmentCapabilities = Field(default_factory=EnvironmentCapabilities)
    environment_context: Optional[Dict] = None
    
    # Advanced orchestration fields
    conceptual_frameworks: List[ConceptualFramework] = []
    workflow_states: List[WorkflowState] = []
    current_workflow_state: Optional[WorkflowState] = None
    session_knowledge: List[SessionKnowledge] = []
```

#### Task

Represents individual tasks with validation and evidence tracking:

```python
class Task(BaseModel):
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4()}")
    description: str
    status: str = "[ ]"  # "[ ]" or "[X]"
    validation_required: bool = False
    validation_criteria: List[str] = []
    evidence: List[dict] = []
    execution_started: bool = False
    execution_evidence: List[str] = []
    # Tool suggestions
    suggested_builtin_tools: List[str] = []
    suggested_mcp_tools: List[str] = []
    suggested_resources: List[str] = []
    # Advanced fields
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = []
    # ...
```

#### EnvironmentCapabilities

Tracks available tools and resources:

```python
class EnvironmentCapabilities(BaseModel):
    built_in_tools: List[BuiltInTool] = []
    mcp_tools: List[MCPTool] = []
    user_resources: List[UserResource] = []
```

### Advanced Models

#### ConceptualFramework

Represents frameworks for self-directed orchestration:

```python
class ConceptualFramework(BaseModel):
    id: str = Field(default_factory=lambda: f"framework_{uuid.uuid4()}")
    type: ConceptualFrameworkType
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    structure: Dict[str, Any]
    task_nodes: List[TaskNode] = []
    relationships: List[Dict[str, Any]] = []
    optimization_rules: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
```

#### WorkflowState

Tracks the current state of workflow execution:

```python
class WorkflowState(BaseModel):
    id: str = Field(default_factory=lambda: f"state_{uuid.uuid4()}")
    current_phase: WorkflowPhase
    current_step: Optional[str] = None
    completed_steps: List[str] = []
    active_frameworks: List[str] = []
    execution_context: Dict[str, Any] = {}
    decision_history: List[Dict[str, Any]] = []
    performance_metrics: Dict[str, Any] = {}
    capability_usage: Dict[str, List[str]] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

#### SessionKnowledge

Represents knowledge learned during session execution:

```python
class SessionKnowledge(BaseModel):
    id: str = Field(default_factory=lambda: f"knowledge_{uuid.uuid4()}")
    type: KnowledgeType
    subject: str
    context: Dict[str, Any]
    insights: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    applicable_scenarios: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = 0
```

## Configuration System

**Location**: `src/maestro/config.py`, `smithery.yaml`

The configuration system uses a hierarchical approach with:

- YAML-based configuration for Smithery deployment
- Environment variable overrides
- Pydantic models for validation

Main configuration sections:
- **Server**: Host, port, workers, timeout, CORS settings
- **Security**: API keys, rate limiting, allowed origins
- **Engine**: Mode, task limits, memory limits, GPU settings
- **Logging**: Log levels, file paths, rotation settings

## Deployment Components

**Location**: `Dockerfile`, `smithery.yaml`

The deployment components define how Maestro is packaged and deployed on Smithery.ai:

- Container-based runtime
- Resource allocation (512Mi memory, 0.5 CPU)
- Scaling configuration (1-10 instances)
- Port and environment configuration 