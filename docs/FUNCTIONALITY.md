# TanukiMCP Maestro: Functionality

This document details the functionality provided by the TanukiMCP Maestro server, focusing on the capabilities of each tool and how they can be used effectively.

## Overview of Available Tools

TanukiMCP Maestro exposes the following tools via the MCP protocol:

1. **maestro_orchestrate**: Session management for complex multi-step tasks
2. **maestro_iae**: Intelligence Amplification Engine for specialized reasoning
3. **maestro_web**: Web research and information gathering
4. **maestro_execute**: Secure code execution in multiple languages
5. **maestro_error_handler**: Error analysis and recovery assistance

## Detailed Tool Functionality

### maestro_orchestrate

The orchestration tool provides session management for complex, multi-step workflows. It enables LLMs to create and manage tasks, track progress, and coordinate tool usage.

#### Actions

The tool supports multiple actions that can be specified via the `action` parameter:

**Basic Session Management:**
- `create_session`: Initialize a new orchestration session
- `declare_capabilities`: Register available tools and resources
- `add_task`: Add a task to the current session
- `execute_next`: Prepare for execution of the next pending task
- `validate_task`: Validate task completion with evidence
- `mark_complete`: Mark a task as completed
- `get_status`: Get the current session status

**Advanced Orchestration:**
- `create_framework`: Create a conceptual framework for orchestration
- `update_workflow_state`: Update the current workflow state
- `add_knowledge`: Add learned knowledge to the session
- `decompose_task`: Decompose a task into subtasks
- `get_relevant_knowledge`: Retrieve relevant session knowledge
- `get_task_hierarchy`: Get the hierarchical task structure
- `validate_capabilities`: Validate that required capabilities are available
- `suggest_capabilities`: Get capability suggestions for a task or framework

#### Example Usage

```python
# Create a new session
result = await maestro_orchestrate(
    action="create_session", 
    session_name="Data Analysis Project"
)

# Declare available capabilities
result = await maestro_orchestrate(
    action="declare_capabilities",
    builtin_tools=["edit_file", "run_terminal_cmd", "read_file"],
    mcp_tools=["maestro_iae", "maestro_web", "maestro_execute"],
    user_resources=["documentation:Python: Python documentation", "dataset:Customer Data: Customer transaction data"]
)

# Add a task
result = await maestro_orchestrate(
    action="add_task",
    task_description="Analyze customer transaction data for seasonal patterns",
    validation_criteria=["Statistical significance test performed", "Visualizations created"]
)

# Execute the next task
result = await maestro_orchestrate(action="execute_next")

# Mark a task as complete
result = await maestro_orchestrate(
    action="mark_complete",
    execution_evidence="Performed time series analysis with seasonal decomposition using Python's statsmodels. Created visualizations showing clear seasonal patterns in Q4 with p-value < 0.01."
)
```

### maestro_iae

The Intelligence Amplification Engine provides access to specialized reasoning engines that extend the computational and analytical capabilities of LLMs.

#### Available Engines

The IAE system supports multiple specialized engines:

- **mathematics**: Mathematical computation and analysis
- **data_analysis**: Statistical analysis and data processing
- **code_quality**: Code analysis and quality assessment
- **language_analysis**: NLP and grammar analysis

#### Parameters

- `engine_name`: The name of the engine to use
- `method_name`: The method to call on the engine
- `parameters`: Parameters to pass to the method

#### Example Usage

```python
# Solve a mathematical equation
result = await maestro_iae(
    engine_name="mathematics",
    method_name="solve_equation",
    parameters={"equation": "x^2 + 5x + 6 = 0"}
)

# Perform statistical analysis
result = await maestro_iae(
    engine_name="data_analysis",
    method_name="descriptive_statistics",
    parameters={"data": [12, 15, 18, 22, 30, 35, 42]}
)
```

### maestro_web

The Web Research tool enables LLMs to search for information on the web, providing access to current data beyond the model's training cutoff.

#### Supported Operations

Currently, only the `search` operation is supported, which performs a web search using a specified search engine.

#### Parameters

- `operation`: The operation to perform (currently only "search")
- `query_or_url`: The search query string
- `search_engine`: The search engine to use (default: "duckduckgo")
- `num_results`: Number of results to return (default: 5)

#### Example Usage

```python
# Search for current information
result = await maestro_web(
    operation="search",
    query_or_url="latest advances in quantum computing 2024",
    num_results=10
)
```

### maestro_execute

The Code Execution tool provides a secure environment for running code in various programming languages.

#### Supported Languages

- **python**: For data analysis, scientific computing, and general scripting
- **javascript**: For web-related tasks and calculations
- **bash**: For system operations and file processing

#### Parameters

- `code`: The source code to execute
- `language`: The programming language to use
- `timeout`: Execution timeout in seconds (default: 60)

#### Example Usage

```python
# Execute Python code
result = await maestro_execute(
    code="""
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 100)
y = np.sin(x)
print(f"Max value: {y.max()}")
""",
    language="python",
    timeout=30
)

# Execute JavaScript code
result = await maestro_execute(
    code="""
const data = [1, 2, 3, 4, 5];
const sum = data.reduce((a, b) => a + b, 0);
console.log(`Sum: ${sum}`);
""",
    language="javascript"
)

# Execute Bash commands
result = await maestro_execute(
    code="ls -la | grep '.py'",
    language="bash"
)
```

### maestro_error_handler

The Error Handler tool provides structured analysis of errors and recovery suggestions.

#### Parameters

- `error_message`: The error message that occurred
- `context`: Context in which the error occurred

#### Example Usage

```python
# Analyze an error
result = await maestro_error_handler(
    error_message="ModuleNotFoundError: No module named 'pandas'",
    context={
        "tool": "maestro_execute",
        "language": "python",
        "code_snippet": "import pandas as pd"
    }
)
```

## Advanced Functionality

### Temporal Awareness

The system includes temporal awareness to help LLMs understand when information might be outdated:

```python
# From tools.py
def _create_temporal_awareness_context() -> Dict[str, Any]:
    """Create temporal awareness context with current date/time and research suggestions."""
    current_time = datetime.datetime.now()
    current_utc = datetime.datetime.utcnow()
    
    # Estimate when information might be outdated
    knowledge_cutoff_estimate = datetime.datetime(2024, 4, 1)
    months_since_cutoff = (current_time.year - knowledge_cutoff_estimate.year) * 12 + \
                         (current_time.month - knowledge_cutoff_estimate.month)
    
    suggests_web_research = months_since_cutoff > 3
    
    return {
        "temporal_awareness": {
            "current_date": current_time.strftime("%Y-%m-%d"),
            "current_time": current_time.strftime("%H:%M:%S"),
            # Additional time-related fields...
        }
    }
```

### Task Decomposition

Complex tasks can be decomposed into subtasks using the hierarchical task management system:

```python
# Decompose a task into subtasks
result = await maestro_orchestrate(
    action="decompose_task",
    parent_task_id="task_123abc",
    subtasks=[
        {
            "description": "Collect and prepare data",
            "phase": "planning",
            "complexity": 0.4
        },
        {
            "description": "Perform statistical analysis",
            "phase": "execution",
            "complexity": 0.7,
            "validation_criteria": ["p-value < 0.05", "R-squared > 0.7"]
        },
        {
            "description": "Generate visualizations",
            "phase": "execution",
            "complexity": 0.5
        }
    ],
    decomposition_strategy="sequential"
)
```

### Knowledge Management

The system can store and retrieve knowledge gained during task execution:

```python
# Add knowledge about a tool's effectiveness
result = await maestro_orchestrate(
    action="add_knowledge",
    knowledge_type="tool_effectiveness",
    knowledge_subject="Data analysis with maestro_iae",
    knowledge_insights=[
        "Statistical analysis is faster with the data_analysis engine than with Python code execution",
        "Complex visualizations still require maestro_execute with matplotlib"
    ],
    knowledge_confidence=0.85
)

# Retrieve relevant knowledge
result = await maestro_orchestrate(
    action="get_relevant_knowledge",
    knowledge_subject="Statistical analysis methods"
)
```

### Conceptual Frameworks

Advanced orchestration is supported through conceptual frameworks:

```python
# Create a workflow optimization framework
result = await maestro_orchestrate(
    action="create_framework",
    framework_type="workflow_optimization",
    framework_name="Data Pipeline Optimization",
    framework_structure={
        "description": "Framework for optimizing data processing pipelines",
        "stages": ["data_collection", "preprocessing", "analysis", "visualization"],
        "optimization_goals": ["reduce_processing_time", "improve_accuracy"],
        "metrics": ["execution_time", "memory_usage", "accuracy"]
    }
)
```

## Security and Error Handling

All tools include robust error handling and security measures:

1. **Sandboxed Execution**: Code execution is performed in isolated environments
2. **Timeouts**: All operations have configurable timeouts to prevent hanging
3. **Input Validation**: Parameters are validated using Pydantic models
4. **Rate Limiting**: Web operations have configurable rate limiting
5. **Error Boundaries**: Structured error handling prevents cascading failures

## Compatibility

The server follows the MCP protocol specification and is compatible with any LLM client that implements the MCP standard, including:

- Claude in Anthropic Console
- Claude in AWS Bedrock
- Custom MCP-compatible clients 