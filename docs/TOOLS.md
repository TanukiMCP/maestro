# TanukiMCP Maestro: Tools Reference

This document provides detailed reference information for all tools available in the TanukiMCP Maestro server.

## Tool Overview

TanukiMCP Maestro provides the following MCP tools:

| Tool Name | Category | Description |
|-----------|----------|-------------|
| `maestro_orchestrate` | Session Management | Intelligent session management for complex multi-step tasks |
| `maestro_iae` | Intelligence Amplification | Access to specialized reasoning engines |
| `maestro_web` | Research | Web search and information gathering |
| `maestro_execute` | Execution | Secure code execution in multiple languages |
| `maestro_error_handler` | Error Handling | Error analysis and recovery assistance |

## Tool Reference

### maestro_orchestrate

Intelligent session management for complex multi-step tasks following MCP principles. Enables LLMs to self-direct orchestration through conceptual frameworks and knowledge management.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | The action to take (see Actions section) |
| `task_description` | string | No | Description of task to add (for 'add_task') |
| `session_name` | string | No | Name for new session (for 'create_session') |
| `validation_criteria` | array | No | List of criteria for task validation (for 'add_task') |
| `evidence` | string | No | Evidence of task completion (for 'validate_task') |
| `execution_evidence` | string | No | Evidence that execution was performed (for tracking completion) |
| `builtin_tools` | array | No | List of built-in tools available (for 'declare_capabilities') |
| `mcp_tools` | array | No | List of MCP server tools available (for 'declare_capabilities') |
| `user_resources` | array | No | List of user-added resources available (for 'declare_capabilities') |
| `next_action_needed` | boolean | No | Whether more actions are needed (default: true) |
| `framework_type` | string | No | Type of framework (for 'create_framework') |
| `framework_name` | string | No | Name for the conceptual framework (for 'create_framework') |
| `framework_structure` | object | No | Dictionary defining the framework structure (for 'create_framework') |
| `task_nodes` | array | No | List of task nodes for decomposition frameworks (for 'create_framework') |
| `workflow_phase` | string | No | Current workflow phase (for 'update_workflow_state') |
| `workflow_state_update` | object | No | Dictionary with workflow state updates (for 'update_workflow_state') |
| `knowledge_type` | string | No | Type of knowledge (for 'add_knowledge') |
| `knowledge_subject` | string | No | What the knowledge is about (for 'add_knowledge') |
| `knowledge_insights` | array | No | List of insights learned (for 'add_knowledge') |
| `knowledge_confidence` | number | No | Confidence level 0.0-1.0 (for 'add_knowledge') |
| `parent_task_id` | string | No | ID of parent task for decomposition (for 'decompose_task') |
| `subtasks` | array | No | List of subtask definitions (for 'decompose_task') |
| `decomposition_strategy` | string | No | Strategy used for decomposition (for 'decompose_task') |

#### Actions

**Basic Session Management:**
- `create_session`: Create new orchestration session
- `declare_capabilities`: Declare available tools and resources
- `add_task`: Add task to session
- `execute_next`: Execute next pending task
- `validate_task`: Validate task completion
- `mark_complete`: Mark task as completed
- `get_status`: Get session status

**Advanced Orchestration:**
- `create_framework`: Create conceptual framework for orchestration
- `update_workflow_state`: Update current workflow state
- `add_knowledge`: Add learned knowledge to session
- `decompose_task`: Decompose task into subtasks
- `get_relevant_knowledge`: Retrieve relevant session knowledge
- `get_task_hierarchy`: Get hierarchical task structure
- `validate_capabilities`: Validate required capabilities
- `suggest_capabilities`: Get capability suggestions

#### Return Value

Returns a dictionary with the current state, suggested actions, and orchestration context. The exact structure depends on the action performed, but generally includes:

```json
{
  "action": "string",
  "session_id": "string",
  "current_task": {
    "id": "string",
    "description": "string",
    "validation_required": boolean,
    "validation_criteria": ["string"]
  },
  "relevant_capabilities": {
    "builtin_tools": [],
    "mcp_tools": [],
    "resources": []
  },
  "suggested_next_actions": ["string"],
  "completion_guidance": "string",
  "status": "string"
}
```

#### Examples

**Creating a Session:**

```python
result = await maestro_orchestrate(
    action="create_session",
    session_name="Data Analysis Project"
)
```

**Declaring Capabilities:**

```python
result = await maestro_orchestrate(
    action="declare_capabilities",
    builtin_tools=[
        "edit_file: Create and edit files",
        "run_terminal_cmd: Execute terminal commands",
        "web_search: Search the web"
    ],
    mcp_tools=[
        "maestro_iae: Intelligence Amplification Engines",
        "maestro_web: Web research tools",
        "maestro_execute: Code execution"
    ],
    user_resources=[
        "documentation:Python: Python language documentation",
        "dataset:Sales Data: Monthly sales figures"
    ]
)
```

**Creating a Framework:**

```python
result = await maestro_orchestrate(
    action="create_framework",
    framework_type="task_decomposition",
    framework_name="Data Analysis Pipeline",
    framework_structure={
        "description": "Framework for structured data analysis",
        "stages": ["data_cleaning", "exploratory_analysis", "modeling", "visualization"],
        "dependencies": {
            "exploratory_analysis": ["data_cleaning"],
            "modeling": ["exploratory_analysis"],
            "visualization": ["modeling"]
        }
    }
)
```

### maestro_iae

Invokes a specific capability from an Intelligence Amplification Engine (IAE) using the MCP-native registry and meta-reasoning logic.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `engine_name` | string | Yes | Name of the IAE engine to invoke |
| `method_name` | string | Yes | Method to call on the engine |
| `parameters` | object | Yes | Parameters to pass to the method |

#### Available Engines and Methods

**mathematics:**
- `solve_equation`: Solves mathematical equations
- `calculus`: Performs calculus operations
- `linear_algebra`: Performs linear algebra operations

**data_analysis:**
- `descriptive_statistics`: Calculates descriptive statistics
- `hypothesis_testing`: Performs statistical hypothesis tests
- `correlation_analysis`: Analyzes correlations between variables

**code_quality:**
- `analyze_complexity`: Analyzes code complexity
- `suggest_improvements`: Suggests code improvements
- `identify_patterns`: Identifies coding patterns

**language_analysis:**
- `grammar_check`: Checks grammar in text
- `sentiment_analysis`: Analyzes sentiment in text
- `syntax_parsing`: Parses syntax in text

#### Return Value

Returns the result of the engine method call, which varies depending on the engine and method used.

#### Examples

**Solving an Equation:**

```python
result = await maestro_iae(
    engine_name="mathematics",
    method_name="solve_equation",
    parameters={"equation": "x^2 + 5x + 6 = 0"}
)
# Returns: {"solutions": [-2, -3], "steps": [...]}
```

**Performing Statistical Analysis:**

```python
result = await maestro_iae(
    engine_name="data_analysis",
    method_name="descriptive_statistics",
    parameters={"data": [12, 15, 18, 22, 30, 35, 42]}
)
# Returns: {"mean": 24.85, "median": 22, "std_dev": 11.13, ...}
```

### maestro_web

Unified web tool for LLM-driven research. Supports only web search (no scraping).

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `operation` | string | Yes | Operation to perform (only 'search' supported) |
| `query_or_url` | string | Yes | Search query string |
| `search_engine` | string | No | Search engine to use (default: "duckduckgo") |
| `num_results` | integer | No | Number of results to return (default: 5) |

#### Return Value

Returns a dictionary with search results:

```json
{
  "operation": "search",
  "query": "string",
  "search_engine": "string",
  "results": [
    {
      "title": "string",
      "url": "string",
      "snippet": "string"
    }
  ]
}
```

#### Examples

**Performing a Web Search:**

```python
result = await maestro_web(
    operation="search",
    query_or_url="latest advances in quantum computing 2024",
    num_results=10
)
```

### maestro_execute

Executes a block of code in a specified language within a secure sandbox.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `code` | string | Yes | The source code to execute |
| `language` | string | Yes | Programming language (e.g., 'python', 'javascript', 'bash') |
| `timeout` | integer | No | Execution timeout in seconds (default: 60) |

#### Return Value

Returns a dictionary with execution results:

```json
{
  "status": "string",  // "success" or "error"
  "exit_code": number,
  "stdout": "string",
  "stderr": "string"
}
```

#### Examples

**Executing Python Code:**

```python
result = await maestro_execute(
    code="""
import math
result = 0
for i in range(1, 10):
    result += math.sqrt(i)
print(f"Sum of square roots from 1 to 9: {result:.2f}")
""",
    language="python",
    timeout=30
)
```

**Executing JavaScript Code:**

```python
result = await maestro_execute(
    code="""
const fibonacci = n => {
  let a = 0, b = 1;
  for(let i = 0; i < n; i++) {
    [a, b] = [b, a + b];
  }
  console.log(`Fibonacci(${n}) = ${a}`);
  return a;
};
fibonacci(10);
""",
    language="javascript"
)
```

### maestro_error_handler

Analyzes an error and provides a structured response for recovery.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `error_message` | string | Yes | The error message that occurred |
| `context` | object | Yes | Context in which the error occurred |

#### Return Value

Returns a dictionary with error analysis:

```json
{
  "original_error": "string",
  "original_context": {},
  "analysis": {
    "error_type": "string",
    "severity": "string",
    "possible_root_cause": "string",
    "recovery_suggestion": "string"
  }
}
```

#### Examples

**Analyzing a Python Import Error:**

```python
result = await maestro_error_handler(
    error_message="ModuleNotFoundError: No module named 'pandas'",
    context={
        "tool": "maestro_execute",
        "language": "python",
        "code_snippet": "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3]})"
    }
)
```

**Analyzing a Timeout Error:**

```python
result = await maestro_error_handler(
    error_message="Execution timed out after 30 seconds",
    context={
        "tool": "maestro_execute",
        "language": "python",
        "code_snippet": "while True: pass"
    }
)
```

## Tool Categories

Tools are organized into the following categories:

| Category | Tools |
|----------|-------|
| Session Management | maestro_orchestrate |
| Intelligence Amplification | maestro_iae |
| Research | maestro_web |
| Execution | maestro_execute |
| Error Handling | maestro_error_handler |

## Tool Discovery

The Maestro server supports fast tool discovery (<100ms) through a static tool dictionary defined in `static_tools_dict.py`. This enables the server to quickly respond to tool discovery requests without initializing the full application.

## Security Considerations

- **Code Execution**: The `maestro_execute` tool runs code in a sandboxed environment with timeouts and resource limits.
- **Web Access**: The `maestro_web` tool is limited to search operations only, with no direct web scraping.
- **Rate Limiting**: Web operations have configurable rate limiting to prevent abuse.
- **Input Validation**: All tool parameters are validated using Pydantic models.

## Version Compatibility

The current tools are compatible with the MCP protocol version `mcp-2024-11-05`.

## Known Limitations

- The `maestro_web` tool currently only supports the `search` operation (no scraping).
- The `maestro_execute` tool supports only Python, JavaScript, and Bash languages.
- The `maestro_iae` engines are limited to the pre-registered capabilities.
- Task sessions are stored on disk and not shared across server instances. 