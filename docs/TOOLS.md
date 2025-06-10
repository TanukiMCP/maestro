# Maestro MCP Server: Tool Registry

## Built-in Tools

### 1. `maestro_orchestrate` (Primary Tool)
- **Description**: Unified workflow orchestration and collaboration handler for the MAESTRO Protocol. Consolidates task orchestration and user collaboration into a single robust tool with comprehensive session management.
- **Parameters**:
  - `task_description` (string, optional): Description of the task to orchestrate (required for new workflows)
  - `available_tools` (array, optional): List of available tools for workflow execution
  - `context_info` (object, optional): Additional context and configuration for workflow execution
  - `workflow_session_id` (string, optional): Existing workflow session ID to continue or None for new workflow
  - `user_response` (any, optional): User response data for collaboration (when operation_mode is "collaborate")
  - `operation_mode` (string, optional): Either "orchestrate" (default) or "collaborate"
- **Usage Examples**:
  - New workflow: `{"task_description": "Create a web app", "available_tools": [...], "operation_mode": "orchestrate"}`
  - Continue workflow: `{"workflow_session_id": "session_123", "operation_mode": "orchestrate"}`
  - Handle collaboration: `{"user_response": {...}, "workflow_session_id": "session_123", "operation_mode": "collaborate"}`

### 2. `maestro_iae`
- **Description**: Invokes a specific capability from an Intelligence Amplification Engine (IAE) using the MCP-native registry and meta-reasoning logic.
- **Parameters**:
  - `engine_name` (string, required): Name of the IAE engine to invoke
  - `method_name` (string, required): Method to call on the engine
  - `parameters` (object, required): Parameters to pass to the method

### 3. `maestro_web`
- **Description**: Unified web tool for LLM-driven research. Supports only web search (no scraping).
- **Parameters**:
  - `operation` (string, required): Operation to perform (only 'search' supported)
  - `query_or_url` (string, required): Search query string
  - `search_engine` (string, optional): Search engine to use (default: duckduckgo)
  - `num_results` (integer, optional): Number of results to return (default: 5)

### 4. `maestro_execute`
- **Description**: Executes a block of code in a specified language within a secure sandbox.
- **Parameters**:
  - `code` (string, required): The source code to execute
  - `language` (string, required): Programming language (e.g., 'python', 'javascript', 'bash')
  - `timeout` (integer, optional): Execution timeout in seconds (default: 60)

### 5. `maestro_error_handler`
- **Description**: Analyzes an error and provides a structured response for recovery.
- **Parameters**:
  - `error_message` (string, required): The error message that occurred
  - `context` (object, required): Context in which the error occurred

---

---

## Migration Guide

### Collaboration workflows:
```javascript
// Use maestro_orchestrate with collaboration mode
{"user_response": {...}, "workflow_session_id": "session_id", "operation_mode": "collaborate"}
```

### Enhanced orchestration (new functionality):
```javascript
// Standard orchestration (default)
{"task_description": "Create app", "available_tools": [...]}

// Explicit orchestration mode
{"task_description": "Create app", "available_tools": [...], "operation_mode": "orchestrate"}

// Collaboration mode  
{"user_response": {...}, "workflow_session_id": "session_123", "operation_mode": "collaborate"}
```

---

## Extending Maestro

- Add new tools in `src/maestro/tools.py` using the FastMCP `@tool` decorator or as async functions
- Register new engines in `src/engines/`
- See code comments for extension points 