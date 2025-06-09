# Maestro MCP Server: Tool Registry

## Built-in Tools

### 1. `maestro_orchestrate`
- **Description**: Orchestrates a complex task by generating and executing a dynamic workflow using a suite of available tools.
- **Parameters**:
  - `task_description` (string, required): The task to orchestrate
  - `available_tools` (array, required): List of available tools for the workflow
  - `context_info` (object, optional): Additional context information

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

### 6. `maestro_collaboration_response`
- **Description**: Handles a response from a user during a collaborative workflow step.
- **Parameters**:
  - `user_response` (any, required): The data received from the user
  - `original_request` (object, required): The original request that prompted the collaboration

---

## Extending Maestro

- Add new tools in `src/maestro/tools.py` using the FastMCP `@tool` decorator or as async functions
- Register new engines in `src/engines/`
- See code comments for extension points 