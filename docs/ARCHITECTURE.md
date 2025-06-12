# TanukiMCP Maestro: Architecture

This document outlines the architecture of the TanukiMCP Maestro server, a Model Context Protocol (MCP) implementation designed for intelligent orchestration and task management.

## System Overview

TanukiMCP Maestro is an MCP server that provides advanced orchestration capabilities for Large Language Models (LLMs). It's designed to extend the capabilities of LLMs by providing:

1. **Session-based orchestration** for multi-step workflows
2. **Intelligence amplification** through specialized reasoning engines
3. **Web research capabilities** for up-to-date information
4. **Secure code execution** in sandboxed environments
5. **Error handling and recovery** for robust operations

The system is built to be deployed as a container on Smithery.ai, following the MCP protocol for standardized tool definition and discovery.

## Core Components

### 1. FastMCP Server

The core server is built using the FastMCP framework, which handles the MCP protocol implementation including tool discovery, execution, and streaming responses. The server is configured in `app_factory.py` and exposes an HTTP API endpoint.

### 2. Session Management

The session management system (`maestro_orchestrate`) provides stateful operations for LLMs, allowing for:

- Creation and management of task sessions
- Declaration of available capabilities
- Task tracking and validation
- Workflow state management
- Knowledge acquisition and utilization

Sessions are persisted to disk to maintain state between API calls.

### 3. Intelligence Amplification Engines (IAE)

The IAE system (`maestro_iae`) provides access to specialized reasoning engines that extend the computational and analytical capabilities of LLMs, including:

- Mathematical computation
- Data analysis
- Domain-specific reasoning
- Algorithm execution

### 4. Web Research

The web research component (`maestro_web`) enables LLMs to access current information from the web through search engine integrations, primarily with DuckDuckGo.

### 5. Code Execution

The code execution system (`maestro_execute`) provides a sandboxed environment for running code in various languages:

- Python for data analysis and general scripting
- JavaScript for web-related tasks
- Bash for system operations

### 6. Error Handling

The error analysis system (`maestro_error_handler`) provides structured error analysis and recovery suggestions.

## Data Models

The system uses Pydantic models (defined in `session_models.py`) for data validation and serialization:

### Core Models

- **Session**: The top-level container for orchestration state
- **Task**: Represents individual tasks with validation and evidence tracking
- **EnvironmentCapabilities**: Tracks available tools and resources

### Advanced Orchestration Models

- **ConceptualFramework**: Represents frameworks for self-directed orchestration
- **WorkflowState**: Tracks the current state of workflow execution
- **SessionKnowledge**: Represents knowledge learned during session execution
- **TaskNode**: Represents nodes in task decomposition frameworks

## Architectural Patterns

### 1. Stateful API with Persistence

Unlike simple stateless MCP servers, Maestro maintains session state between calls by persisting data to disk, allowing for complex multi-step workflows.

```
[LLM] → [API Request] → [Load Session] → [Process Action] → [Save Session] → [API Response] → [LLM]
```

### 2. Tool-Based Composition

The system is designed around composable tools that can be mixed and matched based on task requirements:

```
maestro_orchestrate
    ↓
[Session Management]
    ↓
maestro_iae  ←→  maestro_web  ←→  maestro_execute
```

### 3. Temporal Awareness

The system includes temporal awareness to help LLMs understand when information might be outdated and require web research.

## Security Considerations

1. **Sandboxed Execution**: Code execution happens in isolated environments
2. **Rate Limiting**: Configurable rate limiting for API access
3. **Timeout Controls**: All operations have configurable timeouts
4. **Error Boundaries**: Structured error handling to prevent cascading failures

## Deployment Architecture

Maestro is designed to be deployed as a container on Smithery.ai with the following characteristics:

- **Runtime**: Container-based deployment
- **Scaling**: 1-10 instances with CPU-based autoscaling
- **Resources**: 512Mi memory, 0.5 CPU per instance
- **Configuration**: YAML-based configuration for server, security, engine, and logging

## Implementation Details

### FastMCP Integration

The server uses FastMCP for MCP protocol handling, with static tool definitions for fast discovery:

```python
mcp = FastMCP(
    tools=maestro_tools,
    name="Maestro",
    instructions="An MCP server for advanced, backend-only orchestration...",
    timeout=30,
    json_response=True,
    stateless_http=True
)
```

### Session Persistence

Sessions are persisted to disk using JSON serialization:

```python
def _save_current_session(session):
    if session:
        os.makedirs(os.path.dirname(session_state_file), exist_ok=True)
        with open(session_state_file, 'w') as f:
            session_dict = session.model_dump()
            json.dump(session_dict, f, indent=2, default=str)
```

### Tool Discovery

Tools are defined both dynamically in code and statically in `static_tools_dict.py` for fast discovery without full application initialization.

## Extension Points

The architecture allows for extension in several areas:

1. **Additional Tools**: New tools can be added to the `maestro_tools` list in `app_factory.py`
2. **Intelligence Engines**: New IAE engines can be registered in the IAE registry
3. **Workflow Patterns**: New workflow patterns can be added to the orchestration framework 