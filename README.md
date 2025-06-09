# Maestro MCP Server

**Meta-Agent Orchestration & Intelligence Amplification Platform**

---

## Overview

Maestro MCP Server is a production-grade, backend-only orchestration and intelligence amplification server. It implements the Model Context Protocol (MCP) to provide a suite of agentic tools for advanced reasoning, workflow automation, and multi-agent collaboration.

- **Streamable HTTP & stdio**: FastMCP-powered, supports both HTTP and stdio transports
- **Smithery.ai Ready**: Instant tool discovery, health checks, and container deployment
- **Extensible**: Register your own tools, engines, and workflows
- **Production Quality**: Robust error handling, logging, and configuration

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server (HTTP)
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 3. Run the server (stdio/MCP)
```bash
python src/main.py --stdio
```

### 4. Docker (Smithery/Prod)
```bash
docker build -t maestro-mcp .
docker run -p 8000:8000 maestro-mcp
```

---

## Features

- **Meta-Orchestration**: Automated, multi-phase workflow generation
- **Intelligence Amplification**: Integrates multiple reasoning engines
- **Tool Registry**: Discover and invoke tools instantly
- **Error Handling**: Structured error analysis and recovery
- **Collaboration**: User-in-the-loop and agent collaboration support

---

## Built-in Tools

| Tool                        | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| `maestro_orchestrate`       | Orchestrate complex tasks and workflows                          |
| `maestro_iae`               | Invoke Intelligence Amplification Engines                        |
| `maestro_web`               | LLM-driven web search                                            |
| `maestro_execute`           | Secure code execution sandbox                                    |
| `maestro_error_handler`     | Error analysis and recovery                                      |
| `maestro_collaboration_response` | Handle user/agent collaboration steps                      |

See [`static_tools_dict.py`](static_tools_dict.py) for full parameter details.

---

## Usage Examples

### Orchestrate a Task (HTTP)
Send a POST to `/mcp` with your task and tool list:
```json
{
  "tool": "maestro_orchestrate",
  "parameters": {
    "task_description": "Summarize the latest AI research",
    "available_tools": ["maestro_web", "maestro_iae"]
  }
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Configuration

- Edit environment variables or `src/maestro/config.py` for server, engine, and logging settings
- See `smithery.yaml` for Smithery deployment config

---

## Deployment

- **Smithery.ai**: Push to Git, connect, and deploy via dashboard
- **Docker**: Use provided Dockerfile for containerized deployment
- **Local**: Run with Python or Uvicorn

---

## Extending Maestro

- Add new tools in `src/maestro/tools.py` or as plugins
- Register engines in `src/engines/`
- See code comments for extension points

---

## License & Support

- Non-Commercial License (see LICENSE)
- Commercial use: contact tanukimcp@gmail.com
- Issues/support: open a GitHub issue 