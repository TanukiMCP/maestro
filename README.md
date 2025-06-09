# TanukiMCP Maestro

A local MCP server with advanced orchestration tools, designed to be called by external agentic IDEs like Cursor.

## Quick Start

1.  **Install dependencies:**
```bash
pip install -r requirements.txt
```

2.  **Run the server:**
    This project uses a `src` layout, so the correct way to run it is via the `run.py` script in the root directory.
```bash
    python run.py
```

The server will run on `http://localhost:8000` by default.

## Integration with Cursor or other MCP clients

To use with Cursor or other MCP clients, add the following to your `mcp.json` file:

```json
{
  "mcpServers": {
    "maestro": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## Available Tools

The server exposes a suite of powerful, backend-only tools for agentic workflows:

-   `maestro_orchestrate`: The main entrypoint. It receives a task and a list of available tools from the host (e.g., Cursor) and generates a complete execution plan.
-   `maestro_iae`: Intelligence Amplification Engine for complex analysis.
-   `maestro_search`: Advanced web search capabilities.
-   `maestro_execute`: Secure code execution.
-   `maestro_error_handler`: Adaptive error handling.
-   `maestro_collaboration_response`: Manages collaborative steps in a workflow.

## Architecture

The server is built using `FastMCP`, a Python framework for creating MCP-compliant servers. It is implemented as a pure `ToolServer`, meaning its sole purpose is to expose tools via the Model Context Protocol.

Key principles:

-   **Headless & IDE-Agnostic:** Contains no UI or frontend logic.
-   **External LLM Client:** Does not contain an LLM client. It expects the calling agent/IDE to provide LLM capabilities when invoking tools like `maestro_orchestrate`.
-   **Production Quality:** Built with clear separation of concerns, proper error handling, and a compliant architecture.
-   **No Placeholders:** All tools are implemented with real, production-ready logic.

## Available Endpoints

- `/mcp` - MCP protocol endpoint for tools
- `/health` - Health check endpoint
- `/` - Service information

## Troubleshooting

If tools are not showing up in your client:

1. Make sure the server is running and shows tool loading messages
2. Check if the server returns tools by visiting `http://localhost:8000/` in your browser
3. Ensure your mcp.json configuration is correct
4. Restart your client application after server changes
5. Check your network/firewall settings to ensure the port is accessible
