# TanukiMCP Maestro - Deployment Notes

## ✅ Server Status

The TanukiMCP Maestro server is now properly configured and working:

- **Protocol**: MCP 2024-11-05 
- **Transport**: Server-Sent Events (SSE)
- **Port**: 8000
- **Tools**: 11 production-ready tools
- **Discovery Time**: <100ms (instant via static tool definitions)

## 🔧 Working Endpoints

### Health Check
- **URL**: `http://localhost:8000/health`
- **Method**: GET
- **Response**: `ok` (200 OK)
- **Purpose**: Container health checks and load balancer monitoring

### Tools List (Debug)
- **URL**: `http://localhost:8000/tools`
- **Method**: GET  
- **Response**: JSON with all 11 tools
- **Purpose**: Manual verification of tool registration

### Debug Info
- **URL**: `http://localhost:8000/debug`
- **Method**: GET
- **Response**: Server configuration details
- **Purpose**: Deployment troubleshooting

### MCP Protocol
- **URL**: `http://localhost:8000/mcp`
- **Transport**: Streamable HTTP (MCP 2.0 Standard)
- **Purpose**: Full MCP communication (requires session)

### Smithery Tool Discovery Endpoint
- **URL**: `http://localhost:8000/tools/list`
- **Method**: GET
- **Response**: JSON list of tool names
- **Purpose**: Instant, sessionless tool scanning for Smithery.ai

## ��️ Registered Tools

All 11 tools are properly registered and available:

1. `maestro_orchestrate` - Meta-reasoning orchestration
2. `maestro_collaboration_response` - User collaboration handler  
3. `maestro_iae_discovery` - Intelligence Amplification Engine discovery
4. `maestro_tool_selection` - Intelligent tool selection
5. `maestro_iae` - Intelligence Amplification Engine
6. `get_available_engines` - Available engines listing
7. `maestro_search` - Enhanced web search
8. `maestro_scrape` - Intelligent web scraping
9. `maestro_execute` - Secure code execution
10. `maestro_temporal_context` - Temporal reasoning
11. `maestro_error_handler` - Intelligent error handling

## 🚀 Deployment Status

### ✅ Working
- Docker container builds successfully
- Server starts without errors
- All HTTP endpoints respond correctly (including `/tools/list` for Smithery)
- Tool registration completes instantly (<100ms)
- Health checks pass

### ✅ Smithery.ai Integration Solution

The "TypeError: fetch failed" error from Smithery.ai should now be resolved.

- **Correct Transport**: The server uses `streamable-http` on `/mcp` as recommended by Smithery for full MCP communication.
- **Dedicated Discovery Endpoint**: A simple GET endpoint at `/tools/list` provides an instant, sessionless JSON list of tool names specifically for Smithery's tool scanner. This meets the requirement: "Tool Lists: The `/tools/list` endpoint must be accessible without API keys or configurations."

### 🔍 Troubleshooting Steps

1.  **Verify endpoint accessibility**: `/health`, `/tools`, `/debug`, and `/tools/list` respond correctly. `/mcp` responds as expected for streamable HTTP (requires session for most calls).
2.  **Check transport compatibility**: `streamable-http` is the recommended transport.
3.  **Tool discovery speed**: <100ms via static tool definitions and the dedicated `/tools/list` endpoint. ✅
4.  **Container health**: Health check endpoint working. ✅

### 📋 Recommendations

Configuration for Smithery.ai compatibility:

1.  **Primary MCP Endpoint**: `https://<your-server-url>/mcp` (using `streamable-http`)
2.  **Smithery Tool Discovery URL**: `https://<your-server-url>/tools/list` (simple GET for JSON tool list)

## 🧪 Local Testing

Run the test script to verify all endpoints:

```bash
python test_server_http.py
```

Expected results:
- `/health`: 200 OK
- `/tools`: 200 OK with tool list (debug endpoint)
- `/debug`: 200 OK with config
- `/tools/list`: 200 OK with JSON tool list (for Smithery scanner)
- `/mcp`: 400 Bad Request "Missing session ID" (expected for simple GET/POST without session to streamable HTTP endpoint)

## 🐳 Container Deployment

The server is ready for container deployment with:
- Proper port exposure (8000)
- Health check endpoint
- `/tools/list` endpoint for Smithery discovery
- `/mcp` endpoint for standard MCP communication
- Non-root user for security
- Optimized dependency loading
- Production-ready configuration 

## Deployment Strategy for Smithery.ai

**Final Configuration (After multiple attempts):**

1.  **Primary Transport**: The TanukiMCP Maestro server is configured to use **`stdio` transport exclusively**.
    *   **File**: `server.py`
    *   **Rationale**: Smithery.ai documentation states: "If you deploy a STDIO server, we will wrap your server and proxy it over HTTP for you." This is the most robust way to ensure compatibility, removing complexities of HTTP session management, specific endpoint requirements for scanning, or potential mismatches with `fastmcp`'s `streamable-http` transport and Smithery's scanner/proxy.
    *   The `FastMCP` instance in `server.py` is started with `mcp.run(transport="stdio")`.
    *   All custom HTTP endpoints (`/health`, `/tools`, `/debug`, `/tools/list`) have been **removed** from `server.py` as they are not relevant when relying on Smithery's STDIO wrapping.

2.  **Tool Discovery**: Smithery.ai will perform tool discovery by communicating with the `stdio` server through its WebSocket proxy. The standard MCP `tools/list` method will be used by Smithery's scanner.
    *   Our server uses `STATIC_TOOLS_DICT` for instant tool loading, so discovery via the proxied `stdio` connection will be fast.

3.  **Dockerfile**: `Dockerfile` remains the same, ensuring Python 3.11-slim, copying necessary files, installing dependencies from `requirements.txt`, and running `server.py`.

4.  **`requirements.txt`**: Contains all necessary dependencies, including `fastmcp>=2.6.0`.

**Why this approach?**

*   **Simplicity**: The server's only job is to be a compliant `stdio` MCP server. Smithery takes on the responsibility of web exposure.
*   **Robustness**: Avoids potential issues with HTTP header mismatches, session handling differences, or specific path expectations for tool scanning between `fastmcp` and Smithery's infrastructure.
*   **Direct Compliance**: Directly follows Smithery's documented capability for `stdio` servers.

**Previous Failed Approaches & Learnings:**

*   **Initial `streamable-http` with `/mcp` and custom `/tools/list`**: Despite a dedicated `/tools/list` GET endpoint for easy scanning, Smithery still reported "TypeError: fetch failed". This suggests that either their scanner was still trying to use `/mcp` in a way that was incompatible with `fastmcp`'s session requirements for that path, or there were other underlying HTTP/network issues in their environment.
*   **Various HTTP transport configurations (`sse`, `ws` on different paths)**: These were based on misunderstandings of Smithery's requirements and `fastmcp`'s capabilities. `fastmcp`'s `run()` method doesn't support a generic "http" transport, and SSE is not supported by Smithery for hosting.
*   **Slow tool discovery**: Initially resolved by static tool loading.

This `stdio`-first approach is the most direct and simplified configuration to meet Smithery.ai's deployment model.

**To Test Locally (Simulating STDIO interaction - conceptual):**

Since the server now runs in `stdio` mode, you cannot directly test it with `curl` or a web browser. Local testing would involve running `python server.py` and then using a separate MCP client script that communicates over `stdio`.

Example of how one might interact with an STDIO MCP server (from `mcp-py` docs, adapted):

```python
# client_stdio.py (conceptual - NOT PART OF THIS REPO)
import asyncio
from mcp.client.stdio import StdioClient

async def main():
    client = StdioClient(command=["python", "server.py"]) # Path to your server script
    async with client.connect():
        print("Attempting to list tools...")
        tools = await client.tools.list()
        print(f"Tools from STDIO server: {tools}")

if __name__ == "__main__":
    asyncio.run(main())
```

This type of client is how Smithery's wrapper would interact with our server before exposing it over WebSockets. 