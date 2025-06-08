# TanukiMCP Maestro

A local MCP server with advanced orchestration tools.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python mcp_http_transport.py
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

## Architecture

The server uses a Smithery-compatible MCP transport with these key components:

- `mcp_http_transport.py` - Main HTTP server with MCP protocol support
- `server.py` - Core tool execution logic
- `static_tools_dict.py` - Static tool definitions for instant tool discovery
