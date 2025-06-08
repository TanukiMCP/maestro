@echo off
set PORT=8001
echo Starting TanukiMCP Maestro MCP Server on port %PORT%...
echo.
echo Server will be available at http://localhost:%PORT%/mcp
echo.
echo Press Ctrl+C to stop the server
echo.
set PORT=%PORT%
python mcp_http_transport.py 