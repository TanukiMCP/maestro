#!/usr/bin/env python3
"""
Simple test server using FastMCP
"""
from fastmcp import FastMCP

# Create server
mcp = FastMCP("SimpleTestServer")

@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text"""
    return f"Echo: {text}"

if __name__ == "__main__":
    mcp.run(transport="stdio") 