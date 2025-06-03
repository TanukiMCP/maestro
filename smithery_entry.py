#!/usr/bin/env python3
"""
Smithery entry point for the Maestro MCP Server
"""
from src.main import mcp

def main():
    """Entry point for the MCP server when installed as a package"""
    # Run the MCP server with stdio transport
    mcp.run()

if __name__ == "__main__":
    main() 