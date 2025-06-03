#!/usr/bin/env python3
"""
Test script for Maestro MCP stdio server
Validates that stdio server lists all tools correctly and returns valid responses.
"""

import asyncio
import os
import sys
# Ensure project root and mcp_stdio_server.py is on path
sys.path.insert(0, os.getcwd())

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    # Prepare stdio server parameters
    params = StdioServerParameters(command="python", args=["mcp_stdio_server.py"])
    # Connect to stdio MCP server
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            # Initialize protocol
            await session.initialize()
            # List tools
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]
            expected = [
                "maestro_orchestrate",
                "maestro_iae_discovery",
                "maestro_tool_selection",
                "maestro_iae",
                "maestro_search",
                "maestro_scrape",
                "maestro_execute",
                "maestro_error_handler",
                "maestro_temporal_context",
                "get_available_engines"
            ]
            missing = set(expected) - set(tool_names)
            if missing:
                print(f"❌ Missing tools: {missing}")
                sys.exit(1)
            print("✅ All expected tools are available")
            # Call a simple tool
            result = await session.call_tool("get_available_engines", {})
            if not isinstance(result, dict):
                print("❌ Unexpected result type from get_available_engines")
                sys.exit(1)
            print("✅ get_available_engines returned valid data")
            sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main()) 