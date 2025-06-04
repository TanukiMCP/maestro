"""
TanukiMCP Maestro Test Suite
Comprehensive testing for production-ready MCP server tools

Tests both Smithery.ai compatibility and real-world agentic IDE usage
"""

import os
import sys

# Add project root and src to path for testing
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src')) 