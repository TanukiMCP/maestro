"""
Maestro Tool Discovery Engine

Dynamically discovers and maps available MCP tools and IDE capabilities
to provide intelligent, context-aware orchestration guidance.
"""

import asyncio
import logging
import json
import subprocess
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Information about a discovered tool."""
    name: str
    description: str
    server_name: str
    input_schema: Dict[str, Any]
    annotations: Optional[Dict[str, Any]] = None
    capabilities: List[str] = None
    usage_examples: List[str] = None


@dataclass
class MCPServerInfo:
    """Information about a discovered MCP server."""
    name: str
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    tools: List[ToolInfo] = None
    status: str = "unknown"  # "available", "unavailable", "error"


@dataclass
class IDECapability:
    """Information about IDE built-in capabilities."""
    name: str
    category: str
    description: str
    usage_pattern: str
    parameters: Dict[str, Any] = None


class ToolDiscoveryEngine:
    """
    Dynamically discover available MCP tools and IDE capabilities.
    
    This engine provides the foundation for context-aware orchestration
    by maintaining a real-time inventory of available tools.
    """
    
    def __init__(self):
        self.discovered_servers: Dict[str, MCPServerInfo] = {}
        self.available_tools: Dict[str, ToolInfo] = {}
        self.ide_capabilities: List[IDECapability] = []
        self.tool_categories: Dict[str, Set[str]] = {}
        self._last_discovery_time = None
    
    async def full_discovery_scan(self) -> Dict[str, Any]:
        """
        Perform complete discovery of all available tools and capabilities.
        
        Returns:
            Comprehensive inventory of available tools and capabilities
        """
        logger.info("🔍 Starting full tool discovery scan...")
        
        # Discover MCP servers and their tools
        mcp_results = await self.discover_mcp_ecosystem()
        
        # Discover IDE capabilities
        ide_results = await self.discover_ide_capabilities()
        
        # Create capability mapping
        capability_map = self.create_tool_capability_map()
        
        discovery_results = {
            "mcp_servers": mcp_results,
            "ide_capabilities": ide_results,
            "tool_capability_map": capability_map,
            "total_tools_discovered": len(self.available_tools),
            "discovery_timestamp": asyncio.get_event_loop().time()
        }
        
        self._last_discovery_time = discovery_results["discovery_timestamp"]
        
        logger.info(f"✅ Discovery complete: {len(self.discovered_servers)} servers, "
                   f"{len(self.available_tools)} tools, {len(self.ide_capabilities)} IDE capabilities")
        
        return discovery_results
    
    async def discover_mcp_ecosystem(self) -> Dict[str, MCPServerInfo]:
        """
        Discover available MCP servers and their tools.
        
        Scans for:
        1. Claude Desktop configuration
        2. Environment variables
        3. Common MCP server locations
        """
        logger.info("🔍 Discovering MCP ecosystem...")
        
        # Try to find Claude Desktop config
        claude_config = await self._find_claude_desktop_config()
        if claude_config:
            await self._discover_from_claude_config(claude_config)
        
        # Try to discover from environment
        await self._discover_from_environment()
        
        # Try to discover running MCP processes
        await self._discover_running_mcp_processes()
        
        return self.discovered_servers
    
    async def discover_ide_capabilities(self) -> List[IDECapability]:
        """
        Discover IDE built-in capabilities.
        
        Detects capabilities for:
        1. Cursor IDE
        2. VS Code
        3. General IDE features
        """
        logger.info("🔍 Discovering IDE capabilities...")
        
        # Detect IDE type
        ide_type = self._detect_ide_type()
        
        # Load capability definitions based on IDE
        if ide_type == "cursor":
            self.ide_capabilities = self._load_cursor_capabilities()
        elif ide_type == "vscode":
            self.ide_capabilities = self._load_vscode_capabilities()
        else:
            self.ide_capabilities = self._load_generic_ide_capabilities()
        
        logger.info(f"✅ Discovered {len(self.ide_capabilities)} IDE capabilities for {ide_type}")
        return self.ide_capabilities
    
    def create_tool_capability_map(self) -> Dict[str, List[str]]:
        """
        Create mapping from capabilities to available tools.
        
        Returns:
            Dict mapping capability names to lists of tool names that provide them
        """
        capability_map = {}
        
        # Map MCP tools to capabilities
        for tool_name, tool_info in self.available_tools.items():
            capabilities = self._infer_tool_capabilities(tool_info)
            for capability in capabilities:
                if capability not in capability_map:
                    capability_map[capability] = []
                capability_map[capability].append(tool_name)
        
        # Map IDE capabilities
        for ide_cap in self.ide_capabilities:
            capability_name = ide_cap.category
            if capability_name not in capability_map:
                capability_map[capability_name] = []
            capability_map[capability_name].append(f"ide_{ide_cap.name}")
        
        return capability_map
    
    async def _find_claude_desktop_config(self) -> Optional[Dict[str, Any]]:
        """Find and parse Claude Desktop configuration."""
        potential_paths = [
            Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",  # Windows
            Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",  # macOS
            Path.home() / ".config" / "claude" / "claude_desktop_config.json",  # Linux
            Path.home() / ".claude" / "claude_desktop_config.json",  # Alternative Linux
        ]
        
        for config_path in potential_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    logger.info(f"📋 Found Claude Desktop config at {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse config at {config_path}: {e}")
        
        return None
    
    async def _discover_from_claude_config(self, config: Dict[str, Any]):
        """Discover MCP servers from Claude Desktop configuration."""
        mcp_servers = config.get("mcpServers", {})
        
        for server_name, server_config in mcp_servers.items():
            try:
                server_info = MCPServerInfo(
                    name=server_name,
                    command=server_config.get("command", ""),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {})
                )
                
                # Try to discover tools from this server
                tools = await self._discover_server_tools(server_info)
                server_info.tools = tools
                server_info.status = "available" if tools else "unavailable"
                
                self.discovered_servers[server_name] = server_info
                
                # Add tools to global inventory
                for tool in tools:
                    self.available_tools[f"{server_name}_{tool.name}"] = tool
                
                logger.info(f"✅ Discovered server '{server_name}' with {len(tools)} tools")
                
            except Exception as e:
                logger.error(f"❌ Failed to discover server '{server_name}': {e}")
                server_info.status = "error"
    
    async def _discover_server_tools(self, server_info: MCPServerInfo) -> List[ToolInfo]:
        """
        Attempt to discover tools from an MCP server.
        
        This is a simplified implementation - in practice, you'd want to
        actually connect to the server and query its tools.
        """
        tools = []
        
        # For now, return sample tools based on server name/command patterns
        # In full implementation, this would actually connect to the server
        
        if "filesystem" in server_info.name.lower() or "file" in server_info.command.lower():
            tools.extend(self._get_filesystem_tools(server_info))
        
        if "git" in server_info.name.lower():
            tools.extend(self._get_git_tools(server_info))
        
        if "python" in server_info.command.lower() or "executor" in server_info.name.lower():
            tools.extend(self._get_python_tools(server_info))
        
        return tools
    
    def _get_filesystem_tools(self, server_info: MCPServerInfo) -> List[ToolInfo]:
        """Get filesystem-related tools."""
        return [
            ToolInfo(
                name="read_file",
                description="Read contents of a file",
                server_name=server_info.name,
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                capabilities=["file_reading", "content_access"],
                usage_examples=["read_file({'path': './src/main.py'})"]
            ),
            ToolInfo(
                name="write_file", 
                description="Write content to a file",
                server_name=server_info.name,
                input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
                capabilities=["file_writing", "content_creation"],
                usage_examples=["write_file({'path': './output.txt', 'content': 'Hello World'})"]
            ),
            ToolInfo(
                name="create_directory",
                description="Create a new directory",
                server_name=server_info.name,
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                capabilities=["directory_creation", "project_setup"],
                usage_examples=["create_directory({'path': './new-project'})"]
            )
        ]
    
    def _get_git_tools(self, server_info: MCPServerInfo) -> List[ToolInfo]:
        """Get git-related tools."""
        return [
            ToolInfo(
                name="git_status",
                description="Get git repository status",
                server_name=server_info.name,
                input_schema={"type": "object", "properties": {}},
                capabilities=["version_control", "status_checking"],
                usage_examples=["git_status()"]
            ),
            ToolInfo(
                name="git_commit",
                description="Commit changes to git repository",
                server_name=server_info.name,
                input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
                capabilities=["version_control", "change_tracking"],
                usage_examples=["git_commit({'message': 'Add new feature'})"]
            )
        ]
    
    def _get_python_tools(self, server_info: MCPServerInfo) -> List[ToolInfo]:
        """Get Python execution tools."""
        return [
            ToolInfo(
                name="execute_python",
                description="Execute Python code",
                server_name=server_info.name,
                input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
                capabilities=["code_execution", "testing", "computation"],
                usage_examples=["execute_python({'code': 'print(\"Hello World\")'})"]
            )
        ]
    
    async def _discover_from_environment(self):
        """Discover MCP servers from environment variables."""
        # Look for environment variables that might indicate MCP servers
        mcp_env_vars = {k: v for k, v in os.environ.items() if "MCP" in k.upper()}
        
        if mcp_env_vars:
            logger.info(f"📋 Found {len(mcp_env_vars)} MCP-related environment variables")
            # Implementation would parse these and discover servers
    
    async def _discover_running_mcp_processes(self):
        """Discover running MCP server processes."""
        try:
            # Look for running processes that might be MCP servers
            result = subprocess.run(
                ["ps", "aux"] if os.name != "nt" else ["tasklist"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse output for MCP-related processes
            # This is a simplified implementation
            if "mcp" in result.stdout.lower():
                logger.info("📋 Found potential MCP processes running")
                
        except Exception as e:
            logger.debug(f"Process discovery failed: {e}")
    
    def _detect_ide_type(self) -> str:
        """Detect which IDE/editor is being used."""
        # Check environment variables and process names
        if "CURSOR" in os.environ or "cursor" in str(Path.cwd()).lower():
            return "cursor"
        elif "VSCODE" in os.environ or "code" in str(Path.cwd()).lower():
            return "vscode"
        else:
            return "generic"
    
    def _load_cursor_capabilities(self) -> List[IDECapability]:
        """Load Cursor IDE specific capabilities."""
        return [
            IDECapability(
                name="edit_file",
                category="file_operations",
                description="Edit files with AI assistance",
                usage_pattern="Use Cursor's AI editing for complex modifications",
                parameters={"file_path": "string", "modification_type": "string"}
            ),
            IDECapability(
                name="terminal_access",
                category="execution",
                description="Access to integrated terminal",
                usage_pattern="Run commands in integrated terminal",
                parameters={"command": "string"}
            ),
            IDECapability(
                name="file_tree",
                category="navigation",
                description="File explorer and navigation",
                usage_pattern="Browse and manage project files",
                parameters={}
            ),
            IDECapability(
                name="ai_chat",
                category="ai_assistance",
                description="AI chat interface for code assistance",
                usage_pattern="Ask questions and get AI help",
                parameters={"query": "string"}
            )
        ]
    
    def _load_vscode_capabilities(self) -> List[IDECapability]:
        """Load VS Code specific capabilities."""
        return [
            IDECapability(
                name="command_palette",
                category="commands",
                description="Access to VS Code command palette",
                usage_pattern="Execute VS Code commands",
                parameters={"command": "string"}
            ),
            IDECapability(
                name="extensions",
                category="extensibility",
                description="Access to installed extensions",
                usage_pattern="Leverage installed extensions",
                parameters={}
            )
        ]
    
    def _load_generic_ide_capabilities(self) -> List[IDECapability]:
        """Load generic IDE capabilities."""
        return [
            IDECapability(
                name="text_editing",
                category="editing",
                description="Basic text editing capabilities",
                usage_pattern="Standard text editing operations",
                parameters={}
            )
        ]
    
    def _infer_tool_capabilities(self, tool_info: ToolInfo) -> List[str]:
        """Infer capabilities from tool information."""
        capabilities = tool_info.capabilities or []
        
        # Infer additional capabilities from name and description
        name_lower = tool_info.name.lower()
        desc_lower = tool_info.description.lower()
        
        if "file" in name_lower or "read" in name_lower:
            capabilities.append("file_operations")
        if "git" in name_lower:
            capabilities.append("version_control")
        if "execute" in name_lower or "run" in name_lower:
            capabilities.append("code_execution")
        if "test" in name_lower:
            capabilities.append("testing")
        if "create" in name_lower:
            capabilities.append("creation")
        
        return list(set(capabilities))  # Remove duplicates
    
    def get_tools_by_capability(self, capability: str) -> List[ToolInfo]:
        """Get all tools that provide a specific capability."""
        matching_tools = []
        
        for tool_name, tool_info in self.available_tools.items():
            tool_capabilities = self._infer_tool_capabilities(tool_info)
            if capability in tool_capabilities:
                matching_tools.append(tool_info)
        
        return matching_tools
    
    def is_discovery_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if discovery results are stale and need refresh."""
        if self._last_discovery_time is None:
            return True
        
        current_time = asyncio.get_event_loop().time()
        return (current_time - self._last_discovery_time) > max_age_seconds 