# Maestro 🎭

**Intelligent Workflow Orchestration for Agentic IDEs**

Maestro is an MCP (Model Context Protocol) server that provides advanced workflow orchestration tools to dramatically enhance LLM capabilities in IDEs like Cursor and Claude Desktop. It intelligently discovers your available tools, maps them to workflow phases, and provides explicit execution guidance with context-aware tool recommendations.

## 🚀 What Maestro Does

Maestro transforms the way LLMs handle complex tasks by providing:

- **🔍 Dynamic Tool Discovery**: Automatically finds available MCP servers and IDE capabilities
- **🗺️ Intelligent Tool Mapping**: Maps specific tools to workflow phases with explicit usage instructions  
- **🎯 Context-Aware Orchestration**: Adapts workflows to your specific tool ecosystem
- **📋 Explicit Execution Guidance**: Provides step-by-step tool usage with commands, examples, and prerequisites

## 🎬 Example Usage

Instead of generic responses, Maestro provides specific, actionable guidance:

```
User: "Can you please use maestro_orchestrate to debug this TypeError and implement a resolution?"

Maestro Response:
🎭 Maestro Orchestration Complete

## 🚀 Orchestrated Execution Plan

### Phase 1: Analysis
Step 1: filesystem_read_file (Primary)
- Command: `filesystem_read_file({'path': './src/main.py'})`
- Example: `filesystem_read_file({'path': './src/main.py'}) # Analyze error source`
- Expected Result: File content for analysis

### Phase 2: Implementation  
Step 1: cursor_edit_file (Primary)
- Action: Use Cursor's AI editing for error resolution
- Prerequisites: Error location identified
- Expected Result: Fixed code with proper error handling

### Phase 3: Testing
Step 1: execute_python (Primary)
- Command: `execute_python({'code': 'python -m pytest tests/'})`
- Example: `execute_python({'code': 'python main.py'}) # Verify fix`
- Expected Result: Successful execution without errors
```

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tanukimcp/maestro.git
cd maestro
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Claude Desktop

Add Maestro to your Claude Desktop configuration:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "maestro": {
      "command": "python",
      "args": ["C:\\path\\to\\maestro\\src\\main.py"],
      "description": "Maestro - Intelligent Workflow Orchestration"
    }
  }
}
```

### 4. Restart Claude Desktop

Restart Claude Desktop to load Maestro.

## 🎭 Available Tools

### Primary Tool: `maestro_orchestrate`
The central orchestration tool that handles any workflow request:

```json
{
  "task": "debug this error and implement a fix",
  "context": {
    "error_details": "TypeError: expected str, got int",
    "current_file": "./src/main.py",
    "project_type": "web",
    "priority": "high"
  },
  "focus_phase": "implementation"
}
```

### Advanced Tools (for power users):
- `analyze_task_with_context` - Enhanced task analysis with tool discovery
- `create_tool_aware_execution_plan` - Explicit tool mapping and instructions
- `get_available_tools_with_context` - Dynamic tool inventory
- `analyze_task_for_planning` - Basic task analysis
- `create_execution_plan` - Generic execution planning
- `get_available_templates` - List workflow templates
- `get_template_details` - Template information

## 🎯 Key Features

### 🔍 Intelligent Tool Discovery
- Scans Claude Desktop configuration for MCP servers
- Detects IDE capabilities (Cursor, VS Code, etc.)
- Maintains real-time tool inventory with intelligent caching
- Adapts to your specific tool ecosystem

### 🗺️ Smart Workflow Mapping
- Maps discovered tools to workflow phases (Analysis, Implementation, Testing, etc.)
- Provides explicit usage instructions with commands and examples
- Assigns tool priorities (primary, secondary, fallback)
- Tracks dependencies and prerequisites

### 🎭 Context-Aware Orchestration
- Enhances system prompts with tool ecosystem awareness
- Generates explicit tool execution plans
- Provides fallback strategies when tools are unavailable
- Adapts recommendations based on task context

### 📊 Performance Optimized
- Sub-2-second tool discovery times
- 5-minute intelligent caching for optimal performance
- Handles large tool ecosystems efficiently
- Robust error handling and recovery

## 🏗️ Architecture

Maestro consists of four main components:

1. **Tool Discovery Engine** (`tool_discovery.py`) - Scans and catalogs available tools
2. **Tool Workflow Mapper** (`tool_workflow_mapper.py`) - Maps tools to workflow phases
3. **Context-Aware Orchestrator** (`context_aware_orchestrator.py`) - Provides intelligent orchestration
4. **MCP Server** (`main.py`) - Exposes tools via MCP protocol

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python tests/test_integration.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🎯 Use Cases

- **Debugging**: "Use maestro_orchestrate to debug this error and implement a resolution"
- **Feature Development**: "Use maestro_orchestrate to create a REST API with authentication"
- **Code Refactoring**: "Use maestro_orchestrate to refactor this legacy code for better performance"
- **Testing**: "Use maestro_orchestrate to create comprehensive tests for this module"
- **Documentation**: "Use maestro_orchestrate to generate documentation for this codebase"

## 🌟 Why Maestro?

Traditional LLM interactions in IDEs often lack context about available tools and provide generic guidance. Maestro solves this by:

- **Discovering** what tools you actually have available
- **Mapping** specific tools to specific workflow steps
- **Providing** explicit usage instructions with examples
- **Adapting** to your unique development environment

**Result**: LLMs become dramatically more capable at handling complex development tasks through intelligent orchestration.

---

*"The conductor who knows every instrument in the orchestra can compose symphonies tailored to the specific ensemble at hand."* 🎭✨ 