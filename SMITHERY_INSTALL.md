# Maestro MCP Server - Smithery Installation Guide

## 🚀 Quick Install via Smithery CLI

### For Claude Desktop Users
```bash
# Install via Smithery (automatic configuration)
npx @smithery/cli install tanuki-maestro-mcp --client claude
```

### For Cursor IDE Users
```bash
# Install via Smithery 
npx @smithery/cli install tanuki-maestro-mcp --client cursor
```

### Manual Configuration (Advanced)

#### Option 1: Remote Server (Recommended)
Add to your MCP client configuration:

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "maestro": {
      "url": "http://localhost:8001/sse/"
    }
  }
}
```

**Cursor** (`settings.json`):
```json
{
  "mcp.servers": {
    "maestro": {
      "url": "http://localhost:8001/sse/"
    }
  }
}
```

#### Option 2: Local Installation
```json
{
  "mcpServers": {
    "maestro": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest", 
        "run",
        "tanuki-maestro-mcp",
        "--config",
        "{}"
      ]
    }
  }
}
```

## 🎭 Available Tools (10 Total)

### 🏆 Tier 1: Central Orchestration
- **maestro_orchestrate** - Main workflow orchestration for any development task

### 🧠 Tier 2: Intelligence Amplification  
- **maestro_iae** - Computational engine for complex problem solving
- **amplify_capability** - Direct access to specialized amplification engines

### 🔍 Tier 3: Quality & Verification
- **verify_quality** - Comprehensive quality verification and validation

### 🌐 Tier 4: Enhanced Automation (5 Tools)
- **maestro_search** - Intelligent web search with LLM optimization
- **maestro_scrape** - Smart web scraping with content extraction
- **maestro_execute** - Secure code execution with analysis
- **maestro_error_handler** - Adaptive error handling and resolution
- **maestro_temporal_context** - Time-aware information processing

### 📊 Tier 5: System Information
- **get_available_engines** - List all available computational engines

## 🎯 Usage Examples

### Basic Orchestration
```
Ask Claude/Cursor: "Use maestro_orchestrate to help me build a Python web scraper"
```

### Intelligence Amplification
```
Ask: "Use maestro_iae with quantum_physics domain to calculate wave interference patterns"
```

### Quality Verification
```
Ask: "Use verify_quality to check this code for accuracy and completeness"
```

## 🔧 Requirements

- **Python**: 3.9 or higher
- **Node.js**: 18.0 or higher (for Smithery CLI)
- **Dependencies**: Automatically installed via pip

## 🌟 Features

- ✅ **HTTP/SSE Transport** - Remote deployment ready
- ✅ **Smithery Compatible** - One-command installation
- ✅ **10 Specialized Tools** - Comprehensive AI workflow enhancement
- ✅ **5-Tier Architecture** - Organized tool hierarchy
- ✅ **Quality Verification** - Built-in validation capabilities
- ✅ **Error Handling** - Adaptive problem resolution
- ✅ **Temporal Awareness** - Time-sensitive processing

## 📚 Documentation

- [README.md](README.md) - Full documentation
- [SMITHERY_DEPLOYMENT.md](SMITHERY_DEPLOYMENT.md) - Deployment guide
- [README_HTTP_SSE.md](README_HTTP_SSE.md) - HTTP/SSE technical details

## 🆘 Support

If you encounter issues:

1. **Check server status**: Visit `http://localhost:8001/` for health check
2. **Validate tools**: Run `python validate_all_tools.py`
3. **Restart client**: Restart Claude Desktop/Cursor after installation
4. **Check logs**: Look for connection errors in your MCP client

## 🚀 Ready for AI Enhancement!

Once installed, you'll have access to 10 powerful tools that transform your AI assistant into a comprehensive development and problem-solving partner.

**Start with**: `maestro_orchestrate` for general tasks, then explore specialized tools as needed! 