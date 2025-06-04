# Quick Start Guide

Get MAESTRO up and running in 5 minutes!

## 🚨 License Notice
**MAESTRO is licensed for NON-COMMERCIAL use only.** Commercial use requires approval from TanukiMCP.  
📋 [Commercial License Information](../COMMERCIAL_LICENSE_INFO.md) | 📧 tanukimcp@gmail.com

## 🚀 Installation

### Option 1: Docker (Recommended)
```bash
# Pull and run the MAESTRO container
docker run -p 8000:8000 tanukimcp/maestro:latest
```

### Option 2: Python Package
```bash
# Install from PyPI
pip install tanukimcp-maestro

# Or install from source
git clone https://github.com/tanukimcp/maestro.git
cd maestro
pip install -e .
```

### Option 3: Smithery Deployment
```bash
# Deploy to Smithery (requires Smithery CLI)
smithery deploy tanukimcp/maestro
```

## ⚡ Quick Test

Once installed, test your MAESTRO installation:

```bash
# Start the server
python -m src.main

# Test health endpoint
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "tools_available": 9,
  "engines_loaded": 8
}
```

## 🎯 First Workflow

### 1. Simple Orchestration
```python
import asyncio
from src.maestro_tools import MaestroTools

async def main():
    tools = MaestroTools()
    
    # Orchestrate a complex task
    result = await tools.handle_tool_call("maestro_orchestrate", {
        "task_description": "Analyze the current state of AI research and provide insights",
        "complexity_level": "moderate"
    })
    
    print(result[0].text)

asyncio.run(main())
```

### 2. Intelligence Amplification
```python
# Use the IAE for mathematical analysis
result = await tools.handle_tool_call("maestro_iae", {
    "analysis_request": "Calculate the eigenvalues of a 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]]",
    "engine_type": "mathematical",
    "precision_level": "high"
})
```

### 3. Web Intelligence
```python
# Enhanced web search with LLM analysis
result = await tools.handle_tool_call("maestro_search", {
    "query": "latest developments in quantum computing 2024",
    "max_results": 5,
    "temporal_filter": "recent"
})
```

## 🔧 MCP Integration

### With Claude Desktop
Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "maestro": {
      "command": "python",
      "args": ["-m", "src.main"],
      "env": {}
    }
  }
}
```

### With Other MCP Clients
```python
import mcp

# Connect to MAESTRO MCP server
async with mcp.ClientSession(
    transport=mcp.StdioServerTransport(
        command="python",
        args=["-m", "src.main"]
    )
) as session:
    
    # List available tools
    tools = await session.list_tools()
    print(f"Available tools: {[tool.name for tool in tools.tools]}")
    
    # Call a tool
    result = await session.call_tool("maestro_orchestrate", {
        "task_description": "Your task here"
    })
```

## 🌐 HTTP/SSE Usage

### Direct HTTP Calls
```bash
# Call maestro_orchestrate via HTTP
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "maestro_orchestrate",
      "arguments": {
        "task_description": "Analyze market trends in AI",
        "complexity_level": "moderate"
      }
    }
  }'
```

### JavaScript/TypeScript
```typescript
const response = await fetch('http://localhost:8000/mcp', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/call',
    params: {
      name: 'maestro_search',
      arguments: {
        query: 'AI research papers 2024',
        max_results: 10
      }
    }
  })
});

const result = await response.json();
console.log(result.result);
```

## 🎭 Core Tools Overview

| Tool | Use Case | Example |
|------|----------|---------|
| `maestro_orchestrate` | Complex multi-step tasks | Research analysis, report generation |
| `maestro_iae` | Mathematical/scientific computation | Calculations, data analysis |
| `maestro_search` | Information gathering | Market research, literature review |
| `maestro_scrape` | Content extraction | Website analysis, data collection |
| `maestro_execute` | Code execution | Script running, validation |
| `maestro_error_handler` | Error recovery | Debugging, troubleshooting |

## 🔍 Verification

Test each core tool:

```python
# Test orchestration
await tools.handle_tool_call("maestro_orchestrate", {
    "task_description": "Create a simple project plan"
})

# Test IAE
await tools.handle_tool_call("maestro_iae", {
    "analysis_request": "Calculate 2+2"
})

# Test search
await tools.handle_tool_call("maestro_search", {
    "query": "Python programming"
})

# Test execution
await tools.handle_tool_call("maestro_execute", {
    "command": "print('Hello MAESTRO!')",
    "language": "python"
})
```

## 🎯 Next Steps

1. **Explore Examples**: Check out [Basic Examples](./EXAMPLES.md)
2. **Learn Architecture**: Read [Architecture Overview](./ARCHITECTURE.md)
3. **Advanced Usage**: Try [Advanced Workflows](./ADVANCED_WORKFLOWS.md)
4. **Customize**: See [Configuration Guide](./CONFIGURATION.md)

## 🆘 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Change port
export PORT=8001
python -m src.main
```

**Missing dependencies:**
```bash
# Install all dependencies
pip install -r requirements.txt
```

**Permission errors:**
```bash
# Run with appropriate permissions
sudo python -m src.main  # Linux/Mac
# Or run as administrator on Windows
```

### Getting Help

- **Documentation**: [Complete docs](./README.md)
- **Issues**: [GitHub Issues](https://github.com/tanukimcp/maestro/issues)
- **Discussions**: [Community Forum](https://github.com/tanukimcp/maestro/discussions)

---

**You're ready to go!** 🎉 MAESTRO is now running and ready to amplify your AI capabilities. 