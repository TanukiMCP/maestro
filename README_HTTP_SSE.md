# Maestro MCP Server - HTTP/SSE Transport

> **🌟 NOW COMPATIBLE WITH SMITHERY MCP PLATFORM**

Enhanced Workflow Orchestration for LLM Intelligence Amplification with remote deployment capabilities.

## 🚀 Quick Start

### Development Mode
```bash
# Install dependencies
python deploy.py --install

# Start development server with hot reload
python deploy.py dev
```

Server will be available at:
- **Health Check**: http://127.0.0.1:8000/
- **SSE Endpoint**: http://127.0.0.1:8000/sse/
- **Messages Endpoint**: http://127.0.0.1:8000/messages/

### Production Deployment
```bash
# Production mode
python deploy.py prod --host 0.0.0.0 --port 8000

# Smithery-optimized mode
python deploy.py smithery
```

### Docker Deployment
```bash
# Build image
docker build -t maestro-mcp .

# Run container
docker run -p 8000:8000 maestro-mcp

# For Smithery deployment
docker run -p 8000:8000 maestro-mcp python deploy.py smithery
```

## 🔗 MCP Client Configuration

### Claude Desktop
Add to your MCP configuration file:

```json
{
  "mcpServers": {
    "maestro": {
      "url": "http://localhost:8000/sse",
      "env": {}
    }
  }
}
```

### Cursor IDE
In Cursor Settings > MCP > Add new global MCP server:

```json
{
  "mcpServers": {
    "maestro-http": {
      "url": "http://localhost:8000/sse",
      "env": {}
    }
  }
}
```

### Smithery Platform
The server is fully compatible with Smithery's MCP platform. Simply provide the SSE endpoint URL:

```
http://your-deployed-server.com:8000/sse
```

## 🛠️ Available Tools

### 1. **maestro_orchestrate**
Intelligent workflow orchestration with context analysis and success criteria validation.

```typescript
{
  "name": "maestro_orchestrate",
  "description": "🎭 Intelligent workflow orchestration with context analysis and success criteria validation",
  "parameters": {
    "task": "string - Task description to orchestrate",
    "context": "object - Additional context (optional)"
  }
}
```

### 2. **maestro_iae**
Intelligence Amplification Engine for computational problem solving.

```typescript
{
  "name": "maestro_iae", 
  "description": "🧠 Intelligence Amplification Engine - computational problem solving",
  "parameters": {
    "engine_domain": "string - quantum_physics | advanced_mathematics | computational_modeling",
    "computation_type": "string - Type of computation to perform",
    "parameters": "object - Parameters for computation (optional)"
  }
}
```

## 📊 Server Status Resource

Access server status and capabilities:

```
GET /sse (with MCP client)
Resource URI: maestro://status
```

Returns comprehensive server information including:
- Server status and version
- Available tools and capabilities  
- Deployment compatibility
- Endpoint information

## 🌐 Deployment Options

### Local Development
```bash
python deploy.py dev
```

### Production Server
```bash
python deploy.py prod --workers 4
```

### Cloud Deployment (Smithery)
```bash
python deploy.py smithery --host 0.0.0.0
```

### Docker Container
```bash
docker run -p 8000:8000 maestro-mcp
```

## 🔧 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check and server info |
| `/sse/` | GET | MCP SSE connection endpoint |
| `/messages/` | POST | MCP message handling |

## 🧪 Testing

### Manual Testing
```bash
# Test mode (direct execution)
python deploy.py test

# Check dependencies
python deploy.py --check

# Install and test
python deploy.py --install && python deploy.py dev
```

### Health Check
```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "service": "Maestro MCP Server",
  "status": "active", 
  "version": "1.0.0",
  "transport": "HTTP/SSE",
  "endpoints": {
    "sse": "/sse/",
    "messages": "/messages/",
    "health": "/"
  },
  "smithery_compatible": true
}
```

## 🔐 Security Considerations

- Server binds to `127.0.0.1` by default for local development
- Production mode uses `0.0.0.0` for external access
- Docker container runs as non-root user
- Health checks enabled for container orchestration

## 📋 Requirements

### Python Dependencies
- `mcp>=1.0.0` - Model Context Protocol framework
- `fastapi>=0.100.0` - Web framework
- `starlette>=0.27.0` - ASGI framework  
- `uvicorn[standard]>=0.23.0` - ASGI server

### System Requirements
- Python 3.9+
- Linux/macOS/Windows
- Docker (optional)

## 🚨 Migration from stdio

If migrating from the stdio version:

### Before (stdio)
```python
async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, options)
```

### After (HTTP/SSE)
```python
app = FastAPI()
mcp = FastMCP("maestro")
app.mount("/", create_sse_server(mcp))
```

The HTTP/SSE version provides:
- ✅ Remote deployment capability
- ✅ Smithery platform compatibility
- ✅ Better scalability
- ✅ Standard HTTP interfaces
- ✅ Container deployment support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python deploy.py test`
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**🎭 Ready to orchestrate AI workflows at scale!** 🚀 