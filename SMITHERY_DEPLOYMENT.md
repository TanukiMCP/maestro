# Maestro MCP Server - Smithery Deployment Guide

> **🌟 FULLY COMPATIBLE WITH SMITHERY MCP PLATFORM**

## 🚀 Quick Deployment

### 1. Local Development Testing
```bash
# Install dependencies
python deploy.py --install

# Test locally
python deploy.py dev
```

### 2. Smithery Production Deployment
```bash
# Deploy for Smithery
python deploy.py smithery
```

### 3. Docker Deployment (Recommended for Production)
```bash
# Build and run
docker build -t maestro-mcp .
docker run -p 8000:8000 maestro-mcp
```

## 🔧 Smithery Configuration

### MCP Client Configuration
Configure your Smithery client to connect to:

**SSE Endpoint:** `http://your-server:8000/sse/`

### Expected Response Format
The server provides JSON-RPC over SSE with these endpoints:

| Endpoint | Purpose | Method |
|----------|---------|--------|
| `/` | Health check and server info | GET |
| `/sse/` | **Main MCP SSE connection** | GET |
| `/messages/` | Message handling (if needed) | POST |

## 🛠️ Available MCP Tools

### 1. `maestro_orchestrate`
**Purpose:** Intelligent workflow orchestration with context analysis

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

**Example Usage:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "maestro_orchestrate",
    "arguments": {
      "task": "Create a Python web scraper for e-commerce data",
      "context": {"target_site": "example.com", "data_types": ["prices", "reviews"]}
    }
  }
}
```

### 2. `maestro_iae`
**Purpose:** Intelligence Amplification Engine for computational problem solving

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

**Example Usage:**
```json
{
  "method": "tools/call", 
  "params": {
    "name": "maestro_iae",
    "arguments": {
      "engine_domain": "advanced_mathematics",
      "computation_type": "optimization",
      "parameters": {"variables": 3, "constraints": ["x > 0", "y < 10"]}
    }
  }
}
```

## 📊 Resources Available

### Server Status Resource
**URI:** `maestro://status`

Provides comprehensive server information including:
- Active tools and capabilities
- Performance metrics
- Deployment status
- Compatibility information

## 🔒 Security & Performance

### Security Features
- ✅ CORS-enabled for cross-origin requests
- ✅ Proper HTTP headers for SSE
- ✅ Health check endpoints for monitoring
- ✅ Error handling and logging

### Performance Optimizations  
- ✅ Async/await throughout
- ✅ Connection pooling for SSE
- ✅ Efficient JSON-RPC message handling
- ✅ Resource lifecycle management

## 🚀 Production Deployment Options

### Option 1: Direct Python Deployment
```bash
# Production with multiple workers
python deploy.py prod --workers 4 --host 0.0.0.0 --port 8000
```

### Option 2: Docker Container (Recommended)
```bash
# Build image
docker build -t maestro-mcp .

# Run with resource limits
docker run -d \
  --name maestro-mcp \
  -p 8000:8000 \
  --memory=512m \
  --cpus=1.0 \
  maestro-mcp
```

### Option 3: Docker Compose
```yaml
version: '3.8'
services:
  maestro-mcp:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

## 🧪 Testing & Validation

### Health Check
```bash
curl http://localhost:8000/
```

**Expected Response:**
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

### SSE Connection Test
```bash
curl -H "Accept: text/event-stream" http://localhost:8000/sse/
```

Should establish SSE connection with proper headers.

### Full Integration Test
```bash
python test_server.py
```

## 🔧 Troubleshooting

### Common Issues

**Issue:** SSE connection fails
**Solution:** Ensure proper headers are set:
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Issue:** Tool calls not working
**Solution:** Verify JSON-RPC format:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "maestro_orchestrate",
    "arguments": { ... }
  }
}
```

**Issue:** Connection timeout
**Solution:** Increase timeout settings and check network connectivity.

### Debugging
Enable debug logging:
```bash
PYTHONLOGLEVEL=DEBUG python deploy.py smithery
```

## 📋 Requirements

### System Requirements
- Python 3.9+
- 512MB RAM minimum
- Network access for Smithery platform

### Python Dependencies
- `mcp>=1.0.0` - Model Context Protocol framework
- `fastapi>=0.100.0` - Web framework for HTTP/SSE
- `uvicorn[standard]>=0.23.0` - ASGI server
- `starlette>=0.27.0` - Core ASGI components

### Network Requirements
- Outbound HTTP/HTTPS access
- Port 8000 accessible (or custom port)
- Stable internet connection for Smithery

## 🎯 Migration from stdio

### Before (stdio transport)
```python
async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, options)
```

### After (HTTP/SSE transport)
```python
app = FastAPI()
transport = SseServerTransport("/messages/")
# Server automatically handles SSE connections
```

### Benefits of HTTP/SSE
- ✅ **Remote deployment** capability
- ✅ **Smithery platform** compatibility
- ✅ **Better scalability** and monitoring
- ✅ **Standard HTTP** interfaces
- ✅ **Container deployment** support
- ✅ **Load balancing** ready

## 🌟 Smithery Integration Features

### Optimized for Smithery
- **Event-driven architecture** for real-time communication
- **JSON-RPC over SSE** for efficient message passing
- **Auto-reconnection** handling built-in
- **Resource management** for long-running connections
- **Health monitoring** endpoints for platform integration

### Deployment Verification
1. **Health Check:** Server responds to health endpoint
2. **SSE Connection:** Proper SSE headers and connection
3. **Tool Registration:** Tools appear in Smithery interface
4. **Message Handling:** JSON-RPC messages processed correctly
5. **Resource Access:** Status and capabilities accessible

---

**🎭 Ready for Smithery MCP Platform! 🚀**

The Maestro MCP Server is now fully compatible with Smithery's streamable HTTP/SSE transport requirements and ready for production deployment. 