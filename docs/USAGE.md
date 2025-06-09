# Usage Guide: Maestro MCP Server

## Running the Server

### HTTP (Production/Smithery)
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### stdio (MCP Protocol)
```bash
python src/main.py --stdio
```

### Docker
```bash
docker build -t maestro-mcp .
docker run -p 8000:8000 maestro-mcp
```

## Endpoints

- `/mcp` : Main MCP endpoint (POST, GET, DELETE)
- `/health` : Health check (GET)

## Configuration

- Edit environment variables or `src/maestro/config.py`
- Example: `MAESTRO_PORT=8000 MAESTRO_MODE=production python src/main.py`

## Tool Calls (HTTP)

POST to `/mcp` with JSON body:
```json
{
  "tool": "maestro_orchestrate",
  "parameters": {
    "task_description": "Summarize the latest AI research",
    "available_tools": ["maestro_web", "maestro_iae"]
  }
}
```

## Logs

- Logs are written to `logs/maestro_server.log` (if enabled)

## Example: Health Check

```bash
curl http://localhost:8000/health
``` 