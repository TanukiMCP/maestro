# Maestro MCP Server: Deployment Guide

## Smithery.ai Deployment

1. Commit and push your code to GitHub
2. Connect your repository to Smithery.ai
3. Deploy using the Smithery dashboard
4. Health checks and tool discovery are automatic

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t maestro-mcp .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 maestro-mcp
   ```

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server (HTTP):
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```
3. Or run in stdio mode:
   ```bash
   python src/main.py --stdio
   ```

## Configuration

- Edit environment variables or `src/maestro/config.py` for advanced settings
- See `smithery.yaml` for Smithery-specific config 