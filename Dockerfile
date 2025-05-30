# Maestro MCP Server - HTTP/SSE Transport
# Optimized for Smithery and remote deployment

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY deploy.py .
COPY README.md .
COPY LICENSE .

# Create non-root user
RUN useradd --create-home --shell /bin/bash maestro
RUN chown -R maestro:maestro /app
USER maestro

# Expose port
EXPOSE 8000

# Default command - Run FastMCP server directly
CMD ["python", "src/main.py"]

# Alternative commands (can be overridden):
# For development: docker run -p 8000:8000 maestro-mcp python deploy.py dev --host 0.0.0.0
# For production: docker run -p 8000:8000 maestro-mcp python deploy.py prod
# For Smithery: docker run -p 8000:8000 maestro-mcp python deploy.py smithery 