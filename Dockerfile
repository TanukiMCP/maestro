# Maestro MCP Server - HTTP Transport for Smithery
# Ultra-lightweight for fast tool scanning
# Forced rebuild: 2024-06-04-2 (JSON-RPC format fix)

# Build stage for compiling and preparing dependencies
FROM python:3.11-slim AS builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Final stage - lightweight runtime image
FROM python:3.11-slim

# Set environment variables for runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set work directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy source code
COPY src/ ./src/
COPY mcp_official_server.py .
COPY mcp_http_transport.py .
COPY mcp_stdio_server.py .
COPY static_tools.py .
COPY static_tools_dict.py .
COPY README.md .
COPY LICENSE .

# Create non-root user
RUN useradd --create-home --shell /bin/bash maestro
RUN chown -R maestro:maestro /app
USER maestro

# Expose the port
EXPOSE 8000

# Command to run the HTTP transport wrapper
CMD ["python", "mcp_http_transport.py"]

# Labels for better discoverability
LABEL org.opencontainers.image.title="Maestro MCP Server" \
      org.opencontainers.image.description="Intelligence Amplification MCP Server" \
      org.opencontainers.image.vendor="TanukiMCP" \
      org.opencontainers.image.version="1.0.0" \
      com.smithery.compatible="true" \
      com.smithery.http_transport="enabled"

# Alternative commands (can be overridden):
# For development: docker run -p 8000:8000 maestro-mcp python deploy.py dev --host 0.0.0.0
# For production: docker run -p 8000:8000 maestro-mcp python deploy.py prod 