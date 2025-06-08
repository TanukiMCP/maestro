# TanukiMCP Maestro - Production MCP Server for Smithery.ai
# Optimized for instant tool discovery (<100ms) and production deployment
# Protocol: MCP 2024-11-05 | Smithery.ai Compatible

# Build stage for compiling and preparing dependencies
FROM python:3.11-slim AS builder

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /build

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage - lightweight runtime image
FROM python:3.11-slim

# Set environment variables for production runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DEBUG_MODE=false \
    API_KEY="" \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY src/ ./src/
COPY static_tools_dict.py .
COPY README.md .
COPY LICENSE .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 maestro && \
    chown -R maestro:maestro /app
USER maestro

# Expose the port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use HTTP transport endpoint - run the FastAPI server
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Metadata labels for Smithery.ai and container registries
LABEL org.opencontainers.image.title="TanukiMCP Maestro" \
      org.opencontainers.image.description="Meta-Agent Ensemble for Systematic Task Reasoning and Orchestration with 3-5x LLM capability amplification" \
      org.opencontainers.image.vendor="TanukiMCP" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.url="https://github.com/tanukimcp/maestro" \
      org.opencontainers.image.source="https://github.com/tanukimcp/maestro" \
      org.opencontainers.image.licenses="Non-Commercial" \
      com.smithery.compatible="true" \
      com.smithery.protocol="mcp-2024-11-05" \
      com.smithery.discovery_time="<100ms" \
      com.smithery.production_ready="true" 