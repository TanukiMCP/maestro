#!/usr/bin/env python3
"""
Maestro MCP Server Deployment Script

Usage:
    python deploy.py [mode] [options]

Modes:
    dev     - Development mode with hot reload
    prod    - Production mode  
    test    - Test mode (direct execution)
    smithery - Deploy for Smithery platform

Examples:
    python deploy.py dev
    python deploy.py prod --port 8000
    python deploy.py test
    python deploy.py smithery --host 0.0.0.0
"""

import sys
import subprocess
import argparse
import asyncio
from pathlib import Path
import os
import time # Add time for logging

# VERY EARLY LOGGING
print(f"DEPLOY.PY EXECUTION STARTED AT: {time.time()}", flush=True)

def run_dev_mode(host="127.0.0.1", port=8000):
    """Run in development mode with hot reload."""
    print(f"🎭 Starting Maestro MCP Server in DEV mode")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"🔗 SSE endpoint: http://{host}:{port}/sse/")
    print(f"📡 Messages endpoint: http://{host}:{port}/messages/")
    print(f"🔄 Hot reload enabled - changes will restart the server")
    
    cmd = [
        "uvicorn", 
        "src.main:app",
        "--host", str(host),
        "--port", str(port),
        "--reload"
    ]
    
    subprocess.run(cmd)

def run_prod_mode(host="0.0.0.0", port=8000, workers=1):
    """Run in production mode."""
    print(f"🎭 Starting Maestro MCP Server in PRODUCTION mode")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"🔗 SSE endpoint: http://{host}:{port}/sse/")
    print(f"📡 Messages endpoint: http://{host}:{port}/messages/")
    print(f"👥 Workers: {workers}")
    
    cmd = [
        "uvicorn", 
        "src.main:app",
        "--host", str(host),
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    subprocess.run(cmd)

def run_test_mode():
    """Run in test mode (direct execution)."""
    print("🎭 Starting Maestro MCP Server in TEST mode")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("🔗 SSE endpoint: http://127.0.0.1:8000/sse/")
    
    cmd = [sys.executable, "src/main.py", "direct"]
    subprocess.run(cmd)

def run_smithery_mode(host="0.0.0.0", port=8000):
    """Run optimized for Smithery platform."""
    print(f"🎭 Starting Maestro MCP Server for SMITHERY")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"🔗 SSE endpoint: http://{host}:{port}/sse/")
    print(f"📡 MCP endpoint: http://{host}:{port}/mcp")
    print(f"🧰 Tools endpoint: http://{host}:{port}/tools")
    print("🌟 Optimized for Smithery MCP platform")
    
    # Set environment variables to signal Smithery mode to the application
    # This helps the server optimize for fast tool scanning
    os.environ["SMITHERY_MODE"] = "true"
    os.environ["ENABLE_LAZY_LOADING"] = "true"
    os.environ["OPTIMIZE_TOOL_SCANNING"] = "true"
    os.environ["FASTAPI_PRIORITY_ROUTES"] = "true"  # Signal to prioritize route registration
    os.environ["MCP_DEFERRED_INIT"] = "true"  # Defer full MCP initialization 
    
    cmd = [
        "uvicorn", 
        "src.main:app",
        "--host", str(host),
        "--port", str(port),
        "--workers", "1",
        "--timeout-keep-alive", "120",  # Increase keep-alive timeout for long connections
        "--log-level", "info",
        "--lifespan", "on"  # Ensure lifespan events are processed
    ]
    
    subprocess.run(cmd)

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ["fastapi", "uvicorn", "starlette", "mcp"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Maestro MCP Server Deployment")
    parser.add_argument("mode", nargs="?", choices=["dev", "prod", "test", "smithery"], 
                       help="Deployment mode")
    parser.add_argument("--host", default="127.0.0.1", 
                       help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to bind to (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of worker processes (production mode only)")
    parser.add_argument("--install", action="store_true", 
                       help="Install dependencies before running")
    parser.add_argument("--check", action="store_true", 
                       help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # Check or install dependencies
    if args.install:
        if not install_dependencies():
            sys.exit(1)
    
    if args.check:
        if check_dependencies():
            print("🎭 Maestro MCP Server is ready to deploy!")
            sys.exit(0)
        else:
            print("❌ Please install missing dependencies")
            sys.exit(1)
    
    # Validate mode is provided for non-utility operations
    if not args.mode:
        parser.error("mode is required unless using --check or --install")
    
    # Check dependencies before running
    if not check_dependencies():
        print("❌ Missing dependencies. Run with --install to install them.")
        sys.exit(1)
    
    # Adjust host for different modes
    if args.mode == "dev":
        host = args.host if args.host != "127.0.0.1" else "127.0.0.1"
    elif args.mode in ["prod", "smithery"]:
        host = args.host if args.host != "127.0.0.1" else "0.0.0.0"
    else:
        host = "127.0.0.1"
    
    # Run the appropriate mode
    try:
        if args.mode == "dev":
            run_dev_mode(host, args.port)
        elif args.mode == "prod":
            run_prod_mode(host, args.port, args.workers)
        elif args.mode == "test":
            run_test_mode()
        elif args.mode == "smithery":
            run_smithery_mode(host, args.port)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 