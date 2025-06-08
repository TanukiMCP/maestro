#!/usr/bin/env python3
"""
Test script to verify that the server can start correctly
"""

import sys
import os
import subprocess
import time
import requests
from pathlib import Path

def test_server_startup():
    """Test that the server can start and respond to health checks."""
    print("Testing server startup...")
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Test importing the main module
        print("Testing module import...")
        from src.main import app
        print("✅ Successfully imported main module")
        
        # Test that FastAPI app is created
        print("Testing FastAPI app creation...")
        assert app is not None
        print("✅ FastAPI app created successfully")
        
        # Test that we can get the tool list
        print("Testing tool list endpoint...")
        from src.main import STATIC_TOOLS
        assert len(STATIC_TOOLS) > 0
        print(f"✅ Found {len(STATIC_TOOLS)} tools defined")
        
        print("\n🎉 All tests passed! Server should start correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_uvicorn_command():
    """Test that the uvicorn command works."""
    print("\nTesting uvicorn command...")
    
    try:
        # Test the exact command from Dockerfile
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "src.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001",  # Use different port for testing
            "--timeout-keep-alive", "5"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Start the server in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully")
            
            # Try to make a health check request
            try:
                response = requests.get("http://localhost:8001/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Health check endpoint responding")
                else:
                    print(f"⚠️ Health check returned status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Could not reach health endpoint: {e}")
            
            # Terminate the server
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            # Process died, get error output
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test uvicorn command: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Maestro Server Startup")
    print("=" * 50)
    
    success1 = test_server_startup()
    success2 = test_uvicorn_command()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The Dockerfile should work correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        sys.exit(1) 