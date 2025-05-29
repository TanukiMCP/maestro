#!/usr/bin/env python3
"""
Quick test to verify MAESTRO Protocol MCP Server is working
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

async def test_server_startup():
    """Test that the server starts up correctly"""
    print("🧪 Testing MAESTRO Protocol MCP Server...")
    
    # Path to the main server script
    server_script = Path(__file__).parent / "src" / "main.py"
    
    try:
        # Start the server process
        print("🚀 Starting server process...")
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for startup
        await asyncio.sleep(2)
        
        # Check if process is still running (good sign)
        if process.poll() is None:
            print("✅ Server process started successfully!")
            print("🔗 Server is running and waiting for MCP client connections")
            
            # Terminate the test process
            process.terminate()
            process.wait()
            
            return True
        else:
            # Process exited, check for errors
            stdout, stderr = process.communicate()
            print("❌ Server process exited unexpectedly")
            if stdout:
                print(f"STDOUT: {stdout}")
            if stderr:
                print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start server: {str(e)}")
        return False

def test_imports():
    """Test that all core components can be imported"""
    print("\n🔍 Testing component imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    try:
        # Test core imports
        from maestro import MAESTROOrchestrator
        print("✅ MAESTRO Orchestrator import")
        
        from engines import IntelligenceAmplifier
        print("✅ Intelligence Amplifier import")
        
        from profiles.operator_profiles import OperatorProfileManager
        print("✅ Operator Profile Manager import")
        
        from main import TanukiMCPOrchestra
        print("✅ MCP Server import")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print("🎭 MAESTRO Protocol MCP Server Test Suite")
    print("=" * 50)
    
    # Test imports first
    import_success = test_imports()
    
    if not import_success:
        print("\n❌ Import tests failed. Cannot proceed with server tests.")
        return False
    
    # Test server startup
    server_success = await test_server_startup()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  • Import Tests: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"  • Server Tests: {'✅ PASSED' if server_success else '❌ FAILED'}")
    
    if import_success and server_success:
        print("\n🎉 MAESTRO Protocol MCP Server is ready!")
        print("\n💡 Next Steps:")
        print("  🚀 Start server: python scripts/start_server.py")
        print("  📚 Connect MCP client to stdio transport")
        print("  🧪 Run verification: python scripts/verify_installation.py")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 