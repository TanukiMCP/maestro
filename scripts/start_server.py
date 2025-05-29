#!/usr/bin/env python3
"""
MAESTRO Protocol MCP Server Startup Script

This script provides an easy way to start the MAESTRO Protocol MCP server
with proper configuration, dependency checking, and error handling.
"""

import sys
import os
import subprocess
import logging
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'mcp',
        'pydantic',
        'typing_extensions'
    ]
    
    optional_packages = {
        'sympy': 'Advanced mathematical computation',
        'numpy': 'Numerical analysis',
        'scipy': 'Scientific computing',
        'pandas': 'Data analysis',
        'matplotlib': 'Data visualization',
        'seaborn': 'Statistical visualization',
        'spacy': 'Natural language processing',
        'textstat': 'Text readability analysis',
        'nltk': 'Natural language toolkit',
        'beautifulsoup4': 'HTML parsing',
        'requests': 'HTTP requests',
        'black': 'Code formatting',
        'pylint': 'Code analysis',
        'pytest': 'Testing framework'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (required)")
            missing_required.append(package)
    
    # Check optional packages
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠️  {package} - {description} (optional)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("These packages provide enhanced functionality but are not required")
        print("Install with: pip install -r requirements.txt")
    
    return True

def setup_logging():
    """Set up logging configuration with UTF-8 encoding"""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        log_dir / "maestro_server.log", 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Stream handler for console 
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Only add stream handler if it won't cause encoding issues
    try:
        # Test if we can log with emojis
        test_logger = logging.getLogger("test")
        test_logger.addHandler(stream_handler)
        test_logger.info("Test message with emoji: 🎭")
        logger.addHandler(stream_handler)
    except UnicodeEncodeError:
        # Skip console logging on systems that can't handle UTF-8
        print("Note: Console logging disabled due to Unicode encoding limitations")
    
    return logging.getLogger("maestro_server")

def test_server_components():
    """Test that server components can be imported and initialized"""
    print("\n🧪 Testing server components...")
    
    try:
        # Test main server import
        from main import TanukiMCPOrchestra
        print("✅ Main server module")
        
        # Test orchestrator
        from maestro.orchestrator import MAESTROOrchestrator
        orchestrator = MAESTROOrchestrator()
        print("✅ MAESTRO Orchestrator")
        
        # Test intelligence amplifier
        from engines.intelligence_amplifier import IntelligenceAmplifier
        amplifier = IntelligenceAmplifier()
        print("✅ Intelligence Amplifier")
        
        # Test quality controller
        from maestro.quality_controller import QualityController
        quality_controller = QualityController()
        print("✅ Quality Controller")
        
        # Test profile manager
        from profiles.operator_profiles import OperatorProfileManager
        profile_manager = OperatorProfileManager()
        print("✅ Operator Profile Manager")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

async def start_server():
    """Start the MAESTRO Protocol MCP server"""
    print("\n🚀 Starting MAESTRO Protocol MCP Server...")
    print("=" * 50)
    
    # Change to src directory
    src_dir = Path(__file__).parent.parent / "src"
    os.chdir(src_dir)
    
    try:
        # Import and run the server
        from main import main
        await main()  # Properly await the async function
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        return False

def show_usage_info():
    """Show usage information"""
    print("\n📚 MAESTRO Protocol MCP Server")
    print("Intelligence Amplification > Model Scale")
    print("=" * 50)
    
    print("\n🎯 Available Tools:")
    print("1. orchestrate_workflow - Main meta-orchestration tool")
    print("   - Handles complex tasks with automatic workflow generation")
    print("   - Includes intelligence amplification and quality control")
    print("   - Supports all task types and complexity levels")
    
    print("\n2. verify_quality - Quality verification and validation")
    print("   - Multiple verification methods (mathematical, logical, etc.)")
    print("   - Comprehensive quality metrics")
    print("   - Detailed feedback and recommendations")
    
    print("\n3. amplify_capability - Specific intelligence amplification")
    print("   - Mathematical reasoning and computation")
    print("   - Data analysis and pattern recognition")
    print("   - Language enhancement and style analysis")
    print("   - Code quality analysis and optimization")
    print("   - Web content verification and accessibility")
    
    print("\n🔧 Configuration:")
    print("- Transport: stdio (standard MCP protocol)")
    print("- Logging: Available in logs/maestro_server.log")
    print("- Dependencies: See requirements.txt")
    
    print("\n📖 Documentation:")
    print("- Implementation Guide: MAESTRO_PROTOCOL_IMPLEMENTATION_GUIDE.md")
    print("- Examples: examples/basic_usage.py")
    print("- Tests: tests/test_maestro_basic.py")

async def main():
    """Main startup function"""
    print("🎭 MAESTRO Protocol MCP Server Startup")
    print("Intelligence Amplification > Model Scale")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 To install dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting MAESTRO Protocol MCP Server")
    
    # Test components
    if not test_server_components():
        print("\n❌ Component tests failed. Please check your installation.")
        sys.exit(1)
    
    # Show usage information
    show_usage_info()
    
    # Start the server
    print("\n" + "=" * 60)
    print("🎭 Server is ready! Connect your MCP client to stdio transport.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)
    
    success = await start_server()
    
    if success:
        logger.info("MAESTRO Protocol MCP Server stopped successfully")
        print("\n✅ Server stopped successfully")
    else:
        logger.error("MAESTRO Protocol MCP Server stopped with errors")
        print("\n❌ Server stopped with errors")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 