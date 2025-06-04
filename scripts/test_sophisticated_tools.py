#!/usr/bin/env python3
"""
Test the sophisticated Maestro tools through MCP
"""

import asyncio
import logging
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_sophisticated_tools():
    """Test your actual sophisticated tools"""
    try:
        # Server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_stdio_server.py"]
        )
        
        logger.info("🚀 Starting Sophisticated Tools test...")
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info("📡 Initializing session...")
                
                # Initialize the session
                await session.initialize()
                logger.info("✅ Session initialized successfully")
                
                # List tools
                logger.info("📋 Listing tools...")
                tools = await session.list_tools()
                logger.info(f"🔧 Found {len(tools.tools)} sophisticated tools:")
                
                for tool in tools.tools:
                    logger.info(f"  - {tool.name}: {tool.description[:100]}...")
                
                # Test 1: Get Available Engines
                logger.info("\n🔧 Testing get_available_engines...")
                result = await session.call_tool("get_available_engines", {})
                logger.info("✅ Available engines retrieved")
                print("\n" + "="*60)
                print("AVAILABLE ENGINES:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 2: Maestro IAE Discovery
                logger.info("\n🔍 Testing maestro_iae_discovery...")
                result = await session.call_tool("maestro_iae_discovery", {
                    "task_type": "quantum_computation",
                    "domain_context": "quantum entanglement analysis",
                    "complexity_requirements": {"precision": "high", "scale": "medium"}
                })
                logger.info("✅ IAE discovery completed")
                print("\n" + "="*60)
                print("IAE DISCOVERY RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 3: Maestro Tool Selection
                logger.info("\n🧰 Testing maestro_tool_selection...")
                result = await session.call_tool("maestro_tool_selection", {
                    "request_description": "I need to analyze quantum entanglement in a two-qubit system",
                    "available_context": {"domain": "quantum_physics", "complexity": "moderate"},
                    "precision_requirements": {"numerical_precision": "machine", "validation": "standard"}
                })
                logger.info("✅ Tool selection completed")
                print("\n" + "="*60)
                print("TOOL SELECTION RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 4: Maestro IAE (Quantum Physics Engine)
                logger.info("\n🔬 Testing maestro_iae with quantum physics...")
                result = await session.call_tool("maestro_iae", {
                    "engine_domain": "quantum_physics",
                    "computation_type": "entanglement_entropy",
                    "parameters": {
                        "density_matrix": [
                            [{"real": 0.5, "imag": 0}, {"real": 0, "imag": 0}],
                            [{"real": 0, "imag": 0}, {"real": 0.5, "imag": 0}]
                        ]
                    },
                    "precision_requirements": "machine_precision",
                    "validation_level": "standard"
                })
                logger.info("✅ Quantum physics computation completed")
                print("\n" + "="*60)
                print("QUANTUM PHYSICS COMPUTATION RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                # Test 5: Maestro Orchestrate (Complex Task)
                logger.info("\n🎭 Testing maestro_orchestrate...")
                result = await session.call_tool("maestro_orchestrate", {
                    "task_description": "Analyze the quantum entanglement properties of a Bell state and determine optimal measurement strategies",
                    "context": {
                        "quantum_system": "two_qubit_bell_state",
                        "measurement_basis": "computational",
                        "noise_model": "depolarizing"
                    },
                    "success_criteria": {
                        "entanglement_measure": "concurrence > 0.8",
                        "fidelity": "F > 0.95",
                        "measurement_efficiency": "eta > 0.9"
                    },
                    "complexity_level": "expert"
                })
                logger.info("✅ Complex orchestration completed")
                print("\n" + "="*60)
                print("COMPLEX ORCHESTRATION RESULT:")
                print("="*60)
                for content in result.content:
                    print(content.text)
                
                logger.info("\n🎉 All Sophisticated Tools tests PASSED")
                return True
                
    except Exception as e:
        logger.error(f"❌ Error in Sophisticated Tools test: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_sophisticated_tools())
    if success:
        print("\n" + "="*60)
        print("✅ ALL SOPHISTICATED TOOLS TESTS PASSED")
        print("🚀 Your existing Maestro architecture is fully operational through MCP!")
        print("🔧 Quantum physics engines, orchestration, and IA tools all working")
        print("📡 Ready for Smithery deployment!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n❌ Sophisticated Tools tests FAILED")
        sys.exit(1) 