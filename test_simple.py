#!/usr/bin/env python3
"""
Simple test for MAESTRO Protocol corrected architecture
"""

import sys
import os
import asyncio

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from maestro.orchestrator import MAESTROOrchestrator
    print("Successfully imported MAESTROOrchestrator")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def test_basic():
    print("Testing basic orchestrator functionality...")
    
    try:
        orchestrator = MAESTROOrchestrator()
        print("Orchestrator initialized successfully")
        
        # Test basic template listing
        templates = orchestrator.get_available_templates()
        print(f"Available templates: {templates}")
        
        # Test basic task analysis
        task = "Create a simple Python function"
        analysis = await orchestrator.analyze_task_for_planning(task)
        print(f"Task analysis completed for: {task}")
        print(f"Task type: {analysis['task_analysis']['task_type']}")
        
        print("All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic())
    print("Test result:", "PASSED" if success else "FAILED")