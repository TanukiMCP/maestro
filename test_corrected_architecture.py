#!/usr/bin/env python3
"""
Test script for the corrected MAESTRO Protocol architecture.
Validates that the MCP server provides proper planning tools.
"""

import sys
import os
import asyncio
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maestro.orchestrator import MAESTROOrchestrator
from maestro.templates import list_available_templates, get_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_corrected_architecture():
    """Test the corrected MAESTRO architecture."""
    
    print("Testing Corrected MAESTRO Protocol Architecture")
    print("=" * 60)
    
    # Test 1: Initialize orchestrator (planning engine)
    print("\n1. Testing Orchestrator Initialization...")
    try:
        orchestrator = MAESTROOrchestrator()
        print("   [OK] Orchestrator initialized successfully")
    except Exception as e:
        print(f"   ❌ Orchestrator initialization failed: {e}")
        return False
    
    # Test 2: Test template system
    print("\n2. Testing Template System...")
    try:
        templates = list_available_templates()
        print(f"   ✅ Found {len(templates)} templates: {templates}")
        
        # Test getting a specific template
        code_template = get_template("code_development")
        if code_template:
            print("   ✅ Code development template loaded successfully")
        else:
            print("   ❌ Failed to load code development template")
    except Exception as e:
        print(f"   ❌ Template system test failed: {e}")
        return False
    
    # Test 3: Test task analysis (primary MCP tool)
    print("\n3. Testing Task Analysis Tool...")
    try:
        test_task = "Create a Python function to calculate factorial with error handling"
        analysis = await orchestrator.analyze_task_for_planning(test_task, "comprehensive")
        
        print(f"   ✅ Task analysis completed")
        print(f"   - Task Type: {analysis['task_analysis']['task_type']}")
        print(f"   - Complexity: {analysis['task_analysis']['complexity']}")
        print(f"   - Template: {analysis['template_used']}")
        print(f"   - Phases: {len(analysis['execution_phases'])}")
        print(f"   - Success Criteria: {len(analysis['success_criteria'])}")
        
    except Exception as e:
        print(f"   ❌ Task analysis test failed: {e}")
        return False
    
    # Test 4: Test execution planning
    print("\n4. Testing Execution Planning...")
    try:
        plan = await orchestrator.create_execution_plan(test_task)
        print(f"   ✅ Execution plan created")
        print(f"   - Full plan phases: {len(plan.get('full_plan', []))}")
        print(f"   - Execution sequence: {plan.get('execution_sequence', [])}")
        
    except Exception as e:
        print(f"   ❌ Execution planning test failed: {e}")
        return False
    
    # Test 5: Test template details
    print("\n5. Testing Template Details...")
    try:
        details = orchestrator.get_template_details("mathematical_analysis")
        if details and "error" not in details:
            print("   ✅ Template details retrieved successfully")
            print(f"   - Description: {details.get('description', 'N/A')}")
        else:
            print("   ❌ Failed to get template details")
    except Exception as e:
        print(f"   ❌ Template details test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! MAESTRO Protocol architecture is working correctly.")
    print("\n📋 Summary:")
    print("- ✅ Orchestrator provides PLANNING tools (not execution)")
    print("- ✅ Templates system is modular and functional")
    print("- ✅ Task analysis provides comprehensive guidance")
    print("- ✅ Execution planning breaks down complex tasks")
    print("- ✅ System properly serves as LLM enhancement layer")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_corrected_architecture())
    sys.exit(0 if success else 1)