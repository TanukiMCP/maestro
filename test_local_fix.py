#!/usr/bin/env python3
"""
Quick test to verify the local fixes work
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_orchestrator():
    """Test the MAESTRO orchestrator with fixes"""
    print("🧪 Testing MAESTRO Orchestrator fixes...")
    
    try:
        from maestro.orchestrator import MAESTROOrchestrator
        print("✅ MAESTROOrchestrator imported successfully")
        
        orchestrator = MAESTROOrchestrator()
        print("✅ MAESTROOrchestrator initialized successfully")
        
        # Test amplify_capability with the fixed parameters
        result = await orchestrator.amplify_capability(
            capability_type="mathematics",
            input_data="derivative of x^2",
            requirements={}
        )
        print(f"✅ amplify_capability test: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_profile_import():
    """Test operator profile imports"""
    print("\n🧪 Testing OperatorProfile fixes...")
    
    try:
        from profiles.operator_profiles import OperatorProfile, OperatorProfileManager
        print("✅ OperatorProfile imported successfully")
        
        manager = OperatorProfileManager()
        print("✅ OperatorProfileManager initialized successfully")
        
        profiles = manager.get_all_profiles()
        print(f"✅ Found {len(profiles)} profiles")
        
        # Test that profiles have profile_id
        for name, profile in profiles.items():
            if hasattr(profile, 'profile_id'):
                print(f"✅ Profile '{name}' has profile_id: {profile.profile_id}")
            else:
                print(f"❌ Profile '{name}' missing profile_id")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Profile test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

async def main():
    """Run all tests"""
    print("🎭 MAESTRO Protocol Local Fix Testing")
    print("=" * 50)
    
    # Test profile fixes
    profile_success = test_profile_import()
    
    # Test orchestrator fixes  
    orchestrator_success = await test_orchestrator()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  • Profile Tests: {'✅ PASSED' if profile_success else '❌ FAILED'}")
    print(f"  • Orchestrator Tests: {'✅ PASSED' if orchestrator_success else '❌ FAILED'}")
    
    if profile_success and orchestrator_success:
        print("\n🎉 All fixes working locally!")
        print("🚀 Ready to test on deployed server")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 