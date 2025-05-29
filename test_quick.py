#!/usr/bin/env python3
"""
Quick test script to verify MAESTRO Protocol components
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from maestro.orchestrator import MAESTROOrchestrator
        print("✅ MAESTROOrchestrator imported")
        
        from engines.intelligence_amplifier import IntelligenceAmplifier
        print("✅ IntelligenceAmplifier imported")
        
        from maestro.quality_controller import QualityController
        print("✅ QualityController imported")
        
        from profiles.operator_profiles import OperatorProfileManager
        print("✅ OperatorProfileManager imported")
        
        from main import TanukiMCPOrchestra
        print("✅ TanukiMCPOrchestra imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_initialization():
    """Test that components can be initialized"""
    print("\n🔧 Testing initialization...")
    
    try:
        from maestro.orchestrator import MAESTROOrchestrator
        from engines.intelligence_amplifier import IntelligenceAmplifier
        from maestro.quality_controller import QualityController
        from profiles.operator_profiles import OperatorProfileManager
        
        orchestrator = MAESTROOrchestrator()
        print("✅ MAESTROOrchestrator initialized")
        
        amplifier = IntelligenceAmplifier()
        print("✅ IntelligenceAmplifier initialized")
        
        quality_controller = QualityController()
        print("✅ QualityController initialized")
        
        profile_manager = OperatorProfileManager()
        print("✅ OperatorProfileManager initialized")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n⚡ Testing basic functionality...")
    
    try:
        from engines.mathematics import MathematicsEngine
        from engines.data_analysis import DataAnalysisEngine
        
        # Test math engine
        math_engine = MathematicsEngine()
        result = math_engine.solve_problem("2 + 2", {})
        if result.get('success'):
            print("✅ Mathematics engine working")
        else:
            print("⚠️ Mathematics engine has issues")
        
        # Test data analysis engine
        data_engine = DataAnalysisEngine()
        result = data_engine.analyze_data([1, 2, 3, 4, 5])
        if result.get('success'):
            print("✅ Data analysis engine working")
        else:
            print("⚠️ Data analysis engine has issues")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎭 MAESTRO Protocol Quick Test")
    print("=" * 40)
    
    success = True
    
    success &= test_imports()
    success &= test_initialization()
    success &= test_basic_functionality()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! MAESTRO Protocol is ready!")
        print("🚀 You can now run: python scripts/start_server.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 