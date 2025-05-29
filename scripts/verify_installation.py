#!/usr/bin/env python3
"""
MAESTRO Protocol Installation Verification Script

Verifies that the MAESTRO Protocol is correctly installed and functional.
"""

import sys
import os
import asyncio
import time
import traceback
from typing import Dict, List, Tuple, Any

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class MAESTROVerifier:
    """Comprehensive verification of MAESTRO Protocol installation"""
    
    def __init__(self):
        self.results = {}
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Run a single test and record results"""
        self.test_count += 1
        print(f"🧪 Running: {test_name}")
        
        try:
            result = test_func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            
            if result:
                print(f"  ✅ PASSED: {test_name}")
                self.passed_count += 1
                self.results[test_name] = {'status': 'PASSED', 'error': None}
                return True
            else:
                print(f"  ❌ FAILED: {test_name}")
                self.failed_count += 1
                self.results[test_name] = {'status': 'FAILED', 'error': 'Test returned False'}
                return False
                
        except Exception as e:
            print(f"  ❌ ERROR: {test_name} - {str(e)}")
            self.failed_count += 1
            self.results[test_name] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_core_imports(self) -> bool:
        """Test that core MAESTRO components can be imported"""
        try:
            from maestro import MAESTROOrchestrator
            from maestro.data_models import MAESTROResult, TaskAnalysis
            from maestro.quality_controller import QualityController
            return True
        except ImportError as e:
            print(f"    Import error: {e}")
            return False
    
    def test_engine_imports(self) -> bool:
        """Test that intelligence amplification engines can be imported"""
        try:
            from engines import IntelligenceAmplifier
            from engines.mathematics import MathematicsEngine
            from engines.language import LanguageEnhancementEngine
            from engines.code_quality import CodeQualityEngine
            return True
        except ImportError as e:
            print(f"    Engine import error: {e}")
            return False
    
    def test_profile_imports(self) -> bool:
        """Test that operator profiles can be imported"""
        try:
            from profiles.operator_profiles import OperatorProfileManager, OperatorProfile
            return True
        except ImportError as e:
            print(f"    Profile import error: {e}")
            return False
    
    def test_main_server_import(self) -> bool:
        """Test that the main MCP server can be imported"""
        try:
            from main import TanukiMCPOrchestra
            return True
        except ImportError as e:
            print(f"    Main server import error: {e}")
            return False
    
    def test_orchestrator_initialization(self) -> bool:
        """Test MAESTRO orchestrator initialization"""
        try:
            from maestro import MAESTROOrchestrator
            orchestrator = MAESTROOrchestrator()
            return hasattr(orchestrator, 'operator_factory') and hasattr(orchestrator, 'intelligence_amplifier')
        except Exception as e:
            print(f"    Orchestrator init error: {e}")
            return False
    
    def test_intelligence_amplifier_initialization(self) -> bool:
        """Test intelligence amplifier initialization"""
        try:
            from engines import IntelligenceAmplifier
            amplifier = IntelligenceAmplifier()
            return len(amplifier.engines) > 0
        except Exception as e:
            print(f"    Intelligence amplifier init error: {e}")
            return False
    
    def test_mathematics_engine(self) -> bool:
        """Test mathematics engine basic functionality"""
        try:
            from engines.mathematics import MathematicsEngine
            engine = MathematicsEngine()
            result = engine.solve_problem("2 + 2", {})
            return result.get('success', False)
        except Exception as e:
            print(f"    Mathematics engine error: {e}")
            return False
    
    def test_language_engine(self) -> bool:
        """Test language enhancement engine"""
        try:
            from engines.language import LanguageEnhancementEngine
            engine = LanguageEnhancementEngine()
            # Test with a simple async call
            async def test_language():
                result = await engine.enhance_language_quality("This is a test sentence.")
                return "Enhancement Report" in result
            return asyncio.run(test_language())
        except Exception as e:
            print(f"    Language engine error: {e}")
            return False
    
    def test_code_quality_engine(self) -> bool:
        """Test code quality engine"""
        try:
            from engines.code_quality import CodeQualityEngine
            engine = CodeQualityEngine()
            # Test with a simple async call
            async def test_code():
                result = await engine.analyze_and_improve_code("print('hello world')")
                return result.get('status') == 'Code quality analysis complete'
            return asyncio.run(test_code())
        except Exception as e:
            print(f"    Code quality engine error: {e}")
            return False
    
    def test_quality_controller(self) -> bool:
        """Test quality controller functionality"""
        try:
            from maestro.quality_controller import QualityController
            controller = QualityController()
            return hasattr(controller, 'verification_suite')
        except Exception as e:
            print(f"    Quality controller error: {e}")
            return False
    
    def test_operator_profiles(self) -> bool:
        """Test operator profile management"""
        try:
            from profiles.operator_profiles import OperatorProfileManager
            manager = OperatorProfileManager()
            profiles = manager.get_all_profiles()
            return len(profiles) > 0
        except Exception as e:
            print(f"    Operator profiles error: {e}")
            return False
    
    async def test_basic_orchestration(self) -> bool:
        """Test basic orchestration workflow"""
        try:
            from maestro import MAESTROOrchestrator
            orchestrator = MAESTROOrchestrator()
            
            # Test task analysis
            analysis = await orchestrator.analyze_task_complexity("Calculate 2 + 2")
            return hasattr(analysis, 'task_type') and hasattr(analysis, 'complexity')
        except Exception as e:
            print(f"    Basic orchestration error: {e}")
            return False
    
    def test_mcp_server_initialization(self) -> bool:
        """Test MCP server can be initialized"""
        try:
            from main import TanukiMCPOrchestra
            server = TanukiMCPOrchestra()
            return hasattr(server, 'orchestrator') and hasattr(server, 'app')
        except Exception as e:
            print(f"    MCP server init error: {e}")
            return False
    
    def test_data_models(self) -> bool:
        """Test data model creation and functionality"""
        try:
            from maestro.data_models import MAESTROResult, TaskAnalysis, Workflow, QualityMetrics
            from maestro.data_models import TaskType, ComplexityLevel
            
            # Test enum values
            task_type = TaskType.MATHEMATICS
            complexity = ComplexityLevel.MODERATE
            
            # Test data model creation
            metrics = QualityMetrics(overall_score=0.9)
            analysis = TaskAnalysis(
                task_type=task_type,
                complexity=complexity,
                capabilities=["math"],
                estimated_duration=60,
                required_tools=["calculator"],
                success_criteria=["correct result"],
                quality_requirements={"accuracy": 0.95}
            )
            
            return metrics.meets_threshold(0.8) and analysis.task_type == task_type
        except Exception as e:
            print(f"    Data models error: {e}")
            return False
    
    def run_performance_test(self) -> bool:
        """Run a basic performance test"""
        try:
            start_time = time.time()
            
            # Test multiple engine initializations
            from engines import IntelligenceAmplifier
            from maestro import MAESTROOrchestrator
            
            for _ in range(3):
                amplifier = IntelligenceAmplifier()
                orchestrator = MAESTROOrchestrator()
            
            duration = time.time() - start_time
            print(f"    Performance: Initialized components in {duration:.2f}s")
            
            return duration < 5.0  # Should complete in under 5 seconds
        except Exception as e:
            print(f"    Performance test error: {e}")
            return False
    
    def generate_verification_report(self):
        """Generate a comprehensive verification report"""
        print("\n" + "="*60)
        print("🎭 MAESTRO Protocol Installation Verification Report")
        print("="*60)
        
        print(f"\n📊 Test Results Summary:")
        print(f"  Total Tests: {self.test_count}")
        print(f"  ✅ Passed: {self.passed_count}")
        print(f"  ❌ Failed: {self.failed_count}")
        print(f"  Success Rate: {(self.passed_count/self.test_count*100):.1f}%")
        
        # Detailed results
        if self.failed_count > 0:
            print(f"\n❌ Failed Tests:")
            for test_name, result in self.results.items():
                if result['status'] != 'PASSED':
                    print(f"  - {test_name}: {result['status']}")
                    if result['error']:
                        print(f"    Error: {result['error']}")
        
        # Overall assessment
        if self.failed_count == 0:
            print(f"\n🎉 EXCELLENT! MAESTRO Protocol is fully functional!")
            print(f"✅ All systems operational and ready for superintelligent AI orchestration!")
        elif self.failed_count <= 2:
            print(f"\n✅ GOOD! MAESTRO Protocol is mostly functional!")
            print(f"⚠️  Some optional features may be limited due to missing dependencies.")
        else:
            print(f"\n⚠️  WARNING! MAESTRO Protocol has significant issues!")
            print(f"🔧 Please check the failed tests and resolve dependencies.")
        
        print(f"\n💡 Next Steps:")
        if self.failed_count == 0:
            print(f"  🚀 Start the server: python src/main.py")
            print(f"  📚 Check examples: python examples/basic_usage.py")
            print(f"  🧪 Run tests: python -m pytest tests/")
        else:
            print(f"  🔧 Run configuration: python scripts/configure_maestro.py")
            print(f"  📦 Install missing dependencies")
            print(f"  🔄 Re-run verification: python scripts/verify_installation.py")
    
    def run_full_verification(self):
        """Run the complete verification suite"""
        print("🎭 MAESTRO Protocol Installation Verification")
        print("🔍 Testing all components...\n")
        
        # Core import tests
        self.run_test("Core MAESTRO imports", self.test_core_imports)
        self.run_test("Engine imports", self.test_engine_imports)
        self.run_test("Profile imports", self.test_profile_imports)
        self.run_test("Main server import", self.test_main_server_import)
        
        # Initialization tests
        self.run_test("Orchestrator initialization", self.test_orchestrator_initialization)
        self.run_test("Intelligence amplifier initialization", self.test_intelligence_amplifier_initialization)
        self.run_test("Quality controller initialization", self.test_quality_controller)
        self.run_test("MCP server initialization", self.test_mcp_server_initialization)
        
        # Engine functionality tests
        self.run_test("Mathematics engine", self.test_mathematics_engine)
        self.run_test("Language engine", self.test_language_engine)
        self.run_test("Code quality engine", self.test_code_quality_engine)
        
        # Data model tests
        self.run_test("Data models", self.test_data_models)
        self.run_test("Operator profiles", self.test_operator_profiles)
        
        # Integration tests
        self.run_test("Basic orchestration", self.test_basic_orchestration)
        self.run_test("Performance test", self.run_performance_test)
        
        # Generate report
        self.generate_verification_report()


def main():
    """Main verification entry point"""
    verifier = MAESTROVerifier()
    verifier.run_full_verification()


if __name__ == "__main__":
    main() 