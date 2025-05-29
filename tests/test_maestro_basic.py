#!/usr/bin/env python3
"""
Basic Tests for MAESTRO Protocol MCP Server

Tests core functionality including orchestration, intelligence amplification,
and quality control systems.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maestro.orchestrator import MAESTROOrchestrator
from maestro.quality_controller import QualityController
from maestro.data_models import TaskType, ComplexityLevel, VerificationMethod
from engines.intelligence_amplifier import IntelligenceAmplifier
from engines.mathematics import MathematicsEngine
from engines.data_analysis import DataAnalysisEngine
from profiles.operator_profiles import OperatorProfileManager

class TestMAESTROBasic(unittest.TestCase):
    """Basic tests for MAESTRO Protocol components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = MAESTROOrchestrator()
        self.quality_controller = QualityController()
        self.intelligence_amplifier = IntelligenceAmplifier()
        self.profile_manager = OperatorProfileManager()
    
    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly"""
        self.assertIsNotNone(self.orchestrator)
        self.assertIsNotNone(self.orchestrator.profile_manager)
        self.assertIsNotNone(self.orchestrator.quality_controller)
        self.assertIsNotNone(self.orchestrator.intelligence_amplifier)
    
    def test_task_analysis(self):
        """Test task analysis functionality"""
        task = "Calculate the derivative of x^2 + 3x + 1"
        
        analysis = self.orchestrator._analyze_task(task)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.task_type, TaskType.MATHEMATICAL)
        self.assertIn(analysis.complexity_level, [ComplexityLevel.BASIC, ComplexityLevel.INTERMEDIATE])
        self.assertGreater(len(analysis.required_capabilities), 0)
    
    def test_profile_selection(self):
        """Test operator profile selection"""
        # Test mathematical task
        math_profile = self.profile_manager.select_profile(
            task_type="data_analysis",
            complexity_level="basic"
        )
        self.assertIsNotNone(math_profile)
        self.assertEqual(math_profile.name, "Basic Analyst")
        
        # Test technical task
        tech_profile = self.profile_manager.select_profile(
            task_type="code_review",
            complexity_level="intermediate"
        )
        self.assertIsNotNone(tech_profile)
        self.assertEqual(tech_profile.name, "Technical Specialist")
    
    def test_intelligence_amplification(self):
        """Test intelligence amplification system"""
        # Test mathematical amplification
        result = self.intelligence_amplifier.amplify_capability(
            capability="mathematical_reasoning",
            input_data="2 + 2",
            context={}
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.capability, "mathematical_reasoning")
        self.assertEqual(result.engine_used, "mathematics")
        
        # Test data analysis amplification
        result = self.intelligence_amplifier.amplify_capability(
            capability="data_analysis",
            input_data="[1, 2, 3, 4, 5]",
            context={}
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.capability, "data_analysis")
        self.assertEqual(result.engine_used, "data_analysis")
    
    def test_quality_verification(self):
        """Test quality verification system"""
        # Test mathematical verification
        result = self.quality_controller.verify_quality(
            content="The derivative of x^2 is 2x",
            verification_method=VerificationMethod.MATHEMATICAL,
            context={"original_problem": "Find derivative of x^2"}
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertGreater(result.quality_metrics.accuracy, 0.8)
    
    def test_mathematics_engine(self):
        """Test mathematics engine functionality"""
        engine = MathematicsEngine()
        
        # Test basic arithmetic
        result = engine.solve_problem("2 + 2", {})
        self.assertTrue(result.get('success', False))
        
        # Test equation solving
        result = engine.solve_problem("solve x^2 - 4 = 0", {})
        self.assertTrue(result.get('success', False))
    
    def test_data_analysis_engine(self):
        """Test data analysis engine functionality"""
        engine = DataAnalysisEngine()
        
        # Test simple data analysis
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = engine.analyze_data(test_data)
        
        self.assertTrue(result.get('success', False))
        self.assertIn('statistical_summary', result)
        self.assertIn('data_quality', result)
    
    def test_workflow_generation(self):
        """Test workflow generation"""
        task_analysis = self.orchestrator._analyze_task("Analyze sales data and create a report")
        
        workflow = self.orchestrator._generate_workflow(task_analysis)
        
        self.assertIsNotNone(workflow)
        self.assertGreater(len(workflow.nodes), 0)
        self.assertIsNotNone(workflow.success_criteria)
    
    def test_capability_determination(self):
        """Test capability determination logic"""
        # Mathematical task
        capabilities = self.orchestrator._determine_capabilities(TaskType.MATHEMATICAL, ComplexityLevel.INTERMEDIATE)
        self.assertIn("mathematical_reasoning", capabilities)
        
        # Data analysis task
        capabilities = self.orchestrator._determine_capabilities(TaskType.DATA_ANALYSIS, ComplexityLevel.ADVANCED)
        self.assertIn("data_analysis", capabilities)
        self.assertIn("statistical_analysis", capabilities)
        
        # Code task
        capabilities = self.orchestrator._determine_capabilities(TaskType.CODE_ANALYSIS, ComplexityLevel.INTERMEDIATE)
        self.assertIn("code_analysis", capabilities)
    
    def test_error_handling(self):
        """Test error handling in various components"""
        # Test orchestrator with invalid input
        result = self.orchestrator.orchestrate("", {})
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        
        # Test intelligence amplifier with invalid capability
        result = self.intelligence_amplifier.amplify_capability(
            capability="invalid_capability",
            input_data="test",
            context={}
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

class TestEngineIntegration(unittest.TestCase):
    """Test integration between different engines"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.amplifier = IntelligenceAmplifier()
    
    def test_engine_availability(self):
        """Test that all engines are available"""
        status = self.amplifier.get_engine_status()
        
        self.assertIn('mathematics', status)
        self.assertIn('data_analysis', status)
        self.assertIn('language', status)
        self.assertIn('code_quality', status)
        self.assertIn('web_verification', status)
    
    def test_capability_routing(self):
        """Test that capabilities are routed to correct engines"""
        # Mathematical capabilities
        engine = self.amplifier._select_engine("mathematical_reasoning")
        self.assertEqual(engine, "mathematics")
        
        # Data analysis capabilities
        engine = self.amplifier._select_engine("data_analysis")
        self.assertEqual(engine, "data_analysis")
        
        # Language capabilities
        engine = self.amplifier._select_engine("grammar_checking")
        self.assertEqual(engine, "language")
        
        # Code capabilities
        engine = self.amplifier._select_engine("code_analysis")
        self.assertEqual(engine, "code_quality")
        
        # Web capabilities
        engine = self.amplifier._select_engine("web_verification")
        self.assertEqual(engine, "web_verification")

class TestQualityControl(unittest.TestCase):
    """Test quality control mechanisms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.quality_controller = QualityController()
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        metrics = self.quality_controller._calculate_quality_metrics(
            content="This is a test response with good quality.",
            context={"expected_length": 50}
        )
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.accuracy, 0)
        self.assertGreater(metrics.completeness, 0)
        self.assertGreater(metrics.overall_quality, 0)
    
    def test_verification_methods(self):
        """Test different verification methods"""
        # Test computational verification
        result = self.quality_controller.verify_quality(
            content="2 + 2 = 4",
            verification_method=VerificationMethod.COMPUTATIONAL,
            context={}
        )
        self.assertTrue(result.success)
        
        # Test logical verification
        result = self.quality_controller.verify_quality(
            content="All cats are mammals. Fluffy is a cat. Therefore, Fluffy is a mammal.",
            verification_method=VerificationMethod.LOGICAL,
            context={}
        )
        self.assertTrue(result.success)

def run_basic_tests():
    """Run basic functionality tests"""
    print("🧪 Running MAESTRO Protocol Basic Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestMAESTROBasic))
    suite.addTest(unittest.makeSuite(TestEngineIntegration))
    suite.addTest(unittest.makeSuite(TestQualityControl))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1) 