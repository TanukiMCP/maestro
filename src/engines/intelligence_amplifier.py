"""
Intelligence Amplification Engine for MAESTRO Protocol
Coordinates specialized engines to amplify specific capabilities beyond base model performance.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import specialized engines
from .mathematics import MathematicsEngine
from .language import LanguageEnhancementEngine
from .code_quality import CodeQualityEngine
from .web_verification import WebVerificationEngine
from .data_analysis import DataAnalysisEngine

logger = logging.getLogger(__name__)

@dataclass
class AmplificationResult:
    """Result of intelligence amplification"""
    success: bool
    capability: str
    original_input: str
    amplified_output: Dict[str, Any]
    confidence_score: float
    processing_time: float
    engine_used: str
    recommendations: List[str]
    error_message: Optional[str] = None

class IntelligenceAmplifier:
    """
    Main coordinator for intelligence amplification across multiple domains
    """
    
    def __init__(self):
        # Initialize specialized engines
        self.engines = {
            'mathematics': MathematicsEngine(),
            'language': LanguageEnhancementEngine(),
            'code_quality': CodeQualityEngine(),
            'web_verification': WebVerificationEngine(),
            'data_analysis': DataAnalysisEngine()
        }
        
        # Capability mappings
        self.capability_mappings = {
            'mathematical_reasoning': 'mathematics',
            'calculation': 'mathematics',
            'equation_solving': 'mathematics',
            'statistical_analysis': 'data_analysis',
            'data_analysis': 'data_analysis',
            'pattern_recognition': 'data_analysis',
            'language_enhancement': 'language',
            'grammar_checking': 'language',
            'style_analysis': 'language',
            'code_review': 'code_quality',
            'code_analysis': 'code_quality',
            'syntax_validation': 'code_quality',
            'web_verification': 'web_verification',
            'html_analysis': 'web_verification',
            'accessibility_check': 'web_verification'
        }
        
        # Engine status tracking
        self.engine_status = {engine: True for engine in self.engines.keys()}
    
    async def amplify_capability(self, capability: str, input_data: str, context: Dict[str, Any] = None) -> AmplificationResult:
        """
        Amplify a specific capability using the appropriate specialized engine
        
        Args:
            capability: The capability to amplify
            input_data: Input data for processing
            context: Additional context for processing
            
        Returns:
            AmplificationResult with enhanced output
        """
        import time
        start_time = time.time()
        
        try:
            # Determine which engine to use
            engine_name = self._select_engine(capability)
            if not engine_name:
                return self._create_error_result(
                    capability, input_data, start_time,
                    f"No suitable engine found for capability: {capability}"
                )
            
            # Check engine availability
            if not self.engine_status.get(engine_name, False):
                return self._create_error_result(
                    capability, input_data, start_time,
                    f"Engine '{engine_name}' is currently unavailable"
                )
            
            # Get the engine
            engine = self.engines[engine_name]
            
            # Process based on capability and engine
            result = await self._process_with_engine_async(engine, engine_name, capability, input_data, context)
            
            processing_time = time.time() - start_time
            
            # Create amplification result
            return AmplificationResult(
                success=result.get('success', False),
                capability=capability,
                original_input=input_data,
                amplified_output=result,
                confidence_score=result.get('confidence_score', 0.8),
                processing_time=processing_time,
                engine_used=engine_name,
                recommendations=result.get('recommendations', []),
                error_message=result.get('error') if not result.get('success', False) else None
            )
            
        except Exception as e:
            logger.error(f"Intelligence amplification failed: {str(e)}")
            return self._create_error_result(
                capability, input_data, start_time,
                f"Amplification failed: {str(e)}"
            )
    
    def _select_engine(self, capability: str) -> Optional[str]:
        """Select the appropriate engine for a capability"""
        # Direct mapping
        if capability in self.capability_mappings:
            return self.capability_mappings[capability]
        
        # Fuzzy matching based on keywords
        capability_lower = capability.lower()
        
        # Mathematics keywords
        math_keywords = ['math', 'calculate', 'equation', 'algebra', 'calculus', 'geometry', 'statistics', 'number']
        if any(keyword in capability_lower for keyword in math_keywords):
            return 'mathematics'
        
        # Data analysis keywords
        data_keywords = ['data', 'analysis', 'pattern', 'trend', 'correlation', 'outlier', 'distribution']
        if any(keyword in capability_lower for keyword in data_keywords):
            return 'data_analysis'
        
        # Language keywords
        language_keywords = ['language', 'grammar', 'style', 'writing', 'text', 'readability']
        if any(keyword in capability_lower for keyword in language_keywords):
            return 'language'
        
        # Code keywords
        code_keywords = ['code', 'programming', 'syntax', 'function', 'class', 'variable', 'algorithm']
        if any(keyword in capability_lower for keyword in code_keywords):
            return 'code_quality'
        
        # Web keywords
        web_keywords = ['web', 'html', 'css', 'javascript', 'website', 'accessibility', 'seo']
        if any(keyword in capability_lower for keyword in web_keywords):
            return 'web_verification'
        
        return None
    
    async def _process_with_engine_async(self, engine: Any, engine_name: str, capability: str, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with the selected engine using async methods when possible"""
        try:
            if engine_name == 'mathematics':
                # Use the async method directly
                result = await engine.process_mathematical_task(input_data, context or {})
                return {
                    'success': True,
                    'solution': result,
                    'problem': input_data,
                    'method': 'mathematical_computation'
                }
            
            elif engine_name == 'data_analysis':
                # Data analysis is sync, so call it directly
                analysis_type = context.get('analysis_type', 'comprehensive') if context else 'comprehensive'
                result = engine.analyze_data(input_data, analysis_type)
                return result
            
            elif engine_name == 'language':
                # Use the async method directly
                result = await engine.enhance_language_quality(input_data, context or {})
                return {
                    'success': True,
                    'enhanced_text': result,
                    'original_text': input_data,
                    'enhancement_type': context.get('enhancement_type', 'comprehensive') if context else 'comprehensive'
                }
            
            elif engine_name == 'code_quality':
                # Use the async method directly
                result = await engine.analyze_and_improve_code(input_data, context or {})
                return {
                    'success': True,
                    'analysis': result,
                    'language': context.get('language', 'python') if context else 'python',
                    'confidence_score': 0.85
                }
            
            elif engine_name == 'web_verification':
                # Use the async method directly
                verification_type = context.get('verification_type', 'comprehensive') if context else 'comprehensive'
                result = await engine.verify_web_content(input_data, context or {})
                return result
            
            else:
                return {
                    'success': False,
                    'error': f"Unknown engine: {engine_name}"
                }
                
        except Exception as e:
            logger.error(f"Engine processing failed: {str(e)}")
            return {
                'success': False,
                'error': f"Engine processing failed: {str(e)}"
            }
    
    def get_available_capabilities(self) -> Dict[str, List[str]]:
        """Get list of available capabilities by engine"""
        capabilities = {}
        
        for engine_name, available in self.engine_status.items():
            if available:
                if engine_name == 'mathematics':
                    capabilities[engine_name] = [
                        'mathematical_reasoning', 'calculation', 'equation_solving',
                        'algebra', 'calculus', 'geometry', 'statistics', 'number_theory'
                    ]
                elif engine_name == 'data_analysis':
                    capabilities[engine_name] = [
                        'data_analysis', 'statistical_analysis', 'pattern_recognition',
                        'trend_analysis', 'correlation_analysis', 'outlier_detection'
                    ]
                elif engine_name == 'language':
                    capabilities[engine_name] = [
                        'language_enhancement', 'grammar_checking', 'style_analysis',
                        'readability_analysis', 'sentiment_analysis'
                    ]
                elif engine_name == 'code_quality':
                    capabilities[engine_name] = [
                        'code_review', 'code_analysis', 'syntax_validation',
                        'style_checking', 'security_analysis'
                    ]
                elif engine_name == 'web_verification':
                    capabilities[engine_name] = [
                        'web_verification', 'html_analysis', 'accessibility_check',
                        'seo_analysis', 'performance_analysis'
                    ]
        
        return capabilities
    
    def get_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all engines"""
        status = {}
        
        for engine_name, engine in self.engines.items():
            try:
                # Test engine availability
                test_result = self._test_engine(engine, engine_name)
                status[engine_name] = {
                    'available': test_result['available'],
                    'capabilities': test_result['capabilities'],
                    'dependencies': test_result['dependencies'],
                    'last_tested': test_result['timestamp']
                }
            except Exception as e:
                status[engine_name] = {
                    'available': False,
                    'error': str(e),
                    'capabilities': [],
                    'dependencies': 'unknown'
                }
        
        return status
    
    def _test_engine(self, engine: Any, engine_name: str) -> Dict[str, Any]:
        """Test engine availability and capabilities"""
        import datetime
        
        try:
            # Simple test based on engine type
            if engine_name == 'mathematics':
                test_result = engine.solve_problem("2 + 2", {})
                available = test_result.get('success', False)
                capabilities = ['basic_arithmetic', 'algebra', 'calculus', 'statistics']
                dependencies = 'sympy, numpy, scipy (optional)'
                
            elif engine_name == 'data_analysis':
                test_result = engine.analyze_data([1, 2, 3, 4, 5])
                available = test_result.get('success', False)
                capabilities = ['statistical_analysis', 'pattern_recognition', 'data_quality']
                dependencies = 'pandas, numpy, scipy, matplotlib (optional)'
                
            elif engine_name == 'language':
                test_result = engine.enhance_text("Test text.")
                available = test_result.get('success', False)
                capabilities = ['grammar_check', 'style_analysis', 'readability']
                dependencies = 'spacy, textstat, nltk (optional)'
                
            elif engine_name == 'code_quality':
                test_result = engine.analyze_code("print('hello')")
                available = test_result.get('success', False)
                capabilities = ['syntax_validation', 'style_analysis', 'security_check']
                dependencies = 'ast, black (optional)'
                
            elif engine_name == 'web_verification':
                test_result = engine.verify_web_content("<html><body>Test</body></html>")
                available = test_result.get('success', False)
                capabilities = ['html_validation', 'accessibility_check', 'seo_analysis']
                dependencies = 'beautifulsoup4, requests (optional)'
                
            else:
                available = False
                capabilities = []
                dependencies = 'unknown'
            
            return {
                'available': available,
                'capabilities': capabilities,
                'dependencies': dependencies,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'available': False,
                'capabilities': [],
                'dependencies': 'unknown',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    def _create_error_result(self, capability: str, input_data: str, start_time: float, error_message: str) -> AmplificationResult:
        """Create an error result"""
        import time
        processing_time = time.time() - start_time
        
        return AmplificationResult(
            success=False,
            capability=capability,
            original_input=input_data,
            amplified_output={'error': error_message},
            confidence_score=0.0,
            processing_time=processing_time,
            engine_used='none',
            recommendations=[
                'Check capability name and input format',
                'Verify required dependencies are installed',
                'Try a different capability or engine'
            ],
            error_message=error_message
        ) 