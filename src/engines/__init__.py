"""
Intelligence Amplification Engines for MAESTRO Protocol

This package contains specialized engines that amplify specific capabilities
beyond base model performance through integration with Python libraries
and automated verification systems.

Available Engines:
- MathematicsEngine: Advanced mathematical reasoning and computation
- LanguageEnhancementEngine: Language quality and style analysis
- CodeQualityEngine: Code analysis, review, and improvement
- WebVerificationEngine: Web content verification and analysis
- DataAnalysisEngine: Comprehensive data analysis and insights
- IntelligenceAmplifier: Main coordinator for all engines
"""

from .mathematics import MathematicsEngine
from .language import LanguageEnhancementEngine
from .code_quality import CodeQualityEngine
from .web_verification import WebVerificationEngine
from .data_analysis import DataAnalysisEngine
from .intelligence_amplifier import IntelligenceAmplifier, AmplificationResult

__all__ = [
    'MathematicsEngine',
    'LanguageEnhancementEngine', 
    'CodeQualityEngine',
    'WebVerificationEngine',
    'DataAnalysisEngine',
    'IntelligenceAmplifier',
    'AmplificationResult'
] 