"""
Intelligence Amplification Engines for Maestro

This package contains specialized engines that amplify specific capabilities
beyond base model performance through integration with Python libraries
and automated verification systems.

Available Engines:
- MathematicsEngine: Advanced mathematical reasoning and computation
- LanguageEnhancementEngine: Language quality and style analysis
- CodeQualityEngine: Code analysis, review, and improvement
- WebVerificationEngine: Web content verification and analysis
- DataAnalysisEngine: Comprehensive data analysis and insights
- GrammarEngine: Advanced grammar checking and writing enhancement
- APACitationEngine: APA 7th edition citation formatting and validation
- IntelligenceAmplifier: Main coordinator for all engines
"""

from .mathematics import MathematicsEngine
from .language import LanguageEnhancementEngine
from .code_quality import CodeQualityEngine
from .web_verification import WebVerificationEngine
from .data_analysis import DataAnalysisEngine
from .grammar import GrammarEngine
from .apa_citation import APACitationEngine
from .intelligence_amplifier import IntelligenceAmplificationEngine

# Create alias for backward compatibility
IntelligenceAmplifier = IntelligenceAmplificationEngine

__all__ = [
    'MathematicsEngine',
    'LanguageEnhancementEngine', 
    'CodeQualityEngine',
    'WebVerificationEngine',
    'DataAnalysisEngine',
    'GrammarEngine',
    'APACitationEngine',
    'IntelligenceAmplifier'
] 