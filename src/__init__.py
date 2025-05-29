"""
MAESTRO Protocol: Meta-Agent Ensemble for Systematic Task Reasoning and Orchestration

Transform any LLM into superintelligent AI through advanced orchestration,
quality verification, and automated workflow management.

Core Principle: Intelligence Amplification > Model Scale
"""

__version__ = "1.0.0"
__author__ = "tanukimcp"
__license__ = "MIT"

from .maestro import MAESTROOrchestrator
from .engines import IntelligenceAmplifier
from .profiles import OperatorProfileFactory

__all__ = [
    "MAESTROOrchestrator",
    "IntelligenceAmplifier", 
    "OperatorProfileFactory"
] 