# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Protocol: Meta-Agent Ensemble for Systematic Task Reasoning and Orchestration

Transform any LLM into superintelligent AI through advanced orchestration,
quality verification, and automated workflow management.

Core Principle: Intelligence Amplification > Model Scale
"""

__version__ = "1.0.0"
__author__ = "tanukimcp"
__license__ = "Non-Commercial License - Commercial use requires TanukiMCP approval"

# Remove direct imports to implement lazy loading
# Instead, provide functions to get these components when needed

def get_orchestrator():
    """Get the MAESTROOrchestrator lazily - only imports when called."""
    from .maestro import MAESTROOrchestrator
    return MAESTROOrchestrator

def get_intelligence_amplifier():
    """Get the IntelligenceAmplifier lazily - only imports when called."""
    from .engines import get_intelligence_amplifier
    return get_intelligence_amplifier()

def get_operator_profile_factory():
    """Get the OperatorProfileFactory lazily - only imports when called."""
    from .profiles import OperatorProfileFactory
    return OperatorProfileFactory

# Define __all__ to list available components
__all__ = [
    "get_orchestrator",
    "get_intelligence_amplifier", 
    "get_operator_profile_factory",
    "__version__",
    "__author__",
    "__license__"
] 
