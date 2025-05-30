"""
Operator Profiles for MAESTRO Protocol

This package contains specialized AI operator profiles that define
different personas optimized for specific task types and complexity levels.
Each profile includes custom system prompts, capabilities, and behavioral parameters.

Implements lazy loading for Smithery compatibility.
"""

# Import lightweight enums directly
from .operator_profiles import OperatorType, ComplexityLevel

# Implement lazy loading functions for heavier components
def get_operator_profile():
    """Get OperatorProfile class lazily."""
    from .operator_profiles import OperatorProfile
    return OperatorProfile

def get_operator_profile_manager():
    """Get OperatorProfileManager class lazily."""
    from .operator_profiles import OperatorProfileManager
    return OperatorProfileManager

def get_operator_profile_factory():
    """Get OperatorProfileFactory class lazily."""
    from .operator_profiles import OperatorProfileFactory
    return OperatorProfileFactory

# Define module exports - both direct lightweight and lazy loaded
__all__ = [
    # Lazy loaded components - functions
    'get_operator_profile',
    'get_operator_profile_manager',
    'get_operator_profile_factory',
    
    # Direct imports - lightweight enums
    'OperatorType',
    'ComplexityLevel'
] 