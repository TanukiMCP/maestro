# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Protocol Core Orchestration Components

This module implements lazy loading to ensure Smithery compatibility.
"""

# Import data models directly as they are lightweight
from .data_models import (
    MAESTROResult,
    TaskAnalysis,
    Workflow,
    WorkflowNode,
    VerificationResult,
    QualityMetrics
)

# Implement lazy loading for heavier components
def get_orchestrator():
    """Get MAESTROOrchestrator lazily."""
    from .orchestrator import MAESTROOrchestrator
    return MAESTROOrchestrator

def get_quality_controller():
    """Get QualityController lazily."""
    from .quality_controller import QualityController
    return QualityController

# Define module exports
__all__ = [
    # Lazy loaded components - functions
    "get_orchestrator",
    "get_quality_controller",
    
    # Direct imports - lightweight data models
    "MAESTROResult",
    "TaskAnalysis",
    "Workflow",
    "WorkflowNode",
    "VerificationResult",
    "QualityMetrics"
]

# This file makes src.maestro a Python package 
