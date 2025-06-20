# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

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

import logging
import importlib.util
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Lazy loading implementation
_loaded_engines = {}

def _lazy_import(module_name, class_name):
    """Lazy import function that only imports when the class is actually needed."""
    global _loaded_engines
    
    if class_name in _loaded_engines:
        return _loaded_engines[class_name]
    
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        cls = getattr(module, class_name)
        _loaded_engines[class_name] = cls
        return cls
    except Exception as e:
        logger.warning(f"Failed to import {class_name} from {module_name}: {e}")
        return None

# Lazy class loaders
def get_mathematics_engine():
    return _lazy_import("mathematics", "MathematicsEngine")

def get_language_enhancement_engine():
    return _lazy_import("language", "LanguageEnhancementEngine")

def get_code_quality_engine():
    return _lazy_import("code_quality", "CodeQualityEngine")

def get_web_verification_engine():
    return _lazy_import("web_verification", "WebVerificationEngine")

def get_data_analysis_engine():
    return _lazy_import("data_analysis", "DataAnalysisEngine")

def get_grammar_engine():
    return _lazy_import("grammar", "GrammarEngine")

def get_apa_citation_engine():
    return _lazy_import("apa_citation", "APACitationEngine")

def get_intelligence_amplifier():
    return _lazy_import("intelligence_amplifier", "IntelligenceAmplificationEngine")

def get_quantum_physics_engine():
    return _lazy_import("quantum_physics_engine", "QuantumPhysicsEngine")

def get_scientific_computing_engine():
    return _lazy_import("scientific_computing_engine", "ScientificComputingEngine")

def get_language_arts_engine():
    return _lazy_import("language_arts_engine", "LanguageArtsEngine")

def get_medbio_engine():
    return _lazy_import("medbio_engine", "MedBioEngine")

def get_genomics_engine():
    return _lazy_import("genomics_engine", "GenomicsEngine")

def get_epidemiology_engine():
    return _lazy_import("epidemiology_engine", "EpidemiologyEngine")

def get_molecular_dynamics_engine():
    return _lazy_import("molecular_dynamics_engine", "MolecularDynamicsEngine")

def get_particle_physics_engine():
    return _lazy_import("particle_physics_engine", "ParticlePhysicsEngine")

def get_astrophysics_engine():
    return _lazy_import("astrophysics_engine", "AstrophysicsEngine")

def get_scientific_visualization_engine():
    return _lazy_import("scientific_visualization_engine", "ScientificVisualizationEngine")

def get_vet_nutrition_rag_engine():
    return _lazy_import("vet_nutrition_rag_engine", "VeterinaryNutritionRAGEngine")

def get_vet_rad_onc_engine():
    return _lazy_import("vet_rad_onc_engine", "VeterinaryRadiationOncologyEngine")

# Define the IntelligenceAmplifier as a property for backward compatibility
class LazyEngineLoader:
    @property
    def IntelligenceAmplifier(self):
        return get_intelligence_amplifier()
        
# Create instance for backward compatibility - but it won't load anything until accessed
_engine_instance = LazyEngineLoader()
IntelligenceAmplifier = _engine_instance.IntelligenceAmplifier

# Only define the function names in __all__, not the actual imports
__all__ = [
    'get_mathematics_engine',
    'get_language_enhancement_engine', 
    'get_code_quality_engine',
    'get_web_verification_engine',
    'get_data_analysis_engine',
    'get_grammar_engine',
    'get_apa_citation_engine',
    'get_intelligence_amplifier',
    'get_quantum_physics_engine',
    'get_scientific_computing_engine',
    'get_language_arts_engine',
    'get_medbio_engine',
    'get_genomics_engine',
    'get_epidemiology_engine',
    'get_molecular_dynamics_engine',
    'get_particle_physics_engine',
    'get_astrophysics_engine',
    'get_scientific_visualization_engine',
    'get_vet_nutrition_rag_engine',
    'get_vet_rad_onc_engine',
    'IntelligenceAmplifier'
] 
