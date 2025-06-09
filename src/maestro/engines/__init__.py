# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Lazy loader for MIA (MAESTRO Intelligence Amplification) Protocol engines
"""

__all__ = [
    "quantum_physics",
    "intelligence_amplification",
    "scientific_computing",
    "language_arts",
    "medbio",
    "genomics",
    "epidemiology",
    "molecular_dynamics",
    "particle_physics",
    "astrophysics",
    "scientific_visualization",
    "vet_nutrition_rag",
    "vet_rad_onc",
    "basic_arithmetic_engine"
]

def get_quantum_physics_engine():
    """Lazy loader for QuantumPhysicsEngine to improve startup performance."""
    try:
        from .quantum_physics import QuantumPhysicsEngine
        return QuantumPhysicsEngine
    except ImportError:
        return None

def get_basic_arithmetic_engine():
    """Lazy loader for the BasicArithmeticEngine to improve startup performance."""
    try:
        from .basic_arithmetic_engine import BasicArithmeticEngine
        return BasicArithmeticEngine
    except ImportError:
        return None

def get_intelligence_amplifier():
    """Lazy loader for IntelligenceAmplificationEngine to improve startup performance."""
    try:
        from .intelligence_amplification import IntelligenceAmplificationEngine
        return IntelligenceAmplificationEngine
    except ImportError:
        return None

def get_scientific_computing_engine():
    """Lazy loader for ScientificComputingEngine to improve startup performance."""
    try:
        from .scientific_computing import ScientificComputingEngine
        return ScientificComputingEngine
    except ImportError:
        return None

def get_language_arts_engine():
    """Lazy loader for LanguageArtsEngine to improve startup performance."""
    try:
        from .language_arts import LanguageArtsEngine
        return LanguageArtsEngine
    except ImportError:
        return None

def get_medbio_engine():
    """Lazy loader for MedBioEngine to improve startup performance."""
    try:
        from .medbio import MedBioEngine
        return MedBioEngine
    except ImportError:
        return None

def get_genomics_engine():
    """Lazy loader for GenomicsEngine to improve startup performance."""
    try:
        from .genomics import GenomicsEngine
        return GenomicsEngine
    except ImportError:
        return None

def get_epidemiology_engine():
    """Lazy loader for EpidemiologyEngine to improve startup performance."""
    try:
        from .epidemiology import EpidemiologyEngine
        return EpidemiologyEngine
    except ImportError:
        return None

def get_molecular_dynamics_engine():
    """Lazy loader for MolecularDynamicsEngine to improve startup performance."""
    try:
        from .molecular_dynamics import MolecularDynamicsEngine
        return MolecularDynamicsEngine
    except ImportError:
        return None

def get_particle_physics_engine():
    """Lazy loader for ParticlePhysicsEngine to improve startup performance."""
    try:
        from .particle_physics import ParticlePhysicsEngine
        return ParticlePhysicsEngine
    except ImportError:
        return None

def get_astrophysics_engine():
    """Lazy loader for AstrophysicsEngine to improve startup performance."""
    try:
        from .astrophysics import AstrophysicsEngine
        return AstrophysicsEngine
    except ImportError:
        return None

def get_scientific_visualization_engine():
    """Lazy loader for ScientificVisualizationEngine to improve startup performance."""
    try:
        from .scientific_visualization import ScientificVisualizationEngine
        return ScientificVisualizationEngine
    except ImportError:
        return None

def get_vet_nutrition_rag_engine():
    """Lazy loader for VetNutritionRAG to improve startup performance."""
    try:
        from .vet_nutrition_rag import VetNutritionRAG
        return VetNutritionRAG
    except ImportError:
        return None

def get_vet_rad_onc_engine():
    """Lazy loader for VetRadOncEngine to improve startup performance."""
    try:
        from .vet_rad_onc import VetRadOncEngine
        return VetRadOncEngine
    except ImportError:
        return None 