# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Particle Physics Engine

Provides production-quality computational tools for relativistic kinematics
and particle collision analysis, fundamental to high-energy physics. This engine
leverages the 'vector' library for robust and accurate four-vector calculations.
All functions are fully implemented.
"""

import logging
from typing import Dict, List, Any

import numpy as np

try:
    import vector
except ImportError:
    raise ImportError("The 'vector' library is not installed. Please install it with 'pip install vector'.")

logger = logging.getLogger(__name__)

class ParticlePhysicsEngine:
    """
    Implements computational tools for the particle physics domain.
    """
    
    def __init__(self):
        self.name = "Particle Physics Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            "calculate_invariant_mass",
            "lorentz_boost",
            "analyze_particle_collision"
        ]

    def _create_four_vector(self, particle: Dict[str, float]) -> vector.MomentumNumpy4D:
        """Helper to create a four-vector from a particle dictionary."""
        # Supports multiple formats for specifying a particle's momentum and energy
        if 'px' in particle and 'py' in particle and 'pz' in particle and 'E' in particle:
            return vector.obj(px=particle['px'], py=particle['py'], pz=particle['pz'], E=particle['E'])
        elif 'pt' in particle and 'eta' in particle and 'phi' in particle and 'mass' in particle:
            return vector.obj(pt=particle['pt'], eta=particle['eta'], phi=particle['phi'], mass=particle['mass'])
        else:
            raise ValueError("Particle dictionary must contain ('px', 'py', 'pz', 'E') or ('pt', 'eta', 'phi', 'mass').")

    def calculate_invariant_mass(self, parameters: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Calculates the invariant mass of a system of particles.
        This is a fundamental calculation in experimental particle physics.
        """
        try:
            particles = parameters['particles']
            if len(particles) < 2:
                return {"error": "At least two particles are required to calculate invariant mass."}

            total_vector = self._create_four_vector(particles[0])
            for p_dict in particles[1:]:
                total_vector += self._create_four_vector(p_dict)
            
            invariant_mass = total_vector.mass
            
            return {
                "invariant_mass_gev": float(invariant_mass),
                "total_four_vector": {
                    "px": float(total_vector.px),
                    "py": float(total_vector.py),
                    "pz": float(total_vector.pz),
                    "E": float(total_vector.E)
                },
                "num_particles": len(particles)
            }
        except (KeyError, ValueError) as e:
            return {"error": f"Invalid particle data provided: {e}"}
        except Exception as e:
            logger.error(f"Error in invariant mass calculation: {e}")
            return {"error": f"An unexpected error occurred: {e}"}

    def lorentz_boost(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a Lorentz boost on a particle's four-vector.
        """
        try:
            particle_vec = self._create_four_vector(parameters['particle'])
            boost_vector = vector.obj(
                px=parameters['boost_px'],
                py=parameters['boost_py'],
                pz=parameters['boost_pz']
            )
            
            boosted_particle = particle_vec.boost_beta3(boost_vector)
            
            return {
                "original_particle": {
                    "px": float(particle_vec.px), "py": float(particle_vec.py), "pz": float(particle_vec.pz), "E": float(particle_vec.E)
                },
                "boosted_particle": {
                    "px": float(boosted_particle.px), "py": float(boosted_particle.py), "pz": float(boosted_particle.pz), "E": float(boosted_particle.E)
                },
                "boost_vector": {
                    "px": float(boost_vector.px), "py": float(boost_vector.py), "pz": float(boost_vector.pz)
                }
            }
        except (KeyError, ValueError) as e:
            return {"error": f"Invalid particle or boost data provided: {e}"}
        except Exception as e:
            logger.error(f"Error in Lorentz boost: {e}")
            return {"error": f"An unexpected error occurred: {e}"}

    def analyze_particle_collision(self, parameters: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Analyzes a particle collision event to find key kinematic properties.
        """
        try:
            particles = parameters['particles']
            if not particles:
                return {"error": "Particle list cannot be empty."}

            total_four_vector = self._create_four_vector(particles[0])
            for p_dict in particles[1:]:
                total_four_vector += self._create_four_vector(p_dict)
                
            transverse_momentum = total_four_vector.pt
            pseudorapidity = total_four_vector.eta
            invariant_mass = total_four_vector.mass

            return {
                "num_particles": len(particles),
                "total_four_vector": {
                    "px": float(total_four_vector.px),
                    "py": float(total_four_vector.py),
                    "pz": float(total_four_vector.pz),
                    "E": float(total_four_vector.E)
                },
                "invariant_mass_gev": float(invariant_mass),
                "total_transverse_momentum_pt_gev": float(transverse_momentum),
                "pseudorapidity_eta": float(pseudorapidity)
            }
        except (KeyError, ValueError) as e:
            return {"error": f"Invalid particle data provided for collision analysis: {e}"}
        except Exception as e:
            logger.error(f"Error in collision analysis: {e}")
            return {"error": f"An unexpected error occurred: {e}"} 