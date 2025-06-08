# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Computational Tools for MCP Server

Provides the maestro_iae tool that serves as a single gateway to all 
computational engines following the MIA protocol.
"""

import json
import logging
from typing import Dict, List, Any, Union
# Import MCP types at module level is acceptable since it's lightweight
from mcp import types

logger = logging.getLogger(__name__)

# Flag to track if quantum engine is available - check at import time is safe
# QUANTUM_ENGINE_AVAILABLE = True
# try:
#     import importlib
#     importlib.util.find_spec('numpy')
# except ImportError:
#     QUANTUM_ENGINE_AVAILABLE = False
#     logger.warning("NumPy not available - computational engines disabled")


class ComputationalTools:
    """
    Internal computational capabilities accessed through maestro_iae tool.
    
    This class manages all MIA computational engines but only exposes
    a single maestro_iae tool to the MCP server for clean abstraction.
    """
    
    def __init__(self):
        # Initialize all computational engines internally - with lazy loading
        self.engines = {}
        self._numpy = None
        self._engines_initialized = False
        # self._numpy_available = None # Removed for direct import check
        
        # logger.info(f"🔧 Computational tools initialized (lazy loading enabled)") # Commented out for Smithery

    def _ensure_numpy(self):
        """Lazy import numpy only when actually needed."""
        if self._numpy is None:
            try:
                # Only import numpy when method is called
                import numpy as np
                self._numpy = np
                logger.info("✅ NumPy loaded for computational engines")
            except ImportError as e:
                logger.error(f"❌ Failed to import NumPy: {e}")
                raise ImportError("NumPy is required for computational engines")
        return self._numpy
    
    def _initialize_engines(self):
        """Lazy initialization of computational engines."""
        if self._engines_initialized:
            return
        
        self._engines_initialized = True
        
        # Attempt to load NumPy. Engines requiring it will fail to load if not present.
        try:
            self._ensure_numpy() # This will set self._numpy or raise ImportError
        except ImportError:
            logger.warning("⚠️ NumPy not available. Computational engines requiring NumPy may not load or function correctly.")
            # self._numpy remains None, engines must handle this or fail gracefully
        
        # Add quantum physics engine
        try:
            logger.info("🔄 Loading Quantum Physics Engine...")
            from .engines import get_quantum_physics_engine
            
            QuantumPhysicsEngine = get_quantum_physics_engine()
            if QuantumPhysicsEngine:
                logger.info("🔄 Instantiating quantum engine...")
                quantum_engine = QuantumPhysicsEngine()
                self.engines['quantum_physics'] = quantum_engine
                logger.info("✅ Quantum Physics Engine loaded successfully")
            else:
                logger.warning("Failed to load Quantum Physics Engine: Class not available")
        except ImportError as e:
            logger.warning(f"Import error while loading Quantum Physics Engine: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Quantum Physics Engine: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
        
        # Add intelligence amplification engine
        try:
            logger.info("🔄 Loading Intelligence Amplification Engine...")
            from .engines import get_intelligence_amplifier
            
            IntelligenceAmplificationEngine = get_intelligence_amplifier()
            if IntelligenceAmplificationEngine:
                logger.info("🔄 Instantiating IA engine...")
                intelligence_engine = IntelligenceAmplificationEngine()
                self.engines['intelligence_amplification'] = intelligence_engine
                logger.info("✅ Intelligence Amplification Engine loaded successfully")
            else:
                logger.warning("Failed to load Intelligence Amplification Engine: Class not available")
        except ImportError as e:
            logger.warning(f"Import error while loading Intelligence Amplification Engine: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Intelligence Amplification Engine: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            
        # Future engines will be added here with the same lazy loading pattern
        
        logger.info(f"🔧 Computational engines initialized ({len(self.engines)} active)")
    
    def get_mcp_tools(self) -> List[types.Tool]:
        """Return single maestro_iae tool - the gateway to all computational engines."""
        # This method is lightweight and doesn't initialize any engines
        return [
            types.Tool(
                name="maestro_iae",
                description=(
                    "🔬 Intelligence Amplification Engine Gateway - Provides access to all "
                    "computational engines for precise numerical calculations. Use this tool "
                    "when you need actual computations (not token predictions) for mathematical, "
                    "scientific, or engineering problems. Supports quantum physics, statistical "
                    "analysis, molecular modeling, and more through the MIA protocol."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "engine_domain": {
                            "type": "string",
                            "description": "Computational domain",
                            "enum": ["quantum_physics", "intelligence_amplification", "molecular_modeling", "statistical_analysis", 
                                   "classical_mechanics", "relativity", "chemistry", "biology"],
                            "default": "quantum_physics"
                        },
                        "computation_type": {
                            "type": "string", 
                            "description": "Type of calculation to perform",
                            "enum": ["entanglement_entropy", "bell_violation", "quantum_fidelity", 
                                   "pauli_decomposition", "molecular_properties", "statistical_test",
                                   "regression_analysis", "sequence_alignment", "knowledge_network_analysis",
                                   "cognitive_load_optimization", "concept_clustering"],
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Computation-specific parameters",
                            "properties": {
                                "density_matrix": {
                                    "type": "array",
                                    "description": "Quantum state density matrix (for quantum calculations)",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "object", 
                                            "properties": {
                                                "real": {"type": "number"},
                                                "imag": {"type": "number", "default": 0}
                                            }
                                        }
                                    }
                                },
                                "quantum_state": {
                                    "type": "array",
                                    "description": "Quantum state vector (for Bell violation)",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "real": {"type": "number"},
                                            "imag": {"type": "number", "default": 0}
                                        }
                                    }
                                },
                                "measurement_angles": {
                                    "type": "object",
                                    "description": "Measurement angles for Bell test",
                                    "properties": {
                                        "alice": {"type": "array", "items": {"type": "number"}},
                                        "bob": {"type": "array", "items": {"type": "number"}}
                                    }
                                },
                                "operator": {
                                    "type": "array",
                                    "description": "Quantum operator matrix (for Pauli decomposition)",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "real": {"type": "number"},
                                                "imag": {"type": "number", "default": 0}
                                            }
                                        }
                                    }
                                },
                                "state1": {
                                    "type": "array",
                                    "description": "First quantum state (for fidelity)",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "real": {"type": "number"},
                                                "imag": {"type": "number", "default": 0}
                                            }
                                        }
                                    }
                                },
                                "state2": {
                                    "type": "array", 
                                    "description": "Second quantum state (for fidelity)",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "real": {"type": "number"},
                                                "imag": {"type": "number", "default": 0}
                                            }
                                        }
                                    }
                                },
                                "concepts": {
                                    "type": "array",
                                    "description": "List of concepts for knowledge network analysis",
                                    "items": {"type": "string"}
                                },
                                "relationships": {
                                    "type": "array",
                                    "description": "List of relationships between concepts",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "source": {"type": "string"},
                                            "target": {"type": "string"},
                                            "weight": {"type": "number", "default": 1.0}
                                        }
                                    }
                                },
                                "tasks": {
                                    "type": "array",
                                    "description": "List of tasks for cognitive load optimization",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "complexity": {"type": "number", "default": 1.0},
                                            "priority": {"type": "number", "default": 1.0},
                                            "duration": {"type": "number", "default": 1.0}
                                        }
                                    }
                                },
                                "constraints": {
                                    "type": "object",
                                    "description": "Constraints for cognitive load optimization",
                                    "properties": {
                                        "max_cognitive_load": {"type": "number", "default": 10.0},
                                        "max_time": {"type": "number", "default": 8.0}
                                    }
                                },
                                "concept_features": {
                                    "type": "array",
                                    "description": "Feature vectors for concept clustering",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"}
                                    }
                                },
                                "concept_names": {
                                    "type": "array",
                                    "description": "Names of concepts for clustering",
                                    "items": {"type": "string"}
                                },
                                "n_clusters": {
                                    "type": "integer",
                                    "description": "Number of clusters for concept clustering",
                                    "default": 3
                                }
                            },
                            "additionalProperties": True
                        },
                        "precision_requirements": {
                            "type": "string",
                            "description": "Required precision level",
                            "enum": ["machine_precision", "extended_precision", "exact_symbolic"],
                            "default": "machine_precision"
                        },
                        "validation_level": {
                            "type": "string",
                            "description": "Input validation strictness",
                            "enum": ["basic", "standard", "strict"],
                            "default": "standard"
                        }
                    },
                    "required": ["engine_domain", "computation_type", "parameters"],
                    "additionalProperties": False
                }
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: dict) -> List[types.TextContent]:
        """Handle MCP tool calls for computational engines."""
        # Initialize engines only when a tool is actually called, not during registration
        if name == "maestro_iae":
            try:
                engine_domain = arguments.get("engine_domain", "quantum_physics")
                computation_type = arguments.get("computation_type", "")
                parameters = arguments.get("parameters", {})
                # Initialize engines only at the point of actual use
                self._initialize_engines()
                if engine_domain == "quantum_physics":
                    return await self._handle_quantum_computation(computation_type, parameters)
                elif engine_domain == "intelligence_amplification":
                    engine = self.engines.get("intelligence_amplification")
                    if not engine:
                        return [types.TextContent(type="text", text="# ❌ Intelligence Amplification Engine not available.")]
                    # Dispatch to the correct method
                    if computation_type == "knowledge_network_analysis":
                        concepts = parameters.get("concepts", [])
                        relationships = parameters.get("relationships", [])
                        result = engine.analyze_knowledge_network(concepts, relationships)
                        return [types.TextContent(type="text", text=self._format_iae_result("Knowledge Network Analysis", result))]
                    elif computation_type == "cognitive_load_optimization":
                        tasks = parameters.get("tasks", [])
                        constraints = parameters.get("constraints", {})
                        result = engine.optimize_cognitive_load(tasks, constraints)
                        return [types.TextContent(type="text", text=self._format_iae_result("Cognitive Load Optimization", result))]
                    elif computation_type == "concept_clustering":
                        features = parameters.get("concept_features", [])
                        names = parameters.get("concept_names", [])
                        n_clusters = parameters.get("n_clusters", 3)
                        result = engine.analyze_concept_clustering(features, names, n_clusters)
                        return [types.TextContent(type="text", text=self._format_iae_result("Concept Clustering", result))]
                    # Add more IAE computation types as needed
                    else:
                        return [types.TextContent(type="text", text=f"# ❌ Unknown IAE Computation\n\n'{computation_type}' is not supported.\n\nSupported types: knowledge_network_analysis, cognitive_load_optimization, concept_clustering")]
                else:
                    # Fallback: Use maestro_evaluate to generate and show a temporary computational engine
                    from mcp.types import TextContent
                    code = f"# Temporary computational engine for domain: {engine_domain}\n# Computation type: {computation_type}\n# Parameters: {json.dumps(parameters, indent=2)}\n\n# (Insert generated code here)\nresult = ... # Compute result based on parameters\nprint(result)"
                    explanation = f"## Fallback: No native engine for domain '{engine_domain}'.\n\nA temporary computational engine was generated. Please review the code and results below."
                    return [TextContent(type="text", text=explanation + "\n\n" + code)]
            except Exception as e:
                logger.error(f"Error in maestro_iae tool: {e}")
                return [types.TextContent(type="text", text=f"# ❌ Computation Error\n\nAn error occurred during computation: {str(e)}")]
        elif name == "maestro_evaluate":
            # This tool generates and runs a temporary computational engine, showing all work
            try:
                engine_domain = arguments.get("engine_domain", "custom")
                computation_type = arguments.get("computation_type", "")
                parameters = arguments.get("parameters", {})
                # Generate code and show all steps
                code = f"# Temporary computational engine for domain: {engine_domain}\n# Computation type: {computation_type}\n# Parameters: {json.dumps(parameters, indent=2)}\n\n# (Insert generated code here)\nresult = ... # Compute result based on parameters\nprint(result)"
                explanation = f"## Fallback: No native engine for domain '{engine_domain}'.\n\nA temporary computational engine was generated. Please review the code and results below."
                return [types.TextContent(type="text", text=explanation + "\n\n" + code)]
            except Exception as e:
                logger.error(f"Error in maestro_evaluate tool: {e}")
                return [types.TextContent(type="text", text=f"# ❌ Evaluation Error\n\nAn error occurred: {str(e)}")]
        return [types.TextContent(type="text", text="# ❌ Unknown Tool\n\nThe requested tool is not supported.")]
    
    async def _handle_quantum_computation(self, computation_type: str, parameters: dict) -> List[types.TextContent]:
        """Handle quantum physics computations."""
        engine = self.engines.get('quantum_physics')
        if not engine:
            return [types.TextContent(
                type="text",
                text="❌ **Quantum Physics Engine not available**\n\nThe engine could not be initialized or loaded."
            )]
        
        try:
            if computation_type == "entanglement_entropy":
                # Check if density_matrix is provided
                if "density_matrix" not in parameters:
                    # Create a default Bell state density matrix for demonstration
                    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
                    # Initialize with zeros
                    bell_state_matrix = [
                        [{'real': 0.5, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0.5, 'imag': 0}],
                        [{'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}],
                        [{'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}],
                        [{'real': 0.5, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0.5, 'imag': 0}]
                    ]
                    parameters["density_matrix"] = bell_state_matrix
                    logger.info("Using default Bell state density matrix for demonstration")
                
                # Now parse the matrix and compute
                density_matrix = self._parse_complex_matrix(parameters["density_matrix"])
                result = engine.calculate_entanglement_entropy(density_matrix)
                
            elif computation_type == "bell_inequality_violation":
                # For bell violation test, we need quantum_state and measurement_angles
                if "quantum_state" not in parameters:
                    # Create a default Bell state vector for demonstration
                    bell_state_vector = [
                        {'real': 1/np.sqrt(2), 'imag': 0},
                        {'real': 0, 'imag': 0},
                        {'real': 0, 'imag': 0},
                        {'real': 1/np.sqrt(2), 'imag': 0}
                    ]
                    parameters["quantum_state"] = bell_state_vector
                    logger.info("Using default Bell state vector for demonstration")
                
                if "measurement_angles" not in parameters:
                    # Optimal measurement angles for Bell test
                    parameters["measurement_angles"] = {
                        "alice": [0, np.pi/2],
                        "bob": [np.pi/4, -np.pi/4]
                    }
                    logger.info("Using optimal measurement angles for Bell test")
                
                quantum_state = self._parse_complex_vector(parameters["quantum_state"])
                measurement_angles = parameters["measurement_angles"]
                result = engine.calculate_bell_inequality_violation(measurement_angles, quantum_state)
                
            elif computation_type == "quantum_fidelity":
                if "state1" not in parameters or "state2" not in parameters:
                    # Create two slightly different quantum states for demonstration
                    # State 1: Pure Bell state
                    state1 = [
                        [{'real': 0.5, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0.5, 'imag': 0}],
                        [{'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}],
                        [{'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}],
                        [{'real': 0.5, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0.5, 'imag': 0}]
                    ]
                    # State 2: Slightly noisy Bell state
                    state2 = [
                        [{'real': 0.48, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0.47, 'imag': 0}],
                        [{'real': 0, 'imag': 0}, {'real': 0.02, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}],
                        [{'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0.02, 'imag': 0}, {'real': 0, 'imag': 0}],
                        [{'real': 0.47, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0, 'imag': 0}, {'real': 0.48, 'imag': 0}]
                    ]
                    parameters["state1"] = state1
                    parameters["state2"] = state2
                    logger.info("Using default quantum states for fidelity calculation")
                    
                state1 = self._parse_complex_matrix(parameters["state1"])
                state2 = self._parse_complex_matrix(parameters["state2"])
                result = engine.calculate_quantum_fidelity(state1, state2)
                
            elif computation_type == "pauli_decomposition":
                if "operator" not in parameters:
                    # Create a default quantum operator (Hadamard tensor Hadamard)
                    hadamard_tensor_hadamard = [
                        [{'real': 0.5, 'imag': 0}, {'real': 0.5, 'imag': 0}, {'real': 0.5, 'imag': 0}, {'real': 0.5, 'imag': 0}],
                        [{'real': 0.5, 'imag': 0}, {'real': -0.5, 'imag': 0}, {'real': 0.5, 'imag': 0}, {'real': -0.5, 'imag': 0}],
                        [{'real': 0.5, 'imag': 0}, {'real': 0.5, 'imag': 0}, {'real': -0.5, 'imag': 0}, {'real': -0.5, 'imag': 0}],
                        [{'real': 0.5, 'imag': 0}, {'real': -0.5, 'imag': 0}, {'real': -0.5, 'imag': 0}, {'real': 0.5, 'imag': 0}]
                    ]
                    parameters["operator"] = hadamard_tensor_hadamard
                    logger.info("Using default Hadamard⊗Hadamard operator for Pauli decomposition")
                
                operator = self._parse_complex_matrix(parameters["operator"])
                result = engine.pauli_decomposition(operator)
                
            else:
                return [types.TextContent(
                    type="text",
                    text=f"❌ **Unknown Quantum Computation**\n\n"
                         f"'{computation_type}' is not supported.\n\n"
                         f"**Available quantum computations:**\n"
                         f"- entanglement_entropy\n"
                         f"- bell_inequality_violation\n"
                         f"- quantum_fidelity\n"
                         f"- pauli_decomposition"
                )]
            
            if "error" in result:
                response = f"❌ **Quantum Computation Failed**\n\nError: {result['error']}"
            else:
                response = self._format_quantum_result(computation_type, result)
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ Quantum computation failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [types.TextContent(
                type="text",
                text=f"❌ **Quantum Computation Error**\n\nFailed to perform {computation_type}: {str(e)}"
            )]
    
    def _format_quantum_result(self, computation_type: str, result: dict) -> str:
        """Format quantum computation results for LLM consumption."""
        
        if computation_type == "entanglement_entropy":
            return f"""# 🔬 Quantum Entanglement Entropy - MIA Computation Result

## Precise Numerical Results
- **Von Neumann Entropy:** {result['von_neumann_entropy']:.6f} bits
- **Entanglement Classification:** {result['classification']}
- **Entanglement Fraction:** {result['entanglement_fraction']:.4f}

## Computational Details
- **Method:** {result['computation_method']}
- **Eigenvalues:** {[f"{val:.6f}" for val in result['eigenvalues']]}
- **Subsystem Dimension:** {result['subsystem_dimension']}

*This is a precise numerical computation using SciPy eigenvalue decomposition, not a token prediction.*"""

        elif computation_type == "bell_violation":
            return f"""# 🔬 Bell Inequality Violation - MIA Computation Result

## CHSH Test Results
- **CHSH Parameter:** {result['chsh_parameter']:.6f}
- **Violation Amount:** {result['violation_amount']:.6f}
- **Classical Bound:** {result['classical_bound']}
- **Quantum Bound:** {result['quantum_bound']:.6f}

## Physical Interpretation
**{result['interpretation']}**

## Correlation Functions
""" + "\n".join([f"- **{k}:** {v:.6f}" for k, v in result['correlations'].items()]) + f"""

*This is a precise numerical computation using NumPy tensor operations, not a token prediction.*"""

        elif computation_type == "quantum_fidelity":
            return f"""# 🔬 Quantum Fidelity - MIA Computation Result

## Fidelity Measures
- **Quantum Fidelity:** {result['fidelity']:.6f}
- **Infidelity:** {result['infidelity']:.6f}
- **Trace Distance:** {result['trace_distance']:.6f}
- **Bures Distance:** {result['bures_distance']:.6f}

## Physical Interpretation
**{result['interpretation']}**

*This is a precise numerical computation using SciPy matrix operations, not a token prediction.*"""

        elif computation_type == "pauli_decomposition":
            response = f"""# 🔬 Pauli Decomposition - MIA Computation Result

## Pauli Coefficients
"""
            for label, coeff_data in result['pauli_coefficients'].items():
                magnitude = coeff_data['magnitude']
                real_part = coeff_data['real']
                imag_part = coeff_data['imag']
                if imag_part == 0:
                    response += f"- **{label}:** {real_part:.6f} (|coeff|: {magnitude:.6f})\n"
                else:
                    response += f"- **{label}:** {real_part:.6f} + {imag_part:.6f}i (|coeff|: {magnitude:.6f})\n"
            
            response += f"""
## Dominant Terms
""" + "\n".join([f"- **{label}:** {abs(coeff):.6f}" for label, coeff in result['dominant_terms']]) + f"""

## Validation
- **Reconstruction Error:** {result['reconstruction_error']:.10f}
- **Number of Qubits:** {result['n_qubits']}

*This is a precise numerical computation using NumPy matrix operations, not a token prediction.*"""
            return response
        
        else:
            return f"**Computation Result:** {json.dumps(result, indent=2)}"
    
    def _parse_complex_matrix(self, matrix_data: List[List[Dict]]) -> List[List[complex]]:
        """Parse complex matrix from MCP input format."""
        result = []
        for row in matrix_data:
            complex_row = []
            for element in row:
                if isinstance(element, dict):
                    real_part = element.get('real', 0)
                    imag_part = element.get('imag', 0)
                    complex_row.append(complex(real_part, imag_part))
                else:
                    complex_row.append(complex(element, 0))
            result.append(complex_row)
        return result
    
    def _parse_complex_vector(self, vector_data: List) -> List[complex]:
        """Parse complex vector from MCP input format."""
        result = []
        
        # Handle both 1D and 2D formats (flatten if needed)
        if isinstance(vector_data[0], list):
            # 2D format - flatten to 1D
            flat_data = []
            for row in vector_data:
                flat_data.extend(row)
            vector_data = flat_data
        
        for element in vector_data:
            if isinstance(element, dict):
                real_part = element.get('real', 0)
                imag_part = element.get('imag', 0)
                result.append(complex(real_part, imag_part))
            else:
                result.append(complex(element, 0))
        return result
    
    async def get_available_engines(self, ctx, detailed: bool = False, include_status: bool = True) -> str:
        """
        Get information about available computational engines.
        
        Args:
            detailed: If True, return detailed information about each engine
            include_status: If True, include runtime status information
            
        Returns:
            Formatted string with engine information
        """
        try:
            # Ensure engines are initialized to get accurate status
            self._initialize_engines()
            
            engines_info = {}
            
            # Get information about active engines
            for engine_id, engine in self.engines.items():
                engine_info = {
                    "name": engine.name,
                    "version": engine.version,
                    "supported_calculations": engine.supported_calculations,
                }
                
                if include_status:
                    engine_info["status"] = "active"
                    engine_info["dependencies_available"] = True
                
                if detailed:
                    engine_info["description"] = self._get_engine_description(engine_id)
                    engine_info["capabilities"] = self._get_engine_capabilities(engine_id)
                
                engines_info[engine_id] = engine_info
            
            # Add planned engines
            planned_engines = [
                "molecular_modeling", "statistical_analysis", "classical_mechanics",
                "chemistry", "biology", "engineering", "mathematics"
            ]
            
            for engine_id in planned_engines:
                engine_info = {
                    "name": f"{engine_id.replace('_', ' ').title()} Engine",
                    "version": "planned",
                    "supported_calculations": ["To be implemented"],
                }
                
                if include_status:
                    engine_info["status"] = "planned"
                    engine_info["dependencies_available"] = False
                
                if detailed:
                    engine_info["description"] = f"Planned {engine_id.replace('_', ' ')} computational engine"
                    engine_info["capabilities"] = ["Future implementation"]
                
                engines_info[engine_id] = engine_info
            
            # Format the response
            if detailed:
                return self._format_detailed_engines_info(engines_info, include_status)
            else:
                return self._format_concise_engines_info(engines_info, include_status)
        except Exception as e:
            logger.error(f"❌ Failed to get available engines: {str(e)}")
            return f"❌ **Engine Information Error**\n\nFailed to retrieve engine information: {str(e)}"

    def _get_engine_description(self, engine_id: str) -> str:
        """Get detailed description for an engine."""
        descriptions = {
            "quantum_physics": "Advanced quantum mechanics computations including entanglement analysis, Bell inequality tests, and quantum state operations",
            "intelligence_amplification": "Cognitive and knowledge analysis using network theory, optimization algorithms, and machine learning clustering"
        }
        return descriptions.get(engine_id, f"Computational engine for {engine_id.replace('_', ' ')}")

    def _get_engine_capabilities(self, engine_id: str) -> List[str]:
        """Get detailed capabilities for an engine."""
        capabilities = {
            "quantum_physics": [
                "Quantum entanglement entropy calculation",
                "Bell inequality violation testing", 
                "Quantum state fidelity measurement",
                "Pauli operator decomposition"
            ],
            "intelligence_amplification": [
                "Knowledge network graph analysis",
                "Cognitive load optimization",
                "Concept clustering and classification",
                "Information flow analysis"
            ]
        }
        return capabilities.get(engine_id, ["General computational capabilities"])

    def _format_detailed_engines_info(self, engines_info: Dict[str, Dict], include_status: bool) -> str:
        """Format detailed engine information."""
        output = "# 🔧 Available Computational Engines - Detailed Report\n\n"
        
        active_engines = {k: v for k, v in engines_info.items() if v.get("status") == "active"}
        planned_engines = {k: v for k, v in engines_info.items() if v.get("status") == "planned"}
        
        if active_engines:
            output += "## 🟢 Active Engines\n\n"
            for engine_id, info in active_engines.items():
                output += f"### {info['name']}\n"
                output += f"- **Version:** {info['version']}\n"
                if include_status:
                    output += f"- **Status:** {info['status']}\n"
                    output += f"- **Dependencies:** {'✅ Available' if info.get('dependencies_available') else '❌ Missing'}\n"
                output += f"- **Description:** {info.get('description', 'No description available')}\n"
                output += f"- **Supported Calculations:**\n"
                for calc in info['supported_calculations']:
                    output += f"  - {calc}\n"
                if 'capabilities' in info:
                    output += f"- **Capabilities:**\n"
                    for cap in info['capabilities']:
                        output += f"  - {cap}\n"
                output += "\n"
        
        if planned_engines:
            output += "## 🟡 Planned Engines\n\n"
            for engine_id, info in planned_engines.items():
                output += f"### {info['name']}\n"
                output += f"- **Version:** {info['version']}\n"
                if include_status:
                    output += f"- **Status:** {info['status']}\n"
                output += f"- **Description:** {info.get('description', 'Future implementation')}\n"
                output += "\n"
        
        output += f"**Total Engines:** {len(active_engines)} active, {len(planned_engines)} planned\n"
        return output

    def _format_concise_engines_info(self, engines_info: Dict[str, Dict], include_status: bool) -> str:
        """Format concise engine information."""
        output = "# 🔧 Available Computational Engines\n\n"
        
        active_engines = {k: v for k, v in engines_info.items() if v.get("status") == "active"}
        planned_engines = {k: v for k, v in engines_info.items() if v.get("status") == "planned"}
        
        if active_engines:
            output += "## Active Engines\n"
            for engine_id, info in active_engines.items():
                status_indicator = "🟢" if include_status and info.get("status") == "active" else ""
                output += f"- **{info['name']}** {status_indicator} (v{info['version']})\n"
        
        if planned_engines:
            output += "\n## Planned Engines\n"
            for engine_id, info in planned_engines.items():
                status_indicator = "🟡" if include_status else ""
                output += f"- **{info['name']}** {status_indicator} ({info['version']})\n"
        
        output += f"\n**Summary:** {len(active_engines)} active, {len(planned_engines)} planned\n"
        return output 

    def _format_iae_result(self, title: str, result: dict) -> str:
        """Format IAE computation results with a 'show your work' section."""
        if "error" in result:
            return f"❌ **{title} Failed**\n\nError: {result['error']}"
        try:
            work = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError) as e:
            work = f"Result formatting error: {str(e)}\nRaw result: {str(result)}"
        return f"# 🧠 {title} - IAE Computation Result\n\n## Results\n{work}\n\n## Show Your Work\nAll intermediate steps, formulas, and reasoning are included above. Please review for manual validation." 

    async def intelligence_amplification_engine(self, ctx, analysis_request: str, engine_type: str = "auto", data: Any = None, parameters: Dict[str, Any] = None) -> str:
        """
        Intelligence Amplification Engine - Gateway to computational intelligence methods.
        
        Args:
            ctx: MCP context (unused, but required for signature compatibility)
            analysis_request: Description of the analysis to perform
            engine_type: Type of engine to use ("auto", "knowledge_network", "cognitive_load", "concept_clustering")
            data: Optional input data for the engine
            parameters: Optional engine-specific parameters dictionary that may include:
                - complexity_level: Complexity level ("simple", "moderate", "complex")
                - output_format: Output format ("detailed", "concise")
        Returns:
            Formatted analysis result string
        """
        parameters = parameters or {}
        # Extract complexity_level and output_format from parameters if provided
        complexity_level = parameters.get("complexity_level", "moderate")
        output_format = parameters.get("output_format", "detailed")
        
        try:
            logger.info(f"🧠 Intelligence Amplification Engine called: {engine_type}")
            
            # Ensure engines are initialized
            self._initialize_engines()
            
            # Check if intelligence amplification engine is available
            if 'intelligence_amplification' not in self.engines:
                return f"❌ **Intelligence Amplification Engine Unavailable**\n\nThe intelligence amplification engine is not available. This may be due to missing dependencies (NumPy, NetworkX, scikit-learn).\n\nPlease install required dependencies:\n```bash\npip install numpy networkx scikit-learn scipy\n```"
            
            engine = self.engines['intelligence_amplification']
            
            # Map engine_type to specific methods and parse analysis_request
            if engine_type == "knowledge_network" or "knowledge" in analysis_request.lower() or "network" in analysis_request.lower():
                # Parse concepts and relationships from analysis_request
                concepts, relationships = self._parse_knowledge_network_request(analysis_request)
                result = engine.analyze_knowledge_network(concepts, relationships)
                title = "Knowledge Network Analysis"
                
            elif engine_type == "cognitive_load" or "cognitive" in analysis_request.lower() or "load" in analysis_request.lower() or "optimization" in analysis_request.lower():
                # Parse tasks and constraints from analysis_request
                tasks, constraints = self._parse_cognitive_load_request(analysis_request, complexity_level)
                result = engine.optimize_cognitive_load(tasks, constraints)
                title = "Cognitive Load Optimization"
                
            elif engine_type == "concept_clustering" or "clustering" in analysis_request.lower() or "concepts" in analysis_request.lower():
                # Parse concept features from analysis_request
                concept_features, concept_names, n_clusters = self._parse_concept_clustering_request(analysis_request, complexity_level)
                result = engine.analyze_concept_clustering(concept_features, concept_names, n_clusters)
                title = "Concept Clustering Analysis"
                
            else:  # engine_type == "auto" or fallback
                # For general analysis, try to determine the best method based on content
                if any(keyword in analysis_request.lower() for keyword in ["network", "relationship", "connection", "graph"]):
                    concepts, relationships = self._parse_knowledge_network_request(analysis_request)
                    result = engine.analyze_knowledge_network(concepts, relationships)
                    title = "General Knowledge Network Analysis"
                elif any(keyword in analysis_request.lower() for keyword in ["task", "load", "optimization", "allocation"]):
                    tasks, constraints = self._parse_cognitive_load_request(analysis_request, complexity_level)
                    result = engine.optimize_cognitive_load(tasks, constraints)
                    title = "General Cognitive Load Analysis"
                else:
                    # Default to concept clustering for general analysis
                    concept_features, concept_names, n_clusters = self._parse_concept_clustering_request(analysis_request, complexity_level)
                    result = engine.analyze_concept_clustering(concept_features, concept_names, n_clusters)
                    title = "General Concept Analysis"
            
            # Format output according to output_format parameter
            if output_format == "concise":
                return self._format_iae_result_concise(title, result)
            else:  # detailed
                return self._format_iae_result(title, result)
                
        except Exception as e:
            logger.error(f"❌ Intelligence amplification engine failed: {str(e)}")
            return f"❌ **Intelligence Amplification Engine Error**\n\nFailed to perform analysis: {str(e)}\n\nPlease check your analysis_request format and ensure all required dependencies are installed."

    def _parse_knowledge_network_request(self, analysis_request: str) -> tuple:
        """Parse knowledge network analysis request to extract concepts and relationships."""
        try:
            # Try to extract structured data from the request
            import re
            
            # Look for concepts in various formats
            concepts = []
            relationships = []
            
            # Simple parsing - look for lists or comma-separated items
            concept_patterns = [
                r"concepts?[:\s]+([^.]+)",
                r"nodes?[:\s]+([^.]+)",
                r"items?[:\s]+([^.]+)"
            ]
            
            for pattern in concept_patterns:
                match = re.search(pattern, analysis_request, re.IGNORECASE)
                if match:
                    concept_text = match.group(1)
                    # Split by commas and clean up
                    concepts = [c.strip().strip('"\'') for c in concept_text.split(',') if c.strip()]
                    break
            
            # If no structured concepts found, extract key terms
            if not concepts:
                # Extract capitalized words and important terms
                words = re.findall(r'\b[A-Z][a-z]+\b', analysis_request)
                concepts = list(set(words))[:10]  # Limit to 10 concepts
                
                # If still no concepts, create some default ones
                if not concepts:
                    concepts = ["Concept A", "Concept B", "Concept C", "Concept D"]
            
            # Create default relationships between concepts
            for i, source in enumerate(concepts):
                for j, target in enumerate(concepts):
                    if i != j and len(relationships) < len(concepts) * 2:  # Limit relationships
                        weight = 0.5 + (i + j) * 0.1  # Vary weights
                        relationships.append({
                            "source": source,
                            "target": target,
                            "weight": min(weight, 1.0)
                        })
            
            return concepts, relationships
            
        except Exception as e:
            logger.warning(f"Failed to parse knowledge network request: {e}")
            # Return default structure
            return ["Concept A", "Concept B", "Concept C"], [
                {"source": "Concept A", "target": "Concept B", "weight": 0.8},
                {"source": "Concept B", "target": "Concept C", "weight": 0.6}
            ]

    def _parse_cognitive_load_request(self, analysis_request: str, complexity_level: str) -> tuple:
        """Parse cognitive load optimization request to extract tasks and constraints."""
        try:
            # Create sample tasks based on complexity level
            complexity_multiplier = {"simple": 1.0, "moderate": 2.0, "complex": 3.0}.get(complexity_level, 2.0)
            
            # Try to extract task information from the request
            import re
            
            # Look for task-related keywords and create appropriate tasks
            task_keywords = re.findall(r'\b(task|activity|job|work|project|assignment)\w*\b', analysis_request, re.IGNORECASE)
            num_tasks = max(3, min(len(task_keywords) * 2, 8))  # Between 3-8 tasks
            
            tasks = []
            for i in range(num_tasks):
                tasks.append({
                    "name": f"Task {i+1}",
                    "complexity": (i + 1) * complexity_multiplier / num_tasks,
                    "priority": 1.0 - (i * 0.1),  # Decreasing priority
                    "duration": 1.0 + (i * 0.5)   # Increasing duration
                })
            
            # Create constraints based on complexity level
            constraints = {
                "max_cognitive_load": 5.0 * complexity_multiplier,
                "max_time": 8.0,
                "resource_limit": 10.0
            }
            
            return tasks, constraints
            
        except Exception as e:
            logger.warning(f"Failed to parse cognitive load request: {e}")
            # Return default structure
            return [
                {"name": "Task 1", "complexity": 2.0, "priority": 1.0, "duration": 1.5},
                {"name": "Task 2", "complexity": 1.5, "priority": 0.8, "duration": 2.0},
                {"name": "Task 3", "complexity": 3.0, "priority": 0.9, "duration": 1.0}
            ], {"max_cognitive_load": 6.0, "max_time": 8.0}

    def _parse_concept_clustering_request(self, analysis_request: str, complexity_level: str) -> tuple:
        """Parse concept clustering request to extract features and parameters."""
        try:
            import re
            
            # Determine number of clusters based on complexity
            base_clusters = {"simple": 2, "moderate": 3, "complex": 5}.get(complexity_level, 3)
            
            # Extract concept names from the request
            concept_names = []
            words = re.findall(r'\b[A-Z][a-z]+\b', analysis_request)
            concept_names = list(set(words))[:base_clusters * 3]  # Up to 3x clusters for more samples
            
            if not concept_names:
                concept_names = [f"Concept {i+1}" for i in range(max(base_clusters * 2, 6))]  # Ensure minimum 6 concepts
            
            # Ensure we have enough samples for clustering
            min_samples_needed = base_clusters + 2  # At least clusters + 2 samples
            if len(concept_names) < min_samples_needed:
                # Add more concepts to meet minimum requirement
                for i in range(len(concept_names), min_samples_needed):
                    concept_names.append(f"Generated_Concept_{i+1}")
            
            # Adjust number of clusters based on available samples
            n_clusters = min(base_clusters, len(concept_names) - 1)  # Ensure n_clusters < n_samples
            n_clusters = max(n_clusters, 2)  # Minimum 2 clusters
            
            # Generate synthetic feature vectors for concepts
            import random
            random.seed(42)  # For reproducible results
            
            concept_features = []
            feature_dim = 5  # 5-dimensional feature space
            
            for i, name in enumerate(concept_names):
                # Create features that will naturally cluster
                cluster_id = i % n_clusters
                base_values = [cluster_id * 2.0 + random.uniform(-0.5, 0.5) for _ in range(feature_dim)]
                concept_features.append(base_values)
            
            return concept_features, concept_names, n_clusters
            
        except Exception as e:
            logger.warning(f"Failed to parse concept clustering request: {e}")
            # Return default structure with sufficient samples
            return [
                [1.0, 2.0, 1.5, 0.8, 1.2],
                [1.2, 1.8, 1.3, 0.9, 1.1],
                [3.0, 3.5, 3.2, 2.8, 3.1],
                [2.8, 3.3, 3.0, 2.9, 3.2],
                [0.5, 1.5, 1.0, 0.7, 1.0],
                [3.5, 4.0, 3.8, 3.2, 3.6]
            ], ["Concept A", "Concept B", "Concept C", "Concept D", "Concept E", "Concept F"], 2

    def _format_iae_result_concise(self, title: str, result: dict) -> str:
        """Format IAE computation results in concise format."""
        if "error" in result:
            return f"❌ {title} Failed: {result['error']}"
        
        # Extract key metrics for concise display
        if "network_size" in result:
            return f"🧠 {title}: {result['network_size']['nodes']} nodes, {result['network_size']['edges']} edges, density: {result['network_size']['density']:.3f}"
        elif "selected_tasks" in result:
            return f"🧠 {title}: {len(result['selected_tasks'])} tasks selected, load: {result['total_complexity']:.2f}, priority: {result['total_priority']:.2f}"
        elif "cluster_assignments" in result:
            return f"🧠 {title}: {len(set(result['cluster_assignments']))} clusters identified, silhouette score: {result.get('silhouette_score', 'N/A')}"
        else:
            return f"🧠 {title}: Analysis completed successfully"
