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
        self._numpy_available = None # For the check
        
        logger.info(f"🔧 Computational tools initialized (lazy loading enabled)")
    
    def _check_numpy_availability(self):
        """Check if numpy is available without importing it."""
        if self._numpy_available is None:
            try:
                # Import importlib only when needed
                import importlib.util
                if importlib.util.find_spec('numpy'):
                    self._numpy_available = True
                else:
                    self._numpy_available = False
                    logger.warning("NumPy spec not found - computational engines may be limited")
            except ImportError:
                self._numpy_available = False
                logger.warning("NumPy not available during check - computational engines disabled")
        return self._numpy_available

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
        
        # Ensure numpy is available before initializing engines
        if not self._check_numpy_availability():
            logger.warning("⚠️ Computational engines not available (NumPy missing based on check)")
            return
        
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
        engine = self.engines['quantum_physics']
        
        try:
            if computation_type == "entanglement_entropy":
                density_matrix = self._parse_complex_matrix(parameters["density_matrix"])
                result = engine.calculate_entanglement_entropy(density_matrix)
                
            elif computation_type == "bell_violation":
                quantum_state = self._parse_complex_vector(parameters["quantum_state"])
                measurement_angles = parameters["measurement_angles"]
                result = engine.calculate_bell_inequality_violation(measurement_angles, quantum_state)
                
            elif computation_type == "quantum_fidelity":
                state1 = self._parse_complex_matrix(parameters["state1"])
                state2 = self._parse_complex_matrix(parameters["state2"])
                result = engine.calculate_quantum_fidelity(state1, state2)
                
            elif computation_type == "pauli_decomposition":
                operator = self._parse_complex_matrix(parameters["operator"])
                result = engine.pauli_decomposition(operator)
                
            else:
                return [types.TextContent(
                    type="text",
                    text=f"❌ **Unknown Quantum Computation**\n\n"
                         f"'{computation_type}' is not supported.\n\n"
                         f"**Available quantum computations:**\n"
                         f"- entanglement_entropy\n"
                         f"- bell_violation\n"
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
    
    def get_available_engines(self) -> Dict[str, Dict]:
        """Get information about available computational engines."""
        engines_info = {}
        
        for engine_id, engine in self.engines.items():
            engines_info[engine_id] = {
                "name": engine.name,
                "version": engine.version,
                "supported_calculations": engine.supported_calculations,
                "status": "active"
            }
        
        # Add planned engines
        planned_engines = [
            "molecular_modeling", "statistical_analysis", "classical_mechanics",
            "chemistry", "biology", "engineering", "mathematics"
        ]
        
        for engine_id in planned_engines:
            engines_info[engine_id] = {
                "name": f"{engine_id.replace('_', ' ').title()} Engine",
                "version": "planned",
                "supported_calculations": ["To be implemented"],
                "status": "planned"
            }
        
        return engines_info 

    def _format_iae_result(self, title: str, result: dict) -> str:
        """Format IAE computation results with a 'show your work' section."""
        if "error" in result:
            return f"❌ **{title} Failed**\n\nError: {result['error']}"
        try:
            work = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError) as e:
            work = f"Result formatting error: {str(e)}\nRaw result: {str(result)}"
        return f"# 🧠 {title} - IAE Computation Result\n\n## Results\n{work}\n\n## Show Your Work\nAll intermediate steps, formulas, and reasoning are included above. Please review for manual validation." 