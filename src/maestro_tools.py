from typing import List
from maestro.types import TextContent
from maestro.logger import logger

class MaestroTools:
    async def _handle_orchestrate(self, arguments: dict) -> List[TextContent]:
        """Handle orchestration requests with enhanced Intelligence Amplification Engine integration."""
        try:
            task_description = arguments.get("task_description", "")
            context = arguments.get("context", {})
            success_criteria = arguments.get("success_criteria", {})
            complexity_level = arguments.get("complexity_level", "moderate")
            
            logger.info(f"🎭 Orchestrating task: {task_description[:100]}...")
            
            # Analyze task requirements and map to computational engines
            computational_needs = self._analyze_computational_requirements(task_description, context)
            
            response = f"""# 🎭 Maestro Orchestration Framework

## Task Analysis
**Description:** {task_description}
**Complexity Level:** {complexity_level}

## Computational Amplification Strategy
"""
            
            if computational_needs["requires_computation"]:
                response += f"""
### Intelligence Amplification Engines Required
{computational_needs["engine_recommendations"]}

### Orchestration Workflow
1. **Primary Tool**: `maestro_iae` - Gateway to computational engines
2. **Engine Domain**: {computational_needs["primary_domain"]}
3. **Computation Types**: {', '.join(computational_needs["computation_types"])}

### Recommended Tool Sequence
"""
                for i, step in enumerate(computational_needs["workflow_steps"], 1):
                    response += f"{i}. {step}\n"
                
                response += f"""
### maestro_iae Usage Pattern
```
Tool Call: maestro_iae
Parameters:
  engine_domain: "{computational_needs["primary_domain"]}"
  computation_type: "{computational_needs["computation_types"][0] if computational_needs["computation_types"] else 'custom'}"
  parameters: {{specific computation parameters}}
```
"""
            else:
                response += """
### Analysis Workflow
This task requires strategic analysis rather than computational engines.
Recommended approach: Use reasoning and context analysis tools.
"""
            
            response += f"""
## Success Metrics
{self._format_success_criteria(success_criteria)}

## Enhanced Capabilities
- ✅ **Computational Precision**: Access to MIA engines for exact calculations
- ✅ **Multi-Domain Integration**: Coordinate across scientific domains
- ✅ **Validation Framework**: Verify results through computational engines
- ✅ **Scalable Architecture**: Modular engine selection based on needs

*This orchestration leverages Intelligence Amplification Engines for computational tasks beyond token prediction.*"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ **Orchestration Failed**\n\nError: {str(e)}\n\nPlease refine your task description and try again."
            )]

    async def _handle_iae_discovery(self, arguments: dict) -> List[TextContent]:
        """Handle Intelligence Amplification Engine discovery and mapping."""
        try:
            task_type = arguments.get("task_type", "general")
            domain_context = arguments.get("domain_context", "")
            complexity_requirements = arguments.get("complexity_requirements", {})
            
            logger.info(f"🔍 Discovering IAEs for: {task_type}")
            
            # Get available engines from computational tools
            available_engines = self.computational_tools.get_available_engines()
            
            response = f"""# 🔍 Intelligence Amplification Engine Discovery

## Available Computational Engines

### Active Engines (Ready for Use)
"""
            
            active_engines = {k: v for k, v in available_engines.items() if v["status"] == "active"}
            for engine_id, engine_info in active_engines.items():
                response += f"""
#### {engine_info["name"]} v{engine_info["version"]}
- **Domain**: {engine_id}
- **Capabilities**: {', '.join(engine_info["supported_calculations"])}
- **Access via**: `maestro_iae` with `engine_domain: "{engine_id}"`
"""
            
            response += f"""
### Planned Engines (Under Development)
"""
            planned_engines = {k: v for k, v in available_engines.items() if v["status"] == "planned"}
            for engine_id, engine_info in planned_engines.items():
                response += f"- **{engine_info['name']}**: {engine_id}\n"
            
            # Provide task-specific recommendations
            recommendations = self._get_engine_recommendations(task_type, domain_context)
            
            response += f"""
## Task-Specific Recommendations

### For "{task_type}" tasks:
{recommendations}

## Usage Pattern
To access any computational engine:
```
Tool: maestro_iae
Parameters:
  engine_domain: [select from available engines]
  computation_type: [specific calculation needed]
  parameters: {{computation-specific data}}
```

## Integration Benefits
- **Single Gateway**: All computational engines accessible through `maestro_iae`
- **Standardized Interface**: Consistent MIA protocol across all engines
- **Precise Results**: Machine-precision calculations, not token predictions
- **Modular Growth**: New engines added without changing interface

*The MIA protocol ensures computational amplification through standardized engine interfaces.*"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ IAE discovery failed: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ **Discovery Failed**\n\nError: {str(e)}"
            )]

    async def _handle_tool_selection(self, arguments: dict) -> List[TextContent]:
        """Handle tool selection recommendations."""
        try:
            request_description = arguments.get("request_description", "")
            available_context = arguments.get("available_context", {})
            precision_requirements = arguments.get("precision_requirements", {})
            
            logger.info(f"🎯 Analyzing tool selection for: {request_description[:100]}...")
            
            # Analyze computational vs strategic needs
            computational_analysis = self._analyze_computational_requirements(request_description, available_context)
            
            response = f"""# 🎯 Intelligent Tool Selection Analysis

## Request Analysis
**Description:** {request_description}
**Computational Needs:** {"Yes" if computational_analysis["requires_computation"] else "No"}

## Recommended Tool Strategy
"""
            
            if computational_analysis["requires_computation"]:
                response += f"""
### Primary Recommendation: Computational Approach
**Main Tool**: `maestro_iae` - Intelligence Amplification Engine Gateway

**Configuration:**
- Engine Domain: `{computational_analysis["primary_domain"]}`
- Computation Types: {', '.join(computational_analysis["computation_types"])}
- Precision Level: {precision_requirements.get("level", "machine_precision")}

### Sequential Workflow
1. **Data Preparation**: Organize input parameters for MIA engines
2. **Computation Call**: Use `maestro_iae` with specific engine configuration  
3. **Result Integration**: Process precise numerical results
4. **Analysis Enhancement**: Combine computational results with reasoning

### Alternative Strategic Tools
- `maestro_orchestrate`: For complex multi-step workflows
- `maestro_enhancement`: For integrating computational results with content
"""
            else:
                response += f"""
### Primary Recommendation: Strategic Analysis
**Main Tools**: Orchestration and reasoning tools

**Suggested Sequence:**
1. `maestro_orchestrate`: Strategic workflow planning
2. `maestro_enhancement`: Content improvement and analysis
3. `maestro_iae_discovery`: If computational needs emerge

### Computational Backup
If numerical calculations become necessary:
- **Tool**: `maestro_iae`
- **Benefits**: Precise calculations vs. token predictions
- **Integration**: Seamless result incorporation
"""
            
            response += f"""
## Tool Capability Matrix

| Tool | Computational | Strategic | Coordination | Precision |
|------|-------------|-----------|-------------|-----------|
| `maestro_iae` | ✅ Primary | ⚠️ Limited | ❌ No | ✅ Machine |
| `maestro_orchestrate` | ✅ Routes to IAE | ✅ Primary | ✅ Primary | ✅ Via IAE |
| `maestro_enhancement` | ✅ Integrates | ✅ Primary | ✅ Limited | ✅ Via IAE |
| `maestro_iae_discovery` | ✅ Maps | ✅ Limited | ❌ No | N/A |
| `maestro_tool_selection` | ❌ No | ✅ Primary | ✅ Limited | N/A |

## Key Insights
- **Computational Tasks**: Always prefer `maestro_iae` over token prediction
- **Complex Workflows**: Use `maestro_orchestrate` for multi-engine coordination
- **Precision Matters**: MIA engines provide exact calculations
- **Modular Approach**: Single gateway to all computational capabilities

*Choose tools based on precision requirements: computational engines for exact results, strategic tools for reasoning and coordination.*"""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ Tool selection analysis failed: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ **Tool Selection Failed**\n\nError: {str(e)}"
            )]

    def _analyze_computational_requirements(self, task_description: str, context: dict) -> dict:
        """Analyze if task requires computational engines and which ones."""
        task_lower = task_description.lower()
        
        # Check for computational keywords
        computational_indicators = {
            "quantum": ["quantum_physics", ["entanglement_entropy", "bell_violation", "quantum_fidelity", "pauli_decomposition"]],
            "molecular": ["molecular_modeling", ["molecular_properties", "conformation_analysis"]],
            "statistical": ["statistical_analysis", ["regression_analysis", "statistical_test"]],
            "calculate": ["quantum_physics", ["entanglement_entropy", "quantum_fidelity"]],
            "compute": ["quantum_physics", ["bell_violation", "pauli_decomposition"]],
            "analyze": ["statistical_analysis", ["regression_analysis", "statistical_test"]],
            "entropy": ["quantum_physics", ["entanglement_entropy"]],
            "bell": ["quantum_physics", ["bell_violation"]],
            "fidelity": ["quantum_physics", ["quantum_fidelity"]],
            "pauli": ["quantum_physics", ["pauli_decomposition"]],
            "chemistry": ["chemistry", ["molecular_properties"]],
            "biology": ["biology", ["sequence_alignment"]]
        }
        
        detected_domains = []
        detected_computations = []
        
        for keyword, (domain, computations) in computational_indicators.items():
            if keyword in task_lower:
                if domain not in detected_domains:
                    detected_domains.append(domain)
                detected_computations.extend(computations)
        
        requires_computation = len(detected_domains) > 0
        
        if requires_computation:
            recommendations = f"""
#### Computational Requirements Detected
- **Primary Domain**: {detected_domains[0]}
- **Available Engines**: {', '.join(detected_domains)}
- **Suggested Computations**: {', '.join(set(detected_computations))}

#### Integration Approach
Use `maestro_iae` as the primary computational gateway with:
- Engine routing to appropriate domain
- Precise numerical calculations
- MIA protocol compliance
"""
        else:
            recommendations = """
#### Strategic Analysis Required
No specific computational requirements detected.
Focus on reasoning, planning, and coordination tools.
"""
        
        workflow_steps = []
        if requires_computation:
            workflow_steps = [
                f"Call `maestro_iae` with engine_domain: '{detected_domains[0]}'",
                "Process computational results with machine precision",
                "Integrate numerical findings with strategic analysis",
                "Provide comprehensive response combining computation and reasoning"
            ]
        else:
            workflow_steps = [
                "Use `maestro_orchestrate` for strategic planning",
                "Apply reasoning and context analysis",
                "Provide strategic recommendations",
                "Consider `maestro_iae` if computational needs emerge"
            ]
        
        return {
            "requires_computation": requires_computation,
            "primary_domain": detected_domains[0] if detected_domains else "general",
            "computation_types": list(set(detected_computations)),
            "engine_recommendations": recommendations,
            "workflow_steps": workflow_steps
        } 