# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Maestro Tools - Enhanced tool implementations for MCP

This module provides orchestration, intelligence amplification, and enhanced tools
with proper lazy loading to optimize Smithery scanning performance.
"""

import logging
from typing import List, Dict, Any
from mcp.server.fastmcp import Context
from mcp.types import TextContent # Added for clarity in return types
import json
import traceback

# Set up logging - lightweight operation
logger = logging.getLogger(__name__)

class MaestroTools:
    """
    Provides enhanced tools for the Maestro MCP server.
    
    Implements lazy loading to ensure that Smithery tool scanning
    doesn't timeout by avoiding heavy dependency loading during startup.
    """
    
    def __init__(self):
        # Initialize flags for lazy loading
        self._computational_tools = None
        self._engines_loaded = False
        self._orchestrator_loaded = False
        self._enhanced_tool_handlers = None # Added for other enhanced tools
        
        logger.info("🎭 MaestroTools initialized with lazy loading")
    
    def _ensure_computational_tools(self):
        """Lazy load computational tools only when needed."""
        if self._computational_tools is None:
            try:
                # Import only when method is called, not at initialization
                from .computational_tools import ComputationalTools
                self._computational_tools = ComputationalTools()
                logger.info("✅ ComputationalTools loaded on first use")
            except ImportError as e:
                logger.error(f"❌ Failed to import ComputationalTools: {e}")
                self._computational_tools = None
    
    def _ensure_enhanced_tool_handlers(self):
        """Lazy load enhanced tool handlers only when needed."""
        if self._enhanced_tool_handlers is None:
            try:
                from .maestro.enhanced_tools import EnhancedToolHandlers
                self._enhanced_tool_handlers = EnhancedToolHandlers()
                logger.info("✅ EnhancedToolHandlers loaded on first use")
            except ImportError as e:
                logger.error(f"❌ Failed to import EnhancedToolHandlers: {e}")
                self._enhanced_tool_handlers = None
    
    async def _call_internal_tool(self, ctx: Context, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Helper to call other MCP tools internally and return formatted string output."""
        try:
            if tool_name == "maestro_iae":
                self._ensure_computational_tools()
                if self._computational_tools:
                    results = await self._computational_tools.handle_tool_call(tool_name, arguments)
                    # Assuming results is a list of TextContent; concatenate their text
                    return "\n".join([t.text for t in results])
                else:
                    return f"❌ Error: Computational tools not available for {tool_name}."
            elif tool_name in ["maestro_search", "maestro_scrape", "maestro_execute", "maestro_error_handler", "maestro_temporal_context"]:
                self._ensure_enhanced_tool_handlers()
                if self._enhanced_tool_handlers:
                    # Directly call the handler method for these tools
                    handler_map = {
                        "maestro_search": self._enhanced_tool_handlers.search,
                        "maestro_scrape": self._enhanced_tool_handlers.scrape,
                        "maestro_execute": self._enhanced_tool_handlers.execute,
                        "maestro_error_handler": self._enhanced_tool_handlers.handle_error,
                        "maestro_temporal_context": self._enhanced_tool_handlers.analyze_temporal_context,
                    }
                    if tool_name in handler_map:
                        result_text = await handler_map[tool_name](**arguments)
                        return f"✅ Tool '{tool_name}' executed. Result: {result_text}"
                    else:
                        return f"❌ Error: Handler for '{tool_name}' not found in EnhancedToolHandlers."
                else:
                    return f"❌ Error: Enhanced tool handlers not available for {tool_name}."
            else:
                return f"❌ Error: Internal tool call to '{tool_name}' is not supported via this helper."
        except Exception as e:
            logger.error(f"❌ Internal tool call to {tool_name} failed: {e}")
            return f"❌ Internal tool call failed for '{tool_name}': {str(e)}"

    async def orchestrate_task(self, ctx: Context, task_description: str, context: Dict[str, Any] = None, success_criteria: Dict[str, Any] = None, complexity_level: str = "moderate") -> str:
        """
        Orchestrates complex tasks, leveraging Mixture-of-Agents (MoA) for computational
        amplification and dynamic tool/engine selection. Includes 'show your work' output.
        """
        try:
            # Lazy load necessary components
            self._ensure_computational_tools()
            self._ensure_enhanced_tool_handlers()

            orchestration_plan_prompt = f"""
You are an intelligent workflow orchestrator. Your goal is to break down a complex task
and plan the sequence of actions, including which specialized computational engines or
tools should be used.
If the task requires numerical computations or analysis, identify the appropriate
'maestro_iae' calls (engine_domain, computation_type, parameters).
If it requires other operations (search, scrape, execute, error_handling, temporal_context),
identify calls to 'maestro_search', 'maestro_scrape', 'maestro_execute', 'maestro_error_handler',
or 'maestro_temporal_context'.

Output your plan as a JSON object with the following structure:
{{
    "requires_moa": boolean,
    "steps": [
        {{
            "type": "tool_call",
            "tool_name": "maestro_iae" | "maestro_search" | "maestro_scrape" | "maestro_execute" | "maestro_error_handler" | "maestro_temporal_context",
            "arguments": {{...}}
        }},
        {{
            "type": "reasoning",
            "description": "Explanation of reasoning step"
        }}
        // ... more steps
    ],
    "final_synthesis_required": boolean,
    "moa_aggregation_strategy": "consensus" | "confidence_weighted" | "llm_synthesis" | "none"
}}

Here is the task: "{task_description}"
Context: {context}
Success Criteria: {success_criteria}
Complexity Level: {complexity_level}

Consider if multiple computations/tools are needed and how their results would be aggregated.
Prioritize using available specialized engines via 'maestro_iae' for computations.
"""
            logger.info(f"🎭 Requesting orchestration plan from LLM for task: {task_description[:100]}...")
            
            # Use ctx.sample for internal LLM call
            # This is where the LLM will provide the MoA plan
            orchestration_response_json = await ctx.sample(
                prompt=orchestration_plan_prompt,
                response_format={"type": "json_object"}
            )
            
            orchestration_plan = orchestration_response_json.json()
            logger.debug(f"🎭 Orchestration plan received: {orchestration_plan}")

            full_workflow_output = []
            final_results_for_synthesis = []

            full_workflow_output.append(f"# 🎭 Maestro Orchestration for: {task_description}\n")
            full_workflow_output.append(f"## Orchestration Plan (Generated by LLM)\n```json\n{json.dumps(orchestration_plan, indent=2)}\n```\n")
            full_workflow_output.append(f"## Execution Log & Intermediate Results\n")

            for i, step in enumerate(orchestration_plan.get("steps", []), 1):
                full_workflow_output.append(f"### Step {i}: {step.get('type', 'Unknown Type')}\n")
                if step["type"] == "tool_call":
                    tool_name = step.get("tool_name")
                    arguments = step.get("arguments", {})
                    full_workflow_output.append(f"Calling tool: `{tool_name}` with arguments: ```json\n{json.dumps(arguments, indent=2)}\n```\n")
                    
                    tool_output = await self._call_internal_tool(ctx, tool_name, arguments)
                    full_workflow_output.append(f"Tool Output:\n```\n{tool_output}\n```\n")
                    final_results_for_synthesis.append({
                        "step_name": f"Tool Call: {tool_name}",
                        "output": tool_output
                    })
                elif step["type"] == "reasoning":
                    description = step.get("description", "No description provided.")
                    full_workflow_output.append(f"Reasoning Step: {description}\n")
                    final_results_for_synthesis.append({
                        "step_name": "Reasoning",
                        "output": description
                    })
                else:
                    full_workflow_output.append(f"Unsupported step type: {step.get('type')}\n")
            
            final_synthesis_output = ""
            if orchestration_plan.get("final_synthesis_required", False) and final_results_for_synthesis:
                synthesis_prompt = f"""
You are a highly capable AI assistant tasked with synthesizing diverse results from a complex workflow.
Combine the following intermediate results and observations into a coherent, comprehensive final answer.
Explain your reasoning and how the different pieces of information contribute to the final conclusion.
Highlight any key insights, contradictions, or uncertainties.

Intermediate Results:
{json.dumps(final_results_for_synthesis, indent=2)}

Original Task: "{task_description}"
Original Context: {context}
Original Success Criteria: {success_criteria}

Based on the above, provide the final synthesized answer and a brief explanation of how you arrived at it.
Ensure to include a "show your work" section demonstrating the aggregation strategy.
"""
                logger.info(f"🎭 Requesting final synthesis from LLM for task: {task_description[:100]}...")
                synthesis_response = await ctx.sample(
                    prompt=synthesis_prompt,
                    response_format={"type": "text"} # Expecting rich text output for synthesis
                )
                final_synthesis_output = synthesis_response.text
                full_workflow_output.append(f"## Final Synthesized Answer (Generated by LLM)\n{final_synthesis_output}\n")
            else:
                full_workflow_output.append(f"## Final Answer\nNo specific final synthesis required. The output above contains the direct results of the workflow.\n")

            full_workflow_output.append(f"\n## Overall Success Metrics\n{self._format_success_criteria(success_criteria)}\n")
            full_workflow_output.append(f"\n*This orchestration utilized advanced Mixture-of-Agents (MoA) principles for enhanced computational intelligence and transparency.*")

            return "".join(full_workflow_output)

        except Exception as e:
            logger.error(f"❌ Orchestration failed: {str(e)}\n{traceback.format_exc()}")
            return f"❌ **Orchestration Failed**\n\nAn unexpected error occurred during orchestration: {str(e)}\n\nPlease review the server logs for more details. (Traceback below)\n```\n{traceback.format_exc()}\n```"

    async def _handle_iae_discovery(self, arguments: dict) -> List[TextContent]:
        """Handle Intelligence Amplification Engine discovery and mapping."""
        try:
            # Lazy import TextContent only when needed
            from mcp.types import TextContent
            
            task_type = arguments.get("task_type", "general")
            domain_context = arguments.get("domain_context", "")
            complexity_requirements = arguments.get("complexity_requirements", {})
            
            logger.info(f"🔍 Discovering IAEs for: {task_type}")
            
            # Lazy load computational tools only when needed
            self._ensure_computational_tools()
            
            # Get available engines from computational tools
            available_engines = {}
            if self._computational_tools:
                available_engines = self._computational_tools.get_available_engines()
            
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
            # Import TextContent here to ensure lazy loading
            from mcp.types import TextContent
            logger.error(f"❌ IAE discovery failed: {str(e)}")
            return [TextContent(type="text", text=f"❌ **Discovery Failed**\n\nError: {str(e)}")]

    # Continue with the rest of the class, ensuring imports are inside methods
    
    def _analyze_computational_requirements(self, task_description: str, context: dict) -> dict:
        """Analyze if a task requires computational engines or strategic tools."""
        # Implementation that doesn't require heavy imports
        # Default result with minimal assumptions
        result = {
            "requires_computation": False,
            "primary_domain": "general",
            "computation_types": [],
            "workflow_steps": [],
            "engine_recommendations": "No specialized engines required for this task."
        }
        
        # Simple keyword-based analysis without heavy imports
        computation_keywords = [
            "calculate", "compute", "solve", "equation", "formula", 
            "optimization", "statistics", "probability", "quantum",
            "simulation", "numerical", "matrix", "vector", "algorithm"
        ]
        
        # Check if any computation keywords are present
        if any(keyword in task_description.lower() for keyword in computation_keywords):
            result["requires_computation"] = True
            result["primary_domain"] = "advanced_mathematics"
            result["computation_types"] = ["optimization", "statistical_analysis"]
            result["workflow_steps"] = [
                "Define computational parameters",
                "Select appropriate engine domain",
                "Execute computation",
                "Interpret results"
            ]
            result["engine_recommendations"] = """
- **Mathematics Engine**: For optimization, statistics, and symbolic math
- **Quantum Physics Engine**: For quantum simulations and calculations
- **Data Analysis Engine**: For statistical analysis and modeling
"""
        
        return result
    
    def _format_success_criteria(self, criteria: dict) -> str:
        """Format success criteria for output."""
        if not criteria:
            return "- Task completed according to requirements\n- Results verified for accuracy\n- Documentation provided"
        
        result = ""
        for key, value in criteria.items():
            result += f"- **{key}**: {value}\n"
        
        return result
    
    def _get_engine_recommendations(self, task_type: str, domain_context: str) -> str:
        """Get engine recommendations based on task type."""
        recommendations = {
            "mathematics": "Use the advanced_mathematics engine for symbolic computation, optimization, and numerical analysis.",
            "physics": "The quantum_physics engine provides quantum simulation and physical modeling capabilities.",
            "data_analysis": "Statistical engines can process datasets, perform statistical tests, and generate models.",
            "general": "Start with the mathematics engine for general computational needs."
        }
        
        return recommendations.get(task_type.lower(), recommendations["general"])

    async def _handle_tool_selection(self, arguments: dict) -> List[TextContent]:
        """Handle tool selection recommendations."""
        try:
            # Import TextContent here to ensure lazy loading
            from mcp.types import TextContent
            
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
            # Import TextContent here to ensure lazy loading
            from mcp.types import TextContent
            logger.error(f"❌ Tool selection analysis failed: {str(e)}")
            return [TextContent(type="text", text=f"❌ **Tool Selection Failed**\n\nError: {str(e)}")] 
