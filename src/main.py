"""
MAESTRO Protocol MCP Server - Corrected Architecture

Provides PLANNING and ANALYSIS tools for LLM enhancement.
The LLM uses these tools for guidance and orchestration,
then executes workflows using available IDE tools.
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List

from mcp.server import server
from mcp.server.stdio import stdio_server
from mcp import types
from mcp.server import InitializationOptions

# Import MAESTRO components for planning and analysis
try:
    from .maestro import MAESTROOrchestrator
except ImportError:
    from maestro import MAESTROOrchestrator

logger = logging.getLogger(__name__)


class TanukiMCPOrchestra:
    """
    MAESTRO Protocol MCP Server - Planning and Analysis Engine
    
    Provides orchestration PLANNING tools to enhance LLM capabilities.
    The LLM uses these tools for guidance, then executes using IDE tools.
    """
    
    def __init__(self):
        self._orchestrator = None
        self._initialization_error = None
        self._initialization_attempted = False
        
        # Initialize MCP server
        self.app = server.Server("tanukimcp-orchestra")
        self._register_handlers()
        
        logger.info("🎭 MAESTRO Protocol MCP Server Ready (Planning Engine)")
    
    def _get_orchestrator(self):
        """Get orchestrator with lazy initialization."""
        if self._orchestrator is None and not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                self._orchestrator = MAESTROOrchestrator()
                logger.info("✅ MAESTRO orchestration planner ready")
            except Exception as e:
                self._initialization_error = f"MAESTRO initialization failed: {str(e)}"
                logger.error(f"❌ {self._initialization_error}")
        
        if self._initialization_error:
            raise RuntimeError(self._initialization_error)
        
        return self._orchestrator
    
    def _register_handlers(self):
        """Register MCP server handlers and tools."""
        
        @self.app.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available MAESTRO planning tools."""
            return [
                types.Tool(
                    name="analyze_task_for_planning",
                    description=(
                        "MAESTRO Protocol task analysis and planning tool. "
                        "Analyzes task requirements, selects appropriate workflow template, "
                        "generates execution phases with success criteria, and provides "
                        "system prompt guidance. Use this to get comprehensive orchestration "
                        "guidance before executing a workflow."
                    ),                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Natural language description of the task to analyze"
                            },
                            "detail_level": {
                                "type": "string",
                                "description": "Analysis detail level",
                                "enum": ["fast", "balanced", "comprehensive"],
                                "default": "comprehensive"
                            }
                        },
                        "required": ["task_description"],
                        "additionalProperties": False
                    }
                ),
                types.Tool(
                    name="create_execution_plan",
                    description=(
                        "Create detailed execution plan for task implementation. "
                        "Provides step-by-step guidance, success criteria, and tool recommendations "
                        "for executing a specific workflow phase or entire task."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Task to create execution plan for"
                            },
                            "phase_focus": {
                                "type": "string",
                                "description": "Specific phase to focus on (optional)",
                                "enum": ["Analysis", "Implementation", "Testing", "Quality_Assurance", "Documentation"]
                            }
                        },
                        "required": ["task_description"],
                        "additionalProperties": False
                    }
                ),
                types.Tool(
                    name="get_available_templates",
                    description=(
                        "Get list of available MAESTRO workflow templates. "
                        "Templates provide structured approaches for different types of tasks."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                ),                types.Tool(
                    name="get_template_details",
                    description=(
                        "Get detailed information about a specific workflow template. "
                        "Includes system prompt guidance, execution phases, and quality standards."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template_name": {
                                "type": "string",
                                "description": "Name of the template to get details for"
                            }
                        },
                        "required": ["template_name"],
                        "additionalProperties": False
                    }
                )
            ]
        
        @self.app.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls for MAESTRO planning tools."""
            try:
                orchestrator = self._get_orchestrator()
                
                if name == "analyze_task_for_planning":
                    return await self._handle_analyze_task_for_planning(orchestrator, arguments)
                elif name == "create_execution_plan":
                    return await self._handle_create_execution_plan(orchestrator, arguments)
                elif name == "get_available_templates":
                    return await self._handle_get_available_templates(orchestrator, arguments)
                elif name == "get_template_details":
                    return await self._handle_get_template_details(orchestrator, arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}\n{traceback.format_exc()}")
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error executing {name}: {str(e)}\n\nPlease check your input parameters and try again."
                )]
    
    async def _handle_analyze_task_for_planning(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle task analysis for planning."""
        task_description = arguments["task_description"]
        detail_level = arguments.get("detail_level", "comprehensive")
        
        logger.info(f"🔍 Analyzing task for planning: {task_description[:100]}...")
        
        try:
            analysis = await orchestrator.analyze_task_for_planning(
                task_description=task_description,
                detail_level=detail_level
            )
            
            response = self._format_task_analysis_response(analysis)
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Task analysis failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Task analysis failed: {str(e)}"
            )]    
    async def _handle_create_execution_plan(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle execution plan creation."""
        task_description = arguments["task_description"]
        phase_focus = arguments.get("phase_focus")
        
        try:
            plan = await orchestrator.create_execution_plan(
                task_description=task_description,
                phase_focus=phase_focus
            )
            
            response = self._format_execution_plan_response(plan, phase_focus)
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Execution plan creation failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Execution plan creation failed: {str(e)}"
            )]
    
    async def _handle_get_available_templates(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle getting available templates."""
        try:
            templates = orchestrator.get_available_templates()
            
            response = "## Available MAESTRO Workflow Templates\n\n"
            for i, template in enumerate(templates, 1):
                response += f"{i}. **{template.replace('_', ' ').title()}**\n"
            
            response += "\nUse `get_template_details` with a template name to get detailed information."
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text", 
                text=f"❌ Failed to get templates: {str(e)}"
            )]
    
    async def _handle_get_template_details(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle getting template details."""
        template_name = arguments["template_name"]
        
        try:
            details = orchestrator.get_template_details(template_name)
            
            if "error" in details:
                return [types.TextContent(type="text", text=details["error"])]
            
            response = self._format_template_details_response(details)
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text", 
                text=f"❌ Failed to get template details: {str(e)}"
            )]    
    def _format_task_analysis_response(self, analysis: Dict[str, Any]) -> str:
        """Format task analysis response for user."""
        return f"""## MAESTRO Protocol Task Analysis ✅

**Task Type:** {analysis['task_analysis']['task_type'].title()}
**Complexity:** {analysis['task_analysis']['complexity'].title()}
**Template Used:** {analysis['template_used'].replace('_', ' ').title()}

### System Prompt Guidance
**Role:** {analysis['system_prompt_guidance']['role']}

**Approach Guidelines:**
{chr(10).join(f"- {guideline}" for guideline in analysis['system_prompt_guidance']['approach_guidelines'])}

**Quality Standards:**
{chr(10).join(f"- {standard}" for standard in analysis['system_prompt_guidance']['quality_standards'])}

### Execution Phases
{chr(10).join(f"**{i+1}. {phase['phase'].replace('_', ' ')}:** {phase['description']}" for i, phase in enumerate(analysis['execution_phases']))}

### Success Criteria
{chr(10).join(f"- {criteria}" for criteria in analysis['success_criteria'])}

### Recommended Tools
{chr(10).join(f"- {tool}" for tool in analysis['recommended_tools'])}

**Next Step:** Use `create_execution_plan` to get detailed implementation guidance.
"""
    
    def _format_execution_plan_response(self, plan: Dict[str, Any], phase_focus: str = None) -> str:
        """Format execution plan response."""
        if phase_focus and "focused_phase" in plan:
            phase = plan["focused_phase"]
            return f"""## Focused Execution Plan: {phase['phase'].replace('_', ' ')} ✅

**Phase Description:** {phase['description']}

**Success Criteria:**
{chr(10).join(f"- {criteria}" for criteria in phase['success_criteria'])}

**Detailed Steps:**
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(plan['detailed_steps']))}

**Tools Needed:**
{chr(10).join(f"- {tool}" for tool in plan['tools_needed'])}

**Validation:** Ensure all success criteria are met before proceeding.
"""
        else:
            return f"""## Complete Execution Plan ✅

**Execution Sequence:**
{chr(10).join(f"{i+1}. {phase}" for i, phase in enumerate(plan['execution_sequence']))}

**Critical Success Factors:**
{chr(10).join(f"- {factor}" for factor in plan['critical_success_factors'])}

**Quality Gates:**
{chr(10).join(f"- {gate}" for gate in plan['quality_gates'])}

Use `create_execution_plan` with a specific phase_focus for detailed phase guidance.
"""    
    def _format_template_details_response(self, details: Dict[str, Any]) -> str:
        """Format template details response."""
        return f"""## Template Details: {details['template_name'].replace('_', ' ').title()} ✅

**Description:** {details['description']}

### System Prompt Guidance
**Role:** {details['system_prompt_guidance']['role']}

**Expertise Areas:**
{chr(10).join(f"- {area}" for area in details['system_prompt_guidance']['expertise'])}

**Approach Guidelines:**
{chr(10).join(f"- {approach}" for approach in details['system_prompt_guidance']['approach'])}

### Execution Phases
{chr(10).join(f"**{phase['phase'].replace('_', ' ')}:** {phase['description']}" for phase in details['execution_phases'])}

### Verification Methods
{chr(10).join(f"- {method}" for method in details['verification_methods'])}

This template provides structured guidance for {details['template_name'].replace('_', ' ')} tasks.
"""
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream, 
                write_stream,
                InitializationOptions(
                    server_name="tanukimcp-orchestra",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {}
                    }
                )
            )


async def main():
    """Main entry point for MAESTRO Protocol MCP Server."""
    server_instance = TanukiMCPOrchestra()
    await server_instance.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())