"""
Maestro MCP Server - Enhanced Workflow Orchestration

Provides intelligent workflow orchestration tools for LLM enhancement.
The LLM uses these tools for context-aware planning and tool mapping,
then executes workflows using available IDE tools with explicit guidance.
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List

from mcp.server import Server, InitializationOptions
from mcp import stdio_server
from mcp import types

# Import Maestro components for planning and analysis
try:
    from .maestro import MAESTROOrchestrator
    from .maestro.context_aware_orchestrator import ContextAwareOrchestrator
except ImportError:
    from maestro import MAESTROOrchestrator
    from maestro.context_aware_orchestrator import ContextAwareOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaestroMCPServer:
    """
    Maestro MCP Server - Intelligent Workflow Orchestration
    
    Provides advanced orchestration tools to enhance LLM capabilities
    through dynamic tool discovery and intelligent workflow mapping.
    """
    
    def __init__(self):
        self._orchestrator = None
        self._context_orchestrator = None
        self._initialization_error = None
        self._initialization_attempted = False
        
        # Initialize MCP server with proper configuration
        self.app = Server("maestro")
        self._register_handlers()
        
        logger.info("🎭 Maestro MCP Server Ready (Enhanced Workflow Orchestration)")
    
    def _get_orchestrator(self):
        """Get orchestrator with lazy initialization and proper error handling."""
        if self._orchestrator is None and not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                logger.info("🔄 Initializing Maestro orchestration engine...")
                self._orchestrator = MAESTROOrchestrator()
                self._context_orchestrator = ContextAwareOrchestrator()
                logger.info("✅ Maestro orchestration engine ready")
            except Exception as e:
                self._initialization_error = f"Maestro initialization failed: {str(e)}"
                logger.error(f"❌ {self._initialization_error}")
                logger.error(traceback.format_exc())
        
        if self._initialization_error:
            raise RuntimeError(self._initialization_error)
        
        return self._context_orchestrator or self._orchestrator
    
    def _register_handlers(self):
        """Register MCP server handlers and tools with proper error handling."""
        
        @self.app.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available Maestro orchestration tools."""
            try:
                logger.info("📋 Listing Maestro orchestration tools...")
                tools = [
                    types.Tool(
                        name="maestro_orchestrate",
                        description=(
                            "Central Maestro orchestration tool for intelligent workflow management. "
                            "Simply describe any task (debugging, implementation, testing, etc.) and "
                            "Maestro will automatically discover available tools, generate a context-aware "
                            "workflow, and provide explicit execution guidance with intelligent tool mapping. "
                            "Example: 'debug this TypeError and implement a fix' or 'create a REST API with testing'."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Natural language description of what you want to accomplish"
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Additional context to enhance orchestration",
                                    "properties": {
                                        "error_details": {
                                            "type": "string",
                                            "description": "Error message or details if debugging"
                                        },
                                        "current_file": {
                                            "type": "string", 
                                            "description": "Current file path for context"
                                        },
                                        "project_type": {
                                            "type": "string",
                                            "description": "Type of project (web, cli, library, etc.)"
                                        },
                                        "priority": {
                                            "type": "string",
                                            "description": "Task priority level",
                                            "enum": ["low", "medium", "high", "critical"]
                                        }
                                    },
                                    "additionalProperties": true
                                },
                                "focus_phase": {
                                    "type": "string",
                                    "description": "Specific workflow phase to focus on (optional)",
                                    "enum": ["analysis", "implementation", "testing", "quality_assurance", "documentation", "deployment"]
                                }
                            },
                            "required": ["task"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Maestro Orchestration Engine",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": false
                        }
                    ),
                    types.Tool(
                        name="analyze_task_for_planning",
                        description=(
                            "Maestro task analysis and planning tool. "
                            "Analyzes task requirements, selects appropriate workflow template, "
                            "generates execution phases with success criteria, and provides "
                            "system prompt guidance. Use this to get comprehensive orchestration "
                            "guidance before executing a workflow."
                        ),
                        inputSchema={
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
                        },
                        annotations={
                            "title": "Maestro Task Analysis",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
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
                        },
                        annotations={
                            "title": "Maestro Execution Planner",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    types.Tool(
                        name="get_available_templates",
                        description=(
                            "Get list of available Maestro workflow templates. "
                            "Templates provide structured approaches for different types of tasks."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Maestro Template Catalog",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    types.Tool(
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
                        },
                        annotations={
                            "title": "Maestro Template Details",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    types.Tool(
                        name="analyze_task_with_context",
                        description=(
                            "Enhanced Maestro task analysis with dynamic tool discovery and mapping. "
                            "Discovers available MCP tools and IDE capabilities, then provides "
                            "context-aware orchestration guidance with explicit tool usage instructions."
                        ),
                        inputSchema={
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
                                },
                                "force_tool_refresh": {
                                    "type": "boolean",
                                    "description": "Force refresh of tool discovery cache",
                                    "default": False
                                }
                            },
                            "required": ["task_description"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Enhanced Context-Aware Task Analysis",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    types.Tool(
                        name="create_tool_aware_execution_plan",
                        description=(
                            "Create detailed execution plan with explicit tool mappings and usage instructions. "
                            "Provides step-by-step guidance with specific tool commands, examples, and prerequisites."
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
                                    "enum": ["Analysis", "Implementation", "Testing", "Quality_Assurance", "Documentation", "Deployment"]
                                },
                                "force_tool_refresh": {
                                    "type": "boolean",
                                    "description": "Force refresh of tool discovery",
                                    "default": False
                                }
                            },
                            "required": ["task_description"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Tool-Aware Execution Planner",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    types.Tool(
                        name="get_available_tools_with_context",
                        description=(
                            "Discover and catalog all available MCP tools and IDE capabilities "
                            "with comprehensive context information and usage guidance."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Dynamic Tool Discovery",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    )
                ]
                logger.info(f"✅ Successfully listed {len(tools)} Maestro orchestration tools")
                return tools
                
            except Exception as e:
                logger.error(f"❌ Error listing tools: {str(e)}")
                logger.error(traceback.format_exc())
                # Return empty list rather than failing completely
                return []
        
        @self.app.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls for Maestro orchestration tools with comprehensive error handling."""
            try:
                logger.info(f"🔧 Executing tool: {name}")
                logger.debug(f"Tool arguments: {arguments}")
                
                orchestrator = self._get_orchestrator()
                
                if name == "maestro_orchestrate":
                    return await self._handle_maestro_orchestrate(orchestrator, arguments)
                elif name == "analyze_task_for_planning":
                    return await self._handle_analyze_task_for_planning(orchestrator, arguments)
                elif name == "create_execution_plan":
                    return await self._handle_create_execution_plan(orchestrator, arguments)
                elif name == "get_available_templates":
                    return await self._handle_get_available_templates(orchestrator, arguments)
                elif name == "get_template_details":
                    return await self._handle_get_template_details(orchestrator, arguments)
                elif name == "analyze_task_with_context":
                    return await self._handle_analyze_task_with_context(orchestrator, arguments)
                elif name == "create_tool_aware_execution_plan":
                    return await self._handle_create_tool_aware_execution_plan(orchestrator, arguments)
                elif name == "get_available_tools_with_context":
                    return await self._handle_get_available_tools_with_context(orchestrator, arguments)
                else:
                    error_msg = f"Unknown tool: {name}. Available tools: maestro_orchestrate, analyze_task_for_planning, create_execution_plan, get_available_templates, get_template_details, analyze_task_with_context, create_tool_aware_execution_plan, get_available_tools_with_context"
                    logger.error(error_msg)
                    return [types.TextContent(
                        type="text",
                        text=f"❌ {error_msg}"
                    )]
                    
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return [types.TextContent(
                    type="text",
                    text=f"❌ {error_msg}\n\nPlease check your input parameters and try again. If the problem persists, check the server logs for more details."
                )]
    
    async def _handle_maestro_orchestrate(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle the central Maestro orchestration tool."""
        task = arguments["task"]
        context = arguments.get("context", {})
        focus_phase = arguments.get("focus_phase")
        
        logger.info(f"🎭 Maestro orchestrating task: {task[:100]}...")
        
        try:
            # Step 1: Enhance task description with context
            enhanced_task = self._enhance_task_with_context(task, context)
            
            # Step 2: Perform context-aware analysis
            if hasattr(orchestrator, 'analyze_task_with_context'):
                analysis = await orchestrator.analyze_task_with_context(
                    task_description=enhanced_task,
                    detail_level="comprehensive",
                    force_tool_refresh=False
                )
            else:
                # Fallback to basic analysis
                analysis = await orchestrator.analyze_task_for_planning(
                    task_description=enhanced_task,
                    detail_level="comprehensive"
                )
            
            # Step 3: Generate tool-aware execution plan
            if hasattr(orchestrator, 'create_tool_aware_execution_plan'):
                execution_plan = await orchestrator.create_tool_aware_execution_plan(
                    task_description=enhanced_task,
                    phase_focus=focus_phase,
                    force_tool_refresh=False
                )
            else:
                # Fallback to basic execution plan
                execution_plan = await orchestrator.create_execution_plan(
                    task_description=enhanced_task,
                    phase_focus=focus_phase
                )
            
            # Step 4: Format comprehensive orchestration response
            response = self._format_maestro_orchestration_response(
                task=task,
                context=context,
                analysis=analysis,
                execution_plan=execution_plan,
                focus_phase=focus_phase
            )
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Maestro orchestration failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Maestro orchestration failed: {str(e)}"
            )]
    
    def _enhance_task_with_context(self, task: str, context: dict) -> str:
        """Enhance task description with provided context."""
        enhanced_parts = [task]
        
        if context.get("error_details"):
            enhanced_parts.append(f"Error details: {context['error_details']}")
        
        if context.get("current_file"):
            enhanced_parts.append(f"Current file: {context['current_file']}")
        
        if context.get("project_type"):
            enhanced_parts.append(f"Project type: {context['project_type']}")
        
        if context.get("priority"):
            enhanced_parts.append(f"Priority: {context['priority']}")
        
        return ". ".join(enhanced_parts)
    
    def _format_maestro_orchestration_response(
        self,
        task: str,
        context: dict,
        analysis: dict,
        execution_plan: dict,
        focus_phase: str = None
    ) -> str:
        """Format comprehensive Maestro orchestration response."""
        
        response = f"""# 🎭 Maestro Orchestration Complete

**Task:** {task}
**Context:** {self._format_context_summary(context)}

---

## 🔍 Intelligent Analysis Results

"""
        
        # Add analysis results
        if "context_aware_enhancements" in analysis:
            enhancements = analysis["context_aware_enhancements"]
            tool_discovery = enhancements["tool_discovery_results"]
            
            response += f"""### 🛠️ Available Tool Ecosystem
- **MCP Servers Discovered:** {tool_discovery["total_servers_discovered"]}
- **Tools Available:** {tool_discovery["total_tools_available"]}
- **IDE Capabilities:** {tool_discovery["ide_capabilities"]}

### 🎯 Task Analysis
- **Type:** {analysis["task_analysis"]["task_type"].title()}
- **Complexity:** {analysis["task_analysis"]["complexity"].title()}
- **Approach:** {analysis["template_used"].replace('_', ' ').title()}

"""
        else:
            response += f"""### 🎯 Task Analysis
- **Type:** {analysis["task_analysis"]["task_type"].title()}
- **Complexity:** {analysis["task_analysis"]["complexity"].title()}
- **Approach:** {analysis["template_used"].replace('_', ' ').title()}

"""
        
        # Add execution guidance
        response += "## 🚀 Orchestrated Execution Plan\n\n"
        
        if focus_phase and "explicit_tool_instructions" in execution_plan:
            # Focused phase execution
            instructions = execution_plan["explicit_tool_instructions"]
            response += f"### 🎯 Focused Phase: {focus_phase.title()}\n\n"
            
            for i, instruction in enumerate(instructions, 1):
                response += f"""**Step {i}: {instruction['tool']}** ({instruction['priority_level']})
- **Server:** {instruction['server']}
- **Action:** {instruction['instruction']}
- **Command:** `{instruction['exact_command']}`
- **Example:** `{instruction['example']}`
- **Prerequisites:** {', '.join(instruction['prerequisites']) if instruction['prerequisites'] else 'None'}
- **Expected Result:** {instruction['expected_output']}

"""
        elif "execution_steps" in execution_plan:
            # Complete workflow execution
            for i, phase_step in enumerate(execution_plan["execution_steps"], 1):
                phase_name = phase_step["phase"]
                response += f"### Phase {i}: {phase_name}\n\n"
                
                for j, tool_step in enumerate(phase_step["tools_to_execute"], 1):
                    response += f"""**{i}.{j} {tool_step['step']}**
- **Command:** `{tool_step['command']}`
- **Rationale:** {tool_step['rationale']}
- **Example:** `{tool_step['example']}`
- **Expected Result:** {tool_step['expected_result']}

"""
        else:
            # Fallback to basic execution plan
            response += f"""### Execution Sequence
{chr(10).join(f"{i+1}. {phase}" for i, phase in enumerate(execution_plan.get('execution_sequence', [])))}

### Success Criteria
{chr(10).join(f"- {criteria}" for criteria in execution_plan.get('critical_success_factors', []))}
"""
        
        response += """
---

## 🎯 Next Steps

You now have intelligent, context-aware orchestration guidance! The LLM can follow these explicit tool mappings and workflow steps to accomplish your task efficiently. Each step includes the exact tools to use, when to use them, and what to expect.

**Maestro has orchestrated your workflow for optimal execution.** 🎭✨
"""
        
        return response
    
    def _format_context_summary(self, context: dict) -> str:
        """Format context information for display."""
        if not context:
            return "General task"
        
        summary_parts = []
        if context.get("error_details"):
            summary_parts.append(f"Error: {context['error_details'][:50]}...")
        if context.get("current_file"):
            summary_parts.append(f"File: {context['current_file']}")
        if context.get("project_type"):
            summary_parts.append(f"Project: {context['project_type']}")
        if context.get("priority"):
            summary_parts.append(f"Priority: {context['priority']}")
        
        return " | ".join(summary_parts) if summary_parts else "General task"
    
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
            
            response = "## Available Maestro Workflow Templates\n\n"
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
    
    async def _handle_analyze_task_with_context(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle enhanced context-aware task analysis."""
        task_description = arguments["task_description"]
        detail_level = arguments.get("detail_level", "comprehensive")
        force_tool_refresh = arguments.get("force_tool_refresh", False)
        
        logger.info(f"🔍 Context-aware analysis: {task_description[:100]}...")
        
        try:
            # Use context-aware orchestrator if available
            if hasattr(orchestrator, 'analyze_task_with_context'):
                analysis = await orchestrator.analyze_task_with_context(
                    task_description=task_description,
                    detail_level=detail_level,
                    force_tool_refresh=force_tool_refresh
                )
            else:
                # Fallback to regular analysis
                analysis = await orchestrator.analyze_task_for_planning(
                    task_description=task_description,
                    detail_level=detail_level
                )
            
            response = self._format_context_aware_analysis_response(analysis)
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Context-aware analysis failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Context-aware analysis failed: {str(e)}"
            )]
    
    async def _handle_create_tool_aware_execution_plan(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle tool-aware execution plan creation."""
        task_description = arguments["task_description"]
        phase_focus = arguments.get("phase_focus")
        force_tool_refresh = arguments.get("force_tool_refresh", False)
        
        try:
            # Use context-aware orchestrator if available
            if hasattr(orchestrator, 'create_tool_aware_execution_plan'):
                plan = await orchestrator.create_tool_aware_execution_plan(
                    task_description=task_description,
                    phase_focus=phase_focus,
                    force_tool_refresh=force_tool_refresh
                )
            else:
                # Fallback to regular execution plan
                plan = await orchestrator.create_execution_plan(
                    task_description=task_description,
                    phase_focus=phase_focus
                )
            
            response = self._format_tool_aware_execution_plan_response(plan, phase_focus)
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Tool-aware execution plan creation failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Tool-aware execution plan creation failed: {str(e)}"
            )]
    
    async def _handle_get_available_tools_with_context(self, orchestrator, arguments: dict) -> list[types.TextContent]:
        """Handle tool discovery with context."""
        try:
            # Use context-aware orchestrator if available
            if hasattr(orchestrator, 'get_available_tools_with_context'):
                tools_info = await orchestrator.get_available_tools_with_context()
            else:
                # Fallback to basic template list
                templates = orchestrator.get_available_templates()
                tools_info = {"available_templates": templates}
            
            response = self._format_available_tools_response(tools_info)
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text", 
                text=f"❌ Failed to discover tools: {str(e)}"
            )]
    
    def _format_task_analysis_response(self, analysis: Dict[str, Any]) -> str:
        """Format task analysis response for user."""
        return f"""## Maestro Task Analysis ✅

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
    
    def _format_context_aware_analysis_response(self, analysis: Dict[str, Any]) -> str:
        """Format context-aware analysis response."""
        if "context_aware_enhancements" in analysis:
            # Enhanced analysis with tool context
            base_analysis = {k: v for k, v in analysis.items() if k != "context_aware_enhancements"}
            enhancements = analysis["context_aware_enhancements"]
            
            response = f"""## Enhanced Maestro Task Analysis ✅

**Task Type:** {base_analysis['task_analysis']['task_type'].title()}
**Complexity:** {base_analysis['task_analysis']['complexity'].title()}
**Template Used:** {base_analysis['template_used'].replace('_', ' ').title()}

### 🔧 Tool Discovery Results
- **MCP Servers Discovered:** {enhancements['tool_discovery_results']['total_servers_discovered']}
- **Tools Available:** {enhancements['tool_discovery_results']['total_tools_available']}
- **IDE Capabilities:** {enhancements['tool_discovery_results']['ide_capabilities']}

### 🎯 Enhanced System Prompt Guidance
**Role:** {enhancements['context_aware_system_prompt']['role']}

**Tool Ecosystem Context:**
- Available MCP Servers: {enhancements['context_aware_system_prompt']['tool_ecosystem_context']['available_mcp_servers']}
- Total Tools: {enhancements['context_aware_system_prompt']['tool_ecosystem_context']['total_tools_available']}
- Key Capabilities: {', '.join(enhancements['context_aware_system_prompt']['tool_ecosystem_context']['key_capabilities'])}

**Enhanced Approach Guidelines:**
{chr(10).join(f"- {guideline}" for guideline in enhancements['context_aware_system_prompt']['enhanced_approach_guidelines'])}

**Tool Usage Principles:**
{chr(10).join(f"- {principle}" for principle in enhancements['context_aware_system_prompt']['tool_usage_principles'])}

### 🗺️ Tool-Aware Workflow Phases
"""
            
            for i, phase in enumerate(enhancements['enhanced_workflow'], 1):
                phase_info = phase['phase_info']
                tool_summary = phase['tool_summary']
                
                response += f"""
**Phase {i}: {phase_info['phase'].value}**
- Description: {phase_info['description']}
- Tools Mapped: {tool_summary['total_tools']} ({tool_summary['primary_tools']} primary, {tool_summary['secondary_tools']} secondary)
- Servers Used: {', '.join(tool_summary['tool_categories'])}
"""
            
            response += f"""

### 📋 Tool Execution Plan Summary
- **Total Phases:** {enhancements['tool_execution_plan']['total_phases']}
- **Estimated Tools to Execute:** {enhancements['tool_execution_plan']['estimated_total_tools']}
- **Global Prerequisites:** {', '.join(enhancements['tool_execution_plan']['global_prerequisites']) if enhancements['tool_execution_plan']['global_prerequisites'] else 'None'}

**Next Step:** Use `create_tool_aware_execution_plan` for detailed tool-specific implementation guidance.
"""
            return response
        else:
            # Fallback to standard format
            return self._format_task_analysis_response(analysis)
    
    def _format_tool_aware_execution_plan_response(self, plan: Dict[str, Any], phase_focus: str = None) -> str:
        """Format tool-aware execution plan response."""
        if "explicit_tool_instructions" in plan:
            # Focused phase plan
            phase = plan["focused_phase"]
            instructions = plan["explicit_tool_instructions"]
            
            response = f"""## Tool-Aware Execution Plan: {phase['phase_info']['phase'].value} ✅

**Phase Description:** {phase['phase_info']['description']}

### 🔧 Explicit Tool Instructions
"""
            for i, instruction in enumerate(instructions, 1):
                response += f"""
**{i}. {instruction['tool']}** ({instruction['priority_level']})
- **Server:** {instruction['server']}
- **Instruction:** {instruction['instruction']}
- **Command:** `{instruction['exact_command']}`
- **Example:** `{instruction['example']}`
- **Prerequisites:** {', '.join(instruction['prerequisites']) if instruction['prerequisites'] else 'None'}
- **Expected Output:** {instruction['expected_output']}
"""
            
            response += f"""
### ✅ Success Validation Steps
{chr(10).join(f"- {step}" for step in plan['success_validation_steps'])}
"""
            return response
            
        elif "complete_workflow" in plan:
            # Complete workflow plan
            workflow = plan["complete_workflow"]
            tool_summary = plan["tool_summary"]
            
            response = f"""## Complete Tool-Aware Execution Plan ✅

### 📊 Workflow Overview
- **Total Phases:** {len(workflow)}
- **Workflow Complexity:** {tool_summary['workflow_complexity']}
- **Unique Tools Required:** {tool_summary['total_unique_tools']}
- **MCP Servers Involved:** {tool_summary['server_count']}

### 🗺️ Execution Sequence
{chr(10).join(f"{i+1}. {phase}" for i, phase in enumerate(plan['execution_sequence']))}

### 🔧 Tool Distribution
- **Primary Tools:** {tool_summary['tools_by_priority']['primary']}
- **Secondary Tools:** {tool_summary['tools_by_priority']['secondary']}
- **Fallback Tools:** {tool_summary['tools_by_priority']['fallback']}

### 📋 Tools by Server
"""
            for server, tools in tool_summary['tools_by_server'].items():
                response += f"- **{server}:** {', '.join(tools)}\n"
            
            if plan.get('critical_dependencies'):
                response += f"""
### ⚠️ Critical Dependencies
"""
                for dep in plan['critical_dependencies']:
                    criticality = "🔴 CRITICAL" if dep['critical'] else "🟡 IMPORTANT"
                    response += f"- **{dep['phase']}** {criticality}: {', '.join(dep['requires'])}\n"
            
            response += f"""

**Next Step:** Use `create_tool_aware_execution_plan` with a specific `phase_focus` for detailed tool guidance.
"""
            return response
        else:
            # Fallback to standard format
            return self._format_execution_plan_response(plan, phase_focus)
    
    def _format_available_tools_response(self, tools_info: Dict[str, Any]) -> str:
        """Format available tools response."""
        if "discovery_summary" in tools_info:
            # Enhanced tool discovery response
            discovery = tools_info["discovery_summary"]
            tools_by_capability = tools_info.get("tools_by_capability", {})
            ide_capabilities = tools_info.get("ide_capabilities", [])
            
            response = f"""## Dynamic Tool Discovery Results ✅

### 📊 Discovery Summary
- **MCP Servers Found:** {len(discovery.get('mcp_servers', {}))}
- **Total Tools Discovered:** {discovery.get('total_tools_discovered', 0)}
- **IDE Capabilities:** {len(ide_capabilities)}
- **Discovery Timestamp:** {discovery.get('discovery_timestamp')}

### 🔧 Tools by Capability
"""
            for capability, tools in tools_by_capability.items():
                response += f"""
**{capability.replace('_', ' ').title()}:**
"""
                for tool in tools:
                    response += f"- `{tool['name']}` ({tool['server']}): {tool['description']}\n"
            
            if ide_capabilities:
                response += f"""
### 💻 IDE Capabilities
"""
                for cap in ide_capabilities:
                    response += f"- **{cap['name']}** ({cap['category']}): {cap['description']}\n"
                    response += f"  Usage: {cap['usage_pattern']}\n"
            
            response += f"""

### 🎯 Usage Guidance
Use `analyze_task_with_context` to get task-specific tool mappings and explicit usage instructions.
All discovered tools will be automatically mapped to relevant workflow phases.
"""
            return response
        else:
            # Fallback response
            templates = tools_info.get("available_templates", [])
            response = f"""## Available Resources

### Templates
{chr(10).join(f"- {template}" for template in templates)}

Use the enhanced tools for dynamic tool discovery and context-aware orchestration.
"""
            return response
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream, 
                write_stream,
                InitializationOptions(
                    server_name="maestro",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {}
                    }
                )
            )


async def main():
    """Main entry point for Maestro MCP Server."""
    server_instance = MaestroMCPServer()
    await server_instance.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())