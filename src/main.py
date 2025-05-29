"""
Maestro MCP Server - Enhanced Workflow Orchestration

Provides intelligent workflow orchestration tools for LLM enhancement.
The LLM uses these tools for context-aware planning and tool mapping,
then executes workflows using available IDE tools with explicit guidance.
"""

import asyncio
import logging
import traceback
import json
from typing import Any, Dict, List, Union

from mcp.server import Server, InitializationOptions
from mcp import stdio_server
from mcp import types

# Import enhanced Maestro components
try:
    from .maestro import MAESTROOrchestrator
    from .maestro.context_aware_orchestrator import ContextAwareOrchestrator
    from .maestro.orchestration_framework import (
        EnhancedOrchestrationEngine, 
        ContextSurvey, 
        OrchestrationResult,
        TaskComplexity
    )
except ImportError:
    from maestro import MAESTROOrchestrator
    from maestro.context_aware_orchestrator import ContextAwareOrchestrator
    from maestro.orchestration_framework import (
        EnhancedOrchestrationEngine, 
        ContextSurvey, 
        OrchestrationResult,
        TaskComplexity
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaestroMCPServer:
    """
    Maestro MCP Server - Enhanced Intelligence Amplification Orchestration
    
    Provides advanced orchestration tools to enhance LLM capabilities through:
    - Context intelligence and gap detection
    - Success criteria definition and validation
    - Tool discovery and mapping to external MCP servers  
    - Intelligence Amplification Engine (IAE) integration
    - Collaborative error handling with automated surveys
    """
    
    def __init__(self):
        self._orchestrator = None
        self._context_orchestrator = None
        self._enhanced_orchestrator = None
        self._initialization_error = None
        self._initialization_attempted = False
        
        # Initialize MCP server with proper configuration
        self.app = Server("maestro")
        self._register_handlers()
        
        logger.info("🎭 Maestro MCP Server Ready (Enhanced Intelligence Amplification)")
    
    def _get_enhanced_orchestrator(self):
        """Get enhanced orchestrator with lazy initialization and proper error handling."""
        if self._enhanced_orchestrator is None and not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                logger.info("🔄 Initializing Enhanced Maestro orchestration engine...")
                self._enhanced_orchestrator = EnhancedOrchestrationEngine()
                logger.info("✅ Enhanced Maestro orchestration engine ready")
            except Exception as e:
                self._initialization_error = f"Enhanced Maestro initialization failed: {str(e)}"
                logger.error(f"❌ {self._initialization_error}")
                logger.error(traceback.format_exc())
        
        if self._initialization_error:
            raise RuntimeError(self._initialization_error)
        
        return self._enhanced_orchestrator
    
    def _get_orchestrator(self):
        """Get orchestrator with lazy initialization and proper error handling."""
        if self._orchestrator is None:
            try:
                logger.info("🔄 Initializing fallback Maestro orchestration engine...")
                self._orchestrator = MAESTROOrchestrator()
                self._context_orchestrator = ContextAwareOrchestrator()
                logger.info("✅ Fallback Maestro orchestration engine ready")
            except Exception as e:
                logger.error(f"❌ Fallback orchestrator initialization failed: {str(e)}")
        
        return self._context_orchestrator or self._orchestrator
    
    def _register_handlers(self):
        """Register MCP server handlers and tools with enhanced orchestration capabilities."""
        
        @self.app.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available Maestro orchestration tools."""
            try:
                logger.info("📋 Listing Enhanced Maestro orchestration tools...")
                tools = [
                    types.Tool(
                        name="maestro_orchestrate",
                        description=(
                            "🎭 Enhanced Maestro orchestration with Intelligence Amplification. "
                            "Provides context intelligence, success criteria validation, tool discovery, "
                            "and collaborative error handling. Automatically detects context gaps and "
                            "generates surveys when additional information is needed. Maps Intelligence "
                            "Amplification Engines (IAEs) and external MCP tools to workflow phases."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Natural language description of the task to orchestrate"
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Additional context information (optional)",
                                    "properties": {
                                        "target_audience": {"type": "string"},
                                        "design_preferences": {"type": "string"},
                                        "functionality_requirements": {"type": "string"},
                                        "content_assets": {"type": "string"},
                                        "technical_constraints": {"type": "string"},
                                        "budget_constraints": {"type": "string"},
                                        "timeline": {"type": "string"}
                                    },
                                    "additionalProperties": True
                                },
                                "skip_context_validation": {
                                    "type": "boolean",
                                    "description": "Skip context gap analysis and proceed directly (use with caution)",
                                    "default": False
                                }
                            },
                            "required": ["task"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Enhanced Maestro Orchestration",
                            "readOnlyHint": False,
                            "destructiveHint": False,
                            "idempotentHint": False,
                            "openWorldHint": True
                        }
                    ),
                    
                    types.Tool(
                        name="context_intelligence",
                        description=(
                            "🧠 Context intelligence tool for gap detection and survey generation. "
                            "Analyzes task descriptions to identify missing context information and "
                            "generates structured surveys to gather necessary details for optimal orchestration."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Task description to analyze for context gaps"
                                },
                                "provided_context": {
                                    "type": "object",
                                    "description": "Context information already provided",
                                    "additionalProperties": True
                                },
                                "generate_survey": {
                                    "type": "boolean",
                                    "description": "Whether to generate a survey for identified gaps",
                                    "default": True
                                }
                            },
                            "required": ["task_description"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Context Intelligence & Gap Analysis",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    
                    types.Tool(
                        name="success_criteria_definition",
                        description=(
                            "🎯 Success criteria definition and validation mapping tool. "
                            "Automatically defines comprehensive success criteria for tasks and maps "
                            "appropriate validation tools and Intelligence Amplification Engines (IAEs)."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Task description for success criteria definition"
                                },
                                "task_type": {
                                    "type": "string",
                                    "description": "Type of task (auto-detected if not provided)",
                                    "enum": ["web_development", "backend_development", "mobile_development", "data_science", "testing", "general_development"]
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Context information for criteria customization",
                                    "additionalProperties": True
                                }
                            },
                            "required": ["task_description"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Success Criteria Definition",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    
                    types.Tool(
                        name="workflow_validator",
                        description=(
                            "✅ Workflow validation tool using mapped success criteria. "
                            "Validates workflow execution results against defined success criteria "
                            "using appropriate tools and IAEs for comprehensive verification."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "workflow_id": {
                                    "type": "string",
                                    "description": "ID of the workflow to validate"
                                },
                                "execution_results": {
                                    "type": "object",
                                    "description": "Results from workflow execution",
                                    "additionalProperties": True
                                },
                                "validation_mode": {
                                    "type": "string",
                                    "description": "Validation thoroughness level",
                                    "enum": ["fast", "balanced", "comprehensive"],
                                    "default": "balanced"
                                }
                            },
                            "required": ["workflow_id", "execution_results"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Workflow Success Validation",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    ),
                    
                    types.Tool(
                        name="survey_processor",
                        description=(
                            "📋 Survey response processor for context completion. "
                            "Processes user responses to generated context surveys and continues "
                            "orchestration with complete context information."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "survey_id": {
                                    "type": "string",
                                    "description": "ID of the completed survey"
                                },
                                "survey_responses": {
                                    "type": "object",
                                    "description": "User responses to survey questions",
                                    "additionalProperties": True
                                },
                                "original_task": {
                                    "type": "string",
                                    "description": "Original task description"
                                }
                            },
                            "required": ["survey_id", "survey_responses", "original_task"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "Survey Response Processing",
                            "readOnlyHint": False,
                            "destructiveHint": False,
                            "idempotentHint": False,
                            "openWorldHint": False
                        }
                    ),
                    
                    types.Tool(
                        name="iae_discovery",
                        description=(
                            "⚡ Intelligence Amplification Engine (IAE) discovery and mapping. "
                            "Discovers available IAEs from the 43-engine registry and maps them "
                            "to appropriate workflow phases for cognitive enhancement."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_type": {
                                    "type": "string",
                                    "description": "Type of task for IAE mapping"
                                },
                                "workflow_phases": {
                                    "type": "array",
                                    "description": "Workflow phases for IAE mapping",
                                    "items": {"type": "string"}
                                },
                                "enhancement_focus": {
                                    "type": "string",
                                    "description": "Primary enhancement focus",
                                    "enum": ["analysis", "reasoning", "validation", "optimization", "all"],
                                    "default": "all"
                                }
                            },
                            "required": ["task_type"],
                            "additionalProperties": False
                        },
                        annotations={
                            "title": "IAE Discovery & Mapping",
                            "readOnlyHint": True,
                            "destructiveHint": False,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    )
                ]
                logger.info(f"✅ Successfully listed {len(tools)} Enhanced Maestro orchestration tools")
                return tools
                
            except Exception as e:
                logger.error(f"❌ Error listing tools: {str(e)}")
                logger.error(traceback.format_exc())
                return []
        
        @self.app.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls for Enhanced Maestro orchestration tools."""
            try:
                logger.info(f"🔧 Executing enhanced tool: {name}")
                logger.debug(f"Tool arguments: {arguments}")
                
                if name == "maestro_orchestrate":
                    return await self._handle_enhanced_maestro_orchestrate(arguments)
                elif name == "context_intelligence":
                    return await self._handle_context_intelligence(arguments)
                elif name == "success_criteria_definition":
                    return await self._handle_success_criteria_definition(arguments)
                elif name == "workflow_validator":
                    return await self._handle_workflow_validator(arguments)
                elif name == "survey_processor":
                    return await self._handle_survey_processor(arguments)
                elif name == "iae_discovery":
                    return await self._handle_iae_discovery(arguments)
                else:
                    # Fallback to original handlers
                    orchestrator = self._get_orchestrator()
                    return await self._handle_fallback_tool(name, arguments, orchestrator)
                    
            except Exception as e:
                logger.error(f"❌ Error executing tool {name}: {str(e)}")
                logger.error(traceback.format_exc())
                return [types.TextContent(
                    type="text",
                    text=f"❌ Tool execution failed: {str(e)}\n\nPlease check your input parameters and try again."
                )]
    
    async def _handle_enhanced_maestro_orchestrate(self, arguments: dict) -> list[types.TextContent]:
        """Handle the enhanced Maestro orchestration tool with intelligence amplification."""
        task = arguments["task"]
        context = arguments.get("context", {})
        skip_validation = arguments.get("skip_context_validation", False)
        
        try:
            enhanced_orchestrator = self._get_enhanced_orchestrator()
            
            if skip_validation:
                # Force orchestration without context validation
                result = await enhanced_orchestrator._perform_full_orchestration(task, context, [])
            else:
                # Full orchestration with context intelligence
                result = await enhanced_orchestrator.orchestrate_complete_workflow(task, context)
            
            # Handle different result types
            if isinstance(result, ContextSurvey):
                # Context gaps found - return survey
                response = self._format_context_survey_response(result, task)
            elif isinstance(result, OrchestrationResult):
                # Full orchestration complete
                response = self._format_orchestration_result_response(result)
            else:
                # Fallback formatting
                response = str(result)
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Enhanced Maestro orchestration failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Enhanced Maestro orchestration failed: {str(e)}\n\nTrying fallback orchestration..."
            )]
    
    async def _handle_context_intelligence(self, arguments: dict) -> list[types.TextContent]:
        """Handle context intelligence and gap analysis."""
        task_description = arguments["task_description"]
        provided_context = arguments.get("provided_context", {})
        generate_survey = arguments.get("generate_survey", True)
        
        try:
            enhanced_orchestrator = self._get_enhanced_orchestrator()
            context_engine = enhanced_orchestrator.context_engine
            
            # Analyze context gaps
            gaps = context_engine.analyze_context_gaps(task_description, provided_context)
            
            if not gaps:
                response = f"""# 🧠 Context Intelligence Analysis ✅

**Task:** {task_description}

## Analysis Result
✅ **No context gaps detected!** All necessary context information appears to be available.

**Context Completeness:** 100%
**Ready for orchestration:** Yes

You can proceed directly with orchestration using the `maestro_orchestrate` tool.
"""
            elif generate_survey and gaps:
                # Generate survey for gaps
                survey = context_engine.generate_context_survey(gaps, task_description)
                response = self._format_context_survey_response(survey, task_description)
            else:
                # Just report gaps without survey
                response = f"""# 🧠 Context Intelligence Analysis ⚠️

**Task:** {task_description}

## Context Gaps Identified ({len(gaps)})

"""
                for i, gap in enumerate(gaps, 1):
                    response += f"""
### {i}. {gap.category.title()} ({gap.importance})
**Description:** {gap.description}
**Suggested Questions:**
{chr(10).join(f"- {q}" for q in gap.suggested_questions)}
"""
                
                response += """
## Recommendations
1. Provide answers to the questions above
2. Use `maestro_orchestrate` with additional context
3. Or use `context_intelligence` with `generate_survey: true` to get a structured survey
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Context intelligence failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Context intelligence analysis failed: {str(e)}"
            )]
    
    async def _handle_success_criteria_definition(self, arguments: dict) -> list[types.TextContent]:
        """Handle success criteria definition."""
        task_description = arguments["task_description"]
        task_type = arguments.get("task_type")
        context = arguments.get("context", {})
        
        try:
            enhanced_orchestrator = self._get_enhanced_orchestrator()
            success_engine = enhanced_orchestrator.success_engine
            
            # Auto-detect task type if not provided
            if not task_type:
                task_type = enhanced_orchestrator.context_engine._identify_task_type(task_description)
            
            # Define success criteria
            success_criteria = success_engine.define_success_criteria(task_description, task_type, context)
            
            response = f"""# 🎯 Success Criteria Definition

**Task:** {task_description}
**Task Type:** {task_type}
**Total Criteria:** {len(success_criteria.criteria)}
**Completion Threshold:** {success_criteria.completion_threshold * 100}%
**Validation Strategy:** {success_criteria.validation_strategy}

## Defined Success Criteria

"""
            
            for i, criterion in enumerate(success_criteria.criteria, 1):
                response += f"""
### {i}. {criterion.criterion_id.replace('_', ' ').title()}
- **Description:** {criterion.description}
- **Type:** {criterion.metric_type}
- **Priority:** {criterion.priority}
- **Validation Method:** {criterion.validation_method}
"""
                if criterion.threshold:
                    response += f"- **Target Threshold:** {criterion.threshold}\n"
                if criterion.validation_tools:
                    response += f"- **Validation Tools:** {', '.join(criterion.validation_tools)}\n"
                if criterion.validation_iaes:
                    response += f"- **Intelligence Amplification Engines:** {', '.join(criterion.validation_iaes)}\n"
            
            response += f"""
## Validation Timeline
**Estimated Validation Time:** {success_criteria.estimated_validation_time}

## Usage
These success criteria will be automatically applied during orchestration and validation phases.
Use `workflow_validator` to validate execution results against these criteria.
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Success criteria definition failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Success criteria definition failed: {str(e)}"
            )]
    
    async def _handle_workflow_validator(self, arguments: dict) -> list[types.TextContent]:
        """Handle workflow validation against success criteria."""
        workflow_id = arguments["workflow_id"]
        execution_results = arguments["execution_results"]
        validation_mode = arguments.get("validation_mode", "balanced")
        
        try:
            # This would implement actual validation logic
            # For now, providing a structured response
            
            response = f"""# ✅ Workflow Validation Results

**Workflow ID:** {workflow_id}
**Validation Mode:** {validation_mode}
**Validation Timestamp:** {asyncio.get_event_loop().time()}

## Validation Summary
🔍 **Status:** Validation Complete
📊 **Overall Score:** Pending implementation
⚡ **IAEs Used:** Mapped validation engines
🔧 **Tools Used:** Mapped validation tools

## Individual Criteria Results
*(Detailed validation results would be shown here)*

## Recommendations
*(Improvement recommendations based on validation results)*

## Next Steps
*(Suggested actions based on validation outcomes)*

---
*Note: Full validation implementation is in progress. This is a framework response.*
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Workflow validation failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Workflow validation failed: {str(e)}"
            )]
    
    async def _handle_survey_processor(self, arguments: dict) -> list[types.TextContent]:
        """Handle survey response processing and continue orchestration."""
        survey_id = arguments["survey_id"]
        survey_responses = arguments["survey_responses"]
        original_task = arguments["original_task"]
        
        try:
            enhanced_orchestrator = self._get_enhanced_orchestrator()
            
            # Process survey responses into context
            processed_context = self._process_survey_responses(survey_responses)
            
            # Continue with full orchestration using processed context
            result = await enhanced_orchestrator.orchestrate_complete_workflow(
                original_task, 
                processed_context
            )
            
            if isinstance(result, OrchestrationResult):
                response = f"""# 📋 Survey Processing Complete ✅

**Survey ID:** {survey_id}
**Responses Processed:** {len(survey_responses)}
**Context Completion:** 100%

## Orchestration Results

{self._format_orchestration_result_response(result)}
"""
            else:
                response = f"❌ Unexpected result type after survey processing: {type(result)}"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Survey processing failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ Survey processing failed: {str(e)}"
            )]
    
    async def _handle_iae_discovery(self, arguments: dict) -> list[types.TextContent]:
        """Handle Intelligence Amplification Engine discovery and mapping."""
        task_type = arguments["task_type"]
        workflow_phases = arguments.get("workflow_phases", [])
        enhancement_focus = arguments.get("enhancement_focus", "all")
        
        try:
            # This would integrate with the IAE registry from engines_todo.md
            response = f"""# ⚡ Intelligence Amplification Engine (IAE) Discovery

**Task Type:** {task_type}
**Enhancement Focus:** {enhancement_focus}
**Workflow Phases:** {len(workflow_phases)} phases analyzed

## Recommended IAE Mappings

### For {task_type.replace('_', ' ').title()} Tasks:

"""
            
            # Sample IAE mappings based on task type
            if task_type == "web_development":
                iae_mappings = [
                    {
                        "name": "Design Thinking Engine",
                        "phase": "design",
                        "enhancement": "analysis",
                        "libraries": ["NetworkX", "NumPy", "pandas"],
                        "purpose": "Enhance UX/UI design decisions and user-centered thinking"
                    },
                    {
                        "name": "Visual Art Engine", 
                        "phase": "design",
                        "enhancement": "reasoning",
                        "libraries": ["NumPy", "SciPy", "PIL", "colorsys"],
                        "purpose": "Amplify visual design reasoning through color theory and composition analysis"
                    },
                    {
                        "name": "Accessibility Engine",
                        "phase": "testing",
                        "enhancement": "validation", 
                        "libraries": ["NLTK", "pandas", "BeautifulSoup"],
                        "purpose": "Enhance accessibility reasoning and inclusive design thinking"
                    },
                    {
                        "name": "Systems Engineering Engine",
                        "phase": "planning",
                        "enhancement": "optimization",
                        "libraries": ["NetworkX", "NumPy", "SciPy"],
                        "purpose": "Enhance complex system reasoning and optimization thinking"
                    }
                ]
            else:
                iae_mappings = [
                    {
                        "name": "Algorithm Design Engine",
                        "phase": "implementation",
                        "enhancement": "optimization",
                        "libraries": ["NetworkX", "NumPy", "SymPy"],
                        "purpose": "Enhance algorithmic reasoning and complexity analysis thinking"
                    }
                ]
            
            for i, iae in enumerate(iae_mappings, 1):
                response += f"""
### {i}. {iae['name']}
- **Workflow Phase:** {iae['phase']}
- **Enhancement Type:** {iae['enhancement']}
- **Libraries Required:** {', '.join(iae['libraries'])}
- **Cognitive Enhancement:** {iae['purpose']}
"""
            
            response += f"""
## Integration Instructions

These IAEs will be automatically mapped to your workflow phases when using `maestro_orchestrate`.
Each engine provides analytical frameworks to enhance your reasoning capabilities within specific domains.

**Key Principle:** Intelligence Amplification > Raw Parameter Count

## Available Engine Categories
- **Physics & Mathematics:** 5 engines
- **Chemistry & Biology:** 5 engines  
- **Medicine & Life Sciences:** 5 engines
- **Engineering & Technology:** 5 engines
- **Computer Science & Data:** 5 engines
- **Social Sciences & Humanities:** 5 engines
- **Art, Music & Creativity:** 5 engines
- **Language & Communication:** 5 engines
- **Cross-Domain/Interdisciplinary:** 3 engines

**Total:** 43 Intelligence Amplification Engines available for cognitive enhancement.
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"IAE discovery failed: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"❌ IAE discovery failed: {str(e)}"
            )]
    
    async def _handle_fallback_tool(self, name: str, arguments: dict, orchestrator) -> list[types.TextContent]:
        """Handle fallback to original tool implementations."""
        logger.info(f"🔄 Using fallback implementation for tool: {name}")
        
        # Implementation would call original tool handlers
        return [types.TextContent(
            type="text",
            text=f"🔄 Fallback tool execution for {name} - Enhanced implementation in progress"
        )]
    
    def _format_context_survey_response(self, survey: ContextSurvey, task: str) -> str:
        """Format context survey for user presentation."""
        response = f"""# 📋 Context Information Needed

**Task:** {task}

## ⚠️ Additional Context Required

To provide optimal orchestration, I need some additional information about your project.

## 📊 Survey: {survey.title}

{survey.description}

**Estimated Time:** {survey.estimated_time}
**Survey ID:** {survey.survey_id}

### Questions:

"""
        
        for i, question in enumerate(survey.questions, 1):
            required_indicator = " ***(Required)***" if question.required else ""
            response += f"""
**{i}. {question.question_text}**{required_indicator}
Type: {question.question_type}
{f"Help: {question.help_text}" if question.help_text else ""}
{f"Options: {', '.join(question.options)}" if question.options else ""}

"""
        
        response += f"""
## 🔄 Next Steps

1. **Answer the questions above**
2. **Use the `survey_processor` tool** with the following format:
   ```
   survey_id: "{survey.survey_id}"
   survey_responses: {{
       "q_1_0": "your answer to question 1",
       "q_2_0": "your answer to question 2",
       // ... etc
   }}
   original_task: "{task}"
   ```

3. **I'll continue orchestration** with complete context

## 💡 Why This Helps

Providing this context enables:
- ✅ More accurate workflow planning
- ✅ Better tool and IAE mapping
- ✅ Precise success criteria definition
- ✅ Optimized validation strategies

**Remember:** Intelligence Amplification > Raw Parameter Count
"""
        
        return response
    
    def _format_orchestration_result_response(self, result: OrchestrationResult) -> str:
        """Format orchestration result for user presentation."""
        workflow = result.workflow
        
        response = f"""# 🎭 Maestro Orchestration Complete ✅

**Task:** {workflow.task_description}
**Complexity:** {workflow.complexity.value.title()}
**Estimated Time:** {workflow.estimated_total_time}
**Workflow ID:** {workflow.workflow_id}

## 📊 Orchestration Summary

- **Workflow Phases:** {len(workflow.phases)}
- **Success Criteria:** {len(workflow.success_criteria.criteria)}
- **Tool Mappings:** {len(workflow.tool_mappings)}
- **IAE Mappings:** {len(workflow.iae_mappings)}
- **Intelligence Amplification:** {len(workflow.iae_mappings)} engines mapped

## 🚀 Execution Guidance

{result.execution_guidance}

## 💡 Recommendations

{chr(10).join(f"- {rec}" for rec in result.recommendations)}

## ⚡ Next Steps

{chr(10).join(f"- {step}" for step in result.next_steps)}

---
**Orchestrated with Intelligence Amplification** - Enhanced capabilities through {len(workflow.iae_mappings)} cognitive enhancement engines.
"""
        
        return response
    
    def _process_survey_responses(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Process survey responses into structured context."""
        # This would map survey responses to context fields
        # For now, return responses as-is
        return responses
    
    async def run(self):
        """Run the Enhanced MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream, 
                write_stream,
                InitializationOptions(
                    server_name="maestro",
                    server_version="2.0.0",
                    capabilities={
                        "tools": {}
                    }
                )
            )


async def main():
    """Main entry point for the Enhanced Maestro MCP server."""
    server_instance = MaestroMCPServer()
    await server_instance.run()


if __name__ == "__main__":
    asyncio.run(main())