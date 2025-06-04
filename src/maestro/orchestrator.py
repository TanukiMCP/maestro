# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
MAESTRO Protocol Unified Orchestrator

Provides comprehensive task orchestration using the enhanced MIA protocol framework.
Integrates context intelligence, success criteria validation, and operator profiles.

Core Principle: Intelligence Amplification > Raw Parameter Count
"""

import logging
from typing import Dict, List, Any, Optional
import re

from .data_models import (
    MAESTROResult, TaskAnalysis, Workflow, 
    TaskType, ComplexityLevel, VerificationMethod
)
from .quality_controller import QualityController
from .orchestration_framework import (
    EnhancedOrchestrationEngine,
    ContextIntelligenceEngine, 
    SuccessCriteriaEngine,
    TaskComplexity
)

try:
    from ..engines import IntelligenceAmplifier
except ImportError:
    IntelligenceAmplifier = None

try:
    from ..profiles import OperatorProfileManager
except ImportError:
    OperatorProfileManager = None

logger = logging.getLogger(__name__)


class MAESTROOrchestrator:
    """
    Unified MAESTRO Protocol Orchestrator with MIA compliance.
    
    Provides comprehensive task orchestration through:
    - Context intelligence and gap detection
    - Success criteria definition and validation
    - Operator profile assignment
    - Tool/IAE discovery and mapping
    """
    
    def __init__(self):
        self.quality_controller = QualityController()
        self.context_engine = ContextIntelligenceEngine()
        self.success_engine = SuccessCriteriaEngine()
        self.enhanced_orchestrator = EnhancedOrchestrationEngine()
        
        # Add enhanced error handling and temporal context
        try:
            from .adaptive_error_handler import AdaptiveErrorHandler, TemporalContext
            from .enhanced_tools import EnhancedToolHandlers
            self.error_handler = AdaptiveErrorHandler()
            self.enhanced_tools = EnhancedToolHandlers()
            self.has_enhanced_capabilities = True
        except ImportError:
            logger.warning("Enhanced error handling and tools not available")
            self.error_handler = None
            self.enhanced_tools = None
            self.has_enhanced_capabilities = False
        
        if OperatorProfileManager:
            self.profile_manager = OperatorProfileManager()
        else:
            self.profile_manager = None
            logger.warning("Operator Profile Manager not available")
            
        if IntelligenceAmplifier:
            self.intelligence_amplifier = IntelligenceAmplifier()
        else:
            self.intelligence_amplifier = None
            
        self.task_patterns = self._initialize_task_patterns()
        
        # Add built-in tool registry with fallback capabilities
        self.built_in_tools = {
            "maestro_search": {
                "description": "LLM-driven web search with fallback capabilities",
                "fallback_for": ["web_search", "search", "browser_search"]
            },
            "maestro_scrape": {
                "description": "LLM-driven web scraping with content extraction",
                "fallback_for": ["web_scrape", "scrape", "browser_scrape"]
            },
            "maestro_execute": {
                "description": "LLM-driven code execution for validation",
                "fallback_for": ["execute", "run_code", "validate_code"]
            },
            "maestro_error_handler": {
                "description": "Adaptive error handling with approach reconsideration",
                "fallback_for": ["error_handling", "exception_handling"]
            },
            "maestro_temporal_context": {
                "description": "Temporal context awareness for information currency",
                "fallback_for": ["time_context", "date_context", "freshness_check"]
            }
        }
        
        logger.info(f"🎭 MAESTRO Unified Orchestrator initialized with {'Enhanced' if self.has_enhanced_capabilities else 'Basic'} capabilities")
    
    def _initialize_task_patterns(self) -> Dict[TaskType, List[str]]:
        """Initialize patterns for task type classification."""
        return {
            TaskType.MATHEMATICS: [
                r"calcul(ate|ation)", r"solve.*equation", r"integral", 
                r"derivative", r"statistics", r"math", r"formula"
            ],
            TaskType.WEB_DEVELOPMENT: [
                r"website", r"web.*app", r"html", r"css", r"react", 
                r"frontend", r"backend", r"api"
            ],
            TaskType.CODE_DEVELOPMENT: [
                r"function", r"class", r"algorithm", r"code", 
                r"python", r"script", r"refactor"
            ],
            TaskType.DATA_ANALYSIS: [
                r"data.*analy", r"dataset", r"statistical", r"visualization",
                r"correlation", r"regression", r"machine.*learning"
            ],
            TaskType.RESEARCH: [
                r"research", r"study", r"investigate", r"analyze.*literature",
                r"academic", r"scientific"
            ],
            TaskType.CREATIVE: [
                r"creat(e|ive)", r"design", r"artistic", r"visual",
                r"story", r"narrative", r"music"
            ]
        }
    
    async def orchestrate_task(
        self, 
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        complexity_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        PRIMARY MCP TOOL: Complete task orchestration with MIA protocol and enhanced error handling.
        
        Args:
            task_description: Natural language task description
            context: Additional context information
            complexity_override: Override automatic complexity detection
            
        Returns:
            Complete orchestration result with workflow, validation, and guidance
        """
        logger.info(f"🎭 Orchestrating task: {task_description[:100]}...")
        
        if context is None:
            context = {}
        
        # Add temporal context awareness
        if self.has_enhanced_capabilities:
            from datetime import datetime, timezone
            temporal_context = {
                "current_timestamp": datetime.now(timezone.utc).isoformat(),
                "context_freshness_required": context.get("freshness_required", False),
                "temporal_relevance_window": context.get("relevance_window", "24h")
            }
            context["temporal_context"] = temporal_context
        
        try:
            # Step 1: Context Analysis and Gap Detection
            context_gaps = self.context_engine.analyze_context_gaps(
                task_description, context
            )
            
            if context_gaps:
                # Generate survey for missing context
                survey = self.context_engine.generate_context_survey(
                    task_description, context_gaps
                )
                return {
                    "status": "context_survey_required",
                    "survey": survey,
                    "message": "Additional context needed for optimal orchestration"
                }
            
            # Step 2: Tool Discovery and Availability Check
            available_tools = await self._discover_available_tools(context)
            
            # Step 3: Task Analysis with Operator Profile Assignment
            task_analysis = await self._analyze_task_with_profile(
                task_description, complexity_override
            )
            
            # Step 4: Enhanced Orchestration with Error Recovery
            orchestration_result = await self._orchestrate_with_error_recovery(
                task_description, context, task_analysis, available_tools
            )
            
            # Step 5: Success Criteria Definition with Tool Mapping
            success_criteria = self.success_engine.define_success_criteria(
                task_description, orchestration_result.workflow
            )
            
            # Map validation tools to available tools
            mapped_success_criteria = await self._map_validation_tools(
                success_criteria, available_tools
            )
            
            return {
                "status": "orchestration_complete",
                "task_analysis": {
                    "task_type": task_analysis.task_type.value,
                    "complexity": task_analysis.complexity.value,
                    "assigned_operator": task_analysis.assigned_operator,
                    "capabilities": task_analysis.capabilities,
                    "estimated_duration": task_analysis.estimated_duration
                },
                "workflow": {
                    "workflow_id": orchestration_result.workflow.workflow_id,
                    "phases": [
                        {
                            "phase_id": phase.phase_id,
                            "phase_name": phase.phase_name,
                            "description": phase.description,
                            "tools": [tm.tool_name for tm in phase.tool_mappings],
                            "iaes": [im.iae_name for im in phase.iae_mappings],
                            "dependencies": phase.dependencies
                        }
                        for phase in orchestration_result.workflow.phases
                    ],
                    "estimated_total_time": orchestration_result.workflow.estimated_total_time
                },
                "success_criteria": {
                    "validation_strategy": mapped_success_criteria.validation_strategy,
                    "completion_threshold": mapped_success_criteria.completion_threshold,
                    "criteria": [
                        {
                            "description": criterion.description,
                            "metric_type": criterion.metric_type,
                            "validation_method": criterion.validation_method,
                            "priority": criterion.priority,
                            "validation_tools": getattr(criterion, 'validation_tools', []),
                            "fallback_available": getattr(criterion, 'fallback_available', False)
                        }
                        for criterion in mapped_success_criteria.criteria
                    ]
                },
                "tool_discovery": {
                    "available_tools": available_tools,
                    "built_in_fallbacks": list(self.built_in_tools.keys()),
                    "tool_mapping_performed": True
                },
                "execution_guidance": orchestration_result.execution_guidance,
                "enhanced_capabilities": {
                    "adaptive_error_handling": self.has_enhanced_capabilities,
                    "temporal_context_awareness": self.has_enhanced_capabilities,
                    "fallback_tools_available": bool(self.built_in_tools),
                    "approach_reconsideration": self.has_enhanced_capabilities
                },
                "mia_protocol": {
                    "intelligence_amplification_engines": [
                        f"{mapping.iae_name}: {mapping.cognitive_enhancement}"
                        for mapping in orchestration_result.workflow.iae_mappings.values()
                    ],
                    "cognitive_enhancement_focus": self._get_cognitive_enhancement_focus(
                        task_analysis.task_type
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {str(e)}")
            
            # Enhanced error handling
            if self.has_enhanced_capabilities:
                return await self._handle_orchestration_error(e, task_description, context)
            else:
                return {
                    "status": "orchestration_failed",
                    "error": str(e),
                    "fallback_guidance": self._get_fallback_guidance(task_description)
                }
    
    async def _analyze_task_with_profile(
        self, 
        task_description: str,
        complexity_override: Optional[str] = None
    ) -> TaskAnalysis:
        """Analyze task and assign appropriate operator profile."""
        task_type = self._classify_task_type(task_description.lower())
        
        if complexity_override:
            complexity = ComplexityLevel(complexity_override.lower())
        else:
            complexity = self._assess_complexity(task_description.lower())
        
        # Assign operator profile
        assigned_operator = None
        if self.profile_manager:
            assigned_operator = self.profile_manager.select_optimal_profile(
                task_type.value, complexity.value, task_description
            )
        
        capabilities = self._determine_capabilities(task_type, task_description)
        estimated_duration = self._estimate_duration(complexity, task_type)
        success_criteria = self._define_success_criteria(task_type, complexity)
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            capabilities=capabilities,
            estimated_duration=estimated_duration,
            required_tools=self._get_required_tools(task_type, capabilities),
            success_criteria=success_criteria,
            quality_requirements=self._get_quality_requirements(complexity),
            assigned_operator=assigned_operator
        )
    
    def _classify_task_type(self, task_description: str) -> TaskType:
        """Classify task type using pattern matching."""
        scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, task_description))
                score += matches
            scores[task_type] = score
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return TaskType.GENERAL
    
    def _assess_complexity(self, task_description: str) -> ComplexityLevel:
        """Assess task complexity based on description analysis."""
        complexity_indicators = {
            ComplexityLevel.BASIC: [
                r"simple", r"basic", r"quick", r"easy", r"straightforward"
            ],
            ComplexityLevel.INTERMEDIATE: [
                r"moderate", r"standard", r"typical", r"regular"
            ],
            ComplexityLevel.ADVANCED: [
                r"complex", r"advanced", r"sophisticated", r"detailed",
                r"comprehensive", r"multi.*step", r"integration"
            ],
            ComplexityLevel.EXPERT: [
                r"expert", r"research.*grade", r"cutting.*edge", r"innovative",
                r"novel", r"experimental", r"state.*of.*art"
            ]
        }
        
        scores = {}
        for level, indicators in complexity_indicators.items():
            score = sum(len(re.findall(indicator, task_description)) for indicator in indicators)
            scores[level] = score
        
        # Default logic based on task length and keywords
        word_count = len(task_description.split())
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        elif word_count > 50:
            return ComplexityLevel.ADVANCED
        elif word_count > 20:
            return ComplexityLevel.INTERMEDIATE
        else:
            return ComplexityLevel.BASIC
    
    def _determine_capabilities(self, task_type: TaskType, description: str) -> List[str]:
        """Determine required capabilities based on task type."""
        capability_mapping = {
            TaskType.MATHEMATICS: ["mathematical_reasoning", "symbolic_computation", "numerical_analysis"],
            TaskType.WEB_DEVELOPMENT: ["frontend_development", "backend_development", "ui_design"],
            TaskType.CODE_DEVELOPMENT: ["programming", "algorithm_design", "code_optimization"],
            TaskType.DATA_ANALYSIS: ["statistical_analysis", "data_visualization", "pattern_recognition"],
            TaskType.RESEARCH: ["information_synthesis", "critical_analysis", "source_verification"],
            TaskType.CREATIVE: ["creative_thinking", "design_principles", "artistic_composition"]
        }
        return capability_mapping.get(task_type, ["general_problem_solving"])
    
    def _estimate_duration(self, complexity: ComplexityLevel, task_type: TaskType) -> str:
        """Estimate task duration based on complexity and type."""
        base_times = {
            ComplexityLevel.BASIC: "15-30 minutes",
            ComplexityLevel.INTERMEDIATE: "30-60 minutes", 
            ComplexityLevel.ADVANCED: "1-3 hours",
            ComplexityLevel.EXPERT: "3+ hours"
        }
        return base_times.get(complexity, "30-60 minutes")
    
    def _define_success_criteria(self, task_type: TaskType, complexity: ComplexityLevel) -> List[str]:
        """Define success criteria based on task type and complexity."""
        base_criteria = [
            "Task requirements fully understood and addressed",
            "Solution is functional and meets specifications",
            "Quality standards appropriate for complexity level met"
        ]
        
        type_specific = {
            TaskType.MATHEMATICS: ["Mathematical accuracy verified", "Proper notation and methodology used"],
            TaskType.WEB_DEVELOPMENT: ["Responsive design implemented", "Cross-browser compatibility ensured"],
            TaskType.CODE_DEVELOPMENT: ["Code follows best practices", "Proper error handling implemented"]
        }
        
        return base_criteria + type_specific.get(task_type, [])
    
    def _get_required_tools(self, task_type: TaskType, capabilities: List[str]) -> List[str]:
        """Get required tools based on task type and capabilities."""
        tool_mapping = {
            TaskType.MATHEMATICS: ["mathematical_computation", "symbolic_solver"],
            TaskType.WEB_DEVELOPMENT: ["code_editor", "browser_tools", "development_server"],
            TaskType.CODE_DEVELOPMENT: ["code_editor", "debugger", "version_control"],
            TaskType.DATA_ANALYSIS: ["data_processing", "visualization_tools", "statistical_analysis"]
        }
        return tool_mapping.get(task_type, ["general_tools"])
    
    def _get_quality_requirements(self, complexity: ComplexityLevel) -> Dict[str, float]:
        """Get quality requirements based on complexity."""
        requirements = {
            ComplexityLevel.BASIC: {"accuracy": 0.85, "completeness": 0.80},
            ComplexityLevel.INTERMEDIATE: {"accuracy": 0.90, "completeness": 0.85},
            ComplexityLevel.ADVANCED: {"accuracy": 0.95, "completeness": 0.90},
            ComplexityLevel.EXPERT: {"accuracy": 0.98, "completeness": 0.95}
        }
        return requirements.get(complexity, {"accuracy": 0.85, "completeness": 0.80})
    
    def _get_cognitive_enhancement_focus(self, task_type: TaskType) -> str:
        """Get cognitive enhancement focus for MIA protocol."""
        focus_mapping = {
            TaskType.MATHEMATICS: "Symbolic reasoning and computational precision",
            TaskType.WEB_DEVELOPMENT: "System design and user experience optimization",
            TaskType.CODE_DEVELOPMENT: "Algorithm optimization and code quality enhancement",
            TaskType.DATA_ANALYSIS: "Pattern recognition and statistical insight generation",
            TaskType.RESEARCH: "Information synthesis and critical evaluation",
            TaskType.CREATIVE: "Innovative thinking and aesthetic reasoning"
        }
        return focus_mapping.get(task_type, "General problem-solving enhancement")
    
    def _get_fallback_guidance(self, task_description: str) -> str:
        """Provide fallback guidance when orchestration fails."""
        return f"""
        Fallback Guidance for: {task_description[:100]}...
        
        1. Break down the task into smaller, manageable components
        2. Identify the core requirements and constraints
        3. Use appropriate tools and methodologies for the domain
        4. Validate results against defined success criteria
        5. Iterate and refine based on feedback
        
        Consider using basic task analysis and step-by-step execution.
        """

    async def _discover_available_tools(self, context: Dict[str, Any]) -> List[str]:
        """Discover available tools from client and built-in fallbacks."""
        available_tools = []
        
        # Add built-in MAESTRO tools (always available)
        available_tools.extend(self.built_in_tools.keys())
        
        # In a real MCP environment, we would query the client for available tools
        # For now, we'll assume common tools might be available
        potential_external_tools = [
            "web_search", "browser_tools", "file_operations", 
            "code_execution", "text_analysis", "image_analysis"
        ]
        
        # Simulate tool discovery (in practice, this would be an MCP capability query)
        # For now, assume some tools might be available
        available_tools.extend(potential_external_tools)
        
        logger.info(f"🔍 Tool discovery complete: {len(available_tools)} tools available")
        return available_tools
    
    async def _orchestrate_with_error_recovery(
        self, 
        task_description: str, 
        context: Dict[str, Any], 
        task_analysis: TaskAnalysis,
        available_tools: List[str]
    ) -> Any:
        """Orchestrate workflow with error recovery capabilities."""
        try:
            # Try enhanced orchestration first
            return await self.enhanced_orchestrator.orchestrate_workflow(
                task_description, context, task_analysis.complexity
            )
        except Exception as e:
            logger.warning(f"⚠️ Enhanced orchestration failed, attempting error recovery: {str(e)}")
            
            if self.has_enhanced_capabilities:
                # Analyze error and attempt recovery
                error_details = {
                    "type": "orchestration_failure",
                    "message": str(e),
                    "component": "enhanced_orchestrator",
                    "impact": "workflow_blocking"
                }
                
                from .adaptive_error_handler import TemporalContext
                from datetime import datetime, timezone
                
                temporal_context = TemporalContext(
                    current_timestamp=datetime.now(timezone.utc),
                    information_cutoff=None,
                    task_deadline=None,
                    context_freshness_required=context.get("temporal_context", {}).get("context_freshness_required", False),
                    temporal_relevance_window=context.get("temporal_context", {}).get("temporal_relevance_window", "24h")
                )
                
                error_context = await self.error_handler.analyze_error_context(
                    error_details=error_details,
                    temporal_context=temporal_context,
                    available_tools=available_tools,
                    success_criteria=[]
                )
                
                reconsideration = await self.error_handler.should_reconsider_approach(error_context)
                
                if reconsideration.should_reconsider:
                    logger.info("🔄 Attempting simplified orchestration approach...")
                    return await self._simplified_orchestration(task_description, task_analysis, available_tools)
                else:
                    raise e
            else:
                # Fallback to basic orchestration
                return await self._simplified_orchestration(task_description, task_analysis, available_tools)
    
    async def _simplified_orchestration(self, task_description: str, task_analysis: TaskAnalysis, available_tools: List[str]) -> Any:
        """Simplified orchestration as fallback approach."""
        # Create a basic workflow structure
        from .orchestration_framework import OrchestrationWorkflow, WorkflowPhase
        
        # Create simplified workflow
        workflow = OrchestrationWorkflow(
            workflow_id=f"simplified_{hash(task_description) % 10000}",
            task_description=task_description,
            complexity=task_analysis.complexity,
            phases=[],
            success_criteria=None,
            iae_mappings={},
            tool_mappings={},
            estimated_total_time="30-60 minutes"
        )
        
        # Add basic execution phase
        basic_phase = WorkflowPhase(
            phase_id="execution",
            phase_name="Task Execution",
            description="Execute the task using available tools",
            inputs=["task_requirements"],
            outputs=["task_results"],
            tool_mappings=[],
            iae_mappings=[],
            estimated_duration="30-60 minutes",
            dependencies=[],
            success_criteria=[]
        )
        
        workflow.phases.append(basic_phase)
        
        # Create mock orchestration result
        class SimpleOrchestrationResult:
            def __init__(self, workflow):
                self.workflow = workflow
                self.execution_guidance = f"""
# Simplified Task Execution

**Task:** {task_description}

## Approach:
1. Analyze the task requirements
2. Use available tools: {', '.join(available_tools[:5])}...
3. Execute step by step
4. Validate results

## Available Tools:
{chr(10).join(f"- {tool}" for tool in available_tools[:10])}

This is a simplified execution plan due to orchestration complexity.
"""
        
        return SimpleOrchestrationResult(workflow)
    
    async def _map_validation_tools(self, success_criteria: Any, available_tools: List[str]) -> Any:
        """Map validation tools to available tools with fallback options."""
        if not hasattr(success_criteria, 'criteria'):
            return success_criteria
        
        # Enhance each criterion with tool mapping
        for criterion in success_criteria.criteria:
            # Check if validation tools are available
            validation_tools = getattr(criterion, 'validation_tools', [])
            available_validation_tools = [tool for tool in validation_tools if tool in available_tools]
            
            # Add fallback tools if primary tools not available
            if not available_validation_tools:
                # Map to built-in fallback tools
                if any(keyword in criterion.description.lower() for keyword in ['web', 'search', 'online']):
                    if 'maestro_search' in available_tools:
                        available_validation_tools.append('maestro_search')
                        criterion.fallback_available = True
                
                if any(keyword in criterion.description.lower() for keyword in ['execute', 'run', 'test']):
                    if 'maestro_execute' in available_tools:
                        available_validation_tools.append('maestro_execute')
                        criterion.fallback_available = True
                
                if any(keyword in criterion.description.lower() for keyword in ['scrape', 'extract', 'content']):
                    if 'maestro_scrape' in available_tools:
                        available_validation_tools.append('maestro_scrape')
                        criterion.fallback_available = True
            
            # Update criterion with mapped tools
            criterion.validation_tools = available_validation_tools
            if not hasattr(criterion, 'fallback_available'):
                criterion.fallback_available = False
        
        return success_criteria
    
    async def _handle_orchestration_error(self, error: Exception, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle orchestration errors with adaptive recovery."""
        logger.info("🔧 Applying enhanced error handling...")
        
        try:
            error_details = {
                "type": "orchestration_error",
                "message": str(error),
                "component": "orchestrator",
                "impact": "task_execution_blocked"
            }
            
            from .adaptive_error_handler import TemporalContext
            from datetime import datetime, timezone
            
            temporal_context = TemporalContext(
                current_timestamp=datetime.now(timezone.utc),
                information_cutoff=None,
                task_deadline=None,
                context_freshness_required=context.get("temporal_context", {}).get("context_freshness_required", False),
                temporal_relevance_window=context.get("temporal_context", {}).get("temporal_relevance_window", "24h")
            )
            
            available_tools = await self._discover_available_tools(context)
            
            error_context = await self.error_handler.analyze_error_context(
                error_details=error_details,
                temporal_context=temporal_context,
                available_tools=available_tools,
                success_criteria=[]
            )
            
            reconsideration = await self.error_handler.should_reconsider_approach(error_context)
            
            response = {
                "status": "orchestration_failed_with_recovery",
                "error": str(error),
                "error_analysis": {
                    "error_id": error_context.error_id,
                    "severity": error_context.severity.value,
                    "trigger": error_context.trigger.value
                },
                "recovery_analysis": {
                    "should_reconsider": reconsideration.should_reconsider,
                    "confidence": reconsideration.confidence_score,
                    "reasoning": reconsideration.reasoning
                }
            }
            
            if reconsideration.should_reconsider:
                response["alternative_approaches"] = reconsideration.alternative_approaches
                response["recommended_tools"] = reconsideration.recommended_tools
                response["temporal_adjustments"] = reconsideration.temporal_adjustments
                
                # Provide enhanced guidance
                response["enhanced_guidance"] = f"""
# 🔧 MAESTRO Error Recovery Guidance

**Original Error:** {str(error)}

## Recovery Recommendations:

{reconsideration.reasoning}

### Alternative Approaches:
{chr(10).join(f"- {approach['description']}" for approach in reconsideration.alternative_approaches)}

### Recommended Tools:
{chr(10).join(f"- {tool}" for tool in reconsideration.recommended_tools)}

### Next Steps:
1. Consider using the recommended alternative approaches
2. Use the MAESTRO built-in tools as fallbacks
3. Apply temporal adjustments if information freshness is an issue
4. Try a simplified approach if complexity is the issue

**Built-in Tools Available:** {', '.join(self.built_in_tools.keys())}
"""
            else:
                response["fallback_guidance"] = self._get_fallback_guidance(task_description)
            
            return response
            
        except Exception as recovery_error:
            logger.error(f"❌ Error recovery failed: {str(recovery_error)}")
            return {
                "status": "orchestration_failed",
                "error": str(error),
                "recovery_error": str(recovery_error),
                "fallback_guidance": self._get_fallback_guidance(task_description)
            }

    # Legacy compatibility methods
    async def analyze_task_for_planning(self, task_description: str, detail_level: str = "comprehensive") -> Dict[str, Any]:
        """Legacy compatibility wrapper."""
        result = await self.orchestrate_task(task_description)
        if result["status"] == "orchestration_complete":
            return {
                "task_analysis": result["task_analysis"],
                "execution_phases": result["workflow"]["phases"],
                "success_criteria": result["success_criteria"]["criteria"],
                "system_prompt_guidance": result["execution_guidance"]
            }
        return {"error": "Orchestration failed", "details": result}
