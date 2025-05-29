"""
Enhanced Maestro Orchestrator with Intelligence Amplification Engine Integration
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .context_orchestrator import ContextOrchestrator
from .iae_selector import IAESelector 
from .validation_framework import ValidationFramework
from .success_criteria_analyzer import SuccessCriteriaAnalyzer
from .data_models import (
    MAESTROResult, TaskAnalysis, Workflow, WorkflowNode, 
    VerificationResult, QualityMetrics, ExecutionMetrics
)

logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """Enhanced workflow phases for the Maestro system"""
    CONTEXT_GATHERING = "context_gathering"
    DESIGN_PLANNING = "design_planning"
    IMPLEMENTATION_EXECUTION = "implementation_execution"
    VALIDATION_VERIFICATION = "validation_verification"


@dataclass
class EnhancedWorkflowResult:
    """Result from enhanced workflow execution"""
    success: bool
    phase_completed: WorkflowPhase
    detailed_output: str
    context_gathered: Dict[str, Any]
    iae_engines_used: List[str]
    validation_results: Dict[str, Any]
    files_affected: List[str] = field(default_factory=list)
    execution_metrics: Optional[ExecutionMetrics] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass  
class OrchestrationContext:
    """Complete context for orchestration workflow"""
    original_request: str
    gathered_context: Dict[str, Any]
    success_criteria: List[str]
    validation_metrics: Dict[str, Any]
    iae_mappings: Dict[str, List[str]]
    tool_mappings: Dict[str, List[str]]
    workflow_phases: List[Dict[str, Any]]
    confidence_score: float = 0.0


class EnhancedMAESTROOrchestrator:
    """
    Enhanced Maestro Orchestrator that provides cognitive scaffolding for LLMs
    through Intelligence Amplification Engines (IAEs), dynamic context gathering,
    and comprehensive validation frameworks.
    
    This system proves that intelligence amplification > raw parameter count
    by providing structured reasoning frameworks rather than replacing LLM thinking.
    """
    
    def __init__(self):
        self.context_orchestrator = ContextOrchestrator()
        self.iae_selector = IAESelector()
        self.validation_framework = ValidationFramework()
        self.success_criteria_analyzer = SuccessCriteriaAnalyzer()
        
        # Integration with existing components
        self.tool_discovery = None  # Will be injected
        self.base_orchestrator = None  # Will be injected
        
    def inject_dependencies(self, tool_discovery, base_orchestrator):
        """Inject external dependencies"""
        self.tool_discovery = tool_discovery
        self.base_orchestrator = base_orchestrator
        self.context_orchestrator.set_tool_discovery(tool_discovery)
        
    async def maestro_orchestrate(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        auto_gather_context: bool = True,
        quality_threshold: float = 0.9,
        validation_mode: str = "comprehensive"
    ) -> EnhancedWorkflowResult:
        """
        Primary entry point for enhanced Maestro orchestration.
        
        This method implements the three-phase workflow pattern:
        1. Design/Planning (with context gathering)
        2. Implementation/Execution (with IAE enhancement)
        3. Validation (with success criteria verification)
        
        Args:
            task_description: Natural language description of the task
            context: Optional pre-gathered context
            auto_gather_context: Whether to automatically gather missing context
            quality_threshold: Minimum quality score for completion
            validation_mode: "fast", "balanced", or "comprehensive"
            
        Returns:
            Complete orchestration result with validation
        """
        logger.info(f"🎭 Enhanced Maestro orchestrating: {task_description[:100]}...")
        
        try:
            # PHASE 1: DESIGN/PLANNING WITH CONTEXT ORCHESTRATION
            orchestration_context = await self._phase_1_design_planning(
                task_description, context, auto_gather_context
            )
            
            # PHASE 2: IMPLEMENTATION/EXECUTION WITH IAE ENHANCEMENT  
            implementation_result = await self._phase_2_implementation_execution(
                orchestration_context, quality_threshold
            )
            
            # PHASE 3: VALIDATION WITH SUCCESS CRITERIA VERIFICATION
            validation_result = await self._phase_3_validation_verification(
                orchestration_context, implementation_result, validation_mode
            )
            
            return EnhancedWorkflowResult(
                success=validation_result.success,
                phase_completed=WorkflowPhase.VALIDATION_VERIFICATION,
                detailed_output=implementation_result.get("output", "Task completed"),
                context_gathered=orchestration_context.gathered_context,
                iae_engines_used=list(orchestration_context.iae_mappings.keys()),
                validation_results=validation_result.detailed_results,
                files_affected=implementation_result.get("files_created", []),
                execution_metrics=implementation_result.get("metrics"),
                recommendations=validation_result.recommendations
            )
            
        except Exception as e:
            logger.error(f"Enhanced Maestro orchestration failed: {str(e)}")
            return EnhancedWorkflowResult(
                success=False,
                phase_completed=WorkflowPhase.CONTEXT_GATHERING,
                detailed_output=f"Orchestration failed: {str(e)}",
                context_gathered={},
                iae_engines_used=[],
                validation_results={"error": str(e)},
                recommendations=[
                    "Review task description for clarity",
                    "Check system dependencies",
                    "Try with simplified requirements"
                ]
            )
    
    async def _phase_1_design_planning(
        self,
        task_description: str,
        provided_context: Optional[Dict[str, Any]],
        auto_gather_context: bool
    ) -> OrchestrationContext:
        """
        Phase 1: Design/Planning with intelligent context gathering.
        
        This phase determines if additional context is needed and creates
        a survey framework to gather missing information if required.
        """
        logger.info("📋 Phase 1: Design/Planning with Context Orchestration")
        
        # Analyze task for context requirements
        context_analysis = await self.context_orchestrator.analyze_context_requirements(
            task_description, provided_context
        )
        
        gathered_context = provided_context or {}
        
        # Gather additional context if needed and auto-gathering is enabled
        if context_analysis.needs_additional_context and auto_gather_context:
            logger.info("🔍 Additional context required - generating context gathering survey")
            
            context_survey = await self.context_orchestrator.generate_context_survey(
                task_description, context_analysis.missing_context_areas
            )
            
            # Return survey for user to complete
            # In real implementation, this would pause execution and return survey to user
            gathered_context.update({
                "context_survey_required": True,
                "survey_questions": context_survey.questions,
                "survey_id": context_survey.survey_id
            })
        
        # Analyze success criteria requirements
        success_criteria = await self.success_criteria_analyzer.determine_success_criteria(
            task_description, gathered_context, context_analysis.task_complexity
        )
        
        # Select appropriate IAEs for this task
        iae_mappings = await self.iae_selector.select_engines_for_task(
            task_description, gathered_context, success_criteria
        )
        
        # Map tools for the workflow
        tool_mappings = await self._map_tools_for_workflow(
            task_description, gathered_context, iae_mappings
        )
        
        # Generate workflow phases
        workflow_phases = await self._generate_enhanced_workflow_phases(
            task_description, gathered_context, iae_mappings, tool_mappings
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_planning_confidence(
            context_analysis, len(iae_mappings), len(tool_mappings)
        )
        
        return OrchestrationContext(
            original_request=task_description,
            gathered_context=gathered_context,
            success_criteria=success_criteria.criteria_list,
            validation_metrics=success_criteria.validation_metrics,
            iae_mappings=iae_mappings,
            tool_mappings=tool_mappings,
            workflow_phases=workflow_phases,
            confidence_score=confidence_score
        )
    
    async def _phase_2_implementation_execution(
        self,
        orchestration_context: OrchestrationContext,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Phase 2: Implementation/Execution with IAE cognitive enhancement.
        
        This phase provides the LLM with enhanced reasoning capabilities
        through selected IAEs and structured workflow guidance.
        """
        logger.info("⚡ Phase 2: Implementation/Execution with IAE Enhancement")
        
        # Check if context survey is required
        if orchestration_context.gathered_context.get("context_survey_required"):
            return {
                "output": "Context gathering required before implementation",
                "survey_needed": True,
                "survey_questions": orchestration_context.gathered_context.get("survey_questions", []),
                "status": "awaiting_context"
            }
        
        # Activate selected IAEs to enhance LLM reasoning
        enhanced_reasoning_context = await self._activate_iaes_for_reasoning(
            orchestration_context.iae_mappings,
            orchestration_context.original_request,
            orchestration_context.gathered_context
        )
        
        # Create enhanced system prompt with IAE cognitive scaffolding
        enhanced_prompt = await self._create_iae_enhanced_system_prompt(
            orchestration_context, enhanced_reasoning_context
        )
        
        # Execute workflow with enhanced capabilities
        # This would integrate with the existing orchestrator
        if self.base_orchestrator:
            execution_result = await self.base_orchestrator.execute_workflow_with_context(
                orchestration_context.workflow_phases,
                enhanced_prompt,
                orchestration_context.tool_mappings
            )
        else:
            # Fallback execution simulation
            execution_result = {
                "output": f"Task '{orchestration_context.original_request}' executed with enhanced IAE reasoning",
                "files_created": [],
                "iae_enhancements_applied": list(orchestration_context.iae_mappings.keys()),
                "confidence_score": orchestration_context.confidence_score
            }
        
        return execution_result
    
    async def _phase_3_validation_verification(
        self,
        orchestration_context: OrchestrationContext,
        implementation_result: Dict[str, Any],
        validation_mode: str
    ) -> VerificationResult:
        """
        Phase 3: Validation with comprehensive success criteria verification.
        
        This phase uses both IAEs and tools to validate that the task
        meets the defined success criteria.
        """
        logger.info("✅ Phase 3: Validation with Success Criteria Verification")
        
        # Use validation framework to verify success criteria
        validation_result = await self.validation_framework.comprehensive_validation(
            implementation_result,
            orchestration_context.success_criteria,
            orchestration_context.validation_metrics,
            orchestration_context.iae_mappings,
            validation_mode
        )
        
        return validation_result
    
    async def _activate_iaes_for_reasoning(
        self,
        iae_mappings: Dict[str, List[str]],
        task_description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Activate selected IAEs to provide cognitive enhancement frameworks"""
        enhanced_reasoning = {}
        
        for domain, engines in iae_mappings.items():
            domain_enhancements = []
            
            for engine_name in engines:
                try:
                    # Get reasoning framework from IAE
                    reasoning_framework = await self.iae_selector.get_reasoning_framework(
                        engine_name, task_description, context
                    )
                    domain_enhancements.append(reasoning_framework)
                    
                except Exception as e:
                    logger.warning(f"Failed to activate IAE {engine_name}: {str(e)}")
            
            enhanced_reasoning[domain] = domain_enhancements
        
        return enhanced_reasoning
    
    async def _create_iae_enhanced_system_prompt(
        self,
        orchestration_context: OrchestrationContext,
        enhanced_reasoning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create system prompt enhanced with IAE cognitive frameworks"""
        
        base_prompt = {
            "role": "Enhanced AI Assistant with Cognitive Amplification",
            "task": orchestration_context.original_request,
            "context": orchestration_context.gathered_context,
            "success_criteria": orchestration_context.success_criteria
        }
        
        # Add IAE reasoning frameworks
        iae_frameworks = {}
        for domain, enhancements in enhanced_reasoning_context.items():
            frameworks = []
            for enhancement in enhancements:
                frameworks.append({
                    "engine": enhancement.get("engine_name"),
                    "reasoning_structure": enhancement.get("reasoning_framework"),
                    "analysis_tools": enhancement.get("analysis_capabilities"),
                    "validation_methods": enhancement.get("validation_approaches")
                })
            iae_frameworks[domain] = frameworks
        
        base_prompt["iae_cognitive_enhancements"] = iae_frameworks
        base_prompt["enhanced_capabilities"] = list(orchestration_context.iae_mappings.keys())
        
        return base_prompt
    
    async def _map_tools_for_workflow(
        self,
        task_description: str,
        context: Dict[str, Any],
        iae_mappings: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Map available tools to workflow phases"""
        if not self.tool_discovery:
            return {"general": ["file_operations", "text_processing"]}
        
        # Use existing tool discovery and mapping capabilities
        available_tools = await self.tool_discovery.get_available_tools()
        
        # Map tools based on task requirements and IAE needs
        tool_mappings = {}
        for domain in iae_mappings.keys():
            domain_tools = []
            for tool_name, tool_info in available_tools.items():
                if any(capability in tool_info.get("capabilities", []) 
                       for capability in [domain, "general", "validation"]):
                    domain_tools.append(tool_name)
            tool_mappings[domain] = domain_tools
        
        return tool_mappings
    
    async def _generate_enhanced_workflow_phases(
        self,
        task_description: str,
        context: Dict[str, Any],
        iae_mappings: Dict[str, List[str]],
        tool_mappings: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Generate workflow phases enhanced with IAE and tool context"""
        
        phases = [
            {
                "phase": "analysis",
                "description": "Analyze task requirements with IAE enhancement",
                "iae_engines": iae_mappings.get("analysis", []),
                "tools": tool_mappings.get("analysis", []),
                "success_criteria": ["Requirements understood", "Context gathered", "Approach defined"]
            },
            {
                "phase": "implementation", 
                "description": "Execute task with cognitive amplification",
                "iae_engines": iae_mappings.get("implementation", []),
                "tools": tool_mappings.get("implementation", []),
                "success_criteria": ["Task executed", "Quality maintained", "Files created"]
            },
            {
                "phase": "validation",
                "description": "Validate results against success criteria",
                "iae_engines": iae_mappings.get("validation", []),
                "tools": tool_mappings.get("validation", []),
                "success_criteria": ["Validation complete", "Quality verified", "Success criteria met"]
            }
        ]
        
        return phases
    
    def _calculate_planning_confidence(
        self,
        context_analysis,
        iae_count: int,
        tool_count: int
    ) -> float:
        """Calculate confidence score for the planning phase"""
        base_confidence = 0.7
        
        # Boost confidence based on context completeness
        if not context_analysis.needs_additional_context:
            base_confidence += 0.15
        
        # Boost confidence based on IAE availability
        base_confidence += min(iae_count * 0.05, 0.1)
        
        # Boost confidence based on tool availability
        base_confidence += min(tool_count * 0.02, 0.05)
        
        return min(base_confidence, 1.0)
    
    # Additional methods for context handling
    async def handle_context_survey_response(
        self,
        survey_id: str,
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle user responses to context gathering survey"""
        return await self.context_orchestrator.process_survey_responses(
            survey_id, responses
        )
    
    async def resume_orchestration_with_context(
        self,
        survey_id: str,
        context_responses: Dict[str, Any],
        quality_threshold: float = 0.9,
        validation_mode: str = "comprehensive"
    ) -> EnhancedWorkflowResult:
        """Resume orchestration after context gathering"""
        # Retrieve original context and update with responses
        original_context = await self.context_orchestrator.get_survey_context(survey_id)
        
        updated_context = {**original_context, **context_responses}
        
        # Resume orchestration with complete context
        return await self.maestro_orchestrate(
            task_description=original_context.get("original_task", ""),
            context=updated_context,
            auto_gather_context=False,
            quality_threshold=quality_threshold,
            validation_mode=validation_mode
        ) 