"""
MAESTRO Protocol Orchestrator - Corrected Architecture

Provides PLANNING and ANALYSIS tools for LLM enhancement.
MCP server gives guidance, LLM executes using IDE tools.
"""

import logging
from typing import Dict, List, Any, Optional
import re

from .data_models import (
    MAESTROResult, TaskAnalysis, Workflow, 
    TaskType, ComplexityLevel, VerificationMethod
)
from .quality_controller import QualityController
from .templates import get_template, list_available_templates

try:
    from ..engines import IntelligenceAmplifier
except ImportError:
    IntelligenceAmplifier = None

logger = logging.getLogger(__name__)


class MAESTROOrchestrator:
    """
    MAESTRO Protocol Orchestration Planner.
    Provides analysis and planning tools to enhance LLM capabilities.
    """
    
    def __init__(self):
        self.quality_controller = QualityController()
        
        if IntelligenceAmplifier:
            self.intelligence_amplifier = IntelligenceAmplifier()
        else:
            self.intelligence_amplifier = None
            
        self.task_patterns = self._initialize_task_patterns()
        logger.info("🎭 MAESTRO Orchestrator initialized as planning engine")
    
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
            ]
        }    
    async def analyze_task_for_planning(
        self, 
        task_description: str,
        detail_level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        PRIMARY MCP TOOL: Analyze task and provide planning guidance.
        
        This is what the LLM calls to get orchestration guidance.
        Returns planning information, not execution.
        
        Args:
            task_description: Natural language task description
            detail_level: "fast", "balanced", or "comprehensive"
            
        Returns:
            Dict containing analysis, template, phases, and guidance
        """
        logger.info(f"🔍 Analyzing task for planning: {task_description[:100]}...")
        
        # Step 1: Task Analysis
        task_analysis = await self._analyze_task_complexity(task_description)
        
        # Step 2: Select appropriate template
        template = self._select_workflow_template(task_analysis.task_type)
        
        # Step 3: Generate execution phases with success criteria
        execution_phases = self._generate_execution_phases(
            task_analysis, template, detail_level
        )
        
        # Step 4: Create system prompt guidance
        system_prompt_guidance = self._generate_system_prompt_guidance(
            task_analysis, template
        )
        
        # Step 5: Identify verification methods
        verification_methods = self._get_verification_methods(task_analysis.task_type)
        
        return {
            "task_analysis": {
                "task_type": task_analysis.task_type.value,
                "complexity": task_analysis.complexity.value,
                "capabilities": task_analysis.capabilities,
                "estimated_duration": task_analysis.estimated_duration
            },
            "template_used": template["template_name"],
            "system_prompt_guidance": system_prompt_guidance,
            "execution_phases": execution_phases,
            "verification_methods": [vm.value for vm in verification_methods],
            "success_criteria": task_analysis.success_criteria,
            "recommended_tools": self._get_recommended_tools(task_analysis),
            "quality_standards": task_analysis.quality_requirements
        }    
    async def create_execution_plan(
        self,
        task_description: str,
        phase_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP TOOL: Create detailed execution plan for a specific phase.
        
        Args:
            task_description: Task to plan for
            phase_focus: Specific phase to focus on (optional)
            
        Returns:
            Detailed execution plan with steps and criteria
        """
        analysis = await self.analyze_task_for_planning(task_description)
        
        if phase_focus:
            # Focus on specific phase
            target_phase = None
            for phase in analysis["execution_phases"]:
                if phase["phase"].lower() == phase_focus.lower():
                    target_phase = phase
                    break
            
            if target_phase:
                return {
                    "focused_phase": target_phase,
                    "detailed_steps": self._generate_detailed_steps(target_phase),
                    "tools_needed": target_phase.get("suggested_tools", []),
                    "success_validation": target_phase["success_criteria"]
                }
        
        # Return full execution plan
        return {
            "full_plan": analysis["execution_phases"],
            "execution_sequence": [phase["phase"] for phase in analysis["execution_phases"]],
            "critical_success_factors": analysis["success_criteria"],
            "quality_gates": analysis["verification_methods"]
        }
    
    def get_available_templates(self) -> List[str]:
        """MCP TOOL: Get list of available workflow templates."""
        return list_available_templates()
    
    def get_template_details(self, template_name: str) -> Dict[str, Any]:
        """MCP TOOL: Get detailed information about a specific template."""
        template = get_template(template_name)
        if not template:
            return {"error": f"Template '{template_name}' not found"}
        return template    
    async def _analyze_task_complexity(self, task_description: str) -> TaskAnalysis:
        """Analyze task to determine type, complexity, and requirements."""
        task_type = self._classify_task_type(task_description.lower())
        complexity = self._assess_complexity(task_description.lower())
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
            quality_requirements=self._get_quality_requirements(complexity)
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
    
    def _select_workflow_template(self, task_type: TaskType) -> Dict[str, Any]:
        """Select appropriate workflow template based on task type."""
        template_mapping = {
            TaskType.CODE_DEVELOPMENT: "code_development",
            TaskType.WEB_DEVELOPMENT: "web_development",
            TaskType.MATHEMATICS: "mathematical_analysis",
            TaskType.DATA_ANALYSIS: "data_analysis",
            TaskType.RESEARCH: "research_analysis",
            TaskType.LANGUAGE_PROCESSING: "documentation",
            TaskType.GENERAL: "documentation"
        }
        
        template_name = template_mapping.get(task_type, "documentation")
        return get_template(template_name)    
    def _generate_system_prompt_guidance(
        self, 
        task_analysis: TaskAnalysis, 
        template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate system prompt guidance based on task and template."""
        guidance = template.get("system_prompt_guidance", {})
        
        return {
            "role": guidance.get("role", "Expert Assistant"),
            "expertise_areas": guidance.get("expertise", []),
            "approach_guidelines": guidance.get("approach", []),
            "quality_standards": guidance.get("quality_standards", []),
            "task_specific_context": {
                "task_type": task_analysis.task_type.value,
                "complexity_level": task_analysis.complexity.value,
                "required_capabilities": task_analysis.capabilities,
                "success_criteria": task_analysis.success_criteria
            }
        }
    
    def _generate_execution_phases(
        self, 
        task_analysis: TaskAnalysis, 
        template: Dict[str, Any],
        detail_level: str
    ) -> List[Dict[str, Any]]:
        """Generate execution phases based on template and task analysis."""
        template_phases = template.get("execution_phases", [])
        
        if detail_level == "fast":
            # Return simplified phases for fast mode
            return template_phases[:2] if template_phases else []
        elif detail_level == "balanced":
            # Return core phases
            return template_phases[:3] if template_phases else []
        else:
            # Return all phases for comprehensive mode
            return template_phases
    
    def _generate_detailed_steps(self, phase: Dict[str, Any]) -> List[str]:
        """Generate detailed steps for a specific phase."""
        base_steps = [
            f"Begin {phase['description'].lower()}",
            "Set up necessary tools and environment",
            "Execute core phase activities",
            "Validate against success criteria",
            f"Complete {phase['phase'].lower()} deliverables"
        ]
        return base_steps    
    # Essential helper methods (simplified)
    def _assess_complexity(self, task_description: str) -> ComplexityLevel:
        """Assess task complexity."""
        if len(task_description.split()) > 50:
            return ComplexityLevel.COMPLEX
        elif len(task_description.split()) > 20:
            return ComplexityLevel.MODERATE
        return ComplexityLevel.SIMPLE
    
    def _determine_capabilities(self, task_type: TaskType, task_description: str) -> List[str]:
        """Determine required capabilities."""
        base_capabilities = {
            TaskType.MATHEMATICS: ["mathematics", "symbolic_computation"],
            TaskType.WEB_DEVELOPMENT: ["web_development", "frontend", "backend"],
            TaskType.CODE_DEVELOPMENT: ["code_generation", "testing"],
            TaskType.DATA_ANALYSIS: ["data_processing", "visualization"],
            TaskType.RESEARCH: ["information_gathering", "analysis"],
            TaskType.LANGUAGE_PROCESSING: ["language_analysis", "text_processing"],
            TaskType.GENERAL: ["general_reasoning"]
        }
        return base_capabilities.get(task_type, ["general_reasoning"])
    
    def _estimate_duration(self, complexity: ComplexityLevel, task_type: TaskType) -> int:
        """Estimate task duration in seconds."""
        base_times = {
            ComplexityLevel.SIMPLE: 30,
            ComplexityLevel.MODERATE: 60,
            ComplexityLevel.COMPLEX: 120,
            ComplexityLevel.EXPERT: 300
        }
        return base_times[complexity]
    
    def _define_success_criteria(self, task_type: TaskType, complexity: ComplexityLevel) -> List[str]:
        """Define success criteria."""
        return [
            "Task completion verified",
            "Quality standards met",
            "No critical errors found"
        ]
    
    def _get_required_tools(self, task_type: TaskType, capabilities: List[str]) -> List[str]:
        """Get required tools."""
        return ["general_tools"]
    
    def _get_quality_requirements(self, complexity: ComplexityLevel) -> Dict[str, float]:
        """Get quality requirements."""
        return {"accuracy": 0.95, "completeness": 0.90}
    
    def _get_verification_methods(self, task_type: TaskType) -> List[VerificationMethod]:
        """Get verification methods."""
        return [VerificationMethod.AUTOMATED_TESTING]
    
    def _get_recommended_tools(self, task_analysis: TaskAnalysis) -> List[str]:
        """Get recommended tools for the task."""
        return ["file_operations", "code_execution", "testing_tools"]