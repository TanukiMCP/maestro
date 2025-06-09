# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Maestro Enhanced Orchestration Framework

Provides comprehensive workflow orchestration with:
- Context intelligence and gap detection
- Success criteria definition and validation
- Tool discovery and mapping to external MCP servers
- Intelligence Amplification Engine (IAE) integration
- Collaborative error handling with automated surveys

Core Principle: Intelligence Amplification > Raw Parameter Count
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import datetime

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for appropriate orchestration."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ValidationPhase(Enum):
    """Validation phases in the orchestration lifecycle."""
    PLANNING = "planning"
    EXECUTION = "execution"
    COMPLETION = "completion"


@dataclass
class ContextGap:
    """Represents a gap in context information."""
    category: str
    description: str
    importance: str  # "critical", "important", "optional"
    suggested_questions: List[str]
    default_value: Optional[Any] = None


@dataclass
class SurveyQuestion:
    """Structured survey question for context gathering."""
    question_id: str
    question_text: str
    question_type: str  # "text", "choice", "number", "boolean", "file_path"
    required: bool
    options: Optional[List[str]] = None
    validation_pattern: Optional[str] = None
    help_text: Optional[str] = None


@dataclass
class ContextSurvey:
    """Survey for gathering missing context."""
    survey_id: str
    title: str
    description: str
    questions: List[SurveyQuestion]
    estimated_time: str
    context_gaps: List[ContextGap]


@dataclass
class SuccessCriterion:
    """Individual success criterion with validation mapping."""
    criterion_id: str
    description: str
    metric_type: str  # "functional", "performance", "quality", "accessibility", "security"
    validation_method: str
    target_value: Optional[Any] = None
    threshold: Optional[float] = None
    validation_tools: List[str] = field(default_factory=list)
    validation_iaes: List[str] = field(default_factory=list)
    priority: str = "medium"  # "critical", "high", "medium", "low"


@dataclass
class SuccessCriteria:
    """Complete success criteria for a task."""
    criteria: List[SuccessCriterion]
    validation_strategy: str
    completion_threshold: float  # Percentage of criteria that must pass
    estimated_validation_time: str


@dataclass
class ToolMapping:
    """Mapping of tools to specific workflow phases."""
    tool_id: str
    tool_name: str
    server_name: str
    workflow_phase: str
    usage_context: str
    command_template: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    fallback_tools: List[str] = field(default_factory=list)


@dataclass
class IAEMapping:
    """Mapping of Intelligence Amplification Engines to workflow phases."""
    iae_id: str
    iae_name: str
    workflow_phase: str
    enhancement_type: str  # "analysis", "reasoning", "validation", "optimization"
    application_context: str
    libraries_required: List[str] = field(default_factory=list)
    cognitive_enhancement: str = ""


@dataclass
class WorkflowPhase:
    """Individual phase in the orchestrated workflow."""
    phase_id: str
    phase_name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    tool_mappings: List[ToolMapping]
    iae_mappings: List[IAEMapping]
    success_criteria: List[str]  # References to SuccessCriterion IDs
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: Optional[str] = None


@dataclass
class OrchestrationWorkflow:
    """Complete orchestrated workflow with all mappings."""
    workflow_id: str
    task_description: str
    complexity: TaskComplexity
    phases: List[WorkflowPhase]
    success_criteria: SuccessCriteria
    tool_mappings: Dict[str, ToolMapping]
    iae_mappings: Dict[str, IAEMapping]
    context_requirements: Dict[str, Any]
    estimated_total_time: str
    created_timestamp: str


@dataclass
class ValidationResult:
    """Result of success criteria validation."""
    criterion_id: str
    passed: bool
    actual_value: Optional[Any]
    expected_value: Optional[Any]
    validation_details: str
    tools_used: List[str]
    iaes_used: List[str]
    timestamp: str


@dataclass
class OrchestrationResult:
    """Complete orchestration result with validation."""
    workflow: OrchestrationWorkflow
    execution_guidance: str
    validation_results: List[ValidationResult]
    overall_success: bool
    completion_percentage: float
    recommendations: List[str]
    next_steps: List[str]


class ContextIntelligenceEngine:
    """Engine for detecting context gaps and generating surveys."""
    
    def __init__(self):
        self.task_patterns = self._load_task_patterns()
        self.context_templates = self._load_context_templates()
    
    def analyze_context_gaps(self, task_description: str, provided_context: Dict[str, Any]) -> List[ContextGap]:
        """
        Analyze task description and identify missing context.
        
        Returns list of context gaps that need to be filled.
        """
        logger.info(f"🔍 Analyzing context gaps for task: {task_description[:100]}...")
        
        gaps = []
        task_type = self._identify_task_type(task_description)
        required_context = self._get_required_context_for_task_type(task_type)
        
        for context_key, context_info in required_context.items():
            if context_key not in provided_context or not provided_context[context_key]:
                gap = ContextGap(
                    category=context_info["category"],
                    description=context_info["description"],
                    importance=context_info["importance"],
                    suggested_questions=context_info["questions"],
                    default_value=context_info.get("default")
                )
                gaps.append(gap)
        
        logger.info(f"✅ Identified {len(gaps)} context gaps")
        return gaps
    
    def generate_context_survey(self, gaps: List[ContextGap], task_description: str) -> ContextSurvey:
        """Generate a structured survey to fill context gaps."""
        logger.info(f"📋 Generating context survey for {len(gaps)} gaps...")
        
        questions = []
        for i, gap in enumerate(gaps, 1):
            for j, question_text in enumerate(gap.suggested_questions):
                question = SurveyQuestion(
                    question_id=f"q_{i}_{j}",
                    question_text=question_text,
                    question_type=self._determine_question_type(question_text, gap),
                    required=gap.importance in ["critical", "important"],
                    help_text=gap.description
                )
                questions.append(question)
        
        survey = ContextSurvey(
            survey_id=f"survey_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"Context Information for: {task_description[:50]}...",
            description="Please provide additional information to ensure optimal workflow orchestration.",
            questions=questions,
            estimated_time=f"{len(questions) * 30} seconds",
            context_gaps=gaps
        )
        
        logger.info(f"✅ Generated survey with {len(questions)} questions")
        return survey
    
    def _identify_task_type(self, task_description: str) -> str:
        """Identify the type of task from description."""
        task_lower = task_description.lower()
        
        if any(keyword in task_lower for keyword in ["website", "web", "html", "css", "frontend"]):
            return "web_development"
        elif any(keyword in task_lower for keyword in ["api", "backend", "server", "database"]):
            return "backend_development"
        elif any(keyword in task_lower for keyword in ["app", "mobile", "ios", "android"]):
            return "mobile_development"
        elif any(keyword in task_lower for keyword in ["data", "analysis", "visualization", "machine learning"]):
            return "data_science"
        elif any(keyword in task_lower for keyword in ["test", "testing", "qa", "quality"]):
            return "testing"
        else:
            return "general_development"
    
    def _get_required_context_for_task_type(self, task_type: str) -> Dict[str, Dict[str, Any]]:
        """Get required context fields for specific task types."""
        context_maps = {
            "web_development": {
                "target_audience": {
                    "category": "requirements",
                    "description": "Who will use this website?",
                    "importance": "important",
                    "questions": [
                        "Who is the target audience for this website?",
                        "What age group and demographics should we consider?",
                        "What devices will users primarily access this from?"
                    ]
                },
                "design_preferences": {
                    "category": "design",
                    "description": "Visual and design requirements",
                    "importance": "important",
                    "questions": [
                        "Do you have any color scheme preferences?",
                        "What style should the website have (modern, classic, minimalist, etc.)?",
                        "Are there any websites you'd like to use as inspiration?"
                    ]
                },
                "functionality_requirements": {
                    "category": "features",
                    "description": "Required functionality and features",
                    "importance": "critical",
                    "questions": [
                        "What key features must the website have?",
                        "Do you need contact forms, galleries, e-commerce, etc.?",
                        "Should the website integrate with any external services?"
                    ]
                },
                "content_assets": {
                    "category": "content",
                    "description": "Available content and media",
                    "importance": "important",
                    "questions": [
                        "Do you have existing content (text, images, videos)?",
                        "What content do you need help creating?",
                        "Do you have brand guidelines or existing marketing materials?"
                    ]
                },
                "technical_constraints": {
                    "category": "technical",
                    "description": "Technical requirements and constraints",
                    "importance": "important",
                    "questions": [
                        "Do you have hosting preferences or requirements?",
                        "Are there any technical constraints we should know about?",
                        "Do you need CMS functionality for content updates?"
                    ]
                }
            },
            "backend_development": {
                "data_requirements": {
                    "category": "data",
                    "description": "Data storage and management needs",
                    "importance": "critical",
                    "questions": [
                        "What type of data will the system handle?",
                        "What are your data storage requirements?",
                        "Do you need specific database technologies?"
                    ]
                },
                "performance_requirements": {
                    "category": "performance",
                    "description": "Performance and scalability needs",
                    "importance": "important",
                    "questions": [
                        "How many users do you expect the system to handle?",
                        "What are your performance requirements?",
                        "Do you need real-time capabilities?"
                    ]
                },
                "integration_requirements": {
                    "category": "integration",
                    "description": "External system integrations",
                    "importance": "important",
                    "questions": [
                        "Do you need to integrate with existing systems?",
                        "What APIs or external services will you use?",
                        "Are there any data migration requirements?"
                    ]
                }
            }
        }
        
        return context_maps.get(task_type, {})
    
    def _determine_question_type(self, question_text: str, gap: ContextGap) -> str:
        """Determine appropriate question type based on content."""
        question_lower = question_text.lower()
        
        if "file" in question_lower or "upload" in question_lower or "asset" in question_lower:
            return "file_path"
        elif "color" in question_lower:
            return "text"
        elif "how many" in question_lower or "number" in question_lower:
            return "number"
        elif "yes" in question_lower or "no" in question_lower or "do you" in question_lower:
            return "boolean"
        else:
            return "text"
    
    def _load_task_patterns(self) -> Dict[str, Any]:
        """Load task pattern recognition data."""
        return {}
    
    def _load_context_templates(self) -> Dict[str, Any]:
        """Load context requirement templates."""
        return {}


class SuccessCriteriaEngine:
    """Engine for defining and validating success criteria."""
    
    def __init__(self):
        self.criteria_templates = self._load_criteria_templates()
        self.validation_mappings = self._load_validation_mappings()
    
    def define_success_criteria(self, task_description: str, task_type: str, context: Dict[str, Any]) -> SuccessCriteria:
        """Define comprehensive success criteria for the task."""
        logger.info(f"🎯 Defining success criteria for {task_type} task...")
        
        criteria = []
        
        # Get base criteria for task type
        base_criteria = self._get_base_criteria_for_task_type(task_type)
        
        for criterion_data in base_criteria:
            criterion = SuccessCriterion(
                criterion_id=criterion_data["id"],
                description=criterion_data["description"],
                metric_type=criterion_data["metric_type"],
                validation_method=criterion_data["validation_method"],
                target_value=criterion_data.get("target_value"),
                threshold=criterion_data.get("threshold"),
                validation_tools=criterion_data.get("validation_tools", []),
                validation_iaes=criterion_data.get("validation_iaes", []),
                priority=criterion_data.get("priority", "medium")
            )
            criteria.append(criterion)
        
        # Add context-specific criteria
        context_criteria = self._generate_context_specific_criteria(task_description, context)
        criteria.extend(context_criteria)
        
        success_criteria = SuccessCriteria(
            criteria=criteria,
            validation_strategy="comprehensive",
            completion_threshold=0.85,  # 85% of criteria must pass
            estimated_validation_time=f"{len(criteria) * 15} seconds"
        )
        
        logger.info(f"✅ Defined {len(criteria)} success criteria")
        return success_criteria
    
    def _get_base_criteria_for_task_type(self, task_type: str) -> List[Dict[str, Any]]:
        """Get base success criteria for specific task types."""
        criteria_maps = {
            "web_development": [
                {
                    "id": "functional_requirements",
                    "description": "All functional requirements are implemented correctly",
                    "metric_type": "functional",
                    "validation_method": "automated_testing",
                    "validation_tools": ["playwright", "selenium", "jest"],
                    "validation_iaes": ["Design Thinking Engine", "Systems Engineering Engine"],
                    "priority": "critical"
                },
                {
                    "id": "responsive_design",
                    "description": "Website is responsive across all target devices",
                    "metric_type": "quality",
                    "validation_method": "responsive_testing",
                    "validation_tools": ["browser_testing", "responsive_design_checker"],
                    "validation_iaes": ["Visual Art Engine", "Design Thinking Engine"],
                    "priority": "high"
                },
                {
                    "id": "accessibility_standards",
                    "description": "Website meets WCAG accessibility standards",
                    "metric_type": "accessibility",
                    "validation_method": "accessibility_audit",
                    "validation_tools": ["axe", "lighthouse", "accessibility_checker"],
                    "validation_iaes": ["Accessibility Engine", "Ethics & Bias Analysis Engine"],
                    "priority": "high"
                },
                {
                    "id": "performance_standards",
                    "description": "Website meets performance benchmarks",
                    "metric_type": "performance",
                    "validation_method": "performance_testing",
                    "threshold": 90.0,  # Lighthouse score
                    "validation_tools": ["lighthouse", "pagespeed", "webpagetest"],
                    "validation_iaes": ["Systems Engineering Engine", "Algorithm Design Engine"],
                    "priority": "high"
                },
                {
                    "id": "code_quality",
                    "description": "Code follows best practices and quality standards",
                    "metric_type": "quality",
                    "validation_method": "code_analysis",
                    "validation_tools": ["eslint", "prettier", "sonarqube"],
                    "validation_iaes": ["Software Engineering Engine", "Algorithm Design Engine"],
                    "priority": "medium"
                },
                {
                    "id": "security_standards",
                    "description": "Website implements security best practices",
                    "metric_type": "security",
                    "validation_method": "security_scan",
                    "validation_tools": ["snyk", "security_headers", "ssl_check"],
                    "validation_iaes": ["Cybersecurity Engine", "Ethics & Bias Analysis Engine"],
                    "priority": "high"
                }
            ],
            "backend_development": [
                {
                    "id": "api_functionality",
                    "description": "All API endpoints work correctly",
                    "metric_type": "functional",
                    "validation_method": "api_testing",
                    "validation_tools": ["postman", "insomnia", "pytest"],
                    "validation_iaes": ["Systems Engineering Engine", "Algorithm Design Engine"],
                    "priority": "critical"
                },
                {
                    "id": "performance_benchmarks",
                    "description": "API meets performance requirements",
                    "metric_type": "performance",
                    "validation_method": "load_testing",
                    "validation_tools": ["jmeter", "k6", "artillery"],
                    "validation_iaes": ["Systems Engineering Engine", "Statistical Analysis Engine"],
                    "priority": "high"
                },
                {
                    "id": "data_integrity",
                    "description": "Data operations maintain integrity",
                    "metric_type": "functional",
                    "validation_method": "data_validation",
                    "validation_tools": ["database_tests", "data_validation_tools"],
                    "validation_iaes": ["Data Science Engine", "Statistical Analysis Engine"],
                    "priority": "critical"
                }
            ]
        }
        
        return criteria_maps.get(task_type, [])
    
    def _generate_context_specific_criteria(self, task_description: str, context: Dict[str, Any]) -> List[SuccessCriterion]:
        """Generate additional criteria based on specific context."""
        criteria = []
        
        # Add context-specific criteria based on requirements
        if context.get("target_audience"):
            criteria.append(SuccessCriterion(
                criterion_id="target_audience_alignment",
                description=f"Solution meets needs of target audience: {context['target_audience']}",
                metric_type="quality",
                validation_method="user_experience_review",
                validation_tools=["user_testing", "feedback_collection"],
                validation_iaes=["Behavioral Science Engine", "Design Thinking Engine"],
                priority="high"
            ))
        
        return criteria
    
    def _load_criteria_templates(self) -> Dict[str, Any]:
        """Load success criteria templates."""
        return {}
    
    def _load_validation_mappings(self) -> Dict[str, Any]:
        """Load validation method mappings."""
        return {}


class EnhancedOrchestrationEngine:
    """
    General-purpose, domain-agnostic orchestration engine for LLM-driven workflows.
    Provides context intelligence, tool/IAE discovery, and validation for any task.
    """
    def __init__(self):
        self.context_engine = ContextIntelligenceEngine()
        self.success_engine = SuccessCriteriaEngine()
        self.iae_registry = self._load_iae_registry()
        self.tool_mappings = {}
        self.workflow_templates = self._load_workflow_templates()

    async def _discover_available_tools(self) -> Dict[str, Any]:
        """
        Discover available tools from the provided context.
        This is a backend-only method that relies on tools being provided by the external caller.
        """
        # Get tools from context if available
        context_tools = getattr(self, '_context_tools', {})
        if context_tools:
            return context_tools

        # Default to built-in MAESTRO tools as fallback
        default_tools = {
            "maestro_search": {
                "name": "MAESTRO Search",
                "description": "LLM-driven web search capability",
                "server": "maestro",
                "tool_type": "search",
                "usage_context": "Information gathering and research"
            },
            "maestro_execute": {
                "name": "MAESTRO Execute",
                "description": "Code and command execution capability",
                "server": "maestro",
                "tool_type": "execution",
                "usage_context": "Code execution and validation"
            },
            "maestro_scrape": {
                "name": "MAESTRO Scrape",
                "description": "Web content extraction capability",
                "server": "maestro",
                "tool_type": "scraping",
                "usage_context": "Data extraction and processing"
            }
        }
        
        return default_tools

    async def orchestrate_complete_workflow(
        self,
        task_description: str,
        provided_context: Optional[Dict[str, Any]] = None
    ) -> Union[OrchestrationResult, ContextSurvey]:
        logger.info(f"🎭 Starting general-purpose workflow orchestration for: {task_description[:100]}...")
        if provided_context is None:
            provided_context = {}
            
        # Store available tools in context for discovery
        if "available_tools" in provided_context:
            self._context_tools = {
                tool["name"]: {
                    "name": tool["name"],
                    "description": tool.get("description", "No description available"),
                    "server": tool.get("server", "unknown"),
                    "tool_type": tool.get("tool_type", "unknown"),
                    "usage_context": tool.get("usage_context", "General purpose")
                }
                for tool in provided_context["available_tools"]
            }
            
        # Step 1: Analyze context gaps
        context_gaps = self.context_engine.analyze_context_gaps(task_description, provided_context)
        critical_gaps = [gap for gap in context_gaps if gap.importance == "critical"]
        if critical_gaps:
            logger.info(f"⚠️ Found {len(critical_gaps)} critical context gaps - generating survey")
            survey = self.context_engine.generate_context_survey(context_gaps, task_description)
            return survey
        # Step 2: Proceed with orchestration
        return await self._perform_full_orchestration(task_description, provided_context, context_gaps)

    async def _perform_full_orchestration(
        self,
        task_description: str,
        context: Dict[str, Any],
        context_gaps: List[ContextGap]
    ) -> OrchestrationResult:
        # Dynamically assess complexity (can be improved to use LLM or heuristics)
        complexity = self._assess_task_complexity(task_description, context)
        logger.info(f"📊 Task analysis: complexity={complexity.value}")
        # Define generic success criteria
        success_criteria = self.success_engine.define_success_criteria(task_description, "generic", context)
        # Create generic workflow phases
        workflow_phases = await self._create_generic_workflow_phases(task_description, complexity, context)
        # Map tools and IAEs to phases
        tool_mappings = await self._map_tools_to_phases(workflow_phases)
        iae_mappings = await self._map_iaes_to_phases(workflow_phases, success_criteria)
        # Create complete workflow
        workflow = OrchestrationWorkflow(
            workflow_id=f"workflow_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_description=task_description,
            complexity=complexity,
            phases=workflow_phases,
            success_criteria=success_criteria,
            tool_mappings=tool_mappings,
            iae_mappings=iae_mappings,
            context_requirements=context,
            estimated_total_time=self._estimate_total_time(workflow_phases),
            created_timestamp=datetime.datetime.now().isoformat()
        )
        execution_guidance = self._generate_execution_guidance(workflow)
        result = OrchestrationResult(
            workflow=workflow,
            execution_guidance=execution_guidance,
            validation_results=[],
            overall_success=True,
            completion_percentage=0.0,
            recommendations=self._generate_recommendations(workflow),
            next_steps=self._generate_next_steps(workflow)
        )
        logger.info(f"✅ Orchestration complete: {len(workflow_phases)} phases, {len(tool_mappings)} tools, {len(iae_mappings)} IAEs")
        return result

    async def _create_generic_workflow_phases(self, task_description: str, complexity: TaskComplexity, context: Dict[str, Any]) -> List[WorkflowPhase]:
        """
        Dynamically generate workflow phases for any task using available tools/IAEs and context.
        This method is domain-agnostic and extensible.
        """
        # Discover available tools and IAEs
        available_tools = await self._discover_available_tools()
        iae_registry = self.iae_registry
        # Use LLM or heuristics to break down the task into phases/steps
        # For now, use a simple generic breakdown (can be replaced with LLM-driven planning)
        phases = []
        # Phase 1: Planning/Analysis
        phases.append(WorkflowPhase(
            phase_id="planning",
            phase_name="Planning & Analysis",
            description="Analyze the task, requirements, and context. Plan the workflow.",
            inputs=["task_description", "context"],
            outputs=["plan", "requirements"],
            tool_mappings=[],
            iae_mappings=[],
            success_criteria=["requirements_defined"],
            estimated_duration="30 minutes"
        ))
        # Phase 2: Execution/Implementation
        phases.append(WorkflowPhase(
            phase_id="execution",
            phase_name="Execution",
            description="Execute the planned steps using available tools and IAEs.",
            inputs=["plan", "requirements"],
            outputs=["results", "artifacts"],
            tool_mappings=[],
            iae_mappings=[],
            success_criteria=["execution_successful"],
            dependencies=["planning"],
            estimated_duration="1-2 hours"
        ))
        # Phase 3: Validation/Review
        phases.append(WorkflowPhase(
            phase_id="validation",
            phase_name="Validation & Review",
            description="Validate results, review outcomes, and ensure success criteria are met.",
            inputs=["results", "artifacts"],
            outputs=["validation_report", "final_output"],
            tool_mappings=[],
            iae_mappings=[],
            success_criteria=["validation_passed"],
            dependencies=["execution"],
            estimated_duration="30 minutes"
        ))
        return phases

    async def _map_tools_to_phases(self, phases: List[WorkflowPhase]) -> Dict[str, ToolMapping]:
        """
        Map available tools to workflow phases in a domain-agnostic way.
        """
        tool_mappings = {}
        available_tools = await self._discover_available_tools()
        for phase in phases:
            for tool_name, tool_info in available_tools.items():
                mapping_id = f"{phase.phase_id}_{tool_name}"
                tool_mappings[mapping_id] = ToolMapping(
                    tool_id=mapping_id,
                    tool_name=tool_name,
                    server_name=tool_info.get("server", "maestro"),
                    workflow_phase=phase.phase_id,
                    usage_context=tool_info.get("usage_context", f"Tool for {phase.phase_name}"),
                    command_template=tool_info.get("command_template"),
                    fallback_tools=tool_info.get("fallback_tools", [])
                )
        return tool_mappings

    async def _map_iaes_to_phases(self, phases: List[WorkflowPhase], success_criteria: SuccessCriteria) -> Dict[str, IAEMapping]:
        """
        Map IAEs to workflow phases in a domain-agnostic way.
        """
        iae_mappings = {}
        for phase in phases:
            for iae_id, iae_info in self.iae_registry.items():
                mapping_id = f"{phase.phase_id}_{iae_id}"
                iae_mappings[mapping_id] = IAEMapping(
                    iae_id=mapping_id,
                    iae_name=iae_info["name"],
                    workflow_phase=phase.phase_id,
                    enhancement_type=iae_info["enhancement_types"][0] if iae_info["enhancement_types"] else "analysis",
                    application_context=iae_info["cognitive_focus"],
                    libraries_required=iae_info.get("libraries", []),
                    cognitive_enhancement=iae_info["cognitive_focus"]
                )
        return iae_mappings

    def _assess_task_complexity(self, task_description: str, context: Dict[str, Any]) -> TaskComplexity:
        """Assess task complexity based on description and context."""
        complexity_indicators = 0
        
        # Check for complexity indicators
        complex_keywords = ["integration", "scalable", "enterprise", "machine learning", "ai", "complex"]
        moderate_keywords = ["responsive", "database", "api", "testing", "automation"]
        
        if any(keyword in task_description.lower() for keyword in complex_keywords):
            complexity_indicators += 2
        elif any(keyword in task_description.lower() for keyword in moderate_keywords):
            complexity_indicators += 1
        
        # Check context complexity
        if len(context) > 5:
            complexity_indicators += 1
        
        if complexity_indicators >= 3:
            return TaskComplexity.EXPERT
        elif complexity_indicators >= 2:
            return TaskComplexity.COMPLEX
        elif complexity_indicators >= 1:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    def _generate_execution_guidance(self, workflow: OrchestrationWorkflow) -> str:
        """Generate comprehensive execution guidance for the LLM."""
        
        guidance = f"""# 🎭 Maestro Orchestration Execution Guidance

## 📋 Task Overview
**Description:** {workflow.task_description}
**Complexity:** {workflow.complexity.value}
**Estimated Time:** {workflow.estimated_total_time}
**Total Phases:** {len(workflow.phases)}

## 🚀 Execution Pattern

This orchestration follows the **Intelligence Amplification** principle:
1. **Enhanced Planning** - Use IAEs for cognitive enhancement
2. **Tool-Aware Implementation** - Execute with mapped tool guidance  
3. **Validation-Driven Completion** - Verify against success criteria

## 📊 Workflow Phases

"""
        
        for i, phase in enumerate(workflow.phases, 1):
            guidance += f"""
### Phase {i}: {phase.phase_name}
**Duration:** {phase.estimated_duration}
**Description:** {phase.description}

**🧠 Intelligence Amplification (IAEs):**
"""
            phase_iaes = [iae for iae in workflow.iae_mappings.values() if iae.workflow_phase == phase.phase_id]
            for iae in phase_iaes:
                guidance += f"- **{iae.iae_name}**: {iae.cognitive_enhancement}\n"
            
            guidance += f"""
**🔧 Tool Mappings:**
"""
            phase_tools = [tool for tool in workflow.tool_mappings.values() if tool.workflow_phase == phase.phase_id]
            for tool in phase_tools:
                guidance += f"- **{tool.tool_name}** ({tool.server_name}): {tool.usage_context}\n"
                if tool.command_template:
                    guidance += f"  Command: `{tool.command_template}`\n"
            
            guidance += f"""
**✅ Success Criteria for this Phase:**
"""
            for criterion_id in phase.success_criteria:
                criterion = next((c for c in workflow.success_criteria.criteria if c.criterion_id == criterion_id), None)
                if criterion:
                    guidance += f"- {criterion.description}\n"
            
            guidance += "\n---\n"
        
        guidance += f"""
## 🎯 Overall Success Criteria

**Completion Threshold:** {workflow.success_criteria.completion_threshold * 100}% of criteria must pass
**Validation Strategy:** {workflow.success_criteria.validation_strategy}

"""
        
        for criterion in workflow.success_criteria.criteria:
            guidance += f"""
### {criterion.criterion_id.replace('_', ' ').title()}
- **Description:** {criterion.description}
- **Type:** {criterion.metric_type}
- **Priority:** {criterion.priority}
- **Validation Method:** {criterion.validation_method}
- **Tools for Validation:** {', '.join(criterion.validation_tools) if criterion.validation_tools else 'Manual review'}
- **IAEs for Validation:** {', '.join(criterion.validation_iaes) if criterion.validation_iaes else 'None'}
"""
        
        guidance += f"""
## 🔄 Execution Instructions

1. **Follow Phase Sequence:** Execute phases in dependency order
2. **Apply IAE Enhancement:** Use mapped IAEs for cognitive amplification during each phase
3. **Utilize Tool Mappings:** Execute using discovered and mapped tools where available
4. **Validate Continuously:** Check success criteria at each phase completion
5. **Document Progress:** Track completion and validation results

## 📈 Intelligence Amplification Notes

This workflow leverages **{len(workflow.iae_mappings)} Intelligence Amplification Engines** to enhance your reasoning capabilities:

"""
        
        for iae in workflow.iae_mappings.values():
            guidance += f"- **{iae.iae_name}** ({iae.enhancement_type}): Applied during {iae.workflow_phase} phase\n"
        
        guidance += f"""
## ⚡ Next Steps

1. Begin with Phase 1: {workflow.phases[0].phase_name}
2. Apply mapped IAEs for enhanced reasoning
3. Use tool mappings for efficient execution
4. Validate against success criteria before proceeding to next phase

**Remember:** Intelligence Amplification > Raw Parameter Count
This orchestration enhances your capabilities through structured reasoning frameworks and intelligent tool mapping.
"""
        
        return guidance

    def _generate_recommendations(self, workflow: OrchestrationWorkflow) -> List[str]:
        """Generate recommendations for optimal execution."""
        recommendations = [
            "Use the mapped Intelligence Amplification Engines to enhance reasoning at each phase",
            "Follow the tool mappings for efficient execution",
            "Validate success criteria at each phase before proceeding",
            "Document your progress and any deviations from the planned workflow"
        ]
        
        if workflow.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            recommendations.append("Consider breaking down complex phases into smaller sub-tasks")
            recommendations.append("Pay special attention to the validation phase for complex requirements")
        
        return recommendations

    def _generate_next_steps(self, workflow: OrchestrationWorkflow) -> List[str]:
        """Generate immediate next steps."""
        first_phase = workflow.phases[0] if workflow.phases else None
        
        if first_phase:
            return [
                f"Start with Phase 1: {first_phase.phase_name}",
                f"Review the inputs required: {', '.join(first_phase.inputs)}",
                "Apply the mapped IAEs for cognitive enhancement",
                "Begin execution using the provided tool mappings"
            ]
        else:
            return ["Review the workflow structure and begin execution"]

    def _estimate_total_time(self, phases: List[WorkflowPhase]) -> str:
        """Estimate total time for all phases."""
        # Simple estimation - in practice would be more sophisticated
        total_minutes = 0
        for phase in phases:
            if phase.estimated_duration:
                # Extract minutes from duration string
                duration = phase.estimated_duration.lower()
                if "hour" in duration:
                    if "2-4" in duration:
                        total_minutes += 180  # Average of 3 hours
                    else:
                        total_minutes += 60
                elif "minute" in duration:
                    if "30-60" in duration:
                        total_minutes += 45  # Average
                    elif "30" in duration:
                        total_minutes += 30
                    elif "45" in duration:
                        total_minutes += 45
                    elif "20" in duration:
                        total_minutes += 20
        
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
        else:
            return f"{total_minutes}m"

    def _load_iae_registry(self) -> Dict[str, Any]:
        """Load the registry of available Intelligence Amplification Engines."""
        # Registry of available IAEs with their capabilities
        iae_registry = {
            "design_thinking": {
                "name": "Design Thinking Engine",
                "description": "Enhances UX/UI design decisions through user-centered thinking",
                "enhancement_types": ["analysis", "reasoning"],
                "applicable_phases": ["design", "planning"],
                "libraries": ["NetworkX", "NumPy", "pandas"],
                "cognitive_focus": "Design constraint analysis and solution space mapping"
            },
            "visual_art": {
                "name": "Visual Art Engine", 
                "description": "Enhances visual design reasoning and aesthetic analysis",
                "enhancement_types": ["reasoning", "validation"],
                "applicable_phases": ["design", "implementation"],
                "libraries": ["NumPy", "SciPy", "PIL", "colorsys"],
                "cognitive_focus": "Color theory and composition analysis"
            },
            "accessibility": {
                "name": "Accessibility Engine",
                "description": "Enhances accessibility reasoning and inclusive design",
                "enhancement_types": ["validation", "analysis"],
                "applicable_phases": ["testing", "validation"],
                "libraries": ["NLTK", "pandas", "BeautifulSoup"],
                "cognitive_focus": "Barrier identification and mitigation analysis"
            },
            "systems_engineering": {
                "name": "Systems Engineering Engine",
                "description": "Enhances complex system reasoning and optimization",
                "enhancement_types": ["optimization", "analysis"],
                "applicable_phases": ["planning", "implementation"],
                "libraries": ["NetworkX", "NumPy", "SciPy"],
                "cognitive_focus": "System architecture and dependency reasoning"
            },
            "mathematical_reasoning": {
                "name": "Mathematical Reasoning Engine",
                "description": "Enhances mathematical analysis and problem solving",
                "enhancement_types": ["analysis", "validation"],
                "applicable_phases": ["analysis", "implementation"],
                "libraries": ["NumPy", "SciPy", "SymPy"],
                "cognitive_focus": "Mathematical proof validation and optimization"
            },
            "data_analysis": {
                "name": "Data Analysis Engine",
                "description": "Enhances data processing and statistical reasoning",
                "enhancement_types": ["analysis", "optimization"],
                "applicable_phases": ["analysis", "implementation"],
                "libraries": ["pandas", "NumPy", "scikit-learn"],
                "cognitive_focus": "Statistical analysis and data pattern recognition"
            }
        }
        
        return iae_registry

    def _load_workflow_templates(self) -> Dict[str, Any]:
        """Load workflow templates for different task types."""
        return {}


# Export main classes
__all__ = [
    "EnhancedOrchestrationEngine",
    "ContextIntelligenceEngine", 
    "SuccessCriteriaEngine",
    "OrchestrationWorkflow",
    "OrchestrationResult",
    "ContextSurvey",
    "SuccessCriteria",
    "TaskComplexity"
] 
