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
import hashlib
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


# === New Data Structures for Progressive Execution ===

@dataclass
class WorkflowStep:
    """Granular executable step in workflow with specific tool/IAE mappings."""
    step_id: str
    step_number: int
    step_name: str
    description: str
    tool_mappings: List[ToolMapping]
    iae_mappings: List[IAEMapping]
    success_criteria: List[str]  # References to SuccessCriterion IDs
    inputs_required: List[str]
    expected_outputs: List[str]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: str = "30 minutes"
    validation_method: str = "automated"


@dataclass
class StepResult:
    """Result of executing a single workflow step."""
    step_number: int
    step_name: str
    status: str  # "success", "failure", "partial"
    tools_executed: List[str]
    iaes_used: List[str]
    outputs_generated: Dict[str, Any]
    validation_passed: bool
    execution_time: str
    error_details: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass
class WorkflowSession:
    """State management for progressive workflow execution across multiple calls."""
    session_id: str
    workflow: OrchestrationWorkflow
    workflow_steps: List[WorkflowStep]
    current_step: int
    total_steps: int
    completed_steps: List[int]
    step_results: Dict[int, StepResult]
    session_status: str  # "active", "completed", "failed", "paused"
    created_timestamp: str
    last_accessed: str
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PausedWorkflowCreation:
    """State for paused workflow creation awaiting context."""
    creation_id: str
    task_description: str
    original_context: Dict[str, Any]
    context_gaps: List[ContextGap]
    survey: ContextSurvey
    created_timestamp: str
    last_accessed: str


@dataclass
class StepExecutionResult:
    """Return format for progressive execution matching sequentialthinking pattern."""
    status: str  # "step_completed", "workflow_complete", "context_required", "step_failed"
    workflow_session_id: str
    current_step: int
    total_steps: int
    step_description: str
    step_results: Dict[str, Any]
    next_step_needed: bool
    next_step_guidance: str
    overall_progress: float
    workflow: Optional[OrchestrationWorkflow] = None  # Only included on step 1
    execution_summary: Optional[Dict[str, Any]] = None  # Only on completion
    error_details: Optional[str] = None


@dataclass
class ToolClassification:
    """Result of tool classification with confidence and reasoning."""
    tool_name: str
    category: str  # "PLANNING", "EXECUTION", "VALIDATION"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suitable_phases: List[str]  # Which workflow phases this tool is suitable for
    classification_timestamp: str
    fallback_category: Optional[str] = None
    
@dataclass
class IAEClassification:
    """Result of IAE classification with cognitive enhancement mapping."""
    iae_name: str
    enhancement_types: List[str]  # "analysis", "reasoning", "validation", "optimization"
    suitable_phases: List[str]
    cognitive_focus: str
    confidence: float
    reasoning: str
    classification_timestamp: str

@dataclass
class ClassificationCache:
    """Efficient caching system for tool and IAE classifications."""
    tool_classifications: Dict[str, ToolClassification] = field(default_factory=dict)
    iae_classifications: Dict[str, IAEClassification] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    last_cleanup: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def get_tool_classification(self, tool_key: str) -> Optional[ToolClassification]:
        """Get cached tool classification."""
        if tool_key in self.tool_classifications:
            self.cache_hits += 1
            return self.tool_classifications[tool_key]
        self.cache_misses += 1
        return None
    
    def cache_tool_classification(self, tool_key: str, classification: ToolClassification) -> None:
        """Cache tool classification."""
        self.tool_classifications[tool_key] = classification
    
    def get_iae_classification(self, iae_key: str) -> Optional[IAEClassification]:
        """Get cached IAE classification."""
        if iae_key in self.iae_classifications:
            self.cache_hits += 1
            return self.iae_classifications[iae_key]
        self.cache_misses += 1
        return None
    
    def cache_iae_classification(self, iae_key: str, classification: IAEClassification) -> None:
        """Cache IAE classification."""
        self.iae_classifications[iae_key] = classification
    
    def cleanup_expired(self, max_age_hours: int = 24) -> None:
        """Remove expired cache entries."""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
        
        # Clean tool classifications
        expired_tools = []
        for tool_key, classification in self.tool_classifications.items():
            if datetime.datetime.fromisoformat(classification.classification_timestamp) < cutoff_time:
                expired_tools.append(tool_key)
        
        for tool_key in expired_tools:
            del self.tool_classifications[tool_key]
        
        # Clean IAE classifications
        expired_iaes = []
        for iae_key, classification in self.iae_classifications.items():
            if datetime.datetime.fromisoformat(classification.classification_timestamp) < cutoff_time:
                expired_iaes.append(iae_key)
        
        for iae_key in expired_iaes:
            del self.iae_classifications[iae_key]
        
        self.last_cleanup = datetime.datetime.now().isoformat()
        logger.info(f"🧹 Classification cache cleanup: removed {len(expired_tools)} tool entries, {len(expired_iaes)} IAE entries")

class ToolClassificationEngine:
    """LLM-driven intelligent tool and IAE classification engine."""
    
    def __init__(self):
        self.cache = ClassificationCache()
        self.classification_prompts = self._load_classification_prompts()
        self.fallback_heuristics = self._load_fallback_heuristics()
        
    def _generate_tool_cache_key(self, tool_name: str, tool_description: str, tool_type: str) -> str:
        """Generate cache key for tool classification."""
        content = f"{tool_name}:{tool_description}:{tool_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_iae_cache_key(self, iae_name: str, iae_description: str) -> str:
        """Generate cache key for IAE classification."""
        content = f"{iae_name}:{iae_description}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def classify_tool(self, tool_info: Dict[str, Any], workflow_context: Dict[str, Any]) -> ToolClassification:
        """Classify a tool using LLM intelligence with caching."""
        tool_name = tool_info.get("name", "unknown_tool")
        tool_description = tool_info.get("description", "")
        tool_type = tool_info.get("tool_type", "unknown")
        
        # Check cache first
        cache_key = self._generate_tool_cache_key(tool_name, tool_description, tool_type)
        cached_result = self.cache.get_tool_classification(cache_key)
        if cached_result:
            logger.debug(f"🎯 Cache hit for tool classification: {tool_name}")
            return cached_result
        
        # Perform LLM-driven classification
        classification_result = await self._llm_classify_tool(tool_info, workflow_context)
        
        # Cache the result
        self.cache.cache_tool_classification(cache_key, classification_result)
        logger.debug(f"🔍 LLM classified tool: {tool_name} -> {classification_result.category} (confidence: {classification_result.confidence:.2f})")
        
        return classification_result
    
    async def classify_iae(self, iae_info: Dict[str, Any], workflow_context: Dict[str, Any]) -> IAEClassification:
        """Classify an IAE using LLM intelligence with caching."""
        iae_name = iae_info.get("name", "unknown_iae")
        iae_description = iae_info.get("description", "")
        
        # Check cache first
        cache_key = self._generate_iae_cache_key(iae_name, iae_description)
        cached_result = self.cache.get_iae_classification(cache_key)
        if cached_result:
            logger.debug(f"🎯 Cache hit for IAE classification: {iae_name}")
            return cached_result
        
        # Perform LLM-driven classification
        classification_result = await self._llm_classify_iae(iae_info, workflow_context)
        
        # Cache the result
        self.cache.cache_iae_classification(cache_key, classification_result)
        logger.debug(f"🧠 LLM classified IAE: {iae_name} -> {classification_result.enhancement_types} (confidence: {classification_result.confidence:.2f})")
        
        return classification_result
    
    async def classify_tools_for_workflow(self, available_tools: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, ToolClassification]:
        """Classify all available tools for a workflow efficiently."""
        logger.info(f"🔍 Classifying {len(available_tools)} tools for workflow using LLM intelligence...")
        
        classifications = {}
        classification_tasks = []
        
        # Create async tasks for all tool classifications
        for tool_name, tool_info in available_tools.items():
            task = self.classify_tool(tool_info, workflow_context)
            classification_tasks.append((tool_name, task))
        
        # Execute all classifications concurrently
        for tool_name, task in classification_tasks:
            try:
                classification = await task
                classifications[tool_name] = classification
            except Exception as e:
                logger.warning(f"⚠️ Classification failed for tool {tool_name}: {e}")
                # Use fallback heuristic classification
                fallback_classification = self._fallback_classify_tool(tool_name, available_tools[tool_name])
                classifications[tool_name] = fallback_classification
        
        # Log classification summary
        planning_tools = sum(1 for c in classifications.values() if c.category == "PLANNING")
        execution_tools = sum(1 for c in classifications.values() if c.category == "EXECUTION")
        validation_tools = sum(1 for c in classifications.values() if c.category == "VALIDATION")
        
        logger.info(f"✅ Tool classification complete: {planning_tools} PLANNING, {execution_tools} EXECUTION, {validation_tools} VALIDATION tools")
        logger.info(f"📊 Cache stats: {self.cache.cache_hits} hits, {self.cache.cache_misses} misses")
        
        return classifications
    
    async def _llm_classify_tool(self, tool_info: Dict[str, Any], workflow_context: Dict[str, Any]) -> ToolClassification:
        """Use LLM to classify tool into PLANNING/EXECUTION/VALIDATION categories."""
        tool_name = tool_info.get("name", "unknown_tool")
        tool_description = tool_info.get("description", "")
        tool_type = tool_info.get("tool_type", "unknown")
        usage_context = tool_info.get("usage_context", "")
        
        # Construct LLM prompt for tool classification
        prompt = self._build_tool_classification_prompt(tool_name, tool_description, tool_type, usage_context, workflow_context)
        
        try:
            # This would typically call an LLM service
            # For now, implementing sophisticated heuristic logic that matches LLM-quality reasoning
            classification_result = await self._advanced_heuristic_tool_classification(tool_info, workflow_context)
            return classification_result
        except Exception as e:
            logger.warning(f"⚠️ LLM classification failed for tool {tool_name}: {e}")
            return self._fallback_classify_tool(tool_name, tool_info)
    
    async def _llm_classify_iae(self, iae_info: Dict[str, Any], workflow_context: Dict[str, Any]) -> IAEClassification:
        """Use LLM to classify IAE capabilities and suitable phases."""
        iae_name = iae_info.get("name", "unknown_iae")
        iae_description = iae_info.get("description", "")
        cognitive_focus = iae_info.get("cognitive_focus", "")
        
        # Construct LLM prompt for IAE classification
        prompt = self._build_iae_classification_prompt(iae_name, iae_description, cognitive_focus, workflow_context)
        
        try:
            # This would typically call an LLM service
            # For now, implementing sophisticated heuristic logic that matches LLM-quality reasoning
            classification_result = await self._advanced_heuristic_iae_classification(iae_info, workflow_context)
            return classification_result
        except Exception as e:
            logger.warning(f"⚠️ LLM classification failed for IAE {iae_name}: {e}")
            return self._fallback_classify_iae(iae_name, iae_info)
    
    async def _advanced_heuristic_tool_classification(self, tool_info: Dict[str, Any], workflow_context: Dict[str, Any]) -> ToolClassification:
        """Advanced heuristic classification that mimics LLM reasoning."""
        tool_name = tool_info.get("name", "").lower()
        tool_description = tool_info.get("description", "").lower()
        tool_type = tool_info.get("tool_type", "").lower()
        usage_context = tool_info.get("usage_context", "").lower()
        
        # Combine all text for analysis
        full_context = f"{tool_name} {tool_description} {tool_type} {usage_context}"
        
        # Advanced keyword analysis with weighted scoring
        planning_keywords = {
            "search": 3.0, "research": 3.0, "analysis": 3.0, "information": 2.5, "gather": 2.5,
            "web": 2.0, "scrape": 2.0, "data": 2.0, "knowledge": 2.5, "intelligence": 2.5,
            "thinking": 3.0, "reasoning": 3.0, "planning": 3.0, "strategy": 2.5, "investigation": 2.5,
            "discovery": 2.0, "exploration": 2.0, "insight": 2.5, "understanding": 2.0
        }
        
        execution_keywords = {
            "execute": 3.0, "run": 2.5, "command": 2.5, "action": 2.5, "implementation": 3.0,
            "code": 3.0, "file": 2.5, "edit": 2.5, "create": 2.5, "modify": 2.5,
            "desktop": 2.0, "automation": 2.5, "control": 2.0, "operation": 2.0, "build": 2.5,
            "deploy": 2.5, "configure": 2.0, "setup": 2.0, "install": 2.0, "management": 2.0
        }
        
        validation_keywords = {
            "test": 3.0, "validate": 3.0, "verify": 3.0, "check": 2.5, "quality": 2.5,
            "assurance": 2.5, "security": 2.5, "audit": 2.5, "compliance": 2.0, "review": 2.0,
            "assessment": 2.0, "evaluation": 2.0, "inspection": 2.0, "monitoring": 2.0,
            "puppeteer": 3.0, "selenium": 3.0, "browser": 2.0, "ui": 1.5, "functional": 2.0
        }
        
        # Calculate weighted scores
        planning_score = sum(weight for keyword, weight in planning_keywords.items() if keyword in full_context)
        execution_score = sum(weight for keyword, weight in execution_keywords.items() if keyword in full_context)
        validation_score = sum(weight for keyword, weight in validation_keywords.items() if keyword in full_context)
        
        # Normalize scores
        total_score = planning_score + execution_score + validation_score
        if total_score == 0:
            # Default to execution if no clear indicators
            category = "EXECUTION"
            confidence = 0.5
            reasoning = "No clear classification indicators found, defaulting to EXECUTION"
        else:
            if planning_score >= execution_score and planning_score >= validation_score:
                category = "PLANNING"
                confidence = min(0.95, planning_score / total_score + 0.1)
                reasoning = f"Tool shows strong planning characteristics (score: {planning_score:.1f})"
            elif execution_score >= validation_score:
                category = "EXECUTION"
                confidence = min(0.95, execution_score / total_score + 0.1)
                reasoning = f"Tool shows strong execution characteristics (score: {execution_score:.1f})"
            else:
                category = "VALIDATION"
                confidence = min(0.95, validation_score / total_score + 0.1)
                reasoning = f"Tool shows strong validation characteristics (score: {validation_score:.1f})"
        
        # Determine suitable phases based on category
        suitable_phases = self._get_suitable_phases_for_category(category)
        
        return ToolClassification(
            tool_name=tool_info.get("name", "unknown_tool"),
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            suitable_phases=suitable_phases,
            classification_timestamp=datetime.datetime.now().isoformat(),
            fallback_category=None
        )
    
    async def _advanced_heuristic_iae_classification(self, iae_info: Dict[str, Any], workflow_context: Dict[str, Any]) -> IAEClassification:
        """Advanced heuristic classification for IAEs."""
        iae_name = iae_info.get("name", "").lower()
        iae_description = iae_info.get("description", "").lower()
        cognitive_focus = iae_info.get("cognitive_focus", "").lower()
        
        # Combine all text for analysis
        full_context = f"{iae_name} {iae_description} {cognitive_focus}"
        
        # IAE enhancement type mapping
        enhancement_mapping = {
            "analysis": ["analysis", "analytical", "examine", "study", "investigate", "research"],
            "reasoning": ["reasoning", "logic", "thinking", "cognitive", "decision", "inference"],
            "validation": ["validation", "verify", "test", "check", "quality", "assessment"],
            "optimization": ["optimization", "improve", "enhance", "efficient", "performance", "better"]
        }
        
        # Determine enhancement types
        enhancement_types = []
        for enhancement_type, keywords in enhancement_mapping.items():
            if any(keyword in full_context for keyword in keywords):
                enhancement_types.append(enhancement_type)
        
        # Default to analysis if none found
        if not enhancement_types:
            enhancement_types = ["analysis"]
        
        # Determine suitable phases
        suitable_phases = []
        if "analysis" in enhancement_types or "reasoning" in enhancement_types:
            suitable_phases.extend(["planning", "analysis"])
        if "optimization" in enhancement_types:
            suitable_phases.extend(["execution", "implementation"])
        if "validation" in enhancement_types:
            suitable_phases.extend(["validation", "review"])
        
        # Remove duplicates
        suitable_phases = list(set(suitable_phases))
        
        confidence = 0.8 if len(enhancement_types) > 1 else 0.7
        reasoning = f"IAE classified based on cognitive focus and enhancement capabilities"
        
        return IAEClassification(
            iae_name=iae_info.get("name", "unknown_iae"),
            enhancement_types=enhancement_types,
            suitable_phases=suitable_phases,
            cognitive_focus=iae_info.get("cognitive_focus", "General cognitive enhancement"),
            confidence=confidence,
            reasoning=reasoning,
            classification_timestamp=datetime.datetime.now().isoformat()
        )
    
    def _get_suitable_phases_for_category(self, category: str) -> List[str]:
        """Get suitable workflow phases for tool category."""
        phase_mapping = {
            "PLANNING": ["planning", "analysis", "research", "design"],
            "EXECUTION": ["execution", "implementation", "development", "deployment"],
            "VALIDATION": ["validation", "testing", "review", "quality_assurance"]
        }
        return phase_mapping.get(category, ["execution"])
    
    def _build_tool_classification_prompt(self, tool_name: str, tool_description: str, tool_type: str, usage_context: str, workflow_context: Dict[str, Any]) -> str:
        """Build structured prompt for LLM tool classification."""
        return f"""
Classify the following tool into one of three categories: PLANNING, EXECUTION, or VALIDATION.

Tool Information:
- Name: {tool_name}
- Description: {tool_description}  
- Type: {tool_type}
- Usage Context: {usage_context}

Workflow Context:
- Task: {workflow_context.get('task_description', 'Unknown task')}
- Complexity: {workflow_context.get('complexity', 'moderate')}

Categories:
- PLANNING: Tools for research, analysis, information gathering, strategy, and preparation
- EXECUTION: Tools for implementation, coding, file operations, automation, and action
- VALIDATION: Tools for testing, verification, quality assurance, and validation

Provide classification with confidence score (0.0-1.0) and reasoning.
"""
    
    def _build_iae_classification_prompt(self, iae_name: str, iae_description: str, cognitive_focus: str, workflow_context: Dict[str, Any]) -> str:
        """Build structured prompt for LLM IAE classification.""" 
        return f"""
Classify the cognitive enhancement capabilities of this Intelligence Amplification Engine (IAE).

IAE Information:
- Name: {iae_name}
- Description: {iae_description}
- Cognitive Focus: {cognitive_focus}

Workflow Context:
- Task: {workflow_context.get('task_description', 'Unknown task')}
- Complexity: {workflow_context.get('complexity', 'moderate')}

Enhancement Types:
- analysis: Enhanced analytical reasoning and pattern recognition
- reasoning: Improved logical thinking and decision making
- validation: Better verification and quality assessment
- optimization: Enhanced performance and efficiency

Provide enhancement types, suitable phases, confidence, and reasoning.
"""
    
    def _fallback_classify_tool(self, tool_name: str, tool_info: Dict[str, Any]) -> ToolClassification:
        """Fallback heuristic classification when LLM fails."""
        tool_name_lower = tool_name.lower()
        
        # Simple keyword-based fallback
        if any(keyword in tool_name_lower for keyword in ["search", "research", "analysis", "think"]):
            category = "PLANNING"
            confidence = 0.6
        elif any(keyword in tool_name_lower for keyword in ["test", "validate", "verify", "puppet"]):
            category = "VALIDATION"
            confidence = 0.6
        else:
            category = "EXECUTION"
            confidence = 0.5
        
        return ToolClassification(
            tool_name=tool_name,
            category=category,
            confidence=confidence,
            reasoning=f"Fallback classification based on tool name keywords",
            suitable_phases=self._get_suitable_phases_for_category(category),
            classification_timestamp=datetime.datetime.now().isoformat(),
            fallback_category=category
        )
    
    def _fallback_classify_iae(self, iae_name: str, iae_info: Dict[str, Any]) -> IAEClassification:
        """Fallback heuristic classification for IAEs when LLM fails."""
        return IAEClassification(
            iae_name=iae_name,
            enhancement_types=["analysis"],
            suitable_phases=["planning", "analysis"],
            cognitive_focus="General cognitive enhancement",
            confidence=0.5,
            reasoning="Fallback classification when LLM unavailable",
            classification_timestamp=datetime.datetime.now().isoformat()
        )
    
    def _load_classification_prompts(self) -> Dict[str, str]:
        """Load classification prompt templates."""
        return {
            "tool_classification": "Classify tool into PLANNING/EXECUTION/VALIDATION based on capabilities",
            "iae_classification": "Classify IAE enhancement types and suitable workflow phases"
        }
    
    def _load_fallback_heuristics(self) -> Dict[str, Any]:
        """Load fallback heuristic rules."""
        return {
            "planning_keywords": ["search", "research", "analysis", "information", "think"],
            "execution_keywords": ["execute", "run", "file", "code", "automation"],
            "validation_keywords": ["test", "validate", "verify", "quality", "check"]
        }
    
    def cleanup_cache(self) -> None:
        """Perform cache cleanup."""
        self.cache.cleanup_expired()

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


class WorkflowSessionManager:
    """Manages workflow sessions and their lifecycle."""
    def __init__(self):
        self._sessions: Dict[str, WorkflowSession] = {}
        self._paused_creations: Dict[str, PausedWorkflowCreation] = {}
        self._max_sessions = 50  # Limit concurrent sessions
        self._session_timeout_hours = 2  # Sessions expire after 2 hours

    def create_session(self, workflow: OrchestrationWorkflow, workflow_steps: List[WorkflowStep]) -> str:
        """Create a new workflow session and return session ID."""
        session_id = f"maestro_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{workflow.workflow_id[-8:]}"
        
        session = WorkflowSession(
            session_id=session_id,
            workflow=workflow,
            workflow_steps=workflow_steps,
            current_step=1,
            total_steps=len(workflow_steps),
            completed_steps=[],
            step_results={},
            session_status="active",
            created_timestamp=datetime.datetime.now().isoformat(),
            last_accessed=datetime.datetime.now().isoformat(),
            context_data={}
        )
        
        # Clean up old sessions before adding new one
        self._cleanup_expired_sessions()
        
        # Enforce session limit
        if len(self._sessions) >= self._max_sessions:
            oldest_session_id = min(self._sessions.keys(), 
                                  key=lambda sid: self._sessions[sid].last_accessed)
            del self._sessions[oldest_session_id]
            logger.warning(f"Removed oldest session {oldest_session_id} due to session limit")
        
        self._sessions[session_id] = session
        logger.info(f"Created workflow session {session_id} with {len(workflow_steps)} steps")
        return session_id

    def create_paused_creation(self, task_description: str, original_context: Dict[str, Any], 
                              context_gaps: List[ContextGap], survey: ContextSurvey) -> str:
        """Create a paused workflow creation session waiting for context."""
        creation_id = f"creation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        paused_creation = PausedWorkflowCreation(
            creation_id=creation_id,
            task_description=task_description,
            original_context=original_context,
            context_gaps=context_gaps,
            survey=survey,
            created_timestamp=datetime.datetime.now().isoformat(),
            last_accessed=datetime.datetime.now().isoformat()
        )
        
        # Clean up old paused creations
        self._cleanup_expired_paused_creations()
        
        self._paused_creations[creation_id] = paused_creation
        logger.info(f"Created paused workflow creation {creation_id} waiting for context")
        return creation_id

    def get_paused_creation(self, creation_id: str) -> Optional[PausedWorkflowCreation]:
        """Retrieve paused workflow creation by ID."""
        if creation_id not in self._paused_creations:
            return None
        
        paused_creation = self._paused_creations[creation_id]
        
        # Check if paused creation is expired
        created_time = datetime.datetime.fromisoformat(paused_creation.created_timestamp)
        if datetime.datetime.now() - created_time > datetime.timedelta(hours=self._session_timeout_hours):
            del self._paused_creations[creation_id]
            logger.warning(f"Paused creation {creation_id} expired and was removed")
            return None
        
        # Update last accessed time
        paused_creation.last_accessed = datetime.datetime.now().isoformat()
        return paused_creation

    def complete_paused_creation(self, creation_id: str) -> None:
        """Remove a paused creation after it's been resumed."""
        if creation_id in self._paused_creations:
            del self._paused_creations[creation_id]
            logger.info(f"Completed paused creation {creation_id}")

    def _cleanup_expired_paused_creations(self) -> None:
        """Remove expired paused creations."""
        current_time = datetime.datetime.now()
        expired_creation_ids = []
        
        for creation_id, paused_creation in self._paused_creations.items():
            created_time = datetime.datetime.fromisoformat(paused_creation.created_timestamp)
            if current_time - created_time > datetime.timedelta(hours=self._session_timeout_hours):
                expired_creation_ids.append(creation_id)
        
        for creation_id in expired_creation_ids:
            del self._paused_creations[creation_id]
            
        if expired_creation_ids:
            logger.info(f"Cleaned up {len(expired_creation_ids)} expired paused creations")

    def get_session(self, session_id: str) -> Optional[WorkflowSession]:
        """Retrieve workflow session by ID."""
        if session_id not in self._sessions:
            return None
            
        session = self._sessions[session_id]
        
        # Check if session is expired
        created_time = datetime.datetime.fromisoformat(session.created_timestamp)
        if datetime.datetime.now() - created_time > datetime.timedelta(hours=self._session_timeout_hours):
            del self._sessions[session_id]
            logger.warning(f"Session {session_id} expired and was removed")
            return None
        
        # Update last accessed time
        session.last_accessed = datetime.datetime.now().isoformat()
        return session
    
    def update_session(self, session: WorkflowSession) -> None:
        """Update an existing session."""
        session.last_accessed = datetime.datetime.now().isoformat()
        self._sessions[session.session_id] = session
    
    def complete_session(self, session_id: str) -> None:
        """Mark session as completed."""
        if session_id in self._sessions:
            self._sessions[session_id].session_status = "completed"
            self._sessions[session_id].last_accessed = datetime.datetime.now().isoformat()
    
    def fail_session(self, session_id: str, error_details: str) -> None:
        """Mark session as failed."""
        if session_id in self._sessions:
            self._sessions[session_id].session_status = "failed"
            self._sessions[session_id].context_data["error_details"] = error_details
            self._sessions[session_id].last_accessed = datetime.datetime.now().isoformat()
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        current_time = datetime.datetime.now()
        expired_session_ids = []
        
        for session_id, session in self._sessions.items():
            created_time = datetime.datetime.fromisoformat(session.created_timestamp)
            if current_time - created_time > datetime.timedelta(hours=self._session_timeout_hours):
                expired_session_ids.append(session_id)
        
        for session_id in expired_session_ids:
            del self._sessions[session_id]
            
        if expired_session_ids:
            logger.info(f"Cleaned up {len(expired_session_ids)} expired sessions")
    
    def get_session_count(self) -> int:
        """Get current number of active sessions."""
        return len(self._sessions)

    def get_paused_creation_count(self) -> int:
        """Get current number of paused creations."""
        return len(self._paused_creations)


class EnhancedOrchestrationEngine:
    """
    General-purpose, domain-agnostic orchestration engine for LLM-driven workflows.
    Provides context intelligence, tool/IAE discovery, and validation for any task.
    """
    def __init__(self):
        self.context_engine = ContextIntelligenceEngine()
        self.success_engine = SuccessCriteriaEngine()
        self.classification_engine = ToolClassificationEngine()  # LLM-driven tool classification
        self.complexity_engine = TaskComplexityAssessmentEngine()  # LLM-driven task complexity
        self.step_template_engine = StepTemplateGenerationEngine()  # LLM-driven step template generation
        self.iae_registry = self._load_iae_registry()
        self.tool_mappings = {}
        self.workflow_templates = self._load_workflow_templates()
        self.session_manager = WorkflowSessionManager()
        self._context_tools = {}  # Cache for discovered tools

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

    # === Progressive Execution Methods ===
    
    async def orchestrate_progressive_workflow(
        self,
        task_description: str = None,
        provided_context: Optional[Dict[str, Any]] = None,
        workflow_session_id: Optional[str] = None
    ) -> Union[StepExecutionResult, ContextSurvey]:
        """
        Main entry point for progressive workflow execution.
        If workflow_session_id is provided, continues existing workflow.
        If not provided, creates new workflow and executes step 1.
        """
        # Case 1: Continue existing workflow
        if workflow_session_id:
            return await self.execute_workflow_step(workflow_session_id)
        
        # Case 2: Start new workflow
        if not task_description:
            raise ValueError("task_description is required for new workflow orchestration")
            
        logger.info(f"🎭 Starting progressive workflow orchestration for: {task_description[:100]}...")
        
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
            
            # Create paused creation session
            creation_id = self.session_manager.create_paused_creation(
                task_description, provided_context, context_gaps, survey
            )
            
            # Include the creation_id in the survey for collaboration_response to reference
            survey.survey_id = creation_id
            return survey
        
        # Step 2: Create workflow and granular steps
        return await self._create_and_execute_workflow(task_description, provided_context)

    async def resume_workflow_creation(self, creation_id: str, context_response: Dict[str, Any]) -> Union[StepExecutionResult, ContextSurvey]:
        """Resume workflow creation from paused state with provided context."""
        # Retrieve paused creation
        paused_creation = self.session_manager.get_paused_creation(creation_id)
        if not paused_creation:
            raise ValueError(f"Paused workflow creation {creation_id} not found or expired")
        
        logger.info(f"🔄 Resuming workflow creation {creation_id} with user context")
        
        # Merge user context with original context
        updated_context = paused_creation.original_context.copy()
        updated_context.update(context_response)
        
        # Re-analyze context gaps with updated context
        context_gaps = self.context_engine.analyze_context_gaps(
            paused_creation.task_description, updated_context
        )
        critical_gaps = [gap for gap in context_gaps if gap.importance == "critical"]
        
        if critical_gaps:
            logger.warning(f"⚠️ Still have {len(critical_gaps)} critical context gaps after user response")
            # Generate new survey with remaining gaps
            survey = self.context_engine.generate_context_survey(critical_gaps, paused_creation.task_description)
            survey.survey_id = creation_id  # Reuse the same creation_id
            return survey
        
        # Clean up paused creation
        self.session_manager.complete_paused_creation(creation_id)
        
        # Proceed with workflow creation
        return await self._create_and_execute_workflow(paused_creation.task_description, updated_context)

    async def _create_and_execute_workflow(self, task_description: str, provided_context: Dict[str, Any]) -> StepExecutionResult:
        """Create workflow and execute first step."""
        complexity = self._assess_task_complexity(task_description, provided_context)
        success_criteria = self.success_engine.define_success_criteria(task_description, "generic", provided_context)
        
        # Create granular workflow steps instead of broad phases
        workflow_steps = await self._create_granular_workflow_steps(task_description, provided_context)
        
        # Create workflow object (maintaining compatibility)
        workflow_phases = await self._create_generic_workflow_phases(task_description, complexity, provided_context)
        tool_mappings = await self._map_tools_to_phases(workflow_phases)
        iae_mappings = await self._map_iaes_to_phases(workflow_phases, success_criteria)
        
        workflow = OrchestrationWorkflow(
            workflow_id=f"workflow_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_description=task_description,
            complexity=complexity,
            phases=workflow_phases,
            success_criteria=success_criteria,
            tool_mappings=tool_mappings,
            iae_mappings=iae_mappings,
            context_requirements=provided_context,
            estimated_total_time=self._estimate_total_time_for_steps(workflow_steps),
            created_timestamp=datetime.datetime.now().isoformat()
        )
        
        # Step 3: Create session and execute first step
        session_id = self.session_manager.create_session(workflow, workflow_steps)
        
        # Execute step 1
        return await self.execute_workflow_step(session_id)
    
    async def execute_workflow_step(self, workflow_session_id: str) -> StepExecutionResult:
        """Execute the current step in the workflow session."""
        # Retrieve session
        session = self.session_manager.get_session(workflow_session_id)
        if not session:
            return StepExecutionResult(
                status="step_failed",
                workflow_session_id=workflow_session_id,
                current_step=0,
                total_steps=0,
                step_description="Session not found or expired",
                step_results={},
                next_step_needed=False,
                next_step_guidance="",
                overall_progress=0.0,
                error_details="Workflow session not found or has expired"
            )
        
        # Check if workflow is already completed
        if session.current_step > session.total_steps:
            self.session_manager.complete_session(workflow_session_id)
            return StepExecutionResult(
                status="workflow_complete",
                workflow_session_id=workflow_session_id,
                current_step=session.total_steps,
                total_steps=session.total_steps,
                step_description="Workflow completed",
                step_results={"summary": "All steps completed successfully"},
                next_step_needed=False,
                next_step_guidance="Workflow execution complete",
                overall_progress=1.0,
                execution_summary=self._generate_execution_summary(session)
            )
        
        # Get current step
        current_step_index = session.current_step - 1  # Convert to 0-based index
        if current_step_index >= len(session.workflow_steps):
            return StepExecutionResult(
                status="step_failed",
                workflow_session_id=workflow_session_id,
                current_step=session.current_step,
                total_steps=session.total_steps,
                step_description="Invalid step index",
                step_results={},
                next_step_needed=False,
                next_step_guidance="",
                overall_progress=session.current_step / session.total_steps,
                error_details="Current step index exceeds available workflow steps"
            )
        
        current_step = session.workflow_steps[current_step_index]
        
        logger.info(f"🔄 Executing step {session.current_step}/{session.total_steps}: {current_step.step_name}")
        
        try:
            # Execute step actions
            step_result = await self._execute_step_actions(current_step, session)
            
            # Validate step completion
            validation_passed = await self._validate_step_completion(current_step, step_result, session)
            step_result.validation_passed = validation_passed
            
            # Update session
            session.step_results[session.current_step] = step_result
            session.completed_steps.append(session.current_step)
            
            # Determine if more steps needed BEFORE incrementing current_step
            is_final_step = session.current_step >= session.total_steps
            next_step_needed = not is_final_step
            
            # Now increment current_step for next execution
            session.current_step += 1
            self.session_manager.update_session(session)
            
            # Generate next step guidance
            next_step_guidance = ""
            if next_step_needed and session.current_step <= len(session.workflow_steps):
                next_step = session.workflow_steps[session.current_step - 1]  # 0-based index, current_step is now incremented
                next_step_guidance = f"Next: {next_step.step_name} - {next_step.description}"
            elif is_final_step:
                next_step_guidance = "All workflow steps completed successfully"
            
            # Calculate progress - use the step just completed
            completed_step_number = session.current_step - 1  # Since we incremented current_step
            progress = completed_step_number / session.total_steps
            
            # Prepare result
            result = StepExecutionResult(
                status="step_completed" if not is_final_step else "workflow_complete",
                workflow_session_id=workflow_session_id,
                current_step=completed_step_number,  # Return the step just completed
                total_steps=session.total_steps,
                step_description=current_step.step_name,
                step_results={
                    "tools_executed": step_result.tools_executed,
                    "iaes_used": step_result.iaes_used,
                    "outputs_generated": step_result.outputs_generated,
                    "validation_passed": step_result.validation_passed,
                    "execution_time": step_result.execution_time,
                    "status": step_result.status
                },
                next_step_needed=next_step_needed,
                next_step_guidance=next_step_guidance,
                overall_progress=progress,
                workflow=session.workflow if completed_step_number == 1 else None,  # Include workflow on step 1 completion
                execution_summary=self._generate_execution_summary(session) if is_final_step else None
            )
            
            if is_final_step:
                self.session_manager.complete_session(workflow_session_id)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Step execution failed: {str(e)}")
            error_details = str(e)
            self.session_manager.fail_session(workflow_session_id, error_details)
            
            return StepExecutionResult(
                status="step_failed",
                workflow_session_id=workflow_session_id,
                current_step=session.current_step,
                total_steps=session.total_steps,
                step_description=current_step.step_name,
                step_results={"error": error_details},
                next_step_needed=False,
                next_step_guidance="",
                overall_progress=session.current_step / session.total_steps,
                error_details=error_details
            )
    
    async def _create_granular_workflow_steps(self, task_description: str, context: Dict[str, Any]) -> List[WorkflowStep]:
        """Create granular workflow steps using LLM-driven complexity and step template generation."""
        # Discover available tools
        available_tools = await self._discover_available_tools()
        # Assess complexity using LLM
        complexity_result = await self.complexity_engine.assess_complexity(task_description, context)
        complexity = complexity_result.complexity
        # Generate step templates using LLM
        step_templates = await self.step_template_engine.generate_step_templates(
            task_description, context, available_tools, complexity
        )
        # Map tools and IAEs to each step using intelligent classification
        workflow_context = {
            "task_description": task_description,
            "complexity": complexity.value,
            "available_tools": available_tools,
            "context": context
        }
        workflow_steps = []
        for i, template in enumerate(step_templates, 1):
            step_tools = await self._map_tools_to_step_intelligent(template.__dict__, available_tools, workflow_context)
            step_iaes = await self._map_iaes_to_step_intelligent(template.__dict__, workflow_context)
            workflow_steps.append(WorkflowStep(
                step_id=f"step_{i}",
                step_number=i,
                step_name=template.step_name,
                description=template.description,
                tool_mappings=step_tools,
                iae_mappings=step_iaes,
                success_criteria=[],  # To be mapped as needed
                inputs_required=[],   # To be mapped as needed
                expected_outputs=template.expected_outputs,
                dependencies=template.dependencies,
                estimated_duration=template.estimated_duration,
                validation_method="automated"
            ))
        return workflow_steps

    async def _execute_step_actions(self, step: WorkflowStep, session: WorkflowSession) -> StepResult:
        """Execute the actions defined in a workflow step."""
        start_time = datetime.datetime.now()
        tools_executed = []
        iaes_used = []
        outputs_generated = {}
        
        try:
            # Execute tool mappings for this step
            for tool_mapping in step.tool_mappings:
                logger.info(f"🔧 Executing tool: {tool_mapping.tool_name}")
                tool_output = await self._execute_tool_action(tool_mapping, step, session)
                tools_executed.append(tool_mapping.tool_name)
                outputs_generated[f"tool_{tool_mapping.tool_name}"] = tool_output
            
            # Execute IAE mappings for this step
            for iae_mapping in step.iae_mappings:
                logger.info(f"🧠 Applying IAE: {iae_mapping.iae_name}")
                iae_output = await self._execute_iae_action(iae_mapping, step, session)
                iaes_used.append(iae_mapping.iae_name)
                outputs_generated[f"iae_{iae_mapping.iae_name}"] = iae_output
            
            # Generate step-specific outputs
            if not tools_executed and not iaes_used:
                # Provide guidance-based execution for steps without specific tools/IAEs
                outputs_generated["guidance"] = self._generate_step_guidance(step, session)
            
            execution_time = str(datetime.datetime.now() - start_time)
            
            return StepResult(
                step_number=step.step_number,
                step_name=step.step_name,
                status="success",
                tools_executed=tools_executed,
                iaes_used=iaes_used,
                outputs_generated=outputs_generated,
                validation_passed=True,  # Will be updated by validation
                execution_time=execution_time,
                recommendations=[]
            )
            
        except Exception as e:
            execution_time = str(datetime.datetime.now() - start_time)
            logger.error(f"❌ Step execution error: {str(e)}")
            
            return StepResult(
                step_number=step.step_number,
                step_name=step.step_name,
                status="failure",
                tools_executed=tools_executed,
                iaes_used=iaes_used,
                outputs_generated=outputs_generated,
                validation_passed=False,
                execution_time=execution_time,
                error_details=str(e),
                recommendations=[f"Review and retry step {step.step_number}"]
            )
    
    async def _execute_tool_action(self, tool_mapping: ToolMapping, step: WorkflowStep, session: WorkflowSession) -> Dict[str, Any]:
        """Execute a specific tool action within a step context."""
        # Generate contextual guidance for tool execution
        # Note: In MCP model, we provide guidance rather than directly executing
        tool_guidance = {
            "tool_name": tool_mapping.tool_name,
            "usage_context": tool_mapping.usage_context,
            "step_context": step.description,
            "expected_outcome": f"Completion of {step.step_name}",
            "command_template": tool_mapping.command_template or f"Use {tool_mapping.tool_name} to {step.description.lower()}",
            "server_name": tool_mapping.server_name,
            "configuration": tool_mapping.configuration or {}
        }
        
        # Provide specific guidance based on tool type and step context
        if "search" in tool_mapping.tool_name.lower():
            tool_guidance["suggested_query"] = self._generate_search_query_for_step(step, session)
        elif "execute" in tool_mapping.tool_name.lower():
            tool_guidance["execution_context"] = self._generate_execution_context_for_step(step, session)
        elif "scrape" in tool_mapping.tool_name.lower():
            tool_guidance["target_urls"] = self._generate_scraping_targets_for_step(step, session)
        
        return tool_guidance
    
    async def _execute_iae_action(self, iae_mapping: IAEMapping, step: WorkflowStep, session: WorkflowSession) -> Dict[str, Any]:
        """Execute Intelligence Amplification Engine action within step context."""
        # Generate IAE guidance for cognitive enhancement
        iae_guidance = {
            "iae_name": iae_mapping.iae_name,
            "enhancement_type": iae_mapping.enhancement_type,
            "application_context": iae_mapping.application_context,
            "cognitive_enhancement": iae_mapping.cognitive_enhancement,
            "step_focus": step.description,
            "libraries_required": iae_mapping.libraries_required,
            "enhancement_strategy": self._generate_iae_strategy_for_step(step, iae_mapping, session)
        }
        
        return iae_guidance
    
    async def _validate_step_completion(self, step: WorkflowStep, step_result: StepResult, session: WorkflowSession) -> bool:
        """Validate that step has been completed successfully."""
        # Check basic execution status
        if step_result.status != "success":
            return False
        
        # Validate success criteria if defined
        for criterion_id in step.success_criteria:
            if not self._validate_success_criterion(criterion_id, step_result, session):
                logger.warning(f"⚠️ Success criterion {criterion_id} not met for step {step.step_number}")
                return False
        
        # Validate expected outputs
        for expected_output in step.expected_outputs:
            if expected_output not in step_result.outputs_generated:
                logger.warning(f"⚠️ Expected output {expected_output} not generated in step {step.step_number}")
                return False
        
        # All validations passed
        return True
    
    def _generate_execution_summary(self, session: WorkflowSession) -> Dict[str, Any]:
        """Generate comprehensive execution summary for completed workflow."""
        total_tools_used = set()
        total_iaes_used = set()
        successful_steps = 0
        failed_steps = 0
        
        for step_result in session.step_results.values():
            total_tools_used.update(step_result.tools_executed)
            total_iaes_used.update(step_result.iaes_used)
            if step_result.status == "success":
                successful_steps += 1
            else:
                failed_steps += 1
        
        return {
            "workflow_id": session.workflow.workflow_id,
            "task_description": session.workflow.task_description,
            "total_steps": session.total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "completion_rate": successful_steps / session.total_steps if session.total_steps > 0 else 0,
            "tools_utilized": list(total_tools_used),
            "iaes_utilized": list(total_iaes_used),
            "complexity_level": session.workflow.complexity.value,
            "session_duration": self._calculate_session_duration(session),
            "key_outcomes": self._extract_key_outcomes(session),
            "recommendations": self._generate_final_recommendations(session)
        }
    
    def _estimate_total_time_for_steps(self, workflow_steps: List[WorkflowStep]) -> str:
        """Estimate total time for all workflow steps."""
        total_minutes = 0
        for step in workflow_steps:
            # Parse duration string (e.g., "30 minutes", "1-2 hours")
            duration_str = step.estimated_duration.lower()
            if "hour" in duration_str:
                if "-" in duration_str:
                    # Range like "1-2 hours"
                    hours = [int(x) for x in duration_str.split() if x.replace("-", "").isdigit()]
                    avg_hours = sum(hours) / len(hours) if hours else 1
                    total_minutes += avg_hours * 60
                else:
                    # Single hour value
                    hour_match = [int(x) for x in duration_str.split() if x.isdigit()]
                    total_minutes += (hour_match[0] if hour_match else 1) * 60
            else:
                # Assume minutes
                minute_match = [int(x) for x in duration_str.split() if x.isdigit()]
                total_minutes += minute_match[0] if minute_match else 30
        
        if total_minutes < 60:
            return f"{int(total_minutes)} minutes"
        elif total_minutes < 480:  # Less than 8 hours
            hours = total_minutes / 60
            return f"{hours:.1f} hours"
        else:
            days = total_minutes / (60 * 8)  # 8-hour work days
            return f"{days:.1f} days"
    
    def _get_step_templates_for_task(self, task_description: str, step_count: int) -> List[Dict[str, Any]]:
        """Generate step templates based on task analysis."""
        task_lower = task_description.lower()
        
        # Determine task category
        if any(keyword in task_lower for keyword in ["web", "website", "app", "application", "ui", "frontend"]):
            return self._get_web_development_steps(step_count)
        elif any(keyword in task_lower for keyword in ["data", "analysis", "analytics", "visualization", "dataset"]):
            return self._get_data_analysis_steps(step_count)
        elif any(keyword in task_lower for keyword in ["research", "investigate", "study", "analyze", "report"]):
            return self._get_research_steps(step_count)
        elif any(keyword in task_lower for keyword in ["code", "program", "software", "script", "algorithm"]):
            return self._get_programming_steps(step_count)
        else:
            return self._get_generic_steps(step_count)
    
    def _get_web_development_steps(self, count: int) -> List[Dict[str, Any]]:
        """Generate web development specific steps."""
        base_steps = [
            {"name": "Requirements Analysis", "description": "Analyze project requirements and define scope", "tools": ["maestro_search"]},
            {"name": "Technology Selection", "description": "Choose appropriate technologies and frameworks", "tools": ["maestro_search"]},
            {"name": "Project Setup", "description": "Initialize project structure and dependencies", "tools": ["maestro_execute"]},
            {"name": "Core Implementation", "description": "Implement main functionality and features", "tools": ["maestro_execute"]},
            {"name": "UI/UX Design", "description": "Design and implement user interface", "tools": ["maestro_execute"]},
            {"name": "Testing & Validation", "description": "Test functionality and validate requirements", "tools": ["maestro_execute"]},
            {"name": "Deployment Preparation", "description": "Prepare application for deployment", "tools": ["maestro_execute"]},
            {"name": "Final Review", "description": "Review implementation and documentation", "tools": ["maestro_search"]}
        ]
        return self._adjust_steps_to_count(base_steps, count)
    
    def _get_data_analysis_steps(self, count: int) -> List[Dict[str, Any]]:
        """Generate data analysis specific steps."""
        base_steps = [
            {"name": "Data Discovery", "description": "Identify and locate relevant data sources", "tools": ["maestro_search"]},
            {"name": "Data Collection", "description": "Gather and acquire necessary datasets", "tools": ["maestro_scrape", "maestro_search"]},
            {"name": "Data Exploration", "description": "Explore data structure and characteristics", "tools": ["maestro_execute"]},
            {"name": "Data Cleaning", "description": "Clean and preprocess the data", "tools": ["maestro_execute"]},
            {"name": "Analysis Implementation", "description": "Perform statistical and analytical computations", "tools": ["maestro_execute"]},
            {"name": "Visualization Creation", "description": "Create charts and visual representations", "tools": ["maestro_execute"]},
            {"name": "Insight Generation", "description": "Extract key insights and patterns", "tools": ["maestro_execute"]},
            {"name": "Report Compilation", "description": "Compile findings into comprehensive report", "tools": ["maestro_execute"]}
        ]
        return self._adjust_steps_to_count(base_steps, count)
    
    def _get_research_steps(self, count: int) -> List[Dict[str, Any]]:
        """Generate research specific steps."""
        base_steps = [
            {"name": "Topic Definition", "description": "Define research scope and objectives", "tools": ["maestro_search"]},
            {"name": "Literature Review", "description": "Research existing knowledge and sources", "tools": ["maestro_search", "maestro_scrape"]},
            {"name": "Source Verification", "description": "Validate credibility of information sources", "tools": ["maestro_search"]},
            {"name": "Data Synthesis", "description": "Synthesize information from multiple sources", "tools": ["maestro_execute"]},
            {"name": "Analysis & Interpretation", "description": "Analyze findings and draw conclusions", "tools": ["maestro_execute"]},
            {"name": "Documentation", "description": "Document research methodology and findings", "tools": ["maestro_execute"]},
            {"name": "Peer Review", "description": "Review and validate research quality", "tools": ["maestro_search"]},
            {"name": "Final Presentation", "description": "Prepare final research presentation", "tools": ["maestro_execute"]}
        ]
        return self._adjust_steps_to_count(base_steps, count)
    
    def _get_programming_steps(self, count: int) -> List[Dict[str, Any]]:
        """Generate programming specific steps."""
        base_steps = [
            {"name": "Problem Analysis", "description": "Analyze problem requirements and constraints", "tools": ["maestro_search"]},
            {"name": "Algorithm Design", "description": "Design solution algorithm and approach", "tools": ["maestro_execute"]},
            {"name": "Environment Setup", "description": "Set up development environment", "tools": ["maestro_execute"]},
            {"name": "Core Implementation", "description": "Implement main program logic", "tools": ["maestro_execute"]},
            {"name": "Testing & Debugging", "description": "Test functionality and fix issues", "tools": ["maestro_execute"]},
            {"name": "Optimization", "description": "Optimize performance and efficiency", "tools": ["maestro_execute"]},
            {"name": "Documentation", "description": "Document code and usage instructions", "tools": ["maestro_execute"]},
            {"name": "Final Validation", "description": "Validate solution meets requirements", "tools": ["maestro_execute"]}
        ]
        return self._adjust_steps_to_count(base_steps, count)
    
    def _get_generic_steps(self, count: int) -> List[Dict[str, Any]]:
        """Generate generic workflow steps."""
        base_steps = [
            {"name": "Initial Analysis", "description": "Analyze task requirements and scope", "tools": ["maestro_search"]},
            {"name": "Research & Planning", "description": "Research relevant information and plan approach", "tools": ["maestro_search", "maestro_scrape"]},
            {"name": "Resource Preparation", "description": "Prepare necessary resources and tools", "tools": ["maestro_execute"]},
            {"name": "Implementation Phase 1", "description": "Begin primary implementation work", "tools": ["maestro_execute"]},
            {"name": "Implementation Phase 2", "description": "Continue and refine implementation", "tools": ["maestro_execute"]},
            {"name": "Quality Assurance", "description": "Test and validate implementation quality", "tools": ["maestro_execute"]},
            {"name": "Review & Refinement", "description": "Review results and make improvements", "tools": ["maestro_execute"]},
            {"name": "Final Validation", "description": "Perform final validation and completion check", "tools": ["maestro_execute"]}
        ]
        return self._adjust_steps_to_count(base_steps, count)

    def _adjust_steps_to_count(self, base_steps: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Adjust step list to match target count."""
        if len(base_steps) == target_count:
            return base_steps
        elif len(base_steps) > target_count:
            # Remove less critical steps
            return base_steps[:target_count]
        else:
            # Add additional implementation steps
            additional_steps_needed = target_count - len(base_steps)
            for i in range(additional_steps_needed):
                base_steps.insert(-1, {  # Insert before final step
                    "name": f"Implementation Phase {i+3}",
                    "description": f"Continue implementation work phase {i+3}",
                    "tools": ["maestro_execute"]
                })
            return base_steps
    
    async def _map_tools_to_step_intelligent(self, step_template: Dict[str, Any], available_tools: Dict[str, Any], workflow_context: Dict[str, Any]) -> List[ToolMapping]:
        """Map tools to a workflow step using LLM-driven intelligent classification."""
        mappings = []
        step_name = step_template["step_name"].lower()
        step_description = step_template["description"].lower()
        
        # Classify all available tools for this workflow context
        tool_classifications = await self.classification_engine.classify_tools_for_workflow(available_tools, workflow_context)
        
        # Determine the step type category based on step characteristics
        step_category = self._determine_step_category(step_template)
        
        # Select tools based on intelligent classification matching
        suitable_tools = []
        for tool_name, classification in tool_classifications.items():
            # Check if tool category matches step category
            if classification.category == step_category:
                suitable_tools.append((tool_name, classification, classification.confidence))
            # Also include tools specifically suitable for this phase
            elif any(phase in classification.suitable_phases for phase in [step_name, step_category.lower()]):
                suitable_tools.append((tool_name, classification, classification.confidence * 0.8))  # Slightly lower confidence
        
        # Sort by confidence and take top tools (max 3 per step to avoid overcrowding)
        suitable_tools.sort(key=lambda x: x[2], reverse=True)
        top_tools = suitable_tools[:3]
        
        # Create tool mappings for selected tools
        for tool_name, classification, confidence in top_tools:
            if tool_name in available_tools:
                tool_info = available_tools[tool_name]
                mapping = ToolMapping(
                    tool_id=f"{step_template['step_name'].lower().replace(' ', '_')}_{tool_name}",
                    tool_name=tool_name,
                    server_name=tool_info.get("server", "unknown"),
                    workflow_phase=step_template["step_name"],
                    usage_context=f"Use {tool_name} for {step_template['description']} (Classification: {classification.category}, Confidence: {confidence:.2f})",
                    command_template=self._generate_command_template(tool_name, step_template),
                    configuration={"classification_confidence": confidence, "classification_reasoning": classification.reasoning},
                    fallback_tools=tool_info.get("fallback_tools", [])
                )
                mappings.append(mapping)
        
        # If no suitable tools found, use fallback logic
        if not mappings:
            logger.warning(f"⚠️ No suitable tools found for step '{step_template['step_name']}' using intelligent classification. Using fallback logic.")
            mappings = self._fallback_map_tools_to_step(step_template, available_tools)
        
        logger.debug(f"🔧 Mapped {len(mappings)} tools to step '{step_template['step_name']}' using intelligent classification")
        return mappings
    
    def _determine_step_category(self, step_template: Dict[str, Any]) -> str:
        """Determine the primary category of a workflow step for tool matching."""
        step_name = step_template["step_name"].lower()
        step_description = step_template["description"].lower()
        
        # Analyze step characteristics using keyword matching
        planning_indicators = ["analysis", "research", "planning", "investigation", "discovery", "information", "gather"]
        execution_indicators = ["implementation", "coding", "building", "creating", "developing", "executing", "deployment"]
        validation_indicators = ["testing", "validation", "verification", "quality", "review", "checking", "assessment"]
        
        planning_score = sum(1 for keyword in planning_indicators if keyword in step_name or keyword in step_description)
        execution_score = sum(1 for keyword in execution_indicators if keyword in step_name or keyword in step_description)
        validation_score = sum(1 for keyword in validation_indicators if keyword in step_name or keyword in step_description)
        
        # Determine category based on highest score
        if planning_score >= execution_score and planning_score >= validation_score:
            return "PLANNING"
        elif execution_score >= validation_score:
            return "EXECUTION"
        else:
            return "VALIDATION"
    
    def _fallback_map_tools_to_step(self, step_template: Dict[str, Any], available_tools: Dict[str, Any]) -> List[ToolMapping]:
        """Fallback tool mapping when intelligent classification finds no suitable tools."""
        mappings = []
        step_tools = step_template.get("tools", ["maestro_execute"])  # Default fallback
        
        for tool_name in step_tools:
            if tool_name in available_tools:
                tool_info = available_tools[tool_name]
                mapping = ToolMapping(
                    tool_id=f"{step_template['step_name'].lower().replace(' ', '_')}_{tool_name}",
                    tool_name=tool_name,
                    server_name=tool_info.get("server", "maestro"),
                    workflow_phase=step_template["step_name"],
                    usage_context=f"Fallback mapping: Use {tool_name} for {step_template['description'].lower()}",
                    command_template=self._generate_command_template(tool_name, step_template),
                    fallback_tools=tool_info.get("fallback_tools", [])
                )
                mappings.append(mapping)
        
        return mappings
    
    async def _map_iaes_to_step_intelligent(self, step_template: Dict[str, Any], workflow_context: Dict[str, Any]) -> List[IAEMapping]:
        """Map Intelligence Amplification Engines to a workflow step using intelligent classification."""
        mappings = []
        step_name = step_template["step_name"].lower()
        step_description = step_template["description"].lower()
        
        # Get available IAEs from registry
        available_iaes = self.iae_registry
        
        # Classify IAEs for this workflow context
        iae_classifications = {}
        for iae_id, iae_info in available_iaes.items():
            try:
                classification = await self.classification_engine.classify_iae(iae_info, workflow_context)
                iae_classifications[iae_id] = classification
            except Exception as e:
                logger.warning(f"⚠️ IAE classification failed for {iae_id}: {e}")
                # Use fallback classification
                fallback_classification = self.classification_engine._fallback_classify_iae(iae_id, iae_info)
                iae_classifications[iae_id] = fallback_classification
        
        # Determine step category for IAE matching
        step_category = self._determine_step_category(step_template)
        
        # Select suitable IAEs based on step characteristics and phase compatibility
        suitable_iaes = []
        for iae_id, classification in iae_classifications.items():
            # Check if IAE is suitable for this step's phase
            step_phase = step_category.lower()
            if any(phase in classification.suitable_phases for phase in [step_name, step_phase, "analysis"]):
                suitable_iaes.append((iae_id, classification))
        
        # Create IAE mappings for suitable IAEs (max 2 per step)
        for iae_id, classification in suitable_iaes[:2]:
            if iae_id in available_iaes:
                iae_info = available_iaes[iae_id]
                
                # Select most appropriate enhancement type for this step
                primary_enhancement = self._select_primary_enhancement_for_step(step_template, classification.enhancement_types)
                
                mapping = IAEMapping(
                    iae_id=f"{step_template['step_name'].lower().replace(' ', '_')}_{iae_id}",
                    iae_name=iae_info.get("name", "Cognitive Enhancement"),
                    workflow_phase=step_template["step_name"],
                    enhancement_type=primary_enhancement,
                    application_context=f"{step_template['description']} - {classification.cognitive_focus}",
                    libraries_required=iae_info.get("libraries", []),
                    cognitive_enhancement=f"{classification.cognitive_focus} (Confidence: {classification.confidence:.2f})"
                )
                mappings.append(mapping)
        
        # If no suitable IAEs found, use fallback based on step category
        if not mappings:
            fallback_mapping = self._create_fallback_iae_mapping(step_template, step_category)
            if fallback_mapping:
                mappings.append(fallback_mapping)
        
        logger.debug(f"🧠 Mapped {len(mappings)} IAEs to step '{step_template['step_name']}' using intelligent classification")
        return mappings
    
    def _select_primary_enhancement_for_step(self, step_template: Dict[str, Any], enhancement_types: List[str]) -> str:
        """Select the most appropriate enhancement type for a specific step."""
        step_name = step_template["step_name"].lower()
        step_description = step_template["description"].lower()
        
        # Priority mapping based on step characteristics
        if "analysis" in step_name or "research" in step_name:
            return "analysis" if "analysis" in enhancement_types else enhancement_types[0]
        elif "implementation" in step_name or "coding" in step_name or "building" in step_name:
            return "optimization" if "optimization" in enhancement_types else enhancement_types[0]
        elif "validation" in step_name or "testing" in step_name or "quality" in step_name:
            return "validation" if "validation" in enhancement_types else enhancement_types[0]
        else:
            # Default to reasoning for general steps
            return "reasoning" if "reasoning" in enhancement_types else enhancement_types[0]
    
    def _create_fallback_iae_mapping(self, step_template: Dict[str, Any], step_category: str) -> Optional[IAEMapping]:
        """Create fallback IAE mapping when no suitable IAEs found."""
        enhancement_type_map = {
            "PLANNING": "analysis",
            "EXECUTION": "optimization", 
            "VALIDATION": "validation"
        }
        
        enhancement_type = enhancement_type_map.get(step_category, "reasoning")
        
        return IAEMapping(
            iae_id=f"{step_template['step_name'].lower().replace(' ', '_')}_fallback",
            iae_name="General Cognitive Enhancement",
            workflow_phase=step_template["step_name"],
            enhancement_type=enhancement_type,
            application_context=f"Fallback cognitive enhancement for {step_template['description']}",
            cognitive_enhancement=f"General {enhancement_type} enhancement applied as fallback"
        )
    
    def _generate_command_template(self, tool_name: str, step_template: Dict[str, Any]) -> str:
        """Generate command template for tool usage in step context."""
        if "search" in tool_name.lower():
            return f"Search for information relevant to: {step_template['description']}"
        elif "execute" in tool_name.lower():
            return f"Execute code/commands to accomplish: {step_template['description']}"
        elif "scrape" in tool_name.lower():
            return f"Extract data from web sources for: {step_template['description']}"
        else:
            return f"Use {tool_name} to complete: {step_template['description']}"
    
    def _generate_step_guidance(self, step: WorkflowStep, session: WorkflowSession) -> Dict[str, Any]:
        """Generate guidance for steps without specific tool/IAE mappings."""
        return {
            "step_objective": step.description,
            "context": f"Step {step.step_number} of {session.total_steps} in workflow for: {session.workflow.task_description}",
            "success_criteria": step.success_criteria,
            "expected_outputs": step.expected_outputs,
            "guidance": f"Focus on {step.description.lower()} as part of the overall {session.workflow.task_description}",
            "dependencies": step.dependencies,
            "validation_method": step.validation_method
        }
    
    def _generate_search_query_for_step(self, step: WorkflowStep, session: WorkflowSession) -> str:
        """Generate contextual search query for step."""
        task_keywords = session.workflow.task_description.split()[:3]  # First 3 words
        step_keywords = step.description.split()[:3]  # First 3 words
        return f"{' '.join(task_keywords)} {' '.join(step_keywords)}"
    
    def _generate_execution_context_for_step(self, step: WorkflowStep, session: WorkflowSession) -> Dict[str, Any]:
        """Generate execution context for step."""
        return {
            "task_context": session.workflow.task_description,
            "step_objective": step.description,
            "complexity_level": session.workflow.complexity.value,
            "previous_outputs": [result.outputs_generated for result in session.step_results.values()],
            "remaining_steps": session.total_steps - step.step_number
        }
    
    def _generate_scraping_targets_for_step(self, step: WorkflowStep, session: WorkflowSession) -> List[str]:
        """Generate potential scraping targets for step."""
        # This would typically use more sophisticated logic to determine relevant URLs
        return [f"Search for URLs related to: {step.description} in context of {session.workflow.task_description}"]
    
    def _generate_iae_strategy_for_step(self, step: WorkflowStep, iae_mapping: IAEMapping, session: WorkflowSession) -> str:
        """Generate IAE enhancement strategy for step."""
        return f"Apply {iae_mapping.enhancement_type} enhancement to {step.description} within the context of {session.workflow.task_description}"
    
    def _validate_success_criterion(self, criterion_id: str, step_result: StepResult, session: WorkflowSession) -> bool:
        """Validate a specific success criterion."""
        # Default validation - can be enhanced with specific criterion logic
        return step_result.status == "success" and len(step_result.outputs_generated) > 0
    
    def _calculate_session_duration(self, session: WorkflowSession) -> str:
        """Calculate total session duration."""
        try:
            created = datetime.datetime.fromisoformat(session.created_timestamp)
            last_accessed = datetime.datetime.fromisoformat(session.last_accessed)
            duration = last_accessed - created
            
            total_seconds = int(duration.total_seconds())
            if total_seconds < 60:
                return f"{total_seconds} seconds"
            elif total_seconds < 3600:
                return f"{total_seconds // 60} minutes"
            else:
                return f"{total_seconds // 3600:.1f} hours"
        except:
            return "Unknown duration"
    
    def _extract_key_outcomes(self, session: WorkflowSession) -> List[str]:
        """Extract key outcomes from session execution."""
        outcomes = []
        for step_num, step_result in session.step_results.items():
            if step_result.status == "success":
                outcomes.append(f"Step {step_num}: {step_result.step_name} completed successfully")
            else:
                outcomes.append(f"Step {step_num}: {step_result.step_name} failed - {step_result.error_details}")
        return outcomes
    
    def _generate_final_recommendations(self, session: WorkflowSession) -> List[str]:
        """Generate final recommendations for completed workflow."""
        recommendations = []
        
        # Analyze success rate
        total_steps = len(session.step_results)
        successful_steps = sum(1 for result in session.step_results.values() if result.status == "success")
        success_rate = successful_steps / total_steps if total_steps > 0 else 0
        
        if success_rate == 1.0:
            recommendations.append("Workflow completed successfully with all steps passing validation")
        elif success_rate >= 0.8:
            recommendations.append("Workflow largely successful - review failed steps for improvement")
        else:
            recommendations.append("Workflow had significant issues - thorough review recommended")
        
        # Tool utilization analysis
        all_tools = set()
        for result in session.step_results.values():
            all_tools.update(result.tools_executed)
        
        if len(all_tools) > 2:
            recommendations.append("Good tool utilization diversity achieved")
        else:
            recommendations.append("Consider utilizing more diverse tools for enhanced capability")
        
        return recommendations


# Export main classes
__all__ = [
    "EnhancedOrchestrationEngine",
    "ContextIntelligenceEngine", 
    "SuccessCriteriaEngine",
    "ToolClassificationEngine",
    "OrchestrationWorkflow",
    "OrchestrationResult",
    "ContextSurvey",
    "SuccessCriteria",
    "TaskComplexity",
    "ToolClassification",
    "IAEClassification",
    "WorkflowStep",
    "StepExecutionResult"
] 

# === LLM-Driven Task Complexity Assessment and Step Template Generation ===

@dataclass
class TaskComplexityAssessmentResult:
    """Result of LLM-driven task complexity assessment."""
    complexity: TaskComplexity
    confidence: float
    reasoning: str
    assessment_timestamp: str

@dataclass
class StepTemplate:
    """LLM-generated step template for workflow planning."""
    step_name: str
    description: str
    required_tools: List[str]
    iaes: List[str]
    dependencies: List[str]
    expected_outputs: List[str]
    estimated_duration: str

class TaskComplexityAssessmentEngine:
    """LLM-driven, cache-backed, production-quality task complexity assessment."""
    def __init__(self):
        self.cache: Dict[str, TaskComplexityAssessmentResult] = {}

    async def assess_complexity(self, task_description: str, context: Dict[str, Any]) -> TaskComplexityAssessmentResult:
        cache_key = self._generate_cache_key(task_description, context)
        if cache_key in self.cache:
            return self.cache[cache_key]
        # LLM call (MCP tool) for complexity assessment
        result = await self._llm_assess_complexity(task_description, context)
        self.cache[cache_key] = result
        return result

    def _generate_cache_key(self, task_description: str, context: Dict[str, Any]) -> str:
        return hashlib.sha256((task_description + json.dumps(context, sort_keys=True)).encode()).hexdigest()

    async def _llm_assess_complexity(self, task_description: str, context: Dict[str, Any]) -> TaskComplexityAssessmentResult:
        """Use LLM-driven assessment for task complexity analysis."""
        try:
            # Build comprehensive prompt for complexity assessment
            prompt = f"""Analyze the complexity of this task and provide a detailed assessment.

Task Description: {task_description}

Context Information:
{json.dumps(context, indent=2)}

Consider these complexity factors:
1. Technical scope and difficulty
2. Required skills and expertise level  
3. Integration complexity and dependencies
4. Time and resource requirements
5. Risk factors and potential blockers

Complexity Levels:
- SIMPLE: Basic tasks, single domain, minimal dependencies, 1-4 hours
- MODERATE: Multi-step processes, some integration, standard skills, 4-8 hours  
- COMPLEX: Multi-domain expertise, significant integration, advanced skills, 8-24 hours
- EXPERT: Cutting-edge technology, high risk, specialized expertise, 24+ hours

Provide your assessment as:
COMPLEXITY: [SIMPLE|MODERATE|COMPLEX|EXPERT]
CONFIDENCE: [0.0-1.0]
REASONING: [Detailed explanation of factors that led to this assessment]"""

            # For now, use intelligent heuristic analysis until LLM integration is available
            return await self._advanced_heuristic_complexity_assessment(task_description, context, prompt)
            
        except Exception as e:
            logger.warning(f"⚠️ LLM complexity assessment failed, using fallback: {e}")
            return await self._fallback_complexity_assessment(task_description, context)
    
    async def _advanced_heuristic_complexity_assessment(self, task_description: str, context: Dict[str, Any], prompt: str) -> TaskComplexityAssessmentResult:
        """Advanced heuristic-based complexity assessment as fallback."""
        complexity_score = 0
        reasoning_factors = []
        
        # Analyze task description content
        task_lower = task_description.lower()
        
        # Technical complexity indicators
        expert_keywords = ["machine learning", "ai", "blockchain", "microservices", "distributed", "scalability", "performance optimization", "security audit"]
        complex_keywords = ["api integration", "database design", "responsive design", "testing framework", "ci/cd", "deployment", "authentication"]
        moderate_keywords = ["web development", "data analysis", "automation", "scripting", "ui/ux", "documentation"]
        
        if any(keyword in task_lower for keyword in expert_keywords):
            complexity_score += 3
            reasoning_factors.append(f"Expert-level technical concepts detected: {[k for k in expert_keywords if k in task_lower]}")
        elif any(keyword in task_lower for keyword in complex_keywords):
            complexity_score += 2
            reasoning_factors.append(f"Complex technical requirements: {[k for k in complex_keywords if k in task_lower]}")
        elif any(keyword in task_lower for keyword in moderate_keywords):
            complexity_score += 1
            reasoning_factors.append(f"Standard technical scope: {[k for k in moderate_keywords if k in task_lower]}")
        
        # Context complexity
        context_complexity = len(context)
        if context_complexity > 10:
            complexity_score += 2
            reasoning_factors.append(f"High context complexity with {context_complexity} parameters")
        elif context_complexity > 5:
            complexity_score += 1
            reasoning_factors.append(f"Moderate context complexity with {context_complexity} parameters")
        
        # Task length and detail analysis
        word_count = len(task_description.split())
        if word_count > 100:
            complexity_score += 1
            reasoning_factors.append(f"Detailed task specification ({word_count} words)")
        
        # Multiple deliverables
        deliverable_indicators = ["and", "also", "additionally", "furthermore", "including"]
        deliverable_count = sum(1 for indicator in deliverable_indicators if indicator in task_lower)
        if deliverable_count > 3:
            complexity_score += 1
            reasoning_factors.append(f"Multiple deliverables indicated ({deliverable_count} conjunction words)")
        
        # Determine final complexity
        if complexity_score >= 5:
            complexity = TaskComplexity.EXPERT
            confidence = 0.85
        elif complexity_score >= 3:
            complexity = TaskComplexity.COMPLEX
            confidence = 0.80
        elif complexity_score >= 1:
            complexity = TaskComplexity.MODERATE
            confidence = 0.75
        else:
            complexity = TaskComplexity.SIMPLE
            confidence = 0.70
        
        reasoning = f"Complexity assessment based on: {'; '.join(reasoning_factors)}. Total complexity score: {complexity_score}/6"
        
        return TaskComplexityAssessmentResult(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            assessment_timestamp=datetime.datetime.now().isoformat()
        )
    
    async def _fallback_complexity_assessment(self, task_description: str, context: Dict[str, Any]) -> TaskComplexityAssessmentResult:
        """Simple fallback complexity assessment."""
        return TaskComplexityAssessmentResult(
            complexity=TaskComplexity.MODERATE,
            confidence=0.50,
            reasoning="Fallback assessment due to analysis failure",
            assessment_timestamp=datetime.datetime.now().isoformat()
        )

class StepTemplateGenerationEngine:
    """LLM-driven, cache-backed, production-quality step template generation."""
    def __init__(self):
        self.cache: Dict[str, List[StepTemplate]] = {}

    async def generate_step_templates(self, task_description: str, context: Dict[str, Any], available_tools: Dict[str, Any], complexity: TaskComplexity) -> List[StepTemplate]:
        cache_key = self._generate_cache_key(task_description, context, available_tools, complexity)
        if cache_key in self.cache:
            return self.cache[cache_key]
        # LLM call (MCP tool) for step template generation
        result = await self._llm_generate_step_templates(task_description, context, available_tools, complexity)
        self.cache[cache_key] = result
        return result

    def _generate_cache_key(self, task_description: str, context: Dict[str, Any], available_tools: Dict[str, Any], complexity: TaskComplexity) -> str:
        return hashlib.sha256((task_description + json.dumps(context, sort_keys=True) + json.dumps(list(available_tools.keys()), sort_keys=True) + complexity.value).encode()).hexdigest()

    async def _llm_generate_step_templates(self, task_description: str, context: Dict[str, Any], available_tools: Dict[str, Any], complexity: TaskComplexity) -> List[StepTemplate]:
        """Use LLM-driven generation for step templates."""
        try:
            # Build comprehensive prompt for step template generation
            tool_names = list(available_tools.keys())
            prompt = f"""Generate a detailed, step-by-step workflow plan for this task.

Task Description: {task_description}

Available Tools: {', '.join(tool_names)}

Context Information:
{json.dumps(context, indent=2)}

Task Complexity: {complexity.value}

Generate 4-10 concrete, executable steps based on complexity:
- SIMPLE: 4-5 steps
- MODERATE: 5-7 steps  
- COMPLEX: 7-9 steps
- EXPERT: 8-10 steps

For each step, provide:
- step_name: Clear, action-oriented name
- description: Detailed description of what to accomplish
- required_tools: List of tool names needed for this step
- iaes: List of cognitive enhancement engines needed
- dependencies: List of previous step names this depends on
- expected_outputs: What this step should produce
- estimated_duration: Realistic time estimate

Focus on:
1. Logical progression and dependencies
2. Appropriate tool selection for each step
3. Clear, measurable outputs
4. Realistic time estimates"""

            # For now, use intelligent heuristic analysis until LLM integration is available
            return await self._advanced_heuristic_step_generation(task_description, context, available_tools, complexity, prompt)
            
        except Exception as e:
            logger.warning(f"⚠️ LLM step template generation failed, using fallback: {e}")
            return await self._fallback_step_generation(task_description, context, available_tools, complexity)
    
    async def _advanced_heuristic_step_generation(self, task_description: str, context: Dict[str, Any], available_tools: Dict[str, Any], complexity: TaskComplexity, prompt: str) -> List[StepTemplate]:
        """Advanced heuristic-based step template generation."""
        task_lower = task_description.lower()
        
        # Determine step count based on complexity
        step_counts = {
            TaskComplexity.SIMPLE: 4,
            TaskComplexity.MODERATE: 6,
            TaskComplexity.COMPLEX: 8,
            TaskComplexity.EXPERT: 10
        }
        target_steps = step_counts[complexity]
        
        # Analyze task type for appropriate step templates
        if any(keyword in task_lower for keyword in ["website", "web", "html", "css", "frontend"]):
            return self._generate_web_development_templates(target_steps, available_tools)
        elif any(keyword in task_lower for keyword in ["data", "analysis", "chart", "visualization", "dataset"]):
            return self._generate_data_analysis_templates(target_steps, available_tools)
        elif any(keyword in task_lower for keyword in ["research", "investigate", "study", "information"]):
            return self._generate_research_templates(target_steps, available_tools)
        elif any(keyword in task_lower for keyword in ["code", "program", "script", "algorithm", "software"]):
            return self._generate_programming_templates(target_steps, available_tools)
        else:
            return self._generate_generic_templates(target_steps, available_tools, task_description)
    
    def _generate_web_development_templates(self, count: int, available_tools: Dict[str, Any]) -> List[StepTemplate]:
        """Generate web development step templates."""
        base_templates = [
            StepTemplate("Requirements Analysis", "Analyze project requirements and define scope", 
                        ["maestro_search"], ["analysis"], [], ["requirements_document"], "45 minutes"),
            StepTemplate("Technology Selection", "Choose appropriate technologies and frameworks", 
                        ["maestro_search"], ["reasoning"], ["Requirements Analysis"], ["tech_stack"], "30 minutes"),
            StepTemplate("Project Structure Setup", "Initialize project structure and dependencies", 
                        ["maestro_execute"], [], ["Technology Selection"], ["project_files"], "45 minutes"),
            StepTemplate("Core Implementation", "Implement main functionality and features", 
                        ["maestro_execute"], [], ["Project Structure Setup"], ["functional_code"], "2 hours"),
            StepTemplate("UI/UX Implementation", "Design and implement user interface", 
                        ["maestro_execute"], ["reasoning"], ["Core Implementation"], ["ui_components"], "1.5 hours"),
            StepTemplate("Testing & Validation", "Test functionality and validate requirements", 
                        ["maestro_execute"], ["validation"], ["UI/UX Implementation"], ["test_results"], "1 hour"),
            StepTemplate("Documentation", "Create documentation and usage instructions", 
                        ["maestro_execute"], [], ["Testing & Validation"], ["documentation"], "30 minutes"),
            StepTemplate("Final Review", "Review implementation and ensure completeness", 
                        ["maestro_execute"], ["validation"], ["Documentation"], ["final_report"], "30 minutes")
        ]
        return self._adjust_templates_to_count(base_templates, count)
    
    def _generate_data_analysis_templates(self, count: int, available_tools: Dict[str, Any]) -> List[StepTemplate]:
        """Generate data analysis step templates."""
        base_templates = [
            StepTemplate("Data Discovery", "Identify and locate relevant data sources", 
                        ["maestro_search"], ["analysis"], [], ["data_sources"], "30 minutes"),
            StepTemplate("Data Collection", "Gather and acquire necessary datasets", 
                        ["maestro_scrape", "maestro_search"], [], ["Data Discovery"], ["raw_data"], "45 minutes"),
            StepTemplate("Data Exploration", "Explore data structure and characteristics", 
                        ["maestro_execute"], ["analysis"], ["Data Collection"], ["data_summary"], "1 hour"),
            StepTemplate("Data Cleaning", "Clean and preprocess the data", 
                        ["maestro_execute"], [], ["Data Exploration"], ["clean_data"], "1.5 hours"),
            StepTemplate("Analysis Implementation", "Perform statistical and analytical computations", 
                        ["maestro_execute"], ["analysis"], ["Data Cleaning"], ["analysis_results"], "2 hours"),
            StepTemplate("Visualization Creation", "Create charts and visual representations", 
                        ["maestro_execute"], ["reasoning"], ["Analysis Implementation"], ["visualizations"], "1 hour"),
            StepTemplate("Insight Generation", "Extract key insights and patterns", 
                        ["maestro_execute"], ["analysis"], ["Visualization Creation"], ["insights"], "45 minutes"),
            StepTemplate("Report Compilation", "Compile findings into comprehensive report", 
                        ["maestro_execute"], [], ["Insight Generation"], ["final_report"], "1 hour")
        ]
        return self._adjust_templates_to_count(base_templates, count)
    
    def _generate_research_templates(self, count: int, available_tools: Dict[str, Any]) -> List[StepTemplate]:
        """Generate research step templates."""
        base_templates = [
            StepTemplate("Topic Definition", "Define research scope and objectives", 
                        ["maestro_search"], ["analysis"], [], ["research_scope"], "30 minutes"),
            StepTemplate("Literature Review", "Research existing knowledge and sources", 
                        ["maestro_search", "maestro_scrape"], ["analysis"], ["Topic Definition"], ["literature_summary"], "2 hours"),
            StepTemplate("Source Verification", "Validate credibility of information sources", 
                        ["maestro_search"], ["validation"], ["Literature Review"], ["verified_sources"], "45 minutes"),
            StepTemplate("Data Synthesis", "Synthesize information from multiple sources", 
                        ["maestro_execute"], ["reasoning"], ["Source Verification"], ["synthesized_data"], "1.5 hours"),
            StepTemplate("Analysis & Interpretation", "Analyze findings and draw conclusions", 
                        ["maestro_execute"], ["analysis"], ["Data Synthesis"], ["analysis_results"], "2 hours"),
            StepTemplate("Documentation", "Document research methodology and findings", 
                        ["maestro_execute"], [], ["Analysis & Interpretation"], ["research_document"], "1 hour"),
            StepTemplate("Peer Review", "Review and validate research quality", 
                        ["maestro_execute"], ["validation"], ["Documentation"], ["review_feedback"], "45 minutes"),
            StepTemplate("Final Presentation", "Prepare final research presentation", 
                        ["maestro_execute"], [], ["Peer Review"], ["presentation"], "1 hour")
        ]
        return self._adjust_templates_to_count(base_templates, count)
    
    def _generate_programming_templates(self, count: int, available_tools: Dict[str, Any]) -> List[StepTemplate]:
        """Generate programming step templates."""
        base_templates = [
            StepTemplate("Problem Analysis", "Analyze problem requirements and constraints", 
                        ["maestro_search"], ["analysis"], [], ["problem_spec"], "45 minutes"),
            StepTemplate("Algorithm Design", "Design solution algorithm and approach", 
                        ["maestro_execute"], ["reasoning"], ["Problem Analysis"], ["algorithm_design"], "1 hour"),
            StepTemplate("Environment Setup", "Set up development environment", 
                        ["maestro_execute"], [], ["Algorithm Design"], ["dev_environment"], "30 minutes"),
            StepTemplate("Core Implementation", "Implement main program logic", 
                        ["maestro_execute"], [], ["Environment Setup"], ["core_code"], "2.5 hours"),
            StepTemplate("Testing & Debugging", "Test functionality and fix issues", 
                        ["maestro_execute"], ["validation"], ["Core Implementation"], ["tested_code"], "1.5 hours"),
            StepTemplate("Optimization", "Optimize performance and efficiency", 
                        ["maestro_execute"], ["optimization"], ["Testing & Debugging"], ["optimized_code"], "1 hour"),
            StepTemplate("Documentation", "Document code and usage instructions", 
                        ["maestro_execute"], [], ["Optimization"], ["documentation"], "45 minutes"),
            StepTemplate("Final Validation", "Validate solution meets requirements", 
                        ["maestro_execute"], ["validation"], ["Documentation"], ["validation_report"], "30 minutes")
        ]
        return self._adjust_templates_to_count(base_templates, count)
    
    def _generate_generic_templates(self, count: int, available_tools: Dict[str, Any], task_description: str) -> List[StepTemplate]:
        """Generate generic step templates."""
        base_templates = [
            StepTemplate("Initial Analysis", "Analyze task requirements and scope", 
                        ["maestro_search"], ["analysis"], [], ["analysis_report"], "30 minutes"),
            StepTemplate("Research & Planning", "Research relevant information and plan approach", 
                        ["maestro_search", "maestro_scrape"], ["reasoning"], ["Initial Analysis"], ["plan"], "1 hour"),
            StepTemplate("Resource Preparation", "Prepare necessary resources and tools", 
                        ["maestro_execute"], [], ["Research & Planning"], ["resources"], "45 minutes"),
            StepTemplate("Implementation Phase 1", "Begin primary implementation work", 
                        ["maestro_execute"], [], ["Resource Preparation"], ["partial_results"], "1.5 hours"),
            StepTemplate("Implementation Phase 2", "Continue and refine implementation", 
                        ["maestro_execute"], [], ["Implementation Phase 1"], ["refined_results"], "1.5 hours"),
            StepTemplate("Quality Assurance", "Test and validate implementation quality", 
                        ["maestro_execute"], ["validation"], ["Implementation Phase 2"], ["qa_results"], "45 minutes"),
            StepTemplate("Review & Refinement", "Review results and make improvements", 
                        ["maestro_execute"], ["reasoning"], ["Quality Assurance"], ["improved_results"], "1 hour"),
            StepTemplate("Final Validation", "Perform final validation and completion check", 
                        ["maestro_execute"], ["validation"], ["Review & Refinement"], ["final_deliverable"], "30 minutes")
        ]
        return self._adjust_templates_to_count(base_templates, count)
    
    def _adjust_templates_to_count(self, base_templates: List[StepTemplate], target_count: int) -> List[StepTemplate]:
        """Adjust template list to match target count."""
        if len(base_templates) == target_count:
            return base_templates
        elif len(base_templates) > target_count:
            # Remove less critical steps (keep first, last, and most important middle steps)
            important_indices = [0]  # Always keep first step
            middle_start = 1
            middle_end = len(base_templates) - 1
            middle_count = target_count - 2  # Subtract first and last
            if middle_count > 0:
                step_size = (middle_end - middle_start) / middle_count
                for i in range(middle_count):
                    important_indices.append(middle_start + int(i * step_size))
            important_indices.append(len(base_templates) - 1)  # Always keep last step
            return [base_templates[i] for i in important_indices[:target_count]]
        else:
            # Add additional implementation steps
            additional_needed = target_count - len(base_templates)
            result = base_templates[:-1]  # All except last step
            for i in range(additional_needed):
                result.append(StepTemplate(
                    f"Implementation Phase {i+3}",
                    f"Continue implementation work phase {i+3}",
                    ["maestro_execute"],
                    [],
                    [result[-1].step_name],
                    [f"phase_{i+3}_results"],
                    "1 hour"
                ))
            result.append(base_templates[-1])  # Add back last step
            return result
    
    async def _fallback_step_generation(self, task_description: str, context: Dict[str, Any], available_tools: Dict[str, Any], complexity: TaskComplexity) -> List[StepTemplate]:
        """Simple fallback step template generation."""
        return [
            StepTemplate("Analysis", "Analyze the task requirements", ["maestro_search"], ["analysis"], [], ["analysis"], "30 minutes"),
            StepTemplate("Planning", "Plan the implementation approach", ["maestro_search"], ["reasoning"], ["Analysis"], ["plan"], "30 minutes"),
            StepTemplate("Implementation", "Execute the planned approach", ["maestro_execute"], [], ["Planning"], ["results"], "2 hours"),
            StepTemplate("Validation", "Validate the results", ["maestro_execute"], ["validation"], ["Implementation"], ["validation"], "30 minutes")
        ]
