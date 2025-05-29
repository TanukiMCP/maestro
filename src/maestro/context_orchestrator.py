"""
Context Orchestrator - Dynamic Context Gathering and Survey Framework
"""

import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ContextComplexity(Enum):
    """Complexity levels for context requirements"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class SurveyQuestionType(Enum):
    """Types of questions for context gathering"""
    TEXT_INPUT = "text_input"
    MULTIPLE_CHOICE = "multiple_choice"
    YES_NO = "yes_no"
    NUMERIC = "numeric"
    FILE_UPLOAD = "file_upload"
    MULTI_SELECT = "multi_select"


@dataclass
class SurveyQuestion:
    """Individual question in a context gathering survey"""
    question_id: str
    question_text: str
    question_type: SurveyQuestionType
    required: bool = True
    options: Optional[List[str]] = None
    default_value: Optional[str] = None
    validation_pattern: Optional[str] = None
    help_text: Optional[str] = None
    conditional_logic: Optional[Dict[str, Any]] = None


@dataclass
class ContextSurvey:
    """Complete context gathering survey"""
    survey_id: str
    task_description: str
    questions: List[SurveyQuestion]
    estimated_completion_time: int  # minutes
    priority_level: str = "medium"
    expires_at: Optional[datetime] = None
    context_areas: List[str] = field(default_factory=list)
    

@dataclass
class ContextAnalysis:
    """Analysis of context requirements for a task"""
    needs_additional_context: bool
    missing_context_areas: List[str]
    task_complexity: ContextComplexity
    confidence_score: float
    context_gaps: List[str]
    suggested_information: List[str]
    auto_fillable_context: Dict[str, Any] = field(default_factory=dict)


class ContextOrchestrator:
    """
    Orchestrates dynamic context gathering through intelligent survey generation.
    
    This component determines when additional context is needed and creates
    adaptive surveys to gather missing information from users in a non-intrusive way.
    """
    
    def __init__(self):
        self.active_surveys: Dict[str, ContextSurvey] = {}
        self.survey_responses: Dict[str, Dict[str, Any]] = {}
        self.context_templates = self._initialize_context_templates()
        self.tool_discovery = None
        
    def set_tool_discovery(self, tool_discovery):
        """Set the tool discovery component for enhanced context awareness"""
        self.tool_discovery = tool_discovery
        
    async def analyze_context_requirements(
        self,
        task_description: str,
        provided_context: Optional[Dict[str, Any]] = None
    ) -> ContextAnalysis:
        """
        Analyze a task to determine if additional context is needed.
        
        Args:
            task_description: The user's task description
            provided_context: Any context already provided
            
        Returns:
            Analysis of context requirements
        """
        logger.info("🔍 Analyzing context requirements...")
        
        # Determine task complexity
        task_complexity = self._assess_task_complexity(task_description)
        
        # Identify required context areas
        required_context_areas = self._identify_required_context_areas(task_description)
        
        # Check what context is missing
        provided_context = provided_context or {}
        missing_areas = []
        context_gaps = []
        
        for area in required_context_areas:
            if not self._has_sufficient_context_for_area(area, provided_context):
                missing_areas.append(area)
                gaps = self._identify_specific_gaps_for_area(area, task_description, provided_context)
                context_gaps.extend(gaps)
        
        # Determine if we can auto-fill some context
        auto_fillable = await self._identify_auto_fillable_context(
            task_description, missing_areas
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_context_confidence(
            task_complexity, len(missing_areas), len(context_gaps), provided_context
        )
        
        needs_additional = len(missing_areas) > 0 and confidence_score < 0.8
        
        return ContextAnalysis(
            needs_additional_context=needs_additional,
            missing_context_areas=missing_areas,
            task_complexity=task_complexity,
            confidence_score=confidence_score,
            context_gaps=context_gaps,
            suggested_information=self._generate_context_suggestions(missing_areas),
            auto_fillable_context=auto_fillable
        )
    
    async def generate_context_survey(
        self,
        task_description: str,
        missing_context_areas: List[str]
    ) -> ContextSurvey:
        """
        Generate an adaptive survey to gather missing context.
        
        Args:
            task_description: The original task description
            missing_context_areas: Areas where context is needed
            
        Returns:
            Complete context gathering survey
        """
        logger.info(f"📋 Generating context survey for {len(missing_context_areas)} areas...")
        
        survey_id = f"ctx_{uuid.uuid4().hex[:8]}"
        questions = []
        
        # Generate questions for each missing context area
        for area in missing_context_areas:
            area_questions = self._generate_questions_for_area(area, task_description)
            questions.extend(area_questions)
        
        # Add any conditional or follow-up questions
        questions.extend(self._generate_conditional_questions(task_description, missing_context_areas))
        
        # Estimate completion time
        estimated_time = max(2, len(questions) * 0.5)  # 30 seconds per question minimum
        
        # Set expiration (24 hours from now)
        expires_at = datetime.now() + timedelta(hours=24)
        
        survey = ContextSurvey(
            survey_id=survey_id,
            task_description=task_description,
            questions=questions,
            estimated_completion_time=int(estimated_time),
            context_areas=missing_context_areas,
            expires_at=expires_at
        )
        
        # Store survey for later retrieval
        self.active_surveys[survey_id] = survey
        
        return survey
    
    async def process_survey_responses(
        self,
        survey_id: str,
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process user responses to a context gathering survey.
        
        Args:
            survey_id: ID of the completed survey
            responses: User responses to survey questions
            
        Returns:
            Processed and validated context data
        """
        if survey_id not in self.active_surveys:
            raise ValueError(f"Survey {survey_id} not found or expired")
        
        survey = self.active_surveys[survey_id]
        
        # Validate responses
        validation_results = self._validate_survey_responses(survey, responses)
        
        if not validation_results["valid"]:
            return {
                "success": False,
                "errors": validation_results["errors"],
                "missing_required": validation_results["missing_required"]
            }
        
        # Process and structure the responses
        processed_context = self._process_responses_to_context(survey, responses)
        
        # Store responses
        self.survey_responses[survey_id] = {
            "raw_responses": responses,
            "processed_context": processed_context,
            "completed_at": datetime.now(),
            "survey": survey
        }
        
        return {
            "success": True,
            "processed_context": processed_context,
            "survey_id": survey_id
        }
    
    async def get_survey_context(self, survey_id: str) -> Dict[str, Any]:
        """Retrieve context from a completed survey"""
        if survey_id not in self.survey_responses:
            return {}
        
        return self.survey_responses[survey_id]["processed_context"]
    
    def _assess_task_complexity(self, task_description: str) -> ContextComplexity:
        """Assess the complexity level of a task"""
        task_lower = task_description.lower()
        
        # Expert level indicators
        expert_indicators = [
            "enterprise", "production", "scalable", "security", "compliance",
            "performance optimization", "architecture", "integration"
        ]
        
        # Complex level indicators
        complex_indicators = [
            "database", "api", "authentication", "responsive", "testing",
            "deployment", "workflow", "automation"
        ]
        
        # Moderate level indicators
        moderate_indicators = [
            "website", "application", "form", "interactive", "dynamic",
            "user interface", "features"
        ]
        
        if any(indicator in task_lower for indicator in expert_indicators):
            return ContextComplexity.EXPERT
        elif any(indicator in task_lower for indicator in complex_indicators):
            return ContextComplexity.COMPLEX
        elif any(indicator in task_lower for indicator in moderate_indicators):
            return ContextComplexity.MODERATE
        else:
            return ContextComplexity.SIMPLE
    
    def _identify_required_context_areas(self, task_description: str) -> List[str]:
        """Identify what context areas are required for a task"""
        task_lower = task_description.lower()
        required_areas = []
        
        # Map keywords to context areas
        context_mapping = {
            "target_audience": ["user", "client", "audience", "visitor", "customer"],
            "design_requirements": ["design", "style", "theme", "color", "layout", "branding"],
            "functionality": ["feature", "function", "capability", "behavior", "interaction"],
            "content": ["content", "text", "copy", "images", "media", "photos"],
            "technical_specs": ["technology", "framework", "database", "hosting", "platform"],
            "business_context": ["business", "company", "organization", "industry", "goals"],
            "constraints": ["budget", "timeline", "deadline", "limitation", "requirement"],
            "integration": ["integrate", "connect", "api", "service", "third-party"]
        }
        
        for area, keywords in context_mapping.items():
            if any(keyword in task_lower for keyword in keywords):
                required_areas.append(area)
        
        # Always require target audience for web projects
        if any(term in task_lower for term in ["website", "web", "portfolio", "app"]):
            if "target_audience" not in required_areas:
                required_areas.append("target_audience")
        
        return required_areas
    
    def _has_sufficient_context_for_area(
        self, 
        area: str, 
        provided_context: Dict[str, Any]
    ) -> bool:
        """Check if sufficient context is provided for an area"""
        area_requirements = {
            "target_audience": ["target_users", "audience", "demographics"],
            "design_requirements": ["design_style", "colors", "theme", "branding"],
            "functionality": ["features", "requirements", "functionality"],
            "content": ["content", "text", "copy", "media"],
            "technical_specs": ["technology", "framework", "platform"],
            "business_context": ["business_goals", "industry", "company"],
            "constraints": ["budget", "timeline", "limitations"],
            "integration": ["integrations", "apis", "services"]
        }
        
        required_keys = area_requirements.get(area, [])
        return any(key in provided_context for key in required_keys)
    
    def _identify_specific_gaps_for_area(
        self,
        area: str,
        task_description: str,
        provided_context: Dict[str, Any]
    ) -> List[str]:
        """Identify specific gaps within a context area"""
        gaps = []
        
        gap_mapping = {
            "target_audience": [
                "Who is the target audience?",
                "What are their demographics?",
                "What are their needs and preferences?"
            ],
            "design_requirements": [
                "What design style is preferred?",
                "Are there brand colors or guidelines?",
                "What's the desired visual aesthetic?"
            ],
            "functionality": [
                "What specific features are needed?",
                "How should users interact with it?",
                "What's the core functionality?"
            ],
            "content": [
                "What content needs to be included?",
                "Are there specific images or media?",
                "What's the tone and messaging?"
            ],
            "technical_specs": [
                "What technology should be used?",
                "Are there hosting preferences?",
                "What are the performance requirements?"
            ]
        }
        
        return gap_mapping.get(area, [f"Missing context for {area}"])
    
    async def _identify_auto_fillable_context(
        self,
        task_description: str,
        missing_areas: List[str]
    ) -> Dict[str, Any]:
        """Identify context that can be automatically filled"""
        auto_context = {}
        
        # Auto-fill based on task type
        task_lower = task_description.lower()
        
        if "portfolio" in task_lower:
            auto_context.update({
                "project_type": "portfolio",
                "likely_features": ["gallery", "about", "contact", "showcase"],
                "common_requirements": ["responsive design", "image optimization"]
            })
        
        if "veterinary" in task_lower or "vet" in task_lower:
            auto_context.update({
                "industry": "veterinary",
                "likely_features": ["appointment booking", "services", "contact", "emergency info"],
                "target_audience": "pet owners"
            })
        
        if "directory" in task_lower:
            auto_context.update({
                "project_type": "directory",
                "likely_features": ["search", "listings", "categories", "details pages"]
            })
        
        return auto_context
    
    def _calculate_context_confidence(
        self,
        complexity: ContextComplexity,
        missing_areas_count: int,
        gaps_count: int,
        provided_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for context completeness"""
        base_confidence = 0.5
        
        # Boost for provided context
        base_confidence += min(len(provided_context) * 0.1, 0.3)
        
        # Penalty for missing areas
        base_confidence -= missing_areas_count * 0.15
        
        # Penalty for complexity
        complexity_penalty = {
            ContextComplexity.SIMPLE: 0.0,
            ContextComplexity.MODERATE: 0.05,
            ContextComplexity.COMPLEX: 0.1,
            ContextComplexity.EXPERT: 0.15
        }
        base_confidence -= complexity_penalty[complexity]
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_context_suggestions(self, missing_areas: List[str]) -> List[str]:
        """Generate helpful suggestions for missing context"""
        suggestions = []
        
        suggestion_mapping = {
            "target_audience": "Consider providing information about your target users",
            "design_requirements": "Specify your design preferences and brand guidelines",
            "functionality": "List the key features and functionality needed",
            "content": "Provide or describe the content to be included",
            "technical_specs": "Specify technical requirements and preferences"
        }
        
        for area in missing_areas:
            if area in suggestion_mapping:
                suggestions.append(suggestion_mapping[area])
        
        return suggestions
    
    def _generate_questions_for_area(
        self,
        area: str,
        task_description: str
    ) -> List[SurveyQuestion]:
        """Generate specific questions for a context area"""
        questions = []
        
        if area == "target_audience":
            questions.extend([
                SurveyQuestion(
                    question_id=f"{area}_primary",
                    question_text="Who is your primary target audience?",
                    question_type=SurveyQuestionType.TEXT_INPUT,
                    help_text="e.g., 'Professional photographers', 'Pet owners', 'Small business owners'"
                ),
                SurveyQuestion(
                    question_id=f"{area}_demographics",
                    question_text="What are the key demographics or characteristics?",
                    question_type=SurveyQuestionType.TEXT_INPUT,
                    required=False,
                    help_text="Age range, location, interests, tech-savviness, etc."
                )
            ])
        
        elif area == "design_requirements":
            questions.extend([
                SurveyQuestion(
                    question_id=f"{area}_style",
                    question_text="What design style do you prefer?",
                    question_type=SurveyQuestionType.MULTIPLE_CHOICE,
                    options=["Modern/Minimalist", "Classic/Traditional", "Creative/Artistic", "Professional/Corporate", "Other"]
                ),
                SurveyQuestion(
                    question_id=f"{area}_colors",
                    question_text="Do you have preferred colors or brand guidelines?",
                    question_type=SurveyQuestionType.TEXT_INPUT,
                    required=False,
                    help_text="Specific color codes, brand colors, or general preferences"
                )
            ])
        
        elif area == "functionality":
            questions.extend([
                SurveyQuestion(
                    question_id=f"{area}_core_features",
                    question_text="What are the essential features you need?",
                    question_type=SurveyQuestionType.TEXT_INPUT,
                    help_text="List the key functionality and features"
                ),
                SurveyQuestion(
                    question_id=f"{area}_interactions",
                    question_text="How should users interact with your site?",
                    question_type=SurveyQuestionType.TEXT_INPUT,
                    required=False,
                    help_text="Contact forms, galleries, booking, search, etc."
                )
            ])
        
        elif area == "content":
            questions.extend([
                SurveyQuestion(
                    question_id=f"{area}_text_content",
                    question_text="Do you have existing content (text, copy) to include?",
                    question_type=SurveyQuestionType.YES_NO
                ),
                SurveyQuestion(
                    question_id=f"{area}_media",
                    question_text="Do you have images, videos, or other media?",
                    question_type=SurveyQuestionType.YES_NO
                ),
                SurveyQuestion(
                    question_id=f"{area}_tone",
                    question_text="What tone/style should the content have?",
                    question_type=SurveyQuestionType.MULTIPLE_CHOICE,
                    options=["Professional", "Friendly/Casual", "Creative/Artistic", "Technical/Informative", "Other"],
                    required=False
                )
            ])
        
        elif area == "technical_specs":
            questions.extend([
                SurveyQuestion(
                    question_id=f"{area}_framework",
                    question_text="Do you have any technology preferences?",
                    question_type=SurveyQuestionType.TEXT_INPUT,
                    required=False,
                    help_text="Preferred frameworks, CMS, hosting, etc."
                ),
                SurveyQuestion(
                    question_id=f"{area}_mobile",
                    question_text="Should it be mobile-responsive?",
                    question_type=SurveyQuestionType.YES_NO,
                    default_value="yes"
                )
            ])
        
        return questions
    
    def _generate_conditional_questions(
        self,
        task_description: str,
        missing_areas: List[str]
    ) -> List[SurveyQuestion]:
        """Generate conditional questions based on context"""
        questions = []
        
        # Add timeline question for complex projects
        if any(area in missing_areas for area in ["functionality", "technical_specs"]):
            questions.append(
                SurveyQuestion(
                    question_id="project_timeline",
                    question_text="What's your preferred timeline for completion?",
                    question_type=SurveyQuestionType.MULTIPLE_CHOICE,
                    options=["ASAP", "Within a week", "Within a month", "No rush"],
                    required=False
                )
            )
        
        return questions
    
    def _validate_survey_responses(
        self,
        survey: ContextSurvey,
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate user responses to survey questions"""
        errors = []
        missing_required = []
        
        for question in survey.questions:
            response = responses.get(question.question_id)
            
            # Check required questions
            if question.required and (response is None or response == ""):
                missing_required.append(question.question_id)
                continue
            
            # Validate response format
            if response is not None:
                if question.question_type == SurveyQuestionType.MULTIPLE_CHOICE:
                    if question.options and response not in question.options:
                        errors.append(f"Invalid option for {question.question_id}")
                
                elif question.question_type == SurveyQuestionType.YES_NO:
                    if response.lower() not in ["yes", "no", "true", "false"]:
                        errors.append(f"Invalid yes/no response for {question.question_id}")
        
        return {
            "valid": len(errors) == 0 and len(missing_required) == 0,
            "errors": errors,
            "missing_required": missing_required
        }
    
    def _process_responses_to_context(
        self,
        survey: ContextSurvey,
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process raw responses into structured context"""
        processed = {
            "survey_id": survey.survey_id,
            "original_task": survey.task_description,
            "context_areas_covered": survey.context_areas,
            "responses_processed_at": datetime.now().isoformat()
        }
        
        # Process responses by area
        for area in survey.context_areas:
            area_responses = {}
            
            for question in survey.questions:
                if question.question_id.startswith(area):
                    response = responses.get(question.question_id)
                    if response is not None:
                        # Clean up the question ID to make a nice key
                        key = question.question_id.replace(f"{area}_", "")
                        area_responses[key] = response
            
            if area_responses:
                processed[area] = area_responses
        
        # Add any global responses
        global_questions = ["project_timeline"]
        for q_id in global_questions:
            if q_id in responses:
                processed[q_id] = responses[q_id]
        
        return processed
    
    def _initialize_context_templates(self) -> Dict[str, Any]:
        """Initialize context gathering templates"""
        return {
            "web_development": {
                "required_areas": ["target_audience", "functionality", "design_requirements"],
                "optional_areas": ["content", "technical_specs", "business_context"]
            },
            "portfolio": {
                "required_areas": ["target_audience", "content", "design_requirements"],
                "optional_areas": ["functionality", "technical_specs"]
            },
            "directory": {
                "required_areas": ["target_audience", "functionality", "content"],
                "optional_areas": ["design_requirements", "technical_specs", "integration"]
            }
        } 