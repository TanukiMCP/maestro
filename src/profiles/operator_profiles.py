"""
Operator Profiles for MAESTRO Protocol
Defines specialized AI operator personas optimized for different task types and complexity levels.
Each profile includes custom system prompts, capabilities, and behavioral parameters.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class OperatorType(Enum):
    """Types of specialized operators"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    RESEARCH = "research"
    PROBLEM_SOLVING = "problem_solving"
    QUALITY_ASSURANCE = "quality_assurance"
    COMMUNICATION = "communication"
    STRATEGIC = "strategic"

class ComplexityLevel(Enum):
    """Task complexity levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class OperatorProfile:
    """Complete operator profile definition"""
    profile_id: str
    name: str
    operator_type: OperatorType
    complexity_level: ComplexityLevel
    system_prompt: str
    capabilities: List[str]
    strengths: List[str]
    limitations: List[str]
    preferred_tools: List[str]
    quality_criteria: Dict[str, float]
    behavioral_parameters: Dict[str, Any]
    success_metrics: List[str]

class OperatorProfileManager:
    """
    Manages operator profiles and provides profile selection based on task requirements
    """
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
        self.profile_mappings = self._create_profile_mappings()
    
    def _initialize_profiles(self) -> Dict[str, OperatorProfile]:
        """Initialize all operator profiles"""
        profiles = {}
        
        # Analytical Operators
        profiles["analytical_basic"] = OperatorProfile(
            profile_id="analytical_basic",
            name="Basic Analyst",
            operator_type=OperatorType.ANALYTICAL,
            complexity_level=ComplexityLevel.BASIC,
            system_prompt="""You are a Basic Analytical Operator specialized in fundamental data analysis and logical reasoning.

Your core capabilities:
- Perform basic statistical analysis and data interpretation
- Identify simple patterns and trends in data
- Apply logical reasoning to straightforward problems
- Present findings in clear, structured formats

Your approach:
- Break down problems into manageable components
- Use systematic analysis methods
- Verify conclusions with available data
- Communicate findings clearly and concisely

Quality standards:
- Accuracy in calculations and interpretations
- Clear logical flow in reasoning
- Appropriate use of analytical methods
- Actionable insights and recommendations""",
            capabilities=[
                "basic_statistics", "data_interpretation", "pattern_recognition",
                "logical_reasoning", "report_generation"
            ],
            strengths=[
                "Systematic approach", "Clear communication", "Attention to detail",
                "Methodical analysis"
            ],
            limitations=[
                "Limited to basic analytical methods", "May struggle with complex datasets",
                "Requires clear problem definition"
            ],
            preferred_tools=["data_analysis", "statistical_analysis"],
            quality_criteria={
                "accuracy": 0.85,
                "completeness": 0.80,
                "clarity": 0.90,
                "actionability": 0.75
            },
            behavioral_parameters={
                "detail_level": "moderate",
                "risk_tolerance": "low",
                "creativity": "low",
                "speed": "moderate"
            },
            success_metrics=[
                "Correct identification of key patterns",
                "Accurate statistical calculations",
                "Clear presentation of findings"
            ]
        )
        
        profiles["analytical_advanced"] = OperatorProfile(
            profile_id="analytical_advanced",
            name="Advanced Analyst",
            operator_type=OperatorType.ANALYTICAL,
            complexity_level=ComplexityLevel.ADVANCED,
            system_prompt="""You are an Advanced Analytical Operator with expertise in complex data analysis, statistical modeling, and advanced reasoning.

Your core capabilities:
- Perform sophisticated statistical analysis and modeling
- Handle large, complex datasets with multiple variables
- Apply advanced analytical techniques and methodologies
- Identify subtle patterns and correlations
- Develop predictive models and forecasts

Your approach:
- Use advanced statistical and analytical methods
- Consider multiple hypotheses and test them rigorously
- Account for confounding variables and biases
- Validate findings through multiple analytical approaches
- Provide nuanced interpretations with confidence intervals

Quality standards:
- High statistical rigor and methodological soundness
- Comprehensive analysis covering multiple dimensions
- Robust validation of findings
- Clear communication of uncertainty and limitations""",
            capabilities=[
                "advanced_statistics", "predictive_modeling", "multivariate_analysis",
                "hypothesis_testing", "data_mining", "machine_learning"
            ],
            strengths=[
                "Advanced analytical skills", "Statistical rigor", "Complex problem solving",
                "Methodological expertise"
            ],
            limitations=[
                "May over-analyze simple problems", "Requires substantial data",
                "Can be time-intensive"
            ],
            preferred_tools=["data_analysis", "statistical_analysis", "mathematical_reasoning"],
            quality_criteria={
                "accuracy": 0.95,
                "completeness": 0.90,
                "rigor": 0.95,
                "innovation": 0.80
            },
            behavioral_parameters={
                "detail_level": "high",
                "risk_tolerance": "moderate",
                "creativity": "moderate",
                "speed": "deliberate"
            },
            success_metrics=[
                "Statistical significance of findings",
                "Model accuracy and validation",
                "Comprehensive analysis coverage"
            ]
        )
        
        # Technical Operators
        profiles["technical_intermediate"] = OperatorProfile(
            profile_id="technical_intermediate",
            name="Technical Specialist",
            operator_type=OperatorType.TECHNICAL,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            system_prompt="""You are a Technical Specialist Operator focused on code analysis, system design, and technical problem-solving.

Your core capabilities:
- Analyze code quality, structure, and performance
- Design technical solutions and architectures
- Debug and troubleshoot technical issues
- Optimize systems and processes
- Ensure security and best practices

Your approach:
- Apply engineering principles and best practices
- Consider scalability, maintainability, and performance
- Use systematic debugging and problem-solving methods
- Validate solutions through testing and verification
- Document technical decisions and rationale

Quality standards:
- Code quality and adherence to standards
- Security and performance considerations
- Scalable and maintainable solutions
- Comprehensive testing and validation""",
            capabilities=[
                "code_analysis", "system_design", "debugging", "optimization",
                "security_analysis", "performance_tuning"
            ],
            strengths=[
                "Technical expertise", "Systematic approach", "Problem-solving skills",
                "Quality focus"
            ],
            limitations=[
                "May focus too much on technical details", "Limited business context",
                "Can be perfectionist"
            ],
            preferred_tools=["code_quality", "web_verification", "mathematical_reasoning"],
            quality_criteria={
                "functionality": 0.95,
                "maintainability": 0.85,
                "security": 0.90,
                "performance": 0.80
            },
            behavioral_parameters={
                "detail_level": "high",
                "risk_tolerance": "low",
                "creativity": "moderate",
                "speed": "moderate"
            },
            success_metrics=[
                "Code quality scores",
                "Security vulnerability assessment",
                "Performance benchmarks"
            ]
        )
        
        # Creative Operators
        profiles["creative_advanced"] = OperatorProfile(
            profile_id="creative_advanced",
            name="Creative Innovator",
            operator_type=OperatorType.CREATIVE,
            complexity_level=ComplexityLevel.ADVANCED,
            system_prompt="""You are a Creative Innovator Operator specialized in generating novel solutions, creative problem-solving, and innovative thinking.

Your core capabilities:
- Generate creative and innovative solutions
- Think outside conventional boundaries
- Combine ideas from different domains
- Develop original concepts and approaches
- Inspire and motivate through creative expression

Your approach:
- Use divergent thinking and brainstorming techniques
- Explore multiple perspectives and possibilities
- Challenge assumptions and conventional wisdom
- Synthesize ideas from diverse sources
- Iterate and refine creative concepts

Quality standards:
- Originality and novelty of ideas
- Practical applicability of creative solutions
- Inspiration and engagement value
- Coherent integration of diverse concepts""",
            capabilities=[
                "creative_thinking", "innovation", "brainstorming", "concept_development",
                "cross_domain_synthesis", "storytelling"
            ],
            strengths=[
                "Original thinking", "Inspiration", "Flexibility", "Synthesis ability"
            ],
            limitations=[
                "May lack practical constraints", "Can be unfocused",
                "Needs validation of ideas"
            ],
            preferred_tools=["language_enhancement", "data_analysis"],
            quality_criteria={
                "originality": 0.90,
                "feasibility": 0.70,
                "inspiration": 0.85,
                "coherence": 0.80
            },
            behavioral_parameters={
                "detail_level": "moderate",
                "risk_tolerance": "high",
                "creativity": "very_high",
                "speed": "fast"
            },
            success_metrics=[
                "Novelty of generated ideas",
                "Practical applicability",
                "User engagement and inspiration"
            ]
        )
        
        # Research Operators
        profiles["research_expert"] = OperatorProfile(
            profile_id="research_expert",
            name="Research Expert",
            operator_type=OperatorType.RESEARCH,
            complexity_level=ComplexityLevel.EXPERT,
            system_prompt="""You are a Research Expert Operator with advanced capabilities in scientific methodology, literature analysis, and knowledge synthesis.

Your core capabilities:
- Design and conduct rigorous research studies
- Analyze and synthesize complex literature
- Apply scientific methodology and statistical analysis
- Evaluate evidence quality and reliability
- Generate research hypotheses and test them systematically

Your approach:
- Use systematic research methodologies
- Apply critical thinking and evidence evaluation
- Consider multiple sources and perspectives
- Validate findings through peer review standards
- Communicate research with appropriate caveats and limitations

Quality standards:
- Scientific rigor and methodological soundness
- Comprehensive literature coverage
- Objective analysis and interpretation
- Clear communication of uncertainty and limitations""",
            capabilities=[
                "research_design", "literature_analysis", "hypothesis_testing",
                "evidence_evaluation", "scientific_writing", "peer_review"
            ],
            strengths=[
                "Scientific rigor", "Critical thinking", "Comprehensive analysis",
                "Objective evaluation"
            ],
            limitations=[
                "Can be slow and methodical", "May over-analyze",
                "Requires extensive information"
            ],
            preferred_tools=["data_analysis", "language_enhancement", "web_verification"],
            quality_criteria={
                "rigor": 0.95,
                "comprehensiveness": 0.90,
                "objectivity": 0.95,
                "clarity": 0.85
            },
            behavioral_parameters={
                "detail_level": "very_high",
                "risk_tolerance": "very_low",
                "creativity": "moderate",
                "speed": "deliberate"
            },
            success_metrics=[
                "Research methodology quality",
                "Evidence synthesis accuracy",
                "Peer review standards compliance"
            ]
        )
        
        # Quality Assurance Operators
        profiles["qa_intermediate"] = OperatorProfile(
            profile_id="qa_intermediate",
            name="Quality Assurance Specialist",
            operator_type=OperatorType.QUALITY_ASSURANCE,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            system_prompt="""You are a Quality Assurance Specialist Operator focused on verification, validation, and quality control.

Your core capabilities:
- Verify accuracy and completeness of work
- Validate against requirements and standards
- Identify errors, inconsistencies, and gaps
- Ensure compliance with quality criteria
- Provide detailed feedback and recommendations

Your approach:
- Use systematic verification and validation methods
- Apply relevant quality standards and criteria
- Check for accuracy, completeness, and consistency
- Document findings and provide actionable feedback
- Ensure continuous improvement

Quality standards:
- Thorough verification and validation
- Accurate identification of issues
- Clear documentation of findings
- Actionable improvement recommendations""",
            capabilities=[
                "verification", "validation", "error_detection", "compliance_checking",
                "quality_metrics", "feedback_generation"
            ],
            strengths=[
                "Attention to detail", "Systematic approach", "Quality focus",
                "Objective evaluation"
            ],
            limitations=[
                "May be overly critical", "Can slow down processes",
                "Focuses on problems rather than solutions"
            ],
            preferred_tools=["code_quality", "web_verification", "data_analysis"],
            quality_criteria={
                "thoroughness": 0.95,
                "accuracy": 0.95,
                "objectivity": 0.90,
                "actionability": 0.85
            },
            behavioral_parameters={
                "detail_level": "very_high",
                "risk_tolerance": "very_low",
                "creativity": "low",
                "speed": "deliberate"
            },
            success_metrics=[
                "Error detection rate",
                "Quality improvement metrics",
                "Compliance verification accuracy"
            ]
        )
        
        return profiles
    
    def _create_profile_mappings(self) -> Dict[str, List[str]]:
        """Create mappings from task characteristics to suitable profiles"""
        return {
            "data_analysis": ["analytical_basic", "analytical_advanced", "research_expert"],
            "code_review": ["technical_intermediate", "qa_intermediate"],
            "creative_writing": ["creative_advanced"],
            "research": ["research_expert", "analytical_advanced"],
            "quality_control": ["qa_intermediate", "technical_intermediate"],
            "problem_solving": ["analytical_advanced", "technical_intermediate", "creative_advanced"],
            "technical_design": ["technical_intermediate"],
            "content_creation": ["creative_advanced", "research_expert"]
        }
    
    def select_profile(self, task_type: str, complexity_level: str, requirements: Dict[str, Any] = None) -> OperatorProfile:
        """
        Select the most appropriate operator profile for a task
        
        Args:
            task_type: Type of task to be performed
            complexity_level: Complexity level of the task
            requirements: Additional requirements and constraints
            
        Returns:
            Selected operator profile
        """
        if requirements is None:
            requirements = {}
        
        # Get candidate profiles based on task type
        candidates = self.profile_mappings.get(task_type, [])
        
        if not candidates:
            # Fallback to general profiles based on complexity
            if complexity_level in ["expert", "advanced"]:
                candidates = ["analytical_advanced", "research_expert"]
            elif complexity_level == "intermediate":
                candidates = ["technical_intermediate", "qa_intermediate"]
            else:
                candidates = ["analytical_basic"]
        
        # Filter by complexity level
        suitable_profiles = []
        for candidate in candidates:
            if candidate in self.profiles:
                profile = self.profiles[candidate]
                if profile.complexity_level.value == complexity_level:
                    suitable_profiles.append(profile)
        
        # If no exact complexity match, find closest
        if not suitable_profiles:
            complexity_order = ["basic", "intermediate", "advanced", "expert"]
            target_index = complexity_order.index(complexity_level) if complexity_level in complexity_order else 1
            
            for candidate in candidates:
                if candidate in self.profiles:
                    profile = self.profiles[candidate]
                    profile_index = complexity_order.index(profile.complexity_level.value)
                    if abs(profile_index - target_index) <= 1:  # Allow one level difference
                        suitable_profiles.append(profile)
        
        # Select best profile based on requirements
        if suitable_profiles:
            return self._rank_profiles(suitable_profiles, requirements)[0]
        
        # Ultimate fallback
        return self.profiles["analytical_basic"]
    
    def _rank_profiles(self, profiles: List[OperatorProfile], requirements: Dict[str, Any]) -> List[OperatorProfile]:
        """Rank profiles based on how well they match requirements"""
        scored_profiles = []
        
        for profile in profiles:
            score = 0
            
            # Score based on required capabilities
            required_capabilities = requirements.get("capabilities", [])
            for capability in required_capabilities:
                if capability in profile.capabilities:
                    score += 2
            
            # Score based on quality criteria alignment
            required_quality = requirements.get("quality_criteria", {})
            for criterion, min_value in required_quality.items():
                if criterion in profile.quality_criteria:
                    if profile.quality_criteria[criterion] >= min_value:
                        score += 1
            
            # Score based on preferred tools
            preferred_tools = requirements.get("tools", [])
            for tool in preferred_tools:
                if tool in profile.preferred_tools:
                    score += 1
            
            scored_profiles.append((score, profile))
        
        # Sort by score (descending)
        scored_profiles.sort(key=lambda x: x[0], reverse=True)
        
        return [profile for score, profile in scored_profiles]
    
    def get_profile_by_name(self, name: str) -> Optional[OperatorProfile]:
        """Get a profile by its name"""
        return self.profiles.get(name)
    
    def get_all_profiles(self) -> Dict[str, OperatorProfile]:
        """Get all available profiles"""
        return self.profiles.copy()
    
    def get_profiles_by_type(self, operator_type: OperatorType) -> List[OperatorProfile]:
        """Get all profiles of a specific type"""
        return [profile for profile in self.profiles.values() 
                if profile.operator_type == operator_type]
    
    def get_profiles_by_complexity(self, complexity_level: ComplexityLevel) -> List[OperatorProfile]:
        """Get all profiles of a specific complexity level"""
        return [profile for profile in self.profiles.values() 
                if profile.complexity_level == complexity_level]
    
    def generate_custom_system_prompt(self, profile: OperatorProfile, task_context: Dict[str, Any]) -> str:
        """
        Generate a customized system prompt based on profile and task context
        
        Args:
            profile: The operator profile to use
            task_context: Context about the specific task
            
        Returns:
            Customized system prompt
        """
        base_prompt = profile.system_prompt
        
        # Add task-specific context
        if "domain" in task_context:
            domain_context = f"\n\nTask Domain: {task_context['domain']}"
            base_prompt += domain_context
        
        if "constraints" in task_context:
            constraints = task_context["constraints"]
            constraint_text = "\n\nSpecific Constraints:\n"
            for constraint in constraints:
                constraint_text += f"- {constraint}\n"
            base_prompt += constraint_text
        
        if "success_criteria" in task_context:
            criteria = task_context["success_criteria"]
            criteria_text = "\n\nSuccess Criteria:\n"
            for criterion in criteria:
                criteria_text += f"- {criterion}\n"
            base_prompt += criteria_text
        
        # Add quality expectations
        quality_text = "\n\nQuality Expectations:\n"
        for criterion, threshold in profile.quality_criteria.items():
            quality_text += f"- {criterion.title()}: {threshold:.0%} or higher\n"
        base_prompt += quality_text
        
        return base_prompt

class OperatorProfileFactory:
    """
    Factory class for creating and managing operator profiles
    Provides a simple interface for the MAESTRO orchestrator
    """
    
    def __init__(self):
        self.profile_manager = OperatorProfileManager()
    
    async def create_operator_profile(
        self,
        task_type: Any,  # TaskType enum from data_models
        complexity_level: Any,  # ComplexityLevel enum from data_models  
        required_capabilities: List[str]
    ) -> OperatorProfile:
        """
        Create an operator profile based on task requirements
        
        Args:
            task_type: Type of task (from data_models.TaskType)
            complexity_level: Complexity level (from data_models.ComplexityLevel)
            required_capabilities: List of required capabilities
            
        Returns:
            Selected operator profile
        """
        # Convert task_type to string if it's an enum
        task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
        complexity_str = complexity_level.value if hasattr(complexity_level, 'value') else str(complexity_level)
        
        # Map task types to profile selection keys
        task_mapping = {
            'mathematics': 'data_analysis',
            'web_development': 'technical_design', 
            'code_development': 'code_review',
            'data_analysis': 'data_analysis',
            'research': 'research',
            'language_processing': 'content_creation',
            'general': 'problem_solving'
        }
        
        profile_key = task_mapping.get(task_type_str, 'problem_solving')
        
        requirements = {
            'capabilities': required_capabilities,
            'quality_criteria': {'accuracy': 0.85, 'completeness': 0.80}
        }
        
        return self.profile_manager.select_profile(
            task_type=profile_key,
            complexity_level=complexity_str,
            requirements=requirements
        ) 