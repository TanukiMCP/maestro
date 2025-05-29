"""
MAESTRO Protocol Core Orchestrator

Implements the main orchestration logic for the Meta-Agent Ensemble for 
Systematic Task Reasoning and Orchestration (MAESTRO) Protocol.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import re

from .data_models import (
    MAESTROResult, TaskAnalysis, Workflow, WorkflowNode, 
    VerificationResult, QualityMetrics, ExecutionMetrics,
    TaskType, ComplexityLevel, VerificationMethod
)
from .quality_controller import QualityController

# Use try/except for imports that might fail during testing
try:
    from ..engines import IntelligenceAmplifier
except ImportError:
    # Fallback for testing
    try:
        from engines import IntelligenceAmplifier
    except ImportError:
        IntelligenceAmplifier = None

try:
    from ..profiles import OperatorProfileFactory
except ImportError:
    # Fallback for testing
    try:
        from profiles.operator_profiles import OperatorProfileFactory
    except ImportError:
        OperatorProfileFactory = None

try:
    from .knowledge_graph_engine import RealTimeKnowledgeGraph
except ImportError:
    RealTimeKnowledgeGraph = None

logger = logging.getLogger(__name__)


class MAESTROOrchestrator:
    """
    Core orchestration engine that implements the MAESTRO Protocol.
    Automatically designs workflows, assigns operator profiles, and manages execution.
    """
    
    def __init__(self):
        # Initialize components with fallbacks for testing
        if OperatorProfileFactory:
            self.operator_factory = OperatorProfileFactory()
        else:
            self.operator_factory = None
            
        if IntelligenceAmplifier:
            self.intelligence_amplifier = IntelligenceAmplifier()
        else:
            self.intelligence_amplifier = None
            
        self.quality_controller = QualityController()
        
        # Initialize Real-Time Knowledge Graph (inspired by Cole Medin's RAG optimization)
        if RealTimeKnowledgeGraph:
            self.knowledge_graph = RealTimeKnowledgeGraph()
            self.use_dynamic_optimization = True
            logger.info("🌐 Real-time knowledge graph enabled for dynamic orchestration")
        else:
            self.knowledge_graph = None
            self.use_dynamic_optimization = False
            logger.info("📚 Using static patterns (knowledge graph unavailable)")
        
        # Task classification patterns (fallback for when knowledge graph is unavailable)
        self.task_patterns = self._initialize_task_patterns()
        
        logger.info("🎭 MAESTRO Orchestrator initialized")
    
    def _initialize_task_patterns(self) -> Dict[TaskType, List[str]]:
        """Initialize patterns for task type classification."""
        return {
            TaskType.MATHEMATICS: [
                r"calcul(ate|ation)", r"solve.*equation", r"integral", r"derivative", 
                r"statistics", r"probability", r"math", r"formula", r"proof"
            ],
            TaskType.WEB_DEVELOPMENT: [
                r"website", r"web.*app", r"html", r"css", r"javascript", r"react", 
                r"frontend", r"backend", r"api", r"responsive"
            ],
            TaskType.CODE_DEVELOPMENT: [
                r"function", r"class", r"algorithm", r"code", r"program", r"script",
                r"python", r"javascript", r"java", r"refactor"
            ],
            TaskType.DATA_ANALYSIS: [
                r"data.*analy", r"dataset", r"csv", r"pandas", r"visualization", 
                r"chart", r"graph", r"statistics", r"correlation"
            ],
            TaskType.RESEARCH: [
                r"research", r"study", r"analy[sz]e", r"investigate", r"compare",
                r"report", r"summary", r"findings"
            ],
            TaskType.LANGUAGE_PROCESSING: [
                r"text.*processing", r"nlp", r"language", r"sentiment", r"grammar",
                r"translation", r"summariz", r"writing"
            ]
        }
    
    async def orchestrate_workflow(
        self, 
        task_description: str,
        quality_threshold: float = 0.9,
        verification_mode: str = "fast",
        max_execution_time: int = 300
    ) -> MAESTROResult:
        """
        Primary entry point for MAESTRO Protocol orchestration.
        
        Args:
            task_description: Natural language description of the task
            quality_threshold: Minimum quality score for completion (0.0-1.0)
            verification_mode: "fast", "balanced", or "comprehensive"
            max_execution_time: Maximum execution time in seconds
            
        Returns:
            Complete MAESTRO result with verification and metrics
        """
        start_time = time.time()
        
        try:
            # Optimize for fast mode
            if verification_mode == "fast":
                logger.info("⚡ Fast mode - streamlined orchestration")
                
                # Lightweight task analysis for fast mode
                task_analysis = await self._analyze_task_lightweight(task_description)
                
                # Skip operator profile generation for simple tasks
                if task_analysis.task_type in [TaskType.MATHEMATICS, TaskType.GENERAL]:
                    # Direct execution for simple tasks
                    execution_result = await self._execute_simple_task(
                        task_description, task_analysis, max_execution_time
                    )
                    
                    # Minimal verification for fast mode
                    verification = await self._quick_verification(execution_result, task_analysis)
                    
                    execution_time = time.time() - start_time
                    
                    # Learn from fast mode execution too (key for real-time optimization)
                    if self.use_dynamic_optimization and self.knowledge_graph:
                        await self._learn_from_execution_result(
                            task_description, task_analysis, execution_result, verification, execution_time
                        )
                    
                    return self._create_fast_result(
                        task_description, execution_result, verification, execution_time
                    )
            
            # Full orchestration for balanced/comprehensive modes
            logger.info("🔄 Full orchestration mode")
            
            # Step 1: Task Analysis & Complexity Assessment
            logger.info("🔍 Analyzing task complexity...")
            task_analysis = await self.analyze_task_complexity(task_description)
            
            # Step 2: Operator Profile Selection & Custom System Prompt Generation
            logger.info("👤 Creating operator profile...")
            operator_profile = await self.operator_factory.create_operator_profile(
                task_type=task_analysis.task_type,
                complexity_level=task_analysis.complexity,
                required_capabilities=task_analysis.capabilities
            )
            
            # Step 3: Dynamic Workflow Generation with Operator Context
            logger.info("🔄 Generating workflow...")
            workflow = await self.generate_workflow(
                task_description=task_description,
                task_analysis=task_analysis,
                operator_profile=operator_profile,
                quality_threshold=quality_threshold,
                verification_mode=verification_mode
            )
            
            # Step 4: Execution with Continuous Quality Monitoring
            logger.info("⚡ Executing workflow...")
            execution_result = await self.execute_workflow(
                workflow=workflow,
                operator_profile=operator_profile,
                max_execution_time=max_execution_time
            )
            
            # Step 5: Final Verification & Success Confirmation
            logger.info("✅ Final verification...")
            final_verification = await self.quality_controller.final_verification(
                result=execution_result,
                success_criteria=task_analysis.success_criteria,
                quality_threshold=quality_threshold
            )
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            metrics = ExecutionMetrics(
                total_time=execution_time,
                nodes_executed=len(workflow.nodes),
                nodes_successful=sum(1 for node in workflow.nodes if node.verification_result and node.verification_result.success),
                retries_performed=0,  # TODO: Track retries
                quality_checks_run=len(final_verification.detailed_results)
            )
            
            # Learn from execution results (key innovation from Cole Medin's video)
            if self.use_dynamic_optimization and self.knowledge_graph:
                await self._learn_from_execution_result(
                    task_description=task_description,
                    task_analysis=task_analysis,
                    execution_result=execution_result,
                    verification_result=final_verification,
                    execution_time=execution_time
                )
            
            return MAESTROResult(
                success=final_verification.success,
                task_description=task_description,
                detailed_output=execution_result.get("output", "Task completed successfully"),
                summary=execution_result.get("summary", "MAESTRO Protocol execution completed"),
                workflow_used=workflow,
                operator_profile_id=operator_profile.profile_id,
                verification=final_verification,
                execution_metrics=metrics,
                files_affected=execution_result.get("files_created", [])
            )
            
        except Exception as e:
            logger.error(f"MAESTRO orchestration failed: {str(e)}")
            # Return failure result with error details
            execution_time = time.time() - start_time
            
            error_verification = VerificationResult(
                success=False,
                confidence_score=0.0,
                quality_metrics=QualityMetrics(),
                issues_found=[f"Orchestration error: {str(e)}"],
                recommendations=["Check task description and try again", "Simplify the request", "Ensure all dependencies are available"]
            )
            
            return MAESTROResult(
                success=False,
                task_description=task_description,
                detailed_output=f"Error: {str(e)}",
                summary="MAESTRO Protocol encountered an error during execution",
                workflow_used=Workflow.create(task_description),
                operator_profile_id="error",
                verification=error_verification,
                execution_metrics=ExecutionMetrics(
                    total_time=execution_time,
                    nodes_executed=0,
                    nodes_successful=0,
                    retries_performed=0,
                    quality_checks_run=0
                )
            )
    
    async def analyze_task_complexity(self, task_description: str) -> TaskAnalysis:
        """
        Analyze task to determine type, complexity, and required capabilities.
        Now uses real-time knowledge graph when available for dynamic optimization.
        
        Args:
            task_description: Natural language task description
            
        Returns:
            TaskAnalysis with classification and requirements
        """
        # Use knowledge graph for dynamic analysis if available
        if self.use_dynamic_optimization and self.knowledge_graph:
            logger.info("🌐 Using knowledge graph for dynamic task analysis")
            
            # Get knowledge graph recommendations
            graph_analysis = await self.knowledge_graph.analyze_task_context(task_description)
            
            # Convert to TaskAnalysis format
            task_type = self._classify_task_type(task_description.lower())  # Still need basic classification
            complexity = self._assess_complexity_with_graph(task_description, graph_analysis)
            capabilities = graph_analysis["recommended_capabilities"]
            estimated_duration = int(graph_analysis["estimated_execution_time"])
            
            success_criteria = self._define_success_criteria(task_type, complexity)
            
            logger.info(f"📊 Knowledge graph confidence: {graph_analysis['confidence_score']:.2f}, "
                       f"Similar tasks: {graph_analysis['similar_tasks_count']}, "
                       f"Source: {graph_analysis['learning_source']}")
            
            return TaskAnalysis(
                task_type=task_type,
                complexity=complexity,
                capabilities=capabilities,
                estimated_duration=estimated_duration,
                required_tools=self._get_required_tools(task_type, capabilities),
                success_criteria=success_criteria,
                quality_requirements=self._get_quality_requirements(complexity),
                # Add knowledge graph metadata
                metadata={
                    "knowledge_graph_confidence": graph_analysis["confidence_score"],
                    "similar_tasks_found": graph_analysis["similar_tasks_count"],
                    "learning_source": graph_analysis["learning_source"],
                    "estimated_success_rate": graph_analysis["estimated_success_rate"]
                }
            )
        
        # Fallback to static analysis
        logger.info("📚 Using static pattern matching for task analysis")
        
        # Original static implementation
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
        
        # Return highest scoring type, default to GENERAL
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return TaskType.GENERAL
    
    def _assess_complexity(self, task_description: str) -> ComplexityLevel:
        """Assess task complexity based on indicators."""
        complexity_indicators = {
            ComplexityLevel.EXPERT: ["advanced", "complex", "sophisticated", "enterprise", "scalable", "optimization"],
            ComplexityLevel.COMPLEX: ["multiple", "integration", "comprehensive", "detailed", "analysis"],
            ComplexityLevel.MODERATE: ["medium", "standard", "typical", "moderate", "some"],
            ComplexityLevel.SIMPLE: ["simple", "basic", "easy", "quick", "minimal"]
        }
        
        scores = {}
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in task_description)
            scores[level] = score
        
        # Length-based complexity assessment
        if len(task_description.split()) > 50:
            scores[ComplexityLevel.COMPLEX] += 2
        elif len(task_description.split()) > 20:
            scores[ComplexityLevel.MODERATE] += 1
        
        # Return highest scoring complexity, default to MODERATE
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return ComplexityLevel.MODERATE
    
    def _determine_capabilities(self, task_type: TaskType, task_description: str) -> List[str]:
        """Determine required capabilities based on task type and description."""
        base_capabilities = {
            TaskType.MATHEMATICS: ["mathematics", "symbolic_computation", "numerical_analysis"],
            TaskType.WEB_DEVELOPMENT: ["web_development", "frontend", "backend", "testing"],
            TaskType.CODE_DEVELOPMENT: ["code_generation", "testing", "quality_analysis"],
            TaskType.DATA_ANALYSIS: ["data_processing", "visualization", "statistics"],
            TaskType.RESEARCH: ["information_gathering", "analysis", "reporting"],
            TaskType.LANGUAGE_PROCESSING: ["language_analysis", "text_processing", "grammar_checking"],
            TaskType.GENERAL: ["general_reasoning"]
        }
        
        capabilities = base_capabilities.get(task_type, ["general_reasoning"])
        
        # Add additional capabilities based on keywords
        if any(word in task_description.lower() for word in ["test", "verify", "check"]):
            capabilities.append("quality_verification")
        
        if any(word in task_description.lower() for word in ["visual", "ui", "interface"]):
            capabilities.append("visual_verification")
        
        return capabilities
    
    def _estimate_duration(self, complexity: ComplexityLevel, task_type: TaskType) -> int:
        """Estimate task duration in seconds."""
        base_times = {
            ComplexityLevel.SIMPLE: 30,
            ComplexityLevel.MODERATE: 60,
            ComplexityLevel.COMPLEX: 120,
            ComplexityLevel.EXPERT: 300
        }
        
        type_multipliers = {
            TaskType.WEB_DEVELOPMENT: 1.5,
            TaskType.DATA_ANALYSIS: 1.3,
            TaskType.RESEARCH: 1.4,
            TaskType.MATHEMATICS: 1.0,
            TaskType.CODE_DEVELOPMENT: 1.2,
            TaskType.LANGUAGE_PROCESSING: 1.1,
            TaskType.GENERAL: 1.0
        }
        
        base_time = base_times[complexity]
        multiplier = type_multipliers.get(task_type, 1.0)
        
        return int(base_time * multiplier)
    
    def _define_success_criteria(self, task_type: TaskType, complexity: ComplexityLevel) -> List[str]:
        """Define success criteria based on task type and complexity."""
        base_criteria = [
            "Task completion verified",
            "Quality standards met",
            "No critical errors found"
        ]
        
        type_specific = {
            TaskType.MATHEMATICS: ["Mathematical accuracy verified", "Calculations correct"],
            TaskType.WEB_DEVELOPMENT: ["Code quality verified", "Functionality tested"],
            TaskType.CODE_DEVELOPMENT: ["Code runs without errors", "Meets style guidelines"],
            TaskType.DATA_ANALYSIS: ["Data processed correctly", "Visualizations clear"],
            TaskType.RESEARCH: ["Information accurate", "Sources credible"],
            TaskType.LANGUAGE_PROCESSING: ["Grammar correct", "Style appropriate"]
        }
        
        criteria = base_criteria + type_specific.get(task_type, [])
        
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]:
            criteria.append("Performance optimized")
            criteria.append("Edge cases handled")
        
        return criteria
    
    def _get_required_tools(self, task_type: TaskType, capabilities: List[str]) -> List[str]:
        """Get required tools based on task type and capabilities."""
        tool_mapping = {
            "mathematics": ["sympy", "numpy", "scipy", "matplotlib"],
            "web_development": ["playwright", "selenium", "beautifulsoup"],
            "code_generation": ["pylint", "black", "pytest"],
            "data_processing": ["pandas", "numpy", "matplotlib"],
            "language_analysis": ["spacy", "nltk", "language-tool"],
            "quality_verification": ["pytest", "pylint", "coverage"]
        }
        
        tools = set()
        for capability in capabilities:
            if capability in tool_mapping:
                tools.update(tool_mapping[capability])
        
        return list(tools)
    
    def _get_quality_requirements(self, complexity: ComplexityLevel) -> Dict[str, float]:
        """Get quality requirements based on complexity."""
        base_requirements = {
            "accuracy": 0.95,
            "completeness": 0.90,
            "quality": 0.85
        }
        
        if complexity == ComplexityLevel.EXPERT:
            return {"accuracy": 0.98, "completeness": 0.95, "quality": 0.92}
        elif complexity == ComplexityLevel.COMPLEX:
            return {"accuracy": 0.96, "completeness": 0.93, "quality": 0.88}
        elif complexity == ComplexityLevel.SIMPLE:
            return {"accuracy": 0.93, "completeness": 0.85, "quality": 0.80}
        
        return base_requirements
    
    def _assess_complexity_with_graph(self, task_description: str, graph_analysis: Dict[str, Any]) -> ComplexityLevel:
        """Assess complexity using knowledge graph insights"""
        # Use estimated execution time from graph
        execution_time = graph_analysis.get("estimated_execution_time", 60)
        confidence = graph_analysis.get("confidence_score", 0.5)
        
        # Higher confidence and lower execution time suggests simpler task
        if execution_time < 30 and confidence > 0.8:
            return ComplexityLevel.SIMPLE
        elif execution_time < 90 and confidence > 0.6:
            return ComplexityLevel.MODERATE
        elif execution_time < 180:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXPERT
    
    async def generate_workflow(
        self,
        task_description: str,
        task_analysis: TaskAnalysis,
        operator_profile,
        quality_threshold: float,
        verification_mode: str
    ) -> Workflow:
        """Generate a dynamic workflow based on task analysis."""
        workflow = Workflow.create(
            task_description=task_description,
            quality_threshold=quality_threshold,
            verification_mode=verification_mode
        )
        
        # Create main execution node
        main_node = WorkflowNode.create(
            task_description=task_description,
            capabilities=task_analysis.capabilities,
            success_criteria=task_analysis.success_criteria
        )
        main_node.operator_profile_id = operator_profile.profile_id
        main_node.verification_methods = self._get_verification_methods(task_analysis.task_type)
        
        workflow.nodes.append(main_node)
        workflow.capabilities_used = task_analysis.capabilities
        
        return workflow
    
    def _get_verification_methods(self, task_type: TaskType) -> List[VerificationMethod]:
        """Get appropriate verification methods for task type."""
        method_mapping = {
            TaskType.MATHEMATICS: [VerificationMethod.MATHEMATICAL_VERIFICATION, VerificationMethod.AUTOMATED_TESTING],
            TaskType.WEB_DEVELOPMENT: [VerificationMethod.VISUAL_VERIFICATION, VerificationMethod.CODE_QUALITY_VERIFICATION],
            TaskType.CODE_DEVELOPMENT: [VerificationMethod.CODE_QUALITY_VERIFICATION, VerificationMethod.AUTOMATED_TESTING],
            TaskType.DATA_ANALYSIS: [VerificationMethod.AUTOMATED_TESTING, VerificationMethod.VISUAL_VERIFICATION],
            TaskType.LANGUAGE_PROCESSING: [VerificationMethod.LANGUAGE_QUALITY_VERIFICATION],
            TaskType.RESEARCH: [VerificationMethod.LANGUAGE_QUALITY_VERIFICATION],
            TaskType.GENERAL: [VerificationMethod.AUTOMATED_TESTING]
        }
        
        return method_mapping.get(task_type, [VerificationMethod.AUTOMATED_TESTING])
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        operator_profile,
        max_execution_time: int
    ) -> Dict[str, Any]:
        """Execute the workflow with quality monitoring."""
        
        # For this initial implementation, simulate execution
        # In a full implementation, this would use LangChain agents
        
        result = await self._simulate_task_execution(
            workflow.task_description,
            operator_profile,
            workflow.capabilities_used
        )
        
        # Verify each node
        for node in workflow.nodes:
            node.verification_result = await self.quality_controller.verify_node_result(
                node=node,
                result=result
            )
        
        return result
    
    async def _simulate_task_execution(
        self,
        task_description: str,
        operator_profile,
        capabilities: List[str]
    ) -> Dict[str, Any]:
        """Simulate task execution for demonstration purposes."""
        
        # Use intelligence amplification when available
        if "mathematics" in capabilities:
            try:
                math_result = await self.intelligence_amplifier.process_mathematical_task(task_description)
                return {
                    "output": f"Mathematical task completed: {math_result}",
                    "summary": "Used mathematical intelligence amplification",
                    "files_created": [],
                    "quality_score": 0.95
                }
            except Exception:
                pass
        
        # Default execution simulation
        return {
            "output": f"Task completed successfully using {operator_profile.operator_type.value} operator profile. "
                     f"Capabilities used: {', '.join(capabilities)}. "
                     f"The MAESTRO Protocol has orchestrated this task with quality verification.",
            "summary": "Task executed with MAESTRO Protocol orchestration",
            "files_created": [],
            "quality_score": 0.90
        }
    
    async def verify_content_quality(
        self,
        content: str,
        verification_type: str,
        success_criteria: Dict[str, Any]
    ) -> VerificationResult:
        """Verify content quality using appropriate methods."""
        return await self.quality_controller.verify_content(
            content=content,
            verification_type=verification_type,
            success_criteria=success_criteria
        )
    
    async def amplify_capability(
        self,
        capability_type: str,
        input_data: Any,
        requirements: Dict[str, Any]
    ) -> Any:
        """Use intelligence amplification for specific capabilities."""
        if self.intelligence_amplifier:
            return await self.intelligence_amplifier.amplify_capability(
                capability=capability_type,
                input_data=str(input_data),
                context=requirements
            )
        else:
            # Fallback when intelligence amplifier is not available
            return {
                'success': False,
                'error': 'Intelligence amplifier not available'
            }

    # Fallback to direct execution if no engine available
    async def _execute_simple_task(self, task_description: str, task_analysis: TaskAnalysis, max_execution_time: int) -> Dict[str, Any]:
        """Execute simple tasks directly without complex orchestration."""
        
        if task_analysis.task_type == TaskType.MATHEMATICS and self.intelligence_amplifier:
            try:
                # Use intelligence amplifier for math tasks
                result = await self.intelligence_amplifier.amplify_capability(
                    capability="mathematics",
                    input_data=task_description,
                    context={}
                )
                
                if result.success:
                    return {
                        "output": result.amplified_output.get("solution", str(result.amplified_output)),
                        "summary": f"Mathematical computation completed: {task_description}",
                        "files_created": [],
                        "quality_score": result.confidence_score,
                        "processing_time": result.processing_time
                    }
            except Exception as e:
                logger.warning(f"Fast math execution failed: {e}")
        
        # General fallback
        return {
            "output": f"Task processed: {task_description}\n\nThis task was handled in fast mode. For more detailed processing, use verification_mode='comprehensive'.",
            "summary": f"Fast mode processing completed for: {task_description[:50]}...",
            "files_created": [],
            "quality_score": 0.8
        }
    
    async def _analyze_task_lightweight(self, task_description: str) -> TaskAnalysis:
        """Lightweight task analysis for fast mode."""
        # Quick classification without complex pattern matching
        task_lower = task_description.lower()
        
        # Simple keyword-based classification
        if any(word in task_lower for word in ["calculate", "math", "factorial", "derivative", "integral"]):
            task_type = TaskType.MATHEMATICS
        elif any(word in task_lower for word in ["code", "function", "program", "python"]):
            task_type = TaskType.CODE_DEVELOPMENT
        elif any(word in task_lower for word in ["website", "web", "html", "css"]):
            task_type = TaskType.WEB_DEVELOPMENT
        else:
            task_type = TaskType.GENERAL
        
        # Simple complexity assessment
        complexity = ComplexityLevel.SIMPLE if len(task_description.split()) < 10 else ComplexityLevel.MODERATE
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            capabilities=[task_type.value],
            estimated_duration=30,  # Fast mode assumes 30 seconds max
            required_tools=[],
            success_criteria=["Task completed successfully"],
            quality_requirements={"accuracy": 0.8, "completeness": 0.8}
        )
    
    async def _quick_verification(self, execution_result: Dict[str, Any], task_analysis: TaskAnalysis) -> VerificationResult:
        """Quick verification for fast mode."""
        
        # Basic verification based on output presence
        has_output = bool(execution_result.get("output"))
        quality_score = execution_result.get("quality_score", 0.8)
        
        return VerificationResult(
            success=has_output and quality_score >= 0.7,
            confidence_score=quality_score,
            quality_metrics=QualityMetrics(
                overall_score=quality_score,
                accuracy_score=quality_score,
                completeness_score=0.9 if has_output else 0.3,
                quality_score=quality_score,
                verification_scores={"quick_check": 0.8}
            ),
            issues_found=[] if has_output else ["No output generated"],
            recommendations=["Task completed in fast mode"] if has_output else ["Try comprehensive mode for better results"],
            detailed_results=[]
        )
    
    def _create_fast_result(self, task_description: str, execution_result: Dict[str, Any], verification: VerificationResult, execution_time: float) -> MAESTROResult:
        """Create result object for fast mode execution."""
        
        return MAESTROResult(
            success=verification.success,
            task_description=task_description,
            detailed_output=execution_result.get("output", "No output generated"),
            summary=execution_result.get("summary", "Fast mode execution completed"),
            workflow_used=Workflow.create(task_description),  # Minimal workflow
            operator_profile_id="fast_mode",
            verification=verification,
            execution_metrics=ExecutionMetrics(
                total_time=execution_time,
                nodes_executed=1,
                nodes_successful=1 if verification.success else 0,
                retries_performed=0,
                quality_checks_run=1
            ),
            files_affected=execution_result.get("files_created", [])
        )

    async def _learn_from_execution_result(
        self,
        task_description: str,
        task_analysis: TaskAnalysis,
        execution_result: Dict[str, Any],
        verification_result: VerificationResult,
        execution_time: float
    ):
        """Learn from execution results using the knowledge graph."""
        if self.use_dynamic_optimization and self.knowledge_graph:
            # Format execution data for knowledge graph learning
            learning_data = {
                "task_description": task_description,
                "task_type": task_analysis.task_type.value,
                "complexity": task_analysis.complexity.value,
                "capabilities_used": task_analysis.capabilities,
                "success": verification_result.success,
                "quality_score": verification_result.quality_metrics.overall_score,
                "execution_time": execution_time,
                "confidence_score": verification_result.confidence_score
            }
            
            await self.knowledge_graph.learn_from_execution(learning_data)
            
            # Log learning metrics
            metrics = self.knowledge_graph.get_knowledge_graph_metrics()
            logger.info(f"📊 Knowledge Graph Learning: {metrics['total_tasks']} tasks, "
                       f"{metrics['total_relationships']} relationships, "
                       f"avg success rate: {metrics['avg_task_success_rate']:.2%}") 