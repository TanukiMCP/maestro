"""
Knowledge Graph Engine for MAESTRO Protocol

Inspired by "Make RAG 100x Better with Real-Time Knowledge Graphs" by Cole Medin
Implements dynamic, real-time knowledge graphs for intelligent orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class TaskNode:
    """Represents a task in the knowledge graph"""
    task_id: str
    task_description: str
    task_type: str
    complexity: str
    success_rate: float = 0.0
    execution_count: int = 0
    avg_execution_time: float = 0.0
    preferred_capabilities: List[str] = field(default_factory=list)
    failed_capabilities: List[str] = field(default_factory=list)

@dataclass
class CapabilityNode:
    """Represents a capability in the knowledge graph"""
    capability_id: str
    capability_name: str
    success_rate: float = 0.0
    usage_count: int = 0
    avg_performance_score: float = 0.0
    compatible_tasks: Set[str] = field(default_factory=set)
    incompatible_tasks: Set[str] = field(default_factory=set)

@dataclass
class TaskCapabilityEdge:
    """Represents relationship between task and capability"""
    task_id: str
    capability_id: str
    success_rate: float
    performance_score: float
    execution_count: int
    last_used: float
    confidence: float

class RealTimeKnowledgeGraph:
    """
    Real-time knowledge graph for dynamic orchestration optimization.
    
    Learns from every execution to improve future task routing and 
    capability selection, eliminating static pattern matching.
    """
    
    def __init__(self):
        self.task_nodes: Dict[str, TaskNode] = {}
        self.capability_nodes: Dict[str, CapabilityNode] = {}
        self.task_capability_edges: Dict[Tuple[str, str], TaskCapabilityEdge] = {}
        
        # Real-time learning structures
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize with known capabilities
        self._initialize_base_capabilities()
        
        logger.info("🌐 Real-Time Knowledge Graph initialized")
    
    def _initialize_base_capabilities(self):
        """Initialize with base capabilities from engines"""
        base_capabilities = [
            "mathematics", "language", "code_quality", "web_verification", 
            "data_analysis", "symbolic_computation", "numerical_analysis",
            "text_processing", "visual_verification", "quality_control"
        ]
        
        for cap in base_capabilities:
            self.capability_nodes[cap] = CapabilityNode(
                capability_id=cap,
                capability_name=cap,
                success_rate=0.8,  # Start with reasonable baseline
                usage_count=0,
                avg_performance_score=0.8
            )
    
    async def analyze_task_context(self, task_description: str) -> Dict[str, Any]:
        """
        Real-time analysis using knowledge graph instead of static patterns.
        """
        # Generate task signature for graph lookup
        task_signature = self._generate_task_signature(task_description)
        
        # Check if we've seen similar tasks before
        similar_tasks = await self._find_similar_tasks(task_signature, task_description)
        
        if similar_tasks:
            # Use knowledge graph for recommendations
            recommendations = await self._get_graph_recommendations(similar_tasks)
        else:
            # Fallback to intelligent analysis for new task types
            recommendations = await self._analyze_new_task_type(task_description)
        
        return {
            "recommended_capabilities": recommendations["capabilities"],
            "estimated_success_rate": recommendations["success_rate"],
            "estimated_execution_time": recommendations["execution_time"],
            "confidence_score": recommendations["confidence"],
            "similar_tasks_count": len(similar_tasks),
            "learning_source": "knowledge_graph" if similar_tasks else "intelligent_analysis"
        }
    
    def _generate_task_signature(self, task_description: str) -> str:
        """Generate a signature for task similarity matching"""
        # Extract key terms and normalize
        words = task_description.lower().split()
        key_terms = [
            word for word in words 
            if len(word) > 3 and word not in {'create', 'make', 'build', 'develop', 'write'}
        ]
        return "_".join(sorted(key_terms[:5]))  # Top 5 key terms
    
    async def _find_similar_tasks(self, task_signature: str, description: str) -> List[TaskNode]:
        """Find similar tasks in the knowledge graph"""
        similar = []
        
        for task_node in self.task_nodes.values():
            # Calculate similarity score
            similarity = self._calculate_task_similarity(description, task_node.task_description)
            
            if similarity > 0.6:  # 60% similarity threshold
                similar.append(task_node)
        
        # Sort by success rate and recency
        similar.sort(key=lambda x: (x.success_rate, -x.execution_count), reverse=True)
        return similar[:5]  # Top 5 similar tasks
    
    def _calculate_task_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity between two task descriptions"""
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    async def _get_graph_recommendations(self, similar_tasks: List[TaskNode]) -> Dict[str, Any]:
        """Get recommendations based on similar tasks in the graph"""
        capability_scores = defaultdict(lambda: {"score": 0.0, "count": 0})
        total_success_rate = 0.0
        total_execution_time = 0.0
        
        for task in similar_tasks:
            total_success_rate += task.success_rate
            total_execution_time += task.avg_execution_time
            
            # Score capabilities based on success with similar tasks
            for cap in task.preferred_capabilities:
                edge_key = (task.task_id, cap)
                if edge_key in self.task_capability_edges:
                    edge = self.task_capability_edges[edge_key]
                    capability_scores[cap]["score"] += edge.success_rate * edge.confidence
                    capability_scores[cap]["count"] += 1
        
        # Calculate weighted averages
        num_tasks = len(similar_tasks)
        avg_success_rate = total_success_rate / num_tasks if num_tasks > 0 else 0.8
        avg_execution_time = total_execution_time / num_tasks if num_tasks > 0 else 60.0
        
        # Rank capabilities
        ranked_capabilities = []
        for cap, data in capability_scores.items():
            if data["count"] > 0:
                avg_score = data["score"] / data["count"]
                ranked_capabilities.append((cap, avg_score))
        
        ranked_capabilities.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "capabilities": [cap for cap, _ in ranked_capabilities[:5]],
            "success_rate": min(avg_success_rate, 0.95),
            "execution_time": avg_execution_time,
            "confidence": min(num_tasks / 10.0, 0.95)  # More similar tasks = higher confidence
        }
    
    async def _analyze_new_task_type(self, task_description: str) -> Dict[str, Any]:
        """Intelligent analysis for new task types"""
        description_lower = task_description.lower()
        
        # Intelligent capability mapping based on content analysis
        capability_scores = {}
        
        # Mathematical content
        math_indicators = ["calculate", "math", "equation", "derivative", "integral", "factorial", "solve"]
        math_score = sum(1 for indicator in math_indicators if indicator in description_lower)
        if math_score > 0:
            capability_scores["mathematics"] = min(math_score / 3.0, 1.0)
        
        # Code-related content  
        code_indicators = ["function", "code", "program", "python", "javascript", "algorithm"]
        code_score = sum(1 for indicator in code_indicators if indicator in description_lower)
        if code_score > 0:
            capability_scores["code_quality"] = min(code_score / 3.0, 1.0)
        
        # Web-related content
        web_indicators = ["website", "web", "html", "css", "responsive", "ui", "interface"]
        web_score = sum(1 for indicator in web_indicators if indicator in description_lower)
        if web_score > 0:
            capability_scores["web_verification"] = min(web_score / 3.0, 1.0)
        
        # Data analysis content
        data_indicators = ["data", "analysis", "chart", "graph", "visualization", "csv", "dataset"]
        data_score = sum(1 for indicator in data_indicators if indicator in description_lower)
        if data_score > 0:
            capability_scores["data_analysis"] = min(data_score / 3.0, 1.0)
        
        # Language processing content
        lang_indicators = ["text", "language", "grammar", "writing", "translate", "summarize"]
        lang_score = sum(1 for indicator in lang_indicators if indicator in description_lower)
        if lang_score > 0:
            capability_scores["language"] = min(lang_score / 3.0, 1.0)
        
        # Rank and return top capabilities
        ranked_caps = sorted(capability_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "capabilities": [cap for cap, score in ranked_caps if score > 0.3][:3],
            "success_rate": 0.8,  # Conservative estimate for new tasks
            "execution_time": 90.0,  # Conservative estimate
            "confidence": 0.6  # Lower confidence for new task types
        }
    
    async def learn_from_execution(self, execution_result: Dict[str, Any]):
        """
        Learn from execution results to improve future recommendations.
        This is the key "real-time" aspect that updates the knowledge graph.
        """
        task_desc = execution_result.get("task_description", "")
        capabilities_used = execution_result.get("capabilities_used", [])
        success = execution_result.get("success", False)
        execution_time = execution_result.get("execution_time", 0.0)
        quality_score = execution_result.get("quality_score", 0.0)
        
        # Create or update task node
        task_id = self._generate_task_signature(task_desc)
        
        if task_id not in self.task_nodes:
            self.task_nodes[task_id] = TaskNode(
                task_id=task_id,
                task_description=task_desc,
                task_type=execution_result.get("task_type", "general"),
                complexity=execution_result.get("complexity", "moderate")
            )
        
        task_node = self.task_nodes[task_id]
        
        # Update task node with execution results
        task_node.execution_count += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.3  # Learning rate
        task_node.success_rate = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * task_node.success_rate
        )
        
        # Update execution time (exponential moving average)
        task_node.avg_execution_time = (
            alpha * execution_time + 
            (1 - alpha) * task_node.avg_execution_time
        )
        
        # Update capability relationships
        for capability in capabilities_used:
            await self._update_capability_relationship(
                task_id, capability, success, quality_score, execution_time
            )
        
        # Store execution history for trend analysis
        self.execution_history.append({
            "timestamp": time.time(),
            "task_id": task_id,
            "capabilities": capabilities_used,
            "success": success,
            "quality_score": quality_score,
            "execution_time": execution_time
        })
        
        # Keep history manageable
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-800:]  # Keep last 800
        
        logger.info(f"📊 Knowledge graph updated: Task {task_id}, Success: {success}, Quality: {quality_score:.2f}")
    
    async def _update_capability_relationship(
        self, 
        task_id: str, 
        capability: str, 
        success: bool, 
        quality_score: float, 
        execution_time: float
    ):
        """Update the relationship between a task and capability"""
        
        # Update capability node
        if capability not in self.capability_nodes:
            self.capability_nodes[capability] = CapabilityNode(
                capability_id=capability,
                capability_name=capability
            )
        
        cap_node = self.capability_nodes[capability]
        cap_node.usage_count += 1
        
        # Update capability success rate
        alpha = 0.3
        cap_node.success_rate = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * cap_node.success_rate
        )
        
        # Update performance score
        cap_node.avg_performance_score = (
            alpha * quality_score + 
            (1 - alpha) * cap_node.avg_performance_score
        )
        
        # Update task-capability edge
        edge_key = (task_id, capability)
        
        if edge_key not in self.task_capability_edges:
            self.task_capability_edges[edge_key] = TaskCapabilityEdge(
                task_id=task_id,
                capability_id=capability,
                success_rate=0.8,
                performance_score=0.8,
                execution_count=0,
                last_used=time.time(),
                confidence=0.5
            )
        
        edge = self.task_capability_edges[edge_key]
        edge.execution_count += 1
        edge.last_used = time.time()
        
        # Update edge metrics
        edge.success_rate = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * edge.success_rate
        )
        
        edge.performance_score = (
            alpha * quality_score + 
            (1 - alpha) * edge.performance_score
        )
        
        # Increase confidence with more usage
        edge.confidence = min(edge.execution_count / 10.0, 0.95)
        
        # Update task node preferred/failed capabilities
        task_node = self.task_nodes[task_id]
        
        if success and quality_score > 0.8:
            if capability not in task_node.preferred_capabilities:
                task_node.preferred_capabilities.append(capability)
            # Remove from failed if it was there
            if capability in task_node.failed_capabilities:
                task_node.failed_capabilities.remove(capability)
        elif not success or quality_score < 0.5:
            if capability not in task_node.failed_capabilities:
                task_node.failed_capabilities.append(capability)
    
    def get_knowledge_graph_metrics(self) -> Dict[str, Any]:
        """Get metrics about the knowledge graph state"""
        return {
            "total_tasks": len(self.task_nodes),
            "total_capabilities": len(self.capability_nodes),
            "total_relationships": len(self.task_capability_edges),
            "execution_history_size": len(self.execution_history),
            "avg_task_success_rate": sum(t.success_rate for t in self.task_nodes.values()) / max(len(self.task_nodes), 1),
            "avg_capability_success_rate": sum(c.success_rate for c in self.capability_nodes.values()) / max(len(self.capability_nodes), 1),
            "most_used_capabilities": sorted(
                [(c.capability_name, c.usage_count) for c in self.capability_nodes.values()],
                key=lambda x: x[1], reverse=True
            )[:5]
        } 