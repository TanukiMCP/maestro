"""
Adaptive Error Handling System for MAESTRO Protocol

Provides LLM-driven error detection, approach reconsideration, and adaptive orchestration
with temporal context awareness for enhanced decision-making.

Core Principle: Intelligent Error Recovery > Simple Retry Mechanisms
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for adaptive handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReconsiderationTrigger(Enum):
    """Triggers for approach reconsideration"""
    VALIDATION_FAILURE = "validation_failure"
    TOOL_UNAVAILABILITY = "tool_unavailability"
    SUCCESS_CRITERIA_UNVERIFIABLE = "success_criteria_unverifiable"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    TEMPORAL_CONTEXT_SHIFT = "temporal_context_shift"
    QUALITY_THRESHOLD_UNMET = "quality_threshold_unmet"


@dataclass
class TemporalContext:
    """Temporal context for information currency and relevance"""
    current_timestamp: datetime
    information_cutoff: Optional[datetime]
    task_deadline: Optional[datetime]
    context_freshness_required: bool
    temporal_relevance_window: str  # e.g., "24h", "1w", "1m"
    
    def is_information_current(self, info_timestamp: datetime) -> bool:
        """Check if information is current based on temporal context"""
        if not self.context_freshness_required:
            return True
        
        # Parse temporal relevance window
        if self.temporal_relevance_window.endswith('h'):
            hours = int(self.temporal_relevance_window[:-1])
            cutoff = self.current_timestamp.replace(hour=self.current_timestamp.hour - hours)
        elif self.temporal_relevance_window.endswith('d'):
            days = int(self.temporal_relevance_window[:-1])
            cutoff = self.current_timestamp.replace(day=self.current_timestamp.day - days)
        else:
            cutoff = self.current_timestamp.replace(day=self.current_timestamp.day - 1)  # Default 1 day
        
        return info_timestamp >= cutoff


@dataclass
class ErrorContext:
    """Context for error analysis and handling"""
    error_id: str
    severity: ErrorSeverity
    trigger: ReconsiderationTrigger
    error_message: str
    failed_component: str
    temporal_context: TemporalContext
    available_tools: List[str]
    attempted_approaches: List[str]
    success_criteria: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class ReconsiderationResult:
    """Result of approach reconsideration"""
    should_reconsider: bool
    alternative_approaches: List[Dict[str, Any]]
    modified_success_criteria: List[Dict[str, Any]]
    recommended_tools: List[str]
    temporal_adjustments: Dict[str, Any]
    reasoning: str
    confidence_score: float


class AdaptiveErrorHandler:
    """
    Adaptive error handling system with LLM-driven approach reconsideration
    """
    
    def __init__(self):
        self.name = "Adaptive Error Handler"
        self.version = "1.0.0"
        self.error_history: List[ErrorContext] = []
        self.reconsideration_patterns = self._initialize_reconsideration_patterns()
        
    def _initialize_reconsideration_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for approach reconsideration"""
        return {
            "validation_failure": {
                "triggers": ["missing_validation_tools", "criteria_too_strict", "approach_mismatch"],
                "strategies": ["relax_criteria", "find_alternative_validation", "modify_approach"],
                "confidence_threshold": 0.7
            },
            "tool_unavailability": {
                "triggers": ["tool_not_found", "permission_denied", "service_unavailable"],
                "strategies": ["find_fallback_tool", "modify_workflow", "change_approach"],
                "confidence_threshold": 0.8
            },
            "success_criteria_unverifiable": {
                "triggers": ["no_validation_method", "criteria_ambiguous", "tools_insufficient"],
                "strategies": ["redefine_criteria", "find_proxy_metrics", "manual_validation"],
                "confidence_threshold": 0.6
            },
            "temporal_context_shift": {
                "triggers": ["information_outdated", "deadline_approached", "context_changed"],
                "strategies": ["refresh_information", "adjust_timeline", "modify_scope"],
                "confidence_threshold": 0.8
            }
        }
    
    async def analyze_error_context(
        self, 
        error_details: Dict[str, Any],
        temporal_context: TemporalContext,
        available_tools: List[str],
        success_criteria: List[Dict[str, Any]]
    ) -> ErrorContext:
        """
        Analyze error context for intelligent handling
        
        Args:
            error_details: Details about the error that occurred
            temporal_context: Current temporal context
            available_tools: Tools available for execution
            success_criteria: Current success criteria
            
        Returns:
            Structured error context for analysis
        """
        logger.info("🔍 Analyzing error context for adaptive handling...")
        
        # Determine error severity and trigger
        severity = self._assess_error_severity(error_details)
        trigger = self._identify_reconsideration_trigger(error_details)
        
        error_context = ErrorContext(
            error_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            severity=severity,
            trigger=trigger,
            error_message=error_details.get("message", "Unknown error"),
            failed_component=error_details.get("component", "Unknown"),
            temporal_context=temporal_context,
            available_tools=available_tools,
            attempted_approaches=error_details.get("attempted_approaches", []),
            success_criteria=success_criteria,
            metadata=error_details.get("metadata", {})
        )
        
        self.error_history.append(error_context)
        logger.info(f"📊 Error context analyzed: {trigger.value} ({severity.value} severity)")
        
        return error_context
    
    async def should_reconsider_approach(self, error_context: ErrorContext) -> ReconsiderationResult:
        """
        Determine if approach should be reconsidered and provide alternatives
        
        Args:
            error_context: Analyzed error context
            
        Returns:
            Reconsideration result with alternatives and recommendations
        """
        logger.info("🤔 Evaluating need for approach reconsideration...")
        
        # Check reconsideration patterns
        pattern_key = error_context.trigger.value
        pattern = self.reconsideration_patterns.get(pattern_key, {})
        
        # Analyze error frequency and patterns
        similar_errors = [e for e in self.error_history 
                         if e.trigger == error_context.trigger 
                         and e.failed_component == error_context.failed_component]
        
        should_reconsider = False
        confidence_score = 0.0
        reasoning = ""
        
        # Decision logic based on error context
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            should_reconsider = True
            confidence_score = 0.9
            reasoning = f"High/Critical severity error ({error_context.trigger.value}) requires approach reconsideration"
        elif len(similar_errors) >= 2:
            should_reconsider = True
            confidence_score = 0.8
            reasoning = f"Repeated similar errors ({len(similar_errors)} times) suggest approach needs modification"
        elif error_context.trigger == ReconsiderationTrigger.SUCCESS_CRITERIA_UNVERIFIABLE:
            should_reconsider = True
            confidence_score = 0.85
            reasoning = "Success criteria cannot be verified with available tools - approach modification required"
        elif error_context.trigger == ReconsiderationTrigger.TEMPORAL_CONTEXT_SHIFT:
            should_reconsider = True
            confidence_score = 0.7
            reasoning = "Temporal context has shifted - approach may need updating for current information"
        
        # Generate alternative approaches if reconsideration is recommended
        alternative_approaches = []
        modified_success_criteria = []
        recommended_tools = []
        temporal_adjustments = {}
        
        if should_reconsider:
            alternative_approaches = await self._generate_alternative_approaches(error_context)
            modified_success_criteria = await self._modify_success_criteria(error_context)
            recommended_tools = await self._recommend_alternative_tools(error_context)
            temporal_adjustments = await self._suggest_temporal_adjustments(error_context)
        
        result = ReconsiderationResult(
            should_reconsider=should_reconsider,
            alternative_approaches=alternative_approaches,
            modified_success_criteria=modified_success_criteria,
            recommended_tools=recommended_tools,
            temporal_adjustments=temporal_adjustments,
            reasoning=reasoning,
            confidence_score=confidence_score
        )
        
        logger.info(f"📋 Reconsideration analysis complete: {'RECOMMENDED' if should_reconsider else 'NOT NEEDED'} ({confidence_score:.2f} confidence)")
        
        return result
    
    async def _generate_alternative_approaches(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Generate alternative approaches based on error context"""
        alternatives = []
        
        trigger = error_context.trigger
        available_tools = error_context.available_tools
        
        if trigger == ReconsiderationTrigger.TOOL_UNAVAILABILITY:
            alternatives.extend([
                {
                    "approach": "fallback_tool_strategy",
                    "description": "Use built-in MAESTRO tools (maestro_search, maestro_scrape, maestro_execute) as fallbacks",
                    "tools_required": ["maestro_search", "maestro_scrape", "maestro_execute"],
                    "confidence": 0.8
                },
                {
                    "approach": "manual_validation_strategy",
                    "description": "Modify workflow to use manual validation where tools are unavailable",
                    "tools_required": [],
                    "confidence": 0.6
                }
            ])
        
        elif trigger == ReconsiderationTrigger.SUCCESS_CRITERIA_UNVERIFIABLE:
            alternatives.extend([
                {
                    "approach": "proxy_metrics_strategy",
                    "description": "Define proxy metrics that can be verified with available tools",
                    "tools_required": available_tools,
                    "confidence": 0.7
                },
                {
                    "approach": "staged_validation_strategy", 
                    "description": "Break down validation into stages using available verification methods",
                    "tools_required": available_tools,
                    "confidence": 0.8
                }
            ])
        
        elif trigger == ReconsiderationTrigger.TEMPORAL_CONTEXT_SHIFT:
            alternatives.extend([
                {
                    "approach": "information_refresh_strategy",
                    "description": "Use maestro_search to gather current information before proceeding",
                    "tools_required": ["maestro_search", "web_search"],
                    "confidence": 0.9
                },
                {
                    "approach": "adaptive_timeline_strategy",
                    "description": "Adjust timeline and scope based on current temporal context",
                    "tools_required": [],
                    "confidence": 0.7
                }
            ])
        
        return alternatives
    
    async def _modify_success_criteria(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Modify success criteria to be verifiable with available tools"""
        modified_criteria = []
        
        for criterion in error_context.success_criteria:
            # Check if criterion can be verified with available tools
            verification_possible = any(tool in error_context.available_tools 
                                      for tool in criterion.get("validation_tools", []))
            
            if not verification_possible:
                # Modify criterion to use available tools
                modified_criterion = criterion.copy()
                
                # Add fallback validation methods
                if "maestro_execute" in error_context.available_tools:
                    modified_criterion["validation_tools"] = ["maestro_execute"]
                    modified_criterion["validation_method"] = "execution_based_verification"
                elif "maestro_search" in error_context.available_tools:
                    modified_criterion["validation_tools"] = ["maestro_search"]
                    modified_criterion["validation_method"] = "search_based_verification"
                else:
                    modified_criterion["validation_method"] = "manual_review"
                    modified_criterion["validation_tools"] = []
                
                modified_criteria.append(modified_criterion)
            else:
                modified_criteria.append(criterion)
        
        return modified_criteria
    
    async def _recommend_alternative_tools(self, error_context: ErrorContext) -> List[str]:
        """Recommend alternative tools based on error context"""
        recommendations = []
        
        trigger = error_context.trigger
        
        # Always recommend built-in MAESTRO tools as fallbacks
        maestro_tools = ["maestro_search", "maestro_scrape", "maestro_execute"]
        
        if trigger == ReconsiderationTrigger.TOOL_UNAVAILABILITY:
            recommendations.extend(maestro_tools)
        elif trigger == ReconsiderationTrigger.TEMPORAL_CONTEXT_SHIFT:
            recommendations.extend(["maestro_search", "web_search"])
        elif trigger == ReconsiderationTrigger.SUCCESS_CRITERIA_UNVERIFIABLE:
            recommendations.extend(["maestro_execute", "maestro_scrape"])
        
        return recommendations
    
    async def _suggest_temporal_adjustments(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Suggest temporal adjustments based on context"""
        adjustments = {}
        
        if error_context.trigger == ReconsiderationTrigger.TEMPORAL_CONTEXT_SHIFT:
            adjustments = {
                "refresh_information": True,
                "update_temporal_relevance_window": "6h",  # More recent information
                "add_information_currency_check": True,
                "enable_real_time_updates": True
            }
        
        return adjustments
    
    def _assess_error_severity(self, error_details: Dict[str, Any]) -> ErrorSeverity:
        """Assess error severity based on error details"""
        error_type = error_details.get("type", "unknown")
        impact = error_details.get("impact", "unknown")
        
        if error_type in ["validation_failure", "critical_tool_missing"]:
            return ErrorSeverity.HIGH
        elif error_type in ["tool_unavailable", "timeout"]:
            return ErrorSeverity.MEDIUM
        elif impact == "workflow_blocking":
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.LOW
    
    def _identify_reconsideration_trigger(self, error_details: Dict[str, Any]) -> ReconsiderationTrigger:
        """Identify the trigger for approach reconsideration"""
        error_type = error_details.get("type", "unknown")
        message = error_details.get("message", "").lower()
        
        if "validation" in message and "fail" in message:
            return ReconsiderationTrigger.VALIDATION_FAILURE
        elif "tool" in message and ("not found" in message or "unavailable" in message):
            return ReconsiderationTrigger.TOOL_UNAVAILABILITY
        elif "success criteria" in message and ("verify" in message or "unverifiable" in message):
            return ReconsiderationTrigger.SUCCESS_CRITERIA_UNVERIFIABLE
        elif "timeout" in message or "deadline" in message:
            return ReconsiderationTrigger.RESOURCE_CONSTRAINTS
        elif "outdated" in message or "timestamp" in message:
            return ReconsiderationTrigger.TEMPORAL_CONTEXT_SHIFT
        else:
            return ReconsiderationTrigger.QUALITY_THRESHOLD_UNMET
    
    def get_error_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of error analysis and patterns"""
        return {
            "total_errors": len(self.error_history),
            "error_by_severity": {
                severity.value: len([e for e in self.error_history if e.severity == severity])
                for severity in ErrorSeverity
            },
            "error_by_trigger": {
                trigger.value: len([e for e in self.error_history if e.trigger == trigger])
                for trigger in ReconsiderationTrigger
            },
            "recent_errors": [
                {
                    "error_id": e.error_id,
                    "trigger": e.trigger.value,
                    "severity": e.severity.value,
                    "component": e.failed_component
                }
                for e in self.error_history[-5:]  # Last 5 errors
            ]
        } 