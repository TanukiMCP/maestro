"""
Maestro Context-Aware Orchestrator - Enhanced Architecture

Provides dynamic, tool-aware orchestration that discovers available tools
and maps them to specific workflow steps with explicit usage guidance.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from .orchestrator import MAESTROOrchestrator
from .tool_discovery import ToolDiscoveryEngine, ToolInfo, IDECapability
from .tool_workflow_mapper import ToolWorkflowMapper, EnhancedWorkflowPhase, ToolUsageInstruction
from .quality_controller import QualityController

logger = logging.getLogger(__name__)


class ContextAwareOrchestrator:
    """
    Enhanced Maestro orchestrator with dynamic tool discovery and mapping.
    
    This orchestrator provides intelligent, context-aware workflow guidance
    by dynamically discovering available tools and mapping them to specific
    workflow phases with explicit usage instructions.
    """
    
    def __init__(self):
        self.base_orchestrator = MAESTROOrchestrator()
        self.tool_discovery = ToolDiscoveryEngine()
        self.tool_mapper = ToolWorkflowMapper()
        self.quality_controller = QualityController()
        
        # Cache for tool discovery results
        self._last_discovery_results = None
        self._discovery_cache_timeout = 300  # 5 minutes
        
        logger.info("🎭 Context-Aware Maestro Orchestrator initialized")
    
    async def analyze_task_with_context(
        self, 
        task_description: str,
        detail_level: str = "comprehensive",
        force_tool_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze task with full context awareness of available tools.
        
        This is the enhanced version of analyze_task_for_planning that includes
        dynamic tool discovery and mapping.
        
        Args:
            task_description: Natural language task description
            detail_level: "fast", "balanced", or "comprehensive"
            force_tool_refresh: Force refresh of tool discovery cache
            
        Returns:
            Enhanced analysis with tool-aware workflow guidance
        """
        logger.info(f"🔍 Context-aware task analysis: {task_description[:100]}...")
        
        # Step 1: Discover available tools (with caching)
        if force_tool_refresh or self.tool_discovery.is_discovery_stale(self._discovery_cache_timeout):
            logger.info("🔄 Refreshing tool discovery...")
            discovery_results = await self.tool_discovery.full_discovery_scan()
            self._last_discovery_results = discovery_results
        else:
            logger.info("📋 Using cached tool discovery results")
            discovery_results = self._last_discovery_results
        
        # Step 2: Get base task analysis
        base_analysis = await self.base_orchestrator.analyze_task_for_planning(
            task_description, detail_level
        )
        
        # Step 3: Create tool-aware enhanced workflow
        enhanced_workflow = await self._create_tool_aware_workflow(
            base_analysis,
            discovery_results,
            task_description
        )
        
        # Step 4: Generate context-aware system prompt
        context_aware_prompt = self._generate_context_aware_system_prompt(
            base_analysis,
            enhanced_workflow,
            discovery_results
        )
        
        # Step 5: Create explicit tool execution plan
        tool_execution_plan = self._create_tool_execution_plan(enhanced_workflow)
        
        return {
            **base_analysis,  # Include base analysis
            "context_aware_enhancements": {
                "tool_discovery_results": {
                    "total_servers_discovered": len(discovery_results.get("mcp_servers", {})),
                    "total_tools_available": discovery_results.get("total_tools_discovered", 0),
                    "ide_capabilities": len(discovery_results.get("ide_capabilities", [])),
                    "discovery_timestamp": discovery_results.get("discovery_timestamp")
                },
                "enhanced_workflow": enhanced_workflow,
                "context_aware_system_prompt": context_aware_prompt,
                "tool_execution_plan": tool_execution_plan
            }
        }
    
    async def create_tool_aware_execution_plan(
        self,
        task_description: str,
        phase_focus: Optional[str] = None,
        force_tool_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Create execution plan with explicit tool mappings and usage instructions.
        
        Args:
            task_description: Task to create execution plan for
            phase_focus: Specific phase to focus on (optional)
            force_tool_refresh: Force refresh of tool discovery
            
        Returns:
            Tool-aware execution plan with explicit guidance
        """
        logger.info(f"📋 Creating tool-aware execution plan...")
        
        # Get context-aware analysis
        analysis = await self.analyze_task_with_context(
            task_description, 
            detail_level="comprehensive",
            force_tool_refresh=force_tool_refresh
        )
        
        enhanced_workflow = analysis["context_aware_enhancements"]["enhanced_workflow"]
        
        if phase_focus:
            # Focus on specific phase
            target_phase = None
            for phase in enhanced_workflow:
                if phase["phase_info"]["phase"].value.lower() == phase_focus.lower():
                    target_phase = phase
                    break
            
            if target_phase:
                return {
                    "focused_phase": target_phase,
                    "explicit_tool_instructions": self._format_tool_instructions(
                        target_phase["tool_mappings"]
                    ),
                    "success_validation_steps": target_phase["phase_info"]["success_criteria"]
                }
        
        # Return comprehensive tool-aware plan
        return {
            "complete_workflow": enhanced_workflow,
            "execution_sequence": [phase["phase_info"]["phase"].value for phase in enhanced_workflow],
            "tool_summary": self._create_workflow_tool_summary(enhanced_workflow),
            "critical_dependencies": self._extract_tool_dependencies(enhanced_workflow)
        }
    
    async def get_available_tools_with_context(self) -> Dict[str, Any]:
        """
        Get comprehensive information about available tools with context.
        
        Returns:
            Detailed tool inventory with usage guidance
        """
        # Refresh tool discovery
        discovery_results = await self.tool_discovery.full_discovery_scan()
        
        # Organize tools by capability
        tools_by_capability = {}
        for tool_name, tool_info in self.tool_discovery.available_tools.items():
            capabilities = tool_info.capabilities or ["general"]
            for capability in capabilities:
                if capability not in tools_by_capability:
                    tools_by_capability[capability] = []
                tools_by_capability[capability].append({
                    "name": tool_info.name,
                    "server": tool_info.server_name,
                    "description": tool_info.description,
                    "usage_examples": tool_info.usage_examples
                })
        
        return {
            "discovery_summary": discovery_results,
            "tools_by_capability": tools_by_capability,
            "ide_capabilities": [
                {
                    "name": cap.name,
                    "category": cap.category,
                    "description": cap.description,
                    "usage_pattern": cap.usage_pattern
                }
                for cap in self.tool_discovery.ide_capabilities
            ],
            "tool_capability_map": discovery_results.get("tool_capability_map", {})
        }
    
    async def _create_tool_aware_workflow(
        self,
        base_analysis: Dict[str, Any],
        discovery_results: Dict[str, Any],
        task_description: str
    ) -> List[Dict[str, Any]]:
        """Create enhanced workflow with tool mappings."""
        
        # Get base execution phases
        base_phases = base_analysis.get("execution_phases", [])
        
        # Map tools to workflow phases
        enhanced_phases = self.tool_mapper.map_tools_to_workflow(
            workflow_phases=base_phases,
            available_tools=self.tool_discovery.available_tools,
            ide_capabilities=self.tool_discovery.ide_capabilities,
            task_context={"task_description": task_description}
        )
        
        # Convert to dict format for JSON serialization
        workflow_dicts = []
        for enhanced_phase in enhanced_phases:
            phase_dict = {
                "phase_info": {
                    "phase": enhanced_phase.phase,
                    "description": enhanced_phase.description,
                    "success_criteria": enhanced_phase.success_criteria,
                    "estimated_duration": enhanced_phase.estimated_duration,
                    "dependencies": [dep.value for dep in (enhanced_phase.phase_dependencies or [])]
                },
                "tool_mappings": [
                    {
                        "tool_name": tm.tool_name,
                        "server_name": tm.server_name,
                        "usage_command": tm.usage_command,
                        "when_to_use": tm.when_to_use,
                        "example": tm.example,
                        "priority": tm.priority,
                        "prerequisites": tm.prerequisites or [],
                        "expected_output": tm.expected_output,
                        "parameters": tm.parameters
                    }
                    for tm in enhanced_phase.tool_mappings
                ],
                "tool_summary": self.tool_mapper.get_phase_tool_summary(enhanced_phase)
            }
            workflow_dicts.append(phase_dict)
        
        return workflow_dicts
    
    def _generate_context_aware_system_prompt(
        self,
        base_analysis: Dict[str, Any],
        enhanced_workflow: List[Dict[str, Any]],
        discovery_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate system prompt with tool context awareness."""
        
        base_prompt = base_analysis.get("system_prompt_guidance", {})
        
        # Extract available tool capabilities
        available_capabilities = list(discovery_results.get("tool_capability_map", {}).keys())
        
        # Create tool-aware guidance
        tool_aware_guidance = {
            **base_prompt,
            "tool_ecosystem_context": {
                "available_mcp_servers": len(discovery_results.get("mcp_servers", {})),
                "total_tools_available": discovery_results.get("total_tools_discovered", 0),
                "key_capabilities": available_capabilities[:10],  # Top 10 capabilities
                "ide_integration": len(discovery_results.get("ide_capabilities", []))
            },
            "enhanced_approach_guidelines": [
                *base_prompt.get("approach_guidelines", []),
                "Leverage discovered MCP tools for enhanced workflow execution",
                "Use explicit tool mappings provided in the workflow phases",
                "Follow tool usage instructions and prerequisites carefully",
                "Combine MCP tools with IDE capabilities for optimal results",
                f"You have access to {discovery_results.get('total_tools_discovered', 0)} specialized tools across {len(discovery_results.get('mcp_servers', {}))} servers"
            ],
            "tool_usage_principles": [
                "Always check tool prerequisites before execution",
                "Use primary tools first, fallback to secondary tools if needed",
                "Leverage IDE capabilities alongside MCP tools",
                "Follow the explicit usage commands and examples provided",
                "Validate expected outputs after tool execution"
            ]
        }
        
        return tool_aware_guidance
    
    def _create_tool_execution_plan(self, enhanced_workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create explicit tool execution plan."""
        
        execution_steps = []
        tool_prerequisites = set()
        
        for phase_dict in enhanced_workflow:
            phase_name = phase_dict["phase_info"]["phase"].value
            tool_mappings = phase_dict["tool_mappings"]
            
            phase_steps = {
                "phase": phase_name,
                "tools_to_execute": [],
                "execution_order": []
            }
            
            # Sort tools by priority
            primary_tools = [tm for tm in tool_mappings if tm["priority"] == 1]
            secondary_tools = [tm for tm in tool_mappings if tm["priority"] == 2]
            
            # Add primary tools to execution plan
            for tool_mapping in primary_tools:
                tool_step = {
                    "step": f"Execute {tool_mapping['tool_name']}",
                    "command": tool_mapping["usage_command"],
                    "rationale": tool_mapping["when_to_use"],
                    "example": tool_mapping["example"],
                    "expected_result": tool_mapping["expected_output"]
                }
                phase_steps["tools_to_execute"].append(tool_step)
                phase_steps["execution_order"].append(tool_mapping["tool_name"])
                
                # Collect prerequisites
                tool_prerequisites.update(tool_mapping.get("prerequisites", []))
            
            # Add secondary tools as alternatives
            if secondary_tools:
                phase_steps["alternative_tools"] = [
                    {
                        "tool": tm["tool_name"],
                        "usage": tm["usage_command"],
                        "when": f"Use if primary tools are not available: {tm['when_to_use']}"
                    }
                    for tm in secondary_tools
                ]
            
            execution_steps.append(phase_steps)
        
        return {
            "execution_steps": execution_steps,
            "global_prerequisites": list(tool_prerequisites),
            "total_phases": len(execution_steps),
            "estimated_total_tools": sum(len(step["tools_to_execute"]) for step in execution_steps)
        }
    
    def _format_tool_instructions(self, tool_mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tool mappings into explicit instructions."""
        
        instructions = []
        
        for tm in tool_mappings:
            instruction = {
                "tool": tm["tool_name"],
                "server": tm["server_name"],
                "instruction": f"Use {tm['tool_name']} to {tm['when_to_use'].lower()}",
                "exact_command": tm["usage_command"],
                "example": tm["example"],
                "prerequisites": tm.get("prerequisites", []),
                "expected_output": tm["expected_output"],
                "priority_level": "Primary" if tm["priority"] == 1 else "Secondary" if tm["priority"] == 2 else "Fallback"
            }
            instructions.append(instruction)
        
        return instructions
    
    def _create_workflow_tool_summary(self, enhanced_workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of tools across the entire workflow."""
        
        all_tools = set()
        tools_by_server = {}
        tools_by_priority = {"primary": 0, "secondary": 0, "fallback": 0}
        
        for phase_dict in enhanced_workflow:
            for tm in phase_dict["tool_mappings"]:
                tool_name = tm["tool_name"]
                server_name = tm["server_name"]
                priority = tm["priority"]
                
                all_tools.add(tool_name)
                
                if server_name not in tools_by_server:
                    tools_by_server[server_name] = []
                tools_by_server[server_name].append(tool_name)
                
                if priority == 1:
                    tools_by_priority["primary"] += 1
                elif priority == 2:
                    tools_by_priority["secondary"] += 1
                else:
                    tools_by_priority["fallback"] += 1
        
        return {
            "total_unique_tools": len(all_tools),
            "tools_by_server": tools_by_server,
            "tools_by_priority": tools_by_priority,
            "server_count": len(tools_by_server),
            "workflow_complexity": "High" if len(all_tools) > 10 else "Medium" if len(all_tools) > 5 else "Low"
        }
    
    def _extract_tool_dependencies(self, enhanced_workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract critical tool dependencies from the workflow."""
        
        dependencies = []
        
        for phase_dict in enhanced_workflow:
            phase_name = phase_dict["phase_info"]["phase"].value
            prerequisites = set()
            
            for tm in phase_dict["tool_mappings"]:
                prerequisites.update(tm.get("prerequisites", []))
            
            if prerequisites:
                dependencies.append({
                    "phase": phase_name,
                    "requires": list(prerequisites),
                    "critical": len(prerequisites) > 2
                })
        
        return dependencies
    
    # Delegate other methods to base orchestrator for backward compatibility
    async def get_available_templates(self) -> List[str]:
        """Get available templates."""
        return self.base_orchestrator.get_available_templates()
    
    def get_template_details(self, template_name: str) -> Dict[str, Any]:
        """Get template details."""
        return self.base_orchestrator.get_template_details(template_name) 