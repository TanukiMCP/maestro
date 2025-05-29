"""
Maestro Tool Workflow Mapper

Intelligently maps available tools to workflow phases and provides
explicit tool usage guidance for enhanced orchestration.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from .tool_discovery import ToolInfo, IDECapability

logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """Standard workflow phases for tool mapping."""
    ANALYSIS = "Analysis"
    IMPLEMENTATION = "Implementation" 
    TESTING = "Testing"
    QUALITY_ASSURANCE = "Quality_Assurance"
    DOCUMENTATION = "Documentation"
    DEPLOYMENT = "Deployment"


@dataclass
class ToolUsageInstruction:
    """Explicit instruction for using a tool in a workflow phase."""
    tool_name: str
    server_name: str
    usage_command: str
    when_to_use: str
    parameters: Dict[str, Any]
    example: str
    priority: int = 1  # 1=primary, 2=secondary, 3=fallback
    prerequisites: List[str] = None
    expected_output: str = ""


@dataclass 
class EnhancedWorkflowPhase:
    """Workflow phase enhanced with tool mapping and usage instructions."""
    phase: WorkflowPhase
    description: str
    success_criteria: List[str]
    tool_mappings: List[ToolUsageInstruction]
    phase_dependencies: List[WorkflowPhase] = None
    estimated_duration: int = 0
    quality_gates: List[str] = None


class ToolWorkflowMapper:
    """
    Intelligently map available tools to workflow phases.
    
    Provides explicit, context-aware tool usage guidance that enhances
    the LLM's ability to execute workflows effectively.
    """
    
    def __init__(self):
        self.capability_to_phase_map = self._build_capability_phase_map()
        self.tool_priority_rules = self._build_tool_priority_rules()
        self.phase_tool_requirements = self._build_phase_requirements()
    
    def map_tools_to_workflow(
        self,
        workflow_phases: List[Dict[str, Any]],
        available_tools: Dict[str, ToolInfo],
        ide_capabilities: List[IDECapability],
        task_context: Optional[Dict[str, Any]] = None
    ) -> List[EnhancedWorkflowPhase]:
        """
        Map available tools to workflow phases with explicit usage instructions.
        
        Args:
            workflow_phases: List of workflow phase definitions
            available_tools: Dict of discovered tools keyed by name
            ide_capabilities: List of available IDE capabilities
            task_context: Additional context about the task
            
        Returns:
            List of enhanced workflow phases with tool mappings
        """
        logger.info(f"🗺️ Mapping {len(available_tools)} tools to {len(workflow_phases)} workflow phases")
        
        enhanced_phases = []
        
        for phase_dict in workflow_phases:
            try:
                # Convert to WorkflowPhase enum
                phase_enum = WorkflowPhase(phase_dict.get("phase", "Analysis"))
                
                # Map tools to this phase
                tool_mappings = self._map_tools_to_phase(
                    phase_enum,
                    available_tools,
                    ide_capabilities,
                    task_context
                )
                
                # Create enhanced phase
                enhanced_phase = EnhancedWorkflowPhase(
                    phase=phase_enum,
                    description=phase_dict.get("description", ""),
                    success_criteria=phase_dict.get("success_criteria", []),
                    tool_mappings=tool_mappings,
                    phase_dependencies=self._get_phase_dependencies(phase_enum),
                    estimated_duration=phase_dict.get("estimated_duration", 0),
                    quality_gates=phase_dict.get("quality_gates", [])
                )
                
                enhanced_phases.append(enhanced_phase)
                
                logger.info(f"✅ Mapped {len(tool_mappings)} tools to {phase_enum.value} phase")
                
            except Exception as e:
                logger.error(f"❌ Failed to map tools to phase {phase_dict}: {e}")
                continue
        
        return enhanced_phases
    
    def _map_tools_to_phase(
        self,
        phase: WorkflowPhase,
        available_tools: Dict[str, ToolInfo],
        ide_capabilities: List[IDECapability], 
        task_context: Optional[Dict[str, Any]] = None
    ) -> List[ToolUsageInstruction]:
        """Map specific tools to a workflow phase."""
        
        # Get required capabilities for this phase
        required_capabilities = self.phase_tool_requirements.get(phase, set())
        
        tool_mappings = []
        
        # Map MCP tools
        for tool_name, tool_info in available_tools.items():
            if self._is_tool_relevant_to_phase(tool_info, phase, required_capabilities):
                usage_instruction = self._create_tool_usage_instruction(
                    tool_info, phase, task_context
                )
                tool_mappings.append(usage_instruction)
        
        # Map IDE capabilities  
        for ide_cap in ide_capabilities:
            if self._is_ide_capability_relevant_to_phase(ide_cap, phase):
                usage_instruction = self._create_ide_usage_instruction(
                    ide_cap, phase, task_context
                )
                tool_mappings.append(usage_instruction)
        
        # Sort by priority and relevance
        tool_mappings.sort(key=lambda x: (x.priority, -len(x.usage_command)))
        
        return tool_mappings
    
    def _is_tool_relevant_to_phase(
        self, 
        tool_info: ToolInfo, 
        phase: WorkflowPhase,
        required_capabilities: Set[str]
    ) -> bool:
        """Determine if a tool is relevant to a specific workflow phase."""
        
        tool_capabilities = set(tool_info.capabilities or [])
        
        # Check if tool capabilities match phase requirements
        if tool_capabilities.intersection(required_capabilities):
            return True
        
        # Additional heuristics based on tool name and description
        tool_name_lower = tool_info.name.lower()
        tool_desc_lower = tool_info.description.lower()
        
        phase_keywords = {
            WorkflowPhase.ANALYSIS: ["analyze", "inspect", "read", "examine", "parse"],
            WorkflowPhase.IMPLEMENTATION: ["create", "write", "build", "generate", "implement", "edit"],
            WorkflowPhase.TESTING: ["test", "validate", "verify", "check", "execute", "run"],
            WorkflowPhase.QUALITY_ASSURANCE: ["lint", "format", "quality", "review", "audit"],
            WorkflowPhase.DOCUMENTATION: ["document", "readme", "docs", "comment", "explain"],
            WorkflowPhase.DEPLOYMENT: ["deploy", "publish", "release", "package", "build"]
        }
        
        keywords = phase_keywords.get(phase, [])
        
        for keyword in keywords:
            if keyword in tool_name_lower or keyword in tool_desc_lower:
                return True
        
        return False
    
    def _is_ide_capability_relevant_to_phase(
        self,
        ide_cap: IDECapability,
        phase: WorkflowPhase
    ) -> bool:
        """Determine if an IDE capability is relevant to a workflow phase."""
        
        phase_ide_mappings = {
            WorkflowPhase.ANALYSIS: ["navigation", "file_operations", "ai_assistance"],
            WorkflowPhase.IMPLEMENTATION: ["editing", "file_operations", "ai_assistance"],
            WorkflowPhase.TESTING: ["execution", "commands"],
            WorkflowPhase.QUALITY_ASSURANCE: ["commands", "extensions"],
            WorkflowPhase.DOCUMENTATION: ["editing", "ai_assistance"],
            WorkflowPhase.DEPLOYMENT: ["execution", "commands"]
        }
        
        relevant_categories = phase_ide_mappings.get(phase, [])
        return ide_cap.category in relevant_categories
    
    def _create_tool_usage_instruction(
        self,
        tool_info: ToolInfo,
        phase: WorkflowPhase,
        task_context: Optional[Dict[str, Any]] = None
    ) -> ToolUsageInstruction:
        """Create detailed usage instruction for a tool in a specific phase."""
        
        # Determine priority based on tool relevance to phase
        priority = self._calculate_tool_priority(tool_info, phase)
        
        # Generate usage command
        usage_command = self._generate_usage_command(tool_info)
        
        # Determine when to use this tool
        when_to_use = self._generate_when_to_use_guidance(tool_info, phase)
        
        # Create example
        example = self._generate_usage_example(tool_info, phase, task_context)
        
        # Determine prerequisites
        prerequisites = self._determine_prerequisites(tool_info, phase)
        
        # Expected output
        expected_output = self._generate_expected_output(tool_info, phase)
        
        return ToolUsageInstruction(
            tool_name=tool_info.name,
            server_name=tool_info.server_name,
            usage_command=usage_command,
            when_to_use=when_to_use,
            parameters=tool_info.input_schema.get("properties", {}),
            example=example,
            priority=priority,
            prerequisites=prerequisites,
            expected_output=expected_output
        )
    
    def _create_ide_usage_instruction(
        self,
        ide_cap: IDECapability,
        phase: WorkflowPhase,
        task_context: Optional[Dict[str, Any]] = None
    ) -> ToolUsageInstruction:
        """Create usage instruction for an IDE capability."""
        
        return ToolUsageInstruction(
            tool_name=f"ide_{ide_cap.name}",
            server_name="IDE",
            usage_command=ide_cap.usage_pattern,
            when_to_use=f"Use for {ide_cap.description.lower()} during {phase.value.lower()}",
            parameters=ide_cap.parameters or {},
            example=f"Example: {ide_cap.usage_pattern}",
            priority=2,  # IDE capabilities typically secondary to MCP tools
            prerequisites=[],
            expected_output=f"Enhanced {phase.value.lower()} through IDE integration"
        )
    
    def _calculate_tool_priority(self, tool_info: ToolInfo, phase: WorkflowPhase) -> int:
        """Calculate priority score for a tool in a specific phase."""
        
        # Start with default priority
        priority = 2
        
        # Check priority rules
        tool_name_lower = tool_info.name.lower()
        
        # High priority tools for specific phases
        high_priority_mappings = {
            WorkflowPhase.ANALYSIS: ["read", "analyze", "inspect"],
            WorkflowPhase.IMPLEMENTATION: ["create", "write", "edit"],
            WorkflowPhase.TESTING: ["test", "execute", "run"],
            WorkflowPhase.QUALITY_ASSURANCE: ["lint", "validate"],
            WorkflowPhase.DOCUMENTATION: ["document", "generate"],
            WorkflowPhase.DEPLOYMENT: ["deploy", "build"]
        }
        
        high_priority_keywords = high_priority_mappings.get(phase, [])
        
        for keyword in high_priority_keywords:
            if keyword in tool_name_lower:
                priority = 1
                break
        
        # Apply priority rules from configuration
        for rule in self.tool_priority_rules.get(phase, []):
            if rule["pattern"] in tool_name_lower:
                priority = rule["priority"]
                break
        
        return priority
    
    def _generate_usage_command(self, tool_info: ToolInfo) -> str:
        """Generate the command syntax for using the tool."""
        
        # Get required parameters
        properties = tool_info.input_schema.get("properties", {})
        required = tool_info.input_schema.get("required", [])
        
        # Build parameter list
        params = []
        for param_name, param_info in properties.items():
            if param_name in required:
                params.append(f"'{param_name}': 'value'")
            else:
                params.append(f"'{param_name}': 'optional_value'  # optional")
        
        if params:
            param_str = "{" + ", ".join(params) + "}"
            return f"{tool_info.name}({param_str})"
        else:
            return f"{tool_info.name}()"
    
    def _generate_when_to_use_guidance(self, tool_info: ToolInfo, phase: WorkflowPhase) -> str:
        """Generate guidance on when to use this tool in the phase."""
        
        phase_guidance = {
            WorkflowPhase.ANALYSIS: "Use when you need to examine, read, or analyze existing content",
            WorkflowPhase.IMPLEMENTATION: "Use when creating, modifying, or building new content",
            WorkflowPhase.TESTING: "Use when validating, testing, or verifying functionality", 
            WorkflowPhase.QUALITY_ASSURANCE: "Use when checking quality, formatting, or standards compliance",
            WorkflowPhase.DOCUMENTATION: "Use when creating or updating documentation",
            WorkflowPhase.DEPLOYMENT: "Use when packaging, deploying, or releasing the project"
        }
        
        base_guidance = phase_guidance.get(phase, "Use as needed for this phase")
        
        # Add tool-specific guidance based on capabilities
        tool_capabilities = tool_info.capabilities or []
        
        if "file_operations" in tool_capabilities:
            base_guidance += ". Ideal for file-based operations"
        if "code_execution" in tool_capabilities:
            base_guidance += ". Use for running code or commands"
        if "version_control" in tool_capabilities:
            base_guidance += ". Use for version control operations"
        
        return base_guidance
    
    def _generate_usage_example(
        self,
        tool_info: ToolInfo,
        phase: WorkflowPhase,
        task_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a specific usage example for the tool in this phase."""
        
        # Use provided examples if available
        if tool_info.usage_examples:
            return tool_info.usage_examples[0]
        
        # Generate contextual examples based on phase and tool type
        tool_name = tool_info.name
        
        phase_examples = {
            WorkflowPhase.ANALYSIS: {
                "read_file": "read_file({'path': './src/main.py'}) # Analyze existing code",
                "git_status": "git_status() # Check repository state",
                "execute_python": "execute_python({'code': 'import ast; print(ast.dump(...))'}) # Parse code structure"
            },
            WorkflowPhase.IMPLEMENTATION: {
                "write_file": "write_file({'path': './src/new_feature.py', 'content': '# New feature implementation'}) # Create new module",
                "create_directory": "create_directory({'path': './tests'}) # Set up test structure",
                "execute_python": "execute_python({'code': 'print(\"Testing implementation\")'}) # Test code snippets"
            },
            WorkflowPhase.TESTING: {
                "execute_python": "execute_python({'code': 'pytest tests/'}) # Run test suite",
                "read_file": "read_file({'path': './tests/test_results.txt'}) # Check test output"
            }
        }
        
        phase_specific = phase_examples.get(phase, {})
        if tool_name in phase_specific:
            return phase_specific[tool_name]
        
        # Generic example
        return f"{tool_name}({{}}) # Use during {phase.value.lower()}"
    
    def _determine_prerequisites(self, tool_info: ToolInfo, phase: WorkflowPhase) -> List[str]:
        """Determine prerequisites for using this tool in this phase."""
        
        prerequisites = []
        
        # Tool-specific prerequisites
        if "git" in tool_info.name.lower():
            prerequisites.append("Git repository initialized")
        
        if "test" in tool_info.name.lower():
            prerequisites.append("Test files exist")
        
        if "file" in tool_info.name.lower():
            prerequisites.append("Target directory exists")
        
        # Phase-specific prerequisites
        if phase == WorkflowPhase.TESTING:
            prerequisites.append("Implementation phase completed")
        elif phase == WorkflowPhase.DEPLOYMENT:
            prerequisites.append("Testing phase passed")
        
        return prerequisites
    
    def _generate_expected_output(self, tool_info: ToolInfo, phase: WorkflowPhase) -> str:
        """Generate description of expected output from the tool."""
        
        if "read" in tool_info.name.lower():
            return "File content or data for analysis"
        elif "write" in tool_info.name.lower():
            return "Confirmation of successful file creation/modification"
        elif "execute" in tool_info.name.lower():
            return "Execution results and output"
        elif "git" in tool_info.name.lower():
            return "Version control status or operation confirmation"
        elif "test" in tool_info.name.lower():
            return "Test results and validation status"
        else:
            return f"Results relevant to {phase.value.lower()} phase"
    
    def _build_capability_phase_map(self) -> Dict[str, Set[WorkflowPhase]]:
        """Build mapping from capabilities to relevant workflow phases."""
        return {
            "file_operations": {WorkflowPhase.ANALYSIS, WorkflowPhase.IMPLEMENTATION, WorkflowPhase.DOCUMENTATION},
            "code_execution": {WorkflowPhase.IMPLEMENTATION, WorkflowPhase.TESTING},
            "version_control": {WorkflowPhase.IMPLEMENTATION, WorkflowPhase.QUALITY_ASSURANCE, WorkflowPhase.DEPLOYMENT},
            "testing": {WorkflowPhase.TESTING, WorkflowPhase.QUALITY_ASSURANCE},
            "documentation": {WorkflowPhase.DOCUMENTATION},
            "deployment": {WorkflowPhase.DEPLOYMENT},
            "analysis": {WorkflowPhase.ANALYSIS},
            "quality_assurance": {WorkflowPhase.QUALITY_ASSURANCE}
        }
    
    def _build_tool_priority_rules(self) -> Dict[WorkflowPhase, List[Dict[str, Any]]]:
        """Build priority rules for tools in different phases."""
        return {
            WorkflowPhase.ANALYSIS: [
                {"pattern": "read", "priority": 1},
                {"pattern": "inspect", "priority": 1},
                {"pattern": "analyze", "priority": 1}
            ],
            WorkflowPhase.IMPLEMENTATION: [
                {"pattern": "write", "priority": 1},
                {"pattern": "create", "priority": 1},
                {"pattern": "edit", "priority": 1}
            ],
            WorkflowPhase.TESTING: [
                {"pattern": "test", "priority": 1},
                {"pattern": "execute", "priority": 1},
                {"pattern": "run", "priority": 1}
            ]
        }
    
    def _build_phase_requirements(self) -> Dict[WorkflowPhase, Set[str]]:
        """Build required capabilities for each workflow phase."""
        return {
            WorkflowPhase.ANALYSIS: {"file_operations", "analysis", "inspection"},
            WorkflowPhase.IMPLEMENTATION: {"file_operations", "code_generation", "editing"},
            WorkflowPhase.TESTING: {"code_execution", "testing", "validation"},
            WorkflowPhase.QUALITY_ASSURANCE: {"quality_checking", "linting", "validation"},
            WorkflowPhase.DOCUMENTATION: {"documentation", "file_operations"},
            WorkflowPhase.DEPLOYMENT: {"deployment", "packaging", "distribution"}
        }
    
    def _get_phase_dependencies(self, phase: WorkflowPhase) -> List[WorkflowPhase]:
        """Get dependencies for a workflow phase."""
        dependencies = {
            WorkflowPhase.IMPLEMENTATION: [WorkflowPhase.ANALYSIS],
            WorkflowPhase.TESTING: [WorkflowPhase.IMPLEMENTATION],
            WorkflowPhase.QUALITY_ASSURANCE: [WorkflowPhase.TESTING],
            WorkflowPhase.DOCUMENTATION: [WorkflowPhase.IMPLEMENTATION],
            WorkflowPhase.DEPLOYMENT: [WorkflowPhase.QUALITY_ASSURANCE]
        }
        
        return dependencies.get(phase, [])
    
    def get_phase_tool_summary(self, enhanced_phase: EnhancedWorkflowPhase) -> Dict[str, Any]:
        """Get a summary of tools mapped to a phase."""
        primary_tools = [tm for tm in enhanced_phase.tool_mappings if tm.priority == 1]
        secondary_tools = [tm for tm in enhanced_phase.tool_mappings if tm.priority == 2]
        fallback_tools = [tm for tm in enhanced_phase.tool_mappings if tm.priority >= 3]
        
        return {
            "phase": enhanced_phase.phase.value,
            "total_tools": len(enhanced_phase.tool_mappings),
            "primary_tools": len(primary_tools),
            "secondary_tools": len(secondary_tools),
            "fallback_tools": len(fallback_tools),
            "tool_categories": list(set(tm.server_name for tm in enhanced_phase.tool_mappings))
        } 