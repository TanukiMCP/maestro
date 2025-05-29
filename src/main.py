#!/usr/bin/env python3
"""
MAESTRO Protocol MCP Server

Main entry point for the Meta-Agent Ensemble for Systematic Task Reasoning 
and Orchestration (MAESTRO) Protocol MCP server.

Core Principle: Intelligence Amplification > Model Scale
"""

import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional
import mcp.server as server
import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

# Use try/except for imports that might fail during testing
try:
    from maestro import MAESTROOrchestrator
    from maestro.data_models import MAESTROResult, QualityMetrics, VerificationResult
except ImportError:
    # Fallback import for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    try:
        from maestro import MAESTROOrchestrator
        from maestro.data_models import MAESTROResult, QualityMetrics, VerificationResult
    except ImportError as e:
        print(f"Import error: {e}")
        MAESTROOrchestrator = None
        MAESTROResult = None
        QualityMetrics = None
        VerificationResult = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TanukiMCPOrchestra:
    """
    Main MCP server implementing the MAESTRO Protocol.
    
    Provides a single entry point (orchestrate_workflow) that handles
    all complexity of task analysis, workflow generation, execution,
    and quality verification automatically.
    
    Uses lazy loading to prevent startup timeouts.
    """
    
    def __init__(self):
        # Lazy loading - don't initialize heavy components until needed
        self._orchestrator = None
        self._initialization_error = None
        
        # Initialize MCP server (lightweight)
        self.app = server.Server("tanukimcp-orchestra")
        self._register_handlers()
        
        logger.info("🎭 MAESTRO Protocol MCP Server Initialized (lazy loading enabled)")
    
    def _get_orchestrator(self):
        """Get orchestrator with lazy initialization."""
        if self._orchestrator is None and self._initialization_error is None:
            try:
                if MAESTROOrchestrator:
                    logger.info("🔄 Initializing MAESTRO Orchestrator...")
                    self._orchestrator = MAESTROOrchestrator()
                    logger.info("✅ MAESTRO Orchestrator initialized successfully")
                else:
                    self._initialization_error = "MAESTROOrchestrator not available"
            except Exception as e:
                self._initialization_error = f"Failed to initialize MAESTRO: {str(e)}"
                logger.error(f"❌ MAESTRO initialization failed: {self._initialization_error}")
        
        if self._initialization_error:
            raise RuntimeError(self._initialization_error)
        
        return self._orchestrator
    
    def _register_handlers(self):
        """Register MCP server handlers and tools."""
        
        @self.app.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return [
                types.Tool(
                    name="orchestrate_workflow",
                    description=(
                        "MAESTRO Protocol meta-orchestration tool. "
                        "Automatically designs and executes multi-agent workflows with "
                        "dynamic operator profile creation, intelligence amplification "
                        "for LLM weaknesses, automated quality verification at each step, "
                        "and early stopping when success criteria are met. "
                        "This is the primary entry point - users just describe their goal. "
                        "The system handles ALL complexity, tool selection, and verification."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Natural language description of what to accomplish"
                            },
                            "quality_threshold": {
                                "type": "number",
                                "description": "Minimum quality score for completion (0.0-1.0)",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.9
                            },
                            "verification_mode": {
                                "type": "string",
                                "description": "Verification depth: 'fast', 'balanced', or 'comprehensive'",
                                "enum": ["fast", "balanced", "comprehensive"],
                                "default": "fast"
                            },
                            "max_execution_time": {
                                "type": "integer",
                                "description": "Maximum execution time in seconds",
                                "minimum": 30,
                                "maximum": 1800,
                                "default": 300
                            }
                        },
                        "required": ["task_description"],
                        "additionalProperties": False
                    }
                ),
                
                types.Tool(
                    name="verify_quality",
                    description=(
                        "Verify quality using appropriate verification methods. "
                        "Used internally by orchestration but can be called directly."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to verify"
                            },
                            "verification_type": {
                                "type": "string",
                                "description": "Type of verification to perform",
                                "enum": ["mathematical", "code_quality", "language_quality", "visual", "accessibility"]
                            },
                            "success_criteria": {
                                "type": "object",
                                "description": "Success criteria for verification"
                            }
                        },
                        "required": ["content", "verification_type"],
                        "additionalProperties": False
                    }
                ),
                
                types.Tool(
                    name="amplify_capability",
                    description=(
                        "Use intelligence amplification for specific capability enhancement. "
                        "Compensates for LLM weaknesses using specialized Python libraries."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "capability_type": {
                                "type": "string",
                                "description": "Type of capability to amplify",
                                "enum": ["mathematics", "language", "code_quality", "web_verification", "data_analysis"]
                            },
                            "input_data": {
                                "description": "Input data for capability enhancement"
                            },
                            "requirements": {
                                "type": "object",
                                "description": "Specific requirements for the capability"
                            }
                        },
                        "required": ["capability_type", "input_data"],
                        "additionalProperties": False
                    }
                )
            ]
        
        @self.app.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls."""
            try:
                if name == "orchestrate_workflow":
                    return await self._handle_orchestrate_workflow(arguments)
                elif name == "verify_quality":
                    return await self._handle_verify_quality(arguments)
                elif name == "amplify_capability":
                    return await self._handle_amplify_capability(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}\n{traceback.format_exc()}")
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error executing {name}: {str(e)}\n\nPlease check your input parameters and try again."
                )]
    
    async def _handle_orchestrate_workflow(self, arguments: dict) -> list[types.TextContent]:
        """Handle the primary orchestrate_workflow tool."""
        task_description = arguments["task_description"]
        quality_threshold = arguments.get("quality_threshold", 0.9)
        verification_mode = arguments.get("verification_mode", "fast")
        max_execution_time = arguments.get("max_execution_time", 300)
        
        logger.info(f"🎭 MAESTRO orchestrating task: {task_description[:100]}...")
        
        # Execute MAESTRO orchestration
        try:
            orchestrator = self._get_orchestrator()
            result = await orchestrator.orchestrate_workflow(
                task_description=task_description,
                quality_threshold=quality_threshold,
                verification_mode=verification_mode,
                max_execution_time=max_execution_time
            )
            
            if result.success:
                response = result.format_success_response()
            else:
                response = await self._handle_quality_failure(result, task_description)
            
            return [types.TextContent(type="text", text=response)]
        
        except Exception as e:
            logger.error(f"MAESTRO orchestration failed: {str(e)}")
            error_response = await self._handle_orchestration_error(e, task_description)
            return [types.TextContent(type="text", text=error_response)]
    
    async def _handle_verify_quality(self, arguments: dict) -> list[types.TextContent]:
        """Handle quality verification requests."""
        content = arguments["content"]
        verification_type = arguments["verification_type"]
        success_criteria = arguments.get("success_criteria", {})
        
        try:
            result = await self._get_orchestrator().verify_content_quality(
                content=content,
                verification_type=verification_type,
                success_criteria=success_criteria
            )
            
            response = f"""
## Quality Verification Results

**Verification Type:** {verification_type}
**Overall Score:** {result.quality_metrics.overall_score:.2%}
**Success:** {'✅ Yes' if result.success else '❌ No'}

**Detailed Metrics:**
- Accuracy: {result.quality_metrics.accuracy_score:.2%}
- Completeness: {result.quality_metrics.completeness_score:.2%}
- Quality: {result.quality_metrics.quality_score:.2%}
- Confidence: {result.confidence_score:.2%}

**Issues Found:** {len(result.issues_found)}
{chr(10).join(f"- {issue}" for issue in result.issues_found)}

**Recommendations:**
{chr(10).join(f"- {rec}" for rec in result.recommendations)}
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Quality verification failed: {str(e)}"
            )]
    
    async def _handle_amplify_capability(self, arguments: dict) -> list[types.TextContent]:
        """Handle capability amplification requests."""
        capability_type = arguments["capability_type"]
        input_data = arguments["input_data"]
        requirements = arguments.get("requirements", {})
        
        try:
            result = await self._get_orchestrator().amplify_capability(
                capability_type=capability_type,
                input_data=input_data,
                requirements=requirements
            )
            
            response = f"""
## Capability Amplification Results

**Capability Type:** {capability_type}
**Processing Status:** ✅ Complete

**Enhanced Result:**
{result}

This result has been enhanced using specialized Python libraries to compensate for LLM limitations.
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Capability amplification failed: {str(e)}"
            )]
    
    async def _handle_quality_failure(self, result: MAESTROResult, original_task: str) -> str:
        """Handle cases where automatic quality verification fails."""
        return result.format_success_response()  # This handles both success and failure cases
    
    async def _handle_orchestration_error(self, error: Exception, task_description: str) -> str:
        """Handle orchestration errors with helpful guidance."""
        return f"""
## MAESTRO Protocol Error Handling 🚨

**Original Task:** {task_description}

**Error Encountered:** {str(error)}

**What happened:** The MAESTRO Protocol encountered an unexpected error during orchestration. This could be due to:

1. **Resource limitations** - Complex tasks may require more time or computational resources
2. **Input complexity** - The task description may need to be more specific or broken down
3. **System dependencies** - Some required libraries or services may not be available

**Recommended actions:**
1. **Simplify the task** - Try breaking down complex requests into smaller steps
2. **Provide more context** - Include specific requirements, constraints, or expected outputs
3. **Check dependencies** - Ensure all required systems and libraries are available
4. **Retry with different parameters** - Adjust quality threshold or verification mode

**Example of a well-structured request:**
```
Create a simple Python function that calculates the factorial of a number, 
include unit tests, and verify the code quality meets Python standards.
```

Would you like to try again with a modified approach?
"""

    async def run(self):
        """Run the MCP server."""
        # Use the correct stdio server from mcp.server.stdio
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream, 
                write_stream,
                InitializationOptions(
                    server_name="tanukimcp-orchestra",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {}
                    }
                )
            )


async def main():
    """Main entry point for the MAESTRO Protocol MCP server."""
    # Minimal startup logging for fast tool scanning
    server_instance = TanukiMCPOrchestra()
    await server_instance.run()


if __name__ == "__main__":
    asyncio.run(main()) 