# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries
"""
This module defines the high-level tools exposed by the Maestro MCP server.
These tools are designed to be called by external agentic IDEs and orchestrators.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import anyio
import subprocess
import sys
import platform
import asyncio
import functools
import os
import datetime
import hashlib
from .config import MAESTROConfig
from fastmcp import Context
from dataclasses import asdict, is_dataclass

# Import for type annotation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import MAESTROConfig

logger = logging.getLogger(__name__)

# Add after imports
_orchestration_engine = None

async def maestro_orchestrate(
    task_description: str = None,
    available_tools: List[Dict[str, Any]] = None,
    context_info: Optional[Dict[str, Any]] = None,
    workflow_session_id: Optional[str] = None,
    user_response: Any = None,
    operation_mode: str = "orchestrate",
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Unified workflow orchestration and collaboration handler for the MAESTRO Protocol.
    Consolidates task orchestration and user collaboration into a single robust tool.
    
    Args:
        task_description: Description of the task to orchestrate (required for new workflows)
        available_tools: List of available tools for workflow execution
        context_info: Additional context and configuration for workflow execution
        workflow_session_id: Existing workflow session ID to continue or None for new workflow
        user_response: User response data for collaboration (when operation_mode is "collaborate")
        operation_mode: Either "orchestrate" (default) or "collaborate"
        ctx: MCP context for logging
        
    Returns:
        Dict containing workflow results, session info, or collaboration responses
    """
    global _orchestration_engine
    
    try:
        # Lazy import to prevent delays during tool scanning
        from .orchestration_framework import EnhancedOrchestrationEngine, StepExecutionResult, ContextSurvey
        
        # Use singleton engine instance for consistent session management
        if _orchestration_engine is None:
            _orchestration_engine = EnhancedOrchestrationEngine()
        
        engine = _orchestration_engine
        
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Mode: {operation_mode}, Session: {workflow_session_id or 'new'}")
        
        # Handle collaboration responses
        if operation_mode == "collaborate":
            return await _handle_collaboration_response(
                engine, user_response, workflow_session_id, context_info, ctx
            )
        
        # Handle orchestration (both new and continuing workflows)
        return await _handle_orchestration(
            engine, task_description, available_tools, context_info, workflow_session_id, ctx
        )
        
    except Exception as e:
        if ctx:
            ctx.error(f"[MaestroOrchestrate] Error in {operation_mode} mode: {str(e)}")
        return {
            "status": "error",
            "operation_mode": operation_mode,
            "error": str(e),
            "message": f"Failed to process {operation_mode} request",
            "session_id": workflow_session_id,
            "timestamp": datetime.datetime.now().isoformat()
        }


async def _handle_collaboration_response(
    engine, user_response: Any, workflow_session_id: Optional[str], 
    context_info: Optional[Dict[str, Any]], ctx: Context
) -> Dict[str, Any]:
    """Handle user collaboration responses with robust session management."""
    
    if not user_response:
        return {
            "status": "error",
            "operation_mode": "collaborate",
            "error": "user_response is required for collaboration mode",
            "message": "No user response provided for collaboration",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    # Handle workflow session collaboration
    if workflow_session_id:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Processing collaboration for session: {workflow_session_id}")
        
        try:
            session = engine.session_manager.get_session(workflow_session_id)
            
            if not session:
                if ctx:
                    ctx.error(f"[MaestroOrchestrate] Session {workflow_session_id} not found or expired")
                return {
                    "status": "session_expired",
                    "operation_mode": "collaborate",
                    "session_id": workflow_session_id,
                    "error": f"Session {workflow_session_id} not found or expired",
                    "message": "The workflow session has expired or is invalid. Please start a new workflow.",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Update session context with user response
            _merge_user_response_to_session(session, user_response, context_info)
            
            # Update session in manager
            engine.session_manager.update_session(session)
            
            if ctx:
                ctx.info(f"[MaestroOrchestrate] Session context updated with user response")
            
            # Continue workflow execution with updated context
            try:
                step_result = await engine.execute_workflow_step(workflow_session_id)
                
                if ctx:
                    ctx.info(f"[MaestroOrchestrate] Workflow step executed successfully after collaboration")
                
                return {
                    "status": "workflow_continued",
                    "operation_mode": "collaborate",
                    "session_id": workflow_session_id,
                    "step_result": step_result,
                    "user_response_processed": True,
                    "message": "User response processed and workflow continued successfully",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
            except Exception as step_error:
                if ctx:
                    ctx.error(f"[MaestroOrchestrate] Error executing workflow step: {step_error}")
                
                return {
                    "status": "execution_error",
                    "operation_mode": "collaborate",
                    "session_id": workflow_session_id,
                    "error": str(step_error),
                    "user_response_processed": True,
                    "message": "User response was processed but workflow execution failed",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            if ctx:
                ctx.error(f"[MaestroOrchestrate] Error updating session: {e}")
            
            return {
                "status": "collaboration_error",
                "operation_mode": "collaborate",
                "session_id": workflow_session_id,
                "error": str(e),
                "message": "Failed to process collaboration response",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    # Handle paused workflow creation collaboration
    creation_id = _extract_creation_id_from_context(context_info)
    if creation_id:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Resuming workflow creation: {creation_id}")
        
        try:
            # Prepare context response
            context_response = _prepare_context_response(user_response, context_info)
            
            # Resume workflow creation with user context
            result = await engine.resume_workflow_creation(creation_id, context_response)
            
            if ctx:
                ctx.info(f"[MaestroOrchestrate] Workflow creation resumed successfully")
            
            # Handle different result types
            if isinstance(result, ContextSurvey):
                return {
                    "status": "context_still_required",
                    "operation_mode": "collaborate", 
                    "creation_id": creation_id,
                    "survey": result,
                    "message": "Additional context is still required to continue workflow creation",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            elif isinstance(result, StepExecutionResult):
                return {
                    "status": "workflow_created_and_started",
                    "operation_mode": "collaborate",
                    "session_id": result.workflow_session_id,
                    "step_result": result,
                    "message": "Workflow created successfully and first step executed",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                return {
                    "status": "workflow_created",
                    "operation_mode": "collaborate",
                    "result": result,
                    "message": "Workflow creation completed successfully",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
        except Exception as resume_error:
            if ctx:
                ctx.error(f"[MaestroOrchestrate] Error resuming workflow creation: {resume_error}")
            
            return {
                "status": "creation_error",
                "operation_mode": "collaborate",
                "creation_id": creation_id,
                "error": str(resume_error),
                "message": "Failed to resume workflow creation with provided context",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    # Fallback for non-workflow collaborations
    return {
        "status": "processed",
        "operation_mode": "collaborate",
        "user_response": user_response,
        "message": "User response has been successfully processed and can be used in the next workflow step.",
        "timestamp": datetime.datetime.now().isoformat()
    }


async def _handle_orchestration(
    engine, task_description: str, available_tools: List[Dict[str, Any]], 
    context_info: Optional[Dict[str, Any]], workflow_session_id: Optional[str], ctx: Context
) -> Dict[str, Any]:
    """Handle workflow orchestration with robust session and context management."""
    
    # Map provided context to expected keys for backward compatibility
    if context_info:
        context_info = _map_context_info(context_info)
    
    # Handle progressive workflow execution (continuing existing workflow)
    if workflow_session_id:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Attempting to execute next step in session {workflow_session_id}")
        
        try:
            orchestration_result = await engine.execute_workflow_step(workflow_session_id)
            
            # Convert result to consistent dictionary format
            return _convert_result_to_dict(orchestration_result)
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "session not found" in error_msg or "expired" in error_msg:
                # Session expired or not found, start new workflow with context
                if ctx:
                    ctx.warning(f"[MaestroOrchestrate] Session {workflow_session_id} expired or not found, starting new workflow")
                
                if not task_description or not available_tools:
                    return {
                        "status": "session_expired_insufficient_context",
                        "operation_mode": "orchestrate",
                        "session_id": workflow_session_id,
                        "error": "task_description and available_tools required for new workflow after session expiry",
                        "message": "Session expired and insufficient context to restart workflow",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                
                # Start new workflow with available context
                return await _start_new_workflow(engine, task_description, available_tools, context_info, ctx)
            else:
                raise
    
    # Start new progressive workflow
    if not task_description:
        return {
            "status": "error",
            "operation_mode": "orchestrate",
            "error": "task_description is required for new workflow orchestration",
            "message": "Cannot start new workflow without task description",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    return await _start_new_workflow(engine, task_description, available_tools, context_info, ctx)


async def _start_new_workflow(
    engine, task_description: str, available_tools: List[Dict[str, Any]], 
    context_info: Optional[Dict[str, Any]], ctx: Context
) -> Dict[str, Any]:
    """Start a new workflow with proper context and tool handling."""
    
    # Check if this is a continuation of a paused workflow creation
    creation_id = f"creation_{hashlib.sha256(task_description.encode()).hexdigest()[:10]}"
    paused_creation = engine.session_manager.get_paused_creation(creation_id)

    if paused_creation and context_info:
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Resuming paused workflow creation: {creation_id}")
        
        orchestration_context = context_info.copy()
        if available_tools:
            orchestration_context["available_tools"] = available_tools
        
        result = await engine.resume_workflow_creation(
            creation_id=creation_id,
            context_response=orchestration_context
        )
        
        # Convert result to consistent dictionary format
        return _convert_result_to_dict(result)
    else:
        # Validate required parameters for new workflow
        if not available_tools:
            return {
                "status": "error",
                "operation_mode": "orchestrate",
                "error": "available_tools are required for new workflow orchestration",
                "message": "Cannot start new workflow without available tools",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Merge available_tools into context_info for tool mapping
        orchestration_context = context_info.copy() if context_info else {}
        orchestration_context["available_tools"] = available_tools
        
        if ctx:
            ctx.info(f"[MaestroOrchestrate] Creating new progressive workflow with {len(available_tools)} available tools")
        
        # Run progressive orchestration (creates workflow and executes step 1)
        result = await engine.orchestrate_progressive_workflow(
            task_description=task_description,
            provided_context=orchestration_context
        )
        
        # Convert result to consistent dictionary format
        return _convert_result_to_dict(result)


def _merge_user_response_to_session(session, user_response: Any, context_info: Optional[Dict[str, Any]]):
    """Merge user response into session context data."""
    if isinstance(user_response, dict):
        # If user_response is a dictionary, merge it into context_data
        session.context_data.update(user_response)
    else:
        # If user_response is a string or other type, store it as 'user_context'
        session.context_data['user_context'] = user_response
    
    # Merge additional context if provided
    if context_info:
        session.context_data.update(context_info)
    
    # Store collaboration metadata for reference
    session.context_data['last_collaboration'] = {
        'user_response': user_response,
        'additional_context': context_info,
        'timestamp': datetime.datetime.now().isoformat()
    }


def _extract_creation_id_from_context(context_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract creation ID from context info or survey data."""
    if not context_info:
        return None
        
    # Check direct creation_id
    creation_id = context_info.get('creation_id')
    if creation_id:
        return creation_id
    
    # Check survey data
    survey = context_info.get('survey')
    if survey:
        if isinstance(survey, dict):
            return survey.get('survey_id')
        elif hasattr(survey, 'survey_id'):
            return survey.survey_id
    
    return None


def _prepare_context_response(user_response: Any, context_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare context response for workflow creation resume."""
    context_response = {}
    
    if isinstance(user_response, dict):
        context_response = user_response.copy()
    else:
        context_response['user_context'] = user_response
    
    # Merge additional context if provided
        if context_info:
        context_response.update(context_info)
    
    return context_response


def _map_context_info(context_info: Dict[str, Any]) -> Dict[str, Any]:
    """Map provided context to expected keys for backward compatibility."""
            mapped_context = {
                "target_audience": context_info.get("target_audience"),
                "design_preferences": {
                    "color_scheme": context_info.get("color_scheme"),
                    "design_style": context_info.get("design_style"),
                    "inspiration_sites": context_info.get("inspiration_sites")
                },
                "functionality_requirements": {
                    "key_features": context_info.get("key_features"),
                    "need_forms": context_info.get("need_forms"),
                    "need_galleries": context_info.get("need_galleries"),
                    "need_ecommerce": context_info.get("need_ecommerce"),
                    "external_integrations": context_info.get("external_integrations")
                },
                "content_assets": {
                    "existing_content": context_info.get("existing_content"),
                    "need_content_creation": context_info.get("need_content_creation"),
                    "has_brand_guidelines": context_info.get("has_brand_guidelines")
                },
                "technical_constraints": {
                    "hosting": context_info.get("hosting"),
                    "technical_constraints": context_info.get("technical_constraints"),
                    "cms_needed": context_info.get("cms_needed")
                }
            }
            # Merge original context with mapped context
    return {**context_info, **mapped_context}




async def maestro_search(
    query: str,
    search_engine: str = "duckduckgo",
    num_results: int = 5,
    ctx: Context = None,
) -> List[Dict[str, str]]:
    """
    Performs a web search using a specified search engine and returns the results.

    Args:
        query: The search query.
        search_engine: The search engine to use (e.g., 'duckduckgo', 'google', 'bing').
        num_results: The desired number of search results.
        ctx: The MCP context.

    Returns:
        A list of search result dictionaries, each containing 'title', 'link', and 'snippet'.
    """
    if ctx:
        ctx.info(f"Performing search for '{query}' using {search_engine}")
    try:
        # Lazy import to prevent delays during tool scanning
        from .web import SearchEngine
        engine = SearchEngine(engine=search_engine)
        results = await engine.search(query, num_results=num_results)
        if ctx:
            ctx.info(f"Found {len(results)} results.")
        return results
    except Exception as e:
        if ctx:
            ctx.error(f"Search failed: {e}")
        raise

async def maestro_scrape(
    url: str,
    ctx: Context = None
) -> Dict[str, str]:
    """
    Scrapes the content of a given URL.

    Args:
        url: The URL to scrape.
        ctx: The MCP context.

    Returns:
        A dictionary containing the URL, title, and text content of the page.
    """
    if ctx:
        ctx.info(f"Scraping URL: {url}")
    try:
        # Lazy import to prevent delays during tool scanning
        from .web import Browser
        async with Browser() as browser:
            content = await browser.scrape(url)
            if ctx:
                ctx.info(f"Successfully scraped {len(content.get('text', ''))} characters from {url}.")
            return content
    except Exception as e:
        if ctx:
            ctx.error(f"Scraping failed: {e}")
        raise

async def maestro_web(
    operation: str,
    query_or_url: str,
    search_engine: str = "duckduckgo",
    num_results: int = 5,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Unified web tool for LLM-driven research. Supports only web search (no scraping).

    Args:
        operation: Only 'search' is supported (performs web search).
        query_or_url: Search query string.
        search_engine: The search engine to use (default: duckduckgo).
        num_results: Number of search results to return.
        ctx: The MCP context.

    Returns:
        {"operation": "search", "query": str, "results": List[Dict]}
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if operation != "search":
        raise ValueError("maestro_web only supports 'search' operation. Scraping is not supported.")
    if ctx:
        ctx.info(f"[MaestroWeb] Performing web search for '{query_or_url}' using {search_engine}")
        if config:
            ctx.info(f"[MaestroWeb] Rate limiting enabled: {config.security.rate_limit_enabled}")
    try:
        # Lazy import to prevent delays during tool scanning
        from .web import SearchEngine
        engine = SearchEngine(engine=search_engine)
        results = await engine.search(query_or_url, num_results=num_results)
        if ctx:
            ctx.info(f"[MaestroWeb] Found {len(results)} search results.")
        return {
            "operation": "search",
            "query": query_or_url,
            "search_engine": search_engine,
            "results": results
        }
    except Exception as e:
        if ctx:
            ctx.error(f"[MaestroWeb] Search failed: {e}")
        raise

async def maestro_execute(
    code: str,
    language: str,
    timeout: int = 60,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Executes a block of code in a specified language within a secure sandbox.

    Args:
        code: The source code to execute.
        language: The programming language (e.g., 'python', 'javascript', 'bash').
        timeout: Execution timeout in seconds.
        ctx: The MCP context.

    Returns:
        A dictionary containing the execution status, stdout, stderr, and exit code.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        ctx.info(f"[MaestroExecute] Executing {language} code")
        if config:
            ctx.info(f"[MaestroExecute] Timeout: {config.engine.task_timeout}s")
    from .number_formatter import clean_output
    cmd = []
    if language == 'python':
        cmd = [sys.executable, '-c', code]
    elif language == 'javascript':
        cmd = ['node', '-e', code]
    elif language == 'bash':
        cmd = ['bash', '-c', code]
    else:
        raise ValueError(f"Unsupported language: {language}")
    def run_sync_subprocess(cmd_list: list, timeout_sec: int) -> dict:
        try:
            process_result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            stdout_cleaned = clean_output(process_result.stdout.strip()) if process_result.stdout else ""
            stderr_cleaned = clean_output(process_result.stderr.strip()) if process_result.stderr else ""
            return {
                "exit_code": process_result.returncode,
                "stdout": stdout_cleaned,
                "stderr": stderr_cleaned,
                "status": "success" if process_result.returncode == 0 else "error",
            }
        except FileNotFoundError:
            return {"status": "error", "error": f"Interpreter for '{language}' not found."}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Execution timed out"}
    if ctx:
        await ctx.info(f"Running command in thread: {' '.join(cmd)}")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, functools.partial(run_sync_subprocess, cmd_list=cmd, timeout_sec=timeout)
        )
        if ctx:
            await ctx.info(f"Execution finished with status: {result.get('status')}")
        return result
    except Exception as e:
        if ctx:
            await ctx.error(f"Execution failed in executor: {e}")
        raise

async def maestro_error_handler(
    error_message: str,
    context: Dict[str, Any],
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Analyzes an error and provides a structured response for recovery.

    Args:
        error_message: The error message that occurred.
        context: The context in which the error occurred (e.g., tool name, parameters).
        ctx: The MCP context.

    Returns:
        A dictionary with error analysis and suggested recovery steps.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        ctx.error(f"[MaestroErrorHandler] Analyzing error: {error_message}")
        if config:
            ctx.info(f"[MaestroErrorHandler] Debug mode: {config.engine.mode.value == 'development'}")
    analysis = {
        "error_type": "GenericError",
        "severity": "High",
        "possible_root_cause": "An unknown error occurred.",
        "recovery_suggestion": "Review the error message and context, and try an alternative approach."
    }
    if "timeout" in error_message.lower():
        analysis["error_type"] = "TimeoutError"
        analysis["possible_root_cause"] = "The operation took too long to complete."
        analysis["recovery_suggestion"] = "Try increasing the timeout or simplifying the operation."
    elif "not found" in error_message.lower():
        analysis["error_type"] = "NotFoundError"
        analysis["possible_root_cause"] = "A required resource or file was not found."
        analysis["recovery_suggestion"] = "Verify that all paths are correct and required resources exist."
    return {
        "original_error": error_message,
        "original_context": context,
        "analysis": analysis,
    }

async def maestro_iae(
    engine_name: str,
    method_name: str,
    parameters: Dict[str, Any],
    ctx: Context = None,
) -> Any:
    """
    Invokes a specific capability from an Intelligence Amplification Engine (IAE) using the MCP-native registry and meta-reasoning logic.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        ctx.info(f"[MaestroIAE] Invoking {engine_name}.{method_name}")
        if config:
            ctx.info(f"[MaestroIAE] Engine mode: {config.engine.mode.value}")
    try:
        import sys
        from pathlib import Path
        engines_dir = Path(__file__).parent.parent
        if str(engines_dir) not in sys.path:
            sys.path.insert(0, str(engines_dir))
        from .iae_discovery import IAERegistry
        from .maestro_iae import IAEIntegrationManager
        registry = IAERegistry()
        manager = IAEIntegrationManager(registry)
        await manager.initialize_engines()
        result = await manager.execute_task(
            engine_id=f"iae_{engine_name.lower()}",
            task_name=method_name,
            parameters=parameters,
            context=ctx
        )
        if ctx:
            await ctx.info(f"[MaestroIAE] IAE method {method_name} executed successfully.")
        return result
    except Exception as e:
        if ctx:
            await ctx.error(f"[MaestroIAE] IAE operation failed: {e}")
        raise 

def _convert_result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert StepExecutionResult or other dataclass objects to dictionaries."""
    if is_dataclass(result) and not isinstance(result, type):
        # Convert dataclass to dictionary
        result_dict = asdict(result)
        # Ensure we have consistent fields
        if not result_dict.get("operation_mode"):
            result_dict["operation_mode"] = "orchestrate"
        if not result_dict.get("timestamp"):
            result_dict["timestamp"] = datetime.datetime.now().isoformat()
        return result_dict
    elif isinstance(result, dict):
        # Already a dictionary, just ensure consistent fields
        if not result.get("operation_mode"):
            result["operation_mode"] = "orchestrate"
        if not result.get("timestamp"):
            result["timestamp"] = datetime.datetime.now().isoformat()
        return result
    else:
        # Convert other types to dictionary format
        return {
            "status": "unknown",
            "operation_mode": "orchestrate",
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        } 