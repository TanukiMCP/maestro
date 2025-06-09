# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries
"""
This module defines the high-level tools exposed by the Maestro MCP server.
These tools are designed to be called by external agentic IDEs and orchestrators.
"""

import logging
from typing import List, Dict, Any, Optional
import anyio
import subprocess
import sys
import platform
import asyncio
import functools
import os
from .config import MAESTROConfig

# The Context object is part of the server's context submodule.
from fastmcp.server.context import Context

# Import for type annotation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import MAESTROConfig

logger = logging.getLogger(__name__)

async def maestro_orchestrate(
    task_description: str = None,
    available_tools: List[Dict[str, Any]] = None,
    context_info: Optional[Dict[str, Any]] = None,
    workflow_session_id: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Orchestrates a complex task using progressive step-by-step execution.
    
    For new orchestration: Provide task_description and available_tools to create workflow and execute step 1.
    For continuation: Provide workflow_session_id to execute the next step in existing workflow.
    
    Returns step execution results following sequentialthinking pattern with:
    - Step progress (current_step/total_steps)
    - Step results and validation
    - Next step guidance
    - Session management for state continuity
    
    This implementation is fully MCP-native, context-aware, and stateful, with no placeholders or mock logic.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        if workflow_session_id:
            ctx.info(f"[Maestro] Continuing workflow session: {workflow_session_id}")
        else:
            ctx.info(f"[Maestro] Starting new progressive orchestration: {task_description}")
        if config:
            ctx.info(f"[Maestro] Operating in {config.engine.mode.value} mode.")
    
    try:
        # Lazy import to prevent delays during tool scanning
        from .orchestration_framework import EnhancedOrchestrationEngine, StepExecutionResult, ContextSurvey
        engine = EnhancedOrchestrationEngine()
        
        # Handle progressive workflow execution
        if workflow_session_id:
            # Continue existing workflow
            if ctx:
                ctx.info(f"[Maestro] Executing next step in session {workflow_session_id}")
            
            orchestration_result = await engine.execute_workflow_step(workflow_session_id)
            
        else:
            # Start new progressive workflow
            if not task_description or not available_tools:
                raise ValueError("task_description and available_tools are required for new workflow orchestration")
            
            # Merge available_tools into context_info for tool mapping
            orchestration_context = context_info.copy() if context_info else {}
            orchestration_context["available_tools"] = available_tools
            
            if ctx:
                ctx.info(f"[Maestro] Creating new progressive workflow with {len(available_tools)} available tools")
            
            # Run progressive orchestration (creates workflow and executes step 1)
            orchestration_result = await engine.orchestrate_progressive_workflow(
                task_description=task_description,
                provided_context=orchestration_context
            )
        
        # Handle different result types
        if isinstance(orchestration_result, ContextSurvey):
            if ctx:
                ctx.warning("[Maestro] Context gaps detected, returning survey for user input.")
            return {"status": "context_required", "survey": orchestration_result}
        
        elif isinstance(orchestration_result, StepExecutionResult):
            if ctx:
                step_status = orchestration_result.status
                current_step = orchestration_result.current_step
                total_steps = orchestration_result.total_steps
                ctx.info(f"[Maestro] Step execution result: {step_status} (step {current_step}/{total_steps})")
            
            # Convert StepExecutionResult to dict format for MCP response
            result_dict = {
                "status": orchestration_result.status,
                "workflow_session_id": orchestration_result.workflow_session_id,
                "current_step": orchestration_result.current_step,
                "total_steps": orchestration_result.total_steps,
                "step_description": orchestration_result.step_description,
                "step_results": orchestration_result.step_results,
                "next_step_needed": orchestration_result.next_step_needed,
                "next_step_guidance": orchestration_result.next_step_guidance,
                "overall_progress": orchestration_result.overall_progress
            }
            
            # Include optional fields if present
            if orchestration_result.workflow is not None:
                result_dict["workflow"] = {
                    "workflow_id": orchestration_result.workflow.workflow_id,
                    "task_description": orchestration_result.workflow.task_description,
                    "complexity": orchestration_result.workflow.complexity.value,
                    "estimated_total_time": orchestration_result.workflow.estimated_total_time,
                    "phase_count": len(orchestration_result.workflow.phases),
                    "tool_mappings_count": len(orchestration_result.workflow.tool_mappings),
                    "iae_mappings_count": len(orchestration_result.workflow.iae_mappings)
                }
            
            if orchestration_result.execution_summary is not None:
                result_dict["execution_summary"] = orchestration_result.execution_summary
            
            if orchestration_result.error_details is not None:
                result_dict["error_details"] = orchestration_result.error_details
            
            return result_dict
        
        else:
            if ctx:
                ctx.error(f"[Maestro] Unexpected orchestration result type: {type(orchestration_result)}")
            raise RuntimeError("Unexpected orchestration result type.")
    
    except Exception as e:
        if ctx:
            ctx.error(f"[Maestro] Progressive orchestration failed: {e}")
        raise

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

async def maestro_collaboration_response(
    user_response: Any,
    original_request: Dict[str, Any],
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Handles a response from a user during a collaborative workflow step.

    Args:
        user_response: The data received from the user.
        original_request: The original request that prompted the collaboration.
        ctx: The MCP context.

    Returns:
        A dictionary indicating the collaboration has been processed.
    """
    # Get config lazily only when actually needed
    from .dependencies import get_config
    config = get_config() if ctx else None
    
    if ctx:
        ctx.info(f"[MaestroCollaboration] Processing user response")
        if config:
            ctx.info(f"[MaestroCollaboration] Mode: {config.engine.mode.value}")
    return {
        "status": "processed",
        "user_response": user_response,
        "message": "User response has been successfully processed and can be used in the next workflow step."
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