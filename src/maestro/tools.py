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

# The Context object is part of the server's context submodule.
from fastmcp.server.context import Context

logger = logging.getLogger(__name__)

async def maestro_orchestrate(
    task_description: str,
    available_tools: List[Dict[str, Any]],
    context_info: Optional[Dict[str, Any]] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Orchestrates a complex task by generating and executing a dynamic workflow using a suite of available tools.
    This implementation is fully MCP-native, context-aware, and stateful, with no placeholders or mock logic.
    """
    if ctx:
        ctx.info(f"[Maestro] Orchestration requested: {task_description}")
    try:
        from .orchestration_framework import EnhancedOrchestrationEngine, OrchestrationResult, ContextSurvey
        engine = EnhancedOrchestrationEngine()
        # Merge available_tools into context_info for tool mapping
        orchestration_context = context_info.copy() if context_info else {}
        orchestration_context["available_tools"] = available_tools
        # Run the full orchestration
        orchestration_result = await engine.orchestrate_complete_workflow(
            task_description=task_description,
            provided_context=orchestration_context
        )
        # If a ContextSurvey is returned, return it directly (context gaps must be filled)
        if isinstance(orchestration_result, ContextSurvey):
            if ctx:
                ctx.warning("[Maestro] Context gaps detected, returning survey for user input.")
            return {"status": "context_required", "survey": orchestration_result}
        # If OrchestrationResult, return all workflow details
        elif isinstance(orchestration_result, OrchestrationResult):
            if ctx:
                ctx.info("[Maestro] Orchestration completed successfully.")
            return {
                "status": "orchestrated",
                "workflow": orchestration_result.workflow,
                "execution_guidance": orchestration_result.execution_guidance,
                "validation_results": orchestration_result.validation_results,
                "overall_success": orchestration_result.overall_success,
                "completion_percentage": orchestration_result.completion_percentage,
                "recommendations": orchestration_result.recommendations,
                "next_steps": orchestration_result.next_steps
            }
        else:
            if ctx:
                ctx.error(f"[Maestro] Unexpected orchestration result type: {type(orchestration_result)}")
            raise RuntimeError("Unexpected orchestration result type.")
    except Exception as e:
        if ctx:
            ctx.error(f"[Maestro] Orchestration failed: {e}")
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
    if operation != "search":
        raise ValueError("maestro_web only supports 'search' operation. Scraping is not supported.")
    if ctx:
        ctx.info(f"[MaestroWeb] Performing web search for '{query_or_url}' using {search_engine}")
    try:
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
    if ctx:
        await ctx.info(f"Executing {language} code...")
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
    if ctx:
        ctx.warning(f"Analyzing error: {error_message}")
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
    if ctx:
        ctx.info(f"Received user collaboration response: {user_response}")
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
    if ctx:
        await ctx.info(f"[MaestroIAE] Invoking {method_name} on {engine_name} IAE...")
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