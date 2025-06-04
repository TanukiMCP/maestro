"""
Enhanced Tool Handlers for MAESTRO Protocol

Provides MCP tool handlers for the new enhanced capabilities:
- maestro_search: LLM-driven web search with fallback
- maestro_scrape: LLM-driven web scraping 
- maestro_execute: LLM-driven code execution
- maestro_error_handler: Adaptive error handling
- maestro_temporal_context: Temporal context awareness
"""

import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from mcp import types

from .adaptive_error_handler import (
    AdaptiveErrorHandler,
    TemporalContext,
    ErrorContext,
    ReconsiderationResult
)
from .llm_web_tools import LLMWebTools

logger = logging.getLogger(__name__)


class EnhancedToolHandlers:
    """
    Handlers for enhanced MAESTRO tools with LLM-driven capabilities
    """
    
    def __init__(self):
        self.error_handler = AdaptiveErrorHandler()
        self.llm_web_tools = LLMWebTools()
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure tools are initialized"""
        if not self._initialized:
            logger.info("🔄 Initializing enhanced tool handlers...")
            self._initialized = True
            logger.info("✅ Enhanced tool handlers ready")
    
    async def handle_maestro_search(self, arguments: dict) -> list[types.TextContent]:
        """Handle maestro_search tool calls"""
        await self._ensure_initialized()
        
        try:
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 10)
            search_engine = arguments.get("search_engine", "duckduckgo")
            temporal_filter = arguments.get("temporal_filter", "any")
            result_format = arguments.get("result_format", "structured")
            
            if not query:
                return [types.TextContent(
                    type="text",
                    text="❌ **Search Query Required**\n\nPlease provide a search query."
                )]
            
            logger.info(f"🔍 Executing MAESTRO search: '{query}'")
            
            # Execute search with temporal filtering
            if temporal_filter == "any":
                temporal_filter = None
            
            # Map search_engine to engines list format
            engines = [search_engine] if search_engine else ['duckduckgo']
            
            result = await self.llm_web_tools.llm_driven_search(
                query=query,
                max_results=max_results,
                engines=engines,
                temporal_filter=temporal_filter,
                result_format=result_format,
                llm_analysis=True,
                context=None  # Context will be added when available
            )
            
            if result.get("success", False):
                # Format successful search results
                response = f"# 🔍 LLM-Enhanced MAESTRO Search Results\n\n"
                response += f"**Query:** {query}\n"
                response += f"**Enhanced Query:** {result.get('enhanced_query', query)}\n"
                response += f"**Search Engines:** {', '.join(engines)}\n"
                response += f"**Results:** {result['total_results']} found\n"
                
                if temporal_filter:
                    response += f"**Time Filter:** {temporal_filter}\n"
                
                response += f"\n## Results:\n\n"
                
                for i, search_result in enumerate(result['results'], 1):
                    response += f"### {i}. {search_result['title']}\n"
                    response += f"**URL:** {search_result['url']}\n"
                    response += f"**Domain:** {search_result['domain']}\n"
                    response += f"**Snippet:** {search_result['snippet']}\n"
                    response += f"**Relevance:** {search_result['relevance_score']:.2f}\n"
                    
                    if search_result.get('llm_analysis'):
                        response += f"**LLM Analysis:** {search_result['llm_analysis']}\n"
                    
                    response += "\n"
                
                response += f"\n**Search completed at:** {result['timestamp']}\n"
                
                # Add metadata
                response += f"\n## Metadata\n"
                response += f"- LLM Enhanced: {result['metadata']['llm_enhanced']}\n"
                response += f"- Result format: {result_format}\n"
                response += f"- Engines status: {result['metadata']['engines_status']}\n"
                
            else:
                # Handle search failure with fallback guidance
                response = f"# ❌ MAESTRO Search Failed\n\n"
                response += f"**Query:** {query}\n"
                response += f"**Error:** {result.get('error', 'Unknown error')}\n\n"
                
                if "fallback_results" in result:
                    fallback = result["fallback_results"]
                    response += f"## Fallback Guidance\n\n"
                    response += f"**Suggestion:** {fallback.get('suggestion', '')}\n\n"
                    
                    if "fallback_guidance" in fallback:
                        guidance = fallback["fallback_guidance"]
                        response += f"**Manual Search:** {guidance.get('manual_search', '')}\n"
                        response += f"**Alternative Tools:** {', '.join(guidance.get('alternative_tools', []))}\n"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ MAESTRO search error: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"❌ **MAESTRO Search Error**\n\nError: {str(e)}\n\nPlease check your query and try again."
            )]
    
    async def handle_maestro_scrape(self, arguments: dict) -> list[types.TextContent]:
        """Handle maestro_scrape tool calls"""
        await self._ensure_initialized()
        
        try:
            url = arguments.get("url", "")
            output_format = arguments.get("output_format", "markdown")
            selectors = arguments.get("selectors", [])
            wait_for = arguments.get("wait_for")
            extract_links = arguments.get("extract_links", False)
            extract_images = arguments.get("extract_images", False)
            
            if not url:
                return [types.TextContent(
                    type="text",
                    text="❌ **URL Required**\n\nPlease provide a URL to scrape."
                )]
            
            logger.info(f"🕷️ Executing MAESTRO scrape: {url}")
            
            result = await self.llm_web_tools.llm_driven_scrape(
                url=url,
                output_format=output_format,
                target_content=None,  # Could be enhanced with selector-based targeting
                extract_structured=True,
                take_screenshot=True,
                interact_before_scrape=None,
                context=None  # Context will be added when available
            )
            
            if result.get("success", False):
                scrape_result = result['result']
                response = f"# 🕷️ LLM-Enhanced MAESTRO Scrape Results\n\n"
                response += f"**URL:** {url}\n"
                response += f"**Title:** {scrape_result['title']}\n"
                response += f"**Format:** {output_format}\n"
                response += f"**Content Length:** {scrape_result['metadata']['content_length']} characters\n"
                response += f"**Scraped at:** {result['timestamp']}\n\n"
                
                if scrape_result.get('llm_summary'):
                    response += f"## LLM Summary:\n{scrape_result['llm_summary']}\n\n"
                
                response += f"## Content:\n\n"
                response += scrape_result['content']
                
                # Add additional data
                metadata = scrape_result['metadata']
                if 'additional_data' in metadata:
                    additional_data = metadata['additional_data']
                    
                    if 'links' in additional_data:
                        links = additional_data['links']
                        response += f"\n\n## Links ({len(links)} found):\n\n"
                        for link in links[:10]:  # Limit to first 10 links
                            response += f"- [{link.get('text', 'No text')}]({link.get('href', '#')})\n"
                        if len(links) > 10:
                            response += f"... and {len(links) - 10} more links\n"
                    
                    if 'images' in additional_data:
                        images = additional_data['images']
                        response += f"\n\n## Images ({len(images)} found):\n\n"
                        for img in images[:5]:  # Limit to first 5 images
                            response += f"- {img.get('alt', 'No alt text')} ({img.get('src', '#')})\n"
                        if len(images) > 5:
                            response += f"... and {len(images) - 5} more images\n"
                    
                    if 'structured_data' in additional_data:
                        structured = additional_data['structured_data']
                        if structured:
                            response += f"\n\n## Structured Data:\n\n```json\n{json.dumps(structured, indent=2)}\n```\n"
                
                # Add screenshot if available
                if 'screenshot' in metadata:
                    response += f"\n\n## Screenshot Description:\n{metadata['screenshot'][:200]}...\n"
                
            else:
                response = f"# ❌ MAESTRO Scrape Failed\n\n"
                response += f"**URL:** {url}\n"
                response += f"**Error:** {result.get('error', 'Unknown error')}\n\n"
                
                if "fallback_result" in result:
                    fallback = result["fallback_result"]
                    response += f"## Fallback Guidance\n\n"
                    response += f"**Suggestion:** {fallback.get('suggestion', '')}\n\n"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ MAESTRO scrape error: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"❌ **MAESTRO Scrape Error**\n\nError: {str(e)}\n\nPlease check the URL and try again."
            )]
    
    async def handle_maestro_execute(self, arguments: dict) -> list[types.TextContent]:
        """Handle maestro_execute tool calls"""
        await self._ensure_initialized()
        
        try:
            code = arguments.get("code", "")
            language = arguments.get("language", "python")
            timeout = arguments.get("timeout", 30)
            capture_output = arguments.get("capture_output", True)
            working_directory = arguments.get("working_directory")
            environment_vars = arguments.get("environment_vars", {})
            
            if not code:
                return [types.TextContent(
                    type="text",
                    text="❌ **Code Required**\n\nPlease provide code to execute."
                )]
            
            logger.info(f"⚡ Executing MAESTRO code: {language}")
            
            result = await self.puppeteer_tools.maestro_execute(
                code=code,
                language=language,
                timeout=timeout,
                capture_output=capture_output,
                working_directory=working_directory,
                environment_vars=environment_vars
            )
            
            response = f"# ⚡ MAESTRO Code Execution Results\n\n"
            response += f"**Language:** {language}\n"
            response += f"**Status:** {'✅ SUCCESS' if result['success'] else '❌ FAILED'}\n"
            response += f"**Return Code:** {result['return_code']}\n"
            response += f"**Execution Time:** {result.get('execution_time', 0):.2f}s\n\n"
            
            if result['success']:
                response += f"## Output:\n\n"
                if 'output' in result and result['output']['stdout']:
                    response += f"```\n{result['output']['stdout']}\n```\n\n"
                else:
                    response += "*(No output)*\n\n"
                
                # Add validation results
                if 'validation' in result:
                    validation = result['validation']
                    response += f"## Validation:\n\n"
                    response += f"- Execution successful: {validation['execution_successful']}\n"
                    response += f"- Has output: {validation['has_output']}\n"
                    response += f"- Has errors: {validation['has_errors']}\n"
                    response += f"- Validation status: {validation['validation_status']}\n"
                    
                    if 'output_analysis' in validation:
                        analysis = validation['output_analysis']
                        response += f"- Output lines: {analysis['line_count']}\n"
                        response += f"- Contains JSON: {analysis['contains_json']}\n"
                        response += f"- Contains numbers: {analysis['contains_numbers']}\n"
            
            else:
                response += f"## Error Output:\n\n"
                if 'output' in result and result['output']['stderr']:
                    response += f"```\n{result['output']['stderr']}\n```\n\n"
                
                # Add error analysis
                if 'validation' in result and 'error_types' in result['validation']:
                    error_types = result['validation']['error_types']
                    response += f"**Error Types:** {', '.join(error_types)}\n\n"
            
            response += f"## Metadata:\n"
            response += f"- Code length: {result['metadata']['code_length']} characters\n"
            response += f"- Working directory: {result['metadata']['working_directory']}\n"
            response += f"- Command: {result['metadata']['command']}\n"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ MAESTRO execute error: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"❌ **MAESTRO Execute Error**\n\nError: {str(e)}\n\nPlease check your code and try again."
            )]
    
    async def handle_maestro_error_handler(self, arguments: dict) -> list[types.TextContent]:
        """Handle maestro_error_handler tool calls"""
        await self._ensure_initialized()
        
        try:
            error_details = arguments.get("error_details", {})
            available_tools = arguments.get("available_tools", [])
            success_criteria = arguments.get("success_criteria", [])
            temporal_context_data = arguments.get("temporal_context", {})
            
            # Create temporal context
            temporal_context = TemporalContext(
                current_timestamp=datetime.now(timezone.utc),
                information_cutoff=None,
                task_deadline=None,
                context_freshness_required=temporal_context_data.get("context_freshness_required", False),
                temporal_relevance_window=temporal_context_data.get("temporal_relevance_window", "24h")
            )
            
            logger.info("🔧 Analyzing error context for adaptive handling...")
            
            # Analyze error context
            error_context = await self.error_handler.analyze_error_context(
                error_details=error_details,
                temporal_context=temporal_context,
                available_tools=available_tools,
                success_criteria=success_criteria
            )
            
            # Determine if approach should be reconsidered
            reconsideration = await self.error_handler.should_reconsider_approach(error_context)
            
            response = f"# 🔧 MAESTRO Adaptive Error Analysis\n\n"
            response += f"**Error ID:** {error_context.error_id}\n"
            response += f"**Error Type:** {error_context.trigger.value}\n"
            response += f"**Severity:** {error_context.severity.value}\n"
            response += f"**Component:** {error_context.failed_component}\n\n"
            
            response += f"## Error Analysis:\n\n"
            response += f"**Message:** {error_context.error_message}\n"
            response += f"**Available Tools:** {', '.join(available_tools) if available_tools else 'None'}\n"
            response += f"**Attempted Approaches:** {', '.join(error_context.attempted_approaches) if error_context.attempted_approaches else 'None'}\n\n"
            
            response += f"## Reconsideration Analysis:\n\n"
            response += f"**Should Reconsider:** {'✅ YES' if reconsideration.should_reconsider else '❌ NO'}\n"
            response += f"**Confidence:** {reconsideration.confidence_score:.2f}\n"
            response += f"**Reasoning:** {reconsideration.reasoning}\n\n"
            
            if reconsideration.should_reconsider:
                if reconsideration.alternative_approaches:
                    response += f"## Alternative Approaches:\n\n"
                    for i, approach in enumerate(reconsideration.alternative_approaches, 1):
                        response += f"### {i}. {approach['approach']}\n"
                        response += f"**Description:** {approach['description']}\n"
                        response += f"**Tools Required:** {', '.join(approach['tools_required'])}\n"
                        response += f"**Confidence:** {approach['confidence']:.2f}\n\n"
                
                if reconsideration.recommended_tools:
                    response += f"## Recommended Tools:\n\n"
                    for tool in reconsideration.recommended_tools:
                        response += f"- {tool}\n"
                    response += "\n"
                
                if reconsideration.temporal_adjustments:
                    response += f"## Temporal Adjustments:\n\n"
                    for key, value in reconsideration.temporal_adjustments.items():
                        response += f"- {key.replace('_', ' ').title()}: {value}\n"
                    response += "\n"
                
                if reconsideration.modified_success_criteria:
                    response += f"## Modified Success Criteria:\n\n"
                    for i, criterion in enumerate(reconsideration.modified_success_criteria, 1):
                        response += f"{i}. {criterion.get('description', 'N/A')}\n"
                        validation_method = criterion.get('validation_method', 'N/A')
                        response += f"   - Validation: {validation_method}\n"
                        validation_tools = criterion.get('validation_tools', [])
                        if validation_tools:
                            response += f"   - Tools: {', '.join(validation_tools)}\n"
                        response += "\n"
            
            # Add error history summary
            error_summary = self.error_handler.get_error_analysis_summary()
            response += f"## Error History Summary:\n\n"
            response += f"- Total errors analyzed: {error_summary['total_errors']}\n"
            response += f"- Most frequent trigger: {max(error_summary['error_by_trigger'], key=error_summary['error_by_trigger'].get) if error_summary['error_by_trigger'] else 'None'}\n"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ MAESTRO error handler error: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"❌ **MAESTRO Error Handler Error**\n\nError: {str(e)}\n\nPlease check your error details and try again."
            )]
    
    async def handle_maestro_temporal_context(self, arguments: dict) -> list[types.TextContent]:
        """Handle maestro_temporal_context tool calls"""
        await self._ensure_initialized()
        
        try:
            task_description = arguments.get("task_description", "")
            information_sources = arguments.get("information_sources", [])
            temporal_requirements = arguments.get("temporal_requirements", {})
            
            if not task_description:
                return [types.TextContent(
                    type="text",
                    text="❌ **Task Description Required**\n\nPlease provide a task description for temporal context analysis."
                )]
            
            logger.info("🕐 Analyzing temporal context for information currency...")
            
            # Create temporal context
            current_time = datetime.now(timezone.utc)
            relevance_window = temporal_requirements.get("relevance_window", "24h")
            freshness_required = temporal_requirements.get("freshness_required", True)
            
            temporal_context = TemporalContext(
                current_timestamp=current_time,
                information_cutoff=None,
                task_deadline=None,
                context_freshness_required=freshness_required,
                temporal_relevance_window=relevance_window
            )
            
            response = f"# 🕐 MAESTRO Temporal Context Analysis\n\n"
            response += f"**Task:** {task_description}\n"
            response += f"**Current Time:** {current_time.isoformat()}\n"
            response += f"**Relevance Window:** {relevance_window}\n"
            response += f"**Freshness Required:** {'Yes' if freshness_required else 'No'}\n\n"
            
            if information_sources:
                response += f"## Information Source Analysis:\n\n"
                current_sources = []
                outdated_sources = []
                
                for source in information_sources:
                    source_timestamp_str = source.get("timestamp", "")
                    try:
                        # Try to parse timestamp
                        source_timestamp = datetime.fromisoformat(source_timestamp_str.replace('Z', '+00:00'))
                        is_current = temporal_context.is_information_current(source_timestamp)
                        
                        source_info = {
                            "source": source.get("source", "Unknown"),
                            "timestamp": source_timestamp,
                            "is_current": is_current,
                            "summary": source.get("content_summary", "")
                        }
                        
                        if is_current:
                            current_sources.append(source_info)
                        else:
                            outdated_sources.append(source_info)
                            
                    except Exception as e:
                        response += f"⚠️ **Warning:** Could not parse timestamp for {source.get('source', 'Unknown')}: {e}\n\n"
                
                if current_sources:
                    response += f"### ✅ Current Sources ({len(current_sources)}):\n\n"
                    for source in current_sources:
                        response += f"- **{source['source']}** ({source['timestamp'].strftime('%Y-%m-%d %H:%M UTC')})\n"
                        if source['summary']:
                            response += f"  - {source['summary']}\n"
                        response += "\n"
                
                if outdated_sources:
                    response += f"### ⚠️ Outdated Sources ({len(outdated_sources)}):\n\n"
                    for source in outdated_sources:
                        response += f"- **{source['source']}** ({source['timestamp'].strftime('%Y-%m-%d %H:%M UTC')})\n"
                        if source['summary']:
                            response += f"  - {source['summary']}\n"
                        response += "\n"
                    
                    response += f"### 🔄 Recommendations:\n\n"
                    response += f"The following sources are outdated based on your {relevance_window} relevance window:\n\n"
                    for source in outdated_sources:
                        response += f"- **{source['source']}** - Consider refreshing this information\n"
                    response += "\n"
                    response += f"**Suggested Action:** Use `maestro_search` to gather updated information for these topics.\n\n"
            
            else:
                response += f"## No Information Sources Provided\n\n"
                response += f"To analyze information currency, provide sources with timestamps.\n\n"
            
            # Add general temporal guidance
            response += f"## Temporal Context Guidance:\n\n"
            response += f"- **Information Freshness:** {'Required' if freshness_required else 'Not strictly required'}\n"
            response += f"- **Relevance Window:** Information older than {relevance_window} may need refreshing\n"
            response += f"- **Current UTC Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            
            if freshness_required:
                response += f"### 🔍 Information Refresh Strategy:\n\n"
                response += f"1. Use `maestro_search` with temporal filters to find recent information\n"
                response += f"2. Focus on sources from the last {relevance_window}\n"
                response += f"3. Cross-reference multiple recent sources for accuracy\n"
                response += f"4. Consider using `maestro_scrape` for specific recent content\n\n"
            
            if temporal_requirements.get("deadline"):
                deadline = temporal_requirements["deadline"]
                response += f"### ⏰ Deadline Consideration:\n\n"
                response += f"**Task Deadline:** {deadline}\n"
                response += f"**Recommendation:** Prioritize the most recent and reliable sources given the deadline constraint.\n\n"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"❌ MAESTRO temporal context error: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"❌ **MAESTRO Temporal Context Error**\n\nError: {str(e)}\n\nPlease check your input and try again."
            )]
    
    # Bridge methods for compatibility with existing interface
    async def search(self, query: str, max_results: int = 10, search_engine: str = "duckduckgo", 
                    temporal_filter: str = "any", result_format: str = "structured") -> str:
        """Bridge method for search functionality"""
        result = await self.handle_maestro_search({
            'query': query,
            'max_results': max_results,
            'search_engine': search_engine,
            'temporal_filter': temporal_filter,
            'result_format': result_format
        })
        return result[0].text if result else "Search failed"
    
    async def scrape(self, url: str, output_format: str = "markdown", selectors: list = None,
                    wait_time: int = 3, extract_links: bool = False) -> str:
        """Bridge method for scrape functionality"""
        result = await self.handle_maestro_scrape({
            'url': url,
            'output_format': output_format,
            'selectors': selectors or [],
            'wait_for': None,
            'extract_links': extract_links,
            'extract_images': False
        })
        return result[0].text if result else "Scrape failed"
    
    async def execute(self, command: str, execution_context: dict = None, 
                     timeout_seconds: int = 30, safe_mode: bool = True) -> str:
        """Bridge method for execute functionality"""
        result = await self.handle_maestro_execute({
            'command': command,
            'execution_context': execution_context or {},
            'timeout_seconds': timeout_seconds,
            'safe_mode': safe_mode
        })
        return result[0].text if result else "Execute failed"
    
    async def error_handler(self, error_message: str, error_context: dict = None,
                           recovery_suggestions: bool = True) -> str:
        """Bridge method for error handler functionality"""
        result = await self.handle_maestro_error_handler({
            'error_message': error_message,
            'error_context': error_context or {},
            'recovery_suggestions': recovery_suggestions
        })
        return result[0].text if result else "Error handling failed"
    
    async def temporal_context(self, temporal_query: str, time_range: dict = None,
                              temporal_precision: str = "day") -> str:
        """Bridge method for temporal context functionality"""
        result = await self.handle_maestro_temporal_context({
            'temporal_query': temporal_query,
            'time_range': time_range or {},
            'temporal_precision': temporal_precision
        })
        return result[0].text if result else "Temporal context failed"

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up enhanced tool handlers...")
        # LLM web tools don't need cleanup like puppeteer
        self._initialized = False
        logger.info("✅ Enhanced tool handlers cleaned up") 