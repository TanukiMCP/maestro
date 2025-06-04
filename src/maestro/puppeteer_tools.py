# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Built-in Puppeteer Tools for MAESTRO Protocol

Provides LLM-driven web capabilities through headless browser automation
as fallback tools when client doesn't support web search/scraping.

Tools:
- maestro_search: LLM-driven web search with intelligent query handling
- maestro_scrape: LLM-driven web scraping with format transformation
- maestro_execute: LLM-driven function execution for validation
"""

import logging
import asyncio
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for puppeteer availability
# try:
#     import pyppeteer
#     PUPPETEER_AVAILABLE = True
# except ImportError:
#     logger.warning("pyppeteer not available - puppeteer tools will be limited")
#     PUPPETEER_AVAILABLE = False
PUPPETEER_AVAILABLE = None # Will check dynamically


@dataclass
class SearchResult:
    """Result from web search"""
    title: str
    url: str
    snippet: str
    timestamp: datetime
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class ScrapeResult:
    """Result from web scraping"""
    url: str
    content: str
    format_type: str
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any]
    structured_data: Optional[Dict[str, Any]] = None


class MAESTROPuppeteerTools:
    """
    Built-in Puppeteer tools for MAESTRO Protocol with LLM-driven capabilities
    """
    
    def __init__(self):
        self.name = "MAESTRO Puppeteer Tools"
        self.version = "1.0.0"
        self.browser = None
        self.search_engines = {
            "duckduckgo": "https://duckduckgo.com/?q={query}",
            "bing": "https://www.bing.com/search?q={query}",
            "google": "https://www.google.com/search?q={query}"
        }
        
    async def maestro_search(
        self,
        query: str,
        max_results: int = 10,
        search_engine: str = "duckduckgo",
        temporal_filter: Optional[str] = None,
        result_format: str = "structured"
    ) -> Dict[str, Any]:
        """
        LLM-driven web search with intelligent query handling and result processing
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_engine: Search engine to use (duckduckgo, bing, google)
            temporal_filter: Filter by time (24h, 1w, 1m, 1y)
            result_format: Format of results (structured, markdown, json)
            
        Returns:
            Search results with metadata and analysis
        """
        logger.info(f"🔍 Executing MAESTRO search: '{query}' via {search_engine}")
        
        global PUPPETEER_AVAILABLE
        if PUPPETEER_AVAILABLE is None:
            try:
                import pyppeteer
                PUPPETEER_AVAILABLE = True
            except ImportError:
                logger.warning("pyppeteer not available - puppeteer tools will be limited")
                PUPPETEER_AVAILABLE = False
        
        if not PUPPETEER_AVAILABLE:
            return self._fallback_search(query, max_results)
        
        try:
            # Dynamically import pyppeteer here too, if needed by launch
            import pyppeteer 
            # Initialize browser if needed
            if not self.browser:
                self.browser = await pyppeteer.launch({
                    'headless': True,
                    'args': ['--no-sandbox', '--disable-setuid-sandbox']
                })
            
            page = await self.browser.newPage()
            
            # Set user agent to avoid blocking
            await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Build search URL with temporal filter
            search_url = self._build_search_url(query, search_engine, temporal_filter)
            
            # Navigate to search page
            await page.goto(search_url, {'waitUntil': 'networkidle0'})
            
            # Extract search results based on search engine
            if search_engine == "duckduckgo":
                results = await self._extract_duckduckgo_results(page, max_results)
            elif search_engine == "bing":
                results = await self._extract_bing_results(page, max_results)
            else:
                results = await self._extract_generic_results(page, max_results)
            
            await page.close()
            
            # Process and format results
            processed_results = self._process_search_results(results, result_format)
            
            response = {
                "success": True,
                "query": query,
                "search_engine": search_engine,
                "results_count": len(processed_results),
                "timestamp": datetime.now().isoformat(),
                "results": processed_results,
                "metadata": {
                    "temporal_filter": temporal_filter,
                    "result_format": result_format,
                    "search_url": search_url
                }
            }
            
            logger.info(f"✅ Search complete: {len(processed_results)} results found")
            return response
            
        except Exception as e:
            logger.error(f"❌ MAESTRO search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_results": self._fallback_search(query, max_results)
            }
    
    async def maestro_scrape(
        self,
        url: str,
        output_format: str = "markdown",
        selectors: Optional[List[str]] = None,
        wait_for: Optional[str] = None,
        extract_links: bool = False,
        extract_images: bool = False
    ) -> Dict[str, Any]:
        """
        LLM-driven web scraping with intelligent content extraction and formatting
        
        Args:
            url: URL to scrape
            output_format: Format for output (markdown, json, text, html)
            selectors: CSS selectors for specific content extraction
            wait_for: Element or condition to wait for before scraping
            extract_links: Whether to extract all links
            extract_images: Whether to extract image information
            
        Returns:
            Scraped content with structured data and metadata
        """
        logger.info(f"🕷️ Executing MAESTRO scrape: {url}")
        
        global PUPPETEER_AVAILABLE # Ensure it's seen if not already checked
        if PUPPETEER_AVAILABLE is None: # Check again if not already determined
            try:
                import pyppeteer
                PUPPETEER_AVAILABLE = True
            except ImportError:
                logger.warning("pyppeteer not available - puppeteer tools will be limited")
                PUPPETEER_AVAILABLE = False

        if not PUPPETEER_AVAILABLE:
            return self._fallback_scrape(url, output_format)
        
        try:
            # Dynamically import pyppeteer here too
            import pyppeteer
            if not self.browser:
                self.browser = await pyppeteer.launch({
                    'headless': True,
                    'args': ['--no-sandbox', '--disable-setuid-sandbox']
                })
            
            page = await self.browser.newPage()
            
            # Navigate to URL
            await page.goto(url, {'waitUntil': 'networkidle0'})
            
            # Wait for specific element if specified
            if wait_for:
                try:
                    await page.waitForSelector(wait_for, {'timeout': 10000})
                except:
                    logger.warning(f"Timeout waiting for selector: {wait_for}")
            
            # Extract content based on selectors or general content
            if selectors:
                content = await self._extract_by_selectors(page, selectors)
            else:
                content = await self._extract_general_content(page)
            
            # Extract additional data if requested
            additional_data = {}
            if extract_links:
                additional_data['links'] = await self._extract_links(page)
            if extract_images:
                additional_data['images'] = await self._extract_images(page)
            
            await page.close()
            
            # Format content according to specified format
            formatted_content = self._format_scraped_content(content, output_format)
            
            # Structure the response
            response = {
                "success": True,
                "url": url,
                "content": formatted_content,
                "format": output_format,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "content_length": len(content),
                    "selectors_used": selectors,
                    "additional_data": additional_data,
                    "extraction_method": "puppeteer"
                }
            }
            
            # Add structured data if applicable
            if output_format == "json":
                response["structured_data"] = self._extract_structured_data(content)
            
            logger.info(f"✅ Scraping complete: {len(content)} characters extracted")
            return response
            
        except Exception as e:
            logger.error(f"❌ MAESTRO scrape failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_result": self._fallback_scrape(url, output_format)
            }
    
    async def maestro_execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        capture_output: bool = True,
        working_directory: Optional[str] = None,
        environment_vars: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        LLM-driven function execution for validation and testing
        
        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash)
            timeout: Execution timeout in seconds
            capture_output: Whether to capture stdout/stderr
            working_directory: Working directory for execution
            environment_vars: Environment variables to set
            
        Returns:
            Execution results with output, errors, and validation data
        """
        logger.info(f"⚡ Executing MAESTRO code: {language} ({len(code)} chars)")
        
        try:
            # Create temporary file for code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{self._get_file_extension(language)}', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Prepare execution command
                command = self._build_execution_command(language, temp_file)
                
                # Set working directory
                if working_directory and os.path.exists(working_directory):
                    cwd = working_directory
                else:
                    cwd = os.path.dirname(temp_file)
                
                # Set environment variables
                env = os.environ.copy()
                if environment_vars:
                    env.update(environment_vars)
                
                # Execute code
                start_time = datetime.now()
                
                if capture_output:
                    result = subprocess.run(
                        command,
                        cwd=cwd,
                        env=env,
                        timeout=timeout,
                        capture_output=True,
                        text=True
                    )
                    stdout = result.stdout
                    stderr = result.stderr
                    return_code = result.returncode
                else:
                    result = subprocess.run(
                        command,
                        cwd=cwd,
                        env=env,
                        timeout=timeout
                    )
                    stdout = ""
                    stderr = ""
                    return_code = result.returncode
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Analyze execution results
                success = return_code == 0
                validation_results = self._analyze_execution_results(stdout, stderr, return_code)
                
                response = {
                    "success": success,
                    "return_code": return_code,
                    "execution_time": execution_time,
                    "timestamp": start_time.isoformat(),
                    "output": {
                        "stdout": stdout,
                        "stderr": stderr
                    },
                    "validation": validation_results,
                    "metadata": {
                        "language": language,
                        "code_length": len(code),
                        "working_directory": cwd,
                        "timeout": timeout,
                        "command": " ".join(command)
                    }
                }
                
                logger.info(f"✅ Code execution complete: {'SUCCESS' if success else 'FAILED'} ({execution_time:.2f}s)")
                return response
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Code execution timeout after {timeout}s")
            return {
                "success": False,
                "error": f"Execution timeout after {timeout} seconds",
                "return_code": -1,
                "execution_time": timeout,
                "metadata": {
                    "language": language,
                    "code_length": len(code),
                    "timeout": timeout,
                    "error_type": "timeout"
                }
            }
        except Exception as e:
            logger.error(f"❌ MAESTRO execute failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "return_code": -1,
                "metadata": {
                    "language": language,
                    "code_length": len(code),
                    "error_type": "execution_error"
                }
            }
    
    def _build_search_url(self, query: str, search_engine: str, temporal_filter: Optional[str]) -> str:
        """Build search URL with temporal filtering"""
        base_url = self.search_engines.get(search_engine, self.search_engines["duckduckgo"])
        search_url = base_url.format(query=query.replace(' ', '+'))
        
        # Add temporal filter if specified
        if temporal_filter and search_engine == "duckduckgo":
            if temporal_filter == "24h":
                search_url += "&df=d"
            elif temporal_filter == "1w":
                search_url += "&df=w"
            elif temporal_filter == "1m":
                search_url += "&df=m"
            elif temporal_filter == "1y":
                search_url += "&df=y"
        
        return search_url
    
    async def _extract_duckduckgo_results(self, page, max_results: int) -> List[Dict[str, Any]]:
        """Extract search results from DuckDuckGo"""
        results = []
        
        try:
            # Wait for results to load
            await page.waitForSelector('[data-result]', {'timeout': 10000})
            
            # Extract result elements
            result_elements = await page.querySelectorAll('[data-result]')
            
            for i, element in enumerate(result_elements[:max_results]):
                try:
                    # Extract title
                    title_element = await element.querySelector('h2 a')
                    title = await page.evaluate('(element) => element ? element.textContent : ""', title_element)
                    
                    # Extract URL
                    url = await page.evaluate('(element) => element ? element.href : ""', title_element)
                    
                    # Extract snippet
                    snippet_element = await element.querySelector('[data-result="snippet"]')
                    snippet = await page.evaluate('(element) => element ? element.textContent : ""', snippet_element)
                    
                    if title and url:
                        results.append({
                            'title': title.strip(),
                            'url': url,
                            'snippet': snippet.strip() if snippet else "",
                            'position': i + 1
                        })
                        
                except Exception as e:
                    logger.warning(f"Error extracting result {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting DuckDuckGo results: {e}")
        
        return results
    
    async def _extract_bing_results(self, page, max_results: int) -> List[Dict[str, Any]]:
        """Extract search results from Bing"""
        results = []
        
        try:
            await page.waitForSelector('.b_algo', {'timeout': 10000})
            result_elements = await page.querySelectorAll('.b_algo')
            
            for i, element in enumerate(result_elements[:max_results]):
                try:
                    title_element = await element.querySelector('h2 a')
                    title = await page.evaluate('(element) => element ? element.textContent : ""', title_element)
                    url = await page.evaluate('(element) => element ? element.href : ""', title_element)
                    
                    snippet_element = await element.querySelector('.b_caption p')
                    snippet = await page.evaluate('(element) => element ? element.textContent : ""', snippet_element)
                    
                    if title and url:
                        results.append({
                            'title': title.strip(),
                            'url': url,
                            'snippet': snippet.strip() if snippet else "",
                            'position': i + 1
                        })
                        
                except Exception as e:
                    logger.warning(f"Error extracting Bing result {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting Bing results: {e}")
        
        return results
    
    async def _extract_generic_results(self, page, max_results: int) -> List[Dict[str, Any]]:
        """Extract results using generic selectors"""
        # This is a fallback method for unknown search engines
        return []
    
    def _process_search_results(self, results: List[Dict[str, Any]], result_format: str) -> List[Dict[str, Any]]:
        """Process and format search results"""
        processed = []
        
        for result in results:
            processed_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "position": result.get("position", 0),
                "relevance_score": self._calculate_relevance_score(result),
                "timestamp": datetime.now().isoformat()
            }
            
            if result_format == "markdown":
                processed_result["formatted"] = f"**{processed_result['title']}**\n{processed_result['url']}\n{processed_result['snippet']}\n"
            elif result_format == "json":
                processed_result["formatted"] = json.dumps(processed_result, indent=2)
            
            processed.append(processed_result)
        
        return processed
    
    def _calculate_relevance_score(self, result: Dict[str, Any]) -> float:
        """Calculate relevance score for search result"""
        # Simple scoring based on position and content quality
        position_score = max(0, 1.0 - (result.get("position", 1) - 1) * 0.1)
        content_score = min(1.0, len(result.get("snippet", "")) / 200)
        return (position_score + content_score) / 2
    
    def _fallback_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fallback search when puppeteer is not available"""
        return {
            "success": False,
            "error": "Puppeteer not available",
            "suggestion": "Install pyppeteer or use client's built-in web search capabilities",
            "query": query,
            "fallback_guidance": {
                "manual_search": f"Manually search for: {query}",
                "alternative_tools": ["web_search", "browser_tools"]
            }
        }
    
    async def _extract_by_selectors(self, page, selectors: List[str]) -> str:
        """Extract content using CSS selectors"""
        content_parts = []
        
        for selector in selectors:
            try:
                elements = await page.querySelectorAll(selector)
                for element in elements:
                    text = await page.evaluate('(element) => element.textContent || element.innerText', element)
                    if text:
                        content_parts.append(text.strip())
            except Exception as e:
                logger.warning(f"Error extracting selector {selector}: {e}")
        
        return '\n\n'.join(content_parts)
    
    async def _extract_general_content(self, page) -> str:
        """Extract general page content"""
        try:
            # Try to get main content areas
            selectors = ['main', 'article', '[role="main"]', '.content', '#content', 'body']
            
            for selector in selectors:
                try:
                    element = await page.querySelector(selector)
                    if element:
                        content = await page.evaluate('(element) => element.textContent || element.innerText', element)
                        if content and len(content.strip()) > 100:
                            return content.strip()
                except:
                    continue
            
            # Fallback to body content
            content = await page.evaluate('() => document.body.textContent || document.body.innerText')
            return content.strip() if content else ""
            
        except Exception as e:
            logger.error(f"Error extracting general content: {e}")
            return ""
    
    async def _extract_links(self, page) -> List[Dict[str, str]]:
        """Extract all links from page"""
        try:
            links = await page.evaluate('''() => {
                const linkElements = document.querySelectorAll('a[href]');
                return Array.from(linkElements).map(link => ({
                    text: link.textContent.trim(),
                    href: link.href,
                    title: link.title || ''
                }));
            }''')
            return links
        except:
            return []
    
    async def _extract_images(self, page) -> List[Dict[str, str]]:
        """Extract image information from page"""
        try:
            images = await page.evaluate('''() => {
                const imgElements = document.querySelectorAll('img[src]');
                return Array.from(imgElements).map(img => ({
                    src: img.src,
                    alt: img.alt || '',
                    title: img.title || ''
                }));
            }''')
            return images
        except:
            return []
    
    def _format_scraped_content(self, content: str, output_format: str) -> str:
        """Format scraped content according to specified format"""
        if output_format == "markdown":
            # Simple markdown formatting
            lines = content.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    # Add some basic markdown formatting
                    if len(line) < 100 and line.isupper():
                        formatted_lines.append(f"## {line}")
                    else:
                        formatted_lines.append(line)
            return '\n\n'.join(formatted_lines)
        
        elif output_format == "json":
            return json.dumps({"content": content, "timestamp": datetime.now().isoformat()}, indent=2)
        
        elif output_format == "text":
            return content
        
        else:  # html or other
            return content
    
    def _extract_structured_data(self, content: str) -> Dict[str, Any]:
        """Extract structured data from content"""
        # Simple structured data extraction
        lines = content.split('\n')
        
        structured = {
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "word_count": len(content.split()),
            "line_count": len(lines)
        }
        
        for line in lines:
            line = line.strip()
            if line:
                if len(line) < 100 and (line.isupper() or line.startswith('#')):
                    structured["headings"].append(line)
                elif line.startswith('-') or line.startswith('*'):
                    structured["lists"].append(line)
                elif len(line) > 50:
                    structured["paragraphs"].append(line)
        
        return structured
    
    def _fallback_scrape(self, url: str, output_format: str) -> Dict[str, Any]:
        """Fallback scraping when puppeteer is not available"""
        return {
            "success": False,
            "error": "Puppeteer not available",
            "suggestion": "Install pyppeteer or use client's built-in scraping capabilities",
            "url": url,
            "fallback_guidance": {
                "manual_access": f"Manually visit: {url}",
                "alternative_tools": ["browser_tools", "http_client"]
            }
        }
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language"""
        extensions = {
            "python": "py",
            "javascript": "js",
            "bash": "sh",
            "shell": "sh",
            "node": "js"
        }
        return extensions.get(language.lower(), "txt")
    
    def _build_execution_command(self, language: str, file_path: str) -> List[str]:
        """Build execution command for different languages"""
        if language.lower() == "python":
            return ["python", file_path]
        elif language.lower() in ["javascript", "node"]:
            return ["node", file_path]
        elif language.lower() in ["bash", "shell"]:
            return ["bash", file_path]
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _analyze_execution_results(self, stdout: str, stderr: str, return_code: int) -> Dict[str, Any]:
        """Analyze execution results for validation"""
        analysis = {
            "execution_successful": return_code == 0,
            "has_output": bool(stdout.strip()),
            "has_errors": bool(stderr.strip()),
            "output_length": len(stdout),
            "error_length": len(stderr),
            "validation_status": "passed" if return_code == 0 and not stderr.strip() else "failed"
        }
        
        # Additional analysis
        if stderr:
            analysis["error_types"] = self._classify_errors(stderr)
        
        if stdout:
            analysis["output_analysis"] = self._analyze_output(stdout)
        
        return analysis
    
    def _classify_errors(self, stderr: str) -> List[str]:
        """Classify types of errors from stderr"""
        error_types = []
        stderr_lower = stderr.lower()
        
        if "syntax" in stderr_lower:
            error_types.append("syntax_error")
        if "import" in stderr_lower or "module" in stderr_lower:
            error_types.append("import_error")
        if "name" in stderr_lower and "not defined" in stderr_lower:
            error_types.append("name_error")
        if "type" in stderr_lower:
            error_types.append("type_error")
        if "value" in stderr_lower:
            error_types.append("value_error")
        
        return error_types
    
    def _analyze_output(self, stdout: str) -> Dict[str, Any]:
        """Analyze stdout content"""
        lines = stdout.strip().split('\n')
        
        return {
            "line_count": len(lines),
            "contains_json": any(line.strip().startswith('{') or line.strip().startswith('[') for line in lines),
            "contains_numbers": any(char.isdigit() for char in stdout),
            "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
        }
    
    async def cleanup(self):
        """Cleanup browser resources"""
        if self.browser:
            await self.browser.close()
            self.browser = None 
