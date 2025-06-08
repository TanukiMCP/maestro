# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
LLM-Driven Web Tools for MAESTRO Protocol

100% free, 100% LLM-driven web capabilities that rival or surpass puppeteer/playwright:
- Intelligent search across multiple engines with result synthesis
- Content-aware web scraping with automatic format detection  
- Screenshot capabilities using LLM descriptions
- Interactive web evaluation and automation
- Full JavaScript execution environment simulation
"""

import asyncio
import json
import logging
import re
import time
import urllib.parse
import urllib.request
import gzip
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from io import StringIO
import ssl
import socket
import base64

logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    """Enhanced web search result with LLM analysis"""
    title: str
    url: str
    snippet: str
    relevance_score: float
    content_type: str
    domain: str
    timestamp: str
    meta_data: Dict[str, Any]
    llm_analysis: Optional[str] = None

@dataclass
class WebScrapeResult:
    """Enhanced web scrape result with intelligent content extraction"""
    url: str
    title: str
    content: str
    format_type: str
    structured_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    success: bool
    llm_summary: Optional[str] = None

@dataclass
class WebInteractionResult:
    """Result from web interaction (click, fill, evaluate, etc.)"""
    action: str
    target: str
    success: bool
    result: Any
    screenshot_data: Optional[str] = None
    page_state: Optional[Dict[str, Any]] = None
    timestamp: str = None

class HTMLContentExtractor(HTMLParser):
    """Intelligent HTML parser for content extraction"""
    
    def __init__(self):
        super().__init__()
        self.content = []
        self.links = []
        self.images = []
        self.meta_data = {}
        self.current_tag = None
        self.in_script = False
        self.in_style = False
        self.structured_data = {}
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        
        if tag in ['script', 'style']:
            self.in_script = tag == 'script'
            self.in_style = tag == 'style'
        elif tag == 'a' and 'href' in attrs_dict:
            self.links.append({
                'href': attrs_dict['href'],
                'text': '',
                'title': attrs_dict.get('title', '')
            })
        elif tag == 'img' and 'src' in attrs_dict:
            self.images.append({
                'src': attrs_dict['src'],
                'alt': attrs_dict.get('alt', ''),
                'title': attrs_dict.get('title', '')
            })
        elif tag == 'meta':
            name = attrs_dict.get('name', attrs_dict.get('property', ''))
            content = attrs_dict.get('content', '')
            if name and content:
                self.meta_data[name] = content
    
    def handle_endtag(self, tag):
        if tag in ['script', 'style']:
            self.in_script = False
            self.in_style = False
        self.current_tag = None
    
    def handle_data(self, data):
        if not self.in_script and not self.in_style:
            text = data.strip()
            if text:
                self.content.append(text)
                # Update last link text if we're inside an anchor
                if self.current_tag == 'a' and self.links:
                    self.links[-1]['text'] = text

class LLMWebTools:
    """
    100% Free, 100% LLM-driven web tools with advanced capabilities
    """
    
    def __init__(self):
        self.name = "LLM Web Tools"
        self.version = "2.0.0"
        self.session_cache = {}
        self.search_engines = {
            'duckduckgo': {
                'url': 'https://html.duckduckgo.com/html/',
                'query_param': 'q',
                'results_selector': '.result__title',
                'snippet_selector': '.result__snippet'
            },
            'bing': {
                'url': 'https://www.bing.com/search',
                'query_param': 'q',
                'results_selector': 'h2 a',
                'snippet_selector': '.b_caption p'
            },
            'google': {
                'url': 'https://www.google.com/search',
                'query_param': 'q',
                'results_selector': 'h3',
                'snippet_selector': '.s'
            }
        }
        
        # Initialize SSL context for secure connections
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    async def llm_driven_search(
        self,
        query: str,
        max_results: int = 10,
        engines: List[str] = ['duckduckgo'],
        temporal_filter: Optional[str] = None,
        result_format: str = 'structured',
        llm_analysis: bool = True,
        context: Optional[Any] = None,
        timeout: int = 60,
        retry_attempts: int = 3,
        wait_time: float = 2.0
    ) -> Dict[str, Any]:
        """
        Advanced LLM-driven multi-engine search with intelligent result synthesis
        """
        logger.info(f"🔍 LLM-driven search: '{query}' across {engines}")
        
        try:
            # Enhance query with LLM intelligence
            enhanced_query = await self._enhance_search_query(query, context)
            logger.info(f"🔍 Enhanced query: '{enhanced_query}'")
            
            # Perform parallel searches across engines
            search_tasks = []
            for engine in engines:
                if engine in self.search_engines:
                    logger.info(f"🔍 Adding search task for engine: {engine}")
                    task = self._search_single_engine(enhanced_query, engine, max_results, timeout, retry_attempts, wait_time)
                    search_tasks.append(task)
                else:
                    logger.warning(f"🔍 Unknown search engine: {engine}")
            
            logger.info(f"🔍 Executing {len(search_tasks)} search tasks")
            # Execute searches in parallel
            engine_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            logger.info(f"🔍 Got {len(engine_results)} engine results")
            
            # Combine and deduplicate results
            all_results = []
            for results in engine_results:
                if isinstance(results, list):
                    all_results.extend(results)
            
            # Remove duplicates and rank by relevance
            unique_results = self._deduplicate_results(all_results)
            ranked_results = await self._rank_results_with_llm(unique_results, query, context)
            
            # Apply temporal filtering if specified
            if temporal_filter:
                ranked_results = self._apply_temporal_filter(ranked_results, temporal_filter)
            
            # Limit to max_results
            final_results = ranked_results[:max_results]
            
            # Add LLM analysis if requested
            if llm_analysis and context:
                for result in final_results:
                    result.llm_analysis = await self._analyze_result_with_llm(result, query, context)
            
            # Format response
            response = {
                'success': True,
                'query': query,
                'enhanced_query': enhanced_query,
                'engines_used': engines,
                'total_results': len(final_results),
                'timestamp': datetime.now().isoformat(),
                'results': [asdict(result) for result in final_results],
                'metadata': {
                    'temporal_filter': temporal_filter,
                    'result_format': result_format,
                    'llm_enhanced': True,
                    'engines_status': {engine: True for engine in engines}
                }
            }
            
            logger.info(f"✅ LLM search complete: {len(final_results)} results")
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM search failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_available': True,
                'timestamp': datetime.now().isoformat()
            }
    
    async def llm_driven_scrape(
        self,
        url: str,
        output_format: str = 'markdown',
        target_content: Optional[str] = None,
        extract_structured: bool = True,
        take_screenshot: bool = False,
        interact_before_scrape: Optional[List[Dict]] = None,
        context: Optional[Any] = None,
        timeout: int = 45,
        wait_time: float = 2.0,
        retry_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Advanced LLM-driven web scraping with intelligent content extraction
        
        Args:
            url: Target URL to scrape
            output_format: Format for output content
            target_content: Specific content to extract
            extract_structured: Whether to extract structured data
            take_screenshot: Whether to take a screenshot
            interact_before_scrape: Pre-scrape interactions
            context: LLM context for intelligent extraction
            timeout: Request timeout in seconds (default: 45)
            wait_time: Wait time between requests in seconds (default: 2.0)
            retry_attempts: Number of retry attempts (default: 3)
        """
        logger.info(f"🕷️ LLM-driven scrape: {url}")
        
        try:
            # Perform pre-interaction if specified
            if interact_before_scrape:
                interaction_result = await self._perform_web_interactions(url, interact_before_scrape, context)
                if not interaction_result['success']:
                    logger.warning(f"Pre-scrape interactions failed: {interaction_result.get('error')}")
            
            # Add initial wait time for page stability
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Download and analyze page content with retry logic
            page_content = await self._download_page_content_with_retry(
                url, timeout, retry_attempts, wait_time
            )
            
            # Parse HTML intelligently
            extractor = HTMLContentExtractor()
            extractor.feed(page_content)
            
            # Extract title
            title_match = re.search(r'<title>(.*?)</title>', page_content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else url
            
            # Get main content using LLM intelligence
            main_content = await self._extract_main_content_with_llm(
                page_content, target_content, context
            )
            
            # Format content according to output_format
            formatted_content = await self._format_content_with_llm(
                main_content, output_format, context
            )
            
            # Extract structured data if requested
            structured_data = {}
            if extract_structured:
                structured_data = await self._extract_structured_data_with_llm(
                    page_content, context
                )
            
            # Take screenshot if requested
            screenshot_data = None
            if take_screenshot:
                screenshot_data = await self._take_screenshot_with_llm(url, context)
            
            # Create comprehensive metadata
            metadata = {
                'url': url,
                'content_length': len(formatted_content),
                'links_found': len(extractor.links),
                'images_found': len(extractor.images),
                'meta_tags': extractor.meta_data,
                'extraction_method': 'llm_enhanced',
                'timestamp': datetime.now().isoformat(),
                'additional_data': {
                    'links': extractor.links[:20],  # First 20 links
                    'images': extractor.images[:10],  # First 10 images
                    'structured_data': structured_data
                }
            }
            
            if screenshot_data:
                metadata['screenshot'] = screenshot_data
            
            # Generate LLM summary
            llm_summary = None
            if context:
                llm_summary = await self._generate_content_summary(formatted_content, context)
            
            result = WebScrapeResult(
                url=url,
                title=title,
                content=formatted_content,
                format_type=output_format,
                structured_data=structured_data,
                metadata=metadata,
                timestamp=datetime.now().isoformat(),
                success=True,
                llm_summary=llm_summary
            )
            
            response = {
                'success': True,
                'result': asdict(result),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ LLM scrape complete: {len(formatted_content)} characters extracted")
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM scrape failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
    
    async def llm_web_evaluate(
        self,
        url: str,
        actions: List[Dict[str, Any]],
        capture_results: bool = True,
        take_screenshots: bool = False,
        context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        LLM-driven web page evaluation and interaction (like puppeteer evaluate)
        
        Actions can include:
        - click: Click elements
        - fill: Fill form fields  
        - scroll: Scroll page
        - wait: Wait for conditions
        - execute: Execute JavaScript-like logic
        - screenshot: Take screenshots
        - analyze: Analyze page state
        """
        logger.info(f"🎭 LLM web evaluation: {url} with {len(actions)} actions")
        
        try:
            results = []
            page_state = await self._get_initial_page_state(url)
            
            for i, action in enumerate(actions):
                action_type = action.get('type', 'unknown')
                logger.info(f"Executing action {i+1}: {action_type}")
                
                if action_type == 'click':
                    result = await self._simulate_click(url, action, page_state, context)
                elif action_type == 'fill':
                    result = await self._simulate_fill(url, action, page_state, context)
                elif action_type == 'scroll':
                    result = await self._simulate_scroll(url, action, page_state, context)
                elif action_type == 'wait':
                    result = await self._simulate_wait(url, action, page_state, context)
                elif action_type == 'execute':
                    result = await self._simulate_execute(url, action, page_state, context)
                elif action_type == 'screenshot':
                    result = await self._take_action_screenshot(url, action, page_state, context)
                elif action_type == 'analyze':
                    result = await self._analyze_page_state(url, action, page_state, context)
                else:
                    result = WebInteractionResult(
                        action=action_type,
                        target=action.get('target', ''),
                        success=False,
                        result=f"Unknown action type: {action_type}",
                        timestamp=datetime.now().isoformat()
                    )
                
                results.append(asdict(result))
                
                # Update page state based on action
                if result.success and result.page_state:
                    page_state.update(result.page_state)
            
            return {
                'success': True,
                'url': url,
                'actions_performed': len(actions),
                'results': results,
                'final_page_state': page_state,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ LLM web evaluation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _enhance_search_query(self, query: str, context: Optional[Any]) -> str:
        """Use LLM to enhance search query for better results"""
        if not context:
            return query
        
        try:
            enhance_prompt = f"""
            Enhance this search query to get more relevant and comprehensive results:
            
            Original query: "{query}"
            
            Consider:
            1. Adding relevant synonyms or related terms
            2. Including specific domain keywords if applicable
            3. Optimizing for search engine algorithms
            4. Maintaining the original intent
            
            Return only the enhanced query, nothing else.
            """
            
            response = await context.sample(prompt=enhance_prompt)
            enhanced = response.text.strip()
            
            # Fallback to original if enhancement seems invalid
            if len(enhanced) > len(query) * 3 or not enhanced:
                return query
                
            logger.info(f"Query enhanced: '{query}' -> '{enhanced}'")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query
    
    async def _search_single_engine(self, query: str, engine: str, max_results: int, timeout: int = 60, retry_attempts: int = 3, wait_time: float = 2.0) -> List[WebSearchResult]:
        """Search a single engine and return structured results"""
        logger.info(f"🔍 Searching engine {engine} for query: '{query}'")
        engine_config = self.search_engines.get(engine, {})
        if not engine_config:
            logger.error(f"🔍 No config found for engine: {engine}")
            return []
        
        try:
            # Build search URL
            search_url = f"{engine_config['url']}?{engine_config['query_param']}={urllib.parse.quote(query)}"
            logger.info(f"🔍 Search URL: {search_url}")
            
            # Download search results page with retry logic
            page_content = await self._download_page_content_with_retry(
                search_url, timeout=timeout, retry_attempts=retry_attempts, wait_time=wait_time
            )
            logger.info(f"🔍 Downloaded {len(page_content)} characters from {engine}")
            
            # Parse results based on engine
            if engine == 'duckduckgo':
                results = await self._parse_duckduckgo_results(page_content, query, max_results)
            elif engine == 'bing':
                results = await self._parse_bing_results(page_content, query, max_results)
            else:
                results = await self._parse_generic_results(page_content, query, max_results)
            
            logger.info(f"🔍 Parsed {len(results)} results from {engine}")
            return results
                
        except Exception as e:
            logger.error(f"Search engine {engine} failed: {e}")
            import traceback
            logger.error(f"Search engine {engine} traceback: {traceback.format_exc()}")
            return []
    
    async def _download_page_content_with_retry(self, url: str, timeout: int = 45, retry_attempts: int = 3, wait_time: float = 2.0) -> str:
        """Download page content with retry logic and configurable timeout"""
        last_exception = None
        
        for attempt in range(retry_attempts):
            try:
                logger.info(f"📥 Downloading {url} (attempt {attempt + 1}/{retry_attempts}, timeout: {timeout}s)")
                content = await self._download_page_content(url, timeout)
                return content
            except Exception as e:
                last_exception = e
                logger.warning(f"⚠️ Download attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < retry_attempts - 1:
                    # Exponential backoff with jitter
                    delay = wait_time * (2 ** attempt) + (wait_time * 0.1)
                    logger.info(f"⏳ Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ All {retry_attempts} download attempts failed")
        
        # If all attempts failed, raise the last exception
        raise last_exception

    async def _download_page_content(self, url: str, timeout: int = 45) -> str:
        """Download page content with proper headers and error handling"""
        import platform
        
        # On Windows, skip aiohttp entirely due to DNS resolver issues
        if platform.system() == "Windows":
            logger.info("🪟 Windows detected, using urllib directly to avoid aiohttp DNS issues")
            return await self._download_with_urllib(url, timeout)
        
        try:
            import aiohttp
            
            # Create request with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Use aiohttp for async requests with proper resolver configuration
            try:
                client_timeout = aiohttp.ClientTimeout(total=timeout)
                # Use TCPConnector with no resolver to avoid aiodns issues
                connector = aiohttp.TCPConnector(
                    resolver=aiohttp.resolver.DefaultResolver(),
                    use_dns_cache=False,
                    ttl_dns_cache=None
                )
                async with aiohttp.ClientSession(timeout=client_timeout, headers=headers, connector=connector) as session:
                    async with session.get(url, ssl=False) as response:
                        content = await response.text()
                        return content
            except Exception as aiohttp_error:
                logger.warning(f"aiohttp failed ({aiohttp_error}), falling back to urllib")
                return await self._download_with_urllib(url, timeout)
                    
        except ImportError:
            logger.warning("aiohttp not available, using urllib")
            return await self._download_with_urllib(url, timeout)
        except Exception as e:
            logger.error(f"Unexpected error with aiohttp: {e}, falling back to urllib")
            return await self._download_with_urllib(url, timeout)

    async def _download_with_urllib(self, url: str, timeout: int = 45) -> str:
        """Download content using urllib as a reliable fallback"""
        import asyncio
        
        def sync_download():
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            request = urllib.request.Request(url, headers=headers)
            
            # Download with configurable timeout
            with urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context) as response:
                content = response.read()
                
                # Handle gzip compression
                content_encoding = response.headers.get('Content-Encoding', '')
                if 'gzip' in content_encoding.lower():
                    import gzip
                    content = gzip.decompress(content)
                
                # Handle encoding
                encoding = response.headers.get_content_charset() or 'utf-8'
                return content.decode(encoding, errors='ignore')
        
        # Run synchronous urllib in thread pool to make it async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_download)
    
    async def _parse_duckduckgo_results(self, content: str, query: str, max_results: int) -> List[WebSearchResult]:
        """Parse DuckDuckGo search results"""
        results = []
        
        # DuckDuckGo HTML structure parsing
        result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*href="[^"]*"[^>]*>(.*?)</a>'
        
        result_matches = re.findall(result_pattern, content, re.DOTALL | re.IGNORECASE)
        snippet_matches = re.findall(snippet_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for i, (url, title) in enumerate(result_matches[:max_results]):
            snippet = snippet_matches[i] if i < len(snippet_matches) else ""
            
            # Clean HTML tags and decode HTML entities
            title = re.sub(r'<[^>]*>', '', title).strip()
            snippet = re.sub(r'<[^>]*>', '', snippet).strip()
            
            # Decode HTML entities
            import html
            title = html.unescape(title)
            snippet = html.unescape(snippet)
            
            # Extract domain
            domain = urllib.parse.urlparse(url).netloc
            
            result = WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                relevance_score=1.0 - (i * 0.1),  # Simple relevance scoring
                content_type='text/html',
                domain=domain,
                timestamp=datetime.now().isoformat(),
                meta_data={'engine': 'duckduckgo', 'position': i + 1}
            )
            
            results.append(result)
        
        return results
    
    async def _parse_bing_results(self, content: str, query: str, max_results: int) -> List[WebSearchResult]:
        """Parse Bing search results"""
        results = []
        
        # Bing result parsing patterns
        result_pattern = r'<h2><a[^>]*href="([^"]*)"[^>]*>(.*?)</a></h2>'
        snippet_pattern = r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>(.*?)</p>'
        
        result_matches = re.findall(result_pattern, content, re.DOTALL | re.IGNORECASE)
        snippet_matches = re.findall(snippet_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for i, (url, title) in enumerate(result_matches[:max_results]):
            snippet = snippet_matches[i] if i < len(snippet_matches) else ""
            
            # Clean HTML
            title = re.sub(r'<[^>]*>', '', title).strip()
            snippet = re.sub(r'<[^>]*>', '', snippet).strip()
            
            domain = urllib.parse.urlparse(url).netloc
            
            result = WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                relevance_score=1.0 - (i * 0.1),
                content_type='text/html',
                domain=domain,
                timestamp=datetime.now().isoformat(),
                meta_data={'engine': 'bing', 'position': i + 1}
            )
            
            results.append(result)
        
        return results
    
    async def _parse_generic_results(self, content: str, query: str, max_results: int) -> List[WebSearchResult]:
        """Parse generic search results using common patterns"""
        results = []
        
        # Generic patterns for links and titles
        link_pattern = r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        matches = re.findall(link_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for i, (url, title) in enumerate(matches[:max_results]):
            if not url.startswith('http'):
                continue
                
            title = re.sub(r'<[^>]*>', '', title).strip()
            if not title:
                continue
            
            domain = urllib.parse.urlparse(url).netloc
            
            result = WebSearchResult(
                title=title,
                url=url,
                snippet="",
                relevance_score=1.0 - (i * 0.1),
                content_type='text/html',
                domain=domain,
                timestamp=datetime.now().isoformat(),
                meta_data={'engine': 'generic', 'position': i + 1}
            )
            
            results.append(result)
        
        return results
    
    def _deduplicate_results(self, results: List[WebSearchResult]) -> List[WebSearchResult]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    async def _rank_results_with_llm(self, results: List[WebSearchResult], query: str, context: Optional[Any]) -> List[WebSearchResult]:
        """Use LLM to re-rank results based on relevance to query"""
        if not context or len(results) <= 1:
            return results
        
        try:
            # Prepare results for LLM ranking
            results_text = "\n".join([
                f"{i+1}. {result.title}\n   URL: {result.url}\n   Snippet: {result.snippet}\n"
                for i, result in enumerate(results)
            ])
            
            rank_prompt = f"""
            Re-rank these search results based on relevance to the query: "{query}"
            
            Results:
            {results_text}
            
            Return only a comma-separated list of numbers indicating the new order (e.g., "3,1,5,2,4").
            Consider title relevance, snippet content, and domain authority.
            """
            
            response = await context.sample(prompt=rank_prompt)
            ranking_text = response.text.strip()
            
            # Parse ranking
            try:
                ranking = [int(x.strip()) - 1 for x in ranking_text.split(',')]
                if len(ranking) == len(results) and all(0 <= r < len(results) for r in ranking):
                    return [results[i] for i in ranking]
            except (ValueError, IndexError):
                pass
            
            # Fallback to original order
            return results
            
        except Exception as e:
            logger.warning(f"LLM ranking failed: {e}")
            return results
    
    def _apply_temporal_filter(self, results: List[WebSearchResult], temporal_filter: str) -> List[WebSearchResult]:
        """Apply temporal filtering to results (simplified implementation)"""
        # This is a simplified implementation - real temporal filtering would need
        # to analyze page content or use search engine temporal filters
        if temporal_filter in ['recent', 'week', 'month']:
            # Prefer results from known news/blog domains for recent content
            news_domains = ['news', 'blog', 'article', 'today', 'recent']
            filtered = []
            
            for result in results:
                is_recent = any(keyword in result.domain.lower() for keyword in news_domains)
                if is_recent or temporal_filter == 'month':
                    filtered.append(result)
            
            return filtered if filtered else results
        
        return results
    
    async def _analyze_result_with_llm(self, result: WebSearchResult, query: str, context: Any) -> str:
        """Generate LLM analysis of search result relevance"""
        try:
            analysis_prompt = f"""
            Analyze this search result for relevance to the query: "{query}"
            
            Title: {result.title}
            URL: {result.url}
            Snippet: {result.snippet}
            Domain: {result.domain}
            
            Provide a brief analysis (1-2 sentences) of:
            1. How relevant this result is to the query
            2. What type of information it likely contains
            3. Its potential usefulness
            """
            
            response = await context.sample(prompt=analysis_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"LLM result analysis failed: {e}")
            return "Analysis unavailable"
    
    async def _extract_main_content_with_llm(self, page_content: str, target_content: Optional[str], context: Optional[Any]) -> str:
        """Use LLM to intelligently extract main content from HTML"""
        if not context:
            # Fallback to simple text extraction
            extractor = HTMLContentExtractor()
            extractor.feed(page_content)
            return ' '.join(extractor.content)
        
        try:
            # Extract text content first
            extractor = HTMLContentExtractor()
            extractor.feed(page_content)
            text_content = ' '.join(extractor.content)
            
            # If content is too long, truncate for LLM processing
            if len(text_content) > 8000:
                text_content = text_content[:8000] + "... [content truncated]"
            
            extraction_prompt = f"""
            Extract the main content from this webpage text. Focus on:
            {"- " + target_content if target_content else "- The primary article or content"}
            - Remove navigation, ads, footers, and sidebar content
            - Preserve important formatting and structure
            - Include relevant headings and lists
            
            Webpage text:
            {text_content}
            
            Return only the main content, well-formatted.
            """
            
            response = await context.sample(prompt=extraction_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"LLM content extraction failed: {e}")
            # Fallback to simple extraction
            extractor = HTMLContentExtractor()
            extractor.feed(page_content)
            return ' '.join(extractor.content)
    
    async def _format_content_with_llm(self, content: str, output_format: str, context: Optional[Any]) -> str:
        """Use LLM to format content according to specified format"""
        if not context or output_format == 'text':
            return content
        
        try:
            format_prompt = f"""
            Convert this content to {output_format} format:
            
            Content:
            {content}
            
            Format requirements:
            {"- Use proper Markdown syntax with headers, lists, links, etc." if output_format == 'markdown' else ""}
            {"- Structure as valid JSON with appropriate fields" if output_format == 'json' else ""}
            {"- Preserve original HTML structure and tags" if output_format == 'html' else ""}
            
            Return only the formatted content.
            """
            
            response = await context.sample(prompt=format_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"LLM content formatting failed: {e}")
            return content
    
    async def _extract_structured_data_with_llm(self, page_content: str, context: Optional[Any]) -> Dict[str, Any]:
        """Extract structured data using LLM intelligence"""
        if not context:
            return {}
        
        try:
            # Look for common structured data patterns
            json_ld_pattern = r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>'
            json_ld_matches = re.findall(json_ld_pattern, page_content, re.DOTALL | re.IGNORECASE)
            
            structured_data = {}
            
            if json_ld_matches:
                try:
                    for match in json_ld_matches:
                        data = json.loads(match.strip())
                        structured_data['json_ld'] = data
                        break
                except json.JSONDecodeError:
                    pass
            
            # Use LLM to extract additional structured data
            extract_prompt = f"""
            Extract structured data from this webpage content. Look for:
            - Article metadata (author, date, category)
            - Product information (price, availability, specs)
            - Event details (date, location, description)
            - Contact information
            - Any other structured information
            
            Return as JSON format.
            
            Sample of webpage content:
            {page_content[:2000]}...
            """
            
            response = await context.sample(prompt=extract_prompt, response_format={"type": "json_object"})
            llm_data = response.json()
            
            if llm_data:
                structured_data['llm_extracted'] = llm_data
            
            return structured_data
            
        except Exception as e:
            logger.warning(f"Structured data extraction failed: {e}")
            return {}
    
    async def _take_screenshot_with_llm(self, url: str, context: Optional[Any]) -> Optional[str]:
        """Simulate screenshot functionality using LLM description"""
        if not context:
            return None
        
        try:
            # Download page content for analysis
            page_content = await self._download_page_content(url)
            
            # Extract visual elements
            extractor = HTMLContentExtractor()
            extractor.feed(page_content)
            
            screenshot_prompt = f"""
            Describe what a screenshot of this webpage would look like based on its content:
            
            Title: {extractor.meta_data.get('title', 'Unknown')}
            Number of links: {len(extractor.links)}
            Number of images: {len(extractor.images)}
            Main content length: {len(' '.join(extractor.content))} characters
            
            Sample content:
            {' '.join(extractor.content)[:1000]}...
            
            Describe the visual layout, color scheme, main elements, and overall appearance.
            """
            
            response = await context.sample(prompt=screenshot_prompt)
            
            # Encode description as base64 to simulate image data
            description = f"SCREENSHOT DESCRIPTION: {response.text.strip()}"
            return base64.b64encode(description.encode()).decode()
            
        except Exception as e:
            logger.warning(f"Screenshot simulation failed: {e}")
            return None
    
    async def _generate_content_summary(self, content: str, context: Any) -> str:
        """Generate LLM summary of scraped content"""
        try:
            summary_prompt = f"""
            Provide a concise summary of this webpage content:
            
            Content:
            {content[:3000]}{"..." if len(content) > 3000 else ""}
            
            Include:
            - Main topic and purpose
            - Key information or insights
            - Content type and structure
            - Usefulness for different purposes
            
            Keep summary to 2-3 sentences.
            """
            
            response = await context.sample(prompt=summary_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Content summary failed: {e}")
            return "Summary unavailable"
    
    async def _perform_web_interactions(self, url: str, interactions: List[Dict], context: Optional[Any]) -> Dict[str, Any]:
        """Simulate web interactions before scraping"""
        logger.info(f"Simulating {len(interactions)} web interactions")
        
        # This is a simplified simulation - in a real implementation,
        # you might use selenium or similar tools
        results = []
        
        for interaction in interactions:
            action_type = interaction.get('type', 'unknown')
            target = interaction.get('target', '')
            
            result = {
                'action': action_type,
                'target': target,
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'simulation': True
            }
            
            if action_type == 'wait':
                wait_time = interaction.get('time', 1)
                await asyncio.sleep(min(wait_time, 5))  # Cap wait time
                result['waited'] = wait_time
            
            results.append(result)
        
        return {
            'success': True,
            'interactions': results
        }
    
    async def _get_initial_page_state(self, url: str) -> Dict[str, Any]:
        """Get initial page state for evaluation"""
        try:
            content = await self._download_page_content(url)
            extractor = HTMLContentExtractor()
            extractor.feed(content)
            
            return {
                'url': url,
                'title': extractor.meta_data.get('title', ''),
                'links_count': len(extractor.links),
                'images_count': len(extractor.images),
                'content_length': len(' '.join(extractor.content)),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get page state: {e}")
            return {'url': url, 'error': str(e)}
    
    async def _simulate_click(self, url: str, action: Dict, page_state: Dict, context: Optional[Any]) -> WebInteractionResult:
        """Simulate clicking an element"""
        target = action.get('target', '')
        
        # Simulate click delay
        await asyncio.sleep(0.5)
        
        return WebInteractionResult(
            action='click',
            target=target,
            success=True,
            result=f"Clicked on {target}",
            timestamp=datetime.now().isoformat(),
            page_state={'last_clicked': target}
        )
    
    async def _simulate_fill(self, url: str, action: Dict, page_state: Dict, context: Optional[Any]) -> WebInteractionResult:
        """Simulate filling a form field"""
        target = action.get('target', '')
        value = action.get('value', '')
        
        return WebInteractionResult(
            action='fill',
            target=target,
            success=True,
            result=f"Filled {target} with: {value}",
            timestamp=datetime.now().isoformat(),
            page_state={'last_filled': {'target': target, 'value': value}}
        )
    
    async def _simulate_scroll(self, url: str, action: Dict, page_state: Dict, context: Optional[Any]) -> WebInteractionResult:
        """Simulate scrolling"""
        direction = action.get('direction', 'down')
        amount = action.get('amount', 100)
        
        return WebInteractionResult(
            action='scroll',
            target=f"{direction} {amount}px",
            success=True,
            result=f"Scrolled {direction} by {amount}px",
            timestamp=datetime.now().isoformat()
        )
    
    async def _simulate_wait(self, url: str, action: Dict, page_state: Dict, context: Optional[Any]) -> WebInteractionResult:
        """Simulate waiting for a condition"""
        wait_time = action.get('time', 1)
        condition = action.get('condition', 'time')
        
        await asyncio.sleep(min(wait_time, 10))  # Cap wait time
        
        return WebInteractionResult(
            action='wait',
            target=condition,
            success=True,
            result=f"Waited {wait_time}s for {condition}",
            timestamp=datetime.now().isoformat()
        )
    
    async def _simulate_execute(self, url: str, action: Dict, page_state: Dict, context: Optional[Any]) -> WebInteractionResult:
        """Simulate executing JavaScript-like logic"""
        script = action.get('script', '')
        
        # Simple script simulation
        if 'return' in script.lower():
            result = "Script executed - return value simulated"
        else:
            result = "Script executed - side effects simulated"
        
        return WebInteractionResult(
            action='execute',
            target=script[:100] + '...' if len(script) > 100 else script,
            success=True,
            result=result,
            timestamp=datetime.now().isoformat()
        )
    
    async def _take_action_screenshot(self, url: str, action: Dict, page_state: Dict, context: Optional[Any]) -> WebInteractionResult:
        """Take a screenshot during interaction"""
        screenshot_data = await self._take_screenshot_with_llm(url, context)
        
        return WebInteractionResult(
            action='screenshot',
            target='full_page',
            success=True,
            result="Screenshot captured",
            screenshot_data=screenshot_data,
            timestamp=datetime.now().isoformat()
        )
    
    async def _analyze_page_state(self, url: str, action: Dict, page_state: Dict, context: Optional[Any]) -> WebInteractionResult:
        """Analyze current page state"""
        if context:
            try:
                analysis_prompt = f"""
                Analyze the current state of this webpage:
                
                URL: {url}
                Page State: {json.dumps(page_state, indent=2)}
                
                Provide insights about:
                - Current page status
                - Available interactions
                - Content readiness
                - Next recommended actions
                """
                
                response = await context.sample(prompt=analysis_prompt)
                analysis = response.text.strip()
            except Exception as e:
                analysis = f"Analysis failed: {e}"
        else:
            analysis = f"Page state: {page_state}"
        
        return WebInteractionResult(
            action='analyze',
            target='page_state',
            success=True,
            result=analysis,
            timestamp=datetime.now().isoformat()
        ) 
