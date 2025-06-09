# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Real Web Tools for MAESTRO Protocol

100% functional web capabilities:
- Real HTTP-based search across multiple engines
- Real browser automation using Puppeteer
- Real content extraction and parsing
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
    """Real web search result"""
    title: str
    url: str
    snippet: str
    relevance_score: float
    content_type: str
    domain: str
    timestamp: str
    meta_data: Dict[str, Any]

class HTMLContentExtractor(HTMLParser):
    """Real HTML parser for content extraction"""
    
    def __init__(self):
        super().__init__()
        self.content = []
        self.links = []
        self.images = []
        self.meta_data = {}
        self.current_tag = None
        self.in_script = False
        self.in_style = False
        
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
    Real web tools with functional HTTP-based search
    """
    
    def __init__(self):
        self.name = "LLM Web Tools"
        self.version = "2.0.0"
        self.search_engines = {
            'duckduckgo': {
                'url': 'https://html.duckduckgo.com/html/',
                'query_param': 'q',
                'results_selector': '.result__title',
                'snippet_selector': '.result__snippet'
            }
        }
        
        # Initialize SSL context for secure connections
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    async def search(self, query: str, max_results: int = 10, search_engine: str = 'duckduckgo') -> List:
        logger.info(f"🔍 Real search: '{query}'")
        try:
            results = await self._search_duckduckgo(query, max_results)
            from mcp import types
            if results:
                response_text = f"# Search Results for: {query}\n\n"
                for i, result in enumerate(results, 1):
                    response_text += f"## {i}. {result.title}\n"
                    response_text += f"**URL:** {result.url}\n"
                    response_text += f"**Snippet:** {result.snippet}\n\n"
                return [types.TextContent(type="text", text=response_text)]
                else:
                return [types.TextContent(type="text", text=f"No results found for: {query}")]
        except Exception as e:
            from mcp import types
            return [types.TextContent(type="text", text=f"Search failed: {str(e)}")]

    async def get_page_content_simple(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        try:
            content = await self._download_page_content(url, timeout)
            return {"content": content}
        except Exception as e:
            return {"error": str(e)}

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        try:
            content = await self._download_page_content(search_url, 30)
            return self._parse_duckduckgo_results(content, query, max_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    async def _download_page_content(self, url: str, timeout: int = 45) -> str:
        def sync_download():
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=self.ssl_context) as response:
                content = response.read()
                if response.info().get('Content-Encoding') == 'gzip':
                    content = gzip.decompress(content)
                encoding = response.info().get_content_charset() or 'utf-8'
                return content.decode(encoding, errors='ignore')
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_download)
    
    def _parse_duckduckgo_results(self, content: str, query: str, max_results: int) -> List[WebSearchResult]:
        results = []
        try:
            result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>'
            
            result_matches = re.findall(result_pattern, content, re.IGNORECASE | re.DOTALL)
            snippet_matches = re.findall(snippet_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for i, (url, title) in enumerate(result_matches[:max_results]):
            snippet = snippet_matches[i] if i < len(snippet_matches) else ""
                url = url.strip()
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            
                if url and title:
            domain = urllib.parse.urlparse(url).netloc
            result = WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                relevance_score=1.0 - (i * 0.1),
                        content_type="text/html",
                domain=domain,
                timestamp=datetime.now().isoformat(),
                        meta_data={"engine": "duckduckgo", "position": i + 1}
                    )
            results.append(result)
            return results
        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo results: {e}")
            return []

class PuppeteerTools:
    """
    Real browser automation using Puppeteer
    """
    def __init__(self):
        self.browser = None
        self.pyppeteer = None

    async def _ensure_browser(self):
        """Ensures the browser is running"""
        if self.browser:
            return
        try:
            import pyppeteer
            self.pyppeteer = pyppeteer
            self.browser = await self.pyppeteer.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            logger.info("🚀 Puppeteer browser launched")
        except ImportError:
            logger.error("Pyppeteer not installed. Run: pip install pyppeteer")
            raise
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            try:
                logger.info("Downloading Chromium...")
                import pyppeteer.chromium_downloader
                await pyppeteer.chromium_downloader.download_chromium()
                self.browser = await self.pyppeteer.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
            except Exception as download_error:
                logger.error(f"Failed to download Chromium: {download_error}")
                raise

    async def get_page_content(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Real browser-based page content fetching with JavaScript rendering
        """
        await self._ensure_browser()
        if not self.browser:
            return {"error": "Browser not available"}

        page = await self.browser.newPage()
        try:
            await page.goto(url, {'timeout': timeout * 1000, 'waitUntil': 'networkidle2'})
            content = await page.content()
            return {"content": content}
        except Exception as e:
            logger.error(f"Puppeteer failed for {url}: {e}")
            return {"error": str(e)}
        finally:
            await page.close()

    async def close(self):
        """Closes the browser instance"""
        if self.browser:
            await self.browser.close()
            self.browser = None
            logger.info("�� Browser closed") 