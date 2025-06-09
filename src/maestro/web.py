# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries
"""
This module provides production-quality, asynchronous tools for web browsing and searching.
It uses playwright for robust browser automation and duckduckgo_search for searching.
"""

import logging
from typing import List, Dict
from playwright.async_api import async_playwright, Browser as PlaywrightBrowser, Page
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class Browser:
    """
    An asynchronous wrapper around Playwright for web browsing and scraping.
    """
    def __init__(self, browser: PlaywrightBrowser = None):
        self._playwright = None
        self._browser = browser
        self._ext_browser = browser is not None

    async def __aenter__(self):
        if self._ext_browser:
            return self
        
        logger.info("Launching new browser instance...")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._ext_browser:
            return
            
        logger.info("Closing browser instance...")
        await self._browser.close()
        await self._playwright.stop()

    async def scrape(self, url: str) -> Dict[str, str]:
        """
        Scrapes the given URL and returns its title and text content.
        
        Args:
            url: The URL to scrape.
            
        Returns:
            A dictionary with 'url', 'title', and 'text' content.
        """
        page = await self._browser.new_page()
        try:
            logger.info(f"Navigating to {url}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            title = await page.title()
            # A simple way to get the main text content.
            # For more complex pages, libraries like Readability.js could be used.
            text_content = await page.evaluate("document.body.innerText")
            
            logger.info(f"Scraped '{title}' successfully.")
            return {"url": url, "title": title, "text": text_content}
        finally:
            await page.close()

class SearchEngine:
    """
    An asynchronous wrapper for performing web searches.
    Currently uses DuckDuckGo.
    """
    def __init__(self, engine: str = "duckduckgo"):
        if engine != "duckduckgo":
            raise NotImplementedError(f"Search engine '{engine}' is not supported yet.")
        self.engine = engine

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Performs a search and returns the results.
        
        Args:
            query: The search query.
            num_results: The number of results to return.

        Returns:
            A list of result dictionaries, each with 'title', 'href', and 'body'.
            'href' is the link and 'body' is the snippet.
        """
        logger.info(f"Searching for '{query}' with {self.engine}...")
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
        
        # Standardize the output format
        return [
            {"title": r["title"], "link": r["href"], "snippet": r["body"]}
            for r in results
        ] 