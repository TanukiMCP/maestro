#!/usr/bin/env python3
# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Debug script for testing search functionality
"""

import urllib.request
import urllib.parse
import ssl
import re
import asyncio
import gzip
import io
from typing import List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float
    content_type: str
    domain: str
    timestamp: str
    meta_data: dict

async def test_duckduckgo_search(query: str = "artificial intelligence"):
    """Test DuckDuckGo search parsing"""
    
    # Create SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    print(f"Searching: {search_url}")
    
    try:
        request = urllib.request.Request(search_url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
            content = response.read()
            
            # Handle gzip compression
            content_encoding = response.headers.get('Content-Encoding', '')
            if 'gzip' in content_encoding.lower():
                print("Response is gzip compressed")
                content = gzip.decompress(content)
            
            encoding = response.headers.get_content_charset() or 'utf-8'
            html_content = content.decode(encoding, errors='ignore')
            
            print(f"Response length: {len(html_content)}")
            print(f"Content encoding: {content_encoding}")
            
            # Save the HTML for inspection
            with open('debug_duckduckgo_response.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            print("Saved response to debug_duckduckgo_response.html")
            
            # Show first 1000 characters
            print(f"\nFirst 1000 characters of response:")
            print(html_content[:1000])
            
            # Test different patterns
            print("\n=== Testing Result Patterns ===")
            
            # Original pattern from code
            result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            result_matches = re.findall(result_pattern, html_content, re.DOTALL | re.IGNORECASE)
            print(f"Original pattern matches: {len(result_matches)}")
            
            # Alternative patterns
            patterns_to_test = [
                r'<a[^>]*href="([^"]*)"[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</a>',
                r'<h3[^>]*class="[^"]*result[^"]*"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                r'<div[^>]*class="[^"]*result[^"]*"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                r'<a[^>]*href="(http[^"]*)"[^>]*>(.*?)</a>',
                r'<a[^>]*href="([^"]*)"[^>]*title="[^"]*"[^>]*>(.*?)</a>'
            ]
            
            for i, pattern in enumerate(patterns_to_test):
                matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                print(f"Pattern {i+1} matches: {len(matches)}")
                if matches and len(matches) > 0:
                    print(f"  First match: {matches[0][:2] if len(matches[0]) >= 2 else matches[0]}")
            
            # Look for common classes in the HTML
            print("\n=== Looking for result-related classes ===")
            class_patterns = [
                r'class="([^"]*result[^"]*)"',
                r'class="([^"]*link[^"]*)"',
                r'class="([^"]*title[^"]*)"'
            ]
            
            for pattern in class_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                unique_classes = list(set(matches))
                print(f"Classes matching '{pattern}': {unique_classes[:10]}")  # First 10
            
            # Check if there are any links at all
            all_links = re.findall(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html_content, re.DOTALL)
            print(f"\nTotal links found: {len(all_links)}")
            
            # Show first few links
            if all_links:
                print("First 5 links:")
                for i, (url, text) in enumerate(all_links[:5]):
                    clean_text = re.sub(r'<[^>]*>', '', text).strip()
                    print(f"  {i+1}. {url[:50]}... -> {clean_text[:50]}...")
            
            # Look for specific DuckDuckGo patterns
            print("\n=== DuckDuckGo Specific Patterns ===")
            ddg_patterns = [
                r'<div[^>]*class="[^"]*web-result[^"]*"[^>]*>',
                r'<h2[^>]*class="[^"]*result[^"]*"[^>]*>',
                r'<a[^>]*rel="[^"]*noopener[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                r'data-testid="result"',
                r'<span[^>]*class="[^"]*result[^"]*snippet[^"]*"[^>]*>'
            ]
            
            for i, pattern in enumerate(ddg_patterns):
                matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                print(f"DDG Pattern {i+1} matches: {len(matches)}")
            
            return html_content
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_duckduckgo_search()) 
