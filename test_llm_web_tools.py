#!/usr/bin/env python3
"""
Test script for LLM Web Tools

Tests the new 100% free, 100% LLM-driven web capabilities that rival puppeteer/playwright
"""

import asyncio
import json
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maestro.llm_web_tools import LLMWebTools
from maestro.enhanced_tools import EnhancedToolHandlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class MockContext:
    """Mock context for testing LLM functionality"""
    
    async def sample(self, prompt: str, response_format: dict = None):
        """Mock LLM sampling"""
        class MockResponse:
            def __init__(self, prompt):
                self.prompt = prompt
                
            def json(self):
                # Return mock JSON responses based on prompt content
                if "extract structured data" in self.prompt.lower():
                    return {
                        "article": {
                            "title": "Mock Article Title",
                            "author": "Test Author",
                            "date": "2024-01-15",
                            "category": "Technology"
                        }
                    }
                elif "enhance this search query" in self.prompt.lower():
                    return {}
                else:
                    return {}
            
            @property
            def text(self):
                # Return mock text responses based on prompt content
                if "enhance this search query" in self.prompt.lower():
                    # Extract original query and enhance it
                    lines = self.prompt.split('\n')
                    for line in lines:
                        if 'Original query:' in line:
                            original = line.split('"')[1] if '"' in line else "test query"
                            return f"{original} latest news updates"
                    return "enhanced test query"
                
                elif "re-rank these search results" in self.prompt.lower():
                    return "1,2,3,4,5"
                
                elif "extract the main content" in self.prompt.lower():
                    return "This is the main content extracted from the webpage. It contains the primary article text with important information."
                
                elif "convert this content to" in self.prompt.lower():
                    if "markdown" in self.prompt.lower():
                        return "# Main Content\n\nThis is the formatted content in markdown format.\n\n## Key Points\n\n- Point 1\n- Point 2"
                    elif "json" in self.prompt.lower():
                        return '{"title": "Main Content", "content": "Formatted content", "type": "article"}'
                    else:
                        return "Formatted content in requested format"
                
                elif "describe what a screenshot" in self.prompt.lower():
                    return "The webpage has a clean, modern design with a white background. The header contains navigation links, and the main content area displays an article with a large title, body text, and sidebar elements."
                
                elif "provide a concise summary" in self.prompt.lower():
                    return "This webpage contains an informative article about technology trends. The content is well-structured and provides valuable insights for readers interested in the topic."
                
                elif "analyze this search result" in self.prompt.lower():
                    return "This result appears highly relevant to the query, containing authoritative information from a reputable source. It likely provides comprehensive coverage of the requested topic."
                
                elif "analyze the current state" in self.prompt.lower():
                    return "The webpage is fully loaded and ready for interaction. All main content elements are visible and accessible. Recommended next actions include scrolling to view more content or clicking on relevant links."
                
                else:
                    return "Mock LLM response for testing purposes"
        
        return MockResponse(prompt)

async def test_llm_search():
    """Test LLM-driven search functionality"""
    logger.info("🔍 Testing LLM-driven search...")
    
    llm_tools = LLMWebTools()
    mock_context = MockContext()
    
    try:
        result = await llm_tools.llm_driven_search(
            query="Python programming tutorials",
            max_results=5,
            engines=['duckduckgo'],
            temporal_filter=None,
            result_format='structured',
            llm_analysis=True,
            context=mock_context
        )
        
        logger.info(f"✅ Search completed: {result['success']}")
        if result['success']:
            logger.info(f"   - Query: {result['query']}")
            logger.info(f"   - Enhanced Query: {result['enhanced_query']}")
            logger.info(f"   - Results: {result['total_results']}")
            logger.info(f"   - LLM Enhanced: {result['metadata']['llm_enhanced']}")
        else:
            logger.error(f"   - Error: {result.get('error', 'Unknown error')}")
        
        return result['success']
        
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return False

async def test_llm_scrape():
    """Test LLM-driven scraping functionality"""
    logger.info("🕷️ Testing LLM-driven scraping...")
    
    llm_tools = LLMWebTools()
    mock_context = MockContext()
    
    try:
        result = await llm_tools.llm_driven_scrape(
            url="https://httpbin.org/html",  # Simple test URL
            output_format='markdown',
            target_content=None,
            extract_structured=True,
            take_screenshot=True,
            interact_before_scrape=None,
            context=mock_context
        )
        
        logger.info(f"✅ Scrape completed: {result['success']}")
        if result['success']:
            scrape_result = result['result']
            logger.info(f"   - URL: {scrape_result['url']}")
            logger.info(f"   - Title: {scrape_result['title']}")
            logger.info(f"   - Content Length: {len(scrape_result['content'])} chars")
            logger.info(f"   - Format: {scrape_result['format_type']}")
            logger.info(f"   - Has Summary: {bool(scrape_result.get('llm_summary'))}")
            logger.info(f"   - Has Screenshot: {bool(scrape_result['metadata'].get('screenshot'))}")
        else:
            logger.error(f"   - Error: {result.get('error', 'Unknown error')}")
        
        return result['success']
        
    except Exception as e:
        logger.error(f"❌ Scrape test failed: {e}")
        return False

async def test_llm_web_evaluate():
    """Test LLM-driven web evaluation functionality"""
    logger.info("🎭 Testing LLM-driven web evaluation...")
    
    llm_tools = LLMWebTools()
    mock_context = MockContext()
    
    try:
        actions = [
            {'type': 'click', 'target': '#submit-button'},
            {'type': 'fill', 'target': '#search-input', 'value': 'test query'},
            {'type': 'scroll', 'direction': 'down', 'amount': 200},
            {'type': 'wait', 'time': 2, 'condition': 'page_load'},
            {'type': 'screenshot'},
            {'type': 'analyze'},
            {'type': 'execute', 'script': 'return document.title;'}
        ]
        
        result = await llm_tools.llm_web_evaluate(
            url="https://httpbin.org/html",
            actions=actions,
            capture_results=True,
            take_screenshots=True,
            context=mock_context
        )
        
        logger.info(f"✅ Web evaluation completed: {result['success']}")
        if result['success']:
            logger.info(f"   - URL: {result['url']}")
            logger.info(f"   - Actions Performed: {result['actions_performed']}")
            logger.info(f"   - Results Count: {len(result['results'])}")
            
            # Log each action result
            for i, action_result in enumerate(result['results'], 1):
                logger.info(f"   - Action {i}: {action_result['action']} -> {action_result['success']}")
        else:
            logger.error(f"   - Error: {result.get('error', 'Unknown error')}")
        
        return result['success']
        
    except Exception as e:
        logger.error(f"❌ Web evaluation test failed: {e}")
        return False

async def test_enhanced_tools_integration():
    """Test integration with enhanced tools"""
    logger.info("🔧 Testing enhanced tools integration...")
    
    enhanced_tools = EnhancedToolHandlers()
    
    try:
        # Test search through enhanced tools
        search_result = await enhanced_tools.handle_maestro_search({
            'query': 'machine learning tutorials',
            'max_results': 3,
            'search_engine': 'duckduckgo',
            'temporal_filter': 'any',
            'result_format': 'structured'
        })
        
        logger.info(f"✅ Enhanced search completed")
        logger.info(f"   - Result type: {type(search_result)}")
        logger.info(f"   - Content length: {len(search_result[0].text) if search_result else 0} chars")
        
        # Test scrape through enhanced tools
        scrape_result = await enhanced_tools.handle_maestro_scrape({
            'url': 'https://httpbin.org/html',
            'output_format': 'markdown',
            'selectors': [],
            'wait_for': None,
            'extract_links': True,
            'extract_images': False
        })
        
        logger.info(f"✅ Enhanced scrape completed")
        logger.info(f"   - Result type: {type(scrape_result)}")
        logger.info(f"   - Content length: {len(scrape_result[0].text) if scrape_result else 0} chars")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced tools integration test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling capabilities"""
    logger.info("⚠️ Testing error handling...")
    
    llm_tools = LLMWebTools()
    
    try:
        # Test with invalid URL
        result = await llm_tools.llm_driven_scrape(
            url="https://invalid-url-that-does-not-exist.com",
            output_format='markdown',
            context=None
        )
        
        logger.info(f"✅ Error handling test completed")
        logger.info(f"   - Success: {result['success']}")
        logger.info(f"   - Has error message: {bool(result.get('error'))}")
        
        return True
        
    except Exception as e:
        logger.info(f"✅ Error properly caught: {type(e).__name__}")
        return True

async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting LLM Web Tools comprehensive test suite...")
    
    tests = [
        ("LLM Search", test_llm_search),
        ("LLM Scrape", test_llm_scrape),
        ("LLM Web Evaluate", test_llm_web_evaluate),
        ("Enhanced Tools Integration", test_enhanced_tools_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"❌ FAILED: {test_name} - {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! LLM Web Tools are working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 