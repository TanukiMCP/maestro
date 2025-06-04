# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Web Verification Engine

Uses Playwright, Selenium, and other web testing tools to verify
web content, accessibility, and functionality.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Try to import web testing libraries with graceful fallbacks
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some web libraries not available: {e}")
    WEB_LIBRARIES_AVAILABLE = False


class WebVerificationEngine:
    """
    Provides web content verification and accessibility testing.
    """
    
    def __init__(self):
        logger.info("🌐 Web Verification Engine initialized")
    
    async def verify_web_content(
        self,
        content: str,
        requirements: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Verify web content for quality, accessibility, and best practices.
        
        Args:
            content: HTML content or URL to verify
            requirements: Specific verification requirements
            
        Returns:
            Comprehensive web verification analysis
        """
        if requirements is None:
            requirements = {}
        
        try:
            # Determine if content is HTML or URL
            if self._is_url(content):
                return await self._verify_url(content, requirements)
            else:
                return await self._verify_html_content(content, requirements)
                
        except Exception as e:
            logger.error(f"Web verification failed: {str(e)}")
            return {"error": f"Web verification error: {str(e)}"}
    
    def _is_url(self, content: str) -> bool:
        """Check if content is a URL."""
        try:
            result = urlparse(content.strip())
            return all([result.scheme, result.netloc])
        except:
            return False
    
    async def _verify_url(self, url: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a live URL."""
        
        if not WEB_LIBRARIES_AVAILABLE:
            return {"error": "Web verification requires requests and beautifulsoup4 libraries"}
        
        try:
            # Make request to URL
            response = requests.get(url, timeout=10)
            
            # Basic response analysis
            status_analysis = {
                "status_code": response.status_code,
                "success": 200 <= response.status_code < 300,
                "response_time": response.elapsed.total_seconds(),
                "content_type": response.headers.get('content-type', 'unknown')
            }
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            html_analysis = await self._analyze_html_structure(str(soup))
            accessibility_analysis = await self._check_accessibility(str(soup))
            performance_analysis = await self._analyze_performance(response)
            
            return {
                "url": url,
                "status_analysis": status_analysis,
                "html_analysis": html_analysis,
                "accessibility_analysis": accessibility_analysis,
                "performance_analysis": performance_analysis,
                "overall_score": self._calculate_web_score(
                    status_analysis, html_analysis, accessibility_analysis
                ),
                "status": "Web verification complete"
            }
            
        except requests.RequestException as e:
            return {"error": f"Failed to fetch URL: {str(e)}"}
        except Exception as e:
            return {"error": f"URL verification error: {str(e)}"}
    
    async def _verify_html_content(self, html_content: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Verify HTML content."""
        
        try:
            html_analysis = await self._analyze_html_structure(html_content)
            accessibility_analysis = await self._check_accessibility(html_content)
            seo_analysis = await self._analyze_seo(html_content)
            validation_analysis = await self._validate_html(html_content)
            
            return {
                "content_type": "HTML",
                "html_analysis": html_analysis,
                "accessibility_analysis": accessibility_analysis,
                "seo_analysis": seo_analysis,
                "validation_analysis": validation_analysis,
                "overall_score": self._calculate_html_score(
                    html_analysis, accessibility_analysis, seo_analysis
                ),
                "status": "HTML verification complete"
            }
            
        except Exception as e:
            return {"error": f"HTML verification error: {str(e)}"}
    
    async def _analyze_html_structure(self, html_content: str) -> Dict[str, Any]:
        """Analyze HTML structure and quality."""
        
        if not WEB_LIBRARIES_AVAILABLE:
            return {"error": "HTML analysis requires beautifulsoup4 library"}
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            structure_analysis = {
                "has_doctype": html_content.strip().lower().startswith('<!doctype'),
                "has_html_tag": soup.find('html') is not None,
                "has_head": soup.find('head') is not None,
                "has_body": soup.find('body') is not None,
                "has_title": soup.find('title') is not None,
                "title_text": soup.find('title').get_text() if soup.find('title') else None,
                "meta_tags": len(soup.find_all('meta')),
                "heading_structure": self._analyze_headings(soup),
                "link_analysis": self._analyze_links(soup),
                "image_analysis": self._analyze_images(soup)
            }
            
            # Calculate structure score
            score = 0
            if structure_analysis["has_doctype"]: score += 10
            if structure_analysis["has_html_tag"]: score += 10
            if structure_analysis["has_head"]: score += 15
            if structure_analysis["has_body"]: score += 15
            if structure_analysis["has_title"]: score += 20
            if structure_analysis["meta_tags"] > 0: score += 10
            if structure_analysis["heading_structure"]["has_h1"]: score += 20
            
            structure_analysis["score"] = score
            structure_analysis["status"] = "✅ Good structure" if score >= 80 else "⚠️ Structure issues"
            
            return structure_analysis
            
        except Exception as e:
            return {"error": f"HTML structure analysis error: {str(e)}"}
    
    def _analyze_headings(self, soup) -> Dict[str, Any]:
        """Analyze heading structure."""
        
        headings = {}
        for i in range(1, 7):
            headings[f"h{i}"] = len(soup.find_all(f'h{i}'))
        
        return {
            "headings": headings,
            "has_h1": headings["h1"] > 0,
            "multiple_h1": headings["h1"] > 1,
            "total_headings": sum(headings.values())
        }
    
    def _analyze_links(self, soup) -> Dict[str, Any]:
        """Analyze links in the HTML."""
        
        links = soup.find_all('a')
        
        return {
            "total_links": len(links),
            "external_links": len([link for link in links if link.get('href', '').startswith('http')]),
            "links_without_text": len([link for link in links if not link.get_text().strip()]),
            "links_without_href": len([link for link in links if not link.get('href')])
        }
    
    def _analyze_images(self, soup) -> Dict[str, Any]:
        """Analyze images in the HTML."""
        
        images = soup.find_all('img')
        
        return {
            "total_images": len(images),
            "images_without_alt": len([img for img in images if not img.get('alt')]),
            "images_without_src": len([img for img in images if not img.get('src')])
        }
    
    async def _check_accessibility(self, html_content: str) -> Dict[str, Any]:
        """Check accessibility compliance."""
        
        if not WEB_LIBRARIES_AVAILABLE:
            return {"error": "Accessibility check requires beautifulsoup4 library"}
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            accessibility_issues = []
            
            # Check for alt text on images
            images_without_alt = soup.find_all('img', alt=False)
            if images_without_alt:
                accessibility_issues.append(f"{len(images_without_alt)} images missing alt text")
            
            # Check for form labels
            inputs = soup.find_all('input')
            inputs_without_labels = []
            for input_tag in inputs:
                input_id = input_tag.get('id')
                if input_id:
                    label = soup.find('label', {'for': input_id})
                    if not label:
                        inputs_without_labels.append(input_tag)
            
            if inputs_without_labels:
                accessibility_issues.append(f"{len(inputs_without_labels)} form inputs missing labels")
            
            # Check for heading hierarchy
            headings = []
            for i in range(1, 7):
                headings.extend([(f'h{i}', tag) for tag in soup.find_all(f'h{i}')])
            
            if headings:
                heading_levels = [int(h[0][1]) for h in headings]
                if heading_levels and heading_levels[0] != 1:
                    accessibility_issues.append("Page should start with h1 heading")
            
            # Check for color contrast (basic check for inline styles)
            elements_with_style = soup.find_all(style=True)
            low_contrast_elements = []
            for element in elements_with_style:
                style = element.get('style', '')
                if 'color:' in style and 'background' in style:
                    # This is a very basic check - real contrast checking requires color parsing
                    if 'white' in style and 'yellow' in style:
                        low_contrast_elements.append(element)
            
            if low_contrast_elements:
                accessibility_issues.append(f"Potential low contrast issues found")
            
            # Calculate accessibility score
            score = 100 - len(accessibility_issues) * 15
            score = max(0, score)
            
            return {
                "issues": accessibility_issues,
                "score": score,
                "status": "✅ Accessible" if score >= 80 else "⚠️ Accessibility issues found",
                "recommendations": self._get_accessibility_recommendations(accessibility_issues)
            }
            
        except Exception as e:
            return {"error": f"Accessibility check error: {str(e)}"}
    
    def _get_accessibility_recommendations(self, issues: List[str]) -> List[str]:
        """Get accessibility recommendations based on issues."""
        
        recommendations = []
        
        if any("alt text" in issue for issue in issues):
            recommendations.append("Add descriptive alt text to all images")
        
        if any("labels" in issue for issue in issues):
            recommendations.append("Associate form inputs with descriptive labels")
        
        if any("heading" in issue for issue in issues):
            recommendations.append("Use proper heading hierarchy (h1, h2, h3, etc.)")
        
        if any("contrast" in issue for issue in issues):
            recommendations.append("Ensure sufficient color contrast for text")
        
        return recommendations
    
    async def _analyze_seo(self, html_content: str) -> Dict[str, Any]:
        """Analyze SEO factors."""
        
        if not WEB_LIBRARIES_AVAILABLE:
            return {"error": "SEO analysis requires beautifulsoup4 library"}
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            seo_factors = {
                "title": soup.find('title').get_text() if soup.find('title') else None,
                "meta_description": None,
                "meta_keywords": None,
                "h1_tags": len(soup.find_all('h1')),
                "internal_links": 0,
                "external_links": 0
            }
            
            # Check meta tags
            meta_description = soup.find('meta', attrs={'name': 'description'})
            if meta_description:
                seo_factors["meta_description"] = meta_description.get('content')
            
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                seo_factors["meta_keywords"] = meta_keywords.get('content')
            
            # Analyze links
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href.startswith('http'):
                    seo_factors["external_links"] += 1
                else:
                    seo_factors["internal_links"] += 1
            
            # Calculate SEO score
            score = 0
            if seo_factors["title"]: score += 25
            if seo_factors["meta_description"]: score += 25
            if seo_factors["h1_tags"] == 1: score += 20
            if seo_factors["internal_links"] > 0: score += 15
            if seo_factors["external_links"] > 0: score += 15
            
            seo_factors["score"] = score
            seo_factors["status"] = "✅ SEO optimized" if score >= 70 else "⚠️ SEO improvements needed"
            
            return seo_factors
            
        except Exception as e:
            return {"error": f"SEO analysis error: {str(e)}"}
    
    async def _validate_html(self, html_content: str) -> Dict[str, Any]:
        """Validate HTML syntax."""
        
        validation_issues = []
        
        # Basic HTML validation checks
        if not html_content.strip().lower().startswith('<!doctype'):
            validation_issues.append("Missing DOCTYPE declaration")
        
        # Check for unclosed tags (basic check)
        open_tags = re.findall(r'<(\w+)[^>]*>', html_content)
        close_tags = re.findall(r'</(\w+)>', html_content)
        
        # Self-closing tags that don't need closing
        self_closing = {'img', 'br', 'hr', 'input', 'meta', 'link'}
        
        for tag in open_tags:
            if tag.lower() not in self_closing and tag.lower() not in [t.lower() for t in close_tags]:
                validation_issues.append(f"Unclosed tag: <{tag}>")
        
        # Check for duplicate IDs
        ids = re.findall(r'id=["\']([^"\']+)["\']', html_content)
        duplicate_ids = [id_val for id_val in set(ids) if ids.count(id_val) > 1]
        if duplicate_ids:
            validation_issues.append(f"Duplicate IDs found: {duplicate_ids}")
        
        score = max(0, 100 - len(validation_issues) * 20)
        
        return {
            "issues": validation_issues,
            "score": score,
            "status": "✅ Valid HTML" if score >= 80 else "⚠️ Validation issues found"
        }
    
    async def _analyze_performance(self, response) -> Dict[str, Any]:
        """Analyze performance metrics from response."""
        
        try:
            content_size = len(response.content)
            response_time = response.elapsed.total_seconds()
            
            performance_score = 100
            
            # Penalize slow response times
            if response_time > 3:
                performance_score -= 30
            elif response_time > 1:
                performance_score -= 15
            
            # Penalize large content size
            if content_size > 1000000:  # 1MB
                performance_score -= 20
            elif content_size > 500000:  # 500KB
                performance_score -= 10
            
            return {
                "response_time": response_time,
                "content_size": content_size,
                "score": max(0, performance_score),
                "status": "✅ Good performance" if performance_score >= 70 else "⚠️ Performance issues"
            }
            
        except Exception as e:
            return {"error": f"Performance analysis error: {str(e)}"}
    
    def _calculate_web_score(
        self,
        status_analysis: Dict[str, Any],
        html_analysis: Dict[str, Any],
        accessibility_analysis: Dict[str, Any]
    ) -> int:
        """Calculate overall web quality score."""
        
        if not status_analysis.get("success", False):
            return 0
        
        html_score = html_analysis.get("score", 0)
        accessibility_score = accessibility_analysis.get("score", 0)
        
        # Weighted average
        overall_score = int(0.4 * html_score + 0.6 * accessibility_score)
        
        return max(0, min(100, overall_score))
    
    def _calculate_html_score(
        self,
        html_analysis: Dict[str, Any],
        accessibility_analysis: Dict[str, Any],
        seo_analysis: Dict[str, Any]
    ) -> int:
        """Calculate overall HTML quality score."""
        
        html_score = html_analysis.get("score", 0)
        accessibility_score = accessibility_analysis.get("score", 0)
        seo_score = seo_analysis.get("score", 0)
        
        # Weighted average
        overall_score = int(0.4 * html_score + 0.4 * accessibility_score + 0.2 * seo_score)
        
        return max(0, min(100, overall_score)) 
