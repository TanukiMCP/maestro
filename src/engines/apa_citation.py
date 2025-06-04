# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
APA Citation Engine

Provides comprehensive APA 7th edition citation formatting, validation,
and generation capabilities for academic writing and research.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try to import web scraping libraries for metadata extraction
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logger.warning("requests/beautifulsoup4 not available - install with: pip install requests beautifulsoup4")


@dataclass
class CitationSource:
    """Represents a source for citation."""
    source_type: str  # book, journal, website, etc.
    authors: List[str]
    title: str
    publication_year: str
    publisher: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    access_date: Optional[str] = None
    edition: Optional[str] = None
    city: Optional[str] = None
    isbn: Optional[str] = None


@dataclass
class CitationValidationResult:
    """Result of citation validation."""
    is_valid: bool
    issues: List[str]
    suggestions: List[str]
    corrected_citation: Optional[str] = None
    apa_compliance_score: float = 0.0


@dataclass
class BibliographyEntry:
    """Represents a bibliography entry."""
    citation: str
    source: CitationSource
    sort_key: str
    in_text_format: str


class APACitationEngine:
    """
    Comprehensive APA 7th edition citation engine.
    
    Provides:
    - Citation generation for multiple source types
    - Citation validation and correction
    - In-text citation formatting
    - Reference list generation
    - DOI and URL validation
    - Metadata extraction from URLs
    """
    
    def __init__(self):
        self.apa_rules = self._initialize_apa_rules()
        self.source_templates = self._initialize_source_templates()
        self.common_errors = self._initialize_common_errors()
        logger.info("📚 APA Citation Engine initialized")
    
    def _initialize_apa_rules(self) -> Dict[str, Any]:
        """Initialize APA 7th edition formatting rules."""
        return {
            "author_formatting": {
                "single_author": "{last}, {first_initial}.",
                "multiple_authors": "{authors[:-1]}, & {authors[-1]}",
                "max_displayed": 20,  # Display up to 20 authors
                "et_al_threshold": 21  # Use et al. if more than 20 authors
            },
            "title_formatting": {
                "book": "sentence_case_italic",
                "journal_article": "sentence_case_plain",
                "journal_name": "title_case_italic",
                "website": "sentence_case_plain"
            },
            "date_formatting": {
                "format": "({year})",
                "no_date": "(n.d.)",
                "in_press": "(in press)"
            },
            "doi_formatting": {
                "prefix": "https://doi.org/",
                "required_for": ["journal_articles", "books_with_doi"]
            },
            "url_formatting": {
                "retrieve_format": "Retrieved {date}, from {url}",
                "no_date_format": "{url}"
            }
        }
    
    def _initialize_source_templates(self) -> Dict[str, str]:
        """Initialize APA citation templates for different source types."""
        return {
            "book": "{authors} ({year}). {title}. {publisher}.",
            "book_with_editor": "{authors} ({year}). {title} ({editors}, Ed{s}.). {publisher}.",
            "journal_article": "{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. {doi}",
            "journal_article_no_doi": "{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}.",
            "website": "{authors} ({year}, {month} {day}). {title}. {site_name}. {retrieved_date_url}",
            "website_no_author": "{title}. ({year}, {month} {day}). {site_name}. {retrieved_date_url}",
            "conference_paper": "{authors} ({year}, {month}). {title}. In {conference_editors} (Ed{s}.), {conference_proceedings} (pp. {pages}). {publisher}.",
            "dissertation": "{author} ({year}). {title} [Doctoral dissertation, {institution}]. {database}. {url}",
            "government_report": "{agency}. ({year}). {title} ({report_number}). {publisher}.",
            "edited_book_chapter": "{authors} ({year}). {chapter_title}. In {editors} (Ed{s}.), {book_title} (pp. {pages}). {publisher}."
        }
    
    def _initialize_common_errors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common APA citation errors and corrections."""
        return {
            "author_errors": {
                "patterns": [
                    {"error": r"(\w+)\s+(\w+\.?)", "correction": r"\2, \1", "description": "Author name order incorrect"},
                    {"error": r"&amp;", "correction": "&", "description": "HTML entity in author names"},
                    {"error": r"\s+and\s+", "correction": " & ", "description": "Use & instead of 'and' in citations"}
                ]
            },
            "title_errors": {
                "patterns": [
                    {"error": r'"([^"]+)"', "correction": r"\1", "description": "Remove quotes around titles"},
                    {"error": r"([A-Z][a-z]+ [A-Z][a-z]+)", "correction": lambda m: m.group(1).capitalize(), "description": "Title case should be sentence case"}
                ]
            },
            "date_errors": {
                "patterns": [
                    {"error": r"\b(\d{4})\b(?!\))", "correction": r"(\1)", "description": "Year should be in parentheses"},
                    {"error": r"\((\d{4}),\s*(\w+)\)", "correction": r"(\1, \2)", "description": "Incorrect date formatting"}
                ]
            }
        }
    
    async def generate_citation(
        self,
        source: CitationSource,
        citation_style: str = "reference_list"
    ) -> str:
        """
        Generate APA citation for a source.
        
        Args:
            source: Source information
            citation_style: "reference_list" or "in_text"
            
        Returns:
            Formatted APA citation
        """
        logger.info(f"📖 Generating {citation_style} citation for {source.source_type}")
        
        if citation_style == "in_text":
            return self._generate_in_text_citation(source)
        else:
            return self._generate_reference_citation(source)
    
    def _generate_reference_citation(self, source: CitationSource) -> str:
        """Generate reference list citation."""
        template = self.source_templates.get(source.source_type, self.source_templates["book"])
        
        # Format authors
        authors_formatted = self._format_authors(source.authors)
        
        # Format title
        title_formatted = self._format_title(source.title, source.source_type)
        
        # Format year
        year_formatted = f"({source.publication_year})" if source.publication_year else "(n.d.)"
        
        # Build citation components
        components = {
            "authors": authors_formatted,
            "year": year_formatted,
            "title": title_formatted,
            "publisher": source.publisher or "",
            "journal": self._format_journal_name(source.journal) if source.journal else "",
            "volume": source.volume or "",
            "issue": source.issue or "",
            "pages": self._format_pages(source.pages) if source.pages else "",
            "doi": self._format_doi(source.doi) if source.doi else "",
            "url": source.url or "",
            "retrieved_date_url": self._format_retrieved_date_url(source.url, source.access_date) if source.url else ""
        }
        
        # Apply template
        citation = template.format(**components)
        
        # Clean up formatting
        citation = self._clean_citation_formatting(citation)
        
        return citation
    
    def _generate_in_text_citation(self, source: CitationSource) -> str:
        """Generate in-text citation."""
        if not source.authors:
            # No author - use title
            short_title = source.title.split(":")[0][:50] + ("..." if len(source.title) > 50 else "")
            return f'("{short_title}," {source.publication_year or "n.d."})'
        
        if len(source.authors) == 1:
            # Single author
            author_last = source.authors[0].split(",")[0] if "," in source.authors[0] else source.authors[0].split()[-1]
            return f"({author_last}, {source.publication_year or 'n.d.'})"
        
        elif len(source.authors) == 2:
            # Two authors
            author1 = source.authors[0].split(",")[0] if "," in source.authors[0] else source.authors[0].split()[-1]
            author2 = source.authors[1].split(",")[0] if "," in source.authors[1] else source.authors[1].split()[-1]
            return f"({author1} & {author2}, {source.publication_year or 'n.d.'})"
        
        else:
            # Three or more authors
            first_author = source.authors[0].split(",")[0] if "," in source.authors[0] else source.authors[0].split()[-1]
            return f"({first_author} et al., {source.publication_year or 'n.d.'})"
    
    def _format_authors(self, authors: List[str]) -> str:
        """Format author names according to APA style."""
        if not authors:
            return ""
        
        formatted_authors = []
        for author in authors:
            if "," in author:
                # Already in "Last, F." format
                formatted_authors.append(author)
            else:
                # Convert "First Last" to "Last, F."
                parts = author.strip().split()
                if len(parts) >= 2:
                    last_name = parts[-1]
                    first_initial = parts[0][0] + "."
                    middle_initials = " ".join([name[0] + "." for name in parts[1:-1]])
                    formatted_name = f"{last_name}, {first_initial}"
                    if middle_initials:
                        formatted_name += f" {middle_initials}"
                    formatted_authors.append(formatted_name)
                else:
                    formatted_authors.append(author)
        
        # Join authors with proper formatting
        if len(formatted_authors) == 1:
            return formatted_authors[0]
        elif len(formatted_authors) == 2:
            return f"{formatted_authors[0]}, & {formatted_authors[1]}"
        else:
            return ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
    
    def _format_title(self, title: str, source_type: str) -> str:
        """Format title according to APA style."""
        if not title:
            return ""
        
        # Sentence case (capitalize only first word and proper nouns)
        title = title.lower()
        title = title[0].upper() + title[1:] if title else ""
        
        # Capitalize after colons
        title = re.sub(r':\s*([a-z])', lambda m: ': ' + m.group(1).upper(), title)
        
        # Italicize if needed
        if source_type in ["book", "dissertation"]:
            title = f"*{title}*"
        
        return title
    
    def _format_journal_name(self, journal: str) -> str:
        """Format journal name in title case and italics."""
        if not journal:
            return ""
        
        # Convert to title case
        title_case = journal.title()
        
        # Handle common abbreviations
        abbreviations = {
            "Of": "of", "The": "the", "And": "and", "In": "in", "For": "for",
            "A": "a", "An": "an", "To": "to", "By": "by", "With": "with"
        }
        
        for word, replacement in abbreviations.items():
            title_case = re.sub(rf'\b{word}\b', replacement, title_case)
        
        # Ensure first word is capitalized
        title_case = title_case[0].upper() + title_case[1:] if title_case else ""
        
        return f"*{title_case}*"
    
    def _format_pages(self, pages: str) -> str:
        """Format page numbers according to APA style."""
        if not pages:
            return ""
        
        # Handle different page formats
        if "-" in pages:
            return pages  # Already formatted as range
        elif "," in pages:
            return pages  # Multiple non-consecutive pages
        else:
            return pages  # Single page
    
    def _format_doi(self, doi: str) -> str:
        """Format DOI according to APA style."""
        if not doi:
            return ""
        
        # Remove existing DOI prefix if present
        doi = re.sub(r'^(doi:|DOI:|https?://doi\.org/)', '', doi)
        
        return f"https://doi.org/{doi}"
    
    def _format_retrieved_date_url(self, url: str, access_date: str = None) -> str:
        """Format retrieved date and URL."""
        if not url:
            return ""
        
        if access_date:
            return f"Retrieved {access_date}, from {url}"
        else:
            return url
    
    def _clean_citation_formatting(self, citation: str) -> str:
        """Clean up citation formatting issues."""
        # Remove double spaces
        citation = re.sub(r'\s+', ' ', citation)
        
        # Remove empty parentheses
        citation = re.sub(r'\(\s*\)', '', citation)
        
        # Fix punctuation spacing
        citation = re.sub(r'\s+([,.])', r'\1', citation)
        citation = re.sub(r'([,.])([^\s])', r'\1 \2', citation)
        
        # Remove trailing/leading whitespace
        citation = citation.strip()
        
        return citation
    
    async def validate_citation(self, citation: str) -> CitationValidationResult:
        """
        Validate APA citation formatting.
        
        Args:
            citation: Citation text to validate
            
        Returns:
            Validation result with issues and suggestions
        """
        logger.info("🔍 Validating APA citation formatting")
        
        issues = []
        suggestions = []
        corrected_citation = citation
        
        # Check for common errors
        for error_category, error_data in self.common_errors.items():
            for pattern_info in error_data["patterns"]:
                if re.search(pattern_info["error"], citation):
                    issues.append(pattern_info["description"])
                    corrected_citation = re.sub(
                        pattern_info["error"], 
                        pattern_info["correction"], 
                        corrected_citation
                    )
        
        # Check APA compliance
        compliance_score = self._calculate_apa_compliance(citation)
        
        # Generate suggestions
        if compliance_score < 0.8:
            suggestions.extend([
                "Review APA 7th edition guidelines for proper formatting",
                "Check author name formatting (Last, F. M.)",
                "Ensure titles are in sentence case",
                "Verify date formatting is correct"
            ])
        
        return CitationValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            corrected_citation=corrected_citation if corrected_citation != citation else None,
            apa_compliance_score=compliance_score
        )
    
    def _calculate_apa_compliance(self, citation: str) -> float:
        """Calculate APA compliance score (0-1)."""
        score = 1.0
        
        # Check for basic APA elements
        checks = [
            (r'\(\d{4}\)', "Year in parentheses", 0.2),
            (r'[A-Z][a-z]+,\s+[A-Z]\.', "Author last name, first initial", 0.2),
            (r'\&', "Ampersand for multiple authors", 0.1),
            (r'\.', "Proper punctuation", 0.1),
            (r'[a-z][^A-Z]*[a-z]', "Sentence case titles", 0.2),
            (r'https?://doi\.org/', "Proper DOI formatting", 0.2)
        ]
        
        for pattern, description, weight in checks:
            if not re.search(pattern, citation):
                score -= weight
        
        return max(0.0, score)
    
    async def extract_metadata_from_url(self, url: str) -> CitationSource:
        """
        Extract citation metadata from a URL.
        
        Args:
            url: URL to extract metadata from
            
        Returns:
            CitationSource with extracted metadata
        """
        logger.info(f"🌐 Extracting metadata from URL: {url}")
        
        if not WEB_SCRAPING_AVAILABLE:
            logger.warning("Web scraping not available - returning basic source")
            return CitationSource(
                source_type="website",
                authors=[],
                title="[Title not available]",
                publication_year="n.d.",
                url=url,
                access_date=datetime.now().strftime("%B %d, %Y")
            )
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = self._extract_title(soup)
            authors = self._extract_authors(soup)
            date = self._extract_date(soup)
            
            return CitationSource(
                source_type="website",
                authors=authors,
                title=title,
                publication_year=date,
                url=url,
                access_date=datetime.now().strftime("%B %d, %Y")
            )
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return CitationSource(
                source_type="website",
                authors=[],
                title="[Title not available]",
                publication_year="n.d.",
                url=url,
                access_date=datetime.now().strftime("%B %d, %Y")
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML soup."""
        # Try different title sources
        title_selectors = [
            'meta[property="og:title"]',
            'meta[name="title"]',
            'title',
            'h1'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content', '').strip()
                else:
                    return element.get_text().strip()
        
        return "[Title not available]"
    
    def _extract_authors(self, soup: BeautifulSoup) -> List[str]:
        """Extract authors from HTML soup."""
        authors = []
        
        # Try different author sources
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '.author',
            '.byline'
        ]
        
        for selector in author_selectors:
            elements = soup.select(selector)
            for element in elements:
                if element.name == 'meta':
                    content = element.get('content', '').strip()
                    if content:
                        authors.append(content)
                else:
                    text = element.get_text().strip()
                    if text:
                        authors.append(text)
        
        return authors[:5]  # Limit to 5 authors
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publication date from HTML soup."""
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'time[datetime]',
            '.date'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    date_str = element.get('content', '')
                elif element.name == 'time':
                    date_str = element.get('datetime', '') or element.get_text()
                else:
                    date_str = element.get_text()
                
                # Extract year from date string
                year_match = re.search(r'\b(20\d{2}|19\d{2})\b', date_str)
                if year_match:
                    return year_match.group(1)
        
        return "n.d."
    
    async def generate_bibliography(
        self,
        sources: List[CitationSource],
        sort_order: str = "alphabetical"
    ) -> List[BibliographyEntry]:
        """
        Generate a formatted bibliography.
        
        Args:
            sources: List of citation sources
            sort_order: "alphabetical" or "chronological"
            
        Returns:
            List of formatted bibliography entries
        """
        logger.info(f"📚 Generating bibliography with {len(sources)} sources")
        
        entries = []
        
        for source in sources:
            citation = await self.generate_citation(source, "reference_list")
            in_text = self._generate_in_text_citation(source)
            
            # Create sort key
            if sort_order == "alphabetical":
                sort_key = source.authors[0].split(",")[0] if source.authors else source.title
            else:  # chronological
                sort_key = source.publication_year or "9999"
            
            entries.append(BibliographyEntry(
                citation=citation,
                source=source,
                sort_key=sort_key.lower(),
                in_text_format=in_text
            ))
        
        # Sort entries
        entries.sort(key=lambda x: x.sort_key)
        
        return entries
    
    def check_citation_quality(self, citation: str) -> Dict[str, Any]:
        """Quick synchronous citation quality check."""
        try:
            # Basic APA format checks
            has_author = bool(re.search(r'[A-Z][a-z]+,\s+[A-Z]\.', citation))
            has_year = bool(re.search(r'\(\d{4}\)', citation))
            has_title = len(citation.strip()) > 20
            has_punctuation = citation.endswith('.')
            
            quality_score = sum([has_author, has_year, has_title, has_punctuation]) / 4
            
            issues = []
            if not has_author:
                issues.append("Missing or improperly formatted author")
            if not has_year:
                issues.append("Missing or improperly formatted year")
            if not has_title:
                issues.append("Missing or insufficient title")
            if not has_punctuation:
                issues.append("Missing final punctuation")
            
            return {
                "success": True,
                "quality_score": quality_score,
                "apa_compliance": quality_score > 0.7,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 
