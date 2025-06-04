# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Grammar Engine

Provides advanced grammar checking, style analysis, and writing enhancement
capabilities using multiple language processing libraries and rule sets.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import language processing libraries with graceful fallbacks
try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    logger.warning("language_tool_python not available - install with: pip install language-tool-python")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logger.warning("textstat not available - install with: pip install textstat")


@dataclass
class GrammarIssue:
    """Represents a grammar or style issue found in text."""
    message: str
    rule_id: str
    category: str
    offset: int
    length: int
    suggestions: List[str]
    severity: str = "error"  # error, warning, suggestion


@dataclass
class GrammarAnalysisResult:
    """Result of comprehensive grammar analysis."""
    original_text: str
    corrected_text: str
    issues: List[GrammarIssue]
    readability_score: float
    grade_level: str
    sentence_count: int
    word_count: int
    suggestions: List[str]
    quality_metrics: Dict[str, float]


class GrammarEngine:
    """
    Computational engine for linguistic analysis and text processing.
    
    Provides computational text analysis through algorithmic grammar checking,
    readability metrics, and style analysis using NLP libraries.
    """
    
    def __init__(self):
        self.name = "Grammar Analysis Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            "grammar_analysis",
            "readability_metrics",
            "style_analysis", 
            "sentence_structure_analysis",
            "lexical_diversity_analysis",
            "text_complexity_scoring"
        ]
        self.grammar_checker = self._initialize_grammar_checker()
        self.style_rules = self._initialize_style_rules()
        self.writing_patterns = self._initialize_writing_patterns()
        logger.info("📝 Grammar Analysis Engine initialized")
    
    def _initialize_grammar_checker(self):
        """Initialize the grammar checking tool."""
        if LANGUAGE_TOOL_AVAILABLE:
            try:
                return language_tool_python.LanguageTool('en-US')
            except Exception as e:
                logger.warning(f"Failed to initialize LanguageTool: {e}")
                return None
        return None
    
    def _initialize_style_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize style checking rules."""
        return {
            "passive_voice": {
                "pattern": r"\b(is|are|was|were|being|been|be)\s+\w+ed\b",
                "message": "Consider using active voice for clearer, more direct writing",
                "severity": "suggestion"
            },
            "weak_words": {
                "words": ["very", "really", "quite", "rather", "somewhat", "fairly", "pretty"],
                "message": "Consider using more specific, stronger words",
                "severity": "suggestion"
            },
            "redundancy": {
                "patterns": [
                    r"\bfree gift\b", r"\bfuture plans\b", r"\bpast history\b",
                    r"\badvance planning\b", r"\bexact same\b", r"\bunexpected surprise\b"
                ],
                "message": "Remove redundant words for cleaner writing",
                "severity": "warning"
            },
            "wordiness": {
                "replacements": {
                    "due to the fact that": "because",
                    "in order to": "to",
                    "for the purpose of": "for",
                    "in the event that": "if",
                    "at this point in time": "now",
                    "in spite of the fact that": "although"
                },
                "message": "Consider more concise alternatives",
                "severity": "suggestion"
            }
        }
    
    def _initialize_writing_patterns(self) -> Dict[str, Any]:
        """Initialize advanced writing pattern analysis."""
        return {
            "sentence_starters": {
                "variety_check": True,
                "common_starters": ["The", "This", "That", "It", "There", "I", "You", "We"],
                "target_variety": 0.7  # 70% of sentences should have different starters
            },
            "sentence_length": {
                "ideal_range": (15, 25),  # words per sentence
                "variety_check": True,
                "max_consecutive_long": 2
            },
            "paragraph_structure": {
                "ideal_length_range": (3, 8),  # sentences per paragraph
                "topic_sentence_check": True
            }
        }
    
    async def analyze_grammar_and_style(
        self,
        text: str,
        analysis_type: str = "comprehensive",
        context: Dict[str, Any] = None
    ) -> GrammarAnalysisResult:
        """
        Perform comprehensive grammar and style analysis.
        
        Args:
            text: Text to analyze
            analysis_type: "quick", "standard", or "comprehensive"
            context: Additional context (document type, audience, etc.)
            
        Returns:
            Detailed grammar analysis with suggestions
        """
        if context is None:
            context = {}
        
        logger.info(f"🔍 Analyzing text: {len(text)} characters, {analysis_type} analysis")
        
        # Basic grammar checking
        grammar_issues = await self._check_grammar(text)
        
        # Style analysis
        style_issues = await self._analyze_style(text)
        
        # Readability analysis
        readability_metrics = self._analyze_readability(text)
        
        # Generate corrected text
        corrected_text = self._apply_corrections(text, grammar_issues + style_issues)
        
        # Writing improvement suggestions
        suggestions = self._generate_writing_suggestions(text, context)
        
        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(
            text, grammar_issues, style_issues, readability_metrics
        )
        
        return GrammarAnalysisResult(
            original_text=text,
            corrected_text=corrected_text,
            issues=grammar_issues + style_issues,
            readability_score=readability_metrics.get("flesch_reading_ease", 0),
            grade_level=readability_metrics.get("grade_level", "Unknown"),
            sentence_count=readability_metrics.get("sentence_count", 0),
            word_count=readability_metrics.get("word_count", 0),
            suggestions=suggestions,
            quality_metrics=quality_metrics
        )
    
    async def _check_grammar(self, text: str) -> List[GrammarIssue]:
        """Check grammar using LanguageTool."""
        issues = []
        
        if self.grammar_checker:
            try:
                matches = self.grammar_checker.check(text)
                for match in matches:
                    issue = GrammarIssue(
                        message=match.message,
                        rule_id=match.ruleId,
                        category=match.category,
                        offset=match.offset,
                        length=match.errorLength,
                        suggestions=match.replacements[:3],  # Top 3 suggestions
                        severity="error" if "spelling" not in match.category.lower() else "warning"
                    )
                    issues.append(issue)
            except Exception as e:
                logger.error(f"Grammar checking failed: {e}")
        
        return issues
    
    async def _analyze_style(self, text: str) -> List[GrammarIssue]:
        """Analyze style and writing quality."""
        issues = []
        
        # Check for passive voice
        passive_matches = re.finditer(self.style_rules["passive_voice"]["pattern"], text, re.IGNORECASE)
        for match in passive_matches:
            issue = GrammarIssue(
                message=self.style_rules["passive_voice"]["message"],
                rule_id="PASSIVE_VOICE",
                category="Style",
                offset=match.start(),
                length=match.end() - match.start(),
                suggestions=["Consider rewriting in active voice"],
                severity=self.style_rules["passive_voice"]["severity"]
            )
            issues.append(issue)
        
        # Check for weak words
        weak_words = self.style_rules["weak_words"]["words"]
        for weak_word in weak_words:
            pattern = rf"\b{re.escape(weak_word)}\b"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issue = GrammarIssue(
                    message=f"'{weak_word}' is a weak modifier. {self.style_rules['weak_words']['message']}",
                    rule_id="WEAK_WORDS",
                    category="Style",
                    offset=match.start(),
                    length=match.end() - match.start(),
                    suggestions=[f"Replace '{weak_word}' with a more specific word"],
                    severity=self.style_rules["weak_words"]["severity"]
                )
                issues.append(issue)
        
        # Check for redundancy
        for pattern in self.style_rules["redundancy"]["patterns"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issue = GrammarIssue(
                    message=f"'{match.group()}' is redundant. {self.style_rules['redundancy']['message']}",
                    rule_id="REDUNDANCY",
                    category="Style",
                    offset=match.start(),
                    length=match.end() - match.start(),
                    suggestions=["Remove redundant words"],
                    severity=self.style_rules["redundancy"]["severity"]
                )
                issues.append(issue)
        
        # Check for wordiness
        for wordy_phrase, replacement in self.style_rules["wordiness"]["replacements"].items():
            pattern = re.escape(wordy_phrase)
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issue = GrammarIssue(
                    message=f"'{wordy_phrase}' can be simplified. {self.style_rules['wordiness']['message']}",
                    rule_id="WORDINESS",
                    category="Style",
                    offset=match.start(),
                    length=match.end() - match.start(),
                    suggestions=[f"Replace with '{replacement}'"],
                    severity=self.style_rules["wordiness"]["severity"]
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability and statistics."""
        metrics = {}
        
        if TEXTSTAT_AVAILABLE:
            try:
                metrics = {
                    "flesch_reading_ease": textstat.flesch_reading_ease(text),
                    "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                    "gunning_fog": textstat.gunning_fog(text),
                    "automated_readability_index": textstat.automated_readability_index(text),
                    "grade_level": self._get_grade_level(textstat.flesch_reading_ease(text)),
                    "sentence_count": textstat.sentence_count(text),
                    "word_count": textstat.word_count(text),
                    "character_count": textstat.char_count(text),
                    "syllable_count": textstat.syllable_count(text)
                }
            except Exception as e:
                logger.error(f"Readability analysis failed: {e}")
        
        # Fallback basic metrics
        if not metrics:
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
            metrics = {
                "sentence_count": sentences,
                "word_count": words,
                "avg_words_per_sentence": words / max(sentences, 1),
                "grade_level": "Unknown"
            }
        
        return metrics
    
    def _get_grade_level(self, flesch_score: float) -> str:
        """Convert Flesch Reading Ease score to grade level."""
        if flesch_score >= 90:
            return "5th grade"
        elif flesch_score >= 80:
            return "6th grade"
        elif flesch_score >= 70:
            return "7th grade"
        elif flesch_score >= 60:
            return "8th-9th grade"
        elif flesch_score >= 50:
            return "10th-12th grade"
        elif flesch_score >= 30:
            return "College level"
        else:
            return "Graduate level"
    
    def _apply_corrections(self, text: str, issues: List[GrammarIssue]) -> str:
        """Apply corrections to text based on identified issues."""
        corrected_text = text
        
        # Sort issues by offset in reverse order to maintain positions
        sorted_issues = sorted(issues, key=lambda x: x.offset, reverse=True)
        
        for issue in sorted_issues:
            if issue.suggestions and issue.severity in ["error", "warning"]:
                # Apply the first suggestion for errors and warnings
                start = issue.offset
                end = issue.offset + issue.length
                corrected_text = (
                    corrected_text[:start] + 
                    issue.suggestions[0] + 
                    corrected_text[end:]
                )
        
        return corrected_text
    
    def _generate_writing_suggestions(self, text: str, context: Dict[str, Any]) -> List[str]:
        """Generate general writing improvement suggestions."""
        suggestions = []
        
        # Sentence variety analysis
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 3:
            starters = [sentence.strip().split()[0] if sentence.strip() else "" for sentence in sentences]
            starter_variety = len(set(starters)) / len(starters) if starters else 0
            
            if starter_variety < 0.7:
                suggestions.append("Vary your sentence beginnings to improve flow and readability")
        
        # Paragraph length check
        paragraphs = text.split('\n\n')
        if any(len(p.split('.')) > 8 for p in paragraphs):
            suggestions.append("Consider breaking long paragraphs into smaller, more focused ones")
        
        # Context-specific suggestions
        document_type = context.get("document_type", "").lower()
        if document_type == "academic":
            suggestions.append("Ensure claims are supported with evidence and proper citations")
        elif document_type == "business":
            suggestions.append("Use clear, professional language and actionable statements")
        
        return suggestions
    
    def _calculate_quality_metrics(
        self,
        text: str,
        grammar_issues: List[GrammarIssue],
        style_issues: List[GrammarIssue],
        readability_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        word_count = readability_metrics.get("word_count", len(text.split()))
        
        # Error rates
        grammar_error_rate = len([i for i in grammar_issues if i.severity == "error"]) / max(word_count / 100, 1)
        style_issue_rate = len(style_issues) / max(word_count / 100, 1)
        
        # Quality scores (0-1 scale)
        grammar_score = max(0, 1 - (grammar_error_rate * 0.1))
        style_score = max(0, 1 - (style_issue_rate * 0.05))
        readability_score = min(1, readability_metrics.get("flesch_reading_ease", 50) / 100)
        
        overall_score = (grammar_score * 0.4 + style_score * 0.3 + readability_score * 0.3)
        
        return {
            "overall_quality": overall_score,
            "grammar_accuracy": grammar_score,
            "style_quality": style_score,
            "readability": readability_score,
            "error_density": grammar_error_rate + style_issue_rate
        }
    
    def check_text_quality(self, text: str) -> Dict[str, Any]:
        """Quick synchronous text quality check."""
        try:
            # Basic analysis without async operations
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Simple grammar checks
            basic_issues = []
            
            # Check for common issues
            if re.search(r'\s+([.!?])', text):
                basic_issues.append("Space before punctuation")
            
            if re.search(r'([.!?])[a-zA-Z]', text):
                basic_issues.append("Missing space after punctuation")
            
            return {
                "success": True,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "basic_issues": basic_issues,
                "quality_estimate": max(0.7, 1.0 - len(basic_issues) * 0.1)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 
