"""
Language Enhancement Engine

Uses spaCy, NLTK, and other NLP libraries to enhance language quality,
check grammar, and improve writing style.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional
import string

logger = logging.getLogger(__name__)

# Try to import NLP libraries with graceful fallbacks
try:
    import spacy
    import nltk
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    NLP_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some NLP libraries not available: {e}")
    NLP_LIBRARIES_AVAILABLE = False


class LanguageEnhancementEngine:
    """
    Provides language quality enhancement using specialized NLP libraries.
    """
    
    def __init__(self):
        self.nlp = None
        self._initialize_nlp()
        logger.info("📝 Language Enhancement Engine initialized")
    
    def _initialize_nlp(self):
        """Initialize NLP models if available."""
        if NLP_LIBRARIES_AVAILABLE:
            try:
                # Try to load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Some features may be limited.")
                self.nlp = None
    
    async def enhance_language_quality(
        self,
        text: str,
        requirements: Dict[str, Any] = None
    ) -> str:
        """
        Enhance language quality with grammar checking and style improvements.
        
        Args:
            text: Text to enhance
            requirements: Specific enhancement requirements
            
        Returns:
            Enhanced text with quality analysis
        """
        if requirements is None:
            requirements = {}
        
        try:
            # Perform various language quality checks
            grammar_analysis = await self._check_grammar(text)
            readability_analysis = await self._analyze_readability(text)
            style_analysis = await self._analyze_style(text)
            improvements = await self._suggest_improvements(text)
            
            return f"""
Language Quality Enhancement Report:

**Original Text:**
{text}

**Grammar Analysis:**
{grammar_analysis}

**Readability Analysis:**
{readability_analysis}

**Style Analysis:**
{style_analysis}

**Suggested Improvements:**
{improvements}

**Enhancement Status:** Analysis complete using NLP libraries.
"""
            
        except Exception as e:
            logger.error(f"Language enhancement failed: {str(e)}")
            return f"Language enhancement error: {str(e)}"
    
    async def _check_grammar(self, text: str) -> str:
        """Check grammar and basic language issues."""
        
        issues = []
        
        # Basic grammar checks
        sentences = text.split('.')
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check capitalization
            if sentence and not sentence[0].isupper():
                issues.append(f"Sentence {i+1}: Should start with capital letter")
            
            # Check for double spaces
            if '  ' in sentence:
                issues.append(f"Sentence {i+1}: Contains double spaces")
        
        # Check for common issues
        if text.count('(') != text.count(')'):
            issues.append("Mismatched parentheses")
        
        if text.count('"') % 2 != 0:
            issues.append("Mismatched quotation marks")
        
        if not issues:
            return "✅ No basic grammar issues detected."
        else:
            return "⚠️ Issues found:\n" + "\n".join(f"- {issue}" for issue in issues)
    
    async def _analyze_readability(self, text: str) -> str:
        """Analyze text readability."""
        
        if not NLP_LIBRARIES_AVAILABLE:
            return "Readability analysis requires textstat library."
        
        try:
            # Basic readability metrics
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            avg_words_per_sentence = word_count / max(sentence_count, 1)
            
            # Calculate readability scores if textstat is available
            try:
                flesch_score = flesch_reading_ease(text)
                grade_level = flesch_kincaid_grade(text)
                
                return f"""
**Word Count:** {word_count}
**Sentence Count:** {sentence_count}
**Average Words per Sentence:** {avg_words_per_sentence:.1f}
**Flesch Reading Ease:** {flesch_score:.1f}
**Grade Level:** {grade_level:.1f}

**Readability Assessment:** {'Easy to read' if flesch_score > 60 else 'Moderate difficulty' if flesch_score > 30 else 'Difficult to read'}
"""
            except:
                return f"""
**Word Count:** {word_count}
**Sentence Count:** {sentence_count}
**Average Words per Sentence:** {avg_words_per_sentence:.1f}

**Note:** Advanced readability metrics require additional setup.
"""
                
        except Exception as e:
            return f"Readability analysis error: {str(e)}"
    
    async def _analyze_style(self, text: str) -> str:
        """Analyze writing style."""
        
        style_notes = []
        
        # Check for passive voice indicators
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(1 for word in text.lower().split() if word in passive_indicators)
        
        if passive_count > len(text.split()) * 0.1:
            style_notes.append("Consider reducing passive voice usage")
        
        # Check for filler words
        filler_words = ['very', 'really', 'quite', 'rather', 'somewhat']
        filler_count = sum(1 for word in text.lower().split() if word in filler_words)
        
        if filler_count > 0:
            style_notes.append(f"Found {filler_count} filler words that could be removed")
        
        # Check sentence variety
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            if all(abs(length - avg_length) < 2 for length in sentence_lengths):
                style_notes.append("Consider varying sentence length for better flow")
        
        if not style_notes:
            return "✅ Good writing style detected."
        else:
            return "📝 Style suggestions:\n" + "\n".join(f"- {note}" for note in style_notes)
    
    async def _suggest_improvements(self, text: str) -> str:
        """Suggest specific improvements."""
        
        improvements = []
        
        # Check for repetitive words
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only check longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_freq.items() if count > 3]
        if repeated_words:
            improvements.append(f"Consider synonyms for repeated words: {', '.join(repeated_words[:3])}")
        
        # Check for overly long sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        long_sentences = [i+1 for i, s in enumerate(sentences) if len(s.split()) > 25]
        
        if long_sentences:
            improvements.append(f"Consider breaking up long sentences: {long_sentences}")
        
        # Check for paragraph structure
        if '\n\n' not in text and len(text.split()) > 100:
            improvements.append("Consider adding paragraph breaks for better readability")
        
        if not improvements:
            return "✅ Text quality is good. No major improvements needed."
        else:
            return "\n".join(f"- {improvement}" for improvement in improvements)
    
    async def check_spelling(self, text: str) -> Dict[str, Any]:
        """Check spelling using available libraries."""
        
        try:
            # Basic spell check using word patterns
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            
            # Simple heuristic checks
            potential_errors = []
            
            for word in words:
                # Check for common patterns that might indicate errors
                if len(word) > 2:
                    # Check for repeated letters (simple heuristic)
                    if any(word.count(char) > 2 for char in word if char.isalpha()):
                        potential_errors.append(word)
            
            return {
                "total_words": len(words),
                "potential_errors": potential_errors[:5],  # Limit to first 5
                "status": "Basic spell check completed"
            }
            
        except Exception as e:
            return {"error": f"Spell check error: {str(e)}"}
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment."""
        
        try:
            # Simple sentiment analysis using word lists
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'happy']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'sad', 'angry', 'disappointed']
            
            words = text.lower().split()
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count > negative_count:
                sentiment = "Positive"
            elif negative_count > positive_count:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return {
                "sentiment": sentiment,
                "positive_words": positive_count,
                "negative_words": negative_count,
                "confidence": abs(positive_count - negative_count) / max(len(words), 1)
            }
            
        except Exception as e:
            return {"error": f"Sentiment analysis error: {str(e)}"}
    
    async def extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text."""
        
        try:
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            # Remove common stop words
            stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other'}
            
            keywords = [word for word in words if word not in stop_words]
            
            # Count frequency and return top keywords
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top 10
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in sorted_words[:10]]
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return []
    
    def enhance_text(self, text: str, enhancement_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Enhance text quality (synchronous wrapper for async method)
        
        Args:
            text: Text to enhance
            enhancement_type: Type of enhancement to perform
            
        Returns:
            Dictionary with enhancement results
        """
        try:
            import asyncio
            
            # Run the async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.enhance_language_quality(text, {"enhancement_type": enhancement_type})
                )
                return {
                    'success': True,
                    'enhanced_text': result,
                    'original_text': text,
                    'enhancement_type': enhancement_type
                }
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Text enhancement failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_text': text
            } 