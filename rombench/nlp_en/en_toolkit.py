"""
English-specific NLP toolkit

Implements English text quality analysis for G score.
"""

from typing import Optional
from dataclasses import dataclass
import math

from rombench.nlp import BaseNLPToolkit, TextQualityReport


class EnglishNLPToolkit(BaseNLPToolkit):
    """
    English NLP toolkit for text quality analysis.
    
    Measures:
    - Grammar/spelling (via LanguageTool if available)
    - Text length adequacy
    - Basic style metrics
    """
    
    MIN_WORDS_REQUIRED = 60  # Minimum for full length score
    
    def __init__(self, use_grammar: bool = False):
        self.use_grammar = use_grammar
        self.grammar_checker = None
        
        if use_grammar:
            try:
                from ..nlp_ro.grammar import is_available
                if is_available():
                    from .en_grammar import EnglishGrammarChecker
                    self.grammar_checker = EnglishGrammarChecker()
            except ImportError:
                pass
    
    def analyze(self, text: str) -> TextQualityReport:
        """Analyze English text quality"""
        words = self.tokenize(text)
        word_count = len(words)
        
        # Length score
        length_score = self._compute_length_score(word_count)
        
        # Grammar score (if enabled)
        grammar_score = None
        grammar_details = None
        if self.grammar_checker:
            grammar_result = self.grammar_checker.check(text)
            grammar_score = grammar_result['score']
            grammar_details = grammar_result
        
        # Style score (basic - could be expanded)
        style_score = self._compute_style_score(text, word_count)
        
        # Combined score
        if grammar_score is not None:
            overall_score = (
                0.50 * grammar_score +
                0.30 * style_score +
                0.20 * length_score
            )
        else:
            overall_score = (
                0.70 * style_score +
                0.30 * length_score
            )
        
        return TextQualityReport(
            overall_score=overall_score,
            grammar_score=grammar_score,
            style_score=style_score,
            length_score=length_score,
            total_words=word_count,
            total_tokens=len(text.split()),
            is_too_short=word_count < 50,
            details={
                'grammar': grammar_details,
                'style': {'style_score': style_score},
                'length': {'word_count': word_count, 'length_score': length_score},
            }
        )
    
    def compute_g_score(self, text: str) -> dict:
        """Compute G score for evaluation"""
        report = self.analyze(text)
        
        result = {
            'G': report.overall_score,
            'G_style': report.style_score if report.style_score else 1.0,
            'G_len': report.length_score,
            'total_words': report.total_words,
        }
        
        if report.grammar_score is not None:
            result['G_grammar'] = report.grammar_score
            result['grammar_details'] = report.details.get('grammar', {})
        
        return result
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize into words"""
        # Simple word extraction
        words = []
        for token in text.split():
            # Strip punctuation
            token = token.strip('.,;:!?"()[]{}')
            if token and any(c.isalnum() for c in token):
                words.append(token)
        return words
    
    def normalize(self, text: str) -> str:
        """Normalize text for matching"""
        text = text.lower()
        # Remove punctuation
        for c in '.,;:!?"()[]{}':
            text = text.replace(c, ' ')
        return ' '.join(text.split())
    
    def _compute_style_score(self, text: str, word_count: int) -> float:
        """
        Basic style scoring for English.
        
        Checks:
        - Sentence structure (not all short/long)
        - Basic punctuation use
        - Paragraph formation
        """
        if not text or word_count < 10:
            return 0.0
        
        score = 1.0
        
        # Check for sentence variety
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            score *= 0.8  # Penalize single-sentence responses
        
        # Check for basic punctuation
        has_punctuation = any(c in text for c in '.,;:!?')
        if not has_punctuation:
            score *= 0.7
        
        # Check average sentence length
        if sentences:
            avg_sentence_len = word_count / len(sentences)
            if avg_sentence_len < 5:
                score *= 0.8  # Too short sentences
            elif avg_sentence_len > 100:
                score *= 0.9  # Too long sentences
        
        return score