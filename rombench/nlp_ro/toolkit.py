"""
Romanian NLP Toolkit

Main interface for Romanian text analysis in RomBench.
Combines diacritic analysis, code-switch detection, and text metrics
into a single, easy-to-use toolkit.

All analysis is deterministic - no ML models, no external API calls.
"""

import math
from dataclasses import dataclass
from typing import Optional

from .tokenizer import (
    tokenize,
    tokenize_words,
    normalize_diacritics,
    has_romanian_diacritics,
    count_diacritics,
)
from .diacritics import analyze_diacritics, DiacriticAnalysis
from .codeswitch import detect_code_switching, CodeSwitchAnalysis, is_likely_english_text


@dataclass
class TextQualityReport:
    """
    Complete text quality analysis for Romanian text.

    This is the main output used for the G (Generation Quality) metric.
    """
    # Component scores (0.0-1.0, higher is better)
    diacritic_score: float      # Diacritic correctness
    codeswitch_score: float     # Absence of code-switching (1.0 = no English)
    length_score: float         # Adequate length (not too short)

    # Combined score
    overall_score: float        # Weighted combination of components

    # Detailed analyses
    diacritics: DiacriticAnalysis
    codeswitch: CodeSwitchAnalysis

    # Basic stats
    total_tokens: int
    total_words: int
    has_diacritics: bool

    # Flags
    is_likely_english: bool     # True if text appears to be in English
    is_too_short: bool          # True if text is suspiciously short

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "diacritic_score": self.diacritic_score,
            "codeswitch_score": self.codeswitch_score,
            "length_score": self.length_score,
            "overall_score": self.overall_score,
            "total_tokens": self.total_tokens,
            "total_words": self.total_words,
            "has_diacritics": self.has_diacritics,
            "is_likely_english": self.is_likely_english,
            "is_too_short": self.is_too_short,
            "diacritics_detail": {
                "score": self.diacritics.score,
                "correct": self.diacritics.correct_diacritics,
                "missing": self.diacritics.missing_diacritics,
                "missing_examples": self.diacritics.missing_words[:5],
            },
            "codeswitch_detail": {
                "score": self.codeswitch.score,
                "english_words": self.codeswitch.english_words,
                "english_rate": self.codeswitch.english_rate,
                "flagged_examples": self.codeswitch.flagged_words[:5],
            },
        }


class RomanianNLPToolkit:
    """
    Main toolkit for Romanian text analysis.

    Usage:
        toolkit = RomanianNLPToolkit()
        report = toolkit.analyze(text)
        print(f"Quality score: {report.overall_score}")
    """

    def __init__(
        self,
        min_words_for_full_analysis: int = 10,
        diacritic_weight: float = 0.5,
        codeswitch_weight: float = 0.3,
        length_weight: float = 0.2,
    ):
        """
        Initialize the toolkit.

        Args:
            min_words_for_full_analysis: Minimum words for meaningful analysis
            diacritic_weight: Weight for diacritic score in overall score
            codeswitch_weight: Weight for code-switch score in overall score
            length_weight: Weight for length score in overall score
        """
        self.min_words = min_words_for_full_analysis
        self.diacritic_weight = diacritic_weight
        self.codeswitch_weight = codeswitch_weight
        self.length_weight = length_weight

    def analyze(self, text: str) -> TextQualityReport:
        """
        Perform complete text quality analysis.

        Args:
            text: Romanian text to analyze

        Returns:
            TextQualityReport with all scores and details
        """
        # Normalize text (cedilla -> comma)
        normalized = normalize_diacritics(text)

        # Basic tokenization
        tokens = tokenize(normalized)
        words = [t for t in tokens if t.is_word]
        total_tokens = len(tokens)
        total_words = len(words)

        # Check for diacritics presence
        has_diacritics = has_romanian_diacritics(normalized)

        # Length score - penalize very short texts
        if total_words < 5:
            length_score = 0.3
            is_too_short = True
        elif total_words < self.min_words:
            length_score = 0.5 + 0.5 * (total_words / self.min_words)
            is_too_short = True
        else:
            length_score = 1.0
            is_too_short = False

        # Diacritic analysis
        diacritic_analysis = analyze_diacritics(normalized)
        diacritic_score = diacritic_analysis.score

        # Code-switch detection
        codeswitch_analysis = detect_code_switching(normalized)
        codeswitch_score = codeswitch_analysis.score

        # Check if text is predominantly English
        is_english = is_likely_english_text(normalized)
        if is_english:
            # Severe penalty for responding in wrong language
            codeswitch_score = 0.1
            diacritic_score = min(diacritic_score, 0.3)

        # Compute overall score (weighted combination)
        overall_score = (
            self.diacritic_weight * diacritic_score +
            self.codeswitch_weight * codeswitch_score +
            self.length_weight * length_score
        )

        return TextQualityReport(
            diacritic_score=diacritic_score,
            codeswitch_score=codeswitch_score,
            length_score=length_score,
            overall_score=overall_score,
            diacritics=diacritic_analysis,
            codeswitch=codeswitch_analysis,
            total_tokens=total_tokens,
            total_words=total_words,
            has_diacritics=has_diacritics,
            is_likely_english=is_english,
            is_too_short=is_too_short,
        )

    def compute_g_score(self, text: str) -> dict:
        """
        Compute G (Generation Quality) score for the metrics system.

        This method returns a dictionary in the format expected by
        the compute_G function in metrics.py.

        Args:
            text: Romanian text (explanation from model output)

        Returns:
            Dictionary with G score and component details
        """
        report = self.analyze(text)

        return {
            "G": report.overall_score,
            "G_dia": report.diacritic_score,
            "G_cs": report.codeswitch_score,
            "G_len": report.length_score,
            "n_tokens": report.total_tokens,
            "n_words": report.total_words,
            "has_diacritics": report.has_diacritics,
            "is_likely_english": report.is_likely_english,
            "is_too_short": report.is_too_short,
            "diacritic_details": {
                "correct": report.diacritics.correct_diacritics,
                "missing": report.diacritics.missing_diacritics,
                "examples": report.diacritics.missing_words[:5],
            },
            "codeswitch_details": {
                "english_count": report.codeswitch.english_words,
                "english_rate": report.codeswitch.english_rate,
                "examples": report.codeswitch.flagged_words[:5],
            },
        }


# Convenience functions for direct use

def analyze_romanian_text(text: str) -> TextQualityReport:
    """
    Analyze Romanian text quality (convenience function).

    Args:
        text: Romanian text to analyze

    Returns:
        TextQualityReport
    """
    toolkit = RomanianNLPToolkit()
    return toolkit.analyze(text)


def compute_generation_quality(text: str) -> dict:
    """
    Compute G score for metrics (convenience function).

    Args:
        text: Romanian text

    Returns:
        Dictionary with G score and details
    """
    toolkit = RomanianNLPToolkit()
    return toolkit.compute_g_score(text)
