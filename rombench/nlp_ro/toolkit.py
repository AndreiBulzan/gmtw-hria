"""
Romanian NLP Toolkit

Main interface for Romanian text analysis in RomBench.
Combines diacritic analysis, code-switch detection, and text metrics
into a single, easy-to-use toolkit.

Core analysis is deterministic - no ML models, no external API calls.
Optional grammar checking via LanguageTool is available.
"""

import math
from dataclasses import dataclass
from typing import Optional, Any

from .tokenizer import (
    tokenize,
    tokenize_words,
    normalize_diacritics,
    has_romanian_diacritics,
    count_diacritics,
)
from .diacritics import analyze_diacritics, DiacriticAnalysis
from .codeswitch import detect_code_switching, CodeSwitchAnalysis, is_likely_english_text
from .punctuation import analyze_punctuation, PunctuationAnalysis


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
    punctuation_score: float = 1.0  # Punctuation quality
    grammar_score: Optional[float] = None  # Grammar/spelling (optional, requires LanguageTool)

    # Combined score
    overall_score: float = 0.0  # Weighted combination of components

    # Detailed analyses
    diacritics: DiacriticAnalysis = None
    codeswitch: CodeSwitchAnalysis = None
    punctuation: PunctuationAnalysis = None
    grammar_details: Optional[dict] = None  # Grammar analysis details

    # Basic stats
    total_tokens: int = 0
    total_words: int = 0
    has_diacritics: bool = False

    # Flags
    is_likely_english: bool = False     # True if text appears to be in English
    is_too_short: bool = False          # True if text is suspiciously short
    grammar_available: bool = False     # True if LanguageTool was used

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        result = {
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
            } if self.diacritics else None,
            "codeswitch_detail": {
                "score": self.codeswitch.score,
                "english_words": self.codeswitch.english_words,
                "english_rate": self.codeswitch.english_rate,
                "flagged_examples": self.codeswitch.flagged_words[:5],
            } if self.codeswitch else None,
        }

        # Add grammar details if available
        if self.grammar_available and self.grammar_score is not None:
            result["grammar_score"] = self.grammar_score
            result["grammar_detail"] = self.grammar_details

        return result


class RomanianNLPToolkit:
    """
    Main toolkit for Romanian text analysis.

    Usage:
        toolkit = RomanianNLPToolkit()
        report = toolkit.analyze(text)
        print(f"Quality score: {report.overall_score}")

        # With optional grammar checking (requires language-tool-python)
        toolkit = RomanianNLPToolkit(use_grammar=True)
        report = toolkit.analyze(text)
    """

    # Default weights WITHOUT grammar (sum = 1.0)
    DEFAULT_WEIGHTS = {
        "diacritic": 0.45,
        "codeswitch": 0.30,
        "punctuation": 0.15,
        "length": 0.10,
    }

    # Weights WITH grammar enabled (sum = 1.0)
    WEIGHTS_WITH_GRAMMAR = {
        "diacritic": 0.35,
        "codeswitch": 0.20,
        "punctuation": 0.10,
        "length": 0.10,
        "grammar": 0.25,
    }

    def __init__(
        self,
        min_words_for_full_analysis: int = 10,
        use_grammar: bool = False,
        diacritic_weight: Optional[float] = None,
        codeswitch_weight: Optional[float] = None,
        punctuation_weight: Optional[float] = None,
        length_weight: Optional[float] = None,
        grammar_weight: Optional[float] = None,
    ):
        """
        Initialize the toolkit.

        Args:
            min_words_for_full_analysis: Minimum words for meaningful analysis
            use_grammar: If True, use LanguageTool for grammar checking (requires language-tool-python)
            diacritic_weight: Override weight for diacritic score
            codeswitch_weight: Override weight for code-switch score
            punctuation_weight: Override weight for punctuation score
            length_weight: Override weight for length score
            grammar_weight: Override weight for grammar score (only used if use_grammar=True)
        """
        self.min_words = min_words_for_full_analysis
        self.use_grammar = use_grammar
        self._grammar_module = None

        # Set weights based on mode
        if use_grammar:
            base_weights = self.WEIGHTS_WITH_GRAMMAR.copy()
        else:
            base_weights = self.DEFAULT_WEIGHTS.copy()

        # Apply any overrides
        self.diacritic_weight = diacritic_weight if diacritic_weight is not None else base_weights["diacritic"]
        self.codeswitch_weight = codeswitch_weight if codeswitch_weight is not None else base_weights["codeswitch"]
        self.punctuation_weight = punctuation_weight if punctuation_weight is not None else base_weights["punctuation"]
        self.length_weight = length_weight if length_weight is not None else base_weights["length"]
        self.grammar_weight = grammar_weight if grammar_weight is not None else base_weights.get("grammar", 0.0)

        # Lazy load grammar module if needed
        if use_grammar:
            self._init_grammar()

    def _init_grammar(self):
        """Initialize the grammar module (lazy loading)."""
        try:
            from . import grammar
            if grammar.is_available():
                self._grammar_module = grammar
            else:
                import warnings
                warnings.warn(
                    "use_grammar=True but language-tool-python is not installed. "
                    "Grammar checking will be skipped. "
                    "Install with: pip install language-tool-python"
                )
                self.use_grammar = False
        except ImportError:
            import warnings
            warnings.warn(
                "Grammar module not available. Grammar checking will be skipped."
            )
            self.use_grammar = False

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

        # Punctuation analysis
        punctuation_analysis = analyze_punctuation(normalized)
        punctuation_score = punctuation_analysis.score

        # Check if text is predominantly English
        is_english = is_likely_english_text(normalized)
        if is_english:
            # Severe penalty for responding in wrong language
            codeswitch_score = 0.1
            diacritic_score = min(diacritic_score, 0.3)

        # Optional grammar analysis
        grammar_score = None
        grammar_details = None
        grammar_available = False

        if self.use_grammar and self._grammar_module is not None:
            try:
                grammar_result = self._grammar_module.compute_grammar_score(normalized)
                if grammar_result.get("available", False):
                    grammar_score = grammar_result.get("G_grammar")
                    grammar_details = {
                        "error_count": grammar_result.get("error_count", 0),
                        "weighted_errors": grammar_result.get("weighted_errors", 0),
                        "error_density": grammar_result.get("error_density", 0),
                        "skipped_proper_nouns": grammar_result.get("skipped_proper_nouns", 0),
                        "errors": grammar_result.get("errors", [])[:5],  # Limit examples
                    }
                    grammar_available = True
            except Exception:
                # Grammar check failed, continue without it
                pass

        # Compute overall score (weighted combination)
        if grammar_available and grammar_score is not None:
            # Use grammar-enabled weights
            overall_score = (
                self.diacritic_weight * diacritic_score +
                self.codeswitch_weight * codeswitch_score +
                self.punctuation_weight * punctuation_score +
                self.length_weight * length_score +
                self.grammar_weight * grammar_score
            )
        else:
            # Fall back to default weights (sum to 1.0)
            overall_score = (
                self.DEFAULT_WEIGHTS["diacritic"] * diacritic_score +
                self.DEFAULT_WEIGHTS["codeswitch"] * codeswitch_score +
                self.DEFAULT_WEIGHTS["punctuation"] * punctuation_score +
                self.DEFAULT_WEIGHTS["length"] * length_score
            )

        return TextQualityReport(
            diacritic_score=diacritic_score,
            codeswitch_score=codeswitch_score,
            length_score=length_score,
            punctuation_score=punctuation_score,
            grammar_score=grammar_score,
            overall_score=overall_score,
            diacritics=diacritic_analysis,
            codeswitch=codeswitch_analysis,
            punctuation=punctuation_analysis,
            grammar_details=grammar_details,
            total_tokens=total_tokens,
            total_words=total_words,
            has_diacritics=has_diacritics,
            is_likely_english=is_english,
            is_too_short=is_too_short,
            grammar_available=grammar_available,
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

        result = {
            "G": report.overall_score,
            "G_dia": report.diacritic_score,
            "G_cs": report.codeswitch_score,
            "G_punct": report.punctuation_score,
            "G_len": report.length_score,
            "n_tokens": report.total_tokens,
            "n_words": report.total_words,
            "has_diacritics": report.has_diacritics,
            "is_likely_english": report.is_likely_english,
            "is_too_short": report.is_too_short,
            "diacritic_details": {
                "correct": report.diacritics.correct_diacritics,
                "missing": report.diacritics.missing_diacritics,
                "examples": report.diacritics.missing_words[:15],
            },
            "codeswitch_details": {
                "english_count": report.codeswitch.english_words,
                "english_rate": report.codeswitch.english_rate,
                "examples": report.codeswitch.flagged_words[:10],
            },
            "punctuation_details": {
                "total_issues": report.punctuation.total_issues,
                "space_before_punct": report.punctuation.space_before_punct,
                "missing_space_after": report.punctuation.missing_space_after,
                "double_spaces": report.punctuation.double_spaces,
                "examples": report.punctuation.examples[:5],
            },
        }

        # Add grammar details if available
        if report.grammar_available and report.grammar_score is not None:
            result["G_grammar"] = report.grammar_score
            result["grammar_available"] = True
            result["grammar_details"] = report.grammar_details
        else:
            result["G_grammar"] = None
            result["grammar_available"] = False

        return result


# Convenience functions for direct use

def analyze_romanian_text(text: str, use_grammar: bool = False) -> TextQualityReport:
    """
    Analyze Romanian text quality (convenience function).

    Args:
        text: Romanian text to analyze
        use_grammar: If True, include LanguageTool grammar checking

    Returns:
        TextQualityReport
    """
    toolkit = RomanianNLPToolkit(use_grammar=use_grammar)
    return toolkit.analyze(text)


def compute_generation_quality(text: str, use_grammar: bool = False) -> dict:
    """
    Compute G score for metrics (convenience function).

    Args:
        text: Romanian text
        use_grammar: If True, include LanguageTool grammar checking

    Returns:
        Dictionary with G score and details
    """
    toolkit = RomanianNLPToolkit(use_grammar=use_grammar)
    return toolkit.compute_g_score(text)
