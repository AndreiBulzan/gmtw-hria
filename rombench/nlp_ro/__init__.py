"""
Romanian NLP Toolkit for RomBench

A deterministic, lightweight toolkit for analyzing Romanian text quality.
Used primarily for the G (Generation Quality) metric in GMTW-Ro evaluation.

Main components:
- Diacritic analyzer: Checks correct usage of Romanian diacritics (ă, â, î, ș, ț)
- Code-switch detector: Detects English words in Romanian text
- Text quality scorer: Combines components into overall quality score

All analysis is fully deterministic - no ML models, no randomness.

Usage:
    from rombench.nlp_ro import RomanianNLPToolkit, analyze_romanian_text

    # Full toolkit
    toolkit = RomanianNLPToolkit()
    report = toolkit.analyze("Aceasta este o propoziție în limba română.")
    print(f"Quality: {report.overall_score}")

    # Quick analysis
    report = analyze_romanian_text("Text de analizat")

    # For metrics integration
    g_score = toolkit.compute_g_score(explanation_text)
"""

from .tokenizer import (
    Token,
    tokenize,
    tokenize_words,
    strip_diacritics,
    normalize_diacritics,
    has_romanian_diacritics,
    count_diacritics,
)

from .diacritics import (
    DiacriticAnalysis,
    analyze_diacritics,
    quick_diacritic_check,
)

from .codeswitch import (
    CodeSwitchAnalysis,
    detect_code_switching,
    is_likely_english_text,
)

from .toolkit import (
    TextQualityReport,
    RomanianNLPToolkit,
    analyze_romanian_text,
    compute_generation_quality,
)

# Optional grammar module (requires language-tool-python)
# Import conditionally to avoid ImportError if not installed
from .grammar import (
    is_available as grammar_is_available,
    compute_grammar_score,
    GrammarAnalysis,
)

__all__ = [
    # Tokenizer
    "Token",
    "tokenize",
    "tokenize_words",
    "strip_diacritics",
    "normalize_diacritics",
    "has_romanian_diacritics",
    "count_diacritics",
    # Diacritics
    "DiacriticAnalysis",
    "analyze_diacritics",
    "quick_diacritic_check",
    # Code-switch
    "CodeSwitchAnalysis",
    "detect_code_switching",
    "is_likely_english_text",
    # Toolkit
    "TextQualityReport",
    "RomanianNLPToolkit",
    "analyze_romanian_text",
    "compute_generation_quality",
    # Grammar (optional - requires language-tool-python)
    "grammar_is_available",
    "compute_grammar_score",
    "GrammarAnalysis",
]
