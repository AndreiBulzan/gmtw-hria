"""
Grammar Checker for Romanian using LanguageTool

This module provides optional grammar checking functionality.
LanguageTool is an OPTIONAL dependency - if not installed, functions
will return None or raise ImportError.

Install with: pip install language-tool-python
Or: pip install rombench[grammar]
"""

from dataclasses import dataclass
from typing import Optional

# Lazy import - don't fail if language_tool_python not installed
_language_tool = None
_tool_instance = None


def _get_tool():
    """Lazy initialization of LanguageTool instance."""
    global _language_tool, _tool_instance

    if _tool_instance is not None:
        return _tool_instance

    try:
        import language_tool_python
        _language_tool = language_tool_python
        _tool_instance = language_tool_python.LanguageTool("ro")
        return _tool_instance
    except ImportError:
        raise ImportError(
            "language-tool-python is not installed. "
            "Install with: pip install language-tool-python "
            "or: pip install rombench[grammar]"
        )


def is_available() -> bool:
    """Check if LanguageTool is available."""
    try:
        import language_tool_python
        return True
    except ImportError:
        return False


# Severity weights for different error types
SEVERITY_WEIGHTS = {
    "grammar": 3,
    "misspelling": 2,
    "typographical": 1,
    "style": 1,
}


@dataclass
class GrammarAnalysis:
    """Results of grammar analysis."""
    score: float                    # 0.0-1.0, where 1.0 = no errors
    total_words: int
    error_count: int                # After filtering
    raw_error_count: int            # Before filtering
    weighted_errors: float
    error_density: float            # weighted_errors / total_words
    skipped_proper_nouns: int
    errors: list                    # List of error details
    available: bool = True          # Whether LanguageTool was available


def is_proper_noun_error(match, text: str) -> bool:
    """
    Check if the error is likely a false positive on a proper noun.

    Proper nouns (names, places) often get flagged as misspellings
    because they're not in the dictionary.
    """
    # Only filter MORFOLOGIK (spelling) errors
    if "MORFOLOGIK" not in match.rule_id:
        return False

    # Get the problematic word from context
    offset = match.offset
    length = match.errorLength

    if offset < len(text) and length > 0:
        word = text[offset:offset + length]
        # Check if word starts with uppercase (likely proper noun)
        if word and word[0].isupper():
            return True

    return False


def analyze_grammar(text: str, filter_proper_nouns: bool = True) -> GrammarAnalysis:
    """
    Analyze grammar and spelling in Romanian text using LanguageTool.

    Args:
        text: Romanian text to analyze
        filter_proper_nouns: If True, skip errors on capitalized words (likely names)

    Returns:
        GrammarAnalysis with score and error details

    Raises:
        ImportError: If language-tool-python is not installed
    """
    tool = _get_tool()

    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return GrammarAnalysis(
            score=1.0,
            total_words=0,
            error_count=0,
            raw_error_count=0,
            weighted_errors=0,
            error_density=0,
            skipped_proper_nouns=0,
            errors=[],
        )

    # Get all matches
    all_matches = tool.check(text)
    raw_count = len(all_matches)

    # Filter proper nouns if requested
    if filter_proper_nouns:
        matches = []
        skipped = 0
        for m in all_matches:
            if is_proper_noun_error(m, text):
                skipped += 1
            else:
                matches.append(m)
    else:
        matches = all_matches
        skipped = 0

    # Compute weighted errors
    weighted_sum = 0
    errors = []

    for m in matches:
        issue_type = m.rule_issue_type.lower() if m.rule_issue_type else "unknown"
        weight = SEVERITY_WEIGHTS.get(issue_type, 1)
        weighted_sum += weight

        errors.append({
            "rule_id": m.rule_id,
            "issue_type": issue_type,
            "weight": weight,
            "message": m.message,
            "context": m.context,
            "suggestions": m.replacements[:3] if m.replacements else [],
            "offset": m.offset,
            "length": m.errorLength,
        })

    # Compute score (0-1 scale)
    # Using exponential decay: score = exp(-k * error_density)
    # This means small error rates have minimal impact, but high rates are penalized heavily
    error_density = weighted_sum / word_count if word_count > 0 else 0

    import math
    # Calibrated so that:
    # - 0% errors = 1.0
    # - 5% weighted error density = ~0.78
    # - 10% = ~0.61
    # - 20% = ~0.37
    score = math.exp(-2.5 * error_density)

    return GrammarAnalysis(
        score=score,
        total_words=word_count,
        error_count=len(matches),
        raw_error_count=raw_count,
        weighted_errors=weighted_sum,
        error_density=error_density,
        skipped_proper_nouns=skipped,
        errors=errors,
    )


def compute_grammar_score(text: str) -> dict:
    """
    Compute G_grammar score for the metrics system.

    Returns a dictionary compatible with the G metric structure.

    Args:
        text: Romanian text to analyze

    Returns:
        Dictionary with G_grammar score and details, or None if unavailable
    """
    if not is_available():
        return {
            "G_grammar": None,
            "available": False,
            "error": "language-tool-python not installed",
        }

    try:
        analysis = analyze_grammar(text, filter_proper_nouns=True)

        return {
            "G_grammar": analysis.score,
            "available": True,
            "error_count": analysis.error_count,
            "raw_error_count": analysis.raw_error_count,
            "weighted_errors": analysis.weighted_errors,
            "error_density": analysis.error_density,
            "skipped_proper_nouns": analysis.skipped_proper_nouns,
            "errors": analysis.errors[:10],  # Limit to 10 examples
        }
    except Exception as e:
        return {
            "G_grammar": None,
            "available": False,
            "error": str(e),
        }


def close_tool():
    """Close the LanguageTool instance to free resources."""
    global _tool_instance
    if _tool_instance is not None:
        _tool_instance.close()
        _tool_instance = None
