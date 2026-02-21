"""
Base tokenizer utilities (language-agnostic)
"""

import re
from typing import List


def basic_tokenize(text: str) -> List[str]:
    """
    Basic tokenization - splits on whitespace and punctuation.
    
    Works for most languages but should be overridden for
    language-specific needs.
    """
    # Split on whitespace
    tokens = text.split()
    
    # Further split on punctuation
    result = []
    for token in tokens:
        # Remove leading/trailing punctuation
        token = token.strip('.,;:!?"()[]{}')
        if token:
            result.append(token)
    
    return result


def basic_tokenize_words(text: str) -> List[str]:
    """
    Extract words only (alphanumeric sequences).
    
    Filters out pure punctuation tokens.
    """
    tokens = basic_tokenize(text)
    # Keep only tokens with at least one alphanumeric character
    return [t for t in tokens if any(c.isalnum() for c in t)]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace - collapse multiple spaces"""
    return ' '.join(text.split())


def remove_punctuation(text: str, keep_chars: str = '') -> str:
    """
    Remove punctuation from text.
    
    Args:
        text: Input text
        keep_chars: Characters to keep (e.g., '-' for hyphenated words)
    """
    punct = '.,;:!?"()[]{}\'`'
    for c in punct:
        if c not in keep_chars:
            text = text.replace(c, ' ')
    return normalize_whitespace(text)