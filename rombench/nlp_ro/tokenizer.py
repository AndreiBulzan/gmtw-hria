"""
Simple Romanian Tokenizer

A regex-based tokenizer that handles Romanian text,
preserving diacritics and handling common punctuation.
"""

import re
from dataclasses import dataclass


@dataclass
class Token:
    """A single token with its metadata"""
    text: str           # Original text
    lower: str          # Lowercased
    start: int          # Start position in original text
    end: int            # End position in original text
    is_word: bool       # True if alphabetic word (not punctuation/number)


# Pattern for word tokens (including Romanian diacritics)
# Romanian letters: a-z plus ăâîșț (and their uppercase)
WORD_PATTERN = re.compile(
    r"[a-zA-ZăâîșțĂÂÎȘȚ]+(?:[-'][a-zA-ZăâîșțĂÂÎȘȚ]+)*",
    re.UNICODE
)

# Pattern for any token (words, numbers, punctuation)
TOKEN_PATTERN = re.compile(
    r"[a-zA-ZăâîșțĂÂÎȘȚ]+(?:[-'][a-zA-ZăâîșțĂÂÎȘȚ]+)*"  # Words with optional hyphen/apostrophe
    r"|[0-9]+(?:[.,][0-9]+)*"  # Numbers with optional decimals
    r"|[^\s]",  # Any other non-whitespace character
    re.UNICODE
)


def tokenize(text: str) -> list[Token]:
    """
    Tokenize Romanian text into a list of tokens.

    Args:
        text: Input text string

    Returns:
        List of Token objects
    """
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        token_text = match.group()
        is_word = bool(WORD_PATTERN.fullmatch(token_text))
        tokens.append(Token(
            text=token_text,
            lower=token_text.lower(),
            start=match.start(),
            end=match.end(),
            is_word=is_word
        ))
    return tokens


def tokenize_words(text: str) -> list[str]:
    """
    Extract only word tokens (lowercase) from text.

    Args:
        text: Input text string

    Returns:
        List of lowercase word strings
    """
    return [tok.lower for tok in tokenize(text) if tok.is_word]


def strip_diacritics(text: str) -> str:
    """
    Remove Romanian diacritics from text.

    Args:
        text: Input text with potential diacritics

    Returns:
        Text with diacritics replaced by ASCII equivalents
    """
    replacements = {
        'ă': 'a', 'Ă': 'A',
        'â': 'a', 'Â': 'A',
        'î': 'i', 'Î': 'I',
        'ș': 's', 'Ș': 'S',
        'ț': 't', 'Ț': 'T',
        # Also handle cedilla variants (incorrect but common)
        'ş': 's', 'Ş': 'S',
        'ţ': 't', 'Ţ': 'T',
    }
    result = text
    for diacritic, replacement in replacements.items():
        result = result.replace(diacritic, replacement)
    return result


def normalize_diacritics(text: str) -> str:
    """
    Normalize diacritic variants (cedilla -> comma-below).

    Romanian uses comma-below (ș, ț) not cedilla (ş, ţ),
    but many fonts/systems incorrectly use cedilla.

    Args:
        text: Input text

    Returns:
        Text with normalized diacritics
    """
    replacements = {
        'ş': 'ș', 'Ş': 'Ș',
        'ţ': 'ț', 'Ţ': 'Ț',
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def has_romanian_diacritics(text: str) -> bool:
    """
    Check if text contains any Romanian diacritics.

    Args:
        text: Input text

    Returns:
        True if any Romanian diacritics are present
    """
    diacritics = set('ăâîșțĂÂÎȘȚşţŞŢ')
    return any(c in diacritics for c in text)


def count_diacritics(text: str) -> dict[str, int]:
    """
    Count occurrences of each Romanian diacritic.

    Args:
        text: Input text

    Returns:
        Dictionary mapping diacritic character to count
    """
    diacritics = 'ăâîșț'
    # Normalize cedilla variants first
    normalized = normalize_diacritics(text.lower())
    return {d: normalized.count(d) for d in diacritics}
