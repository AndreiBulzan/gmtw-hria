"""
Punctuation Quality Checker for Romanian

Checks for common punctuation errors:
1. Space BEFORE punctuation (wrong: "text ." → correct: "text.")
2. Missing space AFTER punctuation (wrong: "text.Next" → correct: "text. Next")
3. Double/multiple spaces
4. Improper quote usage

All checks are deterministic and language-agnostic but tuned for Romanian conventions.
"""

import re
from dataclasses import dataclass


@dataclass
class PunctuationAnalysis:
    """Results of punctuation quality analysis"""
    score: float                    # 0.0-1.0, where 1.0 = perfect punctuation
    total_issues: int               # Total punctuation issues found
    space_before_punct: int         # Spaces before . , ; : ! ?
    missing_space_after: int        # Missing spaces after punctuation
    double_spaces: int              # Multiple consecutive spaces
    other_issues: int               # Other punctuation problems
    examples: list[str]             # Sample issues found
    details: dict                   # Additional details


# Punctuation that should NOT have space before it
NO_SPACE_BEFORE = {'.', ',', ';', ':', '!', '?', ')', ']', '}', '"', '"', '»'}

# Punctuation that should have space after it (if followed by text)
SPACE_AFTER = {'.', ',', ';', ':', '!', '?'}


def analyze_punctuation(text: str) -> PunctuationAnalysis:
    """
    Analyze punctuation quality in text.

    Args:
        text: Text to analyze

    Returns:
        PunctuationAnalysis with score and details
    """
    if not text or not text.strip():
        return PunctuationAnalysis(
            score=1.0,
            total_issues=0,
            space_before_punct=0,
            missing_space_after=0,
            double_spaces=0,
            other_issues=0,
            examples=[],
            details={"note": "No text to analyze"}
        )

    issues = []
    space_before_punct = 0
    missing_space_after = 0
    double_spaces = 0
    other_issues = 0

    # Check for space before punctuation
    # Pattern: word + space + punctuation
    space_before_pattern = re.compile(r'\w\s+([.,;:!?])')
    for match in space_before_pattern.finditer(text):
        space_before_punct += 1
        if len(issues) < 10:
            start = max(0, match.start() - 5)
            end = min(len(text), match.end() + 5)
            context = text[start:end].replace('\n', ' ')
            issues.append(f"Space before '{match.group(1)}': ...{context}...")

    # Check for missing space after punctuation
    # Pattern: punctuation + letter (no space)
    # Exclude URLs, numbers (like 3.14), abbreviations
    missing_after_pattern = re.compile(r'([.,;:!?])([A-ZĂÂÎȘȚ])')  # Only flag if next char is uppercase
    for match in missing_after_pattern.finditer(text):
        # Skip if it looks like a decimal number
        if match.group(1) == '.' and match.start() > 0:
            prev_char = text[match.start() - 1]
            if prev_char.isdigit():
                continue
        missing_space_after += 1
        if len(issues) < 10:
            start = max(0, match.start() - 5)
            end = min(len(text), match.end() + 5)
            context = text[start:end].replace('\n', ' ')
            issues.append(f"Missing space after '{match.group(1)}': ...{context}...")

    # Check for double/multiple spaces
    double_space_pattern = re.compile(r'  +')
    double_spaces = len(double_space_pattern.findall(text))
    if double_spaces > 0 and len(issues) < 10:
        issues.append(f"Found {double_spaces} instances of multiple consecutive spaces")

    # Check for space after opening brackets/quotes (less severe)
    space_after_open = len(re.findall(r'[([\[{„«]\s+\w', text))
    if space_after_open > 0:
        other_issues += space_after_open

    # Check for space before closing brackets/quotes (less severe)
    space_before_close = len(re.findall(r'\w\s+[)\]}"»]', text))
    if space_before_close > 0:
        other_issues += space_before_close

    total_issues = space_before_punct + missing_space_after + double_spaces + other_issues

    # Calculate score
    # Each issue reduces score, but we cap the penalty
    if total_issues == 0:
        score = 1.0
    else:
        # Penalize based on issue density
        text_length = len(text.split())
        issue_rate = total_issues / max(text_length, 1)
        # Score decreases with more issues, but floor at 0.3
        score = max(0.3, 1.0 - (issue_rate * 5))

    return PunctuationAnalysis(
        score=score,
        total_issues=total_issues,
        space_before_punct=space_before_punct,
        missing_space_after=missing_space_after,
        double_spaces=double_spaces,
        other_issues=other_issues,
        examples=issues,
        details={
            "text_length_words": len(text.split()),
            "issue_rate": total_issues / max(len(text.split()), 1),
        }
    )


def analyze_repetition(text: str) -> dict:
    """
    Analyze text for repetition and lexical diversity.

    Returns:
        Dictionary with:
        - score: 0.0-1.0 (1.0 = good variety)
        - type_token_ratio: vocabulary diversity
        - repeated_phrases: list of repeated 3+ word sequences
    """
    words = re.findall(r'\b\w+\b', text.lower())

    if len(words) < 10:
        return {
            "score": 1.0,
            "type_token_ratio": 1.0,
            "repeated_phrases": [],
            "note": "Text too short for repetition analysis"
        }

    # Type-Token Ratio (vocabulary diversity)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    # Find repeated phrases (3-grams that appear multiple times)
    repeated_phrases = []
    trigrams = {}
    for i in range(len(words) - 2):
        trigram = ' '.join(words[i:i+3])
        trigrams[trigram] = trigrams.get(trigram, 0) + 1

    for phrase, count in trigrams.items():
        if count >= 3:  # Appears 3+ times
            repeated_phrases.append(f"'{phrase}' (x{count})")

    # Score based on TTR and repeated phrases
    # TTR below 0.4 is low diversity
    ttr_score = min(1.0, ttr / 0.5)  # Normalize: 0.5 TTR = 1.0 score

    # Penalize for excessive repeated phrases
    repetition_penalty = min(0.3, len(repeated_phrases) * 0.05)

    score = max(0.3, ttr_score - repetition_penalty)

    return {
        "score": score,
        "type_token_ratio": round(ttr, 3),
        "repeated_phrases": repeated_phrases[:5],  # Limit examples
        "unique_words": len(unique_words),
        "total_words": len(words),
    }


def analyze_capitalization(text: str) -> dict:
    """
    Analyze capitalization quality.

    Checks:
    - Sentences start with capital letters
    - No random ALL CAPS words (except acronyms)

    Returns:
        Dictionary with score and details
    """
    if not text.strip():
        return {"score": 1.0, "issues": []}

    issues = []

    # Check for sentences not starting with capital
    # Pattern: sentence-ending punctuation + space + lowercase
    lowercase_starts = re.findall(r'[.!?]\s+([a-zăâîșț])', text)
    for char in lowercase_starts:
        if len(issues) < 5:
            issues.append(f"Sentence starts with lowercase: '{char}...'")

    # Check for random ALL CAPS words (not acronyms)
    words = text.split()
    caps_words = []
    for word in words:
        # Remove punctuation
        clean = re.sub(r'[^\w]', '', word)
        if len(clean) > 3 and clean.isupper() and clean.isalpha():
            caps_words.append(word)

    if caps_words:
        issues.append(f"ALL CAPS words: {', '.join(caps_words[:3])}")

    # Score
    total_issues = len(lowercase_starts) + len(caps_words)
    if total_issues == 0:
        score = 1.0
    else:
        score = max(0.5, 1.0 - (total_issues * 0.1))

    return {
        "score": score,
        "lowercase_sentence_starts": len(lowercase_starts),
        "all_caps_words": len(caps_words),
        "issues": issues,
    }
