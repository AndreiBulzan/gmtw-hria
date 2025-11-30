"""
Diacritic Analyzer for Romanian

Analyzes Romanian text for correct diacritic usage using a lexicon-based approach.
This is fully deterministic - no ML models, no external API calls.

The analyzer checks:
1. Are diacritics present where they should be?
2. Are they correct (not just present but correct characters)?

Key insight: Romanian diacritics are not optional stylistic choices - they
change meaning. "și" (and) vs "si" (if read as English, meaningless in RO),
"fără" (without) vs "fara" (not a word).
"""

from dataclasses import dataclass
from .tokenizer import tokenize_words, strip_diacritics, normalize_diacritics, has_romanian_diacritics
from .lexicon import MUST_HAVE_DIACRITICS, DIACRITIC_WORDS


@dataclass
class DiacriticAnalysis:
    """Results of diacritic analysis"""
    score: float                    # 0.0-1.0, where 1.0 = perfect diacritic usage
    total_checkable: int            # Words we could check (in lexicon)
    correct_diacritics: int         # Words with correct diacritics
    missing_diacritics: int         # Words missing required diacritics
    has_any_diacritics: bool        # Whether text has any diacritics at all
    missing_words: list[str]        # Sample of words with missing diacritics
    details: dict                   # Additional details


def analyze_diacritics(text: str) -> DiacriticAnalysis:
    """
    Analyze diacritic usage in Romanian text.

    The algorithm:
    1. Tokenize text into words
    2. For each word, check if it should have diacritics (via lexicon lookup)
    3. If word is in MUST_HAVE_DIACRITICS and appears without them → missing
    4. If word appears with correct diacritics → correct
    5. Score = correct / (correct + missing)

    Args:
        text: Romanian text to analyze

    Returns:
        DiacriticAnalysis with score and details
    """
    # Normalize cedilla variants
    normalized_text = normalize_diacritics(text)

    # Check if text has any diacritics at all
    has_diacritics = has_romanian_diacritics(normalized_text)

    # Get word tokens
    words = tokenize_words(normalized_text)

    if not words:
        return DiacriticAnalysis(
            score=1.0,  # Empty text is "correct" by default
            total_checkable=0,
            correct_diacritics=0,
            missing_diacritics=0,
            has_any_diacritics=False,
            missing_words=[],
            details={"note": "No words to analyze"}
        )

    correct_count = 0
    missing_count = 0
    missing_examples = []

    # Track which words we've seen (avoid double-counting)
    seen_stripped = set()

    for word in words:
        # Get stripped (ASCII) form
        stripped = strip_diacritics(word)

        # Check if this word MUST have diacritics
        if stripped in MUST_HAVE_DIACRITICS:
            expected = MUST_HAVE_DIACRITICS[stripped]

            if word == expected:
                # Correct diacritic usage
                correct_count += 1
            elif word == stripped:
                # Word appears without diacritics but should have them
                missing_count += 1
                if stripped not in seen_stripped and len(missing_examples) < 10:
                    missing_examples.append(f"{word} → {expected}")
                    seen_stripped.add(stripped)
            else:
                # Word has some diacritics but maybe not all correct
                # Check if it matches any valid form in DIACRITIC_WORDS
                if stripped in DIACRITIC_WORDS:
                    valid_forms = DIACRITIC_WORDS[stripped]
                    if word in valid_forms:
                        correct_count += 1
                    else:
                        # Has diacritics but wrong ones
                        missing_count += 1
                        if stripped not in seen_stripped and len(missing_examples) < 10:
                            missing_examples.append(f"{word} → {expected}")
                            seen_stripped.add(stripped)
                else:
                    # Assume correct if has some diacritics
                    correct_count += 1

        # Also check broader DIACRITIC_WORDS lexicon
        elif stripped in DIACRITIC_WORDS:
            valid_forms = DIACRITIC_WORDS[stripped]
            if word in valid_forms:
                correct_count += 1
            elif word == stripped and valid_forms != {stripped}:
                # Word is in ASCII but should have diacritics
                # (unless the ASCII form itself is valid)
                missing_count += 1
                if stripped not in seen_stripped and len(missing_examples) < 10:
                    example_form = next(iter(valid_forms))
                    missing_examples.append(f"{word} → {example_form}")
                    seen_stripped.add(stripped)
            elif word != stripped:
                # Word has diacritics but WRONG ones (e.g., "căsă" instead of "casă")
                missing_count += 1
                if stripped not in seen_stripped and len(missing_examples) < 10:
                    example_form = next(iter(valid_forms - {stripped}), next(iter(valid_forms)))
                    missing_examples.append(f"{word} → {example_form}")
                    seen_stripped.add(stripped)

    total_checkable = correct_count + missing_count

    # Compute score
    if total_checkable == 0:
        # No checkable words - use presence heuristic for long texts
        if len(words) > 20 and not has_diacritics:
            # Long Romanian text with zero diacritics is suspicious
            score = 0.5
        else:
            score = 1.0
    else:
        score = correct_count / total_checkable

    return DiacriticAnalysis(
        score=score,
        total_checkable=total_checkable,
        correct_diacritics=correct_count,
        missing_diacritics=missing_count,
        has_any_diacritics=has_diacritics,
        missing_words=missing_examples,
        details={
            "total_words": len(words),
            "unique_words": len(set(words)),
            "coverage": total_checkable / len(words) if words else 0,
        }
    )


def quick_diacritic_check(text: str) -> float:
    """
    Quick heuristic check for diacritic presence.

    This is a fast check that doesn't do full lexicon lookup.
    Useful for preliminary filtering.

    Args:
        text: Romanian text

    Returns:
        Score from 0.0 to 1.0
    """
    words = tokenize_words(text)
    if not words:
        return 1.0

    # Check presence of common must-have-diacritics words
    critical_words = {"si", "in", "sa", "ca", "la"}  # Often appear without diacritics
    found_critical_stripped = 0
    found_critical_correct = 0

    for word in words:
        stripped = strip_diacritics(word)
        if stripped in critical_words:
            if stripped in MUST_HAVE_DIACRITICS:
                expected = MUST_HAVE_DIACRITICS[stripped]
                if word == expected:
                    found_critical_correct += 1
                elif word == stripped:
                    found_critical_stripped += 1

    total = found_critical_correct + found_critical_stripped
    if total == 0:
        # No critical words found - check if ANY diacritics present
        if len(words) > 30 and not has_romanian_diacritics(text):
            return 0.6  # Suspicious
        return 1.0

    return found_critical_correct / total
