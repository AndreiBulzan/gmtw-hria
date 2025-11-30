"""
Code-Switch Detector for Romanian

Detects English tokens appearing in Romanian text (code-switching).
This indicates the model is mixing languages, which is a quality issue.

The detector uses:
1. A curated list of common English function words (deterministic)
2. A whitelist of acceptable loanwords/technical terms
3. Simple heuristics to avoid false positives

For full language identification, lingua-language-detector is available
but we keep the core metric deterministic and lightweight.
"""

from dataclasses import dataclass
from .tokenizer import tokenize_words, strip_diacritics
from .lexicon import ENGLISH_STOPWORDS, ENGLISH_WHITELIST


@dataclass
class CodeSwitchAnalysis:
    """Results of code-switch detection"""
    score: float                    # 0.0-1.0, where 1.0 = no code-switching
    total_words: int                # Total word tokens analyzed
    english_words: int              # English words detected
    english_rate: float             # english_words / total_words
    flagged_words: list[str]        # Sample of detected English words
    details: dict                   # Additional details


# Additional high-confidence English words not in ENGLISH_STOPWORDS
# These are very unlikely to appear in Romanian naturally
HIGH_CONFIDENCE_ENGLISH: set[str] = {
    # Pronouns/determiners that differ from Romanian
    "i", "me", "myself", "you", "yourself", "he", "him", "himself",
    "she", "her", "herself", "it", "itself", "we", "us", "ourselves",
    "they", "them", "themselves", "mine", "yours", "his", "hers", "ours", "theirs",

    # Common English verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "get", "got", "getting", "make", "made", "making",
    "go", "going", "went", "gone", "come", "coming", "came",
    "see", "saw", "seen", "seeing", "know", "knew", "known", "knowing",
    "think", "thought", "thinking", "want", "wanted", "wanting",
    "use", "used", "using", "find", "found", "finding",
    "give", "gave", "given", "giving", "tell", "told", "telling",
    "work", "worked", "working", "seem", "seemed", "seeming",
    "feel", "felt", "feeling", "try", "tried", "trying",
    "leave", "left", "leaving", "call", "called", "calling",

    # Common English nouns
    "thing", "things", "time", "times", "year", "years", "way", "ways",
    "day", "days", "man", "men", "woman", "women", "child", "children",
    "world", "life", "hand", "hands", "part", "parts", "place", "places",
    "case", "cases", "week", "weeks", "company", "system", "program",
    "question", "questions", "government", "number", "numbers",
    "night", "nights", "point", "points", "home", "water", "room", "rooms",
    "mother", "father", "family", "student", "students", "group", "groups",
    "country", "countries", "problem", "problems", "fact", "facts",

    # Common English adjectives
    "good", "new", "first", "last", "long", "great", "little", "own",
    "other", "old", "right", "big", "high", "different", "small", "large",
    "next", "early", "young", "important", "few", "public", "bad", "same",
    "able", "better", "best", "sure", "free", "true", "full", "special",

    # Common English adverbs
    "also", "just", "only", "now", "then", "more", "very", "well",
    "even", "back", "still", "here", "there", "again", "never", "always",
    "often", "however", "actually", "really", "probably", "maybe",
    "perhaps", "already", "almost", "enough", "quite", "rather",

    # Conjunctions and prepositions
    "and", "or", "but", "if", "because", "although", "while", "when",
    "where", "what", "which", "who", "how", "why", "whether",
    "about", "after", "before", "between", "through", "during", "without",
    "within", "along", "following", "across", "behind", "beyond",

    # Articles (very strong signal)
    "the", "a", "an",
}

# Romanian words that look like English (false positives to avoid)
ROMANIAN_LOOKALIKES: set[str] = {
    # Romanian words that happen to match English words
    "nu",       # "no" in Romanian (not English "nu")
    "de",       # preposition
    "pe",       # preposition
    "care",     # "which/who"
    "este",     # "is" (but not English "is")
    "sunt",     # "are/am"
    "era",      # "was"
    "avea",     # "had"
    "face",     # "does/makes" (not English "face")
    "vine",     # "comes"
    "merge",    # "goes"
    "place",    # "likes" (not English "place")
    "vor",      # "will"
    "fi",       # "be"
    "din",      # "from"
    "cu",       # "with"
    "la",       # "at/to"
    "prin",     # "through"
    "spre",     # "toward"
    "sub",      # "under"
    "asupra",   # "upon"
    "mare",     # "big/sea"
    "mic",      # "small"
    "bun",      # "good"
    "nou",      # "new"
    "alt",      # "other"
    "tot",      # "all"
    "ori",      # "times/or"
    "am",       # "have" (1st person)
    "ai",       # "have" (2nd person)
    "are",      # "has"
    "au",       # "have" (3rd plural)
    "e",        # short for "este"
    "a",        # aux verb / prep
    "o",        # "a" (feminine article)
    "un",       # "a" (masculine article)
    "i",        # dative clitic
    "le",       # accusative clitic
    "se",       # reflexive
    "ne",       # reflexive (us)
    "te",       # accusative "you"
    "ma",       # accusative "me" (no diacritic sometimes)
    "tu",       # "you"
    "el",       # "he"
    "ea",       # "she"
    "noi",      # "we"
    "voi",      # "you" plural
    "ei",       # "they" (masc)
    "ele",      # "they" (fem)
    "ca",       # "as/that"
    "sa",       # subjunctive "să"
    "si",       # "and" (și without diacritic)
    "in",       # "in" (în without diacritic)
    "ce",       # "what"
    "cum",      # "how"
    "cine",     # "who"
    "mai",      # "more/still"
    "dar",      # "but"
    "sau",      # "or"
    "nici",     # "neither"
    "deci",     # "so/therefore"
    "doar",     # "only"
    "chiar",    # "even"
    "fost",     # "been"
    "parte",    # "part"
    "pentru",   # "for"
    "foarte",   # "very"
    "toate",    # "all"
    "acum",     # "now"
    "atunci",   # "then"
    "aici",     # "here"
    "acolo",    # "there"
    "fost",     # "was/been"
    "zile",     # "days"
    "timp",     # "time"
    "ani",      # "years"
    "mod",      # "mode/way"
    "tip",      # "type"
}


def detect_code_switching(text: str) -> CodeSwitchAnalysis:
    """
    Detect English code-switching in Romanian text.

    The algorithm:
    1. Tokenize text into words
    2. For each word, check if it's a high-confidence English word
    3. Exclude Romanian lookalikes and acceptable loanwords
    4. Compute rate and score

    Args:
        text: Romanian text to analyze

    Returns:
        CodeSwitchAnalysis with score and details
    """
    words = tokenize_words(text)

    if not words:
        return CodeSwitchAnalysis(
            score=1.0,
            total_words=0,
            english_words=0,
            english_rate=0.0,
            flagged_words=[],
            details={"note": "No words to analyze"}
        )

    english_count = 0
    flagged = []

    for word in words:
        word_lower = word.lower()
        # Also check stripped version (in case diacritics added)
        word_stripped = strip_diacritics(word_lower)

        # Skip Romanian lookalikes
        if word_lower in ROMANIAN_LOOKALIKES or word_stripped in ROMANIAN_LOOKALIKES:
            continue

        # Skip whitelisted loanwords
        if word_lower in ENGLISH_WHITELIST or word_stripped in ENGLISH_WHITELIST:
            continue

        # Check if it's a high-confidence English word
        if word_lower in HIGH_CONFIDENCE_ENGLISH or word_lower in ENGLISH_STOPWORDS:
            english_count += 1
            if len(flagged) < 10:  # Keep sample
                flagged.append(word)

    total = len(words)
    english_rate = english_count / total if total > 0 else 0.0

    # Score: penalize code-switching exponentially
    # Small amounts (< 1%) are tolerable, but more is bad
    # score = exp(-k * rate) where k controls sensitivity
    import math
    score = math.exp(-20 * english_rate)  # At 5% English, score ~ 0.37

    return CodeSwitchAnalysis(
        score=score,
        total_words=total,
        english_words=english_count,
        english_rate=english_rate,
        flagged_words=flagged,
        details={
            "unique_words": len(set(words)),
            "threshold_used": 0.05,  # 5% would be severe
        }
    )


def is_likely_english_text(text: str, threshold: float = 0.15) -> bool:
    """
    Quick check if text is predominantly English.

    Useful to detect if model responded in wrong language entirely.

    Args:
        text: Text to check
        threshold: English word rate threshold (default 15%)

    Returns:
        True if text appears to be English
    """
    analysis = detect_code_switching(text)
    return analysis.english_rate > threshold
