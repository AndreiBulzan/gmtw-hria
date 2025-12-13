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

    # Additional English-only words (definitely not Romanian)
    # Common verbs with no Romanian cognate
    "said", "says", "saying", "asked", "asking", "told", "telling",
    "looked", "looking", "walked", "walking", "talked", "talking",
    "started", "starting", "stopped", "stopping", "helped", "helping",
    "needed", "needing", "wanted", "wanting", "liked", "liking",
    "loved", "loving", "hated", "hating", "hoped", "hoping",
    "believed", "believing", "understood", "understanding",
    "remembered", "remembering", "forgot", "forgotten", "forgetting",
    "learned", "learning", "taught", "teaching",
    "bought", "buying", "sold", "selling", "paid", "paying",
    "sent", "sending", "received", "receiving",
    "written", "writing", "read", "reading",  # "read" as past tense
    "shown", "showing", "given", "giving", "taken", "taking",
    "brought", "bringing", "kept", "keeping", "let", "letting",
    "heard", "hearing", "meant", "meaning",
    "became", "becoming", "stood", "standing", "sat", "sitting",
    "ran", "running", "held", "holding", "turned", "turning",
    "moved", "moving", "lived", "living", "died", "dying",
    "opened", "opening", "closed", "closing",
    "played", "playing", "watched", "watching",
    "stayed", "staying", "waited", "waiting",
    "happened", "happening", "seemed", "seeming",
    "changed", "changing", "followed", "following",
    "met", "meeting", "led", "leading",
    "lost", "losing", "won", "winning",
    "built", "building", "cut", "cutting",
    "spoke", "speaking", "written",

    # Common English nouns with no Romanian cognate
    "something", "nothing", "everything", "anything",
    "someone", "anyone", "everyone", "nobody", "somebody", "anybody", "everybody",
    "somewhere", "anywhere", "everywhere", "nowhere",
    "sometimes", "always",  # already have but ensuring
    "maybe", "perhaps", "probably", "certainly", "definitely",
    "please", "thanks", "sorry", "hello", "goodbye", "okay",
    "today", "tomorrow", "yesterday",
    "morning", "afternoon", "evening", "tonight",
    "week", "month",  # already have some
    "people", "person", "friend", "friends",
    "job", "jobs", "house", "houses", "car", "cars",
    "school", "teacher", "teachers", "student", "students",
    "money", "price", "cost", "costs",
    "food", "drink", "drinks", "meal", "meals",
    "story", "stories", "news", "game", "games",
    "movie", "movies", "book", "books", "song", "songs",
    "city", "cities", "town", "towns", "street", "streets",
    "door", "doors", "window", "windows", "floor", "floors",
    "side", "sides", "top", "bottom", "front", "middle",
    "end", "ends", "beginning", "start",
    "kind", "kinds", "type", "types", "sort", "sorts",
    "set", "sets", "piece", "pieces", "bit", "bits",
    "lot", "lots", "deal", "deals",
    "issue", "issues", "matter", "matters",
    "power", "powers", "force", "forces",
    "light", "lights", "sound", "sounds",
    "picture", "pictures", "view", "views",
    "word", "words", "letter", "letters",
    "page", "pages", "line", "lines",
    "step", "steps", "move", "moves",
    "change", "changes", "difference", "differences",
    "answer", "answers", "reason", "reasons",
    "sense", "senses", "feeling", "feelings",
    "mind", "minds", "heart", "hearts", "body", "bodies",
    "eye", "eyes", "face", "faces", "head", "heads",
    "hair", "foot", "feet", "arm", "arms", "leg", "legs",
    "blood", "skin", "bone", "bones",
    "air", "fire", "earth", "sky", "sea", "sun", "moon", "star", "stars",
    "tree", "trees", "flower", "flowers", "grass",
    "dog", "dogs", "cat", "cats", "bird", "birds", "fish",
    "horse", "horses", "cow", "cows", "sheep",

    # English adjectives with no Romanian cognate
    "happy", "sad", "angry", "afraid", "sorry", "glad", "proud",
    "tired", "sick", "hungry", "thirsty", "busy", "ready", "quick", "slow",
    "hard", "soft", "hot", "cold", "warm", "cool", "wet", "dry",
    "clean", "dirty", "empty", "full",
    "dark", "bright", "deep", "wide", "narrow", "thick", "thin",
    "cheap", "expensive", "rich", "poor",
    "strong", "weak", "heavy", "light",
    "beautiful", "ugly", "pretty", "handsome",
    "smart", "stupid", "clever", "wise", "crazy", "strange", "weird",
    "nice", "kind", "mean", "funny", "serious",
    "easy", "difficult", "simple", "hard",
    "safe", "dangerous", "healthy", "ill",
    "alive", "dead", "awake", "asleep",
    "alone", "together", "single", "double",
    "whole", "half", "main", "major", "minor",
    "wrong", "fair", "unfair",
    "available", "possible", "impossible", "necessary", "likely", "unlikely",
    "common", "rare", "usual", "unusual", "typical", "obvious", "clear",

    # English adverbs
    "quickly", "slowly", "easily", "hardly", "nearly", "mostly", "mainly",
    "simply", "clearly", "obviously", "certainly", "probably", "possibly",
    "suddenly", "immediately", "finally", "eventually", "recently", "lately",
    "usually", "often", "sometimes", "rarely", "seldom",
    "anyway", "somehow", "somewhat", "otherwise", "anywhere", "everywhere",
    "indeed", "instead", "besides", "meanwhile", "therefore", "thus",
    "forward", "backward", "upward", "downward", "inward", "outward",
    "tonight", "nowadays", "forever", "ago",
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

    # ==========================================================================
    # Romanian words identical to English (cognates/loanwords - valid Romanian)
    # These are commonly used Romanian words that happen to be spelled the same
    # as English words. They should NOT be flagged as code-switching.
    # ==========================================================================

    # Common adjectives (Romanian = English spelling)
    "important",    # RO: important (same meaning)
    "special",      # RO: special (same meaning)
    "normal",       # RO: normal
    "modern",       # RO: modern
    "original",     # RO: original
    "natural",      # RO: natural
    "general",      # RO: general
    "central",      # RO: central
    "cultural",     # RO: cultural
    "actual",       # RO: actual (means "current" in RO)
    "final",        # RO: final
    "local",        # RO: local
    "total",        # RO: total
    "principal",    # RO: principal (main)
    "ideal",        # RO: ideal
    "real",         # RO: real
    "formal",       # RO: formal
    "popular",      # RO: popular
    "similar",      # RO: similar
    "familiar",     # RO: familiar
    "particular",   # RO: particular
    "regulat",      # RO: regular (slightly different spelling)
    "international", # RO: internațional (close enough)
    "national",     # RO: național
    "regional",     # RO: regional
    "traditional",  # RO: tradițional
    "personal",     # RO: personal
    "professional", # RO: profesional
    "social",       # RO: social
    "medical",      # RO: medical
    "legal",        # RO: legal
    "moral",        # RO: moral
    "mental",       # RO: mental
    "vocal",        # RO: vocal
    "vital",        # RO: vital
    "fatal",        # RO: fatal
    "brutal",       # RO: brutal
    "rural",        # RO: rural
    "urban",        # RO: urban

    # Common nouns (Romanian = English spelling)
    "program",      # RO: program
    "moment",       # RO: moment
    "content",      # RO: content (satisfied - adj)
    "plan",         # RO: plan
    "transport",    # RO: transport
    "restaurant",   # RO: restaurant
    "monument",     # RO: monument
    "element",      # RO: element
    "segment",      # RO: segment
    "document",     # RO: document
    "argument",     # RO: argument
    "instrument",   # RO: instrument
    "experiment",   # RO: experiment
    "apartament",   # RO: apartament
    "departament",  # RO: departament
    "sentiment",    # RO: sentiment
    "hotel",        # RO: hotel
    "model",        # RO: model
    "nivel",        # RO: nivel (level)
    "material",     # RO: material
    "animal",       # RO: animal
    "canal",        # RO: canal
    "festival",     # RO: festival
    "capital",      # RO: capital
    "potential",    # RO: potențial
    "serial",       # RO: serial
    "ritual",       # RO: ritual
    "manual",       # RO: manual
    "jurnal",       # RO: jurnal (journal)
    "tribunal",     # RO: tribunal
    "minister",     # RO: ministru (close)
    "profesor",     # RO: profesor
    "doctor",       # RO: doctor
    "director",     # RO: director
    "sector",       # RO: sector
    "factor",       # RO: factor
    "motor",        # RO: motor
    "autor",        # RO: autor
    "actor",        # RO: actor
    "senator",      # RO: senator
    "spectator",    # RO: spectator
    "calculator",   # RO: calculator (computer)
    "indicator",    # RO: indicator

    # Time-related
    "an",           # RO: an (year) - NOT English "an" article!
    "moment",       # RO: moment

    # Other common words
    "roman",        # RO: roman (novel) or român (Romanian)
    "fond",         # RO: fond (fund/background)
    "conflict",     # RO: conflict
    "contact",      # RO: contact
    "contract",     # RO: contract
    "impact",       # RO: impact
    "aspect",       # RO: aspect
    "respect",      # RO: respect
    "concept",      # RO: concept
    "context",      # RO: context
    "text",         # RO: text
    "pretext",      # RO: pretext
    "reflex",       # RO: reflex
    "complex",      # RO: complex
    "index",        # RO: index
    "prefix",       # RO: prefix
    "sufix",        # RO: sufix
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
