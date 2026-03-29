"""
German NLP Toolkit

Implements German text quality analysis for G score.

German-specific quality signals:
- Umlaut correctness (ä, ö, ü, ß vs ae, oe, ue, ss)
- Noun capitalization (German capitalizes all nouns)
- Sentence structure and punctuation
- Text length adequacy
- Code-switching detection (unnecessary English in German text)
"""

from typing import Optional, Any
from dataclasses import dataclass

from rombench.nlp import BaseNLPToolkit, TextQualityReport


# Common German words that should contain umlauts
# Maps incorrect (no umlaut) -> correct (with umlaut) for detection
UMLAUT_WORDS = {
    "muenchen": "münchen", "nuernberg": "nürnberg", "koeln": "köln",
    "duesseldorf": "düsseldorf", "goettingen": "göttingen",
    "wuerzburg": "würzburg", "luebeck": "lübeck", "tuebingen": "tübingen",
    "zuerich": "zürich", "oesterreich": "österreich",
    "ueber": "über", "fuer": "für", "wuerden": "würden",
    "muessen": "müssen", "koennen": "können", "moechte": "möchte",
    "waehrend": "während", "spaeter": "später", "frueh": "früh",
    "glueck": "glück", "bruecke": "brücke", "kuenstler": "künstler",
    "gebaude": "gebäude", "gepaeck": "gepäck",
    "strasse": "straße", "grosse": "große", "heisse": "heiße",
    "schliessen": "schließen", "aussen": "außen", "draussen": "draußen",
    "genuegend": "genügend", "ermaessigung": "ermäßigung",
    "fruehstueck": "frühstück",
    "sehenswuerdigkeit": "sehenswürdigkeit",
    "oeffentlich": "öffentlich",
}

# Common English words that shouldn't appear in German text
# (excluding legitimate loanwords like Computer, Manager, etc.)
ENGLISH_ONLY_WORDS = {
    "the", "is", "are", "were", "have", "has", "had",
    "would", "should", "could", "can", "may", "might",
    "this", "that", "these", "those", "which", "where", "when",
    "because", "although", "however", "therefore", "furthermore",
    "beautiful", "wonderful", "amazing", "excellent", "great",
    "important", "interesting", "different", "several",
    "actually", "really", "very", "just", "well",
    "about", "after", "before", "between", "during", "through",
    "visit", "visited", "recommend", "recommended",
    "enjoy", "enjoyed", "experience", "experienced",
}

# German loanwords from English (acceptable in German text)
GERMAN_LOANWORDS = {
    "computer", "internet", "software", "hardware", "online",
    "email", "website", "app", "smartphone", "tablet",
    "manager", "marketing", "meeting", "team", "office",
    "design", "style", "trend", "cool", "hip",
    "restaurant", "hotel", "bar", "café", "check-in",
    "ticket", "tour", "guide", "shuttle", "transfer",
}

# German words that look like English but are legitimate German.
# Must be excluded from ENGLISH_ONLY_WORDS before flagging.
GERMAN_EXCLUSION_WORDS = {
    "also",    # German: so / therefore / well
    "was",     # German: what
    "will",    # German: wants (to)
    "weil",    # German: because
    "still",   # German: quiet
    "art",     # German: kind / type / manner
    "rat",     # German: council / advice
    "tag",     # German: day
    "hut",     # German: hat
    "rein",    # German: pure / in
    "lager",   # German: warehouse / camp
    "stern",   # German: star
    "hang",    # German: slope
    "hat",     # German: has (3rd person of haben)
    "an",      # German: at / on
    "in",      # German: in
    "so",      # German: so / this way
    "here",    # German: army (Heer, but "here" can appear)
    "her",     # German: towards (direction)
    "die",     # German: the / those
    "wand",    # German: wall
    "bad",     # German: bath
    "fall",    # German: case / fall
    "must",    # German: must (identical meaning)
}


class GermanNLPToolkit(BaseNLPToolkit):
    """
    German NLP toolkit for text quality analysis.

    G score components:
    - Umlaut usage (30%): Proper use of ä, ö, ü, ß
    - Style (40%): Noun capitalization, sentence structure, punctuation
    - Code-switching (15%): Absence of unnecessary English
    - Length (15%): Adequate text length

    Optional grammar checking via LanguageTool (if available).
    """

    MIN_WORDS_REQUIRED = 100

    # Default weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        "umlaut": 0.30,
        "style": 0.40,
        "codeswitch": 0.15,
        "length": 0.15,
    }

    def __init__(self, use_grammar: bool = False):
        self.use_grammar = use_grammar
        self.grammar_checker = None

        if use_grammar:
            try:
                import language_tool_python
                self.grammar_checker = language_tool_python.LanguageTool('de-DE')
            except (ImportError, Exception):
                self.use_grammar = False

    def analyze(self, text: str) -> TextQualityReport:
        """Analyze German text quality."""
        words = self.tokenize(text)
        word_count = len(words)

        # Umlaut score
        umlaut_score = self._compute_umlaut_score(text, words)

        # Style score (capitalization, sentence structure)
        style_score = self._compute_style_score(text, words, word_count)

        # Code-switching score
        codeswitch_score = self._compute_codeswitch_score(words)

        # Length score
        length_score = self._compute_length_score(word_count)

        # Grammar (optional)
        grammar_score = None
        grammar_details = None
        if self.grammar_checker:
            try:
                grammar_result = self._compute_grammar_score(text)
                grammar_score = grammar_result['score']
                grammar_details = grammar_result
            except Exception:
                pass

        # Combined score
        if grammar_score is not None:
            overall_score = (
                0.25 * umlaut_score +
                0.25 * style_score +
                0.10 * codeswitch_score +
                0.15 * length_score +
                0.25 * grammar_score
            )
        else:
            w = self.DEFAULT_WEIGHTS
            overall_score = (
                w["umlaut"] * umlaut_score +
                w["style"] * style_score +
                w["codeswitch"] * codeswitch_score +
                w["length"] * length_score
            )

        return TextQualityReport(
            overall_score=overall_score,
            grammar_score=grammar_score,
            style_score=style_score,
            length_score=length_score,
            language_specific_scores={
                "umlaut_score": umlaut_score,
                "codeswitch_score": codeswitch_score,
            },
            total_words=word_count,
            total_tokens=len(text.split()),
            is_too_short=word_count < 50,
            details={
                'umlaut': {'score': umlaut_score},
                'style': {'score': style_score},
                'codeswitch': {'score': codeswitch_score},
                'length': {'word_count': word_count, 'score': length_score},
                'grammar': grammar_details,
            }
        )

    def compute_g_score(self, text: str) -> dict[str, Any]:
        """Compute G score for evaluation metrics."""
        report = self.analyze(text)

        result = {
            'G': report.overall_score,
            'G_umlaut': report.language_specific_scores.get('umlaut_score', 1.0),
            'G_style': report.style_score if report.style_score else 1.0,
            'G_cs': report.language_specific_scores.get('codeswitch_score', 1.0),
            'G_len': report.length_score,
            'total_words': report.total_words,
        }

        if report.grammar_score is not None:
            result['G_grammar'] = report.grammar_score

        return result

    def tokenize(self, text: str) -> list[str]:
        """Tokenize into words."""
        words = []
        for token in text.split():
            token = token.strip('.,;:!?"()[]{}«»„"')
            if token and any(c.isalnum() for c in token):
                words.append(token)
        return words

    def normalize(self, text: str) -> str:
        """Normalize text for matching."""
        text = text.lower()
        for c in '.,;:!?"()[]{}«»„"':
            text = text.replace(c, ' ')
        return ' '.join(text.split())

    def _compute_umlaut_score(self, text: str, words: list[str]) -> float:
        """
        Check proper use of German umlauts (ä, ö, ü, ß).

        Detects:
        - ae/oe/ue used instead of ä/ö/ü
        - ss used instead of ß (context-dependent)
        - Missing umlauts in known words
        """
        if not words:
            return 1.0

        text_lower = text.lower()

        # Check for umlaut presence
        has_umlauts = any(c in text for c in 'äöüÄÖÜß')

        # Check for substitutions (ae, oe, ue used instead of umlauts)
        substitution_count = 0
        total_checkable = 0

        for word in words:
            w = word.lower()
            # Check known words that should have umlauts
            if w in UMLAUT_WORDS:
                substitution_count += 1
                total_checkable += 1
            elif any(c in w for c in 'äöüß'):
                total_checkable += 1

        # Check for ae/oe/ue patterns
        ae_count = text_lower.count('ae') + text_lower.count('oe') + text_lower.count('ue')
        umlaut_count = sum(text_lower.count(c) for c in 'äöüß')

        if ae_count + umlaut_count == 0:
            # No umlaut-relevant content - neutral score
            return 0.9

        if umlaut_count == 0 and ae_count > 0:
            # Substitutions used instead of umlauts
            return max(0.3, 1.0 - 0.1 * ae_count)

        if substitution_count > 0:
            # Known wrong words found
            return max(0.4, 1.0 - 0.15 * substitution_count)

        # Umlauts present and no obvious substitutions
        return 1.0

    def _compute_style_score(
        self, text: str, words: list[str], word_count: int
    ) -> float:
        """
        German style scoring.

        Checks:
        - Noun capitalization (German capitalizes nouns)
        - Sentence structure
        - Punctuation usage
        """
        if not text or word_count < 10:
            return 0.0

        score = 1.0

        # Check for sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            score *= 0.8

        # Check punctuation
        has_punctuation = any(c in text for c in '.,;:!?')
        if not has_punctuation:
            score *= 0.7

        # Check sentence length variety
        if sentences:
            avg_len = word_count / len(sentences)
            if avg_len < 5:
                score *= 0.8
            elif avg_len > 100:
                score *= 0.9

        # Check for capitalized words (German capitalizes nouns)
        # A good German text should have capitalized words beyond sentence starts
        capitalized = sum(
            1 for w in words
            if w[0].isupper() and len(w) > 1
        )
        cap_ratio = capitalized / max(word_count, 1)

        # German should have ~15-30% capitalized words (nouns + sentence starts)
        if cap_ratio < 0.05:
            score *= 0.7  # Almost nothing capitalized - likely wrong
        elif cap_ratio < 0.10:
            score *= 0.85  # Too few capitals

        return score

    def _compute_codeswitch_score(self, words: list[str]) -> float:
        """
        Detect unnecessary English in German text.

        Allows legitimate loanwords but penalizes English-only words.
        """
        if not words:
            return 1.0

        english_count = 0
        for word in words:
            w = word.lower().strip('.,;:!?"()[]{}')
            if w in ENGLISH_ONLY_WORDS and w not in GERMAN_LOANWORDS and w not in GERMAN_EXCLUSION_WORDS:
                english_count += 1

        if english_count == 0:
            return 1.0

        english_rate = english_count / len(words)

        if english_rate > 0.3:
            return 0.1  # Largely English text
        elif english_rate > 0.15:
            return 0.4
        elif english_rate > 0.05:
            return 0.7
        else:
            return 0.9

    def _compute_grammar_score(self, text: str) -> dict[str, Any]:
        """Compute grammar score using LanguageTool (if available)."""
        if not self.grammar_checker:
            return {'score': 1.0, 'available': False}

        matches = self.grammar_checker.check(text)
        error_count = len(matches)
        word_count = len(text.split())

        if word_count == 0:
            return {'score': 0.0, 'error_count': 0, 'available': True}

        # Error density (errors per 100 words)
        density = (error_count / word_count) * 100

        if density == 0:
            score = 1.0
        elif density < 2:
            score = 0.9
        elif density < 5:
            score = 0.7
        elif density < 10:
            score = 0.5
        else:
            score = 0.3

        return {
            'score': score,
            'error_count': error_count,
            'error_density': density,
            'available': True,
        }