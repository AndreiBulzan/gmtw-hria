"""
Tests for the Romanian NLP Toolkit

Tests cover:
- Tokenization
- Diacritic analysis
- Code-switch detection
- Overall text quality scoring
"""

import pytest
from rombench.nlp_ro import (
    # Tokenizer
    tokenize,
    tokenize_words,
    strip_diacritics,
    normalize_diacritics,
    has_romanian_diacritics,
    # Diacritics
    analyze_diacritics,
    quick_diacritic_check,
    # Code-switch
    detect_code_switching,
    is_likely_english_text,
    # Toolkit
    RomanianNLPToolkit,
    analyze_romanian_text,
)


class TestTokenizer:
    """Tests for tokenizer functions"""

    def test_tokenize_simple(self):
        """Test basic tokenization"""
        tokens = tokenize("Aceasta este o propoziție.")
        assert len(tokens) == 5  # 4 words + 1 punctuation
        assert tokens[0].text == "Aceasta"
        assert tokens[0].is_word is True
        assert tokens[-1].text == "."
        assert tokens[-1].is_word is False

    def test_tokenize_with_diacritics(self):
        """Test tokenization preserves diacritics"""
        tokens = tokenize("România și țară")
        words = [t.text for t in tokens if t.is_word]
        assert words == ["România", "și", "țară"]

    def test_tokenize_words(self):
        """Test word-only tokenization"""
        words = tokenize_words("Aceasta este o propoziție în limba română.")
        assert "aceasta" in words
        assert "română" in words
        assert "." not in words

    def test_strip_diacritics(self):
        """Test diacritic stripping"""
        assert strip_diacritics("ăâîșț") == "aaist"
        assert strip_diacritics("România") == "Romania"
        assert strip_diacritics("fără") == "fara"

    def test_normalize_diacritics(self):
        """Test cedilla to comma normalization"""
        # ş (cedilla) -> ș (comma)
        assert normalize_diacritics("ş") == "ș"
        assert normalize_diacritics("ţ") == "ț"

    def test_has_romanian_diacritics(self):
        """Test diacritic detection"""
        assert has_romanian_diacritics("română") is True
        assert has_romanian_diacritics("Romania") is False
        assert has_romanian_diacritics("și") is True


class TestDiacriticAnalyzer:
    """Tests for diacritic analysis"""

    def test_correct_diacritics(self):
        """Test text with correct diacritics scores high"""
        text = "Aceasta este o propoziție în limba română și este corectă."
        result = analyze_diacritics(text)
        assert result.score > 0.8
        assert result.has_any_diacritics is True

    def test_missing_diacritics(self):
        """Test text missing diacritics scores lower"""
        text = "Aceasta este o propozitie in limba romana si este corecta."
        result = analyze_diacritics(text)
        assert result.score < 0.5
        assert result.missing_diacritics > 0

    def test_specific_missing_words(self):
        """Test specific missing diacritic detection"""
        # "si" should be "și", "in" should be "în"
        text = "Eu merg in oras si cumpar paine."
        result = analyze_diacritics(text)
        assert result.missing_diacritics >= 2  # at least "si" and "in"

    def test_empty_text(self):
        """Test empty text handling"""
        result = analyze_diacritics("")
        assert result.score == 1.0
        assert result.total_checkable == 0

    def test_quick_check(self):
        """Test quick diacritic check"""
        good = quick_diacritic_check("Aceasta este în română și corectă.")
        bad = quick_diacritic_check("Aceasta este in romana si corecta.")
        assert good > bad


class TestCodeSwitchDetector:
    """Tests for code-switch detection"""

    def test_pure_romanian(self):
        """Test pure Romanian text scores high"""
        text = "Aceasta este o propoziție în limba română."
        result = detect_code_switching(text)
        assert result.score > 0.9
        assert result.english_words == 0

    def test_english_contamination(self):
        """Test English words are detected"""
        text = "Aceasta este the best propoziție and very good."
        result = detect_code_switching(text)
        assert result.score < 0.8
        assert result.english_words > 0
        assert "the" in result.flagged_words or "and" in result.flagged_words

    def test_pure_english(self):
        """Test pure English text is detected"""
        text = "This is a sentence in the English language."
        result = detect_code_switching(text)
        assert result.score < 0.5
        assert result.english_rate > 0.3

    def test_is_likely_english(self):
        """Test English detection helper"""
        assert is_likely_english_text("This is English text.") is True
        assert is_likely_english_text("Aceasta este în română.") is False

    def test_romanian_lookalikes_not_flagged(self):
        """Test Romanian words that look English are not flagged"""
        # "nu", "de", "pe" look like English but are Romanian
        text = "Nu voi merge pe stradă de dimineață."
        result = detect_code_switching(text)
        assert "nu" not in result.flagged_words
        assert "de" not in result.flagged_words
        assert "pe" not in result.flagged_words


class TestRomanianNLPToolkit:
    """Tests for the main toolkit"""

    def test_analyze_good_text(self):
        """Test high-quality Romanian text"""
        text = """
        Propun următorul itinerar pentru excursia în Cluj-Napoca.
        În prima zi vom vizita Grădina Botanică și Muzeul de Artă.
        În a doua zi mergem la Cetățuia pentru priveliște.
        """
        toolkit = RomanianNLPToolkit()
        report = toolkit.analyze(text)

        assert report.overall_score > 0.7
        assert report.diacritic_score > 0.5
        assert report.is_likely_english is False
        assert report.is_too_short is False

    def test_analyze_bad_diacritics(self):
        """Test text with missing diacritics"""
        text = """
        Propun urmatorul itinerar pentru excursia in Cluj-Napoca.
        In prima zi vom vizita Gradina Botanica si Muzeul de Arta.
        """
        toolkit = RomanianNLPToolkit()
        report = toolkit.analyze(text)

        assert report.diacritic_score < 0.7
        assert report.diacritics.missing_diacritics > 0

    def test_analyze_short_text(self):
        """Test short text is penalized"""
        text = "Da, ok."
        toolkit = RomanianNLPToolkit()
        report = toolkit.analyze(text)

        assert report.is_too_short is True
        assert report.length_score < 1.0

    def test_analyze_english_response(self):
        """Test English response is heavily penalized"""
        text = "I suggest the following itinerary for the trip to Cluj."
        toolkit = RomanianNLPToolkit()
        report = toolkit.analyze(text)

        assert report.is_likely_english is True
        assert report.overall_score < 0.5

    def test_compute_g_score(self):
        """Test G score computation for metrics"""
        text = "Aceasta este o explicație în limba română și este corectă."
        toolkit = RomanianNLPToolkit()
        g_result = toolkit.compute_g_score(text)

        assert "G" in g_result
        assert "G_dia" in g_result
        assert "G_cs" in g_result
        assert 0.0 <= g_result["G"] <= 1.0

    def test_determinism(self):
        """Test that analysis is deterministic"""
        text = "Aceasta este o propoziție în limba română și este corectă."
        toolkit = RomanianNLPToolkit()

        result1 = toolkit.analyze(text)
        result2 = toolkit.analyze(text)

        assert result1.overall_score == result2.overall_score
        assert result1.diacritic_score == result2.diacritic_score
        assert result1.codeswitch_score == result2.codeswitch_score


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_analyze_romanian_text(self):
        """Test convenience function"""
        report = analyze_romanian_text("Aceasta este în română.")
        assert report.overall_score > 0
        assert report.total_words > 0
