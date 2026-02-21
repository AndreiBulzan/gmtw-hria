"""
English-specific faithfulness checker

Uses simpler morphology than Romanian (mainly plurals, verb forms).
"""

from typing import Any, Set
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.nlp.base_faithfulness import BaseFaithfulness


def normalize_text(text: str) -> str:
    """
    Normalize English text for matching.
    
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    text = text.lower()
    
    for c in '.,;:!?"()[]{}':
        text = text.replace(c, " ")
    
    text = " ".join(text.split())
    return text


def english_morphological_forms(token: str) -> Set[str]:
    """
    Generate English morphological forms.
    
    Simpler than Romanian:
    - Plural: add 's', 'es', 'ies'
    - Past tense: add 'ed', 'd'
    - Progressive: add 'ing'
    - Possessive: add "'s"
    """
    forms = {token}
    w = token.lower()
    
    # Plural forms
    forms.add(w + "s")
    forms.add(w + "es")
    
    # -y -> -ies (city -> cities)
    if w.endswith("y") and len(w) > 1:
        stem = w[:-1]
        forms.add(stem + "ies")
        forms.add(stem + "y")
    
    # Past tense
    forms.add(w + "ed")
    if w.endswith("e"):
        forms.add(w + "d")
    
    # Progressive
    forms.add(w + "ing")
    if w.endswith("e") and len(w) > 2:
        stem = w[:-1]
        forms.add(stem + "ing")
    
    # Possessive
    forms.add(w + "'s")
    
    # Remove final 's' for potential singular
    if w.endswith("s") and len(w) > 2:
        forms.add(w[:-1])
    
    # Remove final 'es' for potential singular
    if w.endswith("es") and len(w) > 3:
        forms.add(w[:-2])
    
    # Remove 'ed' for base form
    if w.endswith("ed") and len(w) > 3:
        forms.add(w[:-2])
        forms.add(w[:-1])  # In case of -e ending
    
    # Remove 'ing' for base form
    if w.endswith("ing") and len(w) > 4:
        forms.add(w[:-3])
        forms.add(w[:-3] + "e")  # In case of dropped -e
    
    return forms


class EnglishFaithfulness(BaseFaithfulness):
    """English-specific faithfulness checking"""
    
    def normalize_text(self, text: str) -> str:
        """Normalize English text"""
        return normalize_text(text)
    
    def generate_forms(self, token: str) -> Set[str]:
        """Generate English morphological forms"""
        return english_morphological_forms(token)
    
    def _generate_multiword_forms(self, phrase: str) -> Set[str]:
        """
        Generate forms for multi-word English phrases.
        
        For English, mostly just need to handle:
        - Possessive forms (Central Park -> Central Park's)
        - Plural forms (apply to last word)
        """
        forms = {phrase.lower()}
        tokens = phrase.split()
        
        if not tokens:
            return forms
        
        # Add possessive of full phrase
        forms.add(phrase.lower() + "'s")
        
        # Apply morphology to last word
        if len(tokens) > 1:
            prefix = " ".join(tokens[:-1]).lower()
            last_forms = english_morphological_forms(tokens[-1])
            for last_form in last_forms:
                forms.add(f"{prefix} {last_form}")
        
        return forms