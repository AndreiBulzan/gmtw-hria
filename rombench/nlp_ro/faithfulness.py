"""
Romanian-specific faithfulness checker

Uses Romanian morphology to match inflected forms.
"""

from typing import Any, Set

from rombench.nlp.base_faithfulness import BaseFaithfulness


def normalize_text(text: str) -> str:
    """
    Normalize Romanian text for matching.
    
    - lowercase
    - remove diacritics
    - remove punctuation
    - collapse whitespace
    """
    text = text.lower()
    
    diacritic_map = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'a', 'Â': 'a', 'Î': 'i', 'Ș': 's', 'Ț': 't'
    }
    for o, r in diacritic_map.items():
        text = text.replace(o, r)
    
    for c in '.,;:!?"()[]{}':
        text = text.replace(c, " ")
    
    text = " ".join(text.split())
    return text


def is_vowel(ch):
    return ch in "aeiouăâî"


def romanian_morphological_forms(token: str) -> Set[str]:
    """
    Romanian morphological forms generator.
    
    Covers:
    - Articulated forms (definite article)
    - Plural
    - Genitive/Dative
    - Gender variations
    """
    forms = {token}
    w = token
    
    # Words ending in -a/-ă (usually feminine)
    if w.endswith("a") or w.endswith("ă"):
        stem = w[:-1]
        forms.add(stem + "a")
        forms.add(stem + "e")
        forms.add(stem + "elor")
        forms.add(stem + "ele")
        
        # Genitive forms
        if w.endswith("ica") or w.endswith("ică"):
            forms.add(stem + "ii")
        elif w.endswith("ina") or w.endswith("ină"):
            forms.add(stem + "ii")
        elif w.endswith("uia"):
            forms.add(stem + "i")
        else:
            forms.add(stem + "ei")
            forms.add(stem + "ii")
        return forms
    
    # Words ending in -e
    if w.endswith("e"):
        stem = w[:-1]
        forms.add(w + "a")
        forms.add(stem + "i")
        forms.add(stem + "ilor")
        forms.add(stem + "ile")
        forms.add(w + "lui")
        return forms
    
    # Words ending in -u
    if w.endswith("u"):
        stem = w[:-1]
        forms.add(stem + "ul")
        forms.add(stem + "ului")
        forms.add(stem + "ui")
        plural = stem + "i"
        forms.add(plural)
        forms.add(plural + "i")
        forms.add(plural + "lor")
        return forms
    
    # Consonant ending
    if not is_vowel(w[-1]):
        forms.add(w + "ul")
        forms.add(w + "ului")
        forms.add(w + "ui")
        forms.add(w + "i")
        forms.add(w + "ii")
        forms.add(w + "ilor")
        forms.add(w + "uri")
        forms.add(w + "urile")
        forms.add(w + "urilor")
        return forms
    
    # Default vowel ending
    forms.add(w + "ul")
    forms.add(w + "ului")
    forms.add(w + "ui")
    forms.add(w + "i")
    forms.add(w + "ii")
    forms.add(w + "ilor")
    return forms


def add_genitive_dative_for_phrase(term: str) -> Set[str]:
    """
    Generate genitive/dative forms for multi-word entities.
    
    Examples:
    - "Parcul Central" → "parcului central"
    - "Grădina Botanică" → "grădinii botanice"
    """
    tokens = term.split()
    if not tokens:
        return {term}
    
    forms = set()
    last = tokens[-1]
    
    # Apply morphology to last word
    if last.endswith("ul"):
        stem = last[:-2]
        forms.add(stem + "ului")
        forms.add(stem + "ui")
    elif last.endswith("ă"):
        stem = last[:-1]
        if last.endswith("ică") or last.endswith("ină"):
            forms.add(stem + "ii")
        else:
            forms.add(stem + "ei")
            forms.add(stem + "ii")
    elif last.endswith("a"):
        stem = last[:-1]
        forms.add(stem + "ei")
        forms.add(stem + "ii")
    
    # Build multi-word forms
    if len(tokens) > 1:
        prefix = " ".join(tokens[:-1])
        # Create a copy of forms to iterate over to avoid "Set changed size during iteration" error
        base_forms = forms.copy()
        for last_form in base_forms:
            forms.add(f"{prefix} {last_form}")
    
    return forms


class RomanianFaithfulness(BaseFaithfulness):
    """Romanian-specific faithfulness checking"""
    
    def __init__(self, use_stanza: bool = False):
        self.use_stanza = use_stanza
        if use_stanza:
            try:
                # Import Stanza version if needed
                from ...gmtw_ro.eval.faithfulness_stanza import (
                    compute_faithfulness_deterministic as compute_with_stanza
                )
                self.stanza_compute = compute_with_stanza
            except ImportError:
                import warnings
                warnings.warn("Stanza not available, falling back to rule-based")
                self.use_stanza = False
    
    def normalize_text(self, text: str) -> str:
        """Normalize Romanian text"""
        return normalize_text(text)
    
    def generate_forms(self, token: str) -> Set[str]:
        """Generate Romanian morphological forms"""
        return romanian_morphological_forms(token)
    
    def _generate_multiword_forms(self, phrase: str) -> Set[str]:
        """Generate forms for multi-word Romanian phrases"""
        return add_genitive_dative_for_phrase(phrase)
    
    def compute_faithfulness(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        """
        Compute faithfulness using Romanian morphology.
        
        If use_stanza=True and available, uses Stanza for better accuracy.
        Otherwise uses rule-based morphology.
        """
        if self.use_stanza and hasattr(self, 'stanza_compute'):
            try:
                return self.stanza_compute(world, plan, explanation)
            except Exception:
                # Fall back to base implementation
                pass
        
        # Use base implementation with Romanian morphology
        return super().compute_faithfulness(world, plan, explanation, **kwargs)
