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

    "Parcul Central" → {"parcului central", ...}
    "Grădina Botanică" → {"grădinei botanice", ...}
    """
    tokens = term.split()
    if not tokens:
        return {term}

    forms = set()
    last = tokens[-1]
    last_forms: Set[str] = set()

    if last.endswith("ul"):
        last_forms.add(last + "ui")
        last_forms.add(last[:-2] + "ului")
    elif last.endswith("u"):
        stem = last[:-1]
        last_forms.add(stem + "ului")
        last_forms.add(stem + "ui")
    elif last.endswith("a"):
        stem = last[:-1]
        last_forms.add(stem + "ei")
    elif last.endswith("e"):
        last_forms.add(last + "lui")
        last_forms.add(last + "i")
    elif last.endswith("i"):
        last_forms.add(last + "lor")
    else:
        last_forms.add(last + "ului")

    for f in last_forms:
        forms.add(" ".join(tokens[:-1] + [f]))

    return forms


def generate_coordinated_genitive_forms(tokens: list) -> Set[str]:
    """
    Generate genitive forms where multiple adjacent tokens are inflected together.

    In Romanian, noun+adjective pairs must agree in case:
    - "Grădina Botanică" → "Grădinei Botanice"
    - "Casa Memorială" → "Casei Memoriale"
    - "Parcul Central" → "Parcului Central"
    """
    forms: Set[str] = set()
    if len(tokens) < 2:
        return forms

    first = tokens[0]

    # Pattern 1: Feminine noun ending in -a
    if first.endswith("a") and len(first) > 2:
        if first.endswith("ica") or first.endswith("ică"):
            first_gen = first[:-1] + "ii"
        elif first.endswith("ina") or first.endswith("ină"):
            first_gen = first[:-1] + "ii"
        elif first.endswith("uia"):
            first_gen = first[:-1] + "ii"
        else:
            first_gen = first[:-1] + "ei"

        rest_original = tokens[1:]
        rest_genitive = []
        for tok in rest_original:
            if tok.endswith("a") and len(tok) > 2:
                rest_genitive.append(tok[:-1] + "e")
            elif tok.endswith("ă") and len(tok) > 2:
                rest_genitive.append(tok[:-1] + "e")
            else:
                rest_genitive.append(tok)

        rest_genitive_reduced = []
        for tok in rest_genitive:
            rest_genitive_reduced.append(tok.replace("ea", "e") if "ea" in tok else tok)

        forms.add(first_gen + " " + " ".join(rest_genitive))
        if rest_genitive_reduced != rest_genitive:
            forms.add(first_gen + " " + " ".join(rest_genitive_reduced))
        forms.add(first_gen + " " + " ".join(rest_original))

    # Pattern 2: Articulated masculine noun ending in -ul
    if first.endswith("ul") and len(first) > 3:
        first_gen = first[:-2] + "ului"
        forms.add(first_gen + " " + " ".join(tokens[1:]))

    # Pattern 3: Feminine noun ending in -ă
    if first.endswith("ă") and len(first) > 2:
        first_gen = first[:-1] + "ei"
        rest_original = tokens[1:]
        rest_genitive = []
        for tok in rest_original:
            if tok.endswith("a") and len(tok) > 2:
                rest_genitive.append(tok[:-1] + "e")
            elif tok.endswith("ă") and len(tok) > 2:
                rest_genitive.append(tok[:-1] + "e")
            else:
                rest_genitive.append(tok)
        forms.add(first_gen + " " + " ".join(rest_genitive))
        forms.add(first_gen + " " + " ".join(rest_original))

    return forms


class RomanianFaithfulness(BaseFaithfulness):
    """Romanian-specific faithfulness checking"""

    def __init__(self, use_stanza: bool = False):
        self.use_stanza = use_stanza
        if use_stanza:
            try:
                from ...gmtw_ro.eval.faithfulness_stanza import (
                    compute_faithfulness_deterministic as compute_with_stanza
                )
                self.stanza_compute = compute_with_stanza
            except ImportError:
                import warnings
                warnings.warn("Stanza not available, falling back to rule-based")
                self.use_stanza = False

    def normalize_text(self, text: str) -> str:
        return normalize_text(text)

    def generate_forms(self, token: str) -> Set[str]:
        return romanian_morphological_forms(token)

    def _generate_multiword_forms(self, phrase: str) -> Set[str]:
        return add_genitive_dative_for_phrase(phrase)

    def _get_search_terms(self, entity: Any) -> list:
        """
        Build all normalized search terms for an entity, expanding:
        - base name + aliases
        - token-wise morphology
        - coordinated genitive forms (noun+adjective pairs)
        - phrase-level genitive/dative
        """
        base_terms = [normalize_text(entity.name)]
        base_terms.extend(normalize_text(a) for a in getattr(entity, 'aliases', []))

        final: Set[str] = set()
        for term in base_terms:
            final.add(term)
            tokens = term.split()

            # Token-wise morphology (each word independently)
            for i, tok in enumerate(tokens):
                if len(tok) < 2:
                    continue
                for form in romanian_morphological_forms(tok):
                    variant = tokens.copy()
                    variant[i] = form
                    final.add(" ".join(variant))

            # Coordinated genitive (noun+adjective together)
            for gd in generate_coordinated_genitive_forms(tokens):
                final.add(gd)

            # Phrase-level genitive (last token only)
            for gd in add_genitive_dative_for_phrase(term):
                final.add(gd)

        return list(final)

    def _resolve_entity_id(self, world: Any, entity_ref: str):
        """Resolve a plan value string to a canonical entity ID."""
        canonical = getattr(world, 'canonical_entities', {})
        if entity_ref in canonical:
            return entity_ref
        ref_norm = normalize_text(entity_ref)
        for eid, ent in canonical.items():
            if normalize_text(ent.name) == ref_norm:
                return eid
            for alias in getattr(ent, 'aliases', []):
                if normalize_text(alias) == ref_norm:
                    return eid
        return None

    def _is_entity_mentioned(self, entity: Any, explanation: str) -> bool:
        exp = normalize_text(explanation)
        for term in self._get_search_terms(entity):
            if term in exp:
                return True
        return False

    def compute_faithfulness(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        if self.use_stanza and hasattr(self, 'stanza_compute'):
            try:
                return self.stanza_compute(world, plan, explanation)
            except Exception:
                pass

        canonical = getattr(world, 'canonical_entities', {})

        # If world has no canonical entities, fall back to base implementation
        if not canonical:
            return super().compute_faithfulness(world, plan, explanation, **kwargs)

        # Handle plan=None
        if not plan:
            plan = {}

        # Collect refs from plan values
        planned_refs = []
        for val in plan.values():
            if isinstance(val, list):
                planned_refs.extend(v for v in val if isinstance(v, str) and v)
            elif isinstance(val, str) and val and val.lower() != "null":
                planned_refs.append(val)

        if not planned_refs:
            return {
                "F": 1.0,
                "F_linear": 1.0,
                "entities_total": 0,
                "entities_mentioned": 0,
                "missing_entities": [],
                "hallucinated": [],
                "note": "Empty plan",
            }

        # Resolve to canonical entities
        planned_entities = []
        planned_ids: Set[str] = set()
        for ref in planned_refs:
            eid = self._resolve_entity_id(world, ref)
            if eid and eid in canonical:
                planned_entities.append(canonical[eid])
                planned_ids.add(eid)

        # Check missing
        missing = [
            ent.id for ent in planned_entities
            if not self._is_entity_mentioned(ent, explanation)
        ]

        # Check hallucinations
        hallucinated = [
            eid for eid, ent in canonical.items()
            if eid not in planned_ids and self._is_entity_mentioned(ent, explanation)
        ]

        severity_exponent = kwargs.get('severity_exponent', 3.0)

        mention_score = (
            (len(planned_entities) - len(missing)) / len(planned_entities)
            if planned_entities else 1.0
        )
        hallucination_penalty = 0.9 ** len(hallucinated) if hallucinated else 1.0
        F_linear = mention_score * hallucination_penalty
        F = F_linear ** severity_exponent

        return {
            "F": F,
            "F_linear": F_linear,
            "F_mention": mention_score,
            "F_hallucination_penalty": hallucination_penalty,
            "entities_total": len(planned_entities),
            "entities_mentioned": len(planned_entities) - len(missing),
            "missing_entities": missing,
            "hallucinated": hallucinated,
        }