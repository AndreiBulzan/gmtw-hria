"""
German-specific faithfulness checker

Handles German morphology for entity matching:
- Compound words (Schwarzwald might appear in Schwarzwaldregion)
- Umlaut normalization (ä→ae, ö→oe, ü→ue, ß→ss)
- Basic German noun declension (cases)
- Plural forms
"""

import itertools
from typing import Any, Set
import unicodedata

from rombench.nlp.base_faithfulness import BaseFaithfulness


def normalize_text(text: str) -> str:
    """
    Normalize German text for matching.

    - lowercase
    - normalize umlauts to base forms (ä→a, ö→o, ü→u, ß→ss)
    - remove punctuation
    - collapse whitespace
    """
    text = text.lower()

    # Normalize umlauts and ß for matching
    umlaut_map = {
        'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
        'Ä': 'a', 'Ö': 'o', 'Ü': 'u',
    }
    for src, dst in umlaut_map.items():
        text = text.replace(src, dst)

    # NOTE: Do NOT blanket-replace ae/oe/ue→a/o/u here.  That destroys
    # legitimate German words such as "Mauer", "Feuer", "Abenteuer",
    # "Museum", etc.  Umlaut normalisation (ä→a, ö→o, ü→u) above is
    # sufficient for matching purposes.

    # Remove punctuation
    for c in '.,;:!?"()[]{}«»„"':
        text = text.replace(c, " ")

    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    text = " ".join(text.split())
    return text


def german_morphological_forms(token: str) -> Set[str]:
    """
    Generate German morphological forms.

    Covers:
    - Plural forms (-e, -en, -er, -n, -s, Umlaut+e)
    - Genitive (-s, -es, -er, -en)
    - Dative (-em, -en, -er)
    - Accusative (-en, -e)
    - Compound word parts (substring matching)
    - Umlaut variants (ä↔ae, ö↔oe, ü↔ue, ß↔ss)
    """
    forms = {token}
    w = token.lower()
    forms.add(w)

    # === Umlaut variants ===
    # Generate both umlaut and non-umlaut forms
    umlaut_pairs = [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]
    current = w
    for uml, repl in umlaut_pairs:
        if uml in current:
            variant = current.replace(uml, repl)
            forms.add(variant)
    # Also reverse: ae->ä etc
    current_rev = w
    for uml, repl in umlaut_pairs:
        if repl in current_rev:
            variant = current_rev.replace(repl, uml)
            forms.add(variant)

    # === Plural forms ===
    forms.add(w + "e")      # Tag→Tage
    forms.add(w + "en")     # Frau→Frauen
    forms.add(w + "er")     # Kind→Kinder
    forms.add(w + "n")      # Straße→Straßen
    forms.add(w + "s")      # Auto→Autos (also genitive)
    forms.add(w + "se")     # Ergebnis→Ergebnisse

    # === Case endings ===
    # Genitive
    forms.add(w + "es")     # des Hauses
    forms.add(w + "s")      # des Mannes

    # Dative
    forms.add(w + "em")     # dem Mann(e)m
    forms.add(w + "er")     # der Frau

    # Adjective declension common endings
    forms.add(w + "en")
    forms.add(w + "es")
    forms.add(w + "er")
    forms.add(w + "em")

    # === Strip common endings to find stems ===
    for suffix in ["en", "er", "es", "em", "e", "s", "n", "se", "ung", "keit", "heit"]:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            stem = w[:-len(suffix)]
            forms.add(stem)

    # === Umlaut plurals (common patterns) ===
    # a→ä, o→ö, u→ü in plural
    for base_vowel, umlaut in [('a', 'ä'), ('o', 'ö'), ('u', 'ü')]:
        if base_vowel in w:
            umlauted = w.replace(base_vowel, umlaut, 1)
            forms.add(umlauted)
            forms.add(umlauted + "e")
            forms.add(umlauted + "er")

    return forms


def _german_adj_forms(token: str) -> Set[str]:
    """
    Generate declined forms of a German adjective (or attributive adjective).

    Strips any existing adjective ending to recover the stem, then adds all
    four-case declension endings (weak, mixed, strong declension all covered
    by -e / -en / -em / -er / -es).
    """
    w = token.lower()
    stem = w
    # Strip the longest matching adjective ending first (order: longer → shorter)
    for ending in ("em", "en", "er", "es", "e"):
        if w.endswith(ending) and len(w) > len(ending) + 3:
            stem = w[:-len(ending)]
            break
    forms = {w}
    for ending in ("e", "en", "em", "er", "es"):
        forms.add(stem + ending)
    return forms


def _german_word_forms(token: str) -> Set[str]:
    """
    Combined noun + adjective forms for a German word used in a non-final
    position of a multi-word entity name.

    A non-final token can be either an adjective ("Ethnographisches") or a
    noun in a compound ("Museum" in "Ethnographisches Museum Siebenbürgens").
    Both need to be covered:
    - Adjective endings via _german_adj_forms  ("-e/-en/-em/-er/-es")
    - Noun case endings via german_morphological_forms ("-s", "-en", …)
    """
    return german_morphological_forms(token) | _german_adj_forms(token)


class GermanFaithfulness(BaseFaithfulness):
    """German-specific faithfulness checking."""

    def normalize_text(self, text: str) -> str:
        """Normalize German text for entity matching."""
        return normalize_text(text)

    def generate_forms(self, token: str) -> Set[str]:
        """Generate German morphological forms."""
        return german_morphological_forms(token)

    def _generate_multiword_forms(self, phrase: str) -> Set[str]:
        """
        Generate forms for multi-word German phrases.

        Handles:
        - Adjective declension across all four cases for every non-last token
          (e.g. "Ethnographisches Museum" → "Ethnographischen Museums" in gen.)
        - Compound word formation (all tokens joined)
        - Genitive "des" prefix forms on the last word
        """
        forms = {phrase.lower()}
        tokens = phrase.split()

        if not tokens:
            return forms

        # Apply full cross-product of adjective × noun inflections
        if len(tokens) > 1:
            # Combined noun+adjective forms for every non-last token so both
            # "Ethnographischen" (adjective) and "Museums" (noun genitive) are
            # covered regardless of whether the token is an adj or a noun.
            adj_form_sets = [_german_word_forms(t) for t in tokens[:-1]]
            noun_forms = german_morphological_forms(tokens[-1])
            for adj_combo in itertools.product(*adj_form_sets):
                for noun_form in noun_forms:
                    forms.add(" ".join(list(adj_combo) + [noun_form]))

        # Try compound word (join all words)
        compound = "".join(t.lower() for t in tokens)
        forms.add(compound)
        # Also generate morphological forms of compound
        for form in german_morphological_forms(compound):
            forms.add(form)

        # Genitive "des" prefix forms
        if len(tokens) >= 1:
            base = tokens[-1].lower()
            forms.add(f"des {base}s")
            forms.add(f"des {base}es")

        return forms

    def check_entity_mentioned(
        self, entity: str, normalized_text: str, debug: bool = False
    ) -> bool:
        """
        Check if entity is mentioned in text.

        Enhanced for German: also checks if the entity appears
        as part of a compound word in the text.
        """
        forms = self.generate_forms(entity.lower())
        if ' ' in entity:
            forms.update(self._generate_multiword_forms(entity))

        # Normalize all forms for matching
        norm_forms = {self.normalize_text(form) for form in forms}

        for form in norm_forms:
            if form and form in normalized_text:
                return True

        if debug:
            print(f"[DEBUG] Entity not matched: '{entity}' | Forms: {norm_forms}")
        return False

    def compute_faithfulness(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        """Compute faithfulness score using German morphology."""
        if not plan or not explanation:
            return {
                "F": 0.0,
                "entities_total": 0,
                "entities_mentioned": 0,
                "missing_entities": [],
            }

        entities = self.extract_entities(world, plan)
        if not entities:
            return {
                "F": 1.0,
                "entities_total": 0,
                "entities_mentioned": 0,
                "missing_entities": [],
                "note": "No entities to check",
            }

        normalized_text = self.normalize_text(explanation)
        mentioned = []
        missing = []

        for entity in entities:
            if self.check_entity_mentioned(entity, normalized_text):
                mentioned.append(entity)
            else:
                missing.append(entity)

        total = len(entities)
        mentioned_count = len(mentioned)
        F_raw = mentioned_count / total if total > 0 else 1.0

        # Hallucination penalty: check for canonical entities that appear
        # in the explanation but were NOT part of the plan.
        canonical = getattr(world, 'canonical_entities', {})
        hallucinated = []
        if canonical and isinstance(plan, dict):
            planned_refs = set()
            for val in plan.values():
                if isinstance(val, list):
                    planned_refs.update(v for v in val if isinstance(v, str) and v)
                elif isinstance(val, str) and val and val.lower() != "null":
                    planned_refs.add(val)

            for eid, ent in canonical.items():
                if eid in planned_refs:
                    continue
                ent_name = getattr(ent, 'name', '')
                all_names = [ent_name] + getattr(ent, 'aliases', [])
                # Skip if any name variant matches a planned reference
                if any(n and (n == ref or n.lower() == ref.lower())
                       for n in all_names for ref in planned_refs):
                    continue
                # Check if any name variant is mentioned in explanation
                if any(n and self.check_entity_mentioned(n, normalized_text)
                       for n in all_names):
                    hallucinated.append(eid)

        hallucination_penalty = 0.9 ** len(hallucinated) if hallucinated else 1.0
        F_linear = F_raw * hallucination_penalty

        severity_exponent = kwargs.get('severity_exponent', 3.0)
        F = F_linear ** severity_exponent

        return {
            "F": F,
            "F_linear": F_linear,
            "F_mention": F_raw,
            "F_hallucination_penalty": hallucination_penalty,
            "entities_total": total,
            "entities_mentioned": mentioned_count,
            "mentioned_entities": mentioned,
            "missing_entities": missing,
            "hallucinated": hallucinated,
        }