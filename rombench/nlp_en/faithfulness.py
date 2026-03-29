"""
English-specific faithfulness checker

Uses simpler morphology than Romanian (mainly plurals, verb forms).
"""

from typing import Any, Set
import unicodedata

from rombench.nlp.base_faithfulness import BaseFaithfulness


def normalize_text(text: str) -> str:
    """
    Normalize English text for matching.

    - lowercase
    - remove apostrophes
    - remove punctuation
    - normalize unicode
    - collapse whitespace
    """
    text = text.lower()
    # Remove apostrophes (straight and curly)
    for c in "\u2018\u2019'`":
        text = text.replace(c, "")
    # Remove punctuation
    for c in '.,;:!?"()[]{}':
        text = text.replace(c, " ")
    # Normalize unicode (NFKC)
    text = unicodedata.normalize('NFKC', text)
    text = " ".join(text.split())
    return text


def english_morphological_forms(token: str) -> Set[str]:
    """
    Generate English morphological forms.

    Covers:
    - Plural: add 's', 'es', 'ies'
    - Past tense: add 'ed', 'd'
    - Progressive: add 'ing'
    - Possessive: add "'s"
    - Reverse inflection: strip endings to find base forms
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
    """English-specific faithfulness checking."""

    def normalize_text(self, text: str) -> str:
        """Normalize English text for matching."""
        return normalize_text(text)

    def generate_forms(self, token: str) -> Set[str]:
        """Generate English morphological forms."""
        return english_morphological_forms(token)

    def _generate_multiword_forms(self, phrase: str) -> Set[str]:
        """
        Generate forms for multi-word English phrases.

        Handles:
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

    def check_entity_mentioned(
        self, entity: str, normalized_text: str, debug: bool = False
    ) -> bool:
        """Check if entity is mentioned in text, with optional debug output."""
        forms = self.generate_forms(entity.lower())
        if ' ' in entity:
            forms.update(self._generate_multiword_forms(entity))
        # Normalize all forms for matching
        norm_forms = {self.normalize_text(form) for form in forms}
        for form in norm_forms:
            if form in normalized_text:
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
        """Compute faithfulness score using English morphology."""
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
            # Collect plan entity IDs
            planned_refs = set()
            for val in plan.values():
                if isinstance(val, list):
                    planned_refs.update(v for v in val if isinstance(v, str) and v)
                elif isinstance(val, str) and val and val.lower() != "null":
                    planned_refs.add(val)

            for eid, ent in canonical.items():
                # Skip entities that are in the plan
                if eid in planned_refs:
                    continue
                # Check by name or alias matching
                ent_name = getattr(ent, 'name', '')
                all_names = [ent_name] + getattr(ent, 'aliases', [])
                if any(n and (n == ref or n.lower() == ref.lower())
                       for n in all_names for ref in planned_refs):
                    continue
                # Is this entity mentioned in the explanation?
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