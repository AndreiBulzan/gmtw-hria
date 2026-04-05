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


# Single-pass translation table built once at import time.
# Maps umlauts to ASCII base forms, ß→ss is handled separately (two→one char).
_NORM_TABLE = str.maketrans({
    'ä': 'a',  'ö': 'o',  'ü': 'u',
    'Ä': 'a',  'Ö': 'o',  'Ü': 'u',
    '.': ' ',  ',': ' ',  ';': ' ',  ':': ' ',  '!': ' ',  '?': ' ',
    '"': ' ',  '(': ' ',  ')': ' ',  '[': ' ',  ']': ' ',
    '{': ' ',  '}': ' ',  '«': ' ',  '»': ' ',  '„': ' ',  '\u201c': ' ',
})


def normalize_text(text: str) -> str:
    """
    Normalize German text for matching.

    Single-pass via str.translate (one table lookup per character) instead of
    22 sequential str.replace calls.  ß→ss still requires a str.replace because
    it expands one character to two, which translate() cannot do.

    NFKC normalization is skipped when the string is already ASCII (the common
    case after umlaut conversion), saving ~half the remaining cost.
    """
    text = text.lower().replace('ß', 'ss').translate(_NORM_TABLE)
    if not text.isascii():
        text = unicodedata.normalize('NFKC', text)
    return " ".join(text.split())


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
    Compact set of case forms for non-final tokens in a multiword entity name.

    Used as the left-hand side of a cross-product, so this set is intentionally
    small (~7 forms) to avoid combinatorial blowup.

    Covers:
    - All five adjective-declension endings (-e/-en/-em/-er/-es), obtained by
      stripping any existing ending to recover the stem first.
    - Genitive noun -s  (e.g. "Museum" → "Museums")
    - The unchanged surface form.

    We do NOT use german_morphological_forms here — it generates 30-40 forms per
    token, which makes a 3-token phrase produce 35x35x35 ≈ 43 000 candidates.
    """
    forms = _german_adj_forms(token)   # stem + -e/-en/-em/-er/-es  (~6 forms)
    w = token.lower()
    forms.add(w + "s")                 # noun genitive -s (e.g. Museums, Gartens)
    return forms


def _german_noun_forms(token: str) -> Set[str]:
    """
    Compact forms for the *last* token of a multi-word entity name.

    Covers:
    - The unchanged surface form.
    - Common noun endings added  (-e, -en, -er, -es, -s).
    - Bare stems obtained by stripping common endings  (needed so that a
      genitively-suffixed source name like "Siebenb\u00fcrgens" also matches
      the nominative "Siebenb\u00fcrgen" if a model drops the genitive).

    This replaces german_morphological_forms() in the cross-product:
    ~13 forms instead of ~35, giving a 2.5\u00d7 reduction in the cross-product
    size (3-token: 7\u00d77\u00d713=637 vs 1715 before).
    """
    w = token.lower()
    forms = {w}
    # Add common noun endings
    for suf in ("e", "en", "er", "es", "s"):
        forms.add(w + suf)
    # Strip common endings to expose the bare stem
    for suf in ("en", "er", "es", "em", "e", "s", "n"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            forms.add(w[: -len(suf)])
    return forms


class GermanFaithfulness(BaseFaithfulness):
    """German-specific faithfulness checking."""

    def __init__(self):
        # Cache normalized form-sets per entity to avoid recomputing across instances.
        self._form_cache: dict[str, frozenset[str]] = {}

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
            # Non-last tokens: compact adj+noun forms (~7 each).
            # Last token: compact noun forms (~13) — enough to cover
            # nominative, genitive, dative, accusative, and bare stems.
            # Using german_morphological_forms() here would give ~35 forms per
            # token and blow the 3-token cross-product to 7×7×35=1715.
            adj_form_sets = [_german_word_forms(t) for t in tokens[:-1]]
            noun_forms = _german_noun_forms(tokens[-1])
            for adj_combo in itertools.product(*adj_form_sets):
                for noun_form in noun_forms:
                    forms.add(" ".join(list(adj_combo) + [noun_form]))

        # Compound word (useful for detecting e.g. "Kunstmuseum" in text
        # when entity name is "Kunst Museum").
        compound = "".join(t.lower() for t in tokens)
        forms.add(compound)
        # Do NOT call german_morphological_forms(compound) — the compound is
        # already a long string and its decorated forms are unlikely to appear.

        # Genitive "des" prefix forms
        if len(tokens) >= 1:
            base = tokens[-1].lower()
            forms.add(f"des {base}s")
            forms.add(f"des {base}es")

        return forms

    def _get_normalized_forms(self, entity: str) -> frozenset[str]:
        """
        Return (and cache) the set of normalized forms for an entity.

        Cache key is lowercased so that "Rosenpark" and "rosenpark" share the
        same entry instead of triggering two independent form builds.
        """
        key = entity.lower()
        if key not in self._form_cache:
            if ' ' in key:
                # Multiword: _generate_multiword_forms already covers all
                # meaningful combinatorial forms.  Calling generate_forms(key)
                # would apply german_morphological_forms to the full phrase
                # string as if it were a single token, producing nonsensical
                # decorated forms like "botanischer gartene" — pure waste.
                forms = self._generate_multiword_forms(key)
            else:
                forms = self.generate_forms(key)
            self._form_cache[key] = frozenset(
                nf for nf in (self.normalize_text(f) for f in forms) if nf
            )
        return self._form_cache[key]

    def check_entity_mentioned(
        self, entity: str, normalized_text: str, debug: bool = False
    ) -> bool:
        """
        Check if entity is mentioned in text.

        Enhanced for German: also checks if the entity appears
        as part of a compound word in the text.
        """
        norm_forms = self._get_normalized_forms(entity)
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
                # For German text, hallucinated references will use the German
                # entity name or (as fallback) the English name.  Romanian
                # names don't need German morphological analysis and would only
                # add spurious cache entries, so we skip them here.
                de_name = ent.attributes.get('name_de') if hasattr(ent, 'attributes') else None
                en_name = getattr(ent, 'aliases', [])
                # Build a deduplicated list: German name first, then English
                # aliases (those that look like English, i.e., ASCII-only and
                # different from de_name), then the primary entity name.
                candidate_names: list[str] = []
                seen_lower: set[str] = set()
                for n in ([de_name] if de_name else []) + [ent_name] + getattr(ent, 'aliases', []):
                    if n:
                        nl = n.lower()
                        if nl not in seen_lower:
                            seen_lower.add(nl)
                            candidate_names.append(n)
                if any(self.check_entity_mentioned(n, normalized_text)
                       for n in candidate_names):
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