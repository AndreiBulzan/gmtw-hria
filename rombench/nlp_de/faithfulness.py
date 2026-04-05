"""
German-specific faithfulness checker

Handles German morphology for entity matching:
- Compound words (Schwarzwald might appear in Schwarzwaldregion)
- Umlaut normalization (ä→ae, ö→oe, ü→ue, ß→ss)
- German noun/adjective declension via lightweight stemming
"""

from typing import Any, Set
import unicodedata

from rombench.nlp.base_faithfulness import BaseFaithfulness


# ---------------------------------------------------------------------------
# Single-pass normalisation table (built once at import time)
# ---------------------------------------------------------------------------
_NORM_TABLE = str.maketrans({
    'ä': 'a',  'ö': 'o',  'ü': 'u',
    'Ä': 'a',  'Ö': 'o',  'Ü': 'u',
    '.': ' ',  ',': ' ',  ';': ' ',  ':': ' ',  '!': ' ',  '?': ' ',
    '"': ' ',  '(': ' ',  ')': ' ',  '[': ' ',  ']': ' ',
    '{': ' ',  '}': ' ',  '«': ' ',  '»': ' ',  '„': ' ',  '\u201c': ' ',
    '-': ' ',  '_': ' ',  '/': ' ',  '\\': ' ',  "'": ' ',
})


def normalize_text(text: str) -> str:
    """
    Normalize German text for matching.

    One translate() pass instead of 22 str.replace calls.
    ß→ss requires a separate replace (one char expands to two).
    NFKC is skipped when the result is already ASCII (common case).
    """
    text = text.lower().replace('ß', 'ss').translate(_NORM_TABLE)
    if not text.isascii():
        text = unicodedata.normalize('NFKC', text)
    return ' '.join(text.split())


# ---------------------------------------------------------------------------
# Lightweight word stemmer
# ---------------------------------------------------------------------------
# Ordered longest-first so we strip "ens" before "en" before "e" etc.
_STEM_SUFFIXES = ('ens', 'em', 'en', 'er', 'es', 'e', 's')
_MIN_AFTER_STEM = 5   # minimum chars remaining after suffix removal


def _stem_word(w: str) -> str:
    """Strip one German declension ending from a lowercased ASCII word."""
    for suf in _STEM_SUFFIXES:
        if w.endswith(suf) and (len(w) - len(suf)) >= _MIN_AFTER_STEM:
            return w[:-len(suf)]
    return w


def _stem_phrase(normalized: str) -> str:
    """
    Stem every word in an already-normalized string and add word-boundary
    padding so substring search is boundary-aware.

    '  botanischen gartens  '  →  ' botanisch garten '
    """
    return ' ' + ' '.join(_stem_word(w) for w in normalized.split()) + ' '


# ---------------------------------------------------------------------------
# Public module-level helpers (kept for backward compatibility / external use)
# ---------------------------------------------------------------------------

def german_morphological_forms(token: str) -> Set[str]:
    """
    Generate German morphological forms (kept for external callers).

    GermanFaithfulness itself no longer uses this — it uses stemming instead.
    """
    forms = {token}
    w = token.lower()
    forms.add(w)
    for uml, repl in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
        if uml in w:
            forms.add(w.replace(uml, repl))
        if repl in w:
            forms.add(w.replace(repl, uml))
    for suf in ('e', 'en', 'er', 'es', 'em', 'n', 's', 'se'):
        forms.add(w + suf)
    for suf in ('en', 'er', 'es', 'em', 'e', 's', 'n'):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            forms.add(w[:-len(suf)])
    return forms


# ---------------------------------------------------------------------------
# GermanFaithfulness
# ---------------------------------------------------------------------------

class GermanFaithfulness(BaseFaithfulness):
    """
    German faithfulness checker using normalize-and-stem matching.

    Strategy:
    1.  Normalize both entity name and explanation text (umlaut→ascii, punct→space).
    2.  Stem every word in each (strip one common German declension ending).
    3.  Check if the stemmed entity phrase appears in the stemmed text (with
        word-boundary padding).

    This is O(tokens) per entity lookup — no cross-product of inflected forms —
    so cold-cache cost is the same as English/Romanian.

    Example:
        entity "Botanischer Garten" → stems to "botanisch garten"
        text   "im Botanischen Gartens" → " im botanisch garten "
        "botanisch garten" found → match ✓
    """

    def __init__(self) -> None:
        # Maps lowercased entity string → ' stemmed entity ' (padded)
        self._stem_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # BaseFaithfulness interface
    # ------------------------------------------------------------------

    def normalize_text(self, text: str) -> str:
        return normalize_text(text)

    def generate_forms(self, token: str) -> Set[str]:
        """Minimal implementation required by ABC; not used internally."""
        return {token}

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def _get_stemmed_entity(self, entity: str) -> str:
        """Return (and cache) the padded stemmed form of entity."""
        key = entity.lower()
        if key not in self._stem_cache:
            self._stem_cache[key] = _stem_phrase(normalize_text(key))
        return self._stem_cache[key]

    def check_entity_mentioned(
        self,
        entity: str,
        normalized_text: str,
        stemmed_text: str = '',
        debug: bool = False,
    ) -> bool:
        """
        Return True if entity is mentioned in normalized_text.

        stemmed_text: pre-computed _stem_phrase(normalized_text).  Pass it
        when checking multiple entities against the same text to avoid
        recomputing the stem on every call.  If omitted it is computed here.
        """
        stemmed_entity = self._get_stemmed_entity(entity)
        if not stemmed_text:
            stemmed_text = _stem_phrase(normalized_text)
        if stemmed_entity in stemmed_text:
            return True
        if debug:
            print(f'[DEBUG] Entity not matched: {entity!r}')
            print(f'        stemmed entity : {stemmed_entity!r}')
            print(f'        stemmed text   : {stemmed_text!r}')
        return False

    def compute_faithfulness(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute F score using German stemmed-matching."""
        if not plan or not explanation:
            return {'F': 0.0, 'entities_total': 0,
                    'entities_mentioned': 0, 'missing_entities': []}

        entities = self.extract_entities(world, plan)
        if not entities:
            return {'F': 1.0, 'entities_total': 0,
                    'entities_mentioned': 0, 'missing_entities': [],
                    'note': 'No entities to check'}

        # Normalise and stem the explanation ONCE and reuse for all entity checks.
        normalized_text = normalize_text(explanation)
        stemmed_text = _stem_phrase(normalized_text)

        mentioned, missing = [], []
        for entity in entities:
            if self.check_entity_mentioned(entity, normalized_text, stemmed_text):
                mentioned.append(entity)
            else:
                missing.append(entity)

        total = len(entities)
        F_raw = len(mentioned) / total if total > 0 else 1.0

        # ------------------------------------------------------------------
        # Hallucination penalty
        # ------------------------------------------------------------------
        canonical = getattr(world, 'canonical_entities', {})
        hallucinated = []
        if canonical and isinstance(plan, dict):
            planned_refs: set[str] = set()
            for val in plan.values():
                if isinstance(val, list):
                    planned_refs.update(v for v in val if isinstance(v, str) and v)
                elif isinstance(val, str) and val and val.lower() != 'null':
                    planned_refs.add(val)

            for eid, ent in canonical.items():
                if eid in planned_refs:
                    continue
                ent_name = getattr(ent, 'name', '')
                all_names = [ent_name] + getattr(ent, 'aliases', [])
                if any(n and (n == ref or n.lower() == ref.lower())
                       for n in all_names for ref in planned_refs):
                    continue
                # Prefer the German name for matching in a German text; fall
                # back to other aliases only when no German name is stored.
                de_name = (ent.attributes.get('name_de')
                           if hasattr(ent, 'attributes') else None)
                seen: set[str] = set()
                candidates: list[str] = []
                for n in ([de_name] if de_name else []) + [ent_name] + all_names:
                    if n:
                        nl = n.lower()
                        if nl not in seen:
                            seen.add(nl)
                            candidates.append(n)
                if any(self.check_entity_mentioned(n, normalized_text, stemmed_text)
                       for n in candidates):
                    hallucinated.append(eid)

        hallucination_penalty = 0.9 ** len(hallucinated) if hallucinated else 1.0
        F_linear = F_raw * hallucination_penalty
        F = F_linear ** kwargs.get('severity_exponent', 3.0)

        return {
            'F': F, 'F_linear': F_linear, 'F_mention': F_raw,
            'F_hallucination_penalty': hallucination_penalty,
            'entities_total': total, 'entities_mentioned': len(mentioned),
            'mentioned_entities': mentioned, 'missing_entities': missing,
            'hallucinated': hallucinated,
        }