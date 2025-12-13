"""
Deterministic Faithfulness Checker for GMTW-Ro

Romanian-aware string matching:
- Diacritic normalization
- Accurate articulated forms
- Plural forms
- Genitive–Dative – much improved
- Multi-word entity handling
"""

from typing import Any
from ..worlds.base import World


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic matching.

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


# ============================================================
# MORPHOLOGY HELPERS
# ============================================================
def is_vowel(ch):
    return ch in "aeiouăâî"


def romanian_morphological_forms(token: str) -> set:
    """
    Improved Romanian morphological rules.

    Covers:
    - articulated forms
    - plural
    - genitive/dative
    - masculine/feminine/neuter heuristics
    - proper nouns
    """
    forms = {token}

    w = token

    if w.endswith("a") or w.endswith("ă"):
        stem = w[:-1]
        forms.add(stem + "a")
        forms.add(stem + "e")
        forms.add(stem + "elor")
        forms.add(stem + "ele")

        # Genitive forms depend on the ending pattern
        # Words ending in -ica/-ină/-ina/-uia use -ii genitive (NOT -ei)
        if w.endswith("ica") or w.endswith("ică"):
            forms.add(stem + "ii")     # biserica → bisericii
        elif w.endswith("ina") or w.endswith("ină"):
            forms.add(stem + "ii")     # grădina → grădinii (NOT grădinei!)
        elif w.endswith("uia"):
            forms.add(stem + "i")      # cetățuia → cetățuii
        else:
            # Standard genitive for other -a/-ă words: -a → -ei
            forms.add(stem + "ei")     # casa → casei
            forms.add(stem + "ii")     # Also add -ii as alternative
        return forms

    if w.endswith("e"):
        stem = w[:-1]
        forms.add(w + "a")           
        forms.add(stem + "i")        
        forms.add(stem + "ilor")     
        forms.add(stem + "ile")      
        forms.add(w + "lui")         
        return forms

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

        forms.add(w + "ului")
        return forms

    forms.add(w + "ul")
    forms.add(w + "ului")
    forms.add(w + "ui")
    forms.add(w + "i")
    forms.add(w + "ii")
    forms.add(w + "ilor")
    return forms


def add_genitive_dative_for_phrase(term: str) -> set:
    """
    Generates genitive/dative forms for multi-word entities.

    "Parcul Central" → {"parcului central", ...}
    "Muzeul Satului" → {"muzeului satului", ...}
    "Grădina Botanică" → {"grădinei botanice", ...}
    """
    tokens = term.split()
    if not tokens:
        return {term}

    forms = set()

    # Single-word handling (original behavior for last token)
    last = tokens[-1]
    last_forms = set()

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

    # Build full phrase variants with only last word changed
    for f in last_forms:
        forms.add(" ".join(tokens[:-1] + [f]))

    return forms


def generate_coordinated_genitive_forms(tokens: list[str]) -> set[str]:
    """
    Generate genitive forms where multiple adjacent tokens are inflected together.

    In Romanian, noun+adjective pairs must agree in case:
    - "Grădina Botanică" → "Grădinei Botanice"
    - "Casa Memorială" → "Casei Memoriale"
    - "Biblioteca Națională" → "Bibliotecii Naționale"
    - "Biserica Neagră" → "Bisericii Negre"

    This handles the common pattern where:
    - Feminine noun (-a/-ă) + feminine adjective (-a/-ă) both change to genitive
    """
    forms = set()

    if len(tokens) < 2:
        return forms

    # Pattern 1: First word is feminine noun, following words are adjectives
    # Examples: "gradina botanica", "casa memoriala", "biblioteca nationala"
    first = tokens[0]
    if first.endswith("a") and len(first) > 2:
        # Special cases for genitive forms:
        # -ica/-ică → -icii (biserica → bisericii)
        # -ina/-ină → -inii (grădina → grădinii)
        # -uia → -uii (cetățuia → cetățuii)
        # -ia → -iei (Maria → Mariei) - but also try -ii
        if first.endswith("ica") or first.endswith("ică"):
            first_gen = first[:-1] + "ii"
        elif first.endswith("ina") or first.endswith("ină"):
            first_gen = first[:-1] + "ii"
        elif first.endswith("uia"):
            first_gen = first[:-1] + "ii"  # cetățuia → cetățuii
        else:
            # Standard genitive of feminine noun: -a → -ei
            first_gen = first[:-1] + "ei"

        # Now inflect any following adjectives that also end in -a/-ă/-e
        rest_original = tokens[1:]
        rest_genitive = []

        for tok in rest_original:
            if tok.endswith("a") and len(tok) > 2:
                # Feminine adjective: -a → -e (genitive)
                gen_form = tok[:-1] + "e"
                rest_genitive.append(gen_form)
            elif tok.endswith("ă") and len(tok) > 2:
                # Alternative feminine ending: -ă → -e
                gen_form = tok[:-1] + "e"
                rest_genitive.append(gen_form)
            else:
                # Keep as-is (proper nouns like names, or masculine adjectives)
                rest_genitive.append(tok)

        # Handle vowel reduction: "ea" → "e" in adjectives like neagră → negre
        rest_genitive_reduced = []
        for tok in rest_genitive:
            if "ea" in tok:
                rest_genitive_reduced.append(tok.replace("ea", "e"))
            else:
                rest_genitive_reduced.append(tok)

        # Full coordinated genitive form
        forms.add(first_gen + " " + " ".join(rest_genitive))

        # Also add form with vowel reduction (neagre → negre)
        if rest_genitive_reduced != rest_genitive:
            forms.add(first_gen + " " + " ".join(rest_genitive_reduced))

        # Also try partial: first word genitive, rest unchanged
        forms.add(first_gen + " " + " ".join(rest_original))

    # Pattern 2: Articulated masculine noun (ending in -ul) followed by adjective
    # Examples: "parcul central" → "parcului central"
    if first.endswith("ul") and len(first) > 3:
        first_gen = first[:-2] + "ului"
        rest = tokens[1:]
        forms.add(first_gen + " " + " ".join(rest))

    # Pattern 3: Feminine noun ending in -ă (alternative to -a)
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



def get_entity_search_terms(entity: Any) -> list[str]:
    """
    Build normalized terms + morphological expansions.

    This includes:
    1. Base forms and aliases
    2. Token-wise morphology (each word inflected independently)
    3. Coordinated genitive forms (noun+adjective inflected together)
    4. Phrase-level genitive/dative forms
    """
    base_terms = [normalize_text(entity.name)]
    base_terms.extend(normalize_text(a) for a in entity.aliases)

    final = set()

    for term in base_terms:
        final.add(term)

        tokens = term.split()

        # Token-wise morphology (original behavior)
        for i, tok in enumerate(tokens):
            if len(tok) < 2:
                continue

            for form in romanian_morphological_forms(tok):
                variant = tokens.copy()
                variant[i] = form
                final.add(" ".join(variant))

        # Coordinated genitive forms (NEW - handles noun+adjective pairs)
        for gd in generate_coordinated_genitive_forms(tokens):
            final.add(gd)

        # Phrase-level genitive/dative (last token only)
        for gd in add_genitive_dative_for_phrase(term):
            final.add(gd)

    return list(final)



def is_entity_mentioned(entity: Any, explanation: str) -> bool:
    """
    Deterministic substring matcher over expanded morphological space.
    """
    exp = normalize_text(explanation)
    terms = get_entity_search_terms(entity)

    for t in terms:
        if t in exp:
            return True
    return False



def compute_faithfulness_deterministic(
    world: World,
    plan: dict,
    explanation: str
) -> dict[str, Any]:

    planned_refs = []

    for key, val in plan.items():
        if isinstance(val, list):
            planned_refs.extend(val)
        elif isinstance(val, str) and val and val != "null":
            planned_refs.append(val)

    if not planned_refs:
        return {
            "F": 1.0,
            "missing": [],
            "planned_entities": [],
            "all_mentioned": True,
            "note": "Empty plan"
        }

    planned_entities = []
    planned_ids = []

    for ref in planned_refs:
        entity_id = _resolve_entity_id(world, ref)
        if entity_id and entity_id in world.canonical_entities:
            ent = world.canonical_entities[entity_id]
            planned_entities.append(ent)
            planned_ids.append(entity_id)

    missing = []
    for eid, ent in zip(planned_ids, planned_entities):
        if not is_entity_mentioned(ent, explanation):
            missing.append(eid)

    from .metrics import SEVERITY_EXPONENT

    if not planned_entities:
        F_linear = 1.0
    else:
        F_linear = (len(planned_entities) - len(missing)) / len(planned_entities)

    F = F_linear ** SEVERITY_EXPONENT

    return {
        "F": F,
        "F_linear": F_linear,
        "missing": missing,
        "planned_entities": planned_ids,
        "all_mentioned": len(missing) == 0,
        "mentioned_count": len(planned_entities) - len(missing),
        "total_count": len(planned_entities),
    }



def _resolve_entity_id(world: World, entity_ref: str) -> str:
    """
    Resolve a name or ID into canonical ID.
    """
    if entity_ref in world.canonical_entities:
        return entity_ref

    ref_norm = normalize_text(entity_ref)

    for eid, ent in world.canonical_entities.items():
        if normalize_text(ent.name) == ref_norm:
            return eid
        for alias in ent.aliases:
            if normalize_text(alias) == ref_norm:
                return eid

    return None
