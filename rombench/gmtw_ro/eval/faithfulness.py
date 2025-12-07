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
        forms.add(stem + "ei")      
        forms.add(stem + "e")        
        forms.add(stem + "elor")     
        forms.add(stem + "ele")      
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
    """
    tokens = term.split()
    if not tokens:
        return {term}

    last = tokens[-1]
    forms = set()

    if last.endswith("ul"):
        forms.add(last + "ui")           
        forms.add(last[:-2] + "ului")
    elif last.endswith("u"):
        stem = last[:-1]
        forms.add(stem + "ului")
        forms.add(stem + "ui")
    elif last.endswith("a"):
        stem = last[:-1]
        forms.add(stem + "ei")
    elif last.endswith("e"):
        forms.add(last + "lui")
        forms.add(last + "i")
    elif last.endswith("i"):
        forms.add(last + "lor")
    else:
        forms.add(last + "ului")

    # build full phrase variants
    phrase_forms = {
        " ".join(tokens[:-1] + [f])
        for f in forms
    }

    return phrase_forms



def get_entity_search_terms(entity: Any) -> list[str]:
    """
    Build normalized terms + morphological expansions.
    """
    base_terms = [normalize_text(entity.name)]
    base_terms.extend(normalize_text(a) for a in entity.aliases)

    final = set()

    for term in base_terms:
        final.add(term)

        # token-wise morphology
        tokens = term.split()
        for i, tok in enumerate(tokens):
            if len(tok) < 2:
                continue

            for form in romanian_morphological_forms(tok):
                variant = tokens.copy()
                variant[i] = form
                final.add(" ".join(variant))

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
