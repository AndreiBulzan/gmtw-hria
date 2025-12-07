"""
Deterministic Faithfulness Checker for GMTW-Ro
— With Romanian Lemmatization + Articulated Forms
"""

from typing import Any

try:
    import stanza
    stanza.download("ro", verbose=False)
    nlp = stanza.Pipeline(
        "ro",
        processors="tokenize,pos,lemma",    
        tokenize_no_ssplit=True
    )
except Exception:
    nlp = None


def lemmatize_ro(text: str) -> list[str]:
    """
    Lemmatize Romanian text using Stanza.
    If stanza is unavailable, return simple tokens as fallback.
    """
    import re
    tokens = re.findall(r"[A-Za-zăâîșțĂÂÎȘȚ]+", text.lower())

    if nlp is None:
        return tokens

    try:
        doc = nlp(text)
        return [w.lemma for s in doc.sentences for w in s.words]
    except Exception:
        return tokens


def normalize_text(text: str) -> str:
    text = text.lower()

    diacritic_map = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'a', 'Â': 'a', 'Î': 'i', 'Ș': 's', 'Ț': 't',
    }
    for a, b in diacritic_map.items():
        text = text.replace(a, b)

    for char in '.,;:!?"()[]{}':
        text = text.replace(char, ' ')

    return ' '.join(text.split())


def generate_articulated_forms(word: str) -> list[str]:
    if not word:
        return []

    forms = {word}
    stem = word

    if word.endswith("ă"):
        stem = word[:-1]
        forms |= {stem + "a", stem + "ei", stem + "ele", stem + "elor"}

    elif word.endswith("e"):
        forms |= {word + "a", word + "le", word + "lui", word + "lor"}

    elif word.endswith("a"):
        forms |= {word + "ua", word[:-1] + "lei"}

    elif word.endswith("iu"):
        forms |= {word + "l", word + "lui", word[:-1] + "ii", word[:-1] + "ilor"}

    else:
        suffix_na = "l" if word.endswith(("u", "i")) else "ul"
        forms.add(word + suffix_na)
        forms.add(word + "lui")
        if not word.endswith(("u", "i")):
            forms.add(word + "ului")

        forms |= {
            word + "ii", word + "ilor",
            word + "urile", word + "urilor",
            word + "ele", word + "elor"
        }

    return list(forms)


def get_entity_search_terms(entity: Any) -> list[str]:
    base_terms = [
        normalize_text(entity.name),
        *[normalize_text(a) for a in entity.aliases],
    ]

    final_terms = set(base_terms)

    # Add articulated forms
    for term in base_terms:
        for token in term.split():
            if len(token) >= 3:
                for form in generate_articulated_forms(token):
                    final_terms.add(term.replace(token, form))

    # Add lemma forms
    for term in base_terms:
        lemmas = lemmatize_ro(term)
        for lemma in lemmas:
            final_terms.add(lemma)

    return list(final_terms)


def is_entity_mentioned(entity: Any, explanation: str) -> bool:
    normalized_exp = normalize_text(explanation)
    lemma_set_exp = set(lemmatize_ro(normalized_exp))

    search_terms = get_entity_search_terms(entity)

    for term in search_terms:

        # Direct substring match
        if term in normalized_exp:
            return True

        # Lemma match
        if term in lemma_set_exp:
            return True

    return False


def compute_faithfulness_deterministic(
    world,
    plan: dict,
    explanation: str
) -> dict[str, Any]:

    # Extract entity refs
    planned_entity_refs = []

    for key, value in plan.items():
        if isinstance(value, list):
            planned_entity_refs.extend(value)
        elif isinstance(value, str) and value and value != "null":
            planned_entity_refs.append(value)

    if not planned_entity_refs:
        return {
            "F": 1.0,
            "missing": [],
            "planned_entities": [],
            "all_mentioned": True,
            "note": "Empty plan - F score is 1.0 by default",
        }

    # Resolve entity objects
    planned_entities = []
    planned_entity_ids = []

    for ref in planned_entity_refs:
        entity_id = _resolve_entity_id(world, ref)

        if entity_id and entity_id in world.canonical_entities:
            entity = world.canonical_entities[entity_id]
            planned_entities.append(entity)
            planned_entity_ids.append(entity_id)

    # Check mentions
    missing_entities = []

    for entity_id, entity in zip(planned_entity_ids, planned_entities):
        if not is_entity_mentioned(entity, explanation):
            missing_entities.append(entity_id)

    # Severity exponent
    from .metrics import SEVERITY_EXPONENT

    if not planned_entities:
        F = 1.0
        F_linear = 1.0
    else:
        mentioned_count = len(planned_entities) - len(missing_entities)
        F_linear = mentioned_count / len(planned_entities)
        F = F_linear ** SEVERITY_EXPONENT

    return {
        "F": F,
        "F_linear": F_linear,
        "missing": missing_entities,
        "planned_entities": planned_entity_ids,
        "all_mentioned": len(missing_entities) == 0,
        "mentioned_count": len(planned_entities) - len(missing_entities),
        "total_count": len(planned_entities),
    }


def _resolve_entity_id(world, entity_ref: str) -> str:
    if entity_ref in world.canonical_entities:
        return entity_ref

    ref_norm = normalize_text(entity_ref)

    for entity_id, entity in world.canonical_entities.items():
        if normalize_text(entity.name) == ref_norm:
            return entity_id
        for alias in entity.aliases:
            if normalize_text(alias) == ref_norm:
                return entity_id

    return None
