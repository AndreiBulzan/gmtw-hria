"""
Deterministic Faithfulness Checker for GMTW-Ro
— With Romanian Lemmatization + Articulated Forms

This module uses Stanza for Romanian lemmatization, which provides
more accurate entity matching than the default morphology rules.

Optimizations:
- GPU acceleration (automatic if CUDA available)
- Batch processing for multiple texts
- LRU cache for repeated lemmatizations

Requires: pip install stanza
Or: pip install rombench[stanza]
"""

from typing import Any
from functools import lru_cache

# Lazy initialization of Stanza (only load when first used)
_nlp = None
_stanza_initialized = False

# Cache for lemmatization results (avoid re-processing same text)
_lemma_cache: dict[str, list[str]] = {}


def _check_gpu_available():
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_stanza_nlp():
    """Lazy initialization of Stanza pipeline with GPU support."""
    global _nlp, _stanza_initialized

    if _stanza_initialized:
        return _nlp

    _stanza_initialized = True
    use_gpu = _check_gpu_available()

    try:
        import stanza

        # Check if model is already downloaded
        try:
            _nlp = stanza.Pipeline(
                "ro",
                processors="tokenize,pos,lemma",
                tokenize_no_ssplit=True,
                download_method=None,  # Don't auto-download
                use_gpu=use_gpu,
                verbose=False,
            )
        except Exception:
            # Model not downloaded, try to download it
            stanza.download("ro", verbose=False)
            _nlp = stanza.Pipeline(
                "ro",
                processors="tokenize,pos,lemma",
                tokenize_no_ssplit=True,
                use_gpu=use_gpu,
                verbose=False,
            )

        if use_gpu:
            import sys
            print("Stanza: Using GPU acceleration", file=sys.stderr)

    except ImportError:
        import warnings
        warnings.warn(
            "stanza is not installed. Falling back to simple tokenization. "
            "Install with: pip install stanza"
        )
        _nlp = None
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to initialize Stanza: {e}. Falling back to simple tokenization.")
        _nlp = None

    return _nlp


def lemmatize_ro(text: str) -> list[str]:
    """
    Lemmatize Romanian text using Stanza with caching.
    If stanza is unavailable, return simple tokens as fallback.
    """
    # Check cache first
    if text in _lemma_cache:
        return _lemma_cache[text]

    import re
    tokens = re.findall(r"[A-Za-zăâîșțĂÂÎȘȚ]+", text.lower())

    nlp = _get_stanza_nlp()
    if nlp is None:
        return tokens

    try:
        doc = nlp(text)
        result = [w.lemma for s in doc.sentences for w in s.words]
        # Cache the result (limit cache size to prevent memory issues)
        if len(_lemma_cache) < 10000:
            _lemma_cache[text] = result
        return result
    except Exception:
        return tokens


def lemmatize_batch(texts: list[str]) -> list[list[str]]:
    """
    Batch lemmatize multiple texts at once (more efficient on GPU).

    Args:
        texts: List of texts to lemmatize

    Returns:
        List of lemma lists, one per input text
    """
    import re

    nlp = _get_stanza_nlp()

    # Check which texts need processing (not in cache)
    results = [None] * len(texts)
    texts_to_process = []
    indices_to_process = []

    for i, text in enumerate(texts):
        if text in _lemma_cache:
            results[i] = _lemma_cache[text]
        else:
            texts_to_process.append(text)
            indices_to_process.append(i)

    # If all cached, return early
    if not texts_to_process:
        return results

    # Fallback if no Stanza
    if nlp is None:
        for i, text in zip(indices_to_process, texts_to_process):
            tokens = re.findall(r"[A-Za-zăâîșțĂÂÎȘȚ]+", text.lower())
            results[i] = tokens
        return results

    # Batch process with Stanza
    try:
        import stanza
        docs = nlp.bulk_process(
            [stanza.Document([], text=t) for t in texts_to_process]
        )

        for idx, (i, doc) in enumerate(zip(indices_to_process, docs)):
            lemmas = [w.lemma for s in doc.sentences for w in s.words]
            results[i] = lemmas
            # Cache results
            if len(_lemma_cache) < 10000:
                _lemma_cache[texts_to_process[idx]] = lemmas

    except Exception:
        # Fallback to simple tokenization
        for i, text in zip(indices_to_process, texts_to_process):
            tokens = re.findall(r"[A-Za-zăâîșțĂÂÎȘȚ]+", text.lower())
            results[i] = tokens

    return results


def clear_cache():
    """Clear the lemmatization cache."""
    global _lemma_cache
    _lemma_cache = {}


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
