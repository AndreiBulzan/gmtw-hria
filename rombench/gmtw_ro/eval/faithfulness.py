"""
Deterministic Faithfulness Checker for GMTW-Ro

Simple, robust approach: Check if planned entities are mentioned in the explanation.
No fuzzy matching, no complex negation detection - just substring presence.
"""

from typing import Any
from ..worlds.base import World


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic matching

    Removes:
    - Diacritics (ă→a, â→a, î→i, ș→s, ț→t)
    - Extra whitespace
    - Punctuation

    Converts to lowercase for case-insensitive matching.
    """
    # Lowercase
    text = text.lower()

    # Remove diacritics
    diacritic_map = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        # Include uppercase versions too (in case normalization order matters)
        'Ă': 'a', 'Â': 'a', 'Î': 'i', 'Ș': 's', 'Ț': 't',
    }

    for original, replacement in diacritic_map.items():
        text = text.replace(original, replacement)

    # Remove common punctuation that might interfere
    for char in '.,;:!?"()[]{}':
        text = text.replace(char, ' ')

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def get_entity_search_terms(entity: Any) -> list[str]:
    """
    Get all possible ways an entity might be mentioned

    Returns:
        List of normalized search terms (name + aliases)
    """
    terms = [normalize_text(entity.name)]

    # Add aliases
    for alias in entity.aliases:
        normalized = normalize_text(alias)
        if normalized and normalized not in terms:
            terms.append(normalized)

    return terms


def is_entity_mentioned(entity: Any, explanation: str) -> bool:
    """
    Check if entity is mentioned in the explanation (deterministic substring match)

    Args:
        entity: The entity to check
        explanation: The explanation text

    Returns:
        True if any form of the entity name appears in the explanation
    """
    normalized_explanation = normalize_text(explanation)
    search_terms = get_entity_search_terms(entity)

    for term in search_terms:
        # Use substring search - simple and deterministic
        if term in normalized_explanation:
            return True

    return False


def compute_faithfulness_deterministic(
    world: World,
    plan: dict,
    explanation: str
) -> dict[str, Any]:
    """
    Compute Faithfulness score using deterministic substring matching

    F measures: "Did the explanation mention all entities that appear in the plan?"

    We ONLY check for missing entities (things in plan but not in explanation).
    We do NOT check for "extra" entities (things mentioned but not in plan) because:
    - It's common to mention alternatives you didn't choose
    - Negation detection is fragile
    - The JSON is the source of truth for what was actually planned

    Args:
        world: The world specification
        plan: The extracted JSON plan
        explanation: The natural language explanation

    Returns:
        Dictionary with F score and details
    """
    # Extract entity IDs from plan
    planned_entity_refs = []

    for key, value in plan.items():
        if isinstance(value, list):
            planned_entity_refs.extend(value)
        elif isinstance(value, str) and value and value != "null":
            planned_entity_refs.append(value)

    if not planned_entity_refs:
        # Empty plan - F is undefined, return 1.0
        return {
            "F": 1.0,
            "missing": [],
            "planned_entities": [],
            "all_mentioned": True,
            "note": "Empty plan - F score is 1.0 by default"
        }

    # Resolve entity refs to actual entities
    planned_entities = []
    planned_entity_ids = []

    for ref in planned_entity_refs:
        # Try to find the entity
        entity_id = _resolve_entity_id(world, ref)

        if entity_id and entity_id in world.canonical_entities:
            entity = world.canonical_entities[entity_id]
            planned_entities.append(entity)
            planned_entity_ids.append(entity_id)

    # Check which planned entities are mentioned in the explanation
    missing_entities = []

    for entity_id, entity in zip(planned_entity_ids, planned_entities):
        if not is_entity_mentioned(entity, explanation):
            missing_entities.append(entity_id)

    # Compute F score with severity exponent
    # Import here to avoid circular imports
    from .metrics import SEVERITY_EXPONENT

    if not planned_entities:
        F = 1.0
        F_linear = 1.0
    else:
        # F = (entities mentioned / total entities planned) ^ severity_exponent
        mentioned_count = len(planned_entities) - len(missing_entities)
        F_linear = mentioned_count / len(planned_entities)
        F = F_linear ** SEVERITY_EXPONENT

    return {
        "F": F,
        "F_linear": F_linear,  # For reference
        "missing": missing_entities,
        "planned_entities": planned_entity_ids,
        "all_mentioned": len(missing_entities) == 0,
        "mentioned_count": len(planned_entities) - len(missing_entities),
        "total_count": len(planned_entities),
    }


def _resolve_entity_id(world: World, entity_ref: str) -> str:
    """
    Resolve entity reference to canonical ID

    Args:
        world: The world
        entity_ref: Entity ID or name

    Returns:
        Canonical entity ID or None if not found
    """
    # Check if it's already an ID
    if entity_ref in world.canonical_entities:
        return entity_ref

    # Try to find by name (case-insensitive, diacritic-insensitive)
    ref_normalized = normalize_text(entity_ref)

    for entity_id, entity in world.canonical_entities.items():
        # Check name
        if normalize_text(entity.name) == ref_normalized:
            return entity_id

        # Check aliases
        for alias in entity.aliases:
            if normalize_text(alias) == ref_normalized:
                return entity_id

    return None
