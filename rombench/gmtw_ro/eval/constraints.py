"""
Constraint and goal checking functions for GMTW-Ro

These functions verify whether model plans satisfy world constraints and goals.
"""

from typing import Any, Callable
from ..worlds.base import World


# ============================================================================
# Travel World Constraints
# ============================================================================

def check_must_include_type(world: World, plan: dict, params: dict) -> bool:
    """
    Check if plan includes at least one entity of required type

    Params:
        type_required: str - the type that must be included
    """
    type_required = params.get("type_required")
    if not type_required:
        return True

    # Extract all entity IDs from plan
    plan_entity_ids = _extract_plan_entity_ids(plan)

    # Check if any entity is of the required type
    for entity_ref in plan_entity_ids:
        # Resolve name to ID
        actual_id = _resolve_entity_id(world, entity_ref)

        if actual_id and actual_id in world.canonical_entities:
            entity = world.canonical_entities[actual_id]
            if entity.attributes.get("type") == type_required:
                return True

    return False


def check_max_outdoor_per_day(world: World, plan: dict, params: dict) -> bool:
    """
    Check if each day has at most max_outdoor outdoor activities

    Params:
        max_outdoor: int - maximum outdoor activities per day
    """
    max_outdoor = params.get("max_outdoor", 1)

    # For travel world, plan is {"day1": [...], "day2": [...]}
    for day_key, entity_ids in plan.items():
        if not day_key.startswith("day"):
            continue

        outdoor_count = 0
        for entity_id in entity_ids:
            # Handle both ID and name formats
            actual_id = _resolve_entity_id(world, entity_id)

            if actual_id and actual_id in world.canonical_entities:
                entity = world.canonical_entities[actual_id]
                if not entity.attributes.get("indoor", True):
                    outdoor_count += 1

        if outdoor_count > max_outdoor:
            return False

    return True


def check_all_family_friendly(world: World, plan: dict, params: dict) -> bool:
    """
    Check if all activities in plan are family-friendly
    """
    plan_entity_ids = _extract_plan_entity_ids(plan)

    for entity_id in plan_entity_ids:
        actual_id = _resolve_entity_id(world, entity_id)

        if actual_id and actual_id in world.canonical_entities:
            entity = world.canonical_entities[actual_id]
            if not entity.attributes.get("family_friendly", False):
                return False

    return True


def check_budget_limit(world: World, plan: dict, params: dict) -> bool:
    """
    Check if total cost of planned activities is within budget

    Params:
        max_budget: int - maximum budget in lei
    """
    max_budget = params.get("max_budget", 0)
    if max_budget <= 0:
        return True

    # Calculate total cost
    total_cost = 0
    plan_entity_ids = _extract_plan_entity_ids(plan)

    for entity_id in plan_entity_ids:
        actual_id = _resolve_entity_id(world, entity_id)

        if actual_id and actual_id in world.canonical_entities:
            entity = world.canonical_entities[actual_id]
            cost = entity.attributes.get("cost_lei", 0)
            total_cost += cost

    return total_cost <= max_budget


def check_min_activities_per_day(world: World, plan: dict, params: dict) -> bool:
    """
    Check if each day has at least min_per_day activities

    Params:
        min_per_day: int - minimum activities per day
    """
    min_per_day = params.get("min_per_day", 1)

    for day_key, entity_ids in plan.items():
        if not day_key.startswith("day"):
            continue

        if len(entity_ids) < min_per_day:
            return False

    return True


def check_max_activities_per_day(world: World, plan: dict, params: dict) -> bool:
    """
    Check if each day has at most max_per_day activities

    Params:
        max_per_day: int - maximum activities per day
    """
    max_per_day = params.get("max_per_day", 3)

    for day_key, entity_ids in plan.items():
        if not day_key.startswith("day"):
            continue

        if len(entity_ids) > max_per_day:
            return False

    return True


# ============================================================================
# Schedule World Constraints
# ============================================================================

def check_max_appointments_per_day(world: World, plan: dict, params: dict) -> bool:
    """
    Check if each day has at most max_per_day appointments

    Params:
        max_per_day: int - maximum appointments per day
    """
    max_per_day = params.get("max_per_day", 2)

    # Count appointments per day
    day_counts = {}

    for slot_key, appointment in plan.items():
        if appointment is None or appointment == "null":
            continue

        # Extract day from slot_key (e.g., "Luni_dimineață" -> "Luni")
        parts = slot_key.split("_")
        if parts:
            day = parts[0]
            day_counts[day] = day_counts.get(day, 0) + 1

    # Check if any day exceeds limit
    for day, count in day_counts.items():
        if count > max_per_day:
            return False

    return True


def check_keep_high_priority(world: World, plan: dict, params: dict) -> bool:
    """
    Check if all high-priority appointments are kept in the schedule
    """
    # Find all high-priority appointments
    high_priority_appointments = []
    for entity in world.canonical_entities.values():
        if entity.attributes.get("priority") == "high":
            high_priority_appointments.append(entity.name)

    # Check if all are in the plan
    planned_appointments = set()
    for slot_key, appointment in plan.items():
        if appointment and appointment != "null":
            planned_appointments.add(appointment)

    for hp_apt in high_priority_appointments:
        if hp_apt not in planned_appointments:
            return False

    return True


# ============================================================================
# Fact World Constraints
# ============================================================================

def check_answer_matches_context(world: World, plan: dict, params: dict) -> bool:
    """
    Check if answer matches the fact database

    The answer should be based ONLY on facts in the provided context,
    not on the model's parametric knowledge.
    """
    # Get the answer from the plan
    answer = plan.get("answer", "")
    if not answer:
        return False

    # Get the fact database from the world payload
    facts = world.payload.get("facts", {})
    if not facts:
        return True  # No facts to check against

    # Normalize answer for comparison
    answer_lower = answer.lower().strip()

    # Check if the answer matches any value in the fact database
    for key, value in facts.items():
        value_lower = str(value).lower().strip()
        # Check for exact match or containment
        if value_lower in answer_lower or answer_lower in value_lower:
            return True

    return False


# ============================================================================
# Recipe World Constraints
# ============================================================================

def check_all_vegetarian(world: World, plan: dict, params: dict) -> bool:
    """
    Check if all dishes in the plan are vegetarian
    """
    plan_entity_ids = _extract_plan_entity_ids(plan)

    for entity_id in plan_entity_ids:
        actual_id = _resolve_entity_id(world, entity_id)

        if actual_id and actual_id in world.canonical_entities:
            entity = world.canonical_entities[actual_id]
            if not entity.attributes.get("vegetarian", False):
                return False

    return True


def check_no_gluten(world: World, plan: dict, params: dict) -> bool:
    """
    Check if no dishes contain gluten
    """
    plan_entity_ids = _extract_plan_entity_ids(plan)

    for entity_id in plan_entity_ids:
        actual_id = _resolve_entity_id(world, entity_id)

        if actual_id and actual_id in world.canonical_entities:
            entity = world.canonical_entities[actual_id]
            if entity.attributes.get("contains_gluten", False):
                return False

    return True


def check_no_lactose(world: World, plan: dict, params: dict) -> bool:
    """
    Check if no dishes contain lactose
    """
    plan_entity_ids = _extract_plan_entity_ids(plan)

    for entity_id in plan_entity_ids:
        actual_id = _resolve_entity_id(world, entity_id)

        if actual_id and actual_id in world.canonical_entities:
            entity = world.canonical_entities[actual_id]
            if entity.attributes.get("contains_lactose", False):
                return False

    return True


def check_max_daily_calories(world: World, plan: dict, params: dict) -> bool:
    """
    Check if daily calorie total is within limit

    Params:
        max_calories: int - maximum calories per day
    """
    max_calories = params.get("max_calories", 2000)

    # Group dishes by day
    daily_calories = {}

    for key, value in plan.items():
        # Parse key format: "day1_mic_dejun" or similar
        parts = key.split("_", 1)
        if len(parts) < 2 or not parts[0].startswith("day"):
            continue

        day = parts[0]
        dish_ref = value

        if not dish_ref:
            continue

        actual_id = _resolve_entity_id(world, dish_ref)

        if actual_id and actual_id in world.canonical_entities:
            entity = world.canonical_entities[actual_id]
            calories = entity.attributes.get("calories", 0)

            if day not in daily_calories:
                daily_calories[day] = 0
            daily_calories[day] += calories

    # Check each day
    for day, total in daily_calories.items():
        if total > max_calories:
            return False

    return True


def check_all_meals_filled(world: World, plan: dict, params: dict) -> bool:
    """
    Check that all meal slots have a dish

    Params:
        num_days: int - number of days
        meals: list[str] - meal types per day
    """
    num_days = params.get("num_days", 2)
    meals = params.get("meals", ["mic_dejun", "pranz", "cina"])

    for day_num in range(1, num_days + 1):
        for meal in meals:
            key = f"day{day_num}_{meal}"
            if key not in plan or not plan[key]:
                return False

    return True


# ============================================================================
# General Goals (Structural Constraints)
# ============================================================================

def check_days_non_empty(world: World, plan: dict, params: dict) -> bool:
    """
    Check that each day in the plan has at least one activity

    Params:
        num_days: int - number of days expected
    """
    num_days = params.get("num_days", 2)

    for day_num in range(1, num_days + 1):
        day_key = f"day{day_num}"
        if day_key not in plan:
            return False
        if not plan[day_key] or len(plan[day_key]) == 0:
            return False

    return True


def check_no_duplicates(world: World, plan: dict, params: dict) -> bool:
    """
    Check that no entity appears more than once in the plan
    """
    seen = set()
    plan_entity_ids = _extract_plan_entity_ids(plan)

    for entity_id in plan_entity_ids:
        actual_id = _resolve_entity_id(world, entity_id)

        if actual_id in seen:
            return False
        seen.add(actual_id)

    return True


def check_valid_entity_ids(world: World, plan: dict, params: dict) -> bool:
    """
    Check that all referenced entity IDs exist in the world

    Params:
        valid_ids: list[str] - list of valid entity IDs
    """
    valid_ids = set(params.get("valid_ids", []))
    plan_entity_ids = _extract_plan_entity_ids(plan)

    for entity_id in plan_entity_ids:
        actual_id = _resolve_entity_id(world, entity_id)

        if actual_id not in valid_ids:
            return False

    return True


def check_no_slot_overlaps(world: World, plan: dict, params: dict) -> bool:
    """
    Check that no time slot has more than one appointment

    Params:
        days: list[str] - list of days
        slots: list[str] - list of slots
    """
    # For schedule world, each slot should have at most one appointment
    # The plan format is {slot_key: appointment_name}
    # No overlap means each slot has only one value, which is guaranteed by dict structure
    return True


def check_no_hallucinated_facts(world: World, plan: dict, params: dict) -> bool:
    """
    Check that provided answers match the fact database

    Params:
        fact_db: dict - the fact database
    """
    fact_db = params.get("fact_db", {})

    # For fact world, check if answer is in the fact_db
    answer = plan.get("answer", "")

    if not answer:
        return False

    # Check if answer matches any value in fact_db
    for key, value in fact_db.items():
        if answer.lower().strip() in value.lower().strip():
            return True

    return False


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_plan_entity_ids(plan: dict) -> list[str]:
    """Extract all entity IDs/names from a plan"""
    entity_ids = []

    for key, value in plan.items():
        if isinstance(value, list):
            entity_ids.extend(value)
        elif isinstance(value, str) and value and value != "null":
            entity_ids.append(value)

    return entity_ids


def _resolve_entity_id(world: World, entity_ref: str) -> str:
    """
    Resolve entity reference to canonical ID

    entity_ref can be:
    - An entity ID (e.g., "A1")
    - An entity name (e.g., "Biserica Neagră")

    Returns the canonical ID or the original ref if not found
    """
    # Check if it's already an ID
    if entity_ref in world.canonical_entities:
        return entity_ref

    # Try to find by name
    entity_ref_lower = entity_ref.lower().strip()

    for entity_id, entity in world.canonical_entities.items():
        if entity.name.lower().strip() == entity_ref_lower:
            return entity_id
        if entity_ref_lower in [alias.lower() for alias in entity.aliases]:
            return entity_id

    # Not found, return original
    return entity_ref


# ============================================================================
# Function Registry
# ============================================================================

CONSTRAINT_FUNCTIONS: dict[str, Callable[[World, dict, dict], bool]] = {
    # Travel
    "check_must_include_type": check_must_include_type,
    "check_max_outdoor_per_day": check_max_outdoor_per_day,
    "check_all_family_friendly": check_all_family_friendly,
    "check_budget_limit": check_budget_limit,
    "check_min_activities_per_day": check_min_activities_per_day,
    "check_max_activities_per_day": check_max_activities_per_day,

    # Schedule
    "check_max_appointments_per_day": check_max_appointments_per_day,
    "check_keep_high_priority": check_keep_high_priority,

    # Fact
    "check_answer_matches_context": check_answer_matches_context,

    # Recipe
    "check_all_vegetarian": check_all_vegetarian,
    "check_no_gluten": check_no_gluten,
    "check_no_lactose": check_no_lactose,
    "check_max_daily_calories": check_max_daily_calories,
    "check_all_meals_filled": check_all_meals_filled,

    # General goals
    "check_days_non_empty": check_days_non_empty,
    "check_no_duplicates": check_no_duplicates,
    "check_valid_entity_ids": check_valid_entity_ids,
    "check_no_slot_overlaps": check_no_slot_overlaps,
    "check_no_hallucinated_facts": check_no_hallucinated_facts,
}


def check_constraint(
    world: World,
    plan: dict,
    constraint_fn_name: str,
    params: dict
) -> bool:
    """
    Check a constraint by name

    Args:
        world: The world specification
        plan: The model's plan
        constraint_fn_name: Name of the constraint function
        params: Parameters for the constraint function

    Returns:
        True if constraint is satisfied, False otherwise
    """
    if constraint_fn_name not in CONSTRAINT_FUNCTIONS:
        raise ValueError(f"Unknown constraint function: {constraint_fn_name}")

    fn = CONSTRAINT_FUNCTIONS[constraint_fn_name]
    return fn(world, plan, params)
