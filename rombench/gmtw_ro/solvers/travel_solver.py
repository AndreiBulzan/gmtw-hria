"""
Constraint solver for Travel world instances.

Uses backtracking/combinatorial search to verify solvability.
"""

from itertools import combinations, permutations
from typing import Any


def solve_travel(world: Any) -> bool:
    """
    Try to find a valid travel plan using brute force backtracking.

    This is a comprehensive solver that checks all constraints including:
    - Must include types (monument, museum, etc.)
    - Must include specific attractions
    - Budget limits (total and per-day)
    - Duration limits per day
    - Family-friendly requirements
    - Indoor/outdoor restrictions (first day indoor, last day outdoor, max outdoor per day)
    - Type diversity requirements
    - No duplicates

    Args:
        world: The travel World instance

    Returns:
        True if at least one valid solution exists
    """
    attrs = world.payload.get("attractions", [])
    num_days = world.payload.get("num_days", 2)
    constraints = world.constraints

    # Parse constraints
    must_types = set()
    must_specific = set()
    family_required = False
    max_budget = float("inf")
    max_budget_per_day = float("inf")
    max_hours_per_day = float("inf")
    max_outdoor_per_day = float("inf")
    first_indoor = False
    last_outdoor = False
    min_types = 0
    no_duplicates = True  # Always enforced

    for c in constraints:
        cid = c.id
        params = c.params if hasattr(c, 'params') else {}

        if cid == "C_MUST_MONUMENT":
            must_types.add("monument")
        elif cid == "C_MUST_MUSEUM":
            must_types.add("muzeu")
        elif cid in ("C_FAMILY", "C_FAMILY_FRIENDLY"):
            family_required = True
        elif cid in ("C_BUDGET", "C_BUDGET_TOTAL"):
            max_budget = params.get("max_budget", float("inf"))
        elif cid == "C_BUDGET_DAILY":
            max_budget_per_day = params.get("max_budget_per_day", float("inf"))
        elif cid == "C_MAX_DURATION":
            max_hours_per_day = params.get("max_hours", float("inf"))
        elif cid == "C_MAX_OUTDOOR":
            max_outdoor_per_day = params.get("max_outdoor", float("inf"))
        elif cid == "C_FIRST_DAY":
            first_indoor = params.get("indoor_only", False)
        elif cid == "C_LAST_DAY":
            last_outdoor = params.get("must_have_outdoor", False)
        elif cid == "C_MUST_SPECIFIC":
            name = params.get("entity_name")
            if name:
                must_specific.add(name)
        elif cid == "C_DIVERSITY":
            min_types = params.get("min_types", 0)

    # Filter by family-friendly if required
    valid_attrs = attrs
    if family_required:
        valid_attrs = [a for a in attrs if a.get("family_friendly")]

    if len(valid_attrs) < num_days:
        return False

    # Check if must_specific attractions are available
    valid_names = {a.get("name") for a in valid_attrs}
    if not must_specific.issubset(valid_names):
        return False

    # Check if must types are available
    valid_types = {a.get("type") for a in valid_attrs}
    if not must_types.issubset(valid_types):
        return False

    # Try all combinations of num_days attractions
    for combo in combinations(range(len(valid_attrs)), num_days):
        selected = [valid_attrs[i] for i in combo]

        # Check must include types
        selected_types = set(a.get("type") for a in selected)
        if not must_types.issubset(selected_types):
            continue

        # Check type diversity
        if len(selected_types) < min_types:
            continue

        # Check must include specific
        selected_names = set(a.get("name") for a in selected)
        if not must_specific.issubset(selected_names):
            continue

        # Check total budget
        total_cost = sum(a.get("cost_lei", 0) for a in selected)
        if total_cost > max_budget:
            continue

        # Check day-specific constraints (try all orderings)
        for perm in permutations(selected):
            valid = True

            # Check first day indoor only
            if first_indoor and not perm[0].get("indoor"):
                valid = False
                continue

            # Check last day has outdoor
            if last_outdoor and perm[-1].get("indoor"):
                valid = False
                continue

            # Check per-day constraints
            for day_idx, day_attr in enumerate(perm):
                # Per-day budget
                if day_attr.get("cost_lei", 0) > max_budget_per_day:
                    valid = False
                    break

                # Per-day duration
                if day_attr.get("duration_hours", 0) > max_hours_per_day:
                    valid = False
                    break

                # Outdoor count per day (with single attraction per day, max is 1 or 0)
                if not day_attr.get("indoor") and max_outdoor_per_day < 1:
                    valid = False
                    break

            if valid:
                return True  # Found a valid solution!

    return False
