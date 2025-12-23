"""
Constraint solver for Recipe world instances.

Uses backtracking to verify solvability with dietary and caloric constraints.
"""

from itertools import product
from typing import Any


def solve_recipe(world: Any) -> bool:
    """
    Try to find a valid meal plan using backtracking.

    This solver checks all constraints including:
    - Dietary restrictions (vegetarian, vegan, gluten-free, lactose-free)
    - Calorie ranges (min/max per day)
    - Meal ordering (lunch heaviest, dinner lightest)
    - No repetition (variety)
    - Prep time limits
    - Quick breakfast requirement

    Args:
        world: The recipe World instance

    Returns:
        True if at least one valid solution exists
    """
    dishes = world.payload.get("dishes", [])
    num_days = world.payload.get("num_days", 3)
    constraints = world.constraints

    # Group dishes by meal type
    by_type = {}
    for d in dishes:
        mt = d.get("type") or d.get("meal_type")
        if mt:
            by_type.setdefault(mt, []).append(d)

    meal_types = list(by_type.keys())
    if not meal_types:
        return True

    # Parse constraints
    dietary = {}
    min_cal = 0
    max_cal = float("inf")
    lunch_heaviest = False
    dinner_lightest = False
    no_repeat = False
    max_prep_time = float("inf")
    max_breakfast_prep = float("inf")
    max_high_calorie = float("inf")
    high_calorie_threshold = 400

    for c in constraints:
        cid = c.id
        params = c.params if hasattr(c, 'params') else {}

        if cid.startswith("C_DIET_") or "vegetarian" in cid.lower():
            # Check the check_fn to determine type
            check_fn = c.check_fn if hasattr(c, 'check_fn') else ""
            if "vegetarian" in check_fn:
                dietary["vegetarian"] = True
            elif "vegan" in check_fn:
                dietary["vegan"] = True
            elif "gluten" in check_fn:
                dietary["no_gluten"] = True
            elif "lactose" in check_fn:
                dietary["no_lactose"] = True

        elif cid == "C_CALORIE_RANGE":
            min_cal = params.get("min_calories", 0)
            max_cal = params.get("max_calories", float("inf"))

        elif cid == "C_LUNCH_HEAVY":
            lunch_heaviest = True

        elif cid == "C_DINNER_LIGHT":
            dinner_lightest = True

        elif cid == "C_NO_DUP":
            no_repeat = True

        elif cid == "C_PREP_TIME":
            max_prep_time = params.get("max_prep_time", float("inf"))

        elif cid == "C_QUICK_BREAKFAST":
            max_breakfast_prep = params.get("max_prep_time", 15)

        elif cid == "C_HIGH_CAL_LIMIT":
            max_high_calorie = params.get("max_high_calorie", float("inf"))
            high_calorie_threshold = params.get("calorie_threshold", 400)

    # Filter dishes by dietary constraints
    def dish_valid(d):
        if dietary.get("vegetarian") and not d.get("vegetarian"):
            return False
        if dietary.get("vegan") and not d.get("vegan"):
            return False
        if dietary.get("no_gluten") and d.get("contains_gluten"):
            return False
        if dietary.get("no_lactose") and d.get("contains_lactose"):
            return False
        return True

    valid_by_type = {}
    for mt, ds in by_type.items():
        valid_by_type[mt] = [d for d in ds if dish_valid(d)]
        if len(valid_by_type[mt]) < num_days:
            return False  # Not enough valid dishes

    # Also filter breakfast by prep time
    breakfast_key = None
    for mt in valid_by_type:
        if "mic_dejun" in mt or "breakfast" in mt:
            breakfast_key = mt
            break

    if breakfast_key and max_breakfast_prep < float("inf"):
        valid_by_type[breakfast_key] = [
            d for d in valid_by_type[breakfast_key]
            if d.get("prep_time_min", 0) <= max_breakfast_prep
        ]
        if len(valid_by_type[breakfast_key]) < num_days:
            return False

    def check_day(day_dishes, high_cal_so_far):
        """Check if a day's dishes satisfy daily constraints."""
        total_cal = sum(d.get("calories", 0) for d in day_dishes)
        total_prep = sum(d.get("prep_time_min", 0) for d in day_dishes)

        if total_cal < min_cal or total_cal > max_cal:
            return False, high_cal_so_far

        if total_prep > max_prep_time:
            return False, high_cal_so_far

        # Count high calorie dishes
        new_high_cal = high_cal_so_far
        for d in day_dishes:
            if d.get("calories", 0) > high_calorie_threshold:
                new_high_cal += 1

        if new_high_cal > max_high_calorie:
            return False, high_cal_so_far

        if lunch_heaviest:
            lunch_d = next((d for d in day_dishes if "pranz" in (d.get("type") or "") or "lunch" in (d.get("type") or "")), None)
            if lunch_d:
                lunch_cal = lunch_d.get("calories", 0)
                for d in day_dishes:
                    dtype = d.get("type") or ""
                    if "pranz" not in dtype and "lunch" not in dtype:
                        if d.get("calories", 0) >= lunch_cal:
                            return False, high_cal_so_far

        if dinner_lightest:
            dinner_d = next((d for d in day_dishes if "cina" in (d.get("type") or "") or "dinner" in (d.get("type") or "")), None)
            if dinner_d:
                dinner_cal = dinner_d.get("calories", 0)
                for d in day_dishes:
                    dtype = d.get("type") or ""
                    if "cina" not in dtype and "dinner" not in dtype:
                        if d.get("calories", 0) <= dinner_cal:
                            return False, high_cal_so_far

        return True, new_high_cal

    # Try to find valid assignments for all days using backtracking
    meal_types_sorted = sorted(valid_by_type.keys())

    def solve_day(day_idx, used_dishes, high_cal_count):
        if day_idx == num_days:
            return True  # All days solved!

        options_per_type = []
        for mt in meal_types_sorted:
            if no_repeat:
                available = [i for i, d in enumerate(valid_by_type[mt])
                            if (mt, i) not in used_dishes]
            else:
                available = list(range(len(valid_by_type[mt])))

            if not available:
                return False
            options_per_type.append([(mt, i) for i in available])

        for combo in product(*options_per_type):
            day_dishes = [valid_by_type[mt][i] for mt, i in combo]

            valid, new_high_cal = check_day(day_dishes, high_cal_count)
            if valid:
                new_used = used_dishes | set(combo) if no_repeat else used_dishes
                if solve_day(day_idx + 1, new_used, new_high_cal):
                    return True

        return False

    return solve_day(0, set(), 0)
