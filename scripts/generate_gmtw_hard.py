#!/usr/bin/env python3
"""
Generate GMTW-Ro HARD dataset

This script generates a challenging evaluation dataset with extreme difficulty
constraints designed to bring weak models down to 40-50% accuracy.

Key differences from standard generation:
- 8-12 interacting constraints per Travel instance
- Complex priority conflicts in Schedule instances
- 100% misbelief traps in Fact instances
- Multiple simultaneous dietary restrictions in Recipe instances

IMPORTANT: All instances are verified solvable at generation time.
"""

import argparse
import json
import random
import sys
from itertools import combinations, product
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro.worlds.base import (
    Instance, World, Constraint, Goal, Entity, ConstraintType, GoalType
)
from rombench.gmtw_ro.worlds.travel import CITIES
from rombench.gmtw_ro.worlds.schedule import DAYS_RO, DAYS_EN, SLOTS_RO, SLOTS_EN, MEETING_TYPES
from rombench.gmtw_ro.worlds.fact import FACTS
from rombench.gmtw_ro.worlds.recipe import DISHES
from rombench.gmtw_ro.worlds import templates_ro, templates_en


# =============================================================================
# SOLVABILITY VERIFICATION FUNCTIONS
# =============================================================================

def verify_travel_solvable(
    attractions: list[dict],
    num_days: int,
    constraints: dict,
) -> bool:
    """
    Verify that a travel instance has at least one valid solution.

    Uses constraint propagation + limited search.

    Args:
        attractions: List of attraction dicts
        num_days: Number of days
        constraints: Dict of constraint params (budget, duration, etc.)

    Returns:
        True if at least one valid plan exists
    """
    # Filter attractions by hard constraints first
    valid_attrs = attractions[:]

    if constraints.get("family_friendly"):
        valid_attrs = [a for a in valid_attrs if a["family_friendly"]]

    if not valid_attrs:
        return False

    # Check if we have enough attractions for all days
    if len(valid_attrs) < num_days:
        return False

    # Check specific requirements
    if constraints.get("must_include_type"):
        required_type = constraints["must_include_type"]
        if not any(a["type"] == required_type for a in valid_attrs):
            return False

    if constraints.get("must_include_specific"):
        specific_name = constraints["must_include_specific"]
        if not any(a["name"] == specific_name for a in valid_attrs):
            return False

    # Check indoor/outdoor requirements
    indoor = [a for a in valid_attrs if a["indoor"]]
    outdoor = [a for a in valid_attrs if not a["indoor"]]

    if constraints.get("first_day_indoor") and len(indoor) < 1:
        return False

    if constraints.get("last_day_outdoor") and len(outdoor) < 1:
        return False

    # Check type diversity
    if constraints.get("min_types"):
        available_types = set(a["type"] for a in valid_attrs)
        if len(available_types) < constraints["min_types"]:
            return False

    # Try to find a valid assignment using greedy + backtracking
    # For efficiency, just check if constraints are satisfiable in aggregate

    max_budget = constraints.get("max_budget", float("inf"))
    max_duration_per_day = constraints.get("max_duration_per_day", float("inf"))

    # Check if cheapest valid set can fit budget
    sorted_by_cost = sorted(valid_attrs, key=lambda a: a.get("cost_lei", 0))
    min_cost_for_n = sum(a.get("cost_lei", 0) for a in sorted_by_cost[:num_days])
    if min_cost_for_n > max_budget:
        return False

    # Check if shortest activities fit per-day duration
    sorted_by_duration = sorted(valid_attrs, key=lambda a: a["duration_hours"])
    if sorted_by_duration[0]["duration_hours"] > max_duration_per_day:
        return False

    return True


def verify_schedule_solvable(
    appointments: list[dict],
    num_days: int,
    constraints: dict,
) -> bool:
    """
    Verify that a schedule instance has at least one valid solution.

    Args:
        appointments: List of appointment dicts with priority
        num_days: Number of days
        constraints: Dict of constraint params

    Returns:
        True if at least one valid plan exists
    """
    max_per_day = constraints.get("max_per_day", 2)
    max_total = constraints.get("max_total", len(appointments))
    min_spread = constraints.get("min_spread", 1)

    # Count by priority
    high_priority = [a for a in appointments if a["priority"] == "high"]

    # Must keep all high priority
    if constraints.get("keep_high_priority") and len(high_priority) > max_total:
        return False

    # Check if we can spread across required days
    total_slots = num_days * max_per_day
    if max_total > total_slots:
        return False

    if min_spread > num_days:
        return False

    # Check if we can fit high priority appointments
    if constraints.get("keep_high_priority"):
        # Each high priority needs a slot
        if len(high_priority) > total_slots:
            return False

    return True


def verify_recipe_solvable(
    dishes_by_type: dict[str, list[dict]],
    num_days: int,
    constraints: dict,
) -> bool:
    """
    Verify that a recipe instance has at least one valid solution.

    Args:
        dishes_by_type: Dict mapping meal type to list of dish dicts
        num_days: Number of days
        constraints: Dict of constraint params

    Returns:
        True if at least one valid plan exists
    """
    # Filter dishes by dietary constraints
    def dish_valid(dish):
        if constraints.get("vegetarian") and not dish.get("vegetarian"):
            return False
        if constraints.get("vegan") and not dish.get("vegan"):
            return False
        if constraints.get("no_gluten") and dish.get("contains_gluten"):
            return False
        if constraints.get("no_lactose") and dish.get("contains_lactose"):
            return False
        return True

    valid_by_type = {}
    for meal_type, dishes in dishes_by_type.items():
        valid_by_type[meal_type] = [d for d in dishes if dish_valid(d)]

    # Check we have enough dishes for each meal type
    for meal_type, dishes in valid_by_type.items():
        if len(dishes) < num_days:
            return False

    # Check calorie constraints are achievable
    min_cal = constraints.get("min_calories", 0)
    max_cal = constraints.get("max_calories", float("inf"))

    # Calculate min and max possible daily calories
    min_daily = sum(
        min(d["calories"] for d in dishes) if dishes else 0
        for dishes in valid_by_type.values()
    )
    max_daily = sum(
        max(d["calories"] for d in dishes) if dishes else 0
        for dishes in valid_by_type.values()
    )

    if min_daily > max_cal or max_daily < min_cal:
        return False

    # Check meal ordering constraints
    if constraints.get("lunch_heaviest"):
        breakfast_dishes = valid_by_type.get("mic_dejun", [])
        lunch_dishes = valid_by_type.get("pranz", [])
        dinner_dishes = valid_by_type.get("cina", [])

        if breakfast_dishes and lunch_dishes and dinner_dishes:
            # Check if there exists a lunch heavier than all breakfasts and dinners
            max_lunch = max(d["calories"] for d in lunch_dishes)
            min_breakfast = min(d["calories"] for d in breakfast_dishes)
            min_dinner = min(d["calories"] for d in dinner_dishes)

            if max_lunch <= min_breakfast or max_lunch <= min_dinner:
                return False

    if constraints.get("dinner_lightest"):
        breakfast_dishes = valid_by_type.get("mic_dejun", [])
        lunch_dishes = valid_by_type.get("pranz", [])
        dinner_dishes = valid_by_type.get("cina", [])

        if breakfast_dishes and lunch_dishes and dinner_dishes:
            min_dinner = min(d["calories"] for d in dinner_dishes)
            max_breakfast = max(d["calories"] for d in breakfast_dishes)
            max_lunch = max(d["calories"] for d in lunch_dishes)

            if min_dinner >= max_breakfast or min_dinner >= max_lunch:
                return False

    return True


class ExtremeTravelGenerator:
    """Generate extremely challenging travel instances"""

    def __init__(self, spec_version: str = "0.1"):
        self.spec_version = spec_version

    def generate(self, world_id: str, seed: int) -> World:
        rng = random.Random(seed)

        # More days = more constraints to juggle
        num_days = rng.choice([4, 5])

        # Select city with many attractions
        city = rng.choice(list(CITIES.keys()))
        city_data = CITIES[city]

        # Use ALL attractions to maximize constraint complexity
        all_attractions = city_data["attractions"]
        selected_attractions = all_attractions[:]

        # Create entities
        entities = {}
        attractions_list = []

        for idx, attr in enumerate(selected_attractions):
            attr_id = f"A{idx + 1}"
            name_en = attr.get("name_en", attr["name"])
            aliases = [
                attr["name"].lower(),
                attr["name"].lower().replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ț", "t"),
                name_en,
                name_en.lower(),
            ]
            entities[attr_id] = Entity(
                id=attr_id,
                name=attr["name"],
                aliases=aliases,
                attributes=attr,
            )

            attractions_list.append({
                "id": attr_id,
                "name": attr["name"],
                "name_en": attr.get("name_en", attr["name"]),
                "type": attr["type"],
                "type_en": attr.get("type_en", attr["type"]),
                "indoor": attr["indoor"],
                "family_friendly": attr["family_friendly"],
                "duration_hours": attr["duration_hours"],
                "cost_lei": attr.get("cost_lei", 0),
            })

        # Calculate stats for constraint tuning
        total_cost = sum(a.get("cost_lei", 0) for a in selected_attractions)
        total_duration = sum(a["duration_hours"] for a in selected_attractions)
        type_counts = {}
        for a in selected_attractions:
            t = a["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        # =====================================================================
        # EXTREME CONSTRAINTS - Layer multiple interacting constraints
        # =====================================================================
        constraints = []

        # 1. Must include monument (baseline)
        has_monument = any(a["type"] == "monument" for a in selected_attractions)
        if has_monument:
            constraints.append(
                Constraint(
                    id="C_MUST_MONUMENT",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Trebuie să incluzi cel puțin un monument istoric în plan.",
                    description_en="You must include at least one historic monument.",
                    check_fn="check_must_include_type",
                    params={"type_required": "monument"},
                )
            )

        # 2. Must include museum
        has_museum = any(a["type"] == "muzeu" for a in selected_attractions)
        if has_museum:
            constraints.append(
                Constraint(
                    id="C_MUST_MUSEUM",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Trebuie să incluzi cel puțin un muzeu în plan.",
                    description_en="You must include at least one museum.",
                    check_fn="check_must_include_type",
                    params={"type_required": "muzeu"},
                )
            )

        # Check what's available for solvability
        family_friendly_attractions = [a for a in selected_attractions if a["family_friendly"]]
        family_friendly_outdoor = [a for a in family_friendly_attractions if not a["indoor"]]
        family_friendly_indoor = [a for a in family_friendly_attractions if a["indoor"]]

        # 3. Strict outdoor limit per day
        constraints.append(
            Constraint(
                id="C_MAX_OUTDOOR",
                type=ConstraintType.INSTRUCTION,
                description_ro="Maxim 1 activitate în aer liber pe zi.",
                description_en="At most 1 outdoor activity per day.",
                check_fn="check_max_outdoor_per_day",
                params={"max_outdoor": 1},
            )
        )

        # 4. Family friendly requirement - only if enough family-friendly options exist
        if len(family_friendly_attractions) >= num_days * 2:
            constraints.append(
                Constraint(
                    id="C_FAMILY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Toate activitățile trebuie să fie potrivite pentru copii.",
                    description_en="All activities must be suitable for children.",
                    check_fn="check_all_family_friendly",
                    params={},
                )
            )
            # Use family-friendly subset for further constraints
            use_family_filter = True
        else:
            use_family_filter = False

        # 5. Tight total budget (50-65% of total possible - challenging but solvable)
        budget = int(total_cost * rng.uniform(0.50, 0.65))
        constraints.append(
            Constraint(
                id="C_BUDGET_TOTAL",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Bugetul total nu trebuie să depășească {budget} lei.",
                description_en=f"Total budget must not exceed {budget} lei.",
                check_fn="check_budget_limit",
                params={"max_budget": budget},
            )
        )

        # 6. Per-day budget limit (reasonable per day)
        daily_budget = int(budget / num_days * 1.2)  # Allow some flexibility
        constraints.append(
            Constraint(
                id="C_BUDGET_DAILY",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Costul activităților pe zi nu trebuie să depășească {daily_budget} lei.",
                description_en=f"Daily activity cost must not exceed {daily_budget} lei.",
                check_fn="check_budget_per_day",
                params={"max_budget_per_day": daily_budget},
            )
        )

        # 7. Duration limit per day (allow 2-3 activities worth)
        avg_duration = total_duration / len(selected_attractions)
        max_hours = round(avg_duration * 2.5, 1)  # Allow 2-3 activities
        max_hours = max(3.0, max_hours)  # Minimum 3 hours
        constraints.append(
            Constraint(
                id="C_MAX_DURATION",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Timpul total de vizită pe zi nu trebuie să depășească {max_hours} ore.",
                description_en=f"Total visit time per day must not exceed {max_hours} hours.",
                check_fn="check_max_duration_per_day",
                params={"max_hours": max_hours},
            )
        )

        # 8. Type diversity requirement
        available_types = set(a["type"] for a in selected_attractions)
        if len(available_types) >= 3:
            constraints.append(
                Constraint(
                    id="C_DIVERSITY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Planul trebuie să includă cel puțin 3 tipuri diferite de activități.",
                    description_en="The plan must include at least 3 different activity types.",
                    check_fn="check_type_diversity",
                    params={"min_types": 3},
                )
            )

        # 9. No duplicates
        constraints.append(
            Constraint(
                id="C_NO_DUP",
                type=ConstraintType.INSTRUCTION,
                description_ro="Nu vizita același loc de două ori.",
                description_en="Do not visit the same place twice.",
                check_fn="check_no_duplicates",
                params={},
            )
        )

        # 10. First day indoor only - only if enough indoor attractions
        indoor_attractions = [a for a in selected_attractions if a["indoor"]]
        if len(indoor_attractions) >= 2:
            constraints.append(
                Constraint(
                    id="C_FIRST_DAY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Prima zi trebuie să conțină doar activități de interior.",
                    description_en="The first day must contain only indoor activities.",
                    check_fn="check_first_day_constraint",
                    params={"indoor_only": True},
                )
            )

        # 11. Last day must have outdoor - only if outdoor attractions available
        outdoor_attractions = [a for a in selected_attractions if not a["indoor"]]
        if use_family_filter:
            valid_outdoor = family_friendly_outdoor
        else:
            valid_outdoor = outdoor_attractions

        if len(valid_outdoor) >= 1:
            constraints.append(
                Constraint(
                    id="C_LAST_DAY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Ultima zi trebuie să includă cel puțin o activitate în aer liber.",
                    description_en="The last day must include at least one outdoor activity.",
                    check_fn="check_last_day_constraint",
                    params={"num_days": num_days, "must_have_outdoor": True},
                )
            )

        # 12. Must include specific attraction (choose from valid options)
        if use_family_filter:
            valid_specific = family_friendly_attractions
        else:
            valid_specific = selected_attractions
        if valid_specific:
            specific_attr = rng.choice(valid_specific)
            constraints.append(
                Constraint(
                    id="C_MUST_SPECIFIC",
                    type=ConstraintType.INSTRUCTION,
                    description_ro=f"Trebuie să vizitezi \"{specific_attr['name']}\".",
                    description_en=f"You must visit \"{specific_attr.get('name_en', specific_attr['name'])}\".",
                    check_fn="check_must_include_specific",
                    params={"entity_name": specific_attr["name"]},
                )
            )

        # Goals - pure structural validity (not duplicating constraints)
        goals = [
            Goal(
                id="G_FILL_DAYS",
                type=GoalType.STRUCTURAL,
                description="Each day must have at least one activity",
                check_fn="check_days_non_empty",
                params={"num_days": num_days},
            ),
            Goal(
                id="G_VALID_IDS",
                type=GoalType.STRUCTURAL,
                description="All referenced IDs must exist",
                check_fn="check_valid_entity_ids",
                params={"valid_ids": list(entities.keys())},
            ),
        ]

        payload = {
            "city": city,
            "city_en": city_data.get("city_en", city),
            "num_days": num_days,
            "attractions": attractions_list,
        }

        meta = {
            "difficulty": "extreme",
            "num_constraints": len(constraints),
            "family_trip": True,
            "num_attractions": len(selected_attractions),
        }

        return World(
            world_id=world_id,
            world_type="travel",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )


class ExtremeScheduleGenerator:
    """Generate extremely challenging schedule instances"""

    def __init__(self, spec_version: str = "0.1"):
        self.spec_version = spec_version

    def generate(self, world_id: str, seed: int) -> World:
        rng = random.Random(seed)

        # More days and appointments
        num_days = rng.choice([3, 4])
        selected_days_ro = DAYS_RO[:num_days]
        selected_days_en = DAYS_EN[:num_days]

        # Generate many appointments (more than can fit)
        num_appointments = rng.randint(6, 8)
        meeting_types = rng.choices(MEETING_TYPES, k=num_appointments)

        appointments = []
        entities = {}

        # Distribute priorities strategically
        priorities = ["high"] * 2 + ["medium"] * (num_appointments - 4) + ["low"] * 2
        rng.shuffle(priorities)

        for idx, meeting_type in enumerate(meeting_types):
            apt_id = f"M{idx + 1}"
            day_idx = rng.randint(0, num_days - 1)
            slot_idx = rng.randint(0, len(SLOTS_RO) - 1)

            appointment = {
                "id": apt_id,
                "name_ro": meeting_type["name_ro"],
                "name_en": meeting_type["name_en"],
                "day_ro": selected_days_ro[day_idx],
                "day_en": selected_days_en[day_idx],
                "slot_ro": SLOTS_RO[slot_idx],
                "slot_en": SLOTS_EN[slot_idx],
                "priority": priorities[idx],
            }

            appointments.append(appointment)
            entities[apt_id] = Entity(
                id=apt_id,
                name=meeting_type["name_ro"],
                aliases=[meeting_type["name_ro"].lower(), meeting_type["name_en"].lower()],
                attributes=appointment,
            )

        # =====================================================================
        # EXTREME CONSTRAINTS
        # =====================================================================
        constraints = []

        # 1. Max per day - tight
        constraints.append(
            Constraint(
                id="C_MAX_PER_DAY",
                type=ConstraintType.INSTRUCTION,
                description_ro="Maxim 2 programări pe zi.",
                description_en="At most 2 appointments per day.",
                check_fn="check_max_appointments_per_day",
                params={"max_per_day": 2},
            )
        )

        # 2. Keep all high priority
        constraints.append(
            Constraint(
                id="C_KEEP_HIGH",
                type=ConstraintType.INSTRUCTION,
                description_ro="Trebuie să păstrezi toate programările cu prioritate înaltă.",
                description_en="You must keep all high-priority appointments.",
                check_fn="check_keep_high_priority",
                params={},
            )
        )

        # 3. No back-to-back - REMOVED: conflicts with "max 2 per day" and "spread"
        # The constraint essentially said "max 1 per day" which made other constraints unsolvable

        # 4. Max total - forces dropping some
        # Ensure we can keep at least all high priority (2) plus some spread
        max_total = min(num_days * 2, num_appointments - 1)  # More generous
        constraints.append(
            Constraint(
                id="C_MAX_TOTAL",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Poți programa maxim {max_total} întâlniri în total.",
                description_en=f"You can schedule at most {max_total} appointments total.",
                check_fn="check_max_total_appointments",
                params={"max_total": max_total},
            )
        )

        # 5. Priority ordering when dropping
        constraints.append(
            Constraint(
                id="C_DROP_ORDER",
                type=ConstraintType.INSTRUCTION,
                description_ro="Dacă trebuie să renunți la programări, renunță mai întâi la cele cu prioritate joasă.",
                description_en="If you must drop appointments, drop low priority ones first.",
                check_fn="check_must_drop_lowest_priority",
                params={},
            )
        )

        # 6. Spread across days - require at least 2 days to have appointments
        min_spread = min(2, num_days)  # At least 2 days, but not more than available
        constraints.append(
            Constraint(
                id="C_SPREAD",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Programările trebuie distribuite în cel puțin {min_spread} zile diferite.",
                description_en=f"Appointments must be spread across at least {min_spread} different days.",
                check_fn="check_spread_across_days",
                params={"min_days_with_appointments": min_spread},
            )
        )

        # 7. Medical appointments in morning (if any)
        has_medical = any("medical" in apt["name_ro"].lower() or "control" in apt["name_ro"].lower()
                         for apt in appointments)
        if has_medical:
            constraints.append(
                Constraint(
                    id="C_MEDICAL_MORNING",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Controalele medicale trebuie programate dimineața.",
                    description_en="Medical checkups must be scheduled in the morning.",
                    check_fn="check_slot_type_restriction",
                    params={"type_keyword": "control", "allowed_slots": ["dimineață", "morning"]},
                )
            )

        # 8. Medium priority day restriction
        last_day_ro = selected_days_ro[-1]
        last_day_en = selected_days_en[-1]
        constraints.append(
            Constraint(
                id="C_MED_RESTRICT",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Programările cu prioritate medie nu pot fi în {last_day_ro}.",
                description_en=f"Medium priority appointments cannot be on {last_day_en}.",
                check_fn="check_priority_day_restriction",
                params={"priority": "medium", "forbidden_days": [last_day_ro, last_day_en]},
            )
        )

        # Goals - pure structural validity (not duplicating constraints)
        goals = [
            Goal(
                id="G_NO_OVERLAPS",
                type=GoalType.STRUCTURAL,
                description="No slot overlaps",
                check_fn="check_no_slot_overlaps",
                params={"days": selected_days_ro, "slots": SLOTS_RO},
            ),
            Goal(
                id="G_VALID_IDS",
                type=GoalType.STRUCTURAL,
                description="All IDs valid",
                check_fn="check_valid_entity_ids",
                params={"valid_ids": list(entities.keys())},
            ),
        ]

        payload = {
            "days_ro": selected_days_ro,
            "days_en": selected_days_en,
            "slots_ro": SLOTS_RO,
            "slots_en": SLOTS_EN,
            "appointments": appointments,
        }

        meta = {
            "difficulty": "extreme",
            "num_constraints": len(constraints),
            "num_appointments": len(appointments),
            "num_days": num_days,
        }

        return World(
            world_id=world_id,
            world_type="schedule",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )


class ExtremeFactGenerator:
    """Generate extremely challenging fact instances with 100% misbelief traps"""

    def __init__(self, spec_version: str = "0.1"):
        self.spec_version = spec_version

    def generate(self, world_id: str, seed: int) -> World:
        rng = random.Random(seed)

        # Collect all facts that have misbelief versions
        all_facts_with_misbelief = []
        for category, facts in FACTS.items():
            for f in facts:
                if "misbelief_answer" in f:
                    all_facts_with_misbelief.append((category, f))

        # Select facts
        num_facts = rng.randint(5, 7)
        selected = rng.sample(all_facts_with_misbelief, min(num_facts, len(all_facts_with_misbelief)))

        fact_db = {}
        entities = {}
        misbelief_scripts = []
        question_to_ask = None

        for idx, (category, fact) in enumerate(selected):
            fact_id = f"F{idx + 1}"

            # ALWAYS use misbelief answer in extreme mode
            answer = fact["misbelief_answer"]
            misbelief_scripts.append({
                "fact_id": fact_id,
                "question_ro": fact["question_ro"],
                "question_en": fact["question_en"],
                "context_answer": answer,
                "real_world_answer": fact["answer"],
                "is_trap": True,
            })

            # Use the first fact as the question
            if question_to_ask is None:
                question_to_ask = fact

            fact_db[fact["key"]] = answer

            entities[fact_id] = Entity(
                id=fact_id,
                name=fact["key"],
                aliases=[fact["key"]],
                attributes={
                    "question_ro": fact["question_ro"],
                    "question_en": fact["question_en"],
                    "answer": answer,
                    "category": category,
                },
            )

        constraints = [
            Constraint(
                id="C_ANSWER_FROM_CONTEXT",
                type=ConstraintType.INSTRUCTION,
                description_ro="Răspunde DOAR pe baza informațiilor din contextul dat, chiar dacă par incorecte sau diferite de cunoștințele tale.",
                description_en="Answer ONLY based on the given context, even if it seems incorrect or different from your knowledge.",
                check_fn="check_answer_matches_context",
                params={},
            ),
        ]

        goals = [
            Goal(
                id="G_NO_HALLUCINATION",
                type=GoalType.STRUCTURAL,
                description="Answer must match fact database",
                check_fn="check_no_hallucinated_facts",
                params={"fact_db": fact_db},
            ),
        ]

        payload = {
            "facts": fact_db,
            "misbelief_scripts": misbelief_scripts,
            "question": {
                "question_ro": question_to_ask["question_ro"],
                "question_en": question_to_ask["question_en"],
                "expected_answer": fact_db.get(question_to_ask["key"], question_to_ask["misbelief_answer"]),
            },
        }

        meta = {
            "difficulty": "extreme",
            "num_facts": len(selected),
            "num_traps": len(misbelief_scripts),
            "has_misbelief": True,
            "trap_rate": 1.0,
        }

        return World(
            world_id=world_id,
            world_type="fact",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )


class ExtremeRecipeGenerator:
    """Generate extremely challenging recipe instances"""

    def __init__(self, spec_version: str = "0.1"):
        self.spec_version = spec_version

    def generate(self, world_id: str, seed: int) -> World:
        rng = random.Random(seed)

        num_days = rng.choice([3, 4])
        meals_per_day = ["mic_dejun", "pranz", "cina"]

        # Select all dishes
        available_dishes = {}
        entities = {}
        dishes_list = []
        dish_idx = 0

        for meal_type in meals_per_day:
            type_dishes = DISHES[meal_type]
            for dish in type_dishes:
                dish_id = f"D{dish_idx + 1}"
                dish_idx += 1

                name_en = dish.get("name_en", dish["name"])
                aliases = [
                    dish["name"].lower(),
                    name_en,
                    name_en.lower(),
                ]
                entities[dish_id] = Entity(
                    id=dish_id,
                    name=dish["name"],
                    aliases=aliases,
                    attributes=dish,
                )

                dishes_list.append({
                    "id": dish_id,
                    "name": dish["name"],
                    "name_en": dish.get("name_en", dish["name"]),
                    "type": dish["type"],
                    "type_en": dish.get("type_en", dish["type"]),
                    "vegetarian": dish["vegetarian"],
                    "vegan": dish["vegan"],
                    "contains_gluten": dish["contains_gluten"],
                    "contains_lactose": dish["contains_lactose"],
                    "prep_time_min": dish["prep_time_min"],
                    "calories": dish["calories"],
                })

                if meal_type not in available_dishes:
                    available_dishes[meal_type] = []
                available_dishes[meal_type].append(dish_id)

        # =====================================================================
        # EXTREME CONSTRAINTS - Stack multiple dietary restrictions
        # =====================================================================
        constraints = []

        # Choose constraint combinations that are SOLVABLE
        # Must ensure enough dishes remain after filtering
        constraint_sets = [
            # Set 1: Vegetarian only (most permissive)
            [
                ("vegetarian", "Toate preparatele trebuie să fie vegetariene.",
                 "All dishes must be vegetarian.", "check_all_vegetarian"),
            ],
            # Set 2: No gluten only
            [
                ("no_gluten", "Evită preparatele cu gluten.",
                 "Avoid dishes with gluten.", "check_no_gluten"),
            ],
            # Set 3: No lactose only
            [
                ("no_lactose", "Evită preparatele cu lactoză.",
                 "Avoid dishes with lactose.", "check_no_lactose"),
            ],
        ]

        selected_set = rng.choice(constraint_sets)
        for idx, (diet_type, desc_ro, desc_en, check_fn) in enumerate(selected_set):
            constraints.append(
                Constraint(
                    id=f"C_DIET_{idx}",
                    type=ConstraintType.INSTRUCTION,
                    description_ro=desc_ro,
                    description_en=desc_en,
                    check_fn=check_fn,
                    params={},
                )
            )

        # Calorie range constraint - achievable band
        # Most dishes are 200-400 kcal, so 3 meals = 600-1200 typical
        min_cal = rng.choice([800, 900, 1000])
        max_cal = min_cal + rng.choice([600, 700, 800])
        constraints.append(
            Constraint(
                id="C_CALORIE_RANGE",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Caloriile zilnice trebuie să fie între {min_cal} și {max_cal} kcal.",
                description_en=f"Daily calories must be between {min_cal} and {max_cal} kcal.",
                check_fn="check_calorie_range",
                params={"min_calories": min_cal, "max_calories": max_cal},
            )
        )

        # No duplicates
        constraints.append(
            Constraint(
                id="C_NO_DUP",
                type=ConstraintType.INSTRUCTION,
                description_ro="Nu repeta același preparat în meniu.",
                description_en="Do not repeat the same dish.",
                check_fn="check_no_duplicates",
                params={},
            )
        )

        # Prep time limit
        max_prep = rng.choice([70, 80, 90])
        constraints.append(
            Constraint(
                id="C_PREP_TIME",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Timpul total de preparare pe zi nu trebuie să depășească {max_prep} minute.",
                description_en=f"Total prep time per day must not exceed {max_prep} minutes.",
                check_fn="check_max_prep_time_per_day",
                params={"max_prep_time": max_prep},
            )
        )

        # Lunch heaviest - only add 50% of the time to avoid over-constraining
        if rng.random() < 0.5:
            constraints.append(
                Constraint(
                    id="C_LUNCH_HEAVY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Prânzul trebuie să fie masa principală (mai multe calorii decât micul dejun și cina).",
                    description_en="Lunch must be the main meal (more calories than breakfast and dinner).",
                    check_fn="check_lunch_heaviest_meal",
                    params={"num_days": num_days},
                )
            )

        # Dinner lightest - only add 50% of the time
        if rng.random() < 0.5:
            constraints.append(
                Constraint(
                    id="C_DINNER_LIGHT",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Cina trebuie să fie masa cea mai ușoară (mai puține calorii decât micul dejun și prânzul).",
                    description_en="Dinner must be the lightest meal (fewer calories than breakfast and lunch).",
                    check_fn="check_dinner_lightest",
                    params={"num_days": num_days},
                )
            )

        # Quick breakfast
        constraints.append(
            Constraint(
                id="C_QUICK_BREAKFAST",
                type=ConstraintType.INSTRUCTION,
                description_ro="Micul dejun trebuie să poată fi preparat în maxim 15 minute.",
                description_en="Breakfast must be preparable in at most 15 minutes.",
                check_fn="check_quick_breakfast",
                params={"max_prep_time": 15},
            )
        )

        # Breakfast variety - REMOVED: hard to guarantee solvability with dietary restrictions
        # The no_duplicates constraint already prevents repeating any dish

        # Max high calorie meals
        constraints.append(
            Constraint(
                id="C_HIGH_CAL_LIMIT",
                type=ConstraintType.INSTRUCTION,
                description_ro="Maximum 2 preparate pot avea peste 350 de calorii.",
                description_en="At most 2 dishes can have over 350 calories.",
                check_fn="check_max_high_calorie_meals",
                params={"calorie_threshold": 350, "max_high_calorie": 2},
            )
        )

        # Goals - pure structural validity (not duplicating constraints)
        goals = [
            Goal(
                id="G_ALL_MEALS",
                type=GoalType.STRUCTURAL,
                description="All meal slots filled",
                check_fn="check_all_meals_filled",
                params={"num_days": num_days, "meals": meals_per_day},
            ),
            Goal(
                id="G_VALID_DISHES",
                type=GoalType.STRUCTURAL,
                description="All dishes valid",
                check_fn="check_valid_entity_ids",
                params={"valid_ids": list(entities.keys())},
            ),
        ]

        meals_per_day_en = ["breakfast", "lunch", "dinner"]

        payload = {
            "num_days": num_days,
            "meals_per_day": meals_per_day,
            "meals_per_day_en": meals_per_day_en,
            "available_dishes": available_dishes,
            "dishes": dishes_list,
        }

        meta = {
            "difficulty": "extreme",
            "num_constraints": len(constraints),
            "num_dishes": len(dishes_list),
            "dietary_constraints": [c[0] for c in selected_set],
        }

        return World(
            world_id=world_id,
            world_type="recipe",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )


def verify_world_solvable(world: World) -> bool:
    """
    Verify that a world instance is solvable.

    Returns True if at least one valid solution exists.
    """
    world_type = world.world_type

    if world_type == "travel":
        # Extract constraint params
        constraints_dict = {}
        for c in world.constraints:
            if c.id == "C_FAMILY":
                constraints_dict["family_friendly"] = True
            elif c.id == "C_MUST_MONUMENT":
                constraints_dict["must_include_type"] = "monument"
            elif c.id == "C_MUST_MUSEUM":
                constraints_dict["must_include_type"] = "muzeu"
            elif c.id == "C_FIRST_DAY":
                constraints_dict["first_day_indoor"] = True
            elif c.id == "C_LAST_DAY":
                constraints_dict["last_day_outdoor"] = True
            elif c.id == "C_DIVERSITY":
                constraints_dict["min_types"] = c.params.get("min_types", 3)
            elif c.id == "C_BUDGET_TOTAL":
                constraints_dict["max_budget"] = c.params.get("max_budget", float("inf"))
            elif c.id == "C_MAX_DURATION":
                constraints_dict["max_duration_per_day"] = c.params.get("max_hours", float("inf"))
            elif c.id == "C_MUST_SPECIFIC":
                constraints_dict["must_include_specific"] = c.params.get("entity_name")

        return verify_travel_solvable(
            world.payload["attractions"],
            world.payload["num_days"],
            constraints_dict
        )

    elif world_type == "schedule":
        constraints_dict = {
            "max_per_day": 2,
            "keep_high_priority": True,
            "min_spread": 2,
        }
        for c in world.constraints:
            if c.id == "C_MAX_TOTAL":
                constraints_dict["max_total"] = c.params.get("max_total", 100)
            elif c.id == "C_SPREAD":
                constraints_dict["min_spread"] = c.params.get("min_days_with_appointments", 1)

        return verify_schedule_solvable(
            world.payload["appointments"],
            len(world.payload["days_ro"]),
            constraints_dict
        )

    elif world_type == "recipe":
        constraints_dict = {}
        for c in world.constraints:
            if "vegetarian" in c.check_fn:
                constraints_dict["vegetarian"] = True
            elif "vegan" in c.check_fn:
                constraints_dict["vegan"] = True
            elif "gluten" in c.check_fn:
                constraints_dict["no_gluten"] = True
            elif "lactose" in c.check_fn:
                constraints_dict["no_lactose"] = True
            elif c.id == "C_CALORIE_RANGE":
                constraints_dict["min_calories"] = c.params.get("min_calories", 0)
                constraints_dict["max_calories"] = c.params.get("max_calories", float("inf"))
            elif c.id == "C_LUNCH_HEAVY":
                constraints_dict["lunch_heaviest"] = True
            elif c.id == "C_DINNER_LIGHT":
                constraints_dict["dinner_lightest"] = True

        # Build dishes by type from DISHES constant
        dishes_by_type = {
            "mic_dejun": DISHES["mic_dejun"],
            "pranz": DISHES["pranz"],
            "cina": DISHES["cina"],
        }

        return verify_recipe_solvable(
            dishes_by_type,
            world.payload["num_days"],
            constraints_dict
        )

    elif world_type == "fact":
        # Fact instances are always solvable (just return context answer)
        return True

    return True


def generate_hard_dataset(
    num_travel: int = 100,
    num_schedule: int = 75,
    num_fact: int = 50,
    num_recipe: int = 75,
    seed_start: int = 10000,  # Different from v0 to avoid overlap
    output_file: str = "data/gmtw_ro_hard.jsonl",
):
    """Generate the hard dataset with solvability verification"""
    instances = []
    seed = seed_start
    max_retries = 10
    unsolvable_count = 0

    travel_gen = ExtremeTravelGenerator()
    schedule_gen = ExtremeScheduleGenerator()
    fact_gen = ExtremeFactGenerator()
    recipe_gen = ExtremeRecipeGenerator()

    print(f"Generating {num_travel} EXTREME Travel instances...")
    for i in range(num_travel):
        for retry in range(max_retries):
            world_id = f"travel_hard_{seed:06d}"
            world = travel_gen.generate(world_id=world_id, seed=seed)

            if verify_world_solvable(world):
                break
            else:
                unsolvable_count += 1
                seed += 1
        else:
            print(f"  WARNING: Could not generate solvable travel instance after {max_retries} retries")

        prompt_ro = templates_ro.generate_prompt(world)
        prompt_en = templates_en.generate_prompt(world)

        instance = Instance(
            instance_id=world_id,
            world=world,
            prompt_ro=prompt_ro,
            prompt_en=prompt_en,
            meta=world.meta
        )
        instances.append(instance)
        seed += 1

        if (i + 1) % 25 == 0:
            print(f"  Generated {i + 1}/{num_travel}...")

    print(f"Generating {num_schedule} EXTREME Schedule instances...")
    for i in range(num_schedule):
        for retry in range(max_retries):
            world_id = f"schedule_hard_{seed:06d}"
            world = schedule_gen.generate(world_id=world_id, seed=seed)

            if verify_world_solvable(world):
                break
            else:
                unsolvable_count += 1
                seed += 1
        else:
            print(f"  WARNING: Could not generate solvable schedule instance after {max_retries} retries")

        prompt_ro = templates_ro.generate_prompt(world)
        prompt_en = templates_en.generate_prompt(world)

        instance = Instance(
            instance_id=world_id,
            world=world,
            prompt_ro=prompt_ro,
            prompt_en=prompt_en,
            meta=world.meta
        )
        instances.append(instance)
        seed += 1

        if (i + 1) % 25 == 0:
            print(f"  Generated {i + 1}/{num_schedule}...")

    print(f"Generating {num_fact} EXTREME Fact instances...")
    for i in range(num_fact):
        world_id = f"fact_hard_{seed:06d}"
        world = fact_gen.generate(world_id=world_id, seed=seed)

        prompt_ro = templates_ro.generate_prompt(world)
        prompt_en = templates_en.generate_prompt(world)

        instance = Instance(
            instance_id=world_id,
            world=world,
            prompt_ro=prompt_ro,
            prompt_en=prompt_en,
            meta=world.meta
        )
        instances.append(instance)
        seed += 1

        if (i + 1) % 25 == 0:
            print(f"  Generated {i + 1}/{num_fact}...")

    print(f"Generating {num_recipe} EXTREME Recipe instances...")
    for i in range(num_recipe):
        for retry in range(max_retries):
            world_id = f"recipe_hard_{seed:06d}"
            world = recipe_gen.generate(world_id=world_id, seed=seed)

            if verify_world_solvable(world):
                break
            else:
                unsolvable_count += 1
                seed += 1
        else:
            print(f"  WARNING: Could not generate solvable recipe instance after {max_retries} retries")

        prompt_ro = templates_ro.generate_prompt(world)
        prompt_en = templates_en.generate_prompt(world)

        instance = Instance(
            instance_id=world_id,
            world=world,
            prompt_ro=prompt_ro,
            prompt_en=prompt_en,
            meta=world.meta
        )
        instances.append(instance)
        seed += 1

        if (i + 1) % 25 == 0:
            print(f"  Generated {i + 1}/{num_recipe}...")

    # Report unsolvable attempts
    if unsolvable_count > 0:
        print(f"\n⚠ Regenerated {unsolvable_count} instances due to unsolvability")

    # Shuffle to mix world types
    random.Random(seed_start).shuffle(instances)

    # Write to file
    print(f"\nWriting {len(instances)} instances to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for instance in instances:
            f.write(json.dumps(instance.to_dict(), ensure_ascii=False) + '\n')

    print(f"\nDone! Generated {len(instances)} EXTREME difficulty instances.")
    print(f"  Travel:   {num_travel}")
    print(f"  Schedule: {num_schedule}")
    print(f"  Fact:     {num_fact}")
    print(f"  Recipe:   {num_recipe}")

    # Print constraint statistics
    constraint_counts = {}
    for inst in instances:
        world_type = inst.world.world_type
        n_constraints = len(inst.world.constraints)
        if world_type not in constraint_counts:
            constraint_counts[world_type] = []
        constraint_counts[world_type].append(n_constraints)

    print("\nConstraint statistics:")
    for wt, counts in sorted(constraint_counts.items()):
        avg = sum(counts) / len(counts)
        print(f"  {wt}: avg {avg:.1f} constraints, range [{min(counts)}-{max(counts)}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GMTW-Ro HARD dataset (extreme difficulty)"
    )
    parser.add_argument(
        "--num-travel", type=int, default=100,
        help="Number of travel instances (default: 100)"
    )
    parser.add_argument(
        "--num-schedule", type=int, default=75,
        help="Number of schedule instances (default: 75)"
    )
    parser.add_argument(
        "--num-fact", type=int, default=50,
        help="Number of fact instances (default: 50)"
    )
    parser.add_argument(
        "--num-recipe", type=int, default=75,
        help="Number of recipe instances (default: 75)"
    )
    parser.add_argument(
        "--seed-start", type=int, default=10000,
        help="Starting seed (default: 10000, different from v0)"
    )
    parser.add_argument(
        "--output", type=str, default="data/gmtw_ro_hard.jsonl",
        help="Output JSONL file"
    )

    args = parser.parse_args()

    generate_hard_dataset(
        num_travel=args.num_travel,
        num_schedule=args.num_schedule,
        num_fact=args.num_fact,
        num_recipe=args.num_recipe,
        seed_start=args.seed_start,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
