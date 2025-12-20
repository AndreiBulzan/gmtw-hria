#!/usr/bin/env python3
"""
Generate GMTW-Ro v0 dataset

This script generates a complete GMTW-Ro evaluation dataset with
Travel, Schedule, Fact, and Recipe world instances.

All instances are verified solvable at generation time using backtracking solvers.
"""

import argparse
import json
import sys
from itertools import combinations, permutations, product
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# SOLVABILITY VERIFICATION (Actual constraint solvers)
# =============================================================================

def solve_travel(world) -> bool:
    """Try to find a valid travel plan using brute force backtracking."""
    attrs = world.payload.get("attractions", [])
    num_days = world.payload.get("num_days", 2)
    constraints = world.constraints

    # Parse constraints
    must_types = set()
    must_specific = set()
    family_required = False
    max_budget = float("inf")
    first_indoor = False
    last_outdoor = False

    for c in constraints:
        if c.id == "C_MUST_MONUMENT":
            must_types.add("monument")
        elif c.id == "C_MUST_MUSEUM":
            must_types.add("muzeu")
        elif c.id == "C_FAMILY_FRIENDLY":
            family_required = True
        elif c.id == "C_BUDGET" and hasattr(c, 'params'):
            max_budget = c.params.get("max_budget", float("inf"))
        elif c.id == "C_FIRST_DAY" and hasattr(c, 'params'):
            first_indoor = c.params.get("indoor", False)
        elif c.id == "C_LAST_DAY" and hasattr(c, 'params'):
            last_outdoor = c.params.get("outdoor", False)
        elif c.id == "C_MUST_SPECIFIC" and hasattr(c, 'params'):
            must_specific.add(c.params.get("entity_name"))

    # Filter by family-friendly if required
    valid_attrs = attrs
    if family_required:
        valid_attrs = [a for a in attrs if a.get("family_friendly")]

    if len(valid_attrs) < num_days:
        return False

    # Try all combinations of num_days attractions
    for combo in combinations(range(len(valid_attrs)), num_days):
        selected = [valid_attrs[i] for i in combo]

        # Check must include types
        selected_types = set(a.get("type") for a in selected)
        if not must_types.issubset(selected_types):
            continue

        # Check must include specific
        selected_names = set(a.get("name") for a in selected)
        if not must_specific.issubset(selected_names):
            continue

        # Check budget
        total_cost = sum(a.get("cost_lei", 0) for a in selected)
        if total_cost > max_budget:
            continue

        # Check day-specific constraints (try all orderings)
        for perm in permutations(selected):
            valid = True

            if first_indoor and not perm[0].get("indoor"):
                valid = False
            if last_outdoor and perm[-1].get("indoor"):
                valid = False

            if valid:
                return True  # Found a valid solution!

    return False


def solve_schedule(world) -> bool:
    """Try to find a valid schedule using brute force."""
    appointments = world.payload.get("appointments", [])
    num_days = world.payload.get("num_days", 5)
    constraints = world.constraints

    # Parse constraints
    max_per_day = 2
    max_total = len(appointments)
    keep_high = False
    min_spread = 1

    for c in constraints:
        if c.id == "C_MAX_PER_DAY" and hasattr(c, 'params'):
            max_per_day = c.params.get("max_per_day", 2)
        elif c.id == "C_MAX_TOTAL" and hasattr(c, 'params'):
            max_total = c.params.get("max_total", len(appointments))
        elif c.id == "C_KEEP_HIGH_PRIORITY":
            keep_high = True
        elif c.id == "C_SPREAD" and hasattr(c, 'params'):
            min_spread = c.params.get("min_days_with_appointments", 1)

    high_priority = [a for a in appointments if a.get("priority") == "high"]

    # Must keep all high priority
    if keep_high:
        must_keep = set(a.get("id") or a.get("name") for a in high_priority)
        if len(must_keep) > max_total:
            return False
    else:
        must_keep = set()

    # Try selecting max_total appointments
    for combo in combinations(range(len(appointments)), min(max_total, len(appointments))):
        selected = [appointments[i] for i in combo]
        selected_ids = set(a.get("id") or a.get("name") for a in selected)

        # Check high priority kept
        if not must_keep.issubset(selected_ids):
            continue

        # Check if we can spread across min_spread days with max_per_day each
        total_slots = num_days * max_per_day
        if len(selected) <= total_slots:
            days_needed = (len(selected) + max_per_day - 1) // max_per_day
            if days_needed <= num_days and days_needed >= min_spread:
                return True
            elif len(selected) >= min_spread:
                return True

    return False


def solve_recipe(world) -> bool:
    """Try to find a valid meal plan using brute force backtracking."""
    dishes = world.payload.get("dishes", [])
    num_days = world.payload.get("num_days", 3)
    constraints = world.constraints

    # Group dishes by meal type
    by_type = {}
    for d in dishes:
        mt = d.get("meal_type")
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

    for c in constraints:
        cid = c.id
        params = c.params if hasattr(c, 'params') else {}

        if cid == "C_DIETARY" or cid.startswith("C_DIET_"):
            r = params.get("restriction")
            if r == "vegetarian":
                dietary["vegetarian"] = True
            elif r == "vegan":
                dietary["vegan"] = True
            elif r == "fara_gluten":
                dietary["no_gluten"] = True
            elif r == "fara_lactoza":
                dietary["no_lactose"] = True
        elif cid == "C_CALORIE_RANGE":
            min_cal = params.get("min_cal", 0)
            max_cal = params.get("max_cal", float("inf"))
        elif cid == "C_MIN_CALORIES":
            min_cal = max(min_cal, params.get("min_cal", 0))
        elif cid == "C_MAX_CALORIES":
            max_cal = min(max_cal, params.get("max_cal", float("inf")))
        elif cid == "C_LUNCH_HEAVIEST":
            lunch_heaviest = True
        elif cid == "C_DINNER_LIGHTEST":
            dinner_lightest = True
        elif cid == "C_VARIETY":
            no_repeat = True

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

    def check_day(day_dishes):
        """Check if a day's dishes satisfy daily constraints."""
        total_cal = sum(d.get("calories", 0) for d in day_dishes)
        if total_cal < min_cal or total_cal > max_cal:
            return False

        if lunch_heaviest:
            lunch_d = next((d for d in day_dishes if d.get("meal_type") == "pranz"), None)
            if lunch_d:
                lunch_cal = lunch_d.get("calories", 0)
                for d in day_dishes:
                    if d.get("meal_type") != "pranz" and d.get("calories", 0) >= lunch_cal:
                        return False

        if dinner_lightest:
            dinner_d = next((d for d in day_dishes if d.get("meal_type") == "cina"), None)
            if dinner_d:
                dinner_cal = dinner_d.get("calories", 0)
                for d in day_dishes:
                    if d.get("meal_type") != "cina" and d.get("calories", 0) <= dinner_cal:
                        return False

        return True

    # Try to find valid assignments for all days using backtracking
    meal_types_sorted = sorted(valid_by_type.keys())

    def solve_day(day_idx, used_dishes):
        if day_idx == num_days:
            return True  # All days solved!

        options_per_type = []
        for mt in meal_types_sorted:
            available = [i for i, d in enumerate(valid_by_type[mt])
                        if (mt, i) not in used_dishes or not no_repeat]
            if not available:
                return False
            options_per_type.append([(mt, i) for i in available])

        for combo in product(*options_per_type):
            day_dishes = [valid_by_type[mt][i] for mt, i in combo]

            if check_day(day_dishes):
                new_used = used_dishes | set(combo) if no_repeat else used_dishes
                if solve_day(day_idx + 1, new_used):
                    return True

        return False

    return solve_day(0, set())


def solve_world(world) -> bool:
    """Verify that a world instance is solvable by finding at least one solution."""
    wtype = world.world_type
    if wtype == "travel":
        return solve_travel(world)
    elif wtype == "schedule":
        return solve_schedule(world)
    elif wtype == "recipe":
        return solve_recipe(world)
    elif wtype == "fact":
        return True  # Facts are always solvable
    return True

from rombench.gmtw_ro.worlds.base import Instance
from rombench.gmtw_ro.worlds.travel import TravelWorldGenerator
from rombench.gmtw_ro.worlds.schedule import ScheduleWorldGenerator
from rombench.gmtw_ro.worlds.fact import FactWorldGenerator
from rombench.gmtw_ro.worlds.recipe import RecipeWorldGenerator
from rombench.gmtw_ro.worlds import templates_ro, templates_en


def get_difficulty_for_index(i: int, total: int, difficulty_mode: str) -> str:
    """
    Determine difficulty level for instance at index i.

    Args:
        i: Current index
        total: Total instances of this type
        difficulty_mode: 'mixed' (default), 'easy', 'medium', or 'hard'

    Returns:
        Difficulty string: 'easy', 'medium', or 'hard'
    """
    if difficulty_mode in ("easy", "medium", "hard"):
        return difficulty_mode

    # Mixed mode: 40% easy, 40% medium, 20% hard
    if i < total * 0.4:
        return "easy"
    elif i < total * 0.8:
        return "medium"
    else:
        return "hard"


def generate_with_verification(generator, world_type: str, world_id: str, seed: int, difficulty: str, max_retries: int = 100):
    """
    Generate a world instance and verify it's solvable.
    Retries with different seeds if unsolvable.

    Returns:
        Tuple of (world, final_seed, retries_needed)
    """
    current_seed = seed
    for retry in range(max_retries):
        world = generator.generate(
            world_id=world_id,
            seed=current_seed,
            difficulty=difficulty
        )

        if solve_world(world):
            return world, current_seed, retry

        # Try next seed
        current_seed = seed + 10000 + retry

    raise RuntimeError(f"Could not generate solvable {world_type} instance after {max_retries} retries")


def generate_dataset(
    num_travel: int = 100,
    num_schedule: int = 100,
    num_fact: int = 50,
    num_recipe: int = 50,
    seed_start: int = 0,
    output_file: str = "instances.jsonl",
    difficulty: str = "mixed"
):
    """
    Generate a complete dataset with solvability verification.

    All instances are verified solvable using backtracking solvers.
    Instances that fail verification are regenerated with different seeds.

    Args:
        num_travel: Number of travel instances
        num_schedule: Number of schedule instances
        num_fact: Number of fact instances
        num_recipe: Number of recipe instances
        seed_start: Starting seed value
        output_file: Output JSONL file
        difficulty: 'mixed' (40% easy, 40% medium, 20% hard), 'easy', 'medium', or 'hard'
    """
    instances = []
    seed = seed_start
    total_retries = 0

    # Initialize generators
    travel_gen = TravelWorldGenerator(spec_version="0.1")
    schedule_gen = ScheduleWorldGenerator(spec_version="0.1")
    fact_gen = FactWorldGenerator(spec_version="0.1")
    recipe_gen = RecipeWorldGenerator(spec_version="0.1")

    print(f"Generating {num_travel} Travel instances (difficulty={difficulty})...")
    for i in range(num_travel):
        world_id = f"travel_{seed:06d}"
        diff = get_difficulty_for_index(i, num_travel, difficulty)

        world, final_seed, retries = generate_with_verification(
            travel_gen, "travel", world_id, seed, diff
        )
        total_retries += retries

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

    print(f"Generating {num_schedule} Schedule instances (difficulty={difficulty})...")
    for i in range(num_schedule):
        world_id = f"schedule_{seed:06d}"
        diff = get_difficulty_for_index(i, num_schedule, difficulty)

        world, final_seed, retries = generate_with_verification(
            schedule_gen, "schedule", world_id, seed, diff
        )
        total_retries += retries

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

    print(f"Generating {num_fact} Fact instances (difficulty={difficulty})...")
    for i in range(num_fact):
        world_id = f"fact_{seed:06d}"
        diff = get_difficulty_for_index(i, num_fact, difficulty)

        # Facts are always solvable, no verification needed
        world = fact_gen.generate(
            world_id=world_id,
            seed=seed,
            difficulty=diff
        )

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

    print(f"Generating {num_recipe} Recipe instances (difficulty={difficulty})...")
    for i in range(num_recipe):
        world_id = f"recipe_{seed:06d}"
        diff = get_difficulty_for_index(i, num_recipe, difficulty)

        world, final_seed, retries = generate_with_verification(
            recipe_gen, "recipe", world_id, seed, diff
        )
        total_retries += retries

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

    # Write to JSONL
    print(f"Writing {len(instances)} instances to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for instance in instances:
            f.write(json.dumps(instance.to_dict(), ensure_ascii=False) + '\n')

    print(f"\nDone! Generated {len(instances)} verified solvable instances.")
    print(f"  Travel: {num_travel}")
    print(f"  Schedule: {num_schedule}")
    print(f"  Fact: {num_fact}")
    print(f"  Recipe: {num_recipe}")
    if total_retries > 0:
        print(f"  Regenerated {total_retries} instances due to unsolvability")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GMTW-Ro v0 dataset"
    )
    parser.add_argument(
        "--num-travel",
        type=int,
        default=100,
        help="Number of travel instances (default: 100)"
    )
    parser.add_argument(
        "--num-schedule",
        type=int,
        default=100,
        help="Number of schedule instances (default: 100)"
    )
    parser.add_argument(
        "--num-fact",
        type=int,
        default=50,
        help="Number of fact instances (default: 50)"
    )
    parser.add_argument(
        "--num-recipe",
        type=int,
        default=50,
        help="Number of recipe instances (default: 50)"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed value (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gmtw_ro_v0/instances.jsonl",
        help="Output JSONL file (default: data/gmtw_ro_v0/instances.jsonl)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["mixed", "easy", "medium", "hard"],
        default="mixed",
        help="Difficulty level: mixed (40%% easy, 40%% medium, 20%% hard), easy, medium, or hard (default: mixed)"
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_dataset(
        num_travel=args.num_travel,
        num_schedule=args.num_schedule,
        num_fact=args.num_fact,
        num_recipe=args.num_recipe,
        seed_start=args.seed_start,
        output_file=args.output,
        difficulty=args.difficulty
    )


if __name__ == "__main__":
    main()
