#!/usr/bin/env python3
"""
Generate GMTW-Ro v0 dataset

This script generates a complete GMTW-Ro evaluation dataset with
Travel, Schedule, and Fact world instances.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    Generate a complete dataset

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

    # Initialize generators
    travel_gen = TravelWorldGenerator(spec_version="0.1")
    schedule_gen = ScheduleWorldGenerator(spec_version="0.1")
    fact_gen = FactWorldGenerator(spec_version="0.1")
    recipe_gen = RecipeWorldGenerator(spec_version="0.1")

    print(f"Generating {num_travel} Travel instances (difficulty={difficulty})...")
    for i in range(num_travel):
        world_id = f"travel_{seed:06d}"
        world = travel_gen.generate(
            world_id=world_id,
            seed=seed,
            difficulty=get_difficulty_for_index(i, num_travel, difficulty)
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

    print(f"Generating {num_schedule} Schedule instances (difficulty={difficulty})...")
    for i in range(num_schedule):
        world_id = f"schedule_{seed:06d}"
        world = schedule_gen.generate(
            world_id=world_id,
            seed=seed,
            difficulty=get_difficulty_for_index(i, num_schedule, difficulty)
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

    print(f"Generating {num_fact} Fact instances (difficulty={difficulty})...")
    for i in range(num_fact):
        world_id = f"fact_{seed:06d}"
        world = fact_gen.generate(
            world_id=world_id,
            seed=seed,
            difficulty=get_difficulty_for_index(i, num_fact, difficulty)
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
        world = recipe_gen.generate(
            world_id=world_id,
            seed=seed,
            difficulty=get_difficulty_for_index(i, num_recipe, difficulty)
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

    # Write to JSONL
    print(f"Writing {len(instances)} instances to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for instance in instances:
            f.write(json.dumps(instance.to_dict(), ensure_ascii=False) + '\n')

    print(f"Done! Generated {len(instances)} instances.")
    print(f"  Travel: {num_travel}")
    print(f"  Schedule: {num_schedule}")
    print(f"  Fact: {num_fact}")
    print(f"  Recipe: {num_recipe}")


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
