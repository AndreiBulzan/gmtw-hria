#!/usr/bin/env python3
"""
Create dummy model outputs for testing the evaluator
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro.worlds.base import Instance


def create_dummy_outputs(instances_file: str, output_file: str):
    """Create dummy outputs that should score reasonably well"""

    outputs = []

    with open(instances_file, 'r', encoding='utf-8') as f:
        for line in f:
            inst = Instance.from_dict(json.loads(line))
            world = inst.world

            # Generate a dummy but valid-ish output based on world type
            if world.world_type == "travel":
                dummy_output = generate_travel_dummy(world)
            elif world.world_type == "schedule":
                dummy_output = generate_schedule_dummy(world)
            elif world.world_type == "fact":
                dummy_output = generate_fact_dummy(world)
            elif world.world_type == "recipe":
                dummy_output = generate_recipe_dummy(world)
            else:
                dummy_output = "Nu pot rezolva această sarcină."

            outputs.append({
                "instance_id": inst.instance_id,
                "output": dummy_output
            })

    # Write outputs
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    print(f"✓ Created {len(outputs)} dummy outputs in {output_file}")


def generate_travel_dummy(world):
    """Generate dummy travel output"""
    city = world.payload['city']
    num_days = world.payload['num_days']
    attractions = world.payload['attractions']

    # Build a simple plan: distribute attractions across days
    plan = {}
    for day in range(1, num_days + 1):
        plan[f"day{day}"] = []

    # Add attractions, respecting family-friendly if needed
    has_family_constraint = any(
        c.id == "C_FAMILY_FRIENDLY" for c in world.constraints
    )

    day_idx = 1
    for attr in attractions:
        if has_family_constraint and not attr.get('family_friendly', False):
            continue  # Skip non-family-friendly

        if day_idx <= num_days:
            plan[f"day{day_idx}"].append(attr['name'])
            day_idx += 1

    # Generate explanation
    explanation = f"Pentru călătoria de {num_days} zile în {city}, propun următorul plan:\n\n"

    for day_num in range(1, num_days + 1):
        day_key = f"day{day_num}"
        if plan[day_key]:
            explanation += f"În ziua {day_num}, vom vizita {', '.join(plan[day_key])}. "
        explanation += "\n"

    explanation += f"\nAcest plan respectă toate cerințele și oferă o experiență variată.\n\n"

    # Add JSON
    json_plan = json.dumps(plan, ensure_ascii=False, indent=2)

    return explanation + json_plan


def generate_schedule_dummy(world):
    """Generate dummy schedule output"""
    appointments = world.payload['appointments']
    days = world.payload['days_ro']
    slots = world.payload['slots_ro']

    # Build schedule
    schedule = {}
    apt_idx = 0

    for day in days:
        for slot in slots:
            key = f"{day}_{slot}"
            if apt_idx < len(appointments):
                schedule[key] = appointments[apt_idx]['name_ro']
                apt_idx += 1
            else:
                schedule[key] = None

    # Generate explanation
    explanation = "Am organizat programările astfel:\n\n"
    for day in days:
        explanation += f"{day}:\n"
        for slot in slots:
            key = f"{day}_{slot}"
            apt = schedule.get(key)
            if apt:
                explanation += f"  - {slot}: {apt}\n"
        explanation += "\n"

    explanation += "Acest plan respectă prioritățile și evită suprapunerile.\n\n"

    # Add JSON
    json_schedule = json.dumps(schedule, ensure_ascii=False, indent=2)

    return explanation + json_schedule


def generate_fact_dummy(world):
    """Generate dummy fact output"""
    facts = world.payload['facts']
    question = world.payload.get('question', {})
    expected = question.get('expected_answer', '')

    if expected:
        answer = expected
        explanation = f"Pe baza informațiilor din contextul dat, răspunsul este: {answer}\n\n"
        explanation += "Aceasta este informația din baza de date furnizată.\n\n"
    else:
        # Just use first fact
        first_key = list(facts.keys())[0] if facts else ''
        answer = facts.get(first_key, 'Necunoscut')
        explanation = f"Răspunsul este: {answer}\n\n"

    json_answer = json.dumps({"answer": answer}, ensure_ascii=False, indent=2)
    return explanation + json_answer


def generate_recipe_dummy(world):
    """Generate dummy recipe output"""
    num_days = world.payload['num_days']
    dishes = world.payload['dishes']

    # Group dishes by type
    by_type = {'mic_dejun': [], 'pranz': [], 'cina': []}
    for dish in dishes:
        dtype = dish['type']
        if dtype in by_type:
            # Check dietary constraints
            is_ok = True
            for c in world.constraints:
                if c.id == 'C_DIET_0' and 'vegetarian' in c.description_ro.lower():
                    if not dish.get('vegetarian', False):
                        is_ok = False
                if 'gluten' in c.check_fn and dish.get('contains_gluten', False):
                    is_ok = False
                if 'lactose' in c.check_fn and dish.get('contains_lactose', False):
                    is_ok = False
            if is_ok:
                by_type[dtype].append(dish['name'])

    # Build plan
    plan = {}
    for day in range(1, num_days + 1):
        for meal in ['mic_dejun', 'pranz', 'cina']:
            key = f"day{day}_{meal}"
            options = by_type.get(meal, [])
            if options:
                plan[key] = options[(day - 1) % len(options)]
            else:
                plan[key] = dishes[0]['name'] if dishes else "Necunoscut"

    # Generate explanation
    meal_names = {'mic_dejun': 'mic dejun', 'pranz': 'prânz', 'cina': 'cină'}
    explanation = f"Am planificat meniurile pentru {num_days} zile.\n\n"

    for day in range(1, num_days + 1):
        explanation += f"Ziua {day}:\n"
        for meal in ['mic_dejun', 'pranz', 'cina']:
            key = f"day{day}_{meal}"
            explanation += f"  - {meal_names[meal]}: {plan[key]}\n"
        explanation += "\n"

    explanation += "Meniurile respectă restricțiile alimentare indicate.\n\n"

    json_plan = json.dumps(plan, ensure_ascii=False, indent=2)
    return explanation + json_plan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dummy model outputs")
    parser.add_argument("instances", help="Input instances JSONL file")
    parser.add_argument("--output", default="dummy_outputs.jsonl", help="Output file")

    args = parser.parse_args()

    create_dummy_outputs(args.instances, args.output)
