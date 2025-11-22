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
    misbeliefs = world.payload.get('misbelief_scripts', [])

    if misbeliefs:
        # Answer the first misbelief question
        script = misbeliefs[0]
        answer = script['true_answer']

        explanation = f"Pe baza informațiilor din contextul dat, răspunsul corect este: {answer}\n\n"
        explanation += "Aceasta este informația din baza de date furnizată.\n\n"

        json_answer = json.dumps({"answer": answer}, ensure_ascii=False, indent=2)
    else:
        # Just list the facts
        explanation = "Faptele din baza de date sunt:\n\n"
        for key, value in facts.items():
            explanation += f"- {key}: {value}\n"
        explanation += "\n"

        json_answer = json.dumps({"answer": "Vezi lista de mai sus"}, ensure_ascii=False, indent=2)

    return explanation + json_answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dummy model outputs")
    parser.add_argument("instances", help="Input instances JSONL file")
    parser.add_argument("--output", default="dummy_outputs.jsonl", help="Output file")

    args = parser.parse_args()

    create_dummy_outputs(args.instances, args.output)
