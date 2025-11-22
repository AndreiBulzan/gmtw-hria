#!/usr/bin/env python3
"""
View GMTW-Ro instances in a readable format
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro.worlds.base import Instance


def view_instances(jsonl_file: str, max_instances: int = None):
    """View instances from JSONL file"""

    instances = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(Instance.from_dict(json.loads(line)))

    if max_instances:
        instances = instances[:max_instances]

    for i, instance in enumerate(instances, 1):
        print("\n" + "=" * 80)
        print(f"INSTANCE {i}/{len(instances)}: {instance.instance_id}")
        print("=" * 80)

        world = instance.world
        print(f"\nType: {world.world_type.upper()}")
        print(f"Difficulty: {world.meta.get('difficulty', 'N/A')}")

        if world.world_type == "travel":
            print(f"City: {world.payload['city']}")
            print(f"Days: {world.payload['num_days']}")
            print(f"\nAttractions:")
            for attr in world.payload['attractions']:
                indoor = "🏠 Indoor" if attr['indoor'] else "🌳 Outdoor"
                family = "👨‍👩‍👧 Family" if attr['family_friendly'] else "❌ Not family"
                print(f"  • {attr['name']} ({attr['type']}) - {indoor}, {family}")

        elif world.world_type == "schedule":
            print(f"Days: {', '.join(world.payload['days_ro'])}")
            print(f"\nAppointments:")
            for apt in world.payload['appointments']:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                emoji = priority_emoji.get(apt['priority'], "⚪")
                print(f"  {emoji} {apt['name_ro']} - {apt['day_ro']} {apt['slot_ro']} (priority: {apt['priority']})")

        elif world.world_type == "fact":
            print(f"\nFacts in database:")
            for key, value in world.payload['facts'].items():
                print(f"  • {key}: {value}")

        print(f"\nConstraints ({len(world.constraints)}):")
        for c in world.constraints:
            print(f"  ✓ {c.description_ro}")

        print(f"\n" + "-" * 80)
        print("ROMANIAN PROMPT:")
        print("-" * 80)
        print(instance.prompt_ro)
        print("-" * 80)

        if i < len(instances):
            response = input("\nPress Enter for next instance, 'q' to quit, or number to jump: ")
            if response.lower() == 'q':
                break
            elif response.isdigit():
                target = int(response) - 1
                if 0 <= target < len(instances):
                    # Restart from that position
                    instances = instances[target:]
                    i = 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View GMTW-Ro instances")
    parser.add_argument("file", help="JSONL file to view")
    parser.add_argument("--max", type=int, help="Maximum instances to show")

    args = parser.parse_args()

    view_instances(args.file, args.max)
