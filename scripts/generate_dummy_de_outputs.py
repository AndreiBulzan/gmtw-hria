#!/usr/bin/env python3
"""
Generate dummy German model outputs for testing the evaluation pipeline.

Reads German instances and produces plausible model outputs (explanation + JSON plan).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro.worlds.base import Instance


def generate_travel_output(instance: Instance) -> str:
    """Generate a plausible German travel output."""
    world = instance.world
    payload = world.payload
    city_de = payload.get("city_de", payload.get("city_en", payload["city"]))
    num_days = payload["num_days"]
    attractions = payload["attractions"]

    # Pick attractions for each day (simple round-robin)
    plan = {}
    for day in range(1, num_days + 1):
        day_attrs = []
        for i, attr in enumerate(attractions):
            if (i % num_days) == (day - 1):
                name_de = attr.get("name_de", attr.get("name_en", attr["name"]))
                day_attrs.append(name_de)
        if not day_attrs:
            # Fallback: use first attraction
            name_de = attractions[0].get("name_de", attractions[0].get("name_en", attractions[0]["name"]))
            day_attrs.append(name_de)
        plan[f"day{day}"] = day_attrs

    # Build a natural German explanation
    all_names = []
    for day_list in plan.values():
        all_names.extend(day_list)

    explanation = f"""Für unseren {num_days}-tägigen Aufenthalt in {city_de} habe ich einen abwechslungsreichen Plan erstellt, der verschiedene Sehenswürdigkeiten und Aktivitäten umfasst. Die Stadt bietet eine reiche Geschichte und Kultur, die es zu entdecken gilt. Dabei habe ich darauf geachtet, die Anforderungen zu berücksichtigen und eine gute Mischung aus verschiedenen Aktivitätstypen zu gewährleisten.

Am ersten Tag beginnen wir mit {all_names[0] if all_names else 'einer Besichtigung'}, um einen guten Überblick über die Stadt zu bekommen. {'Am zweiten Tag besuchen wir ' + all_names[1] if len(all_names) > 1 else ''} {'und am dritten Tag steht ' + all_names[2] + ' auf dem Programm' if len(all_names) > 2 else ''}. Jeder Tag wurde sorgfältig geplant, um ein ausgewogenes Verhältnis zwischen kulturellen Erlebnissen und Erholung zu bieten.

Insgesamt bietet dieser Plan eine umfassende Erkundung von {city_de}, wobei sowohl historische Denkmäler als auch moderne Attraktionen berücksichtigt werden. Die Reihenfolge der Besichtigungen wurde so gewählt, dass die Wege zwischen den Sehenswürdigkeiten möglichst kurz sind und genügend Zeit für jede Aktivität bleibt.

"""

    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    return explanation + "```json\n" + plan_json + "\n```"


def generate_output(instance: Instance) -> str:
    """Generate a plausible output for any world type."""
    if instance.world.world_type == "travel":
        return generate_travel_output(instance)
    else:
        # Fallback for other types
        return generate_travel_output(instance)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate dummy German outputs for testing")
    parser.add_argument("--instances", default="data/gmtw_de_test.jsonl", help="German instance file")
    parser.add_argument("--output", default="data/test_outputs_de.jsonl", help="Output file")
    args = parser.parse_args()

    with open(args.instances, 'r', encoding='utf-8') as f:
        instances = [Instance.from_dict(json.loads(line)) for line in f]

    with open(args.output, 'w', encoding='utf-8') as f:
        for inst in instances:
            output_text = generate_output(inst)
            record = {
                "instance_id": inst.instance_id,
                "output": output_text,
                "language": "de",
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Generated {len(instances)} outputs → {args.output}")


if __name__ == "__main__":
    main()