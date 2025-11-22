#!/usr/bin/env python3
"""
Simple example of GMTW-Ro usage

This script demonstrates:
1. Generating a Travel World instance
2. Creating prompts in Romanian and English
3. Simulating model outputs
4. Evaluating the outputs and computing U/R/G/F metrics
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro import (
    TravelWorldGenerator,
    Instance,
    evaluate_instance,
)
from rombench.gmtw_ro.worlds import templates_ro, templates_en


def main():
    print("=" * 70)
    print("GMTW-Ro Simple Example")
    print("=" * 70)

    # Step 1: Generate a Travel World
    print("\n[1] Generating a Travel World...")
    generator = TravelWorldGenerator(spec_version="0.1")
    world = generator.generate(
        world_id="travel_example_001",
        seed=42,
        difficulty="easy"
    )

    print(f"  ✓ Generated world: {world.world_id}")
    print(f"    City: {world.payload['city']}")
    print(f"    Days: {world.payload['num_days']}")
    print(f"    Attractions: {len(world.payload['attractions'])}")
    print(f"    Constraints: {len(world.constraints)}")

    # Step 2: Generate prompts
    print("\n[2] Generating prompts...")
    prompt_ro = templates_ro.generate_prompt(world)
    prompt_en = templates_en.generate_prompt(world)

    print(f"  ✓ Romanian prompt ({len(prompt_ro)} chars)")
    print(f"  ✓ English prompt ({len(prompt_en)} chars)")

    # Display the Romanian prompt
    print("\n" + "─" * 70)
    print("ROMANIAN PROMPT:")
    print("─" * 70)
    print(prompt_ro)
    print("─" * 70)

    # Step 3: Create an Instance
    instance = Instance(
        instance_id=world.world_id,
        world=world,
        prompt_ro=prompt_ro,
        prompt_en=prompt_en,
        meta=world.meta
    )

    # Step 4: Simulate a model output (with dual-channel structure)
    print("\n[3] Simulating model output...")

    # Get attraction names for the example
    attraction_names = [a["name"] for a in world.payload["attractions"][:3]]

    simulated_output = f"""Propun următorul plan pentru excursia de {world.payload['num_days']} zile în {world.payload['city']}:

Pentru prima zi, vom vizita {attraction_names[0]}, un monument istoric important. Aceasta este o activitate în interior, potrivită pentru familii cu copii.

În a doua zi, vom merge la {attraction_names[1]}, care oferă o experiență frumoasă în aer liber, fiind, de asemenea, potrivită pentru copii.

Acest plan respectă toate cerințele: include un monument istoric și limitează activitățile în aer liber la una pe zi, fiind în același timp potrivit pentru familii cu copii.

{{
  "day1": ["{attraction_names[0]}"],
  "day2": ["{attraction_names[1]}"]
}}
"""

    print("  ✓ Simulated model output:")
    print("\n" + "─" * 70)
    print(simulated_output)
    print("─" * 70)

    # Step 5: Evaluate the output
    print("\n[4] Evaluating output...")
    result = evaluate_instance(instance, simulated_output)

    print(f"\n{'═' * 70}")
    print("EVALUATION RESULTS")
    print(f"{'═' * 70}\n")

    print(f"  Instance ID: {result.instance_id}")
    print(f"  Format OK: {result.format_ok}")
    print(f"  Repaired: {result.repaired}")

    if result.parse_error:
        print(f"  Parse Error: {result.parse_error}")
    else:
        print(f"\n  {'─' * 66}")
        print(f"  METRICS:")
        print(f"  {'─' * 66}")
        print(f"    U (Understanding):   {result.U:.2f}")
        print(f"    R (Reasoning):       {result.R:.2f}")
        print(f"    G (Generation):      {result.G:.2f}")
        print(f"    F (Faithfulness):    {result.F:.2f}")
        print(f"  {'─' * 66}")

        print(f"\n  Understanding Details:")
        print(f"    Constraints satisfied: {result.U_details['satisfied']}/{result.U_details['total']}")
        for c in result.U_details.get('constraints', []):
            status = "✓" if c['satisfied'] else "✗"
            print(f"      {status} {c['id']}: {c['description'][:50]}...")

        print(f"\n  Reasoning Details:")
        print(f"    Goals satisfied: {result.R_details['satisfied']}/{result.R_details['total']}")
        for g in result.R_details.get('goals', []):
            status = "✓" if g['satisfied'] else "✗"
            print(f"      {status} {g['id']}: {g['description'][:50]}...")

        print(f"\n  Faithfulness Details:")
        print(f"    Planned entities: {result.F_details.get('planned_entities', [])}")
        print(f"    Mentioned entities: {result.F_details.get('mentioned_entities', [])}")
        print(f"    Missing from explanation: {result.F_details.get('missing', [])}")
        print(f"    Extra in explanation: {result.F_details.get('extra', [])}")

    print(f"\n{'═' * 70}")
    print("Example completed successfully!")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
