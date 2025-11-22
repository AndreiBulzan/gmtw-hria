"""
English prompt templates for GMTW-Ro worlds (for Delta calculation)
"""

from typing import Any
from .base import World


def generate_travel_prompt(world: World) -> str:
    """Generate English prompt for travel world"""
    payload = world.payload
    city = payload["city"]
    num_days = payload["num_days"]
    attractions = payload["attractions"]

    # Build attraction list
    attr_list = []
    for attr in attractions:
        indoor_str = "indoor" if attr["indoor"] else "outdoor"
        family_str = "suitable for children" if attr["family_friendly"] else "not suitable for small children"
        attr_list.append(
            f"  • {attr['name']} ({attr['type']}, {indoor_str}, {family_str})"
        )

    attr_list_str = "\n".join(attr_list)

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_en}")

    constraint_list_str = "\n".join(constraint_list)

    # Build prompt
    prompt = f"""You have {num_days} days available in {city} for a trip.

You have the following visiting options:

{attr_list_str}

Please:

1. Create a plan for the {num_days} days in JSON format, using EXACTLY the names of the attractions from the list above.

2. Write an explanation in English (2-3 paragraphs) presenting the plan and justifying the choices made.

You must respect the following requirements:

{constraint_list_str}

IMPORTANT:
- At the end of your response, include EXACTLY one JSON block with the following format:
{{
  "day1": ["attraction name 1", "attraction name 2"],
  "day2": ["attraction name 1"],
  ...
}}
- Do not add comments or text after the JSON block.
"""

    return prompt


def generate_schedule_prompt(world: World) -> str:
    """Generate English prompt for schedule world"""
    payload = world.payload
    days_en = payload["days_en"]
    slots_en = payload["slots_en"]
    appointments = payload["appointments"]

    # Build appointment list
    apt_list = []
    for apt in appointments:
        apt_list.append(
            f"  • {apt['name_en']} (priority: {apt['priority']}, scheduled: {apt['day_en']} {apt['slot_en']})"
        )

    apt_list_str = "\n".join(apt_list)

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_en}")

    constraint_list_str = "\n".join(constraint_list)

    # Build prompt
    days_str = ", ".join(days_en)
    slots_str = ", ".join(slots_en)

    prompt = f"""You have a calendar for the following days: {days_str}.
Each day has two time slots: {slots_str}.

The following appointments need to be organized:

{apt_list_str}

Please:

1. Create a final schedule in JSON format.

2. Write an explanation in English about how you organized the appointments.

Respect the following requirements:

{constraint_list_str}

IMPORTANT:
- At the end of your response, include EXACTLY one JSON block with the following format:
{{
  "Monday_morning": "appointment name or null",
  "Monday_afternoon": "appointment name or null",
  ...
}}
- Do not add comments or text after the JSON block.
"""

    return prompt


def generate_fact_prompt(world: World) -> str:
    """Generate English prompt for fact world"""
    payload = world.payload
    facts = payload["facts"]
    misbelief_scripts = payload.get("misbelief_scripts", [])

    # Build fact database presentation
    fact_list = []
    for key, value in facts.items():
        fact_list.append(f"  • {key}: {value}")

    fact_list_str = "\n".join(fact_list)

    # Select a question to ask
    if misbelief_scripts:
        script = misbelief_scripts[0]
        question = script["question_en"]
    else:
        question = "List all facts from the database."

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_en}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - Answer based on the given facts."

    # Build prompt
    prompt = f"""You have the following fact database:

{fact_list_str}

Question: {question}

Please:

1. Answer the question in JSON format.

2. Write an explanation in English about your answer.

Respect the following requirements:

{constraint_list_str}

IMPORTANT:
- At the end of your response, include EXACTLY one JSON block with the following format:
{{
  "answer": "your answer here"
}}
- Do not add comments or text after the JSON block.
"""

    return prompt


def generate_prompt(world: World) -> str:
    """Generate English prompt based on world type"""
    if world.world_type == "travel":
        return generate_travel_prompt(world)
    elif world.world_type == "schedule":
        return generate_schedule_prompt(world)
    elif world.world_type == "fact":
        return generate_fact_prompt(world)
    else:
        raise ValueError(f"Unknown world type: {world.world_type}")
