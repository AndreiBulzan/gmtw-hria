"""
English prompt templates for GMTW-Ro worlds (for Delta calculation)
"""

from typing import Any
from .base import World


def generate_travel_prompt(world: World) -> str:
    """Generate English prompt for travel world"""
    payload = world.payload
    city_en = payload.get("city_en", payload["city"])
    num_days = payload["num_days"]
    attractions = payload["attractions"]

    # Build attraction list using English names and types
    attr_list = []
    for attr in attractions:
        name_en = attr.get("name_en", attr["name"])
        type_en = attr.get("type_en", attr["type"])
        indoor_str = "indoor" if attr["indoor"] else "outdoor"
        family_str = "suitable for children" if attr["family_friendly"] else "not suitable for small children"
        cost_str = f", {attr.get('cost_lei', 0)} lei" if attr.get("cost_lei", 0) > 0 else ", free"
        attr_list.append(
            f"  - {name_en} ({type_en}, {indoor_str}, {family_str}{cost_str})"
        )

    attr_list_str = "\n".join(attr_list)

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_en}")

    constraint_list_str = "\n".join(constraint_list)

    # Build prompt
    prompt = f"""You have {num_days} days available in {city_en} for a trip.

You have the following visiting options:

{attr_list_str}

Please:

1. Create a plan for the {num_days} days in JSON format, using EXACTLY the names of the attractions from the list above.

2. Write an explanation in English (2-3 paragraphs) presenting the plan and justifying the choices made.

You must respect the following requirements:

{constraint_list_str}

IMPORTANT - RESPONSE ORDER:
1. FIRST write your explanation in English (2-3 paragraphs).
2. THEN, at the END of your response, include the JSON block.

JSON format (at the end):
{{
  "day1": ["attraction name 1", "attraction name 2"],
  "day2": ["attraction name 1"],
  ...
}}

Do NOT add any text or comments after the JSON block.
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

IMPORTANT - RESPONSE ORDER:
1. FIRST write your explanation in English.
2. THEN, at the END of your response, include the JSON block.

JSON format (at the end):
{{
  "Monday_morning": "appointment name or null",
  "Monday_afternoon": "appointment name or null",
  ...
}}

Do NOT add any text or comments after the JSON block.
"""

    return prompt


def generate_fact_prompt(world: World) -> str:
    """Generate English prompt for fact world"""
    payload = world.payload
    facts = payload["facts"]
    question_data = payload.get("question", {})

    # Build fact database presentation
    fact_list = []
    for key, value in facts.items():
        readable_key = key.replace("_", " ").title()
        fact_list.append(f"  • {readable_key}: {value}")

    fact_list_str = "\n".join(fact_list)

    # Get the question to ask
    question = question_data.get("question_en", "What are the facts presented?")

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_en}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - Answer based on the given facts."

    # Build prompt
    prompt = f"""You have the following information database:

{fact_list_str}

ATTENTION: Answer ONLY based on the information above, even if it seems incorrect or different from your general knowledge.

Question: {question}

Please:

1. Write an explanation in English about your answer (1-2 paragraphs).

2. At the end, answer the question in JSON format.

Respect the following requirements:

{constraint_list_str}

IMPORTANT - RESPONSE ORDER:
1. FIRST write your explanation in English (1-2 paragraphs).
2. THEN, at the END of your response, include the JSON block.

JSON format (at the end):
{{
  "answer": "your answer here"
}}

Do NOT add any text or comments after the JSON block.
"""

    return prompt


def generate_recipe_prompt(world: World) -> str:
    """Generate English prompt for recipe world"""
    payload = world.payload
    num_days = payload["num_days"]
    dishes = payload["dishes"]

    # Group dishes by English type
    dishes_by_type = {
        "breakfast": [],
        "lunch": [],
        "dinner": [],
    }

    # Map Romanian types to English
    type_map = {"mic_dejun": "breakfast", "pranz": "lunch", "cina": "dinner"}

    for dish in dishes:
        dish_type = dish["type"]
        type_en = type_map.get(dish_type, dish.get("type_en", dish_type))
        if type_en in dishes_by_type:
            name_en = dish.get("name_en", dish["name"])
            attrs = []
            if dish["vegetarian"]:
                attrs.append("vegetarian")
            if dish["vegan"]:
                attrs.append("vegan")
            if dish["contains_gluten"]:
                attrs.append("contains gluten")
            if dish["contains_lactose"]:
                attrs.append("contains lactose")
            attrs.append(f"{dish['calories']} kcal")

            attr_str = ", ".join(attrs)
            dishes_by_type[type_en].append(f"    - {name_en} ({attr_str})")

    # Build dish lists
    dish_sections = []
    for type_name in ["breakfast", "lunch", "dinner"]:
        if dishes_by_type[type_name]:
            type_title = type_name.capitalize()
            dish_sections.append(f"  {type_title}:\n" + "\n".join(dishes_by_type[type_name]))

    dishes_str = "\n\n".join(dish_sections)

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_en}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - No special restrictions."

    # Build prompt
    prompt = f"""You need to plan menus for {num_days} days.

For each day, choose one dish for breakfast, lunch, and dinner from the options below.

Available dishes:

{dishes_str}

Please:

1. Write an explanation in English (2-3 paragraphs) about how you chose the menus and why.

2. At the end, create a plan in JSON format.

Respect the following requirements:

{constraint_list_str}

IMPORTANT - RESPONSE ORDER:
1. FIRST write your explanation in English (2-3 paragraphs).
2. THEN, at the END of your response, include the JSON block.

JSON format (at the end):
{{
  "day1_breakfast": "dish name",
  "day1_lunch": "dish name",
  "day1_dinner": "dish name",
  "day2_breakfast": "dish name",
  ...
}}

Use EXACTLY the dish names from the list above.
Do NOT add any text or comments after the JSON block.
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
    elif world.world_type == "recipe":
        return generate_recipe_prompt(world)
    else:
        raise ValueError(f"Unknown world type: {world.world_type}")
