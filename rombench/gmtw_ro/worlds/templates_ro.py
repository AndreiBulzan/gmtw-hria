"""
Romanian prompt templates for GMTW-Ro worlds
"""

from typing import Any
from .base import World


def generate_travel_prompt(world: World) -> str:
    """Generate Romanian prompt for travel world"""
    payload = world.payload
    city = payload["city"]
    num_days = payload["num_days"]
    attractions = payload["attractions"]

    # Check if we have budget constraint
    has_budget = any(c.id == "C_BUDGET" for c in world.constraints)

    # Build attraction list
    attr_list = []
    for attr in attractions:
        indoor_str = "interior" if attr["indoor"] else "exterior"
        family_str = "potrivit pentru copii" if attr["family_friendly"] else "nu este potrivit pentru copii mici"
        cost = attr.get("cost_lei", 0)
        cost_str = f", {cost} lei" if has_budget else ""
        attr_list.append(
            f"  • {attr['name']} ({attr['type']}, {indoor_str}, {family_str}{cost_str})"
        )

    attr_list_str = "\n".join(attr_list)

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_ro}")

    constraint_list_str = "\n".join(constraint_list)

    # Build prompt
    prompt = f"""Ai {num_days} zile la dispoziție în {city} pentru o excursie.

Ai următoarele opțiuni de vizitare:

{attr_list_str}

Te rog să:

1. Creezi un plan pentru cele {num_days} zile, în format JSON, folosind EXACT numele obiectivelor din lista de mai sus.

2. Scrii o explicație în limba română (2-3 paragrafe) în care prezinți planul și justifici alegerile făcute.

Trebuie să respecți următoarele cerințe:

{constraint_list_str}

IMPORTANT:
- La finalul răspunsului, include EXACT un bloc JSON cu următorul format:
{{
  "day1": ["nume obiectiv 1", "nume obiectiv 2"],
  "day2": ["nume obiectiv 1"],
  ...
}}
- Nu adăuga comentarii sau text după blocul JSON.
"""

    return prompt


def generate_schedule_prompt(world: World) -> str:
    """Generate Romanian prompt for schedule world"""
    payload = world.payload
    days_ro = payload["days_ro"]
    slots_ro = payload["slots_ro"]
    appointments = payload["appointments"]

    # Build appointment list
    apt_list = []
    for apt in appointments:
        apt_list.append(
            f"  • {apt['name_ro']} (prioritate: {apt['priority']}, programat: {apt['day_ro']} {apt['slot_ro']})"
        )

    apt_list_str = "\n".join(apt_list)

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_ro}")

    constraint_list_str = "\n".join(constraint_list)

    # Build prompt
    days_str = ", ".join(days_ro)
    slots_str = ", ".join(slots_ro)

    prompt = f"""Ai un calendar pentru zilele: {days_str}.
Fiecare zi are două intervale: {slots_str}.

Următoarele programări trebuie organizate:

{apt_list_str}

Te rog să:

1. Creezi un plan final de programări în format JSON.

2. Scrii o explicație în limba română despre cum ai organizat programările.

Respectă următoarele cerințe:

{constraint_list_str}

IMPORTANT:
- La finalul răspunsului, include EXACT un bloc JSON cu următorul format:
{{
  "Luni_dimineață": "nume programare sau null",
  "Luni_după-amiază": "nume programare sau null",
  ...
}}
- Nu adăuga comentarii sau text după blocul JSON.
"""

    return prompt


def generate_fact_prompt(world: World) -> str:
    """Generate Romanian prompt for fact world"""
    payload = world.payload
    facts = payload["facts"]
    question_data = payload.get("question", {})

    # Build fact database presentation - more readable format
    fact_list = []
    for key, value in facts.items():
        # Convert key to readable format
        readable_key = key.replace("_", " ").title()
        fact_list.append(f"  • {readable_key}: {value}")

    fact_list_str = "\n".join(fact_list)

    # Get the question to ask
    question = question_data.get("question_ro", "Care sunt faptele prezentate?")

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_ro}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - Răspunde pe baza faptelor date."

    # Build prompt
    prompt = f"""Ai la dispoziție următoarea bază de date cu informații:

{fact_list_str}

ATENȚIE: Răspunde DOAR pe baza informațiilor de mai sus, chiar dacă acestea par incorecte sau diferite de cunoștințele tale generale.

Întrebare: {question}

Te rog să:

1. Scrii o explicație în limba română despre răspunsul tău (1-2 paragrafe).

2. La final, răspunzi la întrebare în format JSON.

Respectă următoarele cerințe:

{constraint_list_str}

IMPORTANT:
- La finalul răspunsului, include EXACT un bloc JSON cu următorul format:
{{
  "answer": "răspunsul tău aici"
}}
- Nu adăuga comentarii sau text după blocul JSON.
"""

    return prompt


def generate_recipe_prompt(world: World) -> str:
    """Generate Romanian prompt for recipe world"""
    payload = world.payload
    num_days = payload["num_days"]
    dishes = payload["dishes"]

    # Group dishes by type
    dishes_by_type = {
        "mic_dejun": [],
        "pranz": [],
        "cina": [],
    }

    for dish in dishes:
        dish_type = dish["type"]
        if dish_type in dishes_by_type:
            # Build dish description
            attrs = []
            if dish["vegetarian"]:
                attrs.append("vegetarian")
            if dish["vegan"]:
                attrs.append("vegan")
            if dish["contains_gluten"]:
                attrs.append("conține gluten")
            if dish["contains_lactose"]:
                attrs.append("conține lactoză")
            attrs.append(f"{dish['calories']} kcal")

            attr_str = ", ".join(attrs)
            dishes_by_type[dish_type].append(f"    • {dish['name']} ({attr_str})")

    # Build dish lists
    dish_sections = []
    type_names = {
        "mic_dejun": "Mic dejun",
        "pranz": "Prânz",
        "cina": "Cină",
    }

    for dish_type, type_name in type_names.items():
        if dishes_by_type[dish_type]:
            dish_sections.append(f"  {type_name}:\n" + "\n".join(dishes_by_type[dish_type]))

    dishes_str = "\n\n".join(dish_sections)

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_ro}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - Nicio restricție specială."

    # Build prompt
    prompt = f"""Trebuie să planifici meniurile pentru {num_days} zile.

Pentru fiecare zi, alege câte un preparat pentru mic dejun, prânz și cină din opțiunile de mai jos.

Preparate disponibile:

{dishes_str}

Te rog să:

1. Scrii o explicație în limba română (2-3 paragrafe) despre cum ai ales meniurile și de ce.

2. La final, creezi un plan în format JSON.

Respectă următoarele cerințe:

{constraint_list_str}

IMPORTANT:
- La finalul răspunsului, include EXACT un bloc JSON cu următorul format:
{{
  "day1_mic_dejun": "numele preparatului",
  "day1_pranz": "numele preparatului",
  "day1_cina": "numele preparatului",
  "day2_mic_dejun": "numele preparatului",
  ...
}}
- Folosește EXACT numele preparatelor din lista de mai sus.
- Nu adăuga comentarii sau text după blocul JSON.
"""

    return prompt


def generate_prompt(world: World) -> str:
    """Generate Romanian prompt based on world type"""
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
