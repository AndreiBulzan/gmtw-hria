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

    # Build attraction list
    attr_list = []
    for attr in attractions:
        indoor_str = "interior" if attr["indoor"] else "exterior"
        family_str = "potrivit pentru copii" if attr["family_friendly"] else "nu este potrivit pentru copii mici"
        attr_list.append(
            f"  • {attr['name']} ({attr['type']}, {indoor_str}, {family_str})"
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
    misbelief_scripts = payload.get("misbelief_scripts", [])

    # Build fact database presentation
    fact_list = []
    for key, value in facts.items():
        fact_list.append(f"  • {key}: {value}")

    fact_list_str = "\n".join(fact_list)

    # Select a question to ask
    if misbelief_scripts:
        script = misbelief_scripts[0]  # For now, just use the first one
        question = script["question_ro"]
        fact_key = [k for k, v in facts.items() if v == script["true_answer"]][0]
    else:
        # Just ask for all facts
        question = "Listează toate faptele din baza de date."
        fact_key = ""

    # Build constraint list
    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            constraint_list.append(f"  - {c.description_ro}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - Răspunde pe baza faptelor date."

    # Build prompt
    prompt = f"""Ai la dispoziție următoarea bază de date cu fapte:

{fact_list_str}

Întrebare: {question}

Te rog să:

1. Răspunzi la întrebare în format JSON.

2. Scrii o explicație în limba română despre răspunsul tău.

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


def generate_prompt(world: World) -> str:
    """Generate Romanian prompt based on world type"""
    if world.world_type == "travel":
        return generate_travel_prompt(world)
    elif world.world_type == "schedule":
        return generate_schedule_prompt(world)
    elif world.world_type == "fact":
        return generate_fact_prompt(world)
    else:
        raise ValueError(f"Unknown world type: {world.world_type}")
