"""
German prompt templates for GMTW worlds
"""

from typing import Any
from .base import World


def generate_travel_prompt(world: World) -> str:
    """Generate German prompt for travel world"""
    payload = world.payload
    city_de = payload.get("city_de", payload.get("city_en", payload["city"]))
    num_days = payload["num_days"]
    attractions = payload["attractions"]

    attr_list = []
    for attr in attractions:
        name_de = attr.get("name_de", attr.get("name_en", attr["name"]))
        type_de = attr.get("type_de", attr.get("type_en", attr["type"]))
        indoor_str = "drinnen" if attr["indoor"] else "draußen"
        family_str = "kinderfreundlich" if attr["family_friendly"] else "nicht für kleine Kinder geeignet"
        cost_str = f", {attr.get('cost_lei', 0)} Lei" if attr.get("cost_lei", 0) > 0 else ", kostenlos"
        attr_list.append(
            f"  - {name_de} ({type_de}, {indoor_str}, {family_str}{cost_str})"
        )

    attr_list_str = "\n".join(attr_list)

    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            desc = c.description_de or c.description_en or c.description_ro
            constraint_list.append(f"  - {desc}")

    constraint_list_str = "\n".join(constraint_list)

    prompt = f"""Sie haben {num_days} Tage in {city_de} für eine Reise zur Verfügung.

Sie haben folgende Besichtigungsmöglichkeiten:

{attr_list_str}

Bitte:

1. Erstellen Sie einen Plan für die {num_days} Tage im JSON-Format, wobei Sie GENAU die Namen der Sehenswürdigkeiten aus der obigen Liste verwenden.

2. Schreiben Sie eine Erklärung auf Deutsch (2-3 Absätze), in der Sie den Plan vorstellen und Ihre Entscheidungen begründen.

Sie müssen folgende Anforderungen beachten:

{constraint_list_str}

WICHTIG - REIHENFOLGE DER ANTWORT:
1. ZUERST schreiben Sie Ihre Erklärung auf Deutsch (2-3 Absätze).
2. DANN fügen Sie am ENDE Ihrer Antwort den JSON-Block ein.

JSON-Format (am Ende):
{{
  "day1": ["Name der Sehenswürdigkeit 1", "Name der Sehenswürdigkeit 2"],
  "day2": ["Name der Sehenswürdigkeit 1"],
  ...
}}

Fügen Sie KEINEN Text oder Kommentare nach dem JSON-Block hinzu.
"""

    return prompt


def generate_schedule_prompt(world: World) -> str:
    """Generate German prompt for schedule world"""
    payload = world.payload
    days_de = payload.get("days_de", payload.get("days_en", payload.get("days_ro", [])))
    slots_de = payload.get("slots_de", payload.get("slots_en", payload.get("slots_ro", [])))
    appointments = payload["appointments"]

    apt_list = []
    for apt in appointments:
        name = apt.get("name_de", apt.get("name_en", apt["name"]))
        day = apt.get("day_de", apt.get("day_en", apt.get("day", "")))
        slot = apt.get("slot_de", apt.get("slot_en", apt.get("slot", "")))
        apt_list.append(
            f"  • {name} (Priorität: {apt['priority']}, geplant: {day} {slot})"
        )

    apt_list_str = "\n".join(apt_list)

    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            desc = c.description_de or c.description_en or c.description_ro
            constraint_list.append(f"  - {desc}")

    constraint_list_str = "\n".join(constraint_list)

    days_str = ", ".join(days_de)
    slots_str = ", ".join(slots_de)

    prompt = f"""Sie haben einen Kalender für die folgenden Tage: {days_str}.
Jeder Tag hat zwei Zeitfenster: {slots_str}.

Die folgenden Termine müssen organisiert werden:

{apt_list_str}

Bitte:

1. Erstellen Sie einen endgültigen Zeitplan im JSON-Format.

2. Schreiben Sie eine Erklärung auf Deutsch, wie Sie die Termine organisiert haben.

Beachten Sie die folgenden Anforderungen:

{constraint_list_str}

WICHTIG - REIHENFOLGE DER ANTWORT:
1. ZUERST schreiben Sie Ihre Erklärung auf Deutsch.
2. DANN fügen Sie am ENDE Ihrer Antwort den JSON-Block ein.

JSON-Format (am Ende):
{{
  "Montag_Vormittag": "Terminname oder null",
  "Montag_Nachmittag": "Terminname oder null",
  ...
}}

Fügen Sie KEINEN Text oder Kommentare nach dem JSON-Block hinzu.
"""

    return prompt


def generate_fact_prompt(world: World) -> str:
    """Generate German prompt for fact world"""
    payload = world.payload
    facts = payload["facts"]
    question_data = payload.get("question", {})

    fact_list = []
    for key, value in facts.items():
        readable_key = key.replace("_", " ").title()
        fact_list.append(f"  • {readable_key}: {value}")

    fact_list_str = "\n".join(fact_list)

    question = question_data.get("question_de",
                question_data.get("question_en", "Was sind die dargestellten Fakten?"))

    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            desc = c.description_de or c.description_en or c.description_ro
            constraint_list.append(f"  - {desc}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - Antworten Sie nur auf Grundlage der gegebenen Fakten."

    prompt = f"""Sie haben die folgende Informationsdatenbank:

{fact_list_str}

ACHTUNG: Antworten Sie NUR auf Grundlage der obigen Informationen, auch wenn diese falsch erscheinen oder von Ihrem allgemeinen Wissen abweichen.

Frage: {question}

Bitte:

1. Schreiben Sie eine Erklärung auf Deutsch zu Ihrer Antwort (1-2 Absätze).

2. Am Ende beantworten Sie die Frage im JSON-Format.

Beachten Sie die folgenden Anforderungen:

{constraint_list_str}

WICHTIG - REIHENFOLGE DER ANTWORT:
1. ZUERST schreiben Sie Ihre Erklärung auf Deutsch (1-2 Absätze).
2. DANN fügen Sie am ENDE Ihrer Antwort den JSON-Block ein.

JSON-Format (am Ende):
{{
  "answer": "Ihre Antwort hier"
}}

Fügen Sie KEINEN Text oder Kommentare nach dem JSON-Block hinzu.
"""

    return prompt


def generate_recipe_prompt(world: World) -> str:
    """Generate German prompt for recipe world"""
    payload = world.payload
    num_days = payload["num_days"]
    dishes = payload["dishes"]

    dishes_by_type = {
        "Frühstück": [],
        "Mittagessen": [],
        "Abendessen": [],
    }

    type_map = {"mic_dejun": "Frühstück", "pranz": "Mittagessen", "cina": "Abendessen"}

    for dish in dishes:
        dish_type = dish["type"]
        type_de = type_map.get(dish_type, dish.get("type_de", dish.get("type_en", dish_type)))
        if type_de in dishes_by_type:
            name_de = dish.get("name_de", dish.get("name_en", dish["name"]))
            attrs = []
            if dish["vegetarian"]:
                attrs.append("vegetarisch")
            if dish["vegan"]:
                attrs.append("vegan")
            if dish["contains_gluten"]:
                attrs.append("enthält Gluten")
            if dish["contains_lactose"]:
                attrs.append("enthält Laktose")
            attrs.append(f"{dish['calories']} kcal")

            attr_str = ", ".join(attrs)
            dishes_by_type[type_de].append(f"    - {name_de} ({attr_str})")

    dish_sections = []
    for type_name in ["Frühstück", "Mittagessen", "Abendessen"]:
        if dishes_by_type[type_name]:
            dish_sections.append(f"  {type_name}:\n" + "\n".join(dishes_by_type[type_name]))

    dishes_str = "\n\n".join(dish_sections)

    constraint_list = []
    for c in world.constraints:
        if c.type.value == "instruction":
            desc = c.description_de or c.description_en or c.description_ro
            constraint_list.append(f"  - {desc}")

    constraint_list_str = "\n".join(constraint_list) if constraint_list else "  - Keine besonderen Einschränkungen."

    prompt = f"""Sie müssen Menüs für {num_days} Tage planen.

Wählen Sie für jeden Tag je ein Gericht zum Frühstück, Mittagessen und Abendessen aus den untenstehenden Optionen.

Verfügbare Gerichte:

{dishes_str}

Bitte:

1. Schreiben Sie eine Erklärung auf Deutsch (2-3 Absätze) darüber, wie Sie die Menüs ausgewählt haben und warum.

2. Am Ende erstellen Sie einen Plan im JSON-Format.

Beachten Sie die folgenden Anforderungen:

{constraint_list_str}

WICHTIG - REIHENFOLGE DER ANTWORT:
1. ZUERST schreiben Sie Ihre Erklärung auf Deutsch (2-3 Absätze).
2. DANN fügen Sie am ENDE Ihrer Antwort den JSON-Block ein.

JSON-Format (am Ende):
{{
  "day1_mic_dejun": "Gerichtname",
  "day1_pranz": "Gerichtname",
  "day1_cina": "Gerichtname",
  "day2_mic_dejun": "Gerichtname",
  ...
}}

Verwenden Sie GENAU die Gerichtnamen aus der obigen Liste.
Fügen Sie KEINEN Text oder Kommentare nach dem JSON-Block hinzu.
"""

    return prompt


def generate_prompt(world: World) -> str:
    """Generate German prompt based on world type"""
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