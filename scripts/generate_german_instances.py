#!/usr/bin/env python3
"""
Generate German test instances from existing Romanian/English instances.

Takes the first N instances from gmtw_ro_v0.jsonl, adds German prompt
(prompt_de) and German names (name_de) to attractions, and writes them out.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rombench.gmtw_ro.worlds.base import Instance
from rombench.gmtw_ro.worlds.templates_de import generate_prompt


# Mapping of attraction names (EN → DE) for the cities in our dataset
NAME_EN_TO_DE = {
    # Timisoara
    "Rose Park": "Rosenpark",
    "Theresia Bastion": "Theresia-Bastion",
    "Art Museum": "Kunstmuseum",
    "Metropolitan Cathedral": "Metropolitankathedrale",
    # Cluj-Napoca
    "Botanical Garden": "Botanischer Garten",
    "Central Park": "Zentralpark",
    "Ethnographic Museum of Transylvania": "Ethnographisches Museum Siebenbürgens",
    "National Art Museum": "Nationales Kunstmuseum",
    "Cetatuia Hill Fortress": "Festung Cetățuia",
    "St. Michael's Church": "St.-Michaels-Kirche",
    # Brasov
    "Poiana Brasov Ski Slope": "Skipiste Poiana Brașov",
    "The Council House": "Das Rathaus",
    "Tampa Cable Car": "Seilbahn Tampa",
    # Sibiu
    "Brukenthal Museum": "Brukenthal-Museum",
    "ASTRA Museum": "ASTRA-Museum",
    "Council Tower": "Ratsturm",
    "Evangelical Cathedral": "Evangelische Kathedrale",
    "Citadel": "Zitadelle",
    "Sub Arini Park": "Sub-Arini-Park",
    "Bridge of Lies": "Lügenbrücke",
    # Bucharest
    "Palace of the Parliament": "Parlamentspalast",
    "Herastrau Park": "Herăstrău-Park",
    "Romanian Athenaeum": "Rumänisches Athenäum",
    "Village Museum": "Dorfmuseum",
    "Old Town": "Altstadt",
    "Cismigiu Garden": "Cișmigiu-Garten",
    "Revolution Square": "Revolutionsplatz",
    "National Museum of Art": "Nationales Kunstmuseum",
    "Triumphal Arch": "Triumphbogen",
    "Carol Park": "Carol-Park",
    "Cotroceni Palace": "Cotroceni-Palast",
}

TYPE_EN_TO_DE = {
    "park": "Park",
    "monument": "Denkmal",
    "museum": "Museum",
    "sports": "Sport",
    "transport": "Transport",
    "restaurant": "Restaurant",
    "church": "Kirche",
    "gallery": "Galerie",
}

CITY_EN_TO_DE = {
    "Timisoara": "Temeswar",
    "Cluj-Napoca": "Klausenburg",
    "Brasov": "Kronstadt",
    "Sibiu": "Hermannstadt",
    "Bucharest": "Bukarest",
    "Iasi": "Jassy",
    "Constanta": "Konstanza",
    "Oradea": "Großwardein",
}

# German translations for common constraint descriptions
CONSTRAINT_EN_TO_DE = {
    "You must include at least one historic monument in the entire plan.":
        "Sie müssen mindestens ein historisches Denkmal im gesamten Plan einbeziehen.",
    "At most 2 outdoor activities per day.":
        "Höchstens 2 Außenaktivitäten pro Tag.",
    "At most 1 outdoor activity per day.":
        "Höchstens 1 Außenaktivität pro Tag.",
    "Do not include activities that are not suitable for small children.":
        "Schließen Sie keine Aktivitäten ein, die nicht für kleine Kinder geeignet sind.",
    "Maximum budget: 50 lei per day.":
        "Maximalbudget: 50 Lei pro Tag.",
    "Maximum budget: 100 lei per day.":
        "Maximalbudget: 100 Lei pro Tag.",
    "At least one different type of activity per day.":
        "Mindestens eine verschiedene Art von Aktivität pro Tag.",
    "No two consecutive activities of the same type.":
        "Keine zwei aufeinanderfolgenden Aktivitäten der gleichen Art.",
}


def add_german_to_instance(data: dict) -> dict:
    """Add German fields to a raw instance dict."""
    world = data["world"]
    payload = world["payload"]

    # Add city_de
    city_en = payload.get("city_en", payload.get("city", ""))
    payload["city_de"] = CITY_EN_TO_DE.get(city_en, city_en)

    # Add name_de and type_de to attractions
    if "attractions" in payload:
        for attr in payload["attractions"]:
            name_en = attr.get("name_en", "")
            type_en = attr.get("type_en", "")
            attr["name_de"] = NAME_EN_TO_DE.get(name_en, name_en)
            attr["type_de"] = TYPE_EN_TO_DE.get(type_en, type_en)

    # Add name_de to canonical_entities aliases
    for eid, ent in world.get("canonical_entities", {}).items():
        name_en = ent.get("attributes", {}).get("name_en", "")
        name_de = NAME_EN_TO_DE.get(name_en, name_en)
        ent["attributes"]["name_de"] = name_de
        # Add German name to aliases
        if name_de and name_de not in ent.get("aliases", []):
            ent.setdefault("aliases", []).append(name_de)
            ent["aliases"].append(name_de.lower())

    # Add description_de to constraints
    for c in world.get("constraints", []):
        desc_en = c.get("description_en", "")
        c["description_de"] = CONSTRAINT_EN_TO_DE.get(desc_en, desc_en)

    return data


def generate_german_instances(input_file: str, output_file: str, count: int = 10):
    """Generate German test instances."""
    instances = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            data = json.loads(line)
            data = add_german_to_instance(data)
            # Parse as Instance to generate prompt_de 
            inst = Instance.from_dict(data)
            prompt_de = generate_prompt(inst.world)
            data["prompt_de"] = prompt_de
            instances.append(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        for inst_data in instances:
            f.write(json.dumps(inst_data, ensure_ascii=False) + '\n')

    print(f"Generated {len(instances)} German instances → {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate German test instances")
    parser.add_argument("--input", default="data/gmtw_ro_v0.jsonl", help="Input instance file")
    parser.add_argument("--output", default="data/gmtw_de_test.jsonl", help="Output file")
    parser.add_argument("--count", type=int, default=10, help="Number of instances")
    args = parser.parse_args()

    generate_german_instances(args.input, args.output, args.count)