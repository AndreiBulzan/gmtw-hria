"""
Travel World Generator for GMTW-Ro

Generates travel planning scenarios with attractions and constraints.
"""

import random
from typing import Any
from .base import World, Constraint, Goal, Entity, ConstraintType, GoalType


# Romanian cities with attractions
# Each attraction has both Romanian (name) and English (name_en) names
# city_en provides ASCII-friendly city names for English prompts
CITIES = {
    "Brașov": {
        "city_en": "Brasov",
        "attractions": [
            {
                "name": "Biserica Neagră",
                "name_en": "The Black Church",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 25,
            },
            {
                "name": "Parcul Central",
                "name_en": "Central Park",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 0,
            },
            {
                "name": "Pârtia Poiana Brașov",
                "name_en": "Poiana Brasov Ski Slope",
                "type": "sport",
                "type_en": "sports",
                "indoor": False,
                "family_friendly": False,
                "duration_hours": 4.0,
                "cost_lei": 150,
            },
            {
                "name": "Muzeul de Istorie",
                "name_en": "History Museum",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Telecabina Tâmpa",
                "name_en": "Tampa Cable Car",
                "type": "transport",
                "type_en": "transport",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 22,
            },
            {
                "name": "Turnul Alb",
                "name_en": "The White Tower",
                "type": "monument",
                "type_en": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 10,
            },
            {
                "name": "Casa Sfatului",
                "name_en": "The Council House",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 15,
            },
        ]
    },
    "Cluj-Napoca": {
        "city_en": "Cluj-Napoca",
        "attractions": [
            {
                "name": "Grădina Botanică",
                "name_en": "Botanical Garden",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 15,
            },
            {
                "name": "Muzeul Național de Artă",
                "name_en": "National Art Museum",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Cetățuia",
                "name_en": "Cetatuia Hill Fortress",
                "type": "monument",
                "type_en": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Parcul Central Simion Bărnuțiu",
                "name_en": "Central Park",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul Etnografic al Transilvaniei",
                "name_en": "Ethnographic Museum of Transylvania",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.5,
                "cost_lei": 25,
            },
            {
                "name": "Biserica Sfântul Mihail",
                "name_en": "St. Michael's Church",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
        ]
    },
    "Sibiu": {
        "city_en": "Sibiu",
        "attractions": [
            {
                "name": "Piața Mare",
                "name_en": "Grand Square",
                "type": "piață",
                "type_en": "square",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul Brukenthal",
                "name_en": "Brukenthal Museum",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.5,
                "cost_lei": 30,
            },
            {
                "name": "Podul Minciunilor",
                "name_en": "Bridge of Lies",
                "type": "monument",
                "type_en": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 0.5,
                "cost_lei": 0,
            },
            {
                "name": "Turnul Sfatului",
                "name_en": "Council Tower",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 10,
            },
            {
                "name": "Grădina Zoologică",
                "name_en": "Zoo",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 3.0,
                "cost_lei": 25,
            },
            {
                "name": "Muzeul ASTRA",
                "name_en": "ASTRA Open-Air Museum",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 4.0,
                "cost_lei": 35,
            },
        ]
    },
    "Timișoara": {
        "city_en": "Timisoara",
        "attractions": [
            {
                "name": "Piața Victoriei",
                "name_en": "Victory Square",
                "type": "piață",
                "type_en": "square",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Catedrala Mitropolitană",
                "name_en": "Metropolitan Cathedral",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
            {
                "name": "Parcul Rozelor",
                "name_en": "Rose Park",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul de Artă",
                "name_en": "Art Museum",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Bastionul Theresia",
                "name_en": "Theresia Bastion",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": False,
                "duration_hours": 1.5,
                "cost_lei": 15,
            },
            {
                "name": "Grădina Zoologică Timișoara",
                "name_en": "Timisoara Zoo",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 3.0,
                "cost_lei": 20,
            },
        ]
    },
    "Iași": {
        "city_en": "Iasi",
        "attractions": [
            {
                "name": "Palatul Culturii",
                "name_en": "Palace of Culture",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 3.0,
                "cost_lei": 30,
            },
            {
                "name": "Grădina Botanică Iași",
                "name_en": "Iasi Botanical Garden",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.5,
                "cost_lei": 15,
            },
            {
                "name": "Mănăstirea Trei Ierarhi",
                "name_en": "Three Hierarchs Monastery",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
            {
                "name": "Teatrul Național Iași",
                "name_en": "Iasi National Theatre",
                "type": "monument",
                "type_en": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 0.5,
                "cost_lei": 0,
            },
            {
                "name": "Parcul Copou",
                "name_en": "Copou Park",
                "type": "parc",
                "type_en": "park",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 0,
            },
            {
                "name": "Casa Memorială Mihai Eminescu",
                "name_en": "Mihai Eminescu Memorial House",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 10,
            },
        ]
    },
    "Constanța": {
        "city_en": "Constanta",
        "attractions": [
            {
                "name": "Cazinoul din Constanța",
                "name_en": "Constanta Casino",
                "type": "monument",
                "type_en": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul de Istorie și Arheologie",
                "name_en": "History and Archaeology Museum",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Delfinariul",
                "name_en": "Dolphinarium",
                "type": "parc",
                "type_en": "park",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 45,
            },
            {
                "name": "Plaja Modern",
                "name_en": "Modern Beach",
                "type": "plajă",
                "type_en": "beach",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 4.0,
                "cost_lei": 0,
            },
            {
                "name": "Acvariul Constanța",
                "name_en": "Constanta Aquarium",
                "type": "muzeu",
                "type_en": "museum",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 30,
            },
            {
                "name": "Farul Genovez",
                "name_en": "Genoese Lighthouse",
                "type": "monument",
                "type_en": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 0.5,
                "cost_lei": 0,
            },
        ]
    },
}


class TravelWorldGenerator:
    """Generator for Travel World instances"""

    def __init__(self, spec_version: str = "0.1"):
        self.spec_version = spec_version

    def generate(
        self,
        world_id: str,
        seed: int,
        difficulty: str = "easy",
        **kwargs: Any
    ) -> World:
        """
        Generate a travel world instance

        Args:
            world_id: Unique identifier
            seed: Random seed for reproducibility
            difficulty: easy|medium|hard
            **kwargs: Additional parameters
        """
        rng = random.Random(seed)

        # Select city
        city = rng.choice(list(CITIES.keys()))
        city_data = CITIES[city]

        # Number of days
        num_days = rng.choice([2, 3]) if difficulty == "easy" else rng.choice([3, 4])

        # Select attractions (subset)
        all_attractions = city_data["attractions"]
        num_attractions = min(len(all_attractions), rng.randint(4, 6))
        selected_attractions = rng.sample(all_attractions, num_attractions)

        # Create entities
        entities = {}
        attractions_list = []

        for idx, attr in enumerate(selected_attractions):
            attr_id = f"A{idx + 1}"
            name_en = attr.get("name_en", attr["name"])
            # Build aliases: Romanian name, Romanian stripped, English name, English lower
            aliases = [
                attr["name"].lower(),
                attr["name"].lower().replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ț", "t"),
                name_en,
                name_en.lower(),
            ]
            entities[attr_id] = Entity(
                id=attr_id,
                name=attr["name"],
                aliases=aliases,
                attributes=attr,
            )

            attractions_list.append({
                "id": attr_id,
                "name": attr["name"],
                "name_en": attr.get("name_en", attr["name"]),
                "type": attr["type"],
                "type_en": attr.get("type_en", attr["type"]),
                "indoor": attr["indoor"],
                "family_friendly": attr["family_friendly"],
                "duration_hours": attr["duration_hours"],
                "cost_lei": attr.get("cost_lei", 0),
            })

        # Calculate total possible cost for budget constraint
        total_possible_cost = sum(a.get("cost_lei", 0) for a in selected_attractions)

        # Generate constraints
        constraints = []

        # Must include at least one monument
        has_monument = any(a["type"] == "monument" for a in selected_attractions)
        if has_monument:
            constraints.append(
                Constraint(
                    id="C_MUST_MONUMENT",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Trebuie să incluzi cel puțin un monument istoric în întregul plan.",
                    description_en="You must include at least one historic monument in the entire plan.",
                    check_fn="check_must_include_type",
                    params={"type_required": "monument"},
                )
            )

        # Must include at least one museum (medium/hard difficulty)
        has_museum = any(a["type"] == "muzeu" for a in selected_attractions)
        if has_museum and difficulty in ("medium", "hard"):
            constraints.append(
                Constraint(
                    id="C_MUST_MUSEUM",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Trebuie să incluzi cel puțin un muzeu în întregul plan.",
                    description_en="You must include at least one museum in the entire plan.",
                    check_fn="check_must_include_type",
                    params={"type_required": "muzeu"},
                )
            )

        # Max outdoor per day
        max_outdoor = rng.choice([1, 2]) if difficulty != "hard" else 1
        activity_word = "activitate" if max_outdoor == 1 else "activități"
        constraints.append(
            Constraint(
                id="C_MAX_OUTDOOR_PER_DAY",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Maxim {max_outdoor} {activity_word} în aer liber pe zi.",
                description_en=f"At most {max_outdoor} outdoor {'activity' if max_outdoor == 1 else 'activities'} per day.",
                check_fn="check_max_outdoor_per_day",
                params={"max_outdoor": max_outdoor},
            )
        )

        # Family friendly constraint
        family_trip = rng.choice([True, False])
        if family_trip:
            constraints.append(
                Constraint(
                    id="C_FAMILY_FRIENDLY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Nu include activități care nu sunt potrivite pentru copii mici.",
                    description_en="Do not include activities that are not suitable for small children.",
                    check_fn="check_all_family_friendly",
                    params={},
                )
            )

        # Budget constraint (medium/hard difficulty)
        if difficulty in ("medium", "hard") and total_possible_cost > 50:
            # Set budget to 60-80% of total possible cost
            budget = int(total_possible_cost * rng.uniform(0.5, 0.75))
            constraints.append(
                Constraint(
                    id="C_BUDGET",
                    type=ConstraintType.INSTRUCTION,
                    description_ro=f"Bugetul total pentru activități nu trebuie să depășească {budget} lei.",
                    description_en=f"The total budget for activities must not exceed {budget} lei.",
                    check_fn="check_budget_limit",
                    params={"max_budget": budget},
                )
            )

        # No duplicates constraint (optional, ~30% chance on medium/hard)
        if difficulty in ("medium", "hard") and rng.random() < 0.3:
            constraints.append(
                Constraint(
                    id="C_NO_DUPLICATES",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Nu vizita același loc de două ori.",
                    description_en="Do not visit the same place twice.",
                    check_fn="check_no_duplicates",
                    params={},
                )
            )

        # =====================================================================
        # HARD MODE CONSTRAINTS
        # =====================================================================

        if difficulty == "hard":
            # Duration limit per day - calculate based on available attractions
            total_duration = sum(a["duration_hours"] for a in selected_attractions)
            avg_duration = total_duration / len(selected_attractions) if selected_attractions else 2.0
            # Allow about 2-3 activities worth of time per day
            max_hours = rng.uniform(avg_duration * 2, avg_duration * 3)
            max_hours = round(max_hours, 1)

            constraints.append(
                Constraint(
                    id="C_MAX_DURATION",
                    type=ConstraintType.INSTRUCTION,
                    description_ro=f"Timpul total de vizită pe zi nu trebuie să depășească {max_hours} ore.",
                    description_en=f"The total visit time per day must not exceed {max_hours} hours.",
                    check_fn="check_max_duration_per_day",
                    params={"max_hours": max_hours},
                )
            )

            # Type diversity - must include at least 3 different types
            available_types = set(a["type"] for a in selected_attractions)
            if len(available_types) >= 3:
                constraints.append(
                    Constraint(
                        id="C_TYPE_DIVERSITY",
                        type=ConstraintType.INSTRUCTION,
                        description_ro="Planul trebuie să includă cel puțin 3 tipuri diferite de activități (ex: muzeu, parc, monument).",
                        description_en="The plan must include at least 3 different types of activities (e.g., museum, park, monument).",
                        check_fn="check_type_diversity",
                        params={"min_types": 3},
                    )
                )

            # Exclude a type (50% chance) - pick a type that has alternatives
            type_counts = {}
            for a in selected_attractions:
                t = a["type"]
                type_counts[t] = type_counts.get(t, 0) + 1

            # Types that are required by other constraints - cannot exclude these
            required_types = set()
            if has_monument:
                required_types.add("monument")
            if has_museum and difficulty in ("medium", "hard"):
                required_types.add("muzeu")

            # Only exclude if there are enough other activities and it's not required
            excludable_types = [
                t for t, c in type_counts.items()
                if c == 1 and len(type_counts) > 2 and t not in required_types
            ]
            if excludable_types and rng.random() < 0.5:
                type_to_exclude = rng.choice(excludable_types)
                type_names_ro = {
                    "monument": "monumente",
                    "parc": "parcuri",
                    "muzeu": "muzee",
                    "sport": "activități sportive",
                    "transport": "transport turistic",
                    "piață": "piețe",
                    "plajă": "plaje",
                }
                type_names_en = {
                    "monument": "monuments",
                    "parc": "parks",
                    "muzeu": "museums",
                    "sport": "sports activities",
                    "transport": "tourist transport",
                    "piață": "squares",
                    "plajă": "beaches",
                }
                constraints.append(
                    Constraint(
                        id="C_EXCLUDE_TYPE",
                        type=ConstraintType.INSTRUCTION,
                        description_ro=f"Nu include {type_names_ro.get(type_to_exclude, type_to_exclude)} în plan.",
                        description_en=f"Do not include {type_names_en.get(type_to_exclude, type_to_exclude)} in the plan.",
                        check_fn="check_must_exclude_type",
                        params={"type_forbidden": type_to_exclude},
                    )
                )

        # Generate goals
        goals = [
            Goal(
                id="G_FILL_DAYS",
                type=GoalType.STRUCTURAL,
                description="Each day must have at least one activity",
                check_fn="check_days_non_empty",
                params={"num_days": num_days},
            ),
            Goal(
                id="G_VALID_IDS",
                type=GoalType.STRUCTURAL,
                description="All referenced IDs must exist in the world",
                check_fn="check_valid_entity_ids",
                params={"valid_ids": list(entities.keys())},
            ),
        ]

        # Build payload
        payload = {
            "city": city,
            "city_en": city_data.get("city_en", city),
            "num_days": num_days,
            "attractions": attractions_list,
        }

        # Metadata
        meta = {
            "difficulty": difficulty,
            "family_trip": family_trip,
            "num_attractions": len(selected_attractions),
        }

        return World(
            world_id=world_id,
            world_type="travel",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )
