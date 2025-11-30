"""
Travel World Generator for GMTW-Ro

Generates travel planning scenarios with attractions and constraints.
"""

import random
from typing import Any
from .base import World, Constraint, Goal, Entity, ConstraintType, GoalType


# Romanian cities with attractions
CITIES = {
    "Brașov": {
        "attractions": [
            {
                "name": "Biserica Neagră",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 25,
            },
            {
                "name": "Parcul Central",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 0,
            },
            {
                "name": "Pârtia Poiana Brașov",
                "type": "sport",
                "indoor": False,
                "family_friendly": False,
                "duration_hours": 4.0,
                "cost_lei": 150,
            },
            {
                "name": "Muzeul de Istorie",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Telecabina Tâmpa",
                "type": "transport",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 22,
            },
            {
                "name": "Turnul Alb",
                "type": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 10,
            },
            {
                "name": "Casa Sfatului",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 15,
            },
        ]
    },
    "Cluj-Napoca": {
        "attractions": [
            {
                "name": "Grădina Botanică",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 15,
            },
            {
                "name": "Muzeul Național de Artă",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Cetățuia",
                "type": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Parcul Central Simion Bărnuțiu",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul Etnografic al Transilvaniei",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.5,
                "cost_lei": 25,
            },
            {
                "name": "Biserica Sfântul Mihail",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
        ]
    },
    "Sibiu": {
        "attractions": [
            {
                "name": "Piața Mare",
                "type": "piață",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul Brukenthal",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.5,
                "cost_lei": 30,
            },
            {
                "name": "Podul Minciunilor",
                "type": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 0.5,
                "cost_lei": 0,
            },
            {
                "name": "Turnul Sfatului",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 10,
            },
            {
                "name": "Grădina Zoologică",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 3.0,
                "cost_lei": 25,
            },
            {
                "name": "Muzeul ASTRA",
                "type": "muzeu",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 4.0,
                "cost_lei": 35,
            },
        ]
    },
    "Timișoara": {
        "attractions": [
            {
                "name": "Piața Victoriei",
                "type": "piață",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 0,
            },
            {
                "name": "Catedrala Mitropolitană",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
            {
                "name": "Parcul Rozelor",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul de Artă",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Bastionul Theresia",
                "type": "monument",
                "indoor": True,
                "family_friendly": False,
                "duration_hours": 1.5,
                "cost_lei": 15,
            },
            {
                "name": "Grădina Zoologică Timișoara",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 3.0,
                "cost_lei": 20,
            },
        ]
    },
    "Iași": {
        "attractions": [
            {
                "name": "Palatul Culturii",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 3.0,
                "cost_lei": 30,
            },
            {
                "name": "Grădina Botanică Iași",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.5,
                "cost_lei": 15,
            },
            {
                "name": "Mănăstirea Trei Ierarhi",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
            {
                "name": "Teatrul Național Iași",
                "type": "monument",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 0.5,
                "cost_lei": 0,
            },
            {
                "name": "Parcul Copou",
                "type": "parc",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 0,
            },
            {
                "name": "Casa Memorială Mihai Eminescu",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 10,
            },
        ]
    },
    "Constanța": {
        "attractions": [
            {
                "name": "Cazinoul din Constanța",
                "type": "monument",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 1.0,
                "cost_lei": 0,
            },
            {
                "name": "Muzeul de Istorie și Arheologie",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 20,
            },
            {
                "name": "Delfinariul",
                "type": "parc",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 2.0,
                "cost_lei": 45,
            },
            {
                "name": "Plaja Modern",
                "type": "plajă",
                "indoor": False,
                "family_friendly": True,
                "duration_hours": 4.0,
                "cost_lei": 0,
            },
            {
                "name": "Acvariul Constanța",
                "type": "muzeu",
                "indoor": True,
                "family_friendly": True,
                "duration_hours": 1.5,
                "cost_lei": 30,
            },
            {
                "name": "Farul Genovez",
                "type": "monument",
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
            entities[attr_id] = Entity(
                id=attr_id,
                name=attr["name"],
                aliases=[
                    attr["name"].lower(),
                    attr["name"].lower().replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ț", "t"),
                ],
                attributes=attr,
            )

            attractions_list.append({
                "id": attr_id,
                "name": attr["name"],
                "type": attr["type"],
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
