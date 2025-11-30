"""
Fact World Generator for GMTW-Ro

Generates fact retrieval scenarios with hallucination traps.
"""

import random
from typing import Any
from .base import World, Constraint, Goal, Entity, ConstraintType, GoalType


# Fact database - organized by category
# Each fact can have a "misbelief" version that contradicts common knowledge
FACTS = {
    "geography": [
        {
            "key": "capital_romania",
            "question_ro": "Care este capitala României?",
            "question_en": "What is the capital of Romania?",
            "answer": "București",
            "typical_wrong": ["Cluj-Napoca", "Brașov"],
            "misbelief_answer": "Sibiu",  # For testing context adherence
        },
        {
            "key": "capital_germany",
            "question_ro": "Care este capitala Germaniei?",
            "question_en": "What is the capital of Germany?",
            "answer": "Berlin",
            "typical_wrong": ["München", "Hamburg"],
        },
        {
            "key": "danube_flows",
            "question_ro": "În ce mare se varsă Dunărea?",
            "question_en": "Which sea does the Danube flow into?",
            "answer": "Marea Neagră",
            "typical_wrong": ["Marea Mediterană", "Marea Adriatică"],
        },
        {
            "key": "longest_river_romania",
            "question_ro": "Care este cel mai lung râu din România?",
            "question_en": "What is the longest river in Romania?",
            "answer": "Dunărea",
            "typical_wrong": ["Mureșul", "Oltul"],
        },
        {
            "key": "highest_peak_romania",
            "question_ro": "Care este cel mai înalt vârf din România?",
            "question_en": "What is the highest peak in Romania?",
            "answer": "Moldoveanu (2544m)",
            "typical_wrong": ["Negoiu", "Omu"],
        },
        {
            "key": "neighbors_romania",
            "question_ro": "Câte țări sunt vecine cu România?",
            "question_en": "How many countries border Romania?",
            "answer": "5 (Ucraina, Moldova, Bulgaria, Serbia, Ungaria)",
            "typical_wrong": ["4", "6"],
        },
    ],
    "history": [
        {
            "key": "romania_independence",
            "question_ro": "În ce an și-a proclamat România independența?",
            "question_en": "In what year did Romania declare independence?",
            "answer": "1877",
            "typical_wrong": ["1918", "1859"],
        },
        {
            "key": "romania_eu",
            "question_ro": "În ce an a aderat România la Uniunea Europeană?",
            "question_en": "In what year did Romania join the European Union?",
            "answer": "2007",
            "typical_wrong": ["2004", "2010"],
        },
        {
            "key": "first_king",
            "question_ro": "Cine a fost primul rege al României?",
            "question_en": "Who was the first king of Romania?",
            "answer": "Carol I",
            "typical_wrong": ["Ferdinand I", "Mihai I"],
        },
        {
            "key": "great_union",
            "question_ro": "În ce an s-a realizat Marea Unire?",
            "question_en": "In what year did the Great Union take place?",
            "answer": "1918",
            "typical_wrong": ["1859", "1920"],
        },
    ],
    "culture": [
        {
            "key": "national_poet",
            "question_ro": "Cine este poetul național al României?",
            "question_en": "Who is the national poet of Romania?",
            "answer": "Mihai Eminescu",
            "typical_wrong": ["Ion Creangă", "George Coșbuc"],
        },
        {
            "key": "national_day",
            "question_ro": "Când este ziua națională a României?",
            "question_en": "When is Romania's national day?",
            "answer": "1 Decembrie",
            "typical_wrong": ["24 Ianuarie", "9 Mai"],
        },
        {
            "key": "currency",
            "question_ro": "Care este moneda națională a României?",
            "question_en": "What is the national currency of Romania?",
            "answer": "Leul (RON)",
            "typical_wrong": ["Euro", "Dolar"],
        },
    ],
    "science": [
        {
            "key": "water_boiling",
            "question_ro": "La ce temperatură fierbe apa la presiune normală?",
            "question_en": "At what temperature does water boil at normal pressure?",
            "answer": "100°C",
            "typical_wrong": ["90°C", "110°C"],
            "misbelief_answer": "85°C",  # For testing context adherence
        },
        {
            "key": "speed_of_light",
            "question_ro": "Care este viteza luminii în vid?",
            "question_en": "What is the speed of light in vacuum?",
            "answer": "aproximativ 300.000 km/s",
            "typical_wrong": ["150.000 km/s", "500.000 km/s"],
        },
    ],
    "general": [
        {
            "key": "population_romania",
            "question_ro": "Care este populația aproximativă a României?",
            "question_en": "What is the approximate population of Romania?",
            "answer": "aproximativ 19 milioane",
            "typical_wrong": ["aproximativ 30 milioane", "aproximativ 10 milioane"],
        },
        {
            "key": "capital_population",
            "question_ro": "Care este populația aproximativă a Bucureștiului?",
            "question_en": "What is the approximate population of Bucharest?",
            "answer": "aproximativ 2 milioane",
            "typical_wrong": ["aproximativ 5 milioane", "aproximativ 1 milion"],
        },
    ],
}


class FactWorldGenerator:
    """Generator for Fact World instances"""

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
        Generate a fact world instance

        Args:
            world_id: Unique identifier
            seed: Random seed for reproducibility
            difficulty: easy|medium|hard
            **kwargs: Additional parameters
        """
        rng = random.Random(seed)

        # Select facts from different categories
        all_facts = []
        for category, facts in FACTS.items():
            all_facts.extend([(category, f) for f in facts])

        num_facts = rng.randint(3, 5) if difficulty == "easy" else rng.randint(4, 6)
        selected = rng.sample(all_facts, min(num_facts, len(all_facts)))

        # Build fact database
        fact_db = {}
        entities = {}
        misbelief_scripts = []
        question_to_ask = None

        for idx, (category, fact) in enumerate(selected):
            fact_id = f"F{idx + 1}"

            # On medium/hard difficulty, sometimes use misbelief answer
            # This tests if the model follows context over parametric knowledge
            use_misbelief = (
                difficulty in ("medium", "hard")
                and "misbelief_answer" in fact
                and rng.random() < 0.4  # 40% chance to use misbelief
            )

            if use_misbelief:
                # Override with "wrong" answer to test context adherence
                answer = fact["misbelief_answer"]
                misbelief_scripts.append({
                    "fact_id": fact_id,
                    "question_ro": fact["question_ro"],
                    "question_en": fact["question_en"],
                    "context_answer": answer,
                    "real_world_answer": fact["answer"],
                    "is_trap": True,
                })
                # This should be the question we ask
                question_to_ask = fact
            else:
                answer = fact["answer"]

            fact_db[fact["key"]] = answer

            # Create entity
            entities[fact_id] = Entity(
                id=fact_id,
                name=fact["key"],
                aliases=[fact["key"]],
                attributes={
                    "question_ro": fact["question_ro"],
                    "question_en": fact["question_en"],
                    "answer": answer,
                    "category": category,
                },
            )

        # If we didn't set a trap question, pick one randomly
        if question_to_ask is None:
            question_to_ask = selected[0][1]

        # Generate constraints
        constraints = [
            Constraint(
                id="C_ANSWER_FROM_CONTEXT",
                type=ConstraintType.INSTRUCTION,
                description_ro="Răspunde DOAR pe baza informațiilor din contextul dat, nu din cunoștințele tale generale.",
                description_en="Answer ONLY based on the information in the given context, not from your general knowledge.",
                check_fn="check_answer_matches_context",
                params={},
            ),
        ]

        # Generate goals
        goals = [
            Goal(
                id="G_NO_HALLUCINATION",
                type=GoalType.STRUCTURAL,
                description="All provided answers must match the fact database",
                check_fn="check_no_hallucinated_facts",
                params={"fact_db": fact_db},
            ),
        ]

        # Build payload
        payload = {
            "facts": fact_db,
            "misbelief_scripts": misbelief_scripts,
            "question": {
                "question_ro": question_to_ask["question_ro"],
                "question_en": question_to_ask["question_en"],
                "expected_answer": fact_db.get(question_to_ask["key"], question_to_ask["answer"]),
            },
        }

        # Metadata
        meta = {
            "difficulty": difficulty,
            "num_facts": len(selected),
            "num_traps": len(misbelief_scripts),
            "has_misbelief": len(misbelief_scripts) > 0,
        }

        return World(
            world_id=world_id,
            world_type="fact",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )
