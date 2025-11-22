"""
Fact World Generator for GMTW-Ro

Generates fact retrieval scenarios with hallucination traps.
"""

import random
from typing import Any
from .base import World, Constraint, Goal, Entity, ConstraintType, GoalType


# Fact database
FACTS = {
    "geography": [
        {
            "key": "capital_romania",
            "question_ro": "Care este capitala României?",
            "question_en": "What is the capital of Romania?",
            "answer": "București",
            "typical_wrong": ["Cluj-Napoca", "Brașov"],
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
    ],
    "history": [
        {
            "key": "romania_independence",
            "question_ro": "În ce an și-a proclamat România independența?",
            "question_en": "In what year did Romania declare independence?",
            "answer": "1877",
            "typical_wrong": ["1918", "1859"],
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

        # Select facts
        all_facts = []
        for category, facts in FACTS.items():
            all_facts.extend(facts)

        num_facts = rng.randint(3, 5)
        selected_facts = rng.sample(all_facts, min(num_facts, len(all_facts)))

        # Build fact database
        fact_db = {}
        entities = {}
        misbelief_scripts = []

        for idx, fact in enumerate(selected_facts):
            fact_id = f"F{idx + 1}"
            fact_db[fact["key"]] = fact["answer"]

            # Create entity
            entities[fact_id] = Entity(
                id=fact_id,
                name=fact["key"],
                aliases=[fact["key"]],
                attributes={
                    "question_ro": fact["question_ro"],
                    "question_en": fact["question_en"],
                    "answer": fact["answer"],
                },
            )

            # Some facts become misbelief traps
            if rng.random() < 0.5:
                misbelief_scripts.append({
                    "fact_id": fact_id,
                    "question_ro": fact["question_ro"],
                    "question_en": fact["question_en"],
                    "true_answer": fact["answer"],
                    "typical_wrong": fact["typical_wrong"],
                })

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
        }

        # Metadata
        meta = {
            "difficulty": difficulty,
            "num_facts": len(selected_facts),
            "num_traps": len(misbelief_scripts),
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
