"""
Schedule World Generator for GMTW-Ro

Generates calendar scheduling scenarios with priority constraints.
"""

import random
from typing import Any
from .base import World, Constraint, Goal, Entity, ConstraintType, GoalType


DAYS_RO = ["Luni", "Marți", "Miercuri", "Joi", "Vineri"]
DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SLOTS_RO = ["dimineață", "după-amiază"]
SLOTS_EN = ["morning", "afternoon"]

MEETING_TYPES = [
    {"name_ro": "Control medical", "name_en": "Medical checkup"},
    {"name_ro": "Ședință de proiect", "name_en": "Project meeting"},
    {"name_ro": "Antrenament sportiv", "name_en": "Sports training"},
    {"name_ro": "Întâlnire cu clientul", "name_en": "Client meeting"},
    {"name_ro": "Prezentare raport", "name_en": "Report presentation"},
    {"name_ro": "Workshop tehnic", "name_en": "Technical workshop"},
    {"name_ro": "Ședință de echipă", "name_en": "Team meeting"},
]


class ScheduleWorldGenerator:
    """Generator for Schedule World instances"""

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
        Generate a schedule world instance

        Args:
            world_id: Unique identifier
            seed: Random seed for reproducibility
            difficulty: easy|medium|hard
            **kwargs: Additional parameters
        """
        rng = random.Random(seed)

        # Number of days
        num_days = 2 if difficulty == "easy" else rng.choice([2, 3])
        selected_days_ro = DAYS_RO[:num_days]
        selected_days_en = DAYS_EN[:num_days]

        # Number of slots per day
        slots = SLOTS_RO

        # Generate appointments
        num_appointments = rng.randint(3, 5)
        appointments = []
        entities = {}

        meeting_types = rng.sample(MEETING_TYPES, min(num_appointments, len(MEETING_TYPES)))

        for idx, meeting_type in enumerate(meeting_types):
            apt_id = f"M{idx + 1}"

            # Assign to a day and slot
            day_idx = rng.randint(0, num_days - 1)
            slot_idx = rng.randint(0, len(slots) - 1)

            priority = rng.choice(["high", "medium", "low"])

            appointment = {
                "id": apt_id,
                "name_ro": meeting_type["name_ro"],
                "name_en": meeting_type["name_en"],
                "day_ro": selected_days_ro[day_idx],
                "day_en": selected_days_en[day_idx],
                "slot_ro": slots[slot_idx],
                "slot_en": SLOTS_EN[slot_idx],
                "priority": priority,
            }

            appointments.append(appointment)

            entities[apt_id] = Entity(
                id=apt_id,
                name=meeting_type["name_ro"],
                aliases=[meeting_type["name_ro"].lower()],
                attributes=appointment,
            )

        # Generate constraints
        constraints = []

        # Max appointments per day
        max_per_day = rng.choice([2, 3]) if difficulty == "easy" else 2
        constraints.append(
            Constraint(
                id="C_MAX_PER_DAY",
                type=ConstraintType.INSTRUCTION,
                description_ro=f"Maxim {max_per_day} programări pe zi.",
                description_en=f"At most {max_per_day} appointments per day.",
                check_fn="check_max_appointments_per_day",
                params={"max_per_day": max_per_day},
            )
        )

        # Must keep high priority
        has_high = any(a["priority"] == "high" for a in appointments)
        if has_high:
            constraints.append(
                Constraint(
                    id="C_KEEP_HIGH_PRIORITY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Trebuie să păstrezi toate programările cu prioritate înaltă.",
                    description_en="You must keep all high-priority appointments.",
                    check_fn="check_keep_high_priority",
                    params={},
                )
            )

        # Generate goals
        goals = [
            Goal(
                id="G_NO_OVERLAPS",
                type=GoalType.STRUCTURAL,
                description="No slot can have more than one appointment",
                check_fn="check_no_slot_overlaps",
                params={"days": selected_days_ro, "slots": slots},
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
            "days_ro": selected_days_ro,
            "days_en": selected_days_en,
            "slots_ro": slots,
            "slots_en": SLOTS_EN,
            "appointments": appointments,
        }

        # Metadata
        meta = {
            "difficulty": difficulty,
            "num_appointments": len(appointments),
            "num_days": num_days,
        }

        return World(
            world_id=world_id,
            world_type="schedule",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )
