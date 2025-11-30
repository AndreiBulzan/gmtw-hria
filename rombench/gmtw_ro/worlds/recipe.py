"""
Recipe World Generator for GMTW-Ro

Generates meal planning scenarios with ingredients and dietary constraints.
Tests: planning, constraint satisfaction, dietary restriction handling.
"""

import random
from typing import Any
from .base import World, Constraint, Goal, Entity, ConstraintType, GoalType


# Dishes with their attributes
# Each dish has both Romanian (name) and English (name_en) names
# type_en provides English meal type for EN prompts
DISHES = {
    "mic_dejun": [  # Breakfast
        {
            "name": "Ouă jumări cu roșii",
            "name_en": "Scrambled Eggs with Tomatoes",
            "type": "mic_dejun",
            "type_en": "breakfast",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 15,
            "calories": 250,
        },
        {
            "name": "Mămăligă cu brânză și smântână",
            "name_en": "Polenta with Cheese and Sour Cream",
            "type": "mic_dejun",
            "type_en": "breakfast",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": True,
            "prep_time_min": 25,
            "calories": 400,
        },
        {
            "name": "Clătite cu gem",
            "name_en": "Pancakes with Jam",
            "type": "mic_dejun",
            "type_en": "breakfast",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": True,
            "contains_lactose": True,
            "prep_time_min": 30,
            "calories": 350,
        },
        {
            "name": "Sandviș cu brânză și legume",
            "name_en": "Cheese and Vegetable Sandwich",
            "type": "mic_dejun",
            "type_en": "breakfast",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": True,
            "contains_lactose": True,
            "prep_time_min": 10,
            "calories": 300,
        },
        {
            "name": "Iaurt cu fructe și granola",
            "name_en": "Yogurt with Fruit and Granola",
            "type": "mic_dejun",
            "type_en": "breakfast",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": True,
            "contains_lactose": True,
            "prep_time_min": 5,
            "calories": 280,
        },
        {
            "name": "Smoothie verde cu spanac",
            "name_en": "Green Spinach Smoothie",
            "type": "mic_dejun",
            "type_en": "breakfast",
            "vegetarian": True,
            "vegan": True,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 10,
            "calories": 180,
        },
    ],
    "pranz": [  # Lunch
        {
            "name": "Ciorbă de legume",
            "name_en": "Vegetable Soup",
            "type": "pranz",
            "type_en": "lunch",
            "vegetarian": True,
            "vegan": True,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 45,
            "calories": 200,
        },
        {
            "name": "Sarmale în foi de viță",
            "name_en": "Stuffed Cabbage Rolls",
            "type": "pranz",
            "type_en": "lunch",
            "vegetarian": False,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 120,
            "calories": 450,
        },
        {
            "name": "Tocană de cartofi cu carne",
            "name_en": "Potato and Meat Stew",
            "type": "pranz",
            "type_en": "lunch",
            "vegetarian": False,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 60,
            "calories": 500,
        },
        {
            "name": "Salată grecească",
            "name_en": "Greek Salad",
            "type": "pranz",
            "type_en": "lunch",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": True,
            "prep_time_min": 15,
            "calories": 250,
        },
        {
            "name": "Paste cu sos de roșii",
            "name_en": "Pasta with Tomato Sauce",
            "type": "pranz",
            "type_en": "lunch",
            "vegetarian": True,
            "vegan": True,
            "contains_gluten": True,
            "contains_lactose": False,
            "prep_time_min": 25,
            "calories": 380,
        },
        {
            "name": "Piept de pui la grătar cu legume",
            "name_en": "Grilled Chicken Breast with Vegetables",
            "type": "pranz",
            "type_en": "lunch",
            "vegetarian": False,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 35,
            "calories": 350,
        },
        {
            "name": "Mâncare de fasole",
            "name_en": "Bean Stew",
            "type": "pranz",
            "type_en": "lunch",
            "vegetarian": True,
            "vegan": True,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 50,
            "calories": 320,
        },
    ],
    "cina": [  # Dinner
        {
            "name": "Supă cremă de ciuperci",
            "name_en": "Cream of Mushroom Soup",
            "type": "cina",
            "type_en": "dinner",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": True,
            "prep_time_min": 40,
            "calories": 220,
        },
        {
            "name": "Orez cu legume",
            "name_en": "Rice with Vegetables",
            "type": "cina",
            "type_en": "dinner",
            "vegetarian": True,
            "vegan": True,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 30,
            "calories": 300,
        },
        {
            "name": "Omletă cu ciuperci și brânză",
            "name_en": "Mushroom and Cheese Omelette",
            "type": "cina",
            "type_en": "dinner",
            "vegetarian": True,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": True,
            "prep_time_min": 20,
            "calories": 280,
        },
        {
            "name": "Pește la cuptor cu cartofi",
            "name_en": "Baked Fish with Potatoes",
            "type": "cina",
            "type_en": "dinner",
            "vegetarian": False,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 50,
            "calories": 400,
        },
        {
            "name": "Salată de ton",
            "name_en": "Tuna Salad",
            "type": "cina",
            "type_en": "dinner",
            "vegetarian": False,
            "vegan": False,
            "contains_gluten": False,
            "contains_lactose": False,
            "prep_time_min": 15,
            "calories": 280,
        },
        {
            "name": "Wrap vegetarian",
            "name_en": "Vegetarian Wrap",
            "type": "cina",
            "type_en": "dinner",
            "vegetarian": True,
            "vegan": True,
            "contains_gluten": True,
            "contains_lactose": False,
            "prep_time_min": 15,
            "calories": 320,
        },
    ],
}


class RecipeWorldGenerator:
    """Generator for Recipe World instances"""

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
        Generate a recipe world instance

        Args:
            world_id: Unique identifier
            seed: Random seed for reproducibility
            difficulty: easy|medium|hard
            **kwargs: Additional parameters
        """
        rng = random.Random(seed)

        # Number of days to plan
        num_days = 2 if difficulty == "easy" else rng.choice([2, 3])

        # Meals per day
        meals_per_day = ["mic_dejun", "pranz", "cina"]

        # Select dishes for each meal type
        available_dishes = {}
        entities = {}
        dishes_list = []
        dish_idx = 0

        for meal_type in meals_per_day:
            type_dishes = DISHES[meal_type]
            num_to_select = min(len(type_dishes), rng.randint(3, 5))
            selected = rng.sample(type_dishes, num_to_select)

            for dish in selected:
                dish_id = f"D{dish_idx + 1}"
                dish_idx += 1

                name_en = dish.get("name_en", dish["name"])
                # Build aliases: Romanian name, English name, both lowercase
                aliases = [
                    dish["name"].lower(),
                    name_en,
                    name_en.lower(),
                ]
                entities[dish_id] = Entity(
                    id=dish_id,
                    name=dish["name"],
                    aliases=aliases,
                    attributes=dish,
                )

                dishes_list.append({
                    "id": dish_id,
                    "name": dish["name"],
                    "name_en": dish.get("name_en", dish["name"]),
                    "type": dish["type"],
                    "type_en": dish.get("type_en", dish["type"]),
                    "vegetarian": dish["vegetarian"],
                    "vegan": dish["vegan"],
                    "contains_gluten": dish["contains_gluten"],
                    "contains_lactose": dish["contains_lactose"],
                    "prep_time_min": dish["prep_time_min"],
                    "calories": dish["calories"],
                })

                if meal_type not in available_dishes:
                    available_dishes[meal_type] = []
                available_dishes[meal_type].append(dish_id)

        # Generate constraints
        constraints = []

        # Dietary constraints (choose one or more based on difficulty)
        dietary_options = [
            ("vegetarian", "Toate preparatele trebuie să fie vegetariene.", "All dishes must be vegetarian.", "check_all_vegetarian"),
            ("no_gluten", "Persoana are intoleranță la gluten - evită preparatele cu gluten.", "The person has gluten intolerance - avoid dishes with gluten.", "check_no_gluten"),
            ("no_lactose", "Persoana are intoleranță la lactoză - evită preparatele cu lactoză.", "The person has lactose intolerance - avoid dishes with lactose.", "check_no_lactose"),
        ]

        if difficulty == "easy":
            num_dietary = 0
        elif difficulty == "medium":
            num_dietary = 1
        else:
            num_dietary = rng.randint(1, 2)

        selected_dietary = rng.sample(dietary_options, min(num_dietary, len(dietary_options)))

        for idx, (diet_type, desc_ro, desc_en, check_fn) in enumerate(selected_dietary):
            constraints.append(
                Constraint(
                    id=f"C_DIET_{idx}",
                    type=ConstraintType.INSTRUCTION,
                    description_ro=desc_ro,
                    description_en=desc_en,
                    check_fn=check_fn,
                    params={},
                )
            )

        # Calorie constraint (medium/hard)
        if difficulty in ("medium", "hard"):
            max_daily_calories = rng.choice([1500, 1800, 2000])
            constraints.append(
                Constraint(
                    id="C_MAX_CALORIES",
                    type=ConstraintType.INSTRUCTION,
                    description_ro=f"Totalul caloriilor pe zi nu trebuie să depășească {max_daily_calories} kcal.",
                    description_en=f"The total daily calories must not exceed {max_daily_calories} kcal.",
                    check_fn="check_max_daily_calories",
                    params={"max_calories": max_daily_calories},
                )
            )

        # Variety constraint (no same dish twice)
        if difficulty == "hard" or (difficulty == "medium" and rng.random() < 0.5):
            constraints.append(
                Constraint(
                    id="C_VARIETY",
                    type=ConstraintType.INSTRUCTION,
                    description_ro="Nu repeta același preparat în meniu.",
                    description_en="Do not repeat the same dish in the menu.",
                    check_fn="check_no_duplicates",
                    params={},
                )
            )

        # Generate goals
        goals = [
            Goal(
                id="G_ALL_MEALS_FILLED",
                type=GoalType.STRUCTURAL,
                description="Each meal slot must have a dish",
                check_fn="check_all_meals_filled",
                params={"num_days": num_days, "meals": meals_per_day},
            ),
            Goal(
                id="G_VALID_DISHES",
                type=GoalType.STRUCTURAL,
                description="All referenced dishes must exist in the menu",
                check_fn="check_valid_entity_ids",
                params={"valid_ids": list(entities.keys())},
            ),
        ]

        # English meal type mapping
        meals_per_day_en = ["breakfast", "lunch", "dinner"]

        # Build payload
        payload = {
            "num_days": num_days,
            "meals_per_day": meals_per_day,
            "meals_per_day_en": meals_per_day_en,
            "available_dishes": available_dishes,
            "dishes": dishes_list,
        }

        # Metadata
        meta = {
            "difficulty": difficulty,
            "num_dishes": len(dishes_list),
            "dietary_constraints": [d[0] for d in selected_dietary],
        }

        return World(
            world_id=world_id,
            world_type="recipe",
            spec_version=self.spec_version,
            seed=seed,
            payload=payload,
            constraints=constraints,
            goals=goals,
            canonical_entities=entities,
            meta=meta,
        )
