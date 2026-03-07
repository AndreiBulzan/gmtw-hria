"""
Base metrics computation (language-agnostic)

Defines the structure for U, G, F metrics that can be
specialized per language.

Architecture:
    - U (Understanding) is language-agnostic (constraint/format checking)
    - G (Generation Quality) is language-specific (subclasses implement)
    - F (Faithfulness) is language-specific (subclasses implement)
    - R (Reasoning) is deprecated, kept for backwards compatibility
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


# Severity exponent for scoring (configurable)
SEVERITY_EXPONENT = 3.0


@dataclass
class MetricScores:
    """Complete set of metric scores"""
    U: float  # Understanding (constraint satisfaction + format)
    R: float  # Reasoning (deprecated, kept for compatibility)
    G: float  # Generation Quality (linguistics)
    F: float  # Faithfulness (explanation mentions plan entities)

    # Detailed breakdowns
    U_details: dict[str, Any]
    R_details: dict[str, Any]
    G_details: dict[str, Any]
    F_details: dict[str, Any]


class BaseMetrics(ABC):
    """
    Base class for language-specific metrics computation.

    Subclasses must implement:
    - compute_generation_quality(): G score with language-specific analysis
    - compute_faithfulness(): F score with language-specific morphology

    U (understanding) is language-agnostic and shared across all languages.

    To add a new language:
        1. Create a subclass of BaseMetrics
        2. Implement compute_generation_quality() and compute_faithfulness()
        3. Create a corresponding evaluator and register it via rombench.registry
    """

    def __init__(self, severity_exponent: float = SEVERITY_EXPONENT):
        self.severity_exponent = severity_exponent

    @abstractmethod
    def compute_generation_quality(
        self, 
        text: str,
        **kwargs
    ) -> dict[str, Any]:
        """
        Compute G score (generation quality).
        
        Should measure:
        - Grammar/spelling quality
        - Proper use of language-specific features (e.g., diacritics)
        - Text style and coherence
        - Adequate length
        
        Returns dict with 'G' score and details
        """
        pass

    @abstractmethod
    def compute_faithfulness(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        """
        Compute F score (faithfulness).
        
        Should measure: Did the explanation mention all entities from the plan?
        
        Uses language-specific:
        - Tokenization
        - Morphology/inflection
        - Entity matching
        
        Returns dict with 'F' score and details
        """
        pass

    def compute_understanding(
        self,
        world: Any,
        plan: dict,
        format_ok: bool = True,
        repaired: bool = False,
    ) -> dict[str, Any]:
        """
        Compute U score (understanding).
        
        Language-agnostic - checks:
        - Constraint satisfaction (85% weight)
        - Format compliance (15% weight)
        
        This implementation works for all languages.
        """
        from ..gmtw_ro.worlds.base import ConstraintType, GoalType
        from ..gmtw_ro.eval.constraints import check_constraint
        instruction_constraints = [
            c for c in world.constraints if c.type == ConstraintType.INSTRUCTION
        ]

        constraint_results = []
        constraints_satisfied = 0

        for constraint in instruction_constraints:
            try:
                is_satisfied = check_constraint(
                    world, plan, constraint.check_fn, constraint.params
                )
                if is_satisfied:
                    constraints_satisfied += 1

                description = constraint.description_en or constraint.description_ro
                constraint_results.append({
                    "id": constraint.id,
                    "description": description,
                    "satisfied": is_satisfied,
                })
            except Exception as e:
                description = constraint.description_en or constraint.description_ro
                constraint_results.append({
                    "id": constraint.id,
                    "description": description,
                    "satisfied": False,
                    "error": str(e),
                })

        # Also check structural goals
        structural_goals = [
            g for g in world.goals if g.type == GoalType.STRUCTURAL
        ]

        for goal in structural_goals:
            try:
                is_satisfied = check_constraint(
                    world, plan, goal.check_fn, goal.params
                )
                if is_satisfied:
                    constraints_satisfied += 1

                constraint_results.append({
                    "id": goal.id,
                    "description": goal.description,
                    "satisfied": is_satisfied,
                })
            except Exception as e:
                constraint_results.append({
                    "id": goal.id,
                    "description": goal.description,
                    "satisfied": False,
                    "error": str(e),
                })

        total_constraints = len(instruction_constraints) + len(structural_goals)

        if total_constraints > 0:
            U_constraints_raw = constraints_satisfied / total_constraints
            U_constraints = U_constraints_raw ** self.severity_exponent
        else:
            U_constraints_raw = 1.0
            U_constraints = 1.0

        # === Part 2: Format compliance (15%) ===
        format_checks = self._check_format_compliance(world, plan, format_ok, repaired)
        format_satisfied = sum(1 for c in format_checks if c.get("satisfied", False))
        total_format = len(format_checks)

        if total_format > 0:
            U_format_raw = format_satisfied / total_format
            U_format = U_format_raw ** self.severity_exponent
        else:
            U_format_raw = 1.0
            U_format = 1.0

        # === Combined U ===
        U = 0.85 * U_constraints + 0.15 * U_format

        return {
            "U": U,
            "U_constraints": U_constraints,
            "U_constraints_linear": U_constraints_raw,
            "U_format": U_format,
            "U_format_linear": U_format_raw,
            "constraints_satisfied": constraints_satisfied,
            "constraints_total": total_constraints,
            "format_satisfied": format_satisfied,
            "format_total": total_format,
            "constraints": constraint_results,
            "format_checks": format_checks,
        }

    def _check_format_compliance(self, world: Any, plan: dict, format_ok: bool, repaired: bool) -> list[dict]:
        """Check format compliance (language-agnostic)"""
        checks = []
        world_type = world.world_type

        # 1. JSON was found
        checks.append({
            "id": "F_JSON_FOUND",
            "description": "Valid JSON produced",
            "satisfied": plan is not None,
        })

        if plan is None:
            checks.extend([
                {"id": "F_JSON_CLEAN", "description": "JSON parsed cleanly", "satisfied": False},
                {"id": "F_KEYS_PRESENT", "description": "Expected keys present", "satisfied": False},
                {"id": "F_NO_EXTRA_KEYS", "description": "No unexpected keys", "satisfied": False},
                {"id": "F_VALUE_TYPES", "description": "Correct value types", "satisfied": False},
            ])
            return checks

        # 2. JSON parsed cleanly
        checks.append({
            "id": "F_JSON_CLEAN",
            "description": "JSON parsed without repair",
            "satisfied": format_ok and not repaired,
        })

        # 3-5. World-type specific checks
        if world_type == "travel":
            num_days = world.payload.get("num_days", 2)
            expected_keys = {f"day{i}" for i in range(1, num_days + 1)}
            actual_keys = set(plan.keys())
            
            checks.append({
                "id": "F_KEYS_PRESENT",
                "description": f"All {num_days} day keys present",
                "satisfied": expected_keys <= actual_keys,
            })
            checks.append({
                "id": "F_NO_EXTRA_KEYS",
                "description": "No unexpected keys",
                "satisfied": actual_keys <= expected_keys,
            })
            checks.append({
                "id": "F_VALUE_TYPES",
                "description": "All values are lists",
                "satisfied": all(isinstance(v, list) for v in plan.values()),
            })

        elif world_type == "recipe":
            num_days = world.payload.get("num_days", 2)
            meals = world.payload.get("meals_per_day", ["mic_dejun", "pranz", "cina"])
            expected_keys = {f"day{d}_{m}" for d in range(1, num_days + 1) for m in meals}
            actual_keys = set(plan.keys())
            
            checks.append({
                "id": "F_KEYS_PRESENT",
                "description": f"All {len(expected_keys)} meal keys present",
                "satisfied": expected_keys <= actual_keys,
            })
            checks.append({
                "id": "F_NO_EXTRA_KEYS",
                "description": "No unexpected keys",
                "satisfied": actual_keys <= expected_keys,
            })
            checks.append({
                "id": "F_VALUE_TYPES",
                "description": "All values are strings or null",
                "satisfied": all(v is None or isinstance(v, str) for v in plan.values()),
            })

        elif world_type == "schedule":
            days, slots = self._get_schedule_keys(world)
            expected_keys = {f"{d}_{s}" for d in days for s in slots}
            actual_keys = set(plan.keys())
            
            checks.append({
                "id": "F_KEYS_PRESENT",
                "description": f"All {len(expected_keys)} time slot keys present",
                "satisfied": expected_keys <= actual_keys,
            })
            checks.append({
                "id": "F_NO_EXTRA_KEYS",
                "description": "No unexpected keys",
                "satisfied": actual_keys <= expected_keys,
            })
            checks.append({
                "id": "F_VALUE_TYPES",
                "description": "All values are strings or null",
                "satisfied": all(v is None or isinstance(v, str) for v in plan.values()),
            })

        elif world_type == "fact":
            checks.append({
                "id": "F_KEYS_PRESENT",
                "description": "'answer' key present",
                "satisfied": "answer" in plan,
            })
            checks.append({
                "id": "F_NO_EXTRA_KEYS",
                "description": "Only 'answer' key present",
                "satisfied": set(plan.keys()) <= {"answer"},
            })
            checks.append({
                "id": "F_VALUE_TYPES",
                "description": "Answer is a string",
                "satisfied": isinstance(plan.get("answer"), str),
            })

        return checks

    def _get_schedule_keys(self, world: Any) -> tuple[list[str], list[str]]:
        """
        Extract schedule day and slot keys from world payload.

        Searches for language-specific keys (days_ro, days_en, days_de, etc.)
        and falls back to the generic 'days' and 'slots' keys.
        """
        payload = world.payload
        days = []
        slots = []

        for key in ["days_ro", "days_en", "days_de", "days"]:
            if key in payload and payload[key]:
                days = payload[key]
                break

        for key in ["slots_ro", "slots_en", "slots_de", "slots"]:
            if key in payload and payload[key]:
                slots = payload[key]
                break

        return days, slots

    def compute_all_metrics(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        format_ok: bool = True,
        repaired: bool = False,
        **kwargs
    ) -> MetricScores:
        """
        Compute all metrics (U, G, F).
        
        R is deprecated and set to U_format for compatibility.
        """
        plan_for_eval = plan if plan is not None else {}

        # Compute metrics
        U_details = self.compute_understanding(world, plan_for_eval, format_ok, repaired)
        G_details = self.compute_generation_quality(explanation, **kwargs)
        F_details = self.compute_faithfulness(world, plan_for_eval, explanation, **kwargs)

        # Check for format violation (explanation too short)
        format_violation = len(explanation.strip()) < 50
        if format_violation:
            U_details["format_violation"] = True
            U_details["U"] = U_details["U"] * 0.5
        else:
            U_details["format_violation"] = False

        # R is deprecated
        R_details = {
            "R": U_details["U_format"],
            "note": "R is deprecated and integrated into U",
            "format_checks": U_details.get("format_checks", []),
        }

        return MetricScores(
            U=U_details["U"],
            R=U_details["U_format"],
            G=G_details["G"],
            F=F_details["F"],
            U_details=U_details,
            R_details=R_details,
            G_details=G_details,
            F_details=F_details,
        )