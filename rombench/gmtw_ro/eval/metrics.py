"""
GMTW-Ro Metrics: U, R, G, F

Implements the four core metrics for evaluating model performance.
"""

import math
from dataclasses import dataclass
from typing import Any
from ..worlds.base import World, ConstraintType, GoalType
from .constraints import check_constraint
from .faithfulness import compute_faithfulness_deterministic


@dataclass
class MetricScores:
    """Complete set of metric scores"""
    U: float  # Understanding
    R: float  # Reasoning
    G: float  # Generation Quality
    F: float  # Faithfulness

    # Detailed breakdowns
    U_details: dict[str, Any]
    R_details: dict[str, Any]
    G_details: dict[str, Any]
    F_details: dict[str, Any]


# ============================================================================
# U - Understanding Score
# ============================================================================

def compute_U(world: World, plan: dict) -> dict[str, Any]:
    """
    Compute Understanding score

    U measures how well the model understood the Romanian constraints.

    Args:
        world: The world specification
        plan: The extracted JSON plan

    Returns:
        Dictionary with U score and details
    """
    # Get instruction-level constraints
    instruction_constraints = [
        c for c in world.constraints if c.type == ConstraintType.INSTRUCTION
    ]

    if not instruction_constraints:
        return {
            "U": 1.0,
            "satisfied": 0,
            "total": 0,
            "constraints": [],
        }

    # Check each constraint
    satisfied_count = 0
    constraint_results = []

    for constraint in instruction_constraints:
        try:
            is_satisfied = check_constraint(
                world, plan, constraint.check_fn, constraint.params
            )
            if is_satisfied:
                satisfied_count += 1

            constraint_results.append({
                "id": constraint.id,
                "description": constraint.description_ro,
                "satisfied": is_satisfied,
            })
        except Exception as e:
            # If check fails, count as unsatisfied
            constraint_results.append({
                "id": constraint.id,
                "description": constraint.description_ro,
                "satisfied": False,
                "error": str(e),
            })

    U = satisfied_count / len(instruction_constraints)

    return {
        "U": U,
        "satisfied": satisfied_count,
        "total": len(instruction_constraints),
        "constraints": constraint_results,
    }


# ============================================================================
# R - Reasoning Score
# ============================================================================

def compute_R(world: World, plan: dict) -> dict[str, Any]:
    """
    Compute Reasoning score

    R measures how well the model's plan satisfies structural/logical goals.

    Args:
        world: The world specification
        plan: The extracted JSON plan

    Returns:
        Dictionary with R score and details
    """
    # Get structural goals
    structural_goals = [
        g for g in world.goals if g.type == GoalType.STRUCTURAL
    ]

    if not structural_goals:
        return {
            "R": 1.0,
            "satisfied": 0,
            "total": 0,
            "goals": [],
        }

    # Check each goal
    satisfied_count = 0
    goal_results = []

    for goal in structural_goals:
        try:
            is_satisfied = check_constraint(
                world, plan, goal.check_fn, goal.params
            )
            if is_satisfied:
                satisfied_count += 1

            goal_results.append({
                "id": goal.id,
                "description": goal.description,
                "satisfied": is_satisfied,
            })
        except Exception as e:
            goal_results.append({
                "id": goal.id,
                "description": goal.description,
                "satisfied": False,
                "error": str(e),
            })

    R = satisfied_count / len(structural_goals)

    return {
        "R": R,
        "satisfied": satisfied_count,
        "total": len(structural_goals),
        "goals": goal_results,
    }


# ============================================================================
# G - Generation Quality Score
# ============================================================================

def compute_G(text: str, nlp_tools: Any = None) -> dict[str, Any]:
    """
    Compute Generation Quality score

    G measures linguistic hygiene: grammar, diacritics, code-switching.

    NOTE: This is a simplified implementation. For full implementation,
    integrate with nlp_ro toolkit.

    Args:
        text: The natural language explanation
        nlp_tools: Optional NLP toolkit (Stanza, etc.)

    Returns:
        Dictionary with G score and details
    """
    # Simplified implementation (to be replaced with full NLP toolkit)

    # Component 1: Grammar/spelling (placeholder)
    # In full implementation: use LanguageTool or grammar checker
    # For now, simple heuristic: check for very short or degenerate text
    words = text.split()
    n_tokens = len(words)

    if n_tokens < 10:
        G_gram = 0.5  # Too short
    else:
        G_gram = 1.0  # Placeholder

    # Component 2: Diacritic coverage
    # Check presence of Romanian diacritics
    diacritic_chars = set('ăâîșț')
    has_diacritics = any(c in text for c in diacritic_chars)

    # Simple heuristic: if text is reasonably long and has no diacritics, penalize
    if n_tokens > 20 and not has_diacritics:
        G_dia = 0.6  # Likely stripped diacritics
    else:
        G_dia = 1.0

    # Component 3: Code-switching
    # Simple heuristic: check for common English words
    common_english = ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'will', 'would']
    text_lower = text.lower()
    english_word_count = sum(1 for word in common_english if f' {word} ' in f' {text_lower} ')

    cs_rate = english_word_count / max(1, n_tokens)
    G_cs = math.exp(-10 * cs_rate)  # Exponential penalty

    # Combined G score
    G = G_gram * G_dia * G_cs

    return {
        "G": G,
        "G_gram": G_gram,
        "G_dia": G_dia,
        "G_cs": G_cs,
        "n_tokens": n_tokens,
        "cs_rate": cs_rate,
        "note": "Simplified implementation - integrate nlp_ro for full scoring",
    }


# ============================================================================
# F - Faithfulness Score
# ============================================================================

def compute_F(
    world: World,
    plan: dict,
    explanation: str,
) -> dict[str, Any]:
    """
    Compute Faithfulness score using deterministic substring matching

    F measures: "Did the explanation mention all entities from the plan?"

    Uses simple, deterministic substring matching (no fuzzy logic, no negation detection).

    Args:
        world: The world specification
        plan: The extracted JSON plan
        explanation: The natural language explanation

    Returns:
        Dictionary with F score and details
    """
    return compute_faithfulness_deterministic(world, plan, explanation)


# ============================================================================
# Combined Scorer
# ============================================================================

def compute_all_metrics(
    world: World,
    plan: dict,
    explanation: str,
    nlp_tools: Any = None
) -> MetricScores:
    """
    Compute all four metrics

    Args:
        world: The world specification
        plan: The extracted JSON plan
        explanation: The natural language explanation
        nlp_tools: Optional NLP toolkit

    Returns:
        MetricScores with all four scores and details
    """
    U_details = compute_U(world, plan)
    R_details = compute_R(world, plan)
    G_details = compute_G(explanation, nlp_tools)
    F_details = compute_F(world, plan, explanation)

    return MetricScores(
        U=U_details["U"],
        R=R_details["R"],
        G=G_details["G"],
        F=F_details["F"],
        U_details=U_details,
        R_details=R_details,
        G_details=G_details,
        F_details=F_details,
    )
