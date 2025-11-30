"""
GMTW-Ro Metrics: U, R, G, F

Implements the four core metrics for evaluating model performance.

Scoring uses a severity exponent to penalize violations more harshly:
- Linear (exponent=1): 2/3 satisfied = 0.67
- Strict (exponent=2): 2/3 satisfied = 0.44
- Harsh (exponent=3): 2/3 satisfied = 0.30

Default is exponent=2 (strict mode).
"""

from dataclasses import dataclass
from typing import Any
from ..worlds.base import World, ConstraintType, GoalType
from .constraints import check_constraint
from .faithfulness import compute_faithfulness_deterministic

# Severity exponent for U and R scores
# Higher = more punishing for violations
# 1.0 = linear (lenient), 2.0 = squared (strict), 3.0 = cubed (harsh)
SEVERITY_EXPONENT = 2.0


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

    # Apply severity exponent: (satisfied/total)^exponent
    # This penalizes violations more harshly than linear scoring
    raw_ratio = satisfied_count / len(instruction_constraints)
    U = raw_ratio ** SEVERITY_EXPONENT

    return {
        "U": U,
        "U_linear": raw_ratio,  # For reference
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

    # Apply severity exponent: (satisfied/total)^exponent
    # This penalizes violations more harshly than linear scoring
    raw_ratio = satisfied_count / len(structural_goals)
    R = raw_ratio ** SEVERITY_EXPONENT

    return {
        "R": R,
        "R_linear": raw_ratio,  # For reference
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

    G measures linguistic hygiene: diacritics, code-switching, text length.

    Uses the Romanian NLP toolkit for deterministic, lexicon-based analysis.

    Components:
    - G_dia: Diacritic correctness (are Romanian diacritics used properly?)
    - G_cs: Code-switch score (absence of English contamination)
    - G_len: Length score (is the text adequately long?)

    Args:
        text: The natural language explanation
        nlp_tools: Optional RomanianNLPToolkit instance (created if not provided)

    Returns:
        Dictionary with G score and component details
    """
    # Use provided toolkit or create one
    if nlp_tools is not None:
        toolkit = nlp_tools
    else:
        # Import here to avoid circular imports
        from ...nlp_ro import RomanianNLPToolkit
        toolkit = RomanianNLPToolkit()

    # Get full analysis from toolkit
    return toolkit.compute_g_score(text)


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

    # Format compliance check: JSON should be at the END, not the beginning
    # If explanation is empty/very short, model put JSON first → penalize U
    format_violation = len(explanation.strip()) < 50  # Less than 50 chars = no real explanation

    if format_violation:
        # Add format violation as a failed constraint in U
        U_details["format_violation"] = True
        U_details["constraints"].append({
            "id": "C_FORMAT_JSON_AT_END",
            "description": "JSON trebuie să fie la finalul răspunsului, nu la început.",
            "satisfied": False,
        })
        # Recalculate U with the format constraint (using severity exponent)
        U_details["total"] += 1
        raw_ratio = U_details["satisfied"] / U_details["total"]
        U_details["U_linear"] = raw_ratio
        U_details["U"] = raw_ratio ** SEVERITY_EXPONENT
    else:
        U_details["format_violation"] = False

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
