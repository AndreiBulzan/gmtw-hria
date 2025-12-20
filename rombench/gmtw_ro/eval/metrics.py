"""
GMTW-Ro Metrics: U, R, G, F

Implements the four core metrics for evaluating model performance.

Scoring uses a severity exponent to penalize violations more harshly:
- Linear (exponent=1): 2/3 satisfied = 0.67
- Strict (exponent=2): 2/3 satisfied = 0.44
- Harsh (exponent=3): 2/3 satisfied = 0.30

Default is exponent=2 (strict mode).

Optional enhancements (require additional dependencies):
- use_languagetool: Adds G_grammar sub-score using LanguageTool
- use_stanza: Uses Stanza for more accurate Romanian lemmatization in F score
"""

from dataclasses import dataclass
from typing import Any, Optional
from ..worlds.base import World, ConstraintType, GoalType
from .constraints import check_constraint

# Default faithfulness implementation (fast, no external deps)
from .faithfulness import compute_faithfulness_deterministic

# Severity exponent for U and F scores
# Higher = more punishing for violations
# 1.0 = linear (lenient), 2.0 = squared (strict), 3.0 = cubed (harsh)
SEVERITY_EXPONENT = 3.0


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

def compute_U(
    world: World,
    plan: dict,
    format_ok: bool = True,
    repaired: bool = False,
) -> dict[str, Any]:
    """
    Compute Understanding score

    U measures how well the model understood the task:
    - 85% weight: Constraint satisfaction (the main task requirements)
    - 15% weight: Format compliance (correct JSON structure)

    This ensures format can't inflate the score, only penalize if wrong.

    Args:
        world: The world specification
        plan: The extracted JSON plan
        format_ok: Whether JSON was found
        repaired: Whether JSON needed repair

    Returns:
        Dictionary with U score and details
    """
    # === Part 1: Constraint satisfaction (85% of U) ===
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

            constraint_results.append({
                "id": constraint.id,
                "description": constraint.description_ro,
                "satisfied": is_satisfied,
            })
        except Exception as e:
            constraint_results.append({
                "id": constraint.id,
                "description": constraint.description_ro,
                "satisfied": False,
                "error": str(e),
            })

    # Also check structural goals (they're part of understanding too)
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
        U_constraints = U_constraints_raw ** SEVERITY_EXPONENT
    else:
        U_constraints_raw = 1.0
        U_constraints = 1.0

    # === Part 2: Format compliance (15% of U) ===
    format_checks = _check_format_compliance(world, plan, format_ok, repaired)
    format_satisfied = sum(1 for c in format_checks if c.get("satisfied", False))
    total_format = len(format_checks)

    if total_format > 0:
        U_format_raw = format_satisfied / total_format
        U_format = U_format_raw ** SEVERITY_EXPONENT
    else:
        U_format_raw = 1.0
        U_format = 1.0

    # === Combined U score ===
    # 85% constraints + 15% format
    # This ensures format can't inflate the score significantly
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


# ============================================================================
# Format Checks (integrated into U with lower weight)
# ============================================================================

def _check_format_compliance(world: World, plan: dict, format_ok: bool, repaired: bool) -> list[dict]:
    """
    Check format compliance - instant JSON structure checks.

    These are O(1) deterministic checks on the parsed JSON structure.
    They are integrated into U with lower weight (15% of U score).
    """
    checks = []
    world_type = world.world_type

    # 1. JSON was found and parseable
    checks.append({
        "id": "F_JSON_FOUND",
        "description": "Modelul a produs JSON valid",
        "satisfied": plan is not None,
    })

    if plan is None:
        # If no JSON, all other checks fail
        checks.append({"id": "F_JSON_CLEAN", "description": "JSON s-a parsat fără reparații", "satisfied": False})
        checks.append({"id": "F_KEYS_PRESENT", "description": "Toate cheile așteptate sunt prezente", "satisfied": False})
        checks.append({"id": "F_NO_EXTRA_KEYS", "description": "Nu există chei neașteptate", "satisfied": False})
        checks.append({"id": "F_VALUE_TYPES", "description": "Valorile au tipurile corecte", "satisfied": False})
        return checks

    # 2. JSON parsed cleanly (no repair needed)
    checks.append({
        "id": "F_JSON_CLEAN",
        "description": "JSON s-a parsat fără reparații",
        "satisfied": format_ok and not repaired,
    })

    # 3. All expected keys present (world-type specific)
    if world_type == "travel":
        num_days = world.payload.get("num_days", 2)
        expected_keys = {f"day{i}" for i in range(1, num_days + 1)}
        actual_keys = set(plan.keys())
        keys_present = expected_keys <= actual_keys
        checks.append({
            "id": "F_KEYS_PRESENT",
            "description": f"Toate cele {num_days} chei de zile sunt prezente",
            "satisfied": keys_present,
        })

    elif world_type == "recipe":
        num_days = world.payload.get("num_days", 2)
        meals = world.payload.get("meals_per_day", ["mic_dejun", "pranz", "cina"])
        expected_keys = {f"day{d}_{m}" for d in range(1, num_days + 1) for m in meals}
        actual_keys = set(plan.keys())
        keys_present = expected_keys <= actual_keys
        checks.append({
            "id": "F_KEYS_PRESENT",
            "description": f"Toate cele {len(expected_keys)} chei de mese sunt prezente",
            "satisfied": keys_present,
        })

    elif world_type == "schedule":
        days = world.payload.get("days_ro", [])
        slots = world.payload.get("slots_ro", [])
        expected_keys = {f"{d}_{s}" for d in days for s in slots}
        actual_keys = set(plan.keys())
        keys_present = expected_keys <= actual_keys
        checks.append({
            "id": "F_KEYS_PRESENT",
            "description": f"Toate cele {len(expected_keys)} chei de intervale sunt prezente",
            "satisfied": keys_present,
        })

    elif world_type == "fact":
        has_answer = "answer" in plan
        checks.append({
            "id": "F_KEYS_PRESENT",
            "description": "Cheia 'answer' este prezentă",
            "satisfied": has_answer,
        })

    # 4. No extra/unexpected keys
    if world_type == "travel":
        num_days = world.payload.get("num_days", 2)
        valid_keys = {f"day{i}" for i in range(1, num_days + 1)}
        actual_keys = set(plan.keys())
        no_extra = actual_keys <= valid_keys
        checks.append({
            "id": "F_NO_EXTRA_KEYS",
            "description": "Nu există chei neașteptate",
            "satisfied": no_extra,
        })

    elif world_type == "recipe":
        num_days = world.payload.get("num_days", 2)
        meals = world.payload.get("meals_per_day", ["mic_dejun", "pranz", "cina"])
        valid_keys = {f"day{d}_{m}" for d in range(1, num_days + 1) for m in meals}
        actual_keys = set(plan.keys())
        no_extra = actual_keys <= valid_keys
        checks.append({
            "id": "F_NO_EXTRA_KEYS",
            "description": "Nu există chei neașteptate",
            "satisfied": no_extra,
        })

    elif world_type == "schedule":
        days = world.payload.get("days_ro", [])
        slots = world.payload.get("slots_ro", [])
        valid_keys = {f"{d}_{s}" for d in days for s in slots}
        actual_keys = set(plan.keys())
        no_extra = actual_keys <= valid_keys
        checks.append({
            "id": "F_NO_EXTRA_KEYS",
            "description": "Nu există chei neașteptate",
            "satisfied": no_extra,
        })

    elif world_type == "fact":
        actual_keys = set(plan.keys())
        no_extra = actual_keys <= {"answer"}
        checks.append({
            "id": "F_NO_EXTRA_KEYS",
            "description": "Doar cheia 'answer' este prezentă",
            "satisfied": no_extra,
        })

    # 5. Value types are correct
    if world_type == "travel":
        all_lists = all(isinstance(v, list) for v in plan.values())
        checks.append({
            "id": "F_VALUE_TYPES",
            "description": "Toate valorile sunt liste",
            "satisfied": all_lists,
        })

    elif world_type in ("recipe", "schedule"):
        all_strings = all(
            v is None or isinstance(v, str)
            for v in plan.values()
        )
        checks.append({
            "id": "F_VALUE_TYPES",
            "description": "Toate valorile sunt șiruri sau null",
            "satisfied": all_strings,
        })

    elif world_type == "fact":
        answer = plan.get("answer")
        is_string = isinstance(answer, str)
        checks.append({
            "id": "F_VALUE_TYPES",
            "description": "Răspunsul este un șir",
            "satisfied": is_string,
        })

    return checks


# ============================================================================
# G - Generation Quality Score
# ============================================================================

def compute_G(
    text: str,
    nlp_tools: Any = None,
    use_languagetool: bool = False
) -> dict[str, Any]:
    """
    Compute Generation Quality score

    G measures linguistic hygiene: diacritics, code-switching, text length.

    Uses the Romanian NLP toolkit for deterministic, lexicon-based analysis.

    Components:
    - G_dia: Diacritic correctness (are Romanian diacritics used properly?)
    - G_cs: Code-switch score (absence of English contamination)
    - G_len: Length score (is the text adequately long?)
    - G_grammar: (optional) Grammar/spelling score via LanguageTool

    Args:
        text: The natural language explanation
        nlp_tools: Optional RomanianNLPToolkit instance (created if not provided)
        use_languagetool: If True, include LanguageTool grammar checking (requires language-tool-python)

    Returns:
        Dictionary with G score and component details
    """
    # Use provided toolkit or create one
    if nlp_tools is not None:
        toolkit = nlp_tools
    else:
        # Import here to avoid circular imports
        from ...nlp_ro import RomanianNLPToolkit
        toolkit = RomanianNLPToolkit(use_grammar=use_languagetool)

    # Get full analysis from toolkit
    return toolkit.compute_g_score(text)


# ============================================================================
# F - Faithfulness Score
# ============================================================================

def compute_F(
    world: World,
    plan: dict,
    explanation: str,
    use_stanza: bool = False,
) -> dict[str, Any]:
    """
    Compute Faithfulness score using deterministic substring matching

    F measures: "Did the explanation mention all entities from the plan?"

    Uses simple, deterministic substring matching (no fuzzy logic, no negation detection).

    Args:
        world: The world specification
        plan: The extracted JSON plan
        explanation: The natural language explanation
        use_stanza: If True, use Stanza for Romanian lemmatization (more accurate but slower)

    Returns:
        Dictionary with F score and details
    """
    if use_stanza:
        try:
            from .faithfulness_stanza import compute_faithfulness_deterministic as compute_f_stanza
            return compute_f_stanza(world, plan, explanation)
        except ImportError:
            import warnings
            warnings.warn(
                "use_stanza=True but stanza is not available. "
                "Falling back to default faithfulness implementation."
            )

    return compute_faithfulness_deterministic(world, plan, explanation)


# ============================================================================
# Combined Scorer
# ============================================================================

def compute_all_metrics(
    world: World,
    plan: dict,
    explanation: str,
    nlp_tools: Any = None,
    use_languagetool: bool = False,
    use_stanza: bool = False,
    format_ok: bool = True,
    repaired: bool = False,
) -> MetricScores:
    """
    Compute all three metrics (U, G, F)

    R has been integrated into U:
    - U = 85% constraint satisfaction + 15% format compliance

    Final score = 50% U + 25% G + 25% F

    Args:
        world: The world specification
        plan: The extracted JSON plan (or None if parsing failed)
        explanation: The natural language explanation
        nlp_tools: Optional NLP toolkit (if provided, use_languagetool is ignored)
        use_languagetool: If True, include LanguageTool grammar checking in G score
        use_stanza: If True, use Stanza for Romanian lemmatization in F score
        format_ok: Whether JSON was found (from parser)
        repaired: Whether JSON needed repair (from parser)

    Returns:
        MetricScores with all scores and details
    """
    # Use empty dict for plan if None (for U calculation)
    plan_for_eval = plan if plan is not None else {}

    # U now includes format checks with 15% weight
    U_details = compute_U(world, plan_for_eval, format_ok=format_ok, repaired=repaired)
    G_details = compute_G(explanation, nlp_tools, use_languagetool=use_languagetool)
    F_details = compute_F(world, plan_for_eval, explanation, use_stanza=use_stanza)

    # Format compliance check: JSON should be at the END, not the beginning
    # If explanation is empty/very short, model put JSON first → penalize U
    format_violation = len(explanation.strip()) < 50  # Less than 50 chars = no real explanation

    if format_violation:
        U_details["format_violation"] = True
        # Penalize by reducing U (format violation is severe)
        U_details["U"] = U_details["U"] * 0.5
    else:
        U_details["format_violation"] = False

    # R is deprecated - set to U_format for backwards compatibility
    R_details = {
        "R": U_details["U_format"],
        "note": "R is deprecated and integrated into U. This shows U_format for compatibility.",
        "format_checks": U_details.get("format_checks", []),
    }

    return MetricScores(
        U=U_details["U"],
        R=U_details["U_format"],  # For backwards compatibility
        G=G_details["G"],
        F=F_details["F"],
        U_details=U_details,
        R_details=R_details,
        G_details=G_details,
        F_details=F_details,
    )
