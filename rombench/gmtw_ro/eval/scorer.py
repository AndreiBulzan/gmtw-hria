"""
Main evaluator for GMTW-Ro

Orchestrates the complete evaluation pipeline.
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional
from ..worlds.base import World, Instance
from .parser import parse_dual_channel_output, ParseResult
from .metrics import compute_all_metrics, MetricScores


@dataclass
class EvaluationResult:
    """Complete evaluation result for one instance"""
    instance_id: str

    # Core metrics
    U: float
    R: float
    G: float
    F: float

    # Parsing info
    format_ok: bool
    repaired: bool
    parse_error: Optional[str]

    # Detailed breakdowns
    U_details: dict[str, Any]
    R_details: dict[str, Any]
    G_details: dict[str, Any]
    F_details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class GMTWEvaluator:
    """Main evaluator for GMTW-Ro instances"""

    def __init__(self, nlp_tools: Any = None):
        """
        Initialize evaluator

        Args:
            nlp_tools: Optional NLP toolkit (Stanza, etc.)
        """
        self.nlp_tools = nlp_tools

    def evaluate_output(
        self,
        instance: Instance,
        output: str
    ) -> EvaluationResult:
        """
        Evaluate a model output for a given instance

        Args:
            instance: The evaluation instance
            output: Raw model output string

        Returns:
            EvaluationResult with all metrics
        """
        world = instance.world

        # Parse the output
        parse_result = parse_dual_channel_output(output)

        # If parsing failed completely, return zero scores
        if parse_result.plan is None:
            return EvaluationResult(
                instance_id=instance.instance_id,
                U=0.0,
                R=0.0,
                G=0.0,
                F=0.0,
                format_ok=False,
                repaired=False,
                parse_error=parse_result.error_message,
                U_details={"U": 0.0, "satisfied": 0, "total": 0, "constraints": []},
                R_details={"R": 0.0, "satisfied": 0, "total": 0, "goals": []},
                G_details={"G": 0.0, "note": "No valid output"},
                F_details={"F": 0.0, "missing": [], "extra": []},
            )

        # Compute all metrics
        try:
            metrics = compute_all_metrics(
                world=world,
                plan=parse_result.plan,
                explanation=parse_result.explanation,
                nlp_tools=self.nlp_tools
            )

            return EvaluationResult(
                instance_id=instance.instance_id,
                U=metrics.U,
                R=metrics.R,
                G=metrics.G,
                F=metrics.F,
                format_ok=parse_result.format_ok,
                repaired=parse_result.repaired,
                parse_error=None,
                U_details=metrics.U_details,
                R_details=metrics.R_details,
                G_details=metrics.G_details,
                F_details=metrics.F_details,
            )

        except Exception as e:
            # If evaluation fails, return partial results
            return EvaluationResult(
                instance_id=instance.instance_id,
                U=0.0,
                R=0.0,
                G=0.0,
                F=0.0,
                format_ok=parse_result.format_ok,
                repaired=parse_result.repaired,
                parse_error=f"Evaluation error: {str(e)}",
                U_details={"U": 0.0, "error": str(e)},
                R_details={"R": 0.0, "error": str(e)},
                G_details={"G": 0.0, "error": str(e)},
                F_details={"F": 0.0, "error": str(e)},
            )


def evaluate_instance(
    instance: Instance,
    output: str,
    nlp_tools: Any = None
) -> EvaluationResult:
    """
    Convenience function to evaluate a single instance

    Args:
        instance: The evaluation instance
        output: Raw model output
        nlp_tools: Optional NLP toolkit

    Returns:
        EvaluationResult
    """
    evaluator = GMTWEvaluator(nlp_tools=nlp_tools)
    return evaluator.evaluate_output(instance, output)
