"""
Base evaluator for GMTW benchmarks (language-agnostic)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Optional

from .base_parser import BaseParser, ParseResult
from .base_metrics import BaseMetrics, MetricScores


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

    # Language info
    language: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BaseEvaluator(ABC):
    """
    Base evaluator for GMTW instances.
    
    Subclasses should:
    - Provide a language-specific metrics implementation
    - Optionally customize the parser
    - Set the language identifier
    """

    def __init__(
        self,
        metrics: BaseMetrics,
        parser: Optional[BaseParser] = None,
        language: str = "unknown",
        **kwargs
    ):
        """
        Initialize evaluator

        Args:
            metrics: Language-specific metrics implementation
            parser: Optional parser (uses BaseParser if not provided)
            language: Language identifier (e.g., 'ro', 'en')
            **kwargs: Additional arguments for metrics
        """
        self.metrics = metrics
        self.parser = parser or BaseParser()
        self.language = language
        self.kwargs = kwargs

    def evaluate_output(
        self,
        instance: Any,
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
        parse_result = self.parser.parse(output)

        # Compute all metrics
        try:
            metric_scores = self.metrics.compute_all_metrics(
                world=world,
                plan=parse_result.plan,
                explanation=parse_result.explanation,
                format_ok=parse_result.format_ok,
                repaired=parse_result.repaired,
                **self.kwargs
            )

            return EvaluationResult(
                instance_id=instance.instance_id,
                U=metric_scores.U,
                R=metric_scores.R,
                G=metric_scores.G,
                F=metric_scores.F,
                format_ok=parse_result.format_ok,
                repaired=parse_result.repaired,
                parse_error=parse_result.error_message if parse_result.error_message else None,
                U_details=metric_scores.U_details,
                R_details=metric_scores.R_details,
                G_details=metric_scores.G_details,
                F_details=metric_scores.F_details,
                language=self.language,
            )

        except Exception as e:
            # If evaluation fails, return zero scores
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
                language=self.language,
            )

    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> 'BaseEvaluator':
        """
        Factory method to create an evaluator with appropriate
        language-specific components.
        
        Example:
            evaluator = RomanianEvaluator.create(use_languagetool=True)
        """
        pass