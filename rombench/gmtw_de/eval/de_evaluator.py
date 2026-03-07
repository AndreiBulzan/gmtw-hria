"""
German evaluator for GMTW-De

Extends BaseEvaluator with German-specific components.
"""

from rombench.eval.base_evaluator import BaseEvaluator
from rombench.eval.base_parser import BaseParser
from .de_metrics import GermanMetrics


class GermanEvaluator(BaseEvaluator):
    """
    Evaluator for German language outputs.

    Uses:
    - GermanMetrics for language-specific scoring
    - BaseParser for JSON extraction (language-agnostic)
    """

    @classmethod
    def create(
        cls,
        use_languagetool: bool = False,
        severity_exponent: float = 3.0,
        **kwargs
    ) -> 'GermanEvaluator':
        """
        Factory method to create German evaluator.

        Args:
            use_languagetool: Enable LanguageTool grammar checking
            severity_exponent: Penalty severity for violations
        """
        metrics = GermanMetrics(
            use_languagetool=use_languagetool,
            severity_exponent=severity_exponent,
        )

        return cls(
            metrics=metrics,
            parser=BaseParser(),
            language="de",
            **kwargs
        )


def evaluate_instance(instance, output, use_languagetool=False, **kwargs):
    """
    Evaluate a single instance (convenience function).

    Args:
        instance: Instance to evaluate
        output: Model output string
        use_languagetool: Enable LanguageTool

    Returns:
        EvaluationResult
    """
    evaluator = GermanEvaluator.create(
        use_languagetool=use_languagetool,
        **kwargs
    )
    return evaluator.evaluate_output(instance, output)