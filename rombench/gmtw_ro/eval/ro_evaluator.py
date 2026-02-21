"""
Romanian evaluator for GMTW-Ro

Extends BaseEvaluator with Romanian-specific components.
"""

from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rombench.eval.base_evaluator import BaseEvaluator
from rombench.eval.base_parser import BaseParser
from .ro_metrics import RomanianMetrics


class RomanianEvaluator(BaseEvaluator):
    """
    Evaluator for Romanian language outputs.
    
    Uses:
    - RomanianMetrics for language-specific scoring
    - BaseParser for JSON extraction (language-agnostic)
    """
    
    @classmethod
    def create(
        cls,
        use_languagetool: bool = False,
        use_stanza: bool = False,
        severity_exponent: float = 3.0,
        **kwargs
    ) -> 'RomanianEvaluator':
        """
        Factory method to create Romanian evaluator.
        
        Args:
            use_languagetool: Enable LanguageTool grammar checking
            use_stanza: Use Stanza for lemmatization in faithfulness
            severity_exponent: Penalty severity for violations
        """
        metrics = RomanianMetrics(
            use_languagetool=use_languagetool,
            use_stanza=use_stanza,
            severity_exponent=severity_exponent,
        )
        
        return cls(
            metrics=metrics,
            parser=BaseParser(),
            language="ro",
            **kwargs
        )


# Convenience function for backwards compatibility
def evaluate_instance(instance, output, use_languagetool=False, use_stanza=False, **kwargs):
    """
    Evaluate a single instance (backwards compatible interface).
    
    Args:
        instance: Instance to evaluate
        output: Model output string
        use_languagetool: Enable LanguageTool
        use_stanza: Enable Stanza
        
    Returns:
        EvaluationResult
    """
    evaluator = RomanianEvaluator.create(
        use_languagetool=use_languagetool,
        use_stanza=use_stanza,
        **kwargs
    )
    return evaluator.evaluate_output(instance, output)