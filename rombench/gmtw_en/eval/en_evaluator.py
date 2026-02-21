"""
English evaluator for GMTW-En

Extends BaseEvaluator with English-specific components.
"""

from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rombench.eval.base_evaluator import BaseEvaluator
from rombench.eval.base_parser import BaseParser
from .en_metrics import EnglishMetrics


class EnglishEvaluator(BaseEvaluator):
    """
    Evaluator for English language outputs.
    
    Uses:
    - EnglishMetrics for language-specific scoring
    - BaseParser for JSON extraction (language-agnostic)
    """
    
    @classmethod
    def create(
        cls,
        use_languagetool: bool = False,
        severity_exponent: float = 3.0,
        **kwargs
    ) -> 'EnglishEvaluator':
        """
        Factory method to create English evaluator.
        
        Args:
            use_languagetool: Enable LanguageTool grammar checking
            severity_exponent: Penalty severity for violations
        """
        metrics = EnglishMetrics(
            use_languagetool=use_languagetool,
            severity_exponent=severity_exponent,
        )
        
        return cls(
            metrics=metrics,
            parser=BaseParser(),
            language="en",
            **kwargs
        )


# Convenience function
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
    evaluator = EnglishEvaluator.create(
        use_languagetool=use_languagetool,
        **kwargs
    )
    return evaluator.evaluate_output(instance, output)