"""
English-specific metrics implementation

Extends base metrics with English language features.
"""

from typing import Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rombench.eval.base_metrics import BaseMetrics
from rombench.nlp_en import EnglishNLPToolkit, EnglishFaithfulness


class EnglishMetrics(BaseMetrics):
    """
    English-specific metrics computation.
    
    Uses:
    - EnglishNLPToolkit for G score (grammar, style, length)
    - EnglishFaithfulness for F score (simple morphology)
    """
    
    def __init__(
        self,
        use_languagetool: bool = False,
        severity_exponent: float = 3.0,
    ):
        super().__init__(severity_exponent=severity_exponent)
        self.use_languagetool = use_languagetool
        self.nlp_toolkit = EnglishNLPToolkit(use_grammar=use_languagetool)
        self.faithfulness_checker = EnglishFaithfulness()
    
    def compute_generation_quality(self, text: str, **kwargs) -> dict[str, Any]:
        """Compute G score using English NLP toolkit"""
        return self.nlp_toolkit.compute_g_score(text)
    
    def compute_faithfulness(
        self, 
        world: Any, 
        plan: dict, 
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        """Compute F score using English faithfulness checker"""
        return self.faithfulness_checker.compute_faithfulness(
            world, plan, explanation, 
            severity_exponent=self.severity_exponent
        )