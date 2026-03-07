"""
German-specific metrics implementation

Extends base metrics with German language features:
- Umlaut analysis (ä, ö, ü, ß)
- Noun capitalization checking
- German morphology for faithfulness
"""

from typing import Any

from rombench.eval.base_metrics import BaseMetrics
from rombench.nlp_de import GermanNLPToolkit, GermanFaithfulness


class GermanMetrics(BaseMetrics):
    """
    German-specific metrics computation.

    Uses:
    - GermanNLPToolkit for G score (umlauts, style, code-switching)
    - GermanFaithfulness for F score (morphology-aware, compound words)
    """

    def __init__(
        self,
        use_languagetool: bool = False,
        severity_exponent: float = 3.0,
    ):
        super().__init__(severity_exponent=severity_exponent)
        self.use_languagetool = use_languagetool
        self.nlp_toolkit = GermanNLPToolkit(use_grammar=use_languagetool)
        self.faithfulness_checker = GermanFaithfulness()

    def compute_generation_quality(self, text: str, **kwargs) -> dict[str, Any]:
        """Compute G score using German NLP toolkit."""
        return self.nlp_toolkit.compute_g_score(text)

    def compute_faithfulness(
        self,
        world: Any,
        plan: dict,
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        """Compute F score using German faithfulness checker."""
        return self.faithfulness_checker.compute_faithfulness(
            world, plan, explanation,
            severity_exponent=self.severity_exponent
        )