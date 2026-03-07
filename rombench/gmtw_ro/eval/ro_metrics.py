"""
Romanian-specific metrics implementation

Extends base metrics with Romanian language features:
- Diacritics analysis
- Code-switching detection (Romanian/English mixing)  
- Romanian morphology for faithfulness
"""

from typing import Any

from rombench.eval.base_metrics import BaseMetrics
from rombench.nlp_ro import RomanianNLPToolkit
from rombench.nlp_ro.faithfulness import RomanianFaithfulness


class RomanianMetrics(BaseMetrics):
    """
    Romanian-specific metrics computation.
    
    Uses:
    - RomanianNLPToolkit for G score (diacritics, code-switching, grammar)
    - RomanianFaithfulness for F score (morphology-aware)
    """
    
    def __init__(
        self,
        use_languagetool: bool = False,
        use_stanza: bool = False,
        severity_exponent: float = 3.0,
    ):
        super().__init__(severity_exponent=severity_exponent)
        self.use_languagetool = use_languagetool
        self.use_stanza = use_stanza
        self.nlp_toolkit = RomanianNLPToolkit(use_grammar=use_languagetool)
        self.faithfulness_checker = RomanianFaithfulness(use_stanza=use_stanza)
    
    def compute_generation_quality(self, text: str, **kwargs) -> dict[str, Any]:
        """Compute G score using Romanian NLP toolkit"""
        return self.nlp_toolkit.compute_g_score(text)
    
    def compute_faithfulness(
        self, 
        world: Any, 
        plan: dict, 
        explanation: str,
        **kwargs
    ) -> dict[str, Any]:
        """Compute F score using Romanian faithfulness checker"""
        return self.faithfulness_checker.compute_faithfulness(
            world, plan, explanation, 
            severity_exponent=self.severity_exponent
        )