"""
Base evaluation framework for GMTW benchmarks

Language-agnostic evaluation components that can be extended
for specific languages (Romanian, English, etc.)
"""

from .base_evaluator import BaseEvaluator, EvaluationResult
from .base_metrics import BaseMetrics, MetricScores
from .base_parser import BaseParser, ParseResult

_all_ = [
    'BaseEvaluator',
    'EvaluationResult',
    'BaseMetrics',
    'MetricScores',
    'BaseParser',
    'ParseResult',
]