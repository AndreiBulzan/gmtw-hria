"""
GMTW: Grounded Multilingual Task Worlds for LLM Evaluation

Supports: Romanian (ro), English (en), German (de)

Usage:
    from rombench.registry import create_evaluator, get_supported_languages

    evaluator = create_evaluator("de", use_languagetool=False)
    result = evaluator.evaluate_output(instance, output)

    from rombench.models import World, Instance
"""

__version__ = "0.2.0"

