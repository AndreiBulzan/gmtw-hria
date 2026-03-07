"""
Language registry for GMTW evaluation framework

Provides a clean extension point for adding new languages.
Register evaluator classes and create evaluators by language code.
"""

from typing import Type, Optional


# Registry of language code -> evaluator class
_EVALUATOR_REGISTRY: dict[str, Type] = {}

# Registry of language code -> human-readable name
_LANGUAGE_NAMES: dict[str, str] = {}


def register_language(
    code: str,
    evaluator_class: Type,
    name: str = None,
):
    """
    Register a language evaluator.

    Args:
        code: Language code (e.g., 'ro', 'en', 'de')
        evaluator_class: Evaluator class with a .create() factory method
        name: Human-readable language name
    """
    _EVALUATOR_REGISTRY[code] = evaluator_class
    _LANGUAGE_NAMES[code] = name or code.upper()


def create_evaluator(language: str, **kwargs):
    """
    Create an evaluator for the given language.

    Args:
        language: Language code (e.g., 'ro', 'en', 'de')
        **kwargs: Passed to the evaluator's create() method

    Returns:
        Language-specific evaluator instance

    Raises:
        ValueError: If language is not registered
    """
    if language not in _EVALUATOR_REGISTRY:
        supported = ", ".join(sorted(_EVALUATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Supported: {supported}"
        )
    return _EVALUATOR_REGISTRY[language].create(**kwargs)


def get_supported_languages() -> list[str]:
    """Get list of supported language codes."""
    return sorted(_EVALUATOR_REGISTRY.keys())


def get_language_name(code: str) -> str:
    """Get human-readable name for a language code."""
    return _LANGUAGE_NAMES.get(code, code.upper())


def is_language_supported(code: str) -> bool:
    """Check if a language is registered."""
    return code in _EVALUATOR_REGISTRY


# --- Auto-register built-in languages ---

def _register_builtins():
    """Register built-in language evaluators."""
    try:
        from .gmtw_ro.eval.ro_evaluator import RomanianEvaluator
        register_language("ro", RomanianEvaluator, "Romanian")
    except ImportError:
        pass

    try:
        from .gmtw_en.eval.en_evaluator import EnglishEvaluator
        register_language("en", EnglishEvaluator, "English")
    except ImportError:
        pass

    try:
        from .gmtw_de.eval.de_evaluator import GermanEvaluator
        register_language("de", GermanEvaluator, "German")
    except ImportError:
        pass


_register_builtins()