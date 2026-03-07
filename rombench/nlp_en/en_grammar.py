"""
English grammar checker using LanguageTool
"""

from typing import Any

from rombench.nlp import BaseGrammarChecker


# Lazy import
_tool_instance = None


def _get_tool():
    """Lazy initialization of LanguageTool instance for English."""
    global _tool_instance
    
    if _tool_instance is not None:
        return _tool_instance
    
    try:
        import language_tool_python
        _tool_instance = language_tool_python.LanguageTool("en-US")
        return _tool_instance
    except ImportError:
        raise ImportError(
            "language-tool-python is not installed. "
            "Install with: pip install language-tool-python"
        )


SEVERITY_WEIGHTS = {
    "grammar": 3,
    "misspelling": 2,
    "typographical": 1,
    "style": 1,
}


class EnglishGrammarChecker(BaseGrammarChecker):
    """English grammar checker using LanguageTool"""
    
    def check(self, text: str) -> dict[str, Any]:
        """Check English grammar and return analysis"""
        if not self.is_available():
            return {
                'score': 1.0,
                'error_count': 0,
                'errors': [],
                'available': False,
            }
        
        tool = _get_tool()
        matches = tool.check(text)
        
        # Compute word count
        word_count = len(text.split())
        
        # Weight errors by severity
        weighted_errors = sum(
            SEVERITY_WEIGHTS.get(m.rule_issue_type.lower(), 1)
            for m in matches
        )
        
        # Compute score
        if word_count == 0:
            score = 0.0
        else:
            error_density = weighted_errors / word_count
            score = max(0.0, 1.0 - error_density)
        
        return {
            'score': score,
            'error_count': len(matches),
            'weighted_errors': weighted_errors,
            'total_words': word_count,
            'errors': [
                {
                    'message': m.message,
                    'rule_id': m.rule_id,
                    'issue_type': m.rule_issue_type,
                    'context': m.context,
                }
                for m in matches[:10]  # Limit to first 10
            ],
            'available': True,
        }
    
    def is_available(self) -> bool:
        """Check if LanguageTool is available"""
        try:
            import language_tool_python
            return True
        except ImportError:
            return False