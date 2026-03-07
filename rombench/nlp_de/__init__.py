"""
German NLP Toolkit for GMTW evaluation

Provides German language support:
- Umlaut analysis (ä, ö, ü, ß)
- Text quality scoring
- German morphological forms for faithfulness
"""

from .de_toolkit import GermanNLPToolkit
from .faithfulness import GermanFaithfulness

__all__ = [
    'GermanNLPToolkit',
    'GermanFaithfulness',
]